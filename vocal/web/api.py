import logging
import os

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from vocal.utils.registry import Registry
from vocal.application.fetch import fetch
from vocal.application.resource import ResourceKind
from vocal.exceptions import VocalError
from vocal.web.models import LibraryView, build_library_view
from vocal.web.utils import check_upload

logger = logging.getLogger(__name__)

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)


def _form_flag(value: object) -> bool:
    """Interpret an optional form field as a boolean flag.

    A checkbox/flag is "on" when present with a truthy string ("1", "true",
    "on", "yes"); absent or any other value is off. Lets the URL-only Add form
    coexist with a future pack "Update" affordance that posts these fields.
    """
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"1", "true", "on", "yes"}


@app.exception_handler(VocalError)
async def handle_vocal_error(request: Request, exc: VocalError) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={"message": exc.message, "hint": exc.hint},
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def handle_unhandled_error(request: Request, exc: Exception) -> HTMLResponse:
    logger.exception("Unhandled error in %s %s", request.method, request.url.path)
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={
            "message": "Something went wrong while handling your request.",
            "hint": "If this keeps happening, check the server log.",
        },
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="about.html")


@app.get("/add", response_class=HTMLResponse)
async def add_get(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="add.html")


@app.post("/add", response_class=HTMLResponse)
async def add_post(request: Request):
    """Fetch whatever a URL points at — project or pack — and land on its tab.

    The form is URL-only, but the handler accepts optional ``git`` / ``update``
    / ``force`` flags (defaulting off) so a later pack "Update" affordance can
    pass them without reshaping the route. ``fetch`` downloads once, classifies
    the tree, and returns the :class:`ResourceKind` it installed; we redirect to
    ``/projects`` for a project or ``/packs`` for a pack.
    """
    form = await request.form()
    url = form.get("url")
    git = _form_flag(form.get("git"))
    update = _form_flag(form.get("update"))
    force = _form_flag(form.get("force"))

    if not url or not isinstance(url, str):
        return templates.TemplateResponse(
            request=request,
            name="add.html",
            context={"error": "Please provide a project or pack URL.", "url": url or ""},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        kind = fetch(url, git=git, update=update, force=force)
    except VocalError as e:
        return templates.TemplateResponse(
            request=request,
            name="add.html",
            context={"error": e.message, "hint": e.hint, "url": url},
            status_code=e.status_code,
        )

    target = "/packs" if kind is ResourceKind.PACK else "/projects"
    return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)


@app.get("/projects", response_class=HTMLResponse)
async def projects(request: Request) -> HTMLResponse:
    try:
        with Registry.open() as registry:
            library = build_library_view(registry)
    except FileNotFoundError:
        library = LibraryView()

    return templates.TemplateResponse(
        request=request, name="projects.html", context={"library": library}
    )


@app.get("/packs", response_class=HTMLResponse)
async def packs(request: Request) -> HTMLResponse:
    try:
        with Registry.open() as registry:
            library = build_library_view(registry)
    except FileNotFoundError:
        library = LibraryView()

    if not library.packs:
        return templates.TemplateResponse(request=request, name="no-packs.html")

    return templates.TemplateResponse(
        request=request, name="packs.html", context={"library": library}
    )


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):

    try:
        with Registry.open() as registry:
            num_projects = len(registry.projects)
    except FileNotFoundError:
        num_projects = 0

    if num_projects == 0:
        return templates.TemplateResponse(request=request, name="no-projects.html")

    return templates.TemplateResponse(request=request, name="checker.html")


@app.post("/", response_class=JSONResponse)
async def upload(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    context = await check_upload(file)

    return templates.TemplateResponse(
        request=request, name="checked.html", context=context.model_dump()
    )
