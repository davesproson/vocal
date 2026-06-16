import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, Request, File, UploadFile, status
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


def _download_flag_context(request: Request) -> dict[str, bool]:
    """Inject ``allow_user_download`` into every template's context.

    Registered as a Jinja2 context processor so the flag is present in *all*
    rendered templates without per-route plumbing — this is what lets the "Add"
    affordance be hidden everywhere it appears (and makes "forgot to hide it on
    one page" structurally impossible). The flag is read from ``app.state`` at
    render time, defaulting to ``False`` so a stray render without it errs safe.
    """
    return {
        "allow_user_download": getattr(request.app.state, "allow_user_download", False)
    }


router = APIRouter()
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates"),
    context_processors=[_download_flag_context],
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


def _downloads_disabled_response(request: Request) -> HTMLResponse:
    """Render the 403 served when downloads are disabled.

    Server-side enforcement for the ``--allow-downloads`` gate: hiding the form
    and nav link is presentation only, so both ``GET /add`` and ``POST /add``
    refuse here regardless of how the request was crafted. The default-off
    posture means a drive-by ``POST /add`` (see the CSRF note on ``add_post``)
    lands here unless the operator opted in.
    """
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={
            "message": "Downloading projects and packs is disabled.",
            "hint": (
                "Restart 'vocal web' with --allow-downloads to let the GUI "
                "fetch projects and packs from URLs."
            ),
        },
        status_code=status.HTTP_403_FORBIDDEN,
    )


async def handle_vocal_error(request: Request, exc: VocalError) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={"message": exc.message, "hint": exc.hint},
        status_code=exc.status_code,
    )


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


@router.get("/about", response_class=HTMLResponse)
async def about(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="about.html")


@router.get("/add", response_class=HTMLResponse)
async def add_get(request: Request) -> HTMLResponse:
    if not request.app.state.allow_user_download:
        return _downloads_disabled_response(request)
    return templates.TemplateResponse(request=request, name="add.html")


@router.post("/add", response_class=HTMLResponse)
async def add_post(request: Request):
    """Fetch whatever a URL points at — project or pack — and land on its tab.

    Gated behind ``--allow-downloads`` (a fetched project's code runs on this
    machine when a file is checked): when downloads are off this refuses with a
    403 *before* touching the form, so a cross-origin drive-by POST (this form
    carries no CSRF token — a documented residual risk) cannot fetch.

    The form is URL-only, but the handler accepts optional ``git`` / ``update``
    / ``force`` flags (defaulting off) so a later pack "Update" affordance can
    pass them without reshaping the route. ``fetch`` downloads once, classifies
    the tree, and returns the :class:`ResourceKind` it installed; we redirect to
    ``/projects`` for a project or ``/packs`` for a pack.
    """
    if not request.app.state.allow_user_download:
        return _downloads_disabled_response(request)

    form = await request.form()
    url = form.get("url")
    git = _form_flag(form.get("git"))
    update = _form_flag(form.get("update"))
    force = _form_flag(form.get("force"))

    if not url or not isinstance(url, str):
        return templates.TemplateResponse(
            request=request,
            name="add.html",
            context={
                "error": "Please provide a project or pack URL.",
                "url": url or "",
            },
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


@router.get("/projects", response_class=HTMLResponse)
async def projects(request: Request) -> HTMLResponse:
    try:
        with Registry.open() as registry:
            library = build_library_view(registry)
    except FileNotFoundError:
        library = LibraryView()

    return templates.TemplateResponse(
        request=request, name="projects.html", context={"library": library}
    )


@router.get("/packs", response_class=HTMLResponse)
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


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):

    try:
        with Registry.open() as registry:
            num_projects = len(registry.projects)
    except FileNotFoundError:
        num_projects = 0

    if num_projects == 0:
        return templates.TemplateResponse(request=request, name="no-projects.html")

    return templates.TemplateResponse(
        request=request,
        name="checker.html",
        context={"upload_enabled": request.app.state.upload_dir is not None},
    )


@router.post("/", response_class=JSONResponse)
async def upload(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    context = await check_upload(
        file, upload_dir=getattr(request.app.state, "upload_dir", None)
    )

    return templates.TemplateResponse(
        request=request, name="checked.html", context=context.model_dump()
    )


def create_app(
    *,
    allow_user_download: bool = False,
    upload_dir: Optional[Path] = None,
) -> FastAPI:
    """Build the web-checker app with the download gate and ingest dir set.

    ``allow_user_download`` is stored on ``app.state`` (the idiomatic place for
    app-scoped config) where the ``/add`` handlers and the template context
    processor read it. Defaults to ``False`` so the safe posture holds unless a
    caller — ``vocal web --allow-downloads`` — opts in.

    ``upload_dir`` is stored on ``app.state`` the same way and read by the
    ``POST /`` route, which passes it to :func:`check_upload`. Defaults to
    ``None`` (feature off): absent ``vocal web --upload-to``, no file is stored.
    The directory is validated by the command before the server binds.
    """
    app = FastAPI()
    app.state.allow_user_download = allow_user_download
    app.state.upload_dir = upload_dir
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
        name="static",
    )
    # Starlette types handlers as taking the base Exception; ours narrows to
    # VocalError (the documented per-type-handler pattern), so silence the stub.
    app.add_exception_handler(VocalError, handle_vocal_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, handle_unhandled_error)
    app.include_router(router)
    return app


# Default instance with downloads disabled, for tooling that imports the app by
# name. The CLI builds its own via ``create_app`` so the flag is an honest
# argument rather than a global.
app = create_app()
