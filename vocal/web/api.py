import logging
import os

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from vocal.utils.registry import Registry
from vocal.application.fetch import fetch_project
from vocal.exceptions import VocalError
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


@app.get("/projects/add", response_class=HTMLResponse)
async def add_project_get(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="add-project.html")


@app.post("/projects/add", response_class=HTMLResponse)
async def add_project_post(request: Request):
    form = await request.form()
    url = form.get("url")

    if not url or not isinstance(url, str):
        return templates.TemplateResponse(
            request=request,
            name="add-project.html",
            context={"error": "Please provide a project URL.", "url": url or ""},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        fetch_project(url, git=False)
    except VocalError as e:
        return templates.TemplateResponse(
            request=request,
            name="add-project.html",
            context={"error": e.message, "hint": e.hint, "url": url},
            status_code=e.status_code,
        )

    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@app.get("/projects", response_class=HTMLResponse)
async def projects(request: Request) -> HTMLResponse:
    with Registry.open() as registry:
        projects = registry.projects

    return templates.TemplateResponse(
        request=request, name="projects.html", context={"projects": projects}
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
