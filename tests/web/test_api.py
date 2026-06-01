"""
Tests for the vocal web API (vocal/web/api.py) and the resolver-driven check
flow behind it (vocal/web/utils.py).

Strategy
--------
``TestClient`` tests cover the HTTP layer only: ``check_upload`` is mocked so
each test exercises exactly one routing/rendering concern.

``TestCheckUploadResolution`` drives ``check_upload`` directly against real
netCDF files and a controlled in-memory registry, asserting the typed
``error: {code, message, hint}`` shape surfaced for each resolver failure
category and for the web-only attribute preconditions. The project import is
patched so no real project package is needed.

Mocking conventions
-------------------
- ``vocal.web.api.Registry.open`` is replaced with a contextmanager function
  that yields a Registry with controlled contents.
- ``vocal.web.api.fetch`` is replaced with a plain Mock/side_effect (its
  return value is a ``ResourceKind`` that drives the redirect target).
- ``vocal.web.api.check_upload`` is an ``async def``; ``unittest.mock.patch``
  automatically uses AsyncMock for coroutine targets, so ``return_value`` and
  ``side_effect`` work as expected.
"""

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Generator
from unittest.mock import patch

import netCDF4
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from vocal.application.fetch import FetchError
from vocal.application.install import derive_url_slug
from vocal.application.resource import ResourceKind
from vocal.manifest import ManifestProduct, build_manifest
from vocal.utils.registry import Pack, Project, Registry
from vocal.web.api import app
from vocal.web.models import CheckContext, ResolverError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _open_registry(projects: dict[str, Project]) -> Generator[Registry, None, None]:
    """Yields a Registry pre-populated with *projects* — used as a drop-in
    replacement for Registry.open() in tests."""
    yield Registry(projects=projects)


@contextmanager
def _open_registry_with_packs(
    packs: list[Pack],
) -> Generator[Registry, None, None]:
    """Yields a Registry pre-populated with *packs* — used as a drop-in
    replacement for Registry.open() in tests."""
    registry = Registry()
    for pack in packs:
        registry.add_pack(pack)
    yield registry


def _make_project(name: str = "test_project") -> Project:
    return Project(
        name=name,
        major=1,
        minor=0,
        project_directory=name,
        local_path=f"/tmp/{name}",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# GET /add
# ---------------------------------------------------------------------------


class TestAddGet:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/add")
        assert response.status_code == 200

    def test_returns_html(self, client: TestClient) -> None:
        response = client.get("/add")
        assert "text/html" in response.headers["content-type"]

    def test_heading_is_kind_neutral(self, client: TestClient) -> None:
        response = client.get("/add")
        assert "Add project or pack" in response.text


# ---------------------------------------------------------------------------
# POST /add — the kind→redirect seam and error re-render
# ---------------------------------------------------------------------------


class TestAddPost:
    def test_no_url_returns_400(self, client: TestClient) -> None:
        response = client.post("/add", data={})
        assert response.status_code == 400

    def test_project_redirects_to_projects(self, client: TestClient) -> None:
        with patch("vocal.web.api.fetch", return_value=ResourceKind.PROJECT):
            response = client.post(
                "/add",
                data={"url": "https://example.com/project"},
                follow_redirects=False,
            )
        assert response.status_code == 302
        assert response.headers["location"] == "/projects"

    def test_pack_redirects_to_packs(self, client: TestClient) -> None:
        with patch("vocal.web.api.fetch", return_value=ResourceKind.PACK):
            response = client.post(
                "/add",
                data={"url": "https://example.com/pack"},
                follow_redirects=False,
            )
        assert response.status_code == 302
        assert response.headers["location"] == "/packs"

    def test_fetch_failure_rerenders_form_with_error(self, client: TestClient) -> None:
        with patch(
            "vocal.web.api.fetch",
            side_effect=FetchError("unreachable host", hint="check your network"),
        ):
            response = client.post(
                "/add",
                data={"url": "https://example.com/bad"},
            )
        assert response.status_code == 422
        assert "unreachable host" in response.text
        assert "check your network" in response.text
        # URL is preserved in the form so the user doesn't have to retype it.
        assert "https://example.com/bad" in response.text

    def test_unknown_failure_renders_error_page(self, client: TestClient) -> None:
        client_no_raise = TestClient(app, raise_server_exceptions=False)
        with patch("vocal.web.api.fetch", side_effect=RuntimeError("kaboom")):
            response = client_no_raise.post(
                "/add",
                data={"url": "https://example.com/bad"},
            )
        assert response.status_code == 500
        assert "We couldn't check that file" in response.text


# ---------------------------------------------------------------------------
# GET /projects
# ---------------------------------------------------------------------------


class TestProjectsGet:
    def test_empty_registry_returns_200(self, client: TestClient) -> None:
        with patch("vocal.web.api.Registry.open", lambda: _open_registry({})):
            response = client.get("/projects")
        assert response.status_code == 200

    def test_project_name_in_response(self, client: TestClient) -> None:
        project = _make_project("my_project")
        with patch(
            "vocal.web.api.Registry.open",
            lambda: _open_registry({"my_project": project}),
        ):
            response = client.get("/projects")
        assert "my_project" in response.text

    def test_targeting_pack_links_to_packs_anchor(self, client: TestClient) -> None:
        project = Project(
            name="MYSTD",
            major=2,
            minor=5,
            project_directory="mystd",
            local_path="/tmp/mystd",
        )
        pack = _pack(url="https://host/widgets", version=1)

        @contextmanager
        def _open() -> Generator[Registry, None, None]:
            registry = Registry(projects={project.key: project})
            registry.add_pack(pack)
            yield registry

        with patch("vocal.web.api.Registry.open", _open):
            response = client.get("/projects")

        assert response.status_code == 200
        assert "https://host/widgets" in response.text
        assert f'/packs#{derive_url_slug("https://host/widgets")}' in response.text


# ---------------------------------------------------------------------------
# GET /packs
# ---------------------------------------------------------------------------


class TestPacksGet:
    def test_no_packs_returns_200(self, client: TestClient) -> None:
        with patch(
            "vocal.web.api.Registry.open", lambda: _open_registry_with_packs([])
        ):
            response = client.get("/packs")
        assert response.status_code == 200

    def test_no_packs_serves_empty_state(self, client: TestClient) -> None:
        with patch(
            "vocal.web.api.Registry.open", lambda: _open_registry_with_packs([])
        ):
            response = client.get("/packs")
        assert "No packs registered" in response.text

    def test_registered_pack_url_in_response(self, client: TestClient) -> None:
        pack = _pack(url="https://host/widgets", version=2)
        with patch(
            "vocal.web.api.Registry.open",
            lambda: _open_registry_with_packs([pack]),
        ):
            response = client.get("/packs")
        assert response.status_code == 200
        assert "https://host/widgets" in response.text


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestRootGet:
    def test_no_registry_file_returns_200(self, client: TestClient) -> None:
        with patch("vocal.web.api.Registry.open", side_effect=FileNotFoundError):
            response = client.get("/")
        assert response.status_code == 200

    def test_no_projects_serves_no_projects_template(self, client: TestClient) -> None:
        with patch("vocal.web.api.Registry.open", side_effect=FileNotFoundError):
            response = client.get("/")
        assert "No projects registered" in response.text

    def test_with_projects_serves_checker_template(self, client: TestClient) -> None:
        project = _make_project()
        with patch(
            "vocal.web.api.Registry.open",
            lambda: _open_registry({"test_project": project}),
        ):
            response = client.get("/")
        assert "Check a NetCDF File" in response.text


# ---------------------------------------------------------------------------
# POST / — HTTP layer (check_upload mocked)
# ---------------------------------------------------------------------------


class TestUploadPost:
    def test_missing_file_returns_422(self, client: TestClient) -> None:
        response = client.post("/")
        assert response.status_code == 422

    def test_returns_200_with_file(self, client: TestClient) -> None:
        with patch("vocal.web.api.check_upload", return_value=CheckContext()):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 200

    def test_returns_html_with_file(self, client: TestClient) -> None:
        with patch("vocal.web.api.check_upload", return_value=CheckContext()):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert "text/html" in response.headers["content-type"]

    def test_resolver_error_renders_banner_with_message_and_hint(
        self, client: TestClient
    ) -> None:
        context = CheckContext(
            error=ResolverError(
                code="pack_missing",
                message="No pack registered for https://host/packs version 3",
                hint="Run 'vocal fetch https://host/packs' to register it.",
            )
        )
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 200
        assert "COULDN'T CHECK FILE" in response.text
        # message and hint both reach the rendered page.
        assert "No pack registered for https://host/packs version 3" in response.text
        # Apostrophes in the hint are HTML-escaped by the template; match the
        # stable parts of the hint that survive escaping.
        assert "vocal fetch https://host/packs" in response.text
        assert "to register it." in response.text

    def test_unknown_check_failure_renders_error_page(
        self, client: TestClient
    ) -> None:
        client_no_raise = TestClient(app, raise_server_exceptions=False)
        with patch(
            "vocal.web.api.check_upload", side_effect=RuntimeError("kaboom")
        ):
            response = client_no_raise.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 500
        assert "We couldn't check that file" in response.text


# ---------------------------------------------------------------------------
# check_upload — resolver-driven flow, asserting the typed error shape
# ---------------------------------------------------------------------------

FILECODEC = {"date": {"regex": r"\d{8}"}}


def _make_nc(
    tmp_path: Path,
    name: str = "foo_20260522.nc",
    *,
    conventions: str | None = None,
    project_url: str | None = None,
    definitions_url: str | None = None,
    definitions_version: int | None = None,
) -> str:
    """Write a minimal netCDF file carrying the given vocal-managed attributes."""
    path = str(tmp_path / name)
    with netCDF4.Dataset(path, "w") as nc:
        if conventions is not None:
            nc.Conventions = conventions
        if project_url is not None:
            nc.vocal_project_url = project_url
        if definitions_url is not None:
            nc.vocal_definitions_url = definitions_url
        if definitions_version is not None:
            nc.vocal_definitions_version = definitions_version
    return path


def _project(name: str = "MYSTD", major: int = 2, minor: int = 3) -> Project:
    return Project(
        name=name,
        major=major,
        minor=minor,
        project_directory="mystd",
        local_path="/cache/projects/mystd",
    )


def _pack(
    url: str = "https://host/packs",
    version: int = 3,
    name: str = "MYSTD",
    major: int = 2,
    min_minor: int = 3,
    local_path: str = "/cache/packs/host-packs/v3",
    products=None,
) -> Pack:
    if products is None:
        products = [
            ManifestProduct(
                name="foo", file_pattern="foo_{date}", schema="product_foo.json"
            )
        ]
    manifest = build_manifest(
        version=version,
        url=url,
        standard_name=name,
        standard_major=major,
        min_minor=min_minor,
        products=products,
    )
    return Pack(manifest=manifest, local_path=local_path)


def _registry(project: Project | None = None, pack: Pack | None = None) -> Registry:
    registry = Registry()
    if project is not None:
        registry.add_project(project)
    if pack is not None:
        registry.add_pack(pack)
    return registry


def _fake_project_module() -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(Dataset=object()),
        filecodec=FILECODEC,
    )


def _run_check_upload(nc_path: str, registry: Registry) -> CheckContext:
    """Drive ``check_upload`` against ``nc_path`` with the registry and project
    import patched, returning the resulting context."""
    from vocal.web import utils as web_utils

    with open(nc_path, "rb") as fh:
        upload = UploadFile(filename=Path(nc_path).name, file=fh)
        with (
            patch.object(web_utils.Registry, "load", return_value=registry),
            patch.object(
                web_utils,
                "import_project_package",
                return_value=_fake_project_module(),
            ),
        ):
            return asyncio.run(web_utils.check_upload(upload))


class TestCheckUploadResolution:
    """The five resolver failure categories each surface as a typed error."""

    def test_project_missing(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            project_url="https://host/mystd.git",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        context = _run_check_upload(nc, _registry())

        assert context.error is not None
        assert context.error.code == "project_missing"
        assert "No project registered for MYSTD-2" in context.error.message
        assert context.error.hint is not None
        assert not context.projects and not context.definitions

    def test_project_too_old(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.5",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        context = _run_check_upload(nc, _registry(project=_project(minor=3)))

        assert context.error is not None
        assert context.error.code == "project_too_old"
        assert (
            "File claims MYSTD-2.5 but registered project is at MYSTD-2.3"
            in context.error.message
        )

    def test_pack_missing(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        context = _run_check_upload(nc, _registry(project=_project()))

        assert context.error is not None
        assert context.error.code == "pack_missing"
        assert (
            "No pack registered for https://host/packs version 3"
            in context.error.message
        )
        assert context.error.hint is not None
        assert "vocal fetch https://host/packs" in context.error.hint

    def test_pack_incompatible(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        pack = _pack(major=3, min_minor=0)
        context = _run_check_upload(nc, _registry(project=_project(), pack=pack))

        assert context.error is not None
        assert context.error.code == "pack_incompatible"
        assert (
            "Pack targets MYSTD-3 but registered project is MYSTD-2"
            in context.error.message
        )

    def test_product_not_found(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            name="unmatched.nc",
            conventions="MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        context = _run_check_upload(nc, _registry(project=_project(), pack=_pack()))

        assert context.error is not None
        assert context.error.code == "product_not_found"
        assert "'unmatched.nc' did not match any product pattern" in (
            context.error.message
        )
        assert context.error.hint is not None
        assert "foo_{date}" in context.error.hint


class TestCheckUploadPreconditions:
    """The web flow rejects files it cannot resolve without a flag fallback."""

    def test_missing_conventions(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            definitions_url="https://host/packs",
            definitions_version=3,
        )  # no Conventions attribute
        context = _run_check_upload(nc, _registry(project=_project()))

        assert context.error is not None
        assert context.error.code == "missing_conventions"
        assert "Conventions" in context.error.message
        assert not context.projects and not context.definitions

    def test_missing_pack_reference(self, tmp_path: Path) -> None:
        # Conventions present, but no vocal_definitions_url / _version: the web
        # UI cannot fall back to a -d flag, so the file is rejected.
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3")
        context = _run_check_upload(nc, _registry(project=_project()))

        assert context.error is not None
        assert context.error.code == "missing_pack_reference"
        assert context.error.hint is not None
        assert "vocal_definitions_url" in context.error.hint
