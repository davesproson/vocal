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
from vocal.checking.shared import (
    CheckOutcome,
    ProjectCheckResult,
    Verdict,
)
from vocal.manifest import ManifestProduct, build_manifest
from vocal.resolution import ProjectTarget, Resolution
from vocal.utils.registry import Pack, Project, Registry
from vocal.versioning import Version, VersionConstraint
from vocal.web.api import app, create_app
from vocal.web.models import CheckContext, Landing, ResolverError, UnverifiedClaim


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
    # Downloads enabled: the /add and rendering tests exercise the fetch route.
    # The default module-level ``app`` has downloads disabled (the safe posture),
    # so build an explicitly-enabled app for these tests.
    return TestClient(create_app(allow_user_download=True), raise_server_exceptions=True)


@pytest.fixture
def client_no_downloads() -> TestClient:
    # The default-off posture: matches the module-level ``app`` but built
    # explicitly so the intent is local to the gate tests.
    return TestClient(create_app(allow_user_download=False), raise_server_exceptions=True)


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
        client_no_raise = TestClient(
            create_app(allow_user_download=True), raise_server_exceptions=False
        )
        with patch("vocal.web.api.fetch", side_effect=RuntimeError("kaboom")):
            response = client_no_raise.post(
                "/add",
                data={"url": "https://example.com/bad"},
            )
        assert response.status_code == 500
        assert "We couldn't check that file" in response.text


# ---------------------------------------------------------------------------
# Download gate — /add refuses and the Add button hides when downloads are off
# ---------------------------------------------------------------------------


class TestDownloadGate:
    """With ``--allow-downloads`` off, /add is refused server-side on both verbs
    and the Add affordance is absent from every page that renders it."""

    def test_get_add_returns_403(self, client_no_downloads: TestClient) -> None:
        response = client_no_downloads.get("/add")
        assert response.status_code == 403
        assert "disabled" in response.text

    def test_post_add_returns_403_without_fetching(
        self, client_no_downloads: TestClient
    ) -> None:
        with patch("vocal.web.api.fetch") as fetch_mock:
            response = client_no_downloads.post(
                "/add", data={"url": "https://example.com/project"}
            )
        assert response.status_code == 403
        # The gate fires before any fetch — the RCE surface never opens.
        fetch_mock.assert_not_called()

    def test_projects_page_hides_add_button(
        self, client_no_downloads: TestClient
    ) -> None:
        with patch("vocal.web.api.Registry.open", lambda: _open_registry({})):
            response = client_no_downloads.get("/projects")
        assert response.status_code == 200
        assert 'href="/add"' not in response.text

    def test_packs_page_hides_add_button(
        self, client_no_downloads: TestClient
    ) -> None:
        pack = _pack(url="https://host/widgets", version=2)
        with patch(
            "vocal.web.api.Registry.open",
            lambda: _open_registry_with_packs([pack]),
        ):
            response = client_no_downloads.get("/packs")
        assert response.status_code == 200
        assert 'href="/add"' not in response.text

    def test_no_projects_page_hides_add_button(
        self, client_no_downloads: TestClient
    ) -> None:
        with patch("vocal.web.api.Registry.open", side_effect=FileNotFoundError):
            response = client_no_downloads.get("/")
        assert response.status_code == 200
        assert 'href="/add"' not in response.text

    def test_projects_page_shows_add_button_when_enabled(
        self, client: TestClient
    ) -> None:
        with patch("vocal.web.api.Registry.open", lambda: _open_registry({})):
            response = client.get("/projects")
        assert response.status_code == 200
        assert 'href="/add"' in response.text


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

    def test_passed_state_renders_distinctly(self, client: TestClient) -> None:
        context = CheckContext(verdict="pass")
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 200
        assert "PASSED" in response.text
        assert "FAILED" not in response.text
        assert "COULDN'T VERIFY" not in response.text

    def test_failed_state_renders_distinctly(self, client: TestClient) -> None:
        context = CheckContext(verdict="fail")
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 200
        assert "FAILED" in response.text
        assert "PASSED" not in response.text
        assert "COULDN'T VERIFY" not in response.text

    def test_indeterminate_state_renders_distinctly_with_fetch_items(
        self, client: TestClient
    ) -> None:
        context = CheckContext(
            verdict="indeterminate",
            unverified=[
                UnverifiedClaim(
                    message="No project registered for https://host/mystd.git",
                    hint="Run 'vocal fetch https://host/mystd.git' to register it.",
                )
            ],
        )
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 200
        assert "COULDN'T VERIFY" in response.text
        assert "PASSED" not in response.text
        # The fetch/update items are surfaced so the user knows what to install.
        assert "No project registered for https://host/mystd.git" in response.text
        assert "vocal fetch https://host/mystd.git" in response.text

    def test_refusal_renders_not_vocal_managed_banner(
        self, client: TestClient
    ) -> None:
        context = CheckContext(
            error=ResolverError(
                code="not_vocal_managed",
                message="This file carries no recognisable vocal claim.",
                hint="Add a vocal_project_url, a vocal_definitions_url, or a "
                "Conventions token naming an installed standard.",
            )
        )
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 200
        # A refusal is a distinct state, not one of the three verdicts.
        assert "NOT A VOCAL-MANAGED FILE" in response.text
        assert "This file carries no recognisable vocal claim." in response.text
        assert "PASSED" not in response.text
        assert "FAILED" not in response.text

    def test_stored_landing_renders_confirmation(self, client: TestClient) -> None:
        context = CheckContext(
            verdict="pass",
            landing=Landing(
                status="stored",
                message="Your file passed validation and was stored.",
            ),
        )
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 200
        assert "Your file passed validation and was stored." in response.text

    def test_refused_landing_renders_reason(self, client: TestClient) -> None:
        context = CheckContext(
            verdict="pass",
            landing=Landing(
                status="refused",
                message=(
                    "Your file passed validation but could not be stored: the "
                    "filename was rejected as unsafe."
                ),
            ),
        )
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 200
        assert "could not be stored" in response.text

    def test_landing_message_omits_absolute_server_path(
        self, client: TestClient
    ) -> None:
        # The landing notice is path-free by design: the message carries no
        # absolute path, so a remote user never learns the server's layout.
        context = CheckContext(
            verdict="pass",
            landing=Landing(
                status="stored",
                message="Your file passed validation and was stored.",
            ),
        )
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert "/data/incoming" not in response.text

    def test_no_landing_notice_for_non_pass_verdict(
        self, client: TestClient
    ) -> None:
        # A FAIL carries no landing result, so the page shows no landing notice.
        context = CheckContext(verdict="fail")
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert "was stored" not in response.text
        assert "could not be stored" not in response.text

    def test_upload_dir_stored_on_app_state(self, tmp_path) -> None:
        # create_app stashes upload_dir on app.state exactly as it does
        # allow_user_download, so the route can read it at request time.
        app = create_app(upload_dir=tmp_path)
        assert app.state.upload_dir == tmp_path

    def test_upload_dir_defaults_to_none(self) -> None:
        app = create_app()
        assert app.state.upload_dir is None

    def test_post_root_passes_upload_dir_to_check_upload(self, tmp_path) -> None:
        app = create_app(upload_dir=tmp_path)
        client = TestClient(app, raise_server_exceptions=True)
        with patch(
            "vocal.web.api.check_upload", return_value=CheckContext()
        ) as check:
            client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert check.call_args.kwargs["upload_dir"] == tmp_path

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
# check_upload — resolver-driven flow, asserting the tri-state verdict
# ---------------------------------------------------------------------------
#
# These tests drive ``check_upload`` directly against real netCDF files and a
# controlled in-memory registry, asserting the observable outcome: the verdict
# (``pass`` / ``fail`` / ``indeterminate``), whether the file was refused
# upfront, and the fetch/update items surfaced for an incomplete check. No
# project package or pack schema is read from disk — cases that would run a
# model/schema check either resolve to "nothing runnable" (INDETERMINATE) or
# patch ``run_check`` so the surface's verdict-mapping is what is under test,
# not the spine's validation mechanics (those are covered by the spine's own
# tests). The surface never fetches: no fetch is mocked, and there is no fetch
# path to mock.

MYSTD_URL = "https://host/mystd.git"
PACKS_URL = "https://host/packs"


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


def _project(
    name: str = "MYSTD",
    major: int = 2,
    minor: int = 3,
    url: str = "",
) -> Project:
    return Project(
        name=name,
        major=major,
        minor=minor,
        project_directory="mystd",
        local_path="/cache/projects/mystd",
        url=url,
    )


def _pack(
    url: str = PACKS_URL,
    version: int = 3,
    satisfies: tuple[VersionConstraint, ...] = (VersionConstraint("MYSTD", 2, 3),),
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
        filecodec={"date": {"regex": r"\d{8}"}},
        satisfies_standards=satisfies,
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


def _pass_outcome(resolution: Resolution) -> CheckOutcome:
    """A passing :class:`CheckOutcome` over ``resolution``'s verifiable targets.

    Used where a fully-resolved file would otherwise run a real model/schema
    check; lets a ``check_upload`` test assert the surface maps a PASS verdict
    through without scaffolding a real project package on disk.
    """
    return CheckOutcome(
        project_results=[
            ProjectCheckResult(target=target)
            for target in resolution.projects
            if target.verifiable
        ],
        pack_result=None,
        failures=[],
        warnings=[],
        comments=[],
        verdict=Verdict.PASS,
    )


def _fail_outcome(resolution: Resolution) -> CheckOutcome:
    """A failing :class:`CheckOutcome` over ``resolution``'s verifiable targets.

    Mirrors :func:`_pass_outcome` but stamps a FAIL verdict, so a ``check_upload``
    test can assert the surface treats a failed file as never-stored without
    scaffolding a project package that genuinely violates a check.
    """
    return CheckOutcome(
        project_results=[
            ProjectCheckResult(target=target)
            for target in resolution.projects
            if target.verifiable
        ],
        pack_result=None,
        failures=[],
        warnings=[],
        comments=[],
        verdict=Verdict.FAIL,
    )


def _run_check_upload(
    nc_path: str,
    registry: Registry,
    *,
    run_check=None,
    upload_dir: Path | None = None,
) -> CheckContext:
    """Drive ``check_upload`` against ``nc_path`` with the registry patched.

    ``run_check`` optionally replaces the check spine (e.g. with
    :func:`_pass_outcome`) for cases that would otherwise read a project package
    or pack schema from disk. ``upload_dir`` is threaded through unchanged to
    exercise the ``--upload-to`` landing.
    """
    from vocal.web import utils as web_utils

    with open(nc_path, "rb") as fh:
        upload = UploadFile(filename=Path(nc_path).name, file=fh)
        patches = [patch("vocal.web.utils.Registry.load", return_value=registry)]
        if run_check is not None:
            patches.append(patch("vocal.web.utils.run_check", side_effect=run_check))
        with patches[0]:
            if run_check is not None:
                with patches[1]:
                    return asyncio.run(
                        web_utils.check_upload(upload, upload_dir=upload_dir)
                    )
            return asyncio.run(
                web_utils.check_upload(upload, upload_dir=upload_dir)
            )


class TestCheckUploadVerdict:
    """A recognisable file is always given a tri-state verdict, never refused."""

    def test_pass_from_own_claims(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            project_url=MYSTD_URL,
        )
        registry = _registry(project=_project(url=MYSTD_URL))
        context = _run_check_upload(
            nc, registry, run_check=lambda res, fn: _pass_outcome(res)
        )

        assert context.error is None
        assert context.verdict == "pass"
        assert "MYSTD-2" in context.projects
        assert not context.unverified

    def test_mandatory_project_missing_is_indeterminate(self, tmp_path: Path) -> None:
        # A vocal_project_url whose project isn't installed is a recognisable
        # claim: accepted and rendered INDETERMINATE (with a fetch hint), never
        # refused and never silently passed.
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3", project_url=MYSTD_URL)
        context = _run_check_upload(nc, _registry())

        assert context.error is None
        assert context.verdict == "indeterminate"
        assert any("No project registered" in c.message for c in context.unverified)
        assert any(
            c.hint and "vocal fetch" in c.hint for c in context.unverified
        )

    def test_missing_pack_is_indeterminate(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path, definitions_url=PACKS_URL, definitions_version=3
        )
        context = _run_check_upload(nc, _registry())

        assert context.error is None
        assert context.verdict == "indeterminate"
        assert any(
            "No pack registered for https://host/packs version 3" in c.message
            for c in context.unverified
        )

    def test_too_old_standard_is_indeterminate_with_update_hint(
        self, tmp_path: Path
    ) -> None:
        # The file claims a newer minor than is installed: the older model must
        # not be run, so the claim is unverifiable → INDETERMINATE with an
        # update hint (not a silent pass, not a fail).
        nc = _make_nc(tmp_path, conventions="MYSTD-2.5", project_url=MYSTD_URL)
        registry = _registry(project=_project(minor=3, url=MYSTD_URL))
        context = _run_check_upload(nc, registry)

        assert context.error is None
        assert context.verdict == "indeterminate"
        assert any(
            c.hint and "--update" in c.hint for c in context.unverified
        )

    def test_product_not_found_is_indeterminate(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            name="unmatched.nc",
            definitions_url=PACKS_URL,
            definitions_version=3,
        )
        context = _run_check_upload(
            nc, _registry(pack=_pack())
        )

        assert context.error is None
        assert context.verdict == "indeterminate"
        assert any(
            "did not match any product pattern" in c.message
            for c in context.unverified
        )


class TestCheckUploadLanding:
    """``--upload-to``: a PASS file is stored; nothing else is."""

    def test_pass_lands_file_and_sets_landing(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3", project_url=MYSTD_URL)
        registry = _registry(project=_project(url=MYSTD_URL))
        dest = tmp_path / "incoming"
        dest.mkdir()

        context = _run_check_upload(
            nc,
            registry,
            run_check=lambda res, fn: _pass_outcome(res),
            upload_dir=dest,
        )

        assert context.verdict == "pass"
        assert context.landing is not None and context.landing.status == "stored"
        # The validated file was copied into the configured directory.
        assert (dest / Path(nc).name).exists()

    def test_feature_off_leaves_landing_none(self, tmp_path: Path) -> None:
        # Absent --upload-to (upload_dir=None), a PASS stores nothing.
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3", project_url=MYSTD_URL)
        registry = _registry(project=_project(url=MYSTD_URL))

        context = _run_check_upload(
            nc, registry, run_check=lambda res, fn: _pass_outcome(res)
        )

        assert context.verdict == "pass"
        assert context.landing is None

    def test_fail_stores_nothing(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3", project_url=MYSTD_URL)
        registry = _registry(project=_project(url=MYSTD_URL))
        dest = tmp_path / "incoming"
        dest.mkdir()

        context = _run_check_upload(
            nc,
            registry,
            run_check=lambda res, fn: _fail_outcome(res),
            upload_dir=dest,
        )

        assert context.verdict == "fail"
        assert context.landing is None
        assert list(dest.iterdir()) == []

    def test_indeterminate_stores_nothing(self, tmp_path: Path) -> None:
        # A mandatory project that isn't installed → INDETERMINATE: never stored.
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3", project_url=MYSTD_URL)
        dest = tmp_path / "incoming"
        dest.mkdir()

        context = _run_check_upload(nc, _registry(), upload_dir=dest)

        assert context.verdict == "indeterminate"
        assert context.landing is None
        assert list(dest.iterdir()) == []


class TestCheckUploadPrecondition:
    """Only a file with no recognisable vocal claim is refused upfront."""

    def test_no_vocal_claim_refused(self, tmp_path: Path) -> None:
        # Only an external CF token, no installed CF project, no vocal URLs:
        # not a vocal-managed file. Refused upfront — a distinct state, not a
        # verdict.
        nc = _make_nc(tmp_path, conventions="CF-1.8")
        context = _run_check_upload(nc, _registry(project=_project()))

        assert context.error is not None
        assert context.error.code == "not_vocal_managed"
        assert context.error.hint is not None
        assert context.verdict is None
        assert not context.projects and not context.definitions

    def test_no_attributes_at_all_refused(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)  # no vocal-managed attributes whatsoever
        context = _run_check_upload(nc, _registry(project=_project()))

        assert context.error is not None
        assert context.error.code == "not_vocal_managed"
        assert context.verdict is None

    def test_conventions_only_no_installed_match_is_indeterminate(
        self, tmp_path: Path
    ) -> None:
        # A Conventions token whose standard name matches an installed project but
        # whose major is not installed: recognisable, so accepted — and rendered
        # INDETERMINATE rather than rejected.
        nc = _make_nc(tmp_path, conventions="MYSTD-3.0")
        context = _run_check_upload(
            nc, _registry(project=_project(major=2, minor=3))
        )

        assert context.error is None
        assert context.verdict == "indeterminate"
        assert any("MYSTD-3.0" in c.message for c in context.unverified)

    def test_conventions_only_installed_match_is_checked(
        self, tmp_path: Path
    ) -> None:
        # The opportunistic standard is installed at the claimed version, so the
        # file is genuinely checked and gets a verdict.
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3")
        registry = _registry(project=_project(major=2, minor=3))
        context = _run_check_upload(
            nc, registry, run_check=lambda res, fn: _pass_outcome(res)
        )

        assert context.error is None
        assert context.verdict == "pass"
        assert "MYSTD-2" in context.projects
