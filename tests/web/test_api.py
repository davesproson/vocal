"""
Tests for the vocal web API (vocal/web/api.py).

Strategy
--------
These tests cover the HTTP layer only. Complex internal logic (file checking,
project fetching, registry I/O) is mocked so that each test exercises exactly
one concern. Integration of the underlying logic is covered by the dedicated
tests for checking.py, validation.py, etc.

Mocking conventions
-------------------
- ``vocal.web.api.Registry.open`` is replaced with a contextmanager function
  that yields a Registry with controlled contents.
- ``vocal.web.api.fetch_project`` is replaced with a plain Mock/side_effect.
- ``vocal.web.api.check_upload`` is an ``async def``; ``unittest.mock.patch``
  automatically uses AsyncMock for coroutine targets, so ``return_value`` and
  ``side_effect`` work as expected.
"""

from contextlib import contextmanager
from typing import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from vocal.application.check import NoConventionsFound, NoMatchingProjects
from vocal.application.fetch import FetchError
from vocal.utils.registry import Project, Registry
from vocal.web.api import app
from vocal.web.models import CheckContext, CheckIssue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _open_registry(projects: dict[str, Project]) -> Generator[Registry, None, None]:
    """Yields a Registry pre-populated with *projects* — used as a drop-in
    replacement for Registry.open() in tests."""
    yield Registry(projects=projects)


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
# GET /projects/add
# ---------------------------------------------------------------------------


class TestAddProjectGet:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/projects/add")
        assert response.status_code == 200

    def test_returns_html(self, client: TestClient) -> None:
        response = client.get("/projects/add")
        assert "text/html" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /projects/add
# ---------------------------------------------------------------------------


class TestAddProjectPost:
    def test_no_url_returns_400(self, client: TestClient) -> None:
        response = client.post("/projects/add", data={})
        assert response.status_code == 400

    def test_successful_fetch_redirects_to_root(self, client: TestClient) -> None:
        with patch("vocal.web.api.fetch_project"):
            response = client.post(
                "/projects/add",
                data={"url": "https://example.com/project"},
                follow_redirects=False,
            )
        assert response.status_code == 302
        assert response.headers["location"] == "/"

    def test_fetch_failure_rerenders_form_with_error(self, client: TestClient) -> None:
        with patch(
            "vocal.web.api.fetch_project",
            side_effect=FetchError("unreachable host", hint="check your network"),
        ):
            response = client.post(
                "/projects/add",
                data={"url": "https://example.com/bad"},
            )
        assert response.status_code == 422
        assert "unreachable host" in response.text
        assert "check your network" in response.text
        # URL is preserved in the form so the user doesn't have to retype it.
        assert "https://example.com/bad" in response.text

    def test_unknown_failure_renders_error_page(self, client: TestClient) -> None:
        client_no_raise = TestClient(app, raise_server_exceptions=False)
        with patch(
            "vocal.web.api.fetch_project", side_effect=RuntimeError("kaboom")
        ):
            response = client_no_raise.post(
                "/projects/add",
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
# POST /
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

    def test_no_conventions_renders_error_page(self, client: TestClient) -> None:
        with patch(
            "vocal.web.api.check_upload",
            side_effect=NoConventionsFound("No conventions in file"),
        ):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 422
        assert "We couldn't check that file" in response.text
        assert "No conventions in file" in response.text

    def test_no_matching_projects_uses_422(self, client: TestClient) -> None:
        with patch(
            "vocal.web.api.check_upload",
            side_effect=NoMatchingProjects("No registered project for conventions X"),
        ):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 422

    def test_inline_errors_render_incomplete_banner(self, client: TestClient) -> None:
        context = CheckContext()
        context.errors.append(
            CheckIssue(
                message="No product definitions registered for FAAM_standard version 2.1",
                hint="Register a project providing definitions for that version.",
            )
        )
        with patch("vocal.web.api.check_upload", return_value=context):
            response = client.post(
                "/",
                files={"file": ("test.nc", b"dummy", "application/octet-stream")},
            )
        assert response.status_code == 200
        assert "INCOMPLETE CHECK" in response.text
        assert "FAAM_standard version 2.1" in response.text
        # message and hint render as separate elements
        assert "Register a project providing definitions" in response.text

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
