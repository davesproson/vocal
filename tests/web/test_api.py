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

from vocal.utils.registry import Project, ProjectSpec, Registry
from vocal.web.api import app
from vocal.web.models import CheckContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _open_registry(projects: dict[str, Project]) -> Generator[Registry, None, None]:
    """Yields a Registry pre-populated with *projects* — used as a drop-in
    replacement for Registry.open() in tests."""
    yield Registry(projects=projects)


def _make_project(name: str = "test_project") -> Project:
    spec = ProjectSpec(name=name, has_major=True, has_minor=True, regex=".*")
    return Project(
        spec=spec,
        path=f"/tmp/{name}",
        definitions=f"/tmp/{name}/definitions",
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

    def test_fetch_failure_returns_400(self, client: TestClient) -> None:
        with patch(
            "vocal.web.api.fetch_project", side_effect=RuntimeError("unreachable host")
        ):
            response = client.post(
                "/projects/add",
                data={"url": "https://example.com/bad"},
            )
        assert response.status_code == 400


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
