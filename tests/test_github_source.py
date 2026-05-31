"""Unit tests for vocal/application/github_source.py.

The module is the single boundary for GitHub source acquisition. Its two
external dependencies — the GitHub HTTP API (``requests.get``) and the git CLI
(``subprocess.run``) — are faked, so the public :func:`materialize_repo`
interface is exercised end-to-end without a network or a real clone:

- the release path is driven by a fake ``requests.get`` that serves a release
  listing and then a real (in-memory) zipball, asserting the flattened tree
  lands at ``target``;
- the clone path is driven by a fake ``subprocess.run``;
- each typed error is asserted to surface through ``materialize_repo``.

The lower-level helpers (``get_latest_release``, ``fetch_with_git``,
``convert_github_repo_to_api_url``) are covered through the same public seam
here, complementing the direct helper tests in ``test_fetch.py``.
"""

import io
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from vocal.application.github_source import (
    FetchError,
    GitCloneFailed,
    GitHubAPIError,
    GitNotInstalled,
    NoReleasesFound,
    NotAGitHubRepo,
    RateLimited,
    RepoNotFound,
    materialize_repo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    status_code: int = 200, json_data: Any = None, text: str = "", content: bytes = b""
) -> MagicMock:
    """Build a fake ``requests.Response`` with controllable behaviour."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.text = text
    resp.content = content
    resp.json.return_value = json_data if json_data is not None else {}

    def _raise_for_status() -> None:
        if not resp.ok:
            raise requests.HTTPError(f"HTTP {status_code}")

    resp.raise_for_status.side_effect = _raise_for_status
    return resp


def _make_zipball(root: str, files: dict[str, str]) -> bytes:
    """Build a GitHub-style source zipball.

    Every entry is nested under a single top-level ``root`` directory, mirroring
    the ``user-repo-sha/`` wrapper GitHub's zipballs carry; ``materialize_repo``
    is expected to flatten that wrapper away.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, body in files.items():
            zf.writestr(f"{root}/{rel}", body)
    return buf.getvalue()


def _release_then_zip(zipball: bytes) -> Any:
    """A ``requests.get`` side effect: release listing first, then the zipball.

    The first call (``…/releases``) returns a one-entry release list; the second
    (the ``zipball_url``) returns the archive bytes.
    """
    releases = [{"tag_name": "v1.0", "zipball_url": "https://example/zip"}]
    responses = iter(
        [
            _make_response(200, json_data=releases),
            _make_response(200, content=zipball),
        ]
    )

    def _get(url: str, *args: Any, **kwargs: Any) -> MagicMock:
        return next(responses)

    return _get


# ---------------------------------------------------------------------------
# materialize_repo — release (default) path
# ---------------------------------------------------------------------------


class TestMaterializeRepoRelease:
    def test_downloads_and_flattens_latest_release(self, tmp_path: Path) -> None:
        zipball = _make_zipball(
            "owner-repo-abc123",
            {"conventions.yaml": "x: 1\n", "pkg/__init__.py": "v = 1\n"},
        )
        target = str(tmp_path / "repo")
        with patch("requests.get", side_effect=_release_then_zip(zipball)):
            materialize_repo("https://github.com/owner/repo", target, git=False)

        # The GitHub wrapper directory is gone; contents sit directly at target.
        assert (Path(target) / "conventions.yaml").read_text() == "x: 1\n"
        assert (Path(target) / "pkg" / "__init__.py").read_text() == "v = 1\n"

    def test_non_github_url_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotAGitHubRepo):
            materialize_repo(
                "https://gitlab.com/u/r", str(tmp_path / "repo"), git=False
            )

    def test_repo_not_found_raises(self, tmp_path: Path) -> None:
        with patch("requests.get", return_value=_make_response(404)):
            with pytest.raises(RepoNotFound):
                materialize_repo(
                    "https://github.com/u/r", str(tmp_path / "repo"), git=False
                )

    def test_no_releases_raises_with_git_hint(self, tmp_path: Path) -> None:
        with patch("requests.get", return_value=_make_response(200, json_data=[])):
            with pytest.raises(NoReleasesFound) as exc_info:
                materialize_repo(
                    "https://github.com/u/r", str(tmp_path / "repo"), git=False
                )
        assert "--git" in (exc_info.value.hint or "")

    def test_rate_limited_raises(self, tmp_path: Path) -> None:
        resp = _make_response(403, text="API rate limit exceeded")
        with patch("requests.get", return_value=resp):
            with pytest.raises(RateLimited):
                materialize_repo(
                    "https://github.com/u/r", str(tmp_path / "repo"), git=False
                )

    def test_other_api_error_raises(self, tmp_path: Path) -> None:
        with patch("requests.get", return_value=_make_response(500, text="boom")):
            with pytest.raises(GitHubAPIError):
                materialize_repo(
                    "https://github.com/u/r", str(tmp_path / "repo"), git=False
                )

    def test_network_error_wrapped_as_fetch_error(self, tmp_path: Path) -> None:
        with patch("requests.get", side_effect=requests.ConnectionError("dns")):
            with pytest.raises(FetchError):
                materialize_repo(
                    "https://github.com/u/r", str(tmp_path / "repo"), git=False
                )


# ---------------------------------------------------------------------------
# materialize_repo — git clone path
# ---------------------------------------------------------------------------


class TestMaterializeRepoGit:
    def test_clone_invokes_git(self, tmp_path: Path) -> None:
        result = MagicMock(returncode=0, stderr="")
        target = str(tmp_path / "repo")
        with patch("subprocess.run", return_value=result) as run:
            materialize_repo("https://github.com/u/r", target, git=True)
        args = run.call_args[0][0]
        assert args[:2] == ["git", "clone"]
        assert args[2] == "https://github.com/u/r"
        assert args[3] == target

    def test_git_not_installed_raises(self, tmp_path: Path) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(GitNotInstalled):
                materialize_repo(
                    "https://github.com/u/r", str(tmp_path / "repo"), git=True
                )

    def test_clone_failure_carries_stderr(self, tmp_path: Path) -> None:
        result = MagicMock(returncode=128, stderr="fatal: repository not found\n")
        with patch("subprocess.run", return_value=result):
            with pytest.raises(GitCloneFailed) as exc_info:
                materialize_repo(
                    "https://github.com/u/r", str(tmp_path / "repo"), git=True
                )
        assert "fatal: repository not found" in exc_info.value.message

    def test_clone_does_not_touch_network(self, tmp_path: Path) -> None:
        result = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", return_value=result), patch(
            "requests.get", side_effect=AssertionError("git path must not call HTTP")
        ):
            materialize_repo(
                "https://github.com/u/r", str(tmp_path / "repo"), git=True
            )
