"""
Tests for vocal/application/fetch.py.

Strategy
--------
Each helper is tested in isolation. Network and subprocess calls are mocked
via ``unittest.mock.patch``; filesystem state uses ``tmp_path``. The full
``fetch_http`` end-to-end path (download + extract a real zip) is not
covered here — its individual pieces (``get_latest_release``, atomic
cleanup) are tested separately.
"""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vocal.application.fetch import (
    FetchError,
    GitCloneFailed,
    GitHubAPIError,
    GitNotInstalled,
    NoReleasesFound,
    NotAGitHubRepo,
    ProjectAlreadyFetched,
    ProjectNotFetched,
    RateLimited,
    RepoNotFound,
    convert_github_repo_to_api_url,
    derive_repo_name,
    fetch_project,
    fetch_with_git,
    get_latest_release,
)
from vocal.conventions_file import InvalidConventionsFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int, json_data: Any = None, text: str = "") -> MagicMock:
    """Build a fake ``requests.Response`` with controllable behaviour."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.text = text
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


def _write_conventions_yaml(root: Path, body: Any) -> None:
    """Drop a conventions.yaml at ``root`` with the given top-level body."""
    with open(root / "conventions.yaml", "w") as f:
        yaml.dump(body, f)


def _make_valid_project(root: Path) -> None:
    """Materialise a minimal but valid fetched-project tree under ``root``."""
    os.makedirs(root / "src", exist_ok=True)
    _write_conventions_yaml(
        root,
        {
            "conventions": {"name": "STD", "major": 1, "minor": 0},
            "layout": {"project_directory": "src"},
        },
    )


# ---------------------------------------------------------------------------
# derive_repo_name
# ---------------------------------------------------------------------------


class TestDeriveRepoName:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/user/repo", "repo"),
            ("https://github.com/user/repo/", "repo"),
            ("https://github.com/user/repo.git", "repo"),
            ("git@github.com:user/repo.git", "repo"),
            ("https://gitlab.example.com/group/sub/project", "project"),
        ],
    )
    def test_derive(self, url: str, expected: str) -> None:
        assert derive_repo_name(url) == expected


# ---------------------------------------------------------------------------
# convert_github_repo_to_api_url
# ---------------------------------------------------------------------------


class TestConvertGithubRepoToApiUrl:
    def test_basic(self) -> None:
        assert (
            convert_github_repo_to_api_url("https://github.com/u/r")
            == "https://api.github.com/repos/u/r"
        )

    def test_strips_trailing_slash(self) -> None:
        assert (
            convert_github_repo_to_api_url("https://github.com/u/r/")
            == "https://api.github.com/repos/u/r"
        )

    def test_strips_git_suffix(self) -> None:
        assert (
            convert_github_repo_to_api_url("https://github.com/u/r.git")
            == "https://api.github.com/repos/u/r"
        )

    def test_non_github_raises(self) -> None:
        with pytest.raises(NotAGitHubRepo) as exc_info:
            convert_github_repo_to_api_url("https://gitlab.com/u/r")
        assert exc_info.value.hint is not None
        assert "--git" in exc_info.value.hint


# ---------------------------------------------------------------------------
# get_latest_release
# ---------------------------------------------------------------------------


class TestGetLatestRelease:
    def test_404_raises_repo_not_found(self) -> None:
        with patch("requests.get", return_value=_make_response(404)):
            with pytest.raises(RepoNotFound):
                get_latest_release("https://api.github.com/repos/u/r")

    def test_403_with_rate_limit_body_raises_rate_limited(self) -> None:
        resp = _make_response(403, text="API rate limit exceeded for user")
        with patch("requests.get", return_value=resp):
            with pytest.raises(RateLimited) as exc_info:
                get_latest_release("https://api.github.com/repos/u/r")
        assert exc_info.value.hint is not None

    def test_403_without_rate_limit_body_raises_api_error(self) -> None:
        resp = _make_response(403, text="forbidden")
        with patch("requests.get", return_value=resp):
            with pytest.raises(GitHubAPIError):
                get_latest_release("https://api.github.com/repos/u/r")

    def test_other_non_2xx_raises_api_error(self) -> None:
        resp = _make_response(500, text="server fell over")
        with patch("requests.get", return_value=resp):
            with pytest.raises(GitHubAPIError) as exc_info:
                get_latest_release("https://api.github.com/repos/u/r")
        assert "500" in exc_info.value.message

    def test_empty_release_list_raises_no_releases_found(self) -> None:
        resp = _make_response(200, json_data=[])
        with patch("requests.get", return_value=resp):
            with pytest.raises(NoReleasesFound):
                get_latest_release("https://api.github.com/repos/u/r")

    def test_returns_first_release(self) -> None:
        release = {"tag_name": "v1.0", "zipball_url": "https://example/zip"}
        resp = _make_response(200, json_data=[release, {"tag_name": "v0.9"}])
        with patch("requests.get", return_value=resp):
            assert get_latest_release("https://api.github.com/repos/u/r") == release

    def test_network_error_wrapped_as_fetch_error(self) -> None:
        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("dns")):
            with pytest.raises(FetchError) as exc_info:
                get_latest_release("https://api.github.com/repos/u/r")
        assert "Network error" in exc_info.value.message


# ---------------------------------------------------------------------------
# fetch_with_git
# ---------------------------------------------------------------------------


class TestFetchWithGit:
    def test_git_not_installed(self, tmp_path: Path) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(GitNotInstalled) as exc_info:
                fetch_with_git("https://github.com/u/r", str(tmp_path / "r"))
        assert exc_info.value.hint is not None

    def test_clone_failure_carries_stderr(self, tmp_path: Path) -> None:
        result = MagicMock(returncode=128, stderr="fatal: repository not found\n")
        with patch("subprocess.run", return_value=result):
            with pytest.raises(GitCloneFailed) as exc_info:
                fetch_with_git("https://github.com/u/r", str(tmp_path / "r"))
        assert "fatal: repository not found" in exc_info.value.message

    def test_clone_failure_empty_stderr_fallback(self, tmp_path: Path) -> None:
        result = MagicMock(returncode=1, stderr="")
        with patch("subprocess.run", return_value=result):
            with pytest.raises(GitCloneFailed) as exc_info:
                fetch_with_git("https://github.com/u/r", str(tmp_path / "r"))
        assert "exited non-zero" in exc_info.value.message

    def test_success_returns_nothing(self, tmp_path: Path) -> None:
        result = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", return_value=result) as run:
            fetch_with_git("https://github.com/u/r", str(tmp_path / "r"))
        # Sanity-check the args we passed to git.
        call_args = run.call_args[0][0]
        assert call_args[:2] == ["git", "clone"]
        assert call_args[2] == "https://github.com/u/r"
        assert call_args[3].endswith("/r")


# ---------------------------------------------------------------------------
# fetch_project — pre-flight flag logic and atomic cleanup
# ---------------------------------------------------------------------------


class TestFetchProjectFlags:
    """Tests for the --update / --force pre-flight logic.

    The fetch path itself is patched out — these tests verify only that the
    pre-flight raises the correct exception or correctly deletes the existing
    target before calling into the download function.
    """

    def _patch_projects_dir(self, tmp_path: Path) -> Any:
        return patch(
            "vocal.application.fetch.get_projects_dir", return_value=str(tmp_path)
        )

    def _patch_download_and_register(self) -> Any:
        """Patch out both download helpers and the registry side-effect."""
        return patch.multiple(
            "vocal.application.fetch",
            fetch_with_git=MagicMock(),
            fetch_http=MagicMock(),
            register_project=MagicMock(),
        )

    def test_exists_no_flag_raises_already_fetched(self, tmp_path: Path) -> None:
        target = tmp_path / "r"
        _make_valid_project(target)

        with self._patch_projects_dir(tmp_path), self._patch_download_and_register():
            with pytest.raises(ProjectAlreadyFetched) as exc_info:
                fetch_project("https://github.com/u/r")
        assert exc_info.value.hint is not None
        assert "--update" in exc_info.value.hint
        # And the existing dir wasn't touched.
        assert target.exists()

    def test_missing_with_update_raises_not_fetched(self, tmp_path: Path) -> None:
        with self._patch_projects_dir(tmp_path), self._patch_download_and_register():
            with pytest.raises(ProjectNotFetched) as exc_info:
                fetch_project("https://github.com/u/r", update=True)
        assert "vocal fetch" in (exc_info.value.hint or "")

    def test_exists_with_update_deletes_before_download(self, tmp_path: Path) -> None:
        target = tmp_path / "r"
        _make_valid_project(target)
        marker = target / "marker.txt"
        marker.write_text("pre-existing")

        # The mocked fetch leaves a fresh, valid project in place.
        def fake_fetch(url: str, tgt: str) -> None:
            os.makedirs(tgt, exist_ok=True)
            _make_valid_project(Path(tgt))

        with self._patch_projects_dir(tmp_path), patch.multiple(
            "vocal.application.fetch",
            fetch_http=MagicMock(side_effect=fake_fetch),
            register_project=MagicMock(),
        ):
            fetch_project("https://github.com/u/r", update=True)

        # The pre-existing marker must be gone (proves the old dir was deleted).
        assert not marker.exists()
        assert target.exists()

    def test_exists_with_force_deletes_before_download(self, tmp_path: Path) -> None:
        target = tmp_path / "r"
        _make_valid_project(target)
        marker = target / "marker.txt"
        marker.write_text("pre-existing")

        def fake_fetch(url: str, tgt: str) -> None:
            os.makedirs(tgt, exist_ok=True)
            _make_valid_project(Path(tgt))

        with self._patch_projects_dir(tmp_path), patch.multiple(
            "vocal.application.fetch",
            fetch_http=MagicMock(side_effect=fake_fetch),
            register_project=MagicMock(),
        ):
            fetch_project("https://github.com/u/r", force=True)

        assert not marker.exists()
        assert target.exists()

    def test_missing_with_force_proceeds(self, tmp_path: Path) -> None:
        def fake_fetch(url: str, tgt: str) -> None:
            os.makedirs(tgt, exist_ok=True)
            _make_valid_project(Path(tgt))

        with self._patch_projects_dir(tmp_path), patch.multiple(
            "vocal.application.fetch",
            fetch_http=MagicMock(side_effect=fake_fetch),
            register_project=MagicMock(),
        ):
            fetch_project("https://github.com/u/r", force=True)

        assert (tmp_path / "r").exists()

    def test_git_flag_routes_to_git_helper(self, tmp_path: Path) -> None:
        def fake_fetch(url: str, tgt: str) -> None:
            os.makedirs(tgt, exist_ok=True)
            _make_valid_project(Path(tgt))

        git_mock = MagicMock(side_effect=fake_fetch)
        http_mock = MagicMock(side_effect=fake_fetch)
        with self._patch_projects_dir(tmp_path), patch.multiple(
            "vocal.application.fetch",
            fetch_with_git=git_mock,
            fetch_http=http_mock,
            register_project=MagicMock(),
        ):
            fetch_project("https://github.com/u/r", git=True)

        git_mock.assert_called_once()
        http_mock.assert_not_called()


class TestFetchProjectCleanup:
    """Atomic cleanup of the target dir on post-download failure."""

    def test_missing_conventions_cleans_up(self, tmp_path: Path) -> None:
        target = tmp_path / "r"

        def fake_fetch(url: str, tgt: str) -> None:
            # Materialise a directory that will fail validation (no
            # conventions.yaml).
            os.makedirs(tgt, exist_ok=True)

        with patch(
            "vocal.application.fetch.get_projects_dir", return_value=str(tmp_path)
        ), patch.multiple(
            "vocal.application.fetch",
            fetch_http=MagicMock(side_effect=fake_fetch),
            register_project=MagicMock(),
        ):
            with pytest.raises(InvalidConventionsFile):
                fetch_project("https://github.com/u/r")

        assert not target.exists()

    def test_register_failure_cleans_up(self, tmp_path: Path) -> None:
        from vocal.application.register import CannotRegisterProjectError

        target = tmp_path / "r"

        def fake_fetch(url: str, tgt: str) -> None:
            os.makedirs(tgt, exist_ok=True)
            _make_valid_project(Path(tgt))

        with patch(
            "vocal.application.fetch.get_projects_dir", return_value=str(tmp_path)
        ), patch.multiple(
            "vocal.application.fetch",
            fetch_http=MagicMock(side_effect=fake_fetch),
            register_project=MagicMock(
                side_effect=CannotRegisterProjectError("boom")
            ),
        ):
            with pytest.raises(FetchError) as exc_info:
                fetch_project("https://github.com/u/r")

        assert "Failed to register project" in exc_info.value.message
        assert not target.exists()


# ---------------------------------------------------------------------------
# FetchError formatting
# ---------------------------------------------------------------------------


class TestFetchErrorFormatting:
    def test_str_with_hint(self) -> None:
        e = FetchError("the thing broke", hint="try X")
        assert str(e) == "the thing broke\n  try X"

    def test_str_without_hint(self) -> None:
        e = FetchError("the thing broke")
        assert str(e) == "the thing broke"
