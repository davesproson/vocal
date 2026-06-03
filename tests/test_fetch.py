"""
Tests for vocal/application/fetch.py.

Strategy
--------
The network/subprocess helpers (``get_latest_release``, ``fetch_with_git``,
``convert_github_repo_to_api_url``) are tested in isolation with mocks.

``fetch_project`` is now a thin shell over the shared install primitive: it
downloads to a temporary location, then hands off to ``install_project``. Its
tests therefore patch only the download step and drive the real
``install_project`` against an isolated ``~/.vocal`` + in-memory registry,
asserting externally observable state — what lands under ``~/.vocal``, what the
registry records, and which errors are raised — rather than implementation
details. A dedicated test pins the convergence guarantee: ``fetch`` and
``register`` reach identical on-disk + registry state for an equivalent source.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vocal.application.fetch import (
    FetchError,
    FetchOutcome,
    MissingProjectURL,
    PackAlreadyFetched,
    ProjectAlreadyFetched,
    ProjectNotFetched,
    UnreadableNetCDF,
    fetch_for_file,
    fetch_pack,
    fetch_project,
)
from vocal.application.github_source import (
    GitCloneFailed,
    GitHubAPIError,
    GitNotInstalled,
    NoReleasesFound,
    NotAGitHubRepo,
    RateLimited,
    RepoNotFound,
    convert_github_repo_to_api_url,
    derive_repo_name,
    fetch_with_git,
    get_latest_release,
)
from vocal.application.register import register_project
from vocal.conventions_file import (
    ConventionsFile,
    InvalidConventionsFile,
    MissingProjectExport,
)
from vocal.utils.registry import Registry


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


def _materialise_project(
    root: Any,
    *,
    name: str = "STD",
    major: int = 1,
    minor: int = 0,
    module: str = "stdmod",
    filecodec: bool = True,
) -> None:
    """Materialise a minimal but importable, contract-satisfying project tree.

    The tree carries a ``conventions.yaml`` and a Python package exposing
    ``defaults``, ``models.Dataset``, and (unless ``filecodec`` is False)
    ``filecodec`` — everything ``install_project``'s validate step requires.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_conventions_yaml(
        root,
        {
            "conventions": {"name": name, "major": major, "minor": minor},
            "layout": {"project_directory": module},
        },
    )
    mod = root / module
    mod.mkdir(parents=True, exist_ok=True)
    init_lines = ["from . import defaults", "from . import models"]
    if filecodec:
        init_lines.append("filecodec = {}")
    (mod / "__init__.py").write_text("\n".join(init_lines) + "\n")
    (mod / "defaults.py").write_text(
        "default_global_attrs = {}\n"
        "default_group_attrs = {}\n"
        "default_variable_attrs = {}\n"
    )
    (mod / "models.py").write_text(
        "from pydantic import BaseModel\n\n\nclass Dataset(BaseModel):\n    pass\n"
    )


def _fake_download(**project_kwargs: Any) -> Any:
    """Return a ``materialize_repo`` stub that drops a valid project.

    Stands in for :func:`materialize_repo`: it materialises a project tree at the
    download ``target`` the real acquisition would have produced. The ``git``
    keyword is accepted (and ignored) so the stub matches the real signature
    ``materialize_repo(url, target, *, git)``.
    """

    def _dl(url: str, target: str, *, git: bool = False) -> None:
        _materialise_project(target, **project_kwargs)

    return _dl


def _snapshot(root: Path) -> dict[str, str]:
    """Return ``{relative_path: contents}`` for every file under ``root``."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_text()
    return out


@contextmanager
def _fetch_env(tmp_path: Path, captured: dict) -> Iterator[str]:
    """Isolate a fetch/register: in-memory registry + a tmp ``~/.vocal`` root.

    Both ``fetch`` and ``install_project`` (via ``register``) read and write the
    registry; all three name bindings are patched to share one in-memory
    ``Registry`` so the install gate sees prior state. ``install``'s
    ``cache_dir`` is redirected so the owned copy lands under the tmp root.
    Yields the vocal root.
    """
    registry = Registry(projects={})
    captured["registry"] = registry
    vocal_root = str(tmp_path / "vocalroot")
    with patch.multiple(
        "vocal.application.register",
        load_registry=lambda: captured["registry"],
        save_registry=lambda r: captured.__setitem__("registry", r),
    ), patch(
        "vocal.application.fetch.load_registry", lambda: captured["registry"]
    ), patch(
        "vocal.application.install.cache_dir", return_value=vocal_root
    ):
        yield vocal_root


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
# fetch_project — thin shell over the shared install primitive
# ---------------------------------------------------------------------------


class TestFetchProjectInstall:
    """fetch downloads to a temp location, then installs via install_project.

    The download is patched out; the real install_project runs against an
    isolated ``~/.vocal`` + in-memory registry.
    """

    def test_installs_owned_copy_under_vocal(self, tmp_path: Path) -> None:
        captured: dict = {}
        with _fetch_env(tmp_path, captured) as vocal_root, patch.multiple(
            "vocal.application.fetch",
            materialize_repo=MagicMock(side_effect=_fake_download(name="STD", major=1)),
        ):
            fetch_project("https://github.com/u/r")

        owned = Path(vocal_root) / "projects" / "STD-1"
        assert (owned / "conventions.yaml").is_file()
        assert (owned / "stdmod" / "__init__.py").is_file()
        # The registry records the owned copy, not a temp download location.
        registered = captured["registry"].projects["STD-1"]
        assert registered.local_path == str(owned)
        assert os.path.isabs(registered.local_path)

    def test_git_flag_forwarded_to_materialize_repo(self, tmp_path: Path) -> None:
        materialize = MagicMock(side_effect=_fake_download())
        captured: dict = {}
        with _fetch_env(tmp_path, captured), patch.multiple(
            "vocal.application.fetch",
            materialize_repo=materialize,
        ):
            fetch_project("https://github.com/u/r", git=True)

        materialize.assert_called_once()
        # fetch_project routes acquisition through the one materialize_repo seam,
        # forwarding --git so it picks the clone path (asserted in its own tests).
        assert materialize.call_args.kwargs["git"] is True


class TestFetchProjectGating:
    """The already-fetched / not-fetched / refresh gating, post-download."""

    def test_already_installed_reports_already_fetched_after_download(
        self, tmp_path: Path
    ) -> None:
        download = MagicMock(side_effect=_fake_download())
        captured: dict = {}
        with _fetch_env(tmp_path, captured), patch.multiple(
            "vocal.application.fetch", materialize_repo=download
        ):
            fetch_project("https://github.com/u/r")
            with pytest.raises(ProjectAlreadyFetched) as exc_info:
                fetch_project("https://github.com/u/r")

        assert "--update" in (exc_info.value.hint or "")
        # The gate runs *after* download: the redundant fetch still downloaded.
        assert download.call_count == 2

    def test_force_overwrites_existing_install(self, tmp_path: Path) -> None:
        captured: dict = {}
        with _fetch_env(tmp_path, captured) as vocal_root:
            with patch.multiple(
                "vocal.application.fetch",
                materialize_repo=MagicMock(side_effect=_fake_download(minor=0)),
            ):
                fetch_project("https://github.com/u/r")
            with patch.multiple(
                "vocal.application.fetch",
                materialize_repo=MagicMock(side_effect=_fake_download(minor=5)),
            ):
                fetch_project("https://github.com/u/r", force=True)

        assert captured["registry"].projects["STD-1"].minor == 5
        owned = Path(vocal_root) / "projects" / "STD-1"
        assert ConventionsFile.load(str(owned)).minor == 5

    def test_update_missing_raises_not_fetched(self, tmp_path: Path) -> None:
        captured: dict = {}
        with _fetch_env(tmp_path, captured), patch.multiple(
            "vocal.application.fetch", materialize_repo=MagicMock(side_effect=_fake_download())
        ):
            with pytest.raises(ProjectNotFetched) as exc_info:
                fetch_project("https://github.com/u/r", update=True)
        assert "vocal fetch" in (exc_info.value.hint or "")

    def test_update_refreshes_existing(self, tmp_path: Path) -> None:
        captured: dict = {}
        with _fetch_env(tmp_path, captured):
            with patch.multiple(
                "vocal.application.fetch",
                materialize_repo=MagicMock(side_effect=_fake_download(minor=0)),
            ):
                fetch_project("https://github.com/u/r")
            with patch.multiple(
                "vocal.application.fetch",
                materialize_repo=MagicMock(side_effect=_fake_download(minor=9)),
            ):
                fetch_project("https://github.com/u/r", update=True)

        assert captured["registry"].projects["STD-1"].minor == 9

    def test_update_changed_major_is_not_fetched(self, tmp_path: Path) -> None:
        """A re-fetch whose major changed is a different identity: cleanly
        "not currently fetched" under --update, not a special identity error."""
        captured: dict = {}
        with _fetch_env(tmp_path, captured):
            with patch.multiple(
                "vocal.application.fetch",
                materialize_repo=MagicMock(side_effect=_fake_download(major=1)),
            ):
                fetch_project("https://github.com/u/r")
            with patch.multiple(
                "vocal.application.fetch",
                materialize_repo=MagicMock(side_effect=_fake_download(major=2)),
            ):
                with pytest.raises(ProjectNotFetched):
                    fetch_project("https://github.com/u/r", update=True)

        # The original major-1 install is the only thing registered.
        assert "STD-1" in captured["registry"].projects
        assert "STD-2" not in captured["registry"].projects


class TestFetchProjectFailureSafety:
    """A broken download never destroys a good install or leaves drift."""

    def test_missing_conventions_installs_nothing(self, tmp_path: Path) -> None:
        def bad_download(url: str, target: str, *, git: bool = False) -> None:
            # A tree that is not a vocal project (no conventions.yaml).
            os.makedirs(target, exist_ok=True)

        captured: dict = {}
        with _fetch_env(tmp_path, captured) as vocal_root, patch.multiple(
            "vocal.application.fetch", materialize_repo=MagicMock(side_effect=bad_download)
        ):
            with pytest.raises(InvalidConventionsFile):
                fetch_project("https://github.com/u/r")

        assert captured["registry"].projects == {}
        assert not (Path(vocal_root) / "projects" / "STD-1").exists()

    def test_broken_force_reinstall_preserves_good_install(
        self, tmp_path: Path
    ) -> None:
        captured: dict = {}
        with _fetch_env(tmp_path, captured) as vocal_root:
            with patch.multiple(
                "vocal.application.fetch",
                materialize_repo=MagicMock(side_effect=_fake_download(filecodec=True)),
            ):
                fetch_project("https://github.com/u/r")
            owned = Path(vocal_root) / "projects" / "STD-1"
            before = _snapshot(owned)

            # Re-fetch the same identity but with a broken package.
            with patch.multiple(
                "vocal.application.fetch",
                materialize_repo=MagicMock(side_effect=_fake_download(filecodec=False)),
            ):
                with pytest.raises(MissingProjectExport):
                    fetch_project("https://github.com/u/r", force=True)

            # The good install survived byte-for-byte.
            assert _snapshot(owned) == before


class TestFetchRegisterConvergence:
    """fetch and register reach identical on-disk + registry state."""

    def test_equivalent_source_reaches_identical_state(self, tmp_path: Path) -> None:
        # register installs from a local source tree...
        source = tmp_path / "src_repo"
        _materialise_project(
            source, name="CONV", major=1, minor=2, module="convmod"
        )
        reg_captured: dict = {}
        with _fetch_env(tmp_path / "reg", reg_captured) as reg_root:
            register_project(str(source))

        # ...fetch installs from an equivalent downloaded tree.
        fetch_captured: dict = {}
        with _fetch_env(tmp_path / "fet", fetch_captured) as fet_root, patch.multiple(
            "vocal.application.fetch",
            materialize_repo=MagicMock(
                side_effect=_fake_download(
                    name="CONV", major=1, minor=2, module="convmod"
                )
            ),
        ):
            fetch_project("https://github.com/u/r")

        reg_owned = Path(reg_root) / "projects" / "CONV-1"
        fet_owned = Path(fet_root) / "projects" / "CONV-1"
        # Identical on-disk shape (relative to each owned root).
        assert _snapshot(reg_owned) == _snapshot(fet_owned)

        # Identical registry record (modulo the root each is installed under).
        reg_rec = reg_captured["registry"].projects["CONV-1"]
        fet_rec = fetch_captured["registry"].projects["CONV-1"]
        assert (
            reg_rec.name,
            reg_rec.major,
            reg_rec.minor,
            reg_rec.project_directory,
        ) == (
            fet_rec.name,
            fet_rec.major,
            fet_rec.minor,
            fet_rec.project_directory,
        )
        assert reg_rec.local_path == str(reg_owned)
        assert fet_rec.local_path == str(fet_owned)


# ---------------------------------------------------------------------------
# fetch_for_file — file-driven fetch (project + pack)
# ---------------------------------------------------------------------------


def _write_nc(
    path: Path,
    *,
    project_url: str | None = None,
    definitions_url: str | None = None,
    definitions_version: int | None = None,
) -> str:
    """Write a minimal netCDF file carrying the given vocal-managed attributes."""
    import netCDF4

    with netCDF4.Dataset(str(path), "w") as nc:
        if project_url is not None:
            nc.vocal_project_url = project_url
        if definitions_url is not None:
            nc.vocal_definitions_url = definitions_url
        if definitions_version is not None:
            nc.vocal_definitions_version = definitions_version
    return str(path)


class TestFetchForFile:
    def test_missing_project_url_raises(self, tmp_path: Path) -> None:
        nc = _write_nc(tmp_path / "f.nc")  # no vocal_project_url
        with patch("vocal.application.fetch.fetch_project") as fp:
            with pytest.raises(MissingProjectURL) as exc_info:
                fetch_for_file(nc)
        assert exc_info.value.hint is not None
        fp.assert_not_called()

    def test_unreadable_path_raises_typed_error(self, tmp_path: Path) -> None:
        bogus = str(tmp_path / "not-a-netcdf.nc")
        (tmp_path / "not-a-netcdf.nc").write_text("definitely not netCDF")
        with pytest.raises(UnreadableNetCDF) as exc_info:
            fetch_for_file(bogus)
        assert exc_info.value.hint is not None

    def test_project_url_present_fetches_project(self, tmp_path: Path) -> None:
        nc = _write_nc(tmp_path / "f.nc", project_url="https://host/std.git")
        with patch("vocal.application.fetch.fetch_project") as fp:
            outcomes = fetch_for_file(nc)

        fp.assert_called_once_with(
            "https://host/std.git", git=False, update=False, force=False
        )
        project = next(o for o in outcomes if o.role == "project")
        assert project == FetchOutcome("project", "https://host/std.git", "fetched")

    def test_no_definitions_url_records_none_declared_pack(self, tmp_path: Path) -> None:
        nc = _write_nc(tmp_path / "f.nc", project_url="https://host/std.git")
        with patch("vocal.application.fetch.fetch_project"):
            outcomes = fetch_for_file(nc)

        pack = next(o for o in outcomes if o.role == "pack")
        assert pack.outcome == "none-declared"

    def test_project_fetched_before_pack(self, tmp_path: Path) -> None:
        nc = _write_nc(tmp_path / "f.nc", project_url="https://host/std.git")
        with patch("vocal.application.fetch.fetch_project"):
            outcomes = fetch_for_file(nc)

        assert [o.role for o in outcomes] == ["project", "pack"]

    def test_flags_forwarded_to_project_fetch(self, tmp_path: Path) -> None:
        nc = _write_nc(tmp_path / "f.nc", project_url="https://host/std.git")
        with patch("vocal.application.fetch.fetch_project") as fp:
            fetch_for_file(nc, git=True, update=True, force=True)

        fp.assert_called_once_with(
            "https://host/std.git", git=True, update=True, force=True
        )

    def test_definitions_url_present_fetches_pack(self, tmp_path: Path) -> None:
        nc = _write_nc(
            tmp_path / "f.nc",
            project_url="https://host/std.git",
            definitions_url="https://host/pack.git",
        )
        with patch("vocal.application.fetch.fetch_project"):
            with patch("vocal.application.fetch.fetch_pack") as fpk:
                outcomes = fetch_for_file(nc)

        fpk.assert_called_once_with(
            "https://host/pack.git", git=False, update=False, force=False
        )
        pack = next(o for o in outcomes if o.role == "pack")
        assert pack == FetchOutcome("pack", "https://host/pack.git", "fetched")

    def test_project_fetched_before_pack_when_both_declared(
        self, tmp_path: Path
    ) -> None:
        nc = _write_nc(
            tmp_path / "f.nc",
            project_url="https://host/std.git",
            definitions_url="https://host/pack.git",
        )
        calls: list[str] = []
        with patch(
            "vocal.application.fetch.fetch_project",
            side_effect=lambda *a, **k: calls.append("project"),
        ):
            with patch(
                "vocal.application.fetch.fetch_pack",
                side_effect=lambda *a, **k: calls.append("pack"),
            ):
                outcomes = fetch_for_file(nc)

        assert calls == ["project", "pack"]
        assert [o.role for o in outcomes] == ["project", "pack"]

    def test_pack_fetch_failure_leaves_project_fetched(self, tmp_path: Path) -> None:
        nc = _write_nc(
            tmp_path / "f.nc",
            project_url="https://host/std.git",
            definitions_url="https://host/pack.git",
        )
        with patch("vocal.application.fetch.fetch_project") as fp:
            with patch(
                "vocal.application.fetch.fetch_pack",
                side_effect=FetchError("pack boom"),
            ):
                with pytest.raises(FetchError):
                    fetch_for_file(nc)

        # Project-first, fail-fast, no rollback: the project fetch ran (and on a
        # re-run the idempotent project install would resume from there).
        fp.assert_called_once()

    def test_flags_forwarded_to_pack_fetch(self, tmp_path: Path) -> None:
        nc = _write_nc(
            tmp_path / "f.nc",
            project_url="https://host/std.git",
            definitions_url="https://host/pack.git",
        )
        with patch("vocal.application.fetch.fetch_project"):
            with patch("vocal.application.fetch.fetch_pack") as fpk:
                fetch_for_file(nc, git=True, update=True, force=True)

        fpk.assert_called_once_with(
            "https://host/pack.git", git=True, update=True, force=True
        )

    def test_orphan_definitions_version_warns_and_fetches_project_only(
        self, tmp_path: Path
    ) -> None:
        nc = _write_nc(
            tmp_path / "f.nc",
            project_url="https://host/std.git",
            definitions_version=3,  # no definitions_url
        )
        with patch("vocal.application.fetch.fetch_project") as fp:
            with patch("vocal.application.fetch.fetch_pack") as fpk:
                with pytest.warns(UserWarning, match="orphaned"):
                    outcomes = fetch_for_file(nc)

        fp.assert_called_once()
        fpk.assert_not_called()
        pack = next(o for o in outcomes if o.role == "pack")
        assert pack.outcome == "none-declared"

    def test_already_fetched_project_records_already_present(
        self, tmp_path: Path
    ) -> None:
        nc = _write_nc(tmp_path / "f.nc", project_url="https://host/std.git")
        with patch(
            "vocal.application.fetch.fetch_project",
            side_effect=ProjectAlreadyFetched("already there"),
        ):
            outcomes = fetch_for_file(nc)

        project = next(o for o in outcomes if o.role == "project")
        assert project == FetchOutcome(
            "project", "https://host/std.git", "already-present"
        )

    def test_already_fetched_pack_records_already_present(
        self, tmp_path: Path
    ) -> None:
        nc = _write_nc(
            tmp_path / "f.nc",
            project_url="https://host/std.git",
            definitions_url="https://host/pack.git",
        )
        with patch("vocal.application.fetch.fetch_project"):
            with patch(
                "vocal.application.fetch.fetch_pack",
                side_effect=PackAlreadyFetched("already there"),
            ):
                outcomes = fetch_for_file(nc)

        pack = next(o for o in outcomes if o.role == "pack")
        assert pack == FetchOutcome(
            "pack", "https://host/pack.git", "already-present"
        )

    def test_already_present_project_still_fetches_missing_pack(
        self, tmp_path: Path
    ) -> None:
        # Re-run safety: an already-present project does not stop the still-missing
        # pack from being fetched.
        nc = _write_nc(
            tmp_path / "f.nc",
            project_url="https://host/std.git",
            definitions_url="https://host/pack.git",
        )
        with patch(
            "vocal.application.fetch.fetch_project",
            side_effect=ProjectAlreadyFetched("already there"),
        ):
            with patch("vocal.application.fetch.fetch_pack") as fpk:
                outcomes = fetch_for_file(nc)

        fpk.assert_called_once()
        project = next(o for o in outcomes if o.role == "project")
        pack = next(o for o in outcomes if o.role == "pack")
        assert project.outcome == "already-present"
        assert pack.outcome == "fetched"


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
