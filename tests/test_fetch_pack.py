"""Integration tests for GitHub-hosted pack fetching and kind dispatch.

Packs are now multi-version GitHub repositories: ``vocal fetch <repo-url>``
acquires the whole tree (latest-release zipball or ``--git`` clone), classifies
it by inspecting the downloaded tree, and registers *every* ``v{Y}`` release it
contains. These tests fake acquisition by patching
``vocal.application.fetch.materialize_repo`` to drop a pack tree at the download
target (mirroring ``test_fetch.py``'s project download stub), and redirect
registry I/O to an in-memory registry so the registered records are inspectable.

The two pure deep modules introduced by this slice — ``classify_resource`` and
``discover_pack_versions`` — are unit-tested directly over temporary trees, with
no network.
"""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator
from unittest.mock import MagicMock, patch

import pytest

from vocal.application.fetch import (
    PackAlreadyFetched,
    PackNotFetched,
    derive_url_slug,
    fetch,
    fetch_pack,
)
from vocal.application.register import register_pack
from vocal.application.resource import (
    NotAVocalResource,
    ResourceKind,
    classify_resource,
    discover_pack_versions,
)
from vocal.manifest import PackInconsistent
from vocal.application.github_source import NoReleasesFound
from vocal.utils.registry import Registry


PACK_URL = "https://github.com/u/packrepo"


# ---------------------------------------------------------------------------
# Building hostable / downloadable pack trees
# ---------------------------------------------------------------------------


def _write_version_dir(
    root: Path, version: int, url: str, manifest_version: int | None = None
) -> Path:
    """Write a single ``v{version}/`` release directory under ``root``."""
    if manifest_version is None:
        manifest_version = version
    vdir = root / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "version": manifest_version,
        "url": url,
        "requires_standard": {"name": "MYSTD", "major": 2, "min_minor": 3},
        "products": [
            {"name": "alpha", "file_pattern": "alpha_{date}.nc", "schema": "alpha.json"}
        ],
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest))
    (vdir / "dataset_schema.json").write_text(json.dumps({"type": "object"}))
    (vdir / "alpha.json").write_text(json.dumps({"meta": {"file_pattern": "alpha"}}))
    return vdir


def _materialise_pack(
    root: Any,
    *,
    versions: Iterable[int] = (3,),
    url: str = PACK_URL,
    with_latest: bool = True,
    manifest_version_overrides: dict[int, int] | None = None,
) -> Path:
    """Materialise a multi-version pack repository tree at ``root``.

    Writes a ``v{Y}/`` directory per version plus, unless suppressed, a
    byte-identical ``latest/`` copy of the highest version — the marker tree
    ``classify_resource`` keys a pack on. ``manifest_version_overrides`` lets a
    single ``v{Y}`` declare a mismatching manifest version, to fabricate a
    hosting bug.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    overrides = manifest_version_overrides or {}
    versions = list(versions)
    for version in versions:
        _write_version_dir(root, version, url, overrides.get(version))

    if with_latest:
        highest = max(versions)
        latest = root / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(root / f"v{highest}", latest)

    return root


def _fake_download(**pack_kwargs: Any) -> Any:
    """Return a ``materialize_repo`` stub that drops a pack tree at ``target``."""

    def _dl(url: str, target: str, *, git: bool = False) -> None:
        _materialise_pack(target, **pack_kwargs)

    return _dl


def _snapshot(root: Path) -> dict[str, str]:
    """Return ``{relative_path: contents}`` for every file under ``root``."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_text()
    return out


@contextmanager
def _pack_env(cache_root: Path, captured: dict) -> Iterator[str]:
    """Isolate an install: in-memory registry + a tmp ``~/.vocal`` root.

    Both ``_install_pack_tree`` (the gate) and ``install_pack`` read the
    registry, so both name bindings are patched to share one in-memory
    ``Registry``; ``install``'s ``cache_dir`` is redirected so the owned copy
    lands under ``cache_root``. Yields the vocal root.
    """
    captured["registry"] = Registry()

    def _load() -> Registry:
        return captured["registry"]

    def _save(r: Registry) -> None:
        captured["registry"] = r

    with patch.multiple(
        "vocal.application.register",
        load_registry=_load,
        save_registry=_save,
    ), patch(
        "vocal.application.fetch.load_registry", _load
    ), patch(
        "vocal.application.install.cache_dir", return_value=str(cache_root)
    ):
        yield str(cache_root)


@pytest.fixture
def registers_into():
    """Redirect install's and fetch's registry I/O to one in-memory registry."""
    captured: dict = {"registry": Registry()}

    def _load():
        return captured["registry"]

    def _save(r: Registry) -> None:
        captured["registry"] = r

    with patch.multiple(
        "vocal.application.register",
        load_registry=_load,
        save_registry=_save,
    ), patch("vocal.application.fetch.load_registry", _load):
        yield captured


@contextmanager
def _patch_cache(tmp_path: Path) -> Iterator[None]:
    """Redirect the install root so owned copies land under ``tmp/cache``."""
    with patch(
        "vocal.application.install.cache_dir",
        return_value=str(tmp_path / "cache"),
    ):
        yield


# ---------------------------------------------------------------------------
# derive_url_slug
# ---------------------------------------------------------------------------


class TestDeriveURLSlug:
    def test_slug_is_filesystem_safe(self) -> None:
        assert derive_url_slug("https://github.com/u/packrepo") == "github-com-u-packrepo"

    def test_slug_normalises_case_and_trailing_slash(self) -> None:
        assert derive_url_slug("https://GitHub.com/U/Repo/") == "github-com-u-repo"


# ---------------------------------------------------------------------------
# classify_resource — pure, no network
# ---------------------------------------------------------------------------


class TestClassifyResource:
    def test_conventions_yaml_is_project(self, tmp_path: Path) -> None:
        (tmp_path / "conventions.yaml").write_text("conventions: {}\n")
        assert classify_resource(str(tmp_path)) is ResourceKind.PROJECT

    def test_latest_manifest_is_pack(self, tmp_path: Path) -> None:
        latest = tmp_path / "latest"
        latest.mkdir()
        (latest / "manifest.json").write_text("{}")
        assert classify_resource(str(tmp_path)) is ResourceKind.PACK

    def test_neither_raises_not_a_vocal_resource(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("hello")
        with pytest.raises(NotAVocalResource) as exc_info:
            classify_resource(str(tmp_path))
        assert exc_info.value.hint is not None

    def test_project_marker_wins_over_pack_marker(self, tmp_path: Path) -> None:
        (tmp_path / "conventions.yaml").write_text("conventions: {}\n")
        latest = tmp_path / "latest"
        latest.mkdir()
        (latest / "manifest.json").write_text("{}")
        assert classify_resource(str(tmp_path)) is ResourceKind.PROJECT


# ---------------------------------------------------------------------------
# discover_pack_versions — pure, no network
# ---------------------------------------------------------------------------


class TestDiscoverPackVersions:
    def test_enumerates_versions_excluding_latest(self, tmp_path: Path) -> None:
        _materialise_pack(tmp_path, versions=[1, 4, 2])
        discovered = discover_pack_versions(str(tmp_path))
        versions = [v for v, _ in discovered]
        assert versions == [1, 2, 4]  # sorted ascending, latest/ excluded
        # Every returned path is the matching v{Y} directory.
        for version, path in discovered:
            assert Path(path).name == f"v{version}"

    def test_empty_tree_yields_nothing(self, tmp_path: Path) -> None:
        assert discover_pack_versions(str(tmp_path)) == []

    def test_ignores_non_versioned_entries_and_files(self, tmp_path: Path) -> None:
        _materialise_pack(tmp_path, versions=[2])
        (tmp_path / "README.md").write_text("x")
        (tmp_path / "vetc").mkdir()  # not a v{Y} directory
        assert [v for v, _ in discover_pack_versions(str(tmp_path))] == [2]


# ---------------------------------------------------------------------------
# Kind dispatch — fetch classifies the downloaded tree
# ---------------------------------------------------------------------------


class TestFetchDispatch:
    def test_pack_tree_registers_all_versions(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        download = MagicMock(side_effect=_fake_download(versions=[2, 3]))
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo", download
        ):
            kind = fetch(PACK_URL)

        registry = registers_into["registry"]
        assert registry.find_pack(PACK_URL, 2) is not None
        assert registry.find_pack(PACK_URL, 3) is not None
        # fetch surfaces the kind it installed, threaded back from the dispatch.
        assert kind is ResourceKind.PACK
        # Acquisition happened exactly once — no pre-download probe.
        download.assert_called_once()

    def test_project_tree_routes_to_project_install(self, tmp_path: Path) -> None:
        def _drop_project(url: str, target: str, *, git: bool = False) -> None:
            Path(target).mkdir(parents=True, exist_ok=True)
            (Path(target) / "conventions.yaml").write_text("conventions: {}\n")

        with patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_drop_project),
        ), patch(
            "vocal.application.fetch._install_project_tree",
            return_value=ResourceKind.PROJECT,
        ) as project, patch(
            "vocal.application.fetch._install_pack_tree"
        ) as pack:
            kind = fetch(PACK_URL)
        project.assert_called_once()
        pack.assert_not_called()
        # fetch threads the project kind back from the dispatch branch.
        assert kind is ResourceKind.PROJECT

    def test_neither_raises_not_a_vocal_resource(self, tmp_path: Path) -> None:
        def _drop_junk(url: str, target: str, *, git: bool = False) -> None:
            Path(target).mkdir(parents=True, exist_ok=True)
            (Path(target) / "README.md").write_text("hi")

        with patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_drop_junk),
        ):
            with pytest.raises(NotAVocalResource):
                fetch(PACK_URL)


# ---------------------------------------------------------------------------
# fetch_pack — register all versions, cache layout, default gating
# ---------------------------------------------------------------------------


class TestFetchPack:
    def test_fetch_registers_every_version(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        packs_dir = tmp_path / "cache" / "packs"
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_fake_download(versions=[2, 3])),
        ):
            fetch_pack(PACK_URL)

        slug = derive_url_slug(PACK_URL)
        for version in (2, 3):
            target = packs_dir / slug / f"v{version}"
            assert (target / "manifest.json").is_file()
            assert (target / "dataset_schema.json").is_file()
            assert (target / "alpha.json").is_file()
            pack = registers_into["registry"].find_pack(PACK_URL, version)
            assert pack is not None
            assert pack.local_path == str(target)

    def test_latest_is_not_registered(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        packs_dir = tmp_path / "cache" / "packs"
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_fake_download(versions=[3])),
        ):
            fetch_pack(PACK_URL)

        slug = derive_url_slug(PACK_URL)
        # Only the v{Y} directory is installed; no "latest" entry on disk or in
        # the registry.
        assert not (packs_dir / slug / "latest").exists()
        assert all(version != "latest" for _, version in registers_into["registry"].packs)

    def test_git_clone_registers_all_versions(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        materialize = MagicMock(side_effect=_fake_download(versions=[1, 5]))
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo", materialize
        ):
            fetch_pack(PACK_URL, git=True)

        assert materialize.call_args.kwargs["git"] is True
        registry = registers_into["registry"]
        assert registry.find_pack(PACK_URL, 1) is not None
        assert registry.find_pack(PACK_URL, 5) is not None

    def test_refetch_default_raises_already_fetched(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_fake_download(versions=[3])),
        ):
            fetch_pack(PACK_URL)
            with pytest.raises(PackAlreadyFetched) as exc_info:
                fetch_pack(PACK_URL)

        hint = exc_info.value.hint or ""
        assert "--update" in hint and "--force" in hint

    def test_no_versioned_dirs_raises_fetch_error(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        def _drop_only_latest(url: str, target: str, *, git: bool = False) -> None:
            root = Path(target)
            (root / "latest").mkdir(parents=True, exist_ok=True)
            (root / "latest" / "manifest.json").write_text("{}")

        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_drop_only_latest),
        ):
            with pytest.raises(Exception) as exc_info:
                fetch_pack(PACK_URL)
        assert "no versioned" in str(exc_info.value).lower()
        assert registers_into["registry"].packs == {}

    def test_inconsistent_version_dir_raises(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        # v9/ on disk but its manifest declares version 3 — a hosting bug that
        # install_pack must surface via load_manifest.
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(
                side_effect=_fake_download(
                    versions=[9], with_latest=False,
                    manifest_version_overrides={9: 3},
                )
            ),
        ):
            with pytest.raises(PackInconsistent):
                fetch_pack(PACK_URL)


class TestFetchPackUpdateForce:
    """Per-URL ``--update`` / ``--force`` gating and additive update semantics."""

    def test_update_on_unregistered_url_fails(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_fake_download(versions=[3])),
        ):
            with pytest.raises(PackNotFetched) as exc_info:
                fetch_pack(PACK_URL, update=True)

        # Nothing registered, and the hint points at a first-time fetch.
        assert registers_into["registry"].packs == {}
        assert PACK_URL in (exc_info.value.hint or "")

    def test_update_picks_up_new_version(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        registry = registers_into["registry"]
        with _patch_cache(tmp_path):
            with patch(
                "vocal.application.fetch.materialize_repo",
                MagicMock(side_effect=_fake_download(versions=[2])),
            ):
                fetch_pack(PACK_URL)
            # A new release adds v3 alongside v2.
            with patch(
                "vocal.application.fetch.materialize_repo",
                MagicMock(side_effect=_fake_download(versions=[2, 3])),
            ):
                fetch_pack(PACK_URL, update=True)

        assert registry.find_pack(PACK_URL, 2) is not None
        assert registry.find_pack(PACK_URL, 3) is not None

    def test_update_refreshes_existing_version(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        slug = derive_url_slug(PACK_URL)
        installed = tmp_path / "cache" / "packs" / slug / "v3" / "alpha.json"
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_fake_download(versions=[3])),
        ):
            fetch_pack(PACK_URL)
            canonical = installed.read_text()
            # Corrupt the owned copy, then update to refresh it back in place.
            installed.write_text("TAMPERED")
            fetch_pack(PACK_URL, update=True)

        assert installed.read_text() == canonical

    def test_update_is_additive_and_never_prunes(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        registry = registers_into["registry"]
        with _patch_cache(tmp_path):
            with patch(
                "vocal.application.fetch.materialize_repo",
                MagicMock(side_effect=_fake_download(versions=[2, 3])),
            ):
                fetch_pack(PACK_URL)
            # The latest release no longer ships v2; update must not prune it.
            with patch(
                "vocal.application.fetch.materialize_repo",
                MagicMock(side_effect=_fake_download(versions=[3])),
            ):
                fetch_pack(PACK_URL, update=True)

        assert registry.find_pack(PACK_URL, 2) is not None
        assert registry.find_pack(PACK_URL, 3) is not None

    def test_force_reinstalls_regardless_of_registration(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        slug = derive_url_slug(PACK_URL)
        installed = tmp_path / "cache" / "packs" / slug / "v3" / "alpha.json"
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_fake_download(versions=[3])),
        ):
            fetch_pack(PACK_URL)
            canonical = installed.read_text()
            installed.write_text("TAMPERED")
            # --force re-installs every version with no gating error.
            fetch_pack(PACK_URL, force=True)

        assert installed.read_text() == canonical
        assert registers_into["registry"].find_pack(PACK_URL, 3) is not None

    def test_force_on_unregistered_url_installs(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        # --force does not require prior registration (unlike --update).
        with _patch_cache(tmp_path), patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_fake_download(versions=[4])),
        ):
            fetch_pack(PACK_URL, force=True)

        assert registers_into["registry"].find_pack(PACK_URL, 4) is not None


class TestFetchNoReleases:
    """A pack repo with no GitHub releases surfaces NoReleasesFound with --git."""

    def test_no_releases_hints_at_git(self, tmp_path: Path) -> None:
        # The release lookup in materialize_repo returns 200 with an empty list.
        resp = MagicMock(status_code=200, ok=True, text="[]")
        resp.json.return_value = []
        with patch("requests.get", return_value=resp):
            with pytest.raises(NoReleasesFound) as exc_info:
                fetch(PACK_URL)
        assert "--git" in (exc_info.value.hint or "")


# ---------------------------------------------------------------------------
# fetch / register convergence
# ---------------------------------------------------------------------------


class TestFetchRegisterConvergence:
    """fetch and register reach identical on-disk + registry state per version."""

    def test_equivalent_source_reaches_identical_state(self, tmp_path: Path) -> None:
        # register installs from a local v{Y} release directory...
        source_root = _materialise_pack(
            tmp_path / "src", versions=[3], with_latest=False
        )
        reg_captured: dict = {}
        with _pack_env(tmp_path / "regcache", reg_captured) as reg_root:
            register_pack(str(source_root / "v3"))

        # ...fetch installs from an equivalent downloaded repository tree.
        fet_captured: dict = {}
        with _pack_env(tmp_path / "fetcache", fet_captured) as fet_root, patch(
            "vocal.application.fetch.materialize_repo",
            MagicMock(side_effect=_fake_download(versions=[3])),
        ):
            fetch_pack(PACK_URL)

        slug = derive_url_slug(PACK_URL)
        reg_owned = Path(reg_root) / "packs" / slug / "v3"
        fet_owned = Path(fet_root) / "packs" / slug / "v3"
        assert _snapshot(reg_owned) == _snapshot(fet_owned)

        reg_pack = reg_captured["registry"].find_pack(PACK_URL, 3)
        fet_pack = fet_captured["registry"].find_pack(PACK_URL, 3)
        assert reg_pack is not None and fet_pack is not None
        assert reg_pack.manifest.to_dict() == fet_pack.manifest.to_dict()
        assert reg_pack.local_path == str(reg_owned)
        assert fet_pack.local_path == str(fet_owned)
