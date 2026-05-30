"""Tests for vocal/application/register.py and the init -> register flow.

These cover the new conventions.yaml-driven registration flow: identity and
layout are read from conventions.yaml, the project package is imported via the
single import path, and the project contract is enforced before a project is
added to the registry.
"""

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from vocal.application.init import init_project
from vocal.application.register import (
    CannotRegisterPackError,
    CannotRegisterProjectError,
    UnknownResourceKind,
    register_pack,
    register_project,
    register_resource,
)
from vocal.conventions_file import ConventionsFile, MissingProjectExport
from vocal.manifest import InvalidManifest, PackInconsistent
from vocal.utils.registry import Registry, project_key


def _fill_in_module(repo: Path, module_name: str, *, filecodec: bool = True) -> None:
    """Simulate the maintainer filling in a valid (importable) project module.

    ``init`` scaffolds a skeleton; here we overwrite it with a minimal module
    that satisfies the project contract.
    """
    mod = repo / module_name
    if mod.exists():
        shutil.rmtree(mod)
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


@contextmanager
def _install_env(tmp_path: Path, captured: dict) -> Iterator[str]:
    """Isolate an install: in-memory registry + a tmp ``~/.vocal`` root.

    Yields the vocal root so tests can assert on the owned copy that
    ``install_project`` writes under ``<root>/projects/{name}-{major}``. The
    registry object is shared across calls within the ``with`` block (mutated in
    place by ``add_project``), so re-install / ``--force`` scenarios see prior
    state.
    """
    registry = Registry(projects={})
    captured["registry"] = registry
    vocal_root = str(tmp_path / "vocalroot")
    with patch.multiple(
        "vocal.application.register",
        load_registry=lambda: registry,
        save_registry=lambda r: captured.__setitem__("registry", r),
    ), patch("vocal.application.install.cache_dir", return_value=vocal_root):
        yield vocal_root


def _snapshot(root: Path) -> dict[str, str]:
    """Return ``{relative_path: contents}`` for every file under ``root``."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_text()
    return out


class TestInitThenRegister:
    def test_scaffold_then_fill_in_registers(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="MYSTD", major=2, minor=3, project_directory="mystd"
        )

        # init writes conventions.yaml with the supplied identity and layout.
        cf = ConventionsFile.load(str(repo))
        assert cf.name == "MYSTD"
        assert cf.major == 2
        assert cf.minor == 3
        assert cf.project_directory == "mystd"

        # The maintainer fills in a valid module.
        _fill_in_module(repo, "mystd")

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_project(str(repo))

        registry = captured["registry"]
        assert "MYSTD-2" in registry.projects
        registered = registry.projects["MYSTD-2"]
        # local_path is the owned copy under ~/.vocal, not the source repo.
        owned = os.path.join(vocal_root, "projects", "MYSTD-2")
        assert registered.local_path == owned
        assert registered.local_path != str(repo)
        assert registered.minor == 3
        assert registered.project_directory == "mystd"
        # The project was actually copied into the owned location.
        assert (Path(owned) / "conventions.yaml").is_file()
        assert (Path(owned) / "mystd" / "__init__.py").is_file()

    def test_register_missing_export_raises_named(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="MYSTD", major=1, minor=0, project_directory="mystd"
        )
        # Fill in a module missing filecodec.
        _fill_in_module(repo, "mystd", filecodec=False)

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            with pytest.raises(MissingProjectExport) as exc:
                register_project(str(repo))

        assert "filecodec" in exc.value.message
        # Nothing was registered, and no owned copy was left behind.
        assert captured["registry"].projects == {}
        assert not (Path(vocal_root) / "projects" / "MYSTD-1").exists()


class TestInstallProject:
    """Integration coverage for ``register`` installing an owned copy.

    Driven through the public ``register_project`` entry point; assertions are
    on externally observable state — what lands under ``~/.vocal`` and what the
    registry records — not on private helpers.
    """

    def test_copies_owned_copy_and_applies_denylist(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="MYSTD", major=2, minor=3, project_directory="mystd"
        )
        _fill_in_module(repo, "mystd")
        # Cruft the install must normalise away, plus a data file it must keep.
        (repo / ".git").mkdir()
        (repo / ".git" / "config").write_text("x")
        (repo / "tests").mkdir()
        (repo / "tests" / "test_x.py").write_text("x")
        (repo / "mystd" / "__pycache__").mkdir()
        (repo / "mystd" / "__pycache__" / "m.pyc").write_text("x")
        (repo / "mystd" / "data.csv").write_text("rows")

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_project(str(repo))

        owned = Path(vocal_root) / "projects" / "MYSTD-2"
        snap = _snapshot(owned)
        # Denylisted entries are gone at every level...
        assert ".git/config" not in snap
        assert "tests/test_x.py" not in snap
        assert "mystd/__pycache__/m.pyc" not in snap
        # ...while the runtime-relevant files (module + data) survive.
        assert snap["conventions.yaml"]
        assert snap["mystd/__init__.py"]
        assert snap["mystd/data.csv"] == "rows"

    def test_relative_source_resolves_location_independently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="REL", major=1, minor=0, project_directory="relmod"
        )
        _fill_in_module(repo, "relmod")

        # Register via a *relative* path from tmp_path.
        monkeypatch.chdir(tmp_path)
        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_project("myrepo")

        registered = captured["registry"].projects["REL-1"]
        owned = os.path.join(vocal_root, "projects", "REL-1")
        # The stored path is the absolute owned copy, not the relative input.
        assert registered.local_path == owned
        assert os.path.isabs(registered.local_path)
        assert (Path(owned) / "relmod" / "__init__.py").is_file()

    def test_duplicate_without_force_raises_and_keeps_install(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="DUP", major=1, minor=0, project_directory="dupmod"
        )
        _fill_in_module(repo, "dupmod")

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_project(str(repo))
            before = _snapshot(Path(vocal_root) / "projects" / "DUP-1")
            with pytest.raises(CannotRegisterProjectError):
                register_project(str(repo))
            # The existing install is untouched.
            assert _snapshot(Path(vocal_root) / "projects" / "DUP-1") == before
        assert project_key("DUP", 1) in captured["registry"].projects

    def test_force_reinstalls_overwriting_copy_and_entry(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="FRC", major=1, minor=2, project_directory="frcmod"
        )
        _fill_in_module(repo, "frcmod")

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_project(str(repo))

            # Bump the source minor and re-register with --force.
            cf = ConventionsFile.load(str(repo))
            cf.minor = 7
            cf.write(str(repo))
            register_project(str(repo), force=True)

            registered = captured["registry"].projects["FRC-1"]
            assert registered.minor == 7
            owned = Path(vocal_root) / "projects" / "FRC-1"
            assert ConventionsFile.load(str(owned)).minor == 7

    def test_broken_force_reinstall_preserves_good_install(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="SAFE", major=1, minor=0, project_directory="safemod"
        )
        _fill_in_module(repo, "safemod")

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_project(str(repo))
            owned = Path(vocal_root) / "projects" / "SAFE-1"
            before = _snapshot(owned)

            # Break the module but keep the same identity, then force-reinstall.
            _fill_in_module(repo, "safemod", filecodec=False)
            with pytest.raises(MissingProjectExport):
                register_project(str(repo), force=True)

            # The good install survived byte-for-byte.
            assert _snapshot(owned) == before

    def test_leftover_dir_without_entry_is_overwritten(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="DRIFT", major=1, minor=0, project_directory="drmod"
        )
        _fill_in_module(repo, "drmod")

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            # Drift: a stale dir on disk with no matching registry entry.
            stale = Path(vocal_root) / "projects" / "DRIFT-1"
            stale.mkdir(parents=True)
            (stale / "stale.txt").write_text("garbage")

            register_project(str(repo))

            snap = _snapshot(stale)
            assert "stale.txt" not in snap
            assert snap["conventions.yaml"]


def _make_pack_dir(
    root: Path, dirname: str, version: int, manifest_version: int | None = None
) -> Path:
    """Materialise a pack release directory on disk and return its path."""
    if manifest_version is None:
        manifest_version = version
    vdir = root / dirname
    vdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "version": manifest_version,
        "url": "https://host/packs",
        "requires_standard": {"name": "MYSTD", "major": 2, "min_minor": 3},
        "products": [
            {"name": "alpha", "file_pattern": "alpha_{date}.nc", "schema": "alpha.json"}
        ],
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest))
    (vdir / "dataset_schema.json").write_text(json.dumps({"type": "object"}))
    (vdir / "alpha.json").write_text(json.dumps({"meta": {}}))
    return vdir


class TestRegisterAutoDetect:
    def test_detects_and_registers_project(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="MYSTD", major=2, minor=3, project_directory="mystd"
        )
        _fill_in_module(repo, "mystd")

        captured: dict = {}
        with _install_env(tmp_path, captured):
            register_resource(str(repo))

        assert "MYSTD-2" in captured["registry"].projects

    def test_detects_and_registers_pack(self, tmp_path: Path) -> None:
        pack_dir = _make_pack_dir(tmp_path, "v3", version=3)

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_resource(str(pack_dir))

        registry = captured["registry"]
        pack = registry.find_pack("https://host/packs", 3)
        assert pack is not None
        # local_path is the owned copy under ~/.vocal, not the source directory.
        owned = os.path.join(vocal_root, "packs", "host-packs", "v3")
        assert pack.local_path == owned
        assert pack.local_path != str(pack_dir)

    def test_path_with_no_marker_raises(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare"
        bare.mkdir()
        with pytest.raises(UnknownResourceKind):
            register_resource(str(bare))


class TestRegisterPack:
    """Integration coverage for ``register`` installing an owned pack copy.

    Driven through the public ``register_pack`` entry point; assertions are on
    externally observable state — what lands under ``~/.vocal`` and what the
    registry records — not on private helpers.
    """

    def test_inconsistent_version_raises(self, tmp_path: Path) -> None:
        # v99/ directory whose manifest declares version 3.
        pack_dir = _make_pack_dir(tmp_path, "v99", version=99, manifest_version=3)

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            with pytest.raises(PackInconsistent):
                register_pack(str(pack_dir))

        # Nothing was registered, and no owned copy was left behind.
        assert captured["registry"].packs == {}
        assert not (Path(vocal_root) / "packs" / "host-packs").exists()

    def test_copies_owned_copy_and_applies_denylist(self, tmp_path: Path) -> None:
        pack_dir = _make_pack_dir(tmp_path, "v3", version=3)
        # Cruft the install must normalise away.
        (pack_dir / ".git").mkdir()
        (pack_dir / ".git" / "config").write_text("x")
        (pack_dir / "tests").mkdir()
        (pack_dir / "tests" / "test_x.py").write_text("x")

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_pack(str(pack_dir))

        owned = Path(vocal_root) / "packs" / "host-packs" / "v3"
        snap = _snapshot(owned)
        # Denylisted entries are gone...
        assert ".git/config" not in snap
        assert "tests/test_x.py" not in snap
        # ...while the pack's own files survive.
        assert snap["manifest.json"]
        assert snap["dataset_schema.json"]
        assert snap["alpha.json"]

    def test_placed_canonically_regardless_of_source_dir_name(
        self, tmp_path: Path
    ) -> None:
        # A source directory not named v{Y}; load trusts the manifest's version.
        pack_dir = _make_pack_dir(tmp_path, "release", version=3)

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_pack(str(pack_dir))

        registered = captured["registry"].find_pack("https://host/packs", 3)
        owned = os.path.join(vocal_root, "packs", "host-packs", "v3")
        assert registered is not None
        assert registered.local_path == owned
        assert (Path(owned) / "manifest.json").is_file()

    def test_already_registered_raises_and_keeps_install(
        self, tmp_path: Path
    ) -> None:
        pack_dir = _make_pack_dir(tmp_path, "v3", version=3)

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_pack(str(pack_dir))
            owned = Path(vocal_root) / "packs" / "host-packs" / "v3"
            before = _snapshot(owned)
            with pytest.raises(CannotRegisterPackError):
                register_pack(str(pack_dir))
            # The existing install is untouched.
            assert _snapshot(owned) == before
            # force re-registers cleanly, overwriting copy + entry.
            register_pack(str(pack_dir), force=True)
        assert (
            captured["registry"].find_pack("https://host/packs", 3) is not None
        )

    def test_broken_force_reinstall_preserves_good_install(
        self, tmp_path: Path
    ) -> None:
        pack_dir = _make_pack_dir(tmp_path, "v3", version=3)

        captured: dict = {}
        with _install_env(tmp_path, captured) as vocal_root:
            register_pack(str(pack_dir))
            owned = Path(vocal_root) / "packs" / "host-packs" / "v3"
            before = _snapshot(owned)

            # Corrupt the source manifest, then force-reinstall.
            (pack_dir / "manifest.json").write_text("{ not valid json")
            with pytest.raises(InvalidManifest):
                register_pack(str(pack_dir), force=True)

            # The good install survived byte-for-byte.
            assert _snapshot(owned) == before
