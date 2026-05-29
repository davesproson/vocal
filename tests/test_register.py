"""Tests for vocal/application/register.py and the init -> register flow.

These cover the new conventions.yaml-driven registration flow: identity and
layout are read from conventions.yaml, the project package is imported via the
single import path, and the project contract is enforced before a project is
added to the registry.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from vocal.application.init import init_project
from vocal.application.register import (
    CannotRegisterPackError,
    UnknownResourceKind,
    register_pack,
    register_project,
    register_resource,
)
from vocal.conventions_file import ConventionsFile, MissingProjectExport
from vocal.manifest import PackInconsistent
from vocal.utils.registry import Registry


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


def _registers_into(captured: dict):
    """Patch register's registry I/O to use an in-memory registry."""
    registry = Registry(projects={})
    captured["registry"] = registry
    return patch.multiple(
        "vocal.application.register",
        load_registry=lambda: registry,
        save_registry=lambda r: captured.__setitem__("registry", r),
    )


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
        with _registers_into(captured):
            register_project(str(repo))

        registry = captured["registry"]
        assert "MYSTD-2" in registry.projects
        registered = registry.projects["MYSTD-2"]
        # local_path is the repo root, not the module subdirectory.
        assert registered.local_path == str(repo)
        assert registered.minor == 3
        assert registered.project_directory == "mystd"

    def test_register_missing_export_raises_named(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        init_project(
            str(repo), name="MYSTD", major=1, minor=0, project_directory="mystd"
        )
        # Fill in a module missing filecodec.
        _fill_in_module(repo, "mystd", filecodec=False)

        captured: dict = {}
        with _registers_into(captured):
            with pytest.raises(MissingProjectExport) as exc:
                register_project(str(repo))

        assert "filecodec" in exc.value.message
        # Nothing was registered.
        assert captured["registry"].projects == {}


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
        with _registers_into(captured):
            register_resource(str(repo))

        assert "MYSTD-2" in captured["registry"].projects

    def test_detects_and_registers_pack(self, tmp_path: Path) -> None:
        pack_dir = _make_pack_dir(tmp_path, "v3", version=3)

        captured: dict = {}
        with _registers_into(captured):
            register_resource(str(pack_dir))

        registry = captured["registry"]
        pack = registry.find_pack("https://host/packs", 3)
        assert pack is not None
        assert pack.local_path == str(pack_dir)

    def test_path_with_no_marker_raises(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare"
        bare.mkdir()
        with pytest.raises(UnknownResourceKind):
            register_resource(str(bare))


class TestRegisterPack:
    def test_inconsistent_version_raises(self, tmp_path: Path) -> None:
        # v99/ directory whose manifest declares version 3.
        pack_dir = _make_pack_dir(tmp_path, "v99", version=99, manifest_version=3)

        captured: dict = {}
        with _registers_into(captured):
            with pytest.raises(PackInconsistent):
                register_pack(str(pack_dir))

        # Nothing was registered.
        assert captured["registry"].packs == {}

    def test_already_registered_raises_without_force(self, tmp_path: Path) -> None:
        pack_dir = _make_pack_dir(tmp_path, "v3", version=3)

        captured: dict = {}
        with _registers_into(captured):
            register_pack(str(pack_dir))
            with pytest.raises(CannotRegisterPackError):
                register_pack(str(pack_dir))
            # force re-registers cleanly.
            register_pack(str(pack_dir), force=True)
