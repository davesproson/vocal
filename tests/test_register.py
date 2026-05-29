"""Tests for vocal/application/register.py and the init -> register flow.

These cover the new conventions.yaml-driven registration flow: identity and
layout are read from conventions.yaml, the project package is imported via the
single import path, and the project contract is enforced before a project is
added to the registry.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from vocal.application.init import init_project
from vocal.application.register import register_project
from vocal.conventions_file import ConventionsFile, MissingProjectExport
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
