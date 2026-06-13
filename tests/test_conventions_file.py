"""Unit tests for vocal/conventions_file.py.

Covers load/write round-tripping of ``conventions.yaml``, malformed-file
rejection, the single project-import path (requiring
``<repo>/<project_directory>/__init__.py``), and project-contract enforcement
(``defaults``, ``models.Dataset``; ``filecodec`` moved to the pack and is no
longer required).
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vocal.conventions_file import (
    ConventionsFile,
    InvalidConventionsFile,
    MissingProjectExport,
    conventions_path,
    import_project_package,
    module_path,
    validate_project_contract,
)
from vocal.versioning import Version


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(root: Path, body) -> None:
    with open(root / "conventions.yaml", "w") as f:
        yaml.dump(body, f)


VALID: dict[str, dict[str, object]] = {
    "conventions": {"name": "MYSTD", "major": 1, "minor": 2},
    "layout": {"project_directory": "mystd"},
}


def _make_valid_package(root: Path, module_name: str = "mystd") -> None:
    """Write a minimal but valid importable project package under ``root``."""
    _write(root, {**VALID, "layout": {"project_directory": module_name}})
    mod = root / module_name
    mod.mkdir(parents=True, exist_ok=True)
    (mod / "__init__.py").write_text(
        "from . import defaults\n"
        "from . import models\n"
        "filecodec = {}\n"
    )
    (mod / "defaults.py").write_text(
        "default_global_attrs = {}\n"
        "default_group_attrs = {}\n"
        "default_variable_attrs = {}\n"
    )
    (mod / "models.py").write_text(
        "from pydantic import BaseModel\n\n\nclass Dataset(BaseModel):\n    pass\n"
    )


# ---------------------------------------------------------------------------
# ConventionsFile.load / write
# ---------------------------------------------------------------------------


class TestLoad:
    def test_loads_valid_file(self, tmp_path: Path) -> None:
        _write(tmp_path, VALID)
        cf = ConventionsFile.load(str(tmp_path))
        assert cf.name == "MYSTD"
        assert cf.major == 1
        assert cf.minor == 2
        assert cf.project_directory == "mystd"

    def test_version_property(self, tmp_path: Path) -> None:
        _write(tmp_path, VALID)
        cf = ConventionsFile.load(str(tmp_path))
        assert cf.version == Version(name="MYSTD", major=1, minor=2)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidConventionsFile) as exc:
            ConventionsFile.load(str(tmp_path))
        assert "not found" in exc.value.message

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        (tmp_path / "conventions.yaml").write_text("conventions: [unclosed\n")
        with pytest.raises(InvalidConventionsFile) as exc:
            ConventionsFile.load(str(tmp_path))
        assert "not valid YAML" in exc.value.message

    def test_missing_conventions_block_raises(self, tmp_path: Path) -> None:
        _write(tmp_path, {"layout": {"project_directory": "mystd"}})
        with pytest.raises(InvalidConventionsFile) as exc:
            ConventionsFile.load(str(tmp_path))
        assert "conventions" in exc.value.message

    def test_missing_layout_block_raises(self, tmp_path: Path) -> None:
        _write(tmp_path, {"conventions": VALID["conventions"]})
        with pytest.raises(InvalidConventionsFile) as exc:
            ConventionsFile.load(str(tmp_path))
        assert "layout" in exc.value.message

    @pytest.mark.parametrize("missing_key", ["name", "major", "minor"])
    def test_missing_identity_key_raises(
        self, tmp_path: Path, missing_key: str
    ) -> None:
        conventions = dict(VALID["conventions"])
        del conventions[missing_key]
        _write(tmp_path, {"conventions": conventions, "layout": VALID["layout"]})
        with pytest.raises(InvalidConventionsFile) as exc:
            ConventionsFile.load(str(tmp_path))
        assert missing_key in exc.value.message

    def test_missing_project_directory_raises(self, tmp_path: Path) -> None:
        _write(tmp_path, {"conventions": VALID["conventions"], "layout": {}})
        with pytest.raises(InvalidConventionsFile) as exc:
            ConventionsFile.load(str(tmp_path))
        assert "project_directory" in exc.value.message

    def test_non_integer_version_raises(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "conventions": {"name": "MYSTD", "major": "one", "minor": 2},
                "layout": VALID["layout"],
            },
        )
        with pytest.raises(InvalidConventionsFile) as exc:
            ConventionsFile.load(str(tmp_path))
        assert "integer" in exc.value.message


class TestWriteRoundTrip:
    def test_write_then_load(self, tmp_path: Path) -> None:
        cf = ConventionsFile(name="MYSTD", major=2, minor=5, project_directory="mystd")
        cf.write(str(tmp_path))
        assert os.path.isfile(conventions_path(str(tmp_path)))
        loaded = ConventionsFile.load(str(tmp_path))
        assert loaded == cf


# ---------------------------------------------------------------------------
# module_path / import_project_package
# ---------------------------------------------------------------------------


class TestModulePath:
    def test_joins_repo_and_project_directory(self, tmp_path: Path) -> None:
        cf = ConventionsFile(name="MYSTD", major=1, minor=0, project_directory="mystd")
        assert module_path(str(tmp_path), cf) == os.path.join(str(tmp_path), "mystd")


class TestImportProjectPackage:
    def test_imports_valid_package(self, tmp_path: Path) -> None:
        _make_valid_package(tmp_path, module_name="goodpkg")
        module = import_project_package(str(tmp_path))
        assert hasattr(module, "defaults")
        assert hasattr(module, "filecodec")
        assert hasattr(module.models, "Dataset")

    def test_missing_init_raises(self, tmp_path: Path) -> None:
        # conventions.yaml points at a directory with no __init__.py.
        _write(tmp_path, VALID)
        (tmp_path / "mystd").mkdir()
        with pytest.raises(InvalidConventionsFile) as exc:
            import_project_package(str(tmp_path))
        assert "__init__.py" in exc.value.message


# ---------------------------------------------------------------------------
# validate_project_contract
# ---------------------------------------------------------------------------


class TestValidateProjectContract:
    def test_passes_with_all_exports(self) -> None:
        module = SimpleNamespace(
            defaults=object(),
            models=SimpleNamespace(Dataset=object()),
            filecodec={},
        )
        validate_project_contract(module)  # type: ignore[arg-type]  # does not raise

    @pytest.mark.parametrize("missing", ["defaults", "models"])
    def test_missing_top_level_export_named(self, missing: str) -> None:
        attrs = {
            "defaults": object(),
            "models": SimpleNamespace(Dataset=object()),
        }
        del attrs[missing]
        module = SimpleNamespace(**attrs)
        with pytest.raises(MissingProjectExport) as exc:
            validate_project_contract(module)  # type: ignore[arg-type]
        assert missing in exc.value.message

    def test_filecodec_no_longer_required(self) -> None:
        # filecodec moved to the pack; a project without one is valid.
        module = SimpleNamespace(
            defaults=object(),
            models=SimpleNamespace(Dataset=object()),
        )
        validate_project_contract(module)  # type: ignore[arg-type]  # does not raise

    def test_legacy_filecodec_tolerated(self) -> None:
        # A legacy project that still exports a filecodec is accepted; it is
        # simply ignored, not rejected.
        module = SimpleNamespace(
            defaults=object(),
            models=SimpleNamespace(Dataset=object()),
            filecodec={"date": {"regex": r"\d{8}"}},
        )
        validate_project_contract(module)  # type: ignore[arg-type]  # does not raise

    def test_missing_dataset_named(self) -> None:
        module = SimpleNamespace(
            defaults=object(),
            models=SimpleNamespace(),  # no Dataset
            filecodec={},
        )
        with pytest.raises(MissingProjectExport) as exc:
            validate_project_contract(module)  # type: ignore[arg-type]
        assert "models.Dataset" in exc.value.message
