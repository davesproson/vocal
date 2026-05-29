"""Integration tests for ``vocal release`` (vocal/application/release.py).

These exercise the release flow end-to-end against a real fixture project and a
real directory of YAML product definitions, asserting the on-disk pack layout,
the ``manifest.json`` content, and each of the locked error paths (URL
fallback/mismatch, first-release-requires-url, release-exists, and
NoProductDefinitions).
"""

import filecmp
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vocal.application.release import (
    FirstReleaseRequiresURL,
    NoProductDefinitions,
    PackURLMismatch,
    ReleaseExists,
    release,
)
from vocal.cli.vocal import app

# ---------------------------------------------------------------------------
# Fixtures: a real, importable project and a directory of definitions
# ---------------------------------------------------------------------------

_CONVENTIONS_YAML = """\
conventions:
  name: MYSTD
  major: 2
  minor: 3
layout:
  project_directory: mystd
"""

_PACKAGE_INIT = """\
from . import defaults
from . import models

filecodec = {"date": {"regex": r"\\\\d{8}"}}
"""

_DEFAULTS = """\
default_global_attrs = {}
default_group_attrs = {}
default_variable_attrs = {}
"""

_MODELS = """\
from pydantic import BaseModel, ConfigDict


class Meta(BaseModel):
    model_config = ConfigDict(extra="allow")
    file_pattern: str


class Dataset(BaseModel):
    model_config = ConfigDict(extra="allow")
    meta: Meta
"""


def _make_project(root: Path) -> str:
    """Scaffold a minimal, importable MYSTD-2.3 project at ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "conventions.yaml").write_text(_CONVENTIONS_YAML)
    module = root / "mystd"
    module.mkdir(parents=True, exist_ok=True)
    (module / "__init__.py").write_text(_PACKAGE_INIT)
    (module / "defaults.py").write_text(_DEFAULTS)
    (module / "models.py").write_text(_MODELS)
    return str(root)


def _write_definition(defs_dir: Path, name: str, file_pattern: str) -> None:
    defs_dir.mkdir(parents=True, exist_ok=True)
    (defs_dir / f"{name}.yaml").write_text(
        "meta:\n"
        f'  file_pattern: "{file_pattern}"\n'
        f"  short_name: {name}\n"
        "attributes:\n"
        f'  title: "{name}"\n'
        "variables: []\n"
    )


@pytest.fixture
def project(tmp_path: Path) -> str:
    return _make_project(tmp_path / "project")


@pytest.fixture
def definitions(tmp_path: Path) -> str:
    defs = tmp_path / "definitions"
    _write_definition(defs, "alpha", "alpha_{date}.nc")
    _write_definition(defs, "bravo", "bravo_{date}.nc")
    return str(defs)


def _read_manifest(output: Path, where: str) -> dict:
    with open(output / where / "manifest.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLayout:
    def test_writes_version_dir_and_latest(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        release(
            project_path=project,
            version=3,
            definitions=definitions,
            output_dir=str(output),
            url="https://host/packs",
        )

        for where in ("v3", "latest"):
            d = output / where
            assert (d / "manifest.json").is_file()
            assert (d / "dataset_schema.json").is_file()
            assert (d / "alpha.json").is_file()
            assert (d / "bravo.json").is_file()

    def test_manifest_product_schemas_are_siblings(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        release(
            project_path=project,
            version=1,
            definitions=definitions,
            output_dir=str(output),
            url="https://host/packs",
        )

        manifest = _read_manifest(output, "v1")
        for product in manifest["products"]:
            schema = product["schema"]
            assert ".." not in schema
            assert not os.path.isabs(schema)
            assert (output / "v1" / schema).is_file()

    def test_latest_is_byte_equal_copy(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        release(
            project_path=project,
            version=2,
            definitions=definitions,
            output_dir=str(output),
            url="https://host/packs",
        )

        cmp = filecmp.dircmp(str(output / "v2"), str(output / "latest"))
        assert cmp.diff_files == []
        assert sorted(cmp.left_only) == [] and sorted(cmp.right_only) == []
        match, mismatch, errors = filecmp.cmpfiles(
            str(output / "v2"), str(output / "latest"), cmp.common_files, shallow=False
        )
        assert mismatch == [] and errors == []

    def test_latest_tracks_highest_version(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        common = dict(
            project_path=project, definitions=definitions, output_dir=str(output)
        )
        release(version=1, url="https://host/packs", **common)
        release(version=3, **common)
        release(version=2, **common)  # releasing an older version last

        # latest still points at the highest version present (v3).
        assert _read_manifest(output, "latest")["version"] == 3


# ---------------------------------------------------------------------------
# Manifest content
# ---------------------------------------------------------------------------


class TestManifestContent:
    def test_manifest_fields(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        release(
            project_path=project,
            version=5,
            definitions=definitions,
            output_dir=str(output),
            url="https://host/packs",
        )

        manifest = _read_manifest(output, "v5")
        assert manifest["schema_version"] == 1
        assert manifest["version"] == 5
        assert manifest["url"] == "https://host/packs"
        assert manifest["requires_standard"] == {
            "name": "MYSTD",
            "major": 2,
            "min_minor": 3,  # defaults to the project's current minor
        }

        products = {p["name"]: p for p in manifest["products"]}
        assert set(products) == {"alpha", "bravo"}
        assert products["alpha"]["file_pattern"] == "alpha_{date}.nc"
        assert products["alpha"]["schema"] == "alpha.json"

    def test_min_minor_override(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        release(
            project_path=project,
            version=1,
            definitions=definitions,
            output_dir=str(output),
            url="https://host/packs",
            min_minor=1,
        )
        assert _read_manifest(output, "v1")["requires_standard"]["min_minor"] == 1

    def test_url_normalised_when_supplied_noncanonical(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        release(
            project_path=project,
            version=1,
            definitions=definitions,
            output_dir=str(output),
            url="https://Host/packs/",  # mixed case + trailing slash
        )
        assert _read_manifest(output, "v1")["url"] == "https://host/packs"


# ---------------------------------------------------------------------------
# URL fallback and mismatch
# ---------------------------------------------------------------------------


class TestURLResolution:
    def test_fallback_from_latest_manifest(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        common = dict(
            project_path=project, definitions=definitions, output_dir=str(output)
        )
        release(version=1, url="https://host/packs", **common)
        # second release omits --url; uses the prior release's URL.
        release(version=2, **common)
        assert _read_manifest(output, "v2")["url"] == "https://host/packs"

    def test_first_release_without_url_raises(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        with pytest.raises(FirstReleaseRequiresURL):
            release(
                project_path=project,
                version=1,
                definitions=definitions,
                output_dir=str(output),
            )

    def test_url_mismatch_raises(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        common = dict(
            project_path=project, definitions=definitions, output_dir=str(output)
        )
        release(version=1, url="https://host/packs", **common)
        with pytest.raises(PackURLMismatch):
            release(version=2, url="https://host/other", **common)

    def test_trailing_slash_and_case_not_a_mismatch(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        common = dict(
            project_path=project, definitions=definitions, output_dir=str(output)
        )
        release(version=1, url="https://host/packs", **common)
        # trailing slash and host case differ but normalise to the same URL.
        release(version=2, url="https://Host/packs/", **common)
        assert _read_manifest(output, "v2")["url"] == "https://host/packs"


# ---------------------------------------------------------------------------
# Release-exists and force
# ---------------------------------------------------------------------------


class TestReleaseExists:
    def test_existing_version_raises(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        common = dict(
            project_path=project,
            definitions=definitions,
            output_dir=str(output),
            url="https://host/packs",
        )
        release(version=1, **common)
        with pytest.raises(ReleaseExists):
            release(version=1, **common)

    def test_force_allows_overwrite(
        self, project: str, definitions: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        common = dict(
            project_path=project,
            definitions=definitions,
            output_dir=str(output),
            url="https://host/packs",
        )
        release(version=1, **common)
        # overwriting with --force succeeds and rewrites the release.
        release(version=1, force=True, **common)
        assert _read_manifest(output, "v1")["version"] == 1


# ---------------------------------------------------------------------------
# No product definitions
# ---------------------------------------------------------------------------


def test_no_product_definitions_raises(project: str, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NoProductDefinitions):
        release(
            project_path=project,
            version=1,
            definitions=str(empty),
            output_dir=str(tmp_path / "out"),
            url="https://host/packs",
        )


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_release_smoke(project: str, definitions: str, tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "release",
            "--project",
            project,
            "--version",
            "4",
            "--definitions",
            definitions,
            "--output",
            str(output),
            "--url",
            "https://host/packs",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _read_manifest(output, "v4")["version"] == 4
