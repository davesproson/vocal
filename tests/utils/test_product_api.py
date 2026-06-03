"""Tests for the programmatic product API in ``vocal.utils``.

``import_project``, ``get_spec``, and ``get_product`` (plus the helpers
``_resolve_version`` / ``_get_product_root``) are a public, user-facing surface:
they let code outside vocal load a project package and pull a product definition
straight out of a released pack, e.g.::

    from vocal.utils import import_project, get_product
    project = import_project('~/.vocal/projects/FAAM-0/faam')
    pack_dir = '~/.vocal/packs/<slug>'
    product = get_product('core_1hz', project, 'latest', pack_dir)

Because nothing inside vocal calls them, they have no other test coverage — this
module is their regression guard. In particular it pins ``'latest'`` resolution
to the post-decoupling pack layout (a ``product_root`` holding ``v{Y}/``
directories), the layout the FAAM packs actually use.
"""

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from vocal.utils import (
    _get_product_root,
    _resolve_version,
    get_product,
    get_spec,
    import_project,
)


# ---------------------------------------------------------------------------
# Fixtures: a self-contained project package and a versioned pack on disk
# ---------------------------------------------------------------------------


# A minimal but self-contained project package. It uses no relative imports so
# ``import_project`` (which loads the package's ``__init__.py`` directly) can
# exec it without submodule resolution. It exposes the project contract pieces
# the API touches: ``models.Dataset`` and ``filecodec``.
_PROJECT_INIT = '''\
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel


class Dataset(BaseModel):
    meta: dict[str, Any]
    attributes: dict[str, Any] = {}
    dimensions: list[Any] = []
    variables: list[Any] = []


models = SimpleNamespace(Dataset=Dataset)
filecodec = {"date": {"regex": r"\\d{8}"}}
defaults = SimpleNamespace(
    default_global_attrs={},
    default_group_attrs={},
    default_variable_attrs={},
)
'''


def _make_project(root: Path, package: str = "myproj") -> str:
    """Write a project package under ``root`` and return its path."""
    pkg_dir = root / package
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(_PROJECT_INIT)
    return str(pkg_dir)


def _write_product(version_dir: Path, name: str, title: str) -> None:
    """Write a single product schema JSON into a ``v{Y}/`` directory."""
    version_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"short_name": name, "file_pattern": f"{name}_{{date}}.nc"},
        "attributes": {"title": title},
        "dimensions": [],
        "variables": [],
    }
    (version_dir / f"{name}.json").write_text(json.dumps(payload))


def _make_pack(root: Path) -> str:
    """Build a multi-version pack and return its ``product_root``.

    Versions are deliberately ``v1`` / ``v2`` / ``v10`` so that ``'latest'``
    resolution is exercised against numeric (not lexical) ordering — lexically
    ``v2`` > ``v10``, numerically it is not.
    """
    pack_root = root / "pack"
    for version, title in ((1, "Alpha v1"), (2, "Alpha v2"), (10, "Alpha v10")):
        _write_product(pack_root / f"v{version}", "alpha", title)
    return str(pack_root)


@pytest.fixture
def project(tmp_path: Path) -> ModuleType:
    return import_project(_make_project(tmp_path / "proj"))


@pytest.fixture
def pack_root(tmp_path: Path) -> str:
    return _make_pack(tmp_path / "packsrc")


# ---------------------------------------------------------------------------
# _resolve_version
# ---------------------------------------------------------------------------


class TestResolveVersion:
    def test_latest_picks_highest_numeric_version(self, tmp_path: Path) -> None:
        root = _make_pack(tmp_path)
        resolved = _resolve_version("latest", root)
        # v10, not v2 — numeric ordering, the crux of the post-decoupling fix.
        assert Path(resolved).name == "v10"

    def test_explicit_int_version(self, tmp_path: Path) -> None:
        root = _make_pack(tmp_path)
        assert Path(_resolve_version(2, root)).name == "v2"

    def test_explicit_str_version(self, tmp_path: Path) -> None:
        root = _make_pack(tmp_path)
        assert Path(_resolve_version("2", root)).name == "v2"

    def test_latest_with_no_version_dirs_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            _resolve_version("latest", str(empty))


# ---------------------------------------------------------------------------
# get_spec
# ---------------------------------------------------------------------------


class TestGetSpec:
    def test_latest_returns_highest_version_spec(
        self, project: ModuleType, pack_root: str
    ) -> None:
        spec = get_spec("alpha", project, "latest", pack_root)
        assert spec is not None
        assert spec["attributes"]["title"] == "Alpha v10"

    def test_pinned_version_returns_that_version_spec(
        self, project: ModuleType, pack_root: str
    ) -> None:
        spec = get_spec("alpha", project, 1, pack_root)
        assert spec is not None
        assert spec["attributes"]["title"] == "Alpha v1"

    def test_unknown_short_name_returns_none(
        self, project: ModuleType, pack_root: str
    ) -> None:
        assert get_spec("does_not_exist", project, "latest", pack_root) is None

    def test_dataset_schema_json_is_excluded(
        self, project: ModuleType, tmp_path: Path
    ) -> None:
        # A dataset_schema.json that *would* match by short_name must still be
        # skipped: lookup is over product schemas only.
        version_dir = tmp_path / "pack" / "v1"
        _write_product(version_dir, "alpha", "The real alpha")
        (version_dir / "dataset_schema.json").write_text(
            json.dumps({"meta": {"short_name": "alpha"}, "attributes": {"title": "schema"}})
        )
        spec = get_spec("alpha", project, "latest", str(tmp_path / "pack"))
        assert spec is not None
        assert spec["attributes"]["title"] == "The real alpha"

    def test_no_project_raises(self, pack_root: str) -> None:
        with pytest.raises(ValueError):
            get_spec("alpha", None, "latest", pack_root)


# ---------------------------------------------------------------------------
# get_product
# ---------------------------------------------------------------------------


class TestGetProduct:
    def test_returns_validated_dataset_model(
        self, project: ModuleType, pack_root: str
    ) -> None:
        product = get_product("alpha", project, "latest", pack_root)
        assert isinstance(product, project.models.Dataset)
        assert product.attributes["title"] == "Alpha v10"

    def test_pinned_version(self, project: ModuleType, pack_root: str) -> None:
        product = get_product("alpha", project, 1, pack_root)
        assert product.attributes["title"] == "Alpha v1"

    def test_unknown_short_name_fails_validation(
        self, project: ModuleType, pack_root: str
    ) -> None:
        # get_spec returns None, which is not a valid Dataset payload.
        with pytest.raises(Exception):
            get_product("does_not_exist", project, "latest", pack_root)

    def test_no_project_raises(self, pack_root: str) -> None:
        with pytest.raises(ValueError):
            get_product("alpha", None, "latest", pack_root)


# ---------------------------------------------------------------------------
# import_project / _get_product_root
# ---------------------------------------------------------------------------


class TestImportProject:
    def test_exposes_project_contract(self, tmp_path: Path) -> None:
        module = import_project(_make_project(tmp_path))
        assert hasattr(module.models, "Dataset")
        assert module.filecodec["date"]["regex"] == r"\d{8}"

    def test_trailing_slash_path(self, tmp_path: Path) -> None:
        # Regression: import_project must tolerate a trailing slash on the path.
        path = _make_project(tmp_path)
        module = import_project(path + "/")
        assert hasattr(module.models, "Dataset")

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            import_project(str(tmp_path / "no_such_project"))


class TestGetProductRoot:
    def test_derives_repo_root_from_module_file(self, tmp_path: Path) -> None:
        # Default product_root is two levels up from the package __init__.py,
        # i.e. the project repo root.
        module = import_project(_make_project(tmp_path / "repo"))
        assert _get_product_root(module) == str(tmp_path / "repo")
