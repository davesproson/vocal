"""Unit tests for vocal/resolution.py — the check-time resolver.

Every path is driven by a fake :class:`Registry` and synthetic file-attribute
arguments, with the project's ``filecodec`` injected as a plain dict, so the
resolver is exercised through its public ``resolve`` surface without importing
a real project package or touching the filesystem. Packs are built through the
``manifest`` module so the tests don't reach into the manifest's internal shape.
"""

import os

import pytest

from vocal.manifest import ManifestProduct, build_manifest
from vocal.resolution import (
    PackIncompatible,
    PackMissing,
    ProductNotFound,
    ProjectMissing,
    ProjectTooOld,
    ResolvedTarget,
    resolve,
)
from vocal.utils.registry import Pack, Project, Registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A project whose filecodec defines a single {date} placeholder, expanded into
# the product file_patterns below.
FILECODEC = {"date": {"regex": r"\d{8}"}}


def _project(
    name: str = "MYSTD",
    major: int = 2,
    minor: int = 3,
    local_path: str = "/cache/projects/mystd",
) -> Project:
    return Project(
        name=name,
        major=major,
        minor=minor,
        project_directory="mystd",
        local_path=local_path,
    )


def _pack(
    url: str = "https://host/packs",
    version: int = 3,
    name: str = "MYSTD",
    major: int = 2,
    min_minor: int = 3,
    local_path: str = "/cache/packs/host-packs/v3",
    products=None,
) -> Pack:
    if products is None:
        products = [
            ManifestProduct(
                name="foo", file_pattern="foo_{date}", schema="product_foo.json"
            )
        ]
    manifest = build_manifest(
        version=version,
        url=url,
        standard_name=name,
        standard_major=major,
        min_minor=min_minor,
        products=products,
    )
    return Pack(manifest=manifest, local_path=local_path)


def _registry(project: Project | None = None, pack: Pack | None = None) -> Registry:
    registry = Registry()
    if project is not None:
        registry.add_project(project)
    if pack is not None:
        registry.add_pack(pack)
    return registry


def _codec(project: Project):
    """A filecodec_loader that ignores the project and returns FILECODEC."""
    return FILECODEC


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_full_flow_resolves_project_pack_and_product(self) -> None:
        project = _project()
        pack = _pack(local_path="/cache/packs/host-packs/v3")
        registry = _registry(project, pack)

        target = resolve(
            registry,
            filename="foo_20260522.nc",
            conventions="CF-1.8 MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
            filecodec_loader=_codec,
        )

        assert isinstance(target, ResolvedTarget)
        assert target.project is project
        assert target.pack is pack
        assert target.product is not None
        assert target.product.name == "foo"
        assert target.schema_path == os.path.join(pack.local_path, "product_foo.json")

    def test_project_minor_above_claimed_minor_resolves(self) -> None:
        # File claims 2.1; registered project is at 2.3 — newer is fine.
        project = _project(minor=3)
        pack = _pack(min_minor=1)
        registry = _registry(project, pack)

        target = resolve(
            registry,
            filename="foo_20260522.nc",
            conventions="MYSTD-2.1",
            definitions_url="https://host/packs",
            definitions_version=3,
            filecodec_loader=_codec,
        )

        assert target.project is project

    def test_vocal_token_selected_among_co_conventions(self) -> None:
        project = _project()
        pack = _pack()
        registry = _registry(project, pack)

        target = resolve(
            registry,
            filename="foo_20260522.nc",
            conventions="CF-1.8 ACDD-1.3 MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
            filecodec_loader=_codec,
        )

        assert target.project is project


# ---------------------------------------------------------------------------
# Project resolution errors
# ---------------------------------------------------------------------------


class TestProjectErrors:
    def test_project_missing_when_no_matching_major(self) -> None:
        # Registry has MYSTD-2; file claims MYSTD-3.
        registry = _registry(_project(major=2))

        with pytest.raises(ProjectMissing) as exc:
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions="MYSTD-3.0",
                definitions_url="https://host/packs",
                definitions_version=3,
                filecodec_loader=_codec,
            )

        assert exc.value.message == "No project registered for MYSTD-3"
        assert exc.value.hint == (
            "Run 'vocal fetch <vocal_project_url>' to register the project, "
            "or pass -p <path>."
        )
        assert exc.value.code == "project_missing"

    def test_project_missing_hint_uses_project_url_when_available(self) -> None:
        registry = _registry()

        with pytest.raises(ProjectMissing) as exc:
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions="MYSTD-2.5",
                project_url="https://github.com/org/mystd",
                definition_override="/some/schema.json",
                filecodec_loader=_codec,
            )

        assert exc.value.message == "No project registered for MYSTD-2"
        assert "https://github.com/org/mystd" in exc.value.hint

    def test_project_too_old_when_registered_minor_below_claim(self) -> None:
        # Registry has MYSTD-2.3; file claims MYSTD-2.5.
        registry = _registry(_project(minor=3))

        with pytest.raises(ProjectTooOld) as exc:
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions="MYSTD-2.5",
                definitions_url="https://host/packs",
                definitions_version=3,
                filecodec_loader=_codec,
            )

        assert exc.value.message == (
            "File claims MYSTD-2.5 but registered project is at MYSTD-2.3"
        )
        assert exc.value.hint == (
            "Update the registered project: 'vocal fetch <vocal_project_url> "
            "--update'."
        )
        assert exc.value.code == "project_too_old"

    def test_project_missing_when_conventions_absent(self) -> None:
        registry = _registry(_project())

        with pytest.raises(ProjectMissing):
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions=None,
                definitions_url="https://host/packs",
                definitions_version=3,
                filecodec_loader=_codec,
            )


# ---------------------------------------------------------------------------
# Pack resolution errors
# ---------------------------------------------------------------------------


class TestPackMissing:
    def test_pack_missing_when_version_not_registered(self) -> None:
        project = _project()
        pack = _pack(version=3)
        registry = _registry(project, pack)

        with pytest.raises(PackMissing) as exc:
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions="MYSTD-2.3",
                definitions_url="https://host/packs",
                definitions_version=4,  # not registered
                filecodec_loader=_codec,
            )

        assert exc.value.message == "No pack registered for https://host/packs version 4"
        assert exc.value.hint == "Run 'vocal fetch https://host/packs/v4' to register it."
        assert exc.value.code == "pack_missing"

    def test_pack_missing_when_url_not_registered(self) -> None:
        project = _project()
        pack = _pack(url="https://host/packs")
        registry = _registry(project, pack)

        with pytest.raises(PackMissing):
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions="MYSTD-2.3",
                definitions_url="https://other/packs",
                definitions_version=3,
                filecodec_loader=_codec,
            )


class TestPackIncompatible:
    def test_min_minor_not_satisfied(self) -> None:
        # Pack requires MYSTD-2.4+; project is at 2.3.
        project = _project(minor=3)
        pack = _pack(name="MYSTD", major=2, min_minor=4)
        registry = _registry(project, pack)

        with pytest.raises(PackIncompatible) as exc:
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions="MYSTD-2.3",
                definitions_url="https://host/packs",
                definitions_version=3,
                filecodec_loader=_codec,
            )

        assert exc.value.message == (
            "Pack requires MYSTD-2.4+ but registered project is at 2.3"
        )
        assert exc.value.hint == (
            "Update the project to 2.4 or later, or pin to an older pack."
        )
        assert exc.value.code == "pack_incompatible"

    def test_major_mismatch(self) -> None:
        # Pack targets MYSTD-3; project is MYSTD-2.
        project = _project(name="MYSTD", major=2, minor=3)
        pack = _pack(name="MYSTD", major=3, min_minor=0)
        registry = _registry(project, pack)

        with pytest.raises(PackIncompatible) as exc:
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions="MYSTD-2.3",
                definitions_url="https://host/packs",
                definitions_version=3,
                filecodec_loader=_codec,
            )

        assert exc.value.message == (
            "Pack targets MYSTD-3 but registered project is MYSTD-2"
        )
        assert exc.value.hint == (
            "Register a project matching the pack's target standard, or pin to "
            "a pack built for MYSTD-2."
        )
        assert exc.value.code == "pack_incompatible"

    def test_name_mismatch(self) -> None:
        # Pack targets OTHER-2; project is MYSTD-2.
        project = _project(name="MYSTD", major=2, minor=3)
        pack = _pack(name="OTHER", major=2, min_minor=0)
        registry = _registry(project, pack)

        with pytest.raises(PackIncompatible) as exc:
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions="MYSTD-2.3",
                definitions_url="https://host/packs",
                definitions_version=3,
                filecodec_loader=_codec,
            )

        assert exc.value.message == (
            "Pack targets OTHER-2 but registered project is MYSTD-2"
        )
        assert exc.value.code == "pack_incompatible"


class TestProductNotFound:
    def test_no_product_pattern_matches(self) -> None:
        project = _project()
        pack = _pack(
            products=[
                ManifestProduct(
                    name="foo", file_pattern="foo_{date}", schema="product_foo.json"
                ),
                ManifestProduct(
                    name="bar", file_pattern="bar_{date}", schema="product_bar.json"
                ),
            ]
        )
        registry = _registry(project, pack)

        with pytest.raises(ProductNotFound) as exc:
            resolve(
                registry,
                filename="baz_20260522.nc",
                conventions="MYSTD-2.3",
                definitions_url="https://host/packs",
                definitions_version=3,
                filecodec_loader=_codec,
            )

        assert exc.value.message == (
            "File 'baz_20260522.nc' did not match any product pattern in pack"
        )
        assert "foo_{date}" in exc.value.hint
        assert "bar_{date}" in exc.value.hint
        assert exc.value.code == "product_not_found"


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------


class TestUrlNormalisation:
    def test_trailing_slash_difference_resolves(self) -> None:
        project = _project()
        pack = _pack(url="https://host/packs")
        registry = _registry(project, pack)

        target = resolve(
            registry,
            filename="foo_20260522.nc",
            conventions="MYSTD-2.3",
            definitions_url="https://host/packs/",  # trailing slash
            definitions_version=3,
            filecodec_loader=_codec,
        )

        assert target.pack is pack

    def test_host_case_difference_resolves(self) -> None:
        project = _project()
        pack = _pack(url="https://host/packs")
        registry = _registry(project, pack)

        target = resolve(
            registry,
            filename="foo_20260522.nc",
            conventions="MYSTD-2.3",
            definitions_url="https://HOST/packs",  # uppercase host
            definitions_version=3,
            filecodec_loader=_codec,
        )

        assert target.pack is pack


# ---------------------------------------------------------------------------
# -d override
# ---------------------------------------------------------------------------


class TestDefinitionOverride:
    def test_override_short_circuits_pack_resolution(self) -> None:
        # No pack registered at all; -d should still succeed.
        project = _project()
        registry = _registry(project)

        target = resolve(
            registry,
            filename="foo_20260522.nc",
            conventions="MYSTD-2.3",
            definition_override="/path/to/chosen_schema.json",
            definitions_url="https://host/packs",
            definitions_version=99,
            filecodec_loader=_codec,
        )

        assert target.project is project
        assert target.schema_path == "/path/to/chosen_schema.json"
        assert target.pack is None
        assert target.product is None

    def test_override_still_resolves_project_and_can_fail_project(self) -> None:
        # Project too old still raises even under -d.
        registry = _registry(_project(minor=3))

        with pytest.raises(ProjectTooOld):
            resolve(
                registry,
                filename="foo_20260522.nc",
                conventions="MYSTD-2.5",
                definition_override="/path/to/chosen_schema.json",
                filecodec_loader=_codec,
            )


# ---------------------------------------------------------------------------
# Project-only resolution (no pack reference, no override)
# ---------------------------------------------------------------------------


class TestProjectOnly:
    def test_no_pack_and_no_override_returns_project_only(self) -> None:
        project = _project()
        registry = _registry(project)

        target = resolve(
            registry,
            filename="foo_20260522.nc",
            conventions="MYSTD-2.3",
            filecodec_loader=_codec,
        )

        assert target.project is project
        assert target.schema_path is None
        assert target.pack is None
        assert target.product is None
