"""Unit tests for vocal/resolution.py — the two-axis check-time resolver.

Every path is driven through the public :func:`resolve_file` surface with a
synthetic :class:`FileConventions` and a fake :class:`Registry`, with packs
built through the ``manifest`` module so the tests never reach into the
manifest's internal shape, import a real project package, or touch the
filesystem. Assertions are made on the returned :class:`Resolution` — its
``projects``, ``pack``, ``failures``, and ``warnings`` — never on private
helpers or intermediate state.
"""

import os

import pytest

from vocal.manifest import ManifestProduct, build_manifest
from vocal.resolution import (
    NothingToCheck,
    PackMissing,
    PackTarget,
    ProductNotFound,
    ProjectMissing,
    ProjectTarget,
    Resolution,
    resolve_file,
)
from vocal.utils.conventions import FileConventions
from vocal.utils.registry import Pack, Project, Registry
from vocal.versioning import Version, VersionConstraint

# The pack's own filecodec, expanding the {date} placeholder in the products'
# file_patterns below.
FILECODEC = {"date": {"regex": r"\d{8}"}}

MYSTD_URL = "https://github.com/org/mystd"
PACK_URL = "https://host/packs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project(
    name: str = "MYSTD",
    major: int = 2,
    minor: int = 3,
    url: str = MYSTD_URL,
    local_path: str = "/cache/projects/mystd",
) -> Project:
    return Project(
        name=name,
        major=major,
        minor=minor,
        project_directory="mystd",
        local_path=local_path,
        url=url,
    )


def _pack(
    url: str = PACK_URL,
    version: int = 3,
    satisfies=None,
    products=None,
    local_path: str = "/cache/packs/host-packs/v3",
) -> Pack:
    if products is None:
        products = [
            ManifestProduct(
                name="foo", file_pattern="foo_{date}", schema="product_foo.json"
            )
        ]
    if satisfies is None:
        satisfies = [VersionConstraint(name="MYSTD", major=2, min_minor=3)]
    manifest = build_manifest(
        version=version,
        url=url,
        filecodec=FILECODEC,
        satisfies_standards=satisfies,
        products=products,
    )
    return Pack(manifest=manifest, local_path=local_path)


def _registry(*items) -> Registry:
    registry = Registry()
    for item in items:
        if isinstance(item, Project):
            registry.add_project(item)
        elif isinstance(item, Pack):
            registry.add_pack(item)
    return registry


def _attrs(
    conventions=None,
    project_urls=None,
    definitions_url=None,
    definitions_version=None,
) -> FileConventions:
    return FileConventions(
        conventions=conventions,
        project_urls=project_urls or [],
        definitions_url=definitions_url,
        definitions_version=definitions_version,
    )


def _resolve(filename="foo_20260522.nc", *, attrs, registry) -> Resolution:
    return resolve_file(filename, attrs=attrs, registry=registry)


def _verifiable(resolution: Resolution) -> list[ProjectTarget]:
    return [t for t in resolution.projects if t.verifiable]


# ---------------------------------------------------------------------------
# Mandatory standards axis (vocal_project_url)
# ---------------------------------------------------------------------------


class TestMandatoryProjects:
    def test_url_selects_mandatory_project_major_from_conventions_token(self) -> None:
        project = _project(major=2, minor=3)
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(conventions="CF-1.8 MYSTD-2.3", project_urls=[MYSTD_URL]),
            registry=registry,
        )

        assert len(resolution.projects) == 1
        target = resolution.projects[0]
        assert target.project is project
        assert target.mandatory is True
        assert target.claimed_version == Version("MYSTD", 2, 3)
        assert target.verifiable is True
        assert resolution.failures == []

    def test_url_major_sourced_from_token_not_url_lookup(self) -> None:
        # The URL's installed project is MYSTD-2, but the file claims MYSTD-3:
        # the claimed major is not installed, so it is a ProjectMissing failure.
        project = _project(major=2, minor=3)
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(conventions="MYSTD-3.0", project_urls=[MYSTD_URL]),
            registry=registry,
        )

        assert resolution.projects == []
        assert len(resolution.failures) == 1
        failure = resolution.failures[0]
        assert isinstance(failure, ProjectMissing)
        assert failure.message == "No project registered for MYSTD-3"
        assert failure.hint is not None and MYSTD_URL in failure.hint

    def test_url_with_no_installed_project_is_missing_with_url_hint(self) -> None:
        registry = _registry()  # nothing installed

        resolution = _resolve(
            attrs=_attrs(conventions="MYSTD-2.3", project_urls=[MYSTD_URL]),
            registry=registry,
        )

        assert resolution.projects == []
        assert len(resolution.failures) == 1
        failure = resolution.failures[0]
        assert isinstance(failure, ProjectMissing)
        assert failure.hint is not None and MYSTD_URL in failure.hint
        assert failure.code == "project_missing"

    def test_url_without_matching_token_resolves_installed_with_no_version(
        self,
    ) -> None:
        # vocal_project_url present but Conventions names no matching token:
        # no version constraint, verify the single installed project at the URL.
        project = _project()
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(conventions="CF-1.8", project_urls=[MYSTD_URL]),
            registry=registry,
        )

        assert len(resolution.projects) == 1
        target = resolution.projects[0]
        assert target.project is project
        assert target.mandatory is True
        assert target.claimed_version is None
        assert target.verifiable is True


# ---------------------------------------------------------------------------
# Opportunistic standards axis (Conventions-only)
# ---------------------------------------------------------------------------


class TestOpportunisticProjects:
    def test_installed_conventions_standard_is_verified_opportunistically(
        self,
    ) -> None:
        project = _project()
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(conventions="MYSTD-2.3"),
            registry=registry,
        )

        assert len(resolution.projects) == 1
        target = resolution.projects[0]
        assert target.project is project
        assert target.mandatory is False
        assert target.verifiable is True

    def test_uninstalled_opportunistic_standard_is_skipped_with_comment(self) -> None:
        # OTHER is named in Conventions but no project is installed for it.
        project = _project()
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(conventions="MYSTD-2.3 OTHER-1.0"),
            registry=registry,
        )

        # MYSTD is verified; OTHER is skipped with a comment, not a failure or
        # a warning (it's an everyday, non-actionable outcome).
        assert [t.project for t in resolution.projects] == [project]
        assert resolution.failures == []
        assert resolution.warnings == []
        assert len(resolution.comments) == 1
        assert "OTHER-1.0" in resolution.comments[0].message
        assert resolution.comments[0].code == "standard_not_verified"

    def test_external_co_conventions_fall_out(self) -> None:
        # CF/ACDD have no URL and no installed project: they produce comments,
        # never failures, and never block resolution of the vocal standard.
        project = _project()
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(conventions="CF-1.8 ACDD-1.3 MYSTD-2.3"),
            registry=registry,
        )

        assert [t.project for t in resolution.projects] == [project]
        assert resolution.failures == []
        commented = {c.message.split(" ")[0] for c in resolution.comments}
        assert "CF-1.8" in commented
        assert "ACDD-1.3" in commented


# ---------------------------------------------------------------------------
# Too-old → unverifiable target (for mandatory and opportunistic alike)
# ---------------------------------------------------------------------------


class TestTooOld:
    def test_mandatory_too_old_is_unverifiable_with_update_hint(self) -> None:
        # Installed MYSTD-2.3; file claims MYSTD-2.5 via a mandatory URL.
        project = _project(minor=3)
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(conventions="MYSTD-2.5", project_urls=[MYSTD_URL]),
            registry=registry,
        )

        assert resolution.failures == []
        assert len(resolution.projects) == 1
        target = resolution.projects[0]
        assert target.verifiable is False
        assert target.mandatory is True
        assert target.hint is not None and "--update" in target.hint
        assert _verifiable(resolution) == []

    def test_opportunistic_too_old_warns_without_target(self) -> None:
        # Installed MYSTD-2.3; file claims MYSTD-2.7 via Conventions only.
        # Best effort: a too-old opportunistic standard is noted as a warning
        # and dropped — no target, so it never taints the verdict.
        project = _project(minor=3)
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(conventions="MYSTD-2.7"),
            registry=registry,
        )

        assert resolution.projects == []
        assert len(resolution.warnings) == 1
        warning = resolution.warnings[0]
        assert warning.code == "standard_not_verified_too_old"
        assert "MYSTD-2.7" in warning.message
        assert warning.hint is not None and "--update" in warning.hint

    def test_newer_installed_minor_is_verifiable(self) -> None:
        # Installed MYSTD-2.5; file claims MYSTD-2.3 — newer is fine.
        project = _project(minor=5)
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(conventions="MYSTD-2.3", project_urls=[MYSTD_URL]),
            registry=registry,
        )

        assert len(resolution.projects) == 1
        assert resolution.projects[0].verifiable is True


# ---------------------------------------------------------------------------
# Product axis (pack)
# ---------------------------------------------------------------------------


class TestProductAxis:
    def test_pack_matched_off_embedded_codec(self) -> None:
        project = _project()
        pack = _pack(version=3)
        registry = _registry(project, pack)

        resolution = _resolve(
            "foo_20260522.nc",
            attrs=_attrs(
                conventions="MYSTD-2.3",
                definitions_url=PACK_URL,
                definitions_version=3,
            ),
            registry=registry,
        )

        assert isinstance(resolution.pack, PackTarget)
        assert resolution.pack.pack is pack
        assert resolution.pack.product.name == "foo"
        assert resolution.pack.schema_path == os.path.join(
            pack.local_path, "product_foo.json"
        )

    def test_pack_version_absent_resolves_to_highest_registered(self) -> None:
        v3 = _pack(version=3, local_path="/cache/packs/host/v3")
        v4 = _pack(version=4, local_path="/cache/packs/host/v4")
        registry = _registry(_project(), v3, v4)

        resolution = _resolve(
            attrs=_attrs(conventions="MYSTD-2.3", definitions_url=PACK_URL),
            registry=registry,
        )

        assert resolution.pack is not None
        assert resolution.pack.pack is v4

    def test_pack_missing_is_a_failure(self) -> None:
        registry = _registry(_project(), _pack(version=3))

        resolution = _resolve(
            attrs=_attrs(
                conventions="MYSTD-2.3",
                definitions_url=PACK_URL,
                definitions_version=4,  # not registered
            ),
            registry=registry,
        )

        assert resolution.pack is None
        assert len(resolution.failures) == 1
        failure = resolution.failures[0]
        assert isinstance(failure, PackMissing)
        assert failure.code == "pack_missing"

    def test_product_not_found_is_a_failure(self) -> None:
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
        registry = _registry(_project(), pack)

        resolution = _resolve(
            "baz_20260522.nc",
            attrs=_attrs(
                conventions="MYSTD-2.3",
                definitions_url=PACK_URL,
                definitions_version=3,
            ),
            registry=registry,
        )

        assert resolution.pack is None
        assert len(resolution.failures) == 1
        failure = resolution.failures[0]
        assert isinstance(failure, ProductNotFound)
        assert failure.hint is not None
        assert "foo_{date}" in failure.hint and "bar_{date}" in failure.hint

    def test_pack_resolves_without_any_standard_installed(self) -> None:
        # Story 5: the product check runs even when no standard is installed.
        pack = _pack(version=3)
        registry = _registry(pack)

        resolution = _resolve(
            attrs=_attrs(definitions_url=PACK_URL, definitions_version=3),
            registry=registry,
        )

        assert resolution.pack is not None
        assert resolution.projects == []


# ---------------------------------------------------------------------------
# satisfies_standards (advisory: warning, never a failure)
# ---------------------------------------------------------------------------


class TestSatisfiesStandards:
    def test_unmet_assertion_warns_not_fails(self) -> None:
        # Pack asserts MYSTD-2.4+; the file claims only MYSTD-2.3 (below floor),
        # so no claimed version satisfies the assertion → a warning.
        pack = _pack(satisfies=[VersionConstraint("MYSTD", 2, 4)])
        registry = _registry(_project(), pack)

        resolution = _resolve(
            attrs=_attrs(
                conventions="MYSTD-2.3",
                definitions_url=PACK_URL,
                definitions_version=3,
            ),
            registry=registry,
        )

        assert resolution.pack is not None
        assert resolution.failures == []
        codes = {w.code for w in resolution.warnings}
        assert "satisfies_standards_unmet" in codes

    def test_met_assertion_emits_no_warning(self) -> None:
        # Pack asserts MYSTD-2.3+; the file claims MYSTD-2.5 → satisfied.
        pack = _pack(satisfies=[VersionConstraint("MYSTD", 2, 3)])
        registry = _registry(_project(minor=5), pack)

        resolution = _resolve(
            attrs=_attrs(
                conventions="MYSTD-2.5",
                definitions_url=PACK_URL,
                definitions_version=3,
            ),
            registry=registry,
        )

        assert resolution.pack is not None
        codes = {w.code for w in resolution.warnings}
        assert "satisfies_standards_unmet" not in codes


# ---------------------------------------------------------------------------
# Nothing to check
# ---------------------------------------------------------------------------


class TestNothingToCheck:
    def test_no_claims_at_all_raises(self) -> None:
        registry = _registry(_project())

        with pytest.raises(NothingToCheck):
            _resolve(attrs=_attrs(), registry=registry)

    def test_conventions_only_with_no_installed_match_raises(self) -> None:
        # CF only, nothing installed for it: a warning would be all there is,
        # which is nothing to act on → raise.
        registry = _registry()

        with pytest.raises(NothingToCheck):
            _resolve(attrs=_attrs(conventions="CF-1.8"), registry=registry)

    def test_mandatory_missing_does_not_raise(self) -> None:
        # A mandatory URL that isn't installed is a failure to report, not
        # "nothing to check".
        registry = _registry()

        resolution = _resolve(
            attrs=_attrs(conventions="MYSTD-2.3", project_urls=[MYSTD_URL]),
            registry=registry,
        )

        assert len(resolution.failures) == 1


# ---------------------------------------------------------------------------
# URL normalisation (pack axis)
# ---------------------------------------------------------------------------


class TestUrlNormalisation:
    def test_pack_trailing_slash_resolves(self) -> None:
        pack = _pack(url=PACK_URL)
        registry = _registry(_project(), pack)

        resolution = _resolve(
            attrs=_attrs(
                conventions="MYSTD-2.3",
                definitions_url=PACK_URL + "/",
                definitions_version=3,
            ),
            registry=registry,
        )

        assert resolution.pack is not None
        assert resolution.pack.pack is pack

    def test_project_url_git_suffix_resolves(self) -> None:
        # The file declares the .git form; the registry stored the bare form.
        project = _project(url=MYSTD_URL)
        registry = _registry(project)

        resolution = _resolve(
            attrs=_attrs(
                conventions="MYSTD-2.3", project_urls=[MYSTD_URL + ".git"]
            ),
            registry=registry,
        )

        assert len(resolution.projects) == 1
        assert resolution.projects[0].project is project
