"""Unit tests for the pure web view-model builder (vocal/web/models.py).

``build_library_view`` is driven directly against in-memory ``Registry``
objects, asserting the shape of the view models it returns. This slice covers
grouping packs by URL and the descending version sort, and the three-state
install status of each version's advisory ``satisfies_standards`` against the
registry's projects.
"""

from vocal.application.install import derive_url_slug
from vocal.manifest import build_manifest
from vocal.utils.registry import Pack, Project, Registry
from vocal.versioning import VersionConstraint
from vocal.web.models import build_library_view


def _pack(
    url: str = "https://host/packs",
    version: int = 1,
    name: str = "MYSTD",
    major: int = 2,
    min_minor: int = 3,
) -> Pack:
    manifest = build_manifest(
        version=version,
        url=url,
        filecodec={},
        satisfies_standards=(VersionConstraint(name, major, min_minor),),
        products=[],
    )
    return Pack(manifest=manifest, local_path=f"/cache/{version}")


def _project(
    name: str = "MYSTD", major: int = 2, minor: int = 3
) -> Project:
    return Project(
        name=name,
        major=major,
        minor=minor,
        project_directory=f"{name.lower()}_{major}",
        local_path=f"/cache/{name}-{major}",
    )


def _registry(*packs: Pack, projects: tuple[Project, ...] = ()) -> Registry:
    registry = Registry()
    for project in projects:
        registry.add_project(project)
    for pack in packs:
        registry.add_pack(pack)
    return registry


class TestBuildLibraryView:
    def test_empty_registry_has_no_packs(self) -> None:
        view = build_library_view(_registry())
        assert view.packs == []

    def test_groups_versions_under_one_url(self) -> None:
        registry = _registry(
            _pack(version=1), _pack(version=2), _pack(version=3)
        )
        view = build_library_view(registry)

        assert len(view.packs) == 1
        assert view.packs[0].url == "https://host/packs"
        assert len(view.packs[0].versions) == 3

    def test_versions_sorted_descending(self) -> None:
        registry = _registry(
            _pack(version=1), _pack(version=3), _pack(version=2)
        )
        view = build_library_view(registry)

        versions = [v.version for v in view.packs[0].versions]
        assert versions == [3, 2, 1]

    def test_latest_is_highest_version(self) -> None:
        registry = _registry(_pack(version=1), _pack(version=5))
        view = build_library_view(registry)

        assert view.packs[0].latest.version == 5

    def test_distinct_urls_are_separate_entries(self) -> None:
        registry = _registry(
            _pack(url="https://host/a", version=1),
            _pack(url="https://host/b", version=1),
        )
        view = build_library_view(registry)

        urls = {pack.url for pack in view.packs}
        assert urls == {"https://host/a", "https://host/b"}

    def test_version_carries_satisfied_standard(self) -> None:
        registry = _registry(_pack(name="MYSTD", major=2, min_minor=4))
        view = build_library_view(registry)

        version = view.packs[0].versions[0]
        assert len(version.satisfies) == 1
        assert version.satisfies[0].constraint == "MYSTD-2.4+"


class TestSatisfiedStandardStatus:
    def test_satisfied_when_project_minor_meets_minimum(self) -> None:
        registry = _registry(
            _pack(name="MYSTD", major=2, min_minor=3),
            projects=(_project(name="MYSTD", major=2, minor=3),),
        )
        std = build_library_view(registry).packs[0].versions[0].satisfies[0]

        assert std.status == "satisfied"
        assert std.label == "Satisfied"

    def test_satisfied_when_project_minor_exceeds_minimum(self) -> None:
        registry = _registry(
            _pack(name="MYSTD", major=2, min_minor=3),
            projects=(_project(name="MYSTD", major=2, minor=5),),
        )
        std = build_library_view(registry).packs[0].versions[0].satisfies[0]

        assert std.status == "satisfied"

    def test_project_missing_when_no_such_standard(self) -> None:
        registry = _registry(_pack(name="MYSTD", major=2, min_minor=3))
        std = build_library_view(registry).packs[0].versions[0].satisfies[0]

        assert std.status == "project_missing"
        assert std.label == "Project not fetched"

    def test_project_missing_when_only_other_major_fetched(self) -> None:
        registry = _registry(
            _pack(name="MYSTD", major=2, min_minor=3),
            projects=(_project(name="MYSTD", major=1, minor=9),),
        )
        std = build_library_view(registry).packs[0].versions[0].satisfies[0]

        assert std.status == "project_missing"

    def test_project_too_old_when_minor_below_minimum(self) -> None:
        registry = _registry(
            _pack(name="MYSTD", major=2, min_minor=3),
            projects=(_project(name="MYSTD", major=2, minor=2),),
        )
        std = build_library_view(registry).packs[0].versions[0].satisfies[0]

        assert std.status == "project_too_old"
        assert std.label == "Project too old"


class TestProjectReverseLinks:
    def _project_view(self, registry: Registry, key: str):
        views = {p.key: p for p in build_library_view(registry).projects}
        return views[key]

    def test_project_lists_packs_satisfying_its_standard(self) -> None:
        registry = _registry(
            _pack(url="https://host/widgets", version=1, name="MYSTD", major=2),
            projects=(_project(name="MYSTD", major=2),),
        )
        project = self._project_view(registry, "MYSTD-2")

        assert len(project.packs) == 1
        assert project.packs[0].url == "https://host/widgets"
        assert project.packs[0].versions == [1]

    def test_project_with_no_satisfying_packs_has_empty_list(self) -> None:
        registry = _registry(
            _pack(name="MYSTD", major=2),
            projects=(_project(name="OTHER", major=1),),
        )
        project = self._project_view(registry, "OTHER-1")

        assert project.packs == []

    def test_packs_grouped_by_url_with_versions_descending(self) -> None:
        registry = _registry(
            _pack(url="https://host/widgets", version=1, name="MYSTD", major=2),
            _pack(url="https://host/widgets", version=3, name="MYSTD", major=2),
            _pack(url="https://host/widgets", version=2, name="MYSTD", major=2),
            projects=(_project(name="MYSTD", major=2),),
        )
        project = self._project_view(registry, "MYSTD-2")

        assert len(project.packs) == 1
        assert project.packs[0].versions == [3, 2, 1]

    def test_reverse_link_filters_to_matching_major_per_project(self) -> None:
        # One pack URL whose later version moved to a different major: v1
        # satisfies MYSTD-2, v2 satisfies MYSTD-3. Each project must see only the
        # versions that satisfy its exact {name}-{major}.
        registry = _registry(
            _pack(url="https://host/widgets", version=1, name="MYSTD", major=2),
            _pack(url="https://host/widgets", version=2, name="MYSTD", major=3),
            projects=(
                _project(name="MYSTD", major=2),
                _project(name="MYSTD", major=3),
            ),
        )

        major2 = self._project_view(registry, "MYSTD-2")
        major3 = self._project_view(registry, "MYSTD-3")

        assert [p.versions for p in major2.packs] == [[1]]
        assert [p.versions for p in major3.packs] == [[2]]

    def test_pack_anchor_id_matches_pack_card_anchor(self) -> None:
        url = "https://host/widgets"
        registry = _registry(
            _pack(url=url, version=1, name="MYSTD", major=2),
            projects=(_project(name="MYSTD", major=2),),
        )
        project = self._project_view(registry, "MYSTD-2")

        assert project.packs[0].anchor_id == derive_url_slug(url)
