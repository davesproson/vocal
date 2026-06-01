"""Unit tests for the pure web view-model builder (vocal/web/models.py).

``build_library_view`` is driven directly against in-memory ``Registry``
objects, asserting the shape of the view models it returns. This slice covers
grouping packs by URL and the descending version sort; the three-state
requirement status is added (and tested) in a later slice.
"""

from vocal.manifest import build_manifest
from vocal.utils.registry import Pack, Registry
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
        standard_name=name,
        standard_major=major,
        min_minor=min_minor,
        products=[],
    )
    return Pack(manifest=manifest, local_path=f"/cache/{version}")


def _registry(*packs: Pack) -> Registry:
    registry = Registry()
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

    def test_version_carries_required_standard(self) -> None:
        registry = _registry(_pack(name="MYSTD", major=2, min_minor=4))
        view = build_library_view(registry)

        version = view.packs[0].versions[0]
        assert version.requires_standard == "MYSTD-2"
        assert version.requires_min_minor == 4
