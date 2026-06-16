"""Unit tests for the pack-index assembly: the slug/href helper and build step.

The slug helper is the single source of truth for both the index's link targets
and the product pages' on-disk filenames (so the two cannot drift); these tests
pin its observable mapping — readable names to ``[a-z0-9-]`` stems, in stable
order. The deeper collision/sanitisation guarantees are proven in the
failure-mode slice (#66).
"""

from vocal.autodoc.pack import build_pack_doc, pack_heading, unique_slugs
from vocal.manifest import Manifest, ManifestProduct
from vocal.versioning import VersionConstraint


def _manifest() -> Manifest:
    return Manifest(
        version=3,
        url="https://host/packs/demo",
        filecodec={},
        satisfies_standards=(VersionConstraint("MYSTD", 2, 4),),
        products=(
            ManifestProduct(name="Alpha", file_pattern="a_{date}.nc", schema="v3/a.json"),
            ManifestProduct(name="Beta", file_pattern="b_{date}.nc", schema="v3/b.json"),
        ),
    )


class TestBuildPackDoc:
    def test_carries_pack_identity_and_standards(self) -> None:
        doc = build_pack_doc(_manifest(), ["alpha.html", "beta.html"])
        assert doc.url == "https://host/packs/demo"
        assert doc.version == 3
        assert doc.satisfies_standards == ["MYSTD-2.4+"]

    def test_one_entry_per_product_with_name_href_and_file_pattern(self) -> None:
        doc = build_pack_doc(_manifest(), ["alpha.html", "beta.html"])
        assert [(e.name, e.href, e.file_pattern) for e in doc.products] == [
            ("Alpha", "alpha.html", "a_{date}.nc"),
            ("Beta", "beta.html", "b_{date}.nc"),
        ]


class TestPackHeading:
    def test_uses_last_path_segment_of_url(self) -> None:
        assert pack_heading("https://host/packs/demo", "fallback") == "demo"

    def test_ignores_a_trailing_slash(self) -> None:
        assert pack_heading("https://host/packs/demo/", "fallback") == "demo"

    def test_falls_back_to_directory_name_when_url_is_bare(self) -> None:
        assert pack_heading("https://host", "my-pack-dir") == "my-pack-dir"


class TestUniqueSlugs:
    def test_readable_name_maps_to_its_slug(self) -> None:
        assert unique_slugs(["Air Temperature"]) == ["air-temperature"]

    def test_ordering_is_stable(self) -> None:
        assert unique_slugs(["beta", "alpha", "gamma"]) == [
            "beta",
            "alpha",
            "gamma",
        ]
