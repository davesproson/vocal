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

    def test_colliding_names_get_distinct_numeric_suffixes(self) -> None:
        # Names that slugify to the same stem must still each get their own page
        # filename, disambiguated by -2, -3, ... in input order.
        assert unique_slugs(["Air Temp", "air temp", "AIR  TEMP"]) == [
            "air-temp",
            "air-temp-2",
            "air-temp-3",
        ]

    def test_unsafe_characters_collapse_to_a_safe_stem(self) -> None:
        # Anything outside [a-z0-9] collapses to single hyphens, trimmed at the
        # ends, so the stem is always a safe filename.
        assert unique_slugs(["Sea/Surface Temp! (°C)"]) == ["sea-surface-temp-c"]

    def test_traversal_style_name_collapses_to_a_safe_stem(self) -> None:
        # A path-traversal-style name cannot escape the output directory: the
        # slug carries no slashes or dots, just a safe stem.
        slug = unique_slugs(["../../etc/passwd"])[0]
        assert slug == "etc-passwd"
        assert "/" not in slug and ".." not in slug

    def test_name_of_only_unsafe_characters_falls_back_to_product(self) -> None:
        # A name that slugifies to nothing still yields a usable stem.
        assert unique_slugs(["///", "!!!"]) == ["product", "product-2"]
