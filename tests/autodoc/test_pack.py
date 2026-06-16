"""Unit tests for the pack-index assembly: the slug/href helper and build step.

The slug helper is the single source of truth for both the index's link targets
and the product pages' on-disk filenames (so the two cannot drift); these tests
pin its observable mapping — readable names to ``[a-z0-9-]`` stems, in stable
order. The deeper collision/sanitisation guarantees are proven in the
failure-mode slice (#66).
"""

from vocal.autodoc.pack import unique_slugs


class TestUniqueSlugs:
    def test_readable_name_maps_to_its_slug(self) -> None:
        assert unique_slugs(["Air Temperature"]) == ["air-temperature"]

    def test_ordering_is_stable(self) -> None:
        assert unique_slugs(["beta", "alpha", "gamma"]) == [
            "beta",
            "alpha",
            "gamma",
        ]
