"""Unit tests for vocal/utils/registry.py — the local registry of projects
and packs.

These exercise the public surface only: load/save round-tripping, the
``find_project`` minor-floor logic, and ``find_pack`` URL+version matching
(including URL normalisation). Packs are built through the ``manifest`` module
so the tests don't reach into the manifest's internal shape.
"""

from pathlib import Path

import pytest

from vocal.manifest import ManifestProduct, build_manifest
from vocal.utils.registry import Pack, Project, Registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project(
    name: str = "MYSTD",
    major: int = 2,
    minor: int = 3,
    project_directory: str = "mystd",
    local_path: str = "/cache/projects/mystd",
    url: str = "",
) -> Project:
    return Project(
        name=name,
        major=major,
        minor=minor,
        project_directory=project_directory,
        local_path=local_path,
        url=url,
    )


def _pack(
    url: str = "https://host/packs",
    version: int = 3,
    name: str = "MYSTD",
    major: int = 2,
    min_minor: int = 4,
    local_path: str = "/cache/packs/host-packs/v3",
) -> Pack:
    manifest = build_manifest(
        version=version,
        url=url,
        standard_name=name,
        standard_major=major,
        min_minor=min_minor,
        products=[
            ManifestProduct(
                name="foo", file_pattern="foo_{date}", schema="product_foo.json"
            )
        ],
    )
    return Pack(manifest=manifest, local_path=local_path)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


class TestKeys:
    def test_project_key_is_name_major(self) -> None:
        assert _project(name="MYSTD", major=2).key == "MYSTD-2"

    def test_pack_key_is_normalised_url_and_version(self) -> None:
        # The pack key uses the manifest's already-normalised URL.
        assert _pack(url="https://host/packs", version=3).key == (
            "https://host/packs",
            3,
        )


# ---------------------------------------------------------------------------
# add_project / add_pack
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_project_keyed_by_name_major(self) -> None:
        reg = Registry()
        reg.add_project(_project(name="MYSTD", major=2))
        assert "MYSTD-2" in reg.projects

    def test_two_majors_coexist(self) -> None:
        reg = Registry()
        reg.add_project(_project(name="MYSTD", major=2, minor=3))
        reg.add_project(_project(name="MYSTD", major=3, minor=0))
        assert set(reg.projects) == {"MYSTD-2", "MYSTD-3"}

    def test_add_duplicate_project_raises_without_force(self) -> None:
        reg = Registry()
        reg.add_project(_project(major=2, minor=3))
        with pytest.raises(ValueError):
            reg.add_project(_project(major=2, minor=5))

    def test_add_duplicate_project_force_overwrites(self) -> None:
        reg = Registry()
        reg.add_project(_project(major=2, minor=3))
        reg.add_project(_project(major=2, minor=5), force=True)
        assert reg.projects["MYSTD-2"].minor == 5

    def test_add_pack_keyed_by_url_version(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        assert ("https://host/packs", 3) in reg.packs

    def test_two_pack_versions_coexist(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(version=3))
        reg.add_pack(_pack(version=4))
        assert set(reg.packs) == {
            ("https://host/packs", 3),
            ("https://host/packs", 4),
        }

    def test_add_duplicate_pack_raises_without_force(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(version=3))
        with pytest.raises(ValueError):
            reg.add_pack(_pack(version=3))


# ---------------------------------------------------------------------------
# find_project
# ---------------------------------------------------------------------------


class TestFindProject:
    def test_found_when_minor_above_floor(self) -> None:
        reg = Registry()
        reg.add_project(_project(name="MYSTD", major=2, minor=5))
        found = reg.find_project("MYSTD", 2, 4)
        assert found is not None
        assert found.minor == 5

    def test_found_when_minor_at_floor(self) -> None:
        reg = Registry()
        reg.add_project(_project(name="MYSTD", major=2, minor=4))
        assert reg.find_project("MYSTD", 2, 4) is not None

    def test_none_when_minor_below_floor(self) -> None:
        reg = Registry()
        reg.add_project(_project(name="MYSTD", major=2, minor=3))
        assert reg.find_project("MYSTD", 2, 4) is None

    def test_none_on_major_mismatch(self) -> None:
        reg = Registry()
        reg.add_project(_project(name="MYSTD", major=2, minor=9))
        assert reg.find_project("MYSTD", 3, 0) is None

    def test_none_on_name_mismatch(self) -> None:
        reg = Registry()
        reg.add_project(_project(name="MYSTD", major=2, minor=9))
        assert reg.find_project("OTHER", 2, 0) is None

    def test_none_when_empty(self) -> None:
        assert Registry().find_project("MYSTD", 2, 0) is None


# ---------------------------------------------------------------------------
# find_project_by_url
# ---------------------------------------------------------------------------


class TestFindProjectByURL:
    def test_found_by_exact_url(self) -> None:
        reg = Registry()
        reg.add_project(_project(url="https://github.com/org/repo"))
        found = reg.find_project_by_url("https://github.com/org/repo")
        assert found is not None
        assert found.key == "MYSTD-2"

    def test_trailing_slash_matches(self) -> None:
        reg = Registry()
        reg.add_project(_project(url="https://github.com/org/repo"))
        assert reg.find_project_by_url("https://github.com/org/repo/") is not None

    def test_dot_git_matches(self) -> None:
        reg = Registry()
        reg.add_project(_project(url="https://github.com/org/repo.git"))
        assert reg.find_project_by_url("https://github.com/org/repo") is not None

    def test_host_case_matches(self) -> None:
        reg = Registry()
        reg.add_project(_project(url="https://github.com/org/repo"))
        assert reg.find_project_by_url("https://GitHub.com/org/repo") is not None

    def test_none_on_distinct_repo(self) -> None:
        reg = Registry()
        reg.add_project(_project(url="https://github.com/org/repo"))
        assert reg.find_project_by_url("https://github.com/org/other") is None

    def test_url_less_record_never_matches(self) -> None:
        # A legacy / locally-registered record carries no url; it must read as
        # "not fetched" rather than matching an empty query.
        reg = Registry()
        reg.add_project(_project(url=""))
        assert reg.find_project_by_url("https://github.com/org/repo") is None
        assert reg.find_project_by_url("") is None

    def test_none_when_empty(self) -> None:
        assert Registry().find_project_by_url("https://github.com/org/repo") is None


# ---------------------------------------------------------------------------
# find_pack
# ---------------------------------------------------------------------------


class TestFindPack:
    def test_found_by_url_and_version(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        assert reg.find_pack("https://host/packs", 3) is not None

    def test_none_on_version_mismatch(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        assert reg.find_pack("https://host/packs", 4) is None

    def test_none_on_url_mismatch(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        assert reg.find_pack("https://host/other", 3) is None

    def test_trailing_slash_normalised(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        assert reg.find_pack("https://host/packs/", 3) is not None

    def test_host_case_normalised(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        assert reg.find_pack("https://HOST/packs", 3) is not None

    def test_none_when_empty(self) -> None:
        assert Registry().find_pack("https://host/packs", 3) is None


# ---------------------------------------------------------------------------
# find_latest_pack
# ---------------------------------------------------------------------------


class TestFindLatestPack:
    def test_returns_highest_version_for_url(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        reg.add_pack(_pack(url="https://host/packs", version=5))
        reg.add_pack(_pack(url="https://host/packs", version=4))

        latest = reg.find_latest_pack("https://host/packs")
        assert latest is not None
        assert latest.version == 5

    def test_single_version_returned(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=2))
        latest = reg.find_latest_pack("https://host/packs")
        assert latest is not None
        assert latest.version == 2

    def test_isolates_by_url(self) -> None:
        # Two URLs with different versions: the lookup never crosses URLs.
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        reg.add_pack(_pack(url="https://host/other", version=9))

        packs = reg.find_latest_pack("https://host/packs")
        other = reg.find_latest_pack("https://host/other")
        assert packs is not None and other is not None
        assert packs.version == 3
        assert other.version == 9

    def test_url_normalised(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        reg.add_pack(_pack(url="https://host/packs", version=4))

        latest = reg.find_latest_pack("https://HOST/packs/")
        assert latest is not None
        assert latest.version == 4

    def test_none_when_url_absent(self) -> None:
        reg = Registry()
        reg.add_pack(_pack(url="https://host/packs", version=3))
        assert reg.find_latest_pack("https://host/other") is None

    def test_none_when_empty(self) -> None:
        assert Registry().find_latest_pack("https://host/packs") is None


# ---------------------------------------------------------------------------
# load / save round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_projects_and_packs(self, tmp_path: Path) -> None:
        reg = Registry()
        reg.add_project(_project(name="MYSTD", major=2, minor=3))
        reg.add_project(_project(name="OTHER", major=1, minor=0))
        reg.add_pack(_pack(url="https://host/packs", version=3))
        reg.add_pack(_pack(url="https://host/packs", version=4))

        path = str(tmp_path / "vocal-registry.yaml")
        reg.save(path)
        loaded = Registry.load(path)

        assert set(loaded.projects) == {"MYSTD-2", "OTHER-1"}
        assert loaded.projects["MYSTD-2"].minor == 3
        assert loaded.projects["MYSTD-2"].local_path == "/cache/projects/mystd"

        assert set(loaded.packs) == {
            ("https://host/packs", 3),
            ("https://host/packs", 4),
        }
        pack = loaded.find_pack("https://host/packs", 3)
        assert pack is not None
        assert pack.local_path == "/cache/packs/host-packs/v3"
        assert pack.manifest.requires_standard.min_minor == 4

    def test_round_trip_preserves_project_url(self, tmp_path: Path) -> None:
        reg = Registry()
        reg.add_project(_project(url="https://github.com/org/repo"))

        path = str(tmp_path / "vocal-registry.yaml")
        reg.save(path)
        loaded = Registry.load(path)

        assert loaded.projects["MYSTD-2"].url == "https://github.com/org/repo"
        assert loaded.find_project_by_url("https://github.com/org/repo/") is not None

    def test_legacy_record_without_url_loads_and_is_not_matched(
        self, tmp_path: Path
    ) -> None:
        # Simulate a registry written before the `url` field existed: its
        # project record has no `url` key at all. It must load (defaulting url to
        # empty) and read as "not fetched" by the lookup, self-healing on the
        # next fetch rather than crashing.
        import yaml

        legacy = {
            "projects": {
                "MYSTD-2": {
                    "name": "MYSTD",
                    "major": 2,
                    "minor": 3,
                    "project_directory": "mystd",
                    "local_path": "/cache/projects/mystd",
                }
            },
            "packs": [],
        }
        path = tmp_path / "vocal-registry.yaml"
        path.write_text(yaml.dump(legacy))

        loaded = Registry.load(str(path))
        assert loaded.projects["MYSTD-2"].url == ""
        assert loaded.find_project_by_url("https://github.com/org/repo") is None

    def test_round_trip_has_two_top_level_keys(self, tmp_path: Path) -> None:
        import yaml

        reg = Registry()
        reg.add_project(_project())
        reg.add_pack(_pack())

        path = str(tmp_path / "vocal-registry.yaml")
        reg.save(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        assert set(raw) == {"projects", "packs"}

    def test_load_empty_file_yields_empty_registry(self, tmp_path: Path) -> None:
        path = tmp_path / "vocal-registry.yaml"
        path.write_text("")
        loaded = Registry.load(str(path))
        assert loaded.projects == {}
        assert loaded.packs == {}

    def test_open_persists_changes(self, tmp_path: Path) -> None:
        path = str(tmp_path / "vocal-registry.yaml")
        Registry().save(path)

        with Registry.open(path) as reg:
            reg.add_project(_project())

        assert "MYSTD-2" in Registry.load(path).projects
