import json

import pytest

from vocal.manifest import (
    SCHEMA_VERSION,
    InvalidManifest,
    InvalidPackURL,
    Manifest,
    ManifestProduct,
    PackInconsistent,
    UnsupportedManifestVersion,
    build_manifest,
    load_manifest,
    normalize_pack_url,
    versioned_dirname,
)
from vocal.versioning import VersionConstraint


# A filecodec mirroring the project-side shape: placeholder -> {"regex": ...}.
FILECODEC = {
    "date": {"regex": r"\d{8}"},
    "platform": {"regex": r"[a-z]+"},
}


def valid_manifest_dict(**overrides):
    data = {
        "schema_version": 1,
        "version": 3,
        "url": "https://host/packs",
        "requires_standard": {"name": "MYSTD", "major": 2, "min_minor": 4},
        "products": [
            {
                "name": "foo",
                "file_pattern": "foo_{date}.nc",
                "schema": "product_foo.json",
            }
        ],
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# normalize_pack_url
# ---------------------------------------------------------------------------


class TestNormalizePackURL:
    def test_lowercases_scheme_and_host(self) -> None:
        assert normalize_pack_url("HTTPS://Host/Packs") == "https://host/Packs"

    def test_leaves_path_case_untouched(self) -> None:
        assert normalize_pack_url("https://host/Packs/Foo") == "https://host/Packs/Foo"

    def test_strips_single_trailing_slash(self) -> None:
        assert normalize_pack_url("https://host/packs/") == "https://host/packs"

    def test_trailing_and_no_trailing_slash_collapse(self) -> None:
        assert normalize_pack_url("https://host/packs/") == normalize_pack_url(
            "https://host/packs"
        )

    def test_strips_surrounding_whitespace(self) -> None:
        assert normalize_pack_url("  https://host/packs  ") == "https://host/packs"

    @pytest.mark.parametrize(
        "bad",
        [
            "https://host/packs?foo=bar",
            "https://host/packs#frag",
            "https://host/packs/?x=1",
        ],
    )
    def test_rejects_query_or_fragment(self, bad: str) -> None:
        with pytest.raises(InvalidPackURL):
            normalize_pack_url(bad)


# ---------------------------------------------------------------------------
# Manifest model + serialisation round-trip
# ---------------------------------------------------------------------------


class TestManifestModel:
    def test_carries_all_fields(self) -> None:
        m = Manifest.from_dict(valid_manifest_dict())
        assert m.schema_version == 1
        assert m.version == 3
        assert m.url == "https://host/packs"
        assert m.requires_standard == VersionConstraint("MYSTD", 2, 4)
        assert m.products == (
            ManifestProduct("foo", "foo_{date}.nc", "product_foo.json"),
        )

    def test_to_dict_round_trips_through_from_dict(self) -> None:
        m = Manifest.from_dict(valid_manifest_dict())
        assert Manifest.from_dict(m.to_dict()) == m

    def test_to_json_round_trips(self) -> None:
        m = Manifest.from_dict(valid_manifest_dict())
        assert Manifest.from_dict(json.loads(m.to_json())) == m

    def test_to_dict_shape(self) -> None:
        m = Manifest.from_dict(valid_manifest_dict())
        d = m.to_dict()
        assert d["schema_version"] == 1
        assert d["requires_standard"] == {"name": "MYSTD", "major": 2, "min_minor": 4}
        assert d["products"] == [
            {"name": "foo", "file_pattern": "foo_{date}.nc", "schema": "product_foo.json"}
        ]


# ---------------------------------------------------------------------------
# from_dict validation
# ---------------------------------------------------------------------------


class TestFromDictValidation:
    def test_rejects_non_object(self) -> None:
        with pytest.raises(InvalidManifest):
            Manifest.from_dict([1, 2, 3])

    @pytest.mark.parametrize(
        "missing", ["schema_version", "version", "url", "requires_standard", "products"]
    )
    def test_rejects_missing_required_field(self, missing: str) -> None:
        data = valid_manifest_dict()
        del data[missing]
        with pytest.raises(InvalidManifest):
            Manifest.from_dict(data)

    def test_rejects_non_int_version(self) -> None:
        with pytest.raises(InvalidManifest):
            Manifest.from_dict(valid_manifest_dict(version="3"))

    def test_rejects_bool_version(self) -> None:
        # bool is an int subclass; it must not be accepted as a version.
        with pytest.raises(InvalidManifest):
            Manifest.from_dict(valid_manifest_dict(version=True))

    def test_rejects_unknown_schema_version_with_upgrade_hint(self) -> None:
        with pytest.raises(UnsupportedManifestVersion) as exc:
            Manifest.from_dict(valid_manifest_dict(schema_version=SCHEMA_VERSION + 1))
        assert "upgrade vocal" in str(exc.value).lower()

    def test_rejects_schema_version_below_one(self) -> None:
        with pytest.raises(InvalidManifest):
            Manifest.from_dict(valid_manifest_dict(schema_version=0))

    def test_rejects_malformed_requires_standard(self) -> None:
        with pytest.raises(InvalidManifest):
            Manifest.from_dict(
                valid_manifest_dict(requires_standard={"name": "MYSTD", "major": 2})
            )

    def test_rejects_products_not_a_list(self) -> None:
        with pytest.raises(InvalidManifest):
            Manifest.from_dict(valid_manifest_dict(products={"foo": "bar"}))

    def test_rejects_product_missing_field(self) -> None:
        with pytest.raises(InvalidManifest):
            Manifest.from_dict(
                valid_manifest_dict(products=[{"name": "foo", "schema": "foo.json"}])
            )

    def test_normalizes_url_on_load(self) -> None:
        m = Manifest.from_dict(valid_manifest_dict(url="HTTPS://Host/packs/"))
        assert m.url == "https://host/packs"

    @pytest.mark.parametrize(
        "bad_path",
        [
            "../other.json",
            "sub/../../escape.json",
            "/abs/product.json",
            "https://host/product.json",
            "C:\\product.json",
        ],
    )
    def test_rejects_escaping_schema_path(self, bad_path: str) -> None:
        data = valid_manifest_dict(
            products=[{"name": "foo", "file_pattern": "x", "schema": bad_path}]
        )
        with pytest.raises(InvalidManifest):
            Manifest.from_dict(data)

    def test_accepts_nested_relative_schema_path(self) -> None:
        data = valid_manifest_dict(
            products=[{"name": "foo", "file_pattern": "x", "schema": "sub/foo.json"}]
        )
        m = Manifest.from_dict(data)
        assert m.products[0].schema == "sub/foo.json"


# ---------------------------------------------------------------------------
# Product lookup
# ---------------------------------------------------------------------------


class TestProductLookup:
    def test_matches_templated_pattern(self) -> None:
        m = Manifest.from_dict(
            valid_manifest_dict(
                products=[
                    {
                        "name": "foo",
                        "file_pattern": "foo_{date}.nc",
                        "schema": "foo.json",
                    }
                ]
            )
        )
        product = m.find_product("foo_20260522.nc", FILECODEC)
        assert product is not None
        assert product.name == "foo"

    def test_returns_none_when_nothing_matches(self) -> None:
        m = Manifest.from_dict(
            valid_manifest_dict(
                products=[
                    {
                        "name": "foo",
                        "file_pattern": "foo_{date}.nc",
                        "schema": "foo.json",
                    }
                ]
            )
        )
        assert m.find_product("bar_nope.nc", FILECODEC) is None

    def test_matches_on_basename_only(self) -> None:
        m = Manifest.from_dict(
            valid_manifest_dict(
                products=[
                    {
                        "name": "foo",
                        "file_pattern": "foo_{date}.nc",
                        "schema": "foo.json",
                    }
                ]
            )
        )
        product = m.find_product("/some/dir/foo_20260522.nc", FILECODEC)
        assert product is not None

    def test_selects_correct_product_among_several(self) -> None:
        m = Manifest.from_dict(
            valid_manifest_dict(
                products=[
                    {
                        "name": "foo",
                        "file_pattern": "foo_{date}.nc",
                        "schema": "foo.json",
                    },
                    {
                        "name": "bar",
                        "file_pattern": "bar_{platform}.nc",
                        "schema": "bar.json",
                    },
                ]
            )
        )
        product = m.find_product("bar_aircraft.nc", FILECODEC)
        assert product is not None
        assert product.name == "bar"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_builds_without_project_import(self) -> None:
        m = build_manifest(
            version=3,
            url="https://host/packs",
            standard_name="MYSTD",
            standard_major=2,
            min_minor=4,
            products=[ManifestProduct("foo", "foo_{date}.nc", "foo.json")],
        )
        assert m.schema_version == SCHEMA_VERSION
        assert m.version == 3
        assert m.requires_standard == VersionConstraint("MYSTD", 2, 4)
        assert m.products == (ManifestProduct("foo", "foo_{date}.nc", "foo.json"),)

    def test_normalises_url(self) -> None:
        m = build_manifest(
            version=1,
            url="HTTPS://Host/packs/",
            standard_name="MYSTD",
            standard_major=2,
            min_minor=0,
            products=[],
        )
        assert m.url == "https://host/packs"

    def test_round_trips_through_serialise_deserialise(self) -> None:
        m = build_manifest(
            version=7,
            url="https://host/packs",
            standard_name="MYSTD",
            standard_major=2,
            min_minor=4,
            products=[ManifestProduct("foo", "foo_{date}.nc", "foo.json")],
        )
        assert Manifest.from_dict(json.loads(m.to_json())) == m


# ---------------------------------------------------------------------------
# load_manifest (filesystem) + PackInconsistent
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def _write(self, directory, data) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        path.write_text(json.dumps(data))
        return str(path)

    def test_loads_from_versioned_directory(self, tmp_path) -> None:
        path = self._write(tmp_path / "v3", valid_manifest_dict(version=3))
        m = load_manifest(path)
        assert m.version == 3

    def test_raises_pack_inconsistent_on_version_mismatch(self, tmp_path) -> None:
        path = self._write(tmp_path / "v3", valid_manifest_dict(version=4))
        with pytest.raises(PackInconsistent):
            load_manifest(path)

    def test_latest_directory_is_not_consistency_checked(self, tmp_path) -> None:
        # latest/ legitimately contains a manifest whose version != 'latest'.
        path = self._write(tmp_path / "latest", valid_manifest_dict(version=4))
        m = load_manifest(path)
        assert m.version == 4

    def test_rejects_invalid_json(self, tmp_path) -> None:
        directory = tmp_path / "v1"
        directory.mkdir()
        path = directory / "manifest.json"
        path.write_text("{ not valid json")
        with pytest.raises(InvalidManifest):
            load_manifest(str(path))

    def test_propagates_unsupported_schema_version(self, tmp_path) -> None:
        path = self._write(
            tmp_path / "v3",
            valid_manifest_dict(version=3, schema_version=SCHEMA_VERSION + 1),
        )
        with pytest.raises(UnsupportedManifestVersion):
            load_manifest(path)


def test_versioned_dirname() -> None:
    assert versioned_dirname(3) == "v3"
