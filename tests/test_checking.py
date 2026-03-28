from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pytest

from vocal.checking import (
    CheckError,
    CheckWarning,
    DimensionCollector,
    ElementDoesNotExist,
    InvalidPlaceholder,
    NotCheckedError,
    ProductChecker,
)


# ---------------------------------------------------------------------------
# State management and property accessors
# ---------------------------------------------------------------------------


class TestProductCheckerProperties:
    def setup_method(self) -> None:
        self.checker = ProductChecker(definition="")

    def test_passing_raises_before_checks(self) -> None:
        with pytest.raises(NotCheckedError):
            _ = self.checker.passing

    def test_errors_raises_before_checks(self) -> None:
        with pytest.raises(NotCheckedError):
            _ = self.checker.errors

    def test_warnings_raises_before_checks(self) -> None:
        with pytest.raises(NotCheckedError):
            _ = self.checker.warnings

    def test_comments_raises_before_checks(self) -> None:
        with pytest.raises(NotCheckedError):
            _ = self.checker.comments

    def test_passing_true_when_all_pass(self) -> None:
        self.checker._check("a check", passed=True)
        assert self.checker.passing is True

    def test_passing_false_when_any_fail(self) -> None:
        self.checker._check("passes", passed=True)
        self.checker._check("fails", passed=False, error=CheckError("oops", "/"))
        assert self.checker.passing is False

    def test_errors_only_includes_failed_checks(self) -> None:
        self.checker._check("passes", passed=True)
        err = CheckError("bad value", "/title")
        self.checker._check("fails", passed=False, error=err)
        errors = self.checker.errors
        assert len(errors) == 1
        assert errors[0].message == "bad value"

    def test_warnings_only_includes_warned_checks(self) -> None:
        self.checker._check("clean", passed=True)
        warned = self.checker._check("warned", passed=True)
        warned.has_warning = True
        warned.warning = CheckWarning("heads up", "/extra")
        assert len(self.checker.warnings) == 1
        assert self.checker.warnings[0].path == "/extra"


# ---------------------------------------------------------------------------
# Placeholder parsing
# ---------------------------------------------------------------------------


class TestGetAttributePropsFromPlaceholder:
    def setup_method(self) -> None:
        self.checker = ProductChecker(definition="")

    def test_required_attribute(self) -> None:
        props = self.checker.get_attribute_props_from_placeholder(
            "<str: derived_from_file>"
        )
        assert props.optional is False
        assert props.regex is None

    def test_optional_attribute(self) -> None:
        props = self.checker.get_attribute_props_from_placeholder(
            "<str: derived_from_file optional>"
        )
        assert props.optional is True

    def test_attribute_with_regex(self) -> None:
        props = self.checker.get_attribute_props_from_placeholder(
            r"<str: derived_from_file regex=\d{4}-\d{2}-\d{2}>"
        )
        assert props.regex is not None

    def test_invalid_placeholder_raises(self) -> None:
        with pytest.raises(InvalidPlaceholder):
            self.checker.get_attribute_props_from_placeholder("not_a_placeholder")


class TestGetTypeFromPlaceholder:
    def setup_method(self) -> None:
        self.checker = ProductChecker(definition="")

    def test_float32(self) -> None:
        dtype, container = self.checker.get_type_from_placeholder(
            "<float32: derived_from_file>"
        )
        assert dtype == np.dtype("float32")
        assert container is None

    def test_float64(self) -> None:
        dtype, container = self.checker.get_type_from_placeholder(
            "<float64: derived_from_file>"
        )
        assert dtype == np.dtype("float64")

    def test_array_placeholder(self) -> None:
        dtype, container = self.checker.get_type_from_placeholder(
            "<Array[float32]: derived_from_file>"
        )
        assert dtype == np.dtype("float32")
        assert container == "Array"

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            self.checker.get_type_from_placeholder("not_a_placeholder")


# ---------------------------------------------------------------------------
# Attribute value checking
# ---------------------------------------------------------------------------


class TestCheckAttributeValue:
    def setup_method(self) -> None:
        self.checker = ProductChecker(definition="")

    def test_matching_string_values_pass(self) -> None:
        self.checker.check_attribute_value("CF-1.8", "CF-1.8", path="/Conventions")
        assert self.checker.passing

    def test_mismatched_string_values_fail(self) -> None:
        self.checker.check_attribute_value("CF-1.8", "CF-1.6", path="/Conventions")
        assert not self.checker.passing

    def test_matching_list_values_pass(self) -> None:
        self.checker.check_attribute_value([1.0, 2.0], [1.0, 2.0], path="/coords")
        assert self.checker.passing

    def test_list_length_mismatch_fails(self) -> None:
        self.checker.check_attribute_value([1.0, 2.0], [1.0], path="/coords")
        assert not self.checker.passing

    def test_placeholder_does_not_raise_value_mismatch(self) -> None:
        # A placeholder defers to type checking; the "value check" check itself passes.
        self.checker.check_attribute_value(
            "<float32: derived_from_file>", np.float32(1.0), path="/fill"
        )
        value_errors = [
            e for e in self.checker.errors if "Unexpected value" in e.message
        ]
        assert not value_errors


# ---------------------------------------------------------------------------
# Attribute comparison
# ---------------------------------------------------------------------------


class TestCompareAttributes:
    def setup_method(self) -> None:
        self.checker = ProductChecker(definition="")

    def test_matching_attributes_pass(self) -> None:
        d = {"title": "My Product", "version": "1.0"}
        f = {"title": "My Product", "version": "1.0"}
        self.checker.compare_attributes(d, f)
        assert self.checker.passing
        assert not self.checker.warnings

    def test_missing_required_attribute_fails(self) -> None:
        d = {"title": "My Product"}
        f: dict[str, Any] = {}
        self.checker.compare_attributes(d, f)
        assert not self.checker.passing

    def test_missing_optional_attribute_passes(self) -> None:
        d = {"comment": "<str: derived_from_file optional>"}
        f: dict[str, Any] = {}
        self.checker.compare_attributes(d, f)
        assert self.checker.passing
        assert not self.checker.errors

    def test_extra_attribute_in_file_generates_warning(self) -> None:
        d = {"title": "My Product"}
        f = {"title": "My Product", "extra_attr": "unexpected"}
        self.checker.compare_attributes(d, f)
        assert self.checker.passing
        assert len(self.checker.warnings) == 1
        assert "extra_attr" in self.checker.warnings[0].message

    def test_wrong_attribute_value_fails(self) -> None:
        d = {"title": "Expected Title"}
        f = {"title": "Wrong Title"}
        self.checker.compare_attributes(d, f)
        assert not self.checker.passing


# ---------------------------------------------------------------------------
# Variable dtype checking
# ---------------------------------------------------------------------------


class TestCheckVariableDtype:
    def setup_method(self) -> None:
        self.checker = ProductChecker(definition="")

    def _var(self, datatype: str) -> dict[str, Any]:
        return {"meta": {"name": "v", "datatype": datatype}, "attributes": {}}

    def test_correct_dtype_passes(self) -> None:
        self.checker.check_variable_dtype(
            self._var("<float32>"), self._var("<float32>")
        )
        assert self.checker.passing

    def test_wrong_dtype_fails(self) -> None:
        self.checker.check_variable_dtype(
            self._var("<float32>"), self._var("<float64>"), path="/temperature"
        )
        assert not self.checker.passing
        assert len(self.checker.errors) == 1

    def test_identical_dtype_strings_add_no_comment(self) -> None:
        self.checker.check_variable_dtype(
            self._var("<float64>"), self._var("<float64>")
        )
        assert not self.checker.comments


# ---------------------------------------------------------------------------
# DimensionCollector
# ---------------------------------------------------------------------------


class TestDimensionCollector:
    def test_collects_root_dimensions(self) -> None:
        container = {
            "dimensions": [{"name": "time", "size": None}, {"name": "lat", "size": 10}]
        }
        dims = DimensionCollector().search(container)
        assert len(dims) == 2
        assert {"name": "time", "size": None} in dims

    def test_collects_nested_group_dimensions(self) -> None:
        container = {
            "dimensions": [{"name": "time", "size": None}],
            "groups": [
                {
                    "meta": {"name": "raw"},
                    "dimensions": [{"name": "sps32", "size": 32}],
                }
            ],
        }
        dims = DimensionCollector().search(container)
        assert len(dims) == 2

    def test_depth_zero_returns_empty(self) -> None:
        container = {"dimensions": [{"name": "time", "size": None}]}
        dims = DimensionCollector().search(container, depth=0)
        assert dims == []

    def test_depth_limits_nested_collection(self) -> None:
        container = {
            "dimensions": [{"name": "root_dim", "size": 10}],
            "groups": [
                {
                    "meta": {"name": "g"},
                    "dimensions": [{"name": "nested_dim", "size": 5}],
                }
            ],
        }
        dims = DimensionCollector().search(container, depth=1)
        assert len(dims) == 1
        assert dims[0]["name"] == "root_dim"


# ---------------------------------------------------------------------------
# Element lookup
# ---------------------------------------------------------------------------


class TestGetElement:
    def setup_method(self) -> None:
        self.checker = ProductChecker(definition="")

    def test_finds_existing_element(self) -> None:
        container = [
            {"meta": {"name": "time"}, "attributes": {}},
            {"meta": {"name": "lat"}, "attributes": {}},
        ]
        elem = self.checker.get_element("time", container)
        assert elem["meta"]["name"] == "time"

    def test_raises_for_missing_element(self) -> None:
        container = [{"meta": {"name": "time"}, "attributes": {}}]
        with pytest.raises(ElementDoesNotExist):
            self.checker.get_element("missing", container)


# ---------------------------------------------------------------------------
# compare_container integration (dict-level, no real files)
# ---------------------------------------------------------------------------


class TestCompareContainerIntegration:
    def setup_method(self) -> None:
        self.checker = ProductChecker(definition="")

    def test_fully_matching_container_passes(
        self, simple_definition_dict: dict[str, Any]
    ) -> None:
        file_dict: dict[str, Any] = {
            "meta": {"file_pattern": "test.nc"},
            "attributes": {"title": "Test Product"},
            "dimensions": [{"name": "time", "size": None}],
            "variables": [
                {
                    "meta": {"name": "time", "datatype": "<float64>"},
                    "dimensions": ["time"],
                    "attributes": {"units": "seconds since 1970-01-01"},
                }
            ],
        }
        self.checker.compare_container(simple_definition_dict, file_dict)
        assert self.checker.passing

    def test_missing_required_variable_fails(self) -> None:
        definition: dict[str, Any] = {
            "attributes": {},
            "variables": [
                {
                    "meta": {
                        "name": "temperature",
                        "datatype": "<float32>",
                        "required": True,
                    },
                    "dimensions": [],
                    "attributes": {},
                }
            ],
        }
        file_dict: dict[str, Any] = {"attributes": {}, "variables": []}
        self.checker.compare_container(definition, file_dict)
        assert not self.checker.passing

    def test_missing_optional_variable_passes(self) -> None:
        definition: dict[str, Any] = {
            "attributes": {},
            "variables": [
                {
                    "meta": {
                        "name": "optional_var",
                        "datatype": "<float32>",
                        "required": False,
                    },
                    "dimensions": [],
                    "attributes": {},
                }
            ],
        }
        file_dict: dict[str, Any] = {"attributes": {}, "variables": []}
        self.checker.compare_container(definition, file_dict)
        assert self.checker.passing

    def test_extra_variable_in_file_fails(self) -> None:
        # Unlike extra attributes (which only warn), extra variables not in the
        # definition cause a check failure.
        definition: dict[str, Any] = {"attributes": {}, "variables": []}
        file_dict: dict[str, Any] = {
            "attributes": {},
            "variables": [
                {
                    "meta": {"name": "extra_var", "datatype": "<float32>"},
                    "dimensions": [],
                    "attributes": {},
                }
            ],
        }
        self.checker.compare_container(definition, file_dict)
        assert not self.checker.passing

    def test_fixed_size_dimension_mismatch_fails(self) -> None:
        definition: dict[str, Any] = {
            "attributes": {},
            "dimensions": [{"name": "sps32", "size": 32}],
            "variables": [],
        }
        file_dict: dict[str, Any] = {
            "attributes": {},
            "dimensions": [{"name": "sps32", "size": 16}],  # wrong size
            "variables": [],
        }
        self.checker.compare_container(definition, file_dict)
        assert not self.checker.passing

    def test_missing_required_group_fails(self) -> None:
        definition: dict[str, Any] = {
            "attributes": {},
            "variables": [],
            "groups": [
                {
                    "meta": {"name": "required_group", "required": True},
                    "attributes": {},
                    "variables": [],
                }
            ],
        }
        file_dict: dict[str, Any] = {"attributes": {}, "variables": [], "groups": []}
        self.checker.compare_container(definition, file_dict)
        assert not self.checker.passing

    def test_missing_optional_group_passes(self) -> None:
        definition: dict[str, Any] = {
            "attributes": {},
            "variables": [],
            "groups": [
                {
                    "meta": {"name": "optional_group", "required": False},
                    "attributes": {},
                    "variables": [],
                }
            ],
        }
        file_dict: dict[str, Any] = {"attributes": {}, "variables": [], "groups": []}
        self.checker.compare_container(definition, file_dict)
        assert self.checker.passing


# ---------------------------------------------------------------------------
# End-to-end: ProductChecker.check() against real netCDF files
# ---------------------------------------------------------------------------


class TestProductCheckerCheckFile:
    def test_valid_file_passes(
        self, simple_nc_file: str, simple_definition_file: str
    ) -> None:
        checker = ProductChecker(definition=simple_definition_file)
        checker.check(simple_nc_file)
        assert checker.passing

    def test_wrong_attribute_value_fails(
        self, tmp_path: Path, simple_definition_file: str
    ) -> None:
        path = str(tmp_path / "wrong_title.nc")
        with netCDF4.Dataset(path, "w") as nc:
            nc.title = "WRONG TITLE"
            nc.createDimension("time", None)
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = "seconds since 1970-01-01"

        checker = ProductChecker(definition=simple_definition_file)
        checker.check(path)
        assert not checker.passing

    def test_missing_required_attribute_fails(
        self, tmp_path: Path, simple_definition_file: str
    ) -> None:
        path = str(tmp_path / "no_title.nc")
        with netCDF4.Dataset(path, "w") as nc:
            nc.createDimension("time", None)
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = "seconds since 1970-01-01"

        checker = ProductChecker(definition=simple_definition_file)
        checker.check(path)
        assert not checker.passing

    def test_extra_attribute_passes_with_warning(
        self, tmp_path: Path, simple_definition_file: str
    ) -> None:
        path = str(tmp_path / "extra_attr.nc")
        with netCDF4.Dataset(path, "w") as nc:
            nc.title = "Test Product"
            nc.undocumented = "surprise"
            nc.createDimension("time", None)
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = "seconds since 1970-01-01"

        checker = ProductChecker(definition=simple_definition_file)
        checker.check(path)
        assert checker.passing
        assert len(checker.warnings) > 0

    def test_wrong_variable_dtype_fails(
        self, tmp_path: Path, simple_definition_file: str
    ) -> None:
        path = str(tmp_path / "wrong_dtype.nc")
        with netCDF4.Dataset(path, "w") as nc:
            nc.title = "Test Product"
            nc.createDimension("time", None)
            # Definition expects float64 ("f8"); create float32 instead
            tv = nc.createVariable("time", "f4", ("time",))
            tv.units = "seconds since 1970-01-01"

        checker = ProductChecker(definition=simple_definition_file)
        checker.check(path)
        assert not checker.passing


# ---------------------------------------------------------------------------
# End-to-end: full fixture (groups, fixed dims, optional elements, placeholders)
# ---------------------------------------------------------------------------


class TestProductCheckerFullFixture:
    def test_valid_full_file_passes(
        self, full_nc_file: str, full_definition_file: str
    ) -> None:
        checker = ProductChecker(definition=full_definition_file)
        checker.check(full_nc_file)
        assert checker.passing

    def test_optional_variable_absent_passes(
        self, full_nc_file: str, full_definition_file: str
    ) -> None:
        # full_nc_file omits the optional 'latitude' variable by design
        checker = ProductChecker(definition=full_definition_file)
        checker.check(full_nc_file)
        assert checker.passing

    def test_optional_attribute_absent_passes(
        self, full_nc_file: str, full_definition_file: str
    ) -> None:
        # full_nc_file omits the optional 'comment' attribute by design
        checker = ProductChecker(definition=full_definition_file)
        checker.check(full_nc_file)
        assert checker.passing

    def test_required_derived_attribute_missing_fails(
        self, tmp_path: Path, full_definition_file: str
    ) -> None:
        path = str(tmp_path / "no_source.nc")
        with netCDF4.Dataset(path, "w") as nc:
            nc.title = "Full Test Product"
            nc.institution = "Test Institution"
            # source omitted — it is required (<str: derived_from_file>)
            nc.createDimension("time", None)
            nc.createDimension("sps32", 32)
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = "seconds since 1970-01-01"
            dv = nc.createVariable("data", "f4", ("time", "sps32"))
            dv.units = "K"
            dv.long_name = "Temperature"
            dv.valid_range = np.array([200.0, 400.0], dtype=np.float32)
            rg = nc.createGroup("raw_data")
            rg.comment = "Raw instrument data"
            rs = rg.createVariable("raw_signal", "f4", ("time", "sps32"))
            rs.units = "V"
            rs.long_name = "Raw Signal"

        checker = ProductChecker(definition=full_definition_file)
        checker.check(path)
        assert not checker.passing

    def test_required_group_missing_fails(
        self, tmp_path: Path, full_definition_file: str
    ) -> None:
        path = str(tmp_path / "no_group.nc")
        with netCDF4.Dataset(path, "w") as nc:
            nc.title = "Full Test Product"
            nc.institution = "Test Institution"
            nc.source = "Synthetic test data"
            nc.createDimension("time", None)
            nc.createDimension("sps32", 32)
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = "seconds since 1970-01-01"
            dv = nc.createVariable("data", "f4", ("time", "sps32"))
            dv.units = "K"
            dv.long_name = "Temperature"
            dv.valid_range = np.array([200.0, 400.0], dtype=np.float32)
            # raw_data group intentionally omitted

        checker = ProductChecker(definition=full_definition_file)
        checker.check(path)
        assert not checker.passing

    def test_fixed_dimension_size_mismatch_fails(
        self, tmp_path: Path, full_definition_file: str
    ) -> None:
        path = str(tmp_path / "wrong_dim.nc")
        with netCDF4.Dataset(path, "w") as nc:
            nc.title = "Full Test Product"
            nc.institution = "Test Institution"
            nc.source = "Synthetic test data"
            nc.createDimension("time", None)
            nc.createDimension("sps32", 16)  # definition expects 32
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = "seconds since 1970-01-01"
            dv = nc.createVariable("data", "f4", ("time", "sps32"))
            dv.units = "K"
            dv.long_name = "Temperature"
            dv.valid_range = np.array([200.0, 400.0], dtype=np.float32)
            rg = nc.createGroup("raw_data")
            rg.comment = "Raw instrument data"
            rs = rg.createVariable("raw_signal", "f4", ("time", "sps32"))
            rs.units = "V"
            rs.long_name = "Raw Signal"

        checker = ProductChecker(definition=full_definition_file)
        checker.check(path)
        assert not checker.passing
