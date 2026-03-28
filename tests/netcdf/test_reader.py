import json
from pathlib import Path
from typing import Any

import netCDF4
import pytest

from vocal.netcdf import NetCDFReader


@pytest.fixture
def simple_nc_file(tmp_path: Path) -> str:
    path = str(tmp_path / "reader_test.nc")
    with netCDF4.Dataset(path, "w") as nc:
        nc.title = "Reader Test"
        nc.createDimension("time", None)  # unlimited
        nc.createDimension("sps32", 32)
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01"
        time_var.long_name = "Time"
        data_var = nc.createVariable("data", "f4", ("time", "sps32"))
        data_var.long_name = "Some Data"
    return path


@pytest.fixture
def grouped_nc_file(tmp_path: Path) -> str:
    path = str(tmp_path / "grouped.nc")
    with netCDF4.Dataset(path, "w") as nc:
        nc.title = "Grouped File"
        nc.createDimension("time", None)
        grp = nc.createGroup("raw_group")
        grp.createDimension("sps8", 8)
        v = grp.createVariable("raw_data", "f4", ("sps8",))
        v.units = "V"
    return path


class TestNetCDFReaderAttributes:
    def test_parses_global_attribute(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        assert reader.dict["attributes"]["title"] == "Reader Test"

    def test_file_pattern_in_meta(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        assert reader.dict["meta"]["file_pattern"] == "reader_test.nc"


class TestNetCDFReaderDimensions:
    def test_parses_dimensions(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        names = [d["name"] for d in reader.dict["dimensions"]]
        assert "time" in names
        assert "sps32" in names

    def test_unlimited_dimension_has_none_size(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        time_dim = next(d for d in reader.dict["dimensions"] if d["name"] == "time")
        assert time_dim["size"] is None

    def test_fixed_dimension_has_correct_size(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        sps_dim = next(d for d in reader.dict["dimensions"] if d["name"] == "sps32")
        assert sps_dim["size"] == 32


class TestNetCDFReaderVariables:
    def test_parses_variable_names(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        var_names = [v["meta"]["name"] for v in reader.dict["variables"]]
        assert "time" in var_names
        assert "data" in var_names

    def test_variable_dtype_mapped_to_spec_string(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        time_var = next(v for v in reader.dict["variables"] if v["meta"]["name"] == "time")
        assert time_var["meta"]["datatype"] == "<float64>"

    def test_float32_variable_dtype(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        data_var = next(v for v in reader.dict["variables"] if v["meta"]["name"] == "data")
        assert data_var["meta"]["datatype"] == "<float32>"

    def test_variable_attributes_parsed(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        time_var = next(v for v in reader.dict["variables"] if v["meta"]["name"] == "time")
        assert time_var["attributes"]["units"] == "seconds since 1970-01-01"
        assert time_var["attributes"]["long_name"] == "Time"


class TestNetCDFReaderGroups:
    def test_groups_parsed(self, grouped_nc_file: str) -> None:
        reader = NetCDFReader(grouped_nc_file)
        assert "groups" in reader.dict
        group_names = [g["meta"]["name"] for g in reader.dict["groups"]]
        assert "raw_group" in group_names

    def test_group_variables_parsed(self, grouped_nc_file: str) -> None:
        reader = NetCDFReader(grouped_nc_file)
        raw_group = next(g for g in reader.dict["groups"] if g["meta"]["name"] == "raw_group")
        var_names = [v["meta"]["name"] for v in raw_group["variables"]]
        assert "raw_data" in var_names

    def test_no_groups_key_absent_when_empty(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        assert "groups" not in reader.dict


class TestNetCDFReaderJson:
    def test_json_property_is_valid_json(self, simple_nc_file: str) -> None:
        reader = NetCDFReader(simple_nc_file)
        data: dict[str, Any] = json.loads(reader.json)
        assert isinstance(data, dict)
        assert "attributes" in data
