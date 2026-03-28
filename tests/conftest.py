import json
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: str, data: dict[str, Any]) -> str:
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Simple fixtures — mirrors tests/fixtures/simple_product.yaml
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_nc_file(tmp_path: Path) -> str:
    """A minimal netCDF file with a known, predictable structure."""
    path = str(tmp_path / "test.nc")
    with netCDF4.Dataset(path, "w") as nc:
        nc.title = "Test Product"
        nc.createDimension("time", None)  # unlimited
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01"
    return path


@pytest.fixture
def simple_definition_dict() -> dict[str, Any]:
    """A product definition dict that matches simple_nc_file exactly."""
    return {
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


@pytest.fixture
def simple_definition_file(tmp_path: Path, simple_definition_dict: dict[str, Any]) -> str:
    """Writes simple_definition_dict to a JSON file and returns its path."""
    return _write_json(str(tmp_path / "definition.json"), simple_definition_dict)


# ---------------------------------------------------------------------------
# Full fixtures — mirrors tests/fixtures/full_product.yaml
# ---------------------------------------------------------------------------


@pytest.fixture
def full_nc_file(tmp_path: Path) -> str:
    """
    A netCDF file that fully satisfies full_product.yaml.
    Optional elements (latitude variable, comment attribute) are intentionally
    absent to exercise the optional code paths.
    """
    path = str(tmp_path / "full_test.nc")
    with netCDF4.Dataset(path, "w") as nc:
        nc.title = "Full Test Product"
        nc.institution = "Test Institution"
        nc.source = "Synthetic test data"
        # comment omitted — it is optional

        nc.createDimension("time", None)   # unlimited
        nc.createDimension("sps32", 32)    # fixed size

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01"

        data_var = nc.createVariable("data", "f4", ("time", "sps32"))
        data_var.units = "K"
        data_var.long_name = "Temperature"
        data_var.valid_range = np.array([200.0, 400.0], dtype=np.float32)

        # latitude omitted — it is optional

        raw_group = nc.createGroup("raw_data")
        raw_group.comment = "Raw instrument data"
        raw_signal = raw_group.createVariable("raw_signal", "f4", ("time", "sps32"))
        raw_signal.units = "V"
        raw_signal.long_name = "Raw Signal"

    return path


@pytest.fixture
def full_definition_dict() -> dict[str, Any]:
    """Product definition dict that matches full_product.yaml."""
    return {
        "attributes": {
            "title": "Full Test Product",
            "institution": "Test Institution",
            "source": "<str: derived_from_file>",
            "comment": "<str: derived_from_file optional>",
        },
        "dimensions": [
            {"name": "time", "size": None},
            {"name": "sps32", "size": 32},
        ],
        "variables": [
            {
                "meta": {"name": "time", "datatype": "<float64>", "required": True},
                "dimensions": ["time"],
                "attributes": {"units": "seconds since 1970-01-01"},
            },
            {
                "meta": {"name": "data", "datatype": "<float32>", "required": True},
                "dimensions": ["time", "sps32"],
                "attributes": {
                    "units": "K",
                    "long_name": "Temperature",
                    "valid_range": "<Array[float32]: derived_from_file>",
                },
            },
            {
                "meta": {"name": "latitude", "datatype": "<float32>", "required": False},
                "dimensions": ["time"],
                "attributes": {
                    "units": "degrees_north",
                    "long_name": "Latitude",
                },
            },
        ],
        "groups": [
            {
                "meta": {"name": "raw_data", "required": True},
                "attributes": {"comment": "Raw instrument data"},
                "variables": [
                    {
                        "meta": {
                            "name": "raw_signal",
                            "datatype": "<float32>",
                            "required": True,
                        },
                        "dimensions": ["time", "sps32"],
                        "attributes": {"units": "V", "long_name": "Raw Signal"},
                    }
                ],
            }
        ],
    }


@pytest.fixture
def full_definition_file(tmp_path: Path, full_definition_dict: dict[str, Any]) -> str:
    """Writes full_definition_dict to a JSON file and returns its path."""
    return _write_json(str(tmp_path / "full_definition.json"), full_definition_dict)
