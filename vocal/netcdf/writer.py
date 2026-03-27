import os
import json

from dataclasses import dataclass, field
from typing import Any, Union

import netCDF4  # type: ignore
import numpy as np
import pydantic
from pydantic.main import BaseModel

from ..types import np_invert

# from .dataset import Dataset

NCContainer = Union[netCDF4.Dataset, netCDF4.Group]


class NumpyEncoder(json.JSONEncoder):
    """
    A JSON encoder which wont fall over with basic numpy types we're
    likely to get from netCDF
    """

    def default(self, obj):
        if isinstance(obj, (np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        return super().default(obj)


@dataclass
class NetCDFReader:
    """
    Class which takes a netCDF file, and converts it to a pydantic model
    describing the
    """

    ncfile: str
    ncdict: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._parse()

    def _parse_variable(self, var: netCDF4.Variable) -> dict:
        """
        Returns a representation of a netCDF4 Variable
        """

        vout: dict[str, Any] = {}
        vout["meta"] = {"name": var.name, "datatype": np_invert[var.dtype]}

        vout["dimensions"] = var.dimensions

        vout["attributes"] = {}
        for attr in var.ncattrs():
            attr_val = getattr(var, attr)
            if isinstance(attr_val, np.ndarray):
                attr_val = list(attr_val)
            vout["attributes"][attr] = attr_val

        return vout

    def _parse_dimension(self, dim: netCDF4.Dimension) -> dict:
        """
        Returns a representation of a netCDF4 Dimension
        """

        return {"name": dim.name, "size": dim.size if not dim.isunlimited() else None}

    def _read_container(self, nc: NCContainer) -> dict:
        """
        Returns a representation of a netCDF container (either a Dataset or a
        Group)
        """

        ret_dict: dict[str, Any] = {}
        ret_dict["dimensions"] = []
        ret_dict["variables"] = []
        ret_dict["attributes"] = {}
        ret_dict["groups"] = []

        for attr in nc.ncattrs():
            ret_dict["attributes"][attr] = getattr(nc, attr)

        for dim in nc.dimensions.values():
            ret_dict["dimensions"].append(self._parse_dimension(dim))

        for var in nc.variables.values():
            ret_dict["variables"].append(self._parse_variable(var))

        for group in nc.groups.values():
            grp = self._read_container(group)
            grp["meta"] = {"name": group.name}
            ret_dict["groups"].append(grp)

        if not ret_dict["groups"]:
            del ret_dict["groups"]

        if not ret_dict["dimensions"]:
            del ret_dict["dimensions"]

        return ret_dict

    def _parse(self) -> None:
        """
        Load/parse a netCDF file, storing the structure of the file in the
        ncdict instance variable
        """

        with netCDF4.Dataset(self.ncfile) as nc:
            self.ncdict["meta"] = {"file_pattern": os.path.basename(self.ncfile)}
            self.ncdict.update(self._read_container(nc))

    @property
    def json(self):
        return json.dumps(self.ncdict, cls=NumpyEncoder)

    @property
    def dict(self):
        return self.ncdict

    def to_model(
        self, model: type[pydantic.BaseModel], validate: bool = True
    ) -> pydantic.BaseModel:
        """
        Attempt to return the internal netCDF file representation as a pydantic
        model.

        Args:
            model: the pydantic model type to return

        Kwargs:
            validate: if True (default) perform validation when constructing
                model.

        Returns:
            An instance of model, containing data from netCDF file
        """

        if validate:
            return model(**self.ncdict)

        return model.construct(**self.ncdict)


@dataclass
class NetCDFWriter:
    model: Union[BaseModel, dict]

    def write(self, ncfile):
        if isinstance(self.model, BaseModel):
            self.model.create_example_file(ncfile)

    def write_dimension(self, nc, dim):
        nc.create_dimension(dim["name"], dim["size"])

    def write_dataset(self, nc):
        for dim in self.model.dimensions:
            self.write_dimension(nc, dim)
