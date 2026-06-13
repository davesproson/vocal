"""Initialise a vocal project."""

import os
import typer

from vocal.conventions_file import ConventionsFile

PACKAGE_INIT = """
from . import defaults
from . import models
"""

ATTRIBUTES_TEMPLATE = """
from pydantic import Field, BaseModel

class {attr_type}Attributes(BaseModel):
    model_config = ConfigDict(
        # Configuration options here
        title='{attr_type} Attributes'
    )

    # Add your attributes here, e.g.
    #
    # my_attribute: str = Field(
    #   description='A description of my attribute',
    #   example='my_attribute_value'
    # )
"""

ATTRIBUTES_INIT = """
from .global_attributes import GlobalAttributes
from .variable_attributes import VariableAttributes
from .group_attributes import GroupAttributes
"""

MODELS_INIT = """
from .dataset import Dataset, DatasetMeta
from .variable import Variable, VariableMeta
from .group import Group, GroupMeta
from .dimension import Dimension
"""

DEFAULTS = """
from vocal.types import *
from .attributes.constants import *

# Add your attribute default values here:

default_global_attrs = {}
default_group_attrs = {}
default_variable_attrs = {}
"""

DATASET_CODE = """
from typing import Optional
import netCDF4 # type: ignore

from pydantic import BaseModel, ConfigDict

from vocal.netcdf.mixins import DatasetNetCDFMixin
from vocal.field import Field

from ..attributes import GlobalAttributes

from .dimension import Dimension
from .group import Group
from .variable import Variable


class DatasetMeta(BaseModel):
    file_pattern: str = Field(description='Canonical filename pattern for this dataset')
    short_name: Optional[str] = Field(description='Unique short name for this dataset')
    description: Optional[str] = Field(description='Description of this dataset')
    references: Optional[list[tuple[str, str]]] = Field(description='References for this dataset')


class Dataset(BaseModel, DatasetNetCDFMixin):
    model_config = ConfigDict(
        title='Dataset Schema'
    )

    meta: DatasetMeta
    attributes: GlobalAttributes
    dimensions: Optional[list[Dimension]]
    groups: Optional[list[Group]]
    variables: list[Variable]
"""

GROUP_CODE = """
from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, ConfigDict
from vocal.netcdf.mixins import GroupNetCDFMixin
from vocal.field import Field

from ..attributes import GroupAttributes

from .dimension import Dimension
from .variable import Variable

class GroupMeta(BaseModel):
    model_config = ConfigDict(
        title='Group Metadata'
    )

    name: str
    required: bool = True

class Group(BaseModel, GroupNetCDFMixin):
    model_config = ConfigDict(
        title='Group Schema'
    )

    meta: GroupMeta
    attributes: GroupAttributes
    dimensions: Optional[list[Dimension]]
    groups: Optional[list[Group]]
    variables: list[Variable]

Group.update_forward_refs()
"""

VARIABLE_CODE = """
import netCDF4 # type: ignore
import numpy as np
import numpy.typing
from pydantic import BaseModel
from typing import List

from vocal.netcdf.mixins import VariableNetCDFMixin
from vocal.field import Field

from ..attributes import VariableAttributes


class VariableMeta(BaseModel):
    datatype: str = Field(description='The type of the data')
    name: str
    required: bool = True


class Variable(BaseModel, VariableNetCDFMixin):
    meta: VariableMeta
    dimensions: Optional[List[str]]
    attributes: VariableAttributes
"""

DIMENSION_CODE = """
from pydantic import BaseModel
from typing import Union

from vocal.netcdf.mixins import DimensionNetCDFMixin

class Dimension(BaseModel, DimensionNetCDFMixin):
    name: str
    size: Union[int, None]
"""


def make_package_init(folder: str) -> None:
    filename = os.path.join(folder, "__init__.py")
    with open(filename, "w") as f:
        f.write(PACKAGE_INIT)


def make_defaults_module(folder: str) -> None:
    filename = os.path.join(folder, "defaults.py")
    with open(filename, "w") as f:
        f.write(DEFAULTS)


def make_attributes_init(folder: str) -> None:
    filename = os.path.join(folder, "__init__.py")
    with open(filename, "w") as f:
        f.write(ATTRIBUTES_INIT)


def make_attributes_file(folder: str, attr_type: str) -> None:
    filename = os.path.join(folder, f"{attr_type.lower()}_attributes.py")
    with open(filename, "w") as f:
        f.write(ATTRIBUTES_TEMPLATE.format(attr_type=attr_type))


def make_attributes_module(parent_folder: str) -> None:
    folder = os.path.join(parent_folder, "attributes")
    os.mkdir(folder)
    for attr_type in ("Global", "Group", "Variable"):
        make_attributes_file(folder, attr_type)
    make_attributes_init(folder)


def make_models_init(folder: str) -> None:
    filename = os.path.join(folder, "__init__.py")
    with open(filename, "w") as f:
        f.write(MODELS_INIT)


def make_model_file(folder: str, model: str) -> None:
    filename = os.path.join(folder, f"{model.lower()}.py")
    text = globals()[f"{model}_CODE"]
    with open(filename, "w") as f:
        f.write(text)


def make_models_module(parent_folder: str) -> None:
    folder = os.path.join(parent_folder, "models")
    os.mkdir(folder)
    for model in ("DIMENSION", "VARIABLE", "GROUP", "DATASET"):
        make_model_file(folder, model)
    make_models_init(folder)


def init_project(
    directory: str,
    name: str,
    major: int,
    minor: int,
    project_directory: str,
) -> None:
    """Scaffold a vocal project.

    Writes ``conventions.yaml`` at ``directory`` carrying the standard's
    identity and module layout, then scaffolds an importable Python package at
    ``<directory>/<project_directory>``.
    """
    os.makedirs(directory, exist_ok=True)

    ConventionsFile(
        name=name,
        major=major,
        minor=minor,
        project_directory=project_directory,
    ).write(directory)

    module_dir = os.path.join(directory, project_directory)
    os.makedirs(module_dir, exist_ok=True)

    make_attributes_module(module_dir)
    make_defaults_module(module_dir)
    make_models_module(module_dir)
    make_package_init(module_dir)


def command(
    directory: str = typer.Option(
        ".",
        "-d",
        "--directory",
        help="The directory in which to create the project. Defaults to cwd.",
    ),
    name: str = typer.Option(
        ...,
        "-n",
        "--name",
        help="The standard's name, e.g. 'MYSTD'.",
    ),
    major: int = typer.Option(
        1,
        "--major",
        help="The standard's major version. One project repo per major version.",
    ),
    minor: int = typer.Option(
        0,
        "--minor",
        help="The standard's current minor version.",
    ),
    project_directory: str = typer.Option(
        None,
        "-p",
        "--project-directory",
        help=(
            "The importable Python module directory to scaffold. "
            "Defaults to the lower-cased standard name."
        ),
    ),
) -> None:
    """Initialise a vocal project."""
    if project_directory is None:
        project_directory = name.lower()
    init_project(directory, name, major, minor, project_directory)
