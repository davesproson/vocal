from __future__ import annotations

import os

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, Tuple, Type
import pydantic

from pydantic import ValidationError

from .utils import FolderManager, dataset_from_partial_yaml
from .writers import VocabularyCreator
from .utils import get_error_locs

ATTRIBUTE_TYPES = ("group", "variable", "globals")

_templates: ContextVar[dict[str, Any]] = ContextVar(
    "_templates", default={i: {} for i in ATTRIBUTE_TYPES}
)


class SupportsCreateVocabulary(Protocol):
    def create_vocabulary(self) -> None: ...


class HasAttributesMembers(Protocol):
    GlobalAttributes: pydantic.BaseModel
    VariableAttributes: pydantic.BaseModel
    GroupAttributes: pydantic.BaseModel


class HasRequiredAttributesMembers(Protocol):
    default_globals_attrs: dict[str, Any]
    default_group_attrs: dict[str, Any]
    default_variable_attrs: dict[str, Any]


def register_defaults(name: str, mapping: dict) -> None:
    f"""
    Register a dictionary of default values
    """
    if name not in ATTRIBUTE_TYPES:
        raise ValueError("Invalid name")
    _templates.set({**_templates.get(), name: mapping})


def register_defaults_module(module: HasRequiredAttributesMembers) -> None:
    _templates.set({
        "globals": getattr(module, "default_global_attrs"),
        "group": getattr(module, "default_group_attrs"),
        "variable": getattr(module, "default_variable_attrs"),
    })


@dataclass
class ProductDefinition:
    """
    Represents a product definition, which can be used to create a dataset.
    """

    path: str
    model: Type[pydantic.BaseModel]

    def __call__(self) -> pydantic.BaseModel:
        """
        Return the dataset from the product definition.

        Returns:
            pydantic.BaseModel: The dataset, as a pydantic model.
        """
        return self._from_yaml(construct=False)

    def construct(self) -> pydantic.BaseModel:
        """
        Construct the dataset from the product definition,

        Returns:
            pydantic.BaseModel: The dataset, as a pydantic model.
        """
        return self._from_yaml(construct=True)

    def _from_yaml(self, construct: bool = True) -> pydantic.BaseModel:
        """
        Create a dataset from the product definition.

        Kwargs:
            construct (bool, optional): If true, construct the dataset. If false,
                return the dataset as a validated pydantic model. Defaults to True.

        Returns:
            pydantic.BaseModel: The dataset, as a pydantic model.
        """

        templates = _templates.get()
        return dataset_from_partial_yaml(
            self.path,
            variable_template=templates["variable"],
            group_template=templates["group"],
            globals_template=templates["globals"],
            model=self.model,
            construct=construct,
        )

    def create_example_file(self, nc_filename: str, find_coords: bool = False) -> None:
        """
        Create an example netCDF file from the product definition.

        Args:
            nc_filename (str): The name of the netCDF file to create.
            find_coords (bool, optional): Whether to find the coordinate variables
                in the dataset. Defaults to False.
        """

        coordinates = self.coordinates() if find_coords else None

        self().create_example_file(nc_filename, coordinates=coordinates)  # type: ignore

    def coordinates(self) -> str:
        """
        Find the coordinate variables in the dataset.

        Returns:
            str: A string of the coordinate variables.
        """
        dataset = self()

        _coords = {"latitude": None, "longitude": None, "altitude": None, "time": None}

        for var in dataset.variables:  # type: ignore
            for _crd in _coords.keys():
                if var.attributes.standard_name == _crd:
                    _coords[_crd] = var.meta.name

        coord_arr = [v for _, v in _coords.items() if v]

        coord_str = " ".join(coord_arr)  # type: ignore

        return coord_str

    def validate(self) -> None:
        """
        Validate the product definition, by trying to create it.
        """
        errors = False
        try:
            self()
        except ValidationError as err:
            # Create a dataset without validation, for error location
            nc_noval = self.construct()

            errors = True
            print(f"Error in dataset: {self.path}")

            # Get the error locations, and print them
            error_locs = get_error_locs(err, nc_noval)
            for err_loc, err_msg in zip(*error_locs):
                print(f"{err_loc}: {err_msg}")

        if errors:
            raise ValueError("Failed to validate dataset")


@dataclass
class ProductCollection:
    """
    Represents a collection of product definitions, which can be used to create
    versioned product definitions.
    """

    model: Type[pydantic.BaseModel]
    version: str
    vocab_creator: Optional[SupportsCreateVocabulary] = None
    definitions: list[ProductDefinition] = field(default_factory=list)

    def __post_init__(self):
        if self.vocab_creator is None:
            self.vocab_creator = VocabularyCreator(
                self, FolderManager, "products", self.version
            )

    def add_product(self, path: str) -> None:
        product = ProductDefinition(path, self.model)
        self.definitions.append(product)

    @property
    def datasets(self) -> Iterator[Tuple[str, pydantic.BaseModel]]:
        for defn in self.definitions:
            name = os.path.basename(defn.path).split(".")[0]
            yield name, defn.construct()

    def write_product_definitions(self):
        for defn in self.definitions:
            defn.validate()
        self.vocab_creator.create_vocabulary()
