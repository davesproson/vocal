from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, Tuple, Type
import pydantic

from pydantic import ValidationError

from .utils import FolderManager, dataset_from_partial_yaml
from .writers import VocabularyCreator
from .utils import get_error_locs


class SupportsCreateVocabulary(Protocol):
    def create_vocabulary(self) -> None: ...


class HasAttributesMembers(Protocol):
    GlobalAttributes: pydantic.BaseModel
    VariableAttributes: pydantic.BaseModel
    GroupAttributes: pydantic.BaseModel


class HasRequiredAttributesMembers(Protocol):
    default_global_attrs: dict[str, Any]
    default_group_attrs: dict[str, Any]
    default_variable_attrs: dict[str, Any]


@dataclass(frozen=True)
class TemplateSet:
    """Attribute template dicts applied when loading a product definition from YAML."""

    globals: dict[str, Any] = field(default_factory=dict)
    group: dict[str, Any] = field(default_factory=dict)
    variable: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "TemplateSet":
        return cls()

    @classmethod
    def from_module(cls, module: HasRequiredAttributesMembers) -> "TemplateSet":
        return cls(
            globals=module.default_global_attrs,
            group=module.default_group_attrs,
            variable=module.default_variable_attrs,
        )

    def merge(self, override: "TemplateSet") -> "TemplateSet":
        """Return a new TemplateSet with override keys winning on conflict."""
        return TemplateSet(
            globals={**self.globals, **override.globals},
            group={**self.group, **override.group},
            variable={**self.variable, **override.variable},
        )


@dataclass
class ProductDefinition:
    """
    Represents a product definition, which can be used to create a dataset.
    """

    path: str
    model: Type[pydantic.BaseModel]
    templates: TemplateSet = field(default_factory=TemplateSet.empty)

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

        return dataset_from_partial_yaml(
            self.path,
            variable_template=self.templates.variable,
            group_template=self.templates.group,
            globals_template=self.templates.globals,
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
    templates: TemplateSet = field(default_factory=TemplateSet.empty)
    vocab_creator: Optional[SupportsCreateVocabulary] = None
    definitions: list[ProductDefinition] = field(default_factory=list)

    def __post_init__(self):
        if self.vocab_creator is None:
            self.vocab_creator = VocabularyCreator(
                self, FolderManager, "products", self.version
            )

    def add_product(self, path: str) -> None:
        product = ProductDefinition(path, self.model, self.templates)
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
