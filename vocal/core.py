from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Tuple, Type
import pydantic
import yaml

from pydantic import ValidationError

from .manifest import ManifestProduct
from .utils import dataset_from_partial_yaml
from .utils import get_error_locs


def product_name(path: str) -> str:
    """Return the manifest product name for a definition file: its basename stem."""
    return os.path.basename(path).split(".")[0]


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

    @property
    def file_pattern(self) -> str:
        """The product's templated ``file_pattern``, read from its YAML ``meta``.

        The pattern is a template; the project's ``filecodec`` supplies the regex
        for each placeholder at check time. It is read straight from the source
        YAML so it does not depend on the project's ``DatasetMeta`` shape.
        """
        with open(self.path, "r") as f:
            raw = yaml.load(f, Loader=yaml.Loader)
        try:
            return raw["meta"]["file_pattern"]
        except (KeyError, TypeError) as err:
            raise ValueError(
                f"Product definition {self.path} is missing meta.file_pattern"
            ) from err

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
    a versioned pack.
    """

    model: Type[pydantic.BaseModel]
    version: int
    templates: TemplateSet = field(default_factory=TemplateSet.empty)
    definitions: list[ProductDefinition] = field(default_factory=list)

    def add_product(self, path: str) -> None:
        product = ProductDefinition(path, self.model, self.templates)
        self.definitions.append(product)

    @property
    def datasets(self) -> Iterator[Tuple[str, pydantic.BaseModel]]:
        for defn in self.definitions:
            yield product_name(defn.path), defn.construct()

    @property
    def manifest_products(self) -> list[ManifestProduct]:
        """The pack's product index: one :class:`ManifestProduct` per definition.

        Each entry's ``schema`` is the relative filename of the product JSON the
        pack writer emits alongside ``manifest.json`` — it agrees with the name
        yielded by :attr:`datasets`.
        """
        products: list[ManifestProduct] = []
        for defn in self.definitions:
            name = product_name(defn.path)
            products.append(
                ManifestProduct(
                    name=name, file_pattern=defn.file_pattern, schema=f"{name}.json"
                )
            )
        return products

    def validate_all(self) -> None:
        """Validate every product definition, raising on the first failure."""
        for defn in self.definitions:
            defn.validate()
