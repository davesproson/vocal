from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ContextManager, Protocol, Container, Type, TYPE_CHECKING
from pydantic import BaseModel
import json
from dataclasses import dataclass

from .utils import FolderManager

if TYPE_CHECKING:
    from .core import ProductCollection


class SupportsInFolder(Protocol):
    def in_folder(self) -> ContextManager[None]: ...


@dataclass  # type: ignore # mypy issue with abstract dataclasses
class BaseWriter(ABC):
    """
    An abstract class, defining an interface for writing vocabs to file.
    """

    model: Any
    name: str
    folder_manager: SupportsInFolder
    indent: int = 2

    @property
    @abstractmethod
    def _json(self) -> str:
        """
        return a string json representation of model
        """
        return NotImplemented

    def write(self) -> None:
        """
        Write the model to file, as json, in a location given by folder_manager
        """
        _filename = f"{self.name}.json"
        with self.folder_manager.in_folder():
            mode = "w"
            with open(f"{self.name}.json", mode) as f:
                f.write(self._json)


class InstanceWriter(BaseWriter):
    """
    Implements a writer intended to write out model instances to file.
    """

    model: BaseModel

    @property
    def _json(self) -> str:
        _dict = self.model.model_dump(exclude_unset=True, by_alias=True, warnings=False)
        return json.dumps(_dict, indent=self.indent)


class SchemaWriter(BaseWriter):
    """
    Implements a writer intended to write model schema to file.
    """

    model: BaseModel

    @property
    def _json(self) -> str:
        return json.dumps(self.model.model_json_schema(), indent=self.indent)


class ContainerWriter(BaseWriter):
    """
    Implements a writer indended to write native containers to file.
    Assumes that these will be json serializable.
    """

    model: Container

    @property
    def _json(self) -> str:
        return json.dumps(self.model, indent=self.indent)


@dataclass
class VocabularyCreator:
    product_collection: ProductCollection
    folder_manager: Type[FolderManager]
    base_folder: str
    version: str

    def write_datasets(self, folder_manager: SupportsInFolder) -> None:
        """
        Write defined datasets to file.
        """
        for name, dataset in self.product_collection.datasets:
            writer = InstanceWriter(
                model=dataset, name=name, folder_manager=folder_manager
            )
            writer.write()

    def write_schemata(self, folder_manager: SupportsInFolder) -> None:
        """
        Write dataset, group, and variable schemata to file.
        """
        models = [self.product_collection.model]
        names = ["dataset_schema"]

        for model, name in zip(models, names):
            writer = SchemaWriter(model=model, name=name, folder_manager=folder_manager)
            writer.write()

    def create_vocabulary(self) -> None:
        """
        Create vocabularies.
        """
        versions = (f"v{self.version}", "latest")

        for version in versions:
            folder_manager = self.folder_manager(
                base_folder=self.base_folder, version=version
            )
            self.write_datasets(folder_manager)
            self.write_schemata(folder_manager)
