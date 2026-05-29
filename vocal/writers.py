from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ContextManager, Container, Generator, Protocol, TYPE_CHECKING

from pydantic import BaseModel

from .manifest import Manifest, versioned_dirname

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
class _TargetFolderManager:
    """A :class:`SupportsInFolder` that writes straight into a fixed directory.

    Unlike :class:`~vocal.utils.FolderManager` it adds no version subfolder — the
    pack writer owns the ``v{Y}/`` directory structure.
    """

    folder: str

    @contextmanager
    def in_folder(self) -> Generator[None, None, None]:
        os.makedirs(self.folder, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(self.folder)
            yield
        finally:
            os.chdir(cwd)


@dataclass
class PackWriter:
    """Write a pack release to disk.

    Emits the product instance JSONs, ``dataset_schema.json``, and
    ``manifest.json`` into ``<output_dir>/v{Y}/`` (where ``Y`` is the manifest's
    version), then refreshes ``<output_dir>/latest/`` to be a byte-equal copy of
    the highest-versioned ``v{N}/`` directory present after the write.

    The version directory is replaced wholesale if it already exists; callers are
    responsible for the ``--force`` gate that authorises overwriting a release.
    """

    product_collection: ProductCollection
    manifest: Manifest
    output_dir: str
    indent: int = 2

    def write(self) -> None:
        version_dir = os.path.join(
            self.output_dir, versioned_dirname(self.manifest.version)
        )
        if os.path.isdir(version_dir):
            shutil.rmtree(version_dir)
        os.makedirs(version_dir, exist_ok=True)

        folder_manager = _TargetFolderManager(version_dir)
        self._write_datasets(folder_manager)
        self._write_schema(folder_manager)
        self._write_manifest(version_dir)

        self._refresh_latest()

    def _write_datasets(self, folder_manager: SupportsInFolder) -> None:
        for name, dataset in self.product_collection.datasets:
            InstanceWriter(
                model=dataset,
                name=name,
                folder_manager=folder_manager,
                indent=self.indent,
            ).write()

    def _write_schema(self, folder_manager: SupportsInFolder) -> None:
        SchemaWriter(
            model=self.product_collection.model,
            name="dataset_schema",
            folder_manager=folder_manager,
            indent=self.indent,
        ).write()

    def _write_manifest(self, version_dir: str) -> None:
        with open(os.path.join(version_dir, "manifest.json"), "w") as f:
            f.write(self.manifest.to_json(indent=self.indent))

    def _refresh_latest(self) -> None:
        """Copy the highest-versioned release directory to ``latest/``."""
        highest = _highest_version_dir(self.output_dir)
        latest_dir = os.path.join(self.output_dir, "latest")
        if os.path.isdir(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(highest, latest_dir)


def _highest_version_dir(output_dir: str) -> str:
    """Return the path of the highest-numbered ``v{N}/`` directory in ``output_dir``."""
    best_version = -1
    best_dir = ""
    for entry in os.listdir(output_dir):
        if not os.path.isdir(os.path.join(output_dir, entry)):
            continue
        if not (entry.startswith("v") and entry[1:].isdigit()):
            continue
        version = int(entry[1:])
        if version > best_version:
            best_version = version
            best_dir = os.path.join(output_dir, entry)
    return best_dir
