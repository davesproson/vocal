"""Read and write a project's ``conventions.yaml``.

``conventions.yaml`` at a project repo's root is the single source of truth for
the standard's identity (``name``, ``major``, ``minor``) and the project's
Python-module layout (``layout.project_directory``). It replaces the previous
``vocal.yaml`` skeleton. A project repo at one path corresponds to one major
version; there is no ``v{X}/`` subdirectory convention.

The file schema is::

    conventions:
      name: MYSTD
      major: 1
      minor: 2
    layout:
      project_directory: mystd            # importable Python module

The project repo carries no reference to product definitions or pack hosting —
those concerns live entirely on the definitions / pack side.

This module also owns the single project-import path: given a repo root, it
resolves ``<repo>/<project_directory>/__init__.py``, imports the package, and
enforces the project contract (``defaults``, ``models.Dataset``, ``filecodec``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType

import yaml

from vocal.exceptions import VocalError
from vocal.utils import import_project
from vocal.versioning import Version

CONVENTIONS_FILENAME = "conventions.yaml"

# The exports a project's importable package must expose to be usable by vocal.
REQUIRED_EXPORTS = ("defaults", "models", "filecodec")


class InvalidConventionsFile(VocalError):
    """Raised when ``conventions.yaml`` is missing, unreadable, or malformed."""


class MissingProjectExport(VocalError):
    """Raised when an imported project package is missing a required export."""


def conventions_path(repo_path: str) -> str:
    """Return the path to ``conventions.yaml`` within ``repo_path``."""
    return os.path.join(repo_path, CONVENTIONS_FILENAME)


@dataclass
class ConventionsFile:
    """The contents of a project's ``conventions.yaml``.

    Carries the standard's identity and the project's module layout only — no
    pack-side fields.
    """

    name: str
    major: int
    minor: int
    project_directory: str

    @property
    def version(self) -> Version:
        """The project's current version as a :class:`~vocal.versioning.Version`."""
        return Version(name=self.name, major=self.major, minor=self.minor)

    @classmethod
    def load(cls, repo_path: str) -> "ConventionsFile":
        """Load and validate ``<repo_path>/conventions.yaml``.

        Raises:
            InvalidConventionsFile: the file is missing, not valid YAML, or does
                not carry the required ``conventions`` / ``layout`` fields.
        """
        path = conventions_path(repo_path)

        try:
            with open(path, "r") as f:
                raw = yaml.load(f, Loader=yaml.Loader)
        except FileNotFoundError:
            raise InvalidConventionsFile(
                f"{CONVENTIONS_FILENAME} not found at {path}.",
                hint="The directory does not look like a vocal project.",
            )
        except yaml.YAMLError as e:
            raise InvalidConventionsFile(f"{CONVENTIONS_FILENAME} is not valid YAML: {e}")

        if not isinstance(raw, dict):
            raise InvalidConventionsFile(
                f"{CONVENTIONS_FILENAME} must be a mapping with 'conventions' "
                "and 'layout' blocks."
            )

        conventions = raw.get("conventions")
        layout = raw.get("layout")

        if not isinstance(conventions, dict):
            raise InvalidConventionsFile(
                f"{CONVENTIONS_FILENAME} is missing the 'conventions' block."
            )
        if not isinstance(layout, dict):
            raise InvalidConventionsFile(
                f"{CONVENTIONS_FILENAME} is missing the 'layout' block."
            )

        for key in ("name", "major", "minor"):
            if key not in conventions:
                raise InvalidConventionsFile(
                    f"{CONVENTIONS_FILENAME} 'conventions' block is missing "
                    f"required key: {key}"
                )

        if "project_directory" not in layout:
            raise InvalidConventionsFile(
                f"{CONVENTIONS_FILENAME} 'layout' block is missing required key: "
                "project_directory"
            )

        try:
            major = int(conventions["major"])
            minor = int(conventions["minor"])
        except (TypeError, ValueError):
            raise InvalidConventionsFile(
                f"{CONVENTIONS_FILENAME} 'major' and 'minor' must be integers."
            )

        return cls(
            name=str(conventions["name"]),
            major=major,
            minor=minor,
            project_directory=str(layout["project_directory"]),
        )

    def to_dict(self) -> dict:
        return {
            "conventions": {
                "name": self.name,
                "major": self.major,
                "minor": self.minor,
            },
            "layout": {
                "project_directory": self.project_directory,
            },
        }

    def write(self, repo_path: str) -> None:
        """Write this configuration to ``<repo_path>/conventions.yaml``."""
        with open(conventions_path(repo_path), "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)


def module_path(repo_path: str, conventions: ConventionsFile) -> str:
    """Return the path to the project's importable module directory."""
    return os.path.join(repo_path, conventions.project_directory)


def import_project_package(repo_path: str) -> ModuleType:
    """Import a project's Python package from a repo root.

    Reads ``conventions.yaml`` to find the module layout, requires
    ``<repo>/<project_directory>/__init__.py`` to exist, and imports it. This is
    the single project-import path — there are no conditionals on layout and no
    ``v{X}/`` fallback.

    Raises:
        InvalidConventionsFile: ``conventions.yaml`` is missing or malformed, or
            the declared ``project_directory`` is not an importable package.
    """
    conventions = ConventionsFile.load(repo_path)
    mod_dir = module_path(repo_path, conventions)

    if not os.path.isfile(os.path.join(mod_dir, "__init__.py")):
        raise InvalidConventionsFile(
            f"Project package not found: {os.path.join(mod_dir, '__init__.py')} "
            "does not exist.",
            hint=(
                f"'{conventions.project_directory}' (from {CONVENTIONS_FILENAME} "
                "layout.project_directory) must be an importable Python package."
            ),
        )

    return import_project(mod_dir)


def validate_project_contract(module: ModuleType) -> None:
    """Verify an imported project package exposes the required exports.

    The package must expose ``defaults``, ``models.Dataset``, and ``filecodec``.

    Raises:
        MissingProjectExport: naming the first missing export.
    """
    for export in REQUIRED_EXPORTS:
        if not hasattr(module, export):
            raise MissingProjectExport(
                f"Project package is missing required export: '{export}'.",
                hint=(
                    "A vocal project package must expose 'defaults', "
                    "'models.Dataset', and 'filecodec'."
                ),
            )

    if not hasattr(module.models, "Dataset"):
        raise MissingProjectExport(
            "Project package is missing required export: 'models.Dataset'.",
            hint=(
                "A vocal project package must expose 'defaults', "
                "'models.Dataset', and 'filecodec'."
            ),
        )
