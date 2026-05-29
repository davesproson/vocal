"""The local registry of fetched projects and packs.

The registry is a single machine-local YAML file with two top-level keys:
``projects:`` and ``packs:``. It records what has been fetched onto this
machine so the resolver can find a project's importable module and a pack's
schema JSONs at check time.

- **Projects** are keyed by ``{name}-{major}``, so two majors of the same
  standard can be registered side by side. A project record carries the
  standard's identity (``name``, ``major``, ``minor``), the project's Python
  module name (``project_directory``), and a ``local_path`` pointing at the
  repo root (NOT the module subdirectory).
- **Packs** are keyed by ``(url, version)`` — both sourced from the pack's
  validated ``manifest.json`` — so multiple releases at the same base URL
  coexist. A pack record carries its :class:`~vocal.manifest.Manifest` and a
  ``local_path`` pointing at the ``v{Y}/`` directory containing
  ``manifest.json``.

Definitions sources are not a registered kind: they are loose directories of
YAMLs that exist only as input to ``vocal release``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import yaml

from vocal.manifest import Manifest, normalize_pack_url
from vocal.utils import cache_dir


def get_default_registry_path() -> str:
    """Return the default path to the local registry file."""
    return os.path.join(cache_dir(), "vocal-registry.yaml")


def project_key(name: str, major: int) -> str:
    """Return the registry key for a project: ``{name}-{major}``."""
    return f"{name}-{major}"


@dataclass
class Project:
    """A registered project: one major version of one standard.

    ``local_path`` is the repo root — the directory holding
    ``conventions.yaml`` and the project's ``project_directory`` module — not
    the module subdirectory itself.
    """

    name: str
    major: int
    minor: int
    project_directory: str
    local_path: str

    @property
    def key(self) -> str:
        return project_key(self.name, self.major)

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        return cls(
            name=d["name"],
            major=int(d["major"]),
            minor=int(d["minor"]),
            project_directory=d["project_directory"],
            local_path=d["local_path"],
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "major": self.major,
            "minor": self.minor,
            "project_directory": self.project_directory,
            "local_path": self.local_path,
        }


@dataclass
class Pack:
    """A registered pack release.

    The pack's identity (``url`` and ``version``) lives in its
    :class:`~vocal.manifest.Manifest`; the registry key is derived from it.
    ``local_path`` points at the cached ``v{Y}/`` directory containing
    ``manifest.json`` and the product schema JSONs.
    """

    manifest: Manifest
    local_path: str

    @property
    def url(self) -> str:
        return self.manifest.url

    @property
    def version(self) -> int:
        return self.manifest.version

    @property
    def key(self) -> tuple[str, int]:
        return (self.manifest.url, self.manifest.version)

    @classmethod
    def from_dict(cls, d: dict) -> "Pack":
        return cls(
            manifest=Manifest.from_dict(d["manifest"]),
            local_path=d["local_path"],
        )

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest.to_dict(),
            "local_path": self.local_path,
        }


@dataclass
class Registry:
    """The local registry of fetched projects and packs."""

    projects: dict[str, Project] = field(default_factory=dict)
    packs: dict[tuple[str, int], Pack] = field(default_factory=dict)

    def find_project(
        self, name: str, major: int, min_minor: int
    ) -> Optional[Project]:
        """Return the registered project for ``name``/``major`` whose ``minor``
        is at least ``min_minor``, or ``None``.

        Returns ``None`` both when no project of that name/major is registered
        and when the registered project's minor is below ``min_minor``.
        """
        project = self.projects.get(project_key(name, major))
        if project is not None and project.minor >= min_minor:
            return project
        return None

    def find_pack(self, url: str, version: int) -> Optional[Pack]:
        """Return the registered pack for the normalised ``url`` and ``version``,
        or ``None``."""
        return self.packs.get((normalize_pack_url(url), version))

    def add_project(self, project: Project, force: bool = False) -> None:
        """Add ``project`` to the registry, keyed by ``{name}-{major}``.

        Raises:
            ValueError: a project with the same key is already registered and
                ``force`` is False.
        """
        if project.key in self.projects and not force:
            raise ValueError(f"Project {project.key} is already registered.")
        self.projects[project.key] = project

    def add_pack(self, pack: Pack, force: bool = False) -> None:
        """Add ``pack`` to the registry, keyed by ``(url, version)``.

        Raises:
            ValueError: a pack with the same key is already registered and
                ``force`` is False.
        """
        if pack.key in self.packs and not force:
            raise ValueError(
                f"Pack {pack.url} version {pack.version} is already registered."
            )
        self.packs[pack.key] = pack

    def remove_project(self, key: str) -> None:
        del self.projects[key]

    def remove_pack(self, url: str, version: int) -> None:
        del self.packs[(normalize_pack_url(url), version)]

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Registry":
        d = d or {}
        projects = {
            k: Project.from_dict(v) for k, v in (d.get("projects") or {}).items()
        }
        packs: dict[tuple[str, int], Pack] = {}
        for entry in d.get("packs") or []:
            pack = Pack.from_dict(entry)
            packs[pack.key] = pack
        return cls(projects=projects, packs=packs)

    def to_dict(self) -> dict:
        return {
            "projects": {k: v.to_dict() for k, v in self.projects.items()},
            "packs": [p.to_dict() for p in self.packs.values()],
        }

    @classmethod
    def load(cls, path: str = get_default_registry_path()) -> "Registry":
        with open(path, "r") as f:
            return cls.from_dict(yaml.load(f, Loader=yaml.Loader))

    def save(self, path: str = get_default_registry_path()) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    @contextmanager
    def open(
        cls, path: str = get_default_registry_path()
    ) -> Generator["Registry", None, None]:
        registry = cls.load(path)
        yield registry
        registry.save(path)
