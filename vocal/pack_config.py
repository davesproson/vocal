"""Read and validate a definitions repo's ``pack.yaml``.

``pack.yaml`` at a definitions repo's root makes the *product axis*
self-describing from its source. It carries the ``filecodec`` (the
placeholder→regex map used to route a file to a product, which used to live in
the project) and the pack's advisory ``satisfies_standards`` assertions. It may
also pin the pack's hosting ``url``.

The file schema is::

    filecodec:
      date:
        regex: '\\d{8}'
      platform:
        regex: '[a-z]+'
    satisfies_standards:        # optional, list of constraint strings
      - MYSTD-2.4+
    url: https://host/packs     # optional

``vocal release`` reads this (slice #57) to build a :class:`~vocal.manifest.Manifest`,
auto-recording the validating standard alongside the author's declared extras.
This module is the loader/validator only — it mirrors
:class:`~vocal.conventions_file.ConventionsFile` and does no network or netCDF
access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml

from vocal.exceptions import VocalError
from vocal.versioning import InvalidVersion, VersionConstraint

PACK_CONFIG_FILENAME = "pack.yaml"


class InvalidPackConfig(VocalError):
    """Raised when ``pack.yaml`` is missing, unreadable, or malformed."""


def pack_config_path(repo_path: str) -> str:
    """Return the path to ``pack.yaml`` within ``repo_path``."""
    return os.path.join(repo_path, PACK_CONFIG_FILENAME)


@dataclass
class PackConfig:
    """The contents of a definitions repo's ``pack.yaml``.

    ``filecodec`` is required (a pack with no routing codec can match no file);
    ``satisfies_standards`` and ``url`` are optional.
    """

    filecodec: dict[str, dict[str, Any]]
    satisfies_standards: tuple[VersionConstraint, ...] = ()
    url: str | None = None

    @classmethod
    def load(cls, repo_path: str) -> "PackConfig":
        """Load and validate ``<repo_path>/pack.yaml``.

        Raises:
            InvalidPackConfig: the file is missing, not valid YAML, or does not
                carry a well-formed ``filecodec`` / ``satisfies_standards`` /
                ``url``.
        """
        path = pack_config_path(repo_path)

        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
        except FileNotFoundError:
            raise InvalidPackConfig(
                f"{PACK_CONFIG_FILENAME} not found at {path}.",
                hint="The directory does not look like a vocal definitions repo.",
            )
        except yaml.YAMLError as e:
            raise InvalidPackConfig(f"{PACK_CONFIG_FILENAME} is not valid YAML: {e}")

        if not isinstance(raw, dict):
            raise InvalidPackConfig(
                f"{PACK_CONFIG_FILENAME} must be a mapping with at least a "
                "'filecodec' block."
            )

        filecodec = _parse_filecodec(raw.get("filecodec"))
        satisfies_standards = _parse_satisfies_standards(raw.get("satisfies_standards"))

        url = raw.get("url")
        if url is not None and not isinstance(url, str):
            raise InvalidPackConfig(
                f"{PACK_CONFIG_FILENAME} 'url' must be a string."
            )

        return cls(
            filecodec=filecodec,
            satisfies_standards=satisfies_standards,
            url=url,
        )


def _parse_filecodec(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise InvalidPackConfig(
            f"{PACK_CONFIG_FILENAME} is missing a 'filecodec' mapping of "
            "placeholder names to {'regex': ...} entries."
        )
    codec: dict[str, dict[str, Any]] = {}
    for name, entry in value.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("regex"), str):
            raise InvalidPackConfig(
                f"{PACK_CONFIG_FILENAME} filecodec entry {name!r} must be a "
                "mapping with a string 'regex'."
            )
        codec[str(name)] = dict(entry)
    return codec


def _parse_satisfies_standards(value: Any) -> tuple[VersionConstraint, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidPackConfig(
            f"{PACK_CONFIG_FILENAME} 'satisfies_standards' must be a list of "
            "constraint strings such as 'MYSTD-2.4+'."
        )
    constraints: list[VersionConstraint] = []
    for entry in value:
        if not isinstance(entry, str):
            raise InvalidPackConfig(
                f"{PACK_CONFIG_FILENAME} 'satisfies_standards' entries must be "
                f"constraint strings, got {type(entry).__name__}."
            )
        try:
            constraints.append(VersionConstraint.parse(entry))
        except InvalidVersion as e:
            raise InvalidPackConfig(
                f"{PACK_CONFIG_FILENAME} 'satisfies_standards' entry {entry!r} "
                f"is not a valid constraint: {e}"
            )
    return tuple(constraints)
