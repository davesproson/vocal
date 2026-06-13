"""Read the vocal-managed global attributes from netCDF files.

This module is concerned only with reading netCDF attributes. Parsing a
conventions token into a structured version lives in :mod:`vocal.versioning`;
the project's identity file lives in :mod:`vocal.conventions_file`; turning a
file's attributes plus the registry into a concrete target lives in
:mod:`vocal.resolution`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import netCDF4


@dataclass(frozen=True)
class FileConventions:
    """The vocal-managed global attributes read from a netCDF file.

    All four attributes are optional on the file. ``conventions`` is the raw
    ``Conventions`` string (which may also carry CF/ACDD tokens); the resolver
    tokenises and parses it. ``project_urls`` is the (possibly empty) list of
    repository URLs parsed from the whitespace-separated ``vocal_project_url``
    attribute — the authoritative set of mandatory standards. ``definitions_url``
    / ``definitions_version`` correspond to the (singular) ``vocal_definitions_url``
    / ``vocal_definitions_version`` attributes.
    """

    conventions: Optional[str]
    project_urls: list[str]
    definitions_url: Optional[str]
    definitions_version: Optional[int]


def read_file_conventions(ncfile: str) -> FileConventions:
    """Read the vocal-managed global attributes from ``ncfile``.

    Args:
        ncfile: the path to the netCDF file.

    Returns:
        a :class:`FileConventions`. ``conventions`` / ``definitions_url`` are
        ``None`` when absent; ``definitions_version`` is coerced to ``int`` when
        present. ``project_urls`` is the whitespace-split ``vocal_project_url``
        attribute — a one-element list for a single URL and an empty list when
        the attribute is absent.
    """
    with netCDF4.Dataset(ncfile) as nc:
        conventions = getattr(nc, "Conventions", None)
        raw_project_url = getattr(nc, "vocal_project_url", None)
        definitions_url = getattr(nc, "vocal_definitions_url", None)
        raw_version = getattr(nc, "vocal_definitions_version", None)

    return FileConventions(
        conventions=conventions,
        project_urls=raw_project_url.split() if raw_project_url is not None else [],
        definitions_url=definitions_url,
        definitions_version=int(raw_version) if raw_version is not None else None,
    )


def get_conventions_string(ncfile: str) -> str | None:
    """
    Get the conventions string from a netCDF file.

    Args:
        ncfile (str): The path to the netCDF file.

    Returns:
        str: The conventions string, or None if the attribute is absent.
    """
    with netCDF4.Dataset(ncfile) as nc:
        return getattr(nc, "Conventions", None)


def get_conventions_list(ncfile: str, delimiter: str = " ") -> list[str] | None:
    """
    Get the conventions list from a netCDF file.

    Args:
        ncfile (str): The path to the netCDF file.

    Returns:
        list[str]: The conventions list, or None if the attribute is absent.
    """
    conventions = get_conventions_string(ncfile)
    if conventions is None:
        return None

    return conventions.split(delimiter)
