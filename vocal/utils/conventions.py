"""Read the ``Conventions`` global attribute from netCDF files.

This module is concerned only with reading the netCDF attribute. Parsing a
conventions token into a structured version lives in :mod:`vocal.versioning`;
the project's identity file lives in :mod:`vocal.conventions_file`.
"""

import netCDF4


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
