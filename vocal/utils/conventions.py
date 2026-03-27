from dataclasses import dataclass
import os
import re
import netCDF4
import yaml


@dataclass
class Conventions:
    name: str
    major_version: int | None = None
    minor_version: int | None = None

    def __str__(self) -> str:
        return f"{self.name}-{self.major_version}.{self.minor_version}"

    @property
    def version_string(self) -> str:
        if self.major_version is None:
            return ""

        if self.minor_version is None:
            return f"v{self.major_version}"

        return f"v{self.major_version}.{self.minor_version}"


def get_conventions_string(ncfile: str) -> str | None:
    """
    Get the conventions string from a netCDF file.

    Args:
        ncfile (str): The path to the netCDF file.

    Returns:
        str: The conventions string.
    """
    with netCDF4.Dataset(ncfile) as nc:
        return getattr(nc, "Conventions", None)

    return None


def get_conventions_list(ncfile: str, delimiter: str = " ") -> list[str] | None:
    """
    Get the conventions list from a netCDF file.

    Args:
        ncfile (str): The path to the netCDF file.

    Returns:
        list[str]: The conventions list.
    """
    conventions = get_conventions_string(ncfile)
    if conventions is None:
        return None

    return conventions.split(delimiter)


def extract_conventions_info(
    ncfile: str, conventions_regex: str, name: str | None = None
) -> Conventions:
    """
    Extract conventions information from a netCDF file.

    Args:
        ncfile: the path to the netCDF file
        conventions_regex: the regular expression to use to extract the
            conventions information

    Returns:
        the extracted conventions information
    """
    with netCDF4.Dataset(ncfile, "r") as nc:
        conventions = nc.getncattr("Conventions")
        matches = re.search(conventions_regex, conventions)
        if not matches:
            raise ValueError("Unable to extract conventions information")

        groups = matches.groupdict()
        major = groups.get("major")
        minor = groups.get("minor")

        if major is not None:
            major = int(major)
        if minor is not None:
            minor = int(minor)

        return Conventions(
            name=name or groups["name"],
            major_version=major,
            minor_version=minor,
        )


def read_conventions_identifier(path: str) -> str:
    """
    Return the regular expression used to extract conventions information from
    a netCDF file.

    Args:
        path: the path to the project

    Returns:
        the regular expression
    """
    conventions_id_file = os.path.join(path, "conventions.yaml")
    if not os.path.isfile(conventions_id_file):
        raise ValueError(
            f"Unable to find conventions identifier file at {conventions_id_file}"
        )

    with open(conventions_id_file, "r") as f:
        y = yaml.load(f, Loader=yaml.Loader)

    name = y["conventions"]["name"]
    regex = rf".*?(?P<name>{name})-(?P<major>[0-9]+)\.(?P<minor>[0-9]+),?\s?.*"
    return regex
