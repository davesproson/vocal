"""Register a vocal project globally."""

import os
import re
from typing import Optional

import typer

from vocal.conventions_file import (
    ConventionsFile,
    import_project_package,
    module_path,
    validate_project_contract,
)
from vocal.utils.registry import (
    ProjectSpec,
    Registry,
    Project,
    get_default_registry_path,
)


class CannotRegisterProjectError(Exception):
    pass


def register_project(
    repo_path: str,
    definitions: str | None = None,
    force: bool = False,
) -> None:
    """
    Register a vocal project globally.

    ``repo_path`` is the project repo root — the directory holding
    ``conventions.yaml``. The project's identity and module layout are read from
    that file; the importable package is imported and checked for the required
    exports before registration.

    Args:
        repo_path: path to the project repo root.
        definitions: path to a product-definitions directory. Defaults to
            ``<repo>/definitions``.
        force: re-register even if a project of the same name is registered.
    """

    print(f"Registering project {repo_path} in userspace")

    conventions = ConventionsFile.load(repo_path)

    # Import the project package via the single import path and enforce the
    # project contract before registering anything.
    module = import_project_package(repo_path)
    validate_project_contract(module)

    if definitions is None:
        definitions = os.path.join(repo_path, "definitions")

    registry = load_registry()

    spec = conventions_to_spec(f"{conventions.name}-[].[]")
    project = Project(
        spec=spec,
        path=module_path(repo_path, conventions),
        definitions=definitions,
    )

    try:
        registry.add_project(project, force=force)
    except ValueError:
        raise CannotRegisterProjectError(
            f"Project for '{spec.name}' is already registered. Use --force to override."
        )

    save_registry(registry)


def load_registry() -> Registry:
    """
    Load the registry of vocal projects.

    Returns:
        Registry: The registry of vocal projects.
    """
    home = os.path.expanduser("~")
    registry_file = os.path.join(home, ".vocal", "vocal-registry.yaml")

    if not os.path.isfile(registry_file):
        return Registry(projects={})

    try:
        return Registry.load(registry_file)
    except Exception as e:
        raise CannotRegisterProjectError(f"Unable to load registry file: {e}") from e


def save_registry(registry: Registry) -> None:
    """
    Save the registry of vocal projects.

    Args:
        registry (Registry): The registry of vocal projects.
    """
    default_path = get_default_registry_path()
    vocal_dir = os.path.dirname(default_path)

    if not os.path.isdir(vocal_dir):
        try:
            os.makedirs(vocal_dir)
        except Exception as e:
            raise CannotRegisterProjectError(
                f"Unable to create vocal directory: {e}"
            ) from e

    try:
        registry.save(default_path)
    except Exception as e:
        raise CannotRegisterProjectError(f"Unable to save registry file: {e}") from e


def conventions_to_spec(conventions_string: str) -> ProjectSpec:
    """
    Convert the given conventions string to a regex pattern.

    Args:
        conventions_string (str): The conventions string to convert.

    Returns:
        str: The regex pattern.
    """
    # Regex which matches STD, STD-[], STD-[].[], indicating standard
    # name, with optional major and minor version numbers
    regex = r"(?P<name>[a-zA-Z0-9]+)(-?(?P<major>\[\])?(\.)?(?P<minor>\[\])?)"

    cmatchd = re.search(regex, conventions_string)

    if cmatchd is None:
        raise ValueError(f"Invalid conventions string: {conventions_string}")

    cmatch = cmatchd.groupdict()
    name = cmatch.get("name")

    if name is None:
        raise ValueError(f"Invalid conventions string: {conventions_string}")

    re_out = name
    if cmatch.get("major"):
        re_out += r"-(?P<major>\d+)"
    if cmatch.get("minor"):
        re_out += r"\.(?P<minor>\d+)"
    re_out += r",?\s?"

    return ProjectSpec(
        name=name,
        has_major=bool(cmatch.get("major")),
        has_minor=bool(cmatch.get("minor")),
        regex=re_out,
    )


def command(
    project: str = typer.Argument(
        help="The vocal project repo root to register (holds conventions.yaml)."
    ),
    definitions: Optional[str] = typer.Option(
        None,
        "-d",
        "--definitions",
        help="The folder to look in for product definitions. Defaults to <project>/definitions.",
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Force registration, even if the project is already registered.",
    ),
) -> None:
    """Register a vocal project globally."""
    register_project(project, definitions, force)
