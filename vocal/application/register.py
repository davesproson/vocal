"""Register a vocal project or pack globally.

``vocal register <path>`` auto-detects the kind of resource at ``<path>`` from
its marker file — ``conventions.yaml`` for a project, ``manifest.json`` for a
pack — and registers it under the correct key. There is no ``-c`` conventions
string flag: a project's conventions string has exactly one source of truth,
its ``conventions.yaml``.
"""

import os

import typer

from vocal.conventions_file import (
    CONVENTIONS_FILENAME,
    ConventionsFile,
    import_project_package,
    validate_project_contract,
)
from vocal.manifest import MANIFEST_FILENAME, load_manifest
from vocal.utils.registry import (
    Registry,
    Pack,
    Project,
    get_default_registry_path,
)


class CannotRegisterError(Exception):
    """Base class for registration failures."""


class CannotRegisterProjectError(CannotRegisterError):
    pass


class CannotRegisterPackError(CannotRegisterError):
    pass


class UnknownResourceKind(CannotRegisterError):
    """Raised when a path carries neither a ``conventions.yaml`` nor a ``manifest.json``."""


def register_resource(path: str, force: bool = False) -> None:
    """Register the resource at ``path``, auto-detecting its kind.

    A directory holding ``conventions.yaml`` is a project; one holding
    ``manifest.json`` is a pack. A path carrying neither marker raises
    :class:`UnknownResourceKind`.
    """
    if os.path.isfile(os.path.join(path, CONVENTIONS_FILENAME)):
        register_project(path, force=force)
    elif os.path.isfile(os.path.join(path, MANIFEST_FILENAME)):
        register_pack(path, force=force)
    else:
        raise UnknownResourceKind(
            f"{path} does not look like a vocal project or pack.",
            f"Expected a '{CONVENTIONS_FILENAME}' (project) or "
            f"'{MANIFEST_FILENAME}' (pack) at the path.",
        )


def register_project(
    repo_path: str,
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
        force: re-register even if a project of the same ``{name}-{major}`` is
            registered.
    """

    print(f"Registering project {repo_path} in userspace")

    conventions = ConventionsFile.load(repo_path)

    # Import the project package via the single import path and enforce the
    # project contract before registering anything.
    module = import_project_package(repo_path)
    validate_project_contract(module)

    registry = load_registry()

    project = Project(
        name=conventions.name,
        major=conventions.major,
        minor=conventions.minor,
        project_directory=conventions.project_directory,
        local_path=repo_path,
    )

    try:
        registry.add_project(project, force=force)
    except ValueError:
        raise CannotRegisterProjectError(
            f"Project '{project.key}' is already registered. Use --force to override."
        )

    save_registry(registry)


def register_pack(
    pack_path: str,
    force: bool = False,
) -> None:
    """Register a vocal pack globally.

    ``pack_path`` is a pack release directory — the directory holding
    ``manifest.json``, ``dataset_schema.json``, and the product schema JSONs.
    The pack's identity (base ``url`` and ``version``) is read from its
    ``manifest.json``; loading enforces the ``v{Y}/`` directory-name vs
    ``manifest.json:version`` equality check and raises ``PackInconsistent`` on
    drift.

    Args:
        pack_path: path to the pack release directory.
        force: re-register even if the same ``(url, version)`` is registered.
    """

    print(f"Registering pack {pack_path} in userspace")

    manifest = load_manifest(os.path.join(pack_path, MANIFEST_FILENAME))

    registry = load_registry()

    pack = Pack(manifest=manifest, local_path=pack_path)

    try:
        registry.add_pack(pack, force=force)
    except ValueError:
        raise CannotRegisterPackError(
            f"Pack '{pack.url}' version {pack.version} is already registered. "
            "Use --force to override."
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


def command(
    path: str = typer.Argument(
        help=(
            "The resource to register. A project repo root (holds "
            "conventions.yaml) or a pack release directory (holds manifest.json); "
            "the kind is auto-detected from the marker file."
        )
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Force registration, even if the resource is already registered.",
    ),
) -> None:
    """Register a vocal project or pack globally."""
    register_resource(path, force=force)
