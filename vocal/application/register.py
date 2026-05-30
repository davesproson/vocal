"""Register a vocal project or pack globally.

``vocal register <path>`` auto-detects the kind of resource at ``<path>`` from
its marker file — ``conventions.yaml`` for a project, ``manifest.json`` for a
pack — and registers it under the correct key. There is no ``-c`` conventions
string flag: a project's conventions string has exactly one source of truth,
its ``conventions.yaml``.
"""

import os
import sys

import typer

from vocal.application.install import (
    DEFAULT_IGNORE,
    pack_install_dir,
    project_install_dir,
    staged_install,
)
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
    project_key,
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


def install_project(
    source: str,
    force: bool = False,
) -> Project:
    """Install an owned copy of the project at ``source`` under ``~/.vocal``.

    The single orchestration both ``register`` and ``fetch`` route a project
    through. It reads the project's identity from ``source``'s
    ``conventions.yaml``, computes the canonical install directory
    ``~/.vocal/projects/{name}-{major}``, and — unless a project of that key is
    already registered (the gate) or ``force`` is passed — stages a normalised
    copy, validates it by importing the package and enforcing the project
    contract against the *staging* copy, then atomically swaps it into place.

    The registry record's ``local_path`` points at the owned copy under
    ``~/.vocal``, so resolution no longer depends on ``source`` continuing to
    exist where it was when registered. A relative ``source`` therefore resolves
    to a location-independent entry. A leftover on-disk directory with no
    matching registry entry passes the registry-key gate and is transparently
    overwritten by the atomic swap; a broken source leaves any existing install
    untouched, because validation runs against the staging copy before ``dest``
    is touched.

    Args:
        source: the project repo root — the directory holding
            ``conventions.yaml``.
        force: re-install even if a project of the same ``{name}-{major}`` is
            registered, overwriting both the on-disk copy and the registry entry.

    Returns:
        the registered :class:`~vocal.utils.registry.Project`.

    Raises:
        CannotRegisterProjectError: a project of the same ``{name}-{major}`` is
            already registered and ``force`` is False.
        InvalidConventionsFile, MissingProjectExport: ``source`` is not a valid,
            importable vocal project; raised before any existing install or
            registry entry is touched.
    """
    conventions = ConventionsFile.load(source)

    registry = load_registry()

    # Gate on the registry key — the source of truth for "is it installed". A
    # leftover on-disk directory without a matching entry is not gated; the
    # atomic swap below overwrites it, so drift self-heals.
    key = project_key(conventions.name, conventions.major)
    if key in registry.projects and not force:
        raise CannotRegisterProjectError(
            f"Project '{key}' is already registered. Use --force to override."
        )

    dest = project_install_dir(conventions)

    def validate(staging: str) -> None:
        # Import + contract-check the byte-identical staging copy, so a broken
        # source aborts the install before any existing dest is torn down.
        # Suppress bytecode writing so the validation import does not seed
        # __pycache__ into the copy that becomes the owned install — the
        # denylist already excludes it on the way in, and it must not creep
        # back in on the way out.
        prior = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            module = import_project_package(staging)
            validate_project_contract(module)
        finally:
            sys.dont_write_bytecode = prior

    staged_install(source, dest, ignore=DEFAULT_IGNORE, validate=validate)

    project = Project(
        name=conventions.name,
        major=conventions.major,
        minor=conventions.minor,
        project_directory=conventions.project_directory,
        local_path=dest,
    )
    registry.add_project(project, force=force)
    save_registry(registry)
    return project


def register_project(
    repo_path: str,
    force: bool = False,
) -> None:
    """Register a vocal project globally by installing an owned copy.

    Thin wrapper over :func:`install_project`: ``repo_path`` is the project repo
    root (the directory holding ``conventions.yaml``); its contents are copied
    into ``~/.vocal/projects/{name}-{major}`` and that owned copy is what the
    registry records, so the registration survives the source being moved or
    deleted.

    Args:
        repo_path: path to the project repo root.
        force: re-register even if a project of the same ``{name}-{major}`` is
            registered.
    """

    print(f"Registering project {repo_path} in userspace")
    install_project(repo_path, force=force)


def install_pack(
    source: str,
    force: bool = False,
) -> Pack:
    """Install an owned copy of the pack at ``source`` under ``~/.vocal``.

    The single orchestration both ``register`` and ``fetch`` route a pack
    through. It reads the pack's identity ``(url, version)`` from ``source``'s
    ``manifest.json``, computes the canonical install directory
    ``~/.vocal/packs/{slug}/v{Y}`` — keyed on the manifest, not the source
    directory name — and, unless a pack of that ``(url, version)`` is already
    registered (the gate) or ``force`` is passed, stages a normalised copy,
    validates it by re-loading the *staging* manifest, then atomically swaps it
    into place.

    Reading the source manifest with :func:`load_manifest` enforces the
    ``v{Y}/`` directory-name vs ``manifest.json:version`` equality check on the
    source: a source already named ``v{Y}/`` (as ``vocal release`` produces)
    must agree with its manifest, but the install destination is derived from
    the manifest's version regardless of what the source directory was called.

    The registry record's ``local_path`` points at the owned copy under
    ``~/.vocal``, so resolution no longer depends on ``source`` continuing to
    exist where it was when registered. A leftover on-disk directory with no
    matching registry entry passes the registry-key gate and is transparently
    overwritten by the atomic swap; a broken manifest leaves any existing
    install untouched, because the source manifest is loaded — and validated —
    before ``dest`` is touched.

    Args:
        source: a pack release directory — the directory holding
            ``manifest.json``, ``dataset_schema.json``, and the product schema
            JSONs.
        force: re-install even if a pack of the same ``(url, version)`` is
            registered, overwriting both the on-disk copy and the registry entry.

    Returns:
        the registered :class:`~vocal.utils.registry.Pack`.

    Raises:
        CannotRegisterPackError: a pack of the same ``(url, version)`` is already
            registered and ``force`` is False.
        PackInconsistent: a source named ``v{Y}/`` disagrees with its manifest's
            version; raised before any existing install or registry entry is
            touched.
        InvalidManifest, UnsupportedManifestVersion: ``source`` is not a valid
            pack; raised before any existing install or registry entry is touched.
    """
    manifest = load_manifest(os.path.join(source, MANIFEST_FILENAME))

    registry = load_registry()

    # Gate on the registry key — the source of truth for "is it installed". A
    # leftover on-disk directory without a matching entry is not gated; the
    # atomic swap below overwrites it, so drift self-heals.
    pack = Pack(manifest=manifest, local_path="")
    if pack.key in registry.packs and not force:
        raise CannotRegisterPackError(
            f"Pack '{pack.url}' version {pack.version} is already registered. "
            "Use --force to override."
        )

    dest = pack_install_dir(manifest)

    def validate(staging: str) -> None:
        # Re-load the byte-identical staging manifest, so a broken source aborts
        # the install before any existing dest is torn down. The staging
        # directory is not named ``v{Y}/``, so this revalidates structure only —
        # the source directory-name consistency check already ran above.
        load_manifest(os.path.join(staging, MANIFEST_FILENAME))

    staged_install(source, dest, ignore=DEFAULT_IGNORE, validate=validate)

    pack.local_path = dest
    registry.add_pack(pack, force=force)
    save_registry(registry)
    return pack


def register_pack(
    pack_path: str,
    force: bool = False,
) -> None:
    """Register a vocal pack globally by installing an owned copy.

    Thin wrapper over :func:`install_pack`: ``pack_path`` is a pack release
    directory — the directory holding ``manifest.json``, ``dataset_schema.json``,
    and the product schema JSONs. Its contents are copied into
    ``~/.vocal/packs/{slug}/v{Y}`` and that owned copy is what the registry
    records, so the registration survives the source being moved or deleted.

    Args:
        pack_path: path to the pack release directory.
        force: re-register even if the same ``(url, version)`` is registered.
    """

    print(f"Registering pack {pack_path} in userspace")
    install_pack(pack_path, force=force)


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
