"""Produce a pack: a self-describing, independently releasable catalogue of
product definitions.

``vocal release`` reads a directory of YAML product definitions plus an
importable project, validates each definition against the project's ``Dataset``
model, and writes a pack to ``<output>/v{Y}/`` with a byte-equal ``<output>/
latest/`` copy of the most recent release. The pack carries a ``manifest.json``
recording its identity (base ``url`` and ``version``), the standard it requires
(``requires_standard`` derived from the project's ``conventions.yaml``), and a
product index used by ``vocal check`` to route a file to its schema.

Persistent pack fields are sourced from CLI flags on the first release and fall
back to the prior release's ``manifest.json`` on subsequent releases:

- ``--url`` (the pack's base URL) falls back to ``<output>/latest/manifest.json``;
  the first release in a fresh output directory requires it explicitly. Supplying
  a ``--url`` that differs (after normalisation) from the prior release is a hard
  error — changing a pack's published URL must be done deliberately.
- ``--min-minor`` (``requires_standard.min_minor``) defaults to the project's
  current ``minor``.
"""

import glob
import os
from typing import Optional

import typer

from vocal.conventions_file import (
    ConventionsFile,
    import_project_package,
    validate_project_contract,
)
from vocal.core import ProductCollection, TemplateSet
from vocal.exceptions import VocalError
from vocal.manifest import (
    build_manifest,
    load_manifest,
    normalize_pack_url,
    versioned_dirname,
)
from vocal.writers import PackWriter


class NoProductDefinitions(VocalError):
    """Raised when the definitions directory holds no top-level ``*.yaml`` files."""


class FirstReleaseRequiresURL(VocalError):
    """Raised when the first release in a fresh output directory omits ``--url``."""


class PackURLMismatch(VocalError):
    """Raised when ``--url`` disagrees with the prior release's ``manifest.json:url``."""


class ReleaseExists(VocalError):
    """Raised when ``<output>/v{Y}/`` already exists and ``--force`` was not given."""


def resolve_full_path(path: str) -> str:
    """Resolve ``path`` to an absolute path against the current working directory."""
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return os.path.normpath(path)


def _resolve_url(url: Optional[str], output_dir: str) -> str:
    """Determine the pack's base URL, applying the ``latest/`` fallback and checks.

    Returns the normalised base URL. Raises :class:`FirstReleaseRequiresURL` when
    no ``--url`` is supplied and there is no prior release to fall back on, and
    :class:`PackURLMismatch` when a supplied ``--url`` differs from the prior
    release's recorded URL.
    """
    latest_manifest = os.path.join(output_dir, "latest", "manifest.json")
    prior_url: Optional[str] = None
    if os.path.isfile(latest_manifest):
        # latest/ is not a v{Y}/ directory, so this never raises PackInconsistent.
        prior_url = load_manifest(latest_manifest).url

    if url is not None:
        url = normalize_pack_url(url)
        if prior_url is not None and url != prior_url:
            raise PackURLMismatch(
                f"Supplied --url {url} differs from the prior release's URL "
                f"{prior_url}.",
                "Changing a pack's published URL must be done deliberately: "
                "remove or update <output>/latest/ first.",
            )
        return url

    if prior_url is None:
        raise FirstReleaseRequiresURL(
            "The first release in a fresh output directory requires --url.",
            "Supply the pack's base URL, e.g. --url https://host/packs.",
        )
    return prior_url


def release(
    *,
    project_path: str,
    version: int,
    definitions: Optional[str] = None,
    output_dir: str = ".",
    url: Optional[str] = None,
    min_minor: Optional[int] = None,
    force: bool = False,
) -> None:
    """Produce a pack from a definitions directory and an importable project.

    Args:
        project_path: the project repo root (holds ``conventions.yaml``).
        version: the pack release number ``Y``.
        definitions: the directory of ``*.yaml`` product definitions. Defaults to
            the current working directory.
        output_dir: where to write ``v{Y}/`` and ``latest/``. Defaults to cwd.
        url: the pack's base URL. Falls back to ``<output>/latest/manifest.json``.
        min_minor: ``requires_standard.min_minor``. Defaults to the project's
            current minor.
        force: allow overwriting an existing ``<output>/v{Y}/``.

    Raises:
        NoProductDefinitions, FirstReleaseRequiresURL, PackURLMismatch,
        ReleaseExists, and the project/conventions errors raised while importing.
    """
    project_path = resolve_full_path(project_path)
    conventions = ConventionsFile.load(project_path)

    module = import_project_package(project_path)
    validate_project_contract(module)
    Dataset = module.models.Dataset
    templates = TemplateSet.from_module(module.defaults)

    defs_dir = resolve_full_path(definitions) if definitions else os.getcwd()
    yaml_files = sorted(glob.glob(os.path.join(defs_dir, "*.yaml")))
    if not yaml_files:
        raise NoProductDefinitions(
            f"No *.yaml product definitions found in {defs_dir}.",
            "vocal release expects a directory of YAML product definitions; "
            "pass --definitions <path> or run from the definitions directory.",
        )

    output_dir = resolve_full_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    resolved_url = _resolve_url(url, output_dir)
    resolved_min_minor = conventions.minor if min_minor is None else min_minor

    version_dir = os.path.join(output_dir, versioned_dirname(version))
    if os.path.isdir(version_dir) and not force:
        raise ReleaseExists(
            f"Release {versioned_dirname(version)} already exists in {output_dir}.",
            "Bump --version, remove the directory, or pass --force to overwrite "
            "(republishing a version with different contents is a footgun).",
        )

    collection = ProductCollection(model=Dataset, version=version, templates=templates)
    for defn in yaml_files:
        collection.add_product(defn)
    collection.validate_all()

    manifest = build_manifest(
        version=version,
        url=resolved_url,
        standard_name=conventions.name,
        standard_major=conventions.major,
        min_minor=resolved_min_minor,
        products=collection.manifest_products,
    )

    PackWriter(
        product_collection=collection, manifest=manifest, output_dir=output_dir
    ).write()


def command(
    project: str = typer.Option(
        ...,
        "-p",
        "--project",
        help="The project repo root to import for templates and the Dataset model.",
    ),
    version: int = typer.Option(
        ...,
        "-v",
        "--version",
        help="The pack release number Y.",
    ),
    url: Optional[str] = typer.Option(
        None,
        "-u",
        "--url",
        help=(
            "The pack's base URL. Falls back to <output>/latest/manifest.json; "
            "the first release in a fresh output directory requires it."
        ),
    ),
    min_minor: Optional[int] = typer.Option(
        None,
        "--min-minor",
        help=(
            "requires_standard.min_minor to write into the manifest. "
            "Defaults to the project's current minor."
        ),
    ),
    definitions: Optional[str] = typer.Option(
        None,
        "-d",
        "--definitions",
        help="The directory of YAML product definitions. Defaults to cwd.",
    ),
    output_dir: str = typer.Option(
        ".",
        "-o",
        "--output",
        help="The directory to write the pack to. Defaults to cwd.",
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Allow overwriting an existing release.",
    ),
) -> None:
    """Produce a pack with a manifest, v{Y}/, and latest/."""
    release(
        project_path=project,
        version=version,
        definitions=definitions,
        output_dir=output_dir,
        url=url,
        min_minor=min_minor,
        force=force,
    )
