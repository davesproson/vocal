"""Create versioned JSON product specifications."""

import glob
import os
from typing import Optional

import typer

from vocal.core import ProductCollection, TemplateSet
from vocal.utils import import_project


def resolve_full_path(path: str) -> str:
    """
    Resolve a path to a full path if it is not already.

    Args:
        path: The path to resolve

    Returns:
        The resolved path
    """

    if not path.startswith("/"):
        path = os.path.join(os.getcwd(), path)
    return path


def release(
    project_path: str,
    version: str,
    output_dir: str,
    definitions: Optional[str],
) -> None:
    project = resolve_full_path(project_path)

    if not os.path.isdir(project):
        raise ValueError(
            f"Project directory {project} does not exist"
            "please supply the full path to a vocal project directory"
        )

    try:
        module = import_project(project)
        defaults = module.defaults

    except ModuleNotFoundError as e:
        raise RuntimeError("Unable to import project defaults") from e

    try:
        Dataset = module.models.Dataset
    except ModuleNotFoundError as e:
        raise RuntimeError("Unable to import project models") from e

    templates = TemplateSet.from_module(defaults)

    if not definitions:
        definitions = os.path.join(project, "definitions")

    defs_dir = resolve_full_path(definitions)
    defs_glob = os.path.join(defs_dir, "*.yaml")

    collection = ProductCollection(model=Dataset, version=version, templates=templates)

    for defn in glob.glob(defs_glob):
        collection.add_product(defn)

    cwd = os.getcwd()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    os.chdir(output_dir)
    try:
        collection.write_product_definitions()
    except Exception as e:
        raise RuntimeError("Failed to create versioned products") from e
    finally:
        os.chdir(cwd)


def command(
    project: str = typer.Argument(
        metavar="PROJECT",
        help="The path of a vocal project for which to create vocabularies.",
    ),
    definitions: Optional[str] = typer.Option(
        None,
        "-d",
        "--definitions",
        help=(
            "The folder to look in for product definitions. "
            "Defaults to <project>/definitions."
        ),
    ),
    version: str = typer.Option(
        ...,
        "-v",
        "--version",
        help="The product version, e.g. 1.0.",
    ),
    output_dir: str = typer.Option(
        ".",
        "-o",
        "--output-dir",
        help="The directory to write the versioned definitions to.",
    ),
) -> None:
    """Create versioned JSON product specifications."""
    release(project, version, output_dir, definitions)
