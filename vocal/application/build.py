"""Create an example data file from a definition."""

import typer

from vocal.core import ProductDefinition, register_defaults_module
from vocal.utils import import_project


def make_example_data(
    project_path: str,
    definition: str,
    output: str,
    find_coordinates: bool,
) -> None:
    project = import_project(project_path)

    try:
        Dataset = project.models.Dataset
    except ModuleNotFoundError as e:
        raise RuntimeError("Unable to import dataset schema") from e

    register_defaults_module(project.defaults)

    product = ProductDefinition(definition, Dataset)
    product.create_example_file(output, find_coords=find_coordinates)


def command(
    project: str = typer.Option(
        ".", "-p", "--project",
        help="The path of a vocal project to use. Defaults to current directory.",
    ),
    definition: str = typer.Option(
        ..., "-d", "--definition",
        help="The product definition file to use for the example data.",
    ),
    output: str = typer.Option(
        ..., "-o", "--output",
        help="The output filename.",
    ),
    find_coordinates: bool = typer.Option(
        False, "--find-coordinates", "-fc",
        help=(
            "Use standard names to generate coordinates attribute "
            "rather than relying on specification/example."
        ),
    ),
) -> None:
    """Create an example data file from a definition."""
    make_example_data(project, definition, output, find_coordinates)
