"""Check a netCDF file against a standard and a product definition.

``vocal check`` is a thin CLI shell over :mod:`vocal.resolution`: it reads the
file's vocal-managed global attributes, hands them to the resolver together
with the local registry, and renders whatever the resolver returns (or the
typed error it raises). All of the "what should this file be validated
against?" logic lives in :mod:`vocal.resolution`; this module only reads
attributes, drives the resolver, runs pydantic / product-spec validation, and
prints results.

The graceful-degradation matrix (see the parent PRD) governs how CLI flags
substitute for missing file attributes:

- ``-p <path>`` supplied: the user takes over project resolution. The resolver
  is bypassed entirely; the given project(s) are imported directly and the
  given ``-d`` definition(s) checked against. This is the only path when the
  file carries no ``Conventions`` attribute.
- no ``-p``: the resolver resolves the project from ``Conventions`` and (when
  ``vocal_definitions_url`` / ``_version`` are present) the pack. An explicit
  ``-d`` overrides the file's declared pack. When the file declares no pack and
  no ``-d`` is given, the run is incomplete and ``-d`` is required.
"""

import os
from typing import Any, Mapping, Optional

import typer
from pydantic import BaseModel
from pydantic import ValidationError

from vocal.conventions_file import import_project_package
from vocal.exceptions import VocalError
from vocal.resolution import ResolutionError, resolve
from vocal.utils.registry import Project, Registry
from ..checking import ProductChecker
from ..netcdf import NetCDFReader
from ..utils import get_error_locs, TextStyles, Printer
from ..utils.conventions import read_file_conventions

LINE_LEN = 50

TS = TextStyles()
p = Printer()


def check_against_standard(
    model: BaseModel, filename: str, project_name: str = ""
) -> bool:
    """
    Check a netCDF file against a standard (a vocal/pydantic model).

    Args:
        model (BaseModel): The model to check against.
        filename (str): The path of the netCDF file to check.
        project_name (str): The name of the project to check against.

    Returns:
        bool: True if all checks pass, False otherwise.
    """

    p.print_err(
        f"Checking {TS.BOLD}{filename}{TS.ENDC} against "
        f"{TS.BOLD}{project_name}{TS.ENDC} standard... ",
        end="",
    )

    nc = NetCDFReader(filename)

    try:
        nc_noval = nc.to_model(model, validate=False)  # type: ignore
        nc.to_model(model)  # type: ignore
    except ValidationError as err:
        p.print_err(f"{TS.FAIL}{TS.BOLD}ERROR!{TS.ENDC}\n")

        error_locs = get_error_locs(err, nc_noval)

        for err_loc, err_msg in zip(*error_locs):
            p.print_err(f"{TS.FAIL}{TS.BOLD}✗{TS.ENDC} {err_loc}: {err_msg}")

        p.print_err()
        return False
    else:
        p.print_err(f"{TS.OKGREEN}{TS.BOLD}OK!{TS.ENDC}\n")

        return True


def print_checks(pc, filename, specification):
    p.print_err(
        f"Checking {TS.BOLD}{filename}{TS.ENDC} against "
        f"{TS.BOLD}{os.path.basename(specification)}{TS.ENDC} specification... ",
        end="",
    )

    failed = any(not check.passed for check in pc.checks)
    if failed:
        p.print_err(f"{TS.FAIL}{TS.BOLD}ERROR!{TS.ENDC}\n")
    else:
        p.print_err(f"{TS.OKGREEN}{TS.BOLD}OK!{TS.ENDC}\n")

    for check in pc.checks:
        if check.passed:
            if check.has_warning and check.warning:
                p.print_warn(f"  {check.description}", end="\r")
                p.print_warn(f"{TS.BOLD}{TS.WARNING}!{TS.ENDC}")
                p.print_warn(
                    f"{TS.BOLD}{TS.WARNING}  --> {check.warning.path}: {TS.ENDC}"
                    f"{TS.WARNING}{check.warning.message}{TS.ENDC}"
                )
            if check.has_comment and check.comment:
                p.print_comment(f"  {check.description}", end="\r")
                p.print_comment(f"{TS.BOLD}{TS.OKBLUE}i{TS.ENDC}")
                p.print_comment(
                    f"{TS.BOLD}{TS.OKBLUE}  --> {check.comment.path}: {TS.ENDC}"
                    f"{TS.OKBLUE}{check.comment.message}{TS.ENDC}"
                )
            else:
                p.print(f"  {check.description}", end="\r")
                p.print(f"{TS.BOLD}{TS.OKGREEN}✔{TS.ENDC}")
        elif check.error:
            p.print_err(f"  {check.description}", end="\r")
            p.print_err(f"{TS.FAIL}{TS.BOLD}✗{TS.ENDC}")
            p.print_err(
                f"{TS.FAIL}  --> {TS.BOLD}{check.error.path}:{TS.ENDC} "
                f"{TS.FAIL}{check.error.message}{TS.ENDC}"
            )

    p.print_err()
    p.print_line_err(LINE_LEN, "=")
    p.print_err(f"{TS.BOLD}{TS.OKGREEN}✔{TS.ENDC} {len(pc.checks)} checks.")
    p.print_err(f"{TS.BOLD}{TS.WARNING}!{TS.ENDC} {len(pc.warnings)} warnings.")
    p.print_err(f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} {len(pc.errors)} errors found.")
    p.print_err(
        f"{TS.BOLD}{TS.OKBLUE}i{TS.ENDC} {len(pc.comments)} comments (run with -c)."
    )
    p.print_line_err(LINE_LEN, "=")
    p.print_err()


def check_against_specification(filename: str, specification: str) -> bool:
    """
    Check a netCDF file against a product specification.

    Args:
        specification (str): The path of the specification to check against.
        filename (str): The path of the netCDF file to check.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    pc = ProductChecker(specification)
    pc.check(filename)
    print_checks(pc, filename, specification)
    return pc.passing


def check_file_against_project(filename: str, project: str) -> bool:
    """
    Check a file against the standard defined by a project at ``project``.

    ``project`` is a project repo root (containing ``conventions.yaml``); its
    package is imported through the single project-import path.

    Args:
        filename (str): The path to the netCDF file.
        project (str): The path to the project repo root.

    Returns:
        bool: True if the file validates against the project's Dataset model.
    """
    p.print_err()

    try:
        project_mod = import_project_package(project)
    except VocalError as e:
        _print_error(e)
        raise typer.Exit(code=1)

    return check_against_standard(
        model=project_mod.models.Dataset,
        filename=filename,
        project_name=os.path.basename(os.path.normpath(project)),
    )


def _print_error(error: VocalError) -> None:
    """Render a typed vocal error — its message and actionable hint — to stderr."""
    p.print_err(f"\n{TS.BOLD}{TS.FAIL}✗{TS.ENDC} {error.message}")
    if error.hint:
        p.print_err(f"  {error.hint}\n")


def _load_filecodec(project: Project) -> Mapping[str, Mapping[str, Any]]:
    """Import a resolved project's package and return its ``filecodec``.

    Passed to :func:`vocal.resolution.resolve` so that product matching can
    expand templated ``file_pattern`` entries. Defined here (rather than relying
    on the resolver's default loader) so tests can patch the import.
    """
    return import_project_package(project.local_path).filecodec


def _run_manual_checks(
    filename: str, projects: list[str], definitions: Optional[list[str]]
) -> bool:
    """Manual mode: check ``filename`` against explicitly supplied paths.

    The resolver is bypassed: each ``-p`` project is imported directly and each
    ``-d`` definition checked against. This is the path taken when the file has
    no ``Conventions`` attribute (the matrix's "absent" row requires both ``-p``
    and ``-d``).
    """
    ok = True
    for project in projects:
        ok = check_file_against_project(filename, project.rstrip("/")) and ok

    for definition in definitions or []:
        ok = check_against_specification(filename, definition) and ok

    return ok


def _run_resolved_checks(
    filename: str, attrs, definitions: Optional[list[str]]
) -> bool:
    """Resolver mode: drive :func:`vocal.resolution.resolve` and render results.

    ``-d`` (the first definition, if any) becomes the resolver's
    ``definition_override``, short-circuiting pack resolution. Typed resolver
    errors are rendered with their locked message and hint.
    """
    override = definitions[0] if definitions else None

    try:
        registry = Registry.load()
    except FileNotFoundError:
        registry = Registry()

    try:
        target = resolve(
            registry,
            filename=filename,
            conventions=attrs.conventions,
            definitions_url=attrs.definitions_url,
            definitions_version=attrs.definitions_version,
            definition_override=override,
            project_url=attrs.project_url,
            filecodec_loader=_load_filecodec,
        )
    except ResolutionError as e:
        _print_error(e)
        return False

    try:
        project_mod = import_project_package(target.project.local_path)
    except VocalError as e:
        _print_error(e)
        return False

    p.print_err()
    project_name = f"{target.project.name}-{target.project.major}"
    ok = check_against_standard(project_mod.models.Dataset, filename, project_name)

    if target.schema_path is None:
        # The project resolved, but the file declares no pack and no -d override
        # was supplied — there is no product definition to check against.
        p.print_err(
            f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} The file declares no product "
            f"definitions (no vocal_definitions_url / vocal_definitions_version)."
        )
        p.print_err(
            "  Pass -d <path> to supply a product definition to check against.\n"
        )
        return False

    return check_against_specification(filename, target.schema_path) and ok


def command(
    filename: str = typer.Argument(metavar="FILE", help="The netCDF file to check"),
    project: Optional[list[str]] = typer.Option(
        None,
        "-p",
        "--project",
        help="Path to one or more vocal projects. Pass multiple times for multiple projects.",
    ),
    definition: Optional[list[str]] = typer.Option(
        None,
        "-d",
        "--definition",
        help="Product definition(s) to test against. Pass multiple times for multiple definitions.",
    ),
    error_only: bool = typer.Option(
        False,
        "-e",
        "--error-only",
        help="Only print errors. Takes precedence over -w/--warnings.",
    ),
    warnings: bool = typer.Option(
        False,
        "-w",
        "--warnings",
        help="Only print warnings and errors.",
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Do not print any output."),
    comments: bool = typer.Option(False, "-c", "--comments", help="Print comments."),
    no_color: bool = typer.Option(
        False, "--no-color", help="Do not print colored output."
    ),
) -> None:
    """Check a netCDF file against standard and product definitions."""
    if error_only:
        p.ignore_info = True
        p.ignore_warnings = True
    if warnings:
        p.ignore_info = True
    if comments:
        p.comments = True

    p.quiet = quiet

    if no_color:
        TS.enabled = False

    if project:
        ok = _run_manual_checks(filename, project, definition)
    else:
        attrs = read_file_conventions(filename)
        ok = _run_resolved_checks(filename, attrs, definition)

    if not ok:
        raise typer.Exit(code=1)
