import os
import tempfile
from typing import Any, Mapping, Optional

from pydantic import ValidationError
from fastapi import UploadFile, HTTPException, status

from vocal.checking import ProductChecker
from vocal.conventions_file import import_project_package
from vocal.netcdf.writer import NetCDFReader
from vocal.resolution import ResolutionError, ResolvedTarget, resolve
from vocal.utils import get_error_locs
from vocal.utils.conventions import FileConventions, read_file_conventions
from vocal.utils.registry import Project, Registry
from vocal.web.models import (
    Check,
    CheckContext,
    CheckDefinition,
    CheckError,
    CheckProject,
    ResolverError,
)


def _web_precondition_error(attrs: FileConventions) -> Optional[ResolverError]:
    """Reject files the web flow cannot resolve for lack of a flag fallback.

    The CLI lets the user supply ``-p`` / ``-d`` when a file does not
    self-describe; the web UI has no such fallback, so the file must carry both
    a ``Conventions`` string and a pack reference
    (``vocal_definitions_url`` + ``vocal_definitions_version``). This mirrors
    the "Web behaviour" column of the parent PRD's graceful-degradation matrix.

    Returns a :class:`ResolverError` describing the missing attribute, or
    ``None`` when the file carries everything the resolver needs.
    """
    if not attrs.conventions:
        return ResolverError(
            code="missing_conventions",
            message="The file has no Conventions attribute.",
            hint=(
                "The web checker resolves a file from the attributes it carries. "
                "Add a Conventions attribute naming the standard and version the "
                "file was authored against (e.g. 'MYSTD-1.2')."
            ),
        )

    if attrs.definitions_url is None or attrs.definitions_version is None:
        return ResolverError(
            code="missing_pack_reference",
            message="The file declares no product definitions pack.",
            hint=(
                "The web checker needs the vocal_definitions_url and "
                "vocal_definitions_version global attributes to locate the pack "
                "the file should be validated against."
            ),
        )

    return None


def _load_filecodec(project: Project) -> Mapping[str, Mapping[str, Any]]:
    """Import a resolved project's package and return its ``filecodec``.

    Passed to :func:`vocal.resolution.resolve` so product matching can expand
    templated ``file_pattern`` entries. Defined here (rather than relying on the
    resolver's default loader) so tests can patch the import in this module.
    """
    return import_project_package(project.local_path).filecodec


def _check_against_standard(
    context: CheckContext, target: ResolvedTarget, file_path: str
) -> None:
    """Validate the file against the resolved project's ``Dataset`` model.

    Populates ``context.projects`` under the project's ``{name}-{major}`` key,
    recording per-attribute validation errors when pydantic rejects the file.
    """
    project = target.project
    project_name = f"{project.name}-{project.major}"
    result = CheckProject(passed=True, errors=[])
    context.projects[project_name] = result

    project_mod = import_project_package(project.local_path)
    nc = NetCDFReader(file_path)

    try:
        nc_noval = nc.to_model(project_mod.models.Dataset, validate=False)
        nc.to_model(project_mod.models.Dataset)
    except ValidationError as err:
        result.passed = False
        locs, msgs = get_error_locs(err, nc_noval)
        for loc, msg in zip(locs, msgs):
            result.errors.append(CheckError(path=loc, message=msg))


def _check_against_definition(
    context: CheckContext, target: ResolvedTarget, file_path: str
) -> None:
    """Check the file against the resolved product definition.

    Populates ``context.definitions`` under the matched product's name with the
    pass/warning/comment state and the individual checks, mirroring the shape
    the results template renders.
    """
    schema_path = target.schema_path
    if schema_path is None:
        return

    def_name = target.product.name if target.product else os.path.basename(schema_path)
    result = CheckDefinition(passed=True, warnings=False, comments=False, checks=[])
    context.definitions[def_name] = result

    pc = ProductChecker(schema_path)
    pc.check(file_path)

    result.passed = all(check.passed for check in pc.checks)

    for check in pc.checks:
        _check = Check(description=check.description)

        if check.passed:
            if check.has_comment and check.comment:
                result.comments = True
                _check.comment = check.comment
            if check.has_warning and check.warning:
                result.warnings = True
                _check.warning = check.warning
        elif check.error:
            _check.error = check.error

        result.checks.append(_check)


async def check_upload(file: UploadFile) -> CheckContext:
    """
    Check the uploaded file against the registered project and pack.

    The web flow drives :mod:`vocal.resolution` exactly as ``vocal check`` does
    on the CLI, but without the ``-p`` / ``-d`` flag fallback: the file must
    self-describe through its ``Conventions`` and ``vocal_definitions_*``
    attributes. Resolution failures (and missing-attribute preconditions) are
    surfaced as a single typed :class:`ResolverError` on the returned context;
    on success the project and product checks populate it.

    Args:
        file (UploadFile): The file to check.

    Returns:
        CheckContext: The context of the check.
    """

    context = CheckContext()

    if not file.filename:
        raise HTTPException(
            detail="No file provided", status_code=status.HTTP_400_BAD_REQUEST
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        file.file.close()

        attrs = read_file_conventions(file_path)

        # The web UI has no flag fallback: a file that does not self-describe
        # cannot be checked. Reject it with a clear, actionable error.
        precondition = _web_precondition_error(attrs)
        if precondition is not None:
            context.error = precondition
            return context

        try:
            registry = Registry.load()
        except FileNotFoundError:
            registry = Registry()

        try:
            target = resolve(
                registry,
                filename=file_path,
                conventions=attrs.conventions,
                definitions_url=attrs.definitions_url,
                definitions_version=attrs.definitions_version,
                project_url=attrs.project_url,
                filecodec_loader=_load_filecodec,
            )
        except ResolutionError as e:
            context.error = ResolverError(code=e.code, message=e.message, hint=e.hint)
            return context

        _check_against_standard(context, target, file_path)
        _check_against_definition(context, target, file_path)

    return context
