import os
import tempfile

from pydantic import ValidationError
from fastapi import UploadFile, HTTPException, status

from vocal.application.check import (
    load_matching_definitions,
    load_matching_projects,
    DefinitionVersionNotFound,
)
from vocal.checking import CheckError, ProductChecker
from vocal.netcdf.writer import NetCDFReader
from vocal.utils import get_error_locs, import_project
from vocal.web.models import (
    Check,
    CheckContext,
    CheckDefinition,
    CheckIssue,
    CheckProject,
)


async def check_upload(file: UploadFile) -> CheckContext:
    """
    Check the uploaded file against the registered projects and definitions.

    Args:
        file (UploadFile): The file to check.

    Returns:
        CheckContext: The context of the check.
    """

    context = CheckContext()

    # If no file is provided, raise an error
    if not file.filename:
        raise HTTPException(
            detail="No file provided", status_code=status.HTTP_400_BAD_REQUEST
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the file to a temporary directory
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        file.file.close()

        # Load the projects which match the file pattern and Conventions.
        # NoConventionsFound / NoMatchingProjects are fatal — they propagate
        # to the global VocalError handler, which renders error.html.
        projects = load_matching_projects(file_path)

        # Definition loading is best-effort. If the version directory for a
        # matched project does not exist, we surface that as an inline
        # error and continue with the project check.
        try:
            definitions = load_matching_definitions(file_path)
        except DefinitionVersionNotFound as e:
            context.errors.append(CheckIssue(message=e.message, hint=e.hint))
            definitions = []

        # Check against each project. A failure inside this loop (for
        # example, a broken project module) becomes an inline error so
        # other projects can still be checked.
        for project in projects:
            try:
                project_mod = import_project(project)
                project_name = project_mod.__name__
                context.projects[project_name] = CheckProject(
                    passed=True,
                    errors=[],
                )
            except Exception as e:
                context.errors.append(
                    CheckIssue(message=f"Failed to load project {project}: {e}")
                )
                continue

            nc = NetCDFReader(file_path)
            try:
                nc_noval = nc.to_model(project_mod.models.Dataset, validate=False)
                nc.to_model(project_mod.models.Dataset)
            except ValidationError as err:
                error_locs = get_error_locs(err, nc_noval)
                context.projects[project_name].passed = False
                for loc, msg in zip(*error_locs):
                    context.projects[project_name].errors.append(
                        CheckError(path=loc, message=msg)
                    )
            except Exception as e:
                context.projects[project_name].passed = False
                context.errors.append(
                    CheckIssue(message=f"Project {project_name} check failed: {e}")
                )

        # Check against each definition. As above, per-definition failures
        # become inline errors.
        for definition in definitions:
            def_name = os.path.basename(definition)
            context.definitions[def_name] = CheckDefinition(
                passed=True, warnings=False, comments=False, checks=[]
            )

            try:
                pc = ProductChecker(definition)
                pc.check(file_path)
            except Exception as e:
                context.errors.append(
                    CheckIssue(message=f"Definition {def_name} check failed: {e}")
                )
                context.definitions.pop(def_name, None)
                continue

            # Parse the results of the check and add them to the context
            context.definitions[def_name].passed = all(
                [r.passed for r in pc.checks]
            )

            for check in pc.checks:
                _check = Check(
                    description=check.description,
                )

                if check.passed:
                    if check.has_comment and check.comment:
                        context.definitions[def_name].comments = True
                        _check.comment = check.comment

                    if check.has_warning and check.warning:
                        context.definitions[def_name].warnings = True
                        _check.warning = check.warning

                elif check.error:
                    _check.error = check.error

                context.definitions[def_name].checks.append(_check)

    return context
