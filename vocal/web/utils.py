import os
import tempfile

from pydantic import ValidationError
from fastapi import UploadFile, HTTPException, status

from vocal.application.check import (
    load_matching_definitions,
    load_matching_projects,
    NoConventionsFound,
    NoMatchingProjects,
)
from vocal.checking import CheckError, ProductChecker
from vocal.netcdf.writer import NetCDFReader
from vocal.utils import get_error_locs, import_project
from vocal.web.models import Check, CheckContext, CheckDefinition, CheckProject


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

        # Load the projects and definitions which match the file
        # pattern and Conventions
        try:
            projects = load_matching_projects(file_path)
        except NoConventionsFound as e:
            raise HTTPException(
                detail=str(e), status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            )
        except NoMatchingProjects as e:
            raise HTTPException(
                detail=str(e), status_code=status.HTTP_400_BAD_REQUEST
            )

        try:
            definitions = load_matching_definitions(file_path)
        except NoConventionsFound as e:
            raise HTTPException(
                detail=str(e), status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            )

        # Check against each project
        for project in projects:
            project_name: str = ""

            # Import the project module
            try:
                project_mod = import_project(project)
                project_name = project_mod.__name__

                context.projects[project_name] = CheckProject(
                    passed=True,
                    errors=[],
                )
            except Exception as e:
                context.errors.append(f"Error loading project {project}: {e}")

            # Load the Dataset model from the project. If the model
            # cannot be parsed, add the error to the context.
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

        # Check against each definition
        for definition in definitions:
            context.definitions[os.path.basename(definition)] = CheckDefinition(
                passed=True, warnings=False, comments=False, checks=[]
            )

            # Instantiate the ProductChecker and check the file against
            # the definition
            pc = ProductChecker(definition)
            pc.check(file_path)

            # Parse the results of the check and add them to the context
            context.definitions[os.path.basename(definition)].passed = all(
                [r.passed for r in pc.checks]
            )

            for check in pc.checks:
                _check = Check(
                    description=check.description,
                )

                if check.passed:
                    if check.has_comment and check.comment:
                        context.definitions[
                            os.path.basename(definition)
                        ].comments = True
                        _check.comment = check.comment

                    if check.has_warning and check.warning:
                        context.definitions[
                            os.path.basename(definition)
                        ].warnings = True
                        _check.warning = check.warning

                elif check.error:
                    _check.error = check.error

                context.definitions[os.path.basename(definition)].checks.append(_check)

    return context
