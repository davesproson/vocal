import os
import re
import tempfile

from pydantic import ValidationError
from fastapi import UploadFile, HTTPException, status

import yaml

from vocal.checking import CheckError, ProductChecker
from vocal.exceptions import VocalError
from vocal.netcdf.writer import NetCDFReader
from vocal.utils import (
    get_error_locs,
    import_project,
    regexify_file_pattern,
    TextStyles,
)
from vocal.utils.conventions import get_conventions_string
from vocal.utils.registry import Registry
from vocal.versioning import InvalidVersion, Version
from vocal.web.models import (
    Check,
    CheckContext,
    CheckDefinition,
    CheckIssue,
    CheckProject,
)

TS = TextStyles()


# ---------------------------------------------------------------------------
# Legacy registry-driven resolution.
#
# These helpers predate the :mod:`vocal.resolution` module and are kept here,
# alongside their only remaining caller (:func:`check_upload`), until the web
# check flow is rewired onto the resolver. The CLI (``vocal check``) no longer
# uses them.
# ---------------------------------------------------------------------------


class NoConventionsFound(VocalError):
    pass


class NoMatchingProjects(VocalError):
    pass


class DefinitionVersionNotFound(VocalError):
    pass


def load_matching_projects(filename: str) -> list[str]:
    """Load all registered projects matching the file's ``Conventions``."""
    conventions = get_conventions_string(filename)

    if conventions is None:
        raise NoConventionsFound(
            "No conventions found in file. Please provide a project or definition."
        )

    try:
        c = Registry.filter(conventions)
    except FileNotFoundError:
        c = Registry(projects={})

    if len(c) == 0:
        raise NoMatchingProjects(
            f"No registered project(s) found for conventions {conventions}"
        )

    print(
        f"\n{TS.BOLD}{TS.OKGREEN}✔{TS.ENDC} Found {len(c)} registered project(s) "
        f"for conventions {conventions}: {', '.join(c.projects.keys())}"
    )

    return [proj.path for proj in c.projects.values()]


def file_version_string(filename: str, project_name: str) -> str:
    """Return the file's claimed ``project_name`` version as ``v{major}.{minor}``."""
    conventions = get_conventions_string(filename)

    if conventions is None:
        raise NoConventionsFound(
            "No conventions found in file. Please provide a project or definition."
        )

    for token in conventions.split():
        token = token.rstrip(",")
        if not token.startswith(f"{project_name}-"):
            continue
        try:
            version = Version.parse(token)
        except InvalidVersion:
            continue
        return f"v{version.major}.{version.minor}"

    raise NoConventionsFound(
        f"No conventions token for '{project_name}' found in file {filename}."
    )


def load_matching_definitions(filename: str) -> list[str]:
    """Load all definitions matching the file's ``Conventions`` and version."""
    conventions = get_conventions_string(filename)

    if conventions is None:
        raise NoConventionsFound(
            "No conventions found in file. Please provide a project or definition."
        )

    registry = Registry.filter(conventions)

    definitions: list[str] = []
    paths: list[str] = []
    filecodecs: list[dict] = []

    for project in registry:
        version_string = file_version_string(filename, project.spec.name)
        path = os.path.join(project.definitions, version_string)
        if not os.path.isdir(path):
            raise DefinitionVersionNotFound(
                f"No product definitions registered for "
                f"{project.spec.name} version {version_string}",
                hint=(
                    f"Register a project providing definitions for "
                    f"{project.spec.name} {version_string}, or upload "
                    f"a file matching a registered version."
                ),
            )
        paths.append(path)
        vocal_project = import_project(project.path)
        filecodecs.append(vocal_project.filecodec)

    for path, codec in zip(paths, filecodecs):
        def_files = [
            f
            for f in os.listdir(path)
            if f.endswith(".json") and f != "dataset_schema.json"
        ]
        for file in def_files:
            with open(os.path.join(path, file), "r") as f:
                data = yaml.load(f, Loader=yaml.Loader)
                rex = regexify_file_pattern(data["meta"]["file_pattern"], codec)
                if re.match(rex, os.path.basename(filename)):
                    definitions.append(os.path.join(path, file))

    return definitions


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
