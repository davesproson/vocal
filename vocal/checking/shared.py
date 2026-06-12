import os

from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from vocal.checking.checking import CheckReport, ProductChecker
from vocal.netcdf import NetCDFReader
from vocal.resolution import ResolvedTarget
from vocal.utils import import_project


@dataclass(frozen=True)
class ProjectCheckResult:
    """
    A class to hold the result of a project check
    """

    error: ValidationError | None = None
    nc_noval: BaseModel | None = None

    @property
    def passed(self) -> bool:
        """
        Returns True if the project check passed, False otherwise.
        """
        return self.error is None


@dataclass(frozen=True)
class DefinitionCheckResult:
    """
    A class to hold the result of a definition check
    """

    report: CheckReport | None = None

    @property
    def passed(self) -> bool:
        """
        Returns True if the definition check passed, False otherwise.
        """
        return self.report is not None and self.report.passing


@dataclass(frozen=True)
class FullCheckResult:
    """
    A class to hold the full result of a check, including both project and
    definition checks.
    """

    project_result: ProjectCheckResult
    definition_result: DefinitionCheckResult | None

    @property
    def passed(self) -> bool:
        """
        Returns True if both the project and definition checks passed, False
        otherwise.
        """
        return self.project_result.passed and (
            self.definition_result is None or self.definition_result.passed
        )


def check_against_project(
    project: type[BaseModel], filename: str
) -> ProjectCheckResult:
    """
    Check a project and return the result.
    """

    nc = NetCDFReader(filename)

    nc_noval: BaseModel | None = None

    try:
        nc_noval = nc.to_model(project, validate=False)
        nc.to_model(project)
    except ValidationError as err:
        return ProjectCheckResult(error=err, nc_noval=nc_noval)

    return ProjectCheckResult()


def check_against_definition(definition: str, filename: str) -> DefinitionCheckResult:
    """
    Check a definition and return the result.
    """

    pc = ProductChecker(definition)
    report = pc.check(filename)

    return DefinitionCheckResult(report=report)


def run_check(target: ResolvedTarget, filename: str) -> FullCheckResult:
    """
    Run a full check against both the project and definition, and return the
    result.
    """

    project_path = os.path.join(
        target.project.local_path, target.project.project_directory
    )

    project = import_project(project_path)

    project_result = check_against_project(project.models.Dataset, filename)

    if target.schema_path is None:
        definition_result = None
    else:
        definition_result = check_against_definition(target.schema_path, filename)

    return FullCheckResult(
        project_result=project_result, definition_result=definition_result
    )
