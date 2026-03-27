from pydantic import BaseModel, Field

from vocal.checking import CheckError, CheckComment, CheckWarning


class Check(BaseModel):
    """A class to hold the context of a check."""

    description: str
    comment: CheckComment | None = None
    warning: CheckWarning | None = None
    error: CheckError | None = None


class CheckProject(BaseModel):
    """A class to hold the context of a project."""

    passed: bool
    errors: list[CheckError]


class CheckDefinition(BaseModel):
    """A class to hold the context of a definition."""

    passed: bool
    warnings: bool
    comments: bool
    checks: list


class CheckContext(BaseModel):
    """
    A class to hold the context of a check, to be used in the web API.

    Attributes:
        projects (dict): A dictionary of projects to check against.
        definitions (dict): A dictionary of definitions to check against.
        errors (list): A list of errors.
    """

    projects: dict[str, CheckProject] = Field(default_factory=dict)
    definitions: dict[str, CheckDefinition] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
