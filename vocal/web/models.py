from pydantic import BaseModel, Field

from vocal.checking import CheckError, CheckComment, CheckWarning


class ResolverError(BaseModel):
    """A typed resolution failure surfaced on the results page.

    Mirrors the ``code`` / ``message`` / ``hint`` shape of the resolver's typed
    errors (:class:`vocal.resolution.ResolutionError`). ``code`` is drawn from
    the resolver's shared vocabulary (``project_missing``, ``project_too_old``,
    ``pack_missing``, ``pack_incompatible``, ``product_not_found``) for genuine
    resolution failures, or from the web-layer attribute-precondition vocabulary
    (``missing_conventions``, ``missing_pack_reference``) for files that cannot
    be resolved at all because the web UI has no ``-p`` / ``-d`` flag fallback.
    The web UI renders ``message`` and ``hint`` directly.
    """

    code: str
    message: str
    hint: str | None = None


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

    When resolution fails (or the file is missing the attributes the web flow
    requires), ``error`` carries the single typed failure and ``projects`` /
    ``definitions`` are empty. When resolution succeeds, ``error`` is ``None``
    and the validation results populate ``projects`` and ``definitions``.

    Attributes:
        projects (dict): The project (standard) validation result, keyed by
            ``{name}-{major}``.
        definitions (dict): The product-definition check result, keyed by
            product name.
        error (ResolverError | None): The typed resolution failure, if any.
    """

    projects: dict[str, CheckProject] = Field(default_factory=dict)
    definitions: dict[str, CheckDefinition] = Field(default_factory=dict)
    error: ResolverError | None = None
