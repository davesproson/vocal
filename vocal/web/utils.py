import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import UploadFile, HTTPException, status

from vocal.checking.shared import (
    CheckOutcome,
    DefinitionCheckResult,
    ProjectCheckResult,
    Verdict,
    run_check,
)
from vocal.resolution import (
    NothingToCheck,
    Resolution,
    resolve_file,
    tokenise_conventions,
)
from vocal.utils import get_error_locs
from vocal.utils.conventions import FileConventions, read_file_conventions
from vocal.utils.registry import Registry
from vocal.web.landing import perform_landing
from vocal.web.models import (
    Check,
    CheckContext,
    CheckDefinition,
    CheckError,
    CheckProject,
    ResolverError,
    UnverifiedClaim,
)


def _load_registry() -> Registry:
    """Load the machine-local registry, falling back to an empty one.

    A machine that has never fetched anything has no registry file; the web
    checker treats that as "nothing registered" rather than an error, so a
    missing file resolves to an empty :class:`Registry`.
    """
    try:
        return Registry.load()
    except FileNotFoundError:
        return Registry()


def _has_vocal_claim(attrs: FileConventions, registry: Registry) -> bool:
    """Whether the file carries *any* recognisable vocal claim.

    The web checker has no ``-p`` / ``-d`` flag fallback, so a file must
    self-describe. It is recognisable when it declares a ``vocal_project_url``, a
    ``vocal_definitions_url``, or a ``Conventions`` token that name-matches an
    installed project. A file that does none of these (e.g. one carrying only
    external CF/ACDD tokens, or nothing at all) is not vocal-managed and is
    refused upfront — distinct from a verdict.

    A recognisable file is *always* given a verdict: even one whose mandatory
    standards aren't installed, or whose only ``Conventions`` standard is
    installed at the wrong major, resolves to INDETERMINATE rather than a
    refusal.
    """
    if attrs.project_urls:
        return True
    if attrs.definitions_url is not None:
        return True
    installed_names = {project.name for project in registry.projects.values()}
    return any(
        token.name in installed_names for token in tokenise_conventions(attrs.conventions)
    )


def _project_view(result: ProjectCheckResult) -> CheckProject:
    """Render one standards-axis model-check result for the results page."""
    if result.passed:
        return CheckProject(passed=True, errors=[])

    assert result.error is not None and result.nc_noval is not None
    locs, msgs = get_error_locs(result.error, result.nc_noval)
    errors = [CheckError(path=loc, message=msg) for loc, msg in zip(locs, msgs)]
    return CheckProject(passed=False, errors=errors)


def _definition_view(result: DefinitionCheckResult) -> CheckDefinition:
    """Render the product-axis structural check report for the results page."""
    view = CheckDefinition(passed=result.passed, warnings=False, comments=False, checks=[])
    if result.report is None:
        return view

    for check in result.report.checks:
        _check = Check(description=check.description)
        if check.passed:
            if check.has_comment and check.comment:
                view.comments = True
                _check.comment = check.comment
            if check.has_warning and check.warning:
                view.warnings = True
                _check.warning = check.warning
        elif check.error:
            _check.error = check.error
        view.checks.append(_check)

    return view


def _unverified_claims(
    resolution: Resolution, outcome: CheckOutcome
) -> list[UnverifiedClaim]:
    """Collect the fetch-it/update-it items that explain an INDETERMINATE verdict.

    Three sources, all carried through from resolution: unresolved mandatory
    claims (a missing project or pack — fetch it), claimed standards installed at
    a minor too old to run (update it), and opportunistic ``Conventions``
    standards whose project isn't installed (an informational comment). Advisory
    ``satisfies_standards`` warnings are not actionable here and are omitted.
    """
    claims: list[UnverifiedClaim] = []

    for failure in outcome.failures:
        claims.append(UnverifiedClaim(message=failure.message, hint=failure.hint))

    for target in resolution.projects:
        if not target.verifiable:
            claims.append(
                UnverifiedClaim(
                    message=(
                        f"{target.claimed_version} could not be verified: the "
                        f"installed project is at an older minor."
                    ),
                    hint=target.hint,
                )
            )

    for comment in outcome.comments:
        if comment.code == "standard_not_verified":
            claims.append(
                UnverifiedClaim(message=comment.message, hint=comment.hint)
            )

    return claims


def _build_context(resolution: Resolution, outcome: CheckOutcome) -> CheckContext:
    """Turn a resolution and its :class:`CheckOutcome` into a render context.

    A pure mapping from the check spine's result to the template's view models:
    the tri-state verdict, the per-axis results, and the fetch/update items.
    """
    context = CheckContext(verdict=outcome.verdict.value)

    for result in outcome.project_results:
        name = f"{result.target.project.name}-{result.target.project.major}"
        context.projects[name] = _project_view(result)

    if outcome.pack_result is not None:
        context.definitions[outcome.pack_result.target.product.name] = _definition_view(
            outcome.pack_result
        )

    context.unverified = _unverified_claims(resolution, outcome)
    return context


async def check_upload(
    file: UploadFile, *, upload_dir: Optional[Path] = None
) -> CheckContext:
    """Check an uploaded file against the standards and product it claims.

    The web flow drives :mod:`vocal.resolution` and the check spine exactly as
    ``vocal check`` does on the CLI, but without the ``-p`` / ``-d`` flag
    fallback: the file must self-describe, and nothing is ever fetched. The
    result is a tri-state :class:`~vocal.web.models.CheckContext`:

    - A file carrying no recognisable vocal claim is refused upfront with a
      distinct ``not_vocal_managed`` :class:`~vocal.web.models.ResolverError` —
      not a verdict.
    - Any other file is given a verdict. A file whose only claims resolve to
      nothing runnable (e.g. a ``Conventions`` standard installed at the wrong
      major) renders INDETERMINATE rather than being rejected.

    Args:
        file (UploadFile): The file to check.
        upload_dir: When set (``vocal web --upload-to``), the directory into
            which a PASS file is copied. On a PASS verdict :func:`perform_landing`
            stores the validated file and attaches the typed
            :class:`~vocal.web.models.Landing` result to the context; FAIL,
            INDETERMINATE, and upfront-refused files store nothing.

    Returns:
        CheckContext: The render context for the results page.
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
        registry = _load_registry()

        # The web UI has no flag fallback: a file that carries no recognisable
        # vocal claim cannot be checked. Refuse it upfront — distinct from a
        # verdict — with a clear "not a vocal-managed file" message.
        if not _has_vocal_claim(attrs, registry):
            context.error = ResolverError(
                code="not_vocal_managed",
                message="This file carries no recognisable vocal claim.",
                hint=(
                    "The web checker validates a file against the standards and "
                    "product it declares. Add a vocal_project_url, a "
                    "vocal_definitions_url, or a Conventions token naming an "
                    "installed standard."
                ),
            )
            return context

        try:
            resolution = resolve_file(file_path, attrs=attrs, registry=registry)
        except NothingToCheck:
            # The file is recognisable (the precondition passed) but nothing it
            # names is runnable here — e.g. a Conventions standard installed at a
            # different major. That is INDETERMINATE, not a refusal: surface the
            # resolution's warnings so the user knows what to fetch.
            return _indeterminate_unresolvable(file_path, attrs, registry)

        outcome = run_check(resolution, file_path)
        context = _build_context(resolution, outcome)

        # Storage is a post-verdict side effect: only a PASS file is copied into
        # the --upload-to directory, and only while the validated temp file
        # still exists (before this block's cleanup). FAIL / INDETERMINATE /
        # upfront-refused files return earlier and are never stored.
        if upload_dir is not None and outcome.verdict is Verdict.PASS:
            context.landing = perform_landing(
                is_pass=True,
                source_path=file_path,
                filename=file.filename,
                upload_dir=upload_dir,
            )

    return context


def _indeterminate_unresolvable(
    file_path: str, attrs: FileConventions, registry: Registry
) -> CheckContext:
    """Build the INDETERMINATE context for a recognisable file that resolves to
    nothing runnable.

    ``resolve_file`` raises :class:`~vocal.resolution.NothingToCheck` in this
    case rather than returning a resolution, so the resolver's warnings are
    re-derived for the surface by recording the opportunistic standards that
    weren't installed. The verdict is INDETERMINATE: the file claimed something
    vocal-managed, but nothing could be verified.
    """
    claims: list[UnverifiedClaim] = [
        UnverifiedClaim(
            message=f"{token} was not verified: no matching project is installed.",
            hint=f"Run 'vocal fetch' for {token.name}-{token.major} to verify it.",
        )
        for token in tokenise_conventions(attrs.conventions)
    ]
    return CheckContext(verdict="indeterminate", unverified=claims)
