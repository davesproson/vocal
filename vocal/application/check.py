"""Check a netCDF file against the standards and product it claims.

``vocal check`` is a thin CLI adapter over the two-axis check spine. It reads the
file's vocal-managed global attributes, builds a :class:`~vocal.resolution.Resolution`
along two independent axes — a *standards* axis (the pydantic ``Dataset`` models
the file is verified against) and a *product* axis (the single pack schema) — hands
it to :func:`~vocal.checking.shared.run_check`, and renders the resulting tri-state
:class:`~vocal.checking.shared.CheckOutcome`.

The verdict drives the exit code: ``0`` on PASS, ``1`` on FAIL, ``2`` on
INDETERMINATE (something the file claims is not installed, or installed at a minor
too old to verify, or there was nothing to run). The distinct INDETERMINATE code
lets a script tell "missing dependency" from "bad file".

``-p`` and ``-d`` are a **per-axis, symmetric** override: ``-p`` overrides only
the standards axis (one or more project repo roots, each treated as a *mandatory*
standard), ``-d`` overrides only the product axis (a single product schema). The
un-named axis is still resolved from the file. ``--specified-only`` suppresses
resolution of the un-named axis entirely, so the file is checked against exactly
— and only — what ``-p``/``-d`` name.

Nothing is ever fetched implicitly. ``--fetch`` (and ``vocal fetch --for``) are
the only paths that fetch, and they run strictly as an opt-in pre-step.
"""

import os
from typing import Optional

import typer

from rich.progress import Progress, SpinnerColumn, TextColumn

from vocal.application.fetch import fetch_for_file, summarise_outcomes
from vocal.application.fetch_gate import confirm_file_fetch
from vocal.checking.checking import CheckReport
from vocal.checking.shared import (
    CheckOutcome,
    DefinitionCheckResult,
    ProjectCheckResult,
    Verdict,
    run_check,
)
from vocal.conventions_file import ConventionsFile
from vocal.exceptions import VocalError
from vocal.manifest import ManifestProduct, build_manifest
from vocal.resolution import (
    NothingToCheck,
    PackMissing,
    PackTarget,
    ProductNotFound,
    ProjectMissing,
    ProjectTarget,
    Resolution,
    ResolutionComment,
    ResolutionError,
    ResolutionWarning,
    resolve_file,
)
from vocal.utils.registry import Pack, Project
from ..utils import get_error_locs, TextStyles, Printer
from ..utils.conventions import FileConventions, read_file_conventions

LINE_LEN = 50

# Exit codes: PASS is 0; FAIL and INDETERMINATE are kept distinct so a script can
# tell "the file is bad" (1) from "the check couldn't be completed" (2).
EXIT_FAIL = 1
EXIT_INDETERMINATE = 2

TS = TextStyles()
p = Printer()


class NoopProgress:
    """A no-op progress context manager for when --quiet suppresses the real one."""

    def __enter__(self) -> "NoopProgress":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def add_task(self, description: str = "", total: Optional[float] = None) -> None:
        pass

    def remove_task(self, task_id: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Building the Resolution (file-resolved axes + per-axis -p/-d overrides).
# ---------------------------------------------------------------------------

# The resolver's failures and warnings are axis-attributable by type/code. When
# one axis is overridden by a flag, the file-resolved problems for *that* axis are
# irrelevant (the user replaced it), so they are filtered out by these predicates.
_PRODUCT_FAILURES = (PackMissing, ProductNotFound)


def _is_product_failure(failure: ResolutionError) -> bool:
    """Whether a resolver failure belongs to the product axis."""
    return isinstance(failure, _PRODUCT_FAILURES)


def _is_product_warning(warning: ResolutionWarning) -> bool:
    """Whether a resolver warning belongs to the product axis."""
    return warning.code == "satisfies_standards_unmet"


def _override_project_target(path: str) -> ProjectTarget:
    """Build a mandatory standards-axis target from a supplied project repo root.

    Reads the repo's ``conventions.yaml`` for the standard's identity; the spine
    imports the package from the same root at check time. A directly supplied
    project is always run (never "too old" — there is no claimed version to be
    older than).
    """
    repo = path.rstrip("/")
    conventions = ConventionsFile.load(repo)
    project = Project(
        name=conventions.name,
        major=conventions.major,
        minor=conventions.minor,
        project_directory=conventions.project_directory,
        local_path=repo,
    )
    return ProjectTarget(
        project=project, mandatory=True, claimed_version=conventions.version
    )


def _override_pack_target(schema_path: str) -> PackTarget:
    """Build a mandatory product-axis target from a supplied product schema path.

    The spine validates against ``schema_path`` directly, so the surrounding pack
    is a thin synthetic carrier (the override bypasses pack resolution and product
    routing entirely — the user named the schema explicitly).
    """
    name = os.path.basename(schema_path)
    product = ManifestProduct(name=name, file_pattern="", schema=name)
    manifest = build_manifest(
        version=0, url="", filecodec={}, satisfies_standards=(), products=(product,)
    )
    pack = Pack(manifest=manifest, local_path=os.path.dirname(os.path.abspath(schema_path)))
    return PackTarget(pack=pack, product=product, schema_path=schema_path)


def _build_resolution(
    filename: str,
    attrs: FileConventions,
    project_paths: list[str],
    definition_path: Optional[str],
    specified_only: bool,
) -> Resolution:
    """Compose a :class:`Resolution` from file-resolved axes and -p/-d overrides.

    Each axis is independent: a named axis (``-p`` standards, ``-d`` product) is
    taken from the supplied path(s) as mandatory; an un-named axis is resolved from
    the file — unless ``specified_only`` suppresses it. The file-resolved problems
    for an overridden axis are dropped (the override replaces them).

    Raises:
        NothingToCheck: nothing was named and nothing resolved from the file.
    """
    override_standards = bool(project_paths)
    override_product = definition_path is not None
    has_overrides = override_standards or override_product

    resolve_standards_from_file = not specified_only and not override_standards
    resolve_product_from_file = not specified_only and not override_product

    file_resolution: Optional[Resolution] = None
    if resolve_standards_from_file or resolve_product_from_file:
        try:
            file_resolution = resolve_file(filename, attrs=attrs)
        except NothingToCheck:
            # The file carries nothing of its own. With overrides present that is
            # fine (we use them); with none, there is genuinely nothing to check.
            if not has_overrides:
                raise
            file_resolution = None

    resolution = Resolution()

    if override_standards:
        resolution.projects.extend(
            _override_project_target(path) for path in project_paths
        )
    elif resolve_standards_from_file and file_resolution is not None:
        resolution.projects.extend(file_resolution.projects)
        resolution.failures.extend(
            f for f in file_resolution.failures if not _is_product_failure(f)
        )
        resolution.warnings.extend(
            w for w in file_resolution.warnings if not _is_product_warning(w)
        )
        # Opportunistic-skip comments are standards-axis; carry them only when the
        # file's standards axis is in play (an override replaces them).
        resolution.comments.extend(file_resolution.comments)

    if override_product:
        assert definition_path is not None  # established by override_product
        resolution.pack = _override_pack_target(definition_path)
    elif resolve_product_from_file and file_resolution is not None:
        resolution.pack = file_resolution.pack
        resolution.failures.extend(
            f for f in file_resolution.failures if _is_product_failure(f)
        )
        resolution.warnings.extend(
            w for w in file_resolution.warnings if _is_product_warning(w)
        )

    if not resolution.projects and resolution.pack is None and not resolution.failures:
        raise NothingToCheck(
            "Nothing to check.",
            "Name an axis with -p (a project) and/or -d (a product definition); "
            "--specified-only ignores the file's own claims.",
        )

    return resolution


# ---------------------------------------------------------------------------
# Rendering a CheckOutcome.
# ---------------------------------------------------------------------------


def _render_project_result(result: ProjectCheckResult, filename: str) -> None:
    """Render one standards-axis model-check result."""
    name = f"{result.target.project.name}-{result.target.project.major}"
    p.print_err(
        f"Checking {TS.BOLD}{filename}{TS.ENDC} against "
        f"{TS.BOLD}{name}{TS.ENDC} standard... ",
        end="",
    )

    if result.passed:
        p.print_err(f"{TS.OKGREEN}{TS.BOLD}OK!{TS.ENDC}\n")
        return

    assert result.error is not None and result.nc_noval is not None
    p.print_err(f"{TS.FAIL}{TS.BOLD}ERROR!{TS.ENDC}\n")
    locs, msgs = get_error_locs(result.error, result.nc_noval)
    for loc, msg in zip(locs, msgs):
        p.print_err(f"{TS.FAIL}{TS.BOLD}✗{TS.ENDC} {loc}: {msg}")
    p.print_err()


def _render_unverifiable(target: ProjectTarget, filename: str) -> None:
    """Render a standards-axis claim that is installed but too old to verify.

    Such a target is deliberately not run (an older minor could spuriously reject
    a legitimately newer-minor file); it forces INDETERMINATE and carries an
    ``--update`` hint.
    """
    name = f"{target.project.name}-{target.project.major}"
    p.print_err(
        f"Checking {TS.BOLD}{filename}{TS.ENDC} against "
        f"{TS.BOLD}{name}{TS.ENDC} standard... ",
        end="",
    )
    p.print_err(f"{TS.WARNING}{TS.BOLD}SKIPPED{TS.ENDC}\n")
    p.print_warn(
        f"{TS.BOLD}{TS.WARNING}!{TS.ENDC} {target.claimed_version} could not be "
        f"verified: the installed project is at an older minor."
    )
    if target.hint:
        p.print_warn(f"  {target.hint}")
    p.print_err()


def print_checks(pc: CheckReport, filename: str, specification: str) -> None:
    """Render a product-axis structural check report."""
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
                p.print_warn(f"{TS.BOLD}{TS.WARNING}!{TS.ENDC} {check.description}")
                p.print_warn(
                    f"{TS.BOLD}{TS.WARNING}  ➜ {check.warning.path}: {TS.ENDC}"
                    f"{TS.WARNING}{check.warning.message}{TS.ENDC}"
                )
            if check.has_comment and check.comment:
                p.print_comment(f"{TS.BOLD}{TS.OKBLUE}i{TS.ENDC} {check.description}")
                p.print_comment(
                    f"{TS.BOLD}{TS.OKBLUE}  ➜ {check.comment.path}: {TS.ENDC}"
                    f"{TS.OKBLUE}{check.comment.message}{TS.ENDC}"
                )
            else:
                p.print(f"{TS.BOLD}{TS.OKGREEN}✔{TS.ENDC} {check.description}")
        elif check.error:
            p.print_err(f"{TS.FAIL}{TS.BOLD}✗{TS.ENDC} {check.description}")
            p.print_err(
                f"{TS.FAIL}  ➜ {TS.BOLD}{check.error.path}:{TS.ENDC} "
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


def _render_pack_result(result: DefinitionCheckResult, filename: str) -> None:
    """Render the product-axis structural check report."""
    if result.report is None:
        return
    print_checks(result.report, filename, result.target.schema_path)


def _render_failure(failure: ResolutionError) -> None:
    """Render an unresolved mandatory claim: its message and the resolver hint."""
    p.print_err(f"\n{TS.BOLD}{TS.FAIL}✗{TS.ENDC} {failure.message}")
    if failure.hint:
        p.print_err(f"  {failure.hint}")


def _render_warning(warning: ResolutionWarning) -> None:
    """Render an advisory resolution warning (e.g. an unmet ``satisfies_standards``
    assertion)."""
    p.print_warn(f"\n{TS.BOLD}{TS.WARNING}!{TS.ENDC} {warning.message}")
    if warning.hint:
        p.print_warn(f"  {warning.hint}")


def _render_comment(comment: ResolutionComment) -> None:
    """Render one standards-axis comment's message and hint (an opportunistic
    standard skipped because no project is installed). Suppressed unless ``-c``."""
    p.print_comment(f"{TS.BOLD}{TS.OKBLUE}i{TS.ENDC} {comment.message}")
    if comment.hint:
        p.print_comment(f"  {comment.hint}")


def _render_comments(outcome: CheckOutcome) -> None:
    """Render the standards-axis comments alongside the standards-axis results.

    The individual notes are ``-c`` gated, but the count line is always shown when
    any comment exists, so a plain check still signals that comments are available
    (the hint drops once ``-c`` is on). Mirrors the product box's comment line."""
    if not outcome.comments:
        return
    for comment in outcome.comments:
        _render_comment(comment)
    hint = "" if p.comments else " (run with -c)"
    n = len(outcome.comments)
    p.print_err(f"{TS.BOLD}{TS.OKBLUE}i{TS.ENDC} {n} comments{hint}.")
    p.print_err()


def _render_fetch_hint(outcome: CheckOutcome, filename: str, fetched: bool) -> None:
    """Point the user at ``vocal fetch --for`` when a mandatory resource is missing.

    Only a missing project/pack is fetchable from the file's own declarations;
    suppressed when ``--fetch`` already ran (re-suggesting it would not help).
    """
    if fetched:
        return
    if any(isinstance(f, (ProjectMissing, PackMissing)) for f in outcome.failures):
        p.print_err(
            f"  Run 'vocal fetch --for {filename}' to fetch what the file declares, "
            f"then re-check.\n"
        )


def _render_verdict(verdict: Verdict) -> None:
    """Render the rolled-up tri-state verdict banner."""
    p.print_err()
    p.print_line_err(LINE_LEN, "=")
    if verdict is Verdict.PASS:
        p.print_err(f"{TS.BOLD}{TS.OKGREEN}✔ PASS{TS.ENDC}")
    elif verdict is Verdict.FAIL:
        p.print_err(f"{TS.BOLD}{TS.FAIL}✗ FAIL{TS.ENDC}")
    else:
        p.print_err(
            f"{TS.BOLD}{TS.WARNING}? INDETERMINATE{TS.ENDC} — "
            f"the check could not be completed."
        )
    p.print_line_err(LINE_LEN, "=")


def _render(
    resolution: Resolution, outcome: CheckOutcome, filename: str, *, fetched: bool
) -> None:
    """Render a full per-axis check outcome and the final verdict."""
    p.print_err()

    for result in outcome.project_results:
        _render_project_result(result, filename)

    for target in resolution.projects:
        if not target.verifiable:
            _render_unverifiable(target, filename)

    # Standards-axis comments (opportunistic standards skipped) belong with the
    # standards-axis results, alongside where a standard's errors would be shown.
    _render_comments(outcome)

    if outcome.pack_result is not None:
        _render_pack_result(outcome.pack_result, filename)

    for failure in outcome.failures:
        _render_failure(failure)

    _render_fetch_hint(outcome, filename, fetched)

    for warning in outcome.warnings:
        _render_warning(warning)

    _render_verdict(outcome.verdict)


def _exit_for(verdict: Verdict) -> None:
    """Raise the typer exit matching the verdict (PASS returns cleanly)."""
    if verdict is Verdict.FAIL:
        raise typer.Exit(code=EXIT_FAIL)
    if verdict is Verdict.INDETERMINATE:
        raise typer.Exit(code=EXIT_INDETERMINATE)


def command(
    filename: str = typer.Argument(metavar="FILE", help="The netCDF file to check"),
    project: Optional[list[str]] = typer.Option(
        None,
        "-p",
        "--project",
        help=(
            "Override the standards axis: path(s) to vocal project repo roots, "
            "each checked as a mandatory standard. Pass multiple times for "
            "multiple standards. The product axis is still resolved from the file "
            "unless --specified-only is given."
        ),
    ),
    definition: Optional[str] = typer.Option(
        None,
        "-d",
        "--definition",
        help=(
            "Override the product axis: a single product definition (schema) to "
            "check against, treated as mandatory. The standards axis is still "
            "resolved from the file unless --specified-only is given."
        ),
    ),
    specified_only: bool = typer.Option(
        False,
        "--specified-only",
        help=(
            "Check the file against only the axes named by -p/-d, suppressing "
            "resolution of the un-named axis entirely."
        ),
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
    fetch: bool = typer.Option(
        False,
        "--fetch",
        help=(
            "Before checking, ensure the resources the file declares about "
            "itself (its vocal_project_url and any vocal_definitions_url) are "
            "fetched, then check in one step. Idempotent — resources already "
            "present are skipped. Cannot be combined with -p (opposed modes: "
            "derive-from-file vs supply-paths)."
        ),
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Do not print any output."),
    yes: bool = typer.Option(
        False,
        "-y",
        "--yes",
        help=(
            "Consent up front to fetching a project declared inside the file "
            "(--fetch). Lets the confirmation gate proceed non-interactively."
        ),
    ),
    comments: bool = typer.Option(False, "-c", "--comments", help="Print comments."),
    no_color: bool = typer.Option(
        False, "--no-color", help="Do not print colored output."
    ),
) -> None:
    """Check a netCDF file against the standards and product it claims."""
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

    if fetch and project:
        # Opposed modes: --fetch derives the sources from the file, while -p
        # supplies them by hand. They cannot be combined.
        p.print_err(
            f"\n{TS.BOLD}{TS.FAIL}✗{TS.ENDC} --fetch cannot be combined with -p."
        )
        p.print_err(
            "  --fetch derives the project (and pack) from the file; -p supplies "
            "a project path manually. Use one or the other.\n"
        )
        raise typer.Exit(code=1)

    # Gate the file-driven fetch before the progress spinner: a project URL
    # declared inside an untrusted file means code that will run on check. The
    # prompt fires here so it does not fight a transient spinner for the terminal.
    if fetch:
        try:
            confirm_file_fetch(
                filename, route="check", yes=yes, quiet=quiet, no_color=no_color
            )
        except VocalError as e:
            # A security refusal (e.g. the can't-prompt BLOCKED error under -q)
            # must always be visible, so render it straight to stderr rather than
            # via the quiet-respecting Printer.
            typer.echo(f"\n{TS.BOLD}{TS.FAIL}✗{TS.ENDC} {e.message}", err=True)
            if e.hint:
                typer.echo(f"  {e.hint}\n", err=True)
            raise typer.Exit(code=1)

    if quiet:
        progress_context: Progress | NoopProgress = NoopProgress()
    else:
        progress_context = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        )

    with progress_context as progress:
        if fetch:
            # Ensure the file's declared resources are present, then fall through
            # into the normal resolve-and-check flow below. Idempotent.
            task = progress.add_task(description="Fetching resources...", total=None)
            try:
                outcomes = fetch_for_file(filename)
            except VocalError as e:
                _render_failure(e)
                raise typer.Exit(code=1)
            finally:
                task and progress.remove_task(task)

            if not quiet and outcomes:
                summarise_outcomes(outcomes)

        task = progress.add_task(description="Checking file...", total=None)
        try:
            attrs = read_file_conventions(filename)
            try:
                resolution = _build_resolution(
                    filename,
                    attrs,
                    project_paths=project or [],
                    definition_path=definition,
                    specified_only=specified_only,
                )
            except VocalError as e:
                _render_failure(e)
                raise typer.Exit(code=1)

            outcome = run_check(resolution, filename)
        finally:
            task and progress.remove_task(task)

    _render(resolution, outcome, filename, fetched=fetch)
    _exit_for(outcome.verdict)
