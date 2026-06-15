"""The check spine: turn a :class:`~vocal.resolution.Resolution` into a
tri-state :class:`CheckOutcome`.

A :class:`~vocal.resolution.Resolution` answers *what should this file be
validated against?* along two independent axes (a set of standards-axis
``ProjectTarget``\\ s and an optional product-axis ``PackTarget``). The spine
answers the next question — *does the file honour those claims?* — by running
the pydantic model check for each **verifiable** project target and the schema
check for the pack target, then rolling the per-check results up into a single
tri-state :class:`Verdict`.

The file is verified *independently* against both the claimed project models and
the pack schema, so a non-compliant hand-authored pack cannot certify a bad file
(and vice versa).

The verdict roll-up (:func:`roll_up_verdict`) is a pure function with a fixed
precedence — **FAIL > INDETERMINATE > PASS**:

- **FAIL** if any check actually ran and the file violated it.
- else **INDETERMINATE** if any mandatory claim is unresolved (the resolution
  carries failures), any claim is unverifiable (installed but at a minor too old
  to run — see :class:`~vocal.resolution.ProjectTarget`), or *zero* checks ran.
- else **PASS**.

Unverifiable targets are deliberately *not* run (an older minor could spuriously
reject a legitimately newer-minor file); they survive in the resolution only to
force INDETERMINATE here. Nothing is ever fetched: the spine consumes exactly
what the resolver already found installed.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from vocal.checking.checking import CheckReport, ProductChecker
from vocal.netcdf import NetCDFReader
from vocal.resolution import (
    PackTarget,
    ProjectTarget,
    Resolution,
    ResolutionComment,
    ResolutionError,
    ResolutionWarning,
)
from vocal.utils import import_project


class Verdict(enum.Enum):
    """The tri-state outcome of a check.

    ``PASS`` — every check that ran passed and nothing was left unverified.
    ``FAIL`` — a check ran and the file violated it. ``INDETERMINATE`` — we could
    not finish verifying (something unresolved or installed too old, or there was
    nothing to run), so the file is neither certified nor condemned.
    """

    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ProjectCheckResult:
    """The result of checking a file against one standards-axis project model.

    ``target`` is the :class:`~vocal.resolution.ProjectTarget` this result is
    for, so a surface can attribute the result to the standard it verified.
    ``error`` is the pydantic :class:`ValidationError` raised by the validating
    pass (``None`` on success); ``nc_noval`` is the model built *without*
    validation, retained for error rendering.
    """

    target: ProjectTarget
    error: ValidationError | None = None
    nc_noval: BaseModel | None = None

    @property
    def passed(self) -> bool:
        """True when the model check raised no validation error."""
        return self.error is None


@dataclass(frozen=True)
class DefinitionCheckResult:
    """The result of checking a file against the product-axis pack schema.

    ``target`` is the :class:`~vocal.resolution.PackTarget` this result is for;
    ``report`` is the structural :class:`~vocal.checking.checking.CheckReport`.
    """

    target: PackTarget
    report: CheckReport | None = None

    @property
    def passed(self) -> bool:
        """True when the schema check produced a passing report."""
        return self.report is not None and self.report.passing


@dataclass(frozen=True)
class CheckOutcome:
    """The aggregated result of checking a file along both axes.

    ``project_results`` holds one entry per *verifiable* standards-axis target
    that was run; ``pack_result`` is the product-axis result (or ``None`` when
    the file declares no resolvable pack). ``failures``, ``warnings`` and
    ``comments`` are carried through from the
    :class:`~vocal.resolution.Resolution` so a surface can render unresolved
    mandatory claims, advisory notes and informational notes alongside the
    verdict. ``verdict`` is the rolled-up tri-state result.
    """

    project_results: list[ProjectCheckResult]
    pack_result: DefinitionCheckResult | None
    failures: list[ResolutionError]
    warnings: list[ResolutionWarning]
    comments: list[ResolutionComment]
    verdict: Verdict

    @property
    def passed(self) -> bool:
        """True only when the verdict is :attr:`Verdict.PASS`."""
        return self.verdict is Verdict.PASS


def check_against_project(target: ProjectTarget, filename: str) -> ProjectCheckResult:
    """Check a file against one project's pydantic ``Dataset`` model.

    Imports the project package from the target's installed location and runs the
    model both without validation (to retain a model for error rendering) and
    with validation (to surface violations as a :class:`ValidationError`).
    """
    project_path = os.path.join(
        target.project.local_path, target.project.project_directory
    )
    project = import_project(project_path)
    model = project.models.Dataset

    nc = NetCDFReader(filename)
    nc_noval: BaseModel | None = None
    try:
        nc_noval = nc.to_model(model, validate=False)
        nc.to_model(model)
    except ValidationError as err:
        return ProjectCheckResult(target=target, error=err, nc_noval=nc_noval)

    return ProjectCheckResult(target=target)


def check_against_definition(target: PackTarget, filename: str) -> DefinitionCheckResult:
    """Check a file against the pack's product schema and return the report."""
    pc = ProductChecker(target.schema_path)
    report = pc.check(filename)
    return DefinitionCheckResult(target=target, report=report)


def roll_up_verdict(
    resolution: Resolution,
    project_results: list[ProjectCheckResult],
    pack_result: DefinitionCheckResult | None,
) -> Verdict:
    """Roll per-check results and the resolution up into a tri-state verdict.

    A pure function with precedence **FAIL > INDETERMINATE > PASS**: a check that
    ran and was violated wins outright; otherwise an unresolved mandatory claim,
    an unverifiable (too-old) target, or zero checks having run yields
    INDETERMINATE; otherwise PASS.
    """
    ran: list[ProjectCheckResult | DefinitionCheckResult] = list(project_results)
    if pack_result is not None:
        ran.append(pack_result)

    if any(not result.passed for result in ran):
        return Verdict.FAIL

    unresolved_mandatory = bool(resolution.failures)
    unverifiable = any(not target.verifiable for target in resolution.projects)
    if unresolved_mandatory or unverifiable or not ran:
        return Verdict.INDETERMINATE

    return Verdict.PASS


def run_check(resolution: Resolution, filename: str) -> CheckOutcome:
    """Run every check the resolution calls for and roll up the verdict.

    Runs the model check for each *verifiable* standards-axis target (unverifiable
    too-old targets are skipped — they only force INDETERMINATE) and the schema
    check for the pack target, then aggregates the results, the resolution's
    failures, warnings and comments, and the rolled-up :class:`Verdict` into a
    :class:`CheckOutcome`.
    """
    project_results = [
        check_against_project(target, filename)
        for target in resolution.projects
        if target.verifiable
    ]

    pack_result = (
        check_against_definition(resolution.pack, filename)
        if resolution.pack is not None
        else None
    )

    verdict = roll_up_verdict(resolution, project_results, pack_result)

    return CheckOutcome(
        project_results=project_results,
        pack_result=pack_result,
        failures=resolution.failures,
        warnings=resolution.warnings,
        comments=resolution.comments,
        verdict=verdict,
    )
