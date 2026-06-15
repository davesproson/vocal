"""Unit tests for the check spine (:mod:`vocal.checking.shared`).

The spine consumes a :class:`~vocal.resolution.Resolution` and produces a
tri-state :class:`~vocal.checking.shared.CheckOutcome`. These tests drive
:func:`~vocal.checking.shared.run_check` with synthetic resolutions and the two
check primitives patched to canned results, asserting on the returned
``CheckOutcome`` and its ``verdict`` through the module's public interface — not
on project import, netCDF reading, or any private state.

The verdict roll-up contract under test (precedence **FAIL > INDETERMINATE >
PASS**):

- FAIL if any check ran and the file violated it;
- else INDETERMINATE if any mandatory claim is unresolved, any claim is
  unverifiable (installed-but-too-old), or zero checks ran;
- else PASS.
"""

from types import SimpleNamespace
from unittest.mock import patch

import vocal.checking.shared as shared
from vocal.checking.shared import (
    CheckOutcome,
    DefinitionCheckResult,
    ProjectCheckResult,
    Verdict,
    run_check,
)
from vocal.resolution import (
    PackTarget,
    ProjectMissing,
    ProjectTarget,
    Resolution,
    ResolutionComment,
    ResolutionWarning,
)
from vocal.utils.registry import Project


def _project(name: str = "MYSTD", major: int = 2, minor: int = 3) -> Project:
    return Project(
        name=name,
        major=major,
        minor=minor,
        project_directory=name.lower(),
        local_path=f"/cache/projects/{name.lower()}",
    )


def _project_target(
    name: str = "MYSTD",
    *,
    mandatory: bool = False,
    verifiable: bool = True,
) -> ProjectTarget:
    return ProjectTarget(
        project=_project(name),
        mandatory=mandatory,
        claimed_version=None,
        verifiable=verifiable,
    )


def _pack_target() -> PackTarget:
    """A PackTarget whose pack/product are irrelevant — the schema check is
    patched, so only its presence (and the patched result) matters."""
    return PackTarget(
        pack=SimpleNamespace(),
        product=SimpleNamespace(),
        schema_path="/cache/packs/host/product_foo.json",
    )


def _project_result(target: ProjectTarget, *, passed: bool) -> ProjectCheckResult:
    # `passed` keys off `error is None`; a sentinel error is enough to flip it.
    return ProjectCheckResult(target=target, error=None if passed else object())


def _pack_result(target: PackTarget, *, passed: bool) -> DefinitionCheckResult:
    return DefinitionCheckResult(target=target, report=SimpleNamespace(passing=passed))


def _run(resolution: Resolution, *, projects_pass=True, pack_pass=True) -> CheckOutcome:
    """Run ``run_check`` with both primitives patched to canned results keyed off
    ``projects_pass`` / ``pack_pass``."""
    with (
        patch.object(
            shared,
            "check_against_project",
            side_effect=lambda target, _f: _project_result(
                target, passed=projects_pass
            ),
        ),
        patch.object(
            shared,
            "check_against_definition",
            side_effect=lambda target, _f: _pack_result(target, passed=pack_pass),
        ),
    ):
        return run_check(resolution, "foo_20260522.nc")


class TestPass:
    def test_all_checks_pass(self) -> None:
        resolution = Resolution(
            projects=[_project_target()], pack=_pack_target()
        )
        outcome = _run(resolution)

        assert outcome.verdict is Verdict.PASS
        assert outcome.passed
        assert len(outcome.project_results) == 1
        assert outcome.pack_result is not None

    def test_pack_only(self) -> None:
        outcome = _run(Resolution(pack=_pack_target()))

        assert outcome.verdict is Verdict.PASS
        assert outcome.project_results == []
        assert outcome.pack_result is not None

    def test_project_only(self) -> None:
        outcome = _run(Resolution(projects=[_project_target()]))

        assert outcome.verdict is Verdict.PASS
        assert outcome.pack_result is None


class TestFail:
    def test_project_violation(self) -> None:
        resolution = Resolution(projects=[_project_target()])
        assert _run(resolution, projects_pass=False).verdict is Verdict.FAIL

    def test_pack_violation(self) -> None:
        resolution = Resolution(projects=[_project_target()], pack=_pack_target())
        assert _run(resolution, pack_pass=False).verdict is Verdict.FAIL

    def test_fail_beats_unresolved_mandatory_and_too_old(self) -> None:
        """A ran-and-violated check is FAIL even alongside a missing/too-old claim."""
        resolution = Resolution(
            projects=[
                _project_target(),  # this one runs and (below) fails
                _project_target("OLDSTD", verifiable=False),  # too-old
            ],
            failures=[ProjectMissing("missing", "fetch it")],
        )
        assert _run(resolution, projects_pass=False).verdict is Verdict.FAIL


class TestIndeterminate:
    def test_zero_checks_ran(self) -> None:
        """No verifiable project and no pack: nothing ran."""
        assert _run(Resolution()).verdict is Verdict.INDETERMINATE

    def test_only_unverifiable_target(self) -> None:
        """An installed-but-too-old target is not run, so zero checks ran."""
        resolution = Resolution(projects=[_project_target(verifiable=False)])
        outcome = _run(resolution)

        assert outcome.verdict is Verdict.INDETERMINATE
        assert outcome.project_results == []

    def test_unresolved_mandatory_claim(self) -> None:
        """A passing check alongside an unresolved mandatory failure is INDETERMINATE."""
        resolution = Resolution(
            projects=[_project_target()],
            failures=[ProjectMissing("missing", "fetch it")],
        )
        assert _run(resolution).verdict is Verdict.INDETERMINATE

    def test_unverifiable_alongside_passing(self) -> None:
        """A too-old target forces INDETERMINATE even when other checks pass."""
        resolution = Resolution(
            projects=[
                _project_target(),
                _project_target("OLDSTD", verifiable=False),
            ],
            pack=_pack_target(),
        )
        assert _run(resolution).verdict is Verdict.INDETERMINATE


class TestRunsAndAggregation:
    def test_unverifiable_target_not_run(self) -> None:
        """Only verifiable targets are run; the too-old one is skipped."""
        verifiable = _project_target("MYSTD")
        too_old = _project_target("OLDSTD", verifiable=False)
        outcome = _run(Resolution(projects=[verifiable, too_old], pack=_pack_target()))

        assert [r.target for r in outcome.project_results] == [verifiable]

    def test_aggregates_n_projects_plus_pack(self) -> None:
        targets = [_project_target(f"STD{i}") for i in range(3)]
        outcome = _run(Resolution(projects=targets, pack=_pack_target()))

        assert outcome.verdict is Verdict.PASS
        assert [r.target for r in outcome.project_results] == targets
        assert outcome.pack_result is not None

    def test_failures_warnings_and_comments_carried_through(self) -> None:
        failure = ProjectMissing("missing", "fetch it")
        warning = ResolutionWarning(code="satisfies_standards_unmet", message="advisory")
        comment = ResolutionComment(code="standard_not_verified", message="skipped")
        resolution = Resolution(
            projects=[_project_target()],
            failures=[failure],
            warnings=[warning],
            comments=[comment],
        )
        outcome = _run(resolution)

        assert outcome.failures == [failure]
        assert outcome.warnings == [warning]
        assert outcome.comments == [comment]
