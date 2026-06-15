"""Integration tests for the ``vocal check`` CLI command.

``vocal check`` is a thin adapter over the two-axis check spine: it reads a
file's vocal-managed global attributes, builds a
:class:`~vocal.resolution.Resolution` (resolving the un-named axes from the file
and overriding the named ones from ``-p``/``-d``), runs
:func:`~vocal.checking.shared.run_check`, and renders the tri-state
:class:`~vocal.checking.shared.CheckOutcome`.

These tests drive the command end-to-end through ``typer.testing.CliRunner``
with a synthetic registry and the spine's two check executors patched, asserting
observable behaviour: the exit code (0 PASS / 1 FAIL / 2 INDETERMINATE), which
axis was actually checked, and the rendered hints.
"""

from pathlib import Path
from unittest.mock import patch

import netCDF4
import typer
from typer.testing import CliRunner

from vocal.application.check import command
from vocal.checking.shared import (
    DefinitionCheckResult,
    ProjectCheckResult,
)
from vocal.manifest import ManifestProduct, build_manifest
from vocal.utils.registry import Pack, Project, Registry
from vocal.versioning import VersionConstraint


runner = CliRunner()

PROJECT_URL = "https://host/mystd.git"
PACK_URL = "https://host/packs"

# The pack's own filecodec, expanding the {date} placeholder in the products'
# file_patterns below.
FILECODEC = {"date": {"regex": r"\d{8}"}}


def _app() -> typer.Typer:
    app = typer.Typer()
    app.command()(command)
    return app


def _make_nc(
    tmp_path: Path,
    name: str = "foo_20260522.nc",
    *,
    conventions: str | None = None,
    project_url: str | None = None,
    definitions_url: str | None = None,
    definitions_version: int | None = None,
) -> str:
    """Write a minimal netCDF file carrying the given vocal-managed attributes."""
    path = str(tmp_path / name)
    with netCDF4.Dataset(path, "w") as nc:
        if conventions is not None:
            nc.Conventions = conventions
        if project_url is not None:
            nc.vocal_project_url = project_url
        if definitions_url is not None:
            nc.vocal_definitions_url = definitions_url
        if definitions_version is not None:
            nc.vocal_definitions_version = definitions_version
    return path


def _project(
    name: str = "MYSTD",
    major: int = 2,
    minor: int = 3,
    url: str = PROJECT_URL,
) -> Project:
    return Project(
        name=name,
        major=major,
        minor=minor,
        project_directory="mystd",
        local_path="/cache/projects/mystd",
        url=url,
    )


def _pack(
    url: str = PACK_URL,
    version: int = 3,
    satisfies=None,
    products=None,
    local_path: str = "/cache/packs/host-packs/v3",
) -> Pack:
    if products is None:
        products = [
            ManifestProduct(
                name="foo", file_pattern="foo_{date}", schema="product_foo.json"
            )
        ]
    if satisfies is None:
        satisfies = [VersionConstraint(name="MYSTD", major=2, min_minor=3)]
    manifest = build_manifest(
        version=version,
        url=url,
        filecodec=FILECODEC,
        satisfies_standards=satisfies,
        products=products,
    )
    return Pack(manifest=manifest, local_path=local_path)


def _registry(project: Project | None = None, pack: Pack | None = None) -> Registry:
    registry = Registry()
    if project is not None:
        registry.add_project(project)
    if pack is not None:
        registry.add_pack(pack)
    return registry


def _invoke(
    args,
    registry: Registry,
    *,
    project_pass: bool = True,
    pack_pass: bool = True,
):
    """Run the command with the registry loaded synthetically and the spine's two
    check executors patched to return pass/fail without touching netCDF/pydantic.

    The spine's ``run_check`` calls these module-globals, so patching them in
    ``vocal.checking.shared`` controls every check the command runs.
    """

    def _fake_project_check(target, filename) -> ProjectCheckResult:
        # ``error`` non-None marks a failed model check; passed otherwise.
        return ProjectCheckResult(
            target=target,
            error=None if project_pass else _PydErr(),
            nc_noval=None if project_pass else object(),
        )

    def _fake_pack_check(target, filename) -> DefinitionCheckResult:
        return DefinitionCheckResult(target=target, report=_Report(pack_pass))

    with (
        patch("vocal.resolution.Registry.load", return_value=registry),
        patch(
            "vocal.checking.shared.check_against_project",
            side_effect=_fake_project_check,
        ),
        patch(
            "vocal.checking.shared.check_against_definition",
            side_effect=_fake_pack_check,
        ),
    ):
        return runner.invoke(_app(), args)


class _Report:
    """A stand-in for a structural CheckReport: only ``passing`` and the lists the
    renderer reads are exercised."""

    def __init__(self, passing: bool) -> None:
        self.passing = passing
        self.checks: list = []
        self.warnings: list = []
        self.errors: list = []
        self.comments: list = []


class _PydErr(Exception):
    """A stand-in for a pydantic ValidationError; the renderer is also patched
    away in tests that exercise a FAIL so error_locs is never reached."""


# ---------------------------------------------------------------------------
# Exit codes: the tri-state verdict drives 0 / 1 / 2.
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_pass_exits_zero(self, tmp_path: Path) -> None:
        """All claims installed and verified, file conforms → PASS → exit 0."""
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            definitions_url=PACK_URL,
            definitions_version=3,
        )
        result = _invoke([nc], _registry(project=_project(), pack=_pack()))

        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_fail_exits_one(self, tmp_path: Path) -> None:
        """A check ran and the file violated it → FAIL → exit 1."""
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            definitions_url=PACK_URL,
            definitions_version=3,
        )
        # The product schema check fails.
        with patch("vocal.application.check.print_checks"):
            result = _invoke(
                [nc], _registry(project=_project(), pack=_pack()), pack_pass=False
            )

        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_indeterminate_exits_two_on_missing_mandatory(
        self, tmp_path: Path
    ) -> None:
        """A mandatory standard (vocal_project_url) is not installed → INDETERMINATE."""
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            project_url=PROJECT_URL,
        )
        result = _invoke([nc], _registry())

        assert result.exit_code == 2
        assert "INDETERMINATE" in result.output

    def test_indeterminate_exits_two_on_too_old(self, tmp_path: Path) -> None:
        """A claimed standard is installed but at a minor too old → INDETERMINATE."""
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.5",
            project_url=PROJECT_URL,
        )
        # Installed minor (3) is older than the claimed minor (5).
        result = _invoke([nc], _registry(project=_project(minor=3)))

        assert result.exit_code == 2
        assert "INDETERMINATE" in result.output


# ---------------------------------------------------------------------------
# Hints: fetch --for on unresolved mandatory; --update on too-old.
# ---------------------------------------------------------------------------


class TestHints:
    def test_fetch_for_hint_on_missing_mandatory(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3", project_url=PROJECT_URL)
        result = _invoke([nc], _registry())

        assert result.exit_code == 2
        assert f"vocal fetch --for {nc}" in result.output

    def test_update_hint_on_too_old(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, conventions="MYSTD-2.5", project_url=PROJECT_URL)
        result = _invoke([nc], _registry(project=_project(minor=3)))

        assert result.exit_code == 2
        assert "--update" in result.output

    def test_no_fetch_for_hint_on_success(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            definitions_url=PACK_URL,
            definitions_version=3,
        )
        result = _invoke([nc], _registry(project=_project(), pack=_pack()))

        assert result.exit_code == 0
        assert "vocal fetch --for" not in result.output


# ---------------------------------------------------------------------------
# Per-axis -p/-d override: each overrides one axis; the un-named axis is still
# resolved from the file.
# ---------------------------------------------------------------------------


class TestPerAxisOverride:
    def test_p_overrides_standards_pack_still_from_file(self, tmp_path: Path) -> None:
        """-p replaces the standards axis; the product axis is still file-resolved."""
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            definitions_url=PACK_URL,
            definitions_version=3,
        )
        # Registry has NO project — but -p supplies one, so the standards axis is
        # satisfied and the file-resolved pack is still checked.
        with patch(
            "vocal.application.check.ConventionsFile.load",
            return_value=_conv(),
        ):
            result = _invoke([nc, "-p", "/some/project"], _registry(pack=_pack()))

        assert result.exit_code == 0
        # The product axis came from the file (the pack schema), not from -d.
        assert "product_foo.json" in result.output

    def test_d_overrides_product_standards_still_from_file(
        self, tmp_path: Path
    ) -> None:
        """-d replaces the product axis; the standards axis is still file-resolved."""
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            # No pack reference on the file at all; -d supplies the product axis.
        )
        result = _invoke(
            [nc, "-d", "/my/override.json"], _registry(project=_project())
        )

        assert result.exit_code == 0
        # The standards axis was resolved from the file (the MYSTD-2 model); the
        # product axis is the overriding schema.
        assert "MYSTD-2" in result.output
        assert "override.json" in result.output

    def test_d_override_makes_product_mandatory(self, tmp_path: Path) -> None:
        """A -d override is mandatory: a failing product schema → FAIL."""
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3")
        with patch("vocal.application.check.print_checks"):
            result = _invoke(
                [nc, "-d", "/my/override.json"],
                _registry(project=_project()),
                pack_pass=False,
            )

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# --specified-only: run only the named axes; ignore the other entirely.
# ---------------------------------------------------------------------------


class TestSpecifiedOnly:
    def test_specified_only_runs_only_named_axis(self, tmp_path: Path) -> None:
        """-d --specified-only checks only the product axis; the file's standards
        claim (with no installed project) is ignored rather than forcing INDETERMINATE."""
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            project_url=PROJECT_URL,  # mandatory, not installed — would be INDETERMINATE
        )
        # Empty registry: were the standards axis resolved, the missing mandatory
        # project would force INDETERMINATE. --specified-only suppresses it.
        result = _invoke(
            [nc, "-d", "/my/override.json", "--specified-only"], _registry()
        )

        assert result.exit_code == 0
        assert "override.json" in result.output
        # The standards axis was not resolved, so no missing-project failure.
        assert "vocal fetch --for" not in result.output

    def test_specified_only_with_no_axis_is_nothing_to_check(
        self, tmp_path: Path
    ) -> None:
        """--specified-only with neither -p nor -d names nothing → nothing to check."""
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3")
        result = _invoke([nc, "--specified-only"], _registry(project=_project()))

        assert result.exit_code == 1
        assert "Nothing to check" in result.output


# ---------------------------------------------------------------------------
# Comments: an opportunistic standard with no installed project is informational
# (not a warning), suppressed unless -c, and rendered with the standards-axis
# results — not folded into the product check's results box.
# ---------------------------------------------------------------------------


class TestStandardComments:
    def test_opportunistic_skip_message_hidden_but_count_reported(
        self, tmp_path: Path
    ) -> None:
        # CF is named in Conventions with no installed project: an opportunistic
        # skip → a comment. Without -c the message is suppressed, but the count
        # line still reports that comments exist and how to surface them.
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3 CF-1.8")
        result = _invoke([nc], _registry(project=_project()))

        assert result.exit_code == 0
        assert "CF-1.8" not in result.output
        assert "1 comments (run with -c)" in result.output

    def test_dash_c_shows_the_comment_with_the_standards_results(
        self, tmp_path: Path
    ) -> None:
        # -c surfaces the skip alongside the standards-axis check output, before
        # the product results box.
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3 CF-1.8",
            definitions_url=PACK_URL,
            definitions_version=3,
        )
        result = _invoke(
            [nc, "-c"], _registry(project=_project(), pack=_pack())
        )

        assert result.exit_code == 0
        assert "CF-1.8" in result.output
        # The standards-axis comment is rendered before the product results box.
        assert result.output.index("CF-1.8") < result.output.index("specification")

    def test_product_box_counts_only_product_comments(self, tmp_path: Path) -> None:
        # The product results box reports the product check's own comment tally; a
        # standards-axis skip does not inflate it.
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3 CF-1.8",
            definitions_url=PACK_URL,
            definitions_version=3,
        )
        result = _invoke(
            [nc], _registry(project=_project(), pack=_pack())
        )

        assert result.exit_code == 0
        assert "0 comments (run with -c)" in result.output


# ---------------------------------------------------------------------------
# No implicit fetch: a plain check never fetches.
# ---------------------------------------------------------------------------


class TestNeverFetchesImplicitly:
    def test_plain_check_does_not_fetch(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3", project_url=PROJECT_URL)
        with patch("vocal.application.check.fetch_for_file") as fetch_mock:
            result = _invoke([nc], _registry())

        assert result.exit_code == 2
        fetch_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers used only by override tests.
# ---------------------------------------------------------------------------


def _conv(name: str = "MYSTD", major: int = 2, minor: int = 3, directory: str = "mystd"):
    from vocal.conventions_file import ConventionsFile

    return ConventionsFile(
        name=name, major=major, minor=minor, project_directory=directory
    )
