"""Tests for the file-driven fetch confirmation gate.

The security-critical decision lives in the pure :func:`decide_fetch_gate`
policy, tested exhaustively here over the full
{project new / present} × {--yes / not} × {can-prompt / cannot} matrix with no
terminal mocking. The thin :func:`confirm_file_fetch` I/O shell is exercised
end-to-end through both CLI routes (``vocal fetch --for`` and
``vocal check --fetch``) with the registry and ``typer.confirm`` patched,
asserting the externally-observable behaviour: whether a prompt fired, whether a
fetch happened, the exit code, and that the red warning panel reaches stderr.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import netCDF4
import pytest
import typer
from typer.testing import CliRunner

from vocal.application.check import command as check_command
from vocal.application.fetch import command as fetch_command
from vocal.application.fetch_gate import (
    FetchGateDecision,
    decide_fetch_gate,
)
from vocal.utils.registry import Project, Registry


# ---------------------------------------------------------------------------
# Pure policy — decide_fetch_gate
# ---------------------------------------------------------------------------


class TestDecideFetchGate:
    """The full {project new?} × {--yes?} × {can-prompt?} matrix.

    A new project is the only thing that confirms; ``--yes`` consents up front;
    a new project that cannot be confirmed and was not pre-consented is blocked.
    """

    @pytest.mark.parametrize("yes", [False, True])
    @pytest.mark.parametrize("can_prompt", [False, True])
    def test_not_new_always_proceeds(self, yes: bool, can_prompt: bool) -> None:
        # Nothing new to confirm: proceed regardless of --yes or can-prompt.
        assert (
            decide_fetch_gate(project_new=False, yes=yes, can_prompt=can_prompt)
            is FetchGateDecision.PROCEED
        )

    @pytest.mark.parametrize("can_prompt", [False, True])
    def test_new_with_yes_proceeds(self, can_prompt: bool) -> None:
        # Consent given up front: proceed whether or not we could have prompted.
        assert (
            decide_fetch_gate(project_new=True, yes=True, can_prompt=can_prompt)
            is FetchGateDecision.PROCEED
        )

    def test_new_no_yes_can_prompt_prompts(self) -> None:
        assert (
            decide_fetch_gate(project_new=True, yes=False, can_prompt=True)
            is FetchGateDecision.PROMPT
        )

    def test_new_no_yes_cannot_prompt_blocks(self) -> None:
        assert (
            decide_fetch_gate(project_new=True, yes=False, can_prompt=False)
            is FetchGateDecision.BLOCKED
        )


# ---------------------------------------------------------------------------
# CLI integration — the confirm_file_fetch shell on both routes
# ---------------------------------------------------------------------------

runner = CliRunner()
err_runner = CliRunner(mix_stderr=False)

PROJECT_URL = "https://host/mystd.git"
PACK_URL = "https://host/packs"


def _fetch_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(fetch_command)
    return app


def _check_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(check_command)
    return app


def _make_nc(
    tmp_path: Path,
    *,
    project_url: str | None = PROJECT_URL,
    definitions_url: str | None = None,
) -> str:
    path = str(tmp_path / "f.nc")
    with netCDF4.Dataset(path, "w") as nc:
        if project_url is not None:
            nc.vocal_project_url = project_url
        if definitions_url is not None:
            nc.vocal_definitions_url = definitions_url
    return path


def _registry_with(*urls: str) -> Registry:
    """A registry holding a project record for each given source URL."""
    registry = Registry()
    for i, url in enumerate(urls):
        registry.add_project(
            Project(
                name=f"STD{i}",
                major=1,
                minor=0,
                project_directory=f"std{i}",
                local_path=f"/cache/std{i}",
                url=url,
            )
        )
    return registry


class TestFetchForRoute:
    """``vocal fetch --for <file>`` — the gate on the fetch route."""

    def test_new_project_accept_proceeds_and_fetches(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with(),  # empty: project is new
            ),
            patch("vocal.application.fetch_gate.typer.confirm", return_value=True),
            patch("vocal.application.fetch.fetch_for_file", return_value=[]) as fff,
        ):
            result = runner.invoke(_fetch_app(), ["--for", nc])

        assert result.exit_code == 0
        fff.assert_called_once()

    def test_new_project_decline_aborts_nothing_fetched(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with(),
            ),
            patch("vocal.application.fetch_gate.typer.confirm", return_value=False),
            patch("vocal.application.fetch.fetch_for_file") as fff,
        ):
            result = runner.invoke(_fetch_app(), ["--for", nc])

        assert result.exit_code == 1
        assert "Aborted — nothing fetched" in result.output
        fff.assert_not_called()

    def test_already_present_project_does_not_prompt(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        confirm = Mock()
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with(PROJECT_URL),
            ),
            patch("vocal.application.fetch_gate.typer.confirm", confirm),
            patch("vocal.application.fetch.fetch_for_file", return_value=[]) as fff,
        ):
            result = runner.invoke(_fetch_app(), ["--for", nc])

        assert result.exit_code == 0
        confirm.assert_not_called()
        fff.assert_called_once()

    def test_trailing_slash_variant_treated_as_present(self, tmp_path: Path) -> None:
        # The file declares …/mystd.git; the registry holds …/mystd/ — same source.
        nc = _make_nc(tmp_path, project_url="https://host/mystd.git")
        confirm = Mock()
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with("https://host/mystd/"),
            ),
            patch("vocal.application.fetch_gate.typer.confirm", confirm),
            patch("vocal.application.fetch.fetch_for_file", return_value=[]),
        ):
            result = runner.invoke(_fetch_app(), ["--for", nc])

        assert result.exit_code == 0
        confirm.assert_not_called()

    def test_pack_only_new_does_not_prompt(self, tmp_path: Path) -> None:
        # Project already present, but the file declares a (new) pack. The gate is
        # project-centric: a pack is data, so no prompt fires.
        nc = _make_nc(tmp_path, definitions_url=PACK_URL)
        confirm = Mock()
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with(PROJECT_URL),
            ),
            patch("vocal.application.fetch_gate.typer.confirm", confirm),
            patch("vocal.application.fetch.fetch_for_file", return_value=[]) as fff,
        ):
            result = runner.invoke(_fetch_app(), ["--for", nc])

        assert result.exit_code == 0
        confirm.assert_not_called()
        fff.assert_called_once()


class TestCheckFetchRoute:
    """``vocal check <file> --fetch`` — the gate on the check route."""

    def test_decline_aborts_before_any_check(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with(),
            ),
            patch("vocal.application.fetch_gate.typer.confirm", return_value=False),
            patch("vocal.application.check.fetch_for_file") as fff,
            patch("vocal.application.check.resolve") as resolve_mock,
        ):
            result = runner.invoke(_check_app(), [nc, "--fetch"])

        assert result.exit_code == 1
        assert "Aborted — nothing fetched" in result.output
        # The whole command aborts: neither the fetch nor the check runs.
        fff.assert_not_called()
        resolve_mock.assert_not_called()

    def test_accept_proceeds_into_fetch(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with(),
            ),
            patch("vocal.application.fetch_gate.typer.confirm", return_value=True),
            patch("vocal.application.check.fetch_for_file", return_value=[]) as fff,
            # Stop after the fetch pre-step — the resolve flow is out of scope here.
            patch(
                "vocal.application.check.read_file_conventions",
                side_effect=typer.Exit(code=0),
            ),
        ):
            result = runner.invoke(_check_app(), [nc, "--fetch"])

        assert result.exit_code == 0
        fff.assert_called_once()


class TestWarningPanel:
    """The red security-warning panel rendered when a new project is gated."""

    def test_panel_lists_project_and_pack_on_stderr(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, definitions_url=PACK_URL)
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with(),
            ),
            patch("vocal.application.fetch_gate.typer.confirm", return_value=False),
            patch("vocal.application.fetch.fetch_for_file"),
        ):
            result = err_runner.invoke(_fetch_app(), ["--for", nc])

        # Written to stderr, leaving stdout clean for piping.
        assert "Security warning" in result.stderr
        assert "runs on check" in result.stderr
        assert PROJECT_URL in result.stderr
        assert "data" in result.stderr
        assert PACK_URL in result.stderr

    def test_check_route_opening_line(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with(),
            ),
            patch("vocal.application.fetch_gate.typer.confirm", return_value=False),
            patch("vocal.application.check.fetch_for_file"),
        ):
            result = err_runner.invoke(_check_app(), [nc, "--fetch"])

        assert "Checking" in result.stderr
        assert "requires fetching" in result.stderr

    def test_no_color_keeps_box_drops_red(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        with (
            patch(
                "vocal.application.fetch_gate.Registry.load",
                return_value=_registry_with(),
            ),
            patch("vocal.application.fetch_gate.typer.confirm", return_value=False),
            patch("vocal.application.check.fetch_for_file"),
        ):
            result = err_runner.invoke(_check_app(), [nc, "--fetch", "--no-color"])

        # The box still renders (panel border glyphs present)...
        assert "─" in result.stderr
        # ...but no red ANSI escape leaks through.
        assert "\x1b[" not in result.stderr
