"""Tests for the `vocal check` CLI command exit-code behavior."""

from unittest.mock import patch

import typer
from typer.testing import CliRunner

from vocal.application.check import DefinitionVersionNotFound, command


runner = CliRunner()


def _app() -> typer.Typer:
    app = typer.Typer()
    app.command()(command)
    return app


class TestCheckCommandIncomplete:
    def test_exit_1_when_definitions_missing_even_if_project_check_passes(
        self,
    ) -> None:
        with (
            patch(
                "vocal.application.check.load_matching_projects",
                return_value=["/tmp/project"],
            ),
            patch(
                "vocal.application.check.load_matching_definitions",
                side_effect=DefinitionVersionNotFound(
                    "No definitions for version 2.1",
                    hint="register a project for 2.1",
                ),
            ),
            patch(
                "vocal.application.check.run_checks", return_value=True
            ),
        ):
            result = runner.invoke(_app(), ["dummy.nc"])

        assert result.exit_code == 1
        # The warning message and hint should both appear in output.
        assert "No definitions for version 2.1" in result.output
        assert "register a project for 2.1" in result.output

    def test_exit_0_when_all_checks_pass(self) -> None:
        with (
            patch(
                "vocal.application.check.load_matching_projects",
                return_value=["/tmp/project"],
            ),
            patch(
                "vocal.application.check.load_matching_definitions",
                return_value=[],
            ),
            patch(
                "vocal.application.check.run_checks", return_value=True
            ),
        ):
            result = runner.invoke(_app(), ["dummy.nc"])

        assert result.exit_code == 0

    def test_exit_1_when_checks_fail(self) -> None:
        with (
            patch(
                "vocal.application.check.load_matching_projects",
                return_value=["/tmp/project"],
            ),
            patch(
                "vocal.application.check.load_matching_definitions",
                return_value=[],
            ),
            patch(
                "vocal.application.check.run_checks", return_value=False
            ),
        ):
            result = runner.invoke(_app(), ["dummy.nc"])

        assert result.exit_code == 1
