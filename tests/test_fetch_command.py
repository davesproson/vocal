"""Integration tests for the ``vocal fetch`` CLI command.

``vocal fetch`` fetches a single resource by URL (auto-detecting the kind) or,
with ``--for <file>``, fetches whatever a netCDF file declares about itself.
These tests exercise the command end-to-end through ``typer.testing.CliRunner``
with the underlying fetch primitives patched, asserting exit codes and that the
locked validation messages and per-resource summary reach the output.
"""

from pathlib import Path
from unittest.mock import patch

import netCDF4
import typer
from typer.testing import CliRunner

from vocal.application.fetch import FetchOutcome, command


runner = CliRunner()


def _app() -> typer.Typer:
    app = typer.Typer()
    app.command()(command)
    return app


def _make_nc(tmp_path: Path, *, project_url: str | None = None) -> str:
    path = str(tmp_path / "f.nc")
    with netCDF4.Dataset(path, "w") as nc:
        if project_url is not None:
            nc.vocal_project_url = project_url
    return path


class TestExactlyOneOf:
    def test_neither_url_nor_for_errors(self) -> None:
        result = runner.invoke(_app(), [])
        assert result.exit_code == 1
        assert "exactly one" in result.output

    def test_both_url_and_for_errors(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, project_url="https://host/std.git")
        result = runner.invoke(_app(), ["https://host/repo", "--for", nc])
        assert result.exit_code == 1
        assert "exactly one" in result.output

    def test_url_alone_drives_plain_fetch(self) -> None:
        with patch("vocal.application.fetch.fetch") as fetch_mock:
            result = runner.invoke(_app(), ["https://host/repo"])
        assert result.exit_code == 0
        fetch_mock.assert_called_once_with(
            "https://host/repo", git=False, update=False, force=False
        )


class TestFetchForFile:
    def test_missing_project_url_renders_typed_error(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)  # no vocal_project_url
        result = runner.invoke(_app(), ["--for", nc])
        assert result.exit_code == 1
        assert "declares no vocal_project_url" in result.output
        assert "not self-describing" in result.output

    def test_per_resource_summary_reaches_output(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, project_url="https://host/std.git")
        outcomes = [
            FetchOutcome("project", "https://host/std.git", "fetched"),
            FetchOutcome("pack", None, "none-declared"),
        ]
        with patch(
            "vocal.application.fetch.fetch_for_file", return_value=outcomes
        ) as fff:
            result = runner.invoke(_app(), ["--for", nc])

        assert result.exit_code == 0
        fff.assert_called_once_with(nc, git=False, update=False, force=False)
        assert "project: https://host/std.git" in result.output
        assert "fetched" in result.output
        assert "no pack to fetch" in result.output

    def test_already_present_rendered_distinctly_from_fetched(
        self, tmp_path: Path
    ) -> None:
        nc = _make_nc(tmp_path, project_url="https://host/std.git")
        outcomes = [
            FetchOutcome("project", "https://host/std.git", "already-present"),
            FetchOutcome("pack", "https://host/pack.git", "fetched"),
        ]
        with patch("vocal.application.fetch.fetch_for_file", return_value=outcomes):
            result = runner.invoke(_app(), ["--for", nc])

        assert result.exit_code == 0
        assert "already present" in result.output
        assert "fetched" in result.output

    def test_flags_forwarded_to_fetch_for_file(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, project_url="https://host/std.git")
        with patch(
            "vocal.application.fetch.fetch_for_file", return_value=[]
        ) as fff:
            result = runner.invoke(
                _app(), ["--for", nc, "--git", "--update", "--force"]
            )

        assert result.exit_code == 0
        fff.assert_called_once_with(nc, git=True, update=True, force=True)
