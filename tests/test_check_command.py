"""Integration tests for the ``vocal check`` CLI command.

``vocal check`` is a thin shell over :mod:`vocal.resolution`: it reads a file's
vocal-managed global attributes, drives the resolver against the local
registry, and renders the result (or the typed error). These tests exercise the
command end-to-end through ``typer.testing.CliRunner`` with the registry and
project import patched, asserting the exit code and — for each of the five
typed resolver errors — that the locked message and hint reach the output.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import netCDF4
import typer
from typer.testing import CliRunner

from vocal.application.check import command
from vocal.manifest import ManifestProduct, build_manifest
from vocal.utils.registry import Pack, Project, Registry


runner = CliRunner()

# A project filecodec defining a single {date} placeholder, used to expand the
# product file_patterns below.
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


def _project(name: str = "MYSTD", major: int = 2, minor: int = 3) -> Project:
    return Project(
        name=name,
        major=major,
        minor=minor,
        project_directory="mystd",
        local_path="/cache/projects/mystd",
    )


def _pack(
    url: str = "https://host/packs",
    version: int = 3,
    name: str = "MYSTD",
    major: int = 2,
    min_minor: int = 3,
    local_path: str = "/cache/packs/host-packs/v3",
    products=None,
) -> Pack:
    if products is None:
        products = [
            ManifestProduct(
                name="foo", file_pattern="foo_{date}", schema="product_foo.json"
            )
        ]
    manifest = build_manifest(
        version=version,
        url=url,
        standard_name=name,
        standard_major=major,
        min_minor=min_minor,
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


def _fake_project_module() -> SimpleNamespace:
    """A stand-in for an imported project package exposing the contract surface."""
    return SimpleNamespace(
        models=SimpleNamespace(Dataset=object()),
        filecodec=FILECODEC,
    )


def _invoke(args, registry: Registry, *, project_module=None):
    """Run the command with the registry and project import patched."""
    if project_module is None:
        project_module = _fake_project_module()
    with (
        patch("vocal.application.check.Registry.load", return_value=registry),
        patch(
            "vocal.application.check.import_project_package",
            return_value=project_module,
        ),
    ):
        return runner.invoke(_app(), args)


# ---------------------------------------------------------------------------
# The five typed resolver errors each surface with the locked message + hint.
# ---------------------------------------------------------------------------


class TestResolverErrorsSurface:
    def test_project_missing(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            project_url="https://host/mystd.git",
        )
        result = _invoke([nc], _registry())

        assert result.exit_code == 1
        assert "No project registered for MYSTD-2" in result.output
        assert "vocal fetch https://host/mystd.git" in result.output
        assert "or pass -p <path>" in result.output

    def test_project_too_old(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.5",
            project_url="https://host/mystd.git",
        )
        result = _invoke([nc], _registry(project=_project(minor=3)))

        assert result.exit_code == 1
        assert "File claims MYSTD-2.5 but registered project is at MYSTD-2.3" in (
            result.output
        )
        assert "Update the registered project" in result.output

    def test_pack_missing(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        result = _invoke([nc], _registry(project=_project()))

        assert result.exit_code == 1
        assert "No pack registered for https://host/packs version 3" in result.output
        assert "vocal fetch https://host/packs" in result.output

    def test_pack_incompatible(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        # Pack targets a different major than the registered project.
        pack = _pack(major=3, min_minor=0)
        result = _invoke([nc], _registry(project=_project(), pack=pack))

        assert result.exit_code == 1
        assert "Pack targets MYSTD-3 but registered project is MYSTD-2" in (
            result.output
        )

    def test_product_not_found(self, tmp_path: Path) -> None:
        # File name does not match the pack's only product pattern (foo_{date}).
        nc = _make_nc(
            tmp_path,
            name="unmatched.nc",
            conventions="MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        result = _invoke([nc], _registry(project=_project(), pack=_pack()))

        assert result.exit_code == 1
        assert "'unmatched.nc' did not match any product pattern" in result.output
        assert "Verify the filename matches one of: foo_{date}" in result.output


# ---------------------------------------------------------------------------
# Graceful-degradation matrix and happy path.
# ---------------------------------------------------------------------------


class TestResolutionFlow:
    def test_full_flow_no_flags(self, tmp_path: Path) -> None:
        """All three attrs present, project + pack registered, product matches."""
        nc = _make_nc(
            tmp_path,
            name="foo_20260522.nc",
            conventions="MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        spec_check = Mock(return_value=True)
        with (
            patch("vocal.application.check.Registry.load", return_value=_registry(
                project=_project(), pack=_pack()
            )),
            patch(
                "vocal.application.check.import_project_package",
                return_value=_fake_project_module(),
            ),
            patch("vocal.application.check.check_against_standard", return_value=True),
            patch("vocal.application.check.check_against_specification", spec_check),
        ):
            result = runner.invoke(_app(), [nc])

        assert result.exit_code == 0
        # The resolver routed the file to the pack's product schema.
        spec_check.assert_called_once_with(
            nc, "/cache/packs/host-packs/v3/product_foo.json"
        )

    def test_conventions_tokenised_picks_vocal_token(self, tmp_path: Path) -> None:
        """A Conventions string carrying CF/ACDD co-conventions still resolves."""
        nc = _make_nc(
            tmp_path,
            name="foo_20260522.nc",
            conventions="CF-1.8 ACDD-1.3 MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        spec_check = Mock(return_value=True)
        with (
            patch(
                "vocal.application.check.Registry.load",
                return_value=_registry(project=_project(), pack=_pack()),
            ),
            patch(
                "vocal.application.check.import_project_package",
                return_value=_fake_project_module(),
            ),
            patch("vocal.application.check.check_against_standard", return_value=True),
            patch("vocal.application.check.check_against_specification", spec_check),
        ):
            result = runner.invoke(_app(), [nc])

        assert result.exit_code == 0
        spec_check.assert_called_once_with(
            nc, "/cache/packs/host-packs/v3/product_foo.json"
        )

    def test_definitions_absent_requires_d(self, tmp_path: Path) -> None:
        """Conventions present but no pack reference and no -d: incomplete."""
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3")
        with patch(
            "vocal.application.check.check_against_standard", return_value=True
        ):
            result = _invoke([nc], _registry(project=_project()))

        assert result.exit_code == 1
        assert "declares no product definitions" in result.output
        assert "Pass -d" in result.output

    def test_d_override_skips_pack_resolution(self, tmp_path: Path) -> None:
        """-d overrides the file's declared pack; no pack lookup is performed."""
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        spec_check = Mock(return_value=True)
        # No pack registered — with -d the resolver must not raise PackMissing.
        with (
            patch(
                "vocal.application.check.Registry.load",
                return_value=_registry(project=_project()),
            ),
            patch(
                "vocal.application.check.import_project_package",
                return_value=_fake_project_module(),
            ),
            patch("vocal.application.check.check_against_standard", return_value=True),
            patch("vocal.application.check.check_against_specification", spec_check),
        ):
            result = runner.invoke(_app(), [nc, "-d", "/my/override.json"])

        assert result.exit_code == 0
        spec_check.assert_called_once_with(nc, "/my/override.json")

    def test_conventions_absent_requires_p_and_d(self, tmp_path: Path) -> None:
        """No Conventions and no -p: the resolver tells the user to pass -p and -d."""
        nc = _make_nc(tmp_path)  # no attributes at all
        result = _invoke([nc], _registry())

        assert result.exit_code == 1
        assert "No Conventions attribute found on the file." in result.output
        assert "Pass -p <path> and -d <path>" in result.output


class TestManualMode:
    def test_p_and_d_bypass_resolver(self, tmp_path: Path) -> None:
        """-p (and -d) bypass the resolver entirely, even with no file attrs."""
        nc = _make_nc(tmp_path)  # no attributes
        spec_check = Mock(return_value=True)
        with (
            patch(
                "vocal.application.check.import_project_package",
                return_value=_fake_project_module(),
            ),
            patch("vocal.application.check.check_against_standard", return_value=True),
            patch("vocal.application.check.check_against_specification", spec_check),
            patch("vocal.application.check.resolve") as resolve_mock,
        ):
            result = runner.invoke(
                _app(), [nc, "-p", "/some/project", "-d", "/some/def.json"]
            )

        assert result.exit_code == 0
        resolve_mock.assert_not_called()
        spec_check.assert_called_once_with(nc, "/some/def.json")

    def test_manual_mode_reports_failure(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        with (
            patch(
                "vocal.application.check.import_project_package",
                return_value=_fake_project_module(),
            ),
            patch("vocal.application.check.check_against_standard", return_value=False),
        ):
            result = runner.invoke(_app(), [nc, "-p", "/some/project"])

        assert result.exit_code == 1


class TestFetchFlag:
    """``vocal check <file> --fetch`` runs the fetch pre-step, then checks."""

    def test_fetch_plus_p_errors(self, tmp_path: Path) -> None:
        """--fetch and -p are opposed modes and cannot be combined."""
        nc = _make_nc(tmp_path, conventions="MYSTD-2.3")
        with (
            patch("vocal.application.check.fetch_for_file") as fetch_mock,
            patch("vocal.application.check.resolve") as resolve_mock,
        ):
            result = runner.invoke(_app(), [nc, "--fetch", "-p", "/some/project"])

        assert result.exit_code == 1
        assert "--fetch cannot be combined with -p" in result.output
        fetch_mock.assert_not_called()
        resolve_mock.assert_not_called()

    def test_fetch_runs_prestep_then_checks(self, tmp_path: Path) -> None:
        """The fetch pre-step runs, then the normal resolved check proceeds."""
        nc = _make_nc(
            tmp_path,
            name="foo_20260522.nc",
            conventions="MYSTD-2.3",
            project_url="https://host/mystd.git",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        spec_check = Mock(return_value=True)
        with (
            patch("vocal.application.check.fetch_for_file") as fetch_mock,
            patch(
                "vocal.application.check.Registry.load",
                return_value=_registry(project=_project(), pack=_pack()),
            ),
            patch(
                "vocal.application.check.import_project_package",
                return_value=_fake_project_module(),
            ),
            patch("vocal.application.check.check_against_standard", return_value=True),
            patch("vocal.application.check.check_against_specification", spec_check),
        ):
            result = runner.invoke(_app(), [nc, "--fetch"])

        assert result.exit_code == 0
        fetch_mock.assert_called_once_with(nc)
        # The resolver still routed the file to the pack's product schema.
        spec_check.assert_called_once_with(
            nc, "/cache/packs/host-packs/v3/product_foo.json"
        )

    def test_fetch_plus_d_allowed_and_overrides_product(self, tmp_path: Path) -> None:
        """--fetch + -d: the pre-step still runs and -d overrides the product."""
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-2.3",
            project_url="https://host/mystd.git",
            definitions_url="https://host/packs",
            definitions_version=3,
        )
        spec_check = Mock(return_value=True)
        # No pack registered: with -d the resolver must not raise PackMissing.
        with (
            patch("vocal.application.check.fetch_for_file") as fetch_mock,
            patch(
                "vocal.application.check.Registry.load",
                return_value=_registry(project=_project()),
            ),
            patch(
                "vocal.application.check.import_project_package",
                return_value=_fake_project_module(),
            ),
            patch("vocal.application.check.check_against_standard", return_value=True),
            patch("vocal.application.check.check_against_specification", spec_check),
        ):
            result = runner.invoke(_app(), [nc, "--fetch", "-d", "/my/override.json"])

        assert result.exit_code == 0
        # The pre-step fetches everything the file declares, independent of -d.
        fetch_mock.assert_called_once_with(nc)
        spec_check.assert_called_once_with(nc, "/my/override.json")

    def test_fetch_missing_project_url_errors_before_check(
        self, tmp_path: Path
    ) -> None:
        """A file with no vocal_project_url surfaces the typed error pre-check."""
        nc = _make_nc(tmp_path)  # no vocal_project_url
        with patch("vocal.application.check.resolve") as resolve_mock:
            result = runner.invoke(_app(), [nc, "--fetch"])

        assert result.exit_code == 1
        assert "nothing to fetch" in result.output
        # The error is raised before the check flow is reached.
        resolve_mock.assert_not_called()
