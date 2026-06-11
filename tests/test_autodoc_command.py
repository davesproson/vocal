"""Tests for the ``vocal autodoc`` command shell.

The IR walk and the HTML rendering have their own tests (``tests/autodoc``); this
exercises the *command seams* through the Typer CLI — the bits that are new with
the subcommand and are where bugs hide: the ``--project``/``--product`` exactly-one
rule, ``--format`` dispatch, the default ``autodoc.<ext>`` output name, the
``--out -`` stdout escape hatch, and that a file is actually written.

The generate step is patched (``import_project`` / ``document_project`` /
``document_product``) so no on-disk project is needed, but the *real* renderer
runs over a *real* tiny IR, so the file/stdout content is genuine output.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

from pydantic import BaseModel
from typer.testing import CliRunner

from vocal.autodoc import document_product, document_project
from vocal.cli.vocal import app as cli_app
from vocal.field import Field
from vocal.mixins import (
    VocalAttributesMixin,
    VocalDatasetMixin,
    VocalDimensionMixin,
    VocalVariableMixin,
)


# --------------------------------------------------------------------------- #
# A tiny hermetic IR, built once and returned by the patched generate step.
# --------------------------------------------------------------------------- #


class _GlobalAttributes(BaseModel, VocalAttributesMixin):
    title: str = Field(description="A brief title", example="My dataset")


class _VariableAttributes(BaseModel, VocalAttributesMixin):
    long_name: str = Field(description="A long name", example="Air temperature")


class _VariableMeta(BaseModel):
    name: str
    datatype: str


class _Variable(BaseModel, VocalVariableMixin):
    meta: _VariableMeta
    dimensions: list[str]
    attributes: _VariableAttributes


class _Dimension(BaseModel, VocalDimensionMixin):
    name: str
    size: Optional[int]


class _DatasetMeta(BaseModel):
    file_pattern: str = Field(description="Pattern", example="thing_{date}.nc")
    short_name: Optional[str] = Field(description="Short name", example="thing", default=None)


class _Dataset(BaseModel, VocalDatasetMixin):
    meta: _DatasetMeta
    attributes: _GlobalAttributes
    dimensions: list[_Dimension]
    variables: list[_Variable]


_PRODUCT = {
    "meta": {"file_pattern": "thing_{date}.nc", "short_name": "thing"},
    "attributes": {"title": "My dataset"},
    "dimensions": [{"name": "time", "size": None}],
    "variables": [
        {
            "meta": {"name": "temperature", "datatype": "<float32>", "required": True},
            "dimensions": ["time"],
            "attributes": {"long_name": "Air temperature"},
        }
    ],
}

_PROJECT_DOC = document_project(_Dataset)
_PRODUCT_DOC = document_product(_PRODUCT)

runner = CliRunner()


def _patch_project():
    """Patch the project generate path so no on-disk project is needed."""
    return (
        patch(
            "vocal.application.autodoc.import_project",
            return_value=SimpleNamespace(Dataset=_Dataset),
        ),
        patch(
            "vocal.application.autodoc.document_project",
            return_value=_PROJECT_DOC,
        ),
    )


def _patch_product():
    return patch(
        "vocal.application.autodoc.document_product",
        return_value=_PRODUCT_DOC,
    )


class TestSourceSelection:
    def test_neither_source_errors(self) -> None:
        result = runner.invoke(cli_app, ["autodoc"])
        assert result.exit_code != 0
        assert "exactly one" in result.output

    def test_both_sources_errors(self) -> None:
        result = runner.invoke(
            cli_app, ["autodoc", "-p", "proj", "--product", "prod.json"]
        )
        assert result.exit_code != 0
        assert "exactly one" in result.output


class TestFormat:
    def test_unknown_format_errors_before_generate(self) -> None:
        # The format is resolved before any IR walk, so this needs no patching;
        # the error must name the bad format and the available ones.
        result = runner.invoke(cli_app, ["autodoc", "-p", "proj", "-f", "pdf"])
        assert result.exit_code != 0
        assert "pdf" in result.output
        assert "html" in result.output


class TestOutput:
    def test_project_writes_html_file(self, tmp_path: Path) -> None:
        out = tmp_path / "doc.html"
        p_import, p_doc = _patch_project()
        with p_import, p_doc:
            result = runner.invoke(
                cli_app, ["autodoc", "-p", "proj", "-o", str(out)]
            )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.read_text().startswith("<!doctype html>")

    def test_default_out_name_derives_from_extension(self) -> None:
        p_import, p_doc = _patch_project()
        with runner.isolated_filesystem(), p_import, p_doc:
            result = runner.invoke(cli_app, ["autodoc", "-p", "proj"])
            assert result.exit_code == 0, result.output
            assert Path("autodoc.html").exists()

    def test_product_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "doc.html"
        with _patch_product():
            result = runner.invoke(
                cli_app, ["autodoc", "--product", "prod.json", "-o", str(out)]
            )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert "product" in out.read_text()

    def test_stdout_hatch_writes_to_stdout_and_no_file(self) -> None:
        p_import, p_doc = _patch_project()
        with runner.isolated_filesystem(), p_import, p_doc:
            result = runner.invoke(cli_app, ["autodoc", "-p", "proj", "--out", "-"])
            assert result.exit_code == 0, result.output
            assert "<!doctype html>" in result.output
            # Nothing should have been written to disk in stdout mode.
            assert not Path("autodoc.html").exists()
