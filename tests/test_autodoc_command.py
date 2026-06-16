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

    def test_pack_with_project_errors(self) -> None:
        # --pack is the third mutually-exclusive source.
        result = runner.invoke(
            cli_app, ["autodoc", "-p", "proj", "--pack", "packdir"]
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

    def test_product_renders_satisfies_standards_from_sibling_manifest(
        self, tmp_path: Path
    ) -> None:
        # A real product JSON beside a real manifest.json: the command reads the
        # manifest's satisfies_standards (not the product JSON) and renders them.
        import json

        (tmp_path / "alpha.json").write_text(json.dumps(_PRODUCT))
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "version": 1,
                    "url": "https://host/packs",
                    "filecodec": {"date": {"regex": r"\d{8}"}},
                    "satisfies_standards": [
                        {"name": "MYSTD", "major": 2, "min_minor": 3}
                    ],
                    "products": [
                        {
                            "name": "alpha",
                            "file_pattern": "thing_{date}.nc",
                            "schema": "alpha.json",
                        }
                    ],
                }
            )
        )
        out = tmp_path / "doc.html"
        result = runner.invoke(
            cli_app,
            ["autodoc", "--product", str(tmp_path / "alpha.json"), "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        rendered = out.read_text()
        assert "Satisfies standards" in rendered
        assert "MYSTD-2.3+" in rendered

    def test_stdout_hatch_writes_to_stdout_and_no_file(self) -> None:
        p_import, p_doc = _patch_project()
        with runner.isolated_filesystem(), p_import, p_doc:
            result = runner.invoke(cli_app, ["autodoc", "-p", "proj", "--out", "-"])
            assert result.exit_code == 0, result.output
            assert "<!doctype html>" in result.output
            # Nothing should have been written to disk in stdout mode.
            assert not Path("autodoc.html").exists()


def _write_pack(directory: Path, names: list[str]) -> None:
    """Write a real pack (manifest.json + a product schema per name) into a dir.

    Pack mode runs the *real* ``document_product`` over these on-disk schemas, so
    the generated pages are genuine output.
    """
    import json

    for name in names:
        (directory / f"{name}.json").write_text(json.dumps(_PRODUCT))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": 1,
                "url": "https://host/packs/demo",
                "filecodec": {"date": {"regex": r"\d{8}"}},
                "satisfies_standards": [],
                "products": [
                    {
                        "name": name,
                        "file_pattern": "thing_{date}.nc",
                        "schema": f"{name}.json",
                    }
                    for name in names
                ],
            }
        )
    )


class TestPack:
    def test_pack_writes_index_and_a_page_per_product(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        _write_pack(pack_dir, ["alpha", "beta"])
        out_dir = tmp_path / "site"

        result = runner.invoke(
            cli_app, ["autodoc", "--pack", str(pack_dir), "-o", str(out_dir)]
        )

        assert result.exit_code == 0, result.output
        assert (out_dir / "index.html").exists()
        assert (out_dir / "alpha.html").exists()
        assert (out_dir / "beta.html").exists()
        # The product pages are genuine product output.
        assert "product" in (out_dir / "alpha.html").read_text()

    def test_index_links_resolve_to_written_pages(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        _write_pack(pack_dir, ["alpha", "beta"])
        out_dir = tmp_path / "site"

        runner.invoke(
            cli_app, ["autodoc", "--pack", str(pack_dir), "-o", str(out_dir)]
        )

        index = (out_dir / "index.html").read_text()
        for href in ("alpha.html", "beta.html"):
            assert f'href="{href}"' in index
            assert (out_dir / href).exists()

    def test_index_heading_is_the_pack_url_tail(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        _write_pack(pack_dir, ["alpha"])
        out_dir = tmp_path / "site"

        runner.invoke(
            cli_app, ["autodoc", "--pack", str(pack_dir), "-o", str(out_dir)]
        )

        # The pack url is .../packs/demo, so its last segment heads the index.
        assert "<h1>demo</h1>" in (out_dir / "index.html").read_text()

    def test_index_shows_each_products_file_pattern(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        _write_pack(pack_dir, ["alpha"])
        out_dir = tmp_path / "site"

        runner.invoke(
            cli_app, ["autodoc", "--pack", str(pack_dir), "-o", str(out_dir)]
        )

        assert "thing_{date}.nc" in (out_dir / "index.html").read_text()

    def test_product_page_titled_with_manifest_name(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        _write_pack(pack_dir, ["alpha"])
        out_dir = tmp_path / "site"

        runner.invoke(
            cli_app, ["autodoc", "--pack", str(pack_dir), "-o", str(out_dir)]
        )

        # The manifest product name overrides the page heading, so it matches the
        # index link text.
        assert "<h1>alpha</h1>" in (out_dir / "alpha.html").read_text()

    def test_default_out_dir_is_autodoc(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        _write_pack(pack_dir, ["alpha"])

        with runner.isolated_filesystem():
            result = runner.invoke(cli_app, ["autodoc", "--pack", str(pack_dir)])
            assert result.exit_code == 0, result.output
            assert Path("autodoc/index.html").exists()
            assert Path("autodoc/alpha.html").exists()

    def test_missing_product_schema_aborts_naming_it_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        # A manifest references a schema file that does not exist. The whole site
        # is rendered in memory first, so this aborts before any write: the error
        # names the offending product and the output directory stays untouched.
        import json

        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        (pack_dir / "alpha.json").write_text(json.dumps(_PRODUCT))
        (pack_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "version": 1,
                    "url": "https://host/packs/demo",
                    "filecodec": {"date": {"regex": r"\d{8}"}},
                    "satisfies_standards": [],
                    "products": [
                        {
                            "name": "alpha",
                            "file_pattern": "a_{date}.nc",
                            "schema": "alpha.json",
                        },
                        {
                            "name": "ghost",
                            "file_pattern": "g_{date}.nc",
                            "schema": "missing.json",
                        },
                    ],
                }
            )
        )
        out_dir = tmp_path / "site"

        result = runner.invoke(
            cli_app, ["autodoc", "--pack", str(pack_dir), "-o", str(out_dir)]
        )

        assert result.exit_code != 0
        assert "ghost" in result.output
        # Nothing is written when any product is broken — not even the good page.
        assert not out_dir.exists()

    def test_unparseable_product_schema_aborts_naming_it(self, tmp_path: Path) -> None:
        import json

        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        (pack_dir / "broken.json").write_text("{ this is not json")
        (pack_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "version": 1,
                    "url": "https://host/packs/demo",
                    "filecodec": {"date": {"regex": r"\d{8}"}},
                    "satisfies_standards": [],
                    "products": [
                        {
                            "name": "wonky",
                            "file_pattern": "w_{date}.nc",
                            "schema": "broken.json",
                        }
                    ],
                }
            )
        )
        out_dir = tmp_path / "site"

        result = runner.invoke(
            cli_app, ["autodoc", "--pack", str(pack_dir), "-o", str(out_dir)]
        )

        assert result.exit_code != 0
        assert "wonky" in result.output
        assert not out_dir.exists()

    def test_stdout_out_is_rejected_in_pack_mode(self, tmp_path: Path) -> None:
        # A pack writes many files into a directory; stdout would concatenate
        # them into broken output, so '-' is refused with a clear error.
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        _write_pack(pack_dir, ["alpha"])

        result = runner.invoke(
            cli_app, ["autodoc", "--pack", str(pack_dir), "--out", "-"]
        )

        assert result.exit_code != 0
        assert "-" in result.output
        assert "--pack" in result.output

    def test_rerun_overwrites_ours_and_leaves_others(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        _write_pack(pack_dir, ["alpha"])
        out_dir = tmp_path / "site"
        out_dir.mkdir()
        keepsake = out_dir / "keep.txt"
        keepsake.write_text("untouched")

        for _ in range(2):
            result = runner.invoke(
                cli_app, ["autodoc", "--pack", str(pack_dir), "-o", str(out_dir)]
            )
            assert result.exit_code == 0, result.output

        assert (out_dir / "index.html").exists()
        # An unrelated file in the output directory is left alone.
        assert keepsake.read_text() == "untouched"
