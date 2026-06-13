"""Generate and render dataset documentation from a project or product.

``vocal autodoc`` is the generate-and-render front door to :mod:`vocal.autodoc`.
It walks one of two sources into the documentation IR — a project's pydantic
model tree (the *abstract standard*, rule-bearing) or a product-pack JSON (a
*concrete instance*, actual values) — then hands that IR to a registered renderer
(see :mod:`vocal.autodoc.renderers`) for the chosen ``--format``.

The IR walk and the renderers are the substance; this module is the thin shell
around them: it resolves which source to document, selects the format, and writes
the rendered result — to a file (default ``autodoc.<ext>``) or, for ``--out -``,
to stdout so the output can be piped.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional

import typer

from vocal.autodoc import document_product, document_project
from vocal.autodoc.ir import ProductDoc, ProjectDoc
from vocal.autodoc.renderers import get_renderer
from vocal.exceptions import VocalError
from vocal.manifest import MANIFEST_FILENAME, load_manifest
from vocal.utils import Printer, TextStyles, import_project

TS = TextStyles()
p = Printer()


def _product_satisfies_standards(product: str) -> list[str]:
    """Return the standards a product's pack asserts it satisfies.

    ``satisfies_standards`` lives in the pack ``manifest.json``, a sibling of the
    product schema JSON, not in the product JSON itself. When that manifest is
    present and readable it is loaded and its constraints returned as canonical
    strings; when it is absent or malformed the product is still documented (just
    without the advisory standards section), so this never raises.
    """
    manifest_path = Path(product).resolve().parent / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return []
    try:
        manifest = load_manifest(manifest_path)
    except VocalError:
        return []
    return [str(constraint) for constraint in manifest.satisfies_standards]


def command(
    project: Optional[str] = typer.Option(
        None,
        "-p",
        "--project",
        help="Path to a vocal project package to document (the abstract standard).",
    ),
    product: Optional[str] = typer.Option(
        None,
        "--product",
        help="Path to a product-pack JSON to document (a concrete instance).",
    ),
    output_format: str = typer.Option(
        "html",
        "-f",
        "--format",
        help="Output format to render (default: html).",
    ),
    out: Optional[str] = typer.Option(
        None,
        "-o",
        "--out",
        help="Output path. Defaults to autodoc.<ext>; pass '-' to write to stdout.",
    ),
    open_result: bool = typer.Option(
        False,
        "--open",
        help="Open the rendered file in a browser (ignored when writing to stdout).",
    ),
) -> None:
    """Generate and render dataset documentation from a project or product."""
    # Exactly one source: --project and --product are opposed modes (abstract
    # standard vs concrete instance) and the generate step forks on which is given.
    if bool(project) == bool(product):
        p.print_err(
            f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} Provide exactly one of "
            f"--project or --product."
        )
        p.print_err(
            "  --project documents an abstract standard; --product documents a "
            "concrete instance."
        )
        raise typer.Exit(code=1)

    # Resolve the format up front so an unknown --format fails before the (more
    # expensive) IR walk.
    try:
        renderer = get_renderer(output_format)
    except ValueError as e:
        p.print_err(f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} {e}")
        raise typer.Exit(code=1)

    if project:
        # A project's display name is its package directory, not anything the IR
        # carries; the renderer takes it as the title override.
        doc: ProjectDoc | ProductDoc = document_project(import_project(project).Dataset)
        title: Optional[str] = Path(project.rstrip("/")).name
    else:
        assert product is not None  # guaranteed by the exactly-one check above
        doc = document_product(
            product, satisfies_standards=_product_satisfies_standards(product)
        )
        title = None

    text = renderer.render(doc, title)

    if out == "-":
        # Stdout mode: emit only the document so it can be piped. No "wrote …"
        # note and no browser open — there is no file to point at.
        typer.echo(text)
        return

    out_path = Path(out) if out else Path(f"autodoc.{renderer.extension}")
    out_path.write_text(text, encoding="utf-8")
    p.print(f"➜ wrote {out_path} — {TS.OKGREEN}{doc.mode}{TS.ENDC}")
    if open_result:
        webbrowser.open(out_path.resolve().as_uri())
