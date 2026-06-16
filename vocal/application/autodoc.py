"""Generate and render dataset documentation from a project or product.

``vocal autodoc`` is the generate-and-render front door to :mod:`vocal.autodoc`.
It walks one of three sources into the documentation IR — a project's pydantic
model tree (``--project``, the *abstract standard*, rule-bearing), a product-pack
JSON (``--product``, a *concrete instance*, actual values), or a whole pack
directory (``--pack``) — then hands that IR to a registered renderer (see
:mod:`vocal.autodoc.renderers`) for the chosen ``--format``.

The IR walk and the renderers are the substance; this module is the thin shell
around them: it resolves which source to document, selects the format, and writes
the rendered result. ``--project`` / ``--product`` write a single file (default
``autodoc.<ext>``) or, for ``--out -``, to stdout so the output can be piped;
``--pack`` writes a small doc site — an index page plus one page per product —
into an output directory (default ``./autodoc/``).
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional

import typer

from vocal.autodoc import document_product, document_project
from vocal.autodoc.ir import ProductDoc, ProjectDoc
from vocal.autodoc.pack import build_pack_doc, unique_slugs
from vocal.autodoc.renderers import Renderer, get_renderer
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
    pack: Optional[str] = typer.Option(
        None,
        "--pack",
        help=(
            "Path to a pack directory (containing manifest.json) to document as "
            "a doc site: an index page plus one page per product."
        ),
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
    """Generate and render dataset documentation from a project, product or pack."""
    # Exactly one source. --project (abstract standard) and --product (concrete
    # instance) are single-file modes the generate step forks on; --pack documents
    # a whole pack directory as a doc site. The three are mutually exclusive.
    if sum(bool(source) for source in (project, product, pack)) != 1:
        p.print_err(
            f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} Provide exactly one of "
            f"--project, --product or --pack."
        )
        p.print_err(
            "  --project documents an abstract standard; --product a concrete "
            "instance; --pack a whole pack directory."
        )
        raise typer.Exit(code=1)

    # Resolve the format up front so an unknown --format fails before the (more
    # expensive) IR walk.
    try:
        renderer = get_renderer(output_format)
    except ValueError as e:
        p.print_err(f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} {e}")
        raise typer.Exit(code=1)

    if pack:
        _render_pack(pack, renderer, out, open_result)
        return

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


def _render_pack(
    pack: str, renderer: Renderer, out: Optional[str], open_result: bool
) -> None:
    """Render a whole pack into a doc site: an index page plus one page per product.

    The pack directory's ``manifest.json`` is loaded with the shared
    :func:`load_manifest` (so the same path handed to ``vocal register`` works
    here), each product schema resolved relative to it. Product names are
    slugified once into the page filenames that double as the index's link
    targets, so links and files cannot drift. Each product page is produced
    exactly as ``--product`` would (manifest ``name`` as the title, the pack-wide
    ``satisfies_standards``). Output lands in a directory (default ``./autodoc/``):
    created if missing, our generated files overwritten, any other files left
    untouched.
    """
    pack_dir = Path(pack)
    manifest = load_manifest(pack_dir / MANIFEST_FILENAME)

    # One slug per product, turned into page filenames; the same hrefs feed both
    # the index links and the on-disk product-page names so they cannot drift.
    hrefs = [
        f"{slug}.{renderer.extension}"
        for slug in unique_slugs(product.name for product in manifest.products)
    ]
    standards = [str(constraint) for constraint in manifest.satisfies_standards]

    pages = {
        href: renderer.render(
            document_product(
                str(pack_dir / product.schema), satisfies_standards=standards
            ),
            product.name,
        )
        for product, href in zip(manifest.products, hrefs)
    }
    index_name = f"index.{renderer.extension}"
    pages[index_name] = renderer.render_index(build_pack_doc(manifest, hrefs))

    out_dir = Path(out) if out else Path("autodoc")
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in pages.items():
        (out_dir / filename).write_text(text, encoding="utf-8")

    p.print(
        f"➜ wrote {out_dir}/ — {TS.OKGREEN}pack{TS.ENDC} "
        f"({len(manifest.products)} product"
        f"{'' if len(manifest.products) == 1 else 's'})"
    )
    if open_result:
        webbrowser.open((out_dir / index_name).resolve().as_uri())
