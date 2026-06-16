"""Assemble a pack manifest into the documentation IR's pack-index nodes.

A *pack* (a ``manifest.json`` indexing many products) documents as a small static
site: one index page routing to one product page per product. This module owns
the two pure, I/O-free steps behind that site:

* :func:`unique_slugs` — the single source of truth for product-page filenames.
  It maps each product ``name`` to a safe ``[a-z0-9-]`` stem; the command turns
  those stems into both the on-disk filenames *and* the index's link targets, so
  links and files cannot drift.
* :func:`build_pack_doc` — assembles a loaded :class:`~vocal.manifest.Manifest`
  plus the computed hrefs into a :class:`~vocal.autodoc.ir.PackDoc`.

Rendering the :class:`PackDoc` to an index page lives in the renderers; selecting
a format and writing files lives in the ``vocal autodoc`` command.
"""

from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urlsplit

from vocal.manifest import Manifest

from .ir import PackDoc, PackEntry

# Anything outside the safe filename alphabet collapses to a single hyphen; the
# slug is then trimmed of leading/trailing hyphens so a name made entirely of
# unsafe characters cannot produce a stem that is just punctuation.
_UNSAFE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    """Reduce a product name to a safe ``[a-z0-9-]`` filename stem."""
    slug = _UNSAFE.sub("-", name.lower()).strip("-")
    return slug or "product"


def unique_slugs(names: Iterable[str]) -> list[str]:
    """Map product names to distinct, safe ``[a-z0-9-]`` slugs, in input order.

    Each name is slugified to the safe alphabet; a stem that has already been
    taken is disambiguated with a numeric suffix (``-2``, ``-3``, …) so every
    product gets its own page filename.
    """
    used: set[str] = set()
    out: list[str] = []
    for name in names:
        base = _slugify(name)
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}-{suffix}"
        used.add(candidate)
        out.append(candidate)
    return out


def pack_heading(url: str, fallback: str) -> str:
    """The pack's display heading: the last path segment of its ``url``.

    The pack ``url`` is the pack's stable identity, so its trailing segment names
    the pack (e.g. ``…/packs/demo`` → ``demo``). When the url carries no path —
    a bare host — there is nothing to name the pack with, so the caller's
    ``fallback`` (the ``--pack`` directory name) is used instead. Pure, so the
    command can resolve the heading without the renderer parsing urls.
    """
    segment = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
    return segment or fallback


def build_pack_doc(manifest: Manifest, hrefs: list[str]) -> PackDoc:
    """Assemble a loaded manifest and the computed hrefs into a ``PackDoc``.

    ``hrefs`` is aligned positionally with ``manifest.products`` — each product's
    page filename, which doubles as the index link target. The pack-wide
    ``satisfies_standards`` is rendered once as canonical constraint strings.
    """
    entries = [
        PackEntry(name=product.name, href=href, file_pattern=product.file_pattern)
        for product, href in zip(manifest.products, hrefs)
    ]
    return PackDoc(
        url=manifest.url,
        version=manifest.version,
        satisfies_standards=[str(c) for c in manifest.satisfies_standards],
        products=entries,
    )
