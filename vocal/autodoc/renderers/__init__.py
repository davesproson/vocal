"""Renderers: standalone consumers that turn an autodoc IR root into a document.

The IR (see :mod:`vocal.autodoc.ir`) is autodoc's core deliverable; each renderer
here is a pure ``(IR root, title) -> str`` function that depends only on the IR,
never the other way round. Formats are registered in :data:`RENDERERS`, keyed by
the name exposed as ``vocal autodoc --format``. Adding a format is a one-liner:
write a ``render`` function in a new module and add a :class:`Renderer` row.

The renderers are deliberately I/O-free — selecting a format, choosing an output
path and writing the bytes all live in the ``vocal autodoc`` command, so the
render step stays a trivially testable string transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..ir import ProductDoc, ProjectDoc
from . import html


@dataclass(frozen=True)
class Renderer:
    """A registered output format: its render function and default extension.

    ``extension`` (no dot) drives both the default ``--out`` filename
    (``autodoc.<extension>``) and how the command labels the format; it lives
    with the renderer so the command needn't know any per-format specifics.
    """

    render: Callable[[ProjectDoc | ProductDoc, str | None], str]
    extension: str


RENDERERS: dict[str, Renderer] = {
    "html": Renderer(render=html.render, extension="html"),
}


def get_renderer(fmt: str) -> Renderer:
    """Look up a registered renderer by format name.

    Raises ``ValueError`` (with the list of available formats) for an unknown
    name, so the command can surface a clean ``--format`` error.
    """
    try:
        return RENDERERS[fmt]
    except KeyError:
        available = ", ".join(sorted(RENDERERS))
        raise ValueError(
            f"unknown format {fmt!r}; available formats: {available}"
        ) from None


__all__ = ["Renderer", "RENDERERS", "get_renderer"]
