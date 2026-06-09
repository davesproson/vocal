"""autodoc: parse projects and products into a shared documentation IR.

Two entry points produce the same node vocabulary in two modes:

* :func:`document_project` walks a project's pydantic model tree into the
  *abstract standard* (rule-bearing), and
* :func:`document_product` walks a product-pack JSON into a *concrete instance*
  (actual values).

Both return a serialisable IR (see :mod:`vocal.autodoc.ir`). No renderer and no
CLI live here — the IR is the deliverable.
"""

from .constraints import normalize_constraints
from .ir import (
    AttributeDoc,
    ConstraintDoc,
    DatasetDoc,
    ProductDoc,
    ProjectDoc,
    RuleDoc,
)
from .placeholder import ParsedValue, parse_value
from .product import document_product
from .project import document_project

__all__ = [
    "AttributeDoc",
    "ConstraintDoc",
    "DatasetDoc",
    "ProductDoc",
    "ProjectDoc",
    "RuleDoc",
    "ParsedValue",
    "parse_value",
    "normalize_constraints",
    "document_product",
    "document_project",
]
