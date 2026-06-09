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
from .diagnostics import mixin_mismatch, undescribed_validator
from .ir import (
    AttributeDoc,
    ConstraintDoc,
    DatasetDoc,
    DimensionDoc,
    GroupDoc,
    NodeRef,
    ProductDoc,
    ProjectDoc,
    RuleDoc,
    VariableDoc,
)
from .placeholder import ParsedValue, parse_value
from .product import document_product
from .project import document_project
from .rules import attribute_rules, model_rules, rule_doc

__all__ = [
    "AttributeDoc",
    "ConstraintDoc",
    "DatasetDoc",
    "DimensionDoc",
    "GroupDoc",
    "NodeRef",
    "ProductDoc",
    "ProjectDoc",
    "RuleDoc",
    "VariableDoc",
    "ParsedValue",
    "parse_value",
    "normalize_constraints",
    "mixin_mismatch",
    "undescribed_validator",
    "document_product",
    "document_project",
    "attribute_rules",
    "model_rules",
    "rule_doc",
]
