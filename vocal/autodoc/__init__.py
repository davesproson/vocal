"""autodoc: parse projects and products into a shared documentation IR.

Two entry points produce the same node vocabulary in two modes:

* :func:`document_project` walks a project's pydantic model tree into the
  *abstract standard* (rule-bearing), and
* :func:`document_product` walks a product-pack JSON into a *concrete instance*
  (actual values).

Both return a serialisable IR (see :mod:`vocal.autodoc.ir`) — the package's core
deliverable. Renderers that turn that IR into a concrete output format live in
the standalone :mod:`vocal.autodoc.renderers` subpackage; format selection and
file I/O live in the ``vocal autodoc`` command. No CLI lives here.
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
