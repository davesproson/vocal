"""Walk a product-pack JSON into the documentation IR.

The product path documents a *concrete instance*: the actual attribute values a
conforming file contains. It is self-contained — it needs only the product JSON,
never the project — because products are produced through vocal and are known to
satisfy the standard. Slice 1 covers global attributes only: each value is
classified by the placeholder parser as a concrete value or a runtime-derived
placeholder with a recovered datatype.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .ir import AttributeDoc, DatasetDoc, ProductDoc
from .placeholder import parse_value


def _load(spec: dict[str, Any] | str | os.PathLike) -> dict[str, Any]:
    """Return the product spec as a dict, loading from a path if needed."""
    if isinstance(spec, dict):
        return spec
    with open(spec) as f:
        return json.load(f)


def _attribute_doc(name: str, raw: Any) -> AttributeDoc:
    """Document a single concrete attribute value as an ``AttributeDoc``."""
    parsed = parse_value(raw)
    return AttributeDoc(
        name=name,
        value=parsed.value,
        derived=parsed.derived,
        datatype=parsed.datatype,
    )


def _document_attributes(attributes: dict[str, Any]) -> list[AttributeDoc]:
    """Document every concrete attribute in a product's ``attributes`` map."""
    return [_attribute_doc(name, raw) for name, raw in attributes.items()]


def document_product(spec: dict[str, Any] | str | os.PathLike) -> ProductDoc:
    """Document a product-pack spec into a :class:`ProductDoc`.

    Accepts the loaded JSON dict or a path to it, walks the raw structure by
    canonical CDM keys, and never imports or validates against the project.
    Slice 1 documents the global attributes only.
    """
    data = _load(spec)
    attributes = data.get("attributes", {})
    doc = DatasetDoc(attributes=_document_attributes(attributes))
    return ProductDoc(dataset=doc)
