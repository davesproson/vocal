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

from vocal.utils.placeholder import Placeholder

from .ir import (
    AttributeDoc,
    ConstraintDoc,
    DatasetDoc,
    DimensionDoc,
    GroupDoc,
    ProductDoc,
    VariableDoc,
)
from .placeholder import parse_value


def _load(spec: dict[str, Any] | str | os.PathLike) -> dict[str, Any]:
    """Return the product spec as a dict, loading from a path if needed."""
    if isinstance(spec, dict):
        return spec
    with open(spec) as f:
        return json.load(f)


def _placeholder_constraints(placeholder: Placeholder) -> list[ConstraintDoc] | None:
    """Translate a placeholder's constraints into ``ConstraintDoc`` form.

    A placeholder's ``regex`` / ``min_len`` / ``max_len`` map onto the same
    ``pattern`` / ``length`` constraint vocabulary the project walk emits, so a
    renderer documents a product's runtime-derived rules with the exact chips it
    already uses for project specs. Returns ``None`` when the placeholder
    declares no constraints (so the IR field stays absent rather than empty).
    """
    c = placeholder.constraints
    constraints: list[ConstraintDoc] = []
    if c.regex is not None:
        constraints.append(ConstraintDoc(kind="pattern", detail={"pattern": c.regex}))
    length_detail = {
        name: value
        for name, value in (("min_length", c.min_len), ("max_length", c.max_len))
        if value is not None
    }
    if length_detail:
        constraints.append(ConstraintDoc(kind="length", detail=length_detail))
    return constraints or None


def _attribute_doc(name: str, raw: Any) -> AttributeDoc:
    """Document a single concrete attribute value as an ``AttributeDoc``.

    A runtime-derived placeholder may declare optionality and constraints; carry
    those onto the doc (``required`` is the inverse of the placeholder's
    ``optional`` flag) so a renderer can surface them. A concrete value declares
    neither, so it keeps the default ``required=True`` and no constraints — the
    renderer gates the required/optional badge on ``derived`` accordingly.
    """
    parsed = parse_value(raw)
    placeholder = parsed.placeholder
    return AttributeDoc(
        name=name,
        value=parsed.value,
        derived=parsed.derived,
        datatype=parsed.datatype,
        required=not placeholder.optional if placeholder else True,
        constraints=_placeholder_constraints(placeholder) if placeholder else None,
    )


def _document_attributes(attributes: dict[str, Any]) -> list[AttributeDoc]:
    """Document every concrete attribute in a product's ``attributes`` map."""
    return [_attribute_doc(name, raw) for name, raw in attributes.items()]


def _datatype(raw: Any) -> Any:
    """Recover a variable's datatype from its ``<dtype>`` notation.

    A product records a variable's datatype as e.g. ``"<float32>"``; strip the
    angle brackets so the IR carries the bare ``"float32"`` (matching the dtype
    the placeholder parser reports for derived attribute values). A non-string
    or already-bare value passes through unchanged.
    """
    if isinstance(raw, str):
        return raw.strip("<>")
    return raw


def _document_variable(raw: dict[str, Any]) -> VariableDoc:
    """Document a single concrete variable from its raw product JSON.

    The variable's ``meta`` carries its ``name``, ``datatype`` and ``required``
    flag (whether a conforming dataset must contain it); ``dimensions`` is the
    list of dimension names it spans; its attributes reuse the concrete
    attribute walk, so they get the same ``AttributeDoc`` shape as the globals.
    """
    meta = raw.get("meta", {})
    return VariableDoc(
        name=meta.get("name"),
        datatype=_datatype(meta.get("datatype")),
        dimensions=raw.get("dimensions"),
        required=meta.get("required"),
        attributes=_document_attributes(raw.get("attributes", {})),
    )


def _document_dimension(raw: dict[str, Any]) -> DimensionDoc:
    """Document a single concrete dimension (its ``name`` and ``size``)."""
    return DimensionDoc(name=raw.get("name"), size=raw.get("size"))


def _document_group(raw: dict[str, Any]) -> GroupDoc:
    """Document a single concrete group, inlining its nested groups.

    A group mirrors the dataset's structure, so it reuses the concrete
    attribute / variable / dimension walks; its child groups are inlined
    recursively (``defs`` is unused in product mode). Its ``meta.name`` is the
    group name.
    """
    meta = raw.get("meta", {})
    return GroupDoc(
        name=meta.get("name"),
        attributes=_document_attributes(raw.get("attributes", {})),
        variables=[_document_variable(v) for v in raw.get("variables", [])],
        dimensions=[_document_dimension(d) for d in raw.get("dimensions", [])],
        groups=[_document_group(g) for g in raw.get("groups") or []],
    )


def document_product(spec: dict[str, Any] | str | os.PathLike) -> ProductDoc:
    """Document a product-pack spec into a :class:`ProductDoc`.

    Accepts the loaded JSON dict or a path to it, walks the raw structure by
    canonical CDM keys, and never imports or validates against the project.
    Documents the global attributes, the ``meta`` section's concrete values
    (file pattern, names, description, references), plus the concrete variables
    and dimensions.
    """
    data = _load(spec)
    doc = DatasetDoc(
        attributes=_document_attributes(data.get("attributes", {})),
        meta=_document_attributes(data.get("meta", {})),
        variables=[_document_variable(v) for v in data.get("variables", [])],
        dimensions=[_document_dimension(d) for d in data.get("dimensions", [])],
        groups=[_document_group(g) for g in data.get("groups") or []],
    )
    return ProductDoc(dataset=doc)
