"""The shared documentation intermediate representation (IR).

One node vocabulary documents both a *project* (the abstract standard: types,
constraints and validator rules) and a *product* (a concrete instance: actual
values, datatypes and derived-at-runtime markers). A node therefore carries two
kinds of optional field:

* **rule-bearing** fields (``description``, ``example``, ``constraints``,
  ``rules``) filled by the project walk, and
* **concrete** fields (``value``, ``derived``, ``datatype``) filled by the
  product walk.

A renderer tells the two modes apart by which fields are present. The IR is
plain pydantic, so it serialises with ``model_dump_json`` and round-trips with
``model_validate_json``.

This module is introduced by autodoc slice 1, which exercises the contract for
global attributes only. ``ConstraintDoc`` and ``RuleDoc`` are intentionally
minimal here; later slices flesh out their fields without changing the node
shape that depends on them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConstraintDoc(BaseModel):
    """A normalised, typed constraint on a field (project mode).

    Populated by the constraint normaliser in a later slice; defined here so the
    ``AttributeDoc.constraints`` field has a stable type.
    """

    kind: str
    detail: dict[str, Any] | None = None


class RuleDoc(BaseModel):
    """A custom ``vocal`` validator rule (project mode).

    Populated by the rule extractor in a later slice; defined here so the
    ``*.rules`` fields have a stable type. ``members`` enumerates a controlled
    vocabulary's allowed values when it can expose them.
    """

    description: str
    members: list[str] | None = None


class AttributeDoc(BaseModel):
    """A single attribute, in either mode.

    Project mode fills ``description`` / ``example`` / ``required`` (and, in
    later slices, ``constraints`` / ``rules``). Product mode fills ``value`` /
    ``derived`` / ``datatype``.
    """

    name: str

    # Rule-bearing fields (project mode).
    description: str | None = None
    example: Any | None = None
    required: bool = True
    constraints: list[ConstraintDoc] | None = None
    rules: list[RuleDoc] | None = None

    # Concrete fields (product mode).
    value: Any | None = None
    derived: bool | None = None
    datatype: str | None = None


class DimensionDoc(BaseModel):
    """A dimension, in either mode.

    Product mode fills the concrete ``name`` and ``size`` (``size`` is ``None``
    for an unlimited dimension). Project mode emits a single template carrying
    only the dimension model's structural ``rules`` (``name`` / ``size`` stay
    ``None`` because the standard does not fix concrete dimensions).
    """

    name: str | None = None
    size: int | None = None

    # Rule-bearing field (project mode): structural rules on the dimension model.
    rules: list[RuleDoc] | None = None


class VariableDoc(BaseModel):
    """A variable, in either mode.

    Project mode emits a single *template* describing what every variable must
    look like: its ``attributes`` are the rule-bearing attribute specs (reusing
    the attribute walk) and ``rules`` are the variable model's structural rules;
    ``name`` / ``datatype`` / ``dimensions`` stay ``None`` because the standard
    does not enumerate concrete variables. Product mode fills the concrete
    ``name`` / ``datatype`` / ``dimensions`` and the attributes' concrete values.
    """

    name: str | None = None

    # Concrete fields (product mode).
    datatype: str | None = None
    dimensions: list[str] | None = None

    # Shared: rule-bearing attribute specs (project) or concrete values (product).
    attributes: list[AttributeDoc] = Field(default_factory=list)

    # Rule-bearing field (project mode): structural rules on the variable model.
    rules: list[RuleDoc] | None = None


class DatasetDoc(BaseModel):
    """The root container node. Holds the global attributes (slice 1),
    plus the variables and dimensions (slice 5).

    ``rules`` carries the container-level (model-bound) validator rules declared
    on the ``Dataset`` model itself — the structural requirements
    (``variable_exists`` / ``dimension_exists`` / … and bespoke model
    validators) that apply to the dataset as a whole rather than to a single
    attribute. ``variables`` holds exactly one template ``VariableDoc`` in
    project mode and the N concrete variables in product mode; ``dimensions``
    likewise. Later slices add ``groups`` / ``meta``.
    """

    attributes: list[AttributeDoc] = Field(default_factory=list)
    rules: list[RuleDoc] | None = None
    variables: list[VariableDoc] = Field(default_factory=list)
    dimensions: list[DimensionDoc] = Field(default_factory=list)


class ProjectDoc(BaseModel):
    """The IR root produced by :func:`document_project`.

    Carries the documented ``dataset`` plus root-level concerns: ``diagnostics``
    (documentation gaps surfaced during the walk) and the project-only ``defs``
    registry used to represent recursive groups without infinite expansion.
    """

    mode: Literal["project"] = "project"
    dataset: DatasetDoc
    diagnostics: list[str] = Field(default_factory=list)
    # Project-only registry of reusable node templates (the recursive ``Group``
    # template lands here from a later slice). Typed loosely until ``GroupDoc``
    # exists; the key is the referenced type name.
    defs: dict[str, Any] = Field(default_factory=dict)


class ProductDoc(BaseModel):
    """The IR root produced by :func:`document_product`.

    Carries the documented ``dataset`` and ``diagnostics``. ``defs`` is unused
    in product mode (groups are inlined), so it is absent here.
    """

    mode: Literal["product"] = "product"
    dataset: DatasetDoc
    diagnostics: list[str] = Field(default_factory=list)
