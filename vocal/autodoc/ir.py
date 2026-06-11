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

from typing import Annotated, Any, Literal, Union

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
    ``None`` because the standard does not fix concrete dimensions). In project
    mode the template is registered once in :attr:`ProjectDoc.defs` and the
    container's ``dimensions`` slot holds a ``NodeRef`` to it; the ``kind`` tag
    discriminates the ``DimensionDoc | NodeRef`` slot union.
    """

    kind: Literal["dimension"] = "dimension"
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
    does not enumerate concrete variables. The template is registered once in
    :attr:`ProjectDoc.defs` and every container's ``variables`` slot holds a
    ``NodeRef`` to it (so the one shared ``Variable`` model is documented once,
    not re-expanded under each group); the ``kind`` tag discriminates the
    ``VariableDoc | NodeRef`` slot union. Product mode fills the concrete
    ``name`` / ``datatype`` / ``dimensions`` / ``required`` and the attributes'
    concrete values.
    """

    kind: Literal["variable"] = "variable"
    name: str | None = None

    # Concrete fields (product mode).
    datatype: str | None = None
    dimensions: list[str] | None = None
    # Whether a conforming dataset must contain this variable. A product-mode
    # fact: the project-mode template is a single abstract ``Variable`` and does
    # not enumerate concrete variables, so ``required`` only has meaning per
    # documented concrete variable (stays ``None`` on the template).
    required: bool | None = None

    # Shared: rule-bearing attribute specs (project) or concrete values (product).
    attributes: list[AttributeDoc] = Field(default_factory=list)

    # Rule-bearing field (project mode): structural rules on the variable model.
    rules: list[RuleDoc] | None = None


class NodeRef(BaseModel):
    """A reference to a node template registered in the root ``defs`` registry.

    Used (project mode only) to represent the recursive ``Group`` slot without
    infinite expansion: the slot holds a ``NodeRef`` whose ``ref`` names the
    template in :attr:`ProjectDoc.defs`. The ``kind`` tag discriminates the
    ``GroupDoc | NodeRef`` union so the IR round-trips through JSON.
    """

    kind: Literal["ref"] = "ref"
    ref: str


# A container's ``variables`` / ``dimensions`` slot is either an inlined concrete
# node (product mode) or a ``NodeRef`` to the single shared template registered
# in ``defs`` (project mode). Discriminated on ``kind`` so product consumers can
# ignore the ref arm and the IR round-trips through JSON.
VariableChild = Annotated[Union[VariableDoc, NodeRef], Field(discriminator="kind")]
DimensionChild = Annotated[Union[DimensionDoc, NodeRef], Field(discriminator="kind")]

# The single structural location where the recursive group union appears: a
# child group is either an inlined ``GroupDoc`` (product mode) or a ``NodeRef``
# to the group template (project mode). Discriminated on ``kind`` so product
# consumers can ignore the ref arm and the IR round-trips through JSON. The
# ``GroupDoc`` arm is a forward reference resolved by ``model_rebuild`` below.
GroupChild = Annotated[Union["GroupDoc", NodeRef], Field(discriminator="kind")]


class GroupDoc(BaseModel):
    """A group container node, in either mode.

    A group mirrors the dataset's structure — attributes, variables, dimensions
    and (recursively) child groups. Project mode emits a single *template*
    (``name`` is ``None``) carrying the rule-bearing attribute specs, the group
    model's structural ``rules`` and the variable/dimension templates; its own
    recursive ``groups`` slot is a ``NodeRef`` back to the template. Product mode
    fills the concrete ``name`` and inlines the child groups.
    """

    kind: Literal["group"] = "group"
    name: str | None = None
    attributes: list[AttributeDoc] = Field(default_factory=list)
    rules: list[RuleDoc] | None = None
    variables: list[VariableChild] = Field(default_factory=list)
    dimensions: list[DimensionChild] = Field(default_factory=list)
    groups: list[GroupChild] = Field(default_factory=list)


class DatasetDoc(BaseModel):
    """The root container node. Holds the global attributes (slice 1),
    plus the variables and dimensions (slice 5), the groups (slice 6) and the
    ``meta`` section (slice 7).

    ``rules`` carries the container-level (model-bound) validator rules declared
    on the ``Dataset`` model itself — the structural requirements
    (``variable_exists`` / ``dimension_exists`` / … and bespoke model
    validators) that apply to the dataset as a whole rather than to a single
    attribute. ``variables`` holds a single ``NodeRef`` to the shared
    ``Variable`` template (project mode, template in ``defs``) or the N concrete
    variables in product mode; ``dimensions`` likewise. ``groups`` holds a single
    ``NodeRef`` (project mode, template in ``defs``) or the N inlined
    ``GroupDoc`` nodes (product mode).

    ``meta`` documents the dataset's headline section — file pattern,
    short/canonical name, description and references — as a list of
    ``AttributeDoc``, reusing the attribute node so a renderer treats it the
    same way: project mode fills each field's spec (description / example /
    required / constraints), product mode fills the concrete value.
    """

    attributes: list[AttributeDoc] = Field(default_factory=list)
    meta: list[AttributeDoc] = Field(default_factory=list)
    rules: list[RuleDoc] | None = None
    variables: list[VariableChild] = Field(default_factory=list)
    dimensions: list[DimensionChild] = Field(default_factory=list)
    groups: list[GroupChild] = Field(default_factory=list)


# A template registered in ``ProjectDoc.defs``: a group, variable or dimension
# template, discriminated on ``kind`` so the heterogeneous registry round-trips
# through JSON. A matching ``NodeRef.ref`` resolves against the registry key.
TemplateDef = Annotated[
    Union[GroupDoc, VariableDoc, DimensionDoc], Field(discriminator="kind")
]


class ProjectDoc(BaseModel):
    """The IR root produced by :func:`document_project`.

    Carries the documented ``dataset`` plus root-level concerns: ``diagnostics``
    (documentation gaps surfaced during the walk) and the project-only ``defs``
    registry used to represent shared/recursive node templates without infinite
    expansion or duplication.
    """

    mode: Literal["project"] = "project"
    dataset: DatasetDoc
    diagnostics: list[str] = Field(default_factory=list)
    # Project-only registry of reusable node templates, keyed by the referenced
    # model's type name. Holds the recursive ``Group`` template plus the shared
    # ``Variable`` / ``Dimension`` templates — each documented once even though a
    # group reuses the same models the dataset does. A matching ``NodeRef.ref``
    # resolves against the key.
    defs: dict[str, TemplateDef] = Field(default_factory=dict)


class ProductDoc(BaseModel):
    """The IR root produced by :func:`document_product`.

    Carries the documented ``dataset`` and ``diagnostics``. ``defs`` is unused
    in product mode (groups are inlined), so it is absent here.
    """

    mode: Literal["product"] = "product"
    dataset: DatasetDoc
    diagnostics: list[str] = Field(default_factory=list)


# Resolve the ``"GroupDoc"`` forward reference in ``GroupChild`` now that every
# node type exists, so the recursive ``groups`` union is fully built.
GroupDoc.model_rebuild()
DatasetDoc.model_rebuild()
