"""Walk a project's pydantic model tree into the documentation IR.

The project path documents the *abstract standard*: the template of what is
allowed/required. Slice 1 covers global attributes only — name, description,
example and required/optional status — read straight off the model fields. The
walk keys on the canonical CDM field name ``attributes`` (the ``Vocal*Mixin`` is
a sanity check only, used from a later diagnostics slice).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ._introspect import field_model
from .constraints import normalize_constraints
from .ir import (
    AttributeDoc,
    DatasetDoc,
    DimensionDoc,
    GroupDoc,
    NodeRef,
    ProjectDoc,
    RuleDoc,
    VariableDoc,
)
from .rules import attribute_rules, model_rules


def _example(field: FieldInfo) -> Any | None:
    """Recover a field's documentation example.

    vocal's ``Field`` wrapper stashes the non-pydantic ``example=`` argument in
    ``json_schema_extra``; fall back to pydantic's native ``examples`` list.
    """
    extra = field.json_schema_extra
    if isinstance(extra, dict) and "example" in extra:
        return extra["example"]
    if field.examples:
        return field.examples[0]
    return None


def _attribute_doc(
    name: str,
    field: FieldInfo,
    fragment: dict[str, Any],
    rules: list[RuleDoc] | None,
) -> AttributeDoc:
    """Document a single attribute field as a rule-bearing ``AttributeDoc``.

    ``fragment`` is the field's JSON-schema fragment, normalised into the
    attribute's typed constraint list; ``rules`` are the custom validator rules
    bound to this attribute (``None`` when it has none).
    """
    return AttributeDoc(
        name=name,
        description=field.description,
        example=_example(field),
        required=field.is_required(),
        constraints=normalize_constraints(fragment),
        rules=rules,
    )


def _document_attributes(model: type[BaseModel] | None) -> list[AttributeDoc]:
    """Document every attribute declared on an attributes container model."""
    if model is None:
        return []
    properties = model.model_json_schema().get("properties", {})
    rules = attribute_rules(model)
    return [
        _attribute_doc(name, field, properties.get(name, {}), rules.get(name))
        for name, field in model.model_fields.items()
    ]


def _variable_template(model: type[BaseModel] | None) -> list[VariableDoc]:
    """Document the variable *template* — what every variable must look like.

    The project does not enumerate concrete variables, so this returns exactly
    one ``VariableDoc`` derived from the ``Variable`` model (or an empty list
    when the dataset declares no ``variables`` field). The template carries the
    rule-bearing attribute specs (reusing the attribute walk, so constraints and
    attribute-bound rules compose in automatically) and the variable model's own
    structural rules; ``name`` / ``datatype`` / ``dimensions`` stay ``None``.
    """
    if model is None:
        return []
    return [
        VariableDoc(
            attributes=_document_attributes(field_model(model, "attributes")),
            rules=model_rules(model) or None,
        )
    ]


def _dimension_template(model: type[BaseModel] | None) -> list[DimensionDoc]:
    """Document the dimension spec as a single template ``DimensionDoc``.

    Like the variable template, the project does not fix concrete dimensions, so
    this returns one ``DimensionDoc`` carrying only the dimension model's
    structural rules (``name`` / ``size`` stay ``None``), or an empty list when
    the dataset declares no ``dimensions`` field.
    """
    if model is None:
        return []
    return [DimensionDoc(rules=model_rules(model) or None)]


def _group_template(group_model: type[BaseModel], name: str) -> GroupDoc:
    """Build the single ``GroupDoc`` template for the recursive group model.

    A group mirrors the dataset's structure, so the template reuses the
    attribute / variable / dimension walks and carries the group model's own
    structural rules. Its own recursive ``groups`` slot is a ``NodeRef`` back to
    itself (``name``), keeping the template finite — the only recursive type in
    a vocal standard is ``Group``.
    """
    nested = field_model(group_model, "groups")
    return GroupDoc(
        attributes=_document_attributes(field_model(group_model, "attributes")),
        rules=model_rules(group_model) or None,
        variables=_variable_template(field_model(group_model, "variables")),
        dimensions=_dimension_template(field_model(group_model, "dimensions")),
        groups=[NodeRef(ref=name)] if nested is not None else [],
    )


def _project_groups(
    container: type[BaseModel], defs: dict[str, GroupDoc]
) -> list[GroupDoc | NodeRef]:
    """Document a container's recursive ``groups`` slot in project mode.

    The slot holds a single ``NodeRef`` to the group template rather than
    expanding the recursive ``Group`` type inline; the template is emitted once
    into the root ``defs`` registry, keyed by the group model's name. Returns an
    empty list when the container declares no ``groups`` field.
    """
    group_model = field_model(container, "groups")
    if group_model is None:
        return []
    name = group_model.__name__
    if name not in defs:
        defs[name] = _group_template(group_model, name)
    return [NodeRef(ref=name)]


def document_project(dataset: type[BaseModel]) -> ProjectDoc:
    """Document a project's root ``Dataset`` model into a :class:`ProjectDoc`.

    Accepts the ``Dataset`` *class* directly (the core owns no importing). The
    global attributes are documented along with the dataset's own model-bound
    (structural) rules, the ``meta`` section's field specs, the variable and
    dimension templates, and the group template (referenced from the dataset
    and registered in ``defs``).
    """
    defs: dict[str, GroupDoc] = {}
    doc = DatasetDoc(
        attributes=_document_attributes(field_model(dataset, "attributes")),
        meta=_document_attributes(field_model(dataset, "meta")),
        rules=model_rules(dataset) or None,
        variables=_variable_template(field_model(dataset, "variables")),
        dimensions=_dimension_template(field_model(dataset, "dimensions")),
        groups=_project_groups(dataset, defs),
    )
    return ProjectDoc(dataset=doc, defs=defs)
