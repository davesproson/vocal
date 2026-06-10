"""Walk a project's pydantic model tree into the documentation IR.

The project path documents the *abstract standard*: the template of what is
allowed/required. Slice 1 covers global attributes only — name, description,
example and required/optional status — read straight off the model fields. The
walk keys on the canonical CDM field name ``attributes`` (the ``Vocal*Mixin`` is
a sanity check only — a mismatch is recorded as a diagnostic, never raised).

The walk threads a ``diagnostics`` list through the recursion so documentation
gaps — undescribed validators and field-name/mixin mismatches — are collected
inline (no separate lint pass) and surfaced on the returned ``ProjectDoc``.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ._introspect import field_model
from .constraints import normalize_constraints
from .diagnostics import record_mixin_mismatch, record_undescribed
from .ir import (
    AttributeDoc,
    DatasetDoc,
    DimensionChild,
    DimensionDoc,
    GroupChild,
    GroupDoc,
    NodeRef,
    ProjectDoc,
    RuleDoc,
    TemplateDef,
    VariableChild,
    VariableDoc,
)
from .rules import attribute_rules, model_rules

# The project-mode child slot a ``NodeRef`` is placed into: ``VariableChild`` /
# ``DimensionChild`` / ``GroupChild`` (each a ``XxxDoc | NodeRef`` union). The
# template walker always emits ``NodeRef``s, but the slot is union-typed for
# product-mode round-tripping. Constraining the return TypeVar to the three child
# unions lets it adapt to the call site rather than narrowing to
# ``list[NodeRef]`` (which list invariance would reject in the union-typed slot).
_Child = TypeVar("_Child", VariableChild, DimensionChild, GroupChild)


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


def _field(
    container: type[BaseModel], field_name: str, diagnostics: list[str]
) -> type[BaseModel] | None:
    """Resolve a canonical field's model, recording any field-name/mixin mismatch.

    The mismatch check is a sanity check only — it appends a diagnostic and the
    walk carries on with whatever model the field actually declares.
    """
    model = field_model(container, field_name)
    record_mixin_mismatch(field_name, model, diagnostics)
    return model


def _document_attributes(
    model: type[BaseModel] | None, diagnostics: list[str]
) -> list[AttributeDoc]:
    """Document every attribute declared on an attributes container model."""
    if model is None:
        return []
    record_undescribed(model, diagnostics)
    properties = model.model_json_schema().get("properties", {})
    rules = attribute_rules(model)
    return [
        _attribute_doc(name, field, properties.get(name, {}), rules.get(name))
        for name, field in model.model_fields.items()
    ]


def _variable_template(
    model: type[BaseModel], defs: dict[str, TemplateDef], diagnostics: list[str]
) -> VariableDoc:
    """Build the variable *template* — what every variable must look like.

    Carries the rule-bearing attribute specs (reusing the attribute walk, so
    constraints and attribute-bound rules compose in automatically) and the
    variable model's own structural rules; ``name`` / ``datatype`` /
    ``dimensions`` stay ``None``. Registered once in ``defs`` and referenced from
    every container's ``variables`` slot (see :func:`_project_template`).
    """
    record_undescribed(model, diagnostics)
    return VariableDoc(
        attributes=_document_attributes(
            _field(model, "attributes", diagnostics), diagnostics
        ),
        rules=model_rules(model) or None,
    )


def _dimension_template(
    model: type[BaseModel], defs: dict[str, TemplateDef], diagnostics: list[str]
) -> DimensionDoc:
    """Build the dimension *template*: only the dimension model's structural
    rules (``name`` / ``size`` stay ``None``). Registered once in ``defs`` and
    referenced from every container's ``dimensions`` slot."""
    record_undescribed(model, diagnostics)
    return DimensionDoc(rules=model_rules(model) or None)


def _group_template(
    group_model: type[BaseModel], defs: dict[str, TemplateDef], diagnostics: list[str]
) -> GroupDoc:
    """Build the single ``GroupDoc`` template for the recursive group model.

    A group mirrors the dataset's structure, so the template reuses the
    attribute walk and *references* the shared variable / dimension templates
    (registered once in ``defs`` — a group's ``Variable`` is the same model the
    dataset uses, so it is documented once, not re-expanded here). It carries the
    group model's own structural rules. Its own recursive ``groups`` slot is a
    ``NodeRef`` back to itself, keeping the template finite — the only recursive
    type in a vocal standard is ``Group``.
    """
    record_undescribed(group_model, diagnostics)
    nested = _field(group_model, "groups", diagnostics)
    return GroupDoc(
        attributes=_document_attributes(
            _field(group_model, "attributes", diagnostics), diagnostics
        ),
        rules=model_rules(group_model) or None,
        variables=_project_template(
            group_model, "variables", defs, diagnostics, _variable_template
        ),
        dimensions=_project_template(
            group_model, "dimensions", defs, diagnostics, _dimension_template
        ),
        groups=[NodeRef(ref=group_model.__name__)] if nested is not None else [],
    )


def _project_template(
    container: type[BaseModel],
    field_name: str,
    defs: dict[str, TemplateDef],
    diagnostics: list[str],
    build: Callable[
        [type[BaseModel], dict[str, TemplateDef], list[str]], TemplateDef
    ],
) -> list[_Child]:
    """Document a container's ``field_name`` slot as a single ``NodeRef``.

    The referenced model's template is built once via ``build`` and registered
    in ``defs`` under the model's type name; repeat occurrences of the same model
    — the shared ``Variable`` / ``Dimension`` reused by every group, or the
    recursive ``Group`` itself — resolve to that one entry instead of being
    re-expanded. Returns an empty list when the container declares no such field.
    Variables, dimensions and groups are all represented this way in project
    mode, so a renderer dereferences every slot uniformly.
    """
    model = _field(container, field_name, diagnostics)
    if model is None:
        return []
    name = model.__name__
    if name not in defs:
        defs[name] = build(model, defs, diagnostics)
    return [NodeRef(ref=name)]


def document_project(dataset: type[BaseModel]) -> ProjectDoc:
    """Document a project's root ``Dataset`` model into a :class:`ProjectDoc`.

    Accepts the ``Dataset`` *class* directly (the core owns no importing). The
    global attributes are documented along with the dataset's own model-bound
    (structural) rules and the ``meta`` section's field specs. The variable,
    dimension and group templates are each registered once in ``defs`` and
    referenced (via ``NodeRef``) from the dataset and from any group that reuses
    the same model, so a shared ``Variable`` is documented once rather than
    re-expanded under every group. Documentation gaps found along the way —
    undescribed validators and field-name/mixin mismatches — are collected
    non-fatally into ``diagnostics``.
    """
    defs: dict[str, TemplateDef] = {}
    diagnostics: list[str] = []
    record_undescribed(dataset, diagnostics)
    doc = DatasetDoc(
        attributes=_document_attributes(
            _field(dataset, "attributes", diagnostics), diagnostics
        ),
        meta=_document_attributes(_field(dataset, "meta", diagnostics), diagnostics),
        rules=model_rules(dataset) or None,
        variables=_project_template(
            dataset, "variables", defs, diagnostics, _variable_template
        ),
        dimensions=_project_template(
            dataset, "dimensions", defs, diagnostics, _dimension_template
        ),
        groups=_project_template(dataset, "groups", defs, diagnostics, _group_template),
    )
    # De-duplicate while preserving first-seen order: a recursive group's slot is
    # re-resolved when its template is built, which can re-report the same gap.
    diagnostics = list(dict.fromkeys(diagnostics))
    return ProjectDoc(dataset=doc, defs=defs, diagnostics=diagnostics)
