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
from .ir import AttributeDoc, DatasetDoc, ProjectDoc, RuleDoc
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


def document_project(dataset: type[BaseModel]) -> ProjectDoc:
    """Document a project's root ``Dataset`` model into a :class:`ProjectDoc`.

    Accepts the ``Dataset`` *class* directly (the core owns no importing). The
    global attributes are documented along with the dataset's own model-bound
    (structural) rules.
    """
    attributes_model = field_model(dataset, "attributes")
    doc = DatasetDoc(
        attributes=_document_attributes(attributes_model),
        rules=model_rules(dataset) or None,
    )
    return ProjectDoc(dataset=doc)
