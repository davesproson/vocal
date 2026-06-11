"""Small introspection helpers shared by the walkers.

The walks key on the netCDF Common Data Model field names (``attributes`` /
``dimensions`` / ``variables`` / ``groups``) plus vocal's ``meta`` — never on a
mixin — so autodoc survives internal class renames. These helpers locate those
fields and unwrap their annotations.
"""

from __future__ import annotations

import typing
from typing import Any

from pydantic import BaseModel


def model_from_annotation(annotation: Any) -> type[BaseModel] | None:
    """Return the pydantic model in ``annotation``, unwrapping ``Optional`` etc.

    Handles a bare model, ``Optional[Model]`` / ``Model | None``, and
    ``list[Model]`` containers — the shapes the CDM fields use. Returns ``None``
    if no ``BaseModel`` subclass is found.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    for arg in typing.get_args(annotation):
        found = model_from_annotation(arg)
        if found is not None:
            return found

    return None


def field_model(model: type[BaseModel], field_name: str) -> type[BaseModel] | None:
    """Return the model class declared by ``model``'s ``field_name`` field.

    Looks the field up by its canonical name (the CDM convention) and unwraps
    the annotation. Returns ``None`` if the field is absent or carries no model.
    Use this for single-model slots (``attributes`` / ``meta`` / ``variables`` /
    ``dimensions``); for a slot whose element type is a union of *flavours* (e.g.
    a ``groups`` field of ``list[ImagerGroup | PlatformGroup]``) use
    :func:`field_models`, which preserves every member.
    """
    field = model.model_fields.get(field_name)
    if field is None:
        return None
    return model_from_annotation(field.annotation)


def models_from_annotation(annotation: Any) -> list[type[BaseModel]]:
    """Return *every* pydantic model in ``annotation``, in declaration order.

    Like :func:`model_from_annotation` but does not collapse a union to its first
    member: ``list[A | B | C]`` yields ``[A, B, C]``. This is what a heterogeneous
    ``groups`` slot needs — each group flavour is a distinct node to document.
    Duplicates are removed while preserving first-seen order.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]

    found: dict[type[BaseModel], None] = {}
    for arg in typing.get_args(annotation):
        for model in models_from_annotation(arg):
            found[model] = None
    return list(found)


def field_models(model: type[BaseModel], field_name: str) -> list[type[BaseModel]]:
    """Return every model class in ``model``'s ``field_name`` field's annotation.

    The plural counterpart of :func:`field_model`: a union-typed slot keeps all
    its members. Returns ``[]`` if the field is absent or carries no model.
    """
    field = model.model_fields.get(field_name)
    if field is None:
        return []
    return models_from_annotation(field.annotation)
