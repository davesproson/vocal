"""Documentation-gap diagnostics collected during the project walk.

Diagnostics are **non-fatal**: the project walk always completes and returns the
IR with a (possibly empty) ``diagnostics`` list — they flag documentation gaps
for a maintainer to fix, never abort the walk. Two gaps are surfaced:

* an **undescribed validator** — a custom ``vocal`` validator whose
  ``.description`` is empty. Its rule still lands in the IR (see
  :mod:`vocal.autodoc.rules`), but with no human-readable meaning, so docs would
  render a blank rule; and
* a **field-name / ``Vocal*Mixin`` mismatch** — a canonical CDM field
  (``attributes`` / ``dimensions`` / ``variables`` / ``groups``) whose model
  type does not carry the mixin the convention expects, or carries the mixin a
  *different* canonical field expects. The walk keys on the field name (so it
  survives class renames); the mixin is only a sanity check, and a mismatch
  means the structure convention and the domain model have drifted apart.

These are pure detectors plus a thin collector; the project walker threads a
``diagnostics`` list through and appends what they report.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from vocal.mixins import (
    VocalAttributesMixin,
    VocalDimensionMixin,
    VocalGroupMixin,
    VocalVariableMixin,
)
from vocal.validation import Attribute, Model

from .rules import iter_validators

# Canonical CDM field name -> the ``Vocal*Mixin`` its model is expected to carry.
# ``meta`` has no anchoring mixin (it is a plain headline model), so it is absent
# and never produces a mixin diagnostic.
_EXPECTED_MIXIN: dict[str, type] = {
    "attributes": VocalAttributesMixin,
    "dimensions": VocalDimensionMixin,
    "variables": VocalVariableMixin,
    "groups": VocalGroupMixin,
}
# Every mixin that anchors a canonical field, for the "carries the wrong one"
# (vice-versa) branch of the mismatch check.
_FIELD_MIXINS: tuple[type, ...] = tuple(_EXPECTED_MIXIN.values())


def _binding_label(binding: Any) -> str:
    """A short human label for a validator's ``.binding``, for diagnostics."""
    if isinstance(binding, Attribute):
        return f"attribute '{binding.name}'"
    if isinstance(binding, Model):
        return f"model.{binding.value}"
    return "unbound"


def undescribed_validator(model: type[BaseModel], func: Any) -> str | None:
    """Return a diagnostic if ``func`` (a validator on ``model``) has no description.

    Returns ``None`` when the validator carries a non-blank ``.description``.
    """
    if (getattr(func, "description", "") or "").strip():
        return None
    return (
        f"{model.__name__}: a {_binding_label(getattr(func, 'binding', None))} "
        f"validator has an empty description"
    )


def mixin_mismatch(field_name: str, model: type[BaseModel] | None) -> str | None:
    """Return a diagnostic if ``model`` on canonical ``field_name`` mismatches its mixin.

    ``model`` is the resolved model class for the field (``None`` when the field
    is absent). Returns ``None`` for a field with no anchoring mixin (e.g.
    ``meta``), for an absent field, or when ``model`` carries the expected mixin.
    """
    expected = _EXPECTED_MIXIN.get(field_name)
    if expected is None or model is None:
        return None
    if issubclass(model, expected):
        return None
    wrong = next(
        (m for m in _FIELD_MIXINS if m is not expected and issubclass(model, m)),
        None,
    )
    if wrong is not None:
        return (
            f"{model.__name__} on field '{field_name}' carries {wrong.__name__} "
            f"but '{field_name}' expects {expected.__name__}"
        )
    return (
        f"{model.__name__} on field '{field_name}' does not carry the expected "
        f"{expected.__name__}"
    )


def record_undescribed(model: type[BaseModel] | None, diagnostics: list[str]) -> None:
    """Append a diagnostic for every undescribed validator declared on ``model``.

    Considers only validators declared directly on ``model`` (see
    :func:`vocal.autodoc.rules.iter_validators`), so a gap is reported against
    the class that declares the validator, not one that merely inherits it.
    """
    if model is None:
        return
    for func in iter_validators(model):
        message = undescribed_validator(model, func)
        if message is not None:
            diagnostics.append(message)


def record_mixin_mismatch(
    field_name: str, model: type[BaseModel] | None, diagnostics: list[str]
) -> None:
    """Append a diagnostic if ``model`` on canonical ``field_name`` mismatches its mixin."""
    message = mixin_mismatch(field_name, model)
    if message is not None:
        diagnostics.append(message)
