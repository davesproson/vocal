"""Normalise a field's JSON-schema fragment into typed constraints.

A project's Field-level constraints (``type``, ``pattern``, numeric bounds,
``min_length``/``max_length``, enum) survive into ``model_json_schema()`` as a
per-property fragment. This module maps such a fragment to a uniform list of
:class:`~vocal.autodoc.ir.ConstraintDoc`, so a renderer never has to
re-implement JSON-schema interpretation and the IR never stores raw schema
dicts.

The single public function :func:`normalize_constraints` is pure: it takes the
fragment dict and returns the constraint list. Optional/union fields (whose
constraints sit inside an ``anyOf`` alongside a ``null`` arm) are unwrapped to
the non-null arm(s); the field's required/optional status is carried separately
on ``AttributeDoc.required``.
"""

from __future__ import annotations

from typing import Any

from .ir import ConstraintDoc

# JSON-schema bound keys -> the Field-argument name used in ``range`` detail.
_RANGE_KEYS = (
    ("minimum", "ge"),
    ("maximum", "le"),
    ("exclusiveMinimum", "gt"),
    ("exclusiveMaximum", "lt"),
)

# JSON-schema length keys -> the Field-argument name used in ``length`` detail.
_LENGTH_KEYS = (
    ("minLength", "min_length"),
    ("maxLength", "max_length"),
)


def normalize_constraints(fragment: dict[str, Any]) -> list[ConstraintDoc]:
    """Map a field's JSON-schema fragment to a list of :class:`ConstraintDoc`.

    Recognises ``type``, ``pattern``, numeric bounds (``minimum`` / ``maximum``
    / ``exclusiveMinimum`` / ``exclusiveMaximum`` collapsed into a single
    ``range`` constraint keyed ``ge`` / ``le`` / ``gt`` / ``lt``), ``minLength``
    / ``maxLength`` (a ``length`` constraint) and ``enum``. The constraints are
    returned in a stable order (type, pattern, range, length, enum). Unknown
    keys are ignored.
    """
    if "anyOf" in fragment:
        # Optional/union: constraints live on the non-null arm(s).
        seen: list[ConstraintDoc] = []
        for arm in fragment["anyOf"]:
            if isinstance(arm, dict) and arm.get("type") == "null":
                continue
            for constraint in normalize_constraints(arm):
                if constraint not in seen:
                    seen.append(constraint)
        return seen

    constraints: list[ConstraintDoc] = []

    if "type" in fragment:
        constraints.append(ConstraintDoc(kind="type", detail={"type": fragment["type"]}))

    if "pattern" in fragment:
        constraints.append(
            ConstraintDoc(kind="pattern", detail={"pattern": fragment["pattern"]})
        )

    range_detail = {
        name: fragment[key] for key, name in _RANGE_KEYS if key in fragment
    }
    if range_detail:
        constraints.append(ConstraintDoc(kind="range", detail=range_detail))

    length_detail = {
        name: fragment[key] for key, name in _LENGTH_KEYS if key in fragment
    }
    if length_detail:
        constraints.append(ConstraintDoc(kind="length", detail=length_detail))

    if "enum" in fragment:
        constraints.append(
            ConstraintDoc(kind="enum", detail={"values": list(fragment["enum"])})
        )

    return constraints
