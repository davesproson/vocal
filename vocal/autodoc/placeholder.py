"""Parse product attribute values into concrete-or-derived form.

A product pack records each attribute value either as a concrete value (a
string, number, list, …) or as a ``<dtype: derived_from_file>`` placeholder
meaning "this is filled in per-file at runtime". Documentation needs to
distinguish the two: a concrete value is shown as-is, a placeholder is shown as
a runtime-derived value of a known datatype.

This module wraps the canonical :class:`vocal.utils.placeholder.Placeholder`
parser in a pure function that classifies *any* raw value, rather than raising
on non-placeholders the way ``Placeholder.parse`` does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vocal.utils.placeholder import PLACEHOLDER_RE, Placeholder

# The marker that distinguishes a runtime-derived placeholder from a concrete
# value. A string only goes through the (stricter) placeholder parser if it
# carries this marker; everything else is a concrete value.
_DERIVED_MARKER = "derived_from_file"


@dataclass(frozen=True)
class ParsedValue:
    """The classification of a single raw product attribute value.

    For a concrete value, ``value`` holds it, ``datatype`` is ``None`` and
    ``placeholder`` is ``None``. For a derived placeholder, ``value`` is ``None``,
    ``datatype`` names the dtype the runtime value will take (e.g. ``"float32"``
    or ``"Array[str]"``) and ``placeholder`` carries the full canonical
    :class:`~vocal.utils.placeholder.Placeholder` — its ``optional`` flag and
    ``constraints`` (regex / length) are read straight off that model rather than
    re-flattened here, so new placeholder constraints surface with no change to
    this class.
    """

    value: Any | None
    derived: bool
    datatype: str | None = None
    placeholder: Placeholder | None = None


def parse_value(raw: Any) -> ParsedValue:
    """Classify a raw product attribute value as concrete or runtime-derived.

    A ``<float32: derived_from_file>`` placeholder becomes
    ``ParsedValue(value=None, derived=True, datatype="float32", placeholder=…)``;
    an ``<Array[str]: derived_from_file>`` placeholder reports
    ``datatype="Array[str]"``. Any other value (a plain string, number, list, …)
    passes through unchanged with ``derived=False`` and ``placeholder=None``.

    A string carrying the ``derived_from_file`` marker that does not parse as a
    valid placeholder raises :class:`~vocal.utils.placeholder.InvalidPlaceholder`
    (via the underlying parser) rather than being silently treated as concrete.
    """
    if isinstance(raw, str) and _DERIVED_MARKER in raw:
        # Validate/normalise through the canonical parser (this raises
        # InvalidPlaceholder on a malformed marker string).
        placeholder = Placeholder.parse(raw)
        # Recover the original dtype *token* for display: Placeholder stores an
        # np.dtype, which doesn't round-trip to the source spelling ("str" etc.),
        # so re-read the token the author wrote.
        match = re.search(PLACEHOLDER_RE, raw)
        assert match is not None  # Placeholder.parse would have raised otherwise
        datatype = match["dtype"]
        if placeholder.is_array:
            datatype = f"Array[{datatype}]"
        return ParsedValue(
            value=None, derived=True, datatype=datatype, placeholder=placeholder
        )

    return ParsedValue(value=raw, derived=False)
