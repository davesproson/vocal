"""Tests for the derived-type vocabulary in :mod:`vocal.types.schema_types`.

The ``OptionalDerived*`` constants are convenience placeholder strings exported
for project authors (re-exported via ``from vocal.types import *``). Each name
encodes a dtype and whether it is an array; this checks that the string each
constant actually holds agrees with its name. A constant naming one dtype but
holding another (a copy-paste error this table is prone to) is caught here.

Expectations are derived from each constant's *name* rather than from its
non-optional sibling, so the check is self-contained and a matching error in
both siblings would not hide the bug.
"""

import numpy as np
import pytest

from vocal.types import schema_types
from vocal.utils.placeholder import Placeholder

# Maps the type fragment in a constant name to the numpy dtype string it should
# encode. e.g. ``OptionalDerivedFloat16`` -> fragment ``Float16`` -> ``float16``.
_FRAGMENT_TO_DTYPE = {
    "String": "str",
    "Byte": "int8",
    "UByte": "uint8",
    "Integer8": "int8",
    "Integer16": "int16",
    "Integer32": "int32",
    "Integer64": "int64",
    "UInteger8": "uint8",
    "UInteger16": "uint16",
    "UInteger32": "uint32",
    "UInteger64": "uint64",
    "Float16": "float16",
    "Float32": "float32",
    "Float64": "float64",
}


def _optional_derived_names() -> list[str]:
    return sorted(
        name for name in vars(schema_types) if name.startswith("OptionalDerived")
    )


def _expected(name: str) -> tuple[np.dtype, bool]:
    """Return the ``(dtype, is_array)`` a constant's name says it should encode."""
    fragment = name[len("OptionalDerived") :]
    is_array = fragment.endswith("Array")
    if is_array:
        fragment = fragment[: -len("Array")]
    return np.dtype(_FRAGMENT_TO_DTYPE[fragment]), is_array


def test_optional_derived_constants_are_discovered():
    # Guards against the introspection silently matching nothing or too little,
    # which would make every parametrised case vacuous.
    names = _optional_derived_names()
    assert len(names) >= len(_FRAGMENT_TO_DTYPE)  # at least every scalar variant
    assert "OptionalDerivedFloat16" in names
    assert "OptionalDerivedFloat16Array" in names


@pytest.mark.parametrize("name", _optional_derived_names())
def test_optional_derived_constant_matches_its_name(name: str):
    value = getattr(schema_types, name)
    expected_dtype, expected_is_array = _expected(name)

    placeholder = Placeholder.parse(value)

    assert placeholder.dtype == expected_dtype, (
        f"{name} encodes dtype {placeholder.dtype}, name implies {expected_dtype}"
    )
    assert placeholder.is_array is expected_is_array
    # Every OptionalDerived* constant must carry the optional flag.
    assert placeholder.optional is True
