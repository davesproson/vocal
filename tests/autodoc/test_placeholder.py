"""Unit tests for the autodoc value classifier (concrete vs runtime-derived)."""

import pytest

from vocal.autodoc.placeholder import parse_value
from vocal.utils.placeholder import InvalidPlaceholder


class TestDerivedPlaceholders:
    def test_scalar_placeholder_recovers_datatype(self) -> None:
        parsed = parse_value("<float32: derived_from_file>")
        assert parsed.value is None
        assert parsed.derived is True
        assert parsed.datatype == "float32"
        assert parsed.placeholder is not None
        assert parsed.placeholder.is_array is False

    @pytest.mark.parametrize(
        "dtype",
        ["float16", "float32", "float64", "int8", "int32", "uint8", "str"],
    )
    def test_various_scalar_dtypes(self, dtype: str) -> None:
        parsed = parse_value(f"<{dtype}: derived_from_file>")
        assert parsed.derived is True
        assert parsed.datatype == dtype
        assert parsed.value is None
        assert parsed.placeholder is not None
        assert parsed.placeholder.is_array is False

    def test_array_placeholder_wraps_datatype(self) -> None:
        parsed = parse_value("<Array[str]: derived_from_file>")
        assert parsed.derived is True
        assert parsed.datatype == "Array[str]"
        assert parsed.placeholder is not None
        assert parsed.placeholder.is_array is True

    def test_optional_placeholder_flagged(self) -> None:
        parsed = parse_value("<str: derived_from_file optional>")
        assert parsed.derived is True
        assert parsed.placeholder is not None
        assert parsed.placeholder.optional is True

    def test_constraints_carried_on_placeholder(self) -> None:
        # The constraints are read straight off the canonical Placeholder model,
        # not re-flattened onto ParsedValue.
        parsed = parse_value("<str: derived_from_file optional,regex=[a-z][0-9]{3}>")
        assert parsed.placeholder is not None
        assert parsed.placeholder.optional is True
        assert parsed.placeholder.constraints.regex == "[a-z][0-9]{3}"
        assert parsed.placeholder.constraints.min_len is None
        assert parsed.placeholder.constraints.max_len is None

    def test_length_constraints_carried_on_placeholder(self) -> None:
        parsed = parse_value("<Array[str]: derived_from_file min_len=1,max_len=5>")
        assert parsed.placeholder is not None
        assert parsed.placeholder.constraints.min_len == 1
        assert parsed.placeholder.constraints.max_len == 5
        assert parsed.placeholder.constraints.regex is None


class TestConcreteValues:
    @pytest.mark.parametrize(
        "value",
        ["degree_north", "EPSG:4979", 42, 3.14, ["a", "b"], 0, None],
    )
    def test_concrete_values_pass_through(self, value: object) -> None:
        parsed = parse_value(value)
        assert parsed.value == value
        assert parsed.derived is False
        assert parsed.datatype is None
        assert parsed.placeholder is None

    def test_plain_string_is_not_derived(self) -> None:
        parsed = parse_value("not a placeholder")
        assert parsed.derived is False
        assert parsed.value == "not a placeholder"


class TestMalformedPlaceholders:
    def test_malformed_derived_marker_raises(self) -> None:
        # Carries the derived marker but is not a valid placeholder: fail loudly
        # rather than silently treat as a concrete value.
        with pytest.raises(InvalidPlaceholder):
            parse_value("derived_from_file but no brackets")
