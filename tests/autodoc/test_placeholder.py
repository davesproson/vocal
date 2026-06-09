"""Unit tests for the autodoc value classifier (concrete vs runtime-derived)."""

import pytest

from vocal.autodoc.placeholder import ParsedValue, parse_value
from vocal.utils.placeholder import InvalidPlaceholder


class TestDerivedPlaceholders:
    def test_scalar_placeholder_recovers_datatype(self) -> None:
        parsed = parse_value("<float32: derived_from_file>")
        assert parsed == ParsedValue(
            value=None, derived=True, datatype="float32", is_array=False
        )

    @pytest.mark.parametrize(
        "dtype",
        ["float16", "float32", "float64", "int8", "int32", "uint8", "str"],
    )
    def test_various_scalar_dtypes(self, dtype: str) -> None:
        parsed = parse_value(f"<{dtype}: derived_from_file>")
        assert parsed.derived is True
        assert parsed.datatype == dtype
        assert parsed.value is None
        assert parsed.is_array is False

    def test_array_placeholder_wraps_datatype(self) -> None:
        parsed = parse_value("<Array[str]: derived_from_file>")
        assert parsed.derived is True
        assert parsed.datatype == "Array[str]"
        assert parsed.is_array is True

    def test_optional_placeholder_flagged(self) -> None:
        parsed = parse_value("<str: derived_from_file optional>")
        assert parsed.derived is True
        assert parsed.optional is True


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
