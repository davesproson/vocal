import pytest

import numpy as np

from vocal.utils.placeholder import (
    Placeholder,
    get_attribute_props_from_placeholder,
    get_type_from_placeholder,
    InvalidPlaceholder,
)

# ---------------------------------------------------------------------------
# Placeholder parsing
# ---------------------------------------------------------------------------


class TestGetAttributePropsFromPlaceholder:
    def test_required_attribute(self) -> None:
        props = get_attribute_props_from_placeholder("<str: derived_from_file>")
        assert props.optional is False
        assert props.regex is None

    def test_optional_attribute(self) -> None:
        props = get_attribute_props_from_placeholder(
            "<str: derived_from_file optional>"
        )
        assert props.optional is True

    def test_attribute_with_regex(self) -> None:
        props = get_attribute_props_from_placeholder(
            r"<str: derived_from_file regex=\d{4}-\d{2}-\d{2}>"
        )
        assert props.regex is not None
        assert props.regex == r"\d{4}-\d{2}-\d{2}"

    def test_invalid_placeholder_raises(self) -> None:
        with pytest.raises(InvalidPlaceholder):
            get_attribute_props_from_placeholder("not_a_placeholder")


class TestGetTypeFromPlaceholder:
    def test_float32(self) -> None:
        dtype, container = get_type_from_placeholder("<float32: derived_from_file>")
        assert dtype == np.dtype("float32")
        assert container is None

    def test_float64(self) -> None:
        dtype, container = get_type_from_placeholder("<float64: derived_from_file>")
        assert dtype == np.dtype("float64")

    def test_array_placeholder(self) -> None:
        dtype, container = get_type_from_placeholder(
            "<Array[float32]: derived_from_file>"
        )
        assert dtype == np.dtype("float32")
        assert container == "Array"

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(InvalidPlaceholder):
            get_type_from_placeholder("not_a_placeholder")


class TestPlaceholderParsingIntegration:
    def test_full_placeholder(self) -> None:
        placeholder_str = (
            "<Array[float64]: derived_from_file optional,regex=\\d{4}-\\d{2}-\\d{2}>"
        )
        props = get_attribute_props_from_placeholder(placeholder_str)
        dtype, container = get_type_from_placeholder(placeholder_str)

        assert dtype == np.dtype("float64")
        assert container == "Array"
        assert props.optional is True
        assert props.regex == r"\d{4}-\d{2}-\d{2}"

    def test_placeholder_parser(self) -> None:
        placeholder_str = (
            "<Array[float64]: derived_from_file optional,regex=\\d{4}-\\d{2}-\\d{2}>"
        )

        ph = Placeholder.parse(placeholder_str)
        assert ph.dtype == np.dtype("float64")
        assert ph.is_array is True
        assert ph.optional is True
        assert ph.regex == r"\d{4}-\d{2}-\d{2}"
