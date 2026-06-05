import pytest

import numpy as np

from vocal.utils.placeholder import (
    Placeholder,
    InvalidPlaceholder,
)

# ---------------------------------------------------------------------------
# Placeholder parsing
# ---------------------------------------------------------------------------

dtypes = [
    ("float16", np.dtype("float16")),
    ("float32", np.dtype("float32")),
    ("float64", np.dtype("float64")),
    ("int8", np.dtype("int8")),
    ("int16", np.dtype("int16")),
    ("int32", np.dtype("int32")),
    ("int64", np.dtype("int64")),
    ("uint8", np.dtype("uint8")),
    ("uint16", np.dtype("uint16")),
    ("uint32", np.dtype("uint32")),
    ("uint64", np.dtype("uint64")),
]


class TestPlaceholderDtype:
    @pytest.mark.parametrize(
        "dtype_str, expected_dtype",
        dtypes,
    )
    def test_various_dtypes(self, dtype_str: str, expected_dtype: np.dtype) -> None:
        ph = Placeholder.parse(f"<{dtype_str}: derived_from_file>")
        assert ph.dtype == expected_dtype
        assert ph.is_array is False

    @pytest.mark.parametrize("dtype_str, expected_dtype", dtypes)
    def test_various_dtypes_in_array(
        self, dtype_str: str, expected_dtype: np.dtype
    ) -> None:
        ph = Placeholder.parse(f"<Array[{dtype_str}]: derived_from_file>")
        assert ph.dtype == expected_dtype
        assert ph.is_array is True

    def test_invalid_raises(self) -> None:
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse("not_a_placeholder")


class TestPlaceholderAttributeProperties:
    def test_required_attribute(self) -> None:
        ph = Placeholder.parse("<str: derived_from_file>")
        assert ph.optional is False
        assert ph.constraints.regex is None

    def test_optional_attribute(self) -> None:
        ph = Placeholder.parse("<str: derived_from_file optional>")
        assert ph.optional is True

    def test_attribute_with_regex(self) -> None:
        ph = Placeholder.parse(r"<str: derived_from_file regex=\d{4}-\d{2}-\d{2}>")
        assert ph.constraints.regex == r"\d{4}-\d{2}-\d{2}"

    def test_regex_may_contain_commas(self) -> None:
        # regex comes last and runs to the end, so commas in the pattern are safe
        ph = Placeholder.parse(r"<str: derived_from_file regex=[A-Z]{2,5}>")
        assert ph.constraints.regex == r"[A-Z]{2,5}"

    def test_optional_and_regex_combined(self) -> None:
        ph = Placeholder.parse(r"<str: derived_from_file optional,regex=[A-Z]{2,5}>")
        assert ph.optional is True
        assert ph.constraints.regex == r"[A-Z]{2,5}"

    def test_unknown_attribute_raises(self) -> None:
        # min_len is not supported yet: fail loudly rather than silently ignore
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse("<str: derived_from_file min_len=2>")

    def test_space_separated_attributes_rejected(self) -> None:
        # attributes are comma-separated; spaces are not supported
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse(r"<str: derived_from_file optional regex=\d{3}>")


class TestPlaceholderParsingIntegration:
    def test_full_placeholder(self) -> None:
        placeholder_str = (
            "<Array[float64]: derived_from_file optional,regex=\\d{4}-\\d{2}-\\d{2}>"
        )

        ph = Placeholder.parse(placeholder_str)
        assert ph.dtype == np.dtype("float64")
        assert ph.is_array is True
        assert ph.optional is True
        assert ph.constraints.regex == r"\d{4}-\d{2}-\d{2}"
