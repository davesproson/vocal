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
        # unrecognised attributes fail loudly rather than being silently ignored
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse("<str: derived_from_file bogus=2>")

    def test_space_separated_attributes_rejected(self) -> None:
        # attributes are comma-separated; spaces are not supported
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse(r"<str: derived_from_file optional regex=\d{3}>")

    def test_regex_on_non_string_dtype_rejected(self) -> None:
        # a regex only makes sense for string-valued attributes
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse(r"<float32: derived_from_file regex=\d+>")

    def test_regex_on_string_array_allowed(self) -> None:
        ph = Placeholder.parse(r"<Array[str]: derived_from_file regex=[A-Z]+>")
        assert ph.is_array is True
        assert ph.constraints.regex == r"[A-Z]+"

    def test_min_len_parsed_and_coerced_to_int(self) -> None:
        ph = Placeholder.parse("<str: derived_from_file min_len=2>")
        assert ph.constraints.min_len == 2

    def test_min_len_with_optional_and_regex(self) -> None:
        ph = Placeholder.parse(
            r"<str: derived_from_file optional,min_len=3,regex=[A-Z]+>"
        )
        assert ph.optional is True
        assert ph.constraints.min_len == 3
        assert ph.constraints.regex == r"[A-Z]+"

    def test_min_len_on_scalar_number_rejected(self) -> None:
        # length only makes sense for strings/arrays
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse("<float32: derived_from_file min_len=2>")

    def test_min_len_on_array_allowed(self) -> None:
        ph = Placeholder.parse("<Array[float32]: derived_from_file min_len=2>")
        assert ph.constraints.min_len == 2

    def test_non_integer_min_len_rejected(self) -> None:
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse("<str: derived_from_file min_len=abc>")

    def test_negative_min_len_rejected(self) -> None:
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse("<str: derived_from_file min_len=-1>")

    def test_max_len_parsed_and_coerced_to_int(self) -> None:
        ph = Placeholder.parse("<str: derived_from_file max_len=5>")
        assert ph.constraints.max_len == 5

    def test_min_and_max_len_combined(self) -> None:
        ph = Placeholder.parse("<str: derived_from_file min_len=2,max_len=5>")
        assert ph.constraints.min_len == 2
        assert ph.constraints.max_len == 5

    def test_max_len_on_scalar_number_rejected(self) -> None:
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse("<float32: derived_from_file max_len=2>")

    def test_max_len_below_one_rejected(self) -> None:
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse("<str: derived_from_file max_len=0>")

    def test_min_len_greater_than_max_len_rejected(self) -> None:
        with pytest.raises(InvalidPlaceholder):
            Placeholder.parse("<str: derived_from_file min_len=5,max_len=2>")


class TestPlaceholderConstraintsCheck:
    def test_scalar_match_passes(self) -> None:
        constraints = Placeholder.parse(
            r"<str: derived_from_file regex=[A-Z]+>"
        ).constraints
        assert all(r.error is None for r in constraints.check("ABC"))

    def test_scalar_mismatch_fails(self) -> None:
        constraints = Placeholder.parse(
            r"<str: derived_from_file regex=[A-Z]+>"
        ).constraints
        assert any(r.error for r in constraints.check("abc"))

    def test_string_array_all_match_passes(self) -> None:
        constraints = Placeholder.parse(
            r"<Array[str]: derived_from_file regex=[A-Z]+>"
        ).constraints
        assert all(r.error is None for r in constraints.check(["ABC", "DEF"]))

    def test_string_array_one_mismatch_fails(self) -> None:
        constraints = Placeholder.parse(
            r"<Array[str]: derived_from_file regex=[A-Z]+>"
        ).constraints
        assert any(r.error for r in constraints.check(["ABC", "de9"]))

    def test_non_string_value_skipped(self) -> None:
        # a non-string value despite a str dtype is a type error, reported by
        # the dtype check; the regex check skips it rather than crashing
        constraints = Placeholder.parse(
            r"<str: derived_from_file regex=[A-Z]+>"
        ).constraints
        assert all(r.error is None for r in constraints.check(3.0))

    def test_min_len_scalar_too_short_fails(self) -> None:
        constraints = Placeholder.parse(
            "<str: derived_from_file min_len=3>"
        ).constraints
        assert any(r.error for r in constraints.check("ab"))

    def test_min_len_scalar_ok_passes(self) -> None:
        constraints = Placeholder.parse(
            "<str: derived_from_file min_len=3>"
        ).constraints
        assert all(r.error is None for r in constraints.check("abc"))

    def test_min_len_array_too_few_elements_fails(self) -> None:
        constraints = Placeholder.parse(
            "<Array[float32]: derived_from_file min_len=2>"
        ).constraints
        assert any(r.error for r in constraints.check([1.0]))

    def test_min_len_array_enough_elements_passes(self) -> None:
        constraints = Placeholder.parse(
            "<Array[float32]: derived_from_file min_len=2>"
        ).constraints
        assert all(r.error is None for r in constraints.check([1.0, 2.0]))

    def test_max_len_scalar_too_long_fails(self) -> None:
        constraints = Placeholder.parse(
            "<str: derived_from_file max_len=3>"
        ).constraints
        assert any(r.error for r in constraints.check("abcd"))

    def test_max_len_scalar_ok_passes(self) -> None:
        constraints = Placeholder.parse(
            "<str: derived_from_file max_len=3>"
        ).constraints
        assert all(r.error is None for r in constraints.check("abc"))

    def test_max_len_array_too_many_elements_fails(self) -> None:
        constraints = Placeholder.parse(
            "<Array[float32]: derived_from_file max_len=2>"
        ).constraints
        assert any(r.error for r in constraints.check([1.0, 2.0, 3.0]))


class TestPlaceholderParsingIntegration:
    def test_full_placeholder(self) -> None:
        placeholder_str = (
            "<Array[str]: derived_from_file optional,regex=\\d{4}-\\d{2}-\\d{2}>"
        )

        ph = Placeholder.parse(placeholder_str)
        assert ph.dtype == np.dtype("str")
        assert ph.is_array is True
        assert ph.optional is True
        assert ph.constraints.regex == r"\d{4}-\d{2}-\d{2}"
