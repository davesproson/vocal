"""
Tests for vocal/validation.py.

Calling convention for validator functions
------------------------------------------
The validator factories (is_exact, is_in, in_vocabulary) return functions that
are intended to be registered as pydantic field validators. Pydantic calls them
as unbound class methods: validator(cls, value). The Validator protocol,
however, models the public signature as validator(value) — the single-arg form
that pydantic exposes after binding.

When calling validators directly in tests we must use the two-argument form
(None, value) to match the actual function signature. mypy flags this as an
error because it sees the Validator protocol, not the underlying function, so
affected locals are typed as Any to acknowledge the deliberate deviation.
"""
from typing import Any

import pytest

from vocal.validation import in_vocabulary, is_exact, is_in, substitute_placeholders
from vocal.vocab import ListVocabulary


# ---------------------------------------------------------------------------
# is_exact
# ---------------------------------------------------------------------------


class TestIsExact:
    def test_correct_value_returns_value(self) -> None:
        validator: Any = is_exact("CF-1.8")
        assert validator(None, "CF-1.8") == "CF-1.8"

    def test_wrong_value_raises(self) -> None:
        validator: Any = is_exact("CF-1.8")
        with pytest.raises(ValueError, match="CF-1.8"):
            validator(None, "CF-1.6")

    def test_validator_has_description(self) -> None:
        validator = is_exact("CF-1.8")
        assert "CF-1.8" in validator.description

    def test_numeric_value_passes(self) -> None:
        validator: Any = is_exact(42)
        assert validator(None, 42) == 42

    def test_numeric_value_wrong_raises(self) -> None:
        validator: Any = is_exact(42)
        with pytest.raises(ValueError):
            validator(None, 43)

    def test_each_call_produces_independent_validator(self) -> None:
        v1: Any = is_exact("a")
        v2: Any = is_exact("b")
        assert v1(None, "a") == "a"
        assert v2(None, "b") == "b"
        with pytest.raises(ValueError):
            v1(None, "b")


# ---------------------------------------------------------------------------
# is_in
# ---------------------------------------------------------------------------


class TestIsIn:
    def test_value_in_collection_passes(self) -> None:
        validator: Any = is_in(["a", "b", "c"])
        assert validator(None, "b") == "b"

    def test_value_not_in_collection_raises(self) -> None:
        validator: Any = is_in(["a", "b", "c"])
        with pytest.raises(ValueError):
            validator(None, "z")

    def test_validator_has_description(self) -> None:
        validator = is_in(["x", "y"])
        assert validator.description

    def test_works_with_non_string_collection(self) -> None:
        validator: Any = is_in([1, 2, 3])
        assert validator(None, 2) == 2
        with pytest.raises(ValueError):
            validator(None, 5)


# ---------------------------------------------------------------------------
# in_vocabulary
# ---------------------------------------------------------------------------


class TestInVocabulary:
    def test_word_in_vocabulary_passes(self) -> None:
        vocab = ListVocabulary("test_vocab", ["air_temperature", "wind_speed"])
        validator: Any = in_vocabulary(vocab)
        assert validator(None, "air_temperature") == "air_temperature"

    def test_word_not_in_vocabulary_raises(self) -> None:
        vocab = ListVocabulary("test_vocab", ["air_temperature"])
        validator: Any = in_vocabulary(vocab)
        with pytest.raises(ValueError):
            validator(None, "not_a_standard_name")

    def test_validator_has_description(self) -> None:
        vocab = ListVocabulary("test_vocab", [])
        validator = in_vocabulary(vocab)
        assert validator.description


# ---------------------------------------------------------------------------
# substitute_placeholders
# ---------------------------------------------------------------------------


class TestSubstitutePlaceholders:
    class MockModel:
        @classmethod
        def model_json_schema(cls) -> dict[str, Any]:
            return {
                "properties": {
                    "title": {"example": "Example Title"},
                    "version": {"example": "1.0"},
                }
            }

    def test_substitutes_derived_placeholder_with_example(self) -> None:
        values: dict[str, Any] = {"title": "<str: derived_from_file>"}
        result = substitute_placeholders(self.MockModel, values)
        assert result["title"] == "Example Title"

    def test_leaves_concrete_value_unchanged(self) -> None:
        values: dict[str, Any] = {"title": "Concrete Title"}
        result = substitute_placeholders(self.MockModel, values)
        assert result["title"] == "Concrete Title"

    def test_skips_key_with_no_example_in_schema(self) -> None:
        values: dict[str, Any] = {"unknown_key": "<str: derived_from_file>"}
        result = substitute_placeholders(self.MockModel, values)
        assert result["unknown_key"] == "<str: derived_from_file>"

    def test_non_string_values_are_skipped(self) -> None:
        values: dict[str, Any] = {"count": 42}
        result = substitute_placeholders(self.MockModel, values)
        assert result["count"] == 42
