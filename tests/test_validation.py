"""
Tests for vocal/validation.py.

Calling convention for validator functions
------------------------------------------
The validator factories (is_exact, is_in, in_vocabulary) bind their validator to
an attribute and return a pydantic descriptor proxy ready to assign as a class
attribute. The underlying function — called by pydantic as an unbound class
method, validator(cls, value) — is reachable via proxy.wrapped.__func__. The
_call helper unwraps and invokes it with the (None, value) form pydantic uses.

Binding metadata (description, binding) is attached to the proxy itself, so it
remains introspectable after binding. Locals are typed as Any because the
Validator protocol models the post-binding single-arg public signature, not the
proxy/underlying-function shapes the tests reach into.
"""

from typing import Any

import pytest

from vocal.validation import in_vocabulary, is_exact, is_in, substitute_placeholders
from vocal.vocab import ListVocabulary


def _call(validator: Any, value: Any) -> Any:
    """Invoke a bound validator's underlying function directly."""
    return validator.wrapped.__func__(None, value)


# ---------------------------------------------------------------------------
# is_exact
# ---------------------------------------------------------------------------


class TestIsExact:
    def test_correct_value_returns_value(self) -> None:
        validator: Any = is_exact("CF-1.8", attribute="conventions")
        assert _call(validator, "CF-1.8") == "CF-1.8"

    def test_wrong_value_raises(self) -> None:
        validator: Any = is_exact("CF-1.8", attribute="conventions")
        with pytest.raises(ValueError, match="CF-1.8"):
            _call(validator, "CF-1.6")

    def test_validator_has_description(self) -> None:
        validator = is_exact("CF-1.8", attribute="conventions")
        assert "CF-1.8" in validator.description

    def test_numeric_value_passes(self) -> None:
        validator: Any = is_exact(42, attribute="answer")
        assert _call(validator, 42) == 42

    def test_numeric_value_wrong_raises(self) -> None:
        validator: Any = is_exact(42, attribute="answer")
        with pytest.raises(ValueError):
            _call(validator, 43)

    def test_each_call_produces_independent_validator(self) -> None:
        v1: Any = is_exact("a", attribute="x")
        v2: Any = is_exact("b", attribute="y")
        assert _call(v1, "a") == "a"
        assert _call(v2, "b") == "b"
        with pytest.raises(ValueError):
            _call(v1, "b")


# ---------------------------------------------------------------------------
# is_in
# ---------------------------------------------------------------------------


class TestIsIn:
    def test_value_in_collection_passes(self) -> None:
        validator: Any = is_in(["a", "b", "c"], attribute="letter")
        assert _call(validator, "b") == "b"

    def test_value_not_in_collection_raises(self) -> None:
        validator: Any = is_in(["a", "b", "c"], attribute="letter")
        with pytest.raises(ValueError):
            _call(validator, "z")

    def test_validator_has_description(self) -> None:
        validator = is_in(["x", "y"], attribute="letter")
        assert validator.description

    def test_works_with_non_string_collection(self) -> None:
        validator: Any = is_in([1, 2, 3], attribute="number")
        assert _call(validator, 2) == 2
        with pytest.raises(ValueError):
            _call(validator, 5)


# ---------------------------------------------------------------------------
# in_vocabulary
# ---------------------------------------------------------------------------


class TestInVocabulary:
    def test_word_in_vocabulary_passes(self) -> None:
        vocab = ListVocabulary("test_vocab", ["air_temperature", "wind_speed"])
        validator: Any = in_vocabulary(vocab, attribute="standard_name")
        assert _call(validator, "air_temperature") == "air_temperature"

    def test_word_not_in_vocabulary_raises(self) -> None:
        vocab = ListVocabulary("test_vocab", ["air_temperature"])
        validator: Any = in_vocabulary(vocab, attribute="standard_name")
        with pytest.raises(ValueError):
            _call(validator, "not_a_standard_name")

    def test_validator_has_description(self) -> None:
        vocab = ListVocabulary("test_vocab", [])
        validator = in_vocabulary(vocab, attribute="standard_name")
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
