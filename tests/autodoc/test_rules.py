"""Unit tests for the attribute-bound rule extractor and vocabulary enumeration.

Fixtures are tiny, hermetic, synthetic models carrying ``vocal`` validators,
following the inline-model style in ``tests/test_validation.py`` — no dependency
on an externally-installed project. Tests assert the external behaviour (the
``RuleDoc`` shape and routing) rather than introspection internals.
"""

from pydantic import BaseModel

from vocal.autodoc import RuleDoc, attribute_rules, document_project, model_rules
from vocal.autodoc.rules import rule_doc
from vocal.field import Field
from vocal.validation import (
    in_vocabulary,
    is_exact,
    is_in,
    variable_exists,
    variable_has_dimensions,
)
from vocal.vocab import CoverageContentTypes


class _NonEnumerableVocab:
    """A vocabulary that cannot enumerate its members (``members() -> None``)."""

    description = "An externally-documented vocabulary, too large to enumerate."

    def __contains__(self, word: str) -> bool:
        return True

    def __str__(self) -> str:
        return "Big Vocab"

    def members(self) -> list[str] | None:
        return None


class _Attributes(BaseModel):
    conventions: str = Field(description="Conventions string", example="CF-1.8")
    kind: str = Field(description="A kind", example="a")
    coverage: str = Field(description="Coverage", example="image")
    standard_name: str = Field(description="Std name", example="air_temperature")
    plain: str = Field(description="No rules", example="x")

    _v_conventions = is_exact("CF-1.8", attribute="conventions")
    _v_kind = is_in(["a", "b", "c"], attribute="kind")
    _v_coverage = in_vocabulary(CoverageContentTypes(), attribute="coverage")
    _v_standard = in_vocabulary(_NonEnumerableVocab(), attribute="standard_name")


# ---------------------------------------------------------------------------
# attribute_rules: extraction + routing
# ---------------------------------------------------------------------------


class TestAttributeRules:
    def test_none_model_yields_no_rules(self) -> None:
        assert attribute_rules(None) == {}

    def test_routes_each_rule_to_its_attribute(self) -> None:
        rules = attribute_rules(_Attributes)
        assert set(rules) == {"conventions", "kind", "coverage", "standard_name"}

    def test_attribute_without_validator_is_absent(self) -> None:
        assert "plain" not in attribute_rules(_Attributes)

    def test_is_exact_rule_description(self) -> None:
        (rule,) = attribute_rules(_Attributes)["conventions"]
        assert "CF-1.8" in rule.description
        assert rule.members is None

    def test_is_in_rule_description(self) -> None:
        (rule,) = attribute_rules(_Attributes)["kind"]
        assert rule.description
        assert rule.members is None

    def test_skips_non_vocal_validators(self) -> None:
        # A plain pydantic validator (no vocal metadata) is not a documented
        # rule; only the four vocal validators above are extracted.
        from pydantic import field_validator

        class WithPlain(BaseModel):
            x: str = "y"

            @field_validator("x")
            @classmethod
            def _check(cls, v: str) -> str:
                return v

        assert attribute_rules(WithPlain) == {}


# ---------------------------------------------------------------------------
# model_rules: model-bound (structural) rule extraction + routing
# ---------------------------------------------------------------------------


class _Structural(BaseModel):
    """A container carrying two model-bound (structural) validators."""

    attributes: _Attributes

    _v_temp_exists = variable_exists("temperature")
    _v_temp_dims = variable_has_dimensions("temperature", ["time"])


class TestModelRules:
    def test_none_model_yields_no_rules(self) -> None:
        assert model_rules(None) == []

    def test_collects_each_model_bound_validator(self) -> None:
        rules = model_rules(_Structural)
        assert len(rules) == 2
        descriptions = {rule.description for rule in rules}
        assert any("temperature" in d and "exist" in d for d in descriptions)
        assert any("dimensions" in d for d in descriptions)

    def test_attribute_bound_validators_excluded(self) -> None:
        # ``_Attributes`` carries only attribute-bound validators, so it
        # contributes no model rules.
        assert model_rules(_Attributes) == []

    def test_model_rules_have_no_members(self) -> None:
        assert all(rule.members is None for rule in model_rules(_Structural))

    def test_only_directly_declared_validators_counted(self) -> None:
        # A subclass that declares no new validators inherits the parent's but
        # documents none of its own.
        class _Sub(_Structural):
            pass

        assert model_rules(_Sub) == []


# ---------------------------------------------------------------------------
# Vocabulary enumeration vs. fallback
# ---------------------------------------------------------------------------


class TestVocabularyEnumeration:
    def test_enumerable_vocab_lists_members(self) -> None:
        (rule,) = attribute_rules(_Attributes)["coverage"]
        assert rule.members == CoverageContentTypes.MEMBERS
        assert "physicalMeasurement" in rule.members

    def test_non_enumerable_vocab_falls_back_to_description(self) -> None:
        (rule,) = attribute_rules(_Attributes)["standard_name"]
        assert rule.members is None
        assert rule.description == _NonEnumerableVocab.description

    def test_rule_doc_for_non_vocab_validator_has_no_members(self) -> None:
        validator = is_exact("z", attribute="thing")
        # ``wrapped`` is pydantic's descriptor-proxy internal (the classmethod);
        # ``__func__`` recovers the raw function ``rule_doc`` consumes, mirroring
        # the ``cls.__dict__[name].__func__`` path the walker uses on a built
        # model. It is not part of the ``Validator`` protocol surface.
        func = validator.wrapped.__func__  # type: ignore[attr-defined]
        doc = rule_doc(func)
        assert isinstance(doc, RuleDoc)
        assert doc.members is None


# ---------------------------------------------------------------------------
# Integration: rules surface on the project walker output
# ---------------------------------------------------------------------------


class TestProjectWalkerRules:
    def _doc_attr(self, name: str):
        class Dataset(BaseModel):
            attributes: _Attributes

        doc = document_project(Dataset)
        return next(a for a in doc.dataset.attributes if a.name == name)

    def test_rules_attached_to_attribute_node(self) -> None:
        coverage = self._doc_attr("coverage")
        assert coverage.rules is not None
        assert coverage.rules[0].members == CoverageContentTypes.MEMBERS

    def test_attribute_without_rules_has_none(self) -> None:
        assert self._doc_attr("plain").rules is None


class TestProjectWalkerModelRules:
    def test_model_rules_attached_to_dataset_node(self) -> None:
        class Dataset(BaseModel):
            attributes: _Attributes

            _v_temp_exists = variable_exists("temperature")

        doc = document_project(Dataset)
        assert doc.dataset.rules is not None
        assert any("temperature" in r.description for r in doc.dataset.rules)

    def test_dataset_without_model_rules_has_none(self) -> None:
        class Dataset(BaseModel):
            attributes: _Attributes

        assert document_project(Dataset).dataset.rules is None
