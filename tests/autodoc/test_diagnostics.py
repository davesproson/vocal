"""Unit tests for the documentation-gap diagnostics (autodoc slice 8).

Fixtures are tiny, hermetic, synthetic models, following the inline-model style
in ``tests/test_validation.py``. Tests assert the external behaviour — which
gaps are reported and that detection is non-fatal — rather than internals.
"""

from pydantic import BaseModel

from vocal.autodoc.diagnostics import (
    mixin_mismatch,
    record_mixin_mismatch,
    record_undescribed,
    undescribed_validator,
)
from vocal.autodoc.rules import iter_validators
from vocal.mixins import VocalAttributesMixin, VocalVariableMixin
from vocal.validation import Attribute, vocal_validator


@vocal_validator(description="", bound=Attribute("title"))
def _undescribed(cls, value):  # pragma: no cover - never invoked
    return value


@vocal_validator(description="A described rule", bound=Attribute("title"))
def _described(cls, value):  # pragma: no cover - never invoked
    return value


class _DescribedAttrs(BaseModel, VocalAttributesMixin):
    title: str
    _v = _described


class _UndescribedAttrs(BaseModel, VocalAttributesMixin):
    title: str
    _v = _undescribed


class _PlainAttrs(BaseModel):
    """An attributes model that forgot to mix in ``VocalAttributesMixin``."""

    title: str


class _WrongMixinAttrs(BaseModel, VocalVariableMixin):
    """An attributes field whose model carries the *variable* mixin instead."""

    title: str


class TestUndescribedValidator:
    def test_records_undescribed_validator(self) -> None:
        diagnostics: list[str] = []
        record_undescribed(_UndescribedAttrs, diagnostics)
        assert len(diagnostics) == 1
        assert "_UndescribedAttrs" in diagnostics[0]
        assert "empty description" in diagnostics[0]

    def test_described_validator_is_silent(self) -> None:
        diagnostics: list[str] = []
        record_undescribed(_DescribedAttrs, diagnostics)
        assert diagnostics == []

    def test_model_without_validators_is_silent(self) -> None:
        diagnostics: list[str] = []
        record_undescribed(_PlainAttrs, diagnostics)
        assert diagnostics == []

    def test_none_model_is_silent(self) -> None:
        diagnostics: list[str] = []
        record_undescribed(None, diagnostics)
        assert diagnostics == []

    def test_pure_detector_returns_message_then_none(self) -> None:
        (described,) = iter_validators(_DescribedAttrs)
        (undescribed,) = iter_validators(_UndescribedAttrs)
        assert undescribed_validator(_DescribedAttrs, described) is None
        assert undescribed_validator(_UndescribedAttrs, undescribed) is not None


class TestMixinMismatch:
    def test_matching_mixin_is_silent(self) -> None:
        assert mixin_mismatch("attributes", _DescribedAttrs) is None

    def test_missing_mixin_is_reported(self) -> None:
        message = mixin_mismatch("attributes", _PlainAttrs)
        assert message is not None
        assert "_PlainAttrs" in message
        assert "VocalAttributesMixin" in message

    def test_wrong_mixin_is_reported(self) -> None:
        message = mixin_mismatch("attributes", _WrongMixinAttrs)
        assert message is not None
        assert "VocalVariableMixin" in message
        assert "VocalAttributesMixin" in message

    def test_meta_field_has_no_expected_mixin(self) -> None:
        assert mixin_mismatch("meta", _PlainAttrs) is None

    def test_absent_field_model_is_silent(self) -> None:
        assert mixin_mismatch("attributes", None) is None

    def test_record_appends_only_on_mismatch(self) -> None:
        diagnostics: list[str] = []
        record_mixin_mismatch("attributes", _DescribedAttrs, diagnostics)
        assert diagnostics == []
        record_mixin_mismatch("attributes", _PlainAttrs, diagnostics)
        assert len(diagnostics) == 1
