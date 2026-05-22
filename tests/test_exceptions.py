"""Tests for the VocalError base class."""

import pytest

from vocal.exceptions import VocalError


class TestVocalErrorFormatting:
    def test_str_with_hint(self) -> None:
        e = VocalError("the thing broke", hint="try X")
        assert str(e) == "the thing broke\n  try X"

    def test_str_without_hint(self) -> None:
        e = VocalError("the thing broke")
        assert str(e) == "the thing broke"

    def test_message_attribute(self) -> None:
        e = VocalError("msg", hint="hint")
        assert e.message == "msg"

    def test_hint_attribute_optional(self) -> None:
        e = VocalError("msg")
        assert e.hint is None

    def test_default_status_code(self) -> None:
        e = VocalError("msg")
        assert e.status_code == 422


class TestVocalErrorSubclassing:
    def test_subclass_inherits_status_code(self) -> None:
        class MyError(VocalError):
            pass

        e = MyError("msg")
        assert e.status_code == 422

    def test_subclass_overrides_status_code(self) -> None:
        class MyError(VocalError):
            status_code = 400

        e = MyError("msg")
        assert e.status_code == 400

    def test_can_be_caught_as_vocal_error(self) -> None:
        class MyError(VocalError):
            pass

        with pytest.raises(VocalError):
            raise MyError("msg")
