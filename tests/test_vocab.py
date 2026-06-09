"""Tests for the Vocabulary implementations' autodoc-facing API.

Covers the ``description`` prose and ``members()`` enumeration/fallback added so
autodoc can document controlled vocabularies. ``CFStandardNames`` is exercised
without network access by stubbing its loader.
"""

import pytest

from vocal.vocab import CFStandardNames, CoverageContentTypes, ListVocabulary


class TestListVocabulary:
    def test_members_enumerates_items(self) -> None:
        vocab = ListVocabulary("v", ["a", "b"])
        assert vocab.members() == ["a", "b"]

    def test_members_is_a_copy(self) -> None:
        items = ["a", "b"]
        vocab = ListVocabulary("v", items)
        vocab.members().append("c")
        assert items == ["a", "b"]

    def test_description_defaults_to_name(self) -> None:
        assert ListVocabulary("v", []).description == "v"

    def test_explicit_description_distinct_from_label(self) -> None:
        vocab = ListVocabulary("v", [], description="The long prose form.")
        assert vocab.description == "The long prose form."
        assert str(vocab) == "v"


class TestCoverageContentTypes:
    def test_members_lists_the_controlled_terms(self) -> None:
        vocab = CoverageContentTypes()
        assert vocab.members() == CoverageContentTypes.MEMBERS
        assert "physicalMeasurement" in vocab.members()

    def test_contains(self) -> None:
        vocab = CoverageContentTypes()
        assert "image" in vocab
        assert "nonsense" not in vocab

    def test_has_description_prose(self) -> None:
        assert CoverageContentTypes().description


class TestCFStandardNames:
    def test_members_is_none_not_enumerable(self, monkeypatch) -> None:
        monkeypatch.setattr(CFStandardNames, "_load", lambda self: None)
        vocab = CFStandardNames()
        assert vocab.members() is None

    def test_description_distinct_from_label(self, monkeypatch) -> None:
        monkeypatch.setattr(CFStandardNames, "_load", lambda self: None)
        vocab = CFStandardNames()
        assert vocab.description != str(vocab)
        assert vocab.description
