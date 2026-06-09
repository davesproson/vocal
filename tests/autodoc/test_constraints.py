"""Unit tests for the constraint normaliser (autodoc slice 2).

The normaliser is a pure function from a field's JSON-schema fragment to a list
of typed ``ConstraintDoc``; these tests exercise each constraint kind and the
edge cases (Optional/union unwrapping, empty fragments, ordering).
"""

from vocal.autodoc import ConstraintDoc, normalize_constraints


class TestType:
    def test_type_constraint(self) -> None:
        assert normalize_constraints({"type": "string"}) == [
            ConstraintDoc(kind="type", detail={"type": "string"})
        ]


class TestPattern:
    def test_pattern_constraint(self) -> None:
        out = normalize_constraints({"type": "string", "pattern": "^a.*"})
        assert ConstraintDoc(kind="pattern", detail={"pattern": "^a.*"}) in out


class TestRange:
    def test_inclusive_bounds_collapse_to_range(self) -> None:
        out = normalize_constraints({"type": "integer", "minimum": 0, "maximum": 10})
        assert ConstraintDoc(kind="range", detail={"ge": 0, "le": 10}) in out

    def test_exclusive_bounds_collapse_to_range(self) -> None:
        out = normalize_constraints(
            {"type": "number", "exclusiveMinimum": 1.5, "exclusiveMaximum": 9.0}
        )
        assert ConstraintDoc(kind="range", detail={"gt": 1.5, "lt": 9.0}) in out

    def test_single_bound(self) -> None:
        out = normalize_constraints({"type": "integer", "minimum": 3})
        assert ConstraintDoc(kind="range", detail={"ge": 3}) in out


class TestLength:
    def test_length_constraint(self) -> None:
        out = normalize_constraints(
            {"type": "string", "minLength": 2, "maxLength": 5}
        )
        assert ConstraintDoc(
            kind="length", detail={"min_length": 2, "max_length": 5}
        ) in out


class TestEnum:
    def test_enum_constraint(self) -> None:
        out = normalize_constraints({"type": "string", "enum": ["a", "b", "c"]})
        assert ConstraintDoc(kind="enum", detail={"values": ["a", "b", "c"]}) in out


class TestEdgeCases:
    def test_empty_fragment_yields_nothing(self) -> None:
        assert normalize_constraints({}) == []

    def test_unknown_keys_ignored(self) -> None:
        assert normalize_constraints({"title": "X", "default": None}) == []

    def test_optional_unwraps_to_non_null_arm(self) -> None:
        # Optional[int] = Field(ge=3) -> anyOf with a null arm.
        fragment = {
            "anyOf": [{"type": "integer", "minimum": 3}, {"type": "null"}],
            "default": None,
        }
        assert normalize_constraints(fragment) == [
            ConstraintDoc(kind="type", detail={"type": "integer"}),
            ConstraintDoc(kind="range", detail={"ge": 3}),
        ]

    def test_union_does_not_duplicate_identical_constraints(self) -> None:
        fragment = {"anyOf": [{"type": "string"}, {"type": "string"}]}
        assert normalize_constraints(fragment) == [
            ConstraintDoc(kind="type", detail={"type": "string"})
        ]

    def test_ordering_is_stable(self) -> None:
        fragment = {
            "type": "string",
            "pattern": "^a",
            "minLength": 1,
            "enum": ["a", "ab"],
        }
        kinds = [c.kind for c in normalize_constraints(fragment)]
        assert kinds == ["type", "pattern", "length", "enum"]
