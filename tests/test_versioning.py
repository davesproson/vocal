import pytest

from vocal.versioning import InvalidVersion, Version, VersionConstraint


# ---------------------------------------------------------------------------
# Version.parse
# ---------------------------------------------------------------------------


class TestVersionParse:
    def test_parses_name_major_minor(self) -> None:
        v = Version.parse("MYSTD-2.5")
        assert v.name == "MYSTD"
        assert v.major == 2
        assert v.minor == 5

    def test_parses_multi_digit_components(self) -> None:
        v = Version.parse("MYSTD-12.34")
        assert v.major == 12
        assert v.minor == 34

    def test_parses_zero_components(self) -> None:
        v = Version.parse("MYSTD-0.0")
        assert v.major == 0
        assert v.minor == 0

    def test_strips_surrounding_whitespace(self) -> None:
        assert Version.parse("  MYSTD-2.5  ") == Version.parse("MYSTD-2.5")

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "MYSTD",
            "MYSTD-2",
            "MYSTD-2.",
            "MYSTD-.5",
            "MYSTD-2.5.1",
            "MYSTD-2.5+",
            "MYSTD-a.b",
            "2.5",
            "MYSTD 2.5",
        ],
    )
    def test_rejects_malformed_input(self, bad: str) -> None:
        with pytest.raises(InvalidVersion):
            Version.parse(bad)

    def test_round_trips_through_str(self) -> None:
        assert str(Version.parse("MYSTD-2.5")) == "MYSTD-2.5"


# ---------------------------------------------------------------------------
# VersionConstraint.parse
# ---------------------------------------------------------------------------


class TestVersionConstraintParse:
    def test_parses_name_major_min_minor(self) -> None:
        c = VersionConstraint.parse("MYSTD-2.4+")
        assert c.name == "MYSTD"
        assert c.major == 2
        assert c.min_minor == 4

    def test_strips_surrounding_whitespace(self) -> None:
        assert VersionConstraint.parse("  MYSTD-2.4+  ") == VersionConstraint.parse(
            "MYSTD-2.4+"
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "MYSTD-2.4",  # missing trailing '+'
            "MYSTD-2+",  # missing minor
            "MYSTD-2.4-",  # not a '+'
            "MYSTD-2.4++",  # double '+'
            "MYSTD-~2.4",  # no tilde ranges
            "MYSTD-2.4+ MYSTD-3.0+",
        ],
    )
    def test_rejects_malformed_input(self, bad: str) -> None:
        with pytest.raises(InvalidVersion):
            VersionConstraint.parse(bad)

    def test_round_trips_through_str(self) -> None:
        assert str(VersionConstraint.parse("MYSTD-2.4+")) == "MYSTD-2.4+"


# ---------------------------------------------------------------------------
# VersionConstraint.satisfied_by
# ---------------------------------------------------------------------------


class TestSatisfiedBy:
    def setup_method(self) -> None:
        self.constraint = VersionConstraint.parse("MYSTD-2.4+")

    def test_minor_above_floor_satisfies(self) -> None:
        assert self.constraint.satisfied_by(Version.parse("MYSTD-2.5"))

    def test_minor_at_floor_satisfies(self) -> None:
        assert self.constraint.satisfied_by(Version.parse("MYSTD-2.4"))

    def test_minor_below_floor_does_not_satisfy(self) -> None:
        assert not self.constraint.satisfied_by(Version.parse("MYSTD-2.3"))

    def test_major_mismatch_does_not_satisfy(self) -> None:
        assert not self.constraint.satisfied_by(Version.parse("MYSTD-3.4"))
        assert not self.constraint.satisfied_by(Version.parse("MYSTD-1.9"))

    def test_name_mismatch_does_not_satisfy(self) -> None:
        assert not self.constraint.satisfied_by(Version.parse("OTHER-2.5"))

    def test_name_match_is_case_sensitive(self) -> None:
        assert not self.constraint.satisfied_by(Version.parse("mystd-2.5"))
