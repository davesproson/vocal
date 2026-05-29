"""Pure-logic version and version-constraint types for vocal standards.

A standard's version is written as ``<name>-<major>.<minor>`` (e.g.
``MYSTD-2.5``). Within a single major, minor increments are non-breaking by
the maintainer's contract, so a constraint only ever carries a lower bound on
the minor: ``<name>-<major>.<min_minor>+`` (e.g. ``MYSTD-2.4+``) means "this
minor or any higher minor within the same major of the same standard." There
are no upper bounds and no range syntax — the non-breaking-minor invariant
makes a lower bound sufficient.

This module is pure logic: it does no filesystem, network, or netCDF access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vocal.exceptions import VocalError

_VERSION_RE = re.compile(r"^(?P<name>\S+?)-(?P<major>\d+)\.(?P<minor>\d+)$")
_CONSTRAINT_RE = re.compile(r"^(?P<name>\S+?)-(?P<major>\d+)\.(?P<min_minor>\d+)\+$")


class InvalidVersion(VocalError):
    """Raised when a version or constraint string cannot be parsed."""


@dataclass(frozen=True)
class Version:
    """A concrete standard version: ``<name>-<major>.<minor>``."""

    name: str
    major: int
    minor: int

    @classmethod
    def parse(cls, value: str) -> "Version":
        """Parse a canonical version string such as ``"MYSTD-2.5"``."""
        match = _VERSION_RE.match(value.strip())
        if not match:
            raise InvalidVersion(
                f"Invalid version string: {value!r}",
                "Expected '<name>-<major>.<minor>', e.g. 'MYSTD-2.5'.",
            )
        return cls(
            name=match["name"],
            major=int(match["major"]),
            minor=int(match["minor"]),
        )

    def __str__(self) -> str:
        return f"{self.name}-{self.major}.{self.minor}"


@dataclass(frozen=True)
class VersionConstraint:
    """A lower-bound constraint: ``<name>-<major>.<min_minor>+``.

    Satisfied by any :class:`Version` of the same name and major whose minor
    is greater than or equal to ``min_minor``.
    """

    name: str
    major: int
    min_minor: int

    @classmethod
    def parse(cls, value: str) -> "VersionConstraint":
        """Parse a canonical constraint string such as ``"MYSTD-2.4+"``."""
        match = _CONSTRAINT_RE.match(value.strip())
        if not match:
            raise InvalidVersion(
                f"Invalid version constraint: {value!r}",
                "Expected '<name>-<major>.<min_minor>+', e.g. 'MYSTD-2.4+'.",
            )
        return cls(
            name=match["name"],
            major=int(match["major"]),
            min_minor=int(match["min_minor"]),
        )

    def satisfied_by(self, version: Version) -> bool:
        """Return whether ``version`` falls within this constraint."""
        return (
            version.name == self.name
            and version.major == self.major
            and version.minor >= self.min_minor
        )

    def __str__(self) -> str:
        return f"{self.name}-{self.major}.{self.min_minor}+"
