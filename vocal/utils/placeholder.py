import re

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError


class InvalidPlaceholder(Exception):
    """
    Raised when an invalid placeholder is used in a check
    """


PLACEHOLDER_RE = (
    r"<(?P<container>Array)?"
    r"\[?(?P<dtype>[a-z0-9]+)\]?"
    r": derived_from_file"
    r"\s?"
    r"(?P<additional>.*)>"
)


@dataclass
class ConstraintResult:
    description: str
    error: str | None = None


class PlaceholderConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    regex: str | None = None
    min_len: int | None = Field(default=None, ge=0)
    max_len: int | None = Field(default=None, ge=1)

    def _check_regex(self, value: Any) -> list[ConstraintResult]:
        if self.regex is None:
            return []

        check = ConstraintResult(description=f"Value matches regex {self.regex}")

        # value may be a scalar string or a list of strings (a string array).
        values = value if isinstance(value, list) else [value]
        for item in values:
            # A non-string value is a dtype error, reported by the type check;
            # skip it here rather than crash on re.fullmatch.
            if not isinstance(item, str):
                continue
            if not re.fullmatch(self.regex, item):
                check.error = (
                    f"Value does not match expected format. Expected to match "
                    f'regex {self.regex}, got "{item}"'
                )
                break

        return [check]

    def _check_min_len(self, value: Any) -> list[ConstraintResult]:
        if self.min_len is None:
            return []

        check = ConstraintResult(
            description=f"Value length is at least {self.min_len}"
        )

        # min_len constrains len(value): character count for a string, element
        # count for an array. A non-sized value is a dtype error, reported by
        # the type check; skip it here rather than crash on len().
        if isinstance(value, (str, list)) and len(value) < self.min_len:
            check.error = (
                f"Value length {len(value)} is less than minimum {self.min_len}"
            )

        return [check]

    def _check_max_len(self, value: Any) -> list[ConstraintResult]:
        if self.max_len is None:
            return []

        check = ConstraintResult(
            description=f"Value length is at most {self.max_len}"
        )

        # As with min_len, this constrains len(value); a non-sized value is a
        # dtype error reported by the type check, so skip it here.
        if isinstance(value, (str, list)) and len(value) > self.max_len:
            check.error = (
                f"Value length {len(value)} exceeds maximum {self.max_len}"
            )

        return [check]

    def check(self, value: Any) -> list[ConstraintResult]:
        checks: list[ConstraintResult] = []
        checks += self._check_regex(value)
        checks += self._check_min_len(value)
        checks += self._check_max_len(value)

        return checks

    @model_validator(mode="after")
    def check_min_max_len(self) -> "PlaceholderConstraints":
        if self.min_len is not None and self.max_len is not None:
            if self.min_len > self.max_len:
                raise ValueError(
                    f"min_len {self.min_len} cannot be greater than max_len {self.max_len}"
                )
        return self


@dataclass(frozen=True)
class Placeholder:
    dtype: np.dtype
    is_array: bool = False
    optional: bool = False
    constraints: PlaceholderConstraints = field(default_factory=PlaceholderConstraints)

    @staticmethod
    def parse(placeholder_str: str) -> "Placeholder":
        matches = re.search(PLACEHOLDER_RE, placeholder_str)
        if not matches:
            raise InvalidPlaceholder(f"Invalid placeholder: {placeholder_str}")

        dtype = np.dtype(matches["dtype"])
        is_array = matches["container"] == "Array"
        optional, constraints = _parse_additional(
            matches["additional"], placeholder_str
        )

        # A regex only makes sense for string-valued attributes (including
        # string arrays). Reject it at parse time so the error points at the
        # definition, not the file.
        if constraints.regex is not None and dtype.kind not in ("U", "S"):
            raise InvalidPlaceholder(
                f"regex constraint requires a string dtype, got '{dtype}' "
                f"in: {placeholder_str}"
            )

        # Length constraints (min_len/max_len) constrain len(value), so they
        # only make sense for sized attributes: strings or arrays.
        has_length_constraint = (
            constraints.min_len is not None or constraints.max_len is not None
        )
        if has_length_constraint and not (is_array or dtype.kind in ("U", "S")):
            raise InvalidPlaceholder(
                f"min_len/max_len constraints require a string or array "
                f"attribute, got '{dtype}' in: {placeholder_str}"
            )

        return Placeholder(
            dtype=dtype,
            is_array=is_array,
            optional=optional,
            constraints=constraints,
        )


def _parse_additional(
    additional: str, placeholder_str: str
) -> tuple[bool, PlaceholderConstraints]:
    """
    Parse the comma-separated attribute list that follows ``derived_from_file``.

    The list is a sequence of flags (e.g. ``optional``) and ``key=value`` pairs
    (e.g. ``min_len=2``), separated by commas with no spaces. ``regex=`` must
    come last: its value runs to the end of the string, so it may itself
    contain commas (e.g. ``regex=[A-Z]{2,5}``).
    """
    optional = False

    constraints: dict[str, Any] = {}

    # regex, if present, is the final attribute and consumes the rest of the
    # string, so split it off before tokenising the remaining attributes.
    if "regex=" in additional:
        additional, regex = additional.split("regex=", 1)
        additional = additional.rstrip(",")
        constraints["regex"] = regex

    for token in filter(None, additional.split(",")):
        match token.split("=", 1):
            case ["optional"]:
                optional = True
            case ["min_len", value]:
                # pydantic coerces the str to int and validates ge=0
                constraints["min_len"] = value
            case ["max_len", value]:
                # pydantic coerces the str to int and validates ge=1
                constraints["max_len"] = value
            case _:
                raise InvalidPlaceholder(
                    f"Unknown placeholder attribute '{token}' in: {placeholder_str}"
                )

    try:
        constraints_obj = PlaceholderConstraints(**constraints)
    except ValidationError as err:
        raise InvalidPlaceholder(
            f"Invalid placeholder constraints in: {placeholder_str}. Error: {err}"
        ) from err

    return optional, constraints_obj
