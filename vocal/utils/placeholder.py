from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np


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


@dataclass(frozen=True)
class PlaceholderConstraints:
    regex: str | None = None


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

        dtype = matches["dtype"]
        is_array = matches["container"] == "Array"
        optional, constraints = _parse_additional(
            matches["additional"], placeholder_str
        )

        return Placeholder(
            dtype=np.dtype(dtype),
            is_array=is_array,
            optional=optional,
            constraints=constraints,
        )


def _parse_additional(
    additional: str, placeholder_str: str
) -> tuple[bool, PlaceholderConstraints]:
    """
    Parse the comma-separated attribute list that follows ``derived_from_file``.

    The list is a sequence of flags (e.g. ``optional``) and, in future,
    ``key=value`` pairs (e.g. ``min_len=2``), separated by commas with no
    spaces. ``regex=`` must come last: its value runs to the end of the string,
    so it may itself contain commas (e.g. ``regex=[A-Z]{2,5}``).
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
            # future key=value attributes (e.g. min_len) add a case here:
            #   case ["min_len", value]: ...
            case _:
                raise InvalidPlaceholder(
                    f"Unknown placeholder attribute '{token}' in: {placeholder_str}"
                )

    return optional, PlaceholderConstraints(**constraints)
