from dataclasses import dataclass
import re
from typing import Any, Literal, Optional

import numpy as np


class InvalidPlaceholder(Exception):
    """
    Raised when an invalid placeholder is used in a check
    """


@dataclass
class AttributeProperties:
    optional: bool = False
    regex: Optional[str] = None


PLACEHOLDER_RE = (
    r"<(?P<container>Array)?"
    r"\[?(?P<dtype>[a-z0-9]+)\]?"
    r": derived_from_file"
    r"\s?"
    r"(?P<additional>.*)>"
)


@dataclass(frozen=True)
class Placeholder:
    dtype: np.dtype
    is_array: bool = False
    optional: bool = False
    regex: Optional[str] = None

    @staticmethod
    def parse(placeholder_str: str) -> "Placeholder":
        rex = re.compile(PLACEHOLDER_RE)
        matches = rex.search(placeholder_str)
        if not matches:
            raise InvalidPlaceholder(f"Invalid placeholder: {placeholder_str}")

        dtype = matches["dtype"]
        is_array = matches["container"] == "Array"

        additional = matches["additional"]
        additional_rex = re.compile("(?P<optional>optional)?,?((regex=)(?P<regex>.+))?")
        additional_matches = additional_rex.search(additional)
        if not additional_matches:
            raise InvalidPlaceholder(f"Invalid placeholder: {placeholder_str}")

        optional = additional_matches["optional"] == "optional"
        regex = additional_matches["regex"]

        return Placeholder(
            dtype=np.dtype(dtype), is_array=is_array, optional=optional, regex=regex
        )


def get_type_from_placeholder(
    placeholder: str,
) -> tuple[np.dtype[Any], Literal["Array"] | None]:
    """
    Returns the type from a placeholder string.

    Args:
        placeholder: the placeholder string

    Returns:
        A tuple of the type, and whether it is an array (as "Array" or None).
    """

    placeholder_info = Placeholder.parse(placeholder)
    return (placeholder_info.dtype, "Array" if placeholder_info.is_array else None)


def get_attribute_props_from_placeholder(placeholder: str) -> AttributeProperties:
    """
    Returns additional attributes from a placeholder string.

    Args:
        placeholder: the placeholder string

    Returns:
        Additional placeholder info, in the form of an AttributeProperties object.
    """

    placeholder_info = Placeholder.parse(placeholder)

    return AttributeProperties(
        optional=placeholder_info.optional, regex=placeholder_info.regex
    )
