from typing import Iterable, Literal

from .checks import Check, CheckError
from .errors import ElementDoesNotExist
from .status import ElementStatus


def get_element(name: str, container: Iterable) -> dict:
    """
    Return an element from an iterable container, using the
    name element of the container meta.

    Args:
        name: the name (variable/group) to find
        container: an iterable yielding variables or groups

    Returns:
        a dict representation of the requested variable.

    Raises:
        ElementDoesNotExist if the variable is not found in the parent
    """
    for i in container:
        if i["meta"]["name"] == name:
            return i

    raise ElementDoesNotExist(f"Element {name} not found")


def check_element_exists(
    name: str,
    parent: Iterable,
    path: str = "",
    from_file: bool = False,
    required: bool = True,
    element_type: Literal["variable", "group"] = "variable",
) -> tuple[list[Check], ElementStatus]:
    """
    Check an element (variable or group) exists in a parent, which is assumed to
    be an iterable yielding a dict representation of the element.

    Args:
        name: the name of the element to check
        container: an iterable yielding dict element representations
        from_file: if True, checking element from file is in definition,
                   if False, checking element from definition is in file.

    Kwargs:
        path: the full path of the element in the netCDF

    Returns:
        A tuple containing a list of Check objects and an ElementStatus enum value
    """

    checks: list[Check] = []

    in_type = "in definition" if from_file else "in file"

    check = Check(description=f"Checking {element_type} {path} exists {in_type}")
    checks.append(check)

    try:
        get_element(name, parent)
    except ElementDoesNotExist:
        if not required:
            return checks, ElementStatus.DOES_NOT_EXIST_AND_NOT_REQUIRED
        check.error = CheckError(
            f"{element_type.capitalize()} does not exist {in_type}", path
        )
        return checks, ElementStatus.DOES_NOT_EXIST_AND_REQUIRED

    return checks, ElementStatus.EXISTS
