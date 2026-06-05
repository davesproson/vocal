from dataclasses import dataclass, field
from typing import Iterable, Any

import numpy as np

from .status import ElementStatus
from .utils import check_element_exists, get_element
from .checks import Check, CheckError, CheckWarning, CheckComment
from vocal.types.schema_types import UnknownDataType, type_from_spec


from vocal.utils.placeholder import Placeholder


def compare_groups(d: Iterable, f: Iterable, path: str = "") -> list[Check]:
    """
    Compare the dict representation of groups from a product specification
    and from file

    Args:
        d: The group representation from the specification
        f: The group representation from file

    Kwargs:
        path: The path to the group container

    Returns:
        list[Check]: A list of Check objects representing the results of the comparison
    """
    checks: list[Check] = []

    for def_group in d:
        group_name = def_group["meta"]["name"]
        group_path = f"{path}/{group_name}"
        group_required = def_group["meta"].get("required", True)

        _checks, group_stat = check_element_exists(
            group_name,
            f,
            path=group_path,
            required=group_required,
            element_type="group",
        )
        checks += _checks

        if group_stat in (
            ElementStatus.DOES_NOT_EXIST_AND_REQUIRED,
            ElementStatus.DOES_NOT_EXIST_AND_NOT_REQUIRED,
        ):
            continue

        f_group = get_element(group_name, f)

        checks += compare_container(def_group, f_group, path=group_path)

    for file_group in f:
        group_name = file_group["meta"]["name"]
        group_path = f"{path}/{group_name}"

        _checks, _ = check_element_exists(
            group_name, d, path=group_path, from_file=True, element_type="group"
        )
        checks += _checks

    return checks


def compare_container(d: dict, f: dict, path: str = "") -> list[Check]:
    """
    Compare the dict representation of a netcdf container from a product
    specification and from file

    Args:
        d: The container representation from a product specification
        f: The container representation from file

    Kwargs:
        path: the path of the container in the netcdf file

    Returns:
        list[Check]: A list of Check objects representing the results of the comparison
    """
    checks: list[Check] = []

    checks += compare_dimensions(d, f, path=path)

    checks += compare_attributes(
        d.get("attributes", {}), f.get("attributes", {}), path=path
    )
    checks += compare_variables(
        d.get("variables", []), f.get("variables", []), path=path
    )

    checks += compare_groups(d.get("groups", []), f.get("groups", []), path=path)

    return checks


@dataclass
class DimensionCollector:
    dimensions: list[dict] = field(default_factory=list)

    def search(self, container: dict, depth: int = 99) -> list[dict]:

        if depth == 0:
            return self.dimensions

        for dim in container.get("dimensions", []):
            self.dimensions.append(dim)

        for group in container.get("groups", []):
            self.search(group, depth=depth - 1)

        return self.dimensions


def compare_dimensions(d: dict, f: dict, path: str = "") -> list[Check]:
    """
    Compare the dict representation of dimensions from a product
    specification and from file.

    We use the DimensionCollector to find the dimensions in the file up
    to the depth of the current path.

    Args:
        d: The dimension representation from the specification
        f: The dimension representation from file

    Kwargs:
        path: The path to the dimension container

    Returns:
        list[Check]: A list of Check objects representing the results of the comparison
    """
    depth = path.count("/") + 1
    def_dims = DimensionCollector().search(d, depth=depth)
    file_dims = DimensionCollector().search(f, depth=depth)

    checks: list[Check] = []

    for dim in file_dims:
        _path = f"{path}/[{dim['name']}]"
        check = Check(description=f"Checking dimension {dim['name']} is in definition")
        checks.append(check)

        # The dimension is in the definition. Checking equality here, which
        # encompasses both size and name.
        if dim in def_dims:
            continue

        # The dimension is not in the definition. Checking if there is a
        # dimension with the same name, but different size.
        dims_with_name = [d for d in def_dims if d["name"] == dim["name"]]
        if dims_with_name:
            # There is a dimension with the same name, but different size.
            message = (
                f"Dimension {dim['name']} found in definition, but "
                f"with different size. Size in file: {dim['size']}, "
                f"size in definition: {dims_with_name[0]['size']}"
            )
        else:
            # There is no dimension with the same name.
            message = f"Dimension {dim['name']} not found in definition"

        check.error = CheckError(message=message, path=_path)

    for dim in def_dims:
        _path = f"{path}/[{dim['name']}]"

        check = Check(description=f"Checking dimension {dim['name']} is in file")
        checks.append(check)

        if dim in file_dims:
            continue

        # The dimension is not in the definition. Checking if there is a
        # dimension with the same name, but different size.
        dims_with_name = [d for d in file_dims if d["name"] == dim["name"]]
        if dims_with_name:
            # There is a dimension with the same name, but different size.
            message = (
                f"Dimension {dim['name']} found in file, but "
                f"with different size. Size in definition: {dim['size']}, "
                f"size in file: {dims_with_name[0]['size']}"
            )
            check.error = CheckError(message=message, path=_path)
        else:
            # There is no dimension with the same name.
            message = f"Dimension {dim['name']} not found in file"

            check.warning = CheckWarning(message=message, path=_path)

    return checks


def compare_variables(d: dict, f: dict, path: str = "") -> list[Check]:
    """
    Compare all of the variables in a container to those in a product
    specification.

    Args:
        d: a dict representation of the container from the specification
        f: a dict representation of the container from the netcdf file

    Kwargs:
        path: the full path to the container in the netcdf file

    Returns:
        list[Check]: A list of Check objects representing the results of the comparison
    """
    checks: list[Check] = []

    for d_var in d:
        var_name = d_var["meta"]["name"]
        var_required = d_var["meta"].get("required", True)
        var_path = f"{path}/{var_name}"

        _checks, variable_stat = check_element_exists(
            var_name,
            f,
            path=var_path,
            required=var_required,
            element_type="variable",
        )
        checks += _checks

        if variable_stat in (
            ElementStatus.DOES_NOT_EXIST_AND_REQUIRED,
            ElementStatus.DOES_NOT_EXIST_AND_NOT_REQUIRED,
        ):
            continue

        f_var = get_element(var_name, f)
        checks += check_variable_dtype(d_var, f_var, path=var_path)

        checks += compare_attributes(
            d_var["attributes"], f_var["attributes"], path=var_path
        )

    for f_var in f:
        var_name = f_var["meta"]["name"]
        var_path = f"{path}/{var_name}"

        _checks, _ = check_element_exists(
            var_name, d, path=var_path, from_file=True, element_type="variable"
        )

        checks += _checks

    return checks


def check_variable_dtype(d: dict, f: dict, path: str = "") -> list[Check]:
    """
    Check the datatype of a variable matches that given in the product
    definition

    Args:
        d: a dict representation of the variable from the specification
        f: a dict representation of the variable from the netCDF

    Kwargs:
        path: the full path to the variable in the netCDF

    Returns:
        A list of Check objects representing the results of the datatype check
    """
    checks: list[Check] = []

    check = Check(description=f"Checking datatype of {path}")
    checks.append(check)

    expected_type_str = d["meta"]["datatype"]
    actual_type_str = f["meta"]["datatype"]

    actual_dtype = type_from_spec(actual_type_str)

    # Check that the expected datatype is known to vocal
    try:
        expected_dtype = type_from_spec(expected_type_str)
    except UnknownDataType:
        check.error = CheckError(
            f"Unknown datatype in specification: {expected_type_str}", path
        )
        return checks

    if actual_dtype != expected_dtype:
        check.error = CheckError(
            f"Incorrect datatype. Found {actual_dtype}, expected {expected_dtype}",
            path,
        )

    if actual_type_str != expected_type_str:
        check.comment = CheckComment(
            (
                f"Found datatype {actual_type_str}. Specification denotes expected "
                f"datatype as {expected_type_str}, but these are considered equivalent."
            ),
            path,
        )

    return checks


def compare_attributes(d: dict, f: dict, path: str = "") -> list[Check]:
    """
    Compare the attributes in a netCDF container against the product
    definition

    Args:
        d: a dict representation of the container from the specification
        f: a dict representation of the container from the netcdf file

    Kwargs:
        path: the full path of the container

    Returns:
        list[Check]: A list of Check objects representing the results of the comparison
    """
    checks = []

    if not path:
        path = "/"

    for def_key, def_value in d.items():
        check = Check(description=f"Checking attribute {path}.{def_key} exists")
        checks.append(check)

        if def_key not in f:
            if isinstance(def_value, str) and def_value.startswith("<"):
                placeholder = Placeholder.parse(def_value)
                if placeholder.optional:
                    continue

            check.error = CheckError(
                message=f"Attribute .{def_key} not in {path}",
                path=f"{path}.{def_key}",
            )
            continue

        checks += check_attribute_value(
            d[def_key], f[def_key], path=f"{path}.{def_key}"
        )

    for file_key in f:
        check = Check(description=f"Checking attribute {path}.{file_key} in definition")
        checks.append(check)

        if file_key not in d:
            check.warning = CheckWarning(
                message=f"Found attribute .{file_key} which is not in definition",
                path=f"{path}.{file_key}",
            )

    return checks


def check_attribute_type(
    placeholder: Placeholder, f: Any, path: str = ""
) -> list[Check]:
    """
    Checks the type of an attribute is correct, given a Placeholder
    in the product definition file.

    Args:
        placeholder: the Placeholder object parsed from the product definition
        f: the attribute in the netcdf file

    Kwargs:
        path: full path of the attribute in the netCDF

    Returns:
        list[Check]: A list of Check objects representing the results of the comparison
    """

    checks: list[Check] = []

    check = Check(description=f"Checking attribute {path} type is correct")
    checks.append(check)

    if placeholder.dtype == type(f):
        return checks

    if placeholder.is_array and type(f) is list:
        if all([type(i) == placeholder.dtype for i in f]):
            return checks

    check.error = CheckError(
        message=f"Type of {path} incorrect. Expected {placeholder.dtype}, got {type(f)}",
        path=path,
    )

    return checks


def check_attribute_against_placeholder(d: Any, f: Any, path: str = "") -> list[Check]:
    """
    Checks the value of an attribute against a regex, where the product definition
    specifies a placeholder with a regex.

    Args:
        d: the attribute in the product definition
        f: the attribute in the netcdf file

    Kwargs:
        path: full path of the attribute in the netCDF
    """
    placeholder = Placeholder.parse(d)
    checks = check_attribute_type(placeholder, f, path=path)

    for constraint_result in placeholder.constraints.check(f):
        check = Check(
            description=f"Checking attribute {path}: {constraint_result.description}"
        )
        checks.append(check)

        if constraint_result.error:
            check.error = CheckError(message=constraint_result.error, path=path)

    return checks


def check_attribute_value(d: Any, f: Any, path: str = "") -> list[Check]:
    """
    Checks the value of an attribute, where it is specified in the
    product definition.

    Args:
        d: the attribute in the product definition
        f: the attribute in the netcdf file

    Kwargs:
        path: full path of the attribute in the netCDF

    Returns:
        None
    """
    checks: list[Check] = []

    check = Check(description=f"Checking value of {path}")
    checks.append(check)

    # If the attribute is a placeholder, all we can do is check the type
    if isinstance(d, str) and "derived_from_file" in d:
        checks += check_attribute_against_placeholder(d, f, path=path)
        return checks

    f_array = np.atleast_1d(f)
    d_array = np.atleast_1d(d)

    if len(f_array) != len(d_array):
        check.error = CheckError(
            message=f"Unexpected value of {path} incorrect. Got {f}, expected {d}",
            path=path,
        )
        return checks

    for f_val, d_val in zip(f_array, d_array):
        if f_val == d_val:
            continue

        if isinstance(d_val, str) and "derived_from_file" in d_val:
            checks += check_attribute_against_placeholder(d_val, f_val, path=path)
            continue

        check.error = CheckError(
            message=f"Unexpected value of {path}. Got {f}, expected: {d}",
            path=path,
        )

    return checks
