from dataclasses import dataclass
import enum
import random

from string import ascii_lowercase, ascii_uppercase
from typing import Any, Callable, Collection, Protocol, cast
from pydantic import model_validator, field_validator

from vocal.vocab import Vocabulary


@dataclass
class Attribute:
    name: str


class Model(enum.Enum):
    before = "before"
    after = "after"


Binding = Attribute | Model


class Validator[I](Protocol):
    """
    A validator is a mask for a function which can be annotated with attributes
    in a type-safe way.
    """

    description: str
    binding: Binding | None

    @staticmethod
    def __call__(value: I) -> I: ...


def _bind_validator(binding: Binding | None) -> Callable:
    """
    Return the appropriate validator function based on the binding

    Args:
        binding: The binding to use

    Returns:
        The appropriate validator function
    """
    if binding is None:
        return lambda x: x

    if isinstance(binding, Model):
        return model_validator(mode=binding.value)

    if isinstance(binding, Attribute):
        return field_validator(binding.name)

    raise ValueError(f"Unknown binding: {binding}")


def vocal_validator(description: str = "", bound: Binding | None = None):
    """
    A decorator for attaching metadata to a validator function.
    """

    def inner[I](func: Callable[[Any, I], I]) -> Validator[I]:
        _func = cast(Validator[I], func)
        _func.description = description
        _func.binding = bound
        return _bind_validator(bound)(_func)

    return inner


def validate[I](binding: Binding, validator: Validator[I]) -> Validator[I]:
    """
    Bind a validator to a specific binding

    Args:
        binding: The binding to use
        validator: The validator to bind

    Returns:
        The validator bound to the given binding
    """
    validator.binding = binding
    return _bind_validator(binding)(validator)


def _randomize_object_name[I: Callable](obj: I) -> I:
    """
    Randomize the name of an object (__name__), to work around a
    bug-or-odd-feature in pydantic

    Args:
        obj: the object to rename

    Returns:
        obj with a random __name__
    """
    _random_str = "".join(random.sample(ascii_lowercase + ascii_uppercase, 12))
    obj.__name__ = _random_str
    return obj


def default_value[I](default: I) -> Validator[I]:
    """
    This has been replaced by is_exact
    """
    return is_exact(default)


def is_exact[I](default: I) -> Validator[I]:
    """
    Provides a validator which ensures an attribute takes a given default
    value

    Args:
        default: The default value to check against

    Returns:
        A validator function
    """

    @vocal_validator(description=f"Value must be exactly '{default}'")
    def _validator(cls, value):
        if value != default:
            raise ValueError(f'text is incorrect. Got: "{value}", expected "{default}"')
        return value

    return _randomize_object_name(_validator)


def is_in(collection: Collection) -> Validator:
    """
    Provides a validator which ensures an attribute takes a value in a
    given collection

    Args:
        collection: The collection of allowed values

    Returns:
        A validator function
    """

    @vocal_validator(description=f"Value must be in {collection}")
    def _validator(cls, value):
        if value not in collection:
            raise ValueError(f"Value should be in {collection}")
        return value

    return _randomize_object_name(_validator)


def variable_exists(variable_name: str) -> Validator:
    """
    Provides a validator which ensures a variable exists in a given
    group.

    Args:
        variable_name: The name of the variable to check

    Returns:
        A validator function
    """

    @vocal_validator(
        description=f"Variable '{variable_name}' must exist in group", bound=Model.after
    )
    def _validator(cls, values):
        try:
            variables = values["variables"]
        except Exception:
            variables = []

        name = getattr(values.meta, "name", "root")
        if variables is None:
            raise ValueError(f"Variable '{variable_name}' not found in {name}")

        for var in variables:
            if var.meta.name == variable_name:
                return values
        raise ValueError(f"Variable '{variable_name}' not found in {name}")

    return _randomize_object_name(_validator)


def variable_has_types(variable_name: str, allowed_types: list[str]) -> Callable:
    """
    Provides a validator which ensures a variable is of a given type(s)

    Args:
        variable_name: The name of the variable to check
        allowed_types: A list of allowed types for the variable

    Returns:
        A validator function
    """

    @vocal_validator(
        description=f"Variable '{variable_name}' must be one of {allowed_types}"
    )
    def _validator(cls, values):
        variables = values.variables
        if variables is None:
            return values
        for var in variables:
            var_name = var.meta.name
            var_type = var.meta.datatype
            if var_name != variable_name:
                continue
            if var_type not in allowed_types:
                raise ValueError(
                    f'Expected datatype of variable "{variable_name}" to be '
                    f"one of [{','.join(allowed_types)}], got {var_type}"
                )
        return values

    return _randomize_object_name(_validator)


def variable_has_dimensions(variable_name: str, dimensions: list[str]) -> Callable:
    """
    Provides a validator which ensures a variable has the given dimensions

    Args:
        variable_name: The name of the variable to check
        dimensions: A list of dimensions the variable should have

    Returns:
        A validator function
    """

    @vocal_validator(
        description=f"Variable '{variable_name}' must have dimensions {dimensions}"
    )
    def _validator(cls, values):
        variables = values.variables
        if variables is None:
            return values
        for var in variables:
            if var.meta.name != variable_name:
                continue

            var_dims = var.dimensions
            for dim in dimensions:
                if dim not in var_dims:
                    raise ValueError(
                        f'Expected variable "{variable_name}" to have dimension "{dim}"'
                    )

            for dim in var_dims:
                if dim not in dimensions:
                    raise ValueError(
                        f'Variable "{variable_name}" has unexpected dimension {dim}'
                    )
        return values

    return _randomize_object_name(_validator)


def group_exists(group_name: str) -> Callable:
    """
    Provides a validator which ensures a group exists in a given
    supergroup

    Args:
        group_name: The name of the group to check

    Returns:
        A validator function
    """

    @vocal_validator(description=f"Group '{group_name}' must exist in supergroup")
    def _validator(cls, values):
        try:
            groups = values.groups
        except Exception:
            groups = []
        name = values.meta.name
        if groups is None:
            raise ValueError(f"Group '{group_name}' not found in {name}")

        for group in groups:
            if group.meta.name == group_name:
                return values
        raise ValueError(f"Group '{group_name}' not found in {name}")

    return _randomize_object_name(_validator)


def dimension_exists(dimension_name: str) -> Callable:
    """
    Provides a validator which ensures a dimension exists in a given
    group

    Args:
        dimension_name: The name of the dimension to check

    Returns:
        A validator function
    """

    @vocal_validator(description=f"Dimension '{dimension_name}' must exist in group")
    def _validator(cls, values):
        dimensions = values.dimensions
        name = values.meta.name
        if dimensions is None:
            raise ValueError(f"Dimension '{dimension_name}' not found in {name}")

        for dim in dimensions:
            if dim.name == dimension_name:
                return values
        raise ValueError(f"Dimension '{dimension_name}' not found in {name}")

    return _randomize_object_name(_validator)


def in_vocabulary(vocabulary: Vocabulary) -> Validator[str]:
    """
    Provides a validator which ensures an attribute takes a value in a
    given vocabulary

    Args:
        vocabulary: The vocabulary of allowed values

    Returns:
        A validator function
    """

    @vocal_validator(description=f"Value must be in {vocabulary}")
    def _validator(cls: Any, value: str) -> str:
        if value not in vocabulary:
            raise ValueError(f"'{value}' not in {vocabulary}")
        return value

    return _randomize_object_name(_validator)


# These were more customised for pydantic v1. They're mostly passthroughs now.
substitutor = model_validator(mode="before")
validator = model_validator(mode="after")


def substitute_placeholders(cls, values: dict) -> dict:
    """
    A root validator, which should be called with pre=True, which turns
    attributes with placeholders (e.g. attr: <str: derived_from_file>)
    into valid values, by substituting them with the example from the attribute
    definition.

    Args:
        cls: The class being validated
        values: The values being validated

    Returns:
        The cls with the placeholders substituted for example values
    """
    DERIVED = "derived_from_file"

    for key, value in values.items():
        if not isinstance(value, (str, list)):
            continue

        try:
            example = cls.model_json_schema()["properties"][key]["example"]
        except KeyError:
            continue

        if DERIVED in value:
            values[key] = example

        # Traverse any lists and replace values
        if isinstance(value, list):
            replaced = []
            for i, list_val in enumerate(value):
                if isinstance(list_val, str) and DERIVED in list_val:
                    replaced.append(example[i])
                else:
                    replaced.append(list_val)

            values[key] = replaced

    return values
