import re

from typing import Union

import numpy as np

Numeric = Union[float, int]


class UnknownDataType(Exception):
    """
    Raised when an unknown type is specified in a schema.
    """


def type_str(typ):
    return f"<{typ}>"


def derived_type(typ: str, optional: bool = False) -> str:
    opt_str = " optional" if optional else ""
    return f"<{typ}: derived_from_file{opt_str}>"


def derived_array(typ: str, optional: bool = False) -> str:
    opt_str = " optional" if optional else ""
    return f"<Array[{typ}]: derived_from_file{opt_str}>"


DerivedString = derived_type("str")
DerivedInteger8 = derived_type("int8")
DerivedInteger16 = derived_type("int16")
DerivedInteger32 = derived_type("int32")
DerivedInteger64 = derived_type("int64")
DerivedUInteger8 = derived_type("uint8")
DeriverUInteger16 = derived_type("uint16")
DerivedUInteger32 = derived_type("uint32")
DerivedUInteger64 = derived_type("uint64")
DerivedByte = derived_type("int8")
DerivedUByte = derived_type("uint8")
DerivedFloat16 = derived_type("float16")
DerivedFloat32 = derived_type("float32")
DerivedFloat64 = derived_type("float64")

OptionalDerivedString = derived_type("str", optional=True)
OptionalDerivedInteger8 = derived_type("int8", optional=True)
OptionalDerivedInteger16 = derived_type("int16", optional=True)
OptionalDerivedInteger32 = derived_type("int32", optional=True)
OptionalDerivedInteger64 = derived_type("int64", optional=True)
OptionalDerivedUInteger8 = derived_type("uint8", optional=True)
OptionalDerivedUInteger16 = derived_type("uint16", optional=True)
OptionalDerivedUInteger32 = derived_type("uint32", optional=True)
OptionalDerivedUInteger64 = derived_type("uint64", optional=True)
OptionalDerivedByte = derived_type("int8", optional=True)
OptionalDerivedUByte = derived_type("uint8", optional=True)
OptionalDerivedFloat16 = derived_type("float64", optional=True)
OptionalDerivedFloat32 = derived_type("float32", optional=True)
OptionalDerivedFloat64 = derived_type("float64", optional=True)

DerivedStringArray = derived_array("str")
DerivedInteger8Array = derived_array("int8")
DerivedInteger16Array = derived_array("int16")
DerivedInteger32Array = derived_array("int32")
DerivedInteger64Array = derived_array("int64")
DerivedUInteger8Array = derived_array("uint8")
DerivedUInteger16Array = derived_array("uint16")
DerivedUInteger32Array = derived_array("uint32")
DerivedUInteger64Array = derived_array("uint64")
DerivedByteArray = derived_array("int8")
DerivedFloat16Array = derived_array("float16")
DerivedFloat32Array = derived_array("float32")
DerivedFloat64Array = derived_array("float64")

OptionalDerivedStringArray = derived_array("str", optional=True)
OptionalDerivedInteger8Array = derived_array("int8", optional=True)
OptionalDerivedInteger16Array = derived_array("int16", optional=True)
OptionalDerivedInteger32Array = derived_array("int32", optional=True)
OptionalDerivedInteger64Array = derived_array("int64", optional=True)
OptionalDerivedUInteger8Array = derived_array("uint8", optional=True)
OptionalDerivedUInteger16Array = derived_array("uint16", optional=True)
OptionalDerivedUInteger32Array = derived_array("uint32", optional=True)
OptionalDerivedUInteger64Array = derived_array("uint64", optional=True)
OptionalDerivedByteArray = derived_array("int8", optional=True)
OptionalDerivedFloat16Array = derived_array("float16", optional=True)
OptionalDerivedFloat32Array = derived_array("float32", optional=True)
OptionalDerivedFloat64Array = derived_array("float64", optional=True)


Byte = type_str("int8")
UByte = type_str("uint8")
Integer8 = type_str("int8")
Integer16 = type_str("int16")
Integer32 = type_str("int32")
Integer64 = type_str("int64")
UInteger8 = type_str("uint8")
UInteger16 = type_str("uint16")
UInteger32 = type_str("uint32")
UInteger64 = type_str("uint64")
Float16 = type_str("float16")
Float32 = type_str("float32")
Float64 = type_str("float64")
String = type_str("str")

np_invert = {
    np.dtype("float16"): Float16,
    np.dtype("float32"): Float32,
    np.dtype("float64"): Float64,
    np.dtype("int8"): Integer8,
    np.dtype("int16"): Integer16,
    np.dtype("int32"): Integer32,
    np.dtype("int64"): Integer64,
    np.dtype("uint8"): UInteger8,
    np.dtype("uint16"): UInteger16,
    np.dtype("uint32"): UInteger32,
    np.dtype("uint64"): UInteger64,
    np.dtype("str"): String,
    str: String,
    float: Float32,
    np.float32: Float32,
    np.float64: Float64,
    np.int64: Integer64,
    np.int32: Integer32,
    np.int16: Integer16,
    np.int8: Integer8,
    np.uint64: UInteger64,
    np.uint32: UInteger32,
    np.uint16: UInteger16,
    np.uint8: UInteger8,
    np.int8: Byte,
    np.uint8: UByte,
    list: list,
}


def type_from_spec(spec: str) -> type:

    rex = r"<(?:Array)?\[?([a-z0-9]+)\]?:?.*?>"
    try:
        _str_type = re.search(rex, spec)
        if _str_type is None:
            raise UnknownDataType(f"Unknown type: {spec}")

        str_type = _str_type.groups()[0]
    except (AttributeError, IndexError):
        raise UnknownDataType(f"Unknown type: {spec}")

    if str_type == "str":
        return str

    if str_type == "list":
        return list

    try:
        return getattr(np, str_type)
    except AttributeError:
        raise UnknownDataType(f"Unknown type: {str_type}")
