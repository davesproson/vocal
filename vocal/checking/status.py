import enum


class ElementStatus(enum.Enum):
    EXISTS = enum.auto()
    DOES_NOT_EXIST_AND_REQUIRED = enum.auto()
    DOES_NOT_EXIST_AND_NOT_REQUIRED = enum.auto()
