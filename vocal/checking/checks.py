from dataclasses import dataclass

from .errors import CheckException


@dataclass
class CheckError:
    """
    Represents an error in a check
    """

    message: str
    path: str

    def raise_err(self) -> None:
        raise CheckException(self.message)


@dataclass
class CheckWarning:
    """
    Represents a warning in a check
    """

    message: str
    path: str


@dataclass
class CheckComment:
    """
    Represents a comment in a check
    """

    message: str
    path: str


@dataclass
class Check:
    """
    Represents a single check
    """

    description: str

    error: CheckError | None = None
    warning: CheckWarning | None = None
    comment: CheckComment | None = None

    @property
    def has_warning(self) -> bool:
        return self.warning is not None

    @property
    def has_comment(self) -> bool:
        return self.comment is not None

    @property
    def passed(self) -> bool:
        return self.error is None
