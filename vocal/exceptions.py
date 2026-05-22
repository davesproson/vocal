"""Base exception hierarchy for user-facing vocal failures."""

from typing import Optional


class VocalError(Exception):
    """Base class for user-facing vocal failures.

    Carries a short ``message`` and an optional ``hint`` describing how to
    resolve the problem. ``status_code`` is used by the web layer when a
    ``VocalError`` propagates to the global exception handler; subclasses
    override it to customise the HTTP response status.
    """

    status_code: int = 422

    def __init__(self, message: str, hint: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n  {self.hint}"
        return self.message
