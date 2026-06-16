"""Decide and perform the storage of a validated upload (``--upload-to``).

When ``vocal web`` runs with ``--upload-to=DIR``, a file that receives a PASS
verdict is copied into ``DIR`` — confined there by a safe name. This module
isolates that side effect from the check spine:

- :func:`decide_landing` is the pure policy: a function of three booleans
  (is the verdict PASS, is the derived name safe, does the target already
  exist), returning a :class:`LandingOutcome`. No I/O, so the matrix is
  exhaustively testable.
- :func:`safe_landing_name` reduces an uploaded filename to its basename and
  reports whether the result is safe to write inside ``DIR``.
- :func:`perform_landing` is the thin I/O shell: it derives the name, tests
  existence, calls the pure decision, performs the copy on ``LAND``, emits the
  server-side log line, and returns a typed :class:`~vocal.web.models.Landing`
  result (or ``None`` when storage was not attempted).
"""

from __future__ import annotations

import enum
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

from vocal.web.models import Landing

logger = logging.getLogger(__name__)


class LandingOutcome(enum.Enum):
    """The outcome of the landing policy for one uploaded file.

    - ``NOT_ATTEMPTED`` — the verdict is not PASS (or the feature is off): nothing
      is stored.
    - ``LAND`` — a PASS file with a safe name and no collision: copy it into the
      directory.
    - ``REFUSE_UNSAFE_NAME`` — a PASS file whose derived name is unsafe (empty,
      ``.``/``..``, or still containing a path separator): refuse, write nothing.
    - ``REFUSE_EXISTS`` — a PASS file with a safe name, but a file of that name
      already exists in the directory: refuse rather than overwrite the existing
      good file.
    """

    NOT_ATTEMPTED = "not_attempted"
    LAND = "land"
    REFUSE_UNSAFE_NAME = "refuse_unsafe_name"
    REFUSE_EXISTS = "refuse_exists"


def safe_landing_name(filename: str) -> tuple[str, bool]:
    """Reduce an uploaded *filename* to its basename and report whether it is safe.

    The returned name is the uploaded filename's basename — the on-disk name a
    PASS file would land under. It is *safe* when it is non-empty, is neither
    ``.`` nor ``..``, and contains no residual path separator (``/`` or ``\\``).
    The separator check catches a cross-platform traversal (e.g. a backslash
    name that :func:`os.path.basename` does not split on POSIX), so a crafted
    upload cannot escape the configured directory.

    Returns:
        A ``(name, safe)`` pair. ``name`` is the derived basename (possibly empty
        or degenerate); callers must consult ``safe`` before writing it.
    """
    name = os.path.basename(filename)
    safe = (
        bool(name) and name not in {".", ".."} and "/" not in name and "\\" not in name
    )
    return name, safe


def decide_landing(
    *, is_pass: bool, name_safe: bool, target_exists: bool
) -> LandingOutcome:
    """Decide the landing outcome — the pure storage policy.

    A function of three booleans and nothing else (no I/O):

    - verdict not PASS → :attr:`LandingOutcome.NOT_ATTEMPTED`;
    - PASS, unsafe name → ``REFUSE_UNSAFE_NAME`` (write nothing outside ``DIR``);
    - PASS, safe name, target already exists → ``REFUSE_EXISTS`` (never
      overwrite the existing good file);
    - PASS, safe name, no collision → ``LAND``.
    """
    if not is_pass:
        return LandingOutcome.NOT_ATTEMPTED
    if not name_safe:
        return LandingOutcome.REFUSE_UNSAFE_NAME
    if target_exists:
        return LandingOutcome.REFUSE_EXISTS
    return LandingOutcome.LAND


def perform_landing(
    *,
    is_pass: bool,
    source_path: Union[str, Path],
    filename: str,
    upload_dir: Path,
) -> Optional[Landing]:
    """Store a PASS file under *upload_dir* — the thin I/O shell over the policy.

    Derives the safe name from *filename*, tests whether the target already
    exists, and consults :func:`decide_landing`. On ``LAND`` it copies the
    already-validated bytes from *source_path* (the temp file ``check_upload``
    wrote) into *upload_dir*, logs an INFO audit line, and returns a
    ``"stored"`` :class:`~vocal.web.models.Landing`. On ``REFUSE_UNSAFE_NAME`` or
    ``REFUSE_EXISTS`` (a name collision) it writes nothing, logs a WARNING, and
    returns a ``"refused"`` result. When storage was not attempted (the verdict
    is not PASS) it returns ``None``.

    The returned message is path-free by design; only the server-side log names
    the directory, for the operator's audit trail.
    """
    name, name_safe = safe_landing_name(filename)
    target = upload_dir / name
    outcome = decide_landing(
        is_pass=is_pass, name_safe=name_safe, target_exists=target.exists()
    )

    if outcome is LandingOutcome.NOT_ATTEMPTED:
        return None

    if outcome is LandingOutcome.REFUSE_UNSAFE_NAME:
        logger.warning(
            "Refused to store upload %r: unsafe filename for landing.", filename
        )
        return Landing(
            status="refused",
            message=(
                "Your file passed validation but could not be stored: the "
                "filename was rejected as unsafe."
            ),
        )

    if outcome is LandingOutcome.REFUSE_EXISTS:
        logger.warning(
            "Refused to store upload '%s' in %s: a file of that name already "
            "exists (not overwritten).",
            name,
            upload_dir,
        )
        return Landing(
            status="refused",
            message=(
                "Your file passed validation but could not be stored: a file "
                "with the same name already exists and was not overwritten."
            ),
        )

    shutil.copyfile(source_path, target)
    logger.info("Stored validated upload '%s' in %s", name, upload_dir)
    return Landing(
        status="stored",
        message="Your file passed validation and was stored.",
    )
