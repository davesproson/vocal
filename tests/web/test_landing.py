"""Tests for the landing module (vocal/web/landing.py).

The landing decision is the deep module behind ``vocal web --upload-to``: a file
that PASSes validation is copied into the configured directory, confined there
by a safe name. ``decide_landing`` is the pure policy (no I/O); ``perform_landing``
is the thin shell that copies, logs, and returns a typed result.

Collision-on-existing-name (``REFUSE_EXISTS``) is deferred to a later slice.
"""

import logging
from pathlib import Path

import pytest

from vocal.web.landing import (
    LandingOutcome,
    decide_landing,
    perform_landing,
    safe_landing_name,
)


def _write_source(tmp_path: Path, name: str = "src.nc", data: bytes = b"netcdf-bytes") -> Path:
    """Write a stand-in for the temp file ``check_upload`` has already created."""
    source = tmp_path / name
    source.write_bytes(data)
    return source


class TestDecideLanding:
    def test_non_pass_verdict_is_not_attempted(self) -> None:
        # Storage is a post-verdict side effect of a PASS only: a FAIL or
        # INDETERMINATE file is never stored, whatever its name.
        outcome = decide_landing(is_pass=False, name_safe=True, target_exists=False)
        assert outcome is LandingOutcome.NOT_ATTEMPTED

    @pytest.mark.parametrize(
        "is_pass, name_safe, target_exists, expected",
        [
            # Verdict not PASS → never attempted (name/existence irrelevant).
            (False, True, False, LandingOutcome.NOT_ATTEMPTED),
            (False, True, True, LandingOutcome.NOT_ATTEMPTED),
            (False, False, False, LandingOutcome.NOT_ATTEMPTED),
            (False, False, True, LandingOutcome.NOT_ATTEMPTED),
            # PASS with an unsafe name → refuse, write nothing.
            (True, False, False, LandingOutcome.REFUSE_UNSAFE_NAME),
            (True, False, True, LandingOutcome.REFUSE_UNSAFE_NAME),
            # PASS with a safe name → land. (Collision handling is a later slice,
            # so an existing target still lands for now.)
            (True, True, False, LandingOutcome.LAND),
            (True, True, True, LandingOutcome.LAND),
        ],
    )
    def test_decision_matrix(
        self,
        is_pass: bool,
        name_safe: bool,
        target_exists: bool,
        expected: LandingOutcome,
    ) -> None:
        assert (
            decide_landing(
                is_pass=is_pass, name_safe=name_safe, target_exists=target_exists
            )
            is expected
        )


class TestSafeLandingName:
    def test_ordinary_name_is_safe(self) -> None:
        name, safe = safe_landing_name("flight123.nc")
        assert (name, safe) == ("flight123.nc", True)

    def test_posix_separator_is_reduced_to_basename(self) -> None:
        # A path is reduced to its basename, which is then safe to write in DIR.
        name, safe = safe_landing_name("some/dir/flight123.nc")
        assert (name, safe) == ("flight123.nc", True)

    def test_residual_backslash_separator_is_unsafe(self) -> None:
        # os.path.basename does not split on a backslash on POSIX, so a
        # Windows-style traversal survives as a residual separator: unsafe.
        _, safe = safe_landing_name("..\\..\\evil.nc")
        assert safe is False

    def test_dotdot_is_unsafe(self) -> None:
        _, safe = safe_landing_name("..")
        assert safe is False

    def test_single_dot_is_unsafe(self) -> None:
        _, safe = safe_landing_name(".")
        assert safe is False

    def test_empty_name_is_unsafe(self) -> None:
        _, safe = safe_landing_name("")
        assert safe is False

    def test_trailing_separator_yields_empty_basename_is_unsafe(self) -> None:
        # A name that is all directory ("evil/") reduces to an empty basename.
        _, safe = safe_landing_name("evil/")
        assert safe is False


class TestPerformLanding:
    def test_pass_lands_bytes_and_returns_stored(self, tmp_path, caplog) -> None:
        source = _write_source(tmp_path, data=b"hello-world")
        dest_dir = tmp_path / "incoming"
        dest_dir.mkdir()

        with caplog.at_level(logging.INFO, logger="vocal.web.landing"):
            landing = perform_landing(
                is_pass=True,
                source_path=source,
                filename="flight123.nc",
                upload_dir=dest_dir,
            )

        # The validated bytes land under the basename, in the configured dir.
        stored = dest_dir / "flight123.nc"
        assert stored.read_bytes() == b"hello-world"
        assert landing is not None and landing.status == "stored"
        # The user-facing message never names the server's path.
        assert str(dest_dir) not in landing.message
        # An INFO audit line names the file and the directory server-side.
        record = next(r for r in caplog.records if r.levelno == logging.INFO)
        assert "flight123.nc" in record.getMessage()
        assert str(dest_dir) in record.getMessage()

    def test_unsafe_name_writes_nothing_and_refuses(self, tmp_path, caplog) -> None:
        source = _write_source(tmp_path)
        dest_dir = tmp_path / "incoming"
        dest_dir.mkdir()

        with caplog.at_level(logging.WARNING, logger="vocal.web.landing"):
            landing = perform_landing(
                is_pass=True,
                source_path=source,
                filename="..",
                upload_dir=dest_dir,
            )

        # No write outside (or inside) the directory for an unsafe name.
        assert list(dest_dir.iterdir()) == []
        assert landing is not None and landing.status == "refused"
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_non_pass_writes_nothing_and_returns_none(self, tmp_path) -> None:
        source = _write_source(tmp_path)
        dest_dir = tmp_path / "incoming"
        dest_dir.mkdir()

        landing = perform_landing(
            is_pass=False,
            source_path=source,
            filename="flight123.nc",
            upload_dir=dest_dir,
        )

        assert landing is None
        assert list(dest_dir.iterdir()) == []
