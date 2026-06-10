"""Tests for the ``vocal web`` launch-safety gate.

The security-critical decisions live in two pure functions tested exhaustively
here with no sockets or uvicorn:

- :func:`is_remote_bind` — classifies a bind host as loopback or remote,
  including the ``0.0.0.0``/``::`` wildcards and non-IP hostnames;
- :func:`decide_launch` — the full
  {remote bind?} × {downloads?} × {acknowledged?} matrix.

A thin slice of the :func:`command` I/O shell is exercised through the Typer CLI
to assert the one refusal branch surfaces a clean error (and that uvicorn is
never reached).
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from vocal.application.web import (
    LaunchDecision,
    RemoteDownloadsBlocked,
    command,
    decide_launch,
    is_remote_bind,
)
from vocal.cli.vocal import app as cli_app


# ---------------------------------------------------------------------------
# Pure predicate — is_remote_bind
# ---------------------------------------------------------------------------


class TestIsRemoteBind:
    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "127.0.0.5", "::1", "localhost", "LocalHost", " localhost "],
    )
    def test_loopback_hosts_are_local(self, host: str) -> None:
        assert is_remote_bind(host) is False

    @pytest.mark.parametrize(
        "host",
        [
            "0.0.0.0",  # IPv4 all-interfaces wildcard — reachable, not loopback
            "::",  # IPv6 all-interfaces wildcard
            "192.168.1.50",  # LAN address
            "10.0.0.1",
            "example.com",  # unknown hostname — fail-safe to remote
            "myhost.local",
        ],
    )
    def test_non_loopback_hosts_are_remote(self, host: str) -> None:
        assert is_remote_bind(host) is True


# ---------------------------------------------------------------------------
# Pure policy — decide_launch
# ---------------------------------------------------------------------------


class TestDecideLaunch:
    """The full {remote?} × {downloads?} × {acknowledged?} matrix."""

    @pytest.mark.parametrize("allow_downloads", [False, True])
    @pytest.mark.parametrize("ack_remote", [False, True])
    def test_loopback_always_proceeds(
        self, allow_downloads: bool, ack_remote: bool
    ) -> None:
        # A loopback bind is safe regardless of downloads or acknowledgement.
        assert (
            decide_launch(
                remote_bind=False,
                allow_downloads=allow_downloads,
                ack_remote=ack_remote,
            )
            is LaunchDecision.PROCEED
        )

    @pytest.mark.parametrize("ack_remote", [False, True])
    def test_remote_downloads_off_warns(self, ack_remote: bool) -> None:
        # Reachable read-only viewer: warn but proceed; ack is irrelevant.
        assert (
            decide_launch(
                remote_bind=True, allow_downloads=False, ack_remote=ack_remote
            )
            is LaunchDecision.WARN_REMOTE
        )

    def test_remote_downloads_on_acknowledged_warns_loudly(self) -> None:
        assert (
            decide_launch(remote_bind=True, allow_downloads=True, ack_remote=True)
            is LaunchDecision.WARN_REMOTE_DOWNLOADS
        )

    def test_remote_downloads_on_unacknowledged_refuses(self) -> None:
        # The critical cell: non-loopback + downloads, no acknowledgement.
        assert (
            decide_launch(remote_bind=True, allow_downloads=True, ack_remote=False)
            is LaunchDecision.REFUSE
        )


# ---------------------------------------------------------------------------
# command — the refusal branch never reaches uvicorn
# ---------------------------------------------------------------------------


class TestCommandRefusal:
    def test_remote_downloads_without_ack_raises_before_serving(self) -> None:
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.create_app") as create,
            patch("vocal.application.web.threading.Thread"),
        ):
            with pytest.raises(RemoteDownloadsBlocked):
                command(
                    host="0.0.0.0",
                    allow_downloads=True,
                    dangerously_allow_remote=False,
                )
        run.assert_not_called()
        create.assert_not_called()

    def test_cli_refusal_exits_nonzero_without_serving(self) -> None:
        # Through the real CLI app: the refusal surfaces as the typed
        # RemoteDownloadsBlocked (which main() renders cleanly), and uvicorn is
        # never started.
        runner = CliRunner()
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.threading.Thread"),
        ):
            result = runner.invoke(
                cli_app,
                ["web", "--host", "0.0.0.0", "--allow-downloads"],
            )
        assert result.exit_code != 0
        run.assert_not_called()
        assert isinstance(result.exception, RemoteDownloadsBlocked)
        assert "remote code execution" in result.exception.message

    def test_remote_downloads_with_ack_serves(self) -> None:
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.create_app") as create,
            patch("vocal.application.web.threading.Thread"),
            patch("vocal.application.web.Console"),
        ):
            command(
                host="0.0.0.0",
                allow_downloads=True,
                dangerously_allow_remote=True,
            )
        create.assert_called_once_with(allow_user_download=True)
        run.assert_called_once()
