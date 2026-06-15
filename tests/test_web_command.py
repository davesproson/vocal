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
    UploadDirUnavailable,
    UploadDownloadsBlocked,
    command,
    decide_launch,
    is_remote_bind,
    upload_downloads_conflict,
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
# Pure policy — upload_downloads_conflict
# ---------------------------------------------------------------------------


class TestUploadDownloadsConflict:
    """--upload-to and --allow-downloads are mutually exclusive."""

    def test_both_set_conflicts(self) -> None:
        # Disk-writing ingest combined with the unauthenticated-RCE download
        # surface is the one combination refused outright.
        assert (
            upload_downloads_conflict(upload_to=True, allow_downloads=True) is True
        )

    @pytest.mark.parametrize(
        ("upload_to", "allow_downloads"),
        [(False, False), (True, False), (False, True)],
    )
    def test_no_conflict_otherwise(
        self, upload_to: bool, allow_downloads: bool
    ) -> None:
        assert (
            upload_downloads_conflict(
                upload_to=upload_to, allow_downloads=allow_downloads
            )
            is False
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
                    upload_to=None,
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
                upload_to=None,
            )
        create.assert_called_once_with(allow_user_download=True, upload_dir=None)
        run.assert_called_once()


# ---------------------------------------------------------------------------
# command — the --upload-to launch gate
# ---------------------------------------------------------------------------


class TestUploadToLaunch:
    def test_upload_to_with_downloads_refuses_before_serving(
        self, tmp_path
    ) -> None:
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.create_app") as create,
            patch("vocal.application.web.threading.Thread"),
        ):
            with pytest.raises(UploadDownloadsBlocked):
                command(
                    allow_downloads=True,
                    upload_to=tmp_path,
                )
        run.assert_not_called()
        create.assert_not_called()

    def test_nonexistent_dir_refuses_and_is_not_created(self, tmp_path) -> None:
        missing = tmp_path / "incoming"
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.create_app") as create,
            patch("vocal.application.web.threading.Thread"),
        ):
            with pytest.raises(UploadDirUnavailable):
                command(upload_to=missing, allow_downloads=False)
        run.assert_not_called()
        create.assert_not_called()
        # A typo'd path must never silently produce a directory tree.
        assert not missing.exists()

    def test_path_that_is_a_file_refuses_before_serving(self, tmp_path) -> None:
        not_a_dir = tmp_path / "incoming"
        not_a_dir.write_text("i am a file")
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.create_app") as create,
            patch("vocal.application.web.threading.Thread"),
        ):
            with pytest.raises(UploadDirUnavailable):
                command(upload_to=not_a_dir, allow_downloads=False)
        run.assert_not_called()
        create.assert_not_called()

    def test_unwritable_dir_refuses_before_serving(self, tmp_path) -> None:
        unwritable = tmp_path / "incoming"
        unwritable.mkdir()
        unwritable.chmod(0o500)
        try:
            with (
                patch("vocal.application.web.uvicorn.run") as run,
                patch("vocal.application.web.create_app") as create,
                patch("vocal.application.web.threading.Thread"),
            ):
                with pytest.raises(UploadDirUnavailable):
                    command(upload_to=unwritable, allow_downloads=False)
            run.assert_not_called()
            create.assert_not_called()
        finally:
            unwritable.chmod(0o700)

    def test_valid_dir_reaches_create_app_and_serves(self, tmp_path) -> None:
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.create_app") as create,
            patch("vocal.application.web.threading.Thread"),
        ):
            command(host="127.0.0.1", upload_to=tmp_path, allow_downloads=False)
        create.assert_called_once_with(
            allow_user_download=False, upload_dir=tmp_path
        )
        run.assert_called_once()

    def test_cli_accepts_upload_to_and_reaches_create_app(self, tmp_path) -> None:
        # Through the real CLI app: --upload-to is a registered option and the
        # validated directory reaches create_app; uvicorn is started.
        runner = CliRunner()
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.create_app") as create,
            patch("vocal.application.web.threading.Thread"),
        ):
            result = runner.invoke(cli_app, ["web", "--upload-to", str(tmp_path)])
        assert result.exit_code == 0
        run.assert_called_once()
        create.assert_called_once_with(
            allow_user_download=False, upload_dir=tmp_path
        )

    def test_cli_upload_to_with_downloads_refuses(self, tmp_path) -> None:
        runner = CliRunner()
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.threading.Thread"),
        ):
            result = runner.invoke(
                cli_app,
                ["web", "--upload-to", str(tmp_path), "--allow-downloads"],
            )
        assert result.exit_code != 0
        run.assert_not_called()
        assert isinstance(result.exception, UploadDownloadsBlocked)

    def test_remote_bind_with_upload_to_warns_and_serves(self, tmp_path) -> None:
        # Storing a validated file is benign next to code execution, so a
        # non-loopback bind with --upload-to (downloads off) keeps today's soft
        # WARN_REMOTE decision rather than requiring an acknowledgement.
        with (
            patch("vocal.application.web.uvicorn.run") as run,
            patch("vocal.application.web.create_app") as create,
            patch("vocal.application.web.threading.Thread"),
            patch("vocal.application.web._render_remote_warning") as warn,
        ):
            command(host="0.0.0.0", upload_to=tmp_path, allow_downloads=False)
        warn.assert_called_once()
        create.assert_called_once_with(
            allow_user_download=False, upload_dir=tmp_path
        )
        run.assert_called_once()
