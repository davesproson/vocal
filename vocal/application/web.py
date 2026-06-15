"""Launch a web-based checker GUI.

The GUI's ``/add`` route fetches a project or pack from a URL, and a fetched
*project's* Python package is imported — i.e. runs — when a file is checked.
That makes the running server a code-execution surface, so launching it is
gated by two independent, default-safe controls:

- ``--allow-downloads`` (default off) enables the ``/add`` fetch route at all;
- a non-loopback ``--host`` *combined with* downloads is the one critical
  combination — unauthenticated remote code execution — and is refused unless
  ``--dangerously-allow-remote`` acknowledges it.

A separate, opt-in ``--upload-to=DIR`` turns the checker into a lightweight
ingest gate (files that PASS are copied into ``DIR``). It is refused alongside
``--allow-downloads`` — never combine a disk-writing ingest with the
unauthenticated-RCE download surface — and the directory is validated (exists,
is a directory, writable; never created) before the server binds.

The security decisions are isolated as pure functions — :func:`is_remote_bind`,
:func:`decide_launch`, and :func:`upload_downloads_conflict` — so the full
matrix can be tested without sockets or uvicorn. :func:`command` is the thin I/O
shell: classify, decide, warn or refuse, then run.
"""

from __future__ import annotations

import enum
import ipaddress
import os
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel

from vocal.exceptions import VocalError
from vocal.web.api import create_app


class RemoteDownloadsBlocked(VocalError):
    """Refuse to enable downloads on a non-loopback bind without acknowledgement.

    Binding off-loopback while allowing user downloads turns the GUI into an
    unauthenticated, network-reachable code-execution surface (a fetched
    project's package is imported when a file is checked). This is the one
    combination vocal refuses outright; the operator must pass
    ``--dangerously-allow-remote`` to proceed.
    """


class UploadDownloadsBlocked(VocalError):
    """Refuse to combine ``--upload-to`` with ``--allow-downloads``.

    Downloads turn the server into an unauthenticated remote code-execution
    surface (a fetched project's code runs when a file is checked); pairing that
    with a process that writes attacker-influenced files to the operator's disk
    is the one combination we refuse outright. The two flags are mutually
    exclusive — see :func:`upload_downloads_conflict`.
    """


class UploadDirUnavailable(VocalError):
    """Refuse to start when the ``--upload-to`` directory cannot receive files.

    Validated before the server binds so a typo or a permissions problem fails
    fast rather than surprising the operator mid-session. The directory must
    already exist, be a directory, and be writable; it is never created.
    """


def upload_downloads_conflict(*, upload_to: bool, allow_downloads: bool) -> bool:
    """Whether ``--upload-to`` and ``--allow-downloads`` are both in play.

    The central safety invariant as a pure function of the two flags (no I/O),
    so the refusal is unit-testable without binding a server: never combine a
    disk-writing ingest with the unauthenticated-RCE surface that downloads
    create.
    """
    return upload_to and allow_downloads


class LaunchDecision(enum.Enum):
    """The outcome of the launch-safety policy.

    - ``PROCEED`` — safe to launch silently (loopback bind, any download state).
    - ``WARN_REMOTE`` — non-loopback bind with downloads off: reachable as a
      read-only viewer; warn but proceed.
    - ``WARN_REMOTE_DOWNLOADS`` — non-loopback bind, downloads on, and the
      operator acknowledged the risk: proceed with a loud warning.
    - ``REFUSE`` — non-loopback bind, downloads on, *no* acknowledgement: refuse.
    """

    PROCEED = "proceed"
    WARN_REMOTE = "warn_remote"
    WARN_REMOTE_DOWNLOADS = "warn_remote_downloads"
    REFUSE = "refuse"


# Only the well-known loopback hostname is trusted by name; every other
# non-IP-literal host is treated as remote (fail-safe — see is_remote_bind).
_LOOPBACK_HOSTNAMES = {"localhost"}


def is_remote_bind(host: str) -> bool:
    """Whether binding to *host* exposes the server beyond the loopback interface.

    The load-bearing predicate for the launch policy. IP literals are classified
    with :func:`ipaddress.ip_address` so ``127.0.0.1``/``::1`` are loopback and
    the ``0.0.0.0``/``::`` wildcards — which bind every interface — correctly
    count as *remote*. A non-literal host (e.g. ``localhost``) is trusted only if
    it is the well-known loopback name; any other hostname is treated as remote
    so an unknown name we cannot prove is local errs toward exposure. No DNS
    resolution is performed.
    """
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.strip().lower() not in _LOOPBACK_HOSTNAMES


def decide_launch(
    *, remote_bind: bool, allow_downloads: bool, ack_remote: bool
) -> LaunchDecision:
    """Decide whether/how to launch — the pure security policy.

    A function of three booleans and nothing else (no I/O), so the matrix is
    exhaustively testable:

    - loopback bind → :attr:`LaunchDecision.PROCEED` (downloads/ack irrelevant);
    - non-loopback, downloads off → ``WARN_REMOTE`` (read-only viewer exposure);
    - non-loopback, downloads on, acknowledged → ``WARN_REMOTE_DOWNLOADS``;
    - non-loopback, downloads on, *not* acknowledged → ``REFUSE``.

    Args:
        remote_bind: whether the bind host is non-loopback (see
            :func:`is_remote_bind`).
        allow_downloads: whether ``--allow-downloads`` enabled the fetch route.
        ack_remote: whether ``--dangerously-allow-remote`` was given.
    """
    if not remote_bind:
        return LaunchDecision.PROCEED
    if not allow_downloads:
        return LaunchDecision.WARN_REMOTE
    if ack_remote:
        return LaunchDecision.WARN_REMOTE_DOWNLOADS
    return LaunchDecision.REFUSE


def _render_remote_warning(console: Console, host: str) -> None:
    """Soft warning: bound off-loopback, downloads disabled."""
    console.print(
        Panel(
            f"Listening on a non-loopback address ({host}).\n"
            "Anyone who can reach this machine on the network can use the "
            "checker.\n"
            "Downloads are disabled, so they cannot fetch new projects or packs.",
            title="Network exposure",
            border_style="yellow",
        )
    )


def _render_remote_downloads_warning(console: Console, host: str) -> None:
    """Loud warning: the operator opted into the critical combination."""
    console.print(
        Panel(
            f"Listening on a non-loopback address ({host}) WITH downloads "
            "enabled.\n\n"
            "Anyone who can reach this machine can fetch arbitrary projects, "
            "whose code runs here when a file is checked. This is "
            "unauthenticated remote code execution by design.\n\n"
            "Only do this on a trusted, access-controlled network.",
            title="⚠ DANGER: remote code execution surface",
            border_style="red",
        )
    )


def _validate_upload_dir(path: Path) -> None:
    """Refuse to launch unless *path* can receive landed files.

    Runs before the server binds so a misconfigured ``--upload-to`` fails fast
    instead of after a user uploads a file. The directory must already exist, be
    a directory, and be writable; it is *never* created — a typo'd path must not
    silently produce an unexpected directory tree. Raises
    :class:`UploadDirUnavailable` on any of these.
    """
    if not path.exists():
        raise UploadDirUnavailable(
            f"Upload directory '{path}' does not exist.",
            hint="Create it first (vocal never creates it), or fix the path.",
        )
    if not path.is_dir():
        raise UploadDirUnavailable(
            f"Upload target '{path}' is not a directory.",
            hint="Point --upload-to at an existing, writable directory.",
        )
    if not os.access(path, os.W_OK):
        raise UploadDirUnavailable(
            f"Upload directory '{path}' is not writable.",
            hint="Grant write permission, or choose a writable directory.",
        )


def command(
    host: str = typer.Option("127.0.0.1", "--host", help="The host to bind to."),
    port: int = typer.Option(8088, "--port", help="The port to bind to."),
    allow_downloads: bool = typer.Option(
        False,
        "--allow-downloads",
        help=(
            "Allow GUI users to fetch projects and packs from URLs. A fetched "
            "project's code runs on this machine when a file is checked. "
            "Disabled by default."
        ),
    ),
    dangerously_allow_remote: bool = typer.Option(
        False,
        "--dangerously-allow-remote",
        help=(
            "Acknowledge that enabling downloads on a non-loopback --host "
            "exposes unauthenticated remote code execution. Required to combine "
            "a non-loopback bind with --allow-downloads."
        ),
    ),
    upload_to: Optional[Path] = typer.Option(
        None,
        "--upload-to",
        help=(
            "Copy files that PASS validation into this existing, writable "
            "directory. Mutually exclusive with --allow-downloads. The "
            "directory is never created."
        ),
    ),
) -> None:
    """Launch a web-based checker GUI."""
    if upload_downloads_conflict(
        upload_to=upload_to is not None, allow_downloads=allow_downloads
    ):
        raise UploadDownloadsBlocked(
            "Refusing to combine --upload-to with --allow-downloads: this would "
            "write uploaded files to disk while exposing unauthenticated remote "
            "code execution.",
            hint="Run an ingest server (--upload-to) without --allow-downloads.",
        )

    if upload_to is not None:
        _validate_upload_dir(upload_to)

    decision = decide_launch(
        remote_bind=is_remote_bind(host),
        allow_downloads=allow_downloads,
        ack_remote=dangerously_allow_remote,
    )

    if decision is LaunchDecision.REFUSE:
        raise RemoteDownloadsBlocked(
            f"Refusing to enable downloads on non-loopback host '{host}': this "
            "exposes unauthenticated remote code execution.",
            hint=(
                "Bind to 127.0.0.1 (the default), drop --allow-downloads, or "
                "pass --dangerously-allow-remote to acknowledge the risk."
            ),
        )

    console = Console(stderr=True)
    if decision is LaunchDecision.WARN_REMOTE:
        _render_remote_warning(console, host)
    elif decision is LaunchDecision.WARN_REMOTE_DOWNLOADS:
        _render_remote_downloads_warning(console, host)

    def _start_browser_in_thread(host: str, port: int) -> None:
        time.sleep(1)
        webbrowser.open_new_tab(f"http://{host}:{port}")

    threading.Thread(target=_start_browser_in_thread, args=(host, port)).start()
    uvicorn.run(
        create_app(allow_user_download=allow_downloads, upload_dir=upload_to),
        host=host,
        port=port,
        log_level="info",
    )
