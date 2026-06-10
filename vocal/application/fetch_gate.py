"""Confirmation gate for file-driven fetches.

When a user runs ``vocal fetch --for <file>`` or ``vocal check <file> --fetch``,
vocal reads a ``vocal_project_url`` declared *inside the netCDF file* and fetches
that project. On the next check the project's Python package is imported — so the
file, an untrusted input, can cause code from a source the user never typed to
run on their machine. This module gates that on a one-time confirmation, fired
the first time a file's declared *project* would be installed.

The security-critical decision is isolated as a pure function,
:func:`decide_fetch_gate`, over three booleans (is the project new? was ``--yes``
given? can we prompt?) so the full matrix can be tested exhaustively without any
terminal mocking. A thin I/O shell, :func:`confirm_file_fetch`, reads the file's
declared URLs and the registry, computes whether the project is new, asks the
policy what to do, and only then performs side effects: render the warning panel
and prompt, or abort on a decline.

The gate is **project-centric**: the project is the only artifact that is
imported/executed, so only a *new project* triggers it. A pack is data (parsed,
not executed), so a pack-only-new fetch never prompts; the pack URL is still
listed for transparency when the gate does fire.

This slice (#47) wires the interactive ``PROCEED`` / ``PROMPT`` branches. The
``BLOCKED`` branch — a new project that cannot be confirmed — together with the
``--yes`` / ``-q`` / non-TTY plumbing is wired in the non-interactive slice
(#48); the policy already returns it so the full matrix is testable now.
"""

from __future__ import annotations

import enum

import typer
from rich.console import Console
from rich.panel import Panel

from vocal.exceptions import VocalError
from vocal.utils.conventions import read_file_conventions
from vocal.utils.registry import Registry


class FetchGateDecision(enum.Enum):
    """The outcome of the fetch-gate policy.

    - ``PROCEED`` — nothing new to confirm (or consent was given up front); the
      fetch may run without a prompt.
    - ``PROMPT`` — a new project would be installed and we can ask the user.
    - ``BLOCKED`` — a new project would be installed but we cannot prompt and no
      consent was given; the caller must refuse with a clean error.
    """

    PROCEED = "proceed"
    PROMPT = "prompt"
    BLOCKED = "blocked"


class FetchDeclined(VocalError):
    """The user declined the confirmation gate; abort with nothing fetched."""


def decide_fetch_gate(
    *, project_new: bool, yes: bool, can_prompt: bool
) -> FetchGateDecision:
    """Decide what the fetch gate should do — the pure security policy.

    The decision is a function of three booleans and nothing else (no file I/O,
    no terminal), so the security-critical matrix can be tested exhaustively:

    - the project is **not** new → :attr:`FetchGateDecision.PROCEED` (a pack-only
      fetch, or an already-consented project, needs no prompt);
    - the project is new but ``yes`` was given → ``PROCEED`` (consent up front);
    - the project is new, no ``yes``, and we can prompt → ``PROMPT``;
    - the project is new, no ``yes``, and we cannot prompt → ``BLOCKED``.

    Args:
        project_new: whether a *new* project would be installed (its URL is not
            already in the registry).
        yes: whether the user consented up front (``--yes``).
        can_prompt: whether vocal can ask the user interactively.

    Returns:
        the :class:`FetchGateDecision` the I/O shell should act on.
    """
    if not project_new:
        return FetchGateDecision.PROCEED
    if yes:
        return FetchGateDecision.PROCEED
    if can_prompt:
        return FetchGateDecision.PROMPT
    return FetchGateDecision.BLOCKED


def _render_warning(
    console: Console, filename: str, project_url: str, pack_url: str | None, route: str
) -> None:
    """Render the red security-warning panel to ``console`` (stderr).

    Leads with the execution consequence, anchors trust to the file's
    provenance, and labels the project as code and the pack (when declared) as
    data. The opening line is route-appropriate so the message reads naturally.
    """
    if route == "check":
        opening = f"Checking '{filename}' requires fetching code declared inside it."
    else:
        opening = f"Fetching the sources declared inside '{filename}'."

    lines = [
        opening,
        "",
        "The project's code will run on your machine when this file is checked.",
        "Only continue if you trust where this file came from.",
        "",
        f"  project (code — runs on check): {project_url}",
    ]
    if pack_url:
        lines.append(f"  pack (data): {pack_url}")

    console.print(
        Panel("\n".join(lines), title="⚠ Security warning", border_style="red")
    )


def confirm_file_fetch(filename: str, *, route: str, no_color: bool = False) -> None:
    """Gate a file-driven fetch on a one-time confirmation.

    Reads ``filename``'s declared ``vocal_project_url`` and the registry,
    computes whether that project is new (not already fetched by normalised URL),
    and asks :func:`decide_fetch_gate`. On ``PROMPT`` it renders the red warning
    panel to stderr and prompts ``[y/N]`` (default No); a decline raises
    :class:`FetchDeclined`, which the caller renders as a clean abort. On
    ``PROCEED`` it returns silently and the caller fetches as normal.

    Reading errors and a missing ``vocal_project_url`` are *not* handled here:
    the gate returns silently so the subsequent ``fetch_for_file`` surfaces the
    same typed error it always did (the file is read once more there — a cheap,
    harmless double read).

    Args:
        filename: the netCDF file whose declared sources are about to be fetched.
        route: ``"check"`` or ``"fetch"`` — selects the opening line.
        no_color: render the panel's box without the red ANSI.
    """
    try:
        attrs = read_file_conventions(filename)
    except (OSError, FileNotFoundError):
        # Let fetch_for_file surface the typed UnreadableNetCDF error.
        return

    if not attrs.project_url:
        # Let fetch_for_file surface the typed MissingProjectURL error.
        return

    try:
        registry = Registry.load()
    except FileNotFoundError:
        registry = Registry()

    project_new = registry.find_project_by_url(attrs.project_url) is None

    # This slice wires PROCEED/PROMPT only: an interactive run can always prompt
    # and no --yes flag exists yet, so BLOCKED is unreachable here (#48 wires it).
    decision = decide_fetch_gate(project_new=project_new, yes=False, can_prompt=True)

    if decision is FetchGateDecision.PROCEED:
        return

    console = Console(stderr=True, no_color=no_color)
    _render_warning(console, filename, attrs.project_url, attrs.definitions_url, route)
    if not typer.confirm("Continue?", default=False):
        raise FetchDeclined("Aborted — nothing fetched.")
