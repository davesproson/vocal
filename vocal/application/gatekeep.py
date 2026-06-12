from __future__ import annotations

import enum
import fcntl
import hashlib
from io import TextIOWrapper
import os
import shlex
import shutil
import sqlite3
import subprocess
import time

from dataclasses import dataclass, replace
from typing import Protocol

import rich
import typer

from vocal.checking.shared import run_check
from vocal.utils import cache_dir
from vocal.utils.conventions import read_file_conventions
from vocal.utils.registry import Registry
from vocal.resolution import ResolutionError, ResolvedTarget, resolve_file


class PassAction(str, enum.Enum):
    """The action to take on files that pass checks."""

    MOVE = "move"
    COMMAND = "command"
    NONE = "none"


class FailAction(str, enum.Enum):
    """The action to take on files that fail checks."""

    MOVE = "move"
    COMMAND = "command"
    DELETE = "delete"


class DataBaseProtocol(Protocol):
    """Protocol for the database used to track processed files."""

    def has_been_processed(self, filepath: str) -> bool: ...
    def mark_as_processed(self, filepath: str) -> None: ...
    def __enter__(self) -> DataBaseProtocol: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...


class DataBase:
    """A simple database to track files that have been processed."""

    def __init__(self) -> None:
        self.conn: sqlite3.Connection | None = None
        self.db_path = os.path.join(cache_dir(), "gatekeep.sqlite")

    def _init_connection(self) -> None:
        if self.conn is not None:
            return

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            "PRAGMA journal_mode=WAL"
        )  # readers don't block the writer; crash-safe
        self.conn.execute(
            "PRAGMA busy_timeout=5000"
        )  # retry on contention instead of "database is locked"

        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS processed_files (watch_folder TEXT, filepath TEXT, size INTEGER, mtime INTEGER, PRIMARY KEY (watch_folder, filepath, size, mtime))"
        )
        self.conn.commit()

    def has_been_processed(self, filepath: str) -> bool:
        """Check if a file has already been processed."""
        self._init_connection()
        assert self.conn is not None

        cursor = self.conn.cursor()
        size = os.path.getsize(filepath)
        mtime = os.stat(filepath).st_mtime_ns
        cursor.execute(
            "SELECT 1 FROM processed_files WHERE watch_folder = ? AND filepath = ? AND size = ? AND mtime = ?",
            (os.path.dirname(filepath), os.path.basename(filepath), size, mtime),
        )
        return cursor.fetchone() is not None

    def mark_as_processed(self, filepath: str) -> None:
        """Mark a file as processed."""
        self._init_connection()
        assert self.conn is not None

        watch_folder = os.path.dirname(filepath)
        name = os.path.basename(filepath)
        size = os.path.getsize(filepath)
        mtime = os.stat(filepath).st_mtime_ns
        self.conn.execute(
            "INSERT OR REPLACE INTO processed_files (watch_folder, filepath, size, mtime) VALUES (?, ?, ?, ?)",
            (watch_folder, name, size, mtime),
        )
        # Keep size/mtime in the key so a changed file (new size or mtime) is
        # rechecked — but prune the file's prior generations so the table holds
        # one row per path rather than accumulating a row per edit forever.
        self.conn.execute(
            "DELETE FROM processed_files WHERE watch_folder = ? AND filepath = ? AND NOT (size = ? AND mtime = ?)",
            (watch_folder, name, size, mtime),
        )
        self.conn.commit()

    def __enter__(self):
        self._init_connection()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


@dataclass(frozen=True)
class GatekeepConfig:
    """Configuration for the gatekeep command."""

    watch_folder: str
    pass_action: PassAction
    fail_action: FailAction
    pass_folder: str | None
    fail_folder: str | None
    pass_command: str | None
    fail_command: str | None
    frequency: int
    database: DataBaseProtocol
    command_timeout: int = 300
    registry: Registry | None = None

    def set_registry(self, registry: Registry) -> "GatekeepConfig":
        """Return a new config with the registry set."""
        return replace(self, registry=registry)


def get_watchfolder_lock(watch_folder: str) -> TextIOWrapper:
    """
    Get a lock for the watch folder to prevent multiple gatekeeper instances from
    running on the same folder.
    """
    key = hashlib.sha1(os.path.abspath(watch_folder).encode()).hexdigest()[:16]
    lock_path = os.path.join(cache_dir(), f"gatekeep-{key}.lock")
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        rich.print(
            f"[bold red]Error:[/bold red] Another instance of gatekeep is already running on the watch folder '{watch_folder}'."
        )
        raise typer.Exit(1)
    return fd


def check_options(config: GatekeepConfig) -> None:
    """Validate the combination of options passed to the command."""
    if config.pass_action == PassAction.COMMAND and not config.pass_command:
        raise typer.BadParameter(
            "The --pass-command option is required when --pass-action is 'command'."
        )
    if config.fail_action == FailAction.COMMAND and not config.fail_command:
        raise typer.BadParameter(
            "The --fail-command option is required when --fail-action is 'command'."
        )
    if config.pass_action == PassAction.MOVE and not config.pass_folder:
        raise typer.BadParameter(
            "The --pass-folder option is required when --pass-action is 'move'."
        )
    if config.fail_action == FailAction.MOVE and not config.fail_folder:
        raise typer.BadParameter(
            "The --fail-folder option is required when --fail-action is 'move'."
        )


def check_watch_folder_exists(folder: str) -> None:
    """Validate that the watch folder exists."""

    if not os.path.isdir(folder):
        raise typer.BadParameter(f"The watch folder '{folder}' does not exist.")


def check_move_folders_exist(config: GatekeepConfig) -> None:
    """Validate that any move-destination folders exist (fail fast at startup)."""

    if config.pass_action == PassAction.MOVE and not os.path.isdir(
        config.pass_folder or ""
    ):
        raise typer.BadParameter(
            f"The pass folder '{config.pass_folder}' does not exist."
        )
    if config.fail_action == FailAction.MOVE and not os.path.isdir(
        config.fail_folder or ""
    ):
        raise typer.BadParameter(
            f"The fail folder '{config.fail_folder}' does not exist."
        )


def file_has_recently_changed(filepath: str, threshold_seconds: int = 30) -> bool:
    """Check if a file has been modified within the last threshold_seconds."""
    current_time = time.time()
    last_modified_time = os.path.getmtime(filepath)
    return (current_time - last_modified_time) < threshold_seconds


def ensure_file_eligibility(registry: Registry, filepath: str) -> ResolvedTarget | None:
    """
    Check if a file is eligible for processing by ensuring it has vocal-managed
    global attributes and that they resolve correctly against the registry.
    """

    try:
        attrs = read_file_conventions(filepath)
    except Exception:
        # The watch folder may hold non-netCDF or unreadable files; those are
        # simply not eligible rather than an error that should stop the loop.
        return None

    try:
        resolved = resolve_file(filepath, attrs=attrs, registry=registry)
    except ResolutionError:
        # The file self-describes but its project/pack/product is not registered
        # (or is incompatible): not eligible for processing.
        return None

    if resolved.is_fully_resolved:
        return resolved

    return None


def _run_command(command: str, filepath: str, timeout: int | None = None) -> bool:
    """
    Run a user-supplied command against ``filepath``.

    A ``{}`` token in the command is substituted with the file path; if no such
    token is present, the path is appended as the final argument (find-style).
    The command is run without a shell (so file names with spaces are safe and
    no shell features are interpreted). Returns True iff the command exits 0.

    ``timeout`` bounds how long the command may run (seconds); a command that
    overruns is killed and treated as a failure rather than being allowed to
    block the watch loop indefinitely.
    """
    tokens = shlex.split(command)
    if "{}" in tokens:
        tokens = [filepath if t == "{}" else t for t in tokens]
    else:
        tokens.append(filepath)

    try:
        return subprocess.run(tokens, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        rich.print(
            f"   ↳ [bold red]✗[/bold red] Command '{command}' timed out after "
            f"{timeout}s on '{filepath}'."
        )
        return False


def fail_file(filepath: str, config: GatekeepConfig) -> bool:
    """
    Take the configured fail action on a file. Returns True if the action
    succeeded (for COMMAND, that the command exited 0).
    """
    rich.print(f"   ↳ Failing file '{filepath}' with action '{config.fail_action}'.")

    match config.fail_action:
        case FailAction.MOVE:
            assert config.fail_folder is not None  # guaranteed by check_options
            shutil.move(filepath, config.fail_folder)
            return True
        case FailAction.DELETE:
            os.remove(filepath)
            return True
        case FailAction.COMMAND:
            assert config.fail_command is not None  # guaranteed by check_options
            return _run_command(config.fail_command, filepath, config.command_timeout)


def pass_file(filepath: str, config: GatekeepConfig) -> bool:
    """
    Take the configured pass action on a file. Returns True if the action
    succeeded (for COMMAND, that the command exited 0).
    """
    rich.print(f"   ↳ Passing file '{filepath}' with action '{config.pass_action}'.")

    match config.pass_action:
        case PassAction.NONE:
            return True
        case PassAction.MOVE:
            assert config.pass_folder is not None  # guaranteed by check_options
            shutil.move(filepath, config.pass_folder)
            return True
        case PassAction.COMMAND:
            assert config.pass_command is not None  # guaranteed by check_options
            return _run_command(config.pass_command, filepath, config.command_timeout)


def check_file(filename: str, target: ResolvedTarget) -> bool:
    """
    Check a file against the resolved target's project and definition.

    Args:
        filename (str): The path of the file to check.
        target (ResolvedTarget): The resolved target to check against.

    Returns:
        bool: True if the file passes all checks, False otherwise.
    """
    rich.print(f":timer_clock:  Checking '{filename}'.")

    check_result = run_check(target, filename)

    if not check_result.passed:
        rich.print(f"   ↳ [bold red]✗[/bold red] File '{filename}' failed checks.")
        return False

    rich.print(f"   ↳ [bold green]✓[/bold green] File '{filename}' passed checks.")
    return True


def dispatch_file_after_check(
    filepath: str, passed: bool, config: GatekeepConfig
) -> None:
    """
    Dispatch a file after it has been checked, taking the appropriate action
    based on whether it passed or failed and the configured actions.

    The file is recorded as processed only if the action succeeded *and* the
    file still exists afterwards: MOVE/DELETE remove it (so there is nothing to
    dedup, and re-stat'ing it would fail), while NONE/COMMAND leave it in place
    and must be marked so it is not re-processed every cycle.
    """
    action_ok = pass_file(filepath, config) if passed else fail_file(filepath, config)

    if action_ok and os.path.exists(filepath):
        config.database.mark_as_processed(filepath)


def _process_file(config: GatekeepConfig, filepath: str) -> None:
    """
    Process a single entry from the watch folder: skip non-files, recently
    changed files, and already-processed files; otherwise check the file and
    dispatch it. ``isfile`` is checked first so a directory or a file that
    vanished between listing and now is skipped before any ``stat`` call.
    """
    assert config.registry is not None

    if not os.path.isfile(filepath):
        rich.print(
            f"[bold yellow]⚠[/bold yellow] '{filepath}' is not a file, skipping."
        )
        return

    if file_has_recently_changed(filepath):
        rich.print(
            f"[bold blue]i[/bold blue] '{filepath}' has been modified recently, skipping for now."
        )
        return

    if config.database.has_been_processed(filepath):
        return

    target = ensure_file_eligibility(config.registry, filepath)

    if target is None:
        rich.print(f":question: File '{filepath}' cannot be processed.")
        passed = False
    else:
        passed = check_file(filepath, target)

    dispatch_file_after_check(filepath, passed, config)


def check_folder(config: GatekeepConfig) -> None:
    for filename in os.listdir(config.watch_folder):
        filepath = os.path.join(config.watch_folder, filename)

        try:
            _process_file(config, filepath)
        except Exception as e:
            # One bad file (vanished mid-cycle, failed action, …) must not stop
            # the rest of the folder from being processed this cycle.
            rich.print(f"[bold red]Error processing '{filepath}':[/bold red] {e}")


def _load_registry() -> Registry:
    """Load the machine-local registry, treating a never-fetched machine as empty."""
    try:
        return Registry.load()
    except FileNotFoundError:
        return Registry()


def gatekeep_loop(config: GatekeepConfig) -> None:
    """
    Run the gatekeep loop, which checks the watch folder for new files at the
    specified frequency and processes them according to the configuration.

    The registry is reloaded at the start of every cycle so a long-running
    watcher picks up projects and packs fetched after it started, without a
    restart. The first scan runs immediately; the sleep follows each cycle.
    """
    with config.database:
        while True:
            try:
                cycle_config = config.set_registry(_load_registry())
                check_folder(cycle_config)
            except Exception as e:
                # A transient bad read (e.g. the registry mid-rewrite by a
                # concurrent `vocal fetch`) must not crash the daemon — skip the
                # cycle and self-heal next tick.
                rich.print(f"[bold red]Error:[/bold red] {e}")

            time.sleep(config.frequency)


def command(
    watch_folder: str = typer.Option(
        ...,
        "--watch-folder",
        "-w",
        help="The folder to watch for new files to check.",
    ),
    pass_action: PassAction = typer.Option(
        PassAction.NONE,
        "--pass-action",
        "-pa",
        case_sensitive=False,
        help=(
            "The action to take on files that pass checks. Defaults to doing nothing. "
            "Options are: 'move' (move passed files to a separate folder), 'command' (run a command on passed files), and 'none' (do nothing). "
            "If 'move' or 'command' is selected, the corresponding option must also be provided."
        ),
    ),
    fail_action: FailAction = typer.Option(
        FailAction.MOVE,
        "--fail-action",
        "-fa",
        case_sensitive=False,
        help=(
            "The action to take on files that fail checks. Defaults to "
            "moving failed files."
        ),
    ),
    pass_folder: str | None = typer.Option(
        None,
        "--pass-folder",
        "-pf",
        help=(
            "The folder to move passed files to if --pass-action is 'move'. "
            "Required if --pass-action is 'move'."
        ),
    ),
    fail_folder: str | None = typer.Option(
        None,
        "--fail-folder",
        "-ff",
        help=(
            "The folder to move failed files to if --fail-action is 'move'. "
            "Required if --fail-action is 'move'."
        ),
    ),
    pass_command: str | None = typer.Option(
        None,
        "--pass-command",
        "-pc",
        help=(
            "The command to run on files that pass checks. Required if "
            "--pass-action is 'command'."
        ),
    ),
    fail_command: str | None = typer.Option(
        None,
        "--fail-command",
        "-fc",
        help=(
            "The command to run on files that fail checks. Required if "
            "--fail-action is 'command'."
        ),
    ),
    frequency: int = typer.Option(
        300,
        "--frequency",
        "-f",
        help=(
            "The frequency, in seconds, with which to check the watch folder for "
            "new files. Defaults to 300 (5 minutes)."
        ),
    ),
    command_timeout: int = typer.Option(
        300,
        "--command-timeout",
        "-ct",
        help=(
            "Maximum time, in seconds, a --pass-command / --fail-command may run "
            "before it is killed and treated as a failure. Stops a hung command "
            "from blocking the watch loop. Defaults to 300 (5 minutes)."
        ),
    ),
) -> None:
    """Watch a folder for new files to check."""

    config = GatekeepConfig(
        watch_folder=watch_folder,
        pass_folder=pass_folder,
        fail_folder=fail_folder,
        pass_action=pass_action,
        fail_action=fail_action,
        pass_command=pass_command,
        fail_command=fail_command,
        frequency=frequency,
        command_timeout=command_timeout,
        database=DataBase(),
    )

    check_options(config)
    check_watch_folder_exists(watch_folder)
    check_move_folders_exist(config)

    lock_fd = get_watchfolder_lock(watch_folder)

    try:
        gatekeep_loop(config)
    finally:
        lock_fd.close()
