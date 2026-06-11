import enum
import os
import time

import typer

from vocal.utils.conventions import read_file_conventions
from vocal.utils.registry import Registry
from vocal.resolution import ResolvedTarget, resolve


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


def check_options(
    pass_action: PassAction,
    fail_action: FailAction,
    pass_command: str | None,
    fail_command: str | None,
) -> None:
    """Validate the combination of options passed to the command."""
    if pass_action == PassAction.COMMAND and not pass_command:
        raise typer.BadParameter(
            "The --pass-command option is required when --pass-action is 'command'."
        )
    if fail_action == FailAction.COMMAND and not fail_command:
        raise typer.BadParameter(
            "The --fail-command option is required when --fail-action is 'command'."
        )


def check_watch_folder_exists(folder: str) -> None:
    """Validate that the watch folder exists."""

    if not os.path.isdir(folder):
        raise typer.BadParameter(f"The watch folder '{folder}' does not exist.")


def ensure_file_eligibility(registry: Registry, filepath: str) -> ResolvedTarget | None:
    """
    Check if a file is eligible for processing by ensuring it has vocal-managed
    global attributes and that they resolve correctly against the registry.
    """

    try:
        attrs = read_file_conventions(filepath)
    except Exception:
        return None

    resolved = resolve(
        registry,
        filename=filepath,
        conventions=attrs.conventions,
        definitions_url=attrs.definitions_url,
        definitions_version=attrs.definitions_version,
        project_url=attrs.project_url,
    )

    if (
        resolved.project is not None
        and resolved.schema_path is not None
        and resolved.pack is not None
        and resolved.product is not None
    ):
        return resolved

    return None


def fail_file(filepath: str, fail_action: FailAction, fail_command: str | None) -> None:
    """
    Take the specified action on a file that failed checks.
    """
    print(f"Failing file '{filepath}' with action '{fail_action}'.")


def pass_file(filepath: str, pass_action: PassAction, pass_command: str | None) -> None:
    """
    Take the specified action on a file that passed checks.
    """
    print(f"Passing file '{filepath}' with action '{pass_action}'.")


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
        FailAction.DELETE,
        "--fail-action",
        "-fa",
        case_sensitive=False,
        help=(
            "The action to take on files that fail checks. Defaults to "
            "deleting failed files."
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
) -> None:
    """Watch a folder for new files to check."""
    check_options(pass_action, fail_action, pass_command, fail_command)
    check_watch_folder_exists(watch_folder)

    try:
        registry = Registry.load()
    except FileNotFoundError:
        registry = Registry()

    while True:
        time.sleep(frequency)

        for filename in os.listdir(watch_folder):
            filepath = os.path.join(watch_folder, filename)

            if os.path.isfile(filepath):
                resolved = ensure_file_eligibility(registry, filepath)
                import pprint

                pprint.pprint(resolved)
                if resolved is None:
                    print("File is not eligible for processing, failing.")

            else:
                print("File is eligible for processing, passing.")
