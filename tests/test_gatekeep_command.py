import pytest

import typer

from vocal.application.gatekeep import (
    PassAction,
    FailAction,
    check_options,
    check_watch_folder_exists,
)


class TestCheckOptions:
    def test_check_options_good(self):
        # Valid combinations
        check_options(PassAction.NONE, FailAction.DELETE, None, None)
        check_options(PassAction.NONE, FailAction.MOVE, None, None)
        check_options(PassAction.MOVE, FailAction.DELETE, None, None)
        check_options(PassAction.MOVE, FailAction.MOVE, None, None)
        check_options(PassAction.COMMAND, FailAction.COMMAND, "echo pass", "echo fail")

    @pytest.mark.parametrize(
        "pass_action, fail_action, pass_command, fail_command",
        [
            (PassAction.COMMAND, FailAction.MOVE, None, "echo fail"),
            (PassAction.NONE, FailAction.COMMAND, "echo pass", None),
            (PassAction.COMMAND, FailAction.DELETE, None, "echo fail"),
            (PassAction.MOVE, FailAction.COMMAND, "echo pass", None),
            (PassAction.COMMAND, FailAction.COMMAND, None, "echo fail"),
            (PassAction.COMMAND, FailAction.COMMAND, "echo pass", None),
        ],
    )
    def test_check_options_bad(
        self, pass_action, fail_action, pass_command, fail_command
    ):
        # Invalid combinations
        with pytest.raises(typer.BadParameter):
            check_options(pass_action, fail_action, pass_command, fail_command)


class TestCheckWatchFolderExists:
    def test_check_watch_folder_exists_good(self, tmp_path):
        # Should not raise for existing directory
        check_watch_folder_exists(str(tmp_path))

    def test_check_watch_folder_exists_bad(self):
        # Should raise for non-existent directory
        with pytest.raises(typer.BadParameter):
            check_watch_folder_exists("/path/that/does/not/exist")
