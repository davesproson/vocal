import os
import types

import pytest

import typer

from vocal.application import gatekeep
from vocal.application.gatekeep import (
    GatekeepConfig,
    PassAction,
    FailAction,
    check_options,
    check_watch_folder_exists,
    check_move_folders_exist,
)
from vocal.utils.registry import Registry


class FakeDataBase:
    """In-memory stand-in for the SQLite dedup DB (keyed on path for tests)."""

    def __init__(self) -> None:
        self.processed: set[str] = set()

    def has_been_processed(self, filepath: str) -> bool:
        return filepath in self.processed

    def mark_as_processed(self, filepath: str) -> None:
        self.processed.add(filepath)

    def __enter__(self) -> "FakeDataBase":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


def make_config(
    *,
    watch_folder: str = "/tmp",
    pass_action: PassAction = PassAction.NONE,
    fail_action: FailAction = FailAction.DELETE,
    pass_folder: str | None = None,
    fail_folder: str | None = None,
    pass_command: str | None = None,
    fail_command: str | None = None,
    frequency: int = 1,
    database=None,
    registry=None,
) -> GatekeepConfig:
    return GatekeepConfig(
        watch_folder=watch_folder,
        pass_action=pass_action,
        fail_action=fail_action,
        pass_folder=pass_folder,
        fail_folder=fail_folder,
        pass_command=pass_command,
        fail_command=fail_command,
        frequency=frequency,
        database=database if database is not None else FakeDataBase(),
        registry=registry,
    )


class TestCheckOptions:
    def test_check_options_good(self):
        # Valid combinations (MOVE now requires its folder).
        check_options(make_config(pass_action=PassAction.NONE, fail_action=FailAction.DELETE))
        check_options(
            make_config(
                pass_action=PassAction.NONE,
                fail_action=FailAction.MOVE,
                fail_folder="/some/folder",
            )
        )
        check_options(
            make_config(
                pass_action=PassAction.MOVE,
                fail_action=FailAction.DELETE,
                pass_folder="/some/folder",
            )
        )
        check_options(
            make_config(
                pass_action=PassAction.MOVE,
                fail_action=FailAction.MOVE,
                pass_folder="/p",
                fail_folder="/f",
            )
        )
        check_options(
            make_config(
                pass_action=PassAction.COMMAND,
                fail_action=FailAction.COMMAND,
                pass_command="echo pass",
                fail_command="echo fail",
            )
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            # COMMAND without the corresponding command
            dict(pass_action=PassAction.COMMAND, fail_action=FailAction.MOVE, fail_folder="/f"),
            dict(pass_action=PassAction.NONE, fail_action=FailAction.COMMAND),
            dict(pass_action=PassAction.COMMAND, fail_action=FailAction.DELETE),
            # MOVE without the corresponding folder
            dict(pass_action=PassAction.MOVE, fail_action=FailAction.DELETE),
            dict(pass_action=PassAction.NONE, fail_action=FailAction.MOVE),
        ],
    )
    def test_check_options_bad(self, kwargs):
        with pytest.raises(typer.BadParameter):
            check_options(make_config(**kwargs))


class TestCheckWatchFolderExists:
    def test_check_watch_folder_exists_good(self, tmp_path):
        check_watch_folder_exists(str(tmp_path))

    def test_check_watch_folder_exists_bad(self):
        with pytest.raises(typer.BadParameter):
            check_watch_folder_exists("/path/that/does/not/exist")


class TestCheckMoveFoldersExist:
    def test_existing_folders_ok(self, tmp_path):
        check_move_folders_exist(
            make_config(
                pass_action=PassAction.MOVE,
                fail_action=FailAction.MOVE,
                pass_folder=str(tmp_path),
                fail_folder=str(tmp_path),
            )
        )

    def test_missing_pass_folder_raises(self):
        with pytest.raises(typer.BadParameter):
            check_move_folders_exist(
                make_config(pass_action=PassAction.MOVE, pass_folder="/no/such/dir")
            )

    def test_missing_fail_folder_raises(self):
        with pytest.raises(typer.BadParameter):
            check_move_folders_exist(
                make_config(fail_action=FailAction.MOVE, fail_folder="/no/such/dir")
            )

    def test_non_move_actions_skip_validation(self):
        # No folders, but neither action is MOVE — must not raise.
        check_move_folders_exist(
            make_config(pass_action=PassAction.NONE, fail_action=FailAction.DELETE)
        )


class TestRunCommand:
    def test_substitutes_placeholder(self, monkeypatch):
        captured = {}

        def fake_run(tokens, *a, **k):
            captured["tokens"] = tokens
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(gatekeep.subprocess, "run", fake_run)
        gatekeep._run_command("process --in {} --verbose", "/data/f.nc")
        assert captured["tokens"] == ["process", "--in", "/data/f.nc", "--verbose"]

    def test_appends_when_no_placeholder(self, monkeypatch):
        captured = {}

        def fake_run(tokens, *a, **k):
            captured["tokens"] = tokens
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(gatekeep.subprocess, "run", fake_run)
        gatekeep._run_command("process", "/data/f.nc")
        assert captured["tokens"] == ["process", "/data/f.nc"]

    def test_nonzero_exit_is_failure(self, monkeypatch):
        monkeypatch.setattr(
            gatekeep.subprocess,
            "run",
            lambda *a, **k: types.SimpleNamespace(returncode=3),
        )
        assert gatekeep._run_command("process", "/data/f.nc") is False

    def test_timeout_is_failure(self, monkeypatch):
        def fake_run(tokens, *a, timeout=None, **k):
            raise gatekeep.subprocess.TimeoutExpired(tokens, timeout)

        monkeypatch.setattr(gatekeep.subprocess, "run", fake_run)
        assert gatekeep._run_command("process", "/data/f.nc", timeout=1) is False


class TestActions:
    def test_pass_none_leaves_file(self, tmp_path):
        f = tmp_path / "f.nc"
        f.write_text("x")
        assert gatekeep.pass_file(str(f), make_config(pass_action=PassAction.NONE)) is True
        assert f.exists()

    def test_pass_move_relocates_file(self, tmp_path):
        dest = tmp_path / "passed"
        dest.mkdir()
        f = tmp_path / "f.nc"
        f.write_text("x")
        config = make_config(pass_action=PassAction.MOVE, pass_folder=str(dest))
        assert gatekeep.pass_file(str(f), config) is True
        assert not f.exists()
        assert (dest / "f.nc").exists()

    def test_fail_delete_removes_file(self, tmp_path):
        f = tmp_path / "f.nc"
        f.write_text("x")
        assert gatekeep.fail_file(str(f), make_config(fail_action=FailAction.DELETE)) is True
        assert not f.exists()

    def test_fail_move_relocates_file(self, tmp_path):
        dest = tmp_path / "failed"
        dest.mkdir()
        f = tmp_path / "f.nc"
        f.write_text("x")
        config = make_config(fail_action=FailAction.MOVE, fail_folder=str(dest))
        assert gatekeep.fail_file(str(f), config) is True
        assert not f.exists()
        assert (dest / "f.nc").exists()

    def test_command_success_and_failure(self, tmp_path):
        f = tmp_path / "f.nc"
        f.write_text("x")
        assert (
            gatekeep.fail_file(
                str(f), make_config(fail_action=FailAction.COMMAND, fail_command="true")
            )
            is True
        )
        assert (
            gatekeep.fail_file(
                str(f), make_config(fail_action=FailAction.COMMAND, fail_command="false")
            )
            is False
        )


class TestProcessFileMarking:
    def _patch_check(self, monkeypatch, *, passed: bool, calls: list):
        """Make every file eligible; record each check_file call; return ``passed``."""
        monkeypatch.setattr(gatekeep, "file_has_recently_changed", lambda *a, **k: False)
        monkeypatch.setattr(
            gatekeep, "ensure_file_eligibility", lambda registry, fp: object()
        )

        def fake_check_file(filepath, target):
            calls.append(filepath)
            return passed

        monkeypatch.setattr(gatekeep, "check_file", fake_check_file)

    def test_passing_none_file_is_marked_and_skipped(self, monkeypatch, tmp_path):
        f = tmp_path / "f.nc"
        f.write_text("x")
        calls: list = []
        self._patch_check(monkeypatch, passed=True, calls=calls)

        db = FakeDataBase()
        config = make_config(
            watch_folder=str(tmp_path),
            pass_action=PassAction.NONE,
            database=db,
            registry=Registry(),
        )

        gatekeep._process_file(config, str(f))
        assert db.has_been_processed(str(f))

        # Second pass: already processed, so check_file is not invoked again.
        gatekeep._process_file(config, str(f))
        assert calls == [str(f)]

    def test_moved_file_is_not_marked(self, monkeypatch, tmp_path):
        dest = tmp_path / "passed"
        dest.mkdir()
        f = tmp_path / "f.nc"
        f.write_text("x")
        self._patch_check(monkeypatch, passed=True, calls=[])

        db = FakeDataBase()
        config = make_config(
            watch_folder=str(tmp_path),
            pass_action=PassAction.MOVE,
            pass_folder=str(dest),
            database=db,
            registry=Registry(),
        )

        gatekeep._process_file(config, str(f))
        assert not f.exists()
        assert (dest / "f.nc").exists()
        assert db.processed == set()  # gone from the folder ⇒ nothing to dedup


class TestDataBasePruning:
    def _db(self, tmp_path):
        db = gatekeep.DataBase()
        db.db_path = str(tmp_path / "gatekeep.sqlite")
        return db

    def _row_count(self, db, filepath):
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM processed_files WHERE watch_folder = ? AND filepath = ?",
            (os.path.dirname(filepath), os.path.basename(filepath)),
        )
        return cursor.fetchone()[0]

    def test_changed_file_is_rechecked_but_old_generation_pruned(self, tmp_path):
        f = tmp_path / "f.nc"
        f.write_text("x")

        with self._db(tmp_path) as db:
            db.mark_as_processed(str(f))
            assert db.has_been_processed(str(f))

            # Rewrite with different content (new size/mtime): the prior
            # generation no longer matches, so the file is eligible again...
            f.write_text("much longer content")
            assert not db.has_been_processed(str(f))

            # ...and once re-marked, only one row survives for the path.
            db.mark_as_processed(str(f))
            assert db.has_been_processed(str(f))
            assert self._row_count(db, str(f)) == 1


class TestCheckFolderIsolation:
    def test_one_failing_file_does_not_stop_the_rest(self, monkeypatch, tmp_path):
        for name in ["a", "b", "c"]:
            (tmp_path / name).write_text("x")

        seen: list = []

        def fake_process(config, filepath):
            seen.append(os.path.basename(filepath))
            if filepath.endswith("b"):
                raise RuntimeError("boom")

        monkeypatch.setattr(gatekeep, "_process_file", fake_process)

        config = make_config(watch_folder=str(tmp_path), registry=Registry())
        # Must not raise despite "b" blowing up.
        gatekeep.check_folder(config)
        assert set(seen) == {"a", "b", "c"}
