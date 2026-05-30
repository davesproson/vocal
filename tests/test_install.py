"""Tests for the shared install primitive and the pure dest/identity helpers.

``staged_install`` is exercised behaviourally: given a source tree, an ignore
callable, and a validate callback, assert what ends up on disk at ``dest``, what
is *preserved* when validation fails, and that staging never leaks. No project or
pack knowledge is required — fake sources and validate callbacks suffice.

The path helpers are pure: they map an identity to a path with no I/O, so they
are checked against ``cache_dir()``-rooted expectations.
"""

import os
from pathlib import Path

import pytest

from vocal.application.install import (
    DEFAULT_IGNORE,
    derive_url_slug,
    pack_install_dir,
    project_install_dir,
    staged_install,
)


def _tree(root: Path, files: dict[str, str]) -> Path:
    """Materialise ``{relative_path: contents}`` under ``root`` and return it."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def _snapshot(root: Path) -> dict[str, str]:
    """Return ``{relative_path: contents}`` for every file under ``root``."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_text()
    return out


def _siblings(dest: Path) -> list[str]:
    """Names alongside ``dest`` in its parent (to catch leaked staging dirs)."""
    return sorted(p.name for p in dest.parent.iterdir())


class TestStagedInstall:
    def test_copies_source_into_dest(self, tmp_path: Path) -> None:
        source = _tree(tmp_path / "src", {"a.txt": "A", "pkg/b.txt": "B"})
        dest = tmp_path / "install" / "thing"

        staged_install(
            str(source), str(dest), ignore=DEFAULT_IGNORE, validate=lambda _: None
        )

        assert _snapshot(dest) == {"a.txt": "A", "pkg/b.txt": "B"}
        # No staging directory left behind beside dest.
        assert _siblings(dest) == ["thing"]

    def test_applies_denylist_at_every_level(self, tmp_path: Path) -> None:
        source = _tree(
            tmp_path / "src",
            {
                "keep.py": "keep",
                ".git/config": "x",
                "__pycache__/keep.cpython.pyc": "x",
                "stale.pyc": "x",
                "pkg.egg-info/PKG": "x",
                "tests/test_x.py": "x",
                "sub/.venv/lib": "x",
                "sub/tests/test_y.py": "x",
                "sub/data.csv": "rows",
            },
        )
        dest = tmp_path / "install" / "thing"

        staged_install(
            str(source), str(dest), ignore=DEFAULT_IGNORE, validate=lambda _: None
        )

        # Everything denylisted is gone — including nested tests/ and .venv —
        # while non-denylisted siblings and data files survive verbatim.
        assert _snapshot(dest) == {"keep.py": "keep", "sub/data.csv": "rows"}

    def test_failing_validate_leaves_existing_dest_byte_identical(
        self, tmp_path: Path
    ) -> None:
        dest = _tree(tmp_path / "install" / "thing", {"old.txt": "GOOD"})
        before = _snapshot(dest)

        source = _tree(tmp_path / "src", {"new.txt": "BAD"})

        def boom(_staging: str) -> None:
            raise RuntimeError("invalid")

        with pytest.raises(RuntimeError):
            staged_install(
                str(source), str(dest), ignore=DEFAULT_IGNORE, validate=boom
            )

        # The existing install is untouched and no staging dir leaked.
        assert _snapshot(dest) == before
        assert _siblings(dest) == ["thing"]

    def test_validate_runs_against_staging_copy(self, tmp_path: Path) -> None:
        source = _tree(tmp_path / "src", {"marker.txt": "hello"})
        dest = tmp_path / "install" / "thing"
        seen: dict[str, str] = {}

        def capture(staging: str) -> None:
            # Validation sees the staged, denylist-applied copy before the swap.
            assert staging != str(source)
            seen["content"] = (Path(staging) / "marker.txt").read_text()

        staged_install(
            str(source), str(dest), ignore=DEFAULT_IGNORE, validate=capture
        )

        assert seen["content"] == "hello"

    def test_existing_dest_replaced_in_place_on_success(self, tmp_path: Path) -> None:
        dest = _tree(tmp_path / "install" / "thing", {"old.txt": "OLD"})
        source = _tree(tmp_path / "src", {"new.txt": "NEW"})

        staged_install(
            str(source), str(dest), ignore=DEFAULT_IGNORE, validate=lambda _: None
        )

        # The old contents are gone; only the new source remains.
        assert _snapshot(dest) == {"new.txt": "NEW"}
        assert _siblings(dest) == ["thing"]

    def test_copy_failure_does_not_clobber_dest_or_leak_staging(
        self, tmp_path: Path
    ) -> None:
        dest = _tree(tmp_path / "install" / "thing", {"old.txt": "GOOD"})
        before = _snapshot(dest)
        missing_source = tmp_path / "does-not-exist"

        with pytest.raises(FileNotFoundError):
            staged_install(
                str(missing_source),
                str(dest),
                ignore=DEFAULT_IGNORE,
                validate=lambda _: None,
            )

        assert _snapshot(dest) == before
        assert _siblings(dest) == ["thing"]

    def test_staging_is_sibling_of_dest(self, tmp_path: Path) -> None:
        source = _tree(tmp_path / "src", {"a.txt": "A"})
        dest = tmp_path / "install" / "thing"
        captured: dict[str, str] = {}

        def capture(staging: str) -> None:
            captured["parent"] = os.path.dirname(os.path.abspath(staging))

        staged_install(
            str(source), str(dest), ignore=DEFAULT_IGNORE, validate=capture
        )

        # Staging shares dest's parent, guaranteeing the same filesystem.
        assert captured["parent"] == str(dest.parent)

    def test_creates_parent_directories_for_dest(self, tmp_path: Path) -> None:
        source = _tree(tmp_path / "src", {"a.txt": "A"})
        dest = tmp_path / "deep" / "nested" / "thing"

        staged_install(
            str(source), str(dest), ignore=DEFAULT_IGNORE, validate=lambda _: None
        )

        assert _snapshot(dest) == {"a.txt": "A"}


class _Conventions:
    def __init__(self, name: str, major: int) -> None:
        self.name = name
        self.major = major


class _Manifest:
    def __init__(self, url: str, version: int) -> None:
        self.url = url
        self.version = version


class TestPathHelpers:
    def test_project_install_dir(self) -> None:
        from vocal.utils import cache_dir

        result = project_install_dir(_Conventions("MYSTD", 2))
        assert result == os.path.join(cache_dir(), "projects", "MYSTD-2")

    def test_pack_install_dir(self) -> None:
        from vocal.utils import cache_dir

        result = pack_install_dir(_Manifest("https://host/packs", 3))
        assert result == os.path.join(
            cache_dir(), "packs", "host-packs", "v3"
        )


class TestDeriveURLSlug:
    def test_slug_is_filesystem_safe(self) -> None:
        assert derive_url_slug("https://host/packs") == "host-packs"

    def test_slug_normalises_case_and_trailing_slash(self) -> None:
        assert derive_url_slug("https://Host/Packs/") == "host-packs"
