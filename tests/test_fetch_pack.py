"""Integration tests for pack fetching, kind dispatch, and the cache layout.

Pack hosting is faked: a :class:`_FakeRemote` serves a pack directory tree over
a patched ``requests.get`` so the tests exercise the real download → cache →
register flow without a network. Registry I/O is redirected to an in-memory
registry (mirroring ``tests/test_register.py``) so the registered records can be
inspected directly.
"""

import json
import os
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from vocal.application.fetch import (
    PackAlreadyFetched,
    derive_url_slug,
    fetch,
    fetch_pack,
    looks_like_pack,
    parse_pack_url,
)
from vocal.manifest import PackInconsistent
from vocal.utils.registry import Registry


# ---------------------------------------------------------------------------
# Fake remote pack hosting
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.ok = path.is_file()
        self.status_code = 200 if self.ok else 404

    @property
    def content(self) -> bytes:
        return self._path.read_bytes()

    @property
    def text(self) -> str:
        return self._path.read_text()

    def json(self) -> Any:
        return json.loads(self._path.read_text())


class _FakeRemote:
    """Serves files under ``root`` for URLs beginning with ``base``."""

    def __init__(self, root: Path, base: str = "https://host/packs") -> None:
        self.root = root
        self.base = base

    def get(self, url: str) -> _FakeResponse:
        prefix = self.base + "/"
        if not url.startswith(prefix):
            return _FakeResponse(self.root / "__missing__")
        rel = url[len(prefix):]
        return _FakeResponse(self.root / rel)


def _make_remote_pack(
    root: Path,
    version: int = 3,
    url: str = "https://host/packs",
    manifest_version: int | None = None,
    with_latest: bool = True,
) -> Path:
    """Materialise a hostable pack tree under ``root``.

    Writes ``v{version}/`` with a manifest, dataset_schema, and one product
    schema, plus a byte-equal ``latest/`` copy. ``manifest_version`` defaults to
    ``version`` but can be set independently to fabricate a hosting bug.
    """
    if manifest_version is None:
        manifest_version = version

    vdir = root / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "version": manifest_version,
        "url": url,
        "requires_standard": {"name": "MYSTD", "major": 2, "min_minor": 3},
        "products": [
            {"name": "alpha", "file_pattern": "alpha_{date}.nc", "schema": "alpha.json"}
        ],
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest))
    (vdir / "dataset_schema.json").write_text(json.dumps({"type": "object"}))
    (vdir / "alpha.json").write_text(json.dumps({"meta": {"file_pattern": "alpha"}}))

    if with_latest:
        latest = root / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(vdir, latest)

    return root


@pytest.fixture
def registers_into():
    """Redirect register's registry I/O to an in-memory registry."""
    captured: dict = {"registry": Registry()}

    def _load():
        return captured["registry"]

    def _save(r: Registry) -> None:
        captured["registry"] = r

    with patch.multiple(
        "vocal.application.register",
        load_registry=_load,
        save_registry=_save,
    ):
        yield captured


# ---------------------------------------------------------------------------
# URL parsing / slug derivation
# ---------------------------------------------------------------------------


class TestParsePackURL:
    def test_base_url_targets_latest(self) -> None:
        base, vdir, manifest, pinned = parse_pack_url("https://host/packs")
        assert base == "https://host/packs"
        assert vdir == "https://host/packs/latest"
        assert manifest == "https://host/packs/latest/manifest.json"
        assert pinned is None

    def test_pinned_version(self) -> None:
        base, vdir, manifest, pinned = parse_pack_url("https://host/packs/v7")
        assert base == "https://host/packs"
        assert vdir == "https://host/packs/v7"
        assert manifest == "https://host/packs/v7/manifest.json"
        assert pinned == 7

    def test_trailing_slash_ignored(self) -> None:
        base, _, _, pinned = parse_pack_url("https://host/packs/v7/")
        assert base == "https://host/packs"
        assert pinned == 7


class TestDeriveURLSlug:
    def test_slug_is_filesystem_safe(self) -> None:
        assert derive_url_slug("https://host/packs") == "host-packs"

    def test_slug_normalises_case_and_trailing_slash(self) -> None:
        assert derive_url_slug("https://Host/Packs/") == "host-packs"


# ---------------------------------------------------------------------------
# Kind detection
# ---------------------------------------------------------------------------


class TestLooksLikePack:
    def test_true_for_served_manifest(self, tmp_path: Path) -> None:
        remote = _FakeRemote(_make_remote_pack(tmp_path / "remote"))
        with patch("requests.get", side_effect=remote.get):
            assert looks_like_pack("https://host/packs") is True

    def test_false_when_no_manifest(self, tmp_path: Path) -> None:
        remote = _FakeRemote(tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        with patch("requests.get", side_effect=remote.get):
            assert looks_like_pack("https://github.com/u/r") is False


class TestFetchDispatch:
    def test_pack_url_routes_to_fetch_pack(self) -> None:
        with patch("vocal.application.fetch.looks_like_pack", return_value=True), \
            patch("vocal.application.fetch.fetch_pack") as pack, \
            patch("vocal.application.fetch.fetch_project") as project:
            fetch("https://host/packs")
        pack.assert_called_once()
        project.assert_not_called()

    def test_non_pack_routes_to_fetch_project(self) -> None:
        with patch("vocal.application.fetch.looks_like_pack", return_value=False), \
            patch("vocal.application.fetch.fetch_pack") as pack, \
            patch("vocal.application.fetch.fetch_project") as project:
            fetch("https://github.com/u/r")
        project.assert_called_once()
        pack.assert_not_called()

    def test_git_flag_never_probes_for_pack(self) -> None:
        with patch("vocal.application.fetch.looks_like_pack") as probe, \
            patch("vocal.application.fetch.fetch_project") as project:
            fetch("https://example.com/u/r", git=True)
        probe.assert_not_called()
        project.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_pack — cache layout, latest/pinned, --update, PackInconsistent
# ---------------------------------------------------------------------------


class TestFetchPack:
    @contextmanager
    def _patch_packs_dir(self, tmp_path: Path) -> Iterator[None]:
        """Redirect both the fetch download dir and the install root to ``tmp``.

        ``register_pack`` now installs an owned copy under ``cache_dir()``, so
        the install root must be sandboxed alongside fetch's download dir;
        pointing both at the same ``tmp/cache`` keeps the owned copy where the
        fetch flow downloaded it.
        """
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "vocal.application.fetch.get_packs_dir",
                    return_value=str(tmp_path / "cache" / "packs"),
                )
            )
            stack.enter_context(
                patch(
                    "vocal.application.install.cache_dir",
                    return_value=str(tmp_path / "cache"),
                )
            )
            yield

    def test_fetch_latest_caches_and_registers(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        remote = _FakeRemote(_make_remote_pack(tmp_path / "remote", version=3))
        packs_dir = tmp_path / "cache" / "packs"

        with self._patch_packs_dir(tmp_path), patch(
            "requests.get", side_effect=remote.get
        ):
            fetch_pack("https://host/packs")

        # Cache layout: ~/.vocal/packs/<slug>/v{Y}/ with the manifest + schemas.
        target = packs_dir / "host-packs" / "v3"
        assert (target / "manifest.json").is_file()
        assert (target / "dataset_schema.json").is_file()
        assert (target / "alpha.json").is_file()

        # The pack is registered keyed by (url, version) with a matching local_path.
        registry = registers_into["registry"]
        pack = registry.find_pack("https://host/packs", 3)
        assert pack is not None
        assert pack.local_path == str(target)
        assert pack.version == 3

    def test_fetch_pinned_version(self, tmp_path: Path, registers_into: dict) -> None:
        remote = _FakeRemote(
            _make_remote_pack(tmp_path / "remote", version=5, with_latest=False)
        )
        packs_dir = tmp_path / "cache" / "packs"

        with self._patch_packs_dir(tmp_path), patch(
            "requests.get", side_effect=remote.get
        ):
            fetch_pack("https://host/packs/v5")

        assert (packs_dir / "host-packs" / "v5" / "manifest.json").is_file()
        assert registers_into["registry"].find_pack("https://host/packs", 5) is not None

    def test_pinned_version_mismatch_raises_inconsistent(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        # v9/ on disk but its manifest declares version 3 — a hosting bug.
        remote = _FakeRemote(
            _make_remote_pack(
                tmp_path / "remote", version=9, manifest_version=3, with_latest=False
            )
        )
        with self._patch_packs_dir(tmp_path), patch(
            "requests.get", side_effect=remote.get
        ):
            with pytest.raises(PackInconsistent):
                fetch_pack("https://host/packs/v9")

    def test_refetch_without_update_raises_already_fetched(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        remote = _FakeRemote(_make_remote_pack(tmp_path / "remote", version=3))
        with self._patch_packs_dir(tmp_path), patch(
            "requests.get", side_effect=remote.get
        ):
            fetch_pack("https://host/packs")
            with pytest.raises(PackAlreadyFetched):
                fetch_pack("https://host/packs")

    def test_update_redownloads_and_overwrites(
        self, tmp_path: Path, registers_into: dict
    ) -> None:
        remote = _FakeRemote(_make_remote_pack(tmp_path / "remote", version=3))
        packs_dir = tmp_path / "cache" / "packs"

        with self._patch_packs_dir(tmp_path), patch(
            "requests.get", side_effect=remote.get
        ):
            fetch_pack("https://host/packs")

            # Leave a stray file in the cache; --update must overwrite the dir.
            stray = packs_dir / "host-packs" / "v3" / "stray.txt"
            stray.write_text("old")

            fetch_pack("https://host/packs", update=True)

        assert not stray.exists()
        assert (packs_dir / "host-packs" / "v3" / "manifest.json").is_file()
        assert registers_into["registry"].find_pack("https://host/packs", 3) is not None
