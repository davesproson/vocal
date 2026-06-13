import textwrap

import pytest

from vocal.pack_config import InvalidPackConfig, PackConfig
from vocal.versioning import VersionConstraint


def _write(directory, text: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pack.yaml").write_text(textwrap.dedent(text))
    return str(directory)


class TestLoad:
    def test_loads_full_config(self, tmp_path) -> None:
        repo = _write(
            tmp_path,
            r"""
            filecodec:
              date:
                regex: '\d{8}'
              platform:
                regex: '[a-z]+'
            satisfies_standards:
              - MYSTD-2.4+
              - OTHER-1.0+
            url: https://host/packs
            """,
        )
        cfg = PackConfig.load(repo)
        assert cfg.filecodec == {
            "date": {"regex": r"\d{8}"},
            "platform": {"regex": r"[a-z]+"},
        }
        assert cfg.satisfies_standards == (
            VersionConstraint("MYSTD", 2, 4),
            VersionConstraint("OTHER", 1, 0),
        )
        assert cfg.url == "https://host/packs"

    def test_satisfies_standards_and_url_are_optional(self, tmp_path) -> None:
        repo = _write(
            tmp_path,
            r"""
            filecodec:
              date:
                regex: '\d{8}'
            """,
        )
        cfg = PackConfig.load(repo)
        assert cfg.satisfies_standards == ()
        assert cfg.url is None

    def test_raises_when_file_missing(self, tmp_path) -> None:
        with pytest.raises(InvalidPackConfig):
            PackConfig.load(str(tmp_path))

    def test_raises_on_invalid_yaml(self, tmp_path) -> None:
        (tmp_path / "pack.yaml").write_text("filecodec: [unclosed")
        with pytest.raises(InvalidPackConfig):
            PackConfig.load(str(tmp_path))

    def test_raises_when_filecodec_missing(self, tmp_path) -> None:
        repo = _write(tmp_path, "url: https://host/packs\n")
        with pytest.raises(InvalidPackConfig):
            PackConfig.load(repo)

    def test_raises_when_filecodec_entry_has_no_regex(self, tmp_path) -> None:
        repo = _write(
            tmp_path,
            """
            filecodec:
              date:
                pattern: nope
            """,
        )
        with pytest.raises(InvalidPackConfig):
            PackConfig.load(repo)

    def test_raises_on_malformed_satisfies_standard(self, tmp_path) -> None:
        repo = _write(
            tmp_path,
            r"""
            filecodec:
              date:
                regex: '\d{8}'
            satisfies_standards:
              - not-a-constraint
            """,
        )
        with pytest.raises(InvalidPackConfig):
            PackConfig.load(repo)

    def test_raises_when_satisfies_standards_not_a_list(self, tmp_path) -> None:
        repo = _write(
            tmp_path,
            r"""
            filecodec:
              date:
                regex: '\d{8}'
            satisfies_standards: MYSTD-2.4+
            """,
        )
        with pytest.raises(InvalidPackConfig):
            PackConfig.load(repo)

    def test_raises_when_url_not_a_string(self, tmp_path) -> None:
        repo = _write(
            tmp_path,
            r"""
            filecodec:
              date:
                regex: '\d{8}'
            url:
              - https://host/packs
            """,
        )
        with pytest.raises(InvalidPackConfig):
            PackConfig.load(repo)
