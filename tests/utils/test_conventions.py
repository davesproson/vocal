"""Unit tests for vocal/utils/conventions.py.

Covers reading the vocal-managed global attributes off a netCDF file into a
:class:`FileConventions`, with particular attention to the multi-valued
``vocal_project_url`` → ``project_urls`` parsing (single, multiple, absent).
"""

from pathlib import Path

import netCDF4

from vocal.utils.conventions import FileConventions, read_file_conventions


def _make_nc(
    tmp_path: Path,
    *,
    conventions: str | None = None,
    project_url: str | None = None,
    definitions_url: str | None = None,
    definitions_version: int | None = None,
) -> str:
    path = str(tmp_path / "f.nc")
    with netCDF4.Dataset(path, "w") as nc:
        if conventions is not None:
            nc.Conventions = conventions
        if project_url is not None:
            nc.vocal_project_url = project_url
        if definitions_url is not None:
            nc.vocal_definitions_url = definitions_url
        if definitions_version is not None:
            nc.vocal_definitions_version = definitions_version
    return path


class TestProjectUrls:
    def test_single_url_yields_one_element_list(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, project_url="https://host/std.git")
        attrs = read_file_conventions(nc)
        assert attrs.project_urls == ["https://host/std.git"]

    def test_multiple_whitespace_separated_urls_split(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            project_url="https://host/a.git https://host/b.git https://host/c.git",
        )
        attrs = read_file_conventions(nc)
        assert attrs.project_urls == [
            "https://host/a.git",
            "https://host/b.git",
            "https://host/c.git",
        ]

    def test_urls_separated_by_arbitrary_whitespace(self, tmp_path: Path) -> None:
        # Splitting on whitespace collapses runs of spaces/newlines/tabs and
        # ignores leading/trailing padding.
        nc = _make_nc(
            tmp_path,
            project_url="  https://host/a.git \t https://host/b.git\n",
        )
        attrs = read_file_conventions(nc)
        assert attrs.project_urls == ["https://host/a.git", "https://host/b.git"]

    def test_absent_attribute_yields_empty_list(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        attrs = read_file_conventions(nc)
        assert attrs.project_urls == []


class TestOtherAttributes:
    def test_definitions_url_stays_singular(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path, definitions_url="https://host/defs.git")
        attrs = read_file_conventions(nc)
        assert attrs.definitions_url == "https://host/defs.git"

    def test_all_attributes_populated(self, tmp_path: Path) -> None:
        nc = _make_nc(
            tmp_path,
            conventions="MYSTD-1.2 CF-1.8",
            project_url="https://host/std.git",
            definitions_url="https://host/defs.git",
            definitions_version=3,
        )
        attrs = read_file_conventions(nc)
        assert attrs == FileConventions(
            conventions="MYSTD-1.2 CF-1.8",
            project_urls=["https://host/std.git"],
            definitions_url="https://host/defs.git",
            definitions_version=3,
        )

    def test_absent_file_attributes_default(self, tmp_path: Path) -> None:
        nc = _make_nc(tmp_path)
        attrs = read_file_conventions(nc)
        assert attrs == FileConventions(
            conventions=None,
            project_urls=[],
            definitions_url=None,
            definitions_version=None,
        )
