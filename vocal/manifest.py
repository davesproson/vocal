"""Pack manifest data model, loader, builder, and product lookup.

A *pack* is a self-describing, independently releasable catalogue of product
definitions. Its identity and routing information live in a ``manifest.json``
that sits alongside the product schema JSONs and ``dataset_schema.json`` inside
each versioned release directory (``v{Y}/``). This module owns:

- the :class:`Manifest` / :class:`ManifestProduct` data model;
- parsing/loading a manifest from a dict or a ``manifest.json`` file, with
  structural validation, ``schema_version`` gating, and rejection of product
  schema paths that escape the versioned directory;
- the ``v{Y}/`` directory-name vs ``manifest.json:version`` equality check,
  surfaced as :class:`PackInconsistent`;
- building a manifest from plain inputs (no project import required);
- product lookup by filename, expanding templated ``file_pattern`` entries with
  a project's ``filecodec``;
- pack-URL normalisation.

The module is pure logic with respect to projects: it never imports a project
module and has no application-layer dependencies. It reads/writes JSON files
only through its explicit ``load_manifest`` / ``Manifest.to_dict`` surface.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from vocal.exceptions import VocalError
from vocal.versioning import VersionConstraint

# The highest manifest ``schema_version`` this build of vocal can parse. A
# manifest declaring a higher value was written by a newer vocal and is
# refused with an "upgrade vocal" hint. This PRD introduces version 1.
SCHEMA_VERSION = 1

# The marker file that identifies a pack release directory. Sits at the root of
# each ``v{Y}/`` (and ``latest/``) directory alongside ``dataset_schema.json``
# and the product schema JSONs.
MANIFEST_FILENAME = "manifest.json"

_VERSIONED_DIR_RE = re.compile(r"^v(?P<version>\d+)$")


class InvalidManifest(VocalError):
    """Raised when a manifest is structurally invalid or malformed."""


class UnsupportedManifestVersion(VocalError):
    """Raised when a manifest declares a ``schema_version`` newer than this vocal."""


class InvalidPackURL(VocalError):
    """Raised when a pack URL carries a query string or fragment."""


class PackInconsistent(VocalError):
    """Raised when a pack's ``v{Y}/`` directory disagrees with ``manifest.json:version``.

    The versioned directory name is an addressing convention; its embedded
    version and the manifest's ``version`` field must agree. Drift between the
    two (including between ``latest/`` and the release it copies) is a hosting
    bug rather than a recoverable state.
    """

    status_code = 500


def normalize_pack_url(url: str) -> str:
    """Return the canonical comparison form of a pack base URL.

    Pack URLs are compared in several places — the ``url`` written into
    ``manifest.json`` at release time, the file's ``vocal_definitions_url`` at
    check time, and the registry's pack key. To avoid silent mismatches from
    trivial differences, all are normalised the same way:

    - scheme and host are lowercased (the path is left case-sensitive);
    - a single trailing slash is stripped;
    - query strings and fragments are rejected.

    Args:
        url: the pack base URL (without a ``v{Y}/`` path component).

    Returns:
        the normalised URL.

    Raises:
        InvalidPackURL: if ``url`` carries a query string or fragment.
    """
    parts = urlsplit(url.strip())

    if parts.query or parts.fragment:
        raise InvalidPackURL(
            f"Pack URL must not contain a query string or fragment: {url!r}",
            "Supply a plain base URL such as 'https://host/packs'.",
        )

    path = parts.path
    if path.endswith("/"):
        path = path[:-1]

    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def versioned_dirname(version: int) -> str:
    """Return the addressing-convention directory name for a pack version."""
    return f"v{version}"


@dataclass(frozen=True)
class ManifestProduct:
    """A single product entry in a pack manifest.

    Carries enough to route a file to its schema without opening every product
    JSON: the product ``name``, its templated ``file_pattern``, and the
    relative ``schema`` path (a sibling of ``manifest.json``).
    """

    name: str
    file_pattern: str
    schema: str

    def matches(self, filename: str, filecodec: Mapping[str, Mapping[str, Any]]) -> bool:
        """Return whether ``filename`` matches this product's ``file_pattern``.

        The ``file_pattern`` is a template; the project's ``filecodec`` supplies
        the regex for each placeholder. Only the basename of ``filename`` is
        considered.
        """
        regex = self.file_pattern.format(
            **{name: codec["regex"] for name, codec in filecodec.items()}
        )
        return re.match(regex, os.path.basename(filename)) is not None

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "file_pattern": self.file_pattern,
            "schema": self.schema,
        }


@dataclass(frozen=True)
class Manifest:
    """A pack manifest: the pack's identity, compatibility, and product index."""

    version: int
    url: str
    requires_standard: VersionConstraint
    products: tuple[ManifestProduct, ...]
    schema_version: int = SCHEMA_VERSION

    def find_product(
        self, filename: str, filecodec: Mapping[str, Mapping[str, Any]]
    ) -> Optional[ManifestProduct]:
        """Return the product whose ``file_pattern`` matches ``filename``, or None.

        Patterns are expanded with ``filecodec`` before matching. The first
        matching product wins.
        """
        for product in self.products:
            if product.matches(filename, filecodec):
                return product
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the ``manifest.json`` wire format."""
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "url": self.url,
            "requires_standard": {
                "name": self.requires_standard.name,
                "major": self.requires_standard.major,
                "min_minor": self.requires_standard.min_minor,
            },
            "products": [product.to_dict() for product in self.products],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a ``manifest.json`` string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Any) -> "Manifest":
        """Parse and validate a manifest from a decoded JSON mapping.

        Raises:
            UnsupportedManifestVersion: if ``schema_version`` exceeds what this
                vocal can parse.
            InvalidManifest: if the manifest is structurally invalid (missing or
                mistyped fields, or a product ``schema`` path that escapes the
                versioned directory).
            InvalidPackURL: if ``url`` carries a query string or fragment.
        """
        if not isinstance(data, Mapping):
            raise InvalidManifest(
                "Manifest must be a JSON object.",
                "Expected an object with 'schema_version', 'version', 'url', "
                "'requires_standard', and 'products'.",
            )

        schema_version = _require(data, "schema_version", int)
        if schema_version > SCHEMA_VERSION:
            raise UnsupportedManifestVersion(
                f"Manifest schema_version {schema_version} is newer than this "
                f"vocal supports (max {SCHEMA_VERSION}).",
                "Upgrade vocal to read this pack.",
            )
        if schema_version < 1:
            raise InvalidManifest(
                f"Manifest schema_version must be >= 1, got {schema_version}.",
            )

        version = _require(data, "version", int)
        url = normalize_pack_url(_require(data, "url", str))
        requires_standard = _parse_requires_standard(data.get("requires_standard"))
        products = _parse_products(data.get("products"))

        return cls(
            version=version,
            url=url,
            requires_standard=requires_standard,
            products=products,
            schema_version=schema_version,
        )


def load_manifest(path: str | os.PathLike[str]) -> Manifest:
    """Load and validate a manifest from a ``manifest.json`` file.

    When the file's parent directory is a versioned release directory
    (``v{Y}/``), the embedded version must equal ``manifest.json:version``; a
    mismatch raises :class:`PackInconsistent`. Directories that are not
    ``v{Y}/`` (e.g. ``latest/``) are not subject to this check — a consumer
    reading ``latest/manifest.json`` learns the version from the manifest
    itself.

    Raises:
        InvalidManifest: if the file is not valid JSON or is structurally
            invalid.
        UnsupportedManifestVersion: if the manifest is too new.
        PackInconsistent: if a ``v{Y}/`` directory name disagrees with the
            manifest's version.
    """
    path = os.fspath(path)
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise InvalidManifest(
            f"Manifest is not valid JSON: {path}",
            f"Parser reported: {err}",
        ) from err

    manifest = Manifest.from_dict(data)

    dirname = os.path.basename(os.path.dirname(os.path.abspath(path)))
    match = _VERSIONED_DIR_RE.match(dirname)
    if match and int(match["version"]) != manifest.version:
        raise PackInconsistent(
            f"Pack directory '{dirname}' disagrees with manifest version "
            f"{manifest.version}.",
            f"Expected the manifest in '{dirname}/' to declare version "
            f"{int(match['version'])}. This is a hosting bug; re-release the pack.",
        )

    return manifest


def build_manifest(
    *,
    version: int,
    url: str,
    standard_name: str,
    standard_major: int,
    min_minor: int,
    products: Iterable[ManifestProduct],
) -> Manifest:
    """Construct a manifest from plain inputs.

    The builder takes the standard's identity (``standard_name`` /
    ``standard_major``, typically sourced from a project's ``conventions.yaml``)
    and ``min_minor`` directly, so it can be exercised independently of any
    project import. ``url`` is normalised before being stored, matching what
    ``vocal release`` writes to disk.
    """
    return Manifest(
        version=version,
        url=normalize_pack_url(url),
        requires_standard=VersionConstraint(
            name=standard_name, major=standard_major, min_minor=min_minor
        ),
        products=tuple(products),
    )


def _require(data: Mapping[str, Any], key: str, type_: type) -> Any:
    """Return ``data[key]``, validating presence and type."""
    if key not in data:
        raise InvalidManifest(f"Manifest is missing required field '{key}'.")
    value = data[key]
    # bool is a subclass of int; reject it where an int is expected.
    if type_ is int and isinstance(value, bool):
        raise InvalidManifest(
            f"Manifest field '{key}' must be of type {type_.__name__}, "
            f"got bool."
        )
    if not isinstance(value, type_):
        raise InvalidManifest(
            f"Manifest field '{key}' must be of type {type_.__name__}, "
            f"got {type(value).__name__}."
        )
    return value


def _parse_requires_standard(value: Any) -> VersionConstraint:
    if not isinstance(value, Mapping):
        raise InvalidManifest(
            "Manifest field 'requires_standard' must be an object with "
            "'name', 'major', and 'min_minor'."
        )
    name = _require(value, "name", str)
    major = _require(value, "major", int)
    min_minor = _require(value, "min_minor", int)
    return VersionConstraint(name=name, major=major, min_minor=min_minor)


def _parse_products(value: Any) -> tuple[ManifestProduct, ...]:
    if not isinstance(value, list):
        raise InvalidManifest("Manifest field 'products' must be a list.")

    products: list[ManifestProduct] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise InvalidManifest(
                f"Manifest product at index {index} must be an object."
            )
        name = _require(entry, "name", str)
        file_pattern = _require(entry, "file_pattern", str)
        schema = _require(entry, "schema", str)
        _validate_schema_path(schema, name)
        products.append(
            ManifestProduct(name=name, file_pattern=file_pattern, schema=schema)
        )
    return tuple(products)


def _validate_schema_path(schema: str, product_name: str) -> None:
    """Reject product schema paths that escape the versioned directory.

    A product ``schema`` must be a relative path resolving to a sibling of
    ``manifest.json`` — no URLs, no absolute paths, and no parent-directory
    traversal.
    """
    if urlsplit(schema).scheme:
        raise InvalidManifest(
            f"Product '{product_name}' schema path must be a relative path, "
            f"not a URL: {schema!r}.",
        )
    if schema.startswith("/") or os.path.isabs(schema) or "\\" in schema:
        raise InvalidManifest(
            f"Product '{product_name}' schema path must be relative, "
            f"got an absolute path: {schema!r}.",
        )
    if ".." in PurePosixPath(schema).parts:
        raise InvalidManifest(
            f"Product '{product_name}' schema path must not escape the "
            f"versioned directory: {schema!r}.",
        )
