"""Post-download classification of a fetched repository tree.

``vocal fetch`` acquires a repository's tree (via the latest release zipball or
a ``git clone``) and must then decide what it just downloaded — a vocal project
or a vocal pack — without a second network round-trip. That decision, and the
enumeration of a pack's versioned release directories, are pure functions over
the on-disk tree, gathered here so they are testable without any network or
subprocess.

Two markers, mutually exclusive in practice, distinguish the kinds at the tree
root:

- a ``conventions.yaml`` ⇒ a project (a pack repo carries none);
- a ``latest/manifest.json`` ⇒ a pack (the byte-identical copy of the highest
  ``v{Y}`` release that every pack repo ships).

A tree carrying neither is not a vocal resource and is rejected with a typed
error, so a mistyped URL or a wrong repository fails clearly rather than
silently registering nothing.
"""

from __future__ import annotations

import os
import re
from enum import Enum

from vocal.application.github_source import FetchError
from vocal.conventions_file import CONVENTIONS_FILENAME
from vocal.manifest import MANIFEST_FILENAME

# A pack's release directories are named ``v{Y}``; ``latest/`` is excluded by
# not matching this pattern, so discovery never registers the redundant copy.
_VERSIONED_DIR_RE = re.compile(r"^v(?P<version>\d+)$")

# The relative marker a pack repo always carries: a ``latest/`` directory with a
# ``manifest.json``, the byte-identical copy of its highest ``v{Y}`` release.
_PACK_MARKER = os.path.join("latest", MANIFEST_FILENAME)


class NotAVocalResource(FetchError):
    """A downloaded tree is neither a vocal project nor a vocal pack."""


class ResourceKind(Enum):
    """The kind of vocal resource a downloaded tree contains."""

    PROJECT = "project"
    PACK = "pack"


def classify_resource(root: str) -> ResourceKind:
    """Classify the repository tree at ``root`` as a project or a pack.

    A pure marker-file check, no network or subprocess: ``conventions.yaml`` at
    the root means a project; ``latest/manifest.json`` means a pack. The two
    markers do not co-occur in practice (a pack repo carries no
    ``conventions.yaml``), and the project marker is checked first.

    Args:
        root: the populated repository tree.

    Returns:
        the resource kind.

    Raises:
        NotAVocalResource: ``root`` carries neither marker.
    """
    if os.path.isfile(os.path.join(root, CONVENTIONS_FILENAME)):
        return ResourceKind.PROJECT
    if os.path.isfile(os.path.join(root, _PACK_MARKER)):
        return ResourceKind.PACK
    raise NotAVocalResource(
        f"{root} is not a vocal project or pack.",
        hint=(
            f"Expected a '{CONVENTIONS_FILENAME}' (project) or a "
            f"'{_PACK_MARKER}' (pack) at the repository root. Check the URL "
            "points at a vocal project or pack repository."
        ),
    )


def discover_pack_versions(root: str) -> list[tuple[int, str]]:
    """Enumerate a pack tree's versioned release directories.

    Returns one ``(version, path)`` pair per ``v{Y}/`` directory directly under
    ``root``, sorted ascending by version. ``latest/`` is excluded — it does not
    match the ``v{Y}`` pattern — so it is never registered; it remains a repo
    artifact. A pure listing of the tree, with no network or subprocess.

    Args:
        root: the populated pack repository tree.

    Returns:
        ``(version, directory_path)`` pairs, sorted by ascending version.
    """
    discovered: list[tuple[int, str]] = []
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        match = _VERSIONED_DIR_RE.match(entry)
        if match is None:
            continue
        discovered.append((int(match["version"]), path))
    return sorted(discovered)
