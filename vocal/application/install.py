"""The shared install primitive and the pure dest/identity helpers.

``register`` and ``fetch`` converge here: both produce a registered resource by
*installing* an owned, normalised copy under ``~/.vocal``. The mechanics that are
easy to get wrong — copy-with-denylist, validate-before-clobber, and the atomic
swap — are concentrated in one deep, project/pack-agnostic function,
:func:`staged_install`, so they are tested in isolation rather than spread across
the two commands.

Alongside it sit the pure path helpers that map a resource's *identity* to its
canonical install directory (:func:`project_install_dir`,
:func:`pack_install_dir`) and :func:`derive_url_slug`, which turns a pack's base
URL into the filesystem-safe slug those paths are keyed on. These do no I/O.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlsplit

from vocal.manifest import normalize_pack_url, versioned_dirname
from vocal.utils import cache_dir

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from vocal.conventions_file import ConventionsFile
    from vocal.manifest import Manifest


# Entries excluded from every install, matched by name at every level of the
# tree. This denylist is what makes "same on-disk state across all entry points"
# a guarantee: ``fetch --git`` carries ``.git``, a local ``register`` may carry
# caches/venvs, and an HTTP ``fetch`` carries neither — all normalise to the
# same shape. Everything not matched here is copied verbatim so runtime imports
# (sibling modules, data files) remain intact.
DENYLIST: tuple[str, ...] = (
    ".git",
    ".venv",
    "__pycache__",
    "*.pyc",
    ".mypy_cache",
    ".pytest_cache",
    "*.egg-info",
    "tests",
)

# An ignore callable in the shape :func:`shutil.copytree` expects:
# ``ignore(dir, names) -> names_to_skip``.
IgnoreFn = Callable[[Any, list[str]], set[str]]

# The canonical ignore callable for installs.
DEFAULT_IGNORE: IgnoreFn = shutil.ignore_patterns(*DENYLIST)


def staged_install(
    source: str,
    dest: str,
    *,
    ignore: IgnoreFn,
    validate: Callable[[str], None],
) -> None:
    """Install ``source`` at ``dest`` via a validated, atomic swap.

    ``source`` is copied into a staging directory created as a *sibling* of
    ``dest`` (so it shares a filesystem and the final move is atomic), applying
    ``ignore`` during the copy. ``validate`` is then run against the staging copy
    — byte-identical to what will be kept — *before* ``dest`` is touched. On
    success ``dest`` is replaced in place (``rmtree`` then ``rename``).

    On any failure (a copy error or a raising ``validate``) the staging directory
    is removed and ``dest`` is left exactly as it was, so a broken or invalid
    source never destroys a working installation.

    Args:
        source: directory to install from.
        dest: canonical install directory; created or replaced in place.
        ignore: a :func:`shutil.copytree`-style ignore callable (see
            :data:`DEFAULT_IGNORE`).
        validate: called with the staging path; raising aborts the install with
            ``dest`` untouched.
    """
    dest = os.path.abspath(dest)
    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)

    # mkdtemp reserves a unique sibling directory; copytree merges into it
    # (dirs_exist_ok) since it is empty. A sibling guarantees the same
    # filesystem, so the closing rename is atomic.
    staging = tempfile.mkdtemp(prefix=".staging-", dir=parent)
    try:
        shutil.copytree(source, staging, ignore=ignore, dirs_exist_ok=True)
        validate(staging)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.rename(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def derive_url_slug(base_url: str) -> str:
    """Derive a stable, filesystem-safe slug from a pack's base URL.

    The slug is the normalised URL's host plus path, lowercased, with runs of
    non-alphanumeric characters collapsed to a single ``-``. Packs fetched from
    the same base URL share a slug directory; their versions coexist as
    ``v{Y}/`` siblings within it.
    """
    parts = urlsplit(normalize_pack_url(base_url))
    raw = f"{parts.netloc}{parts.path}"
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return slug or "pack"


def project_install_dir(conventions: "ConventionsFile") -> str:
    """Return the canonical install directory for a project.

    ``~/.vocal/projects/{name}-{major}`` — identity-named, so the same
    standard+major always lands in one predictable place. The outer name can be
    identity-derived because project import keys off the inner
    ``project_directory`` module, not the repo-root directory name.
    """
    return os.path.join(
        cache_dir(), "projects", f"{conventions.name}-{conventions.major}"
    )


def pack_install_dir(manifest: "Manifest") -> str:
    """Return the canonical install directory for a pack.

    ``~/.vocal/packs/{slug}/v{Y}`` — keyed on the pack's URL slug and version, so
    multiple versions and sources coexist without collision.
    """
    return os.path.join(
        cache_dir(),
        "packs",
        derive_url_slug(manifest.url),
        versioned_dirname(manifest.version),
    )
