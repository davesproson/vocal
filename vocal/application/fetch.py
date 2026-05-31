"""Fetch a vocal project or pack from GitHub and register it.

``vocal fetch <url>`` acquires a GitHub repository's tree — the latest release's
source zipball by default, or a ``git clone`` with ``--git`` — and then
*inspects the downloaded tree* to decide what it got: a ``conventions.yaml`` at
the root is a project; a ``latest/manifest.json`` is a pack; neither is a typed
error. There is no pre-download probe and no version selector in the URL: a pack
repository is a multi-version monorepo, so one fetch retrieves and registers
*every* ``v{Y}`` release it contains.

Both kinds acquire their source through the one boundary in
:mod:`vocal.application.github_source` (:func:`materialize_repo`) and install via
the shared primitives in :mod:`vocal.application.register`, so project and pack
fetch cannot drift apart. The acquisition and classification errors are
re-exported here for callers that catch them by name.
"""

import os
import shutil
import tempfile

import typer

from vocal.application.github_source import (
    FetchError,
    GitCloneFailed,
    GitHubAPIError,
    GitNotInstalled,
    NoReleasesFound,
    NotAGitHubRepo,
    RateLimited,
    RepoNotFound,
    convert_github_repo_to_api_url,
    derive_repo_name,
    fetch_http,
    fetch_with_git,
    get_latest_release,
    materialize_repo,
)
from vocal.application.install import derive_url_slug
from vocal.application.register import (
    install_pack,
    install_project,
    load_registry,
)
from vocal.application.resource import (
    NotAVocalResource,
    ResourceKind,
    classify_resource,
    discover_pack_versions,
)
from vocal.conventions_file import ConventionsFile
from vocal.exceptions import VocalError
from vocal.manifest import normalize_pack_url
from vocal.utils import Printer, TextStyles
from vocal.utils.registry import project_key


TS = TextStyles()
p = Printer()


class ProjectAlreadyFetched(FetchError):
    pass


class ProjectNotFetched(FetchError):
    pass


class PackAlreadyFetched(FetchError):
    pass


class PackNotFetched(FetchError):
    pass


# ---------------------------------------------------------------------------
# Per-kind install of a downloaded tree
# ---------------------------------------------------------------------------


def _install_project_tree(
    download: str,
    url: str,
    *,
    update: bool = False,
    force: bool = False,
) -> None:
    """Install the project tree at ``download`` under ``~/.vocal``, with gating.

    A project's identity is only known after the download, so the
    already-fetched / not-fetched gating runs here, against the registry — the
    source of truth for "is it installed":

    - default: install only if the identity is not already registered, else
      :class:`ProjectAlreadyFetched`;
    - ``--force``: overwrite any existing install;
    - ``--update``: require the identity to already be registered (else
      :class:`ProjectNotFetched`), then refresh it. A re-fetch whose major has
      changed is a *different* identity, so it is simply "not currently fetched"
      under ``--update`` rather than a special identity-changed error.
    """
    # Identity is only known post-download. ConventionsFile.load raises
    # InvalidConventionsFile when conventions.yaml is missing or malformed;
    # that aborts before install_project is reached, with no install touched.
    conventions = ConventionsFile.load(download)
    key = project_key(conventions.name, conventions.major)
    installed = key in load_registry().projects

    if update and not installed:
        raise ProjectNotFetched(
            f"Cannot update '{key}': not currently fetched.",
            hint=f"Run 'vocal fetch {url}' to fetch it for the first time.",
        )

    if installed and not (update or force):
        raise ProjectAlreadyFetched(
            f"Project '{key}' is already fetched.",
            hint="Pass --update to refresh it, or --force to overwrite.",
        )

    install_project(download, force=update or force)


def _install_pack_tree(
    download: str,
    url: str,
    *,
    update: bool = False,
    force: bool = False,
) -> None:
    """Install every version in the pack tree at ``download`` under ``~/.vocal``.

    A pack repository is a multi-version monorepo: :func:`discover_pack_versions`
    enumerates its ``v{Y}/`` release directories (``latest/`` excluded) and each
    is installed and registered independently by looping the shared
    :func:`~vocal.application.register.install_pack` primitive — one atomic swap
    and registry add per version, keyed on the manifest's ``(url, version)``.
    Per-``v{Y}`` manifest consistency is enforced inside ``install_pack`` (a
    ``v{Y}/`` whose manifest disagrees raises :class:`PackInconsistent`).

    Gating is per-repo-URL — any registered version of the URL counts as
    "fetched". This slice implements the default path only; ``--update`` /
    ``--force`` semantics are completed separately:

    - default: install all versions if the URL has no registered versions, else
      :class:`PackAlreadyFetched`.

    Raises:
        FetchError: the pack contains no ``v{Y}/`` release directories.
        PackAlreadyFetched: the URL already has a registered version and neither
            ``--update`` nor ``--force`` was given.
        PackInconsistent: a ``v{Y}/`` directory disagrees with its manifest's
            version (from ``install_pack``).
    """
    versions = discover_pack_versions(download)
    if not versions:
        raise FetchError(
            f"Pack at {url} contains no versioned (v{{Y}}) release directories.",
            hint="The pack repository should hold one or more 'v{Y}/' directories.",
        )

    normalized = normalize_pack_url(url)
    registered = any(u == normalized for (u, _) in load_registry().packs)

    if registered and not (update or force):
        raise PackAlreadyFetched(
            f"Pack {normalized} is already fetched.",
            hint="Pass --update to fetch newly released versions, or --force to "
            "re-install every version.",
        )

    for _version, version_dir in versions:
        install_pack(version_dir, force=update or force)


# ---------------------------------------------------------------------------
# Per-kind fetch (download + install)
# ---------------------------------------------------------------------------


def fetch_project(
    url: str,
    git: bool = False,
    update: bool = False,
    force: bool = False,
) -> None:
    """Download a project and install an owned copy under ``~/.vocal``.

    A thin shell over the shared install primitive: the project is downloaded
    (release-zip extract or ``--git`` clone) into a temporary location via
    :func:`materialize_repo`, then handed to :func:`_install_project_tree`, which
    applies the registry gating and installs an owned copy under
    ``~/.vocal/projects/{name}-{major}``.

    Args:
        url: the URL of the git repository.
        git: clone via git rather than downloading the latest release.
        update: require the project to already exist; refresh it in place.
        force: overwrite any existing fetched copy.
    """
    repo_name = derive_repo_name(url)
    if not repo_name:
        raise FetchError(f"Could not derive a project name from URL: {url}")

    tmp_root = tempfile.mkdtemp(prefix="vocal-fetch-")
    try:
        download = os.path.join(tmp_root, repo_name)
        materialize_repo(url, download, git=git)
        _install_project_tree(download, url, update=update, force=force)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def fetch_pack(
    url: str,
    git: bool = False,
    update: bool = False,
    force: bool = False,
) -> None:
    """Download a GitHub-hosted pack and register every version it contains.

    A thin shell over the shared install primitive: the pack repository is
    downloaded (release-zip extract or ``--git`` clone) into a temporary location
    via :func:`materialize_repo`, then handed to :func:`_install_pack_tree`,
    which discovers the ``v{Y}/`` releases and installs an owned copy of each
    under ``~/.vocal/packs/{slug}/v{Y}`` keyed by ``(url, version)``.

    Args:
        url: the pack's GitHub repository URL.
        git: clone via git rather than downloading the latest release.
        update: pick up newly released versions and refresh existing ones.
        force: re-install every version regardless of what is registered.
    """
    repo_name = derive_repo_name(url)
    if not repo_name:
        raise FetchError(f"Could not derive a pack name from URL: {url}")

    tmp_root = tempfile.mkdtemp(prefix="vocal-fetch-pack-")
    try:
        download = os.path.join(tmp_root, repo_name)
        materialize_repo(url, download, git=git)
        _install_pack_tree(download, url, update=update, force=force)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def fetch(
    url: str,
    git: bool = False,
    update: bool = False,
    force: bool = False,
) -> None:
    """Fetch a project or pack from ``url``, auto-detecting the kind.

    Acquires the repository tree once through :func:`materialize_repo`, then
    classifies the downloaded tree (:func:`classify_resource`) and dispatches:
    a ``conventions.yaml`` root is a project, a ``latest/manifest.json`` root is
    a pack, and neither raises :class:`NotAVocalResource`. The classification is
    a pure inspection of the tree — no pre-download probe.

    Raises:
        NotAVocalResource: the downloaded tree is neither a project nor a pack.
        FetchError and subclasses: from acquisition (release lookup / clone) or
            install.
    """
    repo_name = derive_repo_name(url)
    if not repo_name:
        raise FetchError(f"Could not derive a name from URL: {url}")

    tmp_root = tempfile.mkdtemp(prefix="vocal-fetch-")
    try:
        download = os.path.join(tmp_root, repo_name)
        materialize_repo(url, download, git=git)

        kind = classify_resource(download)
        if kind is ResourceKind.PROJECT:
            _install_project_tree(download, url, update=update, force=force)
        else:
            _install_pack_tree(download, url, update=update, force=force)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def command(
    url: str = typer.Argument(
        help=(
            "The GitHub repository to fetch: a project repo or a pack repo. "
            "The kind is auto-detected by inspecting the downloaded tree."
        )
    ),
    git: bool = typer.Option(
        False,
        "--git",
        help=(
            "Clone the repository with git rather than downloading its latest "
            "release. Required for non-GitHub hosts, private repositories, or "
            "repositories with no published release."
        ),
    ),
    update: bool = typer.Option(
        False,
        "--update",
        help=(
            "Refresh a previously-fetched resource. For a project, re-installs "
            "it; for a pack, picks up newly released versions and refreshes "
            "existing ones. Fails if the resource is not already fetched."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite any existing fetched copy of the resource.",
    ),
) -> None:
    """Fetch a vocal project or pack and register it."""
    try:
        fetch(url, git=git, update=update, force=force)
    except VocalError as e:
        p.print_err(f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} {e.message}")
        if e.hint:
            p.print_err(f"  {e.hint}")
        raise typer.Exit(code=1)
