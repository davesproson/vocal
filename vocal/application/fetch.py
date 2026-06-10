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
import warnings
from dataclasses import dataclass
from typing import Optional

import typer

from vocal.application.github_source import (
    FetchError,
    derive_repo_name,
    materialize_repo,
)
from vocal.application.register import (
    install_pack,
    install_project,
    load_registry,
)
from vocal.application.resource import (
    ResourceKind,
    classify_resource,
    discover_pack_versions,
)
from vocal.conventions_file import ConventionsFile
from vocal.exceptions import VocalError
from vocal.manifest import normalize_pack_url
from vocal.utils import Printer, TextStyles
from vocal.utils.conventions import read_file_conventions
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


class MissingProjectURL(FetchError):
    """The file declares no ``vocal_project_url`` — nothing to bootstrap."""

    pass


class UnreadableNetCDF(FetchError):
    """The path is not a readable netCDF file (open failed before any fetch)."""

    pass


# ---------------------------------------------------------------------------
# File-driven fetch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchOutcome:
    """One resource's outcome from a file-driven fetch.

    ``role`` is the part the resource plays for the file (``"project"`` or
    ``"pack"``); ``url`` is the declared URL (``None`` when nothing was
    declared); ``outcome`` is one of ``"fetched"`` (newly installed),
    ``"already-present"`` (idempotent skip), or ``"none-declared"`` (the file
    declared no such resource).
    """

    role: str
    url: Optional[str]
    outcome: str


def fetch_for_file(
    filename: str,
    *,
    git: bool = False,
    update: bool = False,
    force: bool = False,
) -> list[FetchOutcome]:
    """Fetch the resources a netCDF file declares about itself.

    Reads the file's vocal-managed global attributes and fetches the project it
    references (``vocal_project_url``) and, if declared, the product-definition
    pack it references (``vocal_definitions_url``). ``vocal_project_url`` is the
    prerequisite: with no project URL there is nothing to bootstrap, so this
    raises :class:`MissingProjectURL`. Each URL is fetched via its own fetch
    primitive directly — no auto-detection — so a wrong-kind URL surfaces as a
    clear failure from the underlying install.

    The project is fetched first, then the pack: the common project-only case is
    handled before the pack, and the pack's compatibility check is meaningful
    only once the project is present. A failure on either fetch propagates
    (fail-fast); anything already installed remains, so a re-run resumes from
    where it stopped rather than starting over.

    Pack handling: if ``vocal_definitions_url`` is present, the pack is fetched
    and recorded as ``fetched``. If absent, the pack outcome is
    ``none-declared`` ("no pack to fetch"). If a ``vocal_definitions_version`` is
    present but its URL is absent — a likely authoring mistake — the project is
    still fetched, the pack is ``none-declared``, and a warning about the
    orphaned version is emitted.

    Idempotent skip: a resource already present is reported as
    ``already-present`` rather than erroring. The underlying
    :class:`ProjectAlreadyFetched` / :class:`PackAlreadyFetched` errors are
    caught and converted to that outcome, so both interfaces are safe to re-run
    — only the missing pieces are fetched. The ``fetch_project`` / ``fetch_pack``
    primitives keep their strict already-fetched gating unchanged.

    Args:
        filename: path to the netCDF file to read.
        git: clone the referenced repos rather than downloading releases.
        update: refresh a previously-fetched resource.
        force: overwrite any existing fetched copy.

    Returns:
        a per-resource list of :class:`FetchOutcome`, project first.

    Raises:
        UnreadableNetCDF: ``filename`` is not a readable netCDF file.
        MissingProjectURL: the file declares no ``vocal_project_url``.
        FetchError and subclasses: from the underlying project or pack fetch.
    """
    try:
        attrs = read_file_conventions(filename)
    except (OSError, FileNotFoundError) as e:
        raise UnreadableNetCDF(
            f"Could not read '{filename}' as a netCDF file.",
            hint="Pass the path to a readable vocal-managed netCDF file.",
        ) from e

    if not attrs.project_url:
        raise MissingProjectURL(
            f"File '{filename}' declares no vocal_project_url; nothing to fetch.",
            hint="The file is not self-describing — supply sources manually with "
            "'vocal fetch <url>'.",
        )

    outcomes: list[FetchOutcome] = []

    # Project first, fail-fast: if this raises, nothing further is attempted and
    # nothing already installed is rolled back. An already-present project is an
    # idempotent skip, not an error.
    try:
        fetch_project(attrs.project_url, git=git, update=update, force=force)
        outcomes.append(FetchOutcome("project", attrs.project_url, "fetched"))
    except ProjectAlreadyFetched:
        outcomes.append(FetchOutcome("project", attrs.project_url, "already-present"))

    if attrs.definitions_url:
        # Honour the attribute name as the declared kind: the pack URL is
        # fetched via the pack primitive directly. A wrong-kind URL surfaces as
        # a clear failure from the underlying install, which propagates here.
        # An already-present pack is an idempotent skip, not an error.
        try:
            fetch_pack(attrs.definitions_url, git=git, update=update, force=force)
            outcomes.append(FetchOutcome("pack", attrs.definitions_url, "fetched"))
        except PackAlreadyFetched:
            outcomes.append(
                FetchOutcome("pack", attrs.definitions_url, "already-present")
            )
    else:
        if attrs.definitions_version is not None:
            warnings.warn(
                f"File '{filename}' carries vocal_definitions_version="
                f"{attrs.definitions_version} but no vocal_definitions_url; "
                "the version is orphaned and no pack can be fetched.",
                stacklevel=2,
            )
        outcomes.append(FetchOutcome("pack", None, "none-declared"))

    return outcomes


# ---------------------------------------------------------------------------
# Per-kind install of a downloaded tree
# ---------------------------------------------------------------------------


def _install_project_tree(
    download: str,
    url: str,
    *,
    update: bool = False,
    force: bool = False,
) -> ResourceKind:
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

    Returns:
        :attr:`ResourceKind.PROJECT` — the kind the caller dispatched on, so it
        can be threaded back out of :func:`fetch` to a caller (e.g. the web Add
        handler) that needs to know what was installed.
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

    install_project(download, force=update or force, url=url)
    return ResourceKind.PROJECT


def _install_pack_tree(
    download: str,
    url: str,
    *,
    update: bool = False,
    force: bool = False,
) -> ResourceKind:
    """Install every version in the pack tree at ``download`` under ``~/.vocal``.

    A pack repository is a multi-version monorepo: :func:`discover_pack_versions`
    enumerates its ``v{Y}/`` release directories (``latest/`` excluded) and each
    is installed and registered independently by looping the shared
    :func:`~vocal.application.register.install_pack` primitive — one atomic swap
    and registry add per version, keyed on the manifest's ``(url, version)``.
    Per-``v{Y}`` manifest consistency is enforced inside ``install_pack`` (a
    ``v{Y}/`` whose manifest disagrees raises :class:`PackInconsistent`).

    Gating is per-repo-URL — any registered version of the URL counts as
    "fetched":

    - default: install all versions if the URL has no registered versions, else
      :class:`PackAlreadyFetched`;
    - ``--update``: require the URL to already be registered (else
      :class:`PackNotFetched`), then re-install every version in the latest
      release — refreshing existing versions and adding newly released ones.
      This is *additive*: a version registered locally but absent from this
      release is simply not in ``versions``, so it is left untouched (no
      pruning);
    - ``--force``: re-install every version in the latest release regardless of
      what is registered, repairing a corrupted or partial install.

    Refresh and add reduce to the same operation: ``install_pack`` keyed on
    ``(url, version)`` with ``force`` overwrites an existing version and creates
    a new one, and never removes a version, so the additive guarantee holds for
    both ``--update`` and ``--force``.

    Returns:
        :attr:`ResourceKind.PACK` — the kind the caller dispatched on, so it can
        be threaded back out of :func:`fetch` to a caller (e.g. the web Add
        handler) that needs to know what was installed.

    Raises:
        FetchError: the pack contains no ``v{Y}/`` release directories.
        PackAlreadyFetched: the URL already has a registered version and neither
            ``--update`` nor ``--force`` was given.
        PackNotFetched: ``--update`` was given but the URL has no registered
            version to update.
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

    if update and not registered:
        raise PackNotFetched(
            f"Cannot update pack {normalized}: not currently fetched.",
            hint=f"Run 'vocal fetch {url}' to fetch it for the first time.",
        )

    if registered and not (update or force):
        raise PackAlreadyFetched(
            f"Pack {normalized} is already fetched.",
            hint="Pass --update to fetch newly released versions, or --force to "
            "re-install every version.",
        )

    for _version, version_dir in versions:
        install_pack(version_dir, force=update or force)
    return ResourceKind.PACK


# ---------------------------------------------------------------------------
# Per-kind fetch (download + install)
# ---------------------------------------------------------------------------


def fetch_project(
    url: str,
    git: bool = False,
    update: bool = False,
    force: bool = False,
) -> ResourceKind:
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

    Returns:
        :attr:`ResourceKind.PROJECT`.
    """
    repo_name = derive_repo_name(url)
    if not repo_name:
        raise FetchError(f"Could not derive a project name from URL: {url}")

    tmp_root = tempfile.mkdtemp(prefix="vocal-fetch-")
    try:
        download = os.path.join(tmp_root, repo_name)
        materialize_repo(url, download, git=git)
        return _install_project_tree(download, url, update=update, force=force)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def fetch_pack(
    url: str,
    git: bool = False,
    update: bool = False,
    force: bool = False,
) -> ResourceKind:
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

    Returns:
        :attr:`ResourceKind.PACK`.
    """
    repo_name = derive_repo_name(url)
    if not repo_name:
        raise FetchError(f"Could not derive a pack name from URL: {url}")

    tmp_root = tempfile.mkdtemp(prefix="vocal-fetch-pack-")
    try:
        download = os.path.join(tmp_root, repo_name)
        materialize_repo(url, download, git=git)
        return _install_pack_tree(download, url, update=update, force=force)
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
) -> ResourceKind:
    """Fetch a project or pack from ``url``, auto-detecting the kind.

    Acquires the repository tree once through :func:`materialize_repo`, then
    classifies the downloaded tree (:func:`classify_resource`) and dispatches:
    a ``conventions.yaml`` root is a project, a ``latest/manifest.json`` root is
    a pack, and neither raises :class:`NotAVocalResource`. The classification is
    a pure inspection of the tree — no pre-download probe.

    Returns:
        the :class:`ResourceKind` that was installed, threaded back from the
        install-dispatch helper. Callers that need to know what a URL pointed at
        (e.g. the web Add handler, to land the user on the right tab) read it
        here rather than re-inspecting the tree.

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
            return _install_project_tree(download, url, update=update, force=force)
        return _install_pack_tree(download, url, update=update, force=force)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def summarise_outcomes(outcomes: list[FetchOutcome]) -> None:
    """Print a per-resource summary of a fetch to stdout.

    Shared across every surface that fetches: the bare-URL and ``--for`` forms
    of ``vocal fetch``, and the ``--fetch`` pre-step of ``vocal check``, so the
    summary a user sees is identical however the fetch was triggered.
    """
    labels = {
        "fetched": f"{TS.OKGREEN}fetched{TS.ENDC}",
        "already-present": f"{TS.OKBLUE}already present{TS.ENDC}",
        "none-declared": "no pack to fetch",
    }
    for outcome in outcomes:
        label = labels.get(outcome.outcome, outcome.outcome)
        target = f" {outcome.url}" if outcome.url else ""
        p.print(f"➜ {outcome.role}:{target} — {label}")


def command(
    url: Optional[str] = typer.Argument(
        None,
        help=(
            "The GitHub repository to fetch: a project repo or a pack repo. "
            "The kind is auto-detected by inspecting the downloaded tree. "
            "Mutually exclusive with --for."
        ),
    ),
    for_file: Optional[str] = typer.Option(
        None,
        "--for",
        metavar="FILE",
        help=(
            "Fetch the resources a netCDF file declares about itself "
            "(its vocal_project_url). Mutually exclusive with the URL argument."
        ),
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
    if (url is None) == (for_file is None):
        p.print_err(
            f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} Provide exactly one of a repository "
            f"URL or --for <file>."
        )
        p.print_err(
            "  Pass a URL to fetch a single resource, or --for <file> to fetch "
            "what a file declares."
        )
        raise typer.Exit(code=1)

    try:
        if for_file is not None:
            outcomes = fetch_for_file(for_file, git=git, update=update, force=force)
            summarise_outcomes(outcomes)
        else:
            assert url is not None  # guaranteed by the guard above
            # An already-present resource is a tidy idempotent outcome here, not
            # an error — mirroring the --for path, which catches the same
            # exceptions in fetch_for_file. The exception type carries the role,
            # so we report it without needing fetch() to have returned a kind.
            try:
                kind = fetch(url, git=git, update=update, force=force)
                role = "project" if kind is ResourceKind.PROJECT else "pack"
                summarise_outcomes([FetchOutcome(role, url, "fetched")])
            except ProjectAlreadyFetched:
                summarise_outcomes([FetchOutcome("project", url, "already-present")])
            except PackAlreadyFetched:
                summarise_outcomes([FetchOutcome("pack", url, "already-present")])
    except VocalError as e:
        p.print_err(f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} {e.message}")
        if e.hint:
            p.print_err(f"  {e.hint}")
        raise typer.Exit(code=1)
