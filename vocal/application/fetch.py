"""Fetch a vocal project or pack and register it.

``vocal fetch <url>`` auto-detects whether ``<url>`` refers to a project or a
pack by inspecting the resource's marker file: a downloaded git repo carrying a
``conventions.yaml`` is a project; a URL serving a ``manifest.json`` is a pack.

Pack URL grammar carries the version in the path: ``vocal fetch <base>`` fetches
``<base>/latest/``; ``vocal fetch <base>/v{Y}`` fetches that pinned version.
There is no ``--version`` flag.
"""

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import typer
import requests

from vocal.application.install import derive_url_slug
from vocal.application.register import (
    register_pack,
    register_project,
    CannotRegisterProjectError,
)
from vocal.conventions_file import ConventionsFile, InvalidConventionsFile
from vocal.exceptions import VocalError
from vocal.manifest import (
    Manifest,
    MANIFEST_FILENAME,
    PackInconsistent,
    versioned_dirname,
)
from vocal.utils import cache_dir, flip_to_dir, Printer, TextStyles
from vocal.utils.registry import project_key


TS = TextStyles()
p = Printer()


class FetchError(VocalError):
    """Base class for user-facing fetch failures."""


class NotAGitHubRepo(FetchError):
    pass


class GitHubAPIError(FetchError):
    pass


class RepoNotFound(FetchError):
    pass


class NoReleasesFound(FetchError):
    pass


class RateLimited(FetchError):
    pass


class GitNotInstalled(FetchError):
    pass


class GitCloneFailed(FetchError):
    pass


class ProjectAlreadyFetched(FetchError):
    pass


class ProjectNotFetched(FetchError):
    pass


class ProjectIdentityChanged(FetchError):
    """Raised when a re-fetched project's ``{name}-{major}`` differs from the
    one previously registered at the same local cache directory."""


class PackAlreadyFetched(FetchError):
    pass


def get_projects_dir() -> str:
    """
    Get the directory where projects are stored. On Unix systems, this is
    `~/.vocal/projects`. Creates the directory if it does not exist.

    Returns:
        str: The path to the projects directory.
    """
    projects_dir = os.path.join(cache_dir(), "projects")
    os.makedirs(projects_dir, exist_ok=True)
    return projects_dir


def derive_repo_name(url: str) -> str:
    """
    Derive a stable on-disk directory name from a repository URL.

    Strips any trailing slash and any `.git` suffix so that
    `https://github.com/u/r`, `https://github.com/u/r/`, and
    `git@github.com:u/r.git` all map to `r`.
    """
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def fetch_with_git(url: str, target: str) -> None:
    """
    Fetch a project from a git repository, via the git command line.

    Clones into `target` (which must not already exist).

    Raises:
        GitNotInstalled: if `git` is not on PATH.
        GitCloneFailed: if `git clone` exits non-zero. Message carries the
            captured stderr verbatim so users see git's own diagnostic.
    """
    print(f"Cloning project from git repository: {url}")

    try:
        result = subprocess.run(
            ["git", "clone", url, target],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise GitNotInstalled(
            "git executable not found on PATH.",
            hint=(
                "Install git, or omit --git to fetch the latest release "
                "from a public GitHub repository over HTTPS."
            ),
        )

    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "git clone exited non-zero."
        raise GitCloneFailed(f"git clone failed: {stderr}")


def convert_github_repo_to_api_url(url: str) -> str:
    """
    Convert a GitHub repository URL to the GitHub API URL.
    For example,
    `https://github.com/username/repo` -> `https://api.github.com/repos/username/repo`

    Args:
        url (str): The URL of the GitHub repository.

    Returns:
        str: The URL of the GitHub API.

    Raises:
        NotAGitHubRepo: if the URL does not point at github.com.
    """
    if "github.com" not in url:
        raise NotAGitHubRepo(
            f"Only GitHub repositories are supported in HTTP mode: {url}",
            hint=(
                "Pass --git to clone from a non-GitHub host or a private "
                "repository using the git command line."
            ),
        )

    parts = url.rstrip("/").split("/")
    user = parts[-2]
    repo = parts[-1]
    if repo.endswith(".git"):
        repo = repo[:-4]

    return f"https://api.github.com/repos/{user}/{repo}"


def get_latest_release(api_url: str) -> dict:
    """
    Get the latest published release for a GitHub project by listing
    `/releases` and picking the first entry.

    Using the list endpoint (rather than `/releases/latest`) lets us
    distinguish "repo doesn't exist" (404) from "repo has no releases"
    (200 with an empty array).

    Raises:
        RepoNotFound: GitHub returned 404.
        NoReleasesFound: GitHub returned 200 with no releases.
        RateLimited: GitHub returned 403 with a rate-limit message.
        GitHubAPIError: any other non-2xx response.
        FetchError: wrapping a network-level failure (Tier B).
    """
    try:
        response = requests.get(api_url + "/releases")
    except requests.RequestException as e:
        raise FetchError(f"Network error contacting GitHub: {e}")

    if response.status_code == 404:
        raise RepoNotFound(
            f"GitHub repository not found: {api_url}",
            hint="Check the URL is correct, or use --git for a private repo.",
        )

    if response.status_code == 403:
        body = (response.text or "").lower()
        if "rate limit" in body:
            raise RateLimited(
                "GitHub API rate limit exceeded.",
                hint=(
                    "Wait a few minutes and try again, or use --git which "
                    "authenticates via your local git configuration."
                ),
            )
        raise GitHubAPIError(
            f"GitHub API returned 403 Forbidden for {api_url}.",
            hint="The repository may be private; try --git.",
        )

    if not response.ok:
        excerpt = (response.text or "").strip().splitlines()[:1]
        detail = excerpt[0] if excerpt else ""
        raise GitHubAPIError(
            f"GitHub API returned {response.status_code} for {api_url}: {detail}"
        )

    try:
        releases = response.json()
    except ValueError as e:
        raise GitHubAPIError(f"GitHub API returned invalid JSON: {e}")

    if not isinstance(releases, list) or len(releases) == 0:
        raise NoReleasesFound(
            f"Repository has no published releases: {api_url}",
            hint=(
                "Publish a release on GitHub, or use --git to clone the "
                "default branch directly."
            ),
        )

    return releases[0]


def fetch_http(url: str, target: str) -> None:
    """
    Fetch a project over HTTP by downloading the latest GitHub release zip
    and extracting it into `target`.

    Raises:
        NotAGitHubRepo, RepoNotFound, NoReleasesFound, RateLimited,
        GitHubAPIError: see get_latest_release.
        FetchError: wrapping a download or extraction failure (Tier B).
    """
    print(f"Fetching project from HTTP repository: {url}")

    api_url = convert_github_repo_to_api_url(url)
    release_info = get_latest_release(api_url)

    try:
        zip_url = release_info["zipball_url"]
        tag_name = release_info["tag_name"]
    except (KeyError, TypeError) as e:
        raise GitHubAPIError(f"Unexpected GitHub release payload: missing {e}")

    print(f"Fetching release {tag_name}")

    projects_dir = get_projects_dir()
    zip_path = os.path.join(projects_dir, f".{os.path.basename(target)}.zip")

    try:
        try:
            response = requests.get(zip_url)
            response.raise_for_status()
        except requests.RequestException as e:
            raise FetchError(f"Failed to download release archive: {e}")

        with open(zip_path, "wb") as f:
            f.write(response.content)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                names = zip_ref.namelist()
                if not names:
                    raise FetchError("Downloaded release archive is empty.")
                extracted_root = names[0].split("/")[0]
                with flip_to_dir(projects_dir):
                    zip_ref.extractall(".")
        except zipfile.BadZipFile as e:
            raise FetchError(f"Downloaded release archive is not a valid zip: {e}")

        extracted_path = os.path.join(projects_dir, extracted_root)
        if extracted_path != target:
            os.rename(extracted_path, target)
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def fetch_project(
    url: str,
    git: bool = False,
    update: bool = False,
    force: bool = False,
) -> None:
    """
    Download a project from a git repository and register it with Vocal.

    Args:
        url: The URL of the git repository.
        git: Whether to use git to clone (rather than HTTPS release download).
        update: Require the project to already exist; refresh it in place.
        force: Overwrite any existing fetched copy.
    """
    repo_name = derive_repo_name(url)
    if not repo_name:
        raise FetchError(f"Could not derive a project name from URL: {url}")

    projects_dir = get_projects_dir()
    target = os.path.join(projects_dir, repo_name)
    exists = os.path.exists(target)

    if update and not exists:
        raise ProjectNotFetched(
            f"Cannot update '{repo_name}': not currently fetched.",
            hint=f"Run 'vocal fetch {url}' to fetch it for the first time.",
        )

    if exists and not (update or force):
        raise ProjectAlreadyFetched(
            f"Project already fetched at {target}.",
            hint="Pass --update to refresh it, or --force to overwrite.",
        )

    # On --update, remember the project's prior identity so we can refuse to
    # silently re-register a different {name}-{major} under the same cache dir.
    prior_key: Optional[str] = None
    if exists and update:
        try:
            prior = ConventionsFile.load(target)
            prior_key = project_key(prior.name, prior.major)
        except InvalidConventionsFile:
            prior_key = None

    if exists:
        shutil.rmtree(target)

    if git:
        fetch_with_git(url, target)
    else:
        fetch_http(url, target)

    try:
        # Validate that the fetched tree looks like a vocal project before
        # registering. ConventionsFile.load raises InvalidConventionsFile when
        # conventions.yaml is missing or malformed.
        conventions = ConventionsFile.load(target)

        if prior_key is not None:
            new_key = project_key(conventions.name, conventions.major)
            if new_key != prior_key:
                raise ProjectIdentityChanged(
                    f"Re-fetched project is '{new_key}', but '{prior_key}' was "
                    f"registered at {target}.",
                    hint=(
                        "A new major is a new project: fetch it to a fresh URL "
                        "and register it under its own key rather than updating "
                        "in place."
                    ),
                )

        register_project(target, force=True)
    except CannotRegisterProjectError as e:
        shutil.rmtree(target, ignore_errors=True)
        raise FetchError(f"Failed to register project: {e}")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Packs
# ---------------------------------------------------------------------------

_PINNED_RE = re.compile(r"^v(\d+)$")


def get_packs_dir() -> str:
    """Return the local cache directory for packs (``~/.vocal/packs``).

    Creates the directory if it does not exist.
    """
    packs_dir = os.path.join(cache_dir(), "packs")
    os.makedirs(packs_dir, exist_ok=True)
    return packs_dir


def parse_pack_url(url: str) -> tuple[str, str, str, Optional[int]]:
    """Split a pack fetch URL into its addressing components.

    ``<base>`` resolves to the ``latest/`` release; ``<base>/v{Y}`` pins a
    version. Returns ``(base_url, version_dir_url, manifest_url, pinned)`` where
    ``pinned`` is the requested version for a ``v{Y}`` URL or ``None`` for a
    bare base URL.
    """
    trimmed = url.strip().rstrip("/")
    parts = urlsplit(trimmed)
    segments = parts.path.split("/")
    last = segments[-1] if segments else ""

    match = _PINNED_RE.match(last)
    if match:
        pinned: Optional[int] = int(match.group(1))
        base_path = "/".join(segments[:-1])
        base_url = urlunsplit((parts.scheme, parts.netloc, base_path, "", ""))
        version_dir_url = trimmed
    else:
        pinned = None
        base_url = trimmed
        version_dir_url = f"{trimmed}/latest"

    manifest_url = f"{version_dir_url}/{MANIFEST_FILENAME}"
    return base_url, version_dir_url, manifest_url, pinned


def _http_get(url: str) -> requests.Response:
    """GET ``url``, raising :class:`FetchError` on network or HTTP failure."""
    try:
        response = requests.get(url)
    except requests.RequestException as e:
        raise FetchError(f"Network error fetching {url}: {e}")
    if not response.ok:
        raise FetchError(
            f"Failed to fetch {url}: HTTP {response.status_code}.",
            hint="Check the pack URL is correct and the host is reachable.",
        )
    return response


def _load_remote_manifest(manifest_url: str) -> Manifest:
    """Download and parse a pack manifest from ``manifest_url``."""
    response = _http_get(manifest_url)
    try:
        data = response.json()
    except ValueError as e:
        raise FetchError(f"Pack manifest at {manifest_url} is not valid JSON: {e}")
    try:
        return Manifest.from_dict(data)
    except VocalError as e:
        raise FetchError(
            f"Invalid pack manifest at {manifest_url}: {e.message}", hint=e.hint
        )


def looks_like_pack(url: str) -> bool:
    """Probe ``url`` for a pack manifest, returning whether one was found.

    Used to auto-detect resource kind: a URL serving a valid ``manifest.json``
    (at ``<base>/latest/`` or ``<base>/v{Y}/``) is a pack; anything else is
    treated as a project git URL.
    """
    _, _, manifest_url, _ = parse_pack_url(url)
    try:
        response = requests.get(manifest_url)
    except requests.RequestException:
        return False
    if not response.ok:
        return False
    try:
        Manifest.from_dict(response.json())
    except Exception:
        return False
    return True


def fetch_pack(url: str, update: bool = False, force: bool = False) -> None:
    """Fetch a pack and register it.

    Downloads the pack's ``manifest.json``, ``dataset_schema.json``, and product
    schema JSONs into ``~/.vocal/packs/<url-slug>/v{Y}/`` (where ``Y`` is the
    manifest's canonical version), then registers it keyed by ``(url, version)``.

    Pack versions are immutable, so re-fetching an already-cached version
    requires ``--update`` (or ``--force``); without one this raises
    :class:`PackAlreadyFetched`.

    Raises:
        PackInconsistent: the pinned ``v{Y}`` URL disagrees with the manifest's
            version.
        PackAlreadyFetched: the version is already cached and neither ``--update``
            nor ``--force`` was given.
        FetchError: on a network/HTTP failure or a malformed manifest.
    """
    base_url, version_dir_url, manifest_url, pinned = parse_pack_url(url)

    manifest = _load_remote_manifest(manifest_url)
    version = manifest.version

    if pinned is not None and pinned != version:
        raise PackInconsistent(
            f"Pack at {version_dir_url} declares version {version}, "
            f"not v{pinned}.",
            "The versioned directory name and manifest version must agree; "
            "this is a hosting bug.",
        )

    slug_dir = os.path.join(get_packs_dir(), derive_url_slug(base_url))
    target = os.path.join(slug_dir, versioned_dirname(version))
    exists = os.path.exists(target)

    if exists and not (update or force):
        raise PackAlreadyFetched(
            f"Pack {base_url} version {version} is already fetched at {target}.",
            hint="Pack versions are immutable; pass --update to re-download it.",
        )

    os.makedirs(slug_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".fetch-", dir=slug_dir)
    try:
        _write_text(os.path.join(staging, MANIFEST_FILENAME), manifest.to_json())
        _download_file(
            f"{version_dir_url}/dataset_schema.json",
            os.path.join(staging, "dataset_schema.json"),
        )
        for product in manifest.products:
            _download_file(
                f"{version_dir_url}/{product.schema}",
                os.path.join(staging, product.schema),
            )

        if exists:
            shutil.rmtree(target)
        os.rename(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    register_pack(target, force=True)


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _download_file(url: str, dest: str) -> None:
    """Download ``url`` to ``dest``, creating parent directories as needed."""
    response = _http_get(url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(response.content)


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

    A URL serving a pack ``manifest.json`` is fetched as a pack; anything else
    is fetched as a project git repository. ``--git`` forces the project path
    (a git clone is always a project).
    """
    if not git and looks_like_pack(url):
        fetch_pack(url, update=update, force=force)
    else:
        fetch_project(url, git=git, update=update, force=force)


def command(
    url: str = typer.Argument(
        help=(
            "The resource to fetch: a project git URL, or a pack base URL "
            "(<base> for the latest release, <base>/v{Y} for a pinned version). "
            "The kind is auto-detected."
        )
    ),
    git: bool = typer.Option(
        False,
        "--git",
        help=(
            "Use git to clone the repository. Requires git to be installed. "
            "This is mandatory if using a repository other than GitHub, or "
            "if the repository is private. Always fetches a project."
        ),
    ),
    update: bool = typer.Option(
        False,
        "--update",
        help=(
            "Refresh a previously-fetched resource. For a project, re-clones "
            "and re-registers it; for a pack, re-downloads the immutable "
            "version. Fails if the resource is not already fetched."
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
