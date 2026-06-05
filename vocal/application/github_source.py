"""GitHub source acquisition — the single boundary both fetch kinds sit on.

A vocal resource (a project or a pack) is published from a GitHub repository.
Acquiring its tree means one of two things: download and unpack the latest
release's source zipball, or ``git clone`` the repository. This module hides
both behind one public interface — :func:`materialize_repo` — so the rest of the
codebase never touches the GitHub API, archive extraction, or the git command
line directly.

Everything network- or subprocess-facing is concentrated here: the release
lookup (:func:`get_latest_release`), the zip download/extract/flatten
(:func:`fetch_http`), the clone (:func:`fetch_with_git`), and the typed errors
they raise. ``fetch`` calls :func:`materialize_repo` and works only with the
populated directory tree it leaves behind, so project and pack fetch acquire
their source identically and cannot drift apart.
"""

import os
import subprocess
import zipfile

import requests

from vocal.exceptions import VocalError
from vocal.utils import flip_to_dir


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
    Fetch a repository via the git command line.

    Clones into `target` (which must not already exist).

    Raises:
        GitNotInstalled: if `git` is not on PATH.
        GitCloneFailed: if `git clone` exits non-zero. Message carries the
            captured stderr verbatim so users see git's own diagnostic.
    """

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
    Get the latest published release for a GitHub repository by listing
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
    Fetch a repository over HTTP by downloading the latest GitHub release zip
    and extracting it into `target`.

    Raises:
        NotAGitHubRepo, RepoNotFound, NoReleasesFound, RateLimited,
        GitHubAPIError: see get_latest_release.
        FetchError: wrapping a download or extraction failure (Tier B).
    """

    api_url = convert_github_repo_to_api_url(url)
    release_info = get_latest_release(api_url)

    try:
        zip_url = release_info["zipball_url"]
        _tag_name = release_info["tag_name"]
    except (KeyError, TypeError) as e:
        raise GitHubAPIError(f"Unexpected GitHub release payload: missing {e}")

    # Extract into the target's parent (a caller-owned temp directory), so the
    # download is self-contained and never touches the install location.
    work_dir = os.path.dirname(os.path.abspath(target))
    os.makedirs(work_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, f".{os.path.basename(target)}.zip")

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
                with flip_to_dir(work_dir):
                    zip_ref.extractall(".")
        except zipfile.BadZipFile as e:
            raise FetchError(f"Downloaded release archive is not a valid zip: {e}")

        extracted_path = os.path.join(work_dir, extracted_root)
        if extracted_path != target:
            os.rename(extracted_path, target)
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def materialize_repo(url: str, target: str, *, git: bool) -> None:
    """Populate ``target`` with the contents of the repository at ``url``.

    The single public interface for GitHub source acquisition. ``target`` must
    not already exist; on return it holds the repository's tree, acquired either
    from the latest release's source zipball (the default) or by cloning the
    repository (``git=True``). Both project and pack fetch sit on this boundary,
    so the two acquire their source identically.

    Args:
        url: the repository URL.
        target: the directory to populate (must not already exist).
        git: clone via the git command line instead of downloading the latest
            release zip. Required for non-GitHub hosts, private repositories, or
            repositories with no published release.

    Raises:
        NotAGitHubRepo, RepoNotFound, NoReleasesFound, RateLimited,
        GitHubAPIError: from the release path (see :func:`fetch_http`).
        GitNotInstalled, GitCloneFailed: from the clone path (see
            :func:`fetch_with_git`).
        FetchError: wrapping a network/extraction failure.
    """
    if git:
        fetch_with_git(url, target)
    else:
        fetch_http(url, target)
