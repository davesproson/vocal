"""Fetch a vocal project from a git repository."""

import os
import shutil
import subprocess
import zipfile
from typing import Optional

import typer
import requests
import yaml

from vocal.application.register import register_project, CannotRegisterProjectError
from vocal.utils import cache_dir, flip_to_dir, Printer, TextStyles


TS = TextStyles()
p = Printer()


class FetchError(Exception):
    """Base class for user-facing fetch failures."""

    def __init__(self, message: str, hint: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n  {self.hint}"
        return self.message


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


class InvalidVocalYaml(FetchError):
    pass


class ProjectPathMissing(FetchError):
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


def read_vocal_yaml(project_dir: str) -> dict:
    """
    Read and validate the `vocal.yaml` config at the root of a fetched
    project directory. Returns the validated `vocal:` block as a dict.

    Raises:
        InvalidVocalYaml: file missing, not valid YAML, or missing a
            required key.
        ProjectPathMissing: a referenced path doesn't exist on disk.
    """
    yaml_file = os.path.join(project_dir, "vocal.yaml")

    try:
        with open(yaml_file, "r") as f:
            try:
                raw = yaml.load(f, Loader=yaml.Loader)
            except yaml.YAMLError as e:
                raise InvalidVocalYaml(f"vocal.yaml is not valid YAML: {e}")
    except FileNotFoundError:
        raise InvalidVocalYaml(
            f"vocal.yaml not found at {yaml_file}.",
            hint="The repository does not look like a vocal project.",
        )

    if not isinstance(raw, dict) or "vocal" not in raw:
        raise InvalidVocalYaml("vocal.yaml is missing the top-level 'vocal:' block.")

    data = raw["vocal"]
    if not isinstance(data, dict):
        raise InvalidVocalYaml("'vocal:' block in vocal.yaml must be a mapping.")

    for key in ("project_directory", "products_directory", "conventions"):
        if key not in data:
            raise InvalidVocalYaml(f"vocal.yaml is missing required key: {key}")

    for key in ("project_directory", "products_directory"):
        path = os.path.join(project_dir, data[key])
        if not os.path.isdir(path):
            raise ProjectPathMissing(
                f"vocal.yaml '{key}' points at a directory that does not exist: {path}"
            )

    return data


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

    if exists:
        shutil.rmtree(target)

    if git:
        fetch_with_git(url, target)
    else:
        fetch_http(url, target)

    try:
        data = read_vocal_yaml(target)

        register_project(
            project_path=os.path.join(target, data["project_directory"]),
            definitions=os.path.join(target, data["products_directory"]),
            conventions_string=data["conventions"],
            force=True,
        )
    except CannotRegisterProjectError as e:
        shutil.rmtree(target, ignore_errors=True)
        raise FetchError(f"Failed to register project: {e}")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def command(
    url: str = typer.Argument(
        help="The URL of the git repository to fetch the project from."
    ),
    git: bool = typer.Option(
        False,
        "--git",
        help=(
            "Use git to clone the repository. Requires git to be installed. "
            "This is mandatory if using a repository other than GitHub, or "
            "if the repository is private."
        ),
    ),
    update: bool = typer.Option(
        False,
        "--update",
        help=(
            "Refresh a previously-fetched project. Fails if the project is "
            "not already fetched."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite any existing fetched copy of the project.",
    ),
) -> None:
    """Fetch a vocal project from a git repository."""
    try:
        fetch_project(url, git=git, update=update, force=force)
    except FetchError as e:
        p.print_err(f"{TS.BOLD}{TS.FAIL}✗{TS.ENDC} {e.message}")
        if e.hint:
            p.print_err(f"  {e.hint}")
        raise typer.Exit(code=1)
