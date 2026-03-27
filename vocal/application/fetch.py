"""Fetch a vocal project from a git repository."""

import os
import subprocess
import zipfile

import typer
import requests
import yaml

from vocal.application.register import register_project
from vocal.utils import cache_dir, flip_to_dir


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


def fetch_with_git(url: str) -> str:
    """
    Fetch a project from a git repository, via the git command line.

    Args:
        url (str): The URL of the git repository.

    Returns:
        str: The name of the directory that the project was cloned to.
    """
    print(f"Cloning project from git repository: {url}")

    projects_dir = get_projects_dir()

    with flip_to_dir(projects_dir):
        with open(os.devnull, "w") as devnull:
            subprocess.run(["git", "clone", url], stdout=devnull, stderr=devnull)

    return url.split("/")[-1]


def convert_github_repo_to_api_url(url: str) -> str:
    """
    Convert a GitHub repository URL to the GitHub API URL.
    For example,
    `https://github.com/username/repo` -> `https://api.github.com/repos/username/repo`

    Args:
        url (str): The URL of the GitHub repository.

    Returns:
        str: The URL of the GitHub API.
    """
    if "github.com" not in url:
        raise ValueError("Only GitHub repositories are supported")

    parts = url.split("/")
    user = parts[-2]
    repo = parts[-1]

    return f"https://api.github.com/repos/{user}/{repo}"


def get_latest_release_info(api_url: str) -> dict:
    """
    Get the latest release information for a GitHub project.

    Args:
        api_url (str): The URL of the GitHub API for the project.

    Returns:
        dict: The latest release information.
    """
    response = requests.get(api_url + "/releases/latest")
    return response.json()


def fetch_http(url: str) -> str:
    """
    Fetch a project over HTTP. This is done by fetching the latest release
    from the GitHub API and downloading the release zip file.

    Args:
        url (str): The URL of the project.

    Returns:
        str: The name of the directory that the project was extracted to.
    """
    print(f"Fetching project from HTTP repository: {url}")
    projects_dir = get_projects_dir()

    api_url = convert_github_repo_to_api_url(url)
    release_info = get_latest_release_info(api_url)
    zip_url = release_info["zipball_url"]
    tag_name = release_info["tag_name"]

    print(f"Fetching release {tag_name}")

    with flip_to_dir(projects_dir):
        with open("release.zip", "wb") as f:
            response = requests.get(zip_url)
            f.write(response.content)

        with zipfile.ZipFile("release.zip", "r") as zip_ref:
            dir_name = zip_ref.namelist()[0]
            zip_ref.extractall(".")

        os.remove("release.zip")

    return dir_name


def fetch_project(url: str, git: bool) -> None:
    """
    Download a project from a git repository and register it with Vocal.

    Args:
        url (str): The URL of the git repository.
        git (bool): Whether to use git to clone the repository.

    """
    if git:
        dir_name = fetch_with_git(url)

    else:
        dir_name = fetch_http(url)

    yaml_file = os.path.join(get_projects_dir(), dir_name, "vocal.yaml")

    with open(yaml_file, "r") as f:
        data = yaml.load(f, Loader=yaml.Loader)

    data = data["vocal"]

    register_project(
        project_path=os.path.join(
            get_projects_dir(), dir_name, data["project_directory"]
        ),
        definitions=os.path.join(
            get_projects_dir(),
            dir_name,
            data["products_directory"],
        ),
        conventions_string=data["conventions"],
        force=True,
    )


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
) -> None:
    """Fetch a vocal project from a git repository."""
    fetch_project(url, git)
