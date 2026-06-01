from typing import Literal

from pydantic import BaseModel, Field

from vocal.application.install import derive_url_slug
from vocal.checking import CheckError, CheckComment, CheckWarning
from vocal.utils.registry import Pack, Registry, project_key

RequirementStatus = Literal["satisfied", "project_missing", "project_too_old"]

REQUIREMENT_LABELS: dict[RequirementStatus, str] = {
    "satisfied": "Satisfied",
    "project_missing": "Project not fetched",
    "project_too_old": "Project too old",
}


class ResolverError(BaseModel):
    """A typed resolution failure surfaced on the results page.

    Mirrors the ``code`` / ``message`` / ``hint`` shape of the resolver's typed
    errors (:class:`vocal.resolution.ResolutionError`). ``code`` is drawn from
    the resolver's shared vocabulary (``project_missing``, ``project_too_old``,
    ``pack_missing``, ``pack_incompatible``, ``product_not_found``) for genuine
    resolution failures, or from the web-layer attribute-precondition vocabulary
    (``missing_conventions``, ``missing_pack_reference``) for files that cannot
    be resolved at all because the web UI has no ``-p`` / ``-d`` flag fallback.
    The web UI renders ``message`` and ``hint`` directly.
    """

    code: str
    message: str
    hint: str | None = None


class Check(BaseModel):
    """A class to hold the context of a check."""

    description: str
    comment: CheckComment | None = None
    warning: CheckWarning | None = None
    error: CheckError | None = None


class CheckProject(BaseModel):
    """A class to hold the context of a project."""

    passed: bool
    errors: list[CheckError]


class CheckDefinition(BaseModel):
    """A class to hold the context of a definition."""

    passed: bool
    warnings: bool
    comments: bool
    checks: list


class CheckContext(BaseModel):
    """
    A class to hold the context of a check, to be used in the web API.

    When resolution fails (or the file is missing the attributes the web flow
    requires), ``error`` carries the single typed failure and ``projects`` /
    ``definitions`` are empty. When resolution succeeds, ``error`` is ``None``
    and the validation results populate ``projects`` and ``definitions``.

    Attributes:
        projects (dict): The project (standard) validation result, keyed by
            ``{name}-{major}``.
        definitions (dict): The product-definition check result, keyed by
            product name.
        error (ResolverError | None): The typed resolution failure, if any.
    """

    projects: dict[str, CheckProject] = Field(default_factory=dict)
    definitions: dict[str, CheckDefinition] = Field(default_factory=dict)
    error: ResolverError | None = None


class PackVersionView(BaseModel):
    """One registered release of a pack, as shown on the Packs page.

    ``requires_standard`` is the ``{name}-{major}`` of the project standard the
    version targets; ``requires_min_minor`` is the minimum minor it needs.

    ``requirement_status`` is the three-state status of that requirement against
    the registry, mirroring the resolver's ``project_missing`` /
    ``project_too_old`` distinction: ``satisfied`` when a ``{name}-{major}``
    project exists with ``minor >= min_minor``, ``project_missing`` when no such
    project is fetched, and ``project_too_old`` when it exists but its minor is
    below the required minimum. It is informational only — it does not change
    check gating.
    """

    version: int
    requires_standard: str
    requires_min_minor: int
    requirement_status: RequirementStatus

    @property
    def requirement_label(self) -> str:
        """The human-readable label for :attr:`requirement_status`."""
        return REQUIREMENT_LABELS[self.requirement_status]


class PackView(BaseModel):
    """All registered releases of one pack URL, newest first.

    ``anchor_id`` is a stable, URL-derived slug used as an ``id`` so the
    Projects page can deep-link to a pack's card.
    """

    url: str
    anchor_id: str
    versions: list[PackVersionView]

    @property
    def latest(self) -> PackVersionView:
        """The highest-version release (``versions`` is sorted descending)."""
        return self.versions[0]


class LibraryView(BaseModel):
    """The pre-computed view of what has been fetched onto this machine.

    Built by :func:`build_library_view` from a :class:`~vocal.utils.registry.Registry`
    so templates receive ready-to-render objects rather than the raw,
    tuple-keyed packs mapping that Jinja cannot cleanly iterate.
    """

    packs: list[PackView] = Field(default_factory=list)


def _requirement_status(registry: Registry, pack: Pack) -> RequirementStatus:
    """Return the three-state requirement status of ``pack`` against ``registry``.

    Mirrors the resolver's ``_resolve_project`` lookup (``projects.get`` keyed by
    ``{name}-{major}``, then a ``minor`` comparison) so the UI's notion of
    compatibility matches the resolver's rather than inventing a parallel one.
    """
    requires = pack.manifest.requires_standard
    project = registry.projects.get(project_key(requires.name, requires.major))
    if project is None:
        return "project_missing"
    if project.minor < requires.min_minor:
        return "project_too_old"
    return "satisfied"


def build_library_view(registry: Registry) -> LibraryView:
    """Build the :class:`LibraryView` for ``registry``.

    Packs are grouped by URL (URLs sorted ascending for a stable page order),
    and each group's versions are sorted descending so the latest release leads.
    Each version's three-state requirement status is computed against
    ``registry``'s projects, mirroring the resolver's project lookup.
    """
    by_url: dict[str, list] = {}
    for pack in registry.packs.values():
        by_url.setdefault(pack.url, []).append(pack)

    pack_views: list[PackView] = []
    for url in sorted(by_url):
        packs = sorted(by_url[url], key=lambda pack: pack.version, reverse=True)
        versions = [
            PackVersionView(
                version=pack.version,
                requires_standard=(
                    f"{pack.manifest.requires_standard.name}"
                    f"-{pack.manifest.requires_standard.major}"
                ),
                requires_min_minor=pack.manifest.requires_standard.min_minor,
                requirement_status=_requirement_status(registry, pack),
            )
            for pack in packs
        ]
        pack_views.append(
            PackView(url=url, anchor_id=derive_url_slug(url), versions=versions)
        )

    return LibraryView(packs=pack_views)
