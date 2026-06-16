from typing import Literal

from pydantic import BaseModel, Field

from vocal.application.install import derive_url_slug
from vocal.checking import CheckError, CheckComment, CheckWarning
from vocal.utils.registry import Registry, project_key
from vocal.versioning import VersionConstraint

RequirementStatus = Literal["satisfied", "project_missing", "project_too_old"]

REQUIREMENT_LABELS: dict[RequirementStatus, str] = {
    "satisfied": "Satisfied",
    "project_missing": "Project not fetched",
    "project_too_old": "Project too old",
}


class ResolverError(BaseModel):
    """A typed failure surfaced on the results page *as a refusal, not a verdict*.

    The web checker refuses a file upfront only when it carries no recognisable
    vocal claim at all (``not_vocal_managed``): there is nothing to check and no
    verdict to render. ``code`` / ``message`` / ``hint`` mirror the resolver's
    typed-error shape (:class:`vocal.resolution.ResolutionError`); the web UI
    renders ``message`` and ``hint`` directly. Recoverable, per-claim problems
    (a missing pack, a too-old standard) are *not* refusals — they ride the
    INDETERMINATE verdict as :class:`UnverifiedClaim`\\ s instead.
    """

    code: str
    message: str
    hint: str | None = None


class UnverifiedClaim(BaseModel):
    """A claim the check could not finish verifying — what to fetch or update.

    Surfaced alongside the INDETERMINATE verdict. Covers an unresolved mandatory
    standard or pack (fetch it) and a claimed standard installed at a minor too
    old to run (update it). ``message`` says what wasn't verified; ``hint`` says
    how to complete the check. An opportunistic ``Conventions`` standard whose
    project isn't installed is *not* an unverified claim — it never expected to be
    checked — and is carried as an :class:`InfoComment` instead.
    """

    message: str
    hint: str | None = None


class AdvisoryWarning(BaseModel):
    """An advisory note the user may want to act on, but which never gates a check.

    The web counterpart of :class:`vocal.resolution.ResolutionWarning`: a pack's
    unmet ``satisfies_standards`` assertion (the pack claims to satisfy standards
    the file doesn't), or an opportunistic ``Conventions`` standard installed at a
    minor too old to verify. Advisory — it never changes the verdict — but, unlike
    an :class:`InfoComment`, it flags something worth acting on (often a single
    ``vocal fetch --update``). ``message`` says what; ``hint`` says what to do.
    """

    message: str
    hint: str | None = None


class InfoComment(BaseModel):
    """A purely informational note about what wasn't checked — never actionable.

    The web counterpart of :class:`vocal.resolution.ResolutionComment`: an
    opportunistic ``Conventions`` standard (CF, ACDD, ...) skipped because no
    matching project is installed. It never changes the verdict and carries no
    expectation of action, so the results page renders it as an info note,
    distinct from an :class:`UnverifiedClaim` (which says *fetch or update this*).
    ``message`` says what wasn't checked; ``hint`` says how to check it if wanted.
    """

    message: str
    hint: str | None = None


class Landing(BaseModel):
    """The outcome of storing a PASS file under ``vocal web --upload-to``.

    Surfaced on the results page only when storage was *attempted* — i.e. a PASS
    verdict with the feature on. ``status`` discriminates the two cases the user
    sees: ``"stored"`` (the validated file was collected) and ``"refused"`` (it
    passed but could not be stored, e.g. an unsafe name). ``message`` is a
    human-readable, deliberately **path-free** line: the server's filesystem
    layout is never disclosed to a remote user (the server-side log is the only
    place the directory is named).
    """

    status: Literal["stored", "refused"]
    message: str


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
    """The context of a web check, rendered into the results page.

    The web surface renders a **tri-state verdict** — ``"pass"``, ``"fail"``, or
    ``"indeterminate"`` — driven by the check spine's
    :class:`~vocal.checking.shared.Verdict`. ``projects`` and ``definitions``
    carry the per-axis check results; ``unverified`` carries the
    fetch-it/update-it items that explain an INDETERMINATE verdict.

    ``error`` is set *only* when the file is refused upfront for carrying no
    recognisable vocal claim — a distinct "not a vocal-managed file" state, not a
    verdict. When ``error`` is set, ``verdict`` is ``None`` and the result
    collections are empty; otherwise ``verdict`` is always populated.

    Attributes:
        verdict (str | None): ``"pass"`` / ``"fail"`` / ``"indeterminate"``, or
            ``None`` when the file was refused upfront.
        projects (dict): The standards-axis validation results, keyed by
            ``{name}-{major}``.
        definitions (dict): The product-axis check result, keyed by product name.
        unverified (list[UnverifiedClaim]): What couldn't be verified and how to
            complete the check (populated for the INDETERMINATE state).
        warnings (list[AdvisoryWarning]): Advisory notes the user may want to act
            on but which never change the verdict — an unmet
            ``satisfies_standards`` assertion, or an opportunistic standard
            installed too old to verify. The CLI shows these as warnings too.
        comments (list[InfoComment]): Purely informational notes about what wasn't
            checked — opportunistic standards skipped because their project isn't
            installed. Never actionable; the CLI shows these as comments too.
        error (ResolverError | None): The upfront refusal, if any.
        landing (Landing | None): The storage outcome under ``--upload-to``,
            populated only when a PASS file's storage was attempted; ``None``
            otherwise (feature off, non-PASS verdict, or upfront refusal).
    """

    verdict: str | None = None
    projects: dict[str, CheckProject] = Field(default_factory=dict)
    definitions: dict[str, CheckDefinition] = Field(default_factory=dict)
    unverified: list[UnverifiedClaim] = Field(default_factory=list)
    warnings: list[AdvisoryWarning] = Field(default_factory=list)
    comments: list[InfoComment] = Field(default_factory=list)
    error: ResolverError | None = None
    landing: Landing | None = None


class SatisfiedStandardView(BaseModel):
    """One ``satisfies_standards`` constraint of a pack, plus its install status.

    ``constraint`` is the rendered constraint (e.g. ``MYSTD-2.3+``).
    ``status`` is the three-state status of that standard against the registry,
    mirroring the resolver's ``project_missing`` / ``project_too_old`` distinction:
    ``satisfied`` when a ``{name}-{major}`` project exists with ``minor >=
    min_minor``, ``project_missing`` when none is fetched, ``project_too_old``
    when one exists below the required minimum. It is informational only — the
    ``satisfies_standards`` assertion is advisory and never gates a check.
    """

    constraint: str
    status: RequirementStatus

    @property
    def label(self) -> str:
        """The human-readable label for :attr:`status`."""
        return REQUIREMENT_LABELS[self.status]


class PackVersionView(BaseModel):
    """One registered release of a pack, as shown on the Packs page.

    ``satisfies`` is the advisory list of standards the version's product is
    intended to satisfy, each carrying its install status against the registry.
    A pack may satisfy several standards (or none).
    """

    version: int
    satisfies: list[SatisfiedStandardView] = Field(default_factory=list)


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


class ProjectPackView(BaseModel):
    """A pack satisfying a project's exact standard, as shown on the Projects page.

    Lists only the versions of this pack URL whose ``satisfies_standards``
    includes the project's exact ``{name}-{major}`` (a URL whose later version
    dropped or changed that standard appears here only for the versions that
    still satisfy this project). ``anchor_id`` matches the corresponding
    :class:`PackView`'s, so the entry can deep-link to that pack's card.
    """

    url: str
    anchor_id: str
    versions: list[int]


class ProjectView(BaseModel):
    """A registered project plus the packs that satisfy its exact standard.

    ``key`` is the registry key (``{name}-{major}``). ``packs`` lists the packs
    asserting they satisfy this project, grouped by URL (URLs sorted ascending)
    and filtered to only the versions whose ``satisfies_standards`` includes
    ``key``.
    """

    key: str
    name: str
    major: int
    minor: int
    local_path: str
    packs: list[ProjectPackView] = Field(default_factory=list)


class LibraryView(BaseModel):
    """The pre-computed view of what has been fetched onto this machine.

    Built by :func:`build_library_view` from a :class:`~vocal.utils.registry.Registry`
    so templates receive ready-to-render objects rather than the raw,
    tuple-keyed packs mapping that Jinja cannot cleanly iterate.
    """

    packs: list[PackView] = Field(default_factory=list)
    projects: list[ProjectView] = Field(default_factory=list)


def _constraint_status(
    registry: Registry, constraint: VersionConstraint
) -> RequirementStatus:
    """Return the three-state install status of one ``satisfies_standards``
    constraint against ``registry``.

    Mirrors the resolver's project lookup (``projects.get`` keyed by
    ``{name}-{major}``, then a ``minor`` comparison) so the UI's notion of "is
    this standard installed?" matches the resolver's rather than inventing a
    parallel one.
    """
    project = registry.projects.get(project_key(constraint.name, constraint.major))
    if project is None:
        return "project_missing"
    if project.minor < constraint.min_minor:
        return "project_too_old"
    return "satisfied"


def build_library_view(registry: Registry) -> LibraryView:
    """Build the :class:`LibraryView` for ``registry``.

    Packs are grouped by URL (URLs sorted ascending for a stable page order),
    and each group's versions are sorted descending so the latest release leads.
    Each version's advisory ``satisfies_standards`` are rendered with their
    three-state install status against ``registry``'s projects.
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
                satisfies=[
                    SatisfiedStandardView(
                        constraint=str(constraint),
                        status=_constraint_status(registry, constraint),
                    )
                    for constraint in pack.manifest.satisfies_standards
                ],
            )
            for pack in packs
        ]
        pack_views.append(
            PackView(url=url, anchor_id=derive_url_slug(url), versions=versions)
        )

    return LibraryView(packs=pack_views, projects=_project_views(registry))


def _project_views(registry: Registry) -> list[ProjectView]:
    """Build the per-project view models, including the reverse pack links.

    For each project (keyed ``{name}-{major}``), the packs satisfying it are the
    pack versions whose ``satisfies_standards`` includes that exact key. They are
    grouped by URL (URLs sorted ascending), and within each group only the
    matching versions are kept, sorted descending. A pack URL whose later
    versions dropped that standard therefore appears under the project only for
    the versions that still satisfy it.
    """
    project_views: list[ProjectView] = []
    for key in sorted(registry.projects):
        project = registry.projects[key]
        by_url: dict[str, list[int]] = {}
        for pack in registry.packs.values():
            if any(
                project_key(constraint.name, constraint.major) == project.key
                for constraint in pack.manifest.satisfies_standards
            ):
                by_url.setdefault(pack.url, []).append(pack.version)
        pack_links = [
            ProjectPackView(
                url=url,
                anchor_id=derive_url_slug(url),
                versions=sorted(by_url[url], reverse=True),
            )
            for url in sorted(by_url)
        ]
        project_views.append(
            ProjectView(
                key=project.key,
                name=project.name,
                major=project.major,
                minor=project.minor,
                local_path=project.local_path,
                packs=pack_links,
            )
        )
    return project_views
