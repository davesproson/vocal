"""Check-time resolution: turn a file's claims plus the local registry into a
concrete ``(project, pack, product)`` target.

``vocal check`` (and the web check endpoint) needs to answer a single question
about a netCDF file: *what should this file be validated against?* The file
self-describes through its global attributes — its ``Conventions`` string names
the standard and the minor it claims, and the optional ``vocal_definitions_url``
/ ``vocal_definitions_version`` pair pins the pack it was authored against. This
module owns the logic that resolves those claims against the machine-local
:class:`~vocal.utils.registry.Registry` and returns a :class:`ResolvedTarget`,
or raises one of the typed errors below when something does not line up.

The resolver is pure with respect to the application layer: it takes a registry
and the file's attribute values as plain arguments and returns data. The only
side input it needs from a project is its ``filecodec`` (to expand templated
``file_pattern`` entries when routing a file to a product); that lookup is
injected as a callable so the resolver can be driven entirely from fake
registries and synthetic attribute dicts in tests.

The six-step resolution flow (see the parent PRD's "Check resolution flow"):

1. Parse ``Conventions``: tokenise on whitespace and pick the vocal-managed
   token, parsing it into a :class:`~vocal.versioning.Version`.
2. Look up a registered project with the matching ``{name, major}`` whose
   ``minor`` is at least the file's claimed minor — :class:`ProjectMissing` if
   absent, :class:`ProjectTooOld` if too old.
3. If ``vocal_definitions_url`` is present, look up the registered pack by
   normalised URL: with ``vocal_definitions_version`` also present, the lookup
   is by ``(url, version)``; with the version absent, it resolves to the highest
   registered version for that URL — :class:`PackMissing` if none match.
4. Verify the pack's ``requires_standard`` is satisfied by the registered
   project — :class:`PackIncompatible` otherwise, naming the failing sub-check.
5. Match the file to a product by the manifest's ``file_pattern`` entries,
   expanded with the project's ``filecodec`` — :class:`ProductNotFound` if none
   match.
6. Return a :class:`ResolvedTarget` carrying the project, pack, product, and the
   absolute path of the schema to validate against.

When an explicit ``-d``/``definition_override`` is supplied, steps 3–5 are
replaced by "load the schema at that path directly": the project is still
resolved (steps 1–2), but no pack lookup or compatibility check is performed and
the user takes responsibility for the chosen product schema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Mapping, Optional

from vocal.exceptions import VocalError
from vocal.manifest import ManifestProduct, normalize_pack_url
from vocal.utils.conventions import FileConventions, read_file_conventions
from vocal.utils.registry import Pack, Project, Registry, project_key
from vocal.versioning import InvalidVersion, Version, VersionConstraint

# A loader that maps a registered project to its ``filecodec`` — the mapping of
# template placeholder names to ``{"regex": ...}`` entries used to expand a
# product's ``file_pattern`` at match time. Injected so the resolver stays
# testable without importing a real project package.
FilecodecLoader = Callable[[Project], Mapping[str, Mapping[str, Any]]]


class ResolutionError(VocalError):
    """Base class for the resolver's typed failures.

    Each subclass carries a stable ``code`` drawn from the shared resolver-error
    vocabulary, used by the web API to tag the failure category, plus the
    ``message`` and ``hint`` from :class:`~vocal.exceptions.VocalError`.
    """

    code: str = "resolution_error"


class ProjectMissing(ResolutionError):
    """No registered project matches the standard the file claims."""

    code = "project_missing"


class ProjectTooOld(ResolutionError):
    """The registered project's minor is older than the file's claimed minor."""

    code = "project_too_old"


class PackMissing(ResolutionError):
    """No registered pack matches the file's declared definitions URL/version."""

    code = "pack_missing"


class PackIncompatible(ResolutionError):
    """The pack's ``requires_standard`` is not satisfied by the registered project."""

    code = "pack_incompatible"


class ProductNotFound(ResolutionError):
    """The file matched no product pattern in the resolved pack."""

    code = "product_not_found"


@dataclass(frozen=True)
class ResolvedTarget:
    """The outcome of resolving a file against the registry.

    ``project`` is always present. ``schema_path`` is the absolute path of the
    product schema JSON the file should be validated against — sourced from the
    matched pack product, or directly from a ``-d`` override. ``pack`` and
    ``product`` are populated on the full resolver flow and ``None`` when a
    ``-d`` override short-circuits pack resolution. ``schema_path`` is ``None``
    only when neither a pack nor an override was supplied (project-only
    resolution).
    """

    project: Project
    schema_path: Optional[str]
    pack: Optional[Pack] = None
    product: Optional[ManifestProduct] = None

    @property
    def is_fully_resolved(self) -> bool:
        """Whether the target was resolved to a product schema (vs. project-only)."""
        return (
            self.project is not None
            and self.schema_path is not None
            and self.pack is not None
            and self.product is not None
        )


def _default_filecodec_loader(project: Project) -> Mapping[str, Mapping[str, Any]]:
    """Import the project's package from its cached repo and return its filecodec.

    Imported lazily so the application's project-import machinery is only loaded
    when a real project must be read — tests inject their own loader.
    """
    from vocal.conventions_file import import_project_package

    module: ModuleType = import_project_package(project.local_path)
    return module.filecodec


def resolve(
    registry: Registry,
    *,
    filename: str,
    conventions: Optional[str],
    definitions_url: Optional[str] = None,
    definitions_version: Optional[int] = None,
    definition_override: Optional[str] = None,
    project_url: Optional[str] = None,
    filecodec_loader: FilecodecLoader = _default_filecodec_loader,
) -> ResolvedTarget:
    """Resolve a file's claims against ``registry`` into a :class:`ResolvedTarget`.

    Args:
        registry: the local registry of fetched projects and packs.
        filename: the netCDF file's name — only the basename is used, for product
            pattern matching.
        conventions: the file's ``Conventions`` global attribute, or ``None``.
        definitions_url: the file's ``vocal_definitions_url`` (a pack base URL),
            or ``None``.
        definitions_version: the file's ``vocal_definitions_version``, or ``None``.
        definition_override: an explicit ``-d`` product schema path. When given,
            pack resolution (steps 3–5) is skipped and this path becomes the
            target schema.
        project_url: the file's ``vocal_project_url``, used only to populate the
            ``vocal fetch`` hint on project-related errors.
        filecodec_loader: maps a resolved project to its ``filecodec``. Defaults
            to importing the project's package.

    Returns:
        a :class:`ResolvedTarget`.

    Raises:
        ProjectMissing, ProjectTooOld, PackMissing, PackIncompatible,
        ProductNotFound: per the resolution flow.
    """
    project, claimed = _resolve_project(registry, conventions, project_url)

    # -d override: the user has chosen the product schema explicitly. Resolve the
    # project for the model, then trust the supplied path (steps 3–5 skipped).
    if definition_override is not None:
        return ResolvedTarget(project=project, schema_path=definition_override)

    # No pack reference and no override: resolve the project alone. The CLI
    # layer requires -d in this case; the web layer rejects it. A bare
    # definitions version with no URL cannot name a pack, so it is treated the
    # same way.
    if definitions_url is None:
        return ResolvedTarget(project=project, schema_path=None)

    pack = _resolve_pack(registry, definitions_url, definitions_version)
    _check_compatibility(pack, project)
    product = _match_product(pack, project, filename, filecodec_loader)

    schema_path = os.path.join(pack.local_path, product.schema)
    return ResolvedTarget(
        project=project, schema_path=schema_path, pack=pack, product=product
    )


def _load_registry() -> Registry:
    """Load the machine-local registry, falling back to an empty one.

    A machine that has never fetched anything has no registry file; the check
    surfaces treat that as "nothing registered" rather than an error, so a
    missing file resolves to an empty :class:`Registry`.
    """
    try:
        return Registry.load()
    except FileNotFoundError:
        return Registry()


def resolve_file(
    filename: str,
    *,
    attrs: Optional[FileConventions] = None,
    registry: Optional[Registry] = None,
    definition_override: Optional[str] = None,
    filecodec_loader: FilecodecLoader = _default_filecodec_loader,
) -> ResolvedTarget:
    """Read a file's vocal-managed attributes and resolve them against a registry.

    The shared spine of every check surface — ``vocal check``, the web checker,
    and the gatekeeper — each of which needs to turn a path on disk into a
    :class:`ResolvedTarget`: read the file's :class:`FileConventions`, then drive
    :func:`resolve` with them. Reading the attributes and mapping them onto the
    resolver's keyword arguments is the part that was otherwise copy-pasted per
    surface; everything *around* the resolution (precondition policy, how errors
    are rendered) legitimately differs and stays with the caller.

    ``attrs`` and ``registry`` are accepted pre-built for callers that already
    hold them — the CLI and web layers read the attributes first (to drive a
    ``--fetch`` nudge and a precondition check respectively) and load the
    registry through their own test seam. Callers with neither (the gatekeeper)
    pass just ``filename`` and let this read the file and load the registry.

    Raises the typed :class:`ResolutionError` subclasses from :func:`resolve`,
    plus whatever :func:`read_file_conventions` raises when the file cannot be
    read.
    """
    if attrs is None:
        attrs = read_file_conventions(filename)
    if registry is None:
        registry = _load_registry()

    return resolve(
        registry,
        filename=filename,
        conventions=attrs.conventions,
        definitions_url=attrs.definitions_url,
        definitions_version=attrs.definitions_version,
        definition_override=definition_override,
        project_url=attrs.project_urls[0] if attrs.project_urls else None,
        filecodec_loader=filecodec_loader,
    )


def _resolve_project(
    registry: Registry, conventions: Optional[str], project_url: Optional[str]
) -> tuple[Project, Version]:
    """Steps 1–2: identify the standard from ``Conventions`` and look it up.

    Returns the registered :class:`Project` and the :class:`Version` the file
    claimed. Raises :class:`ProjectMissing` / :class:`ProjectTooOld`.
    """
    claimed = _select_version_token(conventions, registry)
    url_hint = project_url or "<vocal_project_url>"

    project = registry.projects.get(project_key(claimed.name, claimed.major))
    if project is None:
        raise ProjectMissing(
            f"No project registered for {claimed.name}-{claimed.major}",
            f"Run 'vocal fetch {url_hint}' to register the project, or pass -p <path>.",
        )

    if project.minor < claimed.minor:
        raise ProjectTooOld(
            f"File claims {claimed.name}-{claimed.major}.{claimed.minor} but "
            f"registered project is at {project.name}-{project.major}."
            f"{project.minor}",
            f"Update the registered project: 'vocal fetch {url_hint} --update'.",
        )

    return project, claimed


def _select_version_token(conventions: Optional[str], registry: Registry) -> Version:
    """Pick the vocal-managed token out of a ``Conventions`` string.

    The string may carry co-conventions (e.g. ``"CF-1.8 ACDD-1.3 MYSTD-1.2"``).
    Tokens are split on whitespace; each is parsed as a :class:`Version` and
    non-conforming tokens are ignored. The vocal token is the first parseable
    token whose ``name`` matches a registered project's name. When none match a
    registered name (e.g. the standard is not fetched at all), the last
    parseable token is used — vocal tokens conventionally trail the CF/ACDD
    tokens — so the resulting :class:`ProjectMissing` still names the standard.

    Raises:
        ProjectMissing: ``Conventions`` is absent or carries no parseable token.
    """
    if not conventions:
        raise ProjectMissing(
            "No Conventions attribute found on the file.",
            "Pass -p <path> and -d <path> to check the file explicitly.",
        )

    parsed: list[Version] = []
    for token in conventions.split():
        token = token.rstrip(",")
        try:
            parsed.append(Version.parse(token))
        except InvalidVersion:
            continue

    if not parsed:
        raise ProjectMissing(
            f"No recognisable standard version in Conventions: {conventions!r}",
            "Pass -p <path> and -d <path> to check the file explicitly.",
        )

    registered_names = {project.name for project in registry.projects.values()}
    for version in parsed:
        if version.name in registered_names:
            return version

    return parsed[-1]


def _resolve_pack(
    registry: Registry, definitions_url: str, definitions_version: Optional[int]
) -> Pack:
    """Step 3: look up the registered pack for the file's declared URL.

    With an explicit ``definitions_version``, look up ``(url, version)`` exactly.
    With the version absent, resolve to the highest registered version for the
    URL — the file carried no precise pin, so the newest local version is used.
    Raises :class:`PackMissing` (hinting the repo-URL fetch form) when no
    matching pack is registered.
    """
    normalized = normalize_pack_url(definitions_url)

    if definitions_version is None:
        pack = registry.find_latest_pack(definitions_url)
        if pack is None:
            raise PackMissing(
                f"No pack registered for {normalized}",
                f"Run 'vocal fetch {normalized}' to register it.",
            )
        return pack

    pack = registry.find_pack(definitions_url, definitions_version)
    if pack is None:
        raise PackMissing(
            f"No pack registered for {normalized} version {definitions_version}",
            f"Run 'vocal fetch {normalized}' to register it.",
        )
    return pack


def _check_compatibility(pack: Pack, project: Project) -> None:
    """Step 4: verify the pack's ``requires_standard`` matches the project.

    Three sub-checks; the message names the one that failed.
    """
    requires: VersionConstraint = pack.manifest.requires_standard

    if requires.name != project.name or requires.major != project.major:
        raise PackIncompatible(
            f"Pack targets {requires.name}-{requires.major} but registered "
            f"project is {project.name}-{project.major}",
            f"Register a project matching the pack's target standard, or pin to "
            f"a pack built for {project.name}-{project.major}.",
        )

    if requires.min_minor > project.minor:
        raise PackIncompatible(
            f"Pack requires {requires} but registered project is at "
            f"{project.major}.{project.minor}",
            f"Update the project to {requires.major}.{requires.min_minor} or "
            f"later, or pin to an older pack.",
        )


def _match_product(
    pack: Pack,
    project: Project,
    filename: str,
    filecodec_loader: FilecodecLoader,
) -> ManifestProduct:
    """Step 5: route the file to a product via ``file_pattern`` + ``filecodec``."""
    filecodec = filecodec_loader(project)
    product = pack.manifest.find_product(filename, filecodec)
    if product is None:
        patterns = ", ".join(p.file_pattern for p in pack.manifest.products)
        raise ProductNotFound(
            f"File {os.path.basename(filename)!r} did not match any product "
            f"pattern in pack",
            f"Verify the filename matches one of: {patterns}.",
        )
    return product
