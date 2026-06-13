"""Check-time resolution: turn a file's claims plus the local registry into the
two independent axes a check must verify.

``vocal check`` (and the web checker, and the gatekeeper) needs to answer one
question about a netCDF file: *what should this file be validated against?* The
file self-describes through its global attributes, and those claims split into
two axes that never consult each other:

- **A standards axis.** A file may claim many standards. ``vocal_project_url``
  is a whitespace-separated list of repository URLs and is the authoritative set
  of *mandatory* standards: each names (via the registry's
  :meth:`~vocal.utils.registry.Registry.find_project_by_url`) a project the file
  must be verified against. Standards named only in ``Conventions`` are
  *opportunistic* — verified if their project is installed, skipped with a
  :class:`ResolutionWarning` if not (so external co-conventions such as CF and
  ACDD, which have no URL and no installed project, simply fall out). The
  **major** of every URL-claimed standard is sourced from the matching
  ``Conventions`` token *by name*, never from the URL lookup.

- **A product axis.** A file *is* a single product, so it declares exactly one
  pack via ``vocal_definitions_url`` (+ optional ``vocal_definitions_version``).
  The pack's product schema is matched to the file using the pack manifest's own
  embedded ``filecodec`` — the pack is self-contained and needs no project to
  route a file to a product.

The resolver collects per-claim failures rather than raising on the first
problem, so a surface can report everything in one pass. It raises only when
there is *nothing to check at all* — no resolvable target, no pack, and no
failure to report.

A claimed standard that is installed but at a minor *older* than the file claims
cannot be fully verified and — because non-breaking minors are forward-only (a
file conforming to ``STD-1.X`` conforms to ``STD-1.(X+n)`` but not the reverse)
— its model must not be run lest it spuriously reject a legitimately
newer-minor file. Such a claim is recorded as an *unverifiable*
:class:`ProjectTarget` (``verifiable=False``) carrying an ``--update`` hint; the
check spine turns that into an INDETERMINATE verdict. The gate that matters is
**installed-vs-not**, not mandatory-vs-opportunistic.

The resolver is pure with respect to the application and filesystem layers when
driven through :func:`resolve_file` with an explicit ``attrs`` and ``registry``,
so it can be exercised entirely from synthetic registries, :class:`FileConventions`,
and manifests in tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from vocal.exceptions import VocalError
from vocal.manifest import ManifestProduct, normalize_pack_url
from vocal.utils.conventions import FileConventions, read_file_conventions
from vocal.utils.registry import Pack, Project, Registry, project_key
from vocal.versioning import InvalidVersion, Version


class ResolutionError(VocalError):
    """Base class for the resolver's typed, per-claim failures.

    Each subclass carries a stable ``code`` drawn from the shared resolver-error
    vocabulary (used by the web API to tag the failure category), plus the
    ``message`` and ``hint`` from :class:`~vocal.exceptions.VocalError`. These
    are *collected* into a :class:`Resolution`'s ``failures`` rather than raised,
    so a surface can report every unresolved claim at once.
    """

    code: str = "resolution_error"


class ProjectMissing(ResolutionError):
    """A mandatory standard's project is not installed (or not at the claimed major)."""

    code = "project_missing"


class PackMissing(ResolutionError):
    """No registered pack matches the file's declared definitions URL/version."""

    code = "pack_missing"


class ProductNotFound(ResolutionError):
    """The file matched no product pattern in the resolved pack."""

    code = "product_not_found"


class NothingToCheck(ResolutionError):
    """The file carries no recognisable vocal claim at all.

    Raised — not collected — when resolution produces no project target, no
    pack, and no failure to report: there is literally nothing to verify. A
    surface renders this as "not a vocal-managed file", distinct from a verdict.
    """

    code = "nothing_to_check"


@dataclass(frozen=True)
class ProjectTarget:
    """A standard the file is to be verified against, along the standards axis.

    ``mandatory`` is ``True`` when the standard came from ``vocal_project_url``
    and ``False`` for an opportunistic ``Conventions``-only claim.
    ``claimed_version`` is the :class:`~vocal.versioning.Version` the file
    claimed via its ``Conventions`` token, or ``None`` for a URL-claimed standard
    the file names no version for.

    ``verifiable`` is ``False`` when the project is installed but at a minor
    *older* than the file claims: the model must not be run (it could spuriously
    reject a legitimately newer-minor file), so the check spine treats an
    unverifiable target as INDETERMINATE. ``hint`` carries the ``--update``
    nudge in that case.
    """

    project: Project
    mandatory: bool
    claimed_version: Optional[Version]
    verifiable: bool = True
    hint: Optional[str] = None


@dataclass(frozen=True)
class PackTarget:
    """The single product the file is to be verified against, along the product axis.

    ``schema_path`` is the absolute path of the product schema JSON to validate
    the file against, sourced from the matched pack product.
    """

    pack: Pack
    product: ManifestProduct
    schema_path: str


@dataclass(frozen=True)
class ResolutionWarning:
    """A non-fatal note about a claim that was not (fully) verified.

    Warnings never change a verdict on their own; they explain what was skipped
    (an opportunistic standard whose project isn't installed) or that a pack's
    advisory ``satisfies_standards`` assertion doesn't intersect the file's
    claimed standards. ``code`` tags the category for a surface to render.
    """

    code: str
    message: str
    hint: Optional[str] = None


@dataclass
class Resolution:
    """The outcome of resolving a file's claims along both axes.

    ``projects`` holds every standards-axis target (verifiable and unverifiable
    alike — the check spine filters on ``verifiable``). ``pack`` is the single
    product-axis target, or ``None`` when the file declares no pack (or the pack
    could not be resolved — in which case ``failures`` explains why).
    ``failures`` are the typed, per-claim problems for mandatory claims;
    ``warnings`` are advisory notes.
    """

    projects: list[ProjectTarget] = field(default_factory=list)
    pack: Optional[PackTarget] = None
    failures: list[ResolutionError] = field(default_factory=list)
    warnings: list[ResolutionWarning] = field(default_factory=list)


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
) -> Resolution:
    """Resolve a file's claims into a :class:`Resolution` along both axes.

    The shared spine of every check surface. Reads the file's
    :class:`FileConventions` and the machine-local :class:`Registry` when not
    supplied; callers that already hold them (the CLI and web layers read the
    attributes first to drive their preconditions) pass them in, which also makes
    the resolver pure and fully driveable from synthetic inputs in tests.

    Args:
        filename: the netCDF file's name — only the basename is used, for product
            pattern matching.
        attrs: the file's vocal-managed attributes; read from ``filename`` when
            ``None``.
        registry: the local registry; loaded from disk (or empty) when ``None``.

    Returns:
        a :class:`Resolution` carrying ``projects``, ``pack``, ``failures``, and
        ``warnings``.

    Raises:
        NothingToCheck: the file carries no recognisable vocal claim — no
            resolvable project target, no pack, and no failure to report.
    """
    if attrs is None:
        attrs = read_file_conventions(filename)
    if registry is None:
        registry = _load_registry()

    claimed = tokenise_conventions(attrs.conventions)

    resolution = Resolution()
    covered: set[str] = set()

    _resolve_mandatory(attrs.project_urls, claimed, registry, resolution, covered)
    _resolve_opportunistic(claimed, registry, resolution, covered)
    _resolve_product_axis(attrs, filename, registry, claimed, resolution)

    if not resolution.projects and resolution.pack is None and not resolution.failures:
        raise NothingToCheck(
            "The file carries no recognisable vocal claim.",
            "Expected a vocal_project_url, a vocal_definitions_url, or a "
            "Conventions token naming an installed standard. Pass -p/-d to check "
            "the file explicitly.",
        )

    return resolution


def tokenise_conventions(conventions: Optional[str]) -> list[Version]:
    """Tokenise a ``Conventions`` string into the standard versions it claims.

    Tokens are split on whitespace; each is parsed as a :class:`Version` and
    non-conforming tokens (and an absent attribute) are silently dropped (so
    external CF/ACDD tokens fall out). The order in which tokens appear is
    preserved. Exposed as the canonical ``Conventions`` parser so surfaces (e.g.
    the web checker's recognisability precondition) tokenise identically to the
    resolver rather than reimplementing it.
    """
    if not conventions:
        return []
    parsed: list[Version] = []
    for token in conventions.split():
        token = token.rstrip(",")
        try:
            parsed.append(Version.parse(token))
        except InvalidVersion:
            continue
    return parsed


def _find_token(claimed: list[Version], name: str) -> Optional[Version]:
    """Return the first claimed version whose ``name`` matches, or ``None``."""
    for version in claimed:
        if version.name == name:
            return version
    return None


def _resolve_mandatory(
    project_urls: list[str],
    claimed: list[Version],
    registry: Registry,
    resolution: Resolution,
    covered: set[str],
) -> None:
    """Resolve the mandatory standards declared by ``vocal_project_url``.

    Each URL establishes mandatoriness and (via ``find_project_by_url``) the
    project *name*; the **major** is sourced from the matching ``Conventions``
    token by name. A URL with no installed project, or whose claimed major is not
    installed, is a :class:`ProjectMissing` failure carrying the URL as the fetch
    hint. A claim installed but too old becomes an unverifiable target.
    """
    for url in project_urls:
        installed = registry.find_project_by_url(url)
        if installed is None:
            resolution.failures.append(
                ProjectMissing(
                    f"No project registered for {url}",
                    f"Run 'vocal fetch {url}' to register the standard, "
                    f"or pass -p <path>.",
                )
            )
            continue

        name = installed.name
        covered.add(name)
        token = _find_token(claimed, name)

        # A URL'd standard the file names no version for has no version
        # constraint: verify against the single installed project at that URL.
        if token is None:
            resolution.projects.append(
                ProjectTarget(project=installed, mandatory=True, claimed_version=None)
            )
            continue

        project = registry.projects.get(project_key(name, token.major))
        if project is None:
            resolution.failures.append(
                ProjectMissing(
                    f"No project registered for {name}-{token.major}",
                    f"Run 'vocal fetch {url}' to register it, or pass -p <path>.",
                )
            )
            continue

        resolution.projects.append(
            _project_target(project, token, mandatory=True, url=url)
        )


def _resolve_opportunistic(
    claimed: list[Version],
    registry: Registry,
    resolution: Resolution,
    covered: set[str],
) -> None:
    """Resolve the opportunistic standards named only in ``Conventions``.

    A ``Conventions`` standard not already covered by a URL is verified when its
    ``{name, major}`` project is installed, recorded as an unverifiable target
    when installed-but-too-old, and skipped with a :class:`ResolutionWarning`
    when not installed (which is how external co-conventions fall out).
    """
    for token in claimed:
        if token.name in covered:
            continue
        covered.add(token.name)

        project = registry.projects.get(project_key(token.name, token.major))
        if project is None:
            resolution.warnings.append(
                ResolutionWarning(
                    code="standard_not_verified",
                    message=(
                        f"{token} was not verified: no matching project is "
                        f"installed."
                    ),
                    hint=f"Run 'vocal fetch' for {token.name}-{token.major} to "
                    f"verify it.",
                )
            )
            continue

        resolution.projects.append(
            _project_target(project, token, mandatory=False, url=None)
        )


def _project_target(
    project: Project, token: Version, *, mandatory: bool, url: Optional[str]
) -> ProjectTarget:
    """Build a :class:`ProjectTarget` for an installed project, marking it
    unverifiable when its minor is older than the claimed minor."""
    if project.minor < token.minor:
        if url is not None:
            hint = f"Update the registered project: 'vocal fetch {url} --update'."
        else:
            hint = (
                f"Update the registered project to {token.name}-{token.major}."
                f"{token.minor} or later, then re-check ('--update')."
            )
        return ProjectTarget(
            project=project,
            mandatory=mandatory,
            claimed_version=token,
            verifiable=False,
            hint=hint,
        )
    return ProjectTarget(project=project, mandatory=mandatory, claimed_version=token)


def _resolve_product_axis(
    attrs: FileConventions,
    filename: str,
    registry: Registry,
    claimed: list[Version],
    resolution: Resolution,
) -> None:
    """Resolve the single pack declared by ``vocal_definitions_url``.

    Looks up the pack (exact version, else the highest registered for the URL),
    matches the file to a product via the pack's *embedded* ``filecodec``, and
    emits an advisory warning when the pack's ``satisfies_standards`` doesn't
    intersect the file's claimed standards. Pack/product problems are collected
    as failures; the pack is mandatory once named.
    """
    if attrs.definitions_url is None:
        return

    pack = _resolve_pack(
        registry, attrs.definitions_url, attrs.definitions_version, resolution
    )
    if pack is None:
        return

    _check_satisfies(pack, claimed, resolution)

    product = pack.manifest.find_product(filename)
    if product is None:
        patterns = ", ".join(p.file_pattern for p in pack.manifest.products)
        resolution.failures.append(
            ProductNotFound(
                f"File {os.path.basename(filename)!r} did not match any product "
                f"pattern in pack",
                f"Verify the filename matches one of: {patterns}.",
            )
        )
        return

    schema_path = os.path.join(pack.local_path, product.schema)
    resolution.pack = PackTarget(
        pack=pack, product=product, schema_path=schema_path
    )


def _resolve_pack(
    registry: Registry,
    definitions_url: str,
    definitions_version: Optional[int],
    resolution: Resolution,
) -> Optional[Pack]:
    """Look up the registered pack for the file's declared URL, collecting a
    :class:`PackMissing` failure when none matches.

    With an explicit version, look up ``(url, version)`` exactly; with the
    version absent, resolve to the highest registered version for the URL.
    """
    normalized = normalize_pack_url(definitions_url)

    if definitions_version is None:
        pack = registry.find_latest_pack(definitions_url)
    else:
        pack = registry.find_pack(definitions_url, definitions_version)

    if pack is None:
        suffix = (
            f" version {definitions_version}"
            if definitions_version is not None
            else ""
        )
        resolution.failures.append(
            PackMissing(
                f"No pack registered for {normalized}{suffix}",
                f"Run 'vocal fetch {normalized}' to register it.",
            )
        )
        return None
    return pack


def _check_satisfies(
    pack: Pack, claimed: list[Version], resolution: Resolution
) -> None:
    """Emit an advisory warning when the pack's ``satisfies_standards`` doesn't
    intersect the file's claimed standards.

    The assertion is advisory and never a failure: the file is independently
    checked against the pack schema regardless. A warning is emitted only when
    the pack asserts standards *and none* are ``.satisfied_by()`` any claimed
    version (the claimed minor participates, so CF/ACDD — which satisfy no vocal
    constraint — are harmless).
    """
    constraints = pack.manifest.satisfies_standards
    if not constraints:
        return

    intersects = any(
        constraint.satisfied_by(version)
        for constraint in constraints
        for version in claimed
    )
    if intersects:
        return

    asserted = ", ".join(str(constraint) for constraint in constraints)
    resolution.warnings.append(
        ResolutionWarning(
            code="satisfies_standards_unmet",
            message=(
                f"Pack asserts it satisfies {asserted}, none of which the file "
                f"claims."
            ),
            hint="Advisory only; the file is still checked against the pack schema.",
        )
    )
