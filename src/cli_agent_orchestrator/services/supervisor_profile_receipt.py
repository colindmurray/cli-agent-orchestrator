"""Supervisor launch-resolved profile receipts (cond-0817).

The conductor preflights a supervisor profile (reads the file, records the
provider/model/effort it saw) and then asks this runtime to launch. Between
the preflight and the launch the profile bytes on disk can change — an edit,
a ``cao install`` refresh, a retry racing a write — so the runtime re-reads
the profile source at the actual launch boundary and compares what it found
against what the conductor expected *before* any tmux, session, or provider
effect exists.

Contract shapes:

* The request carries an optional ``profile_contract`` shaped
  ``cao-profile-launch-contract-v1``: ``{schema, profile, role, provider,
  model, effort, provenance, source_path, sha256}``. ``sha256`` is the hex
  digest of the exact source bytes the conductor preflighted;
  ``source_path`` is the filesystem path it read (or the ``built-in:<name>``
  pseudo-path for the packaged store); ``model``/``effort`` are the resolved
  route the conductor saw (``None`` when the profile declares none).
* The runtime persists a runtime-authored ``profile_receipt`` shaped
  ``cao-profile-receipt-v1`` with the same fields, derived from the bytes
  the runtime actually loaded — never echoed from the request.

The request contract is an expectation, not identity, authorization, or a
second authority: an absent contract launches normally (and still records a
receipt); a present-but-diverged contract is a typed pre-launch conflict
(:class:`ProfileLaunchConflict`, zero effects, retry with a fresh contract),
never a silent substitution. A malformed contract is refused as a client
error before anything is read twice or launched.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from cli_agent_orchestrator.constants import PROVIDERS
from cli_agent_orchestrator.models.agent_profile import AgentProfile
from cli_agent_orchestrator.providers.base import SealedLaunchMaterial
from cli_agent_orchestrator.utils import agent_profiles
from cli_agent_orchestrator.utils.agent_profiles import (
    INSTALLED_AGENT_STORE_PROVENANCE,
    LEGACY_INSTALLED_STORE_PROVENANCE,
    parse_agent_profile_text,
    read_agent_profile_bytes_with_provenance,
)
from cli_agent_orchestrator.utils.env import resolve_env_vars
from cli_agent_orchestrator.utils.skills import build_skill_catalog
from cli_agent_orchestrator.utils.tool_mapping import resolve_allowed_tools

logger = logging.getLogger(__name__)

PROFILE_LAUNCH_CONTRACT_SCHEMA = "cao-profile-launch-contract-v1"
PROFILE_RECEIPT_SCHEMA = "cao-profile-receipt-v1"

SUPERVISOR_ROLE = "supervisor"

_CONTRACT_FIELDS = (
    "profile",
    "role",
    "provider",
    "model",
    "effort",
    "provenance",
    "source_path",
    "sha256",
)


class ProfileNotFoundError(FileNotFoundError):
    """The named profile is unavailable to CAO's configured stores.

    A launch-boundary error raised before any tmux/session/provider effect:
    the supervisor launch refuses the name rather than degrading to a
    fallback provider with no profile. Subclasses ``FileNotFoundError`` so
    existing handlers keep classifying it as a missing profile, while the
    HTTP boundary narrows on this type so a late, unrelated
    ``FileNotFoundError`` (tmux, FIFO, store) is never misreported as a
    client error.
    """


class ProfileInvalidError(ValueError):
    """The profile source exists but does not parse into an ``AgentProfile``.

    A launch-boundary client error (``ValueError`` maps to 400 at the HTTP
    boundary): the profile bytes are present but unusable, so the operator
    fixes the profile rather than retrying an identical launch.
    """


def canonical_source_path(source_path: str) -> str:
    """Normalize a profile source path into its canonical comparison form.

    Filesystem paths resolve through ``os.path.realpath``, so an
    aliased or symlinked spelling of the same physical profile compares
    equal to the runtime-resolved path. The ``built-in:<name>`` pseudo-path
    for the packaged store is already canonical and compares exactly.
    Canonicalization never weakens identity: both ``source_path`` and
    ``sha256`` must match, so identical bytes at a genuinely different
    canonical path still diverge (no sha-only identity).
    """
    if source_path.startswith("built-in:"):
        return source_path
    return os.path.realpath(source_path)


class ProfileLaunchUnsupported(RuntimeError):
    """A sealed contract names a provider/path that cannot consume the frozen profile.

    Raised after the single launch-boundary read and any contract validation
    but before any tmux/session/DB/provider/sidecar effect: the adapter
    would launch provider-native named artifacts (or resolve prompt/tools
    from a mutable native store), so the runtime refuses to validate or
    persist CAO profile A while the supervisor would consume native
    profile B. Carries the deciding adapter reason and a recovery action;
    the HTTP boundary maps it to an operation-scoped 422. No receipt
    exists for a refused launch, and none is manufactured later.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        source_path: str,
        reason: str,
        recovery: str,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.source_path = source_path
        self.reason = reason
        self.recovery = recovery


class ProfileLaunchConflict(RuntimeError):
    """A supervisor launch contract diverged from the runtime-loaded profile.

    Raised before any tmux, session, database, or provider effect, so the
    failed launch owns nothing and a retry with a freshly preflighted
    contract converges. Carries the divergent fields and the exact values on
    each side so the conductor can re-preflight without guessing.
    """

    def __init__(self, message: str, *, divergent_fields: list, retry: str) -> None:
        super().__init__(message)
        self.divergent_fields = list(divergent_fields)
        self.retry = retry


@dataclass(frozen=True)
class ProfileLaunchContext:
    """The one immutable profile load a supervisor launch consumes.

    Built exactly once per launch from a single source read: the parsed
    profile every downstream consumer (allowed-tools resolution, skill
    catalog, pre-task bootstrap, provider argv) shares, plus the source
    metadata and digest the contract is validated against and the resolved
    provider/model/effort the provider launch pins.
    """

    profile_name: str
    profile: AgentProfile
    source_path: str
    provenance: str
    sha256: str
    provider: str
    model: Optional[str]
    effort: Optional[str]


def _resolved_effort(profile: AgentProfile) -> Optional[str]:
    """The effort channel a launch pins, today Codex-only.

    ``codexConfig.model_reasoning_effort`` is the single profile-declared
    effort knob; anything else is provider-default and recorded as absent
    (``None``), never invented.
    """
    codex_config = getattr(profile, "codexConfig", None) or {}
    if isinstance(codex_config, Mapping):
        raw = codex_config.get("model_reasoning_effort")
    else:
        raw = getattr(codex_config, "model_reasoning_effort", None)
    if raw is None:
        return None
    text = str(raw)
    return text if text else None


def resolve_launch_route(
    profile: AgentProfile,
    *,
    explicit_provider: Optional[str] = None,
    fallback_provider: str = "kiro_cli",
) -> "tuple[str, Optional[str], Optional[str]]":
    """Resolve ``(provider, model, effort)`` from one already-loaded profile.

    Precedence mirrors ``resolve_provider`` without re-reading the store: an
    explicit launch provider wins, then the profile's own valid ``provider``,
    then the fallback. This is the effective route the provider argv pins,
    so sealing it here (rather than re-deriving it downstream) keeps the
    receipt and the launch identical:

    * Codex applies its config seam — ``codexConfig.model`` over the bare
      ``profile.model``, effort from ``codexConfig.model_reasoning_effort``
      — mirroring the pre-task bootstrap's effective route.
    * Every other provider pins the bare ``profile.model`` with no effort:
      passing that through the ``expected_*`` seam yields exactly what those
      adapters derive internally, so the pin changes nothing while making
      the route exact.
    """
    if explicit_provider is not None:
        provider = explicit_provider
    elif profile.provider and profile.provider in PROVIDERS:
        provider = profile.provider
    else:
        if profile.provider:
            logger.warning(
                "Agent profile '%s' has invalid provider '%s'. "
                "Valid providers: %s. Falling back to '%s'.",
                profile.name,
                profile.provider,
                PROVIDERS,
                fallback_provider,
            )
        provider = fallback_provider
    if provider == "codex":
        codex_config = getattr(profile, "codexConfig", None) or {}
        config_model = (
            codex_config.get("model")
            if isinstance(codex_config, Mapping)
            else getattr(codex_config, "model", None)
        )
        model = config_model if isinstance(config_model, str) and config_model else profile.model
        return (provider, model, _resolved_effort(profile))
    return (provider, profile.model, None)


def load_supervisor_launch_context(
    profile_name: str,
    *,
    explicit_provider: Optional[str] = None,
    fallback_provider: str = "kiro_cli",
) -> ProfileLaunchContext:
    """Read the profile source exactly once and freeze the launch context.

    The single binary read supplies the exact source bytes; those bytes are
    decoded once (strict UTF-8), parsed, and hashed — so the parsed profile,
    the resolved route, and the digest all describe the same snapshot, and
    no later stage of the same launch may read the profile by name again
    and observe different bytes.

    Raises :class:`ProfileNotFoundError` when the name resolves to no
    configured store, and :class:`ProfileInvalidError` when the source
    bytes are not valid UTF-8 or do not parse — both before any
    tmux/session/provider effect.
    """
    try:
        raw, source_path, provenance = read_agent_profile_bytes_with_provenance(profile_name)
    except FileNotFoundError as exc:
        raise ProfileNotFoundError(str(exc)) from exc
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        raw_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProfileInvalidError(
            f"agent profile '{profile_name}' from {source_path} is not valid UTF-8: {exc}"
        ) from exc
    try:
        profile = parse_agent_profile_text(resolve_env_vars(raw_text), profile_name)
    except Exception as exc:
        raise ProfileInvalidError(
            f"agent profile '{profile_name}' from {source_path} does not parse: {exc}"
        ) from exc
    provider, model, effort = resolve_launch_route(
        profile, explicit_provider=explicit_provider, fallback_provider=fallback_provider
    )
    return ProfileLaunchContext(
        profile_name=profile_name,
        profile=profile,
        # The receipt records the canonical form, matching what validation
        # compares the request path against.
        source_path=canonical_source_path(source_path),
        provenance=provenance,
        sha256=sha256,
        provider=provider,
        model=model,
        effort=effort,
    )


def build_sealed_launch_material(
    context: ProfileLaunchContext,
    *,
    allowed_tools: Optional[list] = None,
) -> SealedLaunchMaterial:
    """Freeze the capability-gate inputs from an already-loaded context.

    Pure function of the context plus the launch's explicit tool override:
    the resolved route the context carries, the profile's system prompt,
    the composed skill catalog for the profile's skill scope, and the
    effective allowed-tools policy (the explicit launch list when given,
    else exactly what terminal creation would resolve from
    ``allowedTools``/``role``/MCP names). No store read, no tmux, no DB —
    safe to evaluate before any launch effect, and the same object threads
    into provider construction so the gate and the launch cannot disagree.
    """
    profile = context.profile
    mcp_names = list(profile.mcpServers) if profile.mcpServers else None
    if allowed_tools is not None:
        effective_tools = list(allowed_tools)
    else:
        effective_tools = resolve_allowed_tools(profile.allowedTools, profile.role, mcp_names)
    return SealedLaunchMaterial(
        profile=profile,
        model=context.model,
        effort=context.effort,
        system_prompt=profile.system_prompt or "",
        skill_text=build_skill_catalog(profile.skills),
        allowed_tools=tuple(effective_tools),
    )


def _provenance_matches(request_value: str, context: ProfileLaunchContext) -> bool:
    """Compare the contract's provenance against the loaded context.

    Exact match is the rule. One compatibility is allowed: an older
    conductor-era contract may still carry ``"local"`` for the installed
    agent store. It is accepted only when the runtime source really is the
    installed store (canonical path within it) reporting the canonical
    ``installed-agent-store`` label — the ``sha256`` field is compared
    independently, so both must still agree. Any other source, including
    ``custom`` and provider-specific directories, never aliases: a
    ``"local"`` claim over those bytes is a divergence, not a spelling.
    """
    if request_value == context.provenance:
        return True
    return (
        request_value == LEGACY_INSTALLED_STORE_PROVENANCE
        and context.provenance == INSTALLED_AGENT_STORE_PROVENANCE
        and _is_within_installed_store(context.source_path)
    )


def _is_within_installed_store(canonical_source: str) -> bool:
    """Whether a canonical source path lives in the installed agent store."""
    if canonical_source.startswith("built-in:"):
        return False
    store = os.path.realpath(agent_profiles.LOCAL_AGENT_STORE_DIR)
    try:
        return os.path.commonpath([canonical_source, store]) == store
    except (ValueError, OSError):
        return False


def validate_profile_contract(contract: Mapping[str, Any], context: ProfileLaunchContext) -> None:
    """Validate an optional launch contract against the loaded context.

    Raises ``ValueError`` for a malformed contract (wrong schema, wrong role,
    missing keys, wrong types) and :class:`ProfileLaunchConflict` for a
    well-formed contract whose expected values diverge from what the runtime
    loaded. Either raises before any launch effect.

    ``source_path`` compares in canonical form on both sides, so an aliased
    or symlinked spelling of the same physical profile agrees — while a
    genuinely different canonical path still diverges even when the bytes
    (and therefore ``sha256``) are identical. ``provenance`` compares
    exactly, except that a legacy ``"local"`` claim is accepted for a
    source inside the installed agent store (see :func:`_provenance_matches`).
    """
    if not isinstance(contract, Mapping):
        raise ValueError("profile_contract must be an object")
    schema = contract.get("schema")
    if schema != PROFILE_LAUNCH_CONTRACT_SCHEMA:
        raise ValueError(f"profile_contract schema must be {PROFILE_LAUNCH_CONTRACT_SCHEMA!r}")
    role = contract.get("role")
    if role != SUPERVISOR_ROLE:
        raise ValueError("profile_contract role must be 'supervisor'")
    missing = [field for field in _CONTRACT_FIELDS if field not in contract]
    if missing:
        raise ValueError(f"profile_contract is missing fields: {sorted(missing)}")
    for field in ("profile", "provider", "provenance", "source_path", "sha256"):
        if not isinstance(contract[field], str) or not contract[field]:
            raise ValueError(f"profile_contract field {field!r} must be a non-empty string")
    for field in ("model", "effort"):
        value = contract[field]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"profile_contract field {field!r} must be a string or null")

    expected = {
        "profile": context.profile_name,
        "role": SUPERVISOR_ROLE,
        "provider": context.provider,
        "model": context.model,
        "effort": context.effort,
        "provenance": context.provenance,
        "source_path": context.source_path,
        "sha256": context.sha256,
    }
    divergences = []
    # The shape loop above already proved source_path is a non-empty string.
    request_path = contract["source_path"]
    assert isinstance(request_path, str)
    for field in _CONTRACT_FIELDS:
        if field == "source_path":
            same = canonical_source_path(request_path) == canonical_source_path(context.source_path)
        elif field == "provenance":
            same = _provenance_matches(contract[field], context)
        else:
            same = contract[field] == expected[field]
        if not same:
            divergences.append(
                {
                    "field": field,
                    "expected": contract[field],
                    "observed": expected[field],
                }
            )
    divergent = divergences
    if divergent:
        fields = sorted(entry["field"] for entry in divergent)
        raise ProfileLaunchConflict(
            "supervisor profile contract diverged from the runtime-loaded profile "
            f"(fields: {', '.join(fields)}); no launch effect was produced",
            divergent_fields=divergent,
            retry=(
                "re-run conductor preflight against the current profile source "
                "and retry POST /sessions with the fresh profile_contract"
            ),
        )


def build_profile_receipt(context: ProfileLaunchContext) -> Dict[str, Any]:
    """The runtime-authored receipt persisted with the terminal row."""
    return {
        "schema": PROFILE_RECEIPT_SCHEMA,
        "profile": context.profile_name,
        "role": SUPERVISOR_ROLE,
        "provider": context.provider,
        "model": context.model,
        "effort": context.effort,
        "provenance": context.provenance,
        "source_path": context.source_path,
        "sha256": context.sha256,
    }
