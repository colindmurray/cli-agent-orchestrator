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
import json
import logging
import os
import re
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

#: The exact required contract set, shared by the missing-field and
#: extra-field checks and the drift comparison: omitting ``schema`` here
#: once turned a missing schema into a ``KeyError``/HTTP 500 instead of
#: a typed malformed 400.
_CONTRACT_FIELDS = (
    "schema",
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


#: The receipt data fields compared verbatim against a submitted
#: contract during adoption (the ``schema`` values differ by design:
#: receipts carry the receipt schema, contracts the contract schema).
RECEIPT_DATA_FIELDS = (
    "profile",
    "role",
    "provider",
    "model",
    "effort",
    "provenance",
    "source_path",
    "sha256",
)


class ProfileAdoptionMismatch(RuntimeError):
    """A contract-bearing retry names a live duplicate that cannot be adopted.

    Raised with zero mutation when the session already holds terminal
    state but no single live winner matches the submitted contract
    exactly: a divergent field, a missing/corrupt receipt, a
    pending/partial/dead/superseded/stale/ambiguous row, a dead tmux
    identity, a roster incarnation that is not this supervisor's live
    one, or a provider launch that is not ready. The HTTP boundary maps
    it to 409. Deliberately a ``RuntimeError``, not a ``ValueError``:
    the boundary maps ``ValueError`` to 400, which is reserved for the
    ordinary no-contract duplicate.

    Recovery is proportionate: delete the session and retry, retry with
    a fresh session name — or, for an in-flight launch, retry the same
    request after it completes.
    """

    def __init__(
        self,
        message: str,
        *,
        session_name: str,
        reason: str,
        recovery: str,
    ) -> None:
        super().__init__(message)
        self.session_name = session_name
        self.reason = reason
        self.recovery = recovery


def receipt_provenance_matches(
    contract_value: Any, stored_value: Any, stored_source_path: Any
) -> bool:
    """Whether a stored receipt provenance satisfies a contract provenance.

    The same canonical installed-store/legacy equivalence as launch
    validation: an exact match, or a ``local`` contract satisfied by an
    ``installed-agent-store`` receipt whose source path still lives
    under the installed store.
    """
    if contract_value == stored_value:
        return True
    return (
        contract_value == LEGACY_INSTALLED_STORE_PROVENANCE
        and stored_value == INSTALLED_AGENT_STORE_PROVENANCE
        and isinstance(stored_source_path, str)
        and _is_within_installed_store(stored_source_path)
    )


def stored_receipt_divergences(
    contract: Mapping[str, Any], stored: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Field-by-field divergences between a contract and a stored receipt.

    Compares every :data:`RECEIPT_DATA_FIELDS` entry: provenance uses
    the canonical equivalence, ``source_path`` compares canonical
    forms, everything else compares exact values (both sides are
    canonical at this point — the parser normalizes the contract, the
    writer canonicalizes the receipt). Returns the divergent fields in
    contract order; empty means an exact match.
    """
    divergent: list[dict[str, Any]] = []
    for field in RECEIPT_DATA_FIELDS:
        expected = contract.get(field)
        actual = stored.get(field)
        if field == "provenance":
            same = receipt_provenance_matches(expected, actual, stored.get("source_path"))
        elif field == "source_path":
            # A corrupt stored receipt may carry a non-string path: that
            # is a divergence, never a canonicalization crash.
            same = (
                isinstance(expected, str)
                and isinstance(actual, str)
                and canonical_source_path(expected) == canonical_source_path(actual)
            )
        else:
            same = expected == actual
        if not same:
            divergent.append({"field": field, "expected": expected, "actual": actual})
    return divergent


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


class ProfileContractMalformed(ValueError):
    """The profile_contract request field is not a well-formed expectation.

    A launch-boundary client error (``ValueError`` maps to 400 at the HTTP
    boundary): non-mapping input, missing or extra fields, wrong schema or
    role, wrong value types, an invalid source-path form, or a SHA that is
    not exactly 64 hexadecimal characters. Well-formed values that merely
    differ from the loaded profile are drift (:class:`ProfileLaunchConflict`,
    409), not malformation.
    """


_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _is_valid_contract_source_path(value: str) -> bool:
    """Accept absolute filesystem paths and ``built-in:<name>.md`` forms.

    Aliased spellings (symlinks, ``..``, redundant separators) are NOT
    malformed here: validation canonicalizes both sides and compares, so
    an alias of the same physical profile agrees. Only the form is
    checked — a relative path or a malformed ``built-in:`` pseudo-path
    is a 400.
    """
    if value.startswith("built-in:"):
        rest = value[len("built-in:") :]
        return bool(rest) and rest.endswith(".md") and "/" not in rest and "\\" not in rest
    return os.path.isabs(value)


def parse_profile_contract(raw: Any) -> Dict[str, Any]:
    """Endpoint-scoped strict parse of the profile_contract request field.

    The HTTP layer passes the raw JSON value through (``Optional[Any]`` —
    no global FastAPI validation change), and this single function decides
    shape: it returns a normalized contract mapping (SHA lowercased) or
    raises :class:`ProfileContractMalformed`. Uppercase SHA hex is
    accepted and normalized for comparison; receipts always emit the
    lowercase digest. Any non-empty-string provenance passes shape here —
    a well-formed value that differs is drift (409), including the legacy
    ``"local"`` compatibility the validator applies.
    """
    if not isinstance(raw, Mapping):
        raise ProfileContractMalformed("profile_contract must be an object")
    # The missing and extra checks share the one exact required set, so
    # no field can be required-but-unlisted (KeyError/500) or
    # listed-but-unrequired.
    missing = [field for field in _CONTRACT_FIELDS if field not in raw]
    if missing:
        raise ProfileContractMalformed(f"profile_contract is missing fields: {sorted(missing)}")
    extra = sorted(key for key in raw if key not in _CONTRACT_FIELDS)
    if extra:
        raise ProfileContractMalformed(f"profile_contract has unknown fields: {extra}")
    if raw["schema"] != PROFILE_LAUNCH_CONTRACT_SCHEMA:
        raise ProfileContractMalformed(
            f"profile_contract schema must be {PROFILE_LAUNCH_CONTRACT_SCHEMA!r}"
        )
    if raw["role"] != SUPERVISOR_ROLE:
        raise ProfileContractMalformed("profile_contract role must be 'supervisor'")
    for field in ("profile", "provider", "provenance"):
        if not isinstance(raw[field], str) or not raw[field]:
            raise ProfileContractMalformed(
                f"profile_contract field {field!r} must be a non-empty string"
            )
    for field in ("model", "effort"):
        if raw[field] is not None and not isinstance(raw[field], str):
            raise ProfileContractMalformed(
                f"profile_contract field {field!r} must be a string or null"
            )
    source_path = raw["source_path"]
    if not isinstance(source_path, str) or not _is_valid_contract_source_path(source_path):
        raise ProfileContractMalformed(
            "profile_contract field 'source_path' must be an absolute path "
            "or a 'built-in:<name>.md' pseudo-path"
        )
    sha256 = raw["sha256"]
    if not isinstance(sha256, str) or not _SHA256_HEX_RE.match(sha256):
        raise ProfileContractMalformed(
            "profile_contract field 'sha256' must be exactly 64 hexadecimal characters"
        )
    parsed = dict(raw)
    parsed["sha256"] = sha256.lower()
    return parsed


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
        "schema": PROFILE_LAUNCH_CONTRACT_SCHEMA,
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


SUPERVISOR_CREATE_REQUEST_SCHEMA = "cao-supervisor-create-request-v1"

#: Request values that spawn the memory-manager sidecar in the new
#: session. Shared by the HTTP boundary and the request fingerprint so
#: the fingerprinted decision and the spawned sidecar can never disagree.
MEMORY_MANAGER_TRUTHY_VALUES = ("true", "1", "yes")


def memory_manager_enabled(value: Any) -> bool:
    """Whether a memory_manager request value spawns the sidecar."""
    return value is not None and str(value).lower() in MEMORY_MANAGER_TRUTHY_VALUES


def canonical_env_hash(env_vars: Any) -> str:
    """SHA-256 over the canonical effective env map.

    Keys sort, so ``{"B": "2", "A": "1"}`` and ``{"A": "1", "B": "2"}``
    share an identity. ``None`` and ``{}`` share the empty-map identity.
    Only the digest ever persists — raw env values (which may carry
    secrets) are never written to a row. Non-string keys or values are
    a malformed request (``ValueError`` maps to 400), never coerced.
    """
    mapping = dict(env_vars or {})
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"env_vars has non-string key {key!r}")
        if not isinstance(item, str):
            raise ValueError(f"env_vars key {key!r} must be a string")
    canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_request_tools(allowed_tools: Any) -> Optional[list]:
    """The request's explicit tool policy in fingerprint form.

    ``None`` (resolve from the profile at load) stays ``None``: it is a
    distinct request fact from an explicit list, even when resolution
    would agree. Order is preserved — a reordered list is a different
    request, refused rather than mis-adopted. Non-string entries are
    malformed (``ValueError``), never coerced.
    """
    if allowed_tools is None:
        return None
    if not isinstance(allowed_tools, (list, tuple)):
        raise ValueError("allowed_tools must be a list of strings or null")
    tools = list(allowed_tools)
    for item in tools:
        if not isinstance(item, str):
            raise ValueError(f"allowed_tools entries must be strings, got {item!r}")
    return tools


def build_supervisor_create_request(
    *,
    session_name: str,
    agent_profile: str,
    provider: Optional[str],
    contract: Mapping[str, Any],
    working_directory: Optional[str],
    allowed_tools: Any,
    env_vars: Any,
    memory_manager: Any,
) -> Dict[str, Any]:
    """One canonical internal request document for a supervisor create.

    Pure function of the request alone — no profile read, no store
    access — so the claim-owned sequence computes it before inspecting
    the durable row. ``session_name`` must already be normalized,
    ``contract`` already strict-parsed. ``working_directory`` resolves
    exactly like pane creation does for activated providers (``None``
    becomes the canonical current directory, aliases resolve); for
    other providers it is the same canonicalization of intent, so
    byte-identical retries always agree. Only the env digest enters
    the document — raw env values never persist.
    """
    return {
        "schema": SUPERVISOR_CREATE_REQUEST_SCHEMA,
        "session_name": session_name,
        "role": SUPERVISOR_ROLE,
        "agent_profile": agent_profile,
        "provider": provider,
        "profile_contract": dict(contract),
        "working_directory": os.path.realpath(working_directory or os.getcwd()),
        "allowed_tools": normalize_request_tools(allowed_tools),
        "env_vars_sha256": canonical_env_hash(env_vars),
        "memory_manager": memory_manager_enabled(memory_manager),
    }


def supervisor_create_request_fingerprint(request: Mapping[str, Any]) -> str:
    """SHA-256 hex identity of a canonical create-request document.

    Deterministic compact JSON (sorted keys): two byte-identical
    requests share a fingerprint, and any request-only difference —
    cwd, tools, env, provider, model/effort via the contract — changes
    it. Persisted on the terminal row as the request identity the
    adoption lookup compares exactly.
    """
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
