"""Provider manager as module singleton with direct terminal_id → provider mapping."""

import logging
from typing import Any, Dict, List, Optional, Type

from sqlalchemy.exc import OperationalError

from cli_agent_orchestrator.clients.database import get_terminal_metadata, get_terminal_metadata_v2
from cli_agent_orchestrator.models.provider import ProviderType
from cli_agent_orchestrator.providers.antigravity_cli import AntigravityCliProvider
from cli_agent_orchestrator.providers.base import (
    BaseProvider,
    PreparedSealedLaunch,
    SealedLaunchMaterial,
    SealedPreparationUnsupported,
    SealedProfileSupport,
)
from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider
from cli_agent_orchestrator.providers.codex import CodexProvider
from cli_agent_orchestrator.providers.copilot_cli import CopilotCliProvider
from cli_agent_orchestrator.providers.cursor_cli import CursorCliProvider
from cli_agent_orchestrator.providers.hermes import HermesProvider
from cli_agent_orchestrator.providers.kimi_cli import KimiCliProvider
from cli_agent_orchestrator.providers.kiro_cli import KiroCliProvider
from cli_agent_orchestrator.providers.mock_cli import MockCliProvider
from cli_agent_orchestrator.providers.muse_cli import MuseCliProvider
from cli_agent_orchestrator.providers.opencode_cli import OpenCodeCliProvider

logger = logging.getLogger(__name__)

# Sealed-profile capability routing (cond-0817). Maps each constructed
# provider type to its adapter class WITHOUT constructing it, so the
# session boundary can refuse a sealed contract before any effect. This
# mirrors the create_provider construction chain below — it is a class
# lookup, not a supported-provider allowlist: the decision itself lives on
# each adapter (conservative base default unsupported), and the capability
# matrix test pins the two together so a new branch cannot drift. Unknown
# or unmapped types are unsupported, never silently supported.
_ADAPTER_CLASS_BY_TYPE: Dict[str, Type[BaseProvider]] = {
    ProviderType.KIRO_CLI.value: KiroCliProvider,
    ProviderType.CLAUDE_CODE.value: ClaudeCodeProvider,
    ProviderType.CODEX.value: CodexProvider,
    ProviderType.COPILOT_CLI.value: CopilotCliProvider,
    ProviderType.KIMI_CLI.value: KimiCliProvider,
    ProviderType.MUSE_CLI.value: MuseCliProvider,
    ProviderType.OPENCODE_CLI.value: OpenCodeCliProvider,
    ProviderType.HERMES.value: HermesProvider,
    ProviderType.CURSOR_CLI.value: CursorCliProvider,
    ProviderType.ANTIGRAVITY_CLI.value: AntigravityCliProvider,
    ProviderType.MOCK_CLI.value: MockCliProvider,
}


class TerminalAssignedRouteIncompleteError(Exception):
    """A generation-bound terminal lacks a complete assigned route pin."""


class TerminalMetadataCollisionError(ValueError):
    """One terminal ID is present in both isolated metadata vintages."""


def _is_missing_v2_table(error: OperationalError) -> bool:
    """Return whether an OperationalError is only an uncreated v2 table.

    A pre-v2 database legitimately has no ``managed_launch_v2_terminals``
    table.  Every other database/read failure must remain visible to callers;
    treating a locked or corrupt v2 surface as an absent terminal could cause
    restart code to reconstruct the wrong provider or overwrite state.
    """
    # SQLAlchemy keeps the database's actual missing relation in ``orig``;
    # the statement text may name the v2 relation even when SQLite reports a
    # different missing table, so classify from the driver error only.
    origin = str(getattr(error, "orig", "")).lower()
    return "no such table" in origin and "managed_launch_v2_terminals" in origin


class ProviderManager:
    """Simplified provider manager with direct mapping."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseProvider] = {}
        # Identity for providers reconstructed from durable metadata.  Providers
        # created directly by callers predate this opt-in cache contract and are
        # intentionally left untagged; the managed-v2 path revalidates those
        # entries against the durable row before returning them.
        self._provider_identities: Dict[str, tuple] = {}

    def sealed_launch_support(
        self, provider_type: str, material: Optional[SealedLaunchMaterial]
    ) -> SealedProfileSupport:
        """Evaluate sealed-launch capability without constructing anything.

        Pure class-level dispatch to the adapter's own predicate: no
        provider instance, no tmux, no DB, no side effects. This table is a
        lookup, not a supported-provider allowlist — the decision lives per
        adapter. Unknown or unmapped provider types are unsupported.
        """
        adapter = _ADAPTER_CLASS_BY_TYPE.get(provider_type)
        if adapter is None:
            return SealedProfileSupport(
                supported=False,
                reason=(
                    f"unknown provider type {provider_type!r}: no adapter declares "
                    "frozen-launch-material support"
                ),
            )
        return adapter.supports_sealed_launch(material)

    def prepare_sealed_launch(
        self, provider_type: str, material: Optional[SealedLaunchMaterial]
    ) -> PreparedSealedLaunch:
        """Validate + serialize the frozen material without any effect.

        Pure class-level dispatch to the adapter's own preparation: no
        provider instance, no tmux, no DB, no files. The session
        boundary runs this exactly once after the capability gate and
        before any effect; malformed material raises
        :class:`SealedPreparationUnsupported` (mapped to 422 with zero
        effects), unexpected failures propagate (still pre-effect).
        Unknown or unmapped provider types are unpreparable, never
        silently preparable.
        """
        adapter = _ADAPTER_CLASS_BY_TYPE.get(provider_type)
        if adapter is None:
            raise SealedPreparationUnsupported(
                f"unknown provider type {provider_type!r}: no adapter prepares "
                "frozen-launch-material"
            )
        return adapter.prepare_sealed_launch(material)

    def create_provider(
        self,
        provider_type: str,
        terminal_id: str,
        tmux_session: str,
        tmux_window: str,
        agent_profile: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        skill_prompt: Optional[str] = None,
        model: Optional[str] = None,
        trusted_project_root: Optional[str] = None,
        expected_model: Optional[str] = None,
        expected_effort: Optional[str] = None,
        native_session_id: Optional[str] = None,
        codex_profile_material: Optional[dict] = None,
        codex_executable: Optional[str] = None,
        # The launch's already-loaded profile (cond-0817). Forwarded to the
        # adapter constructor so the launch argv/config consumes this exact
        # object and never reloads the profile by name. ``None`` keeps the
        # legacy per-adapter load for direct construction.
        launch_profile: Optional[Any] = None,
        # The sealed material the gate froze (cond-0817). Forwarded to the
        # adapters whose launch spans bootstrap and resume (Codex,
        # Antigravity) so both consume the admitted inputs verbatim.
        # ``None`` keeps legacy per-adapter resolution.
        sealed_launch_material: Optional[SealedLaunchMaterial] = None,
        # The one prepared sealed-launch value (cond-0817 repair).
        # Forwarded to the adapters whose launch serializes MCP material
        # (Codex, Claude, Kimi, Cursor) so they consume the pre-effect
        # validated artifact instead of re-resolving. ``None`` keeps
        # legacy per-adapter resolution.
        prepared_sealed_launch: Optional[PreparedSealedLaunch] = None,
    ) -> BaseProvider:
        """Create and store provider instance."""
        try:
            provider: BaseProvider
            if provider_type == ProviderType.KIRO_CLI.value:
                if not agent_profile:
                    raise ValueError("Kiro CLI provider requires agent_profile parameter")
                provider = KiroCliProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    launch_profile=launch_profile,
                )
            elif provider_type == ProviderType.CLAUDE_CODE.value:
                provider = ClaudeCodeProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    skill_prompt=skill_prompt,
                    native_session_id=native_session_id,
                    launch_profile=launch_profile,
                    prepared_sealed_launch=prepared_sealed_launch,
                )
            elif provider_type == ProviderType.CODEX.value:
                provider = CodexProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    skill_prompt=skill_prompt,
                    trusted_project_root=trusted_project_root,
                    expected_model=expected_model,
                    expected_effort=expected_effort,
                    native_session_id=native_session_id,
                    codex_profile_material=codex_profile_material,
                    codex_executable=codex_executable,
                    launch_profile=launch_profile,
                    sealed_launch_material=sealed_launch_material,
                )
            elif provider_type == ProviderType.COPILOT_CLI.value:
                provider = CopilotCliProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    model=model,
                )
            elif provider_type == ProviderType.KIMI_CLI.value:
                provider = KimiCliProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    skill_prompt=skill_prompt,
                    expected_model=expected_model,
                    expected_effort=expected_effort,
                    launch_profile=launch_profile,
                    prepared_sealed_launch=prepared_sealed_launch,
                )
            elif provider_type == ProviderType.MUSE_CLI.value:
                provider = MuseCliProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    skill_prompt=skill_prompt,
                    expected_model=expected_model,
                    expected_effort=expected_effort,
                    launch_profile=launch_profile,
                )
            elif provider_type == ProviderType.OPENCODE_CLI.value:
                provider = OpenCodeCliProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    model=model,
                )
            elif provider_type == ProviderType.HERMES.value:
                provider = HermesProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    skill_prompt=skill_prompt,
                    launch_profile=launch_profile,
                )
            elif provider_type == ProviderType.CURSOR_CLI.value:
                provider = CursorCliProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    model=model,
                    skill_prompt=skill_prompt,
                    launch_profile=launch_profile,
                    prepared_sealed_launch=prepared_sealed_launch,
                )
            elif provider_type == ProviderType.ANTIGRAVITY_CLI.value:
                provider = AntigravityCliProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    agent_profile,
                    allowed_tools,
                    model=expected_model or model,
                    skill_prompt=skill_prompt,
                    native_session_id=native_session_id,
                    effort=expected_effort,
                    launch_profile=launch_profile,
                    sealed_launch_material=sealed_launch_material,
                )
            # --- Credentials-free mock provider (test/CI infrastructure) ---
            elif provider_type == ProviderType.MOCK_CLI.value:
                provider = MockCliProvider(
                    terminal_id,
                    tmux_session,
                    tmux_window,
                    allowed_tools,
                )
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")

            # Store in direct mapping
            self._provider_identities.pop(terminal_id, None)
            self._providers[terminal_id] = provider
            logger.info(f"Created {provider_type} provider for terminal: {terminal_id}")
            return provider

        except Exception as e:
            logger.error(
                f"Failed to create provider {provider_type} for terminal {terminal_id}: {e}"
            )
            raise

    def get_provider(
        self, terminal_id: str, *, include_managed_v2: bool = False
    ) -> Optional[BaseProvider]:
        """Get provider instance, creating on-demand if not found.

        Args:
            terminal_id: Terminal ID to get provider for

        Returns:
            Provider instance

        Raises:
            ValueError: If terminal not found in database or provider creation fails
            RuntimeError/OperationalError: If metadata cannot be read (distinct from absence)

        ``include_managed_v2`` is intentionally opt-in.  Most callers operate
        on the legacy projection and must not make v2 native-TUI terminals
        visible to old status/projection behavior.  The FIFO/ACP status path
        opts in because it is a cross-vintage consumer that must recover the
        provider for a v2 managed row after restart.
        """
        if not include_managed_v2:
            # Keep the default path's direct in-memory behavior unchanged.
            provider = self._providers.get(terminal_id)
            cached_identity = self._provider_identities.get(terminal_id)
            if provider and (cached_identity is None or cached_identity[0] != "v2"):
                return provider
            metadata = get_terminal_metadata(terminal_id)
        else:
            # Probe both isolated metadata vintages.  The v1 row remains the
            # authoritative shape for old terminals, while a v2-only row must
            # be recoverable after a server restart.  Always perform the second
            # probe so an accidental cross-vintage ID collision cannot silently
            # select a provider from whichever table happened to be queried
            # first.
            # Do not catch exceptions: an unreadable projection must propagate
            # distinctly from an actually absent terminal, otherwise callers
            # cannot distinguish "no row" from "could not read row" and may
            # treat an unreadable managed terminal as a legacy row.
            # This is the first tier of a deliberate v1-then-v2 probe.  A v1
            # miss is expected for every healthy v2-only terminal, so suppress
            # the legacy lookup's recurring missing-row warning; only a caller
            # that needs to report both tiers empty should emit a not-found
            # diagnostic.
            v1_metadata = get_terminal_metadata(terminal_id, warn_if_missing=False)
            try:
                v2_metadata = get_terminal_metadata_v2(terminal_id)
            except OperationalError as error:
                if not _is_missing_v2_table(error):
                    raise
                v2_metadata = None

            if v1_metadata is not None and v2_metadata is not None:
                raise TerminalMetadataCollisionError(
                    f"Terminal {terminal_id} exists in both v1 and v2 metadata stores"
                )

            metadata = v1_metadata
            if metadata is None and v2_metadata is not None:
                # The v2 schema prefixes fields whose names would otherwise
                # collide with v1 columns.  Provider constructors intentionally
                # consume the common ``native_session_id`` name, so normalize
                # only that input at this manager boundary; assigned route pins
                # remain absent/nullable for v2 and must not trigger v1's
                # completeness guard.
                metadata = dict(v2_metadata)
                metadata["native_session_id"] = metadata.get("v2_native_session_id")

        if not metadata:
            raise ValueError(f"Terminal {terminal_id} not found in database")

        provider = self._providers.get(terminal_id)
        if include_managed_v2 and provider is not None:
            cached_identity = self._provider_identities.get(terminal_id)
            durable_identity = self._provider_identity(metadata)
            if cached_identity == durable_identity:
                return provider
            if metadata.get("protocol_vintage") != "v2" and cached_identity is None:
                # A provider created directly by a legacy caller has no
                # durable identity tag, but the v1 lookup is still the exact
                # legacy row that this opt-in call resolved.
                return provider
            # The opt-in path must not let an untagged or stale provider hide a
            # v2-only row (or a changed v2 incarnation).  Drop the mapping so
            # create_provider's normal assignment installs the reconstruction.
            self._providers.pop(terminal_id, None)
            self._provider_identities.pop(terminal_id, None)

        # Managed v1 rows are classified by non-null ``generation`` alone.
        # A row with generation and missing assigned fields must not reconstruct
        # on ambient defaults even when native_session_id is still None.
        if metadata.get("protocol_vintage") != "v2" and metadata.get("generation") is not None:
            missing: list[str] = []
            if metadata.get("assigned_model") is None:
                missing.append("assigned_model")
            if metadata.get("assigned_effort") is None:
                missing.append("assigned_effort")
            if missing:
                raise TerminalAssignedRouteIncompleteError(
                    f"Terminal {terminal_id} generation {metadata.get('generation')!r} "
                    f"native_session_id {metadata.get('native_session_id')!r} "
                    f"missing {', '.join(missing)}"
                )

        # Create provider on-demand from persisted assigned route pin.
        create_kwargs = {"native_session_id": metadata.get("native_session_id")}
        if metadata.get("protocol_vintage") != "v2":
            # Assigned route pins are a v1 managed-terminal contract.  Do not
            # invent or enforce them for the isolated v2 observation row.
            create_kwargs.update(
                expected_model=metadata.get("assigned_model"),
                expected_effort=metadata.get("assigned_effort"),
            )
        provider = self.create_provider(
            metadata["provider"],
            terminal_id,
            metadata["tmux_session"],
            metadata["tmux_window"],
            metadata["agent_profile"],
            **create_kwargs,
        )
        # Restore shell_command baseline from DB so get_status() can detect kiro exit.
        # The terminal already exists in the DB, so its CLI has long since
        # launched — mark the provider as initialized so KiroCliProvider's
        # post-launch checks (Check 3) trust the restored baseline. Without
        # this, a restored terminal that has returned to the shell would be
        # misreported as PROCESSING indefinitely.
        if metadata.get("shell_command"):
            provider.shell_baseline = metadata["shell_command"]
            if hasattr(provider, "_initialized"):
                provider._initialized = True
        if include_managed_v2:
            self._provider_identities[terminal_id] = self._provider_identity(metadata)
        logger.info(f"Created provider on-demand for terminal {terminal_id}")
        return provider

    @staticmethod
    def _provider_identity(metadata: dict) -> tuple:
        """Return the durable identity used to validate opt-in cache hits."""
        return (
            metadata.get("protocol_vintage", "v1"),
            metadata.get("generation"),
            metadata.get("native_session_id"),
            metadata.get("provider"),
        )

    def cleanup_provider(self, terminal_id: str) -> None:
        """Cleanup provider and remove from map (used when terminal is deleted)."""
        try:
            provider = self._providers.pop(terminal_id, None)
            self._provider_identities.pop(terminal_id, None)
            if provider:
                provider.cleanup()
                logger.info(f"Cleaned up provider for terminal: {terminal_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup provider for terminal {terminal_id}: {e}")

    def list_providers(self) -> Dict[str, str]:
        """List all active providers (for debugging)."""
        return {
            terminal_id: provider.__class__.__name__
            for terminal_id, provider in self._providers.items()
        }


# Module-level singleton
provider_manager = ProviderManager()
