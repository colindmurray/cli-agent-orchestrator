"""Session service for session-level operations.

This module provides session management functionality for CAO, where a "session"
corresponds to a tmux session that may contain multiple terminal windows (agents).

Session Hierarchy:
- Session: A tmux session (e.g., "cao-my-project")
  - Terminal: A tmux window within the session (e.g., "developer-abc123")
    - Provider: The CLI agent running in the terminal (e.g., KiroCliProvider)

Key Operations:
- list_sessions(): Get all CAO-managed sessions (filtered by SESSION_PREFIX)
- get_session(): Get session details including all terminal metadata
- delete_session(): Clean up session, providers, database records, and tmux session

Session Lifecycle:
1. create_terminal() with new_session=True creates a new tmux session
2. Additional terminals are added via create_terminal() with new_session=False
3. delete_session() removes the entire session and all contained terminals
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cli_agent_orchestrator.backends.registry import get_backend
from cli_agent_orchestrator.clients.database import (
    PROVIDER_READINESS_FAILED,
    PROVIDER_READINESS_PENDING,
    PROVIDER_READINESS_READY,
    cas_terminal_provider_readiness,
    list_terminals_by_session,
)
from cli_agent_orchestrator.constants import SESSION_PREFIX
from cli_agent_orchestrator.models.terminal import Terminal
from cli_agent_orchestrator.plugins import (
    PluginRegistry,
    PostCreateSessionEvent,
    PostKillSessionEvent,
)
from cli_agent_orchestrator.providers.base import SealedPreparationUnsupported
from cli_agent_orchestrator.providers.manager import provider_manager
from cli_agent_orchestrator.services.plugin_dispatch import dispatch_plugin_event
from cli_agent_orchestrator.services.session_env import clear_session_env
from cli_agent_orchestrator.services.stable_agent_roster import ROLE_SUPERVISOR
from cli_agent_orchestrator.services.supervisor_profile_receipt import (
    ProfileLaunchUnsupported,
    build_profile_receipt,
    build_sealed_launch_material,
    build_supervisor_create_request,
    load_supervisor_launch_context,
    parse_profile_contract,
    supervisor_create_request_fingerprint,
    validate_profile_contract,
)
from cli_agent_orchestrator.services.terminal_service import (
    _terminal_from_adoption_row,
    create_terminal,
    find_supervisor_adoption_candidate,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CreateOrAdoptResult:
    """Internal create-or-adopt outcome for a session creation.

    ``terminal`` is the fresh or adopted terminal; ``adopted`` tells the
    HTTP boundary whether this request created (201) or adopted the
    exact durable ready winner with zero effects (200 plus the adoption
    indicator). Internal only — the public Terminal shape is unchanged.
    """

    terminal: Terminal
    adopted: bool


class SessionStopRefused(RuntimeError):
    """A zero-effect precondition refused the stop before any mutation.

    No pane was collected and no lifecycle row changed (e.g. an open callback
    recovery). Distinct from a partial collection so the API boundary can map
    it to a retryable conflict rather than an opaque failure.
    """


class SessionStopPartial(RuntimeError):
    """Some panes refused collection after the lifecycle row was already stopped.

    The row is preserved and a retry converges, so this is a visible, recoverable
    state — not a silent half-teardown. Carries the IDs that were collected and
    the structured per-terminal errors for server-side logging; only bounded,
    redacted guidance crosses the HTTP boundary.
    """

    def __init__(
        self,
        message: str,
        *,
        collected_terminal_ids: List[str],
        errors: List[Dict],
    ) -> None:
        super().__init__(message)
        self.collected_terminal_ids = list(collected_terminal_ids)
        self.errors = list(errors)


async def create_session(
    provider: str | None,
    agent_profile: str,
    session_name: str | None = None,
    working_directory: str | None = None,
    allowed_tools: list[str] | None = None,
    registry: PluginRegistry | None = None,
    env_vars: dict[str, str] | None = None,
    profile_contract: dict | None = None,
    memory_manager: str | None = None,
) -> CreateOrAdoptResult:
    """Create a new session by creating its initial terminal.

    ``env_vars`` are operator-forwarded env vars from ``cao launch --env``.
    They are persisted on the session record so every worker spawned later
    in the same session inherits them. See issue #248.

    ``profile_contract`` is the optional ``cao-profile-launch-contract-v1``
    expectation the conductor preflighted. It is validated against the
    profile source the runtime loads exactly once, before any tmux, session,
    or provider effect: a divergence raises :class:`ProfileLaunchConflict`
    with zero effects and a retry path, and a malformed contract raises
    ``ValueError``. An absent contract launches normally — the contract is
    an expectation, not a second authority — and a receipt is recorded
    either way.

    A **stopped** name is refused. Stopping records what the session would
    restore to and, in time, what to relaunch; a new campaign taking that
    name silently inherits all of it, and a later resume would relaunch the
    wrong workers against the wrong provider sessions. The refusal is the
    cheap half of that defence, and it matters most for reserved reusable
    names — a repair session on a fixed name is guaranteed to hit this.
    """
    if session_name:
        from cli_agent_orchestrator.services import session_lifecycle

        declared = session_lifecycle.describe(session_name)
        if declared["lifecycle"] == session_lifecycle.STOPPED:
            raise ValueError(
                f"session {session_name!r} is stopped and still holds what a resume would "
                f"restore ({declared['restore_to']!r}); delete the session to release the "
                "name, or pick another"
            )

    # A sealed contract arrives as the raw request value: strict-parse it
    # once (malformed shape is a typed 400). Validation against the
    # loaded profile happens later, under the claim, and only when no
    # durable winner adopts first — a response-loss retry must survive
    # later source drift without consulting mutable source at all.
    parsed_contract = None
    if profile_contract is not None:
        parsed_contract = parse_profile_contract(profile_contract)

    # Create-or-adopt request identity (cond-0817 repair): one canonical
    # document over the request alone — normalized session, supervisor
    # role, contract, normalized cwd, explicit tools, env digest, and
    # memory-manager input — fingerprinted to the SHA-256 the adoption
    # lookup compares exactly. Pure: no profile read, no store access.
    # Only for a contract-bearing retry on a fixed name; an absent
    # contract, or an auto-generated name no prior launch can match,
    # launches the ordinary path with no request identity.
    request_fingerprint: Optional[str] = None
    claim_session_name: Optional[str] = None
    if parsed_contract is not None and session_name is not None:
        claim_session_name = normalise_claim_session_name(session_name)
        request_fingerprint = supervisor_create_request_fingerprint(
            build_supervisor_create_request(
                session_name=claim_session_name,
                agent_profile=agent_profile,
                provider=provider,
                contract=parsed_contract,
                working_directory=working_directory,
                allowed_tools=allowed_tools,
                env_vars=env_vars,
                memory_manager=memory_manager,
            )
        )

    # One lifecycle-claim-owned sequence for a contract-bearing retry on
    # a fixed name: the durable adoption lookup runs first (before any
    # profile read), so a response-loss retry adopts the exact ready
    # winner even after the profile source drifted; only when no winner
    # exists does the launch read the profile, validate, prepare, and
    # create — all under the same claim, so a concurrent loser blocks on
    # the claim, then adopts without preparing or dispatching. Any other
    # launch keeps the ordinary pre-claim flow below.
    if (
        parsed_contract is not None
        and session_name is not None
        and request_fingerprint is not None
        and claim_session_name is not None
    ):
        from cli_agent_orchestrator.services import callback_recovery

        session_claim = callback_recovery.async_session_lifecycle_claim(
            type(get_backend()).__name__, claim_session_name
        )
        await session_claim.__aenter__()
        try:
            return await _create_or_adopt_supervisor_session(
                provider=provider,
                agent_profile=agent_profile,
                session_name=session_name,
                working_directory=working_directory,
                allowed_tools=allowed_tools,
                registry=registry,
                env_vars=env_vars,
                parsed_contract=parsed_contract,
                request_fingerprint=request_fingerprint,
                session_claim=session_claim,
            )
        finally:
            await session_claim.__aexit__(None, None, None)

    # The ONE profile read for an ordinary launch: the parsed profile,
    # its source metadata/digest, and the resolved provider/model/effort
    # every downstream stage consumes. Nothing below reloads the profile
    # by name, so the bytes validated here are the bytes the provider
    # launches with. Raises before any tmux/session/provider effect:
    # ProfileNotFoundError for a missing profile, ProfileInvalidError
    # for unparseable bytes, ProfileLaunchConflict for a diverged
    # contract.
    launch_context = load_supervisor_launch_context(
        agent_profile,
        explicit_provider=provider,
        fallback_provider="kiro_cli",
    )
    if parsed_contract is not None:
        validate_profile_contract(parsed_contract, launch_context)

    # Sealed-launch capability gate: the immutable launch material is built
    # from the already-frozen context (no second store read), evaluated
    # after any contract validation but before create_terminal owns any
    # tmux, session, DB, provider, or sidecar effect. The query itself
    # constructs nothing. A sealed contract on an adapter that would drop
    # nonempty frozen material (prompt/skills/policy it never forwards) or
    # launch provider-native named artifacts is refused outright —
    # validating or persisting CAO profile A while the supervisor consumes
    # something else is worse than refusing. Without a contract the same
    # adapter keeps its ordinary legacy launch path, but records no exact
    # receipt.
    sealed_material = build_sealed_launch_material(launch_context, allowed_tools=allowed_tools)
    sealed = provider_manager.sealed_launch_support(launch_context.provider, sealed_material)
    if not sealed.supported and profile_contract is not None:
        raise ProfileLaunchUnsupported(
            f"provider {launch_context.provider!r} cannot launch exactly from the "
            f"frozen profile {agent_profile!r}; no launch effect was produced",
            provider=launch_context.provider,
            source_path=launch_context.source_path,
            reason=sealed.reason,
            recovery=(
                "use a provider whose adapter consumes the frozen CAO profile "
                "exactly (see the adapter sealed-profile capability), or retry "
                "without profile_contract for an ordinary legacy launch with "
                "no exact receipt"
            ),
        )

    # Sealed-launch preparation (cond-0817 repair): the adapter validates
    # and serializes every field it actually consumes — the same composer
    # and serializers the launch runs — exactly once, after contract
    # validation/material construction and before ANY effect. A malformed
    # provider shape (e.g. a transport-less MCP entry) is refused here as
    # an operation-scoped 422 with zero mutation; previously it failed
    # late inside create_terminal, after the persisted session env had
    # already been cleared. An unexpected preparation failure propagates
    # (still pre-effect) and keeps its 500 classification.
    prepared_sealed_launch = None
    if sealed.supported:
        try:
            prepared_sealed_launch = provider_manager.prepare_sealed_launch(
                launch_context.provider, sealed_material
            )
        except SealedPreparationUnsupported as exc:
            if profile_contract is not None:
                recovery = (
                    "use a provider whose adapter consumes the frozen CAO profile "
                    "exactly (see the adapter sealed-profile capability), or retry "
                    "without profile_contract for an ordinary legacy launch with "
                    "no exact receipt"
                )
            else:
                recovery = (
                    "repair the frozen profile fields the adapter cannot consume "
                    "exactly (see the adapter reason), or switch to a provider "
                    "whose adapter consumes them"
                )
            raise ProfileLaunchUnsupported(
                f"provider {launch_context.provider!r} cannot launch exactly from the "
                f"frozen profile {agent_profile!r}; no launch effect was produced",
                provider=launch_context.provider,
                source_path=launch_context.source_path,
                reason=str(exc),
                recovery=recovery,
            ) from exc

    create_kwargs: dict = {
        "provider": launch_context.provider,
        "agent_profile": agent_profile,
        "session_name": session_name,
        "new_session": True,
        "working_directory": working_directory,
        "allowed_tools": allowed_tools,
        "registry": registry,
        "env_vars": env_vars,
        # Session creation owns the initial supervisor role and
        # passes it explicitly — role is launch truth, never a
        # profile-name heuristic.
        "stable_agent_role": ROLE_SUPERVISOR,
    }
    if sealed.supported:
        # The same loaded profile/context/material flows through terminal
        # creation, the terminal row (receipt), the pre-task bootstrap, and
        # provider construction — including the exact expected model/effort
        # the provider argv pins and the skill/policy inputs the gate
        # decided on, so the launch cannot disagree with the decision.
        # The prepared value is the one pre-effect validation artifact:
        # create_terminal binds it structurally at the recorded injection
        # sites and consumes it, never re-running composition.
        create_kwargs["profile_launch_context"] = launch_context
        create_kwargs["sealed_launch_material"] = sealed_material
        create_kwargs["prepared_sealed_launch"] = prepared_sealed_launch
        create_kwargs["expected_model"] = launch_context.model
        create_kwargs["expected_effort"] = launch_context.effort
        create_kwargs["profile_contract"] = parsed_contract
    terminal = await create_terminal(**create_kwargs)
    dispatch_plugin_event(
        registry,
        "post_create_session",
        PostCreateSessionEvent(
            session_id=terminal.session_name,
            session_name=terminal.session_name,
        ),
    )
    return CreateOrAdoptResult(terminal=terminal, adopted=False)


async def _create_or_adopt_supervisor_session(
    *,
    provider: str | None,
    agent_profile: str,
    session_name: str,
    working_directory: str | None,
    allowed_tools: list[str] | None,
    registry: PluginRegistry | None,
    env_vars: dict[str, str] | None,
    parsed_contract: Dict[str, Any],
    request_fingerprint: str,
    session_claim: Any,
) -> CreateOrAdoptResult:
    """Run the claim-owned create-or-adopt sequence for one supervisor retry.

    The caller holds the session lifecycle claim; every step below runs
    under it, in order: (1) inspect the durable row before reading the
    mutable profile source, adopting the exact ready winner with zero
    effects; (2) load the profile once and validate the contract (drift
    is a typed 409 with zero effects); (3) capability-gate and prepare
    the immutable artifact exactly once (unsupported/malformed material
    is a typed 422 with zero effects); (4) create through
    ``create_terminal`` under the same held claim (persisting the row
    as ``pending`` with the request fingerprint); (5) dispatch
    ``post_create_session`` exactly once, still under the claim;
    (6) CAS ``pending`` → ``ready`` and return the fresh terminal.
    A concurrent loser blocks on the claim at step 1, then adopts —
    it never prepares and never dispatches.
    """
    # Step 1: the durable lookup before any mutable-source read. An
    # exact ready winner adopts here even when the profile source has
    # since drifted — the stored receipt and fingerprint, not the live
    # profile, decide. A non-adoptable duplicate raises the typed 409
    # with zero effects; no rows at all returns None and the launch
    # proceeds to first creation below.
    candidate = find_supervisor_adoption_candidate(
        session_name=normalise_claim_session_name(session_name),
        contract=parsed_contract,
        request_fingerprint=request_fingerprint,
    )
    if candidate is not None:
        stored_receipt = candidate["profile_receipt"]
        assert isinstance(stored_receipt, dict)
        return CreateOrAdoptResult(
            terminal=_terminal_from_adoption_row(candidate, stored_receipt),
            adopted=True,
        )

    # Step 2: the ONE profile read, then contract validation. Both run
    # under the claim, before any effect.
    launch_context = load_supervisor_launch_context(
        agent_profile,
        explicit_provider=provider,
        fallback_provider="kiro_cli",
    )
    validate_profile_contract(parsed_contract, launch_context)

    # Step 3: capability gate, then exactly-once preparation.
    sealed_material = build_sealed_launch_material(launch_context, allowed_tools=allowed_tools)
    sealed = provider_manager.sealed_launch_support(launch_context.provider, sealed_material)
    if not sealed.supported:
        raise ProfileLaunchUnsupported(
            f"provider {launch_context.provider!r} cannot launch exactly from the "
            f"frozen profile {agent_profile!r}; no launch effect was produced",
            provider=launch_context.provider,
            source_path=launch_context.source_path,
            reason=sealed.reason,
            recovery=(
                "use a provider whose adapter consumes the frozen CAO profile "
                "exactly (see the adapter sealed-profile capability), or retry "
                "without profile_contract for an ordinary legacy launch with "
                "no exact receipt"
            ),
        )
    try:
        prepared_sealed_launch = provider_manager.prepare_sealed_launch(
            launch_context.provider, sealed_material
        )
    except SealedPreparationUnsupported as exc:
        raise ProfileLaunchUnsupported(
            f"provider {launch_context.provider!r} cannot launch exactly from the "
            f"frozen profile {agent_profile!r}; no launch effect was produced",
            provider=launch_context.provider,
            source_path=launch_context.source_path,
            reason=str(exc),
            recovery=(
                "use a provider whose adapter consumes the frozen CAO profile "
                "exactly (see the adapter sealed-profile capability), or retry "
                "without profile_contract for an ordinary legacy launch with "
                "no exact receipt"
            ),
        ) from exc

    # Step 4: create under the same held claim — the preflight
    # re-checks adoption (covering direct construction), then runs the
    # one effect sequence: clear, tmux, DB row (born ``pending`` with
    # the request fingerprint), roster, provider construction.
    terminal = await create_terminal(
        provider=launch_context.provider,
        agent_profile=agent_profile,
        session_name=session_name,
        new_session=True,
        working_directory=working_directory,
        allowed_tools=allowed_tools,
        registry=registry,
        env_vars=env_vars,
        stable_agent_role=ROLE_SUPERVISOR,
        profile_launch_context=launch_context,
        sealed_launch_material=sealed_material,
        prepared_sealed_launch=prepared_sealed_launch,
        expected_model=launch_context.model,
        expected_effort=launch_context.effort,
        profile_contract=parsed_contract,
        create_request_fingerprint=request_fingerprint,
        held_session_claim=session_claim,
    )

    # Steps 5-6: the post-create hook exactly once while the claim is
    # still held, then the pending→ready transition. A hook or
    # transition failure settles the row failed (best-effort) and
    # raises, so a half-created launch is never adoptable.
    try:
        dispatch_plugin_event(
            registry,
            "post_create_session",
            PostCreateSessionEvent(
                session_id=terminal.session_name,
                session_name=terminal.session_name,
            ),
        )
        settled = cas_terminal_provider_readiness(
            terminal.id, PROVIDER_READINESS_PENDING, PROVIDER_READINESS_READY
        )
        if not settled:
            raise RuntimeError(f"supervisor launch {terminal.id} did not settle provider-ready")
    except Exception:
        try:
            cas_terminal_provider_readiness(
                terminal.id, PROVIDER_READINESS_PENDING, PROVIDER_READINESS_FAILED
            )
        except Exception:
            pass  # Marking is best-effort; the primary failure propagates.
        raise
    return CreateOrAdoptResult(terminal=terminal, adopted=False)


def normalise_claim_session_name(session_name: str) -> str:
    """Canonical session name for the claim-owned adoption lookup.

    Same prefix-then-normalise derivation creation admission applies
    under the claim, so the fingerprint, the claim key, and the lookup
    always name one session. Callers pass a fixed (non-None) name —
    the claim-owned sequence never runs for auto-generated names.
    """
    from cli_agent_orchestrator.services.session_lifecycle import (
        normalise_session_name,
    )

    prefixed = (
        session_name
        if session_name.startswith(SESSION_PREFIX)
        else f"{SESSION_PREFIX}{session_name}"
    )
    return normalise_session_name(prefixed)


def list_sessions() -> List[Dict]:
    """List all sessions from tmux."""
    try:
        tmux_sessions = get_backend().list_sessions()
        return [s for s in tmux_sessions if s["id"].startswith(SESSION_PREFIX)]
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return []


def get_session(session_name: str) -> Dict:
    """Get session with terminals."""
    try:
        if not get_backend().session_exists(session_name):
            raise ValueError(f"Session '{session_name}' not found")

        tmux_sessions = get_backend().list_sessions()
        session_data = next((s for s in tmux_sessions if s["id"] == session_name), None)

        if not session_data:
            raise ValueError(f"Session '{session_name}' not found")

        # Read through the projection, which is the one authority on what a
        # terminal is: it observes liveness rather than trusting the stored
        # row, reports a lifecycle instead of a provider status for a pane
        # that no longer resolves, and covers both protocol vintages.
        #
        # This route is what the dashboard and ``conduct status`` read. While
        # it returned raw rows, a terminal whose window had been deleted
        # rendered as provider ``Unknown`` forever — indistinguishable from a
        # healthy worker awaiting detection — and a managed v2 worker
        # appeared in neither view, because its row lives in a separate
        # table. Meanwhile ``cao session status`` *was* projected, so the two
        # human views disagreed by construction.
        #
        # The projection derives the provider status itself, for a live pane
        # only, so nothing is enriched here.
        #
        # Deliberately not applied to ``delete_session``, the watchdog or
        # cleanup: those are the machine paths the v2 store's write/consume
        # isolation is about, and they must keep seeing v1 rows only. The
        # boundary this crosses is *human visibility*, which was never the
        # thing being isolated.
        from cli_agent_orchestrator.services import terminal_projection

        terminals = terminal_projection.project_session(session_name)
        return {"session": session_data, "terminals": terminals}

    except Exception as e:
        logger.error(f"Failed to get session {session_name}: {e}")
        raise


def delete_session(session_name: str, registry: PluginRegistry | None = None) -> Dict:
    """Delete session and cleanup.

    Returns:
        Dict with 'deleted' (list of deleted session names) and 'errors' (list of error dicts).
    """
    result: Dict = {"deleted": [], "errors": []}
    session_claim = None
    try:
        from cli_agent_orchestrator.services import callback_recovery
        from cli_agent_orchestrator.services import session_lifecycle as sl
        from cli_agent_orchestrator.services import terminal_service

        # Canonicalize before every physical or durable effect. Otherwise a
        # bare name can tear down an absent session while forget() alone
        # releases the lifecycle row belonging to the real prefixed session.
        session_name = sl.normalise_session_name(session_name)

        backend = get_backend()
        session_claim = callback_recovery.session_lifecycle_claim(
            type(backend).__name__, session_name
        )
        session_claim.__enter__()
        # This final existence observation, snapshot, physical teardown, and
        # durable environment cleanup are one session/workspace lifecycle.
        session_alive = backend.session_exists(session_name)
        terminals = list_terminals_by_session(session_name)

        # Clean up each terminal (snapshot, kill window, FIFO reader,
        # status buffer, provider, DB) via the event-driven teardown path.
        terminal_errors = []
        claim_keys = callback_recovery.terminal_lifecycle_claim_set(*terminals)
        with callback_recovery.generation_lifecycle_claims(claim_keys):
            for terminal in terminals:
                if callback_recovery.terminal_has_open_recovery(
                    terminal["id"], terminal.get("generation")
                ):
                    terminal_errors.append(
                        {
                            "terminal_id": terminal["id"],
                            "detail": "open callback recovery",
                        }
                    )

            if not terminal_errors:
                for terminal in terminals:
                    try:
                        generation = terminal.get("generation")
                        kwargs = {}
                        if generation:
                            kwargs = {
                                "expected_generation": generation,
                                "expected_session": terminal.get("tmux_session") or session_name,
                            }
                        terminal_service.delete_terminal(
                            terminal["id"], registry=registry, **kwargs
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cleanup terminal {terminal['id']}: {e}")
                        terminal_errors.append({"terminal_id": terminal["id"], "detail": str(e)})

            if terminal_errors:
                raise RuntimeError(
                    "session deletion held because terminal cleanup failed: " f"{terminal_errors}"
                )

        # A deleted session must not leave its declaration behind. A later
        # session taking the name would inherit a stranger's `complete` or
        # `paused` — and both of those are marshal suppressors, so a
        # brand-new live campaign would start out invisible to the thing
        # whose job is to notice it wedging.
        try:
            sl.forget(session_name)
        except Exception as exc:  # noqa: BLE001 - deletion must still complete
            logger.warning("Could not forget the declared state of %s: %s", session_name, exc)

        # Re-check under the session claim: a concurrent create cannot add a
        # replacement window between this observation and kill_session.
        session_alive = backend.session_exists(session_name)
        # Kill backend session only if it still exists
        if session_alive:
            backend.kill_session(session_name)

        # Drop the per-session forwarded-env mapping (issue #248). Safe
        # even when no vars were forwarded — the helper is a no-op then.
        # Strict (cond-0050): a delete that cannot complete durably raises
        # rather than leaving a stale row behind to be silently inherited.
        clear_session_env(session_name)

        result["deleted"].append(session_name)
        logger.info(f"Deleted session: {session_name}")
        dispatch_plugin_event(
            registry,
            "post_kill_session",
            PostKillSessionEvent(session_id=session_name, session_name=session_name),
        )
        return result

    except Exception as e:
        logger.error(f"Failed to delete session {session_name}: {e}")
        raise
    finally:
        if session_claim is not None:
            session_claim.__exit__(None, None, None)


def stop_session(
    session_name: str,
    *,
    declared_by: str,
    restore_to: Optional[str] = None,
    note: Optional[str] = None,
    expected_epoch: Optional[int] = None,
    registry: PluginRegistry | None = None,
    archived: bool = False,
) -> Dict:
    """Collect a session's panes while preserving its declared stopped state.

    The row-preserving counterpart to ``delete_session``. A lifecycle stop
    must both tear the fleet down and keep the truth a resume needs: the
    lifecycle row stays in ``stopped`` with its ``restore_to`` target, the
    forwarded environment survives so a resume relaunches against it, and the
    per-terminal snapshots/recovery artifacts survive because collection goes
    through the same event-driven ``delete_terminal`` path deletion does.

    The order is load-bearing:

    * the session-lifecycle claim is held across the whole operation, so a
      concurrent create cannot add a window between the stopped check and the
      teardown;
    * admission (an open callback recovery) refuses *before* anything is
      written or collected — collecting a terminal mid-recovery would lose
      the one-shot refusal the recovery is adjudicating, and a stop recorded
      while one is open would be a false state;
    * the stopped declaration is written *before* any pane is collected, so a
      write or admission failure deletes nothing, and a fully collected fleet
      can never be left declared working;
    * on a mid-collection failure the row is already stopped, so the
      divergence is visible (not silent) and a retry re-collects what remains
      — re-stopping is idempotent and keeps ``restore_to``.

    Deliberately not symmetric with deletion: this never calls ``forget`` and
    never clears the forwarded env. Releasing the name is the destructive
    ``DELETE`` path's job, kept separate so an ordinary stop can never become
    an accidental cleanup.
    """
    from cli_agent_orchestrator.services import callback_recovery
    from cli_agent_orchestrator.services import session_lifecycle as sl
    from cli_agent_orchestrator.services import terminal_service

    # cond-0221: canonicalize once, before any physical or admission effect, so
    # a bare name (`repair`) reaches the physical claim, the terminal listing,
    # the exact-generation deletion's expected_session, backend existence/kill,
    # and the durable row as `cao-repair`. It also makes the lifecycle write
    # claim taken here and the one taken inside ``sl.stop()``/``set_archived()``
    # (which normalize internally) the same key — re-entrant — rather than a
    # raw-vs-canonical pair that inverts the canonical lock order.
    session_name = sl.normalise_session_name(session_name)

    backend = get_backend()
    collected: List[str] = []
    record: Dict = {}
    with callback_recovery.session_lifecycle_claim(type(backend).__name__, session_name):
        # cond-0221: hold the lifecycle write claim across admission, the stop
        # write, and collection. Every lifecycle mutation takes it, so a
        # competing ``declare``/``set_kind``/… cannot commit during the stop —
        # it waits, then sees ``stopped`` and is refused by the transition.
        # Acquired before the generation claims: the write claim sorts after the
        # physical session-workspace claim and before terminal-generation claims
        # (physical < write < generation), and ``sl.stop()``'s own ``_write``
        # takes it re-entrantly via the thread-local, so it does not self-deadlock.
        with callback_recovery.session_lifecycle_write_claim(session_name):
            terminals = list_terminals_by_session(session_name)
            claim_keys = callback_recovery.terminal_lifecycle_claim_set(*terminals)
            with callback_recovery.generation_lifecycle_claims(claim_keys):
                # Admission: an open callback recovery blocks both collection and
                # the declaration. Checked before the lifecycle write so a refusal
                # leaves no false stopped state, and before any deletion so it
                # collects nothing.
                for terminal in terminals:
                    if callback_recovery.terminal_has_open_recovery(
                        terminal["id"], terminal.get("generation")
                    ):
                        raise SessionStopRefused(
                            f"session stop held because terminal {terminal['id']} has an "
                            "open callback-recovery operation; collection is refused until "
                            "callback completion or a terminal refusal/manual disposition"
                        )

                # Record the stop (preserving the row and restore_to) before any
                # collection. A write failure here raises and collects nothing.
                if archived:
                    record = sl.set_archived(
                        session_name,
                        True,
                        declared_by=declared_by,
                        note=note,
                        expected_epoch=expected_epoch,
                    )
                else:
                    record = sl.stop(
                        session_name,
                        declared_by=declared_by,
                        restore_to=restore_to,
                        note=note,
                        expected_epoch=expected_epoch,
                    )

                # Collect the panes via the same event-driven teardown deletion
                # uses: delete_terminal snapshots before killing, so recovery
                # artifacts survive. The forwarded env is intentionally left.
                errors = []
                for terminal in terminals:
                    try:
                        generation = terminal.get("generation")
                        kwargs = {}
                        if generation:
                            kwargs = {
                                "expected_generation": generation,
                                "expected_session": terminal.get("tmux_session") or session_name,
                            }
                        terminal_service.delete_terminal(
                            terminal["id"], registry=registry, **kwargs
                        )
                        collected.append(terminal["id"])
                    except Exception as e:  # noqa: BLE001 - one pane must not erase the rest
                        logger.warning(
                            "Failed to collect terminal %s during stop of %s: %s",
                            terminal["id"],
                            session_name,
                            e,
                        )
                        errors.append({"terminal_id": terminal["id"], "detail": str(e)})
                if errors:
                    # The row is already stopped and the env is preserved, so the
                    # divergence is visible and a retry converges. Surface the
                    # structured detail for server-side logging; the HTTP layer
                    # exposes only bounded, redacted guidance.
                    raise SessionStopPartial(
                        "session stop partially collected; lifecycle preserved; " "retry converges",
                        collected_terminal_ids=collected,
                        errors=errors,
                    )

            # Kill the enclosing backend session if anything remains. Per-window
            # teardown already removed each pane; this clears empty shells.
            if backend.session_exists(session_name):
                backend.kill_session(session_name)

    return {**record, "collected_terminal_ids": collected, "errors": []}
