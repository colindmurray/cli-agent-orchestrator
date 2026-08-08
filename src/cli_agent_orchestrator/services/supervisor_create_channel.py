"""The supervisor-creation channel: one verb, over AF_UNIX, decided by lineage.

Supervisor terminal creation is the one place this server must tell *the
operator asked for this* from *a managed worker asked for this*. The HTTP API
cannot answer that question. It is served over TCP (``constants.SERVER_HOST`` /
``SERVER_PORT``), and ``SO_PEERCRED`` / ``LOCAL_PEERCRED`` are ``AF_UNIX``
socket options that do not exist for a TCP connection on either platform — so
on ``POST /sessions/{name}/terminals`` there is no peer pid, hence no ancestry,
hence no server-derived discriminator at all. The only signals left there are
``agent_profile``, ``caller_id``, ``role`` frontmatter, ``CAO_TERMINAL_ID``,
same-UID trust, and a caller's own assertion. Every one of those is caller data
or a guess, and a supervisor discriminator built on caller data is not a
discriminator: any local process that reaches loopback could name itself the
supervisor, and a managed worker does reach loopback.

So supervisor creation — and *only* supervisor creation — moves here, to a
server-scoped mode-``0600`` ``AF_UNIX`` socket carrying exactly one verb. The
TCP creation endpoints keep their present signatures and their present
``agent_profile`` parameter, and they remain unable to establish authority under
any parameter value, because the authority decision is not reachable from them.

**The predicate is negative, and that inverts a fail-closed default.**
``actor_broker`` asks a *positive* question — "does this peer's ancestry reach
the provider tree?" — so its walker can answer ``False`` on an unreadable chain
and be safe: an unprovable peer is refused. Here the question is the mirror
image — "does this peer's ancestry reach *no* managed process?" — and a bare
``False`` would mean *operator origin*, i.e. **admit**. An unreadable chain
would become an admission. This module therefore classifies origin in three
states, never two (:class:`PeerOrigin`), and an unreadable chain is
``UNPROVEN`` → ``authority-lineage-unproven``. Collapsing ``UNPROVEN`` into
``OPERATOR`` would hand the channel to exactly the process it exists to refuse.

**Which projects reach the authority pipeline.** Building this channel, the
predicate, and the A0/A/B/C pipeline is gate 2's *own* subject — a gate whose
subject may not be built is a gate that can never close — while everything a
*later* gate proves stays out. But gate 2 closes only on evidence that the
pipeline produced, so the pipeline has to run somewhere before the gate is
closed. It runs for exactly one project: the disposable one named by the
operator-written designation of
:mod:`cli_agent_orchestrator.services.gate2_proof_designation`. Every other
project keeps the ordinary bypass, in which no decision is taken and no epoch is
allocated — phase B creates the terminal and the authority side effect refuses
``authority-bootstrap-unavailable`` with detail ``g10-unproven``.

That bypass refusal is **not** a bring-up failure. It is the one outcome where
the verb succeeds and returns a terminal while carrying no authority, and
treating it as a failure would break every bring-up on every deployment until
gate 2 closes. Every *other* refusal either precedes creation or tears the
created terminal down.
"""

from __future__ import annotations

import enum
import fcntl
import json
import logging
import os
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

from cli_agent_orchestrator.services.actor_broker import PeerCredentials, peer_credentials

logger = logging.getLogger(__name__)

#: The channel's filename under the state root. The state root itself is the
#: fork's single existing decision (``constants.CAO_HOME_DIR``, resolved from
#: ``CAO_STATE_ROOT``); this module never re-derives, probes, or falls back to
#: another location, because a client that guesses a second convention is how
#: two lanes end up with two sockets.
SOCKET_BASENAME = "supervisor-create.sock"

#: The sibling lock proving no live server owns the socket path.
LOCK_BASENAME = "supervisor-create.sock.lock"

#: The one verb. There is no ``establish``, on this channel or any other: an
#: ``establish`` verb would bind authority to something a caller names, and on a
#: fresh project there is no recorded supervisor tree to check such a call
#: against, so it could only be dead or a "no row -> allow" special case.
VERB_SUPERVISOR_TERMINAL_CREATE = "supervisor-terminal-create"

#: Set A — launch parameters, forwarded unchanged to the same creation code the
#: TCP endpoints call, and read by no part of the authority decision. This is
#: exactly the set the pinned callers already pass; it is deliberately not
#: narrower, because a verb that cannot reproduce ``conduct up``'s
#: ``working_directory`` and shim ``env_vars`` would silently launch every
#: supervisor with a defaulted environment.
SET_A_FIELDS: FrozenSet[str] = frozenset(
    {
        "session_name",
        "agent_profile",
        "working_directory",
        "env_vars",
        "caller_id",
        "initial_message",
        "orchestration_type",
        "allowed_tools",
        "defer_init",
    }
)

#: Set B — authority inputs. Empty, and enforced by refusal rather than by
#: ignoring: silently dropping a field a caller believed was load-bearing is how
#: a caller comes to think it can name the authority. Anything naming an
#: existing terminal as the subject of authority belongs here too.
SET_B_REFUSED_FIELDS: FrozenSet[str] = frozenset(
    {
        "terminal_id",
        "target_terminal_id",
        "supervisor_terminal_id",
        "role",
        "authority_epoch",
        "project_incarnation",
        "supervisor_generation",
        "terminal_generation",
    }
)

# Reason codes. Each is a member of the design's closed v1 vocabulary; this
# module adds none. Detail fields are permitted by that vocabulary and are how
# two paths sharing one code stay distinguishable on the wire.
REASON_DISCRIMINATOR_ABSENT = "supervisor-creation-discriminator-absent"
REASON_LINEAGE_UNPROVEN = "authority-lineage-unproven"
REASON_BOOTSTRAP_UNAVAILABLE = "authority-bootstrap-unavailable"
REASON_SET_B_PRESENT = "operation-admission-unproven"

#: ``authority-bootstrap-unavailable`` is emitted on two paths whose terminals
#: have *opposite* fates — retained here, torn down at the phase-C precondition.
#: The code is the wire contract; this detail is what tells an operator whether
#: a terminal survived.
DETAIL_G10_UNPROVEN = "g10-unproven"
DETAIL_BOOTSTRAP_PRECONDITION = "bootstrap-precondition"

#: `authority-lineage-unproven` carries the cause, so an operator can tell a
#: broken host from a refused peer. These are detail strings, not reason codes:
#: the closed vocabulary permits added detail fields and forbids added codes.
DETAIL_ANCESTRY_UNREADABLE = "ancestry-unreadable"
DETAIL_ANCESTRY_CYCLIC = "ancestry-cyclic"
DETAIL_ANCESTRY_BUDGET_EXHAUSTED = "ancestry-budget-exhausted"
DETAIL_PEER_NOT_LIVE = "peer-not-live"
DETAIL_MANAGED_SET_UNENUMERABLE = "managed-set-unenumerable"
DETAIL_KERNEL_CREDENTIALS_UNAVAILABLE = "kernel-credentials-unavailable"

#: Every `UNPROVEN` cause, for the completeness assertion in the suite.
LINEAGE_UNPROVEN_DETAILS = frozenset(
    {
        DETAIL_ANCESTRY_UNREADABLE,
        DETAIL_ANCESTRY_CYCLIC,
        DETAIL_ANCESTRY_BUDGET_EXHAUSTED,
        DETAIL_PEER_NOT_LIVE,
        DETAIL_MANAGED_SET_UNENUMERABLE,
        DETAIL_KERNEL_CREDENTIALS_UNAVAILABLE,
    }
)

#: The fork's existing safe bound for an ``AF_UNIX`` pathname. A longer path
#: fails closed at startup rather than binding a silently truncated name.
AF_UNIX_SAFE_PATH_BYTES = 100

#: Hop budget for an ancestry walk. Exhausting it without reaching init is not
#: a proof of anything, so it answers ``UNPROVEN``.
_MAX_ANCESTRY_HOPS = 64


#: The gate-2 proof designation as it stood **at server start**, and the only
#: value any decision consults. Read once so it cannot be flipped mid-run to
#: change a decision already in flight, and ``None`` — no designation, the
#: pipeline runs nowhere — is both the default and the safe state.
_DESIGNATION_AT_START = None


def load_designation_at_start() -> None:
    """Read the designation once, at server start. Fails closed on a bad one.

    Absence is ordinary and leaves the pipeline running nowhere. A malformed,
    ambiguous, or multi-project designation raises, because an operator who
    wrote one believes a proof run is scoped, and continuing as if it were
    absent would silently falsify that belief.
    """
    global _DESIGNATION_AT_START

    from cli_agent_orchestrator.services import gate2_proof_designation

    _DESIGNATION_AT_START = gate2_proof_designation.load_designation()


def _set_designation_for_test(designation) -> None:
    """Test seam for the start-only read. Not reachable from any request path."""
    global _DESIGNATION_AT_START

    _DESIGNATION_AT_START = designation


class SupervisorCreateChannelError(RuntimeError):
    """The channel cannot be served as specified. Startup fails closed."""


class PeerOrigin(enum.Enum):
    """Three states, because a negative predicate cannot live with two.

    ``OPERATOR`` and ``MANAGED`` are both *proofs*. ``UNPROVEN`` is the absence
    of one, and it must never be read as ``OPERATOR``.
    """

    OPERATOR = "operator"
    MANAGED = "managed"
    UNPROVEN = "unproven"


@dataclass(frozen=True)
class ManagedPidSet:
    """The server's own record of which pids are managed.

    ``enumerable`` is separate from emptiness on purpose: a genuinely fresh
    server legitimately records no managed pane, while a failed query also
    yields no rows. The first admits an operator; the second can prove nothing
    and must refuse.
    """

    pids: FrozenSet[int]
    enumerable: bool


@dataclass(frozen=True)
class ChannelOutcome:
    """One verb invocation's result, surfaced without collapsing the cases.

    ``ok`` answers "did the command succeed", which is **not** the same question
    as "did authority attach". The ``g10-unproven`` row is ``ok=True`` with
    ``authority_granted=False``; every other refusal is ``ok=False``.
    """

    ok: bool
    reason_code: Optional[str] = None
    detail: Optional[str] = None
    authority_granted: bool = False
    terminal: Optional[Dict[str, Any]] = None
    terminal_created: bool = False

    def to_wire(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "authority_granted": self.authority_granted,
            "terminal_created": self.terminal_created,
            "terminal": self.terminal,
        }


def g10_proofs_recorded() -> bool:
    """Whether G10's two deployment proofs are recorded for this deployment.

    Hard ``False``. G10 closes only on proofs taken against a target deployment
    — that the supervisor-rooted broker distinguishes the supervisor tree from
    every same-UID sibling, and that supervisor creation is reachable only over
    this channel — and no source change can constitute either. Flipping this is
    a release-gate action performed with those receipts in hand, not an
    implementation detail.

    While it answers ``False`` the authority decision is never taken: no
    authority row, no epoch allocation, no grant, no phase C. Creation itself is
    unaffected, which is what keeps bring-up working while the capability is
    dark.
    """
    return False


def socket_path() -> Path:
    """``<state-root>/supervisor-create.sock``, by the fork's one state-root rule.

    Resolved at call time rather than at import so an isolated state root set by
    a test or a second install is honoured, exactly as the pane-input arbiter
    resolves its lock directory.
    """
    from cli_agent_orchestrator.constants import CAO_HOME_DIR

    return Path(CAO_HOME_DIR) / SOCKET_BASENAME


def lock_path() -> Path:
    from cli_agent_orchestrator.constants import CAO_HOME_DIR

    return Path(CAO_HOME_DIR) / LOCK_BASENAME


def validate_socket_path(path: Path) -> None:
    """Refuse a path the kernel would truncate, before anything binds it."""
    encoded = os.fsencode(str(path))
    if len(encoded) > AF_UNIX_SAFE_PATH_BYTES:
        raise SupervisorCreateChannelError(
            f"supervisor-create channel path is {len(encoded)} bytes, over the "
            f"{AF_UNIX_SAFE_PATH_BYTES}-byte AF_UNIX bound: {path}. Binding a "
            "truncated name would put the channel somewhere no client resolves."
        )


def managed_pid_set() -> ManagedPidSet:
    """Every pid this server records as managed, plus its own tree roots.

    The set is the union of four server-derived sources and contains no caller
    input: every live ``terminals.pane_pid``, every
    ``managed_launch_v2_terminals.v2_pane_pid``, the tmux server backing those
    panes, and this API server process. The tmux server pid is deliberately in
    the set: both ``pane_pid`` columns are nullable, so it is the robust
    catch-all for a pane whose own pid was never recorded. The consequence is
    that *every* pane on that tmux server — including an operator's ordinary
    tmux windows, since the fork attaches to the default server — is
    managed-origin, and the operator must therefore invoke bring-up from
    outside it. That is a legible zero-effect refusal rather than a silent
    admission, which is the right direction for this trade.
    """
    pids: set[int] = set()
    try:
        from cli_agent_orchestrator.clients.database import (
            ManagedLaunchV2TerminalModel,
            SessionLocal,
            TerminalModel,
        )

        with SessionLocal() as db:
            for (pid,) in db.query(TerminalModel.pane_pid).all():
                if pid is None:
                    # A recorded terminal whose own pane pid is missing leaves
                    # the set incomplete. This refuses even though the tmux
                    # server pid below would in practice still catch anything
                    # inside a pane — deliberate extra conservatism, not a hole
                    # being plugged. It costs availability until the row is
                    # repaired and buys independence from the catch-all's
                    # continued correctness.
                    return ManagedPidSet(pids=frozenset(), enumerable=False)
                pids.add(int(pid))
            for (pid,) in db.query(ManagedLaunchV2TerminalModel.v2_pane_pid).all():
                if pid is None:
                    return ManagedPidSet(pids=frozenset(), enumerable=False)
                pids.add(int(pid))
    except Exception:
        # A store that cannot be read proves nothing about origin.
        logger.exception("supervisor-create: managed pid enumeration failed")
        return ManagedPidSet(pids=frozenset(), enumerable=False)

    pids.add(os.getpid())

    tmux_server_pid = _tmux_server_pid()
    if tmux_server_pid is not None:
        pids.add(tmux_server_pid)

    return ManagedPidSet(pids=frozenset(pids), enumerable=True)


def _tmux_server_pid() -> Optional[int]:
    """The pid of the tmux server backing this fork's sessions, or ``None``.

    ``None`` is not treated as an enumeration failure: the two ``pane_pid``
    sources still bound the managed set, and a server with no tmux running has
    no panes to descend from.
    """
    try:
        out = subprocess.run(
            ["tmux", "display-message", "-p", "#{pid}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = out.stdout.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def classify_peer_origin(peer_pid: int, managed: ManagedPidSet) -> Tuple[PeerOrigin, Optional[str]]:
    """Classify the peer's origin, returning ``(origin, cause_detail)``.

    Re-walked live rather than cached: kernel peer credentials are connect-time
    fixed on both platforms, so re-reading them returns the original pid even
    after the peer exits. The walk is the detector, not the pid.

    ``MANAGED`` is decidable on a **partial** walk — a positive hit
    short-circuits and never needs the chain completed. ``OPERATOR`` requires
    the opposite: an **exhaustive** walk to init meeting no managed pid, with
    the managed set proven enumerable. Everything else is ``UNPROVEN``, and the
    cause is returned so an operator can tell a broken host from a refused peer.
    """
    if not managed.enumerable:
        return PeerOrigin.UNPROVEN, DETAIL_MANAGED_SET_UNENUMERABLE
    if peer_pid <= 0:
        return PeerOrigin.UNPROVEN, DETAIL_PEER_NOT_LIVE

    seen: set[int] = set()
    current = peer_pid
    first_hop = True
    for _ in range(_MAX_ANCESTRY_HOPS):
        if current in managed.pids:
            return PeerOrigin.MANAGED, None
        if current <= 1:
            # Walked exhaustively to init without meeting a managed pid. This is
            # the only admitting answer.
            return PeerOrigin.OPERATOR, None
        if current in seen:
            return PeerOrigin.UNPROVEN, DETAIL_ANCESTRY_CYCLIC
        seen.add(current)
        parent = _parent_pid(current)
        if parent is None:
            # A chain that stops answering cannot prove the absence of managed
            # ancestry. On the very first hop this is overwhelmingly a peer that
            # has already exited; deeper in, a host that stopped answering.
            return (
                PeerOrigin.UNPROVEN,
                DETAIL_PEER_NOT_LIVE if first_hop else DETAIL_ANCESTRY_UNREADABLE,
            )
        current = parent
        first_hop = False
    # The budget ran out before reaching init, so the walk was never exhaustive
    # and proves nothing.
    return PeerOrigin.UNPROVEN, DETAIL_ANCESTRY_BUDGET_EXHAUSTED


def _parent_pid(pid: int) -> Optional[int]:
    """One hop up, via ``ps`` — portable across Linux and macOS.

    ``None`` means "could not read", never "no parent". A dead pid yields empty
    output and lands here, which is what makes an exited peer unprovable rather
    than silently operator-origin.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = out.stdout.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def validate_request(payload: Any) -> Dict[str, Any]:
    """Accept exactly Set A; refuse Set B and anything unknown.

    Unknown keys are refused rather than dropped for the same reason Set B is:
    a caller that can add a field it believes matters, and be answered ``200``,
    has been taught that the field was read.
    """
    if not isinstance(payload, dict):
        raise SupervisorCreateChannelError("request must be a JSON object")

    verb = payload.get("verb")
    if verb != VERB_SUPERVISOR_TERMINAL_CREATE:
        raise SupervisorCreateChannelError(
            f"unknown verb {verb!r}; this channel carries only "
            f"{VERB_SUPERVISOR_TERMINAL_CREATE!r}"
        )

    args = payload.get("args", {})
    if not isinstance(args, dict):
        raise SupervisorCreateChannelError("args must be a JSON object")

    present_set_b = sorted(set(args) & SET_B_REFUSED_FIELDS)
    if present_set_b:
        raise SupervisorCreateChannelError(
            "authority inputs are not accepted on this channel: "
            f"{', '.join(present_set_b)}. Authority is derived from the terminal "
            "this server creates, never from a field a caller supplies."
        )

    unknown = sorted(set(args) - SET_A_FIELDS)
    if unknown:
        raise SupervisorCreateChannelError(f"unknown launch parameters: {', '.join(unknown)}")

    if "agent_profile" not in args or not args["agent_profile"]:
        raise SupervisorCreateChannelError("agent_profile is required to launch a terminal")

    return args


def evaluate_phase_a0(
    credentials: Optional[PeerCredentials],
    managed: ManagedPidSet,
) -> Optional[ChannelOutcome]:
    """Phase A0: verify the peer before anything is created or allocated.

    Returns the refusal, or ``None`` when the peer is proven operator-origin and
    the call may proceed. Nothing is created, allocated, or written on either
    refusal — A0 runs with no transaction held and before phase A exists.
    """
    if credentials is None:
        return ChannelOutcome(
            ok=False,
            reason_code=REASON_LINEAGE_UNPROVEN,
            detail=DETAIL_KERNEL_CREDENTIALS_UNAVAILABLE,
        )

    origin, cause = classify_peer_origin(credentials.pid, managed)
    if origin is PeerOrigin.MANAGED:
        return ChannelOutcome(ok=False, reason_code=REASON_DISCRIMINATOR_ABSENT)
    if origin is PeerOrigin.UNPROVEN:
        return ChannelOutcome(ok=False, reason_code=REASON_LINEAGE_UNPROVEN, detail=cause)
    return None


def bind_channel_socket() -> Tuple[socket.socket, int]:
    """Bind the channel, clearing only a socket node that is provably stale.

    A stale ``AF_UNIX`` node does not clear itself, and unlinking one that is
    still in use would steal a live server's channel. The sibling lock decides
    between those: holding it exclusively and non-blocking proves no live server
    owns this path, so whatever node is there is residue. Failing to take it
    means someone live owns the channel, and startup fails closed rather than
    unlinking underneath them.
    """
    path = socket_path()
    validate_socket_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor = os.open(lock_path(), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise SupervisorCreateChannelError(
            "another live server already owns the supervisor-create channel"
        ) from None

    try:
        _unlink_stale_socket(path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(path))
            os.chmod(path, 0o600)
            server.listen(8)
        except Exception:
            server.close()
            raise
    except Exception:
        os.close(descriptor)
        raise

    return server, descriptor


def _unlink_stale_socket(path: Path) -> None:
    """Remove a residual node, refusing to touch anything that is not a socket."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(info.st_mode):
        raise SupervisorCreateChannelError(
            f"{path} exists and is not a socket; refusing to unlink it"
        )
    path.unlink()


def read_peer_credentials(conn: socket.socket) -> Optional[PeerCredentials]:
    """Kernel peer identity, or ``None`` when the platform cannot answer."""
    try:
        return peer_credentials(conn)
    except Exception:
        # ``ActorUnavailable`` on an unsupported platform, ``OSError`` from the
        # getsockopt itself, or anything else: none of them is an identity, so
        # all of them answer None and the caller refuses.
        logger.warning("supervisor-create: peer credentials unavailable", exc_info=True)
        return None


async def handle_supervisor_terminal_create(
    args: Dict[str, Any],
    *,
    credentials: Optional[PeerCredentials],
    managed: ManagedPidSet,
) -> ChannelOutcome:
    """Run A0, then create, then answer the authority question.

    The order is what makes the refusals honest: A0 precedes creation, so a
    wrong-origin or unprovable peer never causes a terminal to exist. With G10
    unproven there is no authority decision to take after creation, so the
    terminal is created, retained, and reported as carrying no authority.
    """
    from cli_agent_orchestrator.services import supervisor_authority as authority

    refusal = evaluate_phase_a0(credentials, managed)
    if refusal is not None:
        return refusal

    project = _project_for(args)
    pipeline = _pipeline_admitted(project)

    if not pipeline:
        # Ordinary operation on a deployment whose gate-2 receipt records no
        # proofs: no decision, no allocation, no phase-C bind. The terminal is
        # still created with full launch fidelity and retained — this is the one
        # outcome where the command succeeds while carrying no authority, and
        # treating it as a failure would break bring-up everywhere.
        terminal = await _create_terminal_from_set_a(args)
        return ChannelOutcome(
            ok=True,
            reason_code=REASON_BOOTSTRAP_UNAVAILABLE,
            detail=DETAIL_G10_UNPROVEN,
            authority_granted=False,
            terminal=terminal,
            terminal_created=True,
        )

    # ---- phase A: decide and commit the allocation, before anything is created
    decision = authority.decide(
        authority.SupervisorTuple(
            project=project, supervisor_terminal_id="", supervisor_generation=""
        ),
        witness=_existing_run_witness(project),
    )
    if decision.decision is authority.Decision.REFUSE:
        # A decision-table refusal is reached in phase A, before phase B, so it
        # allocates nothing and creates nothing. Leaving a stray
        # non-authoritative supervisor terminal behind here would be wrong.
        return ChannelOutcome(ok=False, reason_code=decision.reason_code, detail=decision.detail)

    if decision.decision is not authority.Decision.NOOP:
        # A non-refusing, non-noop decision always carries both values; the
        # asserts state that rather than leaving it to the reader.
        assert decision.project_incarnation is not None
        assert decision.authority_epoch is not None
        authority.phase_a_allocate(project, decision.project_incarnation, decision.authority_epoch)

    # ---- phase B: create the terminal; no transaction and no lock held here
    terminal = await _create_terminal_from_set_a(args)
    terminal_id = str(terminal.get("id") or "")
    generation = str(terminal.get("generation") or "")
    tmux_session = str(terminal.get("tmux_session") or "")

    # ---- phase C: re-verify on the same still-open connection, then bind
    recheck = evaluate_phase_a0(credentials, managed_pid_set())
    if recheck is not None:
        # A peer that exited, whose pid was recycled into the managed set, or
        # whose chain stopped answering during the 15-30 s settle wait. The
        # phase-B terminal is torn down and the phase-A epoch stays consumed.
        await _teardown(terminal_id)
        return ChannelOutcome(
            ok=False,
            reason_code=recheck.reason_code,
            detail=recheck.detail,
            terminal_created=True,
        )

    if decision.decision is authority.Decision.BOOTSTRAP and tmux_session:
        if authority.session_has_other_live_managed_terminal(tmux_session, terminal_id):
            # Excluding phase B's own terminal is what keeps this from
            # self-refusing every genuine bootstrap.
            await _teardown(terminal_id)
            return ChannelOutcome(
                ok=False,
                reason_code=REASON_BOOTSTRAP_UNAVAILABLE,
                detail=DETAIL_BOOTSTRAP_PRECONDITION,
                terminal_created=True,
            )

    bound, conflict = authority.phase_c_bind(
        authority.SupervisorTuple(
            project=project,
            supervisor_terminal_id=terminal_id,
            supervisor_generation=generation,
        ),
        decision,
        channel_socket_path=str(socket_path()),
    )
    if not bound:
        await _teardown(terminal_id)
        return ChannelOutcome(ok=False, reason_code=conflict, terminal_created=True)

    return ChannelOutcome(
        ok=True,
        authority_granted=True,
        terminal=terminal,
        terminal_created=True,
    )


def _project_for(args: Dict[str, Any]) -> str:
    """The project key this creation belongs to.

    Derived from the session name, which is a Set A launch parameter: it says
    *where to run*, and using it as the project key grants the caller no say in
    who the authority is. The authority decision reads only server-derived state
    for that key.
    """
    return str(args.get("session_name") or "")


def _pipeline_admitted(project: str) -> bool:
    """Whether the pre-closure pipeline may run for this project.

    Two ways in, and only two: the deployment's gate-2 receipt records both
    proofs, or this exact project is the one named by the operator-written
    designation. Absence of a designation means the pipeline runs nowhere.
    """
    if g10_proofs_recorded():
        return True
    designation = _designation()
    return designation is not None and bool(project) and designation.project == project


def _designation():
    """The designation loaded at server start, or ``None``.

    Read from the module-level cache rather than the filesystem, so it cannot be
    flipped mid-run to affect a decision in flight.
    """
    return _DESIGNATION_AT_START


def _existing_run_witness(project: str):
    """Whether a prior run of this project is known to survive.

    A designated proof project is disposable by construction — created for the
    run and destroyed with it — so its prior-run state is positively absent.
    For any other project this server cannot see ``project.json``, so the honest
    answer is ``UNKNOWN``, which fails closed rather than minting ``(1, 1)``.
    """
    from cli_agent_orchestrator.services import supervisor_authority as authority

    designation = _designation()
    if designation is not None and designation.project == project:
        return authority.ExistingRunWitness.ABSENT
    return authority.ExistingRunWitness.UNKNOWN


async def _teardown(terminal_id: str) -> None:
    """Remove a phase-B terminal after a phase-C abort.

    Best-effort by necessity — the abort has already happened and the epoch is
    already consumed — but a failure is logged rather than swallowed, because a
    surviving non-authoritative supervisor terminal is exactly what the
    create/no-create contract forbids.
    """
    if not terminal_id:
        return
    try:
        import asyncio

        from cli_agent_orchestrator.services import terminal_service

        # Synchronous, and it tears down tmux — kept off the event loop.
        await asyncio.to_thread(terminal_service.delete_terminal, terminal_id)
    except Exception:
        logger.exception(
            "supervisor-create: phase-C teardown of %s failed; a non-authoritative "
            "supervisor terminal may survive and must be removed by an operator",
            terminal_id,
        )


async def _create_terminal_from_set_a(args: Dict[str, Any]) -> Dict[str, Any]:
    """Forward Set A to the same creation path the TCP endpoints call.

    Two modes, matching the two calls this verb replaces: an absent target
    session creates the session and its first terminal, a present one adds the
    supervisor terminal to it. ``provider`` is not a Set A field — it is
    defaulted here exactly as the TCP endpoints default it.
    """
    from cli_agent_orchestrator.constants import DEFAULT_PROVIDER
    from cli_agent_orchestrator.services import terminal_service

    session_name = args.get("session_name")
    new_session = not _session_exists(session_name)

    allowed_tools = args.get("allowed_tools")
    if isinstance(allowed_tools, str):
        allowed_tools = [item for item in allowed_tools.split(",") if item]

    terminal = await terminal_service.create_terminal(
        provider=DEFAULT_PROVIDER,
        agent_profile=args["agent_profile"],
        session_name=session_name,
        new_session=new_session,
        working_directory=args.get("working_directory"),
        allowed_tools=allowed_tools,
        env_vars=args.get("env_vars"),
        caller_id=args.get("caller_id"),
        defer_init=bool(args.get("defer_init", False)),
        initial_message=args.get("initial_message"),
        initial_message_orchestration_type=args.get("orchestration_type"),
    )
    return terminal.model_dump(mode="json") if hasattr(terminal, "model_dump") else dict(terminal)


def _session_exists(session_name: Optional[str]) -> bool:
    """Whether the target session already has terminals recorded.

    Absence selects the atomic session-plus-first-terminal mode. A store that
    cannot be read answers "absent", which is the ordinary fresh-project shape;
    the phase-B session-name conflict is what serializes a genuine race.
    """
    if not session_name:
        return False
    try:
        from cli_agent_orchestrator.clients.database import list_terminals_by_session
        from cli_agent_orchestrator.constants import SESSION_PREFIX

        effective = (
            session_name
            if session_name.startswith(SESSION_PREFIX)
            else f"{SESSION_PREFIX}{session_name}"
        )
        return bool(list_terminals_by_session(effective))
    except Exception:
        logger.exception("supervisor-create: session lookup failed")
        return False


def encode_response(outcome: ChannelOutcome) -> bytes:
    return (json.dumps(outcome.to_wire(), separators=(",", ":")) + "\n").encode("utf-8")


#: The channel is a new listener, and this repository holds an explicit
#: contract that ``cao-server`` with no flags opens none beyond its TCP port
#: (``test/api/test_default_off_listeners.py``). It is therefore off unless
#: asked for. That is also the correct posture for this stage: G10 is unproven,
#: so nothing may use the channel yet, and the conductor still creates
#: supervisors over the TCP endpoints. Turning it on is a deployment action
#: taken with gate-2's proofs in hand — never a side effect of upgrading.
CHANNEL_ENABLED_ENV = "CAO_SUPERVISOR_CREATE_CHANNEL_ENABLED"


def channel_enabled() -> bool:
    """Whether this server should serve the supervisor-creation channel."""
    raw = os.environ.get(CHANNEL_ENABLED_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class SupervisorCreateChannel:
    """Serves the one verb on the bound socket, one request per connection.

    The connection is held open across terminal creation because the design
    requires the peer to remain verifiable across the settle wait; no lock and
    no database transaction is held there.
    """

    def __init__(self) -> None:
        self._server: Optional[socket.socket] = None
        self._lock_fd: Optional[int] = None
        self._asyncio_server: Any = None

    async def start(self) -> None:
        import asyncio

        self._server, self._lock_fd = bind_channel_socket()
        self._asyncio_server = await asyncio.start_unix_server(
            self._handle_connection, sock=self._server
        )
        logger.info("supervisor-create channel listening at %s", socket_path())

    async def _handle_connection(self, reader: Any, writer: Any) -> None:
        try:
            raw = await reader.readline()
            conn = writer.get_extra_info("socket")
            credentials = read_peer_credentials(conn) if conn is not None else None
            try:
                payload = json.loads(raw.decode("utf-8"))
                args = validate_request(payload)
            except (ValueError, SupervisorCreateChannelError) as exc:
                writer.write(
                    encode_response(
                        ChannelOutcome(
                            ok=False,
                            reason_code=REASON_SET_B_PRESENT,
                            detail=str(exc),
                        )
                    )
                )
                await writer.drain()
                return

            outcome = await handle_supervisor_terminal_create(
                args, credentials=credentials, managed=managed_pid_set()
            )
            writer.write(encode_response(outcome))
            await writer.drain()
        except Exception:
            logger.exception("supervisor-create: connection handling failed")
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def aclose(self) -> None:
        """Unlink the socket on a clean shutdown; a crash leaves it for the lock."""
        if self._asyncio_server is not None:
            self._asyncio_server.close()
            self._asyncio_server = None
        self._server = None
        try:
            path = socket_path()
            if path.exists() and stat.S_ISSOCK(path.lstat().st_mode):
                path.unlink()
        except OSError:
            logger.warning("supervisor-create: socket unlink failed", exc_info=True)
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
