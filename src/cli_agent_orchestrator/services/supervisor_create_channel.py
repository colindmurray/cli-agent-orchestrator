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

**What this module deliberately does not contain.** The authority record, epoch
allocation, history, the recovery high-water calculation, and the phase-A/phase-C
bind are absent. They sit behind G10, whose two deployment proofs are not
recorded (:func:`g10_proofs_recorded`), and the design forbids building behind a
fail-closed refusal. With G10 unproven no authority decision is taken and no
epoch is allocated: phase A0 verifies the peer, phase B creates the terminal,
and the authority side effect refuses ``authority-bootstrap-unavailable`` with
detail ``g10-unproven``. That refusal is **not** a bring-up failure — it is the
one outcome where the verb succeeds and returns a terminal while carrying no
authority, and treating it as a failure would break every bring-up on every
deployment until G10 closes.
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

#: The fork's existing safe bound for an ``AF_UNIX`` pathname. A longer path
#: fails closed at startup rather than binding a silently truncated name.
AF_UNIX_SAFE_PATH_BYTES = 100

#: Hop budget for an ancestry walk. Exhausting it without reaching init is not
#: a proof of anything, so it answers ``UNPROVEN``.
_MAX_ANCESTRY_HOPS = 64


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
                if pid:
                    pids.add(int(pid))
            for (pid,) in db.query(ManagedLaunchV2TerminalModel.v2_pane_pid).all():
                if pid:
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


def classify_peer_origin(peer_pid: int, managed: ManagedPidSet) -> PeerOrigin:
    """Walk the peer's live ancestry and answer which of the three states holds.

    Re-walked live rather than cached: kernel peer credentials are connect-time
    fixed on both platforms, so a pid alone cannot tell a live peer from an
    exited one whose pid has since been recycled. The walk is the detector.
    """
    if not managed.enumerable:
        return PeerOrigin.UNPROVEN
    if peer_pid <= 0:
        return PeerOrigin.UNPROVEN

    seen: set[int] = set()
    current = peer_pid
    for _ in range(_MAX_ANCESTRY_HOPS):
        if current in managed.pids:
            return PeerOrigin.MANAGED
        if current <= 1:
            # Reached init without meeting a managed pid: proven outside.
            return PeerOrigin.OPERATOR
        if current in seen:
            return PeerOrigin.UNPROVEN
        seen.add(current)
        parent = _parent_pid(current)
        if parent is None:
            # A chain that stops answering cannot prove absence of ancestry.
            return PeerOrigin.UNPROVEN
        current = parent
    return PeerOrigin.UNPROVEN


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
    the call may proceed. Nothing is written on either refusal.
    """
    if credentials is None:
        return ChannelOutcome(ok=False, reason_code=REASON_LINEAGE_UNPROVEN)

    origin = classify_peer_origin(credentials.pid, managed)
    if origin is PeerOrigin.MANAGED:
        return ChannelOutcome(ok=False, reason_code=REASON_DISCRIMINATOR_ABSENT)
    if origin is PeerOrigin.UNPROVEN:
        return ChannelOutcome(ok=False, reason_code=REASON_LINEAGE_UNPROVEN)
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
    refusal = evaluate_phase_a0(credentials, managed)
    if refusal is not None:
        return refusal

    terminal = await _create_terminal_from_set_a(args)

    if not g10_proofs_recorded():
        # The sole create-without-authority outcome. ``ok`` is True because the
        # command did what it was asked to do; authority simply does not exist
        # on this deployment yet.
        return ChannelOutcome(
            ok=True,
            reason_code=REASON_BOOTSTRAP_UNAVAILABLE,
            detail=DETAIL_G10_UNPROVEN,
            authority_granted=False,
            terminal=terminal,
            terminal_created=True,
        )

    # Unreachable while G10 is unproven. Phases A and C — the epoch allocation,
    # the recovery high-water, the history append, and the single-row CAS — are
    # deliberately absent rather than stubbed: the design forbids building
    # behind a fail-closed refusal, and a stub here would be indistinguishable
    # from an implementation that had been reviewed.
    raise SupervisorCreateChannelError(
        "authority establishment is not implemented: G10 is unproven and the "
        "phase-A/phase-C bind is out of scope until it closes"
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
