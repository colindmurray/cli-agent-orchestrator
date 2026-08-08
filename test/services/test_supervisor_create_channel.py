"""Supervisor-creation channel: origin discrimination and the G10 posture.

The rows here are the fork-owned negative boundaries of the route-attestation
design's gate-2 proof 2. They are deliberately weighted toward refusal: the
channel's whole purpose is to answer "may this peer ask for a supervisor
terminal?", and every wrong answer in the admitting direction is an authority
hole.
"""

from __future__ import annotations

import os
import shutil
import socket
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from cli_agent_orchestrator.services import supervisor_create_channel as channel
from cli_agent_orchestrator.services.actor_broker import PeerCredentials


def _managed(*pids: int, enumerable: bool = True) -> channel.ManagedPidSet:
    return channel.ManagedPidSet(pids=frozenset(pids), enumerable=enumerable)


@pytest.fixture
def short_state_root(monkeypatch):
    """A state root short enough to hold an ``AF_UNIX`` path.

    pytest's ``tmp_path`` is ~138 bytes on macOS
    (``/private/var/folders/...``), which the channel correctly refuses as over
    the 100-byte bound — so the socket-lifecycle rows need a short root rather
    than a relaxed bound. The real default root
    (``~/.aws/cli-agent-orchestrator``) is comfortably inside it.
    """
    import cli_agent_orchestrator.constants as constants

    root = Path(tempfile.mkdtemp(prefix="cao-sc-", dir="/tmp"))
    monkeypatch.setattr(constants, "CAO_HOME_DIR", root)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------
# The predicate is negative, so "unreadable" must never read as "operator".
# --------------------------------------------------------------------------


def test_unreadable_ancestry_is_unproven_not_operator(monkeypatch):
    """A chain that stops answering cannot prove absence of managed ancestry.

    This is the inversion that makes this module's walker different from
    ``actor_broker``'s: there, False means "refuse"; here, a two-state False
    would mean "admit".
    """
    monkeypatch.setattr(channel, "_parent_pid", lambda pid: None)
    assert channel.classify_peer_origin(4242, _managed(99)) is channel.PeerOrigin.UNPROVEN


def test_dead_peer_is_unproven(monkeypatch):
    """An exited peer yields no ps output, so its origin is unprovable."""
    monkeypatch.setattr(
        channel,
        "_parent_pid",
        lambda pid: None,
    )
    origin = channel.classify_peer_origin(999_999, _managed(1234))
    assert origin is channel.PeerOrigin.UNPROVEN


def test_non_enumerable_managed_set_is_unproven():
    """A store that cannot be read proves nothing about origin."""
    assert (
        channel.classify_peer_origin(1234, _managed(enumerable=False))
        is channel.PeerOrigin.UNPROVEN
    )


def test_managed_ancestor_anywhere_in_chain_is_managed(monkeypatch):
    """Depth does not launder origin: a worker's grandchild is still managed."""
    chain = {500: 400, 400: 300, 300: 200, 200: 1}
    monkeypatch.setattr(channel, "_parent_pid", lambda pid: chain.get(pid))
    assert channel.classify_peer_origin(500, _managed(300)) is channel.PeerOrigin.MANAGED


def test_peer_that_is_itself_a_managed_pane_is_managed(monkeypatch):
    monkeypatch.setattr(channel, "_parent_pid", lambda pid: 1)
    assert channel.classify_peer_origin(777, _managed(777)) is channel.PeerOrigin.MANAGED


def test_chain_reaching_init_without_managed_pid_is_operator(monkeypatch):
    chain = {800: 700, 700: 1}
    monkeypatch.setattr(channel, "_parent_pid", lambda pid: chain.get(pid))
    assert channel.classify_peer_origin(800, _managed(300, 400)) is channel.PeerOrigin.OPERATOR


def test_ancestry_cycle_is_unproven(monkeypatch):
    chain = {10: 11, 11: 10}
    monkeypatch.setattr(channel, "_parent_pid", lambda pid: chain.get(pid))
    assert channel.classify_peer_origin(10, _managed(999)) is channel.PeerOrigin.UNPROVEN


def test_hop_budget_exhaustion_is_unproven(monkeypatch):
    """A chain longer than the budget was never walked to init, so it proves nothing."""
    monkeypatch.setattr(channel, "_parent_pid", lambda pid: pid + 1)
    assert channel.classify_peer_origin(2, _managed(10**9)) is channel.PeerOrigin.UNPROVEN


def test_nonpositive_peer_pid_is_unproven():
    assert channel.classify_peer_origin(0, _managed(1)) is channel.PeerOrigin.UNPROVEN


# --------------------------------------------------------------------------
# Phase A0 refusals: nothing is created on either.
# --------------------------------------------------------------------------


def test_phase_a0_refuses_managed_origin_with_discriminator_absent(monkeypatch):
    monkeypatch.setattr(channel, "classify_peer_origin", lambda pid, m: channel.PeerOrigin.MANAGED)
    outcome = channel.evaluate_phase_a0(PeerCredentials(pid=5, uid=501), _managed(5))
    assert outcome is not None
    assert outcome.ok is False
    assert outcome.reason_code == channel.REASON_DISCRIMINATOR_ABSENT
    assert outcome.terminal_created is False
    assert outcome.authority_granted is False


def test_phase_a0_refuses_unproven_with_lineage_unproven(monkeypatch):
    monkeypatch.setattr(channel, "classify_peer_origin", lambda pid, m: channel.PeerOrigin.UNPROVEN)
    outcome = channel.evaluate_phase_a0(PeerCredentials(pid=5, uid=501), _managed(5))
    assert outcome is not None
    assert outcome.reason_code == channel.REASON_LINEAGE_UNPROVEN
    assert outcome.terminal_created is False


def test_phase_a0_refuses_absent_credentials():
    """No kernel identity is not an admission."""
    outcome = channel.evaluate_phase_a0(None, _managed(1))
    assert outcome is not None
    assert outcome.reason_code == channel.REASON_LINEAGE_UNPROVEN


def test_phase_a0_admits_proven_operator(monkeypatch):
    monkeypatch.setattr(channel, "classify_peer_origin", lambda pid, m: channel.PeerOrigin.OPERATOR)
    assert channel.evaluate_phase_a0(PeerCredentials(pid=5, uid=501), _managed(9)) is None


# --------------------------------------------------------------------------
# Set A / Set B: the partition is what keeps this from being an establish door.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(channel.SET_B_REFUSED_FIELDS))
def test_every_set_b_field_is_refused_not_ignored(field):
    payload = {
        "verb": channel.VERB_SUPERVISOR_TERMINAL_CREATE,
        "args": {"agent_profile": "code_supervisor", field: "anything"},
    }
    with pytest.raises(channel.SupervisorCreateChannelError) as excinfo:
        channel.validate_request(payload)
    assert field in str(excinfo.value)


def test_full_set_a_is_accepted():
    args = {
        "session_name": "p1",
        "agent_profile": "code_supervisor",
        "working_directory": "/tmp/wt",
        "env_vars": {"PATH": "/x", "ZDOTDIR": "/y"},
        "caller_id": "abcd1234",
        "initial_message": "hello",
        "orchestration_type": "assign",
        "allowed_tools": "a,b",
        "defer_init": True,
    }
    assert set(args) == set(channel.SET_A_FIELDS)
    got = channel.validate_request({"verb": channel.VERB_SUPERVISOR_TERMINAL_CREATE, "args": args})
    assert got == args


def test_unknown_launch_parameter_is_refused():
    with pytest.raises(channel.SupervisorCreateChannelError):
        channel.validate_request(
            {
                "verb": channel.VERB_SUPERVISOR_TERMINAL_CREATE,
                "args": {"agent_profile": "x", "provider": "codex"},
            }
        )


def test_no_establish_verb_exists():
    for verb in ("establish", "supervisor-authority-establish", "grant mint", "rotate"):
        with pytest.raises(channel.SupervisorCreateChannelError):
            channel.validate_request({"verb": verb, "args": {"agent_profile": "x"}})


def test_agent_profile_is_required_but_is_not_a_discriminator():
    """It must be present to launch, and absent from every decision input.

    Asserted structurally: the A0 decision takes only credentials and the
    server's pid set, so no profile value can reach it.
    """
    with pytest.raises(channel.SupervisorCreateChannelError):
        channel.validate_request({"verb": channel.VERB_SUPERVISOR_TERMINAL_CREATE, "args": {}})

    import inspect

    signature = inspect.signature(channel.evaluate_phase_a0)
    assert set(signature.parameters) == {"credentials", "managed"}


def test_request_must_be_an_object():
    for bad in ([], "x", 3, None):
        with pytest.raises(channel.SupervisorCreateChannelError):
            channel.validate_request(bad)


# --------------------------------------------------------------------------
# The G10 posture: dark, and not a bring-up failure.
# --------------------------------------------------------------------------


def test_g10_proofs_are_not_recorded_in_source():
    """Source can never constitute a deployment proof."""
    assert channel.g10_proofs_recorded() is False


@pytest.mark.asyncio
async def test_g10_unproven_creates_the_terminal_and_refuses_only_authority(monkeypatch):
    """The sole create-without-authority outcome, and the command still succeeds.

    An implementation that reported this as a failure would break every
    ``conduct up`` until gate 2 closes.
    """
    monkeypatch.setattr(channel, "classify_peer_origin", lambda pid, m: channel.PeerOrigin.OPERATOR)

    async def fake_create(args):
        return {"id": "term1234", "agent_profile": args["agent_profile"]}

    monkeypatch.setattr(channel, "_create_terminal_from_set_a", fake_create)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor"},
        credentials=PeerCredentials(pid=42, uid=501),
        managed=_managed(999),
    )

    assert outcome.ok is True
    assert outcome.terminal_created is True
    assert outcome.terminal == {"id": "term1234", "agent_profile": "code_supervisor"}
    assert outcome.reason_code == channel.REASON_BOOTSTRAP_UNAVAILABLE
    assert outcome.detail == channel.DETAIL_G10_UNPROVEN
    assert outcome.authority_granted is False


@pytest.mark.asyncio
async def test_managed_peer_creates_no_terminal(monkeypatch):
    """The reproduced hole: a worker in situ gets nothing, whatever it sends."""
    monkeypatch.setattr(channel, "classify_peer_origin", lambda pid, m: channel.PeerOrigin.MANAGED)

    called = False

    async def fake_create(args):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(channel, "_create_terminal_from_set_a", fake_create)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor"},
        credentials=PeerCredentials(pid=42, uid=501),
        managed=_managed(42),
    )

    assert called is False, "a refused origin must never reach creation"
    assert outcome.reason_code == channel.REASON_DISCRIMINATOR_ABSENT
    assert outcome.terminal_created is False


@pytest.mark.asyncio
async def test_every_profile_value_is_refused_for_a_managed_peer(monkeypatch):
    """No ``agent_profile`` value buys a managed peer anything."""
    monkeypatch.setattr(channel, "classify_peer_origin", lambda pid, m: channel.PeerOrigin.MANAGED)
    for profile in ("code_supervisor", "supervisor", "developer", "memory_manager"):
        outcome = await channel.handle_supervisor_terminal_create(
            {"agent_profile": profile},
            credentials=PeerCredentials(pid=7, uid=501),
            managed=_managed(7),
        )
        assert outcome.reason_code == channel.REASON_DISCRIMINATOR_ABSENT
        assert outcome.authority_granted is False


def test_two_bootstrap_unavailable_paths_have_distinct_details():
    """Same code, opposite terminal fates, so the wire must distinguish them."""
    assert channel.DETAIL_G10_UNPROVEN != channel.DETAIL_BOOTSTRAP_PRECONDITION


def test_reason_codes_are_from_the_closed_vocabulary():
    """This module invents no code.

    The four it emits are members of the design's closed v1 list; detail fields
    are permitted by that list and are used instead of new codes.
    """
    assert channel.REASON_DISCRIMINATOR_ABSENT == "supervisor-creation-discriminator-absent"
    assert channel.REASON_LINEAGE_UNPROVEN == "authority-lineage-unproven"
    assert channel.REASON_BOOTSTRAP_UNAVAILABLE == "authority-bootstrap-unavailable"
    assert channel.REASON_SET_B_PRESENT == "operation-admission-unproven"


# --------------------------------------------------------------------------
# Socket lifecycle: path bound, staleness, mode, and a live owner.
# --------------------------------------------------------------------------


def test_socket_path_over_the_af_unix_bound_fails_closed():
    long_path = Path("/" + ("x" * 200)) / channel.SOCKET_BASENAME
    with pytest.raises(channel.SupervisorCreateChannelError) as excinfo:
        channel.validate_socket_path(long_path)
    assert "AF_UNIX" in str(excinfo.value)


def test_socket_path_within_the_bound_is_accepted(short_state_root):
    channel.validate_socket_path(short_state_root / channel.SOCKET_BASENAME)


def test_socket_path_follows_the_state_root(short_state_root):
    """Resolved at call time, by the fork's one state-root rule and no other."""
    assert channel.socket_path() == short_state_root / channel.SOCKET_BASENAME
    assert channel.lock_path() == short_state_root / channel.LOCK_BASENAME


def test_bind_creates_a_0600_socket(short_state_root):
    server, descriptor = channel.bind_channel_socket()
    try:
        info = (short_state_root / channel.SOCKET_BASENAME).lstat()
        assert stat.S_ISSOCK(info.st_mode)
        assert stat.S_IMODE(info.st_mode) == 0o600
    finally:
        server.close()
        os.close(descriptor)


def test_bind_clears_a_stale_socket_node(short_state_root):
    """A crash leaves a node behind; the lock proves it is residue."""
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(short_state_root / channel.SOCKET_BASENAME))
    stale.close()  # node persists on disk

    server, descriptor = channel.bind_channel_socket()
    try:
        assert (short_state_root / channel.SOCKET_BASENAME).exists()
    finally:
        server.close()
        os.close(descriptor)


def test_bind_refuses_when_a_live_server_holds_the_lock(short_state_root):
    """Startup fails closed rather than unlinking a channel in use."""
    first_server, first_descriptor = channel.bind_channel_socket()
    try:
        with pytest.raises(channel.SupervisorCreateChannelError) as excinfo:
            channel.bind_channel_socket()
        assert "already owns" in str(excinfo.value)
    finally:
        first_server.close()
        os.close(first_descriptor)


def test_bind_refuses_to_unlink_a_non_socket(short_state_root):
    (short_state_root / channel.SOCKET_BASENAME).write_text("not a socket")
    with pytest.raises(channel.SupervisorCreateChannelError) as excinfo:
        channel.bind_channel_socket()
    assert "not a socket" in str(excinfo.value)


# --------------------------------------------------------------------------
# The managed pid set is server-derived, and its failure mode is explicit.
# --------------------------------------------------------------------------


def test_managed_pid_set_includes_this_process(monkeypatch):
    monkeypatch.setattr(channel, "_tmux_server_pid", lambda: None)
    got = channel.managed_pid_set()
    assert got.enumerable is True
    assert os.getpid() in got.pids


def test_managed_pid_set_includes_the_tmux_server(monkeypatch):
    monkeypatch.setattr(channel, "_tmux_server_pid", lambda: 31337)
    assert 31337 in channel.managed_pid_set().pids


def test_managed_pid_set_reports_non_enumerable_on_store_failure(monkeypatch):
    """A failed query must not masquerade as "no managed pids"."""

    def boom():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        channel,
        "_tmux_server_pid",
        lambda: None,
    )
    import cli_agent_orchestrator.clients.database as database

    monkeypatch.setattr(database, "SessionLocal", boom)
    got = channel.managed_pid_set()
    assert got.enumerable is False
    assert got.pids == frozenset()


def test_absent_tmux_server_is_not_an_enumeration_failure(monkeypatch):
    monkeypatch.setattr(channel, "_tmux_server_pid", lambda: None)
    assert channel.managed_pid_set().enumerable is True


def test_parent_pid_returns_none_when_ps_fails(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no ps")

    monkeypatch.setattr(subprocess, "run", boom)
    assert channel._parent_pid(123) is None


def test_parent_pid_returns_none_on_empty_output(monkeypatch):
    class Result:
        stdout = "\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    assert channel._parent_pid(123) is None


# --------------------------------------------------------------------------
# Wire surface.
# --------------------------------------------------------------------------


def test_wire_response_never_carries_authority_material():
    outcome = channel.ChannelOutcome(
        ok=True,
        reason_code=channel.REASON_BOOTSTRAP_UNAVAILABLE,
        detail=channel.DETAIL_G10_UNPROVEN,
        terminal={"id": "t1"},
        terminal_created=True,
    )
    wire = outcome.to_wire()
    assert set(wire) == {
        "ok",
        "reason_code",
        "detail",
        "authority_granted",
        "terminal_created",
        "terminal",
    }
    serialized = channel.encode_response(outcome).decode()
    for forbidden in ("epoch", "credential", "secret", "token", "signing", "hmac"):
        assert forbidden not in serialized.lower()


# --------------------------------------------------------------------------
# Default-off posture: an upgrade opens no new listener by itself.
# --------------------------------------------------------------------------


def test_channel_is_disabled_without_the_flag(monkeypatch):
    monkeypatch.delenv(channel.CHANNEL_ENABLED_ENV, raising=False)
    assert channel.channel_enabled() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_channel_enables_on_explicit_truthy_values(monkeypatch, raw):
    monkeypatch.setenv(channel.CHANNEL_ENABLED_ENV, raw)
    assert channel.channel_enabled() is True


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "maybe"])
def test_channel_stays_off_for_anything_else(monkeypatch, raw):
    monkeypatch.setenv(channel.CHANNEL_ENABLED_ENV, raw)
    assert channel.channel_enabled() is False


def test_server_opens_no_channel_socket_by_default(monkeypatch, short_state_root):
    """The build's no-extra-listener contract, asserted for this surface."""
    from fastapi.testclient import TestClient

    from cli_agent_orchestrator.api.main import app

    monkeypatch.delenv(channel.CHANNEL_ENABLED_ENV, raising=False)
    with TestClient(app, base_url="http://localhost"):
        assert getattr(app.state, "supervisor_create_channel", None) is None
        assert not (short_state_root / channel.SOCKET_BASENAME).exists()
