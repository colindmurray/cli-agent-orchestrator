"""T-A22f and the A0/A/B/C pipeline as one orchestration.

These rows exercise the boundaries the phase split introduces: the window
between A0 and the phase-C bind is 15-30 s of settle wait, and a peer can exit
or have its pid recycled inside it. Every abort here must tear the phase-B
terminal down and leave its allocated epoch consumed, never reissued.

The pipeline only runs pre-closure for the project named by the operator-written
designation, so each row that needs it installs one — through the start-only
seam, never through a request.
"""

from __future__ import annotations

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.services import supervisor_authority as authority
from cli_agent_orchestrator.services import supervisor_create_channel as channel
from cli_agent_orchestrator.services.actor_broker import PeerCredentials
from cli_agent_orchestrator.services.gate2_proof_designation import Gate2ProofDesignation

PROOF_PROJECT = "cao-gate2-scratch"
CREDS = PeerCredentials(pid=4242, uid=501)


@pytest.fixture(autouse=True)
def _reset_designation():
    """The start-only designation is module state; never leak it between rows."""
    channel._set_designation_for_test(None)
    yield
    channel._set_designation_for_test(None)


@pytest.fixture
def designated():
    channel._set_designation_for_test(
        Gate2ProofDesignation(project=PROOF_PROJECT, sha256="0" * 64, path="/tmp/x.json")
    )


@pytest.fixture
def operator_peer(monkeypatch):
    monkeypatch.setattr(
        channel, "classify_peer_origin", lambda pid, m: (channel.PeerOrigin.OPERATOR, None)
    )


def _managed(*pids: int, enumerable: bool = True) -> channel.ManagedPidSet:
    return channel.ManagedPidSet(pids=frozenset(pids), enumerable=enumerable)


def _install_create(monkeypatch, *, terminal_id="phase-b", session=PROOF_PROJECT):
    """Record what phase B was asked to create, and create the row it would."""
    created: dict = {}

    async def fake_create(args):
        created["args"] = dict(args)
        with database.SessionLocal() as db:
            db.add(
                database.TerminalModel(
                    id=terminal_id,
                    tmux_session=session,
                    tmux_window="supervisor",
                    provider="codex",
                    generation=f"gen-{terminal_id}",
                )
            )
            db.commit()
        return {
            "id": terminal_id,
            "generation": f"gen-{terminal_id}",
            "tmux_session": session,
        }

    monkeypatch.setattr(channel, "_create_terminal_from_set_a", fake_create)
    return created


def _install_teardown(monkeypatch):
    torn: list = []

    async def fake_teardown(terminal_id):
        torn.append(terminal_id)

    monkeypatch.setattr(channel, "_teardown", fake_teardown)
    return torn


# --------------------------------------------------------------------------
# The bypass is scoped to ordinary projects.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undesignated_project_keeps_the_g10_bypass(
    isolated_memory_db, monkeypatch, operator_peer
):
    """An ordinary project takes no decision and allocates no epoch."""
    _install_create(monkeypatch, session="cao-ordinary")
    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": "cao-ordinary"},
        credentials=CREDS,
        managed=_managed(999),
    )
    assert outcome.ok is True
    assert outcome.terminal_created is True
    assert outcome.authority_granted is False
    assert outcome.detail == channel.DETAIL_G10_UNPROVEN
    assert authority.compute_high_water("cao-ordinary").any_source is False


@pytest.mark.asyncio
async def test_designation_admits_only_its_exact_project(
    isolated_memory_db, monkeypatch, designated, operator_peer
):
    """Every other project on the deployment keeps the ordinary bypass."""
    _install_create(monkeypatch, session="cao-other")
    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": "cao-other"},
        credentials=CREDS,
        managed=_managed(999),
    )
    assert outcome.detail == channel.DETAIL_G10_UNPROVEN
    assert outcome.authority_granted is False


# --------------------------------------------------------------------------
# The admitting path: A0 -> A -> B -> C.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_binds_authority_for_the_proof_project(
    isolated_memory_db, monkeypatch, designated, operator_peer
):
    monkeypatch.setattr(channel, "managed_pid_set", lambda: _managed(999))
    created = _install_create(monkeypatch)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": PROOF_PROJECT},
        credentials=CREDS,
        managed=_managed(999),
    )

    assert outcome.ok is True
    assert outcome.authority_granted is True
    assert outcome.reason_code is None
    with database.SessionLocal() as db:
        row = db.query(database.ProjectSupervisorAuthorityModel).one()
        assert (row.project_incarnation, row.authority_epoch) == (1, 1)
        assert row.supervisor_terminal_id == "phase-b"
        assert row.supervisor_generation == "gen-phase-b"
    # The allocation was committed in phase A, before creation.
    assert authority.compute_high_water(PROOF_PROJECT).epoch_max == 1
    assert created["args"]["agent_profile"] == "code_supervisor"


# --------------------------------------------------------------------------
# T-A22f: phase-C re-verification and its aborts.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_c_peer_exit_tears_down_and_consumes_the_epoch(
    isolated_memory_db, monkeypatch, designated
):
    """The peer was operator-origin at A0 and is gone by phase C."""
    calls = {"n": 0}

    def classify(pid, m):
        calls["n"] += 1
        if calls["n"] == 1:
            return channel.PeerOrigin.OPERATOR, None
        return channel.PeerOrigin.UNPROVEN, channel.DETAIL_PEER_NOT_LIVE

    monkeypatch.setattr(channel, "classify_peer_origin", classify)
    monkeypatch.setattr(channel, "managed_pid_set", lambda: _managed(999))
    _install_create(monkeypatch)
    torn = _install_teardown(monkeypatch)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": PROOF_PROJECT},
        credentials=CREDS,
        managed=_managed(999),
    )

    assert outcome.ok is False
    assert outcome.reason_code == channel.REASON_LINEAGE_UNPROVEN
    assert outcome.detail == channel.DETAIL_PEER_NOT_LIVE
    assert outcome.terminal_created is True
    assert torn == ["phase-b"], "the phase-B terminal must be torn down"
    with database.SessionLocal() as db:
        assert db.query(database.ProjectSupervisorAuthorityModel).count() == 0
    # Consumed, never reissued.
    assert authority.compute_high_water(PROOF_PROJECT).epoch_max == 1
    nxt = authority.decide(
        authority.SupervisorTuple(PROOF_PROJECT, "t2", "g2"),
        witness=authority.ExistingRunWitness.ABSENT,
    )
    assert nxt.authority_epoch == 2


@pytest.mark.asyncio
async def test_phase_c_pid_recycled_into_managed_set_tears_down(
    isolated_memory_db, monkeypatch, designated
):
    """A recycled pid that is now managed refuses with the discriminator code.

    Distinct from the UNPROVEN case, and distinct in the create/no-create
    contract: this one created a terminal in B, so it must be torn down.
    """
    calls = {"n": 0}

    def classify(pid, m):
        calls["n"] += 1
        if calls["n"] == 1:
            return channel.PeerOrigin.OPERATOR, None
        return channel.PeerOrigin.MANAGED, None

    monkeypatch.setattr(channel, "classify_peer_origin", classify)
    monkeypatch.setattr(channel, "managed_pid_set", lambda: _managed(4242))
    _install_create(monkeypatch)
    torn = _install_teardown(monkeypatch)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": PROOF_PROJECT},
        credentials=CREDS,
        managed=_managed(999),
    )

    assert outcome.reason_code == channel.REASON_DISCRIMINATOR_ABSENT
    assert outcome.terminal_created is True
    assert torn == ["phase-b"]
    with database.SessionLocal() as db:
        assert db.query(database.ProjectSupervisorAuthorityModel).count() == 0


@pytest.mark.asyncio
async def test_bootstrap_precondition_refuses_on_a_foreign_live_terminal(
    isolated_memory_db, monkeypatch, designated, operator_peer
):
    monkeypatch.setattr(channel, "managed_pid_set", lambda: _managed(999))
    _install_create(monkeypatch)
    torn = _install_teardown(monkeypatch)
    with database.SessionLocal() as db:
        db.add(
            database.TerminalModel(
                id="squatter",
                tmux_session=PROOF_PROJECT,
                tmux_window="worker",
                provider="codex",
            )
        )
        db.commit()

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": PROOF_PROJECT},
        credentials=CREDS,
        managed=_managed(999),
    )

    assert outcome.reason_code == channel.REASON_BOOTSTRAP_UNAVAILABLE
    assert outcome.detail == channel.DETAIL_BOOTSTRAP_PRECONDITION
    assert outcome.terminal_created is True
    assert torn == ["phase-b"]


@pytest.mark.asyncio
async def test_phase_b_own_terminal_never_triggers_the_precondition(
    isolated_memory_db, monkeypatch, designated, operator_peer
):
    """T-A22f together with a genuinely fresh bootstrap must be satisfiable."""
    monkeypatch.setattr(channel, "managed_pid_set", lambda: _managed(999))
    _install_create(monkeypatch)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": PROOF_PROJECT},
        credentials=CREDS,
        managed=_managed(999),
    )
    assert outcome.authority_granted is True


@pytest.mark.asyncio
async def test_lost_phase_c_cas_tears_down_and_reports_rotation_conflict(
    isolated_memory_db, monkeypatch, designated, operator_peer
):
    """The other creator won between our decision and our CAS."""
    monkeypatch.setattr(channel, "managed_pid_set", lambda: _managed(999))
    _install_create(monkeypatch)
    torn = _install_teardown(monkeypatch)

    real_bind = authority.phase_c_bind

    def losing_bind(tuple_, decision, **kwargs):
        with database.SessionLocal() as db:
            db.add(
                database.ProjectSupervisorAuthorityModel(
                    project=PROOF_PROJECT,
                    project_incarnation=1,
                    supervisor_terminal_id="winner",
                    supervisor_generation="gen-winner",
                    authority_epoch=1,
                    state=authority.STATE_LIVE,
                )
            )
            db.commit()
        return real_bind(tuple_, decision, **kwargs)

    monkeypatch.setattr(authority, "phase_c_bind", losing_bind)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": PROOF_PROJECT},
        credentials=CREDS,
        managed=_managed(999),
    )

    assert outcome.ok is False
    assert outcome.reason_code == authority.REASON_ROTATION_CONFLICT
    assert torn == ["phase-b"]
    with database.SessionLocal() as db:
        assert (
            db.query(database.ProjectSupervisorAuthorityModel).one().supervisor_terminal_id
            == "winner"
        )


# --------------------------------------------------------------------------
# A phase-A decision refusal creates nothing at all.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_supervisor_refusal_creates_no_terminal(
    isolated_memory_db, monkeypatch, designated, operator_peer
):
    """Reached in phase A, before phase B: no allocation and no terminal.

    Leaving a stray non-authoritative supervisor terminal behind here would be
    wrong, so the assertion is that creation was never called at all.
    """
    monkeypatch.setattr(channel, "managed_pid_set", lambda: _managed(999))
    created = _install_create(monkeypatch)
    with database.SessionLocal() as db:
        db.add(
            database.ProjectSupervisorAuthorityModel(
                project=PROOF_PROJECT,
                project_incarnation=7,
                supervisor_terminal_id="live-t",
                supervisor_generation="live-g",
                authority_epoch=11,
                state=authority.STATE_LIVE,
            )
        )
        db.commit()
    monkeypatch.setattr(authority, "is_supervisor_live", lambda t, g: True)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": PROOF_PROJECT},
        credentials=CREDS,
        managed=_managed(999),
    )

    assert outcome.ok is False
    assert outcome.reason_code == authority.REASON_LIVE_SUPERVISOR_PRESENT
    assert outcome.terminal_created is False
    assert created == {}, "phase A refused, so phase B never ran"
    assert authority.compute_high_water(PROOF_PROJECT).epoch_max == 11, "no new allocation"


@pytest.mark.asyncio
async def test_a0_refusal_short_circuits_before_any_phase(
    isolated_memory_db, monkeypatch, designated
):
    monkeypatch.setattr(
        channel, "classify_peer_origin", lambda pid, m: (channel.PeerOrigin.MANAGED, None)
    )
    created = _install_create(monkeypatch)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": PROOF_PROJECT},
        credentials=CREDS,
        managed=_managed(4242),
    )

    assert outcome.reason_code == channel.REASON_DISCRIMINATOR_ABSENT
    assert outcome.terminal_created is False
    assert created == {}
    assert authority.compute_high_water(PROOF_PROJECT).any_source is False


# --------------------------------------------------------------------------
# Set A still reaches creation unchanged, whatever the authority outcome.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_fidelity_holds_independently_of_the_authority_outcome(
    isolated_memory_db, monkeypatch, designated, operator_peer
):
    """Set A must survive both the bypass and the full pipeline."""
    monkeypatch.setattr(channel, "managed_pid_set", lambda: _managed(999))
    created = _install_create(monkeypatch)
    args = {
        "session_name": PROOF_PROJECT,
        "agent_profile": "code_supervisor",
        "working_directory": "/wt/proj",
        "env_vars": {"PATH": "/shim/bin", "ZDOTDIR": "/shim/zsh"},
        "caller_id": "abcd1234",
        "initial_message": "bootstrap",
        "orchestration_type": "assign",
        "allowed_tools": "a,b",
        "defer_init": True,
    }

    outcome = await channel.handle_supervisor_terminal_create(
        dict(args), credentials=CREDS, managed=_managed(999)
    )
    assert outcome.authority_granted is True
    assert created["args"]["working_directory"] == "/wt/proj"
    assert created["args"]["env_vars"] == {"PATH": "/shim/bin", "ZDOTDIR": "/shim/zsh"}
    assert created["args"]["defer_init"] is True
