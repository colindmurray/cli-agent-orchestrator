"""T-A22a: the three intervals, their details, and where interval choice comes from.

The interval a deployment is in is read from durable start-state and from nothing
else. The two bypass intervals must be distinguishable, because emitting
``g10-unproven`` after gate 2 has closed would be a false statement about the
deployment — G10 *is* proven by then.
"""

from __future__ import annotations

import json

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.services import gate2_proof_receipt_state as rs
from cli_agent_orchestrator.services import project_state_binding as psb
from cli_agent_orchestrator.services import supervisor_authority as authority
from cli_agent_orchestrator.services import supervisor_create_channel as channel
from cli_agent_orchestrator.services.actor_broker import PeerCredentials
from cli_agent_orchestrator.services.gate2_proof_designation import Gate2ProofDesignation

ORDINARY = "cao-ordinary"
PROOF_PROJECT = "cao-gate2-scratch"
CREDS = PeerCredentials(pid=4242, uid=501)


@pytest.fixture(autouse=True)
def _clean_start_state():
    channel._set_designation_for_test(None)
    channel._set_receipt_state_for_test(None)
    yield
    channel._set_designation_for_test(None)
    channel._set_receipt_state_for_test(None)


@pytest.fixture
def state_root(monkeypatch, tmp_path):
    import cli_agent_orchestrator.constants as constants

    root = tmp_path / "state"
    root.mkdir()
    monkeypatch.setattr(constants, "CAO_HOME_DIR", root)
    return root


@pytest.fixture
def operator_peer(monkeypatch):
    monkeypatch.setattr(
        channel, "classify_peer_origin", lambda pid, m: (channel.PeerOrigin.OPERATOR, None)
    )
    monkeypatch.setattr(channel, "managed_pid_set", lambda: _managed(999))


def _managed(*pids: int) -> channel.ManagedPidSet:
    return channel.ManagedPidSet(pids=frozenset(pids), enumerable=True)


def _recorded_state() -> rs.Gate2ReceiptState:
    return rs.Gate2ReceiptState(
        receipt_sha256="a" * 64,
        proofs_recorded=rs.REQUIRED_PROOFS,
        sha256="b" * 64,
        path="/tmp/receipt-state.json",
    )


def _install_create(monkeypatch, *, terminal_id="phase-b", session=ORDINARY):
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


async def _run(project: str) -> channel.ChannelOutcome:
    return await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": project},
        credentials=CREDS,
        managed=_managed(999),
    )


# --------------------------------------------------------------------------
# T-A22a — interval i: gate 2 open.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interval_one_emits_g10_unproven_and_retains_the_terminal(
    isolated_memory_db, state_root, monkeypatch, operator_peer
):
    _install_create(monkeypatch)
    outcome = await _run(ORDINARY)
    assert outcome.ok is True
    assert outcome.terminal_created is True
    assert outcome.authority_granted is False
    assert outcome.reason_code == channel.REASON_BOOTSTRAP_UNAVAILABLE
    assert outcome.detail == channel.DETAIL_G10_UNPROVEN
    with database.SessionLocal() as db:
        assert db.query(database.ProjectSupervisorAuthorityModel).count() == 0
    assert authority.compute_high_water(ORDINARY).any_source is False


# --------------------------------------------------------------------------
# T-A22a — interval ii: gate 2 closed, binding not yet recorded.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interval_two_emits_witness_binding_unavailable(
    isolated_memory_db, state_root, monkeypatch, operator_peer
):
    """Emitting `g10-unproven` here would be false, and T-A22a fails that."""
    channel._set_receipt_state_for_test(_recorded_state())
    _install_create(monkeypatch)

    outcome = await _run(ORDINARY)

    assert outcome.detail == channel.DETAIL_WITNESS_BINDING_UNAVAILABLE
    assert outcome.detail != channel.DETAIL_G10_UNPROVEN
    assert outcome.ok is True
    assert outcome.terminal_created is True, "bring-up must keep working"
    assert outcome.authority_granted is False
    assert authority.compute_high_water(ORDINARY).any_source is False


@pytest.mark.asyncio
async def test_interval_two_is_not_reached_on_a_partial_proof_list(
    isolated_memory_db, state_root, monkeypatch, operator_peer
):
    """A partial `proofs_recorded` satisfies nothing, so the deployment stays in i."""
    channel._set_receipt_state_for_test(
        rs.Gate2ReceiptState(
            receipt_sha256="a" * 64,
            proofs_recorded=(rs.PROOF_LINEAGE_ISOLATION,),
            sha256="b" * 64,
            path="/tmp/x.json",
        )
    )
    _install_create(monkeypatch)
    outcome = await _run(ORDINARY)
    assert outcome.detail == channel.DETAIL_G10_UNPROVEN


# --------------------------------------------------------------------------
# T-A22a — interval iii: gate 2 closed and the binding recorded.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interval_three_runs_the_pipeline_and_mints_on_observed_absent(
    isolated_memory_db, state_root, tmp_path, monkeypatch, operator_peer
):
    channel._set_receipt_state_for_test(_recorded_state())
    project_dir = tmp_path / "conductor-ordinary"
    project_dir.mkdir()  # readable, and demonstrably lacking project.json
    psb.write_binding_for_project(psb.binding_path(ORDINARY), ORDINARY, str(project_dir))
    _install_create(monkeypatch)

    outcome = await _run(ORDINARY)

    assert outcome.authority_granted is True
    assert outcome.reason_code is None
    with database.SessionLocal() as db:
        row = db.query(database.ProjectSupervisorAuthorityModel).one()
        assert (row.project_incarnation, row.authority_epoch) == (1, 1)


@pytest.mark.asyncio
async def test_interval_three_unknown_witness_still_refuses(
    isolated_memory_db, state_root, tmp_path, monkeypatch, operator_peer
):
    """The bypass never means UNKNOWN is treated as ABSENT."""
    channel._set_receipt_state_for_test(_recorded_state())
    psb.write_binding_for_project(
        psb.binding_path(ORDINARY), ORDINARY, str(tmp_path / "does-not-exist")
    )
    created = _install_create(monkeypatch)

    outcome = await _run(ORDINARY)

    assert outcome.ok is False
    assert outcome.reason_code == authority.REASON_RECOVERY_HIGH_WATER_UNAVAILABLE
    assert outcome.detail == "existing-run-unknown"
    assert outcome.terminal_created is False
    assert created == {}, "a phase-A refusal precedes creation"


@pytest.mark.asyncio
async def test_interval_three_present_witness_refuses(
    isolated_memory_db, state_root, tmp_path, monkeypatch, operator_peer
):
    channel._set_receipt_state_for_test(_recorded_state())
    project_dir = tmp_path / "conductor-ordinary"
    project_dir.mkdir()
    (project_dir / psb.PROJECT_JSON_BASENAME).write_bytes(b'{"project_incarnation": 7}')
    psb.write_binding_for_project(psb.binding_path(ORDINARY), ORDINARY, str(project_dir))
    _install_create(monkeypatch)

    outcome = await _run(ORDINARY)
    assert outcome.reason_code == authority.REASON_RECOVERY_HIGH_WATER_UNAVAILABLE
    assert outcome.detail == "existing-run-present"


# --------------------------------------------------------------------------
# The designated proof project runs the pipeline in interval i — that is how
# gate 2 gets proven at all.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_designated_proof_project_runs_the_pipeline_while_gate_two_is_open(
    isolated_memory_db, state_root, monkeypatch, operator_peer
):
    channel._set_designation_for_test(
        Gate2ProofDesignation(project=PROOF_PROJECT, sha256="0" * 64, path="/tmp/d.json")
    )
    _install_create(monkeypatch, session=PROOF_PROJECT)

    outcome = await _run(PROOF_PROJECT)
    assert outcome.authority_granted is True


@pytest.mark.asyncio
async def test_designation_does_not_admit_a_different_project(
    isolated_memory_db, state_root, monkeypatch, operator_peer
):
    channel._set_designation_for_test(
        Gate2ProofDesignation(project=PROOF_PROJECT, sha256="0" * 64, path="/tmp/d.json")
    )
    _install_create(monkeypatch)
    outcome = await _run(ORDINARY)
    assert outcome.detail == channel.DETAIL_G10_UNPROVEN
    assert outcome.authority_granted is False


# --------------------------------------------------------------------------
# The honor window closes at server start, not per request.
# --------------------------------------------------------------------------


def test_designation_after_recorded_receipt_refuses_server_start(state_root, tmp_path):
    """The proof run is over; a leftover designation is stale, not honored."""
    from cli_agent_orchestrator.services import gate2_proof_designation as gd

    gd.write_designation_for_proof_run(state_root / gd.DESIGNATION_BASENAME, PROOF_PROJECT)
    rs.write_receipt_state_for_proof_run(state_root / rs.RECEIPT_STATE_BASENAME, "a" * 64)

    with pytest.raises(channel.SupervisorCreateChannelError) as excinfo:
        channel.load_designation_at_start()
    message = str(excinfo.value)
    assert PROOF_PROJECT in message
    assert "already records" in message


def test_designation_alone_starts_cleanly(state_root):
    from cli_agent_orchestrator.services import gate2_proof_designation as gd

    gd.write_designation_for_proof_run(state_root / gd.DESIGNATION_BASENAME, PROOF_PROJECT)
    channel.load_designation_at_start()
    assert channel._designation().project == PROOF_PROJECT


def test_receipt_state_alone_starts_cleanly(state_root):
    rs.write_receipt_state_for_proof_run(state_root / rs.RECEIPT_STATE_BASENAME, "a" * 64)
    channel.load_designation_at_start()
    assert channel._designation() is None
    assert channel._gate2_receipt_recorded() is True


def test_neither_artifact_starts_cleanly_and_keeps_interval_one(state_root):
    channel.load_designation_at_start()
    assert channel._designation() is None
    assert channel._gate2_receipt_recorded() is False


def test_malformed_receipt_state_refuses_server_start(state_root):
    """Server-scoped blast radius, matching the designation."""
    import os

    path = state_root / rs.RECEIPT_STATE_BASENAME
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, json.dumps({"schema": "wrong"}).encode())
    finally:
        os.close(descriptor)

    with pytest.raises(rs.ReceiptStateError):
        channel.load_designation_at_start()


def test_malformed_binding_does_not_refuse_server_start(state_root):
    """The deliberate asymmetry: per-project, so start survives."""
    import os

    psb.bindings_dir().mkdir(parents=True, exist_ok=True)
    path = psb.binding_path(ORDINARY)
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, b"not json")
    finally:
        os.close(descriptor)

    channel.load_designation_at_start()  # must not raise
    assert psb.observe_witness(ORDINARY).witness == "unknown"


# --------------------------------------------------------------------------
# Interval choice is never request-derived.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "witness",
        "existing_run_witness",
        "project_json_sha256",
        "project_state_dir",
        "project_json_observed_absent",
    ],
)
def test_a_request_carrying_a_witness_is_refused_not_ignored(field):
    payload = {
        "verb": channel.VERB_SUPERVISOR_TERMINAL_CREATE,
        "args": {"agent_profile": "code_supervisor", field: "absent"},
    }
    with pytest.raises(channel.SupervisorCreateChannelError) as excinfo:
        channel.validate_request(payload)
    assert field in str(excinfo.value)


def test_every_witness_field_is_in_set_b():
    for field in (
        "witness",
        "existing_run_witness",
        "project_json_sha256",
        "project_state_dir",
        "project_json_observed_absent",
    ):
        assert field in channel.SET_B_REFUSED_FIELDS
        assert field not in channel.SET_A_FIELDS


def test_interval_selection_reads_only_start_state(monkeypatch, state_root):
    """Flipping the file after start must not change the interval in flight."""
    channel.load_designation_at_start()
    assert channel._gate2_receipt_recorded() is False

    rs.write_receipt_state_for_proof_run(state_root / rs.RECEIPT_STATE_BASENAME, "a" * 64)
    # No re-read happens without another start.
    assert channel._gate2_receipt_recorded() is False

    channel.load_designation_at_start()
    assert channel._gate2_receipt_recorded() is True


# --------------------------------------------------------------------------
# Provenance reaches the audit record and the authority row.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authority_row_records_the_witness_basis(
    isolated_memory_db, state_root, tmp_path, monkeypatch, operator_peer
):
    """An auditor must see on what basis (1, 1) was minted."""
    channel._set_receipt_state_for_test(_recorded_state())
    project_dir = tmp_path / "conductor-ordinary"
    project_dir.mkdir()
    psb.write_binding_for_project(psb.binding_path(ORDINARY), ORDINARY, str(project_dir))
    _install_create(monkeypatch)

    await _run(ORDINARY)

    with database.SessionLocal() as db:
        row = db.query(database.ProjectSupervisorAuthorityModel).one()
        provenance = json.loads(row.witness_provenance_json)
    assert provenance["witness"] == "absent"
    assert provenance["witness_project_json_observed_absent"] is True
    assert provenance["witness_source_path"].endswith("project.json")
    assert provenance["witness_detail"] == "project-json-observed-absent"


def test_decision_carries_provenance_even_when_refusing(isolated_memory_db):
    got = authority.decide(
        authority.SupervisorTuple("p", "t", "g"),
        witness=authority.ExistingRunWitness.UNKNOWN,
        witness_provenance={"witness": "unknown", "witness_detail": "binding-absent"},
    )
    assert got.decision is authority.Decision.REFUSE
    assert got.witness_provenance == {"witness": "unknown", "witness_detail": "binding-absent"}


@pytest.mark.asyncio
async def test_proof_project_provenance_names_the_designation(
    isolated_memory_db, state_root, monkeypatch, operator_peer
):
    """The proof project's ABSENT is designation-derived, and says so."""
    channel._set_designation_for_test(
        Gate2ProofDesignation(project=PROOF_PROJECT, sha256="0" * 64, path="/tmp/d.json")
    )
    _install_create(monkeypatch, session=PROOF_PROJECT)

    await _run(PROOF_PROJECT)

    with database.SessionLocal() as db:
        row = db.query(database.ProjectSupervisorAuthorityModel).one()
        provenance = json.loads(row.witness_provenance_json)
    assert provenance["witness"] == "absent"
    assert provenance["witness_detail"] == "designated-proof-project"
    assert provenance["witness_source_path"] == "/tmp/d.json"
