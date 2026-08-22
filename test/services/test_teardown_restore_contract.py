"""Teardown-time restore-contract publication (cond-0485).

Production teardown publishes the immutable restore contract for a managed
incarnation whose launch facts are durably recorded (the managed-launch
reservations), then retires through the roster's dormant transition — the
producer every resurrection gate reads.  Every failure at that seam
degrades to the ordinary contract-free retirement; teardown is never
blocked, and an agent retired before this seam existed stays contract-free.

No provider, tmux, or network I/O: the roster bind, reservation rows, and
terminal rows are written directly against an isolated database.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.services import cohort_journal, cohort_resume
from cli_agent_orchestrator.services import restore_contract as rc
from cli_agent_orchestrator.services import stable_agent_roster as roster
from cli_agent_orchestrator.services import supervisor_worker_ops
from cli_agent_orchestrator.services import terminal_service as ts

_TERMINAL_ID = "a1b2c3d4"
_GENERATION = "00000000-0000-4000-8000-000000000001"
_NATIVE_ID = "11111111-2222-4333-8444-555555555555"
_SESSION = "cao-campaign-a"


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _bind_worker(**changes):
    """Bind a roster worker/lineage/incarnation; returns the bind dict."""
    payload = {
        "agent_id": str(uuid.uuid4()),
        "session_name": _SESSION,
        "role": roster.ROLE_WORKER,
        "profile_family": "developer",
        "harness": "claude_code",
        "native_session_id": _NATIVE_ID,
        "acquisition_method": "chosen_session_id",
        "route_provenance": {"provider_route": "anthropic"},
        "terminal_id": _TERMINAL_ID,
        "generation": _GENERATION,
        "pane_id": "%101",
        "pane_pid": 4242,
        "process_identity": {"pid": 4242, "start_marker": "2026-08-09T00:00:00Z"},
        "execution_mode": "native_tui",
    }
    payload.update(changes)
    return roster.bind_generation(roster.BindingContract(**payload))


def _terminal_row():
    return database.create_terminal(
        _TERMINAL_ID,
        _SESSION,
        "worker-1",
        "claude_code",
        agent_profile="developer",
        generation=_GENERATION,
    )


def _reserve_v1(working_directory: str, trusted_project_root: str | None = None, **changes):
    payload = {
        "reservation_id": str(uuid.uuid4()),
        "terminal_id": _TERMINAL_ID,
        "generation": _GENERATION,
        "session_name": _SESSION,
        "provider": "claude_code",
        "agent_profile": "developer",
        "caller_id": "operator",
        "working_directory": working_directory,
        "trusted_project_root": trusted_project_root,
        "state": "ready",
        "request_json": "{}",
        "observations_json": "[]",
        "created_at": _stamp(),
        "updated_at": _stamp(),
    }
    payload.update(changes)
    with database.SessionLocal() as session:
        session.add(database.ManagedLaunchReservationModel(**payload))
        session.commit()


def _reserve_v2(working_directory: str, trusted_project_root: str | None = None, **changes):
    payload = {
        "reservation_id": str(uuid.uuid4()),
        "terminal_id": _TERMINAL_ID,
        "generation": _GENERATION,
        "session_name": _SESSION,
        "provider": "claude_code",
        "agent_profile": "developer",
        "caller_id": "operator",
        "working_directory": working_directory,
        "trusted_project_root": trusted_project_root,
        "obligation_generation": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "launch_nonce_digest": "a" * 64,
        "state": "ready",
        "request_json": "{}",
        "created_at": _stamp(),
        "updated_at": _stamp(),
    }
    payload.update(changes)
    with database.SessionLocal() as session:
        session.add(database.ManagedLaunchV2ReservationModel(**payload))
        session.commit()


@pytest.fixture
def workdir(tmp_path):
    """A canonical real directory (no symlink aliasing) for the contract's
    required working-directory fact."""
    path = os.path.realpath(str(tmp_path / "worktree"))
    os.mkdir(path)
    return path


def test_teardown_publishes_contract_and_transitions_dormant(isolated_memory_db, workdir):
    """A managed incarnation's teardown publishes the immutable contract and
    retires through the dormant transition, leaving the stable agent
    exactly resumable at the roster gates."""
    bind = _bind_worker()
    _terminal_row()
    _reserve_v1(workdir, trusted_project_root=workdir)

    ts._roster_retire_incarnation_best_effort(_TERMINAL_ID, _GENERATION)

    stored = rc.get_contract_by_incarnation(_TERMINAL_ID, _GENERATION)
    assert stored is not None
    assert stored["agent_id"] == bind["agent"]["agent_id"]
    assert stored["native_session_id"] == _NATIVE_ID
    payload = stored["contract"]
    assert payload["working_directory"] == workdir
    assert payload["trusted_project_root"] == workdir
    assert payload["provider"] == "claude_code"
    assert payload["route_provenance"] == {"provider_route": "anthropic"}
    assert payload["execution_mode"] == "native_tui"
    # Facts teardown cannot truthfully supply are typed unavailable, never
    # inferred.
    for field in ("model", "effort", "executable", "profile_material", "provider_home_facts"):
        assert payload[field]["state"] == rc.FACT_UNAVAILABLE
        assert payload[field]["reason"]

    agent = roster.get_agent(stored["agent_id"])
    assert agent["disposition"] == roster.DISPOSITION_DORMANT
    assert agent["current_incarnation"]["disposition"] == roster.INCARNATION_RETIRED
    assert agent["current_incarnation"]["retirement_reason"] == "terminal teardown"
    # The roster gate that was False for every real agent now holds.
    assert supervisor_worker_ops.has_exact_resume_identity(agent) is True


def test_teardown_publishes_contract_from_a_v2_reservation(isolated_memory_db, workdir):
    """The v2 reservation surface is an equally authoritative launch-facts
    source (it is the surface fleet campaign workers launch through)."""
    _bind_worker()
    _terminal_row()
    _reserve_v2(workdir)

    ts._roster_retire_incarnation_best_effort(_TERMINAL_ID, _GENERATION)

    stored = rc.get_contract_by_incarnation(_TERMINAL_ID, _GENERATION)
    assert stored is not None
    assert stored["contract"]["working_directory"] == workdir
    agent = roster.get_agent(stored["agent_id"])
    assert agent["disposition"] == roster.DISPOSITION_DORMANT


def test_cohort_boundary_observation_carries_the_teardown_contract(isolated_memory_db, workdir):
    """The Resume-time cohort boundary is re-observed live from the roster:
    once teardown has published, the member snapshot carries the resume
    identity ``cohort_resume`` hard-requires instead of reading unresumable."""
    _bind_worker()
    _terminal_row()
    _reserve_v2(workdir)

    ts._roster_retire_incarnation_best_effort(_TERMINAL_ID, _GENERATION)

    boundary = cohort_journal.observe_boundary(_SESSION)
    member = next(m for m in boundary["members"] if m["terminal_id"] == _TERMINAL_ID)
    stored = rc.get_contract_by_incarnation(_TERMINAL_ID, _GENERATION)
    assert member["restore_contract_id"] == stored["contract_id"]
    assert member["restore_contract_digest"] == stored["contract_digest"]
    assert cohort_resume._has_resume_identity(member) is True


def test_teardown_without_durable_launch_facts_degrades_to_plain_retire(isolated_memory_db):
    """An unmanaged incarnation (no reservation) has no durably recorded
    working directory, so nothing is published — the retirement is exactly
    today's contract-free path and the agent stays unresurrectable."""
    bind = _bind_worker()
    _terminal_row()

    ts._roster_retire_incarnation_best_effort(_TERMINAL_ID, _GENERATION)

    assert rc.get_contract_by_incarnation(_TERMINAL_ID, _GENERATION) is None
    agent = roster.get_agent(bind["agent"]["agent_id"])
    assert agent["disposition"] == roster.DISPOSITION_DORMANT
    assert agent["current_incarnation"]["disposition"] == roster.INCARNATION_RETIRED
    assert supervisor_worker_ops.has_exact_resume_identity(agent) is False


def test_failed_publish_never_blocks_teardown(isolated_memory_db, workdir, monkeypatch):
    """A refused/failed publish degrades to the plain retirement: teardown
    is never blocked and the incarnation is still retired."""
    bind = _bind_worker()
    _terminal_row()
    _reserve_v1(workdir)

    def _boom(contract, db=None):
        raise rc.RestoreContractUnavailable("store down")

    monkeypatch.setattr(rc, "publish_contract", _boom)

    ts._roster_retire_incarnation_best_effort(_TERMINAL_ID, _GENERATION)

    assert rc.get_contract_by_incarnation(_TERMINAL_ID, _GENERATION) is None
    agent = roster.get_agent(bind["agent"]["agent_id"])
    assert agent["disposition"] == roster.DISPOSITION_DORMANT
    assert agent["current_incarnation"]["disposition"] == roster.INCARNATION_RETIRED


def test_failed_transition_never_blocks_teardown(isolated_memory_db, workdir, monkeypatch):
    """A transition_dormant failure after a successful publish degrades to
    the plain retirement: teardown is never blocked, and the contract that
    was already published survives for a later exact resume."""
    bind = _bind_worker()
    _terminal_row()
    _reserve_v1(workdir)

    def _boom(*args, **kwargs):
        raise RuntimeError("roster store down")

    monkeypatch.setattr(roster, "transition_dormant", _boom)

    ts._roster_retire_incarnation_best_effort(_TERMINAL_ID, _GENERATION)

    assert rc.get_contract_by_incarnation(_TERMINAL_ID, _GENERATION) is not None
    agent = roster.get_agent(bind["agent"]["agent_id"])
    assert agent["current_incarnation"]["disposition"] == roster.INCARNATION_RETIRED


def test_incarnation_retired_before_this_seam_stays_contract_free(isolated_memory_db, workdir):
    """An agent retired before the publish seam existed has no contract and
    never gains one through a later teardown replay: it degrades to today's
    behaviour (unresumable), not to an error or a retroactive record."""
    bind = _bind_worker()
    _terminal_row()
    _reserve_v1(workdir)
    roster.retire_incarnation(
        terminal_id=_TERMINAL_ID, generation=_GENERATION, reason="pre-seam retirement"
    )

    ts._roster_retire_incarnation_best_effort(_TERMINAL_ID, _GENERATION)

    assert rc.get_contract_by_incarnation(_TERMINAL_ID, _GENERATION) is None
    agent = roster.get_agent(bind["agent"]["agent_id"])
    assert agent["disposition"] == roster.DISPOSITION_DORMANT
    assert agent["current_incarnation"]["retirement_reason"] == "pre-seam retirement"
    assert supervisor_worker_ops.has_exact_resume_identity(agent) is False
