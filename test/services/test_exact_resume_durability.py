"""P0-A launch-time durability for exact resume (cond-0573).

The P0 rescue chain was self-blocking: the only restore contract the system
writes (cond-0485's teardown seam) recorded ``model`` / ``effort`` /
``executable`` as typed ``unavailable`` because the pane that could report
them is dead by teardown, so ``exact_executor._fact_refusal`` refused every
resurrection contract that existed.  This lane makes those three facts
durable at MANAGED LAUNCH ADMISSION (the reservation rows) and consumes them
at teardown, so an exact resume of a teardown-published worker passes the gate
that refused before — without weakening any gate.

No provider, tmux, or network I/O: the roster bind, reservation rows, and
terminal rows are written directly against an isolated database, and the
executable fact is a real digest-pinned file (the resume gate re-hashes the
bytes at the recorded path, so the fact must be a genuine filesystem fact).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from datetime import datetime, timezone

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.managed_launch import (
    PROTOCOL_VERSION,
    ManagedLaunchReserveRequest,
)
from cli_agent_orchestrator.models.managed_launch_v2 import (
    PROTOCOL_VERSION_V2,
    ManagedLaunchV2ReserveRequest,
)
from cli_agent_orchestrator.services import exact_executor as ee
from cli_agent_orchestrator.services import managed_launch as ml
from cli_agent_orchestrator.services import managed_launch_v2 as v2
from cli_agent_orchestrator.services import restore_contract as rc
from cli_agent_orchestrator.services import stable_agent_roster as roster
from cli_agent_orchestrator.services import terminal_service as ts

_NATIVE_ID = "11111111-2222-4333-8444-555555555555"
_SESSION = "cao-campaign-a"
_MODEL = "claude-sonnet-4-5"
_EFFORT = "high"


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.fixture
def launch_root(tmp_path):
    """A real canonical working directory and a real digest-pinned binary.

    The executor's ``_fact_refusal`` gate requires the contract's recorded
    executable path to exist and to re-hash to the recorded digest, so the
    stored executable fact must be a genuine filesystem fact.
    """
    workdir = os.path.realpath(str(tmp_path / "work"))
    os.makedirs(workdir, exist_ok=True)
    binary = os.path.realpath(str(tmp_path / "bin" / "claude"))
    os.makedirs(os.path.dirname(binary), exist_ok=True)
    with open(binary, "wb") as handle:
        handle.write(b"#!/bin/sh\nsleep 60\n")
    os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    digest = hashlib.sha256(open(binary, "rb").read()).hexdigest()
    return {"workdir": workdir, "binary": binary, "binary_sha256": digest}


def _v2_reserve_request(launch_root, **changes):
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "reservation_id": str(uuid.uuid4()),
        "session_name": _SESSION,
        "provider": "claude_code",
        "agent_profile": "developer",
        "caller_id": "deadbeef",
        "working_directory": launch_root["workdir"],
        "trusted_project_root": None,
        "expected_model": _MODEL,
        "expected_effort": _EFFORT,
        "provider_executable": launch_root["binary"],
        "provider_executable_sha256": launch_root["binary_sha256"],
        "obligation_generation": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "delivery_id": str(uuid.uuid4()),
        "launch_nonce": "n" * 40,
        "execution_mode": "native_tui",
    }
    payload.update(changes)
    return ManagedLaunchV2ReserveRequest(**payload)


def _v1_reserve_request(launch_root, **changes):
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reservation_id": str(uuid.uuid4()),
        "session_name": _SESSION,
        "provider": "claude_code",
        "agent_profile": "developer",
        "caller_id": "deadbeef",
        "project": "p0a-project",
        "task_id": "p0a-task",
        "delivery_id": str(uuid.uuid4()),
        "working_directory": launch_root["workdir"],
        "trusted_project_root": None,
        "expected_model": _MODEL,
        "expected_effort": _EFFORT,
        "provider_executable": launch_root["binary"],
        "provider_executable_sha256": launch_root["binary_sha256"],
        "provider_route": "anthropic",
    }
    payload.update(changes)
    return ManagedLaunchReserveRequest(**payload)


def _bind_worker(terminal_id, generation, **changes):
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
        "terminal_id": terminal_id,
        "generation": generation,
        "pane_id": "%101",
        "pane_pid": 4242,
        "process_identity": {"pid": 4242, "start_marker": "2026-08-09T00:00:00Z"},
        "execution_mode": "native_tui",
    }
    payload.update(changes)
    return roster.bind_generation(roster.BindingContract(**payload))


def _terminal_row(terminal_id, generation):
    return database.create_terminal(
        terminal_id,
        _SESSION,
        "worker-1",
        "claude_code",
        agent_profile="developer",
        generation=generation,
    )


def _stored_contract_payload(terminal_id, generation):
    stored = rc.get_contract_by_incarnation(terminal_id, generation)
    assert stored is not None
    return stored


# ---------------------------------------------------------------------------
# 1. Managed admission persists model/effort/executable durably
# ---------------------------------------------------------------------------


def test_v2_admission_persists_launch_facts_durably(isolated_memory_db, launch_root):
    """The v2 reserve/create seam records the resolved model, effort, and
    executable path + sha256 digest on the reservation row."""
    record, created = v2.reserve(_v2_reserve_request(launch_root))
    assert created is True
    # The public read surfaces the durable facts as a first-class field.
    facts = v2.get(record["reservation_id"])["launch_facts"]
    assert facts == {
        "model": _MODEL,
        "effort": _EFFORT,
        "provider_executable": launch_root["binary"],
        "provider_executable_sha256": launch_root["binary_sha256"],
    }
    # The same facts round-trip through the stored row bytes.
    with database.SessionLocal() as session:
        row = (
            session.query(database.ManagedLaunchV2ReservationModel)
            .filter(
                database.ManagedLaunchV2ReservationModel.reservation_id == record["reservation_id"]
            )
            .one()
        )
        assert json.loads(row.launch_facts_json) == facts


def test_v1_admission_persists_launch_facts_durably(isolated_memory_db, launch_root):
    """The v1 reserve seam records the same durable facts."""
    record, created = ml.reserve(_v1_reserve_request(launch_root))
    assert created is True
    facts = ml.get(record["reservation_id"])["launch_facts"]
    assert facts == {
        "model": _MODEL,
        "effort": _EFFORT,
        "provider_executable": launch_root["binary"],
        "provider_executable_sha256": launch_root["binary_sha256"],
    }


# ---------------------------------------------------------------------------
# 2. Teardown publishes present facts (digest matches admission) and the
#    exact-resume gate that refused before now passes
# ---------------------------------------------------------------------------


def test_teardown_publishes_present_facts_and_resume_gate_passes(isolated_memory_db, launch_root):
    """A v2 managed incarnation admitted with durable facts publishes a
    contract whose model/effort/executable are PRESENT (not unavailable) and
    whose executable digest matches the admission, and the exact-resume gate
    that refused every teardown contract before now passes it."""
    record, _created = v2.reserve(_v2_reserve_request(launch_root))
    terminal_id = record["terminal_id"]
    generation = record["generation"]
    _bind_worker(terminal_id, generation)
    _terminal_row(terminal_id, generation)

    ts._roster_retire_incarnation_best_effort(terminal_id, generation)

    stored = _stored_contract_payload(terminal_id, generation)
    payload = stored["contract"]
    # The facts teardown used to type unavailable are now present and carry
    # the exact values recorded at admission.
    assert payload["model"] == {"state": rc.FACT_PRESENT, "value": _MODEL, "reason": None}
    assert payload["effort"] == {"state": rc.FACT_PRESENT, "value": _EFFORT, "reason": None}
    assert payload["executable"]["state"] == rc.FACT_PRESENT
    assert payload["executable"]["value"] == {
        "path": launch_root["binary"],
        "sha256": launch_root["binary_sha256"],
    }
    # The facts that are still not durably recorded stay unavailable — only
    # the launch-admission facts gained a producer.
    for field in ("profile_material", "provider_home_facts"):
        assert payload[field]["state"] == rc.FACT_UNAVAILABLE

    # The exact-resume gate that refused before now passes the published
    # contract: decoded through the same constructor the executor uses.
    contract = rc.decode_stored_contract(payload)
    assert contract is not None
    assert ee._fact_refusal(contract) is None


def test_teardown_facts_are_present_for_a_v1_reservation(isolated_memory_db, launch_root):
    """The v1 reservation surface publishes present facts too."""
    record, _created = ml.reserve(_v1_reserve_request(launch_root))
    terminal_id = record["terminal_id"]
    generation = record["generation"]
    _bind_worker(terminal_id, generation)
    _terminal_row(terminal_id, generation)

    ts._roster_retire_incarnation_best_effort(terminal_id, generation)

    stored = _stored_contract_payload(terminal_id, generation)
    payload = stored["contract"]
    assert payload["model"]["state"] == rc.FACT_PRESENT
    assert payload["effort"]["state"] == rc.FACT_PRESENT
    assert payload["executable"]["state"] == rc.FACT_PRESENT
    assert payload["executable"]["value"]["sha256"] == launch_root["binary_sha256"]
    contract = rc.decode_stored_contract(payload)
    assert contract is not None
    assert ee._fact_refusal(contract) is None


# ---------------------------------------------------------------------------
# 3. In-flight / pre-migration rows (facts absent) still degrade honestly
# ---------------------------------------------------------------------------


def test_pre_migration_row_degrades_to_unavailable_and_gate_refuses(
    isolated_memory_db, launch_root
):
    """A reservation row that predates this change (``launch_facts_json``
    null) publishes unavailable facts exactly as before, and the exact-resume
    gate still refuses it — no inferred facts."""
    terminal_id = "a1b2c3d4"
    generation = "00000000-0000-4000-8000-000000000001"
    _bind_worker(terminal_id, generation)
    _terminal_row(terminal_id, generation)
    _reserve_v2_without_facts(terminal_id, generation, working_directory=launch_root["workdir"])

    ts._roster_retire_incarnation_best_effort(terminal_id, generation)

    stored = _stored_contract_payload(terminal_id, generation)
    payload = stored["contract"]
    for field in ("model", "effort", "executable"):
        assert payload[field]["state"] == rc.FACT_UNAVAILABLE
        assert payload[field]["reason"]
    # The gate still refuses the unavailable executable fact honestly.
    contract = rc.decode_stored_contract(payload)
    assert contract is not None
    refusal = ee._fact_refusal(contract)
    assert refusal is not None
    assert "executable" in refusal


def _reserve_v2_without_facts(terminal_id, generation, working_directory):
    with database.SessionLocal() as session:
        session.add(
            database.ManagedLaunchV2ReservationModel(
                reservation_id=str(uuid.uuid4()),
                terminal_id=terminal_id,
                generation=generation,
                protocol_vintage="v2",
                session_name=_SESSION,
                provider="claude_code",
                agent_profile="developer",
                caller_id="deadbeef",
                working_directory=working_directory,
                trusted_project_root=None,
                obligation_generation=str(uuid.uuid4()),
                run_id=str(uuid.uuid4()),
                launch_nonce_digest="a" * 64,
                state="ready",
                request_json="{}",
                created_at=_stamp(),
                updated_at=_stamp(),
            )
        )
        session.commit()


# ---------------------------------------------------------------------------
# 4. Override variations still require a compatibility cell (Option C remains)
# ---------------------------------------------------------------------------


def _present_facts_contract(launch_root):
    return rc.RestoreContract(
        agent_id=str(uuid.uuid4()),
        lineage_id=str(uuid.uuid4()),
        terminal_id="a1b2c3d4",
        generation="00000000-0000-4000-8000-000000000001",
        native_session_id=_NATIVE_ID,
        harness="claude_code",
        provider="claude_code",
        route_provenance={"provider_route": "anthropic"},
        execution_mode="native_tui",
        model=rc.ContractFact.present(_MODEL),
        effort=rc.ContractFact.present(_EFFORT),
        working_directory=launch_root["workdir"],
        trusted_project_root=None,
        executable=rc.ContractFact.present(
            {"path": launch_root["binary"], "sha256": launch_root["binary_sha256"]}
        ),
        profile_material=rc.ContractFact.unavailable("no profile material at this seam"),
        provider_home_facts=rc.ContractFact.unavailable("no provider-home facts at this seam"),
    )


def test_exact_resume_request_against_present_facts_needs_no_cell(isolated_memory_db, launch_root):
    """An exact resume request (facts equal to the present contract) produces
    no variations, so the compatibility-cell gate passes without naming a
    cell — the full fact-present path the P0 contract now enables."""
    contract = _present_facts_contract(launch_root)
    request = _operation_request(contract, model_requested=_MODEL)
    assert ee._variations(request, contract, ee.LaunchMaterial()) == []
    assert ee._cell_refusal(request, []) is None


def test_override_variation_still_requires_a_compatibility_cell(isolated_memory_db, launch_root):
    """A resume that varies the recorded contract (a different model) still
    needs the operation to name an exact compatibility cell — this lane does
    not claim to fix overrides."""
    contract = _present_facts_contract(launch_root)
    request = _operation_request(contract, model_requested="claude-opus-4-5")
    variations = ee._variations(request, contract, ee.LaunchMaterial())
    assert any("model" in variation for variation in variations)
    # Without a named cell, the variation is typed-disabled, never inferred.
    assert ee._cell_refusal(request, variations) is not None


def _operation_request(contract, model_requested):
    from cli_agent_orchestrator.services import operation_journal as oj
    from cli_agent_orchestrator.services import session_lifecycle as sl

    return oj.OperationRequest(
        operation_id=str(uuid.uuid4()),
        session_name=_SESSION,
        agent_id=contract.agent_id,
        roster_revision=0,
        role="worker",
        profile_family="developer",
        lineage_id=contract.lineage_id,
        harness=contract.harness,
        native_session_id=contract.native_session_id or "",
        prior_terminal_id=contract.terminal_id,
        prior_generation=contract.generation,
        prior_incarnation_id=str(uuid.uuid4()),
        lifecycle_epoch=0,
        lifecycle_observation=sl.WORKING,
        restore_contract_id="restore-contract-placeholder",
        restore_contract_digest=contract.digest(),
        restore_contract_schema=rc.SCHEMA_VERSION,
        route_provider="claude_code",
        model_requested=model_requested,
        effort_requested=_EFFORT,
        execution_mode_requested="native_tui",
    )
