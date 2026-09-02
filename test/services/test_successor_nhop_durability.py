"""N-hop exact-resume durability for successors (cond-0573 P0-A follow-up 3).

Exact-resume resurrection worked for ONE hop: a managed original pinned its
launch facts on its reservation row, but the exact executor's successor was a
native attachment with no reservation row, so the successor's own teardown
could never publish a restore contract and a second-generation resume died.
This lane records the successor's launch facts durably on the reincarnation
operation row that reserved its terminal id + generation — sourced from the
restore-contract facts the executor verified at launch — and makes the
teardown seam read that source after the managed reservations, so an
N-hop exact-resume chain publishes a complete restore contract at every hop.

What is proven here:

- **N-hop positive**: managed original (full facts, version banner) ->
  retire -> exact-resume successor A -> retire A -> A's teardown publishes
  COMPLETE facts -> exact-resume successor B accepted, and B's launch
  consumes A's facts.  Two hops past the original, indefinitely repeatable.
- **Back-compat negative**: a journal row whose stored facts record absence
  (constructed directly — the executor's fact gate refuses a pre-P0-A source
  contract before any successor is created, so this state is reader-side
  degradation only) publishes teardown typed-unavailable IDENTICAL to today's
  managed pre-P0-A output — the exact refusal strings are pinned, and the
  next hop stays refused fail-closed with no crash and no partial acceptance.
- **One-hop unchanged**: a managed reservation row always wins over the
  operation-journal source, so every existing managed teardown behaves
  identically; order of checks preserved.

No provider, tmux, or network I/O: the roster bind, reservation rows,
operation rows, and terminal rows are written directly against an isolated
database, and the executable fact is a real digest-pinned file (the resume
gate re-hashes the bytes at the recorded path).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.managed_launch_v2 import (
    PROTOCOL_VERSION_V2,
    ManagedLaunchV2ReserveRequest,
)
from cli_agent_orchestrator.services import exact_executor as ee
from cli_agent_orchestrator.services import managed_launch_v2 as v2
from cli_agent_orchestrator.services import native_attachment as na
from cli_agent_orchestrator.services import native_tui_launch
from cli_agent_orchestrator.services import operation_journal as oj
from cli_agent_orchestrator.services import restore_contract as rc
from cli_agent_orchestrator.services import session_lifecycle as sl
from cli_agent_orchestrator.services import stable_agent_roster as roster
from cli_agent_orchestrator.services import terminal_service as ts

_NATIVE_ID = "11111111-2222-4333-8444-555555555555"
_SESSION = "cao-campaign-a"
_MODEL = "claude-sonnet-4-5"
_EFFORT = "high"
_VERSION = "muse-spark-1.3-contributor (full banner)"


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


@pytest.fixture
def launch_root(tmp_path):
    """A real canonical working directory and a real digest-pinned binary.

    The executor's ``_fact_refusal`` gate requires the contract's recorded
    executable path to exist and to re-hash to the recorded digest, so every
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
    profile = os.path.realpath(str(tmp_path / "profile" / "settings.json"))
    os.makedirs(os.path.dirname(profile), exist_ok=True)
    with open(profile, "wb") as handle:
        handle.write(b'{"permissions": {"allow": ["*"]}}\n')
    return {
        "workdir": workdir,
        "binary": binary,
        "binary_sha256": digest,
        "profile_path": profile,
        "profile_sha256": hashlib.sha256(open(profile, "rb").read()).hexdigest(),
    }


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


def _bind_worker(agent_id=None, terminal_id="a1b2c3d4", generation=None, **bind_changes):
    """Bind a roster worker/lineage/incarnation; returns the bind dict."""
    payload = {
        "agent_id": agent_id or str(uuid.uuid4()),
        "session_name": _SESSION,
        "role": roster.ROLE_WORKER,
        "profile_family": "developer",
        "harness": "claude_code",
        "native_session_id": _NATIVE_ID,
        "acquisition_method": "chosen_session_id",
        "route_provenance": {"provider_route": "anthropic"},
        "terminal_id": terminal_id,
        "generation": generation or "00000000-0000-4000-8000-000000000001",
        "pane_id": "%101",
        "pane_pid": 4242,
        "process_identity": {"pid": 4242, "start_marker": "2026-08-09T00:00:00Z"},
        "execution_mode": "native_tui",
    }
    payload.update(bind_changes)
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


def _prior_attachment(
    *,
    pid: Optional[int] = None,
    terminal_id: str = "a1b2c3d4",
    generation: str = "00000000-0000-4000-8000-000000000001",
    native_session_id: str = _NATIVE_ID,
    provider: str = "claude_code",
) -> dict:
    """A real attachment row owned by the exact prior incarnation (dead pid)."""
    owner = {
        "terminal_id": terminal_id,
        "generation": generation,
        "execution_mode": "native_tui",
    }
    na.declare(
        provider=provider,
        native_session_id=native_session_id,
        intent=na.acquire_intent(
            acquisition_method=na.ACQUISITION_CHOSEN_SESSION_ID,
            acquisition_receipt={"kind": "chosen", "session_id": native_session_id},
            admits_only_new_instructions=True,
            replays_task_bytes=False,
        ),
        **owner,
    )
    na.mark_starting(provider=provider, native_session_id=native_session_id, **owner)
    return na.mark_attached(
        provider=provider,
        native_session_id=native_session_id,
        process_identity=na.process_identity(
            pid=pid if pid is not None else _dead_pid(),
            start_marker="2026-08-09T00:00:00Z",
        ),
        **owner,
    )


@dataclass
class FakeTransport:
    """A deterministic ``NativePaneTransport`` over the real launch path."""

    workdir: str
    native_session_id: str = _NATIVE_ID
    argv: Optional[list] = None
    created: int = 0
    observed_argv: Optional[list] = None
    pane_pid: int = field(default_factory=_dead_pid)
    create_error: Optional[Exception] = None
    observe_error: Optional[Exception] = None

    def create_pane(self, *, argv) -> str:
        if self.create_error is not None:
            raise self.create_error
        self.created += 1
        self.argv = list(argv)
        return f"%{self.created + 41}"

    def observe(self):
        if self.observe_error is not None:
            raise self.observe_error
        if self.created == 0:
            return None
        argv = self.observed_argv if self.observed_argv is not None else list(self.argv or [])
        return {
            "pane_id": "%42",
            "pid": self.pane_pid,
            "start_marker": "2026-08-14T00:00:00Z",
            "argv": argv,
            "cwd": self.workdir,
        }

    def capture_render(self, pane_id: str) -> list[str]:
        return []


@dataclass
class ReapRecorder:
    """Records the exact-generation teardown the executor performs."""

    calls: list = field(default_factory=list)

    def __call__(self, terminal_id, registry=None, **kwargs):
        self.calls.append({"terminal_id": terminal_id, **kwargs})
        return True


def _operation_request(bind, contract, operation_id=None, **changes):
    agent = roster.get_agent(bind["agent"]["agent_id"])
    payload = {
        "operation_id": operation_id or str(uuid.uuid4()),
        "session_name": bind["agent"]["session_name"],
        "agent_id": bind["agent"]["agent_id"],
        "roster_revision": agent["revision"],
        "role": bind["agent"]["role"],
        "profile_family": bind["agent"]["profile_family"],
        "lineage_id": bind["lineage"]["lineage_id"],
        "harness": bind["lineage"]["harness"],
        "native_session_id": bind["lineage"]["native_session_id"],
        "prior_terminal_id": bind["incarnation"]["terminal_id"],
        "prior_generation": bind["incarnation"]["generation"],
        "prior_incarnation_id": bind["incarnation"]["incarnation_id"],
        "lifecycle_epoch": 0,
        "lifecycle_observation": sl.WORKING,
        "restore_contract_digest": contract.digest(),
        "restore_contract_schema": rc.SCHEMA_VERSION,
        "route_provider": "claude_code",
        "model_requested": _MODEL,
        "effort_requested": _EFFORT,
        "execution_mode_requested": "native_tui",
        "compatibility_cell_ref": "claude_code:anthropic:native_tui",
        "compatibility_cell_digest": "c" * 64,
    }
    if "restore_contract_id" not in changes:
        payload["restore_contract_id"] = rc.get_contract_by_incarnation(
            terminal_id=bind["incarnation"]["terminal_id"],
            generation=bind["incarnation"]["generation"],
        )["contract_id"]
    payload.update(changes)
    return oj.OperationRequest(**payload)


def _resume_material(launch_root):
    """The operator-selected profile material every exact resume carries.

    A teardown-published contract records profile/provider-home material as
    typed ``unavailable`` (the operator reserves that content), so the resume
    supplies the selected profile reference — exactly the production exact
    resume.  The profile file is a real digest-pinned file the executor
    re-hashes at launch.
    """
    return ee.LaunchMaterial(
        profile_material={
            "profile_config_path": launch_root["profile_path"],
            "profile_config_sha256": launch_root["profile_sha256"],
        }
    )


def _execute(request, transport, material=None):
    return asyncio.run(
        ee.execute(
            request,
            material=material if material is not None else ee.LaunchMaterial(),
            transport_factory=lambda: transport,
        )
    )


def _recorded_successor_facts(operation_id):
    with database.SessionLocal() as session:
        row = (
            session.query(database.ReincarnationOperationModel)
            .filter(database.ReincarnationOperationModel.operation_id == operation_id)
            .one()
        )
        return json.loads(row.successor_launch_facts_json)


def _contract_from(payload):
    return rc.decode_stored_contract(payload)


# ---------------------------------------------------------------------------
# 1. N-hop positive: managed original -> A -> B, complete facts every hop
# ---------------------------------------------------------------------------


def test_two_hops_past_a_managed_original_publish_complete_facts(
    isolated_memory_db, launch_root, monkeypatch
):
    """A managed original with full facts (including the version banner)
    publishes a complete restore contract at every successor hop: A's teardown
    publishes COMPLETE facts from the operation journal, and B's launch
    consumes A's facts and is accepted — two hops past the original."""
    monkeypatch.setattr(ts, "delete_terminal", ReapRecorder())
    material = _resume_material(launch_root)

    # -- hop 0: managed v2 original, full facts + version banner ----------
    record, _created = v2.reserve(_v2_reserve_request(launch_root))
    original_terminal = record["terminal_id"]
    original_generation = record["generation"]
    v2._record_launch_executable_version(record["reservation_id"], _VERSION)
    original_bind = _bind_worker(terminal_id=original_terminal, generation=original_generation)
    _terminal_row(original_terminal, original_generation)
    _prior_attachment(
        terminal_id=original_terminal, generation=original_generation, native_session_id=_NATIVE_ID
    )
    ts._roster_retire_incarnation_best_effort(original_terminal, original_generation)
    original_contract = rc.get_contract_by_incarnation(original_terminal, original_generation)
    assert original_contract is not None
    original_payload = original_contract["contract"]
    assert original_payload["executable"]["value"] == {
        "path": launch_root["binary"],
        "sha256": launch_root["binary_sha256"],
        "version": _VERSION,
    }
    agent_id = original_bind["agent"]["agent_id"]

    # -- hop 1: exact-resume successor A from the original ------------------
    a_request = _operation_request(original_bind, _contract_from(original_payload))
    a_result = _execute(a_request, FakeTransport(workdir=launch_root["workdir"]), material)
    assert a_result["outcome"] == ee.OUTCOME_ACCEPTED
    a_terminal = a_result["successor_terminal_id"]
    a_generation = a_result["successor_generation"]

    # A's operation journal recorded COMPLETE facts from the original's
    # contract — the model/effort/executable identity INCLUDING the banner.
    a_facts = _recorded_successor_facts(a_request.operation_id)
    assert a_facts == {
        "working_directory": launch_root["workdir"],
        "trusted_project_root": None,
        "model": _MODEL,
        "effort": _EFFORT,
        "provider_executable": launch_root["binary"],
        "provider_executable_sha256": launch_root["binary_sha256"],
        "provider_executable_version": _VERSION,
    }

    # -- retire A: its teardown publishes COMPLETE facts --------------------
    ts._roster_retire_incarnation_best_effort(a_terminal, a_generation)
    a_contract = rc.get_contract_by_incarnation(a_terminal, a_generation)
    assert a_contract is not None, "A's teardown must publish a restore contract"
    a_payload = a_contract["contract"]
    assert a_payload["model"] == {"state": rc.FACT_PRESENT, "value": _MODEL, "reason": None}
    assert a_payload["effort"] == {"state": rc.FACT_PRESENT, "value": _EFFORT, "reason": None}
    assert a_payload["executable"]["state"] == rc.FACT_PRESENT
    assert a_payload["executable"]["value"] == {
        "path": launch_root["binary"],
        "sha256": launch_root["binary_sha256"],
        "version": _VERSION,
    }
    agent = roster.get_agent(agent_id)
    assert agent["current_incarnation"]["disposition"] == roster.INCARNATION_RETIRED

    # -- hop 2: exact-resume successor B consumes A's facts and is accepted --
    bind_for_b = {
        "agent": agent,
        "lineage": agent["current_lineage"],
        "incarnation": agent["current_incarnation"],
    }
    b_request = _operation_request(bind_for_b, _contract_from(a_payload))
    # B's launch must consume A's facts: spy the exact launch seam and let it
    # proceed, so B is genuinely accepted on the strength of A's contract.
    start_calls: list = []
    original_start = native_tui_launch.start

    def _spy_start(**kwargs):
        start_calls.append(kwargs)
        return original_start(**kwargs)

    monkeypatch.setattr(native_tui_launch, "start", _spy_start)
    b_transport = FakeTransport(workdir=launch_root["workdir"])
    b_result = _execute(b_request, b_transport, material)
    assert b_result["outcome"] == ee.OUTCOME_ACCEPTED
    assert len(start_calls) == 1
    call = start_calls[0]
    assert call["binary"] == launch_root["binary"]
    assert call["binary_sha256"] == launch_root["binary_sha256"]
    assert call["provider_version"] == _VERSION
    assert call["working_directory"] == launch_root["workdir"]
    assert call["terminal_id"] == b_result["successor_terminal_id"]
    # B recorded the same complete facts its launch used — A's facts flowed
    # into B's own journal, so the chain extends indefinitely.
    b_facts = _recorded_successor_facts(b_request.operation_id)
    assert b_facts == a_facts


# ---------------------------------------------------------------------------
# 2. Back-compat negative: successor records absence -> identical refusal
# ---------------------------------------------------------------------------


def test_successor_recording_absence_publishes_identical_typed_unavailable(
    isolated_memory_db, launch_root
):
    """A journal row whose stored facts record absence — model/effort/
    executable all None, as a pre-P0-A-sourced successor WOULD record if one
    could exist — publishes a contract whose model/effort/executable are
    typed unavailable with the EXACT same reason strings a managed pre-P0-A
    reservation produces today, and the exact-resume gate refuses it
    fail-closed with no partial acceptance.  The row is constructed directly:
    the executor's fact gate refuses a contract lacking the executable fact
    before any successor is created, so this state is reader-side degradation
    only — and the teardown never invents facts for it."""
    # The successor terminal/generation the operation reserved.
    a_terminal = "cccc1111"
    a_generation = "00000000-0000-4000-8000-000000000099"
    agent_id = str(uuid.uuid4())

    # The successor's operation row, recording ABSENCE: working directory and
    # trusted root present, model/effort/executable absent.  Inserted directly
    # because the executor never reaches this state (its fact gate refuses a
    # pre-P0-A source contract before successor creation); the teardown's
    # handling of such a payload is the reader-side degradation pinned here.
    _insert_successor_operation(
        a_terminal,
        a_generation,
        {
            "working_directory": launch_root["workdir"],
            "trusted_project_root": None,
            "model": None,
            "effort": None,
            "provider_executable": None,
            "provider_executable_sha256": None,
            "provider_executable_version": None,
        },
    )
    _bind_worker(
        agent_id=agent_id,
        terminal_id=a_terminal,
        generation=a_generation,
        native_session_id=str(uuid.uuid4()),
    )

    ts._roster_retire_incarnation_best_effort(a_terminal, a_generation)

    stored = rc.get_contract_by_incarnation(a_terminal, a_generation)
    assert stored is not None
    payload = stored["contract"]
    assert payload["working_directory"] == launch_root["workdir"]

    # The exact typed-unavailable strings a managed pre-P0-A row publishes
    # today — the successor must produce byte-identical reasons.
    expected = {
        "model": {
            "state": rc.FACT_UNAVAILABLE,
            "value": None,
            "reason": "the observed/assigned model was not durably recorded at launch",
        },
        "effort": {
            "state": rc.FACT_UNAVAILABLE,
            "value": None,
            "reason": "the observed/assigned effort was not durably recorded at launch",
        },
        "executable": {
            "state": rc.FACT_UNAVAILABLE,
            "value": None,
            "reason": (
                "the exact resolved executable identity was not durably " "recorded at launch"
            ),
        },
    }
    for field, wanted in expected.items():
        assert payload[field] == wanted, field

    # The exact-resume gate refuses the successor's contract fail-closed with
    # the exact refusal string — no crash, no partial acceptance.
    contract = rc.decode_stored_contract(payload)
    assert contract is not None
    refusal = ee._fact_refusal(contract)
    assert refusal == (
        "the restore contract's executable identity is 'unavailable' "
        "(the exact resolved executable identity was not durably recorded at "
        "launch); an exact restore requires the present, digest-matched "
        "executable"
    )

    # And these are byte-identical to what a managed pre-P0-A reservation
    # (launch_facts_json NULL) publishes at its own teardown.
    legacy_terminal = "dddd2222"
    legacy_generation = "00000000-0000-4000-8000-000000000001"
    _insert_legacy_v2_reservation(legacy_terminal, legacy_generation, launch_root["workdir"])
    _bind_worker(
        agent_id=str(uuid.uuid4()),
        terminal_id=legacy_terminal,
        generation=legacy_generation,
        native_session_id=str(uuid.uuid4()),
    )
    _terminal_row(legacy_terminal, legacy_generation)
    ts._roster_retire_incarnation_best_effort(legacy_terminal, legacy_generation)
    legacy_payload = rc.get_contract_by_incarnation(legacy_terminal, legacy_generation)["contract"]
    for field in expected:
        assert legacy_payload[field] == expected[field], field


def test_successor_without_recorded_facts_degrades_to_contract_free(
    isolated_memory_db, launch_root
):
    """A successor whose operation row predates this lane (NULL launch facts)
    degrades exactly like today: no facts, no contract, contract-free
    retirement — never a fabricated path."""
    a_terminal = "eeee3333"
    a_generation = "00000000-0000-4000-8000-000000000002"
    agent_id = str(uuid.uuid4())
    _insert_successor_operation(a_terminal, a_generation, None)
    bind = _bind_worker(agent_id=agent_id, terminal_id=a_terminal, generation=a_generation)

    ts._roster_retire_incarnation_best_effort(a_terminal, a_generation)

    assert rc.get_contract_by_incarnation(a_terminal, a_generation) is None
    agent = roster.get_agent(bind["agent"]["agent_id"])
    assert agent["current_incarnation"]["disposition"] == roster.INCARNATION_RETIRED


# ---------------------------------------------------------------------------
# 3. One-hop unchanged: managed reservations still win the check order
# ---------------------------------------------------------------------------


def test_managed_reservation_wins_over_an_operation_journal_row(isolated_memory_db, launch_root):
    """The teardown check order is preserved: a managed reservation row for
    the same terminal/generation supplies the facts, never the operation
    journal — every existing managed teardown behaves identically."""
    terminal_id = "ffff4444"
    generation = "00000000-0000-4000-8000-000000000003"
    # An operation journal row that would claim the same terminal.
    _insert_successor_operation(
        terminal_id,
        generation,
        {
            "working_directory": launch_root["workdir"],
            "trusted_project_root": None,
            "model": "op-journal-model",
            "effort": "low",
            "provider_executable": launch_root["binary"],
            "provider_executable_sha256": launch_root["binary_sha256"],
        },
    )
    # And a managed v2 reservation with the authoritative facts.
    _insert_managed_v2_reservation(
        terminal_id,
        generation,
        launch_root,
        model="managed-model",
        effort="medium",
    )
    _bind_worker(agent_id=str(uuid.uuid4()), terminal_id=terminal_id, generation=generation)
    _terminal_row(terminal_id, generation)

    facts = ts._teardown_launch_facts(terminal_id, generation)
    assert facts is not None
    # The managed row's facts win — the journal row is never consulted.
    assert facts["model"] == "managed-model"
    assert facts["effort"] == "medium"


def _insert_successor_operation(terminal_id, generation, facts):
    """Insert one reincarnation operation row naming the successor."""
    payload = {
        "operation_id": str(uuid.uuid4()),
        "request_digest": "d" * 64,
        "schema_version": "cao-m3-operation-journal-v1",
        "session_name": _SESSION,
        "agent_id": str(uuid.uuid4()),
        "roster_revision": 2,
        "role": "worker",
        "profile_family": "developer",
        "lineage_id": str(uuid.uuid4()),
        "harness": "claude_code",
        "native_session_id": _NATIVE_ID,
        "prior_terminal_id": "99999999",
        "prior_generation": None,
        "prior_incarnation_id": str(uuid.uuid4()),
        "lifecycle_epoch": 0,
        "lifecycle_observation": "working",
        "restore_contract_id": "rc-placeholder",
        "restore_contract_digest": "d" * 64,
        "restore_contract_schema": rc.SCHEMA_VERSION,
        "route_provider": "claude_code",
        "model_requested": None,
        "effort_requested": None,
        "execution_mode_requested": "native_tui",
        "phase": "verify_identity",
        "request_json": "{}",
        "successor_terminal_id": terminal_id,
        "successor_generation": generation,
        "successor_launch_facts_json": (
            json.dumps(facts, sort_keys=True, separators=(",", ":")) if facts is not None else None
        ),
        "created_at": _stamp(),
        "updated_at": _stamp(),
    }
    with database.SessionLocal() as session:
        session.add(database.ReincarnationOperationModel(**payload))
        session.commit()


def _insert_legacy_v2_reservation(terminal_id, generation, working_directory):
    """A pre-P0-A v2 reservation row: ``launch_facts_json`` NULL."""
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


def _insert_managed_v2_reservation(
    terminal_id,
    generation,
    launch_root,
    *,
    model,
    effort,
):
    """A v2 reservation row WITH complete launch facts."""
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
                working_directory=launch_root["workdir"],
                trusted_project_root=None,
                obligation_generation=str(uuid.uuid4()),
                run_id=str(uuid.uuid4()),
                launch_nonce_digest="a" * 64,
                state="ready",
                request_json="{}",
                launch_facts_json=json.dumps(
                    {
                        "model": model,
                        "effort": effort,
                        "provider_executable": launch_root["binary"],
                        "provider_executable_sha256": launch_root["binary_sha256"],
                    },
                    sort_keys=True,
                ),
                created_at=_stamp(),
                updated_at=_stamp(),
            )
        )
        session.commit()
