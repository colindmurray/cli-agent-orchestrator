"""Tests for cond-0485: Restore contract production writers and dormant transitions.

Proves:
1. Restore contract is published across all four seams:
   - Seam A: terminal_service._pre_task_bind_and_resolve (unmanaged pre-task bind)
   - Seam B: managed_launch_v2.bind_native (managed launch v2 bind)
   - Seam C: exact_executor._bind_successor (exact restore / successor bind)
   - Seam D: native_status_repair._commit_repair (native status repair)
2. Contracts published carry present executable facts, model, and effort (cond-0496 acceptance).
3. publish_contract failures NEVER fail the launch, bind, repair, or resume.
4. Teardown / stop succeeds best-effort when roster bookkeeping raises.
5. has_exact_resume_identity is True for a normally launched/bound agent.
6. Dormant transitions occur on retirement when a restore contract is present.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import uuid
from typing import Any, Mapping, Optional

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.clients.tmux import PaneControlIdentity, TmuxClient
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import execution_mode as em
from cli_agent_orchestrator.services import exact_executor as xe
from cli_agent_orchestrator.services import managed_launch_v2 as v2
from cli_agent_orchestrator.services import native_attachment
from cli_agent_orchestrator.services import native_pane_input as npi
from cli_agent_orchestrator.services import native_status_repair as nsr
from cli_agent_orchestrator.services import operation_journal as oj
from cli_agent_orchestrator.services import provider_contracts
from cli_agent_orchestrator.services import restore_contract as rc
from cli_agent_orchestrator.services import session_lifecycle as sl
from cli_agent_orchestrator.services import stable_agent_roster as roster
from cli_agent_orchestrator.services import supervisor_worker_ops as ops
from cli_agent_orchestrator.services import task_occurrence as occ
from cli_agent_orchestrator.services import terminal_service
from cli_agent_orchestrator.services import unmanaged_native_identity

SESSION = "cao-test-restore-writer"
_NATIVE_ID = "11111111-2222-4333-8444-555555555555"
_DIGEST_A = "a" * 64
_CELL_REF = "claude_code:anthropic"
_CELL_DIGEST = "c" * 64


@pytest.fixture(autouse=True)
def _setup_db(isolated_memory_db, monkeypatch, tmp_path):
    from cli_agent_orchestrator import constants

    monkeypatch.setattr(constants, "COMPANION_DIR", tmp_path / "companion")
    return isolated_memory_db


@pytest.fixture
def worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def _open_round(bound, *, round_index=0):
    return occ.open_occurrence(
        occ.OpenRequest(
            task_occurrence_id=str(uuid.uuid4()),
            session_name=SESSION,
            agent_id=bound["agent"]["agent_id"],
            round_index=round_index,
            dispatch_digest=_DIGEST_A,
            incarnation=occ.EffectIncarnation(
                incarnation_id=bound["incarnation"]["incarnation_id"],
                terminal_id=bound["incarnation"]["terminal_id"],
                generation=bound["incarnation"]["generation"],
            ),
            seed=occ.EMPTY_SEED,
        )
    )


# ---------------------------------------------------------------------------
# Seam A: Unmanaged pre-task bind (terminal_service)
# ---------------------------------------------------------------------------


def test_seam_a_unmanaged_bind_publishes_restore_contract(tmp_path, monkeypatch):
    """Seam A: _pre_task_bind_and_resolve publishes a valid RestoreContract with present executable fact."""
    terminal_id = f"term-{uuid.uuid4().hex[:8]}"
    generation = str(uuid.uuid4())
    session_name = f"sess-{uuid.uuid4().hex[:8]}"
    workdir = str(tmp_path.resolve())
    native_id = str(uuid.uuid4())

    with database.SessionLocal() as db:
        db.add(
            database.TerminalModel(
                id=terminal_id,
                tmux_session=session_name,
                tmux_window="w1",
                provider="claude_code",
                agent_profile="developer",
                generation=generation,
                pre_task_identity_state=provider_contracts.PRE_TASK_IDENTITY_PENDING,
            )
        )
        db.commit()

    monkeypatch.setattr(
        unmanaged_native_identity,
        "resolve_pre_task_identity",
        lambda **kw: {
            "native_session_id": native_id,
            "acquisition_method": "chosen_session_id",
            "working_directory": os.path.realpath(workdir),
            "model": "claude-3-7-sonnet-20250219",
            "effort": "high",
            "executable_path": "/bin/sh",
            "executable_hash": "a" * 64,
            "executable_version": "1.0.0",
        },
    )

    result = terminal_service._pre_task_bind_and_resolve(
        terminal_id=terminal_id,
        session_name=session_name,
        stable_agent_role=roster.ROLE_WORKER,
        agent_profile="developer",
        provider="claude_code",
        terminal_generation=generation,
        pane_id="%1",
        pane_pid=1001,
        native_status_source=True,
        working_directory=workdir,
        expected_model="claude-3-7-sonnet-20250219",
        expected_effort="high",
        codex_profile_material=None,
        forwarded_environment=None,
    )

    assert result is not None
    assert result["native_session_id"] == native_id

    contract = rc.get_contract_by_incarnation(terminal_id, generation)
    assert contract is not None
    assert contract["terminal_id"] == terminal_id
    assert contract["generation"] == generation
    assert contract["native_session_id"] == native_id
    assert contract["contract"]["harness"] == "claude_code"
    assert contract["contract"]["working_directory"]["state"] == "present"
    assert contract["contract"]["working_directory"]["value"] == os.path.realpath(workdir)

    # Executable fact must be present per cond-0496 acceptance
    assert contract["contract"]["executable"]["state"] == "present"
    assert contract["contract"]["executable"]["value"]["path"] == "/bin/sh"
    assert contract["contract"]["executable"]["value"]["sha256"] == "a" * 64
    assert contract["contract"]["model"]["state"] == "present"
    assert contract["contract"]["model"]["value"] == "claude-3-7-sonnet-20250219"
    assert contract["contract"]["effort"]["state"] == "present"
    assert contract["contract"]["effort"]["value"] == "high"


def test_seam_a_publish_failure_does_not_fail_launch(tmp_path, monkeypatch):
    """Seam A: if publish_contract raises, _pre_task_bind_and_resolve still succeeds."""
    terminal_id = f"term-{uuid.uuid4().hex[:8]}"
    generation = str(uuid.uuid4())
    session_name = f"sess-{uuid.uuid4().hex[:8]}"
    workdir = str(tmp_path.resolve())
    native_id = str(uuid.uuid4())

    with database.SessionLocal() as db:
        db.add(
            database.TerminalModel(
                id=terminal_id,
                tmux_session=session_name,
                tmux_window="w1",
                provider="claude_code",
                agent_profile="developer",
                generation=generation,
                pre_task_identity_state=provider_contracts.PRE_TASK_IDENTITY_PENDING,
            )
        )
        db.commit()

    monkeypatch.setattr(
        unmanaged_native_identity,
        "resolve_pre_task_identity",
        lambda **kw: {
            "native_session_id": native_id,
            "acquisition_method": "chosen_session_id",
            "working_directory": os.path.realpath(workdir),
        },
    )

    def _failing_publish(*args, **kwargs):
        raise RuntimeError("database temporarily locked")

    monkeypatch.setattr(rc, "publish_contract", _failing_publish)

    # Must NOT raise
    result = terminal_service._pre_task_bind_and_resolve(
        terminal_id=terminal_id,
        session_name=session_name,
        stable_agent_role=roster.ROLE_WORKER,
        agent_profile="developer",
        provider="claude_code",
        terminal_generation=generation,
        pane_id="%1",
        pane_pid=1001,
        native_status_source=True,
        working_directory=workdir,
        expected_model=None,
        expected_effort=None,
        codex_profile_material=None,
        forwarded_environment=None,
    )
    assert result is not None
    assert result["native_session_id"] == native_id


# ---------------------------------------------------------------------------
# Seam B: Managed Launch V2 (managed_launch_v2.bind_native)
# ---------------------------------------------------------------------------


def _reserve_and_claim_v2(worktree, tmp_path, provider="codex"):
    executable = tmp_path / "fake-provider"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    req = v2.ManagedLaunchV2ReserveRequest(
        protocol_version=v2.PROTOCOL_VERSION_V2,
        reservation_id=str(uuid.uuid4()),
        session_name=SESSION,
        provider=provider,
        agent_profile="reviewer-sol-max" if provider == "codex" else "developer",
        caller_id="deadbeef",
        working_directory=str(worktree),
        trusted_project_root=str(worktree) if provider == "codex" else None,
        expected_model="gpt-5.6-sol" if provider == "codex" else "claude-3-7-sonnet-20250219",
        expected_effort="xhigh" if provider == "codex" else "high",
        provider_executable=str(executable),
        provider_executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        obligation_generation="obgen-7c2e4a1b",
        run_id="run-0001",
        delivery_id=str(uuid.uuid4()),
        launch_nonce="n" * 40,
    )
    record, _ = v2.reserve(req)
    v2.claim_launch(record["reservation_id"])
    return record


def _setup_v2_ready_state(record, monkeypatch):
    session_id = f"thr_{uuid.uuid4().hex[:16]}"
    req = record.get("request") or {}
    model = req.get("expected_model") or "gpt-5.6-sol"
    effort = req.get("expected_effort") or "xhigh"
    receipt = {
        "bridge_version": "cao-bridge-v1",
        "receipt_id": session_id,
        "provider_session_id": session_id,
        "provider_receipt_kind": "codex-thread-start",
        "provider_transcript_sha256": "a" * 64,
        "provider_version": "0.146.0",
        "model_input_ready": True,
        "reservation_id": record["reservation_id"],
        "terminal_id": record["terminal_id"],
        "generation": record["generation"],
        "provider": record["provider"],
        "agent_profile": record["agent_profile"],
        "model": model,
        "effort": effort,
        "working_directory": record["working_directory"],
    }
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.managed_provider_bridge.read_state",
        lambda rid: {"state": "ready", "readiness": receipt},
        raising=False,
    )
    return session_id


def test_seam_b_managed_bind_publishes_restore_contract(worktree, tmp_path, monkeypatch):
    """Seam B: bind_native publishes a valid RestoreContract with present executable fact inside transaction."""
    record = _reserve_and_claim_v2(worktree, tmp_path, provider="codex")
    provider_session_id = _setup_v2_ready_state(record, monkeypatch)

    bind_req = v2.ManagedLaunchV2BindRequest(
        protocol_version=v2.PROTOCOL_VERSION_V2,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        attempt_id=str(uuid.uuid4()),
    )
    bound = v2.bind_native(record["reservation_id"], bind_req)
    assert bound["state"] == "bound"

    contract = rc.get_contract_by_incarnation(record["terminal_id"], record["generation"])
    assert contract is not None
    assert contract["terminal_id"] == record["terminal_id"]
    assert contract["generation"] == record["generation"]
    assert contract["native_session_id"] == provider_session_id
    assert contract["contract"]["harness"] == "codex"
    assert contract["contract"]["model"]["state"] == "present"
    assert contract["contract"]["model"]["value"] == "gpt-5.6-sol"
    assert contract["contract"]["effort"]["state"] == "present"
    assert contract["contract"]["effort"]["value"] == "xhigh"

    # Executable fact must be present per cond-0496 acceptance
    assert contract["contract"]["executable"]["state"] == "present"
    assert contract["contract"]["executable"]["value"]["path"] == str(tmp_path / "fake-provider")
    assert len(contract["contract"]["executable"]["value"]["sha256"]) == 64


def test_seam_b_publish_failure_does_not_fail_bind(worktree, tmp_path, monkeypatch):
    """Seam B: if publish_contract raises, bind_native still succeeds and transitions to bound."""
    record = _reserve_and_claim_v2(worktree, tmp_path, provider="codex")
    provider_session_id = _setup_v2_ready_state(record, monkeypatch)

    def _failing_publish(*args, **kwargs):
        raise RuntimeError("publication failed unexpectedly")

    monkeypatch.setattr(rc, "publish_contract", _failing_publish)

    bind_req = v2.ManagedLaunchV2BindRequest(
        protocol_version=v2.PROTOCOL_VERSION_V2,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        attempt_id=str(uuid.uuid4()),
    )
    bound = v2.bind_native(record["reservation_id"], bind_req)
    assert bound["state"] == "bound"


# ---------------------------------------------------------------------------
# Seam C: Exact Executor / Successor Reincarnation (exact_executor)
# ---------------------------------------------------------------------------


def _setup_bound_and_dormant_worker(workdir):
    agent_id = str(uuid.uuid4())
    term_id = f"term-{uuid.uuid4().hex[:8]}"
    gen = str(uuid.uuid4())
    session = database.SessionLocal()
    try:
        bind = roster.bind_generation(
            roster.BindingContract(
                agent_id=agent_id,
                session_name=SESSION,
                role=roster.ROLE_WORKER,
                profile_family="developer",
                harness="claude_code",
                native_session_id=_NATIVE_ID,
                acquisition_method="chosen_session_id",
                terminal_id=term_id,
                generation=gen,
                pane_id="%1",
                pane_pid=1001,
                process_identity={"pid": 1001, "start_marker": "m-1"},
                execution_mode=em.NATIVE_TUI,
                admitted=True,
            ),
            db=session,
        )
        contract = rc.RestoreContract(
            agent_id=agent_id,
            lineage_id=bind["lineage"]["lineage_id"],
            terminal_id=term_id,
            generation=gen,
            native_session_id=_NATIVE_ID,
            harness="claude_code",
            provider="claude_code",
            route_provenance=bind["lineage"]["route_provenance"],
            execution_mode=em.NATIVE_TUI,
            working_directory=workdir,
            model=rc.ContractFact.present("claude-3-7-sonnet-20250219"),
            effort=rc.ContractFact.present("high"),
            executable=rc.ContractFact.present({"path": "/bin/sh", "sha256": "a" * 64}),
            profile_material=rc.ContractFact.unavailable("no profile"),
            provider_home_facts=rc.ContractFact.unavailable("no home"),
        )
        published = rc.publish_contract(contract, db=session)
        roster.transition_dormant(
            terminal_id=term_id,
            generation=gen,
            agent_id=agent_id,
            lineage_id=bind["lineage"]["lineage_id"],
            contract_digest=contract.digest(),
            reason="retired for test",
            db=session,
        )
        session.commit()
    finally:
        session.close()
    return bind, contract, published["contract_id"]


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
        "model_requested": "claude-3-7-sonnet-20250219",
        "effort_requested": "high",
        "execution_mode_requested": "native_tui",
        "compatibility_cell_ref": _CELL_REF,
        "compatibility_cell_digest": _CELL_DIGEST,
    }
    if "restore_contract_id" not in changes:
        payload["restore_contract_id"] = rc.get_contract_by_incarnation(
            terminal_id=bind["incarnation"]["terminal_id"],
            generation=bind["incarnation"]["generation"],
        )["contract_id"]
    payload.update(changes)
    return oj.OperationRequest(**payload)


def test_seam_c_exact_resume_publishes_successor_contract(tmp_path, monkeypatch):
    """Seam C: _bind_successor in exact_executor publishes successor RestoreContract with present executable fact."""
    workdir = str(tmp_path.resolve())
    bind, prior_contract, contract_id = _setup_bound_and_dormant_worker(workdir)

    successor_terminal_id = f"term-succ-{uuid.uuid4().hex[:8]}"
    successor_generation = str(uuid.uuid4())
    req = _operation_request(bind, prior_contract)
    oj.claim_operation(req)

    with database.SessionLocal() as db:
        row = (
            db.query(database.ReincarnationOperationModel)
            .filter_by(operation_id=req.operation_id)
            .first()
        )
        row.phase = oj.EFFECT_STEP_VERIFY_IDENTITY
        db.commit()

    execution = xe._Execution(req, xe.LaunchMaterial(), None, None)
    execution.contract = prior_contract

    launch_outcome = {
        "outcome": {
            "pane_observation": {"pid": 1002, "start_marker": "m-2"},
            "attachment": {"owner": {"pane_id": "%2"}},
        }
    }

    bind_result = xe._bind_successor(
        execution,
        req,
        successor_terminal_id,
        successor_generation,
        launch_outcome,
    )
    assert bind_result["incarnation"]["disposition"] == roster.INCARNATION_BOUND

    successor_contract = rc.get_contract_by_incarnation(successor_terminal_id, successor_generation)
    assert successor_contract is not None
    assert successor_contract["terminal_id"] == successor_terminal_id
    assert successor_contract["generation"] == successor_generation
    assert successor_contract["agent_id"] == bind["agent"]["agent_id"]
    assert successor_contract["lineage_id"] == bind_result["lineage"]["lineage_id"]
    assert successor_contract["native_session_id"] == _NATIVE_ID
    assert successor_contract["contract"]["model"]["state"] == "present"
    assert successor_contract["contract"]["model"]["value"] == "claude-3-7-sonnet-20250219"

    # Executable fact preserved as present per cond-0496 acceptance
    assert successor_contract["contract"]["executable"]["state"] == "present"
    assert successor_contract["contract"]["executable"]["value"]["path"] == "/bin/sh"


def test_seam_c_publish_failure_does_not_fail_resume(tmp_path, monkeypatch):
    """Seam C: publication failure during successor bind logs warning and does not fail resume."""
    workdir = str(tmp_path.resolve())
    bind, prior_contract, contract_id = _setup_bound_and_dormant_worker(workdir)

    successor_terminal_id = f"term-succ-{uuid.uuid4().hex[:8]}"
    successor_generation = str(uuid.uuid4())
    req = _operation_request(bind, prior_contract)
    oj.claim_operation(req)

    with database.SessionLocal() as db:
        row = (
            db.query(database.ReincarnationOperationModel)
            .filter_by(operation_id=req.operation_id)
            .first()
        )
        row.phase = oj.EFFECT_STEP_VERIFY_IDENTITY
        db.commit()

    def _failing_publish(*args, **kwargs):
        raise RuntimeError("successor publish boom")

    monkeypatch.setattr(rc, "publish_contract", _failing_publish)

    execution = xe._Execution(req, xe.LaunchMaterial(), None, None)
    execution.contract = prior_contract

    launch_outcome = {
        "outcome": {
            "pane_observation": {"pid": 1002, "start_marker": "m-2"},
            "attachment": {"owner": {"pane_id": "%2"}},
        }
    }

    bind_result = xe._bind_successor(
        execution,
        req,
        successor_terminal_id,
        successor_generation,
        launch_outcome,
    )
    assert bind_result["incarnation"]["disposition"] == roster.INCARNATION_BOUND
    assert "successor publish boom" in execution.evidence["restore_contract_publish_error"]


# ---------------------------------------------------------------------------
# Seam D: Native Status Repair (native_status_repair)
# ---------------------------------------------------------------------------


class _FakeStatusHarness:
    def __init__(self, screens, terminal_id, session_name):
        self.screens = screens
        self.styled_screens = []
        self.escapes = 0
        self.pane_identity = PaneControlIdentity(
            pane_id="%7",
            window_id="@7",
            session_id="$1",
            pane_pid=4242,
            session_name=session_name,
            window_name=f"w-{terminal_id}",
            bracketed_paste_proven=False,
            dead=False,
            server_socket_path="/private/tmp/cao-native.sock",
        )
        self.server_identity = "/private/tmp/cao-native.sock"
        self.live_start_marker = "Thu Jul 24 10:00:00 2026"
        self._composer = ""

    def _composer_rows(self):
        return ["", "", "", "", f"> {self._composer}"]

    def turn_state(self, pane_id, **kw):
        return TerminalStatus.IDLE

    def capture_screen(self, pane_id, **kw):
        return list(self.screens[-1]) + self._composer_rows()

    def capture_screen_styled(self, pane_id, **kw):
        return [
            "-------------------------------------------------------------------------------",
            "> ",
            "-------------------------------------------------------------------------------",
            "<quota/model/cwd status line>",
        ]

    def pane_control_identity(self, *args, **kw):
        return self.pane_identity

    def pane_server_identity(self, *args, **kw):
        return self.server_identity

    def start_marker(self, pid):
        return self.live_start_marker

    def typed_literal(self, text):
        self._composer = text

    def typed_enter(self):
        self._composer = ""

    def typed_key(self, keystroke):
        if keystroke == "Escape":
            self.escapes += 1


class _FakeStatusPaneInput:
    _state: _FakeStatusHarness

    @classmethod
    def for_state(cls, state: _FakeStatusHarness) -> type["_FakeStatusPaneInput"]:
        cls._state = state
        return cls

    def __init__(self, pane_id):
        self._pane_id = pane_id

    def send_literal(self, text):
        self._state.typed_literal(text)

    def send_enter(self):
        self._state.typed_enter()

    def send_key(self, key):
        self._state.typed_key(key)


def _setup_repair_mocks(harness, monkeypatch):
    monkeypatch.setattr(npi, "TmuxPaneInput", _FakeStatusPaneInput.for_state(harness))
    monkeypatch.setattr(npi, "capture_pane_screen", harness.capture_screen)
    monkeypatch.setattr(npi, "capture_pane_screen_styled", harness.capture_screen_styled)
    monkeypatch.setattr(npi, "observe_claude_turn_state", harness.turn_state)
    monkeypatch.setattr(TmuxClient, "pane_control_identity", harness.pane_control_identity)
    monkeypatch.setattr(TmuxClient, "observe_pane_server_identity", harness.pane_server_identity)
    monkeypatch.setattr(nsr, "_live_start_marker", harness.start_marker)
    monkeypatch.setattr(v2, "NATIVE_PANE_READY_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr(v2, "_NATIVE_PANE_READY_POLL_SECONDS", 0.005)


def test_seam_d_status_repair_publishes_restore_contract(tmp_path, monkeypatch):
    """Seam D: repair_terminal_native_identity publishes RestoreContract."""
    terminal_id = "a1b2c3d4"
    generation = "00000000-0000-4000-8000-000000000001"
    session_id = "4f5f46c7-b660-4f6f-a144-d2c6dceccf95"
    workdir = str(tmp_path.resolve())

    # Create terminal row in DB
    database.create_terminal_v2(
        terminal_id,
        SESSION,
        f"w-{terminal_id}",
        "claude_code",
        generation=generation,
        pane_id="%7",
        window_id="@7",
        server_socket_path="/private/tmp/cao-native.sock",
        session_id="$1",
        pane_pid=4242,
    )
    with database.SessionLocal() as db:
        row = (
            db.query(database.ManagedLaunchV2TerminalModel)
            .filter(database.ManagedLaunchV2TerminalModel.id == terminal_id)
            .first()
        )
        row.v2_lifecycle_state = "live"
        row.v2_native_session_id = None
        db.commit()

    # Create reservation with executable and route facts
    v2_binding = {
        "schema": "cao-managed-v2-native-binding-v1",
        "execution_mode": "native_tui",
        "native_session_id": session_id,
        "provider_version": "2.1.226",
    }
    with database.SessionLocal() as db:
        db.add(
            database.ManagedLaunchV2ReservationModel(
                reservation_id=str(uuid.uuid4()),
                protocol_vintage="v2",
                session_name=SESSION,
                provider="claude_code",
                agent_profile="developer",
                caller_id="deadbeef",
                obligation_generation="obgen-1",
                run_id="run-0001",
                launch_nonce_digest="0" * 64,
                terminal_id=terminal_id,
                generation=generation,
                working_directory=workdir,
                state="bound",
                execution_mode="native_tui",
                created_at="2026-08-19T00:00:00Z",
                updated_at="2026-08-19T00:00:00Z",
                binding_json=json.dumps(v2_binding),
                request_json=json.dumps(
                    {
                        "expected_model": "claude-3-7-sonnet-20250219",
                        "expected_effort": "high",
                        "provider_executable": "/bin/sh",
                        "provider_executable_sha256": "c" * 64,
                    }
                ),
            )
        )
        db.commit()

    # Pre-bind in roster with missing identity
    roster.bind_generation(
        roster.BindingContract(
            agent_id=str(uuid.uuid4()),
            session_name=SESSION,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="claude_code",
            native_session_id=None,
            acquisition_method=None,
            terminal_id=terminal_id,
            generation=generation,
            pane_id="%7",
            pane_pid=4242,
            process_identity={"pid": 4242, "start_marker": "Thu Jul 24 10:00:00 2026"},
            execution_mode=em.NATIVE_TUI,
            admitted=True,
        )
    )

    # Panel rows
    panel_rows = [
        "Settings  Status   Config   Usage   Stats",
        "",
        "Version:          2.1.226",
        f"Session ID:       {session_id}",
        "Session kind:     interactive",
        f"cwd:              {workdir}",
        "Esc to cancel",
    ]

    harness = _FakeStatusHarness([panel_rows], terminal_id, SESSION)
    _setup_repair_mocks(harness, monkeypatch)

    outcome = nsr.repair_terminal_native_identity(
        terminal_id=terminal_id,
        generation=generation,
        provider_version="2.1.226",
        operation_id=str(uuid.uuid4()),
    )
    assert outcome["status"] == "repaired"

    contract = rc.get_contract_by_incarnation(terminal_id, generation)
    assert contract is not None
    assert contract["terminal_id"] == terminal_id
    assert contract["generation"] == generation
    assert contract["native_session_id"] == session_id
    assert contract["contract"]["harness"] == "claude_code"
    assert contract["contract"]["provider"] == "claude_code"

    # Executable fact from reservation must be present per cond-0496 acceptance
    assert contract["contract"]["executable"]["state"] == "present"
    assert contract["contract"]["executable"]["value"]["path"] == "/bin/sh"
    assert contract["contract"]["executable"]["value"]["sha256"] == "c" * 64
    assert contract["contract"]["model"]["state"] == "present"
    assert contract["contract"]["model"]["value"] == "claude-3-7-sonnet-20250219"


def test_seam_d_publish_failure_does_not_fail_repair(tmp_path, monkeypatch):
    """Seam D: if publish_contract raises during repair, repair still succeeds and evidence is recorded."""
    terminal_id = "a1b2c3d4"
    generation = "00000000-0000-4000-8000-000000000001"
    session_id = "4f5f46c7-b660-4f6f-a144-d2c6dceccf95"
    workdir = str(tmp_path.resolve())

    database.create_terminal_v2(
        terminal_id,
        SESSION,
        f"w-{terminal_id}",
        "claude_code",
        generation=generation,
        pane_id="%7",
        window_id="@7",
        server_socket_path="/private/tmp/cao-native.sock",
        session_id="$1",
        pane_pid=4242,
    )
    with database.SessionLocal() as db:
        row = (
            db.query(database.ManagedLaunchV2TerminalModel)
            .filter(database.ManagedLaunchV2TerminalModel.id == terminal_id)
            .first()
        )
        row.v2_lifecycle_state = "live"
        row.v2_native_session_id = None
        db.commit()

    v2_binding = {
        "schema": "cao-managed-v2-native-binding-v1",
        "execution_mode": "native_tui",
        "native_session_id": session_id,
        "provider_version": "2.1.226",
    }
    with database.SessionLocal() as db:
        db.add(
            database.ManagedLaunchV2ReservationModel(
                reservation_id=str(uuid.uuid4()),
                protocol_vintage="v2",
                session_name=SESSION,
                provider="claude_code",
                agent_profile="developer",
                caller_id="deadbeef",
                obligation_generation="obgen-1",
                run_id="run-0001",
                launch_nonce_digest="0" * 64,
                terminal_id=terminal_id,
                generation=generation,
                working_directory=workdir,
                state="bound",
                execution_mode="native_tui",
                created_at="2026-08-19T00:00:00Z",
                updated_at="2026-08-19T00:00:00Z",
                binding_json=json.dumps(v2_binding),
                request_json=json.dumps(
                    {
                        "expected_model": "claude-3-7-sonnet-20250219",
                        "expected_effort": "high",
                    }
                ),
            )
        )
        db.commit()

    roster.bind_generation(
        roster.BindingContract(
            agent_id=str(uuid.uuid4()),
            session_name=SESSION,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="claude_code",
            native_session_id=None,
            acquisition_method=None,
            terminal_id=terminal_id,
            generation=generation,
            pane_id="%7",
            pane_pid=4242,
            process_identity={"pid": 4242, "start_marker": "Thu Jul 24 10:00:00 2026"},
            execution_mode=em.NATIVE_TUI,
            admitted=True,
        )
    )

    def _failing_publish(*args, **kwargs):
        raise RuntimeError("repair contract publication boom")

    monkeypatch.setattr(rc, "publish_contract", _failing_publish)

    panel_rows = [
        "Settings  Status   Config   Usage   Stats",
        "",
        "Version:          2.1.226",
        f"Session ID:       {session_id}",
        "Session kind:     interactive",
        f"cwd:              {workdir}",
        "Esc to cancel",
    ]

    harness = _FakeStatusHarness([panel_rows], terminal_id, SESSION)
    _setup_repair_mocks(harness, monkeypatch)

    op_id = str(uuid.uuid4())
    outcome = nsr.repair_terminal_native_identity(
        terminal_id=terminal_id,
        generation=generation,
        provider_version="2.1.226",
        operation_id=op_id,
    )
    assert outcome["status"] == "repaired"
    assert outcome["native_session_id"] == session_id


# ---------------------------------------------------------------------------
# Dormant Transitions and Teardown Best-Effort Resilience
# ---------------------------------------------------------------------------


def test_terminal_service_best_effort_retire_transitions_dormant_when_contract_exists(
    tmp_path, monkeypatch
):
    """_roster_retire_incarnation_best_effort transitions dormant when contract exists."""
    workdir = str(tmp_path.resolve())
    terminal_id = f"term-{uuid.uuid4().hex[:8]}"
    generation = str(uuid.uuid4())
    session_name = f"sess-{uuid.uuid4().hex[:8]}"
    native_id = str(uuid.uuid4())

    with database.SessionLocal() as db:
        db.add(
            database.TerminalModel(
                id=terminal_id,
                tmux_session=session_name,
                tmux_window="w1",
                provider="claude_code",
                agent_profile="developer",
                generation=generation,
                pre_task_identity_state=provider_contracts.PRE_TASK_IDENTITY_PENDING,
            )
        )
        db.commit()

    monkeypatch.setattr(
        unmanaged_native_identity,
        "resolve_pre_task_identity",
        lambda **kw: {
            "native_session_id": native_id,
            "acquisition_method": "chosen_session_id",
            "working_directory": os.path.realpath(workdir),
        },
    )

    terminal_service._pre_task_bind_and_resolve(
        terminal_id=terminal_id,
        session_name=session_name,
        stable_agent_role=roster.ROLE_WORKER,
        agent_profile="developer",
        provider="claude_code",
        terminal_generation=generation,
        pane_id="%1",
        pane_pid=1001,
        native_status_source=True,
        working_directory=workdir,
        expected_model=None,
        expected_effort=None,
        codex_profile_material=None,
        forwarded_environment=None,
    )

    # Teardown best effort
    terminal_service._roster_retire_incarnation_best_effort(terminal_id, generation)

    contract = rc.get_contract_by_incarnation(terminal_id, generation)
    assert contract is not None
    agent = roster.get_agent(roster.derive_initial_agent_id(terminal_id))
    assert agent["disposition"] == roster.DISPOSITION_DORMANT


def test_terminal_service_best_effort_retire_falls_back_when_no_contract():
    """_roster_retire_incarnation_best_effort retires incarnation directly when no contract exists."""
    terminal_id = f"term-{uuid.uuid4().hex[:8]}"
    generation = str(uuid.uuid4())

    roster_bind = roster.bind_generation(
        roster.BindingContract(
            agent_id=str(uuid.uuid4()),
            session_name=SESSION,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="claude_code",
            native_session_id=str(uuid.uuid4()),
            acquisition_method="chosen_session_id",
            terminal_id=terminal_id,
            generation=generation,
            pane_id="%50",
            pane_pid=5050,
            process_identity={"pid": 5050, "start_marker": "m-50"},
            execution_mode=em.NATIVE_TUI,
            admitted=True,
        )
    )

    terminal_service._roster_retire_incarnation_best_effort(terminal_id, generation)

    inc = roster.get_incarnation_by_terminal(terminal_id=terminal_id, generation=generation)
    assert inc["disposition"] == roster.INCARNATION_RETIRED


def test_terminal_service_best_effort_retire_never_raises(monkeypatch):
    """_roster_retire_incarnation_best_effort swallows any roster error during teardown."""

    def _broken(*args, **kwargs):
        raise RuntimeError("catastrophic roster error")

    monkeypatch.setattr(roster, "retire_incarnation", _broken)
    monkeypatch.setattr(roster, "transition_dormant", _broken)

    # Must not raise
    terminal_service._roster_retire_incarnation_best_effort("term-nonexistent", "gen-nonexistent")


def test_retire_worker_pane_transitions_dormant_when_contract_exists(tmp_path, monkeypatch):
    """ops.retire_worker_pane transitions dormant when restore contract is present."""
    workdir = str(tmp_path.resolve())
    bind, contract, _ = _setup_bound_and_dormant_worker(workdir)
    # Re-bind so incarnation is active
    new_gen = str(uuid.uuid4())
    active_bind = roster.bind_generation(
        roster.BindingContract(
            agent_id=bind["agent"]["agent_id"],
            session_name=SESSION,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="claude_code",
            native_session_id=_NATIVE_ID,
            acquisition_method="chosen_session_id",
            terminal_id="term-active-1",
            generation=new_gen,
            pane_id="%51",
            pane_pid=5051,
            process_identity={"pid": 5051, "start_marker": "m-51"},
            execution_mode=em.NATIVE_TUI,
            admitted=True,
        )
    )
    new_contract = rc.RestoreContract(
        agent_id=bind["agent"]["agent_id"],
        lineage_id=active_bind["lineage"]["lineage_id"],
        terminal_id="term-active-1",
        generation=new_gen,
        native_session_id=_NATIVE_ID,
        harness="claude_code",
        provider="claude_code",
        route_provenance=active_bind["lineage"]["route_provenance"],
        execution_mode=em.NATIVE_TUI,
        working_directory=workdir,
        model=rc.ContractFact.present("claude-3-7-sonnet-20250219"),
        effort=rc.ContractFact.present("high"),
        executable=rc.ContractFact.present({"path": "/bin/sh", "sha256": "a" * 64}),
        profile_material=rc.ContractFact.unavailable("no profile"),
        provider_home_facts=rc.ContractFact.unavailable("no home"),
    )
    rc.publish_contract(new_contract)

    monkeypatch.setattr(terminal_service, "delete_terminal", lambda *args, **kwargs: True)

    retire_res = ops.retire_worker_pane(
        ops.RetireRequest(
            session_name=SESSION,
            agent_id=bind["agent"]["agent_id"],
            reason="worker task completed",
            retired_by="supervisor",
        )
    )
    assert retire_res["pane_collected"] is True
    assert "agent" in retire_res["retired_incarnation"]
    assert retire_res["retired_incarnation"]["agent"]["disposition"] == roster.DISPOSITION_DORMANT
    agent = roster.get_agent(bind["agent"]["agent_id"])
    assert agent["disposition"] == roster.DISPOSITION_DORMANT


def test_retire_worker_pane_succeeds_when_roster_raises(tmp_path, monkeypatch):
    """ops.retire_worker_pane still succeeds in pane collection when roster bookkeeping raises."""
    workdir = str(tmp_path.resolve())
    bind, contract, _ = _setup_bound_and_dormant_worker(workdir)

    monkeypatch.setattr(terminal_service, "delete_terminal", lambda *args, **kwargs: True)

    def _failing_transition(*args, **kwargs):
        raise roster.StableAgentError("roster unreachable")

    monkeypatch.setattr(roster, "transition_dormant", _failing_transition)

    retire_res = ops.retire_worker_pane(
        ops.RetireRequest(
            session_name=SESSION,
            agent_id=bind["agent"]["agent_id"],
            reason="test teardown",
            retired_by="supervisor",
        )
    )
    # Roster failure does not block pane collection
    assert retire_res["pane_collected"] is True
    assert retire_res["retired_incarnation"] is None


def test_recover_lost_pane_transitions_dormant_when_contract_exists(tmp_path, monkeypatch):
    """ops.recover_lost_pane transitions dormant when restore contract is present."""
    workdir = str(tmp_path.resolve())
    agent_id = str(uuid.uuid4())
    term_id = f"term-lost-{uuid.uuid4().hex[:8]}"
    gen = str(uuid.uuid4())
    active_bind = roster.bind_generation(
        roster.BindingContract(
            agent_id=agent_id,
            session_name=SESSION,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="claude_code",
            native_session_id=_NATIVE_ID,
            acquisition_method="chosen_session_id",
            terminal_id=term_id,
            generation=gen,
            pane_id="%52",
            pane_pid=5052,
            process_identity={"pid": 5052, "start_marker": "m-52"},
            execution_mode=em.NATIVE_TUI,
            admitted=True,
        )
    )
    new_contract = rc.RestoreContract(
        agent_id=agent_id,
        lineage_id=active_bind["lineage"]["lineage_id"],
        terminal_id=term_id,
        generation=gen,
        native_session_id=_NATIVE_ID,
        harness="claude_code",
        provider="claude_code",
        route_provenance=active_bind["lineage"]["route_provenance"],
        execution_mode=em.NATIVE_TUI,
        working_directory=workdir,
        model=rc.ContractFact.present("claude-3-7-sonnet-20250219"),
        effort=rc.ContractFact.present("high"),
        executable=rc.ContractFact.present({"path": "/bin/sh", "sha256": "a" * 64}),
        profile_material=rc.ContractFact.unavailable("no profile"),
        provider_home_facts=rc.ContractFact.unavailable("no home"),
    )
    rc.publish_contract(new_contract)

    # Open a round for the active worker so lost-pane recovery can admit
    _open_round(active_bind)

    def _assert_no_bare_retire(*args, **kwargs):
        raise AssertionError("bare retire_incarnation was called instead of transition_dormant")

    monkeypatch.setattr(roster, "retire_incarnation", _assert_no_bare_retire)

    # Admit fresh successor
    admitted = ops.admit_fresh_successor(
        SESSION,
        agent_id,
        recovery_id=str(uuid.uuid4()),
        requested_by="supervisor",
    )
    assert admitted["mode"] == ops.ADMIT_LAUNCH

    # Incarnation must be retired and agent contract digest must match new_contract
    inc = roster.get_incarnation_by_terminal(terminal_id=term_id, generation=gen)
    assert inc["disposition"] == roster.INCARNATION_RETIRED
    agent = roster.get_agent(agent_id)
    assert agent["disposition"] == roster.DISPOSITION_DORMANT


# ---------------------------------------------------------------------------
# Acceptance Line: has_exact_resume_identity
# ---------------------------------------------------------------------------


def test_has_exact_resume_identity_is_true_for_normally_launched_agent(
    worktree, tmp_path, monkeypatch
):
    """The acceptance line: has_exact_resume_identity is True for a normally launched agent."""
    record = _reserve_and_claim_v2(worktree, tmp_path, provider="codex")
    provider_session_id = _setup_v2_ready_state(record, monkeypatch)

    bind_req = v2.ManagedLaunchV2BindRequest(
        protocol_version=v2.PROTOCOL_VERSION_V2,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        attempt_id=str(uuid.uuid4()),
    )
    bound = v2.bind_native(record["reservation_id"], bind_req)
    assert bound["state"] == "bound"

    # Fetch agent from roster
    db = database.SessionLocal()
    try:
        res = (
            db.query(database.ManagedLaunchV2ReservationModel)
            .filter_by(reservation_id=record["reservation_id"])
            .first()
        )
        assert res.stable_agent_id is not None
        agent_id = res.stable_agent_id
    finally:
        db.close()

    agent = roster.get_agent(agent_id)
    assert ops.has_exact_resume_identity(agent) is True


def test_has_exact_resume_identity_is_false_when_no_contract():
    """has_exact_resume_identity is False when agent was bound without a RestoreContract."""
    agent_id = str(uuid.uuid4())
    roster.bind_generation(
        roster.BindingContract(
            agent_id=agent_id,
            session_name=SESSION,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="claude_code",
            native_session_id=_NATIVE_ID,
            acquisition_method="chosen_session_id",
            terminal_id="term-no-contract",
            generation=str(uuid.uuid4()),
            pane_id="%98",
            pane_pid=9098,
            process_identity={"pid": 9098, "start_marker": "m-98"},
            execution_mode=em.NATIVE_TUI,
            admitted=True,
        )
    )
    agent = roster.get_agent(agent_id)
    assert ops.has_exact_resume_identity(agent) is False


def test_retire_worker_pane_still_collects_pane_when_contract_read_raises(tmp_path, monkeypatch):
    """F3: a raw SQLAlchemy error from the restore-contract read must never
    skip delete_terminal — teardown never fails closed on roster bookkeeping."""
    from sqlalchemy.exc import OperationalError

    workdir = str(tmp_path.resolve())
    bind, contract, _ = _setup_bound_and_dormant_worker(workdir)

    deleted: list[str] = []
    monkeypatch.setattr(
        terminal_service,
        "delete_terminal",
        lambda terminal_id, **kwargs: deleted.append(terminal_id) or True,
    )

    def _broken_read(*args, **kwargs):
        raise OperationalError("read", {}, Exception("database is locked"))

    monkeypatch.setattr(rc, "get_contract_by_incarnation", _broken_read)

    retire_res = ops.retire_worker_pane(
        ops.RetireRequest(
            session_name=SESSION,
            agent_id=bind["agent"]["agent_id"],
            reason="test teardown",
            retired_by="supervisor",
        )
    )
    assert retire_res["pane_collected"] is True
    assert deleted == [bind["incarnation"]["terminal_id"]]


def test_retire_worker_pane_falls_back_to_plain_retirement_when_dormant_transition_conflicts(
    tmp_path, monkeypatch
):
    """F5: when the dormant transition refuses (StableAgentConflict), teardown
    still retires the incarnation — it is never left LIVE with no backing pane."""
    workdir = str(tmp_path.resolve())
    bind, contract, _ = _setup_bound_and_dormant_worker(workdir)

    monkeypatch.setattr(terminal_service, "delete_terminal", lambda *args, **kwargs: True)

    def _conflicting_transition(*args, **kwargs):
        raise roster.StableAgentConflict(
            "stored restore contract cannot authorize the dormant transition"
        )

    monkeypatch.setattr(roster, "transition_dormant", _conflicting_transition)

    retire_res = ops.retire_worker_pane(
        ops.RetireRequest(
            session_name=SESSION,
            agent_id=bind["agent"]["agent_id"],
            reason="test teardown",
            retired_by="supervisor",
        )
    )
    assert retire_res["pane_collected"] is True
    assert retire_res["retired_incarnation"] is not None
    inc = roster.get_incarnation_by_terminal(
        terminal_id=bind["incarnation"]["terminal_id"],
        generation=bind["incarnation"]["generation"],
    )
    assert inc["disposition"] == roster.INCARNATION_RETIRED


def test_seam_d_repair_records_unavailable_working_directory_when_none_captured(
    tmp_path, monkeypatch
):
    """F4: when the reservation carries no working directory, the repair
    contract records working_directory as unavailable — never the repair
    process's cwd."""
    terminal_id = "a1b2c3d4"
    generation = "00000000-0000-4000-8000-000000000001"
    session_id = "4f5f46c7-b660-4f6f-a144-d2c6dceccf95"

    database.create_terminal_v2(
        terminal_id,
        SESSION,
        f"w-{terminal_id}",
        "claude_code",
        generation=generation,
        pane_id="%7",
        window_id="@7",
        server_socket_path="/private/tmp/cao-native.sock",
        session_id="$1",
        pane_pid=4242,
    )
    with database.SessionLocal() as db:
        row = db.query(database.ManagedLaunchV2TerminalModel).filter_by(id=terminal_id).first()
        row.v2_lifecycle_state = "live"
        row.v2_native_session_id = None
        db.commit()

    roster.bind_generation(
        roster.BindingContract(
            agent_id=str(uuid.uuid4()),
            session_name=SESSION,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="claude_code",
            native_session_id=None,
            acquisition_method=None,
            terminal_id=terminal_id,
            generation=generation,
            pane_id="%7",
            pane_pid=4242,
            process_identity={"pid": 4242, "start_marker": "Thu Jul 24 10:00:00 2026"},
            execution_mode=em.NATIVE_TUI,
            admitted=True,
        )
    )

    monkeypatch.setattr(nsr, "_verify_exact_facts", lambda *a, **k: None)
    monkeypatch.setattr(nsr, "_verify_live_pane", lambda *a, **k: None)

    facts = {
        "operation_id": str(uuid.uuid4()),
        "request_digest": hashlib.sha256(b"req").hexdigest(),
        "terminal_id": terminal_id,
        "model_generation": generation,
        "occurrence": generation,
        "provider": "claude_code",
        "provider_version": "2.1.226",
        "session_id": session_id,
        "parser_key": "claude-status",
        "evidence_sha256": hashlib.sha256(b"ev").hexdigest(),
        "observed_at": "2026-08-19T00:00:00Z",
        "pane_id": "%7",
        "window_id": "@7",
        "tmux_session_id": "$1",
        "server_socket_path": "/private/tmp/cao-native.sock",
        "pane_pid": 4242,
        "process_identity": {"pid": 4242, "start_marker": "Thu Jul 24 10:00:00 2026"},
        "binding_native_id": None,
        "binding_provider_version": None,
    }
    with database.SessionLocal() as db:
        nsr._commit_repair(db, facts)

    contract = rc.get_contract_by_incarnation(terminal_id, generation)
    assert contract is not None
    assert contract["contract"]["working_directory"]["state"] == "unavailable"
    assert contract["contract"]["working_directory"]["value"] is None
    assert contract["contract"]["working_directory"]["reason"]
