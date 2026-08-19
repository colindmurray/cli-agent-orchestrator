"""Tests for surfacing task_occurrence_id in managed-launch v2 (cond-0518)."""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cli_agent_orchestrator.api.main import app
from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.managed_launch_v2 import (
    PROTOCOL_VERSION_V2,
    ManagedLaunchV2AdmitRequest,
    ManagedLaunchV2BindRequest,
    ManagedLaunchV2ReserveRequest,
)
from cli_agent_orchestrator.services import managed_launch_v2 as v2
from cli_agent_orchestrator.services import managed_provider_bridge as bridge
from cli_agent_orchestrator.services import vintage_migration as vm
from cli_agent_orchestrator.services.managed_provider_bridge import BRIDGE_VERSION

DELIVERY_ID = "33333333-3333-4333-8333-333333333333"


@pytest.fixture(autouse=True)
def _companion(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "COMPANION_DIR", tmp_path / "companion")


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


def _reserve_request(worktree, tmp_path, **changes):
    executable = tmp_path / "fake-provider"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "reservation_id": str(uuid.uuid4()),
        "session_name": "cao-test",
        "provider": "codex",
        "agent_profile": "reviewer-sol-max",
        "caller_id": "deadbeef",
        "working_directory": str(worktree),
        "trusted_project_root": str(worktree),
        "expected_model": "gpt-5.6-sol",
        "expected_effort": "xhigh",
        "provider_executable": str(executable),
        "provider_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "obligation_generation": "obgen-7c2e4a1b",
        "task_id": "self-heal-demo-task",
        "run_id": "run-0001",
        "delivery_id": DELIVERY_ID,
        "launch_nonce": "n" * 40,
    }
    payload.update(changes)
    return ManagedLaunchV2ReserveRequest(**payload)


def _ready_bridge_state(record, monkeypatch, **changes):
    session_id = f"thr_{uuid.uuid4().hex[:16]}"
    receipt = {
        "bridge_version": BRIDGE_VERSION,
        "receipt_id": session_id,
        "provider_session_id": session_id,
        "provider_receipt_kind": "codex-thread-start",
        "provider_transcript_sha256": "a" * 64,
        "provider_version": "0.146.0",
        "model_input_ready": True,
        "reservation_id": record["reservation_id"],
        "terminal_id": record["terminal_id"],
        "generation": record["generation"],
        "provider": "codex",
        "agent_profile": "reviewer-sol-max",
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "working_directory": record["working_directory"],
    }
    receipt.update(changes)
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.managed_provider_bridge.read_state",
        lambda rid: {"state": "ready", "readiness": receipt},
        raising=False,
    )
    return receipt


def _bind_request(record, **changes):
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "terminal_id": record["terminal_id"],
        "generation": record["generation"],
        "attempt_id": str(uuid.uuid4()),
    }
    payload.update(changes)
    return ManagedLaunchV2BindRequest(**payload)


def _admit_request(record, digest, **changes):
    message = "review the exact head"
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "delivery_id": DELIVERY_ID,
        "message": message,
        "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
        "sender_id": "deadbeef",
        "orchestration_type": "assign",
        "context": {
            "boot_id": "11111111-1111-4111-8111-111111111111",
            "project": "test-project",
            "task_id": "test-task",
            "run_id": "test-task",
            "task_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "dossier_sha256": "3" * 64,
            "lease_sha256": "4" * 64,
            "command_packet_sha256": "5" * 64,
            "source_chain_sha256": "6" * 64,
        },
        "native_binding_digest": digest,
    }
    payload.update(changes)
    return ManagedLaunchV2AdmitRequest(**payload)


def test_reserve_and_admit_surfaces_task_occurrence_id_when_provided(
    isolated_memory_db, worktree, tmp_path, monkeypatch
):
    occurrence_id = str(uuid.uuid4())
    request = _reserve_request(worktree, tmp_path, task_occurrence_id=occurrence_id)
    record, created = v2.reserve(request)
    assert created
    assert record["task_occurrence_id"] == occurrence_id
    assert v2.get(record["reservation_id"])["task_occurrence_id"] == occurrence_id

    launching_record, won = v2.claim_launch(record["reservation_id"])
    assert won
    assert launching_record["task_occurrence_id"] == occurrence_id

    _ready_bridge_state(record, monkeypatch)
    bound = v2.bind_native(record["reservation_id"], _bind_request(record))
    assert bound["state"] == "bound"
    assert bound["task_occurrence_id"] == occurrence_id

    digest = v2.native_binding_digest(bound)
    admit = _admit_request(bound, digest)
    claimed, should_send = v2.claim_admission(record["reservation_id"], admit)
    assert should_send
    assert claimed["state"] == "admitting"
    assert claimed["task_occurrence_id"] == occurrence_id

    receipt = {
        "receipt_id": "turn-1",
        "provider_session_id": bound["binding"]["native_session_id"],
        "provider_turn_id": "turn-1",
        "provider_receipt_kind": "codex-turn-start",
    }
    completed = v2.complete_admission(record["reservation_id"], admit.delivery_id, receipt)
    assert completed["state"] == "admitted"
    assert completed["task_occurrence_id"] == occurrence_id


def test_reserve_and_admit_absent_task_occurrence_id_is_none_and_succeeds(
    isolated_memory_db, worktree, tmp_path, monkeypatch
):
    request = _reserve_request(worktree, tmp_path)
    assert request.task_occurrence_id is None

    record, created = v2.reserve(request)
    assert created
    assert record["task_occurrence_id"] is None
    assert v2.get(record["reservation_id"])["task_occurrence_id"] is None

    launching_record, won = v2.claim_launch(record["reservation_id"])
    assert won
    assert launching_record["task_occurrence_id"] is None

    _ready_bridge_state(record, monkeypatch)
    bound = v2.bind_native(record["reservation_id"], _bind_request(record))
    assert bound["state"] == "bound"
    assert bound["task_occurrence_id"] is None

    digest = v2.native_binding_digest(bound)
    admit = _admit_request(bound, digest)
    claimed, should_send = v2.claim_admission(record["reservation_id"], admit)
    assert should_send
    assert claimed["state"] == "admitting"
    assert claimed["task_occurrence_id"] is None

    receipt = {
        "receipt_id": "turn-1",
        "provider_session_id": bound["binding"]["native_session_id"],
        "provider_turn_id": "turn-1",
        "provider_receipt_kind": "codex-turn-start",
    }
    completed = v2.complete_admission(record["reservation_id"], admit.delivery_id, receipt)
    assert completed["state"] == "admitted"
    assert completed["task_occurrence_id"] is None


def test_reserve_request_validation_for_task_occurrence_id(worktree, tmp_path):
    valid_id = str(uuid.uuid4())
    req = _reserve_request(worktree, tmp_path, task_occurrence_id=valid_id)
    assert req.task_occurrence_id == valid_id

    req_none = _reserve_request(worktree, tmp_path, task_occurrence_id=None)
    assert req_none.task_occurrence_id is None

    with pytest.raises(ValueError):
        _reserve_request(worktree, tmp_path, task_occurrence_id="not-a-uuid")

    with pytest.raises(ValueError):
        _reserve_request(worktree, tmp_path, task_occurrence_id=valid_id.upper())


def test_vintage_migration_creates_and_backfills_task_occurrence_id_column(tmp_path):
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE managed_launch_v2_reservations ("
            "reservation_id TEXT PRIMARY KEY, terminal_id TEXT NOT NULL UNIQUE, "
            "generation TEXT NOT NULL UNIQUE, protocol_vintage TEXT NOT NULL DEFAULT 'v2' "
            "CHECK (protocol_vintage = 'v2'), session_name TEXT NOT NULL, "
            "provider TEXT NOT NULL, agent_profile TEXT NOT NULL, caller_id TEXT NOT NULL, "
            "working_directory TEXT NOT NULL, trusted_project_root TEXT, "
            "obligation_generation TEXT NOT NULL, task_id TEXT, run_id TEXT NOT NULL, "
            "launch_nonce_digest TEXT NOT NULL, state TEXT NOT NULL, request_json TEXT NOT NULL, "
            "binding_json TEXT, admission_json TEXT, created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()

    vm.migrate_v2(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(managed_launch_v2_reservations)")
        }
        assert "task_occurrence_id" in columns
    finally:
        conn.close()


def test_row_dict_surfaces_none_for_legacy_row_missing_task_occurrence_id(
    isolated_memory_db, worktree, tmp_path
):
    request = _reserve_request(worktree, tmp_path)
    record, _ = v2.reserve(request)

    class _LegacyRow:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            if name == "task_occurrence_id":
                raise AttributeError(name)
            return getattr(self._real, name)

    with database.SessionLocal() as db:
        row = v2._query(db, record["reservation_id"])
        projected = v2._row_dict(_LegacyRow(row))

    assert "task_occurrence_id" in projected
    assert projected["task_occurrence_id"] is None


def test_http_api_v2_reservations_surface_task_occurrence_id(
    isolated_memory_db, worktree, tmp_path
):
    client = TestClient(app, base_url="http://localhost")
    occurrence_id = str(uuid.uuid4())

    executable = tmp_path / "fake-provider"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)

    payload_with_occ = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "reservation_id": str(uuid.uuid4()),
        "session_name": "cao-test",
        "provider": "codex",
        "agent_profile": "reviewer-sol-max",
        "caller_id": "deadbeef",
        "working_directory": str(worktree),
        "trusted_project_root": str(worktree),
        "expected_model": "gpt-5.6-sol",
        "expected_effort": "xhigh",
        "provider_executable": str(executable),
        "provider_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "obligation_generation": "obgen-7c2e4a1b",
        "task_id": "self-heal-demo-task",
        "task_occurrence_id": occurrence_id,
        "run_id": "run-0001",
        "delivery_id": str(uuid.uuid4()),
        "launch_nonce": "n" * 40,
    }

    res = client.post("/managed-launch/v2/reservations", json=payload_with_occ)
    assert res.status_code == 201, res.text
    data = res.json()
    assert data["task_occurrence_id"] == occurrence_id

    get_res = client.get(f"/managed-launch/v2/reservations/{payload_with_occ['reservation_id']}")
    assert get_res.status_code == 200
    assert get_res.json()["task_occurrence_id"] == occurrence_id

    payload_without_occ = dict(payload_with_occ)
    payload_without_occ["reservation_id"] = str(uuid.uuid4())
    payload_without_occ["delivery_id"] = str(uuid.uuid4())
    del payload_without_occ["task_occurrence_id"]

    res2 = client.post("/managed-launch/v2/reservations", json=payload_without_occ)
    assert res2.status_code == 201, res2.text
    data2 = res2.json()
    assert data2["task_occurrence_id"] is None

    get_res2 = client.get(
        f"/managed-launch/v2/reservations/{payload_without_occ['reservation_id']}"
    )
    assert get_res2.status_code == 200
    assert get_res2.json()["task_occurrence_id"] is None
