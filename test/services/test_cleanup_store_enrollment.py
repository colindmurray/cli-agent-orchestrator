"""Lane X2 — cleanup enrollment for campaign-era stores.

Each rule is exercised against a real SQLite database built from the ORM
metadata, so the retention predicates run against the actual schema and
fail-closed guards.  The four stores enrolled into ``cleanup_old_data``:

1. ``route_observation_operations`` — aged terminal results removable once
   their inbox wake claim is no longer pending (spec §5/§8 coordination).
2. ``restore_contracts`` — retired AND superseded incarnations past the
   retention age, unless a live reincarnation operation still reads them.
3. ``registered_waits`` + ``wait_message_admissions`` — aged terminal waits
   removable (their admission verdicts follow the wait), unless the wake is
   still pending or a monitor owns the wait.
4. ``wake_receipts`` / ``companion_receipts`` sidecar files — terminal /
   retired-generation sidecars past the retention age.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.inbox import MessageStatus
from cli_agent_orchestrator.services import cleanup_service


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _aged(days: int = 40) -> str:
    return _iso(datetime.now(timezone.utc) - timedelta(days=days))


def _fresh(days: int = 1) -> str:
    return _iso(datetime.now(timezone.utc) - timedelta(days=days))


def _sha(value: str) -> str:
    """A 64-hex digest-shaped value for a not-NULL digest column."""
    return (value * 64)[:64]


class _NullManager:
    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A real per-test database plus isolated sidecar dirs for one store."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'app.db'}", connect_args={"check_same_thread": False}
    )
    database.Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)
    monkeypatch.setattr(cleanup_service, "SessionLocal", session)
    monkeypatch.setattr(cleanup_service, "WAKE_RECEIPT_DIR", tmp_path / "wake-receipts")
    monkeypatch.setattr(cleanup_service, "COMPANION_DIR", tmp_path / "companion")
    # Keep the unrelated cleanup paths inert for any test that runs the full
    # pass rather than a single rule.
    monkeypatch.setattr(cleanup_service, "TERMINAL_LOG_DIR", tmp_path / "terminal-logs")
    monkeypatch.setattr(cleanup_service, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(cleanup_service, "fifo_manager", _NullManager())
    monkeypatch.setattr(cleanup_service, "status_monitor", _NullManager())
    try:
        yield session
    finally:
        engine.dispose()


def _insert(session, model, **kwargs):
    row = model(**kwargs)
    session.add(row)
    session.commit()
    return row


def _count(session, model):
    return session.query(model).count()


# ---------------------------------------------------------------------------
# Rule 1 — route_observation_operations
# ---------------------------------------------------------------------------


class TestRouteObservationRetention:
    def _op(self, *, operation_id: str, state: str, updated_at: str, inbox_message_id=None):
        return {
            "operation_id": operation_id,
            "schema_version": "cao-m10-route-observation-op-v1",
            "request_digest": _sha("req"),
            "target_terminal_id": "target-1",
            "target_generation": "gen-1",
            "native_session_id": "ns-1",
            "provider": "codex",
            "provider_version": "0.1",
            "provider_artifact_sha256": _sha("art"),
            "requester_terminal_id": "requester-1",
            "requester_generation": "gen-1",
            "state": state,
            "inbox_message_id": inbox_message_id,
            "created_at": updated_at,
            "updated_at": updated_at,
        }

    def test_removes_aged_terminal_without_pending_wake(self, env):
        session = env()
        _insert(
            session,
            database.RouteObservationOperationModel,
            **self._op(
                operation_id="11111111-1111-1111-1111-111111111111",
                state="observed-closed",
                updated_at=_aged(),
            ),
        )
        assert cleanup_service._prune_route_observation_operations() == 1
        assert _count(session, database.RouteObservationOperationModel) == 0

    def test_retains_fresh_terminal(self, env):
        session = env()
        _insert(
            session,
            database.RouteObservationOperationModel,
            **self._op(
                operation_id="11111111-1111-1111-1111-111111111111",
                state="observed-closed",
                updated_at=_fresh(),
            ),
        )
        assert cleanup_service._prune_route_observation_operations() == 0
        assert _count(session, database.RouteObservationOperationModel) == 1

    def test_retains_terminal_with_pending_inbox_wake(self, env):
        session = env()
        inbox = _insert(
            session,
            database.InboxModel,
            sender_id="target-1",
            receiver_id="requester-1",
            message="wake",
            status=MessageStatus.PENDING.value,
        )
        _insert(
            session,
            database.RouteObservationOperationModel,
            **self._op(
                operation_id="11111111-1111-1111-1111-111111111111",
                state="zero-effect-refusal",
                updated_at=_aged(),
                inbox_message_id=inbox.id,
            ),
        )
        assert cleanup_service._prune_route_observation_operations() == 0
        assert _count(session, database.RouteObservationOperationModel) == 1

    def test_removes_terminal_when_wake_delivered(self, env):
        session = env()
        inbox = _insert(
            session,
            database.InboxModel,
            sender_id="target-1",
            receiver_id="requester-1",
            message="wake",
            status=MessageStatus.DELIVERED.value,
        )
        _insert(
            session,
            database.RouteObservationOperationModel,
            **self._op(
                operation_id="11111111-1111-1111-1111-111111111111",
                state="ambiguous-after-possible-effect",
                updated_at=_aged(),
                inbox_message_id=inbox.id,
            ),
        )
        assert cleanup_service._prune_route_observation_operations() == 1
        assert _count(session, database.RouteObservationOperationModel) == 0

    def test_retains_nonterminal_requested(self, env):
        session = env()
        _insert(
            session,
            database.RouteObservationOperationModel,
            **self._op(
                operation_id="11111111-1111-1111-1111-111111111111",
                state="requested",
                updated_at=_aged(),
            ),
        )
        assert cleanup_service._prune_route_observation_operations() == 0
        assert _count(session, database.RouteObservationOperationModel) == 1


# ---------------------------------------------------------------------------
# Rule 2 — restore_contracts (retired AND superseded AND aged)
# ---------------------------------------------------------------------------


class TestRestoreContractRetention:
    def _incarnation(
        self,
        *,
        incarnation_id,
        terminal_id,
        generation,
        disposition,
        agent_id="11111111-1111-1111-1111-111111111111",
    ):
        return {
            "incarnation_id": incarnation_id,
            "agent_id": agent_id,
            "terminal_id": terminal_id,
            "generation": generation,
            "disposition": disposition,
            "created_at": _aged(),
            "updated_at": _aged(),
        }

    def _contract(
        self,
        *,
        contract_id,
        terminal_id,
        generation,
        agent_id,
        created_at,
        lineage_id="22222222-2222-2222-2222-222222222222",
    ):
        return {
            "contract_id": contract_id,
            "contract_digest": _sha("c"),
            "schema_version": "cao-m3-restore-contract-v1",
            "agent_id": agent_id,
            "lineage_id": lineage_id,
            "terminal_id": terminal_id,
            "generation": generation,
            "contract_json": "{}",
            "created_at": created_at,
        }

    def test_removes_retired_superseded_aged(self, env):
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(
                incarnation_id="inc-1", terminal_id="t-1", generation="g-1", disposition="retired"
            ),
        )
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-1",
                terminal_id="t-1",
                generation="g-1",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_aged(),
            ),
        )
        # Superseding: a newer contract for the same stable agent.
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-2",
                terminal_id="t-2",
                generation="g-2",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_fresh(),
            ),
        )
        assert cleanup_service._prune_restore_contracts() == 1
        assert _count(session, database.RestoreContractModel) == 1
        assert session.get(database.RestoreContractModel, "c-2") is not None

    def test_retains_live_incarnation_contract(self, env):
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(
                incarnation_id="inc-1", terminal_id="t-1", generation="g-1", disposition="bound"
            ),
        )
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-1",
                terminal_id="t-1",
                generation="g-1",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_aged(),
            ),
        )
        assert cleanup_service._prune_restore_contracts() == 0
        assert _count(session, database.RestoreContractModel) == 1

    def test_retains_live_incarnation_even_when_superseded(self, env):
        # The publish-before-retire crash window: the contract exists but the
        # roster incarnation is still live (bound).  Even though a newer
        # contract supersedes it for the same agent, the missing retirement
        # proof must preserve it — only the roster can retire an incarnation.
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(
                incarnation_id="inc-1", terminal_id="t-1", generation="g-1", disposition="bound"
            ),
        )
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-1",
                terminal_id="t-1",
                generation="g-1",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_aged(),
            ),
        )
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-2",
                terminal_id="t-2",
                generation="g-2",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_fresh(),
            ),
        )
        assert cleanup_service._prune_restore_contracts() == 0
        assert _count(session, database.RestoreContractModel) == 2

    def test_retains_retired_not_superseded(self, env):
        # A dormant agent with exactly one contract keeps it: nothing newer
        # supersedes it, so the resurrection basis must survive.
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(
                incarnation_id="inc-1", terminal_id="t-1", generation="g-1", disposition="retired"
            ),
        )
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-1",
                terminal_id="t-1",
                generation="g-1",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_aged(),
            ),
        )
        assert cleanup_service._prune_restore_contracts() == 0
        assert _count(session, database.RestoreContractModel) == 1

    def test_retains_retired_superseded_but_fresh(self, env):
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(
                incarnation_id="inc-1", terminal_id="t-1", generation="g-1", disposition="retired"
            ),
        )
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-1",
                terminal_id="t-1",
                generation="g-1",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_fresh(),
            ),
        )
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-2",
                terminal_id="t-2",
                generation="g-2",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_fresh(),
            ),
        )
        assert cleanup_service._prune_restore_contracts() == 0
        assert _count(session, database.RestoreContractModel) == 2

    def test_retains_when_live_reincarnation_operation_references_it(self, env):
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(
                incarnation_id="inc-1", terminal_id="t-1", generation="g-1", disposition="retired"
            ),
        )
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-1",
                terminal_id="t-1",
                generation="g-1",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_aged(),
            ),
        )
        _insert(
            session,
            database.RestoreContractModel,
            **self._contract(
                contract_id="c-2",
                terminal_id="t-2",
                generation="g-2",
                agent_id="11111111-1111-1111-1111-111111111111",
                created_at=_fresh(),
            ),
        )
        _insert(
            session,
            database.ReincarnationOperationModel,
            operation_id="33333333-3333-3333-3333-333333333333",
            request_digest=_sha("r"),
            schema_version="cao-m3-reincarnation-op-v1",
            session_name="s",
            agent_id="11111111-1111-1111-1111-111111111111",
            roster_revision=1,
            role="role",
            profile_family="family",
            lineage_id="22222222-2222-2222-2222-222222222222",
            harness="codex",
            native_session_id="ns-1",
            prior_terminal_id="t-1",
            prior_incarnation_id="inc-1",
            lifecycle_epoch=1,
            lifecycle_observation="observed",
            restore_contract_id="c-1",
            restore_contract_digest=_sha("c"),
            restore_contract_schema="v1",
            phase="claimed",
            request_json="{}",
            result_state="pending",
            created_at=_aged(),
            updated_at=_aged(),
        )
        assert cleanup_service._prune_restore_contracts() == 0
        assert _count(session, database.RestoreContractModel) == 2


# ---------------------------------------------------------------------------
# Rule 3 — registered_waits + wait_message_admissions
# ---------------------------------------------------------------------------


class TestRegisteredWaitRetention:
    def _wait(self, *, wait_id, state, updated_at, expiry_operation_id, wake_message_id=None):
        return {
            "wait_id": wait_id,
            "operation_id": f"op-{wait_id}",
            "request_digest": _sha("w"),
            "request_json": "{}",
            "session_name": "sess",
            "owner_agent_id": "11111111-1111-1111-1111-111111111111",
            "owner_incarnation_id": "inc-1",
            "owner_terminal_id": "t-1",
            "owner_generation": "g-1",
            "state": state,
            "deadline_at": _aged(60),
            "expiry_operation_id": expiry_operation_id,
            "wake_message_id": wake_message_id,
            "created_at": updated_at,
            "updated_at": updated_at,
        }

    def _admission(self, *, admission_id, operation_id):
        return {
            "admission_id": admission_id,
            "schema_version": "cao-wait-admission-v1",
            "message_schema_version": "cao-wait-message-v1",
            "operation_id": operation_id,
            "message_id": f"msg-{admission_id}",
            "session_name": "sess",
            "message_kind": "expiry",
            "owner_agent_id": "11111111-1111-1111-1111-111111111111",
            "owner_incarnation_id": "inc-1",
            "owner_terminal_id": "t-1",
            "owner_generation": "g-1",
            "owner_identity_digest": _sha("id"),
            "request_digest": _sha("a"),
            "message_digest": _sha("m"),
            "message_json": "{}",
            "admission_state": "admitted",
            "receipt_digest": _sha("rec"),
            "created_at": _aged(),
        }

    def test_removes_aged_terminal_wait_and_its_admissions(self, env):
        session = env()
        _insert(
            session,
            database.RegisteredWaitModel,
            **self._wait(
                wait_id="w-1", state="resolved", updated_at=_aged(), expiry_operation_id="exp-1"
            ),
        )
        _insert(
            session,
            database.WaitMessageAdmissionModel,
            **self._admission(admission_id="a-1", operation_id="exp-1"),
        )
        assert cleanup_service._prune_registered_waits() == 1
        assert _count(session, database.RegisteredWaitModel) == 0
        assert _count(session, database.WaitMessageAdmissionModel) == 0

    def test_retains_fresh_terminal_wait(self, env):
        session = env()
        _insert(
            session,
            database.RegisteredWaitModel,
            **self._wait(
                wait_id="w-1", state="resolved", updated_at=_fresh(), expiry_operation_id="exp-1"
            ),
        )
        assert cleanup_service._prune_registered_waits() == 0
        assert _count(session, database.RegisteredWaitModel) == 1

    def test_retains_nonterminal_wait(self, env):
        session = env()
        _insert(
            session,
            database.RegisteredWaitModel,
            **self._wait(
                wait_id="w-1", state="acknowledged", updated_at=_aged(), expiry_operation_id="exp-1"
            ),
        )
        assert cleanup_service._prune_registered_waits() == 0
        assert _count(session, database.RegisteredWaitModel) == 1

    def test_retains_terminal_wait_with_pending_wake(self, env):
        session = env()
        inbox = _insert(
            session,
            database.InboxModel,
            sender_id="t-1",
            receiver_id="t-1",
            message="wake",
            status=MessageStatus.PENDING.value,
        )
        _insert(
            session,
            database.RegisteredWaitModel,
            **self._wait(
                wait_id="w-1",
                state="resolved",
                updated_at=_aged(),
                expiry_operation_id="exp-1",
                wake_message_id=inbox.id,
            ),
        )
        assert cleanup_service._prune_registered_waits() == 0
        assert _count(session, database.RegisteredWaitModel) == 1

    def test_retains_monitor_owned_wait(self, env):
        session = env()
        _insert(
            session,
            database.RegisteredWaitModel,
            **self._wait(
                wait_id="w-1", state="resolved", updated_at=_aged(), expiry_operation_id="exp-1"
            ),
        )
        _insert(
            session,
            database.RegisteredWaitMonitorModel,
            wait_id="w-1",
            request_digest=_sha("m"),
            run_dir="/tmp/nonexistent",
            state="completed",
            created_at=_aged(),
            updated_at=_aged(),
        )
        assert cleanup_service._prune_registered_waits() == 0
        assert _count(session, database.RegisteredWaitModel) == 1

    def test_removes_admission_only_for_the_pruned_wait(self, env):
        # Another live wait's admission verdict must survive.
        session = env()
        _insert(
            session,
            database.RegisteredWaitModel,
            **self._wait(
                wait_id="w-1", state="resolved", updated_at=_aged(), expiry_operation_id="exp-1"
            ),
        )
        _insert(
            session,
            database.RegisteredWaitModel,
            **self._wait(
                wait_id="w-2",
                state="acknowledged",
                updated_at=_fresh(),
                expiry_operation_id="exp-2",
            ),
        )
        _insert(
            session,
            database.WaitMessageAdmissionModel,
            **self._admission(admission_id="a-1", operation_id="exp-1"),
        )
        _insert(
            session,
            database.WaitMessageAdmissionModel,
            **self._admission(admission_id="a-2", operation_id="exp-2"),
        )
        assert cleanup_service._prune_registered_waits() == 1
        assert _count(session, database.RegisteredWaitModel) == 1
        assert _count(session, database.WaitMessageAdmissionModel) == 1
        assert session.get(database.WaitMessageAdmissionModel, "a-2") is not None


# ---------------------------------------------------------------------------
# Rule 4 — wake_receipts / companion_receipts sidecar files
# ---------------------------------------------------------------------------


class TestWakeReceiptSweep:
    def _record(self, *, terminal_id, message_id, state):
        return {
            "schema": "cao-unmanaged-wake-receipt-v1",
            "schema_version": 1,
            "message_id": message_id,
            "terminal_id": terminal_id,
            "state": state,
            "source": "status-transition",
        }

    def _write(self, env, *, terminal_id, message_id, state, age_days):
        path = cleanup_service.WAKE_RECEIPT_DIR
        path.mkdir(parents=True, exist_ok=True)
        import json
        import os
        import time as _time

        file_path = path / f"{terminal_id}-{message_id}.json"
        file_path.write_text(
            json.dumps(self._record(terminal_id=terminal_id, message_id=message_id, state=state))
        )
        old = _time.time() - age_days * 86400
        os.utime(file_path, (old, old))
        return file_path

    def test_removes_aged_terminal_records(self, env):
        from cli_agent_orchestrator.services import wake_receipts

        self._write(
            env,
            terminal_id="t-1",
            message_id="m-1",
            state=wake_receipts.WAKE_CONFIRMED,
            age_days=40,
        )
        self._write(
            env,
            terminal_id="t-2",
            message_id="m-2",
            state=wake_receipts.WAKE_UNCONFIRMED,
            age_days=40,
        )
        assert cleanup_service._sweep_wake_receipts() == 2
        assert list(cleanup_service.WAKE_RECEIPT_DIR.glob("*.json")) == []

    def test_retains_fresh_terminal_record(self, env):
        from cli_agent_orchestrator.services import wake_receipts

        self._write(
            env, terminal_id="t-1", message_id="m-1", state=wake_receipts.WAKE_CONFIRMED, age_days=1
        )
        assert cleanup_service._sweep_wake_receipts() == 0
        assert len(list(cleanup_service.WAKE_RECEIPT_DIR.glob("*.json"))) == 1

    def test_retains_watching_record(self, env):
        from cli_agent_orchestrator.services import wake_receipts

        self._write(
            env, terminal_id="t-1", message_id="m-1", state=wake_receipts.WATCHING, age_days=40
        )
        assert cleanup_service._sweep_wake_receipts() == 0
        assert len(list(cleanup_service.WAKE_RECEIPT_DIR.glob("*.json"))) == 1


class TestOldDatabaseShapeGuard:
    """Older databases legitimately predate the enrolled tables.

    Every enrolled store must fail closed when its table does not exist:
    report zero and preserve, never raise through the startup pass.
    """

    @pytest.fixture
    def old_db(self, tmp_path, monkeypatch):
        engine = create_engine(
            f"sqlite:///{tmp_path / 'old.db'}", connect_args={"check_same_thread": False}
        )
        # Only the pre-existing inbox table exists — none of the four stores.
        database.Base.metadata.create_all(bind=engine, tables=[database.InboxModel.__table__])
        session = sessionmaker(bind=engine)
        monkeypatch.setattr(cleanup_service, "SessionLocal", session)
        monkeypatch.setattr(cleanup_service, "WAKE_RECEIPT_DIR", tmp_path / "wake-receipts")
        monkeypatch.setattr(cleanup_service, "COMPANION_DIR", tmp_path / "companion")
        try:
            yield session
        finally:
            engine.dispose()

    def test_all_stores_fail_closed_on_missing_tables(self, old_db):
        assert cleanup_service._prune_route_observation_operations() == 0
        assert cleanup_service._prune_restore_contracts() == 0
        assert cleanup_service._prune_registered_waits() == 0
        assert cleanup_service._sweep_wake_receipts() == 0
        assert cleanup_service._sweep_companion_receipts() == 0


class TestCompanionReceiptSweep:
    def _incarnation(self, *, terminal_id, generation, disposition):
        return {
            "incarnation_id": f"inc-{terminal_id}-{generation}",
            "agent_id": "11111111-1111-1111-1111-111111111111",
            "terminal_id": terminal_id,
            "generation": generation,
            "disposition": disposition,
            "created_at": _aged(),
            "updated_at": _aged(),
        }

    def _write(self, env, *, terminal_id, generation, age_days):
        path = cleanup_service.COMPANION_DIR
        path.mkdir(parents=True, exist_ok=True)
        import json
        import os
        import time as _time

        file_path = path / f"{terminal_id}-{generation}.json"
        file_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "terminal_id": terminal_id,
                    "generation": generation,
                    "message_acks": {},
                }
            )
        )
        old = _time.time() - age_days * 86400
        os.utime(file_path, (old, old))
        return file_path

    def test_removes_aged_retired_generation_record(self, env):
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(terminal_id="t-1", generation="g-1", disposition="retired"),
        )
        self._write(env, terminal_id="t-1", generation="g-1", age_days=40)
        assert cleanup_service._sweep_companion_receipts() == 1
        assert list(cleanup_service.COMPANION_DIR.glob("*.json")) == []

    def test_retains_live_generation_record(self, env):
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(terminal_id="t-1", generation="g-1", disposition="bound"),
        )
        self._write(env, terminal_id="t-1", generation="g-1", age_days=40)
        assert cleanup_service._sweep_companion_receipts() == 0
        assert len(list(cleanup_service.COMPANION_DIR.glob("*.json"))) == 1

    def test_retains_fresh_retired_generation_record(self, env):
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(terminal_id="t-1", generation="g-1", disposition="retired"),
        )
        self._write(env, terminal_id="t-1", generation="g-1", age_days=1)
        assert cleanup_service._sweep_companion_receipts() == 0
        assert len(list(cleanup_service.COMPANION_DIR.glob("*.json"))) == 1

    def test_retains_unverifiable_generation_record(self, env):
        # A generation absent from the roster cannot prove retirement.
        env()
        self._write(env, terminal_id="t-1", generation="g-1", age_days=40)
        assert cleanup_service._sweep_companion_receipts() == 0
        assert len(list(cleanup_service.COMPANION_DIR.glob("*.json"))) == 1

    def test_retains_record_without_generation_body(self, env):
        # A sidecar whose body never recorded its generation cannot be matched
        # to the roster incarnation; it must never fall through to a
        # NULL-generation match that deletes it on a guess.
        session = env()
        _insert(
            session,
            database.StableAgentIncarnationModel,
            **self._incarnation(terminal_id="t-1", generation=None, disposition="retired"),
        )
        path = cleanup_service.COMPANION_DIR
        path.mkdir(parents=True, exist_ok=True)
        import json
        import os
        import time as _time

        file_path = path / "t-1.json"
        file_path.write_text(json.dumps({"schema_version": 1, "terminal_id": "t-1"}))
        old = _time.time() - 40 * 86400
        os.utime(file_path, (old, old))
        assert cleanup_service._sweep_companion_receipts() == 0
        assert file_path.exists()
