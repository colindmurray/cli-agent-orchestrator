"""Focused M7 timer lifecycle tests for cond-0534."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.inbox import MessageStatus
from cli_agent_orchestrator.services import cohort_effects, generation_fence, inbox_service
from cli_agent_orchestrator.services import registered_waits as waits
from cli_agent_orchestrator.services import stable_agent_roster as roster
from cli_agent_orchestrator.services import wait_admission
from cli_agent_orchestrator.services.control_input_contract import ACCEPTED
from cli_agent_orchestrator.services.inbox_service import InboxService

SESSION = "cao-timer-tests"
NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _db(isolated_memory_db, monkeypatch):
    monkeypatch.delenv("CAO_M7_WAIT_REGISTRATION_ENABLED", raising=False)
    monkeypatch.delenv("CAO_M7_WAIT_CONSUMER_ENABLED", raising=False)
    return isolated_memory_db


def _bind(*, agent_id=None, suffix="1"):
    return roster.bind_generation(
        roster.BindingContract(
            agent_id=agent_id or str(uuid.uuid4()),
            session_name=SESSION,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="codex",
            native_session_id=f"native-{suffix}",
            acquisition_method="chosen_session_id",
            terminal_id=f"term-{suffix}",
            generation=str(uuid.uuid4()),
            pane_id=f"%{suffix}",
            pane_pid=8100 + int(suffix),
            process_identity={"pid": 8100 + int(suffix), "start_marker": f"m-{suffix}"},
            execution_mode="native_tui",
            admitted=True,
        )
    )


def _owner(bound):
    incarnation = bound["incarnation"]
    lineage = bound["lineage"]
    return wait_admission.WaitOwner(
        agent_id=incarnation["agent_id"],
        incarnation_id=incarnation["incarnation_id"],
        terminal_id=incarnation["terminal_id"],
        generation=incarnation["generation"],
        lineage_id=lineage["lineage_id"],
        native_session_id=lineage["native_session_id"],
    )


def _request(bound, *, operation_id=None, duration=60, name="coffee", estimated=30):
    return waits.RegistrationRequest(
        operation_id=operation_id or str(uuid.uuid4()),
        session_name=SESSION,
        project="proj",
        task_id="cond-0534",
        name=name,
        description="Resume the exact timer test after the scheduled boundary.",
        duration_seconds=duration,
        estimated_seconds=estimated,
        owner=_owner(bound),
    )


def _inbox_rows():
    with database.SessionLocal() as db:
        return db.query(database.InboxModel).all()


def _deliver_native_accepted(monkeypatch, bound):
    owner = _owner(bound)
    monkeypatch.setattr(
        inbox_service.managed_launch,
        "managed_control_identity",
        lambda terminal_id: {
            "reservation_id": f"rsv-{terminal_id}",
            "terminal_id": terminal_id,
            "generation": owner.generation,
            "provider": "codex",
            "state": "admitted",
            "controllable": True,
            "vintage": "v2",
            "execution_mode": "native_tui",
        },
    )
    monkeypatch.setattr(generation_fence, "installed_receipt", lambda *_args: None)
    monkeypatch.setattr(
        inbox_service.control_input_service,
        "deliver_native_inbox_payload",
        lambda *_args, **_kwargs: SimpleNamespace(outcome=ACCEPTED, reason_code="accepted"),
    )
    InboxService().deliver_pending(owner.terminal_id)


def test_happy_registration_status_list_restart_and_replay_adoption():
    bound = _bind()
    request = _request(bound)
    first = waits.register(request, now=NOW)
    replay = waits.register(request, now=NOW + timedelta(seconds=2))

    assert first["state"] == waits.STATE_ACKNOWLEDGED
    assert first["condition"]["name"] == "coffee"
    assert first["condition"]["description"]
    assert first["round"] == {
        "number": 1,
        "max_seconds": 60,
        "estimated_seconds": 30,
    }
    assert first["totals"] == {"elapsed_seconds": 0}
    assert replay["totals"] == {"elapsed_seconds": 2}
    assert replay["adopted"] is True
    assert replay["wait_id"] == first["wait_id"]
    persisted_owner = waits.get(first["wait_id"])["owner"]
    assert persisted_owner == {
        "project": "proj",
        "task_id": "cond-0534",
        "session_name": SESSION,
        "terminal_id": _owner(bound).terminal_id,
        "stable_agent_id": _owner(bound).agent_id,
        "incarnation_id": _owner(bound).incarnation_id,
        "generation": _owner(bound).generation,
        "lineage_id": _owner(bound).lineage_id,
        "native_session_id": _owner(bound).native_session_id,
    }
    assert waits.get_by_operation(request.operation_id)["wait_id"] == first["wait_id"]
    assert [row["wait_id"] for row in waits.list_waits(session_name=SESSION)] == [first["wait_id"]]

    without_estimate = waits.register(_request(bound, name="lunch", estimated=None), now=NOW)
    assert without_estimate["round"]["estimated_seconds"] is None


def test_registration_divergence_and_round_bound_refuse():
    bound = _bind()
    operation_id = str(uuid.uuid4())
    waits.register(_request(bound, operation_id=operation_id), now=NOW)
    with pytest.raises(waits.RegisteredWaitConflict):
        waits.register(_request(bound, operation_id=operation_id, name="different"), now=NOW)
    with pytest.raises(waits.RegisteredWaitInvalid):
        _request(bound, duration=waits.MAX_ROUND_SECONDS + 1)


def test_expiry_persists_one_exact_message_and_confirmed_receipt_resolves():
    bound = _bind()
    record = waits.register(_request(bound, duration=10), now=NOW)
    deliveries = []

    result = waits.process_due(
        now=NOW + timedelta(seconds=10),
        deliver=lambda terminal: deliveries.append(terminal),
        receipt_probe=lambda terminal, message: {
            "terminal_id": terminal,
            "message_id": str(message),
            "state": "submitted",
        },
    )

    assert result[-1]["state"] == waits.STATE_RESOLVED
    assert result[-1]["outcome"]["reason_code"] == "wake-confirmed"
    assert deliveries == [_owner(bound).terminal_id]
    assert len(_inbox_rows()) == 1
    assert _inbox_rows()[0].sender_id == _owner(bound).terminal_id
    assert _inbox_rows()[0].receiver_id == _owner(bound).terminal_id
    assert _inbox_rows()[0].sender_generation == _owner(bound).generation
    assert _inbox_rows()[0].expected_receiver_generation == _owner(bound).generation
    assert len(wait_admission.list_admissions(SESSION)) == 1

    # Restart/reconcile reads the terminal state; it neither recreates nor
    # resends the exact operation.
    assert (
        waits.process_due(
            now=NOW + timedelta(seconds=11),
            deliver=lambda terminal: deliveries.append(terminal),
        )
        == []
    )
    assert deliveries == [_owner(bound).terminal_id]
    assert waits.get(record["wait_id"])["state"] == waits.STATE_RESOLVED


def test_native_accepted_inbox_delivery_resolves_without_a_stronger_receipt(monkeypatch):
    bound = _bind()
    record = waits.register(_request(bound, duration=1), now=NOW)
    waits.process_due(now=NOW + timedelta(seconds=1))

    _deliver_native_accepted(monkeypatch, bound)
    assert _inbox_rows()[0].status == MessageStatus.DELIVERED

    result = waits.process_due(now=NOW + timedelta(seconds=2))[-1]
    assert result["state"] == waits.STATE_RESOLVED
    assert result["outcome"]["reason_code"] == "wake-delivered"
    assert waits.get(record["wait_id"])["state"] == waits.STATE_RESOLVED


def test_pending_response_loss_queries_same_operation_without_resending():
    bound = _bind()
    waits.register(_request(bound, duration=1), now=NOW)
    deliveries = []
    waits.process_due(
        now=NOW + timedelta(seconds=1),
        deliver=lambda terminal: deliveries.append(terminal),
    )
    waits.process_due(
        now=NOW + timedelta(seconds=20),
        deliver=lambda terminal: deliveries.append(terminal),
    )
    assert len(deliveries) == 1
    assert len(_inbox_rows()) == 1
    assert len(wait_admission.list_admissions(SESSION)) == 1


def test_cancellation_is_idempotent_and_prevents_later_wake():
    bound = _bind()
    record = waits.register(_request(bound, duration=10), now=NOW)
    cancel_id = str(uuid.uuid4())
    first = waits.cancel(record["wait_id"], operation_id=cancel_id, actor="codex:worker", now=NOW)
    replay = waits.cancel(record["wait_id"], operation_id=cancel_id, actor="codex:worker", now=NOW)
    assert first["state"] == waits.STATE_CANCELLED
    assert first["outcome"]["actor"] == "codex:worker"
    assert replay["adopted"] is True
    with pytest.raises(waits.RegisteredWaitConflict):
        waits.cancel(
            record["wait_id"],
            operation_id=cancel_id,
            actor="different-actor",
            now=NOW,
        )
    assert waits.process_due(now=NOW + timedelta(seconds=20)) == []
    assert _inbox_rows() == []


def test_cancellation_race_wins_while_exact_inbox_row_is_still_pending():
    bound = _bind()
    record = waits.register(_request(bound, duration=1), now=NOW)

    def cancel_during_delivery(_terminal):
        waits.cancel(
            record["wait_id"],
            operation_id=str(uuid.uuid4()),
            actor="codex:worker",
            now=NOW + timedelta(seconds=1),
        )

    waits.process_due(now=NOW + timedelta(seconds=1), deliver=cancel_during_delivery)
    final = waits.get(record["wait_id"])
    assert final["state"] == waits.STATE_CANCELLED
    assert _inbox_rows()[0].status == MessageStatus.FAILED


def test_refused_wake_invalidates_and_ambiguity_uses_grace_without_resend():
    bound = _bind()
    refused = waits.register(_request(bound, duration=1), now=NOW)
    waits.process_due(now=NOW + timedelta(seconds=1))
    with database.SessionLocal() as db:
        row = db.get(database.RegisteredWaitModel, refused["wait_id"])
        db.get(database.InboxModel, row.wake_message_id).status = MessageStatus.FAILED.value
        db.commit()
    result = waits.process_due(now=NOW + timedelta(seconds=2))[-1]
    assert result["state"] == waits.STATE_INVALID
    assert result["outcome"]["reason_code"] == "wake-refused"

    ambiguous = waits.register(_request(bound, duration=1, name="ambiguous"), now=NOW)
    waits.process_due(now=NOW + timedelta(seconds=1))
    pending = waits.get(ambiguous["wait_id"])
    assert pending["state"] == waits.STATE_EXPIRY_WAKE_PENDING
    result = waits.process_due(now=NOW + timedelta(seconds=62))[-1]
    assert result["state"] == waits.STATE_INVALID
    assert result["outcome"]["reason_code"] == "expiry-wake-ambiguous"
    assert len(_inbox_rows()) == 2
    assert _inbox_rows()[1].status == MessageStatus.FAILED


def test_failed_wake_settles_from_durable_evidence_before_owner_reverification(monkeypatch):
    bound = _bind()
    refused = waits.register(_request(bound, duration=1), now=NOW)
    waits.process_due(now=NOW + timedelta(seconds=1))
    with database.SessionLocal() as db:
        row = db.get(database.RegisteredWaitModel, refused["wait_id"])
        db.get(database.InboxModel, row.wake_message_id).status = MessageStatus.FAILED.value
        db.commit()

    def unavailable(*_args, **_kwargs):
        raise wait_admission.WaitAdmissionUnavailable("owner store unavailable")

    monkeypatch.setattr(wait_admission, "verify_owner", unavailable)
    result = waits.process_due(now=NOW + timedelta(seconds=2))[-1]

    assert result["state"] == waits.STATE_INVALID
    assert result["outcome"]["reason_code"] == "wake-refused"


def test_unreadable_due_wait_does_not_starve_a_later_valid_wait():
    bound = _bind()
    unreadable = waits.register(_request(bound, duration=1, name="unreadable"), now=NOW)
    valid = waits.register(_request(bound, duration=2, name="valid"), now=NOW)
    with database.SessionLocal() as db:
        db.get(database.RegisteredWaitModel, unreadable["wait_id"]).request_json = "{broken"
        db.commit()

    results = waits.process_due(now=NOW + timedelta(seconds=2))

    assert results[0]["wait_id"] == unreadable["wait_id"]
    assert results[0]["state"] == "unreadable"
    assert results[0]["reason_code"] == waits.RegisteredWaitUnavailable.code
    assert waits.get(valid["wait_id"])["state"] == waits.STATE_EXPIRY_WAKE_PENDING
    assert [row.receiver_id for row in _inbox_rows()] == [_owner(bound).terminal_id]


def test_due_scan_does_not_hide_an_unexpected_programming_failure(monkeypatch):
    bound = _bind()
    waits.register(_request(bound, duration=1), now=NOW)
    monkeypatch.setattr(
        waits,
        "_request_from_row",
        lambda _row: (_ for _ in ()).throw(AssertionError("unexpected defect")),
    )

    with pytest.raises(AssertionError, match="unexpected defect"):
        waits.process_due(now=NOW + timedelta(seconds=1))


def test_owner_replacement_invalidates_instead_of_transferring():
    first = _bind()
    owner = _owner(first)
    record = waits.register(_request(first, duration=1), now=NOW)
    roster.retire_incarnation(
        terminal_id=owner.terminal_id, generation=owner.generation, reason="replaced"
    )
    _bind(agent_id=owner.agent_id, suffix="2")
    result = waits.process_due(now=NOW + timedelta(seconds=1))[-1]
    assert result["wait_id"] == record["wait_id"]
    assert result["state"] == waits.STATE_INVALID
    assert result["outcome"]["reason_code"] == wait_admission.DENY_OWNER_REPLACED
    assert _inbox_rows() == []


def test_w8_hold_precedes_wake_then_release_uses_the_same_expiry_operation():
    bound = _bind()
    record = waits.register(_request(bound, duration=1), now=NOW)
    held = waits.process_due(
        now=NOW + timedelta(seconds=1), input_held=lambda _terminal, _generation: True
    )
    assert held == [{"wait_id": record["wait_id"], "state": waits.STATE_EXPIRY_INTENT}]
    assert _inbox_rows() == []

    waits.process_due(
        now=NOW + timedelta(seconds=2), input_held=lambda _terminal, _generation: False
    )
    assert len(_inbox_rows()) == 1
    assert len(wait_admission.list_admissions(SESSION)) == 1


def test_absent_and_unreadable_are_distinct_deadman_dispositions():
    bound = _bind()
    owner = _owner(bound)
    assert waits.deadman_disposition(owner.terminal_id, owner.generation) == {
        "state": "absent",
        "suppress_ordinary_deadman": False,
    }
    record = waits.register(_request(bound), now=NOW)
    with database.SessionLocal() as db:
        row = db.get(database.RegisteredWaitModel, record["wait_id"])
        row.request_json = "{broken"
        db.commit()
    disposition = waits.deadman_disposition(owner.terminal_id, owner.generation)
    assert disposition["state"] == "unreadable"
    assert disposition["suppress_ordinary_deadman"] is True


def test_stop_interrupts_active_waits_and_clears_suppression():
    bound = _bind()
    owner = _owner(bound)
    observed = datetime.now(timezone.utc)
    record = waits.register(_request(bound), now=observed)
    assert waits.deadman_disposition(owner.terminal_id, owner.generation)[
        "suppress_ordinary_deadman"
    ]
    results = cohort_effects._default_wait_interruptor(SESSION, str(uuid.uuid4()))
    assert results == [{"wait_id": record["wait_id"], "state": waits.STATE_INTERRUPTED}]
    assert (
        waits.deadman_disposition(owner.terminal_id, owner.generation)["suppress_ordinary_deadman"]
        is False
    )


def test_stop_waits_for_the_exact_inflight_delivery_stripe():
    bound = _bind()
    record = waits.register(_request(bound, duration=1), now=NOW)
    waits.process_due(now=NOW + timedelta(seconds=1))
    message_id = waits.get(record["wait_id"])["wake_message_id"]
    lock = InboxService._managed_delivery_lock(message_id)
    completed = threading.Event()
    results = []

    def interrupt():
        results.extend(cohort_effects._default_wait_interruptor(SESSION, str(uuid.uuid4())))
        completed.set()

    lock.acquire()
    thread = threading.Thread(target=interrupt)
    try:
        thread.start()
        assert completed.wait(timeout=0.1) is False
    finally:
        lock.release()
    thread.join(timeout=5)

    assert completed.is_set()
    assert results == [{"wait_id": record["wait_id"], "state": waits.STATE_INTERRUPTED}]
    assert _inbox_rows()[0].status == MessageStatus.FAILED


def test_reverse_rollback_disables_registration_while_consumer_drains(monkeypatch):
    bound = _bind()
    record = waits.register(_request(bound, duration=1), now=NOW)
    monkeypatch.setenv("CAO_M7_WAIT_REGISTRATION_ENABLED", "false")
    capability = waits.capability()
    assert capability["registration_enabled"] is False
    assert capability["consumer_attached"] is True
    with pytest.raises(waits.RegisteredWaitConflict):
        waits.register(_request(bound, name="new"), now=NOW)
    waits.process_due(
        now=NOW + timedelta(seconds=1), receipt_probe=lambda _terminal, _message: {"ok": True}
    )
    assert waits.get(record["wait_id"])["state"] == waits.STATE_RESOLVED


def test_deadman_absent_when_registered_waits_table_missing(isolated_memory_db):
    """Table genuinely absent proves vacuity: no table means no rows means writer never ran."""
    from sqlalchemy import text

    with isolated_memory_db.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS registered_waits"))
    result = waits.deadman_disposition("term-absent-0000", "gen-absent-0000")
    assert result == {"state": "absent", "suppress_ordinary_deadman": False}


def test_deadman_unreadable_when_registered_waits_table_has_incompatible_schema(
    isolated_memory_db,
):
    """Existing table with older/incompatible minimal schema is unreadable, not absent."""
    from sqlalchemy import text

    with isolated_memory_db.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS registered_waits"))
        conn.execute(text("CREATE TABLE registered_waits (wait_id TEXT PRIMARY KEY)"))
    result = waits.deadman_disposition("term-unreadable-0000", "gen-unreadable-0000")
    assert result["state"] == "unreadable"
    assert result["suppress_ordinary_deadman"] is True
    assert "detail" in result
