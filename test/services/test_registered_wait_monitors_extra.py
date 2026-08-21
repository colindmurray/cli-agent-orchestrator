"""Deterministic state-machine tests for registered wait monitors."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.inbox import MessageStatus
from cli_agent_orchestrator.services import registered_wait_monitors as monitors
from cli_agent_orchestrator.services import (
    registered_waits,
)
from cli_agent_orchestrator.services import stable_agent_roster as roster
from cli_agent_orchestrator.services import (
    wait_admission,
)
from cli_agent_orchestrator.services.registered_waits import RegistrationRequest
from cli_agent_orchestrator.services.wait_runner import (
    RESULT_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    compute_sha256,
)

NOW = datetime(2026, 8, 21, 8, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path}/state.db", connect_args={"check_same_thread": False}
    )
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(
        database,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=engine),
    )
    database.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(monitors, "CAO_HOME_DIR", tmp_path)
    monkeypatch.setenv("CAO_WAIT_RUNNER_FAKE_MARKER", "test-marker")
    monkeypatch.setenv("CAO_M7_WAIT_MONITOR_CONSUMER_ENABLED", "true")
    yield
    engine.dispose()


def _owner(session: str, terminal: str = "term-1") -> wait_admission.WaitOwner:
    bound = roster.bind_generation(
        roster.BindingContract(
            agent_id=str(uuid.uuid4()),
            session_name=session,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="codex",
            native_session_id=f"native-{uuid.uuid4()}",
            acquisition_method="chosen_session_id",
            terminal_id=terminal,
            generation=str(uuid.uuid4()),
            pane_id="%1",
            pane_pid=9001,
            process_identity={"pid": 9001, "start_marker": "owner-marker"},
            execution_mode="native_tui",
            admitted=True,
        )
    )
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


def _adapter(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    executable = root / "probe"
    executable.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    executable.chmod(0o755)
    return {
        "kind": "process",
        "executable": str(executable),
        "executable_sha256": compute_sha256(str(executable)),
        "cwd": str(root),
        "argv": [str(executable)],
    }


def _pending(
    tmp_path, monkeypatch, name: str, duration: int = 60, terminal: str = "term-1"
) -> dict[str, Any]:
    monkeypatch.setattr(monitors, "launch_dormant_runner", lambda _paths: None)
    request = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name=name,
        project="p",
        task_id="t",
        name=name,
        description="deterministic monitor test",
        duration_seconds=duration,
        owner=_owner(name, terminal),
        adapter=_adapter(tmp_path / name),
    )
    return registered_waits.register(request, now=NOW)


def _monitor(wait_id: str) -> Any:
    with database.SessionLocal() as db:
        return db.get(database.RegisteredWaitMonitorModel, wait_id)


def _paths(wait_id: str) -> dict[str, Path]:
    return monitors.monitor_paths_for_monitor(_monitor(wait_id))


def _spec(wait_id: str) -> dict[str, Any]:
    return json.loads(_paths(wait_id)["spec"].read_text(encoding="utf-8"))


def _ready(wait_id: str, *, child: bool = False) -> dict[str, Any]:
    spec = _spec(wait_id)
    value = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "wait_id": wait_id,
        "request_digest": spec["request_digest"],
        "pid": 1111,
        "start_marker": "helper-1111",
        "phase": "running" if child else "waiting-for-activation",
        "started_at": "2026-08-21T08:00:00Z",
        "adapter": "process",
        "timeout_seconds": spec["timeout_seconds"],
    }
    if child:
        value.update(child_pid=2222, child_start_marker="child-2222", pgid=2222)
    return value


def _result(wait_id: str) -> dict[str, Any]:
    spec = _spec(wait_id)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "wait_id": wait_id,
        "request_digest": spec["request_digest"],
        "adapter": spec["adapter"],
        "outcome": "completed",
        "reason": "process-exit",
        "observed": {"exit_code": 0},
        "started_at": "2026-08-21T08:00:00Z",
        "finished_at": "2026-08-21T08:00:01Z",
        "elapsed_seconds": 1.0,
        "process": {"exit_code": 0},
    }


def _write_result(wait_id: str) -> None:
    _paths(wait_id)["result"].write_text(json.dumps(_result(wait_id)), encoding="utf-8")


def _attach(receipt: dict[str, Any]) -> Any:
    return lambda _result: receipt


def _good_attachment(result: dict[str, Any]) -> dict[str, str]:
    return {
        "communication_id": "communication-1",
        "attachment_id": "attachment-1",
        "digest": result["result_digest"],
    }


def _inbox_count() -> int:
    with database.SessionLocal() as db:
        return db.query(database.InboxModel).count()


def test_ready_and_result_schemas_are_exact():
    spec = {
        "wait_id": "wait-1",
        "request_digest": "a" * 64,
        "timeout_seconds": 60,
        "adapter": {"kind": "process"},
    }
    ready = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "wait_id": "wait-1",
        "request_digest": "a" * 64,
        "pid": 1,
        "start_marker": "marker",
        "phase": "waiting-for-activation",
        "started_at": "2026-08-21T08:00:00Z",
        "adapter": "process",
        "timeout_seconds": 60,
    }
    assert monitors._validate_ready(ready, spec) == ready
    for mutation in (
        {**ready, "schema_version": "future"},
        {**ready, "timeout_seconds": 61},
        {**ready, "extra": True},
        {**ready, "pid": True},
        {**ready, "child_pid": "2"},
    ):
        assert monitors._validate_ready(mutation, spec) is None

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "wait_id": "wait-1",
        "request_digest": "a" * 64,
        "adapter": {"kind": "process"},
        "outcome": "completed",
        "reason": "process-exit",
        "observed": {},
        "started_at": "a",
        "finished_at": "b",
        "elapsed_seconds": 1,
    }
    assert monitors._validate_result(result, spec) == result
    assert monitors._validate_result({**result, "schema_version": "future"}, spec) is None
    assert monitors._validate_result({**result, "foreign": True}, spec) is None
    assert monitors._validate_result({**result, "reason": ""}, spec) is None


def test_no_ack_without_exact_ready_and_replay_never_relaunches(tmp_path, monkeypatch):
    launches = []
    monkeypatch.setattr(monitors, "launch_dormant_runner", lambda paths: launches.append(paths))
    owner = _owner("ambiguous")
    request = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="ambiguous",
        project="p",
        task_id="t",
        name="ambiguous",
        description="ambiguous launch",
        duration_seconds=60,
        owner=owner,
        adapter=_adapter(tmp_path / "ambiguous"),
    )
    first = registered_waits.register(request, now=NOW)
    second = registered_waits.register(request, now=NOW + timedelta(seconds=1))
    assert first["state"] == second["state"] == "registration-pending"
    assert first["monitor_health"] == "unmonitored"
    assert second["adopted"] is True
    assert len(launches) == 1


def test_launch_intent_has_bounded_stale_exit(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "deadline", duration=10)
    before = registered_waits.process_monitors(now=NOW + timedelta(seconds=9))
    assert before[0]["state"] == "launch-intent"
    assert registered_waits.get(record["wait_id"])["state"] == "registration-pending"
    after = registered_waits.process_monitors(now=NOW + timedelta(seconds=10))
    assert after[0]["state"] == "invalid"
    final = registered_waits.get(record["wait_id"])
    assert final["state"] == "invalid"
    assert final["outcome"]["reason_code"] == "monitor-stale"


def test_input_hold_before_attachment_has_zero_effect(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "hold-before")
    _write_result(record["wait_id"])
    attachments = []
    result = registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1),
        attach_result=lambda value: attachments.append(value),
        input_held=lambda _terminal, _generation: True,
    )
    assert result[0]["detail"] == "input-held"
    assert attachments == []
    assert _inbox_count() == 0
    assert _monitor(record["wait_id"]).state == "result-ready"


def test_input_hold_after_attachment_has_zero_wake_effect(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "hold-after")
    _write_result(record["wait_id"])
    registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1), attach_result=_good_attachment
    )
    with database.SessionLocal() as db:
        monitor = db.get(database.RegisteredWaitMonitorModel, record["wait_id"])
        monitor.state = "result-ready"
        monitor.wake_message_id = None
        monitor.wake_pending_since = None
        db.query(database.InboxModel).delete()
        db.commit()
    holds = iter((False, True))
    result = registered_waits.process_monitors(
        now=NOW + timedelta(seconds=2),
        attach_result=_good_attachment,
        input_held=lambda _terminal, _generation: next(holds),
    )
    assert result[0]["detail"] == "input-held"
    assert _inbox_count() == 0


def test_wrong_attachment_digest_never_creates_wake(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "digest")
    _write_result(record["wait_id"])
    result = registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1),
        attach_result=_attach(
            {
                "communication_id": "communication-1",
                "attachment_id": "attachment-1",
                "digest": "0" * 64,
            }
        ),
    )
    assert result[0]["detail"] == "attachment-receipt-invalid"
    assert _inbox_count() == 0
    assert _monitor(record["wait_id"]).state == "result-ready"


def test_delivery_is_once_and_wake_ambiguity_is_bounded(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "delivery")
    _write_result(record["wait_id"])
    deliveries = []
    kwargs = {
        "attach_result": _good_attachment,
        "deliver": lambda terminal: deliveries.append(terminal),
    }
    first = registered_waits.process_monitors(now=NOW + timedelta(seconds=1), **kwargs)
    assert first[0]["state"] == "wake-pending"
    registered_waits.process_monitors(now=NOW + timedelta(seconds=30), **kwargs)
    assert deliveries == [record["owner"]["terminal_id"]]
    final = registered_waits.process_monitors(now=NOW + timedelta(seconds=61), **kwargs)
    assert final[0]["state"] == "invalid"
    wait = registered_waits.get(record["wait_id"])
    assert wait["outcome"]["reason_code"] == "monitor-wake-ambiguous"


def test_owner_is_reverified_before_attachment(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "owner")
    _write_result(record["wait_id"])
    attachments = []
    monkeypatch.setattr(
        wait_admission,
        "verify_owner",
        lambda _owner, db=None: {
            "denial_reason": wait_admission.DENY_OWNER_RETIRED,
            "detail": "retired before result delivery",
        },
    )
    result = registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1),
        attach_result=lambda value: attachments.append(value),
    )
    assert result[0]["state"] == "invalid"
    assert attachments == []
    assert _inbox_count() == 0
    assert registered_waits.get(record["wait_id"])["outcome"]["reason_code"] == "owner-retired"


def test_wake_admission_preserves_the_observed_denial_reason(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "wake-denial")
    _write_result(record["wait_id"])
    original_admit = wait_admission.admit

    def deny_wake(request, *, db=None):
        if request.message.kind == wait_admission.KIND_WORKER_WAKE:
            return {
                "admission_state": wait_admission.STATE_DENIED,
                "denial_reason": wait_admission.DENY_OWNER_RETIRED,
                "detail": "owner retired before wake admission",
            }
        return original_admit(request, db=db)

    monkeypatch.setattr(wait_admission, "admit", deny_wake)
    result = registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1), attach_result=_good_attachment
    )

    assert result[0]["state"] == "invalid"
    assert registered_waits.get(record["wait_id"])["outcome"]["reason_code"] == "owner-retired"


@pytest.mark.parametrize("verb", ["cancel", "interrupt"])
def test_failed_monitor_wake_is_settled_not_mistaken_for_delivery(tmp_path, monkeypatch, verb):
    record = _pending(tmp_path, monkeypatch, f"failed-{verb}")
    _write_result(record["wait_id"])
    registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1), attach_result=_good_attachment
    )
    with database.SessionLocal() as db:
        monitor = db.get(database.RegisteredWaitMonitorModel, record["wait_id"])
        db.get(database.InboxModel, monitor.wake_message_id).status = MessageStatus.FAILED.value
        db.commit()

    if verb == "cancel":
        registered_waits.cancel(record["wait_id"], operation_id=str(uuid.uuid4()), actor="test")
    else:
        registered_waits.interrupt_session_waits(record["owner"]["session_name"], str(uuid.uuid4()))

    final = registered_waits.get(record["wait_id"])
    assert final["state"] == "invalid"
    assert final["outcome"]["reason_code"] == "wake-refused"


def test_timer_consumer_skips_adapter_even_after_deadline(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "timer-skip", duration=1)
    results = registered_waits.process_due(now=NOW + timedelta(seconds=2))
    assert results == []
    assert registered_waits.get(record["wait_id"])["state"] == "registration-pending"
    assert _inbox_count() == 0


def test_missing_helper_is_stale_not_monitored(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "missing-helper")
    with database.SessionLocal() as db:
        monitor = db.get(database.RegisteredWaitMonitorModel, record["wait_id"])
        wait = db.get(database.RegisteredWaitModel, record["wait_id"])
        monitor.state = "active"
        monitor.helper_pid = None
        monitor.helper_start_marker = None
        wait.state = "acknowledged"
        db.commit()
    result = registered_waits.process_monitors(now=NOW + timedelta(seconds=1))
    assert result[0]["health"] == "monitor-stale"
    assert result[0]["detail"] == "missing-helper-no-result"
    assert registered_waits.get(record["wait_id"])["state"] == "invalid"


def test_stop_refreshes_and_terminates_child_then_helper(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "stop")
    ready = _ready(record["wait_id"], child=True)
    _paths(record["wait_id"])["ready"].write_text(json.dumps(ready), encoding="utf-8")
    with database.SessionLocal() as db:
        monitor = db.get(database.RegisteredWaitMonitorModel, record["wait_id"])
        wait = db.get(database.RegisteredWaitModel, record["wait_id"])
        monitor.state = "active"
        monitor.helper_pid = ready["pid"]
        monitor.helper_start_marker = ready["start_marker"]
        wait.state = "acknowledged"
        db.commit()
    alive = {1111: True, 2222: True}
    terminated = []
    monkeypatch.setattr(monitors, "_helper_alive", lambda pid, _marker: alive[pid])
    monkeypatch.setattr(monitors, "_group_absent", lambda pgid: not alive[pgid])

    def terminate(pgid, grace=2.0):
        terminated.append(pgid)
        alive[pgid] = False
        return True

    monkeypatch.setattr(monitors, "_terminate_pgid", terminate)
    result = monitors.stop_monitor(record["wait_id"], str(uuid.uuid4()), "test")
    assert terminated == [2222, 1111]
    assert result["state"] == "interrupted-by-stop"
    stored = _monitor(record["wait_id"])
    assert (stored.child_pid, stored.child_start_marker, stored.pgid) == (
        2222,
        "child-2222",
        2222,
    )


@pytest.mark.parametrize(
    ("message_status", "expected_state", "reason"),
    [
        (MessageStatus.DELIVERED.value, "resolved", "wake-delivered"),
        (MessageStatus.FAILED.value, "invalid", "wake-refused"),
    ],
)
def test_delivered_and_failed_wakes_have_distinct_outcomes(
    tmp_path, monkeypatch, message_status, expected_state, reason
):
    record = _pending(tmp_path, monkeypatch, f"wake-{message_status}")
    _write_result(record["wait_id"])
    registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1), attach_result=_good_attachment
    )
    with database.SessionLocal() as db:
        monitor = db.get(database.RegisteredWaitMonitorModel, record["wait_id"])
        inbox = db.get(database.InboxModel, monitor.wake_message_id)
        inbox.status = message_status
        db.commit()
    registered_waits.process_monitors(
        now=NOW + timedelta(seconds=2), attach_result=_good_attachment
    )
    final = registered_waits.get(record["wait_id"])
    assert final["state"] == expected_state
    assert final["outcome"]["reason_code"] == reason


def test_unreadable_spec_and_request_are_not_absence(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "unreadable")
    _paths(record["wait_id"])["spec"].write_text("{broken", encoding="utf-8")
    health = monitors.monitor_health_for(record["wait_id"])
    assert health["health"] == "unmonitored"
    assert health["detail"] == "unreadable:spec"
    result = registered_waits.process_monitors(now=NOW + timedelta(seconds=1))
    assert result[0] == {
        "wait_id": record["wait_id"],
        "state": "unreadable",
        "detail": "unreadable:spec",
    }
    assert registered_waits.get(record["wait_id"])["state"] == "registration-pending"


def test_get_monitor_projects_durable_nonsecret_state(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "projection")
    projected = monitors.get_monitor(record["wait_id"])
    assert projected["wait_id"] == record["wait_id"]
    assert projected["operation_id"] == record["operation_id"]
    assert projected["adapter_kind"] == "process"
    assert projected["state"] == "launch-intent"
    assert projected["run_dir"] == str(_paths(record["wait_id"])["dir"])
    assert "created_at" not in projected
    assert "updated_at" not in projected
    assert monitors.get_monitor(str(uuid.uuid4())) is None


def test_adapter_cancel_records_cancelled_while_stop_records_interrupted(tmp_path, monkeypatch):
    cancelled = _pending(tmp_path, monkeypatch, "cancel-vocabulary")
    operation_id = str(uuid.uuid4())
    result = registered_waits.cancel(
        cancelled["wait_id"], operation_id=operation_id, actor="worker"
    )
    assert result["state"] == "cancelled"
    assert _monitor(cancelled["wait_id"]).state == "cancelled"
    assert (
        registered_waits.cancel(cancelled["wait_id"], operation_id=operation_id, actor="worker")[
            "adopted"
        ]
        is True
    )
    with pytest.raises(registered_waits.RegisteredWaitConflict, match="divergent replay"):
        registered_waits.cancel(
            cancelled["wait_id"], operation_id=operation_id, actor="other-worker"
        )

    stopped = _pending(tmp_path, monkeypatch, "stop-vocabulary", terminal="term-2")
    result = registered_waits.interrupt_session_waits("stop-vocabulary", str(uuid.uuid4()))
    assert result == [{"wait_id": stopped["wait_id"], "state": "interrupted-by-stop"}]
    assert _monitor(stopped["wait_id"]).state == "interrupted-by-stop"


@pytest.mark.parametrize(
    ("inbox_status", "state", "reason"),
    [
        (MessageStatus.DELIVERED.value, "resolved", "wake-delivered"),
        (MessageStatus.FAILED.value, "invalid", "wake-refused"),
    ],
)
def test_installed_monitor_wake_settles_before_owner_reverification(
    tmp_path, monkeypatch, inbox_status, state, reason
):
    record = _pending(tmp_path, monkeypatch, f"settle-{inbox_status}")
    _write_result(record["wait_id"])
    registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1), attach_result=_good_attachment
    )
    with database.SessionLocal() as db:
        monitor = db.get(database.RegisteredWaitMonitorModel, record["wait_id"])
        db.get(database.InboxModel, monitor.wake_message_id).status = inbox_status
        db.commit()
    monkeypatch.setattr(
        wait_admission,
        "verify_owner",
        lambda _owner, db=None: {
            "denial_reason": wait_admission.DENY_OWNER_RETIRED,
            "detail": "owner retired after wake installation",
        },
    )

    registered_waits.process_monitors(now=NOW + timedelta(seconds=2))

    final = registered_waits.get(record["wait_id"])
    assert final["state"] == state
    assert final["outcome"]["reason_code"] == reason


def test_active_adoption_repairs_missing_activation_after_commit_crash(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "activation-repair")
    paths = _paths(record["wait_id"])
    paths["ready"].write_text(json.dumps(_ready(record["wait_id"])), encoding="utf-8")
    monkeypatch.setattr(monitors, "_helper_alive", lambda _pid, _marker: True)
    assert monitors.adopt_monitor_evidence(record["wait_id"])["state"] == "active"
    paths["activate"].unlink()

    assert monitors.adopt_monitor_evidence(record["wait_id"])["state"] == "active"
    assert paths["activate"].exists()


def test_terminal_commit_between_wake_creation_and_install_prevents_late_delivery(
    tmp_path, monkeypatch
):
    record = _pending(tmp_path, monkeypatch, "wake-cas")
    _write_result(record["wait_id"])
    deliveries = []

    def terminal_wins(db, wait_row, monitor, _request):
        wait_id = wait_row.wait_id
        db.rollback()
        with database.SessionLocal() as winner:
            current_wait = winner.get(database.RegisteredWaitModel, wait_id)
            current_monitor = winner.get(database.RegisteredWaitMonitorModel, wait_id)
            current_wait.state = "cancelled"
            current_monitor.state = "cancelled"
            inbox = database.InboxModel(
                sender_id=current_wait.owner_terminal_id,
                receiver_id=current_wait.owner_terminal_id,
                message="must not deliver",
                status=MessageStatus.PENDING.value,
            )
            winner.add(inbox)
            winner.commit()
            return inbox.id

    monkeypatch.setattr(monitors, "_create_wake", terminal_wins)
    registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1),
        attach_result=_good_attachment,
        deliver=deliveries.append,
    )

    assert deliveries == []
    assert registered_waits.get(record["wait_id"])["state"] == "cancelled"
    assert _monitor(record["wait_id"]).state == "cancelled"


def test_wake_installation_and_cancel_share_the_wait_stripe(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "wake-stripe")
    _write_result(record["wait_id"])
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()
    original_create = monitors._create_wake

    def paused_create(*args):
        entered.set()
        assert release.wait(timeout=5)
        return original_create(*args)

    monkeypatch.setattr(monitors, "_create_wake", paused_create)
    processor = threading.Thread(
        target=lambda: registered_waits.process_monitors(
            now=NOW + timedelta(seconds=1), attach_result=_good_attachment
        )
    )

    def cancel_wait():
        registered_waits.cancel(record["wait_id"], operation_id=str(uuid.uuid4()), actor="worker")
        cancelled.set()

    processor.start()
    assert entered.wait(timeout=5)
    canceller = threading.Thread(target=cancel_wait)
    canceller.start()
    serialized = not cancelled.wait(timeout=0.2)
    release.set()
    processor.join(timeout=5)
    canceller.join(timeout=5)

    assert serialized
    assert not processor.is_alive()
    assert not canceller.is_alive()
    assert registered_waits.get(record["wait_id"])["state"] == "cancelled"


def _race_terminal(record, action):
    operation_id = str(uuid.uuid4())
    if action == "cancel":
        registered_waits.cancel(record["wait_id"], operation_id=operation_id, actor="racer")
    else:
        with database.SessionLocal() as db:
            session_name = db.get(database.RegisteredWaitModel, record["wait_id"]).session_name
        registered_waits.interrupt_session_waits(session_name, operation_id)


@pytest.mark.parametrize(
    ("action", "terminal_state"),
    [("cancel", "cancelled"), ("stop", "interrupted-by-stop")],
)
def test_terminal_commit_racing_ready_adoption_cannot_be_overwritten(
    tmp_path, monkeypatch, action, terminal_state
):
    record = _pending(tmp_path, monkeypatch, f"adopt-ready-{action}")
    paths = _paths(record["wait_id"])
    paths["ready"].write_text(json.dumps(_ready(record["wait_id"])), encoding="utf-8")
    entered = threading.Event()
    release = threading.Event()
    terminal_done = threading.Event()

    def paused_alive(_pid, _marker):
        entered.set()
        assert release.wait(timeout=5)
        return True

    monkeypatch.setattr(monitors, "_helper_alive", paused_alive)
    adopter = threading.Thread(target=monitors.adopt_monitor_evidence, args=(record["wait_id"],))
    adopter.start()
    assert entered.wait(timeout=5)

    def terminalize():
        _race_terminal(record, action)
        terminal_done.set()

    terminal = threading.Thread(target=terminalize)
    terminal.start()
    serialized = not terminal_done.wait(timeout=0.2)
    release.set()
    adopter.join(timeout=5)
    terminal.join(timeout=5)

    assert not adopter.is_alive()
    assert not terminal.is_alive()
    assert registered_waits.get(record["wait_id"])["state"] == terminal_state
    assert _monitor(record["wait_id"]).state == terminal_state
    assert serialized


@pytest.mark.parametrize(
    ("action", "terminal_state"),
    [("cancel", "cancelled"), ("stop", "interrupted-by-stop")],
)
def test_terminal_commit_racing_result_adoption_has_no_late_wake(
    tmp_path, monkeypatch, action, terminal_state
):
    record = _pending(tmp_path, monkeypatch, f"adopt-result-{action}")
    _write_result(record["wait_id"])
    entered = threading.Event()
    release = threading.Event()
    terminal_done = threading.Event()
    original_digest = monitors._canonical_result_digest

    def paused_digest(result):
        entered.set()
        assert release.wait(timeout=5)
        return original_digest(result)

    monkeypatch.setattr(monitors, "_canonical_result_digest", paused_digest)
    adopter = threading.Thread(target=monitors.adopt_monitor_evidence, args=(record["wait_id"],))
    adopter.start()
    assert entered.wait(timeout=5)

    def terminalize():
        _race_terminal(record, action)
        terminal_done.set()

    terminal = threading.Thread(target=terminalize)
    terminal.start()
    serialized = not terminal_done.wait(timeout=0.2)
    release.set()
    adopter.join(timeout=5)
    terminal.join(timeout=5)
    deliveries = []
    registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1),
        attach_result=_good_attachment,
        deliver=deliveries.append,
    )

    assert not adopter.is_alive()
    assert not terminal.is_alive()
    assert registered_waits.get(record["wait_id"])["state"] == terminal_state
    assert _monitor(record["wait_id"]).state == terminal_state
    assert deliveries == []
    assert _inbox_count() == 0
    assert serialized


def test_committed_terminal_between_result_read_and_adoption_cas_wins(tmp_path, monkeypatch):
    record = _pending(tmp_path, monkeypatch, "adopt-result-cas")
    _write_result(record["wait_id"])
    original_digest = monitors._canonical_result_digest

    def terminal_then_digest(result):
        with database.SessionLocal() as winner:
            wait_row = winner.get(database.RegisteredWaitModel, record["wait_id"])
            monitor = winner.get(database.RegisteredWaitMonitorModel, record["wait_id"])
            wait_row.state = "cancelled"
            wait_row.outcome_json = json.dumps({"reason_code": "cancelled"})
            monitor.state = "cancelled"
            monitor.outcome_json = json.dumps({"reason_code": "cancelled"})
            winner.commit()
        return original_digest(result)

    monkeypatch.setattr(monitors, "_canonical_result_digest", terminal_then_digest)
    assert monitors.adopt_monitor_evidence(record["wait_id"]) is None
    deliveries = []
    registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1),
        attach_result=_good_attachment,
        deliver=deliveries.append,
    )

    assert registered_waits.get(record["wait_id"])["state"] == "cancelled"
    assert _monitor(record["wait_id"]).state == "cancelled"
    assert deliveries == []
    assert _inbox_count() == 0


def test_adapter_capability_stays_dark_without_consumer(monkeypatch):
    monkeypatch.delenv("CAO_M7_WAIT_MONITOR_CONSUMER_ENABLED")
    assert registered_waits.capability()["adapter_support"] == []


def test_adapter_registration_refuses_without_consumer(tmp_path, monkeypatch):
    monkeypatch.delenv("CAO_M7_WAIT_MONITOR_CONSUMER_ENABLED")
    with pytest.raises(registered_waits.RegisteredWaitConflict, match="monitor consumer"):
        registered_waits.register(
            RegistrationRequest(
                operation_id=str(uuid.uuid4()),
                session_name="dark",
                project="p",
                task_id="t",
                name="dark",
                description="consumer absent",
                duration_seconds=60,
                owner=_owner("dark"),
                adapter=_adapter(tmp_path / "dark"),
            ),
            now=NOW,
        )


@pytest.mark.parametrize("failure", ["consumer-absent", "exception", "invalid-receipt"])
def test_result_attachment_failure_settles_at_registered_deadline(tmp_path, monkeypatch, failure):
    record = _pending(tmp_path, monkeypatch, "attachment-deadline", duration=1)
    _write_result(record["wait_id"])
    if failure == "consumer-absent":
        attach_result = None
    elif failure == "invalid-receipt":
        attach_result = lambda _result: {}
    else:
        attach_result = lambda _result: (_ for _ in ()).throw(RuntimeError("bridge down"))

    result = registered_waits.process_monitors(
        now=NOW + timedelta(seconds=1),
        attach_result=attach_result,
    )

    assert result[0]["state"] == "invalid"
    final = registered_waits.get(record["wait_id"])
    assert final["outcome"]["reason_code"] == "monitor-attachment-deadline"
