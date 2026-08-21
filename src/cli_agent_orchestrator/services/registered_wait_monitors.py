"""Durable lifecycle for bounded process and GitHub wait monitors."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from sqlalchemy.exc import OperationalError, SQLAlchemyError

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.constants import CAO_HOME_DIR
from cli_agent_orchestrator.models.inbox import MessageStatus
from cli_agent_orchestrator.services import wait_admission, wait_runner
from cli_agent_orchestrator.services.registered_waits import _isots, _now, _parse_time

MONITOR_LAUNCH_INTENT = "launch-intent"
MONITOR_ACTIVE = "active"
MONITOR_RESULT_READY = "result-ready"
MONITOR_WAKE_PENDING = "wake-pending"
MONITOR_COMPLETED = "completed"
MONITOR_INTERRUPTED = "interrupted-by-stop"
MONITOR_INVALID = "invalid"
MONITOR_CANCELLED = "cancelled"
MONITOR_ACTIVE_STATES = frozenset(("launch-intent", "active", "result-ready", "wake-pending"))
MONITOR_TERMINAL_STATES = frozenset(("completed", "interrupted-by-stop", "invalid", "cancelled"))

HEALTH_MONITORED = "monitored"
HEALTH_STALE = "monitor-stale"
HEALTH_UNMONITORED = "unmonitored"

CONTROL_SCHEMA_VERSION = wait_runner.CONTROL_SCHEMA_VERSION
RUNTIME_SCHEMA_VERSION = wait_runner.RUNTIME_SCHEMA_VERSION
RESULT_SCHEMA_VERSION = wait_runner.RESULT_SCHEMA_VERSION
SPEC_SCHEMA_VERSION = wait_runner.SPEC_SCHEMA_VERSION

_get_start_marker = wait_runner._get_start_marker
_group_absent = wait_runner._group_absent
_terminate_pgid = wait_runner._terminate_pgid

_MESSAGE_NAMESPACE = uuid.UUID("07000000-2000-4700-b7e2-000000000009")
_WAKE_NAMESPACE = uuid.UUID("07000000-2000-4700-b7e2-000000000008")
_WAIT_TERMINAL_STATES = (MONITOR_TERMINAL_STATES - {MONITOR_COMPLETED}) | {"resolved"}


def _monitor_root() -> Path:
    root = CAO_HOME_DIR / "wait_monitors"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return root


def monitor_run_dir(wait_id: str) -> Path:
    return _monitor_root() / wait_id


def monitor_paths(wait_id: str, *, run_dir: Optional[Path | str] = None) -> dict[str, Path]:
    base = Path(run_dir) if run_dir is not None else monitor_run_dir(wait_id)
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(base, 0o700)
    except OSError:
        pass
    return {
        "dir": base,
        "spec": base / "spec.json",
        "ready": base / "ready.json",
        "result": base / "result.json",
        "activate": base / "activate.json",
        "stop": base / "stop.json",
    }


def monitor_paths_for_monitor(monitor: Any) -> dict[str, Path]:
    run_dir = getattr(monitor, "run_dir", None)
    return (
        monitor_paths(monitor.wait_id, run_dir=run_dir)
        if run_dir
        else monitor_paths(monitor.wait_id)
    )


def _helper_alive(pid: int, marker: Optional[str]) -> bool:
    return bool(pid and marker and _get_start_marker(pid) == marker)


def _canonical_result_digest(result: dict[str, Any]) -> str:
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_control(path: Path, action: str, wait_id: str, digest: str) -> None:
    wait_runner.atomic_write_json(
        path,
        {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "action": action,
            "wait_id": wait_id,
            "request_digest": digest,
        },
    )


def _validate_ready(data: Any, spec: dict[str, Any]) -> Optional[dict[str, Any]]:
    required = {
        "schema_version",
        "wait_id",
        "request_digest",
        "pid",
        "start_marker",
        "phase",
        "started_at",
        "adapter",
        "timeout_seconds",
    }
    allowed = required | {"child_pid", "child_start_marker", "pgid"}
    if not isinstance(data, dict) or not required <= set(data) <= allowed:
        return None
    expected = (
        data.get("schema_version") == RUNTIME_SCHEMA_VERSION
        and data.get("wait_id") == spec.get("wait_id")
        and data.get("request_digest") == spec.get("request_digest")
        and data.get("adapter") == spec.get("adapter", {}).get("kind")
        and data.get("timeout_seconds") == spec.get("timeout_seconds")
        and data.get("phase") in {"waiting-for-activation", "running"}
    )
    if not expected:
        return None
    if type(data.get("pid")) is not int or not isinstance(data.get("start_marker"), str):
        return None
    if not data["start_marker"] or not isinstance(data.get("started_at"), str):
        return None
    optional_types = {
        "child_pid": int,
        "child_start_marker": str,
        "pgid": int,
    }
    if any(key in data and type(data[key]) is not kind for key, kind in optional_types.items()):
        return None
    return data


def _validate_result(data: Any, spec: dict[str, Any]) -> Optional[dict[str, Any]]:
    required = {
        "schema_version",
        "wait_id",
        "request_digest",
        "adapter",
        "outcome",
        "reason",
        "observed",
        "started_at",
        "finished_at",
        "elapsed_seconds",
    }
    allowed = required | {"process", "github", "error", "exception"}
    if not isinstance(data, dict) or not required <= set(data) <= allowed:
        return None
    expected = (
        data.get("schema_version") == RESULT_SCHEMA_VERSION
        and data.get("wait_id") == spec.get("wait_id")
        and data.get("request_digest") == spec.get("request_digest")
        and isinstance(data.get("adapter"), dict)
        and data.get("adapter", {}).get("kind") == spec.get("adapter", {}).get("kind")
    )
    if not expected:
        return None
    scalar_shape = (
        isinstance(data.get("outcome"), str)
        and bool(data["outcome"])
        and isinstance(data.get("reason"), str)
        and bool(data["reason"])
        and isinstance(data.get("observed"), dict)
        and isinstance(data.get("started_at"), str)
        and isinstance(data.get("finished_at"), str)
        and isinstance(data.get("elapsed_seconds"), (int, float))
    )
    return data if scalar_shape else None


def _load_json(path: Path) -> tuple[str, Any]:
    if not path.exists():
        return "absent", None
    try:
        with path.open(encoding="utf-8") as handle:
            return "ok", json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return "unreadable", str(exc)


def _load_spec(monitor: Any, paths: dict[str, Path]) -> Optional[dict[str, Any]]:
    status, value = _load_json(paths["spec"])
    if status != "ok":
        return None
    try:
        spec = wait_runner.validate_spec(value)
    except ValueError:
        return None
    if spec["wait_id"] != monitor.wait_id or spec["request_digest"] != monitor.request_digest:
        return None
    return spec


def create_monitor_intent(
    wait_id: str,
    request_digest: str,
    adapter: dict[str, Any],
    timeout_seconds: int,
    run_dir: Optional[Path | str] = None,
) -> dict[str, Path]:
    paths = monitor_paths(wait_id, run_dir=run_dir)
    spec = wait_runner.validate_spec(
        {
            "schema_version": SPEC_SCHEMA_VERSION,
            "wait_id": wait_id,
            "request_digest": request_digest,
            "timeout_seconds": timeout_seconds,
            "adapter": adapter,
        }
    )
    wait_runner.atomic_write_json(paths["spec"], spec)
    return paths


def launch_dormant_runner(paths: dict[str, Path]) -> subprocess.Popen:
    argv = [
        sys.executable,
        "-m",
        "cli_agent_orchestrator.services.wait_runner",
        "--spec",
        str(paths["spec"]),
        "--ready",
        str(paths["ready"]),
        "--activate",
        str(paths["activate"]),
        "--stop",
        str(paths["stop"]),
        "--result",
        str(paths["result"]),
    ]
    env = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "CAO_WAIT_RUNNER_FAKE_MARKER")
        if key in os.environ
    }
    return subprocess.Popen(
        argv,
        shell=False,
        start_new_session=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _invalid_observation(value: Any, spec: dict[str, Any], kind: str) -> str:
    if not isinstance(value, dict):
        return f"{kind}-unreadable:shape"
    if value.get("wait_id") != spec.get("wait_id") or value.get("request_digest") != spec.get(
        "request_digest"
    ):
        return f"{kind}-mismatch"
    if value.get("schema_version") != (
        RUNTIME_SCHEMA_VERSION if kind == "ready" else RESULT_SCHEMA_VERSION
    ):
        return f"{kind}-unreadable:schema"
    return f"{kind}-unreadable:shape"


def monitor_health_for_row(
    monitor: Any, spec: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    if monitor.state in MONITOR_TERMINAL_STATES:
        return {"health": HEALTH_UNMONITORED, "detail": f"monitor-{monitor.state}"}
    if monitor.state in {MONITOR_RESULT_READY, MONITOR_WAKE_PENDING}:
        return {"health": HEALTH_UNMONITORED, "detail": "result-present"}

    status, raw_result = _load_json(paths["result"])
    if status == "unreadable":
        return {"health": HEALTH_UNMONITORED, "detail": f"unreadable:result:{raw_result}"}
    if status == "ok":
        if _validate_result(raw_result, spec) is not None:
            return {"health": HEALTH_UNMONITORED, "detail": "result-present"}
        return {
            "health": HEALTH_UNMONITORED,
            "detail": _invalid_observation(raw_result, spec, "result"),
        }

    status, raw_ready = _load_json(paths["ready"])
    if status == "unreadable":
        return {"health": HEALTH_UNMONITORED, "detail": f"unreadable:ready:{raw_ready}"}
    if status == "ok":
        ready = _validate_ready(raw_ready, spec)
        if ready is None:
            return {
                "health": HEALTH_UNMONITORED,
                "detail": _invalid_observation(raw_ready, spec, "ready"),
            }
        if monitor.helper_pid is not None:
            if (
                ready["pid"] != monitor.helper_pid
                or ready["start_marker"] != monitor.helper_start_marker
            ):
                return {"health": HEALTH_STALE, "detail": "helper-mismatch"}
        if _helper_alive(ready["pid"], ready["start_marker"]):
            detail = "ready-adoptable" if monitor.state == MONITOR_LAUNCH_INTENT else "helper-live"
            return {"health": HEALTH_MONITORED, "detail": detail}
        return {"health": HEALTH_STALE, "detail": "helper-dead-no-result"}

    if monitor.helper_pid is not None:
        if _helper_alive(monitor.helper_pid, monitor.helper_start_marker):
            return {"health": HEALTH_MONITORED, "detail": "helper-live-by-db"}
        return {"health": HEALTH_STALE, "detail": "helper-dead-no-result"}
    if monitor.state == MONITOR_LAUNCH_INTENT:
        return {"health": HEALTH_UNMONITORED, "detail": "launch-intent-without-proof"}
    return {"health": HEALTH_STALE, "detail": "missing-helper-no-result"}


def monitor_health_for(wait_id: str) -> dict[str, Any]:
    try:
        with database.SessionLocal() as db:
            wait_row = db.get(database.RegisteredWaitModel, wait_id)
            monitor = db.get(database.RegisteredWaitMonitorModel, wait_id)
            if wait_row is None:
                return {"health": HEALTH_UNMONITORED, "detail": "wait-absent"}
            if wait_row.state in _WAIT_TERMINAL_STATES:
                return {"health": HEALTH_UNMONITORED, "detail": f"wait-{wait_row.state}"}
            try:
                request = json.loads(wait_row.request_json)
            except (TypeError, json.JSONDecodeError) as exc:
                return {"health": HEALTH_UNMONITORED, "detail": f"unreadable:request:{exc}"}
            if not isinstance(request, dict) or not isinstance(request.get("adapter"), dict):
                return {"health": HEALTH_UNMONITORED, "detail": "timer-wait"}
            if monitor is None:
                return {"health": HEALTH_UNMONITORED, "detail": "monitor-absent"}
            paths = monitor_paths_for_monitor(monitor)
            spec = _load_spec(monitor, paths)
            if spec is None:
                return {"health": HEALTH_UNMONITORED, "detail": "unreadable:spec"}
            return monitor_health_for_row(monitor, spec, paths)
    except (OperationalError, SQLAlchemyError) as exc:
        return {"health": HEALTH_UNMONITORED, "detail": f"unreadable:store:{exc}"}


def _ack_wait(wait_row: Any, observed: datetime) -> None:
    if wait_row.state == "registration-pending":
        wait_row.state = "acknowledged"
        wait_row.updated_at = _isots(observed)


def adopt_monitor_evidence(wait_id: str) -> Optional[dict[str, Any]]:
    observed = _now()
    activate: Optional[tuple[Path, str]] = None
    with database.SessionLocal() as db:
        monitor = db.get(database.RegisteredWaitMonitorModel, wait_id)
        wait_row = db.get(database.RegisteredWaitModel, wait_id)
        if monitor is None or wait_row is None or monitor.state in MONITOR_TERMINAL_STATES:
            return None
        if monitor.state in {MONITOR_RESULT_READY, MONITOR_WAKE_PENDING}:
            return {"state": monitor.state}
        paths = monitor_paths_for_monitor(monitor)
        spec = _load_spec(monitor, paths)
        if spec is None:
            return None

        status, raw_result = _load_json(paths["result"])
        result = _validate_result(raw_result, spec) if status == "ok" else None
        if result is not None:
            canonical = json.dumps(result, sort_keys=True)
            digest = _canonical_result_digest(result)
            if monitor.result_json not in {None, canonical} or monitor.result_digest not in {
                None,
                digest,
            }:
                return None
            monitor.result_json = canonical
            monitor.result_digest = digest
            monitor.state = MONITOR_RESULT_READY
            monitor.updated_at = _isots(observed)
            _ack_wait(wait_row, observed)
            db.commit()
            return {"state": MONITOR_RESULT_READY, "result": result}

        if monitor.state != MONITOR_LAUNCH_INTENT:
            return None
        status, raw_ready = _load_json(paths["ready"])
        ready = _validate_ready(raw_ready, spec) if status == "ok" else None
        if ready is None or not _helper_alive(ready["pid"], ready["start_marker"]):
            return None
        monitor.helper_pid = ready["pid"]
        monitor.helper_start_marker = ready["start_marker"]
        monitor.child_pid = ready.get("child_pid")
        monitor.child_start_marker = ready.get("child_start_marker")
        monitor.pgid = ready.get("pgid")
        monitor.state = MONITOR_ACTIVE
        monitor.updated_at = _isots(observed)
        _ack_wait(wait_row, observed)
        db.commit()
        activate = (paths["activate"], monitor.request_digest)

    if activate is not None and not activate[0].exists():
        _write_control(activate[0], "activate", wait_id, activate[1])
    return {"state": MONITOR_ACTIVE, "ready": ready}


def _view(wait_id: str, state: str, **extra: Any) -> dict[str, Any]:
    return {"wait_id": wait_id, "state": state, **extra}


def _set_terminal(
    wait_row: Any,
    monitor: Any,
    *,
    wait_state: str,
    monitor_state: str,
    reason: str,
    detail: Optional[str],
    observed: datetime,
) -> None:
    outcome = {"reason_code": reason, "detail": detail, "at": _isots(observed)}
    wait_row.state = wait_state
    wait_row.outcome_json = json.dumps(outcome, sort_keys=True)
    wait_row.updated_at = _isots(observed)
    monitor.state = monitor_state
    monitor.outcome_json = json.dumps({"reason_code": reason, "detail": detail}, sort_keys=True)
    monitor.updated_at = _isots(observed)


def _input_disposition(
    callback: Optional[Callable[[str, str], bool]], terminal_id: str, generation: str
) -> Optional[str]:
    if callback is None:
        return None
    try:
        return "input-held" if callback(terminal_id, generation) else None
    except Exception as exc:
        return f"input-unreadable:{exc}"


def process_monitors(
    *,
    now: Optional[datetime] = None,
    deliver: Optional[Callable[[str], None]] = None,
    receipt_probe: Optional[Callable[[str, int], Optional[Mapping[str, Any]]]] = None,
    attach_result: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    input_held: Optional[Callable[[str, str], bool]] = None,
    ambiguity_grace_seconds: int = 60,
) -> list[dict[str, Any]]:
    observed = now or _now()
    try:
        with database.SessionLocal() as db:
            ids = [
                str(row.wait_id)
                for row in db.query(database.RegisteredWaitMonitorModel)
                .filter(database.RegisteredWaitMonitorModel.state.in_(MONITOR_ACTIVE_STATES))
                .all()
            ]
    except (OperationalError, SQLAlchemyError) as exc:
        return [_view("*", "unreadable", detail=f"unreadable:store:{exc}")]
    results = []
    for wait_id in ids:
        try:
            result = _process_one_monitor(
                wait_id,
                observed=observed,
                deliver=deliver,
                receipt_probe=receipt_probe,
                attach_result=attach_result,
                input_held=input_held,
                ambiguity_grace_seconds=ambiguity_grace_seconds,
            )
        except (OperationalError, SQLAlchemyError) as exc:
            result = _view(wait_id, "unreadable", detail=f"unreadable:store:{exc}")
        if result is not None:
            results.append(result)
    return results


def _settle_stale(
    db: Any, wait_row: Any, monitor: Any, observed: datetime, detail: str
) -> dict[str, Any]:
    _set_terminal(
        wait_row,
        monitor,
        wait_state="invalid",
        monitor_state=MONITOR_INVALID,
        reason="monitor-stale",
        detail=detail,
        observed=observed,
    )
    db.commit()
    return _view(wait_row.wait_id, wait_row.state, health=HEALTH_STALE, detail=detail)


def _attachment_valid(value: Any, digest: Optional[str]) -> bool:
    if not isinstance(value, dict):
        return False
    strings = (value.get("communication_id"), value.get("attachment_id"))
    return (
        all(isinstance(item, str) and item.strip() for item in strings)
        and value.get("digest") == digest
    )


def _create_wake(db: Any, wait_row: Any, monitor: Any, request: dict[str, Any]) -> int:
    result = json.loads(monitor.result_json or "{}")
    message_id = str(uuid.uuid5(_MESSAGE_NAMESPACE, f"{wait_row.operation_id}-monitor"))
    wake_operation_id = str(uuid.uuid5(_WAKE_NAMESPACE, f"{wait_row.operation_id}-wake"))
    text = (
        f"Wait monitor completed: {wait_row.wait_id} "
        f"outcome={result.get('outcome')} reason={result.get('reason')}"
    )
    owner = wait_admission.WaitOwner(**request["owner"])
    admission = wait_admission.admit(
        wait_admission.AdmissionRequest(
            operation_id=wake_operation_id,
            session_name=wait_row.session_name,
            owner=owner,
            message=wait_admission.WaitMessage(
                message_id=message_id,
                kind=wait_admission.KIND_WORKER_WAKE,
                reason_code="monitor-completed",
                payload_digest=monitor.result_digest,
                source_operation_id=wait_row.operation_id,
                text=text[:512],
            ),
        ),
        db=db,
    )
    if admission["admission_state"] != wait_admission.STATE_ADMITTED:
        error = wait_admission.WaitAdmissionError(str(admission.get("detail") or "wake refused"))
        error.code = str(admission["denial_reason"])
        raise error
    inbox = database.InboxModel(
        sender_id=wait_row.owner_terminal_id,
        receiver_id=wait_row.owner_terminal_id,
        message=text,
        status=MessageStatus.PENDING.value,
        sender_generation=wait_row.owner_generation,
        expected_receiver_generation=wait_row.owner_generation,
    )
    db.add(inbox)
    db.flush()
    return int(inbox.id)


def _settle_wake(
    wait_id: str,
    wake_id: int,
    terminal_id: str,
    observed: datetime,
    receipt_probe: Optional[Callable[[str, int], Optional[Mapping[str, Any]]]],
    ambiguity_grace_seconds: int,
) -> Optional[dict[str, Any]]:
    receipt = receipt_probe(terminal_id, wake_id) if receipt_probe else None
    with database.SessionLocal() as db:
        monitor = db.get(database.RegisteredWaitMonitorModel, wait_id)
        wait_row = db.get(database.RegisteredWaitModel, wait_id)
        if monitor is None or wait_row is None or monitor.state != MONITOR_WAKE_PENDING:
            return None
        inbox = db.get(database.InboxModel, wake_id)
        if receipt is not None or (inbox and inbox.status == MessageStatus.DELIVERED.value):
            reason = "wake-confirmed" if receipt is not None else "wake-delivered"
            detail = f"monitor result delivered via inbox {wake_id}"
            _set_terminal(
                wait_row,
                monitor,
                wait_state="resolved",
                monitor_state=MONITOR_COMPLETED,
                reason=reason,
                detail=detail,
                observed=observed,
            )
            wait_row.wake_message_id = wake_id
        elif inbox and inbox.status == MessageStatus.FAILED.value:
            _set_terminal(
                wait_row,
                monitor,
                wait_state="invalid",
                monitor_state=MONITOR_INVALID,
                reason="wake-refused",
                detail=f"monitor inbox {wake_id} refused",
                observed=observed,
            )
        else:
            try:
                pending_since = _parse_time(monitor.wake_pending_since)
            except (TypeError, ValueError):
                return _view(wait_id, MONITOR_WAKE_PENDING, detail="unreadable:wake-pending-since")
            if observed < pending_since + timedelta(seconds=ambiguity_grace_seconds):
                return _view(wait_id, MONITOR_WAKE_PENDING)
            if inbox and inbox.status == MessageStatus.PENDING.value:
                inbox.status = MessageStatus.FAILED.value
            _set_terminal(
                wait_row,
                monitor,
                wait_state="invalid",
                monitor_state=MONITOR_INVALID,
                reason="monitor-wake-ambiguous",
                detail=f"no delivery evidence after {ambiguity_grace_seconds} seconds",
                observed=observed,
            )
        db.commit()
        return _view(wait_id, wait_row.state, monitor_state=monitor.state)


def _process_one_monitor(
    wait_id: str,
    *,
    observed: datetime,
    deliver: Optional[Callable[[str], None]],
    receipt_probe: Optional[Callable[[str, int], Optional[Mapping[str, Any]]]],
    attach_result: Optional[Callable[[dict[str, Any]], dict[str, Any]]],
    input_held: Optional[Callable[[str, str], bool]],
    ambiguity_grace_seconds: int,
) -> Optional[dict[str, Any]]:
    adopt_monitor_evidence(wait_id)
    with database.SessionLocal() as db:
        monitor = db.get(database.RegisteredWaitMonitorModel, wait_id)
        wait_row = db.get(database.RegisteredWaitModel, wait_id)
        if monitor is None or wait_row is None:
            return None
        if wait_row.state in _WAIT_TERMINAL_STATES:
            if wait_row.state == "resolved" and monitor.state not in MONITOR_TERMINAL_STATES:
                monitor.state = MONITOR_COMPLETED
                monitor.updated_at = _isots(observed)
                db.commit()
            return None
        paths = monitor_paths_for_monitor(monitor)
        spec = _load_spec(monitor, paths)
        if spec is None:
            return _view(wait_id, "unreadable", detail="unreadable:spec")
        health = monitor_health_for_row(monitor, spec, paths)
        if monitor.state == MONITOR_LAUNCH_INTENT:
            if observed >= _parse_time(wait_row.deadline_at):
                return _settle_stale(db, wait_row, monitor, observed, "launch-intent-deadline")
            if health["health"] == HEALTH_STALE:
                return _settle_stale(db, wait_row, monitor, observed, health["detail"])
            return _view(wait_id, monitor.state, **health)
        if monitor.state == MONITOR_ACTIVE:
            if health["health"] == HEALTH_STALE:
                return _settle_stale(db, wait_row, monitor, observed, health["detail"])
            return _view(wait_id, monitor.state, **health)

        try:
            request = json.loads(wait_row.request_json)
            result = json.loads(monitor.result_json or "{}")
            owner = wait_admission.WaitOwner(**request["owner"])
        except (TypeError, KeyError, json.JSONDecodeError) as exc:
            return _view(wait_id, "unreadable", detail=f"request-unreadable:{exc}")
        verdict = wait_admission.verify_owner(owner, db=db)
        denial = verdict["denial_reason"]
        if denial == wait_admission.DENY_OWNER_UNREADABLE:
            return _view(wait_id, "owner-unreadable", detail=verdict["detail"])
        if denial is not None:
            _set_terminal(
                wait_row,
                monitor,
                wait_state="invalid",
                monitor_state=MONITOR_INVALID,
                reason=str(denial),
                detail=verdict["detail"],
                observed=observed,
            )
            db.commit()
            return _view(wait_id, wait_row.state, health=HEALTH_UNMONITORED)

        hold = _input_disposition(input_held, wait_row.owner_terminal_id, wait_row.owner_generation)
        if hold:
            return _view(wait_id, monitor.state, health=HEALTH_UNMONITORED, detail=hold)
        if monitor.state == MONITOR_RESULT_READY and monitor.attachment_id is None:
            if attach_result is None:
                return _view(wait_id, MONITOR_RESULT_READY, health=HEALTH_UNMONITORED)
            try:
                attachment = attach_result(
                    {
                        "wait_id": wait_id,
                        "request": request,
                        "result": result,
                        "request_digest": monitor.request_digest,
                        "result_digest": monitor.result_digest,
                    }
                )
            except Exception:
                return _view(wait_id, MONITOR_RESULT_READY, health=HEALTH_UNMONITORED)
            if not _attachment_valid(attachment, monitor.result_digest):
                return _view(
                    wait_id,
                    MONITOR_RESULT_READY,
                    health=HEALTH_UNMONITORED,
                    detail="attachment-receipt-invalid",
                )
            monitor.communication_id = attachment["communication_id"]
            monitor.attachment_id = attachment["attachment_id"]
            monitor.attachment_digest = attachment["digest"]
            monitor.updated_at = _isots(observed)
            db.commit()

        created = False
        if monitor.wake_message_id is None:
            hold = _input_disposition(
                input_held, wait_row.owner_terminal_id, wait_row.owner_generation
            )
            if hold:
                return _view(wait_id, monitor.state, health=HEALTH_UNMONITORED, detail=hold)
            try:
                wake_id = _create_wake(db, wait_row, monitor, request)
            except wait_admission.WaitAdmissionError as exc:
                _set_terminal(
                    wait_row,
                    monitor,
                    wait_state="invalid",
                    monitor_state=MONITOR_INVALID,
                    reason=exc.code,
                    detail=str(exc),
                    observed=observed,
                )
                db.commit()
                return _view(wait_id, wait_row.state)
            monitor.wake_message_id = wake_id
            monitor.wake_pending_since = _isots(observed)
            monitor.state = MONITOR_WAKE_PENDING
            monitor.updated_at = _isots(observed)
            db.commit()
            created = True
        else:
            wake_id = int(monitor.wake_message_id)
        terminal_id = wait_row.owner_terminal_id

    if created and deliver is not None:
        try:
            deliver(terminal_id)
        except Exception:
            pass
    return _settle_wake(
        wait_id,
        wake_id,
        terminal_id,
        observed,
        receipt_probe,
        ambiguity_grace_seconds,
    )


def _stop_result(wait_id: str, wait_row: Any, monitor: Any) -> dict[str, Any]:
    return _view(wait_id, wait_row.state, monitor_state=monitor.state if monitor else None)


def _terminate_bound_group(
    *, pid: int, marker: str, pgid: int, label: str, unavailable: type[Exception]
) -> None:
    if not _helper_alive(pid, marker):
        if _group_absent(pgid):
            return
        raise unavailable(f"{label} identity inconclusive")
    if not _terminate_pgid(pgid):
        raise unavailable(f"{label} termination inconclusive")
    if _helper_alive(pid, marker) and not _group_absent(pgid):
        raise unavailable(f"{label} still live after termination")


def stop_monitor(wait_id: str, operation_id: str, actor: str) -> dict[str, Any]:
    from cli_agent_orchestrator.services.registered_waits import (
        RegisteredWaitInvalid,
        RegisteredWaitUnavailable,
    )

    try:
        uuid.UUID(operation_id)
    except ValueError as exc:
        raise RegisteredWaitUnavailable(str(exc)) from exc
    with database.SessionLocal() as db:
        wait_row = db.get(database.RegisteredWaitModel, wait_id)
        monitor = db.get(database.RegisteredWaitMonitorModel, wait_id)
        if wait_row is None:
            raise RegisteredWaitInvalid(f"unknown wait {wait_id}")
        if wait_row.state in _WAIT_TERMINAL_STATES:
            return _stop_result(wait_id, wait_row, monitor)
        if monitor is None:
            raise RegisteredWaitUnavailable("no monitor for wait")
        if monitor.state in MONITOR_TERMINAL_STATES:
            return _stop_result(wait_id, wait_row, monitor)

        if monitor.wake_message_id is not None:
            inbox = db.get(database.InboxModel, monitor.wake_message_id)
            if inbox and inbox.status == MessageStatus.DELIVERED.value:
                return _stop_result(wait_id, wait_row, monitor)
            if inbox and inbox.status == MessageStatus.FAILED.value:
                _set_terminal(
                    wait_row,
                    monitor,
                    wait_state="invalid",
                    monitor_state=MONITOR_INVALID,
                    reason="wake-refused",
                    detail=f"monitor inbox {monitor.wake_message_id} refused",
                    observed=_now(),
                )
                db.commit()
                return _stop_result(wait_id, wait_row, monitor)

        paths = monitor_paths_for_monitor(monitor)
        spec = _load_spec(monitor, paths)
        if spec is None:
            raise RegisteredWaitUnavailable("spec unreadable or mismatched")
        if monitor.state == MONITOR_ACTIVE:
            status, raw_ready = _load_json(paths["ready"])
            ready = _validate_ready(raw_ready, spec) if status == "ok" else None
            if ready is None:
                raise RegisteredWaitUnavailable("ready unreadable or mismatched")
            if (
                ready["pid"] != monitor.helper_pid
                or ready["start_marker"] != monitor.helper_start_marker
            ):
                raise RegisteredWaitUnavailable("ready helper identity mismatch")
            monitor.child_pid = ready.get("child_pid")
            monitor.child_start_marker = ready.get("child_start_marker")
            monitor.pgid = ready.get("pgid")
            monitor.updated_at = _isots(_now())
            db.commit()
            _write_control(paths["stop"], "stop", wait_id, monitor.request_digest)
            groups = [(ready["pid"], ready["start_marker"], ready["pid"], "helper")]
            if ready.get("child_pid") is not None:
                if not ready.get("child_start_marker") or ready.get("pgid") is None:
                    raise RegisteredWaitUnavailable("child identity incomplete")
                groups.append(
                    (
                        ready["child_pid"],
                        ready["child_start_marker"],
                        ready["pgid"],
                        "child",
                    )
                )
            seen = set()
            for pid, marker, pgid, label in reversed(groups):
                if pgid in seen:
                    continue
                seen.add(pgid)
                _terminate_bound_group(
                    pid=pid,
                    marker=marker,
                    pgid=pgid,
                    label=label,
                    unavailable=RegisteredWaitUnavailable,
                )
        elif monitor.state == MONITOR_LAUNCH_INTENT:
            _write_control(paths["stop"], "stop", wait_id, monitor.request_digest)

        for message_id in {monitor.wake_message_id, wait_row.wake_message_id} - {None}:
            inbox = db.get(database.InboxModel, message_id)
            if inbox and inbox.status == MessageStatus.DELIVERED.value:
                return _stop_result(wait_id, wait_row, monitor)
            if inbox and inbox.status == MessageStatus.PENDING.value:
                inbox.status = MessageStatus.FAILED.value
        observed = _now()
        _set_terminal(
            wait_row,
            monitor,
            wait_state="interrupted-by-stop",
            monitor_state=MONITOR_INTERRUPTED,
            reason="interrupted-by-stop",
            detail=f"operation {operation_id} by {actor}",
            observed=observed,
        )
        db.commit()
        return _stop_result(wait_id, wait_row, monitor)


def get_monitor(wait_id: str) -> Optional[dict[str, Any]]:
    with database.SessionLocal() as db:
        row = db.get(database.RegisteredWaitMonitorModel, wait_id)
        wait_row = db.get(database.RegisteredWaitModel, wait_id)
        if row is None:
            return None
        adapter_kind = None
        if wait_row is not None:
            try:
                adapter_kind = json.loads(wait_row.request_json).get("adapter", {}).get("kind")
            except (TypeError, json.JSONDecodeError):
                pass
        omitted = {"created_at", "updated_at"}
        result = {
            column.name: getattr(row, column.name)
            for column in database.RegisteredWaitMonitorModel.__table__.columns
            if column.name not in omitted
        }
        result.update(
            operation_id=wait_row.operation_id if wait_row else None,
            adapter_kind=adapter_kind,
        )
        return result
