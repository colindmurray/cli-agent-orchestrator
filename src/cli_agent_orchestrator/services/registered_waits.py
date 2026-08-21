"""Durable exact-owner scheduled waits (M7 timer adapter).

The fork owns the lifecycle because it also owns the stable roster, inbox and
Stop barrier.  Registration intent is committed before acknowledgement;
expiry intent is committed before inbox I/O.  Registration, expiry and cancel
operations all replay by exact operation id and divergent bytes refuse.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.inbox import MessageStatus
from cli_agent_orchestrator.services import wait_admission
from cli_agent_orchestrator.services.canonical_json import canonical_sha256, encode_canonical

SCHEMA_VERSION = "cao-registered-wait-v2"
CAPABILITY_SCHEMA_VERSION = 1
MAX_ROUND_SECONDS = 8 * 60 * 60
DEFAULT_AMBIGUITY_GRACE_SECONDS = 60

STATE_REGISTRATION_PENDING = "registration-pending"
STATE_ACKNOWLEDGED = "acknowledged"
STATE_EXPIRY_INTENT = "expiry-intent"
STATE_EXPIRY_WAKE_PENDING = "expiry-wake-pending"
STATE_RESOLVED = "resolved"
STATE_CANCELLED = "cancelled"
STATE_INVALID = "invalid"
STATE_INTERRUPTED = "interrupted-by-stop"
TERMINAL_STATES = frozenset({STATE_RESOLVED, STATE_CANCELLED, STATE_INVALID, STATE_INTERRUPTED})
ACTIVE_STATES = frozenset(
    {STATE_REGISTRATION_PENDING, STATE_ACKNOWLEDGED, STATE_EXPIRY_INTENT, STATE_EXPIRY_WAKE_PENDING}
)

_WAIT_NAMESPACE = uuid.UUID("07000000-2000-4700-b7e2-000000000007")
_EXPIRY_NAMESPACE = uuid.UUID("07000000-2000-4700-b7e2-000000000008")
_MESSAGE_NAMESPACE = uuid.UUID("07000000-2000-4700-b7e2-000000000009")


class RegisteredWaitError(RuntimeError):
    code = "registered-wait-error"


class RegisteredWaitInvalid(RegisteredWaitError):
    code = "registered-wait-invalid"


class RegisteredWaitConflict(RegisteredWaitError):
    code = "registered-wait-conflict"


class RegisteredWaitUnavailable(RegisteredWaitError):
    code = "registered-wait-unavailable"


def _enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def capability() -> dict[str, Any]:
    registration = _enabled("CAO_M7_WAIT_REGISTRATION_ENABLED")
    consumer = _enabled("CAO_M7_WAIT_CONSUMER_ENABLED")
    monitor_consumer = _enabled("CAO_M7_WAIT_MONITOR_CONSUMER_ENABLED", default=False)
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "capability": "m7-scheduled-waits",
        "contract_schema_version": SCHEMA_VERSION,
        "enabled": registration or consumer,
        "registration_enabled": registration,
        "consumer_attached": consumer,
        "stop_interruptor_attached": True,
        "public_surface": True,
        "reverse_rollback": "disable registration first; consumer drains acknowledged waits",
        "max_round_seconds": MAX_ROUND_SECONDS,
        "ambiguity_grace_seconds": DEFAULT_AMBIGUITY_GRACE_SECONDS,
        "adapter_support": ["process", "github-actions"] if monitor_consumer else [],
        "monitor_consumer_attached": monitor_consumer,
        "monitor_health_vocabulary": ["monitored", "monitor-stale", "unmonitored"],
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _isots(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _uuid(value: Any, field: str) -> str:
    try:
        parsed = str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise RegisteredWaitInvalid(f"{field} must be a canonical UUID") from exc
    if parsed != value:
        raise RegisteredWaitInvalid(f"{field} must be a canonical lowercase UUID")
    return parsed


def _text(value: Any, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegisteredWaitInvalid(f"{field} must be a non-empty string")
    text = str(value).strip()
    if len(text.encode("utf-8")) > maximum:
        raise RegisteredWaitInvalid(f"{field} must encode to at most {maximum} bytes")
    return text


def _wait_row(db: Any, wait_id: str) -> Any:
    return db.get(database.RegisteredWaitModel, wait_id)


def _inbox_row(db: Any, message_id: int) -> Any:
    return db.get(database.InboxModel, message_id)


def _operation_row(db: Any, operation_id: str) -> Any:
    return (
        db.query(database.RegisteredWaitModel)
        .filter(database.RegisteredWaitModel.operation_id == operation_id)
        .one_or_none()
    )


@dataclass(frozen=True)
class RegistrationRequest:
    operation_id: str
    session_name: str
    project: str
    task_id: str
    name: str
    description: str
    duration_seconds: int
    owner: wait_admission.WaitOwner
    estimated_seconds: Optional[int] = None
    adapter: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _uuid(self.operation_id, "operation_id"))
        for field in ("session_name", "project", "task_id", "name"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "description", _text(self.description, "description", 256))
        if not isinstance(self.owner, wait_admission.WaitOwner):
            raise RegisteredWaitInvalid("owner must be an exact WaitOwner")
        if not isinstance(self.duration_seconds, int) or isinstance(self.duration_seconds, bool):
            raise RegisteredWaitInvalid("duration_seconds must be an integer")
        if not 0 < self.duration_seconds <= MAX_ROUND_SECONDS:
            raise RegisteredWaitInvalid(
                f"duration_seconds must be between 1 and {MAX_ROUND_SECONDS}"
            )
        if self.estimated_seconds is not None and (
            not isinstance(self.estimated_seconds, int)
            or isinstance(self.estimated_seconds, bool)
            or self.estimated_seconds <= 0
        ):
            raise RegisteredWaitInvalid("estimated_seconds must be a positive integer")
        if self.adapter is not None:
            if not isinstance(self.adapter, dict):
                raise RegisteredWaitInvalid("adapter must be an object when present")
            kind = self.adapter.get("kind")
            if kind == "process":
                try:
                    from cli_agent_orchestrator.services.wait_runner import validate_process_adapter

                    validate_process_adapter(self.adapter)
                except ValueError as exc:
                    raise RegisteredWaitInvalid(str(exc)) from exc
            elif kind == "github-actions":
                try:
                    from cli_agent_orchestrator.services.wait_runner import validate_github_adapter

                    validate_github_adapter(self.adapter)
                except ValueError as exc:
                    raise RegisteredWaitInvalid(str(exc)) from exc
            else:
                raise RegisteredWaitInvalid("adapter kind must be process or github-actions")

    def canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "operation_id": self.operation_id,
            "session_name": self.session_name,
            "project": self.project,
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "duration_seconds": self.duration_seconds,
            "estimated_seconds": self.estimated_seconds,
            "owner": self.owner.canonical(),
        }
        if self.adapter is not None:
            payload["adapter"] = self.adapter
        return payload


def wait_id_for(operation_id: str) -> str:
    return str(uuid.uuid5(_WAIT_NAMESPACE, _uuid(operation_id, "operation_id")))


def expiry_operation_id_for(wait_id: str) -> str:
    return str(uuid.uuid5(_EXPIRY_NAMESPACE, _uuid(wait_id, "wait_id")))


def _request_from_row(row: Any) -> dict[str, Any]:
    try:
        request = json.loads(row.request_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RegisteredWaitUnavailable(f"wait {row.wait_id} request is unreadable") from exc
    if not isinstance(request, dict) or canonical_sha256(request) != row.request_digest:
        raise RegisteredWaitUnavailable(f"wait {row.wait_id} request contradicts its digest")
    return request


def _record(row: Any, *, now: Optional[datetime] = None) -> dict[str, Any]:
    request = _request_from_row(row)
    outcome = json.loads(row.outcome_json) if row.outcome_json else None
    observed = now or _now()
    if row.state in TERMINAL_STATES and outcome and outcome.get("at"):
        observed = _parse_time(outcome["at"])
    created = _parse_time(row.created_at)
    elapsed = max(0, int((observed - created).total_seconds()))
    adapter = request.get("adapter")
    if isinstance(adapter, dict) and adapter.get("kind") == "process":
        condition_kind = "process"
    elif isinstance(adapter, dict) and adapter.get("kind") == "github-actions":
        condition_kind = "github-actions"
    else:
        condition_kind = "scheduled-time"
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "wait_id": row.wait_id,
        "operation_id": row.operation_id,
        "condition": {
            "kind": condition_kind,
            "id": request["name"],
            "name": request["name"],
            "description": request["description"],
        },
        "round": {
            "number": 1,
            "max_seconds": request["duration_seconds"],
            "estimated_seconds": request.get("estimated_seconds"),
        },
        "totals": {"elapsed_seconds": elapsed},
        "owner": {
            "project": request["project"],
            "task_id": request["task_id"],
            "session_name": row.session_name,
            "terminal_id": row.owner_terminal_id,
            "stable_agent_id": row.owner_agent_id,
            "incarnation_id": row.owner_incarnation_id,
            "generation": row.owner_generation,
            "lineage_id": request["owner"].get("lineage_id"),
            "native_session_id": request["owner"].get("native_session_id"),
        },
        "state": row.state,
        "registered_at": row.created_at,
        "deadline_at": row.deadline_at,
        "wake_message_id": row.wake_message_id,
        "wake_pending_since": row.wake_pending_since,
        "outcome": outcome,
        "updated_at": row.updated_at,
    }
    if adapter is not None:
        # adapter path: expose monitor health and durable monitor state
        try:
            from cli_agent_orchestrator.services.registered_wait_monitors import monitor_health_for

            health = monitor_health_for(row.wait_id)
            health_value = health.get("health", "unmonitored")
            health_detail = health.get("detail")
        except Exception as exc:
            health_value = "unmonitored"
            health_detail = f"unreadable:{exc}"
        record["monitor_health"] = health_value
        record["monitor_health_detail"] = health_detail
        record["adapter"] = adapter
        # durable monitor state without secrets/files
        try:
            with database.SessionLocal() as db2:
                mon = db2.get(database.RegisteredWaitMonitorModel, row.wait_id)
                if mon is not None:
                    mon_outcome = json.loads(mon.outcome_json) if mon.outcome_json else None
                    result_info: dict[str, Any] = {}
                    if mon.result_json:
                        try:
                            rj = json.loads(mon.result_json)
                            result_info = {
                                "outcome": rj.get("outcome"),
                                "reason": rj.get("reason"),
                                "elapsed_seconds": rj.get("elapsed_seconds"),
                            }
                        except Exception:
                            result_info = {}
                    record["monitor"] = {
                        "state": mon.state,
                        "result_digest": mon.result_digest,
                        "result": result_info if result_info else None,
                        "communication_id": mon.communication_id,
                        "attachment_id": mon.attachment_id,
                        "attachment_digest": mon.attachment_digest,
                        "wake_message_id": mon.wake_message_id,
                        "wake_pending_since": mon.wake_pending_since,
                        "outcome": mon_outcome,
                    }
                else:
                    record["monitor"] = None
        except Exception as exc:
            record["monitor"] = {"detail": f"unreadable:{exc}"}
    else:
        # timer path: no monitor read, preserve shape
        record["monitor_health"] = "unmonitored"
        record["monitor_health_detail"] = "timer-wait"
    return record


def _owner(request: Mapping[str, Any]) -> wait_admission.WaitOwner:
    return wait_admission.WaitOwner(**request["owner"])


def register(request: RegistrationRequest, *, now: Optional[datetime] = None) -> dict[str, Any]:
    """Persist registration intent, then verify and acknowledge its exact owner.

    For adapter waits, intent is persisted BEFORE ``Popen``, helper identity
    BEFORE ``Activate``, and the result BEFORE any wake.  Replay adopts exact
    durable rows/files or refuses divergent requests.  An ambiguous launch
    window (intent with neither ready nor result) reports ``unmonitored`` and
    never relaunches.  Acknowledgement occurs only in the same transaction that
    records exact live helper or exact durable result.
    """
    if not _enabled("CAO_M7_WAIT_REGISTRATION_ENABLED"):
        raise RegisteredWaitConflict(
            "new wait registration is disabled; acknowledged waits continue to drain"
        )
    if request.adapter is not None and not _enabled(
        "CAO_M7_WAIT_MONITOR_CONSUMER_ENABLED", default=False
    ):
        raise RegisteredWaitConflict("adapter registration requires an attached monitor consumer")
    observed = now or _now()
    canonical = request.canonical()
    digest = canonical_sha256(canonical)
    wait_id = wait_id_for(request.operation_id)
    adopting_pending = False
    has_adapter = request.adapter is not None
    # --- intent phase: wait row + monitor intent ---
    try:
        with database.SessionLocal() as db:
            existing = _operation_row(db, request.operation_id)
            if existing is not None and existing.state != STATE_REGISTRATION_PENDING:
                if existing.request_digest != digest:
                    raise RegisteredWaitConflict(
                        f"operation {request.operation_id} is a divergent registration replay"
                    )
                if has_adapter:
                    mon = db.get(database.RegisteredWaitMonitorModel, existing.wait_id)
                    if mon is not None and mon.request_digest != digest:
                        raise RegisteredWaitConflict(
                            f"operation {request.operation_id} monitor digest diverges"
                        )
                result = _record(existing, now=observed)
                result["adopted"] = True
                return result
            if existing is not None and existing.request_digest != digest:
                raise RegisteredWaitConflict(
                    f"operation {request.operation_id} is a divergent registration replay"
                )
            if existing is not None:
                adopting_pending = True
                if has_adapter:
                    mon = db.get(database.RegisteredWaitMonitorModel, wait_id)
                    if mon is not None and mon.request_digest != digest:
                        raise RegisteredWaitConflict(
                            f"operation {request.operation_id} monitor digest diverges"
                        )
            else:
                row = database.RegisteredWaitModel(
                    wait_id=wait_id,
                    operation_id=request.operation_id,
                    request_digest=digest,
                    request_json=encode_canonical(canonical).decode("utf-8"),
                    session_name=request.session_name,
                    owner_agent_id=request.owner.agent_id,
                    owner_incarnation_id=request.owner.incarnation_id,
                    owner_terminal_id=request.owner.terminal_id,
                    owner_generation=request.owner.generation,
                    state=STATE_REGISTRATION_PENDING,
                    deadline_at=_isots(observed + timedelta(seconds=request.duration_seconds)),
                    expiry_operation_id=expiry_operation_id_for(wait_id),
                    created_at=_isots(observed),
                    updated_at=_isots(observed),
                )
                db.add(row)
                db.commit()
                if has_adapter:
                    from cli_agent_orchestrator.services.registered_wait_monitors import (
                        create_monitor_intent,
                        monitor_run_dir,
                    )

                    run_dir = monitor_run_dir(wait_id)
                    paths = create_monitor_intent(
                        wait_id, digest, request.adapter, request.duration_seconds, run_dir
                    )
                    mon = database.RegisteredWaitMonitorModel(
                        wait_id=wait_id,
                        request_digest=digest,
                        run_dir=str(run_dir),
                        state="launch-intent",
                        created_at=_isots(observed),
                        updated_at=_isots(observed),
                    )
                    db.add(mon)
                    db.commit()
            if has_adapter and adopting_pending:
                mon = db.get(database.RegisteredWaitMonitorModel, wait_id)
                if mon is None:
                    from cli_agent_orchestrator.services.registered_wait_monitors import (
                        create_monitor_intent,
                        monitor_run_dir,
                    )

                    run_dir = monitor_run_dir(wait_id)
                    paths = create_monitor_intent(
                        wait_id, digest, request.adapter, request.duration_seconds, run_dir
                    )
                    mon = database.RegisteredWaitMonitorModel(
                        wait_id=wait_id,
                        request_digest=digest,
                        run_dir=str(run_dir),
                        state="launch-intent",
                        created_at=_isots(observed),
                        updated_at=_isots(observed),
                    )
                    db.add(mon)
                    db.commit()
    except RegisteredWaitError:
        raise
    except (OperationalError, SQLAlchemyError) as exc:
        raise RegisteredWaitUnavailable(f"registration intent write failed: {exc}") from exc

    verdict = wait_admission.verify_owner(request.owner)
    state = STATE_ACKNOWLEDGED if verdict["denial_reason"] is None else STATE_INVALID
    outcome = None
    if state == STATE_INVALID:
        outcome = {
            "reason_code": verdict["denial_reason"],
            "detail": verdict["detail"],
            "at": _isots(observed),
        }
    if has_adapter:
        if state == STATE_INVALID:
            try:
                with database.SessionLocal() as db:
                    stored = _wait_row(db, wait_id)
                    mon = db.get(database.RegisteredWaitMonitorModel, wait_id)
                    if stored is not None and stored.state == STATE_REGISTRATION_PENDING:
                        stored.state = state
                        stored.outcome_json = (
                            json.dumps(outcome, sort_keys=True) if outcome else None
                        )
                        stored.updated_at = _isots(observed)
                    if mon is not None and mon.state == "launch-intent":
                        mon.state = "invalid"
                        mon.outcome_json = json.dumps(outcome, sort_keys=True) if outcome else None
                        mon.updated_at = _isots(observed)
                    db.commit()
                    result = _record(stored, now=observed) if stored else {}
                    result["adopted"] = adopting_pending
                    return result
            except (OperationalError, SQLAlchemyError) as exc:
                raise RegisteredWaitUnavailable(
                    f"registration acknowledgement failed: {exc}"
                ) from exc
        # owner valid: adopt or launch
        try:
            from cli_agent_orchestrator.services.registered_wait_monitors import (
                _helper_alive,
                adopt_monitor_evidence,
                launch_dormant_runner,
                monitor_paths,
                monitor_paths_for_monitor,
            )

            with database.SessionLocal() as db:
                mon = db.get(database.RegisteredWaitMonitorModel, wait_id)
                wait_row = _wait_row(db, wait_id)
                if mon is None or wait_row is None:
                    raise RegisteredWaitUnavailable("registration intent disappeared")
                if mon.state != "launch-intent":
                    if wait_row.state == STATE_REGISTRATION_PENDING:
                        # only ack if monitor has proof (active/result)
                        if mon.state in {"active", "result-ready", "wake-pending", "completed"}:
                            wait_row.state = STATE_ACKNOWLEDGED
                            wait_row.updated_at = _isots(observed)
                            db.commit()
                    result = _record(wait_row, now=observed)
                    result["adopted"] = adopting_pending
                    return result
                # launch-intent
                paths = monitor_paths_for_monitor(mon)
                has_ready = paths["ready"].exists()
                has_result = paths["result"].exists()
                if adopting_pending and not has_ready and not has_result:
                    # ambiguous window: stay pending, unmonitored, never relaunch
                    result = _record(wait_row, now=observed)
                    result["adopted"] = True
                    return result
                if not adopting_pending and not has_ready and not has_result:
                    # fresh: launch dormant runner
                    try:
                        proc = launch_dormant_runner(paths)
                        import time as _time

                        deadline = _time.monotonic() + 2.0
                        while _time.monotonic() < deadline:
                            if paths["ready"].exists() or paths["result"].exists():
                                break
                            _time.sleep(0.05)
                    except Exception as exc:
                        raise RegisteredWaitUnavailable(f"monitor launch failed: {exc}") from exc
                    # try to adopt whatever appeared
                    adopted = adopt_monitor_evidence(wait_id)
                    with database.SessionLocal() as db2:
                        w2 = _wait_row(db2, wait_id)
                        m2 = db2.get(database.RegisteredWaitMonitorModel, wait_id)
                        # ack only if adoption succeeded (helper or result recorded)
                        if (
                            adopted is not None
                            and w2 is not None
                            and w2.state == STATE_REGISTRATION_PENDING
                        ):
                            # adopted already acked inside adopt if valid; ensure ack
                            if m2 is not None and m2.state in {"active", "result-ready"}:
                                if w2.state == STATE_REGISTRATION_PENDING:
                                    w2.state = STATE_ACKNOWLEDGED
                                    w2.updated_at = _isots(observed)
                                    db2.commit()
                        result = _record(w2, now=observed) if w2 else {}
                        result["adopted"] = False
                        return result
                # adopting_pending with ready/result present: adopt
                adopted = adopt_monitor_evidence(wait_id)
                with database.SessionLocal() as db2:
                    w2 = _wait_row(db2, wait_id)
                    m2 = db2.get(database.RegisteredWaitMonitorModel, wait_id)
                    if (
                        adopted is not None
                        and w2 is not None
                        and w2.state == STATE_REGISTRATION_PENDING
                    ):
                        if m2 is not None and m2.state in {"active", "result-ready"}:
                            w2.state = STATE_ACKNOWLEDGED
                            w2.updated_at = _isots(observed)
                            db2.commit()
                    # if adoption failed (invalid/mismatch), stay pending
                    result = _record(w2, now=observed) if w2 else {}
                    result["adopted"] = adopting_pending
                    return result
        except RegisteredWaitError:
            raise
        except (OperationalError, SQLAlchemyError) as exc:
            raise RegisteredWaitUnavailable(f"registration acknowledgement failed: {exc}") from exc

    # timer path
    try:
        with database.SessionLocal() as db:
            stored = _wait_row(db, wait_id)
            if stored is None:
                raise RegisteredWaitUnavailable("registration intent disappeared")
            if stored.state == STATE_REGISTRATION_PENDING:
                stored.state = state
                stored.outcome_json = json.dumps(outcome, sort_keys=True) if outcome else None
                stored.updated_at = _isots(observed)
                db.commit()
            result = _record(stored, now=observed)
            result["adopted"] = adopting_pending
            return result
    except RegisteredWaitError:
        raise
    except (OperationalError, SQLAlchemyError) as exc:
        raise RegisteredWaitUnavailable(f"registration acknowledgement failed: {exc}") from exc


def get(wait_id: str) -> Optional[dict[str, Any]]:
    _uuid(wait_id, "wait_id")
    try:
        with database.SessionLocal() as db:
            row = _wait_row(db, wait_id)
            return _record(row) if row is not None else None
    except RegisteredWaitError:
        raise
    except (OperationalError, SQLAlchemyError) as exc:
        raise RegisteredWaitUnavailable(f"wait read failed: {exc}") from exc


def get_by_operation(operation_id: str) -> Optional[dict[str, Any]]:
    _uuid(operation_id, "operation_id")
    try:
        with database.SessionLocal() as db:
            row = _operation_row(db, operation_id)
            return _record(row) if row is not None else None
    except RegisteredWaitError:
        raise
    except (OperationalError, SQLAlchemyError) as exc:
        raise RegisteredWaitUnavailable(f"wait operation read failed: {exc}") from exc


def list_waits(
    *, session_name: Optional[str] = None, terminal_id: Optional[str] = None
) -> list[dict[str, Any]]:
    try:
        with database.SessionLocal() as db:
            query = db.query(database.RegisteredWaitModel)
            if session_name is not None:
                query = query.filter(database.RegisteredWaitModel.session_name == session_name)
            if terminal_id is not None:
                query = query.filter(database.RegisteredWaitModel.owner_terminal_id == terminal_id)
            return [_record(row) for row in query.order_by(database.RegisteredWaitModel.created_at)]
    except RegisteredWaitError:
        raise
    except (OperationalError, SQLAlchemyError) as exc:
        raise RegisteredWaitUnavailable(f"wait list failed: {exc}") from exc


def cancel(
    wait_id: str, *, operation_id: str, actor: str, now: Optional[datetime] = None
) -> dict[str, Any]:
    """Cancel before wake delivery; replaying the same terminal result is harmless."""
    wait_id = _uuid(wait_id, "wait_id")
    operation_id = _uuid(operation_id, "operation_id")
    actor = _text(actor, "actor")
    observed = now or _now()
    try:
        with database.SessionLocal() as db:
            initial = _wait_row(db, wait_id)
            if initial is None:
                raise RegisteredWaitInvalid(f"unknown wait {wait_id}")
            message_id = initial.wake_message_id
            # For adapter waits, also consider monitor wake
            monitor = db.get(database.RegisteredWaitMonitorModel, wait_id)
            monitor_wake = monitor.wake_message_id if monitor is not None else None
            adapter_monitor = monitor is not None
            # need both ids for fencing
            combined_wake = message_id if message_id is not None else monitor_wake
            if adapter_monitor and initial.state in TERMINAL_STATES:
                outcome = json.loads(initial.outcome_json) if initial.outcome_json else None
                if initial.state == STATE_CANCELLED and outcome:
                    if (
                        outcome.get("operation_id") == operation_id
                        and outcome.get("actor") != actor
                    ):
                        raise RegisteredWaitConflict(
                            f"cancellation operation {operation_id} is a divergent replay"
                        )
                result = _record(initial, now=observed)
                result["adopted"] = True
                return result
        if adapter_monitor:
            from cli_agent_orchestrator.services.registered_wait_monitors import stop_monitor

            try:
                stop_monitor(
                    wait_id,
                    operation_id,
                    actor,
                    disposition=STATE_CANCELLED,
                )
            except RegisteredWaitUnavailable:
                raise
            except Exception as exc:
                raise RegisteredWaitUnavailable(f"monitor cancel failed: {exc}") from exc
            with database.SessionLocal() as db:
                row = _wait_row(db, wait_id)
                if row is None:
                    raise RegisteredWaitInvalid(f"unknown wait {wait_id}")
                result = _record(row, now=observed)
                result["adopted"] = False
                return result
        lock: Any = nullcontext()
        if combined_wake is not None:
            from cli_agent_orchestrator.services.inbox_service import InboxService

            lock = InboxService._managed_delivery_lock(combined_wake)
        with lock, database.SessionLocal() as db:
            row = _wait_row(db, wait_id)
            if row is None:
                raise RegisteredWaitInvalid(f"unknown wait {wait_id}")
            if row.wake_message_id != message_id:
                # Expiry installed the immutable inbox row between the first
                # read and lock selection. Retry once through its delivery
                # stripe before deciding which terminal outcome can win.
                return cancel(wait_id, operation_id=operation_id, actor=actor, now=observed)
            outcome = json.loads(row.outcome_json) if row.outcome_json else None
            if row.state == STATE_CANCELLED:
                if outcome and outcome.get("operation_id") == operation_id:
                    if outcome.get("actor") != actor:
                        raise RegisteredWaitConflict(
                            f"cancellation operation {operation_id} is a divergent replay"
                        )
                result = _record(row, now=observed)
                result["adopted"] = True
                return result
            if row.state in TERMINAL_STATES:
                result = _record(row, now=observed)
                result["adopted"] = True
                return result
            if row.wake_message_id is not None:
                message = _inbox_row(db, row.wake_message_id)
                if message is not None and message.status != MessageStatus.PENDING.value:
                    # A provider effect may already exist. Cancellation cannot
                    # erase it or claim it did not happen.
                    return _record(row, now=observed)
                if message is not None:
                    message.status = MessageStatus.FAILED.value
            outcome_json = json.dumps(
                {
                    "reason_code": "cancelled",
                    "operation_id": operation_id,
                    "actor": actor,
                    "at": _isots(observed),
                },
                sort_keys=True,
            )
            updated = (
                db.query(database.RegisteredWaitModel)
                .filter(
                    database.RegisteredWaitModel.wait_id == wait_id,
                    database.RegisteredWaitModel.state.in_(ACTIVE_STATES),
                )
                .update(
                    {
                        database.RegisteredWaitModel.state: STATE_CANCELLED,
                        database.RegisteredWaitModel.outcome_json: outcome_json,
                        database.RegisteredWaitModel.updated_at: _isots(observed),
                    },
                    synchronize_session=False,
                )
            )
            if updated != 1:
                db.rollback()
                return get(wait_id)  # type: ignore[return-value]
            db.commit()
            row = _wait_row(db, wait_id)
            result = _record(row, now=observed)
            result["adopted"] = False
            return result
    except RegisteredWaitError:
        raise
    except (OperationalError, SQLAlchemyError) as exc:
        raise RegisteredWaitUnavailable(f"wait cancellation failed: {exc}") from exc


Delivery = Callable[[str], None]
ReceiptProbe = Callable[[str, int], Optional[Mapping[str, Any]]]
InputHold = Callable[[str, str], bool]


def _mark_terminal(
    row: Any, *, state: str, reason: str, detail: Optional[str], now: datetime
) -> None:
    row.state = state
    row.outcome_json = json.dumps(
        {"reason_code": reason, "detail": detail, "at": _isots(now)}, sort_keys=True
    )
    row.updated_at = _isots(now)


def _wake_text(record: Mapping[str, Any]) -> str:
    condition = record["condition"]
    round_ = record["round"]
    return (
        f"Scheduled wait expired: {condition['name']} ({record['wait_id']}). "
        f"{condition['description']} Round {round_['number']} after "
        f"{round_['max_seconds']} seconds; total elapsed "
        f"{record['totals']['elapsed_seconds']} seconds."
    )


def process_due(
    *,
    now: Optional[datetime] = None,
    deliver: Optional[Delivery] = None,
    receipt_probe: Optional[ReceiptProbe] = None,
    input_held: Optional[InputHold] = None,
    ambiguity_grace_seconds: int = DEFAULT_AMBIGUITY_GRACE_SECONDS,
) -> list[dict[str, Any]]:
    """Advance due waits, querying one stored wake operation without resending."""
    if not _enabled("CAO_M7_WAIT_CONSUMER_ENABLED"):
        return []
    observed = now or _now()
    results: list[dict[str, Any]] = []
    # Snapshot ids only. Every transition reloads and checks current state.
    try:
        with database.SessionLocal() as db:
            ids = [
                str(row.wait_id)
                for row in db.query(database.RegisteredWaitModel)
                .filter(database.RegisteredWaitModel.state.in_(ACTIVE_STATES))
                .order_by(database.RegisteredWaitModel.deadline_at)
            ]
    except (OperationalError, SQLAlchemyError) as exc:
        raise RegisteredWaitUnavailable(f"due-wait scan failed: {exc}") from exc

    for wait_id in ids:
        try:
            result = _process_due_wait(
                wait_id,
                observed=observed,
                deliver=deliver,
                receipt_probe=receipt_probe,
                input_held=input_held,
                ambiguity_grace_seconds=ambiguity_grace_seconds,
            )
        except (RegisteredWaitError, wait_admission.WaitAdmissionError) as exc:
            results.append(
                {
                    "wait_id": wait_id,
                    "state": "unreadable",
                    "reason_code": exc.code,
                    "detail": str(exc),
                }
            )
            continue
        if result is not None:
            results.append(result)
    return results


def _settle_expiry_wake(
    wait_id: str,
    message_id: int,
    terminal_id: str,
    observed: datetime,
    receipt_probe: Optional[ReceiptProbe],
    ambiguity_grace_seconds: int,
) -> Optional[dict[str, Any]]:
    """Settle an installed wake from durable evidence without repeating admission."""
    receipt = receipt_probe(terminal_id, message_id) if receipt_probe else None
    with database.SessionLocal() as db:
        current = _wait_row(db, wait_id)
        if current is None or current.state != STATE_EXPIRY_WAKE_PENDING:
            return None
        inbox = _inbox_row(db, message_id)
        if receipt is not None:
            _mark_terminal(
                current,
                state=STATE_RESOLVED,
                reason="wake-confirmed",
                detail=f"durable receiver receipt for inbox message {message_id}",
                now=observed,
            )
        elif inbox is not None and inbox.status == MessageStatus.DELIVERED.value:
            _mark_terminal(
                current,
                state=STATE_RESOLVED,
                reason="wake-delivered",
                detail=f"inbox message {message_id} reached durable DELIVERED",
                now=observed,
            )
        elif inbox is not None and inbox.status == MessageStatus.FAILED.value:
            _mark_terminal(
                current,
                state=STATE_INVALID,
                reason="wake-refused",
                detail=f"inbox message {message_id} was refused",
                now=observed,
            )
        elif observed >= _parse_time(current.wake_pending_since) + timedelta(
            seconds=ambiguity_grace_seconds
        ):
            if inbox is not None and inbox.status == MessageStatus.PENDING.value:
                inbox.status = MessageStatus.FAILED.value
            _mark_terminal(
                current,
                state=STATE_INVALID,
                reason="expiry-wake-ambiguous",
                detail=f"no durable delivery evidence after {ambiguity_grace_seconds} seconds",
                now=observed,
            )
        db.commit()
        return _record(current, now=observed)


def _process_due_wait(
    wait_id: str,
    *,
    observed: datetime,
    deliver: Optional[Delivery],
    receipt_probe: Optional[ReceiptProbe],
    input_held: Optional[InputHold],
    ambiguity_grace_seconds: int,
) -> Optional[dict[str, Any]]:
    """Advance one wait so an unreadable row cannot starve later rows."""
    pending_wake: Optional[tuple[str, int]] = None
    with database.SessionLocal() as db:
        row = _wait_row(db, wait_id)
        if row is None or row.state not in ACTIVE_STATES:
            return None
        request = _request_from_row(row)
        # Adapter waits are never processed as scheduled timers.
        if isinstance(request, dict) and request.get("adapter") is not None:
            return None
        if row.state == STATE_EXPIRY_WAKE_PENDING:
            if row.wake_message_id is None:
                raise RegisteredWaitUnavailable(f"wait {wait_id} has no installed wake")
            pending_wake = (row.owner_terminal_id, int(row.wake_message_id))
        else:
            owner = _owner(request)
            verdict = wait_admission.verify_owner(owner, db=db)
            if verdict["denial_reason"] is not None:
                if verdict["denial_reason"] == wait_admission.DENY_OWNER_UNREADABLE:
                    return {"wait_id": wait_id, "state": "owner-unreadable"}
                _mark_terminal(
                    row,
                    state=STATE_INVALID,
                    reason=str(verdict["denial_reason"]),
                    detail=verdict["detail"],
                    now=observed,
                )
                db.commit()
                return _record(row, now=observed)
            if row.state == STATE_REGISTRATION_PENDING:
                row.state = STATE_ACKNOWLEDGED
                row.updated_at = _isots(observed)
                db.commit()
                if observed < _parse_time(row.deadline_at):
                    return None
            if observed < _parse_time(row.deadline_at) and row.state == STATE_ACKNOWLEDGED:
                return None
            if row.state == STATE_ACKNOWLEDGED:
                row.state = STATE_EXPIRY_INTENT
                row.updated_at = _isots(observed)
                db.commit()  # expiry intent before any delivery I/O
            if input_held and input_held(row.owner_terminal_id, row.owner_generation):
                return {"wait_id": wait_id, "state": STATE_EXPIRY_INTENT}

    if pending_wake is not None:
        terminal_id, message_id = pending_wake
        return _settle_expiry_wake(
            wait_id,
            message_id,
            terminal_id,
            observed,
            receipt_probe,
            ambiguity_grace_seconds,
        )

    # Admit once, outside the state transaction. A retry adopts this exact
    # immutable verdict. Cancellation may win while admission runs.
    record = get(wait_id)
    if record is None:
        return None
    created_message = False
    with database.SessionLocal() as db:
        current = _wait_row(db, wait_id)
        if current is None or current.state not in {
            STATE_EXPIRY_INTENT,
            STATE_EXPIRY_WAKE_PENDING,
        }:
            return None
        request = _request_from_row(current)
        owner = _owner(request)
        message_id = str(uuid.uuid5(_MESSAGE_NAMESPACE, current.expiry_operation_id))
        admission = wait_admission.admit(
            wait_admission.AdmissionRequest(
                operation_id=current.expiry_operation_id,
                session_name=current.session_name,
                owner=owner,
                message=wait_admission.WaitMessage(
                    message_id=message_id,
                    kind=wait_admission.KIND_EXPIRY,
                    reason_code="scheduled-wait-expired",
                    payload_digest=current.request_digest,
                    source_operation_id=current.operation_id,
                    text=_wake_text(record),
                ),
            ),
            db=db,
        )
        if admission["admission_state"] != wait_admission.STATE_ADMITTED:
            _mark_terminal(
                current,
                state=STATE_INVALID,
                reason=str(admission["denial_reason"]),
                detail=admission.get("detail"),
                now=observed,
            )
            db.commit()
            return _record(current, now=observed)
        if current.wake_message_id is None:
            inbox = database.InboxModel(
                sender_id=current.owner_terminal_id,
                receiver_id=current.owner_terminal_id,
                message=_wake_text(record),
                status=MessageStatus.PENDING.value,
                sender_generation=current.owner_generation,
                expected_receiver_generation=current.owner_generation,
            )
            db.add(inbox)
            db.flush()
            installed = (
                db.query(database.RegisteredWaitModel)
                .filter(
                    database.RegisteredWaitModel.wait_id == wait_id,
                    database.RegisteredWaitModel.state == STATE_EXPIRY_INTENT,
                    database.RegisteredWaitModel.wake_message_id.is_(None),
                )
                .update(
                    {
                        database.RegisteredWaitModel.state: STATE_EXPIRY_WAKE_PENDING,
                        database.RegisteredWaitModel.wake_message_id: inbox.id,
                        database.RegisteredWaitModel.wake_pending_since: _isots(observed),
                        database.RegisteredWaitModel.updated_at: _isots(observed),
                    },
                    synchronize_session=False,
                )
            )
            if installed != 1:
                db.rollback()
                return None
            message_id_int = int(inbox.id)
            created_message = True
        else:
            message_id_int = int(current.wake_message_id)
        terminal_id = current.owner_terminal_id
        db.commit()

    # A response-loss retry never calls this for a row whose inbox message
    # has already left PENDING; it only queries the durable evidence below.
    with database.SessionLocal() as db:
        inbox = _inbox_row(db, message_id_int)
        should_deliver = inbox is not None and inbox.status == MessageStatus.PENDING.value
    if created_message and should_deliver and deliver is not None:
        try:
            deliver(terminal_id)
        except Exception:
            # Outcome is ambiguous. The stored message and operation are
            # queried during grace; they are never recreated or resent.
            pass

    return _settle_expiry_wake(
        wait_id,
        message_id_int,
        terminal_id,
        observed,
        receipt_probe,
        ambiguity_grace_seconds,
    )


def deadman_disposition(terminal_id: str, generation: str) -> dict[str, Any]:
    """Owner-scoped ordinary-deadman decision; unreadable is not absence.

    Pure read: settling of stale monitors is performed by ``process_monitors``;
    deadman only reads durable truth and suppresses only exact monitored+
    acknowledged before deadline.  Unreadable suppresses only this exact wait.
    """
    try:
        with database.SessionLocal() as db:
            if not sa_inspect(db.get_bind()).has_table(database.RegisteredWaitModel.__tablename__):
                return {"state": "absent", "suppress_ordinary_deadman": False}
            rows = (
                db.query(database.RegisteredWaitModel)
                .filter(
                    database.RegisteredWaitModel.owner_terminal_id == terminal_id,
                    database.RegisteredWaitModel.owner_generation == generation,
                    database.RegisteredWaitModel.state.in_(ACTIVE_STATES),
                )
                .all()
            )
            if not rows:
                return {"state": "absent", "suppress_ordinary_deadman": False}
            for row in rows:
                try:
                    request = _request_from_row(row)
                except RegisteredWaitUnavailable as exc:
                    return {
                        "state": "unreadable",
                        "suppress_ordinary_deadman": True,
                        "detail": str(exc),
                    }
                verdict = wait_admission.verify_owner(_owner(request), db=db)
                if verdict["denial_reason"] == wait_admission.DENY_OWNER_UNREADABLE:
                    return {
                        "state": "unreadable",
                        "suppress_ordinary_deadman": True,
                        "detail": verdict["detail"],
                    }
                # Adapter waits: health drives suppression
                if isinstance(request, dict) and request.get("adapter") is not None:
                    try:
                        from cli_agent_orchestrator.services.registered_wait_monitors import (
                            monitor_health_for,
                        )

                        health = monitor_health_for(row.wait_id)
                    except Exception as exc:
                        return {
                            "state": "unreadable",
                            "suppress_ordinary_deadman": True,
                            "detail": str(exc),
                        }
                    detail = str(health.get("detail") or "")
                    if health.get("health") == "monitor-stale" or detail.startswith("unreadable"):
                        if detail.startswith("unreadable"):
                            return {
                                "state": "unreadable",
                                "suppress_ordinary_deadman": True,
                                "detail": detail,
                            }
                        # stale: do not suppress, let process_monitors settle
                        continue
                    if (
                        verdict["denial_reason"] is None
                        and row.state == STATE_ACKNOWLEDGED
                        and _now() < _parse_time(str(row.deadline_at))
                        and health.get("health") == "monitored"
                    ):
                        return {
                            "state": STATE_ACKNOWLEDGED,
                            "wait_id": row.wait_id,
                            "suppress_ordinary_deadman": True,
                        }
                    continue
                if (
                    verdict["denial_reason"] is None
                    and row.state == STATE_ACKNOWLEDGED
                    and _now() < _parse_time(str(row.deadline_at))
                ):
                    return {
                        "state": STATE_ACKNOWLEDGED,
                        "wait_id": row.wait_id,
                        "suppress_ordinary_deadman": True,
                    }
            return {"state": "not-active", "suppress_ordinary_deadman": False}
    except (OperationalError, SQLAlchemyError) as exc:
        return {
            "state": "unreadable",
            "suppress_ordinary_deadman": True,
            "detail": str(exc),
        }


def interrupt_session_waits(session_name: str, operation_id: str) -> Sequence[Mapping[str, Any]]:
    """Stop exit verb: interrupt every still-live wait in the stopped cohort."""
    operation_id = _uuid(operation_id, "operation_id")
    observed = _now()
    try:
        with database.SessionLocal() as db:
            wait_ids = [
                str(row.wait_id)
                for row in (
                    db.query(database.RegisteredWaitModel)
                    .filter(
                        database.RegisteredWaitModel.session_name == session_name,
                        database.RegisteredWaitModel.state.in_(ACTIVE_STATES),
                    )
                    .all()
                )
            ]
        results = []
        for wait_id in wait_ids:
            with database.SessionLocal() as db:
                initial = _wait_row(db, wait_id)
                monitor = db.get(database.RegisteredWaitMonitorModel, wait_id)
                # For adapter, need monitor wake id for fencing
                message_id = initial.wake_message_id if initial is not None else None
                monitor_wake = monitor.wake_message_id if monitor is not None else None
                adapter_monitor = monitor is not None
                combined = message_id if message_id is not None else monitor_wake
            if adapter_monitor:
                from cli_agent_orchestrator.services.registered_wait_monitors import stop_monitor

                try:
                    result = stop_monitor(wait_id, operation_id, "stop")
                except RegisteredWaitUnavailable:
                    raise
                except Exception as exc:
                    raise RegisteredWaitUnavailable(f"monitor stop failed: {exc}") from exc
                results.append({"wait_id": wait_id, "state": result["state"]})
                continue
            lock: Any = nullcontext()
            if combined is not None:
                from cli_agent_orchestrator.services.inbox_service import InboxService

                lock = InboxService._managed_delivery_lock(combined)
            with lock, database.SessionLocal() as db:
                row = _wait_row(db, wait_id)
                if row is None or row.state not in ACTIVE_STATES:
                    continue
                if row.wake_message_id != message_id:
                    # Expiry installed a message while Stop selected the row.
                    # Revisit it after the outer pass through its exact delivery
                    # stripe rather than claiming interruption without fencing.
                    wait_ids.append(wait_id)
                    continue
                if row.wake_message_id is not None:
                    inbox = _inbox_row(db, row.wake_message_id)
                    if inbox is not None and inbox.status == MessageStatus.PENDING.value:
                        inbox.status = MessageStatus.FAILED.value
                _mark_terminal(
                    row,
                    state=STATE_INTERRUPTED,
                    reason="interrupted-by-stop",
                    detail=f"session Stop operation {operation_id}",
                    now=observed,
                )
                results.append({"wait_id": row.wait_id, "state": row.state})
                db.commit()
        return results
    except RegisteredWaitError:
        raise
    except (OperationalError, SQLAlchemyError) as exc:
        raise RegisteredWaitUnavailable(f"Stop wait interruption failed: {exc}") from exc


def process_monitors(
    *,
    now: Optional[datetime] = None,
    deliver: Optional[Callable[[str], None]] = None,
    receipt_probe: Optional[Callable[[str, int], Optional[Mapping[str, Any]]]] = None,
    attach_result: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    input_held: Optional[Callable[[str, str], bool]] = None,
    ambiguity_grace_seconds: int = DEFAULT_AMBIGUITY_GRACE_SECONDS,
) -> list[dict[str, Any]]:
    """Sentinel entry point for monitor result persistence and wake delivery."""
    from cli_agent_orchestrator.services.registered_wait_monitors import process_monitors as _pm

    return _pm(
        now=now,
        deliver=deliver,
        receipt_probe=receipt_probe,
        attach_result=attach_result,
        input_held=input_held,
        ambiguity_grace_seconds=ambiguity_grace_seconds,
    )
