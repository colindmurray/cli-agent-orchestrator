"""Project home and CAO-session history projections.

The issue tracker owns project grouping.  This module turns that grouping into
read-only dashboard views by joining evidence CAO already records: explicit
session scopes, issue filing context, managed-launch workdirs, live pane
workdirs, terminal rows, and teardown snapshots.  It does not create a second
project or session registry.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from sqlalchemy.exc import OperationalError

from cli_agent_orchestrator.backends.registry import get_backend
from cli_agent_orchestrator.clients.database import (
    ManagedLaunchReservationModel,
    ManagedLaunchV2ReservationModel,
    ManagedLaunchV2TerminalModel,
    SessionLocal,
    TerminalModel,
    TrackerIssueModel,
    TrackerProjectModel,
    TrackerScopeModel,
)
from cli_agent_orchestrator.constants import SESSION_PREFIX, TERMINAL_LOG_DIR
from cli_agent_orchestrator.services import issue_tracker

_TERMINAL_ID = re.compile(r"^[a-f0-9]{8}$")
_OPEN_STATUSES = tuple(
    s for s in issue_tracker.STATUSES if s not in issue_tracker.TERMINAL_STATUSES
)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    return text or None


def _under_any(path: Optional[str], roots: Iterable[str]) -> bool:
    if not path:
        return False
    try:
        candidate = os.path.realpath(os.path.abspath(path))
    except (OSError, TypeError, ValueError):
        return False
    for root in roots:
        try:
            if os.path.commonpath((candidate, root)) == root:
                return True
        except (OSError, TypeError, ValueError):
            continue
    return False


def _issue_brief(row: TrackerIssueModel) -> Dict[str, Any]:
    return {
        "key": row.key,
        "kind": getattr(row, "kind", "bug") or "bug",
        "title": row.title,
        "status": row.status,
        "severity": row.severity,
        "assignee": row.assignee,
        "favorite": bool(getattr(row, "favorite", False)),
        "session_name": row.session_name,
        "terminal_id": row.terminal_id,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def _snapshot_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        paths = sorted(Path(TERMINAL_LOG_DIR).glob("*.snapshot.json"))[-10_000:]
    except OSError:
        return records
    for path in paths:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            terminal_id = body.get("terminal_id")
            session_name = body.get("session_name")
            if not isinstance(terminal_id, str) or not _TERMINAL_ID.fullmatch(terminal_id):
                continue
            if not isinstance(session_name, str) or not session_name:
                continue
            observed = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            records.append(
                {
                    "terminal_id": terminal_id,
                    "session_name": session_name,
                    "name": body.get("window_name"),
                    "provider": body.get("provider"),
                    "agent_profile": body.get("agent_profile"),
                    "caller_id": body.get("caller_id"),
                    "working_directory": body.get("working_directory"),
                    "protocol_vintage": "snapshot",
                    "lifecycle_state": "historical",
                    "status": "historical",
                    "native_session_id": None,
                    "generation": None,
                    "last_active": _iso(observed),
                    "first_seen": _iso(observed),
                    "last_seen": _iso(observed),
                    "snapshot_available": True,
                }
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return records


def _model_records(db: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    terminals: List[Dict[str, Any]] = []
    reservations: List[Dict[str, Any]] = []
    try:
        v1_rows = db.query(TerminalModel).all()
    except OperationalError:
        v1_rows = []
    for row in v1_rows:
        terminals.append(
            {
                "terminal_id": row.id,
                "session_name": row.tmux_session,
                "name": row.tmux_window,
                "provider": row.provider,
                "agent_profile": row.agent_profile,
                "caller_id": row.caller_id,
                "generation": row.generation,
                "native_session_id": row.native_session_id,
                "protocol_vintage": "v1",
                "lifecycle_state": row.lifecycle_state,
                "status": row.lifecycle_state,
                "last_active": _iso(row.last_active),
                "pane_id": row.pane_id,
            }
        )
    try:
        v2_rows = db.query(ManagedLaunchV2TerminalModel).all()
    except OperationalError:
        v2_rows = []
    for row in v2_rows:
        terminals.append(
            {
                "terminal_id": row.id,
                "session_name": row.tmux_session,
                "name": row.tmux_window,
                "provider": row.provider,
                "agent_profile": row.agent_profile,
                "caller_id": row.caller_id,
                "generation": row.generation,
                "native_session_id": row.v2_native_session_id,
                "protocol_vintage": "v2",
                "lifecycle_state": row.v2_lifecycle_state,
                "status": row.v2_lifecycle_state,
                "last_active": _iso(row.last_active),
                "pane_id": row.pane_id,
            }
        )
    for model, vintage in (
        (ManagedLaunchReservationModel, "v1"),
        (ManagedLaunchV2ReservationModel, "v2"),
    ):
        try:
            rows = db.query(model).all()
        except OperationalError:
            rows = []
        for row in rows:
            reservations.append(
                {
                    "terminal_id": row.terminal_id,
                    "session_name": row.session_name,
                    "working_directory": row.working_directory,
                    "trusted_project_root": row.trusted_project_root,
                    "created_at": _iso(row.created_at),
                    "updated_at": _iso(row.updated_at),
                    "protocol_vintage": vintage,
                }
            )
    return terminals, reservations


def _live_context(terminals: List[Dict[str, Any]]) -> Tuple[Set[str], Dict[str, Set[str]], Any]:
    try:
        backend = get_backend()
        listed = backend.list_sessions()
    except Exception:  # an unreadable backend must not erase durable history
        return set(), {}, None
    live = {
        str(row.get("id") or row.get("name"))
        for row in listed
        if isinstance(row, dict)
        and str(row.get("id") or row.get("name") or "").startswith(SESSION_PREFIX)
    }
    workdirs: Dict[str, Set[str]] = defaultdict(set)
    for terminal in terminals:
        session_name = terminal.get("session_name")
        window_name = terminal.get("name")
        if session_name not in live or not window_name:
            continue
        try:
            cwd = backend.get_pane_working_directory(session_name, window_name)
        except Exception:
            cwd = None
        if cwd:
            terminal["working_directory"] = cwd
            workdirs[session_name].add(cwd)
    return live, workdirs, backend


def _merge_terminal(target: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(target)
    for key, value in incoming.items():
        if value is not None and (result.get(key) is None or key in {"status", "lifecycle_state"}):
            result[key] = value
    return result


def _session_projection(project_id: str) -> Dict[str, Any]:
    slug = issue_tracker._validate_slug(project_id)
    snapshots = _snapshot_records()
    with SessionLocal() as db:
        if db.get(TrackerProjectModel, slug) is None:
            raise issue_tracker.TrackerError("not-found", f"no such project: {slug}")
        scopes = db.query(TrackerScopeModel).filter(TrackerScopeModel.project_id == slug).all()
        issues = db.query(TrackerIssueModel).filter(TrackerIssueModel.project_id == slug).all()
        terminals, reservations = _model_records(db)

    path_roots = [os.path.realpath(scope.value) for scope in scopes if scope.kind == "path"]
    associated: Dict[str, Set[str]] = defaultdict(set)
    for scope in scopes:
        if scope.kind == "session":
            associated[scope.value].add("session scope")
    for issue in issues:
        if issue.session_name:
            associated[issue.session_name].add("issue filing")
    for reservation in reservations:
        if _under_any(reservation.get("working_directory"), path_roots) or _under_any(
            reservation.get("trusted_project_root"), path_roots
        ):
            associated[reservation["session_name"]].add("managed worktree")
    for snapshot in snapshots:
        if _under_any(snapshot.get("working_directory"), path_roots):
            associated[snapshot["session_name"]].add("archived worktree")

    live, live_workdirs, _backend = _live_context(terminals)
    for session_name, workdirs in live_workdirs.items():
        if any(_under_any(path, path_roots) for path in workdirs):
            associated[session_name].add("live worktree")

    # A conductor campaign alias conventionally maps to cao-<alias>. Only add
    # it when there is durable or live evidence for that session; the name
    # convention alone never fabricates a historical run.
    known_sessions = {
        str(row.get("session_name"))
        for row in [*terminals, *reservations, *snapshots]
        if row.get("session_name")
    } | live
    for scope in scopes:
        if scope.kind == "project_id":
            candidate = f"{SESSION_PREFIX}{scope.value}"
            if candidate in known_sessions:
                associated[candidate].add("campaign alias")

    by_terminal: Dict[str, Dict[str, Any]] = {}
    reservation_by_terminal = {r["terminal_id"]: r for r in reservations}
    for snapshot in snapshots:
        if snapshot["session_name"] in associated:
            by_terminal[snapshot["terminal_id"]] = snapshot
    for terminal in terminals:
        if terminal["session_name"] not in associated:
            continue
        terminal_id = terminal["terminal_id"]
        merged = _merge_terminal(by_terminal.get(terminal_id, {}), terminal)
        reservation = reservation_by_terminal.get(terminal_id)
        if reservation:
            merged = _merge_terminal(merged, reservation)
            merged["first_seen"] = reservation.get("created_at")
            merged["last_seen"] = reservation.get("updated_at") or merged.get("last_active")
        by_terminal[terminal_id] = merged

    # For live associated sessions, use the shared human projection so the
    # project page and the main fleet dashboard report the same worker state.
    for session_name in sorted(set(associated) & live):
        try:
            from cli_agent_orchestrator.services import terminal_projection

            projected = terminal_projection.project_session(session_name)
        except Exception:
            projected = []
        for row in projected:
            terminal_id = row.get("terminal_id") or row.get("id")
            if not terminal_id:
                continue
            incoming = {
                "terminal_id": terminal_id,
                "session_name": session_name,
                "name": row.get("name") or row.get("tmux_window"),
                "provider": row.get("provider"),
                "agent_profile": row.get("agent_profile"),
                "caller_id": row.get("caller_id"),
                "generation": row.get("generation"),
                "native_session_id": row.get("native_session_id"),
                "protocol_vintage": row.get("protocol_vintage"),
                "lifecycle_state": row.get("lifecycle_state"),
                "status": row.get("status"),
                "last_active": _iso(row.get("last_active")),
                "pane_id": row.get("pane_id"),
                "wedged": bool(row.get("wedged")),
            }
            by_terminal[terminal_id] = _merge_terminal(by_terminal.get(terminal_id, {}), incoming)

    issues_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    issues_by_terminal: Dict[str, List[str]] = defaultdict(list)
    for issue in issues:
        brief = _issue_brief(issue)
        if issue.session_name:
            issues_by_session[issue.session_name].append(brief)
        if issue.terminal_id:
            issues_by_terminal[issue.terminal_id].append(issue.key)

    terminals_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for terminal in by_terminal.values():
        terminal_id = terminal["terminal_id"]
        terminal["issue_keys"] = sorted(set(issues_by_terminal.get(terminal_id, [])))
        terminal["snapshot_available"] = bool(
            terminal.get("snapshot_available")
            or (Path(TERMINAL_LOG_DIR) / f"{terminal_id}.snapshot.json").is_file()
        )
        terminal["log_available"] = any(
            (Path(TERMINAL_LOG_DIR) / f"{terminal_id}{suffix}").is_file()
            for suffix in (".scrollback", ".log")
        )
        terminals_by_session[terminal["session_name"]].append(terminal)

    sessions: List[Dict[str, Any]] = []
    for session_name, reasons in associated.items():
        workers = sorted(
            terminals_by_session.get(session_name, []),
            key=lambda row: (
                row.get("last_seen") or row.get("last_active") or "",
                row["terminal_id"],
            ),
            reverse=True,
        )
        timestamps = [
            value
            for row in workers
            for value in (row.get("first_seen"), row.get("last_seen"), row.get("last_active"))
            if value
        ]
        workdirs = sorted(
            {row["working_directory"] for row in workers if row.get("working_directory")}
        )
        providers = sorted({row["provider"] for row in workers if row.get("provider")})
        session_issues = sorted(
            issues_by_session.get(session_name, []),
            key=lambda row: row.get("updated_at") or "",
            reverse=True,
        )
        is_live = session_name in live
        sessions.append(
            {
                "name": session_name,
                "status": "active" if is_live else "historical",
                "live": is_live,
                "associated_by": sorted(reasons),
                "worker_count": len(workers),
                "active_workers": sum(
                    1 for worker in workers if is_live and worker.get("lifecycle_state") == "live"
                ),
                "providers": providers,
                "workdirs": workdirs,
                "issue_count": len(session_issues),
                "artifact_count": sum(
                    int(bool(worker.get("snapshot_available")))
                    + int(bool(worker.get("log_available")))
                    for worker in workers
                ),
                "first_seen": min(timestamps) if timestamps else None,
                "last_seen": max(timestamps) if timestamps else None,
                "terminals": workers,
                "issues": session_issues,
            }
        )
    sessions.sort(key=lambda row: (row.get("last_seen") or "", row["name"]), reverse=True)
    sessions.sort(key=lambda row: not row["live"])
    return {
        "project_id": slug,
        "total": len(sessions),
        "active": sum(1 for row in sessions if row["live"]),
        "historical": sum(1 for row in sessions if not row["live"]),
        "sessions": sessions,
    }


def project_sessions(project_id: str) -> Dict[str, Any]:
    projection = _session_projection(project_id)
    return {
        **projection,
        "sessions": [
            {key: value for key, value in row.items() if key not in {"terminals", "issues"}}
            for row in projection["sessions"]
        ],
    }


def project_session(project_id: str, session_name: str) -> Dict[str, Any]:
    projection = _session_projection(project_id)
    for row in projection["sessions"]:
        if row["name"] == session_name:
            return {"project_id": projection["project_id"], "session": row}
    raise issue_tracker.TrackerError(
        "not-found", f"session {session_name!r} is not associated with project {project_id!r}"
    )


def project_home(project_id: str) -> Dict[str, Any]:
    slug = issue_tracker._validate_slug(project_id)
    with SessionLocal() as db:
        if db.get(TrackerProjectModel, slug) is None:
            raise issue_tracker.TrackerError("not-found", f"no such project: {slug}")
        rows = db.query(TrackerIssueModel).filter(TrackerIssueModel.project_id == slug).all()
    rows.sort(key=lambda row: (_iso(row.updated_at) or "", row.key), reverse=True)
    open_rows = [row for row in rows if row.status in _OPEN_STATUSES]
    favorites = [_issue_brief(row) for row in rows if bool(getattr(row, "favorite", False))][:8]
    urgent = [_issue_brief(row) for row in open_rows if row.severity in {"P0", "P1"}][:8]
    recent = [_issue_brief(row) for row in rows[:8]]
    sessions = project_sessions(slug)
    return {
        "project_id": slug,
        "issues": {
            "open": len(open_rows),
            "in_progress": sum(1 for row in rows if row.status == "in-progress"),
            "favorites": favorites,
            "urgent": urgent,
            "recent": recent,
        },
        "sessions": {
            "total": sessions["total"],
            "active": sessions["active"],
            "historical": sessions["historical"],
            "recent": sessions["sessions"][:5],
        },
    }


def terminal_log(
    project_id: str,
    session_name: str,
    terminal_id: str,
    *,
    mode: str = "last",
) -> Dict[str, Any]:
    if not _TERMINAL_ID.fullmatch(terminal_id):
        raise issue_tracker.TrackerError(
            "invalid", "terminal_id must be eight lowercase hex characters"
        )
    detail = project_session(project_id, session_name)["session"]
    if not any(row["terminal_id"] == terminal_id for row in detail["terminals"]):
        raise issue_tracker.TrackerError(
            "not-found", f"terminal {terminal_id!r} is not part of session {session_name!r}"
        )
    if mode not in {"last", "full"}:
        raise issue_tracker.TrackerError("invalid", "log mode must be last or full")
    candidates = (
        Path(TERMINAL_LOG_DIR) / f"{terminal_id}.scrollback",
        Path(TERMINAL_LOG_DIR) / f"{terminal_id}.log",
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise issue_tracker.TrackerError("not-found", f"no captured log for terminal {terminal_id}")
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise issue_tracker.TrackerError(
            "conflict", f"captured log for {terminal_id} is unreadable"
        ) from exc
    truncated = False
    if mode == "last":
        lines = text.splitlines()
        truncated = len(lines) > 200
        text = "\n".join(lines[-200:])
    elif len(text) > 2_000_000:
        truncated = True
        text = text[-2_000_000:]
    return {
        "terminal_id": terminal_id,
        "session_name": session_name,
        "mode": mode,
        "output": text,
        "truncated": truncated,
        "source": "archived-scrollback" if source.suffix == ".scrollback" else "terminal-log",
    }
