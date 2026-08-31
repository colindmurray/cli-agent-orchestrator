"""Project-scoped issue tracking.

A project is a *declared* grouping: one name, any number of directories,
sessions and git remotes, one issue log. That shape is the whole point — the
CAO system spans two repositories (``cao-conductor`` and
``cli-agent-orchestrator``) plus their worktrees plus a rotating cast of tmux
sessions, and its defects belong to one list rather than to whichever checkout
happened to notice them.

Resolution is deliberately ordered and deliberately strict:

    explicit project id  ->  session name  ->  path  ->  git remote

Explicit wins because a caller that names a project has already answered the
question. Session beats path because a supervisor and its workers share a
session while working across several directories. Path beats git remote
because a worktree is a directory fact before it is a remote fact. And an
identifier resolving to two projects is refused at *registration* time rather
than resolved by row order at read time: a scope that silently picks one of
two issue logs is worse than no scope, because the filing appears to succeed.

Nothing here reads or writes ``ProjectAliasModel``. That table is the memory
subsystem's automatic identity cache; grouping two repos under one issue log
must not merge their memory recall as a side effect.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import func, text
from sqlalchemy.exc import OperationalError

from cli_agent_orchestrator.clients.database import (
    SessionLocal,
    TrackerCommentModel,
    TrackerEventModel,
    TrackerIssueModel,
    TrackerLinkModel,
    TrackerProjectModel,
    TrackerScopeModel,
)
from cli_agent_orchestrator.services.tracker_filters import StructuredFilters

logger = logging.getLogger(__name__)

ITEM_KINDS: Tuple[str, ...] = (
    "project",
    "bug",
    "feature",
    "milestone",
    "goal",
    "epic",
    "story",
    "task",
)

# Workflow vocabulary. `open` and `closed` are the load-bearing ends; the
# middle states exist so a long-running project can distinguish "nobody has
# looked at this" from "somebody is on it" from "it is waiting on something
# else", which is the distinction that makes a stale tracker readable.
STATUSES: Tuple[str, ...] = (
    "open",
    "triage",
    "in-progress",
    "blocked",
    "resolved",
    "closed",
    "wontfix",
    "duplicate",
)
# Statuses that mean the issue no longer needs attention. `resolved` is NOT
# terminal: it means a fix landed and verification has not been recorded.
TERMINAL_STATUSES: frozenset = frozenset({"closed", "wontfix", "duplicate"})

# P0 exists because the ledger being migrated uses it for the most severe
# class ("Lane D live acceptance can kill the production tmux server"). Two
# entries carry it; folding them into P1 would erase the one distinction their
# author made. Ordered most-severe-first, which is also their sort order.
SEVERITIES: Tuple[str, ...] = ("P0", "P1", "P2", "P3", "P4", "unset")

SCOPE_KINDS: Tuple[str, ...] = ("path", "session", "git_remote", "project_id")

# Directed link vocabulary. `blocks` runs blocker -> blocked ("a blocks b"
# means b waits on a). `part-of` runs child -> parent/map and is the only
# membership edge: `list_children`/`frontier` read it, and nothing else
# implies membership.
LINK_KINDS: Tuple[str, ...] = ("blocks", "relates", "duplicates", "caused-by", "part-of")

PROJECT_STATUSES: Tuple[str, ...] = ("active", "archived")

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_PREFIX_RE = re.compile(r"^[a-z][a-z0-9-]{0,15}$")
_KEY_RE = re.compile(r"^[a-z][a-z0-9-]{0,15}-\d{1,9}$")

# Fields a caller may PATCH, and whether an empty string clears them.
_EDITABLE_FIELDS: Tuple[str, ...] = (
    "title",
    "body",
    "status",
    "severity",
    "component",
    "assignee",
    "reporter",
    "collaborators",
    "branches",
    "worktrees",
    "pull_requests",
    "failing_command",
    "reproduction_steps",
    "expected_outcome",
    "actual_outcome",
    "evidence",
    "observed_revision",
    "resolution",
    "duplicate_of",
    "labels",
    "kind",
    "favorite",
    # Atomic label deltas (cond-0394). One strategy per update: `labels`
    # replaces the whole set, `clear_labels` replaces it with nothing, and
    # add/remove apply a delta — the three never combine.
    "add_labels",
    "remove_labels",
    "clear_labels",
)

MAX_TITLE = 300
MAX_BODY = 200_000
MAX_LABELS = 32
MAX_LABEL_LEN = 64
MAX_CONTEXT_VALUES = 64
MAX_CONTEXT_VALUE_LEN = 4096


class TrackerError(Exception):
    """A refusal the caller can act on.

    ``code`` maps to an HTTP status at the API boundary and to an exit code at
    the CLI boundary, so the same refusal reads the same way from both.

    ``details`` carries the observed record state a conflict was raised from
    (the current assignee on a lost claim, the current version on a stale
    write), so a programmatic caller does not have to parse the message to
    retry.
    """

    def __init__(self, code: str, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: Dict[str, Any] = dict(details or {})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    """Serialize a stored timestamp as an explicit-UTC ISO-8601 string.

    Every timestamp in these tables is written by ``_utcnow``, so a naive value
    read back from SQLite is UTC that lost its tzinfo in transit — stamping it
    UTC restores the truth rather than assuming it. (Contrast the conductor's
    ``last_active``, which is naive *local* time; assuming UTC there shifts
    every reading by the host offset.)
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(text: str, *, field: str) -> datetime:
    """Parse an ISO-8601 timestamp a caller read back from an issue payload.

    A naive value is read as UTC, the same convention ``_iso`` writes with —
    every stored timestamp in these tables is UTC.
    """
    raw = str(text or "").strip()
    if not raw:
        raise TrackerError("invalid", f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError:
        raise TrackerError("invalid", f"{field} is not an ISO-8601 timestamp: {text!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _stored_moment(value: Optional[datetime]) -> Optional[datetime]:
    """A stored timestamp as an aware UTC datetime for comparison."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_labels(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, (str, int, float))]


def _parse_string_list(raw: Optional[str]) -> List[str]:
    """Read a stored JSON string list, tolerating historical malformed data."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str) and item.strip()]


def normalise_string_list(values: Optional[Iterable[Any]], *, field: str) -> List[str]:
    """Trim, de-duplicate, and bound an open-ended repeatable issue field."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out: List[str] = []
    seen = set()
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        if len(text) > MAX_CONTEXT_VALUE_LEN:
            raise TrackerError(
                "invalid",
                f"{field} value too long (max {MAX_CONTEXT_VALUE_LEN} characters)",
            )
        seen.add(text)
        out.append(text)
        if len(out) > MAX_CONTEXT_VALUES:
            raise TrackerError("invalid", f"too many {field} values (max {MAX_CONTEXT_VALUES})")
    return out


def normalise_labels(labels: Optional[Iterable[Any]]) -> List[str]:
    """Trim, de-duplicate and bound a label list, preserving first-seen order."""
    if labels is None:
        return []
    if isinstance(labels, str):
        labels = [chunk for chunk in re.split(r"[,\s]+", labels) if chunk]
    out: List[str] = []
    seen = set()
    for item in labels:
        text = str(item).strip()
        if not text:
            continue
        if len(text) > MAX_LABEL_LEN:
            raise TrackerError("invalid", f"label too long (max {MAX_LABEL_LEN}): {text[:32]}...")
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) > MAX_LABELS:
            raise TrackerError("invalid", f"too many labels (max {MAX_LABELS})")
    return out


def slugify(text: str) -> str:
    """Derive a project slug from a display name.

    Never invents characters: it lowercases, collapses runs of anything
    non-alphanumeric into a single hyphen, and trims. An input that reduces to
    nothing is an error rather than a generated id, because a project called
    ``"???"`` silently becoming ``project-1`` is a naming decision the caller
    did not make.
    """
    lowered = str(text or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    slug = slug[:64].rstrip("-")
    if not slug:
        raise TrackerError("invalid", f"cannot derive a project id from {text!r}")
    return slug


def _validate_slug(slug: str) -> str:
    slug = str(slug or "").strip().lower()
    if not _SLUG_RE.match(slug):
        raise TrackerError(
            "invalid",
            f"invalid project id {slug!r}: use lowercase letters, digits, '.', '_' or '-'",
        )
    return slug


def _validate_prefix(prefix: str) -> str:
    prefix = str(prefix or "").strip().lower()
    if not _PREFIX_RE.match(prefix):
        raise TrackerError(
            "invalid",
            f"invalid issue prefix {prefix!r}: use 1-16 lowercase letters, digits or '-'",
        )
    return prefix


def _validate_choice(value: str, allowed: Sequence[str], label: str) -> str:
    text = str(value or "").strip()
    if text not in allowed:
        raise TrackerError(
            "invalid", f"invalid {label} {text!r}: expected one of {', '.join(allowed)}"
        )
    return text


def _validate_kind(kind: str) -> str:
    text = str(kind or "").strip().lower()
    if text not in ITEM_KINDS:
        raise TrackerError(
            "invalid", f"invalid kind {text!r}: expected one of {', '.join(ITEM_KINDS)}"
        )
    return text


def normalise_scope_value(kind: str, value: str) -> str:
    """Canonicalise a scope value so equal things compare equal.

    Paths are realpath'd (a worktree symlinked into place must not register as
    a second, competing scope) and stripped of a trailing separator. Git
    remotes lose their scheme, credentials, ``.git`` suffix and trailing slash,
    so ``git@github.com:o/r.git`` and ``https://github.com/o/r`` are one scope.
    """
    kind = _validate_choice(kind, SCOPE_KINDS, "scope kind")
    text = str(value or "").strip()
    if not text:
        raise TrackerError("invalid", "scope value must not be empty")
    if kind == "path":
        expanded = os.path.expanduser(text)
        # Checked BEFORE realpath, which would resolve a relative path against
        # whatever directory the server happens to be running in and register
        # a scope the caller never named.
        if not os.path.isabs(expanded):
            raise TrackerError("invalid", f"path scope must be absolute: {text!r}")
        try:
            resolved = os.path.realpath(expanded)
        except OSError:
            resolved = os.path.abspath(expanded)
        return resolved.rstrip(os.sep) or os.sep
    if kind == "git_remote":
        return _normalise_remote(text)
    return text


def _normalise_remote(url: str) -> str:
    """Strip a git remote down to ``host/owner/repo``.

    Credentials are dropped rather than stored: a remote can carry
    ``https://user:token@host/...`` and this row is readable by every dashboard
    client.
    """
    text = url.strip()
    text = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", text)
    text = re.sub(r"^[^/@]*@", "", text)  # user[:token]@host
    text = re.sub(r"^([^/:]+):(\d+)/", r"\1/", text)  # ssh://host:22/o/r -> host/o/r
    text = text.replace(":", "/", 1) if "/" not in text.split(":", 1)[0] else text
    text = re.sub(r"\.git$", "", text)
    text = text.rstrip("/")
    return text.lower()


def _path_contains(scope: str, candidate: str) -> bool:
    """Whether ``candidate`` is ``scope`` or lives inside it.

    Compares path components, never string prefixes: ``/a/foo`` must not claim
    ``/a/foobar``, and that difference is exactly the
    ``cao-conductor`` / ``cao-conductor-worktrees`` case this system has to get
    right on day one.
    """
    if candidate == scope:
        return True
    return candidate.startswith(scope.rstrip(os.sep) + os.sep)


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------


def _project_row(
    row: TrackerProjectModel, *, counts: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": row.id,
        "name": row.name,
        "description": row.description or "",
        "status": row.status,
        "issue_prefix": row.issue_prefix,
        "next_issue_number": row.next_issue_number,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }
    if counts is not None:
        out["counts"] = counts
    return out


def create_project(
    *,
    name: str,
    project_id: Optional[str] = None,
    description: str = "",
    issue_prefix: Optional[str] = None,
    scopes: Optional[Iterable[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Create a project and, atomically, its initial scopes.

    Scopes are validated and inserted in the same transaction as the project.
    A half-created project whose scopes were rejected would resolve nothing and
    look like a working project, so either all of it exists or none of it does.
    """
    display = str(name or "").strip()
    if not display:
        raise TrackerError("invalid", "project name must not be empty")
    slug = _validate_slug(project_id) if project_id else slugify(display)
    prefix = _validate_prefix(issue_prefix) if issue_prefix else _default_prefix(slug)

    prepared = [
        (
            normalise_scope_value(s.get("kind", ""), s.get("value", "")),
            _validate_choice(s.get("kind", ""), SCOPE_KINDS, "scope kind"),
        )
        for s in (scopes or [])
    ]
    # Two scopes in ONE request that normalise to the same value used to reach
    # the unique constraint and surface as a raw 500. It is the same conflict as
    # a cross-project clash and deserves the same answer.
    seen_values = set()
    for value, _kind in prepared:
        if value in seen_values:
            raise TrackerError("conflict", f"scope {value!r} is listed twice in this request")
        seen_values.add(value)

    with SessionLocal() as db:
        if db.get(TrackerProjectModel, slug) is not None:
            raise TrackerError("conflict", f"project {slug!r} already exists")
        _assert_prefix_free(db, prefix, owner=slug)
        for value, _kind in prepared:
            clash = db.query(TrackerScopeModel).filter(TrackerScopeModel.value == value).first()
            if clash is not None:
                raise TrackerError(
                    "conflict",
                    f"scope {value!r} is already registered to project {clash.project_id!r}",
                )
        row = TrackerProjectModel(
            id=slug,
            name=display,
            description=str(description or ""),
            status="active",
            issue_prefix=prefix,
            next_issue_number=1,
        )
        db.add(row)
        for value, kind in prepared:
            db.add(TrackerScopeModel(project_id=slug, kind=kind, value=value))
        db.commit()
        db.refresh(row)
        return _project_row(row)


def _assert_prefix_free(db: Any, prefix: str, *, owner: str) -> None:
    """Refuse a key prefix another project already uses.

    Issue keys are unique across the whole installation, not per project, so
    that `cond-0242` in a commit message, a report or an evidence path means
    exactly one thing. Two projects sharing a prefix would therefore collide at
    key-allocation time with a confusing "issue already exists" from a project
    the caller never mentioned. Refusing here moves that failure to the moment
    somebody chooses the prefix, where it is actionable.
    """
    clash = (
        db.query(TrackerProjectModel)
        .filter(TrackerProjectModel.issue_prefix == prefix, TrackerProjectModel.id != owner)
        .first()
    )
    if clash is not None:
        raise TrackerError(
            "conflict",
            f"issue prefix {prefix!r} is already used by project {clash.id!r}; "
            "issue keys are unique across the installation, so each project needs its own",
        )

    # A project that VACATED a prefix leaves its issues behind still holding
    # those keys, so "no project owns it" is not the same as "the namespace is
    # free". Handing it to a new project made every filing there collide with a
    # key that already exists — and because the counter bump and the insert
    # share one transaction, the rollback took the bump with it, so the counter
    # never advanced and the wedge was permanent. The key table is the real
    # namespace; ask it.
    taken = (
        db.query(TrackerIssueModel)
        .filter(TrackerIssueModel.key.like(f"{prefix}-%"))
        .filter(TrackerIssueModel.project_id != owner)
        .first()
    )
    if taken is not None:
        raise TrackerError(
            "conflict",
            f"issue prefix {prefix!r} is still held by existing keys "
            f"(e.g. {taken.key} in project {taken.project_id!r}); keys outlive the "
            "project that minted them, so the prefix is not free",
        )


def _default_prefix(slug: str) -> str:
    """Derive an issue prefix from a slug: initials for multiword, else the slug."""
    parts = [p for p in re.split(r"[^a-z0-9]+", slug) if p]
    if len(parts) > 1:
        initials = "".join(p[0] for p in parts)[:16]
        if _PREFIX_RE.match(initials):
            return initials
    candidate = (parts[0] if parts else slug)[:16]
    return candidate if _PREFIX_RE.match(candidate) else "issue"


def list_projects(*, include_archived: bool = False) -> List[Dict[str, Any]]:
    """List projects with per-status issue counts (issue-only legacy) plus by_kind."""
    with SessionLocal() as db:
        query = db.query(TrackerProjectModel)
        if not include_archived:
            query = query.filter(TrackerProjectModel.status == "active")
        rows = query.order_by(TrackerProjectModel.name.asc()).all()
        # Single snapshot for all counts to avoid concurrency drift (PR1-2 gate 4)
        all_issues = db.query(TrackerIssueModel).all()
        # Compute tallies in Python from one snapshot
        tallies: Dict[str, int] = {}
        open_tallies: Dict[str, int] = {}
        all_tallies: Dict[str, int] = {}
        all_open: Dict[str, int] = {}
        by_kind_total: Dict[str, Dict[str, int]] = {k: {} for k in ITEM_KINDS}
        by_kind_open: Dict[str, Dict[str, int]] = {k: {} for k in ITEM_KINDS}
        for iss in all_issues:
            pid = iss.project_id
            kind = getattr(iss, "kind", "bug")
            is_open = iss.status not in TERMINAL_STATUSES
            # all
            all_tallies[pid] = all_tallies.get(pid, 0) + 1
            if is_open:
                all_open[pid] = all_open.get(pid, 0) + 1
            # by_kind
            by_kind_total[kind][pid] = by_kind_total[kind].get(pid, 0) + 1
            if is_open:
                by_kind_open[kind][pid] = by_kind_open[kind].get(pid, 0) + 1
            # legacy issue-only
            if kind == "bug":
                tallies[pid] = tallies.get(pid, 0) + 1
                if is_open:
                    open_tallies[pid] = open_tallies.get(pid, 0) + 1
        out = []
        for row in rows:
            counts: Dict[str, Any] = {
                "total": int(tallies.get(row.id, 0)),
                "open": int(open_tallies.get(row.id, 0)),
                "all_total": int(all_tallies.get(row.id, 0)),
                "all_open": int(all_open.get(row.id, 0)),
                "by_kind": {
                    k: {
                        "total": int(by_kind_total[k].get(row.id, 0)),
                        "open": int(by_kind_open[k].get(row.id, 0)),
                    }
                    for k in ITEM_KINDS
                },
            }
            out.append(_project_row(row, counts=counts))
        return out


def get_project(project_id: str) -> Dict[str, Any]:
    """Return one project with its scopes and issue counts."""
    slug = _validate_slug(project_id)
    with SessionLocal() as db:
        row = db.get(TrackerProjectModel, slug)
        if row is None:
            raise TrackerError("not-found", f"no such project: {slug}")
        scopes = (
            db.query(TrackerScopeModel)
            .filter(TrackerScopeModel.project_id == slug)
            .order_by(TrackerScopeModel.kind.asc(), TrackerScopeModel.value.asc())
            .all()
        )
        # Single snapshot for project stats (PR1-2 gate 4)
        all_issues = db.query(TrackerIssueModel).filter(TrackerIssueModel.project_id == slug).all()
        by_status: Dict[str, int] = {}
        all_by_status: Dict[str, int] = {}
        by_kind: Dict[str, Any] = {k: {"total": 0, "open": 0, "by_status": {}} for k in ITEM_KINDS}
        for iss in all_issues:
            kind = getattr(iss, "kind", "bug")
            all_by_status[iss.status] = all_by_status.get(iss.status, 0) + 1
            if kind == "bug":
                by_status[iss.status] = by_status.get(iss.status, 0) + 1
            # per-kind
            sub = by_kind[kind]
            sub["by_status"][iss.status] = sub["by_status"].get(iss.status, 0) + 1
        for k in ITEM_KINDS:
            sub = by_kind[k]["by_status"]
            by_kind[k]["total"] = sum(int(v) for v in sub.values())
            by_kind[k]["open"] = sum(int(v) for kk, v in sub.items() if kk not in TERMINAL_STATUSES)
        total = sum(int(v) for v in by_status.values())
        open_count = sum(int(v) for k, v in by_status.items() if k not in TERMINAL_STATUSES)
        payload = _project_row(
            row,
            counts={
                "total": total,
                "open": open_count,
                "by_status": {k: int(v) for k, v in by_status.items()},
                "by_kind": by_kind,
                "all_total": sum(int(v) for v in all_by_status.values()),
                "all_open": sum(
                    int(v) for kk, v in all_by_status.items() if kk not in TERMINAL_STATUSES
                ),
            },
        )
        payload["scopes"] = [
            {"id": s.id, "kind": s.kind, "value": s.value, "created_at": _iso(s.created_at)}
            for s in scopes
        ]
        return payload


def update_project(
    project_id: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    issue_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Rename, re-describe, archive or re-prefix a project.

    Changing ``issue_prefix`` affects only keys allocated afterwards. Existing
    keys are never rewritten — they are quoted in commit messages, reports and
    evidence paths that this database cannot reach.
    """
    slug = _validate_slug(project_id)
    with SessionLocal() as db:
        row = db.get(TrackerProjectModel, slug)
        if row is None:
            raise TrackerError("not-found", f"no such project: {slug}")
        if name is not None:
            display = str(name).strip()
            if not display:
                raise TrackerError("invalid", "project name must not be empty")
            row.name = display
        if description is not None:
            row.description = str(description)
        if status is not None:
            row.status = _validate_choice(status, PROJECT_STATUSES, "project status")
        if issue_prefix is not None:
            candidate = _validate_prefix(issue_prefix)
            _assert_prefix_free(db, candidate, owner=slug)
            row.issue_prefix = candidate
        db.commit()
        db.refresh(row)
        return _project_row(row)


def delete_project(project_id: str, *, force: bool = False) -> Dict[str, Any]:
    """Delete a project, its scopes, and (only with ``force``) its issues.

    Refuses by default when issues exist. Deleting an issue log because
    somebody tidied a project list is not recoverable, and the archived state
    covers every non-destructive reason to make a project go away.
    """
    slug = _validate_slug(project_id)
    with SessionLocal() as db:
        row = db.get(TrackerProjectModel, slug)
        if row is None:
            raise TrackerError("not-found", f"no such project: {slug}")
        issues = db.query(TrackerIssueModel).filter(TrackerIssueModel.project_id == slug).all()
        if issues and not force:
            raise TrackerError(
                "conflict",
                f"project {slug!r} still holds {len(issues)} issue(s); archive it, or pass force to delete them",
            )
        keys = [issue.key for issue in issues]
        if keys:
            db.query(TrackerCommentModel).filter(TrackerCommentModel.issue_key.in_(keys)).delete(
                synchronize_session=False
            )
            db.query(TrackerEventModel).filter(TrackerEventModel.issue_key.in_(keys)).delete(
                synchronize_session=False
            )
            db.query(TrackerLinkModel).filter(
                (TrackerLinkModel.from_key.in_(keys)) | (TrackerLinkModel.to_key.in_(keys))
            ).delete(synchronize_session=False)
            db.query(TrackerIssueModel).filter(TrackerIssueModel.project_id == slug).delete(
                synchronize_session=False
            )
        db.query(TrackerScopeModel).filter(TrackerScopeModel.project_id == slug).delete(
            synchronize_session=False
        )
        db.delete(row)
        db.commit()
        return {"id": slug, "deleted": True, "issues_deleted": len(keys)}


# --------------------------------------------------------------------------
# scopes
# --------------------------------------------------------------------------


def add_scope(project_id: str, *, kind: str, value: str) -> Dict[str, Any]:
    """Register one identifier as resolving to this project."""
    slug = _validate_slug(project_id)
    kind = _validate_choice(kind, SCOPE_KINDS, "scope kind")
    canonical = normalise_scope_value(kind, value)
    with SessionLocal() as db:
        if db.get(TrackerProjectModel, slug) is None:
            raise TrackerError("not-found", f"no such project: {slug}")
        existing = db.query(TrackerScopeModel).filter(TrackerScopeModel.value == canonical).first()
        if existing is not None:
            if existing.project_id == slug and existing.kind == kind:
                return {
                    "id": existing.id,
                    "project_id": existing.project_id,
                    "kind": existing.kind,
                    "value": existing.value,
                    "created_at": _iso(existing.created_at),
                    "created": False,
                }
            raise TrackerError(
                "conflict",
                f"scope {canonical!r} is already registered to project {existing.project_id!r}",
            )
        row = TrackerScopeModel(project_id=slug, kind=kind, value=canonical)
        db.add(row)
        db.commit()
        db.refresh(row)
        return {
            "id": row.id,
            "project_id": row.project_id,
            "kind": row.kind,
            "value": row.value,
            "created_at": _iso(row.created_at),
            "created": True,
        }


def remove_scope(project_id: str, scope_id: int) -> Dict[str, Any]:
    """Drop one scope row."""
    slug = _validate_slug(project_id)
    with SessionLocal() as db:
        row = db.get(TrackerScopeModel, int(scope_id))
        if row is None or row.project_id != slug:
            raise TrackerError("not-found", f"no scope {scope_id} on project {slug}")
        db.delete(row)
        db.commit()
        return {"id": int(scope_id), "deleted": True}


def list_scopes(project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with SessionLocal() as db:
        query = db.query(TrackerScopeModel)
        if project_id:
            query = query.filter(TrackerScopeModel.project_id == _validate_slug(project_id))
        rows = query.order_by(
            TrackerScopeModel.project_id.asc(), TrackerScopeModel.kind.asc()
        ).all()
        return [
            {
                "id": r.id,
                "project_id": r.project_id,
                "kind": r.kind,
                "value": r.value,
                "created_at": _iso(r.created_at),
            }
            for r in rows
        ]


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    """How a filing site resolved, and by what evidence.

    ``matched_by`` is part of the answer, not decoration: a caller that filed
    into the wrong project needs to see whether it was the session, the path or
    the remote that put it there.
    """

    project_id: Optional[str]
    matched_by: Optional[str]
    matched_value: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "matched_by": self.matched_by,
            "matched_value": self.matched_value,
        }


def resolve_project(
    *,
    project: Optional[str] = None,
    session: Optional[str] = None,
    alias: Optional[str] = None,
    cwd: Optional[str] = None,
    git_remote: Optional[str] = None,
) -> Resolution:
    """Resolve a filing site to a project.

    ``alias`` matches a ``project_id``-kind scope: an identifier from some
    other system that names this project — a conductor campaign name, a CAO
    memory project id. It ranks just under ``session`` because, like a session,
    it is something the caller stated rather than something inferred from where
    a process happens to be running.

    Returns an *unmatched* Resolution rather than raising when nothing matches:
    "which project is this?" has a legitimate answer of "none registered", and
    callers differ on whether that is fatal.
    """
    if project:
        slug = _validate_slug(project)
        with SessionLocal() as db:
            if db.get(TrackerProjectModel, slug) is None:
                raise TrackerError("not-found", f"no such project: {slug}")
        return Resolution(slug, "explicit", slug)

    with SessionLocal() as db:
        if session:
            row = (
                db.query(TrackerScopeModel)
                .filter(TrackerScopeModel.kind == "session", TrackerScopeModel.value == session)
                .first()
            )
            if row is not None:
                return Resolution(row.project_id, "session", row.value)

        if alias:
            row = (
                db.query(TrackerScopeModel)
                .filter(TrackerScopeModel.kind == "project_id", TrackerScopeModel.value == alias)
                .first()
            )
            if row is not None:
                return Resolution(row.project_id, "alias", row.value)

        if cwd:
            resolved = normalise_scope_value("path", cwd)
            candidates = db.query(TrackerScopeModel).filter(TrackerScopeModel.kind == "path").all()
            # Longest scope wins: a project registered on
            # ~/Projects/cao-conductor/gateway is more specific than one
            # registered on ~/Projects, and the specific answer is the one the
            # operator meant.
            best: Optional[TrackerScopeModel] = None
            for candidate in candidates:
                if _path_contains(candidate.value, resolved):
                    if best is None or len(candidate.value) > len(best.value):
                        best = candidate
            if best is not None:
                return Resolution(best.project_id, "path", best.value)

        remote = git_remote or (_git_remote_for(cwd) if cwd else None)
        if remote:
            canonical = _normalise_remote(remote)
            row = (
                db.query(TrackerScopeModel)
                .filter(
                    TrackerScopeModel.kind == "git_remote", TrackerScopeModel.value == canonical
                )
                .first()
            )
            if row is not None:
                return Resolution(row.project_id, "git_remote", row.value)

    return Resolution(None, None, None)


def _git_remote_for(cwd: str) -> Optional[str]:
    """Read ``remote.origin.url`` for a directory, or None.

    Bounded and non-fatal: resolution runs on the filing path, and a slow or
    broken git must degrade to "no remote match", never hang the filing.
    """
    import subprocess

    try:
        path = Path(os.path.expanduser(cwd))
        if not path.is_dir():
            return None
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"git remote lookup failed in {cwd}: {exc}")
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


# --------------------------------------------------------------------------
# issues
# --------------------------------------------------------------------------


def _issue_row(row: TrackerIssueModel) -> Dict[str, Any]:
    return {
        "key": row.key,
        "project_id": row.project_id,
        "kind": getattr(row, "kind", "bug") or "bug",
        "title": row.title,
        "body": row.body or "",
        "status": row.status,
        "severity": row.severity,
        "component": row.component,
        "reporter": row.reporter,
        "assignee": row.assignee,
        "labels": _parse_labels(row.labels),
        "collaborators": _parse_string_list(row.collaborators),
        "branches": _parse_string_list(row.branches),
        "worktrees": _parse_string_list(row.worktrees),
        "pull_requests": _parse_string_list(row.pull_requests),
        "failing_command": row.failing_command,
        "reproduction_steps": row.reproduction_steps,
        "expected_outcome": row.expected_outcome,
        "actual_outcome": row.actual_outcome,
        "evidence": row.evidence,
        "observed_revision": row.observed_revision,
        "resolution": row.resolution,
        "session_name": row.session_name,
        "terminal_id": row.terminal_id,
        "source_path": row.source_path,
        "duplicate_of": row.duplicate_of,
        "origin": row.origin,
        "favorite": bool(getattr(row, "favorite", False)),
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "closed_at": _iso(row.closed_at),
    }


def _allocate_key(db: Any, project: TrackerProjectModel) -> str:
    """Allocate the next issue key by compare-and-swap on the counter.

    SQLite has no ``SELECT ... FOR UPDATE``, so the counter is claimed with a
    conditional UPDATE and the caller retries if somebody else won. The unique
    constraint on ``key`` is the backstop: two racers can never both persist
    the same key even if this loop were wrong.
    """
    for _ in range(50):
        current = int(project.next_issue_number or 1)
        claimed = (
            db.query(TrackerProjectModel)
            .filter(
                TrackerProjectModel.id == project.id,
                TrackerProjectModel.next_issue_number == current,
            )
            .update({TrackerProjectModel.next_issue_number: current + 1}, synchronize_session=False)
        )
        if claimed:
            return f"{project.issue_prefix}-{current:04d}"
        db.refresh(project)
    raise TrackerError("conflict", f"could not allocate an issue key for {project.id}")


def create_issue(
    *,
    project_id: Optional[str] = None,
    title: str,
    body: str = "",
    status: str = "open",
    severity: str = "unset",
    component: Optional[str] = None,
    reporter: Optional[str] = None,
    assignee: Optional[str] = None,
    labels: Optional[Iterable[Any]] = None,
    collaborators: Optional[Iterable[Any]] = None,
    branches: Optional[Iterable[Any]] = None,
    worktrees: Optional[Iterable[Any]] = None,
    pull_requests: Optional[Iterable[Any]] = None,
    failing_command: Optional[str] = None,
    reproduction_steps: Optional[str] = None,
    expected_outcome: Optional[str] = None,
    actual_outcome: Optional[str] = None,
    evidence: Optional[str] = None,
    observed_revision: Optional[str] = None,
    session_name: Optional[str] = None,
    terminal_id: Optional[str] = None,
    source_path: Optional[str] = None,
    origin: str = "api",
    key: Optional[str] = None,
    created_at: Optional[datetime] = None,
    cwd: Optional[str] = None,
    alias: Optional[str] = None,
    kind: str = "bug",
    favorite: bool = False,
    force: bool = False,
    enforce_bug_details: bool = False,
) -> Dict[str, Any]:
    """File an issue.

    With no ``project_id`` the filing site is resolved from ``session_name`` /
    ``cwd``; an unresolvable site is refused rather than dropped into a default
    bucket, because an issue filed into the wrong log is harder to find than
    one that failed loudly at filing time.
    """
    heading = str(title or "").strip()
    if not heading:
        raise TrackerError("invalid", "issue title must not be empty")
    if len(heading) > MAX_TITLE:
        raise TrackerError("invalid", f"title too long (max {MAX_TITLE} chars)")
    text = str(body or "")
    if len(text) > MAX_BODY:
        raise TrackerError("invalid", f"body too long (max {MAX_BODY} chars)")

    status = _validate_choice(status, STATUSES, "status")
    severity = _validate_choice(severity, SEVERITIES, "severity")
    kind = _validate_kind(kind)
    # An opaque revision string recorded verbatim when the caller knows it.
    # Never inferred from source_path, the filing worktree, or a checkout.
    observed_revision = str(observed_revision or "").strip() or None
    if observed_revision and len(observed_revision) > MAX_BODY:
        raise TrackerError("invalid", f"observed_revision too long (max {MAX_BODY} chars)")
    if status == "duplicate":
        raise TrackerError(
            "invalid",
            "duplicate status requires duplicate_of canonical key (set via update after creation)",
        )
    bug_details = {
        "reproduction_steps": reproduction_steps,
        "expected_outcome": expected_outcome,
        "actual_outcome": actual_outcome,
    }
    for field, value in bug_details.items():
        if value is not None and len(str(value)) > MAX_BODY:
            raise TrackerError("invalid", f"{field} too long (max {MAX_BODY} chars)")
    label_list = normalise_labels(labels)
    collaborator_list = normalise_string_list(collaborators, field="collaborators")
    branch_list = normalise_string_list(branches, field="branches")
    worktree_list = normalise_string_list(worktrees, field="worktrees")
    pull_request_list = normalise_string_list(pull_requests, field="pull_requests")
    assignee = str(assignee or "").strip() or None
    if status == "in-progress" and assignee is None and not force:
        raise TrackerError(
            "invalid",
            "in-progress issues require an assignee; assign an owner or pass force=true "
            "to record an explicit exception",
        )

    resolution = resolve_project(
        project=project_id, session=session_name, alias=alias, cwd=cwd or source_path
    )
    if resolution.project_id is None:
        raise TrackerError(
            "unresolved",
            "cannot resolve a project for this issue: pass an explicit project, "
            "or register a path/session/git_remote scope for this filing site",
        )

    # Filing policy is evaluated only after the destination is known. A caller
    # filing from an unregistered site needs the actionable scope error first;
    # supplying diagnostics cannot make an unresolved write legal.
    if kind == "bug":
        missing = [name for name, value in bug_details.items() if not str(value or "").strip()]
        if missing and enforce_bug_details and not force:
            raise TrackerError(
                "invalid",
                "bug filings should include reproduction_steps, expected_outcome, and "
                f"actual_outcome; missing {', '.join(missing)} (pass force=true to record "
                "an explicit exception)",
            )
    elif failing_command or any(str(value or "").strip() for value in bug_details.values()):
        raise TrackerError(
            "invalid",
            "failing_command, reproduction_steps, expected_outcome, and actual_outcome "
            "are supported only for bugs",
        )

    if key is not None:
        key = str(key).strip().lower()
        if not _KEY_RE.match(key):
            raise TrackerError("invalid", f"invalid issue key {key!r}")

    with SessionLocal() as db:
        project = db.get(TrackerProjectModel, resolution.project_id)
        if project is None:
            raise TrackerError("not-found", f"no such project: {resolution.project_id}")
        if key is None:
            issue_key = _allocate_key(db, project)
        else:
            if db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first():
                raise TrackerError("conflict", f"issue {key} already exists")
            issue_key = key
            # An explicit key must not be handed out again later.
            number = int(key.rsplit("-", 1)[1])
            if number >= int(project.next_issue_number or 1):
                project.next_issue_number = number + 1

        stamp = created_at or _utcnow()
        row = TrackerIssueModel(
            key=issue_key,
            project_id=project.id,
            kind=kind,
            title=heading,
            body=text,
            status=status,
            severity=severity,
            component=(component or None),
            reporter=(reporter or None),
            assignee=assignee,
            labels=json.dumps(label_list),
            collaborators=json.dumps(collaborator_list),
            branches=json.dumps(branch_list),
            worktrees=json.dumps(worktree_list),
            pull_requests=json.dumps(pull_request_list),
            failing_command=(failing_command or None),
            reproduction_steps=(reproduction_steps or None),
            expected_outcome=(expected_outcome or None),
            actual_outcome=(actual_outcome or None),
            evidence=(evidence or None),
            observed_revision=observed_revision,
            session_name=(session_name or None),
            terminal_id=(terminal_id or None),
            source_path=(source_path or None),
            origin=str(origin or "api"),
            favorite=bool(favorite),
            created_at=stamp,
            updated_at=stamp,
            closed_at=stamp if status in TERMINAL_STATUSES else None,
        )
        db.add(row)
        db.add(
            TrackerEventModel(
                issue_key=issue_key,
                actor=(reporter or None),
                kind="created",
                field=None,
                old_value=None,
                new_value=heading,
                created_at=stamp,
            )
        )
        db.commit()
        db.refresh(row)
        payload = _issue_row(row)
        payload["resolved_by"] = resolution.matched_by
        return payload


def create_feature(
    *,
    project_id: Optional[str] = None,
    title: str,
    body: str = "",
    status: str = "open",
    severity: str = "unset",
    component: Optional[str] = None,
    reporter: Optional[str] = None,
    assignee: Optional[str] = None,
    labels: Optional[Iterable[Any]] = None,
    collaborators: Optional[Iterable[Any]] = None,
    branches: Optional[Iterable[Any]] = None,
    worktrees: Optional[Iterable[Any]] = None,
    pull_requests: Optional[Iterable[Any]] = None,
    failing_command: Optional[str] = None,
    evidence: Optional[str] = None,
    session_name: Optional[str] = None,
    terminal_id: Optional[str] = None,
    source_path: Optional[str] = None,
    origin: str = "api",
    key: Optional[str] = None,
    created_at: Optional[datetime] = None,
    cwd: Optional[str] = None,
    alias: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Create a feature request (thin wrapper over create_issue with kind=feature)."""
    if failing_command:
        raise TrackerError("invalid", "failing_command is not allowed for feature requests")
    return create_issue(
        project_id=project_id,
        title=title,
        body=body,
        status=status,
        severity=severity,
        component=component,
        reporter=reporter,
        assignee=assignee,
        labels=labels,
        collaborators=collaborators,
        branches=branches,
        worktrees=worktrees,
        pull_requests=pull_requests,
        failing_command=None,
        evidence=evidence,
        session_name=session_name,
        terminal_id=terminal_id,
        source_path=source_path,
        origin=origin,
        key=key,
        created_at=created_at,
        cwd=cwd,
        alias=alias,
        kind="feature",
        force=force,
    )


def get_issue(key: str) -> Dict[str, Any]:
    """Return one issue with its comments, events and links."""
    key = str(key or "").strip().lower()
    with SessionLocal() as db:
        row = db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first()
        if row is None:
            raise TrackerError("not-found", f"no such issue: {key}")
        payload = _issue_row(row)
        payload["comments"] = [
            {
                "id": c.id,
                "author": c.author,
                "body": c.body,
                "important": bool(getattr(c, "important", False)),
                "created_at": _iso(c.created_at),
            }
            for c in db.query(TrackerCommentModel)
            .filter(TrackerCommentModel.issue_key == key)
            .order_by(TrackerCommentModel.created_at.asc(), TrackerCommentModel.id.asc())
            .all()
        ]
        payload["events"] = [
            {
                "id": e.id,
                "actor": e.actor,
                "kind": e.kind,
                "field": e.field,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "created_at": _iso(e.created_at),
            }
            for e in db.query(TrackerEventModel)
            .filter(TrackerEventModel.issue_key == key)
            .order_by(TrackerEventModel.created_at.asc(), TrackerEventModel.id.asc())
            .all()
        ]
        payload["links"] = [
            {"id": l.id, "kind": l.kind, "from_key": l.from_key, "to_key": l.to_key}
            for l in db.query(TrackerLinkModel)
            .filter((TrackerLinkModel.from_key == key) | (TrackerLinkModel.to_key == key))
            .order_by(TrackerLinkModel.id.asc())
            .all()
        ]
        return payload


def snapshot_issues(*, project_id: str, keys: Sequence[str]) -> Dict[str, Any]:
    """Return a deterministic issue/reference closure from one read instant."""
    project_id = _validate_slug(project_id)
    normalised_keys = [str(key or "").strip().lower() for key in keys]
    if not normalised_keys:
        raise TrackerError("invalid", "snapshot selection must contain at least one issue key")
    invalid_keys = sorted({key for key in normalised_keys if not _KEY_RE.match(key)})
    if invalid_keys:
        raise TrackerError(
            "invalid",
            f"invalid issue keys in snapshot selection: {', '.join(invalid_keys)}",
            details={"invalid_keys": invalid_keys},
        )
    duplicate_keys = sorted({key for key in normalised_keys if normalised_keys.count(key) > 1})
    if duplicate_keys:
        raise TrackerError(
            "invalid",
            f"duplicate issue keys in snapshot selection: {', '.join(duplicate_keys)}",
            details={"duplicate_keys": duplicate_keys},
        )
    selected_keys = sorted(normalised_keys)
    digest_input = "".join(f"{key}\n" for key in selected_keys).encode("utf-8")

    with SessionLocal() as db:
        # sqlite3's legacy transaction control does not start a read
        # transaction for SELECT. SQLAlchemy's Session autobegin therefore is
        # not enough: issue and comment reads could otherwise observe two WAL
        # instants.
        db.execute(text("BEGIN"))
        issue_rows = (
            db.query(TrackerIssueModel)
            .filter(TrackerIssueModel.key.in_(selected_keys))
            .order_by(TrackerIssueModel.key.asc())
            .all()
        )
        rows_by_key = {row.key: row for row in issue_rows}
        missing_selected = sorted(set(selected_keys) - set(rows_by_key))
        if missing_selected:
            raise TrackerError(
                "not-found",
                f"selected issue keys do not exist: {', '.join(missing_selected)}",
                details={"missing_keys": missing_selected},
            )
        mismatches = [
            {"key": row.key, "project_id": row.project_id}
            for row in issue_rows
            if row.project_id != project_id
        ]
        if mismatches:
            raise TrackerError(
                "conflict",
                f"selected issue keys do not belong to project {project_id}",
                details={"project_mismatches": mismatches},
            )

        comments_by_key: Dict[str, List[Dict[str, Any]]] = {}
        links_by_id: Dict[int, Dict[str, Any]] = {}
        hydrated: set[str] = set()
        attempted: set[str] = set(selected_keys)
        unresolved_sources: Dict[str, set[str]] = {}
        issue_ref_pattern = re.compile(r"\b[a-z][a-z0-9-]{0,15}-\d{1,9}\b", re.IGNORECASE)

        while set(rows_by_key) - hydrated:
            wave = sorted(set(rows_by_key) - hydrated)
            comments = (
                db.query(TrackerCommentModel)
                .filter(TrackerCommentModel.issue_key.in_(wave))
                .order_by(
                    TrackerCommentModel.issue_key.asc(),
                    TrackerCommentModel.created_at.asc(),
                    TrackerCommentModel.id.asc(),
                )
                .all()
            )
            for key in wave:
                comments_by_key[key] = []
            for comment in comments:
                comments_by_key[comment.issue_key].append(
                    {
                        "id": comment.id,
                        "author": comment.author,
                        "body": comment.body,
                        "important": bool(getattr(comment, "important", False)),
                        "created_at": _iso(comment.created_at),
                    }
                )

            links = (
                db.query(TrackerLinkModel)
                .filter((TrackerLinkModel.from_key.in_(wave)) | (TrackerLinkModel.to_key.in_(wave)))
                .order_by(
                    TrackerLinkModel.from_key.asc(),
                    TrackerLinkModel.to_key.asc(),
                    TrackerLinkModel.kind.asc(),
                    TrackerLinkModel.id.asc(),
                )
                .all()
            )
            discovered: Dict[str, set[str]] = {}
            for link in links:
                links_by_id[link.id] = {
                    "id": link.id,
                    "kind": link.kind,
                    "from_key": link.from_key,
                    "to_key": link.to_key,
                }
                for endpoint in (link.from_key, link.to_key):
                    if endpoint not in rows_by_key:
                        discovered.setdefault(endpoint, set()).add("native-link")

            for key in wave:
                row = rows_by_key[key]
                texts = [row.title or "", row.body or ""]
                texts.extend(comment["body"] for comment in comments_by_key[key])
                for match in issue_ref_pattern.finditer("\n".join(texts)):
                    reference = match.group(0).lower()
                    if reference not in rows_by_key:
                        discovered.setdefault(reference, set()).add("text")

            for key, sources in discovered.items():
                if key in attempted and key not in rows_by_key:
                    unresolved_sources.setdefault(key, set()).update(sources)
            pending = sorted(set(discovered) - attempted)
            attempted.update(pending)
            if pending:
                found = (
                    db.query(TrackerIssueModel)
                    .filter(TrackerIssueModel.key.in_(pending))
                    .order_by(TrackerIssueModel.key.asc())
                    .all()
                )
                rows_by_key.update((row.key, row) for row in found)
                missing = set(pending) - {row.key for row in found}
                for key in missing:
                    unresolved_sources.setdefault(key, set()).update(discovered[key])
            hydrated.update(wave)

        part_of_parents: Dict[str, set[str]] = {}
        for link in links_by_id.values():
            if link["kind"] == "part-of":
                part_of_parents.setdefault(link["from_key"], set()).add(link["to_key"])
        root_keys_set: set[str] = set()
        frontier = list(selected_keys)
        while frontier:
            child = frontier.pop()
            for parent in part_of_parents.get(child, set()):
                if (
                    parent in rows_by_key
                    and parent not in root_keys_set
                    and parent not in selected_keys
                ):
                    root_keys_set.add(parent)
                    frontier.append(parent)

        root_keys = sorted(root_keys_set)
        reference_keys = sorted(set(rows_by_key) - set(selected_keys) - root_keys_set)
        roles = []
        for key in sorted(rows_by_key):
            role = (
                "selected"
                if key in selected_keys
                else "root" if key in root_keys_set else "reference"
            )
            roles.append({"key": key, "role": role})

        issues = []
        for key in sorted(rows_by_key):
            issue = _issue_row(rows_by_key[key])
            issue["comments"] = comments_by_key[key]
            issue["links"] = sorted(
                (
                    link
                    for link in links_by_id.values()
                    if link["from_key"] == key or link["to_key"] == key
                ),
                key=lambda link: (
                    link["from_key"],
                    link["to_key"],
                    link["kind"],
                    link["id"],
                ),
            )
            issues.append(issue)

        project_ids = sorted({row.project_id for row in rows_by_key.values()})
        project_rows = (
            db.query(TrackerProjectModel)
            .filter(TrackerProjectModel.id.in_(project_ids))
            .order_by(TrackerProjectModel.id.asc())
            .all()
        )
        projects = [_project_row(row) for row in project_rows]
        missing_projects = sorted(set(project_ids) - {row.id for row in project_rows})
        unresolved_references: List[Dict[str, Any]] = [
            {
                "kind": "issue",
                "key": key,
                "reason": "not-found",
                "sources": sorted(sources),
            }
            for key, sources in sorted(unresolved_sources.items())
        ]
        unresolved_references.extend(
            {
                "kind": "project",
                "id": missing,
                "reason": "not-found",
                "sources": ["issue.project_id"],
            }
            for missing in missing_projects
        )

        return {
            "schema": "cao-tracker-issue-snapshot-v1",
            "project_id": project_id,
            "selected_keys": selected_keys,
            "selected_keys_digest": {
                "algorithm": "sha256-sorted-newline-v1",
                "value": hashlib.sha256(digest_input).hexdigest(),
            },
            "roles": roles,
            "root_keys": root_keys,
            "reference_keys": reference_keys,
            "issues": issues,
            "projects": projects,
            "unresolved_references": unresolved_references,
            "consistency": {"kind": "sqlite-read-transaction"},
        }


def list_issues(
    *,
    project_id: Optional[str] = None,
    status: Optional[Sequence[str]] = None,
    severity: Optional[Sequence[str]] = None,
    component: Optional[str] = None,
    assignee: Optional[str] = None,
    reporter: Optional[str] = None,
    label: Optional[Sequence[str] | str] = None,
    without_label: Optional[Sequence[str] | str] = None,
    unlabeled: bool = False,
    observed_revision: Optional[Sequence[str] | str] = None,
    query: Optional[str] = None,
    open_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    order: str = "created_desc",
    kind: Optional[str] = "bug",
) -> Dict[str, Any]:
    """List issues with filters, returning rows plus the unpaged total.

    ``total`` is the count BEFORE limit/offset so a caller can page without
    guessing whether a short page means the end.

    ``without_label`` excludes issues carrying any named label. Repeated
    exclusions compose as AND (the row must carry none of them), which lets a
    triage pass ignore workflow-state labels without losing provenance labels.

    ``unlabeled`` selects issues with an empty label set. It composes with
    every other filter.

    ``observed_revision`` is the exact-match revision filter (one or more
    values, OR within the family), shared with ranked search through the
    structured-filter builder so the two surfaces cannot drift.

    Filter families are expressed once in ``services.tracker_filters``; this
    path keeps its own ordering, substring query, and pagination semantics.
    """
    limit = max(1, min(int(limit or 100), 500))
    offset = max(0, int(offset or 0))
    # Preserve the historical earliest-error order: kind, then project id,
    # then the remaining families through the shared builder.
    if kind not in (None, "all"):
        _validate_kind(kind)
    if project_id:
        validated_project_id = _validate_slug(project_id)
    else:
        validated_project_id = None
    families = StructuredFilters(
        kinds=() if kind in (None, "all") else (kind,),
        statuses=tuple(status or ()),
        severities=tuple(severity or ()),
        components=(component,) if component else (),
        observed_revisions=_repeatable_values(observed_revision),
        labels=_repeatable_values(label),
        without_labels=_repeatable_values(without_label),
        assignee=assignee,
        reporter=reporter,
        open_only=open_only,
        unlabeled=unlabeled,
    ).validated()

    with SessionLocal() as db:
        q = db.query(TrackerIssueModel)
        if validated_project_id:
            q = q.filter(TrackerIssueModel.project_id == validated_project_id)
        for condition in families.orm_conditions():
            q = q.filter(condition)
        if query:
            needle = f"%{query.strip()}%"
            q = q.filter(
                TrackerIssueModel.title.ilike(needle)
                | TrackerIssueModel.body.ilike(needle)
                | TrackerIssueModel.key.ilike(needle)
                | TrackerIssueModel.failing_command.ilike(needle)
                | TrackerIssueModel.reproduction_steps.ilike(needle)
                | TrackerIssueModel.expected_outcome.ilike(needle)
                | TrackerIssueModel.actual_outcome.ilike(needle)
                | TrackerIssueModel.evidence.ilike(needle)
            )

        total = q.count()
        q = _apply_order(q, order)
        rows = q.limit(limit).offset(offset).all()
        return {
            "total": int(total),
            "limit": limit,
            "offset": offset,
            "issues": [_issue_row(r) for r in rows],
        }


def _repeatable_values(value: Optional[Sequence[str] | str]) -> Tuple[str, ...]:
    """Normalize a scalar-or-sequence parameter into a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def list_features(
    *,
    project_id: Optional[str] = None,
    status: Optional[Sequence[str]] = None,
    severity: Optional[Sequence[str]] = None,
    component: Optional[str] = None,
    assignee: Optional[str] = None,
    reporter: Optional[str] = None,
    label: Optional[str] = None,
    without_label: Optional[Sequence[str] | str] = None,
    unlabeled: bool = False,
    observed_revision: Optional[Sequence[str] | str] = None,
    query: Optional[str] = None,
    open_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    order: str = "created_desc",
) -> Dict[str, Any]:
    """List feature requests (thin wrapper with kind=feature)."""
    return list_issues(
        project_id=project_id,
        status=status,
        severity=severity,
        component=component,
        assignee=assignee,
        reporter=reporter,
        label=label,
        without_label=without_label,
        unlabeled=unlabeled,
        observed_revision=observed_revision,
        query=query,
        open_only=open_only,
        limit=limit,
        offset=offset,
        order=order,
        kind="feature",
    )


def _apply_order(q: Any, order: str) -> Any:
    if order == "created_asc":
        return q.order_by(TrackerIssueModel.created_at.asc(), TrackerIssueModel.id.asc())
    if order == "updated_desc":
        return q.order_by(TrackerIssueModel.updated_at.desc(), TrackerIssueModel.id.desc())
    if order == "severity":
        # SQLite sorts the literal strings, and "P0" < "P1" < ... < "unset"
        # lexicographically already, which is the intended rank.
        return q.order_by(TrackerIssueModel.severity.asc(), TrackerIssueModel.created_at.asc())
    if order == "key":
        return q.order_by(TrackerIssueModel.key.asc())
    return q.order_by(TrackerIssueModel.created_at.desc(), TrackerIssueModel.id.desc())


class _UpdateRaceLost(Exception):
    """The guarded write matched no row: a concurrent edit landed between this
    transaction's read and its write."""


class _ImportanceRaceLost(Exception):
    """A concurrent importance setter flipped the flag between this
    transaction's read of it and its guarded write."""


class _CommentRaceLost(Exception):
    """A comment's parent changed after its expected-clock precheck."""


class _LinkRaceLost(Exception):
    """One endpoint changed after a link operation read its clock."""


def update_issue(
    issue_key: str,
    *,
    actor: Optional[str] = None,
    expected_updated_at: Optional[str] = None,
    force: bool = False,
    drop_previous_assignee: bool = False,
    **changes: Any,
) -> Dict[str, Any]:
    """Apply field changes, recording one audit event per field that moved.

    Only fields the caller actually passed are touched, and a field whose new
    value equals the old one writes no event — an audit trail full of no-ops is
    an audit trail nobody reads.

    ``expected_updated_at`` is an optimistic-concurrency precondition for
    callers that read then write (a wayfinder session rewriting a map body):
    when given, the update applies only if the issue's ``updated_at`` still
    equals it, and a mismatch refuses THIS write with a conflict carrying the
    current observable version. Callers that do not pass it get the historical
    unconditional edit.

    The precondition is enforced at the write seam, not by a Python comparison
    before it: every persisting update is applied by ONE conditional UPDATE
    guarded on the ``updated_at`` this transaction read. SQLite serialises
    writers, so a concurrent commit turns the guard into a no-match instead of
    a lost update — with ``expected_updated_at`` the loser gets the typed
    conflict; without it the update is retried against the fresh row (a label
    delta re-merges, so two cooperative add-label edits both land). A failed or
    superseded attempt writes nothing and records no events: the audit trail
    only ever records the transition that was actually established.

    The positional parameter is ``issue_key`` rather than ``key`` so that
    ``update_issue(k, key=...)`` reports the intended "not editable" refusal
    instead of a TypeError about duplicate arguments.
    """
    key = str(issue_key or "").strip().lower()
    unknown = set(changes) - set(_EDITABLE_FIELDS)
    if unknown:
        raise TrackerError("invalid", f"not editable: {', '.join(sorted(unknown))}")

    attempts = 8
    for attempt in range(attempts):
        try:
            with SessionLocal() as db:
                return _apply_issue_update(
                    db,
                    key,
                    actor=actor,
                    expected_updated_at=expected_updated_at,
                    force=force,
                    drop_previous_assignee=drop_previous_assignee,
                    changes=dict(changes),
                )
        except _UpdateRaceLost:
            if expected_updated_at is not None:
                # The precondition held at read time and the write still lost:
                # a concurrent edit landed in between. Report the version that
                # is observable now, exactly as the up-front check would.
                current = _current_updated_at(key)
                raise TrackerError(
                    "conflict",
                    f"{key} has changed since {expected_updated_at} "
                    f"(current updated_at {current}); re-read and retry",
                    details={"current_updated_at": current},
                )
            time.sleep(0.005 * (attempt + 1))
        except OperationalError as exc:
            # SQLite reports a lost lock race (or a snapshot a concurrent
            # commit made stale) as "database is locked". Retrying re-reads
            # the committed state, so the guarded write — not luck — decides
            # whether this update applies.
            msg = str(exc).lower()
            if "locked" not in msg and "busy" not in msg:
                raise
            time.sleep(0.005 * (attempt + 1))
    raise TrackerError(
        "conflict",
        f"{key} is being updated concurrently and this edit could not be "
        f"applied after {attempts} attempts; retry it",
    )


def _current_updated_at(key: str) -> Optional[str]:
    with SessionLocal() as db:
        row = db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first()
        return _iso(row.updated_at) if row is not None else None


def _apply_issue_update(
    db: Any,
    key: str,
    *,
    actor: Optional[str],
    expected_updated_at: Optional[str],
    force: bool,
    drop_previous_assignee: bool,
    changes: Dict[str, Any],
) -> Dict[str, Any]:
    """One attempt of ``update_issue``: read, validate, guarded write, events.

    All column writes go through ``assignments`` and land in a single UPDATE
    whose WHERE pins the ``updated_at`` read at the top of this transaction.
    """
    row = db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first()
    if row is None:
        raise TrackerError("not-found", f"no such issue: {key}")

    if expected_updated_at is not None:
        expected = _parse_timestamp(expected_updated_at, field="expected_updated_at")
        if _stored_moment(row.updated_at) != expected:
            raise TrackerError(
                "conflict",
                f"{key} has changed since {expected_updated_at} "
                f"(current updated_at {_iso(row.updated_at)}); re-read and retry",
                details={"current_updated_at": _iso(row.updated_at)},
            )

    target_status = row.status
    if "status" in changes and changes["status"] is not None:
        target_status = _validate_choice(changes["status"], STATUSES, "status")
    target_assignee = row.assignee
    if "assignee" in changes and changes["assignee"] is not None:
        target_assignee = str(changes["assignee"]).strip() or None
    if (
        not force
        and target_status == "in-progress"
        and target_assignee is None
        and ("status" in changes or "assignee" in changes)
    ):
        raise TrackerError(
            "invalid",
            "in-progress issues require an assignee; assign an owner in the same edit "
            "or pass force=true to record an explicit exception",
        )

    now = _utcnow()
    events: List[TrackerEventModel] = []
    assignments: Dict[str, Any] = {}

    def _record(field: str, old: Any, new: Any) -> None:
        assignments[field] = new
        events.append(
            TrackerEventModel(
                issue_key=key,
                actor=actor,
                kind="field",
                field=field,
                old_value=_as_text(old),
                new_value=_as_text(new),
                created_at=now,
            )
        )

    # A reassignment is a handoff by default. Preserve the former owner in the
    # collaborator paper trail whether or not the caller also replaces the
    # collaborator list. The explicit override suppresses only this addition;
    # it never clears collaborators already present.
    if (
        "assignee" in changes
        and changes["assignee"] is not None
        and row.assignee
        and target_assignee != row.assignee
        and not drop_previous_assignee
    ):
        collaborators = (
            normalise_string_list(changes["collaborators"], field="collaborators")
            if changes.get("collaborators") is not None
            else _parse_string_list(row.collaborators)
        )
        if row.assignee not in collaborators:
            collaborators.append(row.assignee)
        changes["collaborators"] = collaborators

    # Duplicate status requires canonical key (P1)
    if (
        changes.get("status") == "duplicate"
        and not changes.get("duplicate_of")
        and not getattr(row, "duplicate_of", None)
    ):
        raise TrackerError("invalid", "duplicate status requires duplicate_of canonical key")
    # Determine target kind after change (for kind-switch validation)
    # N3: explicit null/empty is treated as no-op (skip), not 400
    if "kind" in changes and (changes["kind"] is None or not str(changes["kind"]).strip()):
        # Remove null/empty kind from changes so loop skips it
        del changes["kind"]
    target_kind = (
        _validate_kind(changes["kind"]) if "kind" in changes else getattr(row, "kind", "bug")
    )
    bug_only_fields = (
        "failing_command",
        "reproduction_steps",
        "expected_outcome",
        "actual_outcome",
    )
    if target_kind != "bug":
        supplied = [
            field
            for field in bug_only_fields
            if field in changes and str(changes.get(field) or "").strip()
        ]
        if supplied:
            raise TrackerError(
                "invalid",
                f"{', '.join(supplied)} are supported only for bugs",
            )
        # A type conversion has a complete exit: diagnostic fields that no
        # longer apply are cleared in the same transition.
        if "kind" in changes:
            for field in bug_only_fields:
                old = getattr(row, field, None)
                if old:
                    _record(field, old, None)
                changes.pop(field, None)
    else:
        required_bug_fields = (
            "reproduction_steps",
            "expected_outcome",
            "actual_outcome",
        )
        validates_bug_details = ("kind" in changes and getattr(row, "kind", "bug") != "bug") or any(
            field in changes and not str(changes.get(field) or "").strip()
            for field in required_bug_fields
        )
        if validates_bug_details and not force:
            missing = []
            for field in required_bug_fields:
                value = changes.get(field, getattr(row, field, None))
                if not str(value or "").strip():
                    missing.append(field)
            if missing:
                raise TrackerError(
                    "invalid",
                    "bugs should include reproduction_steps, expected_outcome, and "
                    f"actual_outcome; missing {', '.join(missing)} (pass force=true "
                    "to record an explicit exception)",
                )

    # Atomic label deltas. The merged set is computed from the row read inside
    # this transaction — and if that read proves stale at write time, the
    # guarded UPDATE below loses and the whole attempt is retried against the
    # fresh labels, so a delta never silently drops a concurrent actor's label.
    add_labels = normalise_labels(changes.pop("add_labels", None) or [])
    remove_labels = set(normalise_labels(changes.pop("remove_labels", None) or []))
    clear_labels = bool(changes.pop("clear_labels", None))
    if "labels" in changes and (add_labels or remove_labels or clear_labels):
        raise TrackerError(
            "invalid",
            "labels replaces the whole label set; it cannot be combined with "
            "add_labels/remove_labels/clear_labels in one update",
        )
    if clear_labels and (add_labels or remove_labels):
        raise TrackerError(
            "invalid",
            "clear_labels replaces the whole label set with nothing; combine it "
            "with nothing (pass labels to replace with a specific set)",
        )
    if add_labels or remove_labels or clear_labels:
        current_labels = [] if clear_labels else _parse_labels(row.labels)
        merged = [l for l in current_labels if l not in remove_labels]
        merged += [l for l in add_labels if l not in merged]
        merged = normalise_labels(merged)  # the bounds apply to the result
        if merged != _parse_labels(row.labels):
            _record("labels", row.labels, json.dumps(merged))

    for field, raw in changes.items():
        if raw is None:
            continue
        old, new = _coerce_field(row, field, raw, db=db)
        if old == new:
            continue
        _record(field, old, new)
        if field == "status":
            assignments["closed_at"] = now if new in TERMINAL_STATUSES else None

    if not events:
        return _issue_row(row)

    assignments["updated_at"] = now
    # The guarded write: apply only to the row version this transaction read.
    # A concurrent commit between our read and this write makes the rowcount 0
    # (or fails the write with a lock error the caller retries) — never a
    # silent lost update reported as success.
    written = (
        db.query(TrackerIssueModel)
        .filter(
            TrackerIssueModel.key == key,
            TrackerIssueModel.updated_at == row.updated_at,
        )
        .update(assignments, synchronize_session=False)
    )
    if written != 1:
        db.rollback()
        raise _UpdateRaceLost()
    for event in events:
        db.add(event)
    db.flush()
    effect_ids = [int(event.id) for event in events]
    db.commit()
    db.refresh(row)
    result = _issue_row(row)
    # A status-only write has one durable audit effect.  Keep the plural form
    # for multi-field edits while making the ordinary status/close seam easy
    # for an audit publisher to acknowledge exactly.
    result["effect_ids"] = effect_ids
    if len(effect_ids) == 1:
        result["effect_id"] = effect_ids[0]
    return result


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value)


def _coerce_field(row: TrackerIssueModel, field: str, raw: Any, *, db: Any) -> Tuple[Any, Any]:
    """Validate one field change and return ``(old, new)`` in stored form."""
    if field == "kind":
        return getattr(row, "kind", "bug"), _validate_kind(raw)
    if field == "status":
        return row.status, _validate_choice(raw, STATUSES, "status")
    if field == "severity":
        return row.severity, _validate_choice(raw, SEVERITIES, "severity")
    if field == "favorite":
        if not isinstance(raw, bool):
            raise TrackerError("invalid", "favorite must be a boolean")
        return bool(getattr(row, "favorite", False)), raw
    if field == "labels":
        return row.labels, json.dumps(normalise_labels(raw))
    if field in {"collaborators", "branches", "worktrees", "pull_requests"}:
        return getattr(row, field), json.dumps(normalise_string_list(raw, field=field))
    if field == "title":
        text = str(raw).strip()
        if not text:
            raise TrackerError("invalid", "issue title must not be empty")
        if len(text) > MAX_TITLE:
            raise TrackerError("invalid", f"title too long (max {MAX_TITLE} chars)")
        return row.title, text
    if field == "body":
        text = str(raw)
        if len(text) > MAX_BODY:
            raise TrackerError("invalid", f"body too long (max {MAX_BODY} chars)")
        return row.body, text
    if field in {"reproduction_steps", "expected_outcome", "actual_outcome"}:
        text = str(raw)
        if len(text) > MAX_BODY:
            raise TrackerError("invalid", f"{field} too long (max {MAX_BODY} chars)")
        return getattr(row, field), (text if text.strip() else None)
    if field == "observed_revision":
        # An empty string is an explicit clear, like the other free-text fields.
        text = str(raw).strip()
        if len(text) > MAX_BODY:
            raise TrackerError("invalid", f"observed_revision too long (max {MAX_BODY} chars)")
        return row.observed_revision, (text or None)
    if field == "duplicate_of":
        text = str(raw).strip().lower()
        if not text:
            return row.duplicate_of, None
        if text == row.key:
            raise TrackerError("invalid", "an issue cannot duplicate itself")
        if db.query(TrackerIssueModel).filter(TrackerIssueModel.key == text).first() is None:
            raise TrackerError("not-found", f"no such issue: {text}")
        return row.duplicate_of, text
    # Remaining free-text fields: an empty string is an explicit clear.
    text = str(raw)
    return getattr(row, field), (text if text.strip() else None)


def delete_issue(key: str) -> Dict[str, Any]:
    """Delete an issue and everything attached to it."""
    key = str(key or "").strip().lower()
    with SessionLocal() as db:
        row = db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first()
        if row is None:
            raise TrackerError("not-found", f"no such issue: {key}")
        db.query(TrackerCommentModel).filter(TrackerCommentModel.issue_key == key).delete(
            synchronize_session=False
        )
        db.query(TrackerEventModel).filter(TrackerEventModel.issue_key == key).delete(
            synchronize_session=False
        )
        db.query(TrackerLinkModel).filter(
            (TrackerLinkModel.from_key == key) | (TrackerLinkModel.to_key == key)
        ).delete(synchronize_session=False)
        db.delete(row)
        db.commit()
        return {"key": key, "deleted": True}


def add_comment(
    key: str,
    *,
    body: str,
    author: Optional[str] = None,
    important: bool = False,
    expected_updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a comment and record it in the audit trail.

    ``important`` defaults to false (ordinary/routine weight); a caller that
    already knows a comment is high-signal may file it flagged. The flag is
    reversible afterwards through :func:`set_comment_importance`.
    """
    key = str(key or "").strip().lower()
    text = str(body or "").strip()
    if not text:
        raise TrackerError("invalid", "comment body must not be empty")
    if len(text) > MAX_BODY:
        raise TrackerError("invalid", f"comment too long (max {MAX_BODY} chars)")
    for attempt in range(8):
        try:
            with SessionLocal() as db:
                row = db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first()
                if row is None:
                    raise TrackerError("not-found", f"no such issue: {key}")
                if expected_updated_at is not None:
                    expected = _parse_timestamp(expected_updated_at, field="expected_updated_at")
                    if _stored_moment(row.updated_at) != expected:
                        raise TrackerError(
                            "conflict",
                            f"{key} has changed since {expected_updated_at} "
                            f"(current updated_at {_iso(row.updated_at)}); re-read and retry",
                            details={"current_updated_at": _iso(row.updated_at)},
                        )
                now = _utcnow()
                # The parent clock is the compare-and-swap fence.  Do this before
                # flushing the comment/audit rows so a late conflicting mutation
                # leaves no visible partial effect.
                written = (
                    db.query(TrackerIssueModel)
                    .filter(
                        TrackerIssueModel.key == key,
                        TrackerIssueModel.updated_at == row.updated_at,
                    )
                    .update({"updated_at": now}, synchronize_session=False)
                )
                if written != 1:
                    db.rollback()
                    raise _CommentRaceLost()
                comment = TrackerCommentModel(
                    issue_key=key,
                    author=author,
                    body=text,
                    important=bool(important),
                    created_at=now,
                )
                event = TrackerEventModel(
                    issue_key=key,
                    actor=author,
                    kind="comment",
                    new_value=text[:200],
                    created_at=now,
                )
                db.add_all([comment, event])
                db.flush()
                result = {
                    "id": int(comment.id),
                    "issue_key": key,
                    "author": comment.author,
                    "body": comment.body,
                    "important": bool(comment.important),
                    "created_at": _iso(comment.created_at),
                    "updated_at": _iso(now),
                    "effect_id": int(event.id),
                }
                db.commit()
                return result
        except _CommentRaceLost:
            if expected_updated_at is not None:
                current = _current_updated_at(key)
                raise TrackerError(
                    "conflict",
                    f"{key} changed while creating the comment (current updated_at {current}); re-read and retry",
                    details={"current_updated_at": current},
                )
            time.sleep(0.005 * (attempt + 1))
        except OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            if expected_updated_at is not None:
                current = _current_updated_at(key)
                raise TrackerError(
                    "conflict",
                    f"{key} changed while creating the comment (current updated_at {current}); re-read and retry",
                    details={"current_updated_at": current},
                )
            time.sleep(0.005 * (attempt + 1))
    raise TrackerError("conflict", f"{key} could not be updated after 8 attempts; retry it")


def set_comment_importance(
    key: str,
    comment_id: int,
    *,
    important: bool,
    actor: Optional[str] = None,
) -> Dict[str, Any]:
    """Set or clear a comment's ``important`` flag idempotently.

    The row change is applied by ONE conditional UPDATE guarded on the value
    this transaction read (``WHERE important <> :new``), and only the setter
    whose UPDATE actually changed the row appends the ``comment-field`` audit
    event and bumps the parent issue's ``updated_at`` — all in one
    transaction. A same-value retry writes nothing at all, and when two
    concurrent setters race to flip the same direction SQLite serialises them:
    the loser's guard matches zero rows, so exactly one transition is ever
    recorded.
    """
    key = str(key or "").strip().lower()
    important = bool(important)
    attempts = 8
    for attempt in range(attempts):
        try:
            with SessionLocal() as db:
                comment = db.get(TrackerCommentModel, int(comment_id))
                if comment is None or comment.issue_key != key:
                    raise TrackerError("not-found", f"no comment {comment_id} on {key}")
                current = bool(comment.important)
                if current == important:
                    # Same-value retry: no row change, no event, no bump.
                    return _comment_payload(comment)
                now = _utcnow()
                written = (
                    db.query(TrackerCommentModel)
                    .filter(
                        TrackerCommentModel.id == int(comment_id),
                        # The design's conditional guard (``WHERE important <>
                        # :new_value``): applies only when the stored value still
                        # differs, so a superseded setter's UPDATE matches zero
                        # rows instead of double-recording an established
                        # transition.
                        TrackerCommentModel.important != important,
                    )
                    .update({"important": important}, synchronize_session=False)
                )
                if written != 1:
                    raise _ImportanceRaceLost()
                parent = db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first()
                db.add(
                    TrackerEventModel(
                        issue_key=key,
                        actor=actor,
                        kind="comment-field",
                        field="important",
                        old_value=_as_text(current),
                        new_value=_as_text(important),
                        created_at=now,
                    )
                )
                if parent is not None:
                    parent.updated_at = now
                db.commit()
                return {
                    "id": int(comment_id),
                    "issue_key": key,
                    "important": important,
                    "changed": True,
                    "updated_at": _iso(parent.updated_at) if parent is not None else None,
                }
        except _ImportanceRaceLost:
            # Another setter flipped the flag between our read and write. The
            # next attempt re-reads: if it converged to the requested value
            # this converges to the same-value response, otherwise it retries.
            time.sleep(0.005 * (attempt + 1))
        except OperationalError as exc:
            msg = str(exc).lower()
            if "locked" not in msg and "busy" not in msg:
                raise
            time.sleep(0.005 * (attempt + 1))
    raise TrackerError(
        "conflict",
        f"comment {comment_id} importance could not be set after {attempts} " "attempts; retry it",
    )


def _comment_payload(comment: TrackerCommentModel) -> Dict[str, Any]:
    """Serialize one comment row for API responses."""
    return {
        "id": comment.id,
        "issue_key": comment.issue_key,
        "author": comment.author,
        "body": comment.body,
        "important": bool(comment.important),
        "changed": False,
        "created_at": _iso(comment.created_at),
    }


def delete_comment(key: str, comment_id: int, *, actor: Optional[str] = None) -> Dict[str, Any]:
    """Delete a comment, recording the deletion before the row is removed.

    The deletion event (carrying the removed body excerpt) and the parent
    timestamp bump are written in the SAME transaction as the row removal, so
    the audit trail can never disagree with the table about what happened.
    """
    key = str(key or "").strip().lower()
    with SessionLocal() as db:
        comment = db.get(TrackerCommentModel, int(comment_id))
        if comment is None or comment.issue_key != key:
            raise TrackerError("not-found", f"no comment {comment_id} on {key}")
        parent = db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first()
        now = _utcnow()
        db.add(
            TrackerEventModel(
                issue_key=key,
                actor=actor,
                kind="comment-deleted",
                field=None,
                old_value=str(comment.body)[:200],
                new_value=str(int(comment_id)),
                created_at=now,
            )
        )
        if parent is not None:
            parent.updated_at = now
        db.delete(comment)
        db.commit()
        return {
            "id": int(comment_id),
            "issue_key": key,
            "deleted": True,
            "updated_at": _iso(parent.updated_at) if parent is not None else None,
        }


def add_link(
    from_key: str,
    *,
    to_key: str,
    kind: str,
    actor: Optional[str] = None,
    expected_from_updated_at: Optional[str] = None,
    expected_to_updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Relate two issues, optionally fencing both endpoint clocks.

    A new relationship changes both issues' visible graph, so it advances both
    clocks and records an audit event on each endpoint in the same transaction.
    Callers without clocks retain the historical idempotent behavior; callers
    that did review clocks get a typed conflict instead of publishing against a
    stale endpoint.
    """
    from_key = str(from_key or "").strip().lower()
    to_key = str(to_key or "").strip().lower()
    kind = _validate_choice(kind, LINK_KINDS, "link kind")
    if from_key == to_key:
        raise TrackerError("invalid", "an issue cannot link to itself")
    attempted_fence = expected_from_updated_at is not None or expected_to_updated_at is not None
    for attempt in range(8):
        try:
            with SessionLocal() as db:
                from_row = _require_issue(db, from_key)
                to_row = _require_issue(db, to_key)
                endpoint_specs = (
                    ("from", from_key, from_row, expected_from_updated_at),
                    ("to", to_key, to_row, expected_to_updated_at),
                )
                for label, endpoint_key, endpoint_row, expected_text in endpoint_specs:
                    if expected_text is None:
                        continue
                    expected = _parse_timestamp(expected_text, field=f"expected_{label}_updated_at")
                    if _stored_moment(endpoint_row.updated_at) != expected:
                        raise TrackerError(
                            "conflict",
                            f"{endpoint_key} has changed since {expected_text} "
                            f"(current updated_at {_iso(endpoint_row.updated_at)}); re-read and retry",
                            details={
                                "endpoint": label,
                                "current_updated_at": _iso(endpoint_row.updated_at),
                            },
                        )
                existing = (
                    db.query(TrackerLinkModel)
                    .filter(
                        TrackerLinkModel.from_key == from_key,
                        TrackerLinkModel.to_key == to_key,
                        TrackerLinkModel.kind == kind,
                    )
                    .first()
                )
                if existing is not None:
                    return {
                        "id": int(existing.id),
                        "from_key": from_key,
                        "to_key": to_key,
                        "kind": kind,
                        "created": False,
                        "from_updated_at": _iso(from_row.updated_at),
                        "to_updated_at": _iso(to_row.updated_at),
                    }
                now = _utcnow()
                # Pin each conditional update to the exact version observed in
                # this transaction.  A failure rolls back the other endpoint,
                # link, and both audit rows as one all-or-nothing effect.
                for endpoint_key, endpoint_row in ((from_key, from_row), (to_key, to_row)):
                    written = (
                        db.query(TrackerIssueModel)
                        .filter(
                            TrackerIssueModel.key == endpoint_key,
                            TrackerIssueModel.updated_at == endpoint_row.updated_at,
                        )
                        .update({"updated_at": now}, synchronize_session=False)
                    )
                    if written != 1:
                        db.rollback()
                        raise _LinkRaceLost()
                link = TrackerLinkModel(from_key=from_key, to_key=to_key, kind=kind)
                events = [
                    TrackerEventModel(
                        issue_key=from_key, actor=actor, kind="link", field=kind, new_value=to_key
                    ),
                    TrackerEventModel(
                        issue_key=to_key, actor=actor, kind="link", field=kind, new_value=from_key
                    ),
                ]
                db.add(link)
                db.add_all(events)
                db.flush()
                result = {
                    "id": int(link.id),
                    "from_key": from_key,
                    "to_key": to_key,
                    "kind": kind,
                    "created": True,
                    "from_updated_at": _iso(now),
                    "to_updated_at": _iso(now),
                    "effect_ids": [int(event.id) for event in events],
                }
                db.commit()
                return result
        except _LinkRaceLost:
            if attempted_fence:
                from_current = _current_updated_at(from_key)
                to_current = _current_updated_at(to_key)
                raise TrackerError(
                    "conflict",
                    "a link endpoint changed while creating the relationship; re-read and retry",
                    details={
                        "from_current_updated_at": from_current,
                        "to_current_updated_at": to_current,
                    },
                )
            time.sleep(0.005 * (attempt + 1))
        except OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            if attempted_fence:
                from_current = _current_updated_at(from_key)
                to_current = _current_updated_at(to_key)
                raise TrackerError(
                    "conflict",
                    "a link endpoint changed while creating the relationship; re-read and retry",
                    details={
                        "from_current_updated_at": from_current,
                        "to_current_updated_at": to_current,
                    },
                )
            time.sleep(0.005 * (attempt + 1))
    raise TrackerError("conflict", "link could not be created after 8 attempts; retry it")


def remove_link(link_id: int, *, actor: Optional[str] = None) -> Dict[str, Any]:
    with SessionLocal() as db:
        row = db.get(TrackerLinkModel, int(link_id))
        if row is None:
            raise TrackerError("not-found", f"no such link: {link_id}")
        db.add(
            TrackerEventModel(
                issue_key=row.from_key,
                actor=actor,
                kind="unlink",
                field=row.kind,
                old_value=row.to_key,
            )
        )
        db.delete(row)
        db.commit()
        return {"id": int(link_id), "deleted": True}


# --------------------------------------------------------------------------
# map membership and the frontier (cond-0394)
# --------------------------------------------------------------------------


def _children_query(db: Any, parent: str) -> Any:
    return (
        db.query(TrackerIssueModel)
        .join(TrackerLinkModel, TrackerLinkModel.from_key == TrackerIssueModel.key)
        .filter(TrackerLinkModel.to_key == parent, TrackerLinkModel.kind == "part-of")
    )


def _require_issue(db: Any, key: str) -> TrackerIssueModel:
    row = db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first()
    if row is None:
        raise TrackerError("not-found", f"no such issue: {key}")
    return row


def list_children(parent_key: str) -> Dict[str, Any]:
    """List the direct children of a parent/map issue.

    Membership is exactly the ``part-of`` links (child -> parent); there is no
    transitive closure, no rollup, no derived tree. Ordered by creation
    (``created_at``, ties by row id) — the order the map was charted in.
    """
    parent = str(parent_key or "").strip().lower()
    with SessionLocal() as db:
        _require_issue(db, parent)
        rows = (
            _children_query(db, parent)
            .order_by(TrackerIssueModel.created_at.asc(), TrackerIssueModel.id.asc())
            .all()
        )
        return {"parent": parent, "children": [_issue_row(r) for r in rows]}


def _incoming_blockers(db: Any, keys: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """``blocks`` edges pointing at each key, as ``key -> [(blocker, status)]``.

    Shared by ``frontier`` and ``map_projection`` so the takeable rule is
    written once: a child is benched by an incoming ``blocks`` edge whose
    blocker is itself nonterminal (``resolved`` still blocks — landed, not
    verified).
    """
    if not keys:
        return {}
    rows = (
        db.query(TrackerLinkModel.from_key, TrackerLinkModel.to_key, TrackerIssueModel.status)
        .join(TrackerIssueModel, TrackerIssueModel.key == TrackerLinkModel.from_key)
        .filter(TrackerLinkModel.kind == "blocks", TrackerLinkModel.to_key.in_(keys))
        .all()
    )
    out: Dict[str, List[Tuple[str, str]]] = {}
    for blocker, blocked, blocker_status in rows:
        out.setdefault(blocked, []).append((blocker, blocker_status))
    return out


def _frontier_keys(
    children: List[TrackerIssueModel], blockers: Dict[str, List[Tuple[str, str]]]
) -> List[str]:
    """The canonical frontier rule: nonterminal, unassigned, no nonterminal
    incoming blocker. Input order (creation order) is preserved."""
    return [
        row.key
        for row in children
        if row.status not in TERMINAL_STATUSES
        and row.assignee is None
        and not any(status not in TERMINAL_STATUSES for _key, status in blockers.get(row.key, []))
    ]


def frontier(parent_key: str) -> Dict[str, Any]:
    """The takeable edge of a map: its direct children that are nonterminal,
    unassigned, and have no nonterminal incoming ``blocks`` edge.

    Everything is computed from the canonical issue and link records at query
    time — there is no stored "blocked" flag to drift. Ordering is
    deterministic: creation order (``created_at``, ties by row id), so the
    first row is the oldest takeable ticket and concurrent wayfinder sessions
    picking "the first frontier ticket" see the same answer.
    """
    parent = str(parent_key or "").strip().lower()
    with SessionLocal() as db:
        _require_issue(db, parent)
        children = (
            _children_query(db, parent)
            .order_by(TrackerIssueModel.created_at.asc(), TrackerIssueModel.id.asc())
            .all()
        )
        by_key = {row.key: row for row in children}
        keys = _frontier_keys(children, _incoming_blockers(db, list(by_key)))
        return {"parent": parent, "frontier": [_issue_row(by_key[k]) for k in keys]}


def map_projection(map_key: str) -> Dict[str, Any]:
    """The one-request projection behind the dashboard's map view.

    Returns the map itself, every direct child with its classification
    (``blocked_by``: the nonterminal blockers benching it; ``frontier``: the
    same canonical rule ``frontier()`` uses), every link touching the map or a
    child, every link endpoint that is not a member (``external``), and
    progress counts. One request, one derivation — the UI renders it rather
    than reconstructing classifications from N detail fetches.

    ``external`` covers EVERY endpoint of a returned link that is neither the
    map nor a child — a ``relates``/``duplicates``/``caused-by`` neighbour is
    materialized exactly like a blocker, so no returned link ever points at an
    issue the caller cannot see. Each external row carries ``blocking``: the
    member children it actually benches (its nonterminal incoming ``blocks``
    edges to them). A non-empty ``blocking`` is what makes an external issue an
    external blocker; anything else is there for context, not because it holds
    a ticket back.

    This is deliberately NOT a hierarchy engine: children are direct
    ``part-of`` members only, and progress counts direct children — no
    transitive closure, no rollup.
    """
    key = str(map_key or "").strip().lower()
    with SessionLocal() as db:
        map_row = _require_issue(db, key)
        children = (
            _children_query(db, key)
            .order_by(TrackerIssueModel.created_at.asc(), TrackerIssueModel.id.asc())
            .all()
        )
        child_keys = [c.key for c in children]
        members = {key, *child_keys}
        blockers = _incoming_blockers(db, child_keys)
        frontier_set = set(_frontier_keys(children, blockers))

        links = (
            db.query(TrackerLinkModel)
            .filter(
                (TrackerLinkModel.from_key.in_(members)) | (TrackerLinkModel.to_key.in_(members))
            )
            .order_by(TrackerLinkModel.id.asc())
            .all()
        )

        children_payload = []
        for child in children:
            row = _issue_row(child)
            row["blocked_by"] = [
                bk for bk, s in blockers.get(child.key, []) if s not in TERMINAL_STATUSES
            ]
            row["frontier"] = child.key in frontier_set
            children_payload.append(row)

        # External endpoints: every issue a returned link names that is neither
        # the map nor a child — included so every link renders without a second
        # request, and a benched child explains itself. ``blocking`` inverts
        # the children's blocked_by lists: the member children this issue
        # actually benches. A terminal blocker or a relates/duplicates
        # neighbour benches nobody, so its list is empty.
        external_keys = sorted(
            {ep for link in links for ep in (link.from_key, link.to_key)} - members
        )
        benches: Dict[str, List[str]] = {}
        for child_row in children_payload:
            for blocker_key in child_row["blocked_by"]:
                benches.setdefault(blocker_key, []).append(child_row["key"])
        external = []
        if external_keys:
            for r in (
                db.query(TrackerIssueModel)
                .filter(TrackerIssueModel.key.in_(external_keys))
                .order_by(TrackerIssueModel.key.asc())
                .all()
            ):
                row = _issue_row(r)
                row["blocking"] = benches.get(row["key"], [])
                external.append(row)

        terminal = sum(1 for c in children if c.status in TERMINAL_STATUSES)
        return {
            "map": _issue_row(map_row),
            "children": children_payload,
            "frontier": [k for k in child_keys if k in frontier_set],
            "links": [
                {"id": l.id, "kind": l.kind, "from_key": l.from_key, "to_key": l.to_key}
                for l in links
            ],
            "external": external,
            "progress": {
                "total": len(children),
                "open": len(children) - terminal,
                "terminal": terminal,
                "resolved": sum(1 for c in children if c.status == "resolved"),
                "claimed": sum(
                    1 for c in children if c.status not in TERMINAL_STATUSES and c.assignee
                ),
                "frontier": len(frontier_set),
            },
        }


def graph_projection(
    root_key: str,
    *,
    max_depth: int = 8,
    max_nodes: int = 300,
) -> Dict[str, Any]:
    """A bounded transitive issue hierarchy plus its relationship context.

    ``part-of`` is the only hierarchy edge and runs child -> parent. Starting
    from any issue, breadth-first traversal discovers descendants without
    prescribing which issue kinds may contain which others. All links touching
    the resulting hierarchy are returned, with non-member endpoints
    materialized in ``external`` so the relationship view has no dangling
    edges. Bounds are explicit response metadata, never a silent partial list.
    """
    key = str(root_key or "").strip().lower()
    depth_limit = max(1, min(int(max_depth or 8), 12))
    node_limit = max(1, min(int(max_nodes or 300), 500))
    external_limit = 200

    with SessionLocal() as db:
        root = _require_issue(db, key)
        rows_by_key: Dict[str, TrackerIssueModel] = {key: root}
        depths: Dict[str, int] = {key: 0}
        frontier_keys = [key]
        truncated_reasons: set[str] = set()

        for depth in range(1, depth_limit + 1):
            if not frontier_keys or len(rows_by_key) >= node_limit:
                break
            pairs = (
                db.query(TrackerLinkModel, TrackerIssueModel)
                .join(TrackerIssueModel, TrackerIssueModel.key == TrackerLinkModel.from_key)
                .filter(
                    TrackerLinkModel.kind == "part-of",
                    TrackerLinkModel.to_key.in_(frontier_keys),
                )
                .order_by(TrackerIssueModel.created_at.asc(), TrackerIssueModel.id.asc())
                .all()
            )
            next_keys: List[str] = []
            for _link, child in pairs:
                if child.key in rows_by_key:
                    continue
                if len(rows_by_key) >= node_limit:
                    truncated_reasons.add("node-limit")
                    break
                rows_by_key[child.key] = child
                depths[child.key] = depth
                next_keys.append(child.key)
            frontier_keys = next_keys

        if (
            len(rows_by_key) >= node_limit
            and frontier_keys
            and db.query(TrackerLinkModel.id)
            .filter(
                TrackerLinkModel.kind == "part-of",
                TrackerLinkModel.to_key.in_(frontier_keys),
                ~TrackerLinkModel.from_key.in_(list(rows_by_key)),
            )
            .first()
            is not None
        ):
            truncated_reasons.add("node-limit")

        # If the last included level still has children, depth bounded the
        # projection. This extra existence query records that fact explicitly.
        deepest = [node_key for node_key, depth in depths.items() if depth == depth_limit]
        if (
            deepest
            and db.query(TrackerLinkModel.id)
            .filter(
                TrackerLinkModel.kind == "part-of",
                TrackerLinkModel.to_key.in_(deepest),
            )
            .first()
            is not None
        ):
            truncated_reasons.add("depth-limit")

        members = set(rows_by_key)
        touching = (
            db.query(TrackerLinkModel)
            .filter(
                (TrackerLinkModel.from_key.in_(members)) | (TrackerLinkModel.to_key.in_(members))
            )
            .order_by(TrackerLinkModel.id.asc())
            .all()
        )

        # A child omitted by a hierarchy bound must not sneak back in as an
        # "external" relationship node. Other neighbours are useful context.
        omitted_hierarchy_children = {
            link.from_key
            for link in touching
            if link.kind == "part-of" and link.to_key in members and link.from_key not in members
        }
        live_omitted_children: set[str] = set()
        if omitted_hierarchy_children:
            live_omitted_children = {
                row.key
                for row in db.query(TrackerIssueModel)
                .filter(TrackerIssueModel.key.in_(omitted_hierarchy_children))
                .all()
                if row.status not in TERMINAL_STATUSES
            }
        live_children_beyond_bound = sorted(
            {
                link.to_key
                for link in touching
                if link.kind == "part-of" and link.from_key in live_omitted_children
            }
        )
        candidate_external: List[str] = []
        for link in touching:
            for endpoint in (link.from_key, link.to_key):
                if endpoint in members or endpoint in candidate_external:
                    continue
                if endpoint in omitted_hierarchy_children:
                    continue
                candidate_external.append(endpoint)
        if len(candidate_external) > external_limit:
            candidate_external = candidate_external[:external_limit]
            truncated_reasons.add("external-limit")
        external_set = set(candidate_external)

        external_rows = []
        if external_set:
            by_external_key = {
                row.key: row
                for row in db.query(TrackerIssueModel)
                .filter(TrackerIssueModel.key.in_(external_set))
                .all()
            }
            external_rows = [
                _issue_row(by_external_key[external_key])
                for external_key in candidate_external
                if external_key in by_external_key
            ]
            external_set = {row["key"] for row in external_rows}

        visible = members | external_set
        links = [link for link in touching if link.from_key in visible and link.to_key in visible]
        parent_keys: Dict[str, List[str]] = {node_key: [] for node_key in members}
        child_counts: Dict[str, int] = {node_key: 0 for node_key in members}
        for link in links:
            if link.kind != "part-of":
                continue
            if link.from_key in members and link.to_key in members:
                parent_keys[link.from_key].append(link.to_key)
                child_counts[link.to_key] += 1

        ordered_keys = sorted(
            members,
            key=lambda node_key: (
                depths[node_key],
                rows_by_key[node_key].created_at,
                rows_by_key[node_key].id,
            ),
        )
        nodes = []
        for node_key in ordered_keys:
            payload = _issue_row(rows_by_key[node_key])
            payload.update(
                {
                    "depth": depths[node_key],
                    "parent_keys": parent_keys[node_key],
                    "child_count": child_counts[node_key],
                }
            )
            nodes.append(payload)

        return {
            "root": _issue_row(root),
            "nodes": nodes,
            "external": external_rows,
            "links": [
                {"id": link.id, "kind": link.kind, "from_key": link.from_key, "to_key": link.to_key}
                for link in links
            ],
            "bounds": {
                "max_depth": depth_limit,
                "max_nodes": node_limit,
                "truncated": bool(truncated_reasons),
                "reasons": sorted(truncated_reasons),
                "live_children_beyond_bound": live_children_beyond_bound,
            },
            "stats": {
                "nodes": len(nodes),
                "descendants": max(0, len(nodes) - 1),
                "external": len(external_rows),
                "links": len(links),
                "depth": max(depths.values()),
            },
        }


def _strongly_connected_components(
    nodes: Iterable[str], edges: Iterable[Tuple[str, str]]
) -> List[List[str]]:
    """Return deterministic directed cycles as strongly connected components."""

    node_set = set(nodes)
    adjacency: Dict[str, List[str]] = {key: [] for key in node_set}
    self_loops: set[str] = set()
    for source, target in edges:
        if source not in node_set or target not in node_set:
            continue
        adjacency[source].append(target)
        if source == target:
            self_loops.add(source)
    for values in adjacency.values():
        values.sort()

    index = 0
    indexes: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: set[str] = set()
    components: List[List[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbour in adjacency[node]:
            if neighbour not in indexes:
                visit(neighbour)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbour])
            elif neighbour in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[neighbour])

        if lowlinks[node] != indexes[node]:
            return
        component: List[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        component.sort()
        if len(component) > 1 or component[0] in self_loops:
            components.append(component)

    for node in sorted(node_set):
        if node not in indexes:
            visit(node)
    return sorted(components)


def hierarchy_audit(
    root_key: str,
    *,
    max_depth: int = 8,
    max_nodes: int = 300,
) -> Dict[str, Any]:
    """Audit a bounded hierarchy and derive its actionable leaf frontier.

    Parent/child links remain policy-flexible: multiple parents and cycles are
    reported rather than refused.  The projection is one read of canonical
    issue/link state, so counts, blockers, findings and frontier cannot drift
    between separate client-side walks.
    """

    projection = graph_projection(root_key, max_depth=max_depth, max_nodes=max_nodes)
    nodes = projection["nodes"]
    members = {row["key"] for row in nodes}
    by_key = {row["key"]: row for row in [*nodes, *projection["external"]]}
    links = projection["links"]
    internal_hierarchy_links = [
        link
        for link in links
        if link["kind"] == "part-of" and link["from_key"] in members and link["to_key"] in members
    ]
    member_parent_links = [
        link for link in links if link["kind"] == "part-of" and link["from_key"] in members
    ]
    visible_child_links = [
        link for link in links if link["kind"] == "part-of" and link["to_key"] in members
    ]
    block_links = [link for link in links if link["kind"] == "blocks"]

    parents: Dict[str, List[str]] = {key: [] for key in members}
    for link in member_parent_links:
        parents[link["from_key"]].append(link["to_key"])
    for values in parents.values():
        values.sort()

    unresolved_blockers = []
    blocked_keys: set[str] = set()
    for link in block_links:
        blocked = link["to_key"]
        blocker = by_key.get(link["from_key"])
        if blocked not in members or blocker is None:
            continue
        if blocker["status"] in TERMINAL_STATUSES:
            continue
        blocked_keys.add(blocked)
        unresolved_blockers.append({"blocker": link["from_key"], "blocked": blocked})
    unresolved_blockers.sort(key=lambda row: (row["blocked"], row["blocker"]))

    parents_with_live_children = {
        link["to_key"]
        for link in visible_child_links
        if by_key[link["from_key"]]["status"] not in TERMINAL_STATUSES
    }
    parents_with_live_children.update(projection["bounds"].get("live_children_beyond_bound", []))
    root = projection["root"]
    frontier = []
    for row in nodes:
        if row["key"] == root["key"] and len(nodes) > 1:
            continue
        if row["status"] in TERMINAL_STATUSES or row["status"] in {"blocked", "in-progress"}:
            continue
        if row.get("assignee") is not None:
            continue
        if row["key"] in parents_with_live_children or row["key"] in blocked_keys:
            continue
        frontier.append(row)

    counts_by_kind: Dict[str, int] = {}
    counts_by_status: Dict[str, int] = {}
    counts_by_link: Dict[str, int] = {}
    for row in nodes:
        counts_by_kind[row["kind"]] = counts_by_kind.get(row["kind"], 0) + 1
        counts_by_status[row["status"]] = counts_by_status.get(row["status"], 0) + 1
    for link in links:
        counts_by_link[link["kind"]] = counts_by_link.get(link["kind"], 0) + 1

    hierarchy_edges = [(link["from_key"], link["to_key"]) for link in internal_hierarchy_links]
    blocker_edges = [
        (link["from_key"], link["to_key"])
        for link in block_links
        if link["from_key"] in members and link["to_key"] in members
    ]
    findings = {
        "hierarchy_cycles": _strongly_connected_components(members, hierarchy_edges),
        "blocker_cycles": _strongly_connected_components(members, blocker_edges),
        "multiple_parents": [
            {"key": key, "parents": parent_keys}
            for key, parent_keys in sorted(parents.items())
            if len(parent_keys) > 1
        ],
        "root_parents": parents.get(root["key"], []),
        "active_without_assignee": sorted(
            row["key"]
            for row in nodes
            if row["status"] == "in-progress" and row.get("assignee") is None
        ),
    }
    return {
        "root": root,
        "counts": {
            "nodes": len(nodes),
            "descendants": max(0, len(nodes) - 1),
            "part_of": len(internal_hierarchy_links),
            "blocks": len(block_links),
            "relationships": sum(link["kind"] != "part-of" for link in links),
            "by_kind": dict(sorted(counts_by_kind.items())),
            "by_status": dict(sorted(counts_by_status.items())),
            "by_link_kind": dict(sorted(counts_by_link.items())),
        },
        "findings": findings,
        "unresolved_blockers": unresolved_blockers,
        "frontier": frontier,
        "bounds": projection["bounds"],
    }


def field_options(
    project_id: str,
    *,
    field: str,
    query: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return a bounded, searchable vocabulary for an open-ended issue field.

    Components and actor names are deliberately not mechanism-level enums:
    callers may create a new value. This endpoint lets the dashboard reuse
    existing spellings without downloading an unbounded corpus. Labels live in
    a JSON array; the other fields are scalar columns, but the response shape
    is identical for every picker.
    """
    allowed = {
        "label",
        "component",
        "assignee",
        "reporter",
        "collaborator",
        "branch",
        "worktree",
        "pull_request",
    }
    if field not in allowed:
        raise TrackerError(
            "invalid",
            f"invalid option field {field!r}: expected one of {', '.join(sorted(allowed))}",
        )
    slug = _validate_slug(project_id)
    limit = max(1, min(int(limit or 20), 50))
    needle = str(query or "").strip().casefold()
    with SessionLocal() as db:
        if db.get(TrackerProjectModel, slug) is None:
            raise TrackerError("not-found", f"no such project: {slug}")
        rows = db.query(TrackerIssueModel).filter(TrackerIssueModel.project_id == slug).all()

    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        if field == "label":
            values = _parse_labels(row.labels)
        elif field in {"collaborator", "branch", "worktree", "pull_request"}:
            column = {
                "collaborator": "collaborators",
                "branch": "branches",
                "worktree": "worktrees",
                "pull_request": "pull_requests",
            }[field]
            values = _parse_string_list(getattr(row, column))
        else:
            values = [getattr(row, field)]
        for raw in values:
            value = str(raw or "").strip()
            if not value or (needle and needle not in value.casefold()):
                continue
            entry = counts.setdefault(value, {"total": 0, "open": 0})
            entry["total"] += 1
            if row.status not in TERMINAL_STATUSES:
                entry["open"] += 1

    ordered = sorted(
        counts.items(), key=lambda item: (-item[1]["open"], -item[1]["total"], item[0])
    )
    return {
        "project_id": slug,
        "field": field,
        "query": str(query or ""),
        "matching_total": len(ordered),
        "options": [
            {"value": value, **counts_for_value} for value, counts_for_value in ordered[:limit]
        ],
    }


def label_facets(project_id: str) -> Dict[str, Any]:
    """Label discovery for one project: every label with total and open counts.

    Covers all item kinds — labels like ``effort:``/``wayfinder:map`` span bugs
    and features, and a per-kind split would hide exactly the labels an operator
    is hunting for. Ordered most-alive-first (open desc, total desc, label asc)
    and deterministic. ``unlabeled``/``unlabeled_open`` count the never-triaged
    bucket the ``unlabeled`` list filter selects.
    """
    slug = _validate_slug(project_id)
    with SessionLocal() as db:
        if db.get(TrackerProjectModel, slug) is None:
            raise TrackerError("not-found", f"no such project: {slug}")
        rows = db.query(TrackerIssueModel).filter(TrackerIssueModel.project_id == slug).all()
    counts: Dict[str, Dict[str, int]] = {}
    unlabeled = 0
    unlabeled_open = 0
    for row in rows:
        is_open = row.status not in TERMINAL_STATUSES
        labels = _parse_labels(row.labels)
        if not labels:
            unlabeled += 1
            if is_open:
                unlabeled_open += 1
        for label in labels:
            entry = counts.setdefault(label, {"total": 0, "open": 0})
            entry["total"] += 1
            if is_open:
                entry["open"] += 1
    return {
        "project_id": slug,
        "labels": [
            {"label": label, **entry}
            for label, entry in sorted(
                counts.items(), key=lambda kv: (-kv[1]["open"], -kv[1]["total"], kv[0])
            )
        ],
        "unlabeled": unlabeled,
        "unlabeled_open": unlabeled_open,
    }


# --------------------------------------------------------------------------
# claim lifecycle (cond-0394)
# --------------------------------------------------------------------------


def claim_issue(issue_key: str, *, claimant: str) -> Dict[str, Any]:
    """Claim an issue and begin active work in one guarded transition.

    ``open`` and ``triage`` work is promoted to ``in-progress`` as it is
    assigned. More specific nonterminal states such as ``blocked`` are
    preserved. A retry by the current claimant is idempotent, and also repairs
    an old open claim by promoting it. A different claimant gets a conflict
    naming the owner that was actually observed.
    """
    key = str(issue_key or "").strip().lower()
    who = str(claimant or "").strip()
    if not who:
        raise TrackerError("invalid", "claimant must not be empty")
    attempts = 8
    for attempt in range(attempts):
        try:
            with SessionLocal() as db:
                row = db.query(TrackerIssueModel).filter(TrackerIssueModel.key == key).first()
                if row is None:
                    raise TrackerError("not-found", f"no such issue: {key}")
                observed_status = row.status
                if row.assignee == who:
                    if observed_status in {"open", "triage"}:
                        now = _utcnow()
                        promoted = (
                            db.query(TrackerIssueModel)
                            .filter(
                                TrackerIssueModel.key == key,
                                TrackerIssueModel.assignee == who,
                                TrackerIssueModel.status == observed_status,
                            )
                            .update(
                                {
                                    TrackerIssueModel.status: "in-progress",
                                    TrackerIssueModel.updated_at: now,
                                },
                                synchronize_session=False,
                            )
                        )
                        if promoted != 1:
                            db.rollback()
                            raise _UpdateRaceLost()
                        db.add(
                            TrackerEventModel(
                                issue_key=key,
                                actor=who,
                                kind="field",
                                field="status",
                                old_value=observed_status,
                                new_value="in-progress",
                                created_at=now,
                            )
                        )
                        db.commit()
                        row = _require_issue(db, key)
                    payload = _issue_row(row)
                    payload["claimed"] = True
                    payload["already_claimed"] = True
                    return payload
                if row.assignee:
                    raise TrackerError(
                        "conflict",
                        f"{key} is already claimed by {row.assignee}",
                        details={"observed_assignee": row.assignee},
                    )
                if observed_status in TERMINAL_STATUSES:
                    raise TrackerError(
                        "conflict",
                        f"{key} is {observed_status}; terminal issues cannot be claimed",
                        details={"observed_status": observed_status},
                    )

                now = _utcnow()
                next_status = (
                    "in-progress" if observed_status in {"open", "triage"} else observed_status
                )
                claimed = (
                    db.query(TrackerIssueModel)
                    .filter(
                        TrackerIssueModel.key == key,
                        TrackerIssueModel.assignee.is_(None),
                        TrackerIssueModel.status == observed_status,
                    )
                    .update(
                        {
                            TrackerIssueModel.assignee: who,
                            TrackerIssueModel.status: next_status,
                            TrackerIssueModel.updated_at: now,
                        },
                        synchronize_session=False,
                    )
                )
                if claimed != 1:
                    db.rollback()
                    raise _UpdateRaceLost()
                db.add(
                    TrackerEventModel(
                        issue_key=key,
                        actor=who,
                        kind="claim",
                        field="assignee",
                        old_value=None,
                        new_value=who,
                        created_at=now,
                    )
                )
                if next_status != observed_status:
                    db.add(
                        TrackerEventModel(
                            issue_key=key,
                            actor=who,
                            kind="field",
                            field="status",
                            old_value=observed_status,
                            new_value=next_status,
                            created_at=now,
                        )
                    )
                db.commit()
                row = _require_issue(db, key)
                payload = _issue_row(row)
                payload["claimed"] = True
                payload["already_claimed"] = False
                return payload
        except _UpdateRaceLost:
            time.sleep(0.005 * (attempt + 1))
        except OperationalError as exc:
            msg = str(exc).lower()
            if "locked" not in msg and "busy" not in msg:
                raise
            time.sleep(0.005 * (attempt + 1))
    raise TrackerError(
        "conflict",
        f"{key} is being updated concurrently and the claim could not be "
        f"applied after {attempts} attempts; retry it",
    )


def unclaim_issue(issue_key: str, *, actor: Optional[str] = None) -> Dict[str, Any]:
    """Release a claim — the ordinary recovery exit from a stale assignment.

    Any actor may release: a supervisor cleaning up after a dead worker must
    not be blocked by the claim it is clearing. Idempotent — unclaiming an
    unclaimed issue succeeds without writing an event. Releasing active work
    also returns ``in-progress`` to ``open`` in the same guarded write, so the
    tracker does not manufacture an active but ownerless issue.

    The release is one conditional UPDATE guarded on the assignee this call
    actually observed, pinned for the whole request. Two concurrent releasers
    therefore cannot both report the transition: the winner writes the single
    ``unclaim`` event, and the loser resolves idempotently from a fresh read
    (``was_claimed=False``, no event) rather than throwing a lock error or
    duplicating the event. The same guard means a call can never clear a
    *successor* claim that appeared after the one it observed — it releases
    the claim it saw, or none. To release whoever holds the issue now,
    re-read and call again.
    """
    key = str(issue_key or "").strip().lower()
    observed: Optional[str] = None
    first_read = True
    attempts = 8
    for attempt in range(attempts):
        try:
            with SessionLocal() as db:
                row = _require_issue(db, key)
                if first_read:
                    # The one claim this call is allowed to release. Retries
                    # keep pinning it, so a successor claim that appears
                    # mid-request is never cleared by a stale attempt.
                    observed = row.assignee
                    first_read = False
                if observed is None or row.assignee != observed:
                    # Either there was no claim, or the observed claim is
                    # already gone — this call established nothing.
                    payload = _issue_row(row)
                    payload["unclaimed"] = True
                    payload["was_claimed"] = False
                    return payload
                now = _utcnow()
                observed_status = row.status
                next_status = "open" if observed_status == "in-progress" else observed_status
                released = (
                    db.query(TrackerIssueModel)
                    .filter(
                        TrackerIssueModel.key == key,
                        TrackerIssueModel.assignee == observed,
                        TrackerIssueModel.status == observed_status,
                    )
                    .update(
                        {
                            TrackerIssueModel.assignee: None,
                            TrackerIssueModel.status: next_status,
                            TrackerIssueModel.updated_at: now,
                        },
                        synchronize_session=False,
                    )
                )
                if released != 1:
                    db.rollback()
                    raise _UpdateRaceLost()
                db.add(
                    TrackerEventModel(
                        issue_key=key,
                        actor=actor,
                        kind="unclaim",
                        field="assignee",
                        old_value=observed,
                        new_value=None,
                        created_at=now,
                    )
                )
                if next_status != observed_status:
                    db.add(
                        TrackerEventModel(
                            issue_key=key,
                            actor=actor,
                            kind="field",
                            field="status",
                            old_value=observed_status,
                            new_value=next_status,
                            created_at=now,
                        )
                    )
                db.commit()
                db.refresh(row)
                payload = _issue_row(row)
                payload["unclaimed"] = True
                payload["was_claimed"] = True
                return payload
        except _UpdateRaceLost:
            # The observed claim changed under us (released, or changed
            # hands). The next attempt re-reads and resolves idempotently.
            time.sleep(0.005 * (attempt + 1))
        except OperationalError as exc:
            msg = str(exc).lower()
            if "locked" not in msg and "busy" not in msg:
                raise
            time.sleep(0.005 * (attempt + 1))
    raise TrackerError(
        "conflict",
        f"{key} is being updated concurrently and the unclaim could not be "
        f"applied after {attempts} attempts; retry it",
    )


def stats(project_id: Optional[str] = None, *, kind: Optional[str] = "bug") -> Dict[str, Any]:
    """Aggregate counts for a project (or the whole install).

    Default ``kind="bug"`` returns bug-only counts.
    ``kind=None`` or ``kind="all"`` aggregates across all kinds and also returns ``by_kind``.
    """
    with SessionLocal() as db:
        q = db.query(TrackerIssueModel)
        if kind is not None:
            if kind == "all":
                pass
            else:
                _validate_kind(kind)
                q = q.filter(TrackerIssueModel.kind == kind)
        else:
            # kind=None means all kinds (explicit generic surface)
            pass
        if project_id:
            q = q.filter(TrackerIssueModel.project_id == _validate_slug(project_id))
        rows = q.all()
    by_status: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    by_component: Dict[str, int] = {}
    for row in rows:
        by_status[row.status] = by_status.get(row.status, 0) + 1
        by_severity[row.severity] = by_severity.get(row.severity, 0) + 1
        key = row.component or "(none)"
        by_component[key] = by_component.get(key, 0) + 1
    result: Dict[str, Any] = {
        "project_id": project_id,
        "total": len(rows),
        "open": sum(1 for r in rows if r.status not in TERMINAL_STATUSES),
        "by_status": by_status,
        "by_severity": by_severity,
        "by_component": by_component,
    }
    if kind is None or kind == "all":
        with SessionLocal() as db:
            base = db.query(TrackerIssueModel)
            if project_id:
                base = base.filter(TrackerIssueModel.project_id == _validate_slug(project_id))
            all_rows = base.all()
        by_kind: Dict[str, Dict[str, Any]] = {}
        for k in ITEM_KINDS:
            subset = [r for r in all_rows if getattr(r, "kind", "bug") == k]
            by_status_k: Dict[str, int] = {}
            by_sev_k: Dict[str, int] = {}
            for r in subset:
                by_status_k[r.status] = by_status_k.get(r.status, 0) + 1
                by_sev_k[r.severity] = by_sev_k.get(r.severity, 0) + 1
            by_kind[k] = {
                "total": len(subset),
                "open": sum(1 for r in subset if r.status not in TERMINAL_STATUSES),
                "by_status": by_status_k,
                "by_severity": by_sev_k,
            }
        result["by_kind"] = by_kind
        result["all_total"] = len(all_rows)
        result["all_open"] = sum(1 for r in all_rows if r.status not in TERMINAL_STATUSES)
    return result


def render_markdown(project_id: str, *, open_only: bool = True, kind: Optional[str] = "bug") -> str:
    """Render an issue log as markdown.

    The markdown ledger this replaces is now an *export*: a view produced from
    the database on demand, not a second source of truth that has to be kept in
    step with it.
    """
    project = get_project(project_id)

    # Paged rather than capped. `list_issues` bounds a page at 500, and this
    # export replaces a file that held every entry — a truncated log that
    # announces nothing is the exact failure the markdown ledger never had.
    issues: List[Dict[str, Any]] = []
    total = 0
    while True:
        page = list_issues(
            project_id=project_id,
            open_only=open_only,
            limit=500,
            offset=len(issues),
            order="created_asc",
            kind=kind,
        )
        total = page["total"]
        issues.extend(page["issues"])
        if not page["issues"] or len(issues) >= total:
            break

    lines = [
        f"# {'Open' if open_only else 'All'} issues — {project['name']}",
        "",
        f"Rendered from the CAO issue tracker. Project id `{project['id']}`; " f"{total} issue(s).",
        "",
        "---",
        "",
    ]
    for issue in issues:
        severity = f"[{issue['severity']}] " if issue["severity"] != "unset" else ""
        kind_prefix = f"[{issue.get('kind', 'bug')}] " if kind is None or kind == "all" else ""
        lines.append(f"## {issue['key']} — {kind_prefix}{severity}{issue['title']}")
        lines.append("")
        if kind is None or kind == "all":
            lines.append(f"- **kind:** {issue.get('kind', 'bug')}")
        lines.append(f"- **filed:** {issue['created_at']}")
        lines.append(f"- **reporter:** {issue['reporter'] or 'unknown'}")
        lines.append(f"- **status:** {issue['status']}")
        if issue["component"]:
            lines.append(f"- **component:** {issue['component']}")
        if issue["assignee"]:
            lines.append(f"- **assignee:** {issue['assignee']}")
        if issue["collaborators"]:
            lines.append(f"- **collaborators:** {', '.join(issue['collaborators'])}")
        if issue["branches"]:
            lines.append(f"- **branches:** {', '.join(issue['branches'])}")
        if issue["worktrees"]:
            lines.append(f"- **worktrees:** {', '.join(issue['worktrees'])}")
        if issue["pull_requests"]:
            lines.append(f"- **pull requests:** {', '.join(issue['pull_requests'])}")
        if issue["labels"]:
            lines.append(f"- **labels:** {', '.join(issue['labels'])}")
        if issue["failing_command"]:
            lines.append(f"- **failing command:** `{issue['failing_command']}`")
        if issue["reproduction_steps"]:
            lines.append("- **reproduction steps:**")
            lines.append("")
            lines.append(issue["reproduction_steps"].rstrip())
        if issue["expected_outcome"]:
            lines.append("- **expected outcome:**")
            lines.append("")
            lines.append(issue["expected_outcome"].rstrip())
        if issue["actual_outcome"]:
            lines.append("- **actual outcome:**")
            lines.append("")
            lines.append(issue["actual_outcome"].rstrip())
        if issue.get("favorite"):
            lines.append("- **project Home:** favorite")
        if issue["evidence"]:
            lines.append(f"- **evidence:** {issue['evidence']}")
        if issue["resolution"]:
            lines.append(f"- **resolution:** {issue['resolution']}")
        lines.append("")
        if issue["body"]:
            lines.append(issue["body"].rstrip())
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
