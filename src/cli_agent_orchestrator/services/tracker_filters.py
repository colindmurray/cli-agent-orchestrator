"""Shared structured-filter and subtree-scope builder for tracker retrieval.

Design §10.2: one builder keeps ``issue list`` and ranked-search filter
semantics aligned, so the two surfaces cannot drift apart while list ordering
stays untouched. Every §10.2 family is expressed exactly once here as an ORM
predicate over :class:`TrackerIssueModel`; the ranked service reuses the same
predicates for its candidate selection, and vocabulary validation goes through
the same ``issue_tracker`` helpers the list path has always used.

The module imports ``issue_tracker`` lazily inside functions: it is imported
by that module at load time, and the tracker vocabulary (statuses, kinds,
severities, label rules) must have exactly one source.

The subtree closure (design §10.1) is deliberately a cycle-safe recursive CTE
over ``child --part-of--> parent`` links using ``UNION``, never ``UNION ALL``
— a membership cycle among issues must terminate, not loop. It is computed
directly against the link table rather than reused from the graph projection,
which is bounded to depth 12 / 500 nodes and may silently truncate; search
scope is either complete or refuses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from sqlalchemy import text as _sql_text

from cli_agent_orchestrator.clients.database import TrackerIssueModel

# LIKE escaping for exact-label matching inside the JSON-array text column,
# shared verbatim with the historical inline implementation so list behavior
# is byte-identical after the delegation refactor.
_LABEL_LIKE_ESCAPE = "\\"


def _issue_tracker():
    """Late binding of the tracker vocabulary/validation helpers."""
    from cli_agent_orchestrator.services.issue_tracker import (
        TERMINAL_STATUSES,
        _validate_choice,
        _validate_kind,
        _validate_slug,
        normalise_labels,
    )

    return {
        "TERMINAL_STATUSES": TERMINAL_STATUSES,
        "_validate_choice": _validate_choice,
        "_validate_kind": _validate_kind,
        "_validate_slug": _validate_slug,
        "normalise_labels": normalise_labels,
    }


def _label_like_needle(label: str) -> str:
    """Escape a label for exact matching inside the JSON-array text column."""
    return label.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass(frozen=True)
class StructuredFilters:
    """The §10.2 filter families, before validation.

    Repeated values inside one family compose as OR; different families
    compose as AND. Included labels compose as AND across values (an issue
    must carry every listed label), excluded labels as none-of.
    """

    kinds: Tuple[str, ...] = ()
    statuses: Tuple[str, ...] = ()
    severities: Tuple[str, ...] = ()
    components: Tuple[str, ...] = ()
    observed_revisions: Tuple[str, ...] = ()
    labels: Tuple[str, ...] = ()
    without_labels: Tuple[str, ...] = ()
    assignee: Optional[str] = None
    reporter: Optional[str] = None
    open_only: bool = False
    unlabeled: bool = False

    def validated(self) -> "StructuredFilters":
        """Validate every family through the tracker's own vocabulary rules.

        Raises the same :class:`TrackerError` refusals the list path raises
        today, so direct CLI, API, and ranked-search callers share refusal
        semantics rather than growing a second dialect.

        Choice vocabularies (kind/status/severity) are validated here because
        both surfaces validate them identically today. Scalar text families
        (components, observed revisions, assignee, reporter) are deliberately
        passed through unmodified so the list path's historical exact-match
        behavior is preserved byte-for-byte; a caller that wants trimmed
        matching normalizes its input before constructing this object.
        """
        helpers = _issue_tracker()
        kinds = tuple(helpers["_validate_kind"](str(k)) for k in self.kinds)
        statuses = tuple(
            helpers["_validate_choice"](s, _statuses(), "status") for s in self.statuses
        )
        severities = tuple(
            helpers["_validate_choice"](s, _severities(), "severity") for s in self.severities
        )
        labels = tuple(helpers["normalise_labels"](self.labels))
        without_labels = tuple(helpers["normalise_labels"](self.without_labels))
        return StructuredFilters(
            kinds=kinds,
            statuses=statuses,
            severities=severities,
            components=self.components,
            observed_revisions=self.observed_revisions,
            labels=labels,
            without_labels=without_labels,
            assignee=self.assignee,
            reporter=self.reporter,
            open_only=self.open_only,
            unlabeled=self.unlabeled,
        )

    def orm_conditions(self) -> List[Any]:
        """AND-composed SQLAlchemy predicates over :class:`TrackerIssueModel`.

        Each family contributes at most one predicate group; an empty family
        contributes nothing. This is the single expression of §10.2 semantics
        shared by ``issue list`` and ranked-search candidate selection.
        """
        conditions: List[Any] = []
        if self.kinds:
            conditions.append(TrackerIssueModel.kind.in_(self.kinds))
        if self.statuses:
            conditions.append(TrackerIssueModel.status.in_(self.statuses))
        if self.open_only:
            conditions.append(TrackerIssueModel.status.notin_(tuple(_terminal_statuses())))
        if self.severities:
            conditions.append(TrackerIssueModel.severity.in_(self.severities))
        if self.components:
            conditions.append(TrackerIssueModel.component.in_(self.components))
        if self.observed_revisions:
            # Exact values OR'd within the family (§10.2): a caller pinning a
            # build revision gets every row recorded against any listed form.
            conditions.append(TrackerIssueModel.observed_revision.in_(self.observed_revisions))
        for label in self.labels:
            needle = _label_like_needle(label)
            conditions.append(
                TrackerIssueModel.labels.like(f'%"{needle}"%', escape=_LABEL_LIKE_ESCAPE)
            )
        for label in self.without_labels:
            needle = _label_like_needle(label)
            conditions.append(
                ~TrackerIssueModel.labels.like(f'%"{needle}"%', escape=_LABEL_LIKE_ESCAPE)
            )
        if self.unlabeled:
            # Stored label sets are always JSON arrays ("[]" when empty).
            conditions.append(TrackerIssueModel.labels == "[]")
        if self.assignee:
            conditions.append(TrackerIssueModel.assignee == self.assignee)
        if self.reporter:
            conditions.append(TrackerIssueModel.reporter == self.reporter)
        return conditions


def _statuses() -> Sequence[str]:
    from cli_agent_orchestrator.services.issue_tracker import STATUSES

    return STATUSES


def _severities() -> Sequence[str]:
    from cli_agent_orchestrator.services.issue_tracker import SEVERITIES

    return SEVERITIES


def _terminal_statuses() -> FrozenSet[str]:
    from cli_agent_orchestrator.services.issue_tracker import TERMINAL_STATUSES

    return TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# Subtree scope (design §10.1)
# ---------------------------------------------------------------------------


def subtree_closure(executor: Any, roots: Sequence[str]) -> Dict[str, Any]:
    """Complete transitive ``part-of`` descendant closure of ``roots``.

    ``executor`` is any SQLAlchemy-executable (Session or Connection); every
    statement runs through ``text()`` binding. Returns
    ``{"keys": frozenset, "root_presence": {root: bool}}``. Each root is
    included itself; every transitive child joined by a directed
    ``from_key --part-of--> to_key`` link is included. The recursive CTE uses
    ``UNION`` so a membership cycle terminates instead of looping, and no
    depth bound can silently omit descendants. A root naming a nonexistent
    issue is a typed refusal scoped to this request, never a silent empty set.
    """
    from cli_agent_orchestrator.services.issue_tracker import TrackerError

    wanted = list(dict.fromkeys(str(r).strip() for r in roots if str(r).strip()))
    if not wanted:
        return {"keys": frozenset(), "root_presence": {}}
    placeholders = ", ".join(f":root{i}" for i in range(len(wanted)))
    params: Dict[str, Any] = {f"root{i}": key for i, key in enumerate(wanted)}
    found = {
        str(row[0])
        for row in executor.execute(
            _sql_text(f"SELECT key FROM tracker_issues WHERE key IN ({placeholders})"),
            params,
        ).fetchall()
    }
    missing = [key for key in wanted if key not in found]
    if missing:
        raise TrackerError(
            "invalid",
            f"unknown subtree root(s): {', '.join(missing)}",
        )
    closure_sql = (
        "WITH RECURSIVE subtree(key) AS (\n"
        f"  SELECT key FROM tracker_issues WHERE key IN ({placeholders})\n"
        "  UNION\n"
        "  SELECT l.from_key FROM tracker_issue_links AS l\n"
        "  JOIN subtree ON l.to_key = subtree.key\n"
        "  WHERE l.kind = 'part-of'\n"
        ")\n"
        "SELECT key FROM subtree"
    )
    keys = frozenset(
        str(row[0]) for row in executor.execute(_sql_text(closure_sql), params).fetchall()
    )
    return {"keys": keys, "root_presence": {key: True for key in wanted}}


@dataclass(frozen=True)
class ScopeResolution:
    """One resolved request scope (design §10.1: exactly one scope form).

    ``allowed_keys`` is ``None`` when nothing constrains the candidate set
    (all-projects with no subtree roots); otherwise it is the complete set of
    issue keys the request may return — selected projects intersected with
    every subtree closure, so a cross-project descendant outside the selected
    tracker projects is excluded.
    """

    project_ids: Tuple[str, ...] = ()
    all_projects: bool = False
    subtree_roots: Tuple[str, ...] = ()
    closure_keys: FrozenSet[str] = field(default=frozenset())
    allowed_keys: Optional[FrozenSet[str]] = None


def resolve_scope(
    executor: Any,
    *,
    project_ids: Sequence[str],
    all_projects: bool,
    subtree_roots: Sequence[str],
) -> ScopeResolution:
    """Resolve the request scope into one concrete candidate-key constraint.

    ``executor`` is any SQLAlchemy-executable (Session or Connection).
    ``project_ids`` and ``all_projects`` are mutually exclusive; both/neither
    is a typed invalid request (§10.1). Subtree roots are optional and further
    constrain the selected projects; their closure is cycle-safe and complete.
    """
    from cli_agent_orchestrator.services.issue_tracker import TrackerError, _validate_slug

    if bool(project_ids) == bool(all_projects):
        raise TrackerError(
            "invalid-scope",
            "exactly one scope form is required: tracker project id(s) or all_projects",
        )
    projects = tuple(_validate_slug(p) for p in project_ids) if project_ids else ()
    roots = tuple(str(r).strip() for r in subtree_roots if str(r).strip())

    closure = subtree_closure(executor, roots)

    allowed: Optional[FrozenSet[str]] = None
    if projects:
        placeholders = ", ".join(f":p{i}" for i in range(len(projects)))
        params = {f"p{i}": pid for i, pid in enumerate(projects)}
        project_keys = {
            str(row[0])
            for row in executor.execute(
                _sql_text(f"SELECT key FROM tracker_issues WHERE project_id IN ({placeholders})"),
                params,
            ).fetchall()
        }
        allowed = frozenset(project_keys)
    if closure["keys"]:
        allowed = closure["keys"] if allowed is None else (allowed & closure["keys"])
    return ScopeResolution(
        project_ids=projects,
        all_projects=all_projects,
        subtree_roots=roots,
        closure_keys=closure["keys"],
        allowed_keys=allowed,
    )


# A query containing no alphanumeric character carries no literal term after
# FTS tokenization: quoting cannot save it, and treating it as empty matches
# the documented nonempty-normalized-text rule.
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def is_effectively_empty_query(text: str) -> bool:
    """True when normalized free-form text carries no searchable literal term."""
    return not (text or "").strip() or bool(_PUNCT_ONLY_RE.match(text.strip()))
