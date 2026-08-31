"""HTTP surface for project-scoped issue tracking.

Kept out of ``api/main.py`` on purpose: that module is already 5,500 lines,
and a tracker that has to be read alongside the managed-launch state machine
to be understood is a tracker nobody will extend.

Every route carries the same scope dependency as the rest of the API. A read
route accepts read/write/admin; a mutating route demands write or admin. The
scope check is a no-op while auth is disabled (the default localhost posture),
which is exactly why it must be present now rather than added the day auth is
turned on.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field

from cli_agent_orchestrator.security.auth import (
    SCOPE_ADMIN,
    SCOPE_READ,
    SCOPE_WRITE,
    require_any_scope,
)
from cli_agent_orchestrator.services import embedding_adapter
from cli_agent_orchestrator.services import issue_similar as similar
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services import project_dashboard
from cli_agent_orchestrator.services import search_index_maintenance as maintenance
from cli_agent_orchestrator.services import tracker_ranked_search as ranked

logger = logging.getLogger(__name__)

TRACKER_API_VERSION = 2
TRACKER_CAPABILITIES = (
    "atomic-issue-snapshot",
    "generic-item-kinds",
    "bug-diagnostics",
    "claims",
    "issue-graph",
    "hierarchy-audit",
    "project-dashboard",
    "searchable-field-options",
)

router = APIRouter(tags=["issue-tracker"])

_READ = Depends(require_any_scope(SCOPE_READ, SCOPE_WRITE, SCOPE_ADMIN))
_WRITE = Depends(require_any_scope(SCOPE_WRITE, SCOPE_ADMIN))

# A refusal the service classified keeps that classification at the HTTP
# boundary. Collapsing them all to 400 would make "you spelled the status
# wrong" and "that project is gone" indistinguishable to a CLI client that can
# only branch on the status code.
_STATUS_FOR_CODE = {
    "invalid": status.HTTP_400_BAD_REQUEST,
    "not-found": status.HTTP_404_NOT_FOUND,
    "conflict": status.HTTP_409_CONFLICT,
    "unresolved": status.HTTP_422_UNPROCESSABLE_ENTITY,
}


def _http(exc: tracker.TrackerError) -> HTTPException:
    # A refusal that observed record state carries it in ``details`` so a
    # programmatic caller (a wayfinder session retrying a stale map edit, a
    # worker that lost a claim) can branch on facts, not on message text.
    detail: Any = exc.message
    if exc.details:
        detail = {"message": exc.message, "code": exc.code, **exc.details}
    return HTTPException(
        status_code=_STATUS_FOR_CODE.get(exc.code, status.HTTP_400_BAD_REQUEST),
        detail=detail,
    )


# --------------------------------------------------------------------------
# request bodies
# --------------------------------------------------------------------------


class StrictBody(BaseModel):
    """Reject unknown fields instead of ignoring them.

    Pydantic's default is to drop what it does not recognise, which turns a
    misspelled or non-editable field into a silent no-op answered with 200 —
    `PATCH {"project_id": "other"}` looked like it moved an issue between
    projects and did nothing at all. A 422 naming the field is the only answer
    a client can act on.
    """

    model_config = ConfigDict(extra="forbid")


class ScopeBody(StrictBody):
    kind: str
    value: str


class ProjectCreateBody(StrictBody):
    name: str
    id: Optional[str] = None
    description: str = ""
    issue_prefix: Optional[str] = None
    scopes: List[ScopeBody] = Field(default_factory=list)


class ProjectUpdateBody(StrictBody):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    issue_prefix: Optional[str] = None


class IssueCreateBody(StrictBody):
    title: str
    kind: str = "bug"
    project_id: Optional[str] = None
    body: str = ""
    status: str = "open"
    severity: str = "unset"
    component: Optional[str] = None
    reporter: Optional[str] = None
    assignee: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    collaborators: List[str] = Field(default_factory=list)
    branches: List[str] = Field(default_factory=list)
    worktrees: List[str] = Field(default_factory=list)
    pull_requests: List[str] = Field(default_factory=list)
    failing_command: Optional[str] = None
    reproduction_steps: Optional[str] = None
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    evidence: Optional[str] = None
    observed_revision: Optional[str] = None
    session_name: Optional[str] = None
    terminal_id: Optional[str] = None
    source_path: Optional[str] = None
    cwd: Optional[str] = None
    alias: Optional[str] = None
    key: Optional[str] = None
    origin: str = "api"
    favorite: bool = False
    force: bool = False


class IssueUpdateBody(StrictBody):
    """Every field optional; only the ones present are applied.

    ``model_fields_set`` distinguishes "not sent" from "sent as null", which is
    what lets the dashboard clear an assignee (send ``""``) without every other
    unsent field being wiped.

    ``expected_updated_at`` is not a field but an optimistic-concurrency
    precondition on the write itself: when sent, the PATCH applies only if the
    issue's ``updated_at`` still equals it; a stale value answers 409 carrying
    the current version. ``add_labels``/``remove_labels``/``clear_labels`` are
    atomic label deltas; ``labels`` (full replacement) never combines with them.
    """

    title: Optional[str] = None
    body: Optional[str] = None
    status: Optional[str] = None
    severity: Optional[str] = None
    component: Optional[str] = None
    reporter: Optional[str] = None
    assignee: Optional[str] = None
    labels: Optional[List[str]] = None
    collaborators: Optional[List[str]] = None
    branches: Optional[List[str]] = None
    worktrees: Optional[List[str]] = None
    pull_requests: Optional[List[str]] = None
    add_labels: Optional[List[str]] = None
    remove_labels: Optional[List[str]] = None
    clear_labels: Optional[bool] = None
    failing_command: Optional[str] = None
    reproduction_steps: Optional[str] = None
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    evidence: Optional[str] = None
    observed_revision: Optional[str] = None
    resolution: Optional[str] = None
    duplicate_of: Optional[str] = None
    kind: Optional[str] = None
    favorite: Optional[bool] = None
    expected_updated_at: Optional[str] = None
    actor: Optional[str] = None
    force: bool = False
    drop_previous_assignee: bool = False


class FeatureCreateBody(StrictBody):
    title: str
    project_id: Optional[str] = None
    body: str = ""
    status: str = "open"
    severity: str = "unset"
    component: Optional[str] = None
    reporter: Optional[str] = None
    assignee: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    collaborators: List[str] = Field(default_factory=list)
    branches: List[str] = Field(default_factory=list)
    worktrees: List[str] = Field(default_factory=list)
    pull_requests: List[str] = Field(default_factory=list)
    evidence: Optional[str] = None
    session_name: Optional[str] = None
    terminal_id: Optional[str] = None
    source_path: Optional[str] = None
    cwd: Optional[str] = None
    alias: Optional[str] = None
    key: Optional[str] = None
    origin: str = "api"
    force: bool = False


class FeatureUpdateBody(StrictBody):
    """Feature edits share the issue update machinery, so they accept the same
    atomic label deltas and the same ``expected_updated_at`` precondition —
    the dashboard must not lie about generic support depending on kind."""

    title: Optional[str] = None
    body: Optional[str] = None
    status: Optional[str] = None
    severity: Optional[str] = None
    component: Optional[str] = None
    reporter: Optional[str] = None
    assignee: Optional[str] = None
    labels: Optional[List[str]] = None
    collaborators: Optional[List[str]] = None
    branches: Optional[List[str]] = None
    worktrees: Optional[List[str]] = None
    pull_requests: Optional[List[str]] = None
    add_labels: Optional[List[str]] = None
    remove_labels: Optional[List[str]] = None
    clear_labels: Optional[bool] = None
    evidence: Optional[str] = None
    resolution: Optional[str] = None
    duplicate_of: Optional[str] = None
    kind: Optional[str] = None
    expected_updated_at: Optional[str] = None
    actor: Optional[str] = None
    force: bool = False
    drop_previous_assignee: bool = False


class CommentBody(StrictBody):
    body: str
    author: Optional[str] = None
    # Optional at creation; defaults to ordinary/routine weight.
    important: bool = False


class CommentImportanceBody(StrictBody):
    """PATCH body for the one reversible importance update surface."""

    important: bool
    actor: Optional[str] = None


class LinkBody(StrictBody):
    to_key: str
    kind: str
    actor: Optional[str] = None


class ClaimBody(StrictBody):
    claimant: str


class UnclaimBody(StrictBody):
    actor: Optional[str] = None


class SimilarIssuesBody(StrictBody):
    """POST /tracker/issues/similar — an advisory duplicate probe.

    Exactly one of ``issue_key`` / ``draft`` and exactly one of
    ``project_ids`` / ``all_projects``; both XOR rules are validated by the
    service so the CLI refuses with the same typed codes. ``draft`` is a free
    dict at this boundary because its allowed-field set is owned by the
    service, which refuses undeclared fields by name instead of ignoring them.
    """

    issue_key: Optional[str] = None
    draft: Optional[Dict[str, Any]] = None
    project_ids: Optional[List[str]] = None
    all_projects: bool = False
    limit: int = Field(ranked.DEFAULT_LIMIT, ge=ranked.MIN_LIMIT, le=ranked.MAX_LIMIT)


class IssueSnapshotBody(StrictBody):
    project_id: str
    keys: List[str] = Field(min_length=1)


class SearchIndexRefreshBody(StrictBody):
    """POST /tracker/issues/search-index/refresh — outbox maintenance.

    ``all`` selects the full drain (and the activation offer for a finished
    building generation) over the bounded batch. ``retry_failed`` resets the
    backoff of documents whose embedding failed; ``limit`` bounds either the
    retry reset or the batch. Every field defaults, so a bare ``POST`` is the
    bounded read-shaped refresh.
    """

    all: bool = False
    retry_failed: bool = False
    limit: Optional[int] = Field(None, ge=1, le=1_000_000)


class SearchIndexRebuildBody(StrictBody):
    """POST /tracker/issues/search-index/rebuild — the derived-state repair verb.

    ``scope`` is exactly one of ``lexical`` (FTS documents), ``vectors`` (a
    fresh generation built and activated), or ``all``. Both scopes rebuild
    derived rows only and never touch an issue, comment, link, or event.
    """

    scope: str = Field("all", description="lexical|vectors|all")


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------


@router.get("/tracker/vocabulary")
async def tracker_vocabulary(_scopes: List[str] = _READ) -> Dict[str, Any]:
    """The enumerations the server will accept.

    Served rather than duplicated in the dashboard so a status added here can
    never disagree with the dropdown that offers it.
    """
    return {
        "statuses": list(tracker.STATUSES),
        "terminal_statuses": sorted(tracker.TERMINAL_STATUSES),
        "statuses_by_kind": {kind: list(tracker.STATUSES) for kind in tracker.ITEM_KINDS},
        "terminal_statuses_by_kind": {
            kind: sorted(tracker.TERMINAL_STATUSES) for kind in tracker.ITEM_KINDS
        },
        "item_kinds": list(tracker.ITEM_KINDS),
        "severities": list(tracker.SEVERITIES),
        "scope_kinds": list(tracker.SCOPE_KINDS),
        "link_kinds": list(tracker.LINK_KINDS),
        "project_statuses": list(tracker.PROJECT_STATUSES),
    }


@router.get("/tracker/projects")
async def list_projects(
    include_archived: bool = Query(False),
    _scopes: List[str] = _READ,
) -> List[Dict[str, Any]]:
    return tracker.list_projects(include_archived=include_archived)


@router.post("/tracker/projects", status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreateBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.create_project(
            name=body.name,
            project_id=body.id,
            description=body.description,
            issue_prefix=body.issue_prefix,
            scopes=[s.model_dump() for s in body.scopes],
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/resolve")
async def resolve_project(
    project: Optional[str] = Query(None),
    session: Optional[str] = Query(None),
    alias: Optional[str] = Query(None, description="a project_id-kind scope value"),
    cwd: Optional[str] = Query(None),
    git_remote: Optional[str] = Query(None),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    """Answer "which project would an issue filed here belong to?".

    Declared before ``/tracker/projects/{project_id}`` so the literal path wins
    over the parameterised one; ``resolve`` is a legal slug and would otherwise
    be swallowed as a project id.
    """
    try:
        return tracker.resolve_project(
            project=project, session=session, alias=alias, cwd=cwd, git_remote=git_remote
        ).as_dict()
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/{project_id}")
async def get_project(project_id: str, _scopes: List[str] = _READ) -> Dict[str, Any]:
    try:
        return tracker.get_project(project_id)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/{project_id}/dashboard")
async def project_home_dashboard(
    project_id: str,
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    try:
        return project_dashboard.project_home(project_id)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/{project_id}/sessions")
async def project_sessions(
    project_id: str,
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    try:
        return project_dashboard.project_sessions(project_id)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/{project_id}/sessions/{session_name}")
async def project_session_detail(
    project_id: str,
    session_name: str,
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    try:
        return project_dashboard.project_session(project_id, session_name)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/{project_id}/sessions/{session_name}/terminals/{terminal_id}/log")
async def project_terminal_log(
    project_id: str,
    session_name: str,
    terminal_id: str,
    mode: str = Query("last", pattern="^(last|full)$"),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    try:
        return project_dashboard.terminal_log(
            project_id,
            session_name,
            terminal_id,
            mode=mode,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.patch("/tracker/projects/{project_id}")
async def update_project(
    project_id: str,
    body: ProjectUpdateBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.update_project(
            project_id,
            name=body.name,
            description=body.description,
            status=body.status,
            issue_prefix=body.issue_prefix,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.delete("/tracker/projects/{project_id}")
async def delete_project(
    project_id: str,
    force: bool = Query(False, description="also delete the project's issues"),
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.delete_project(project_id, force=force)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/{project_id}/labels")
async def project_label_facets(project_id: str, _scopes: List[str] = _READ) -> Dict[str, Any]:
    """Every label on the project's issues with total/open counts — the
    discovery surface behind the dashboard's label filter bar."""
    try:
        return tracker.label_facets(project_id)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/{project_id}/options")
async def project_field_options(
    project_id: str,
    field: str = Query(..., description="label|component|assignee|reporter"),
    q: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    """Search a bounded vocabulary for an open-ended issue field."""
    try:
        return tracker.field_options(project_id, field=field, query=q, limit=limit)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/{project_id}/export")
async def export_project_markdown(
    project_id: str,
    open_only: bool = Query(True),
    kind: Optional[str] = Query(None),
    _scopes: List[str] = _READ,
) -> Response:
    """Render the issue log as markdown.

    The ledger files this replaces become a view produced on demand, not a
    second source of truth somebody has to keep in step.
    """
    effective_kind = kind if kind is not None else "bug"
    try:
        text = tracker.render_markdown(project_id, open_only=open_only, kind=effective_kind)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc
    return Response(content=text, media_type="text/markdown; charset=utf-8")


@router.post("/tracker/projects/{project_id}/scopes", status_code=status.HTTP_201_CREATED)
async def add_scope(
    project_id: str,
    body: ScopeBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.add_scope(project_id, kind=body.kind, value=body.value)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.delete("/tracker/projects/{project_id}/scopes/{scope_id}")
async def remove_scope(
    project_id: str,
    scope_id: int,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.remove_scope(project_id, scope_id)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


# --------------------------------------------------------------------------
# issues
# --------------------------------------------------------------------------


@router.get("/tracker/issues")
async def list_issues(
    project_id: Optional[str] = Query(None),
    status_filter: Optional[List[str]] = Query(None, alias="status"),
    severity: Optional[List[str]] = Query(None),
    component: Optional[str] = Query(None),
    assignee: Optional[str] = Query(None),
    reporter: Optional[str] = Query(None),
    label_filter: Optional[List[str]] = Query(None, alias="label"),
    without_label: Optional[List[str]] = Query(
        None, description="exclude issues carrying any of these exact labels"
    ),
    unlabeled: bool = Query(False, description="only issues with an empty label set"),
    q: Optional[str] = Query(None),
    open_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("created_desc"),
    kind: Optional[str] = Query(
        None, description="project|bug|feature|milestone|goal|epic|story|task|all"
    ),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    effective_kind = kind if kind is not None else "bug"
    try:
        return tracker.list_issues(
            project_id=project_id,
            status=status_filter,
            severity=severity,
            component=component,
            assignee=assignee,
            reporter=reporter,
            label=label_filter,
            without_label=without_label,
            unlabeled=unlabeled,
            query=q,
            open_only=open_only,
            limit=limit,
            offset=offset,
            order=order,
            kind=effective_kind,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/issues/search")
async def search_issues(
    q: str = Query(..., description="free-form ranked-search text"),
    project_id: Optional[List[str]] = Query(None, description="tracker project id, repeatable"),
    all_projects: bool = Query(False, description="search every tracker project"),
    under: Optional[List[str]] = Query(
        None, description="restrict to the part-of subtree rooted at this issue key, repeatable"
    ),
    kind: Optional[List[str]] = Query(None, description="item kind filter, repeatable"),
    status_filter: Optional[List[str]] = Query(None, alias="status", description="repeatable"),
    severity: Optional[List[str]] = Query(None, description="repeatable"),
    component: Optional[List[str]] = Query(None, description="exact component, repeatable"),
    observed_revision: Optional[List[str]] = Query(
        None, description="exact observed revision, repeatable"
    ),
    label: Optional[List[str]] = Query(
        None, description="required exact label, repeatable (AND-composed)"
    ),
    without_label: Optional[List[str]] = Query(
        None, description="excluded exact label, repeatable (none-of)"
    ),
    assignee: Optional[str] = Query(None),
    reporter: Optional[str] = Query(None),
    open_only: bool = Query(False),
    unlabeled: bool = Query(False),
    include_comments: bool = Query(True),
    mode: str = Query("lexical", description="lexical|semantic|hybrid; uninstalled modes degrade"),
    limit: int = Query(ranked.DEFAULT_LIMIT, ge=ranked.MIN_LIMIT, le=ranked.MAX_LIMIT),
    offset: int = Query(0, ge=0),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    """Explained ranked search over issues and their comments.

    Declared before ``/tracker/issues/{issue_key}`` so the literal path wins
    over the parameterised one — ``search`` would otherwise be swallowed as an
    issue key, exactly as ``resolve`` and ``stats`` are protected below.

    The route is a thin adapter: every filter family forwards to the shared
    ranked-search service request, repeated query parameters become repeated
    filter values, and scope is exactly one of ``project_id`` (one or more) or
    ``all_projects`` — both or neither is a typed invalid request. The service
    owns validation, bounds, degradation metadata, and the refusal codes this
    boundary maps onto HTTP statuses.
    """
    request = ranked.RankedSearchRequest(
        query=q,
        project_ids=tuple(project_id or ()),
        all_projects=all_projects,
        subtree_roots=tuple(under or ()),
        kinds=tuple(kind or ()),
        statuses=tuple(status_filter or ()),
        severities=tuple(severity or ()),
        components=tuple(component or ()),
        observed_revisions=tuple(observed_revision or ()),
        labels=tuple(label or ()),
        without_labels=tuple(without_label or ()),
        assignee=assignee,
        reporter=reporter,
        open_only=open_only,
        unlabeled=unlabeled,
        include_comments=include_comments,
        mode=mode,
        limit=limit,
        offset=offset,
    )
    try:
        return ranked.ranked_search(request)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.post("/tracker/issues/similar")
async def similar_issues(
    body: SimilarIssuesBody,
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    """Explained similar-issue candidates for an existing issue or a draft.

    Declared before ``/tracker/issues/{issue_key}`` so the literal path wins
    over the parameterised one — ``similar`` would otherwise be swallowed as
    an issue key, exactly as ``search``, ``resolve``, and ``stats`` are
    protected above. Read-scoped and advisory by contract: nothing here sits
    in the create path, so a similarity failure can never block filing.
    """
    try:
        return similar.find_similar_issues(
            similar.SimilarIssuesRequest(
                issue_key=body.issue_key,
                draft=body.draft,
                project_ids=tuple(body.project_ids or ()),
                all_projects=body.all_projects,
                limit=body.limit,
            )
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.post("/tracker/issues/snapshot")
async def snapshot_issues(
    body: IssueSnapshotBody,
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    """One deterministic tracker export from a single SQLite read instant.

    This literal route is declared before ``/tracker/issues/{issue_key}`` so
    ``snapshot`` cannot be interpreted as an issue key by a parameterised
    route in this or a later API version.
    """
    try:
        return tracker.snapshot_issues(project_id=body.project_id, keys=body.keys)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/issues/stats")
async def issue_stats(
    project_id: Optional[str] = Query(None),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    try:
        return tracker.stats(project_id)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


# --------------------------------------------------------------------------
# search-index maintenance (cond-0770)
# --------------------------------------------------------------------------

# A maintenance refusal keeps the reason the orchestrator observed. Most of
# them are "the installation is not in the state this verb needs" rather than
# "the request is malformed", so 409 — not 400 — is the honest default; a
# caller branches on ``reason`` in the detail either way.
_STATUS_FOR_MAINTENANCE_REASON = {
    "invalid-scope": status.HTTP_400_BAD_REQUEST,
    "unknown-generation": status.HTTP_404_NOT_FOUND,
}
_MAINTENANCE_DEFAULT_STATUS = status.HTTP_409_CONFLICT


def _maintenance_http(exc: Exception) -> HTTPException:
    """Carry a typed maintenance refusal, its reason, and its remedy over HTTP."""
    reason = getattr(exc, "reason", type(exc).__name__)
    message = str(getattr(exc, "message", None) or exc)
    detail: Any = {"message": message, "reason": reason}
    action = getattr(exc, "action", None)
    if action:
        detail["action"] = action
    return HTTPException(
        status_code=_STATUS_FOR_MAINTENANCE_REASON.get(reason, _MAINTENANCE_DEFAULT_STATUS),
        detail=detail,
    )


@router.get("/tracker/issues/search-index/status")
async def search_index_status(_scopes: List[str] = _READ) -> Dict[str, Any]:
    """Capability, engine, lexical, and semantic state of the search index.

    Declared before ``/tracker/issues/{issue_key}`` so the literal path wins
    over the parameterised one, exactly like ``search``, ``similar``, and
    ``stats``. Read-only and cheap enough to poll: it never loads model
    weights, and every degraded state names the operator action that repairs
    it under ``next_actions``.
    """
    try:
        return maintenance.index_status()
    except maintenance.SearchIndexMaintenanceError as exc:
        raise _maintenance_http(exc) from exc


@router.get("/tracker/issues/search-index/integrity-check")
async def search_index_integrity(_scopes: List[str] = _READ) -> Dict[str, Any]:
    """Read-only §13.4 integrity report over the derived index.

    Reports FTS internal integrity, source-to-FTS coverage, duplicate and
    orphan document keys, dirty/failed/ready vector counts, stale vectors,
    generation provenance, per-project coverage, and last failures. It
    repairs nothing — ``rebuild`` is the counterpart verb.
    """
    try:
        return maintenance.integrity_check()
    except maintenance.SearchIndexMaintenanceError as exc:
        raise _maintenance_http(exc) from exc


@router.post("/tracker/issues/search-index/refresh")
async def search_index_refresh(
    body: SearchIndexRefreshBody = SearchIndexRefreshBody(),
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    """Embed the queued documents of every active and building generation.

    Write-scoped maintenance per §9.2: unlike the query-time bounded drain a
    semantic search performs, ``all=true`` drains completely and offers a
    finished building generation for activation. An incomplete build is kept
    from going live by the coverage proof inside the activation transaction,
    not by this route's judgement.
    """
    try:
        return maintenance.refresh_index(
            all=body.all,
            retry_failed=body.retry_failed,
            limit=body.limit,
        )
    except (maintenance.SearchIndexMaintenanceError, embedding_adapter.EmbeddingCapabilityError) as exc:
        raise _maintenance_http(exc) from exc


@router.post("/tracker/issues/search-index/rebuild")
async def search_index_rebuild(
    body: SearchIndexRebuildBody = SearchIndexRebuildBody(),
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    """Repair the derived index; never rewrites authoritative tracker rows.

    ``scope=lexical`` repopulates the FTS documents with fresh content
    versions and requeues every live document; ``scope=vectors`` builds a
    fresh generation and activates it only after the coverage proof passes;
    ``scope=all`` does both in that order.
    """
    try:
        return maintenance.rebuild_index(scope=body.scope)
    except maintenance.SearchIndexMaintenanceError as exc:
        raise _maintenance_http(exc) from exc


@router.post("/tracker/issues", status_code=status.HTTP_201_CREATED)
async def create_issue(
    body: IssueCreateBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.create_issue(
            project_id=body.project_id,
            title=body.title,
            kind=body.kind,
            body=body.body,
            status=body.status,
            severity=body.severity,
            component=body.component,
            reporter=body.reporter,
            assignee=body.assignee,
            labels=body.labels,
            collaborators=body.collaborators,
            branches=body.branches,
            worktrees=body.worktrees,
            pull_requests=body.pull_requests,
            failing_command=body.failing_command,
            reproduction_steps=body.reproduction_steps,
            expected_outcome=body.expected_outcome,
            actual_outcome=body.actual_outcome,
            evidence=body.evidence,
            observed_revision=body.observed_revision,
            session_name=body.session_name,
            terminal_id=body.terminal_id,
            source_path=body.source_path,
            cwd=body.cwd,
            alias=body.alias,
            key=body.key,
            origin=body.origin,
            favorite=body.favorite,
            force=body.force,
            enforce_bug_details=True,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/issues/{issue_key}")
async def get_issue(issue_key: str, _scopes: List[str] = _READ) -> Dict[str, Any]:
    try:
        return tracker.get_issue(issue_key)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.patch("/tracker/issues/{issue_key}")
async def update_issue(
    issue_key: str,
    body: IssueUpdateBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    changes = {
        name: value
        for name, value in body.model_dump().items()
        if name not in ("actor", "expected_updated_at", "force", "drop_previous_assignee")
        and name in body.model_fields_set
    }
    try:
        existing = tracker.get_issue(issue_key)
        _assert_issue(existing)
        # Duplicate status requires canonical key validation (P1)
        if (
            changes.get("status") == "duplicate"
            and not changes.get("duplicate_of")
            and not existing.get("duplicate_of")
        ):
            raise tracker.TrackerError(
                "invalid", "duplicate status requires duplicate_of canonical key"
            )
        return tracker.update_issue(
            issue_key,
            actor=body.actor,
            expected_updated_at=body.expected_updated_at,
            force=body.force,
            drop_previous_assignee=body.drop_previous_assignee,
            **changes,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.delete("/tracker/issues/{issue_key}")
async def delete_issue(issue_key: str, _scopes: List[str] = _WRITE) -> Dict[str, Any]:
    try:
        existing = tracker.get_issue(issue_key)
        _assert_issue(existing)
        return tracker.delete_issue(issue_key)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.post("/tracker/issues/{issue_key}/comments", status_code=status.HTTP_201_CREATED)
async def add_comment(
    issue_key: str,
    body: CommentBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.add_comment(
            issue_key, body=body.body, author=body.author, important=body.important
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.patch("/tracker/issues/{issue_key}/comments/{comment_id}")
async def set_comment_importance(
    issue_key: str,
    comment_id: int,
    body: CommentImportanceBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    """Set or clear a comment's ``important`` flag (idempotent and reversible)."""
    try:
        return tracker.set_comment_importance(
            issue_key,
            comment_id,
            important=body.important,
            actor=body.actor,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.delete("/tracker/issues/{issue_key}/comments/{comment_id}")
async def delete_comment(
    issue_key: str,
    comment_id: int,
    actor: Optional[str] = Query(None),
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.delete_comment(issue_key, comment_id, actor=actor)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.post("/tracker/issues/{issue_key}/links", status_code=status.HTTP_201_CREATED)
async def add_link(
    issue_key: str,
    body: LinkBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.add_link(issue_key, to_key=body.to_key, kind=body.kind, actor=body.actor)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.delete("/tracker/issues/{issue_key}/links/{link_id}")
async def remove_link(
    issue_key: str,
    link_id: int,
    actor: Optional[str] = Query(None),
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.remove_link(link_id, actor=actor)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


# --------------------------------------------------------------------------
# map membership, frontier, and claim lifecycle (cond-0394)
#
# These routes serve the wayfinding workflow: a map is an issue, its tickets
# are `part-of` children, the frontier is what a session may take next, and
# claim/unclaim is how concurrent sessions avoid picking the same ticket.
# They are deliberately generic over the shared store — a feature can be a map
# or a ticket exactly as an issue can.
# --------------------------------------------------------------------------


@router.get("/tracker/issues/{issue_key}/children")
async def list_children(issue_key: str, _scopes: List[str] = _READ) -> Dict[str, Any]:
    """Direct children of a parent/map issue (its `part-of` members)."""
    try:
        return tracker.list_children(issue_key)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/issues/{issue_key}/frontier")
async def issue_frontier(issue_key: str, _scopes: List[str] = _READ) -> Dict[str, Any]:
    """Takeable children: nonterminal, unassigned, no nonterminal blocker.

    Ordered by creation (``created_at``, ties by row id) — deterministic, so
    concurrent sessions picking "the first frontier ticket" see the same list.
    """
    try:
        return tracker.frontier(issue_key)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/issues/{issue_key}/map")
async def issue_map_projection(issue_key: str, _scopes: List[str] = _READ) -> Dict[str, Any]:
    """The one-request map projection: map, classified children, ordered
    frontier, member links, every external link endpoint (each carrying the
    member children it actually blocks), and progress counts."""
    try:
        return tracker.map_projection(issue_key)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/issues/{issue_key}/graph")
async def issue_graph_projection(
    issue_key: str,
    max_depth: int = Query(default=8, ge=1, le=12),
    max_nodes: int = Query(default=300, ge=1, le=500),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    """Bounded transitive ``part-of`` hierarchy rooted at any issue, plus
    every visible relationship and its materialized external endpoint."""
    try:
        return tracker.graph_projection(issue_key, max_depth=max_depth, max_nodes=max_nodes)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/issues/{issue_key}/audit")
async def issue_hierarchy_audit(
    issue_key: str,
    max_depth: int = Query(default=8, ge=1, le=12),
    max_nodes: int = Query(default=300, ge=1, le=500),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    """Audit one bounded transitive hierarchy and derive its leaf frontier."""
    try:
        return tracker.hierarchy_audit(issue_key, max_depth=max_depth, max_nodes=max_nodes)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.post("/tracker/issues/{issue_key}/claim")
async def claim_issue(
    issue_key: str,
    body: ClaimBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    """Atomically claim an open issue. A second claimant gets 409 naming the
    observed owner; a retry by the current claimant is idempotent."""
    try:
        return tracker.claim_issue(issue_key, claimant=body.claimant)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.post("/tracker/issues/{issue_key}/unclaim")
async def unclaim_issue(
    issue_key: str,
    body: UnclaimBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    """Release a claim — the ordinary exit from a stale assignment."""
    try:
        return tracker.unclaim_issue(issue_key, actor=body.actor)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


# --------------------------------------------------------------------------
# features — typed aliases over shared storage (D4)
# --------------------------------------------------------------------------


def _assert_feature(row: dict) -> None:
    if row.get("kind") != "feature":
        raise tracker.TrackerError("not-found", f"no such feature: {row.get('key')}")


def _assert_issue(row: dict) -> None:
    # The generic issue surface addresses every planning-item type. The
    # feature routes remain compatibility aliases with a stricter assertion.
    return None


@router.get("/tracker/features")
async def list_features(
    project_id: Optional[str] = Query(None),
    status_filter: Optional[List[str]] = Query(None, alias="status"),
    severity: Optional[List[str]] = Query(None),
    component: Optional[str] = Query(None),
    assignee: Optional[str] = Query(None),
    reporter: Optional[str] = Query(None),
    label: Optional[str] = Query(None),
    without_label: Optional[List[str]] = Query(
        None, description="exclude features carrying any of these exact labels"
    ),
    unlabeled: bool = Query(False, description="only features with an empty label set"),
    q: Optional[str] = Query(None),
    open_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("created_desc"),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    try:
        return tracker.list_features(
            project_id=project_id,
            status=status_filter,
            severity=severity,
            component=component,
            assignee=assignee,
            reporter=reporter,
            label=label,
            without_label=without_label,
            unlabeled=unlabeled,
            query=q,
            open_only=open_only,
            limit=limit,
            offset=offset,
            order=order,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/features/stats")
async def feature_stats(
    project_id: Optional[str] = Query(None),
    _scopes: List[str] = _READ,
) -> Dict[str, Any]:
    try:
        return tracker.stats(project_id, kind="feature")
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.post("/tracker/features", status_code=201)
async def create_feature(
    body: FeatureCreateBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        return tracker.create_feature(
            project_id=body.project_id,
            title=body.title,
            body=body.body,
            status=body.status,
            severity=body.severity,
            component=body.component,
            reporter=body.reporter,
            assignee=body.assignee,
            labels=body.labels,
            collaborators=body.collaborators,
            branches=body.branches,
            worktrees=body.worktrees,
            pull_requests=body.pull_requests,
            evidence=body.evidence,
            session_name=body.session_name,
            terminal_id=body.terminal_id,
            source_path=body.source_path,
            cwd=body.cwd,
            alias=body.alias,
            key=body.key,
            origin=body.origin,
            force=body.force,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/features/{feature_key}")
async def get_feature(feature_key: str, _scopes: List[str] = _READ) -> Dict[str, Any]:
    try:
        row = tracker.get_issue(feature_key)
        _assert_feature(row)
        return row
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.patch("/tracker/features/{feature_key}")
async def update_feature(
    feature_key: str,
    body: FeatureUpdateBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        existing = tracker.get_issue(feature_key)
        _assert_feature(existing)
        changes = {
            name: value
            for name, value in body.model_dump().items()
            if name not in ("actor", "expected_updated_at", "force", "drop_previous_assignee")
            and name in body.model_fields_set
        }
        return tracker.update_issue(
            feature_key,
            actor=body.actor,
            expected_updated_at=body.expected_updated_at,
            force=body.force,
            drop_previous_assignee=body.drop_previous_assignee,
            **changes,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.delete("/tracker/features/{feature_key}")
async def delete_feature(feature_key: str, _scopes: List[str] = _WRITE) -> Dict[str, Any]:
    try:
        existing = tracker.get_issue(feature_key)
        _assert_feature(existing)
        return tracker.delete_issue(feature_key)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.post("/tracker/features/{feature_key}/comments", status_code=201)
async def add_feature_comment(
    feature_key: str,
    body: CommentBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        existing = tracker.get_issue(feature_key)
        _assert_feature(existing)
        return tracker.add_comment(
            feature_key, body=body.body, author=body.author, important=body.important
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.patch("/tracker/features/{feature_key}/comments/{comment_id}")
async def set_feature_comment_importance(
    feature_key: str,
    comment_id: int,
    body: CommentImportanceBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        existing = tracker.get_issue(feature_key)
        _assert_feature(existing)
        return tracker.set_comment_importance(
            feature_key,
            comment_id,
            important=body.important,
            actor=body.actor,
        )
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.delete("/tracker/features/{feature_key}/comments/{comment_id}")
async def delete_feature_comment(
    feature_key: str,
    comment_id: int,
    actor: Optional[str] = Query(None),
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        existing = tracker.get_issue(feature_key)
        _assert_feature(existing)
        return tracker.delete_comment(feature_key, comment_id, actor=actor)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.post("/tracker/features/{feature_key}/links", status_code=201)
async def add_feature_link(
    feature_key: str,
    body: LinkBody,
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        existing = tracker.get_issue(feature_key)
        _assert_feature(existing)
        return tracker.add_link(feature_key, to_key=body.to_key, kind=body.kind, actor=body.actor)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.delete("/tracker/features/{feature_key}/links/{link_id}")
async def remove_feature_link(
    feature_key: str,
    link_id: int,
    actor: Optional[str] = Query(None),
    _scopes: List[str] = _WRITE,
) -> Dict[str, Any]:
    try:
        existing = tracker.get_issue(feature_key)
        _assert_feature(existing)
        # Verify link belongs to this feature (URL key must match link's from_key/to_key)
        from cli_agent_orchestrator.clients.database import SessionLocal, TrackerLinkModel

        with SessionLocal() as db:
            link = db.get(TrackerLinkModel, int(link_id))
            if link is None or (
                link.from_key != feature_key.lower() and link.to_key != feature_key.lower()
            ):
                raise tracker.TrackerError(
                    "not-found", f"no link {link_id} on feature {feature_key}"
                )
        return tracker.remove_link(link_id, actor=actor)
    except tracker.TrackerError as exc:
        raise _http(exc) from exc


@router.get("/tracker/projects/{project_id}/features/export")
async def export_project_features_markdown(
    project_id: str,
    open_only: bool = Query(True),
    _scopes: List[str] = _READ,
) -> Response:
    try:
        text = tracker.render_markdown(project_id, open_only=open_only, kind="feature")
    except tracker.TrackerError as exc:
        raise _http(exc) from exc
    return Response(content=text, media_type="text/markdown; charset=utf-8")
