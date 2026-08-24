"""Similar-issue detection ahead of issue creation (cond-0645).

One question, two inputs: an existing issue key ("what looks like this
issue?") or a create-shaped draft ("would filing this duplicate something?").
Retrieval reuses the merged ranked-search service in lexical mode with
comments disabled, and scope resolution reuses the shared builder, so neither
candidate ranking nor project scoping can drift from ``issue search``.

Contract (tracker cond-0645):

- Exactly one of ``issue_key`` / ``draft`` and exactly one of ``project_ids``
  / ``all_projects``; every violation is a typed refusal, never a default.
- A draft carries only declared create/search fields — server-owned identity,
  status, and relation fields are refused, never silently ignored.
- The search spans open and terminal statuses, excludes the source issue
  itself, defaults the draft kind to ``bug`` (the create-path default), and
  keeps comment documents out of the match set.
- Every candidate returns the full ranked-search explanation; confirmed
  duplicates of hits (issues whose ``duplicate_of`` names a hit) are expanded
  one level beside their hit.
- NON-GATING by contract: this surface is advisory. Nothing here sits in the
  create path, so a similarity failure can never block issue creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cli_agent_orchestrator.clients.database import SessionLocal, TrackerIssueModel
from cli_agent_orchestrator.services import tracker_ranked_search as ranked

#: The create/search fields a draft may carry. Server-owned identity
#: (``key``, ``project_id``), status, and relation fields (``duplicate_of``,
#: collaborators, links) are deliberately absent: this surface describes work,
#: it does not file it.
DRAFT_FIELDS: Tuple[str, ...] = (
    "title",
    "kind",
    "body",
    "severity",
    "component",
    "reporter",
    "assignee",
    "labels",
    "failing_command",
    "reproduction_steps",
    "expected_outcome",
    "actual_outcome",
    "evidence",
    "observed_revision",
)

#: A draft without an explicit kind searches the kind the create path would
#: have filed it as.
DEFAULT_DRAFT_KIND = "bug"

#: Prose fields beyond title/body folded into the composed query text —
#: bug diagnostics carry the discriminating signal for near-duplicates.
QUERY_PROSE_FIELDS: Tuple[str, ...] = (
    "failing_command",
    "reproduction_steps",
    "expected_outcome",
    "actual_outcome",
    "evidence",
)


@dataclass(frozen=True)
class SimilarIssuesRequest:
    """One similar-issue lookup request.

    Exactly one of ``issue_key`` / ``draft`` and exactly one of
    ``project_ids`` / ``all_projects`` is required; the service owns both
    validations so every caller shares the same refusals.
    """

    issue_key: Optional[str] = None
    draft: Optional[Dict[str, Any]] = None
    project_ids: Tuple[str, ...] = ()
    all_projects: bool = False
    limit: int = ranked.DEFAULT_LIMIT
    include_comments: bool = False


def _validate_request(request: SimilarIssuesRequest) -> None:
    from cli_agent_orchestrator.services.issue_tracker import TrackerError

    source_key = str(request.issue_key).strip() if request.issue_key else ""
    if bool(source_key) == (request.draft is not None):
        raise TrackerError(
            "invalid",
            "exactly one of issue_key or draft is required",
        )
    if request.draft is not None:
        if not isinstance(request.draft, dict):
            raise TrackerError("invalid", "draft must be a JSON object")
        undeclared = sorted(str(name) for name in request.draft if name not in DRAFT_FIELDS)
        if undeclared:
            raise TrackerError(
                "invalid",
                "draft carries undeclared field(s): " + ", ".join(undeclared),
            )
    if bool(request.project_ids) == bool(request.all_projects):
        raise TrackerError(
            "invalid-scope",
            "exactly one scope form is required: tracker project id(s) or all_projects",
        )
    if request.limit < ranked.MIN_LIMIT or request.limit > ranked.MAX_LIMIT:
        raise TrackerError(
            "invalid",
            f"limit must be between {ranked.MIN_LIMIT} and {ranked.MAX_LIMIT}",
        )


def _compose_query(fields: Dict[str, Any]) -> str:
    """Fold the draft/issue prose into one literal query for ranked search.

    Title leads; body and bug-diagnostics prose follow. The composed text is
    capped at the ranked service's request bounds so a large issue body can
    never turn into a size refusal — truncation is the intended behavior for
    a similarity probe, not an error.
    """
    parts = [str(fields.get("title") or ""), str(fields.get("body") or "")]
    parts.extend(str(fields.get(name) or "") for name in QUERY_PROSE_FIELDS)
    text = " ".join(part.strip() for part in parts if part and part.strip())
    text = text[: ranked.MAX_QUERY_CHARS]
    return " ".join(text.split()[: ranked.MAX_QUERY_UNITS])


def _expand_duplicate_chains(
    hit_keys: List[str], *, exclude_key: Optional[str]
) -> List[Dict[str, Any]]:
    """Confirmed duplicates of the returned hits, expanded one level.

    An issue whose ``duplicate_of`` names a hit is a confirmed duplicate of
    that hit regardless of its own status. Expansion is exactly one level —
    chains deeper than that belong to the issue graph, not to this advisory
    answer. The source issue never appears in its own expansion.
    """
    from cli_agent_orchestrator.services.issue_tracker import _issue_row

    if not hit_keys:
        return []
    with SessionLocal() as db:
        rows = (
            db.query(TrackerIssueModel)
            .filter(TrackerIssueModel.duplicate_of.in_(hit_keys))
            .order_by(TrackerIssueModel.duplicate_of, TrackerIssueModel.key)
            .all()
        )
    expansions: List[Dict[str, Any]] = []
    for row in rows:
        if exclude_key and str(row.key).casefold() == str(exclude_key).casefold():
            continue
        expansions.append({"duplicate_of": str(row.duplicate_of), "issue": _issue_row(row)})
    return expansions


def find_similar_issues(request: SimilarIssuesRequest) -> Dict[str, Any]:
    """Answer "what already exists that looks like this?" with explanations.

    The ranked-search envelope is narrowed, not reshaped: candidates keep the
    full per-result explanation objects, and the wrapper adds only what this
    surface owns — the query source, self-exclusion, and the one-level
    duplicate-chain expansion beside their hits.
    """
    from cli_agent_orchestrator.services.issue_tracker import get_issue

    _validate_request(request)

    source_key: Optional[str] = None
    fields: Dict[str, Any]
    if request.issue_key:
        row = get_issue(str(request.issue_key).strip())
        source_key = str(row["key"])
        fields = {
            "title": row.get("title"),
            "body": row.get("body"),
            **{name: row.get(name) for name in QUERY_PROSE_FIELDS},
        }
        effective_kind = str(row.get("kind") or DEFAULT_DRAFT_KIND)
    else:
        fields = dict(request.draft or {})
        effective_kind = str(fields.get("kind") or DEFAULT_DRAFT_KIND)

    query_text = _compose_query(fields)
    # Self-exclusion runs after retrieval, so issue-key mode fetches one
    # extra candidate: when the source itself ranks within ``limit``, a
    # near-duplicate just outside it still survives exclusion.
    fetch_limit = (
        min(request.limit + 1, ranked.MAX_LIMIT) if source_key else request.limit
    )
    response = ranked.ranked_search(
        ranked.RankedSearchRequest(
            query=query_text,
            project_ids=request.project_ids,
            all_projects=request.all_projects,
            kinds=(effective_kind,),
            include_comments=request.include_comments,
            limit=fetch_limit,
        )
    )

    candidates: List[Dict[str, Any]] = []
    for result in response["results"]:
        issue = result.get("issue") or {}
        if source_key and str(issue.get("key", "")).casefold() == source_key.casefold():
            continue
        candidates.append(result)
    if source_key:
        candidates = candidates[: request.limit]

    hit_keys = [
        str((candidate.get("issue") or {}).get("key"))
        for candidate in candidates
        if (candidate.get("issue") or {}).get("key")
    ]
    expansions = _expand_duplicate_chains(hit_keys, exclude_key=source_key)

    return {
        "query_source": {
            "mode": "issue_key" if source_key else "draft",
            "issue_key": source_key,
            "kind": effective_kind,
        },
        "query": response["query"],
        "scope": response["scope"],
        "include_comments": request.include_comments,
        "limit": request.limit,
        "total": len(candidates),
        "candidates": candidates,
        "duplicate_expansions": expansions,
    }


__all__ = [
    "DEFAULT_DRAFT_KIND",
    "DRAFT_FIELDS",
    "SimilarIssuesRequest",
    "find_similar_issues",
]
