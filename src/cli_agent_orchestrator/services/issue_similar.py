"""Similar-issue detection ahead of issue creation (cond-0645).

One question, two inputs: an existing issue key ("what looks like this
issue?") or a create-shaped draft ("would filing this duplicate something?").
Retrieval reuses the merged ranked-search service through bounded literal-safe
probes, and scope resolution reuses the shared builder, so neither candidate
ranking nor project scoping can drift from ``issue search``.  Probe fusion is
local to this advisory surface; normal free-form search keeps its all-terms
semantics.

Contract (tracker cond-0645):

- Exactly one of ``issue_key`` / ``draft`` and exactly one of ``project_ids``
  / ``all_projects``; every violation is a typed refusal, never a default.
- A draft carries only declared create/search fields — server-owned identity,
  status, and relation fields are refused, never silently ignored.
- The search spans open and terminal statuses, excludes the source issue
  itself, defaults the draft kind to ``bug`` (the create-path default), and
  keeps comment documents out of the match set.
- Every candidate returns the full ranked-search explanation plus the bounded
  probe contributions that produced it, and the envelope carries
  mode/degradation/coverage facts; confirmed duplicates of hits (via either
  the native directional link or legacy ``duplicate_of`` field) are expanded
  one level beside their hit. Native links are authoritative when a legacy
  field disagrees with them.
- NON-GATING by contract: this surface is advisory. Nothing here sits in the
  create path, so a similarity failure can never block issue creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from cli_agent_orchestrator.clients.database import (
    SessionLocal,
    TrackerIssueModel,
    TrackerLinkModel,
)
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
QUERY_CONTEXT_FIELDS: Tuple[str, ...] = ("component", "observed_revision", "resolution")

# Similarity is allowed to issue a small number of independent, literal-safe
# ranked searches.  This is deliberately local to the advisory probe: normal
# free-form ``issue search`` remains an all-terms query.  The first query is
# the complete draft; the bounded field/drop-one probes preserve recall when a
# draft has one wording drift without turning generic prose into a broad OR.
SIMILARITY_MAX_PROBES = 12
SIMILARITY_PRIMARY_WEIGHT = 2.0
SIMILARITY_FALLBACK_WEIGHT = 1.0
SIMILARITY_DROP_ONE_WEIGHT = 0.5
SIMILARITY_RRF_K = ranked.RRF_K


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
    mode: str = "hybrid"


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
    if request.mode not in ("lexical", "semantic", "hybrid"):
        raise TrackerError("invalid", f"unknown search mode {request.mode!r}")


def _compose_query(fields: Dict[str, Any]) -> str:
    """Fold the draft/issue prose into one literal query for ranked search.

    Title leads; body and bug-diagnostics prose follow. The composed text is
    capped at the ranked service's request bounds so a large issue body can
    never turn into a size refusal — truncation is the intended behavior for
    a similarity probe, not an error.
    """
    parts = [str(fields.get("title") or ""), str(fields.get("body") or "")]
    parts.extend(str(fields.get(name) or "") for name in QUERY_PROSE_FIELDS + QUERY_CONTEXT_FIELDS)
    text = " ".join(part.strip() for part in parts if part and part.strip())
    text = text[: ranked.MAX_QUERY_CHARS]
    return " ".join(text.split()[: ranked.MAX_QUERY_UNITS])


def _bounded_query(value: Any) -> str:
    """Normalize one field into a bounded query without interpreting syntax."""
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text[: ranked.MAX_QUERY_CHARS].split()[: ranked.MAX_QUERY_UNITS])


def _similarity_probes(fields: Dict[str, Any], primary: str) -> List[Tuple[str, str, float]]:
    """Return a deterministic, bounded set of literal ranked-search probes.

    The complete draft remains the highest-weight probe.  Independent field
    probes avoid making a single diagnostic token an implicit AND requirement;
    bounded drop-one probes cover a near duplicate with one token drift while
    still requiring the remaining field terms.  Every query goes through
    ``ranked_search`` and therefore through its literal-safe FTS renderer.
    """
    probes: List[Tuple[str, str, float]] = []
    seen: Set[str] = set()

    def add(label: str, query: str, weight: float, *, dedupe: bool = True) -> None:
        normalized = _bounded_query(query)
        if (
            not normalized
            or (dedupe and normalized in seen)
            or len(probes) >= SIMILARITY_MAX_PROBES
        ):
            return
        seen.add(normalized)
        probes.append((label, normalized, weight))

    add("draft", primary, SIMILARITY_PRIMARY_WEIGHT)
    # Reserve one probe for every populated create/search field before adding
    # any surplus drop-one variants.  A long title must never consume the
    # bounded budget before a precise failing command or outcome is searched.
    # Field queries intentionally do not dedupe against the primary/each other:
    # their labels are part of the audit trail, and the reserve is only ten
    # slots (draft + ten fields = eleven) under the twelve-probe cap.
    field_names = ("title", "body", *QUERY_PROSE_FIELDS, *QUERY_CONTEXT_FIELDS)
    for field_name in field_names:
        value = _bounded_query(fields.get(field_name))
        if not value:
            continue
        add(field_name, value, SIMILARITY_FALLBACK_WEIGHT, dedupe=False)

    # Title first because it is the most useful identity signal.  Body and
    # diagnostics then provide recall for create-shaped bug drafts.  These are
    # surplus probes only; every populated field above already has a slot.
    for field_name in field_names:
        value = _bounded_query(fields.get(field_name))
        if not value:
            continue
        units = ranked.normalize_query_units(value)
        # Removing each term is intentionally bounded.  Three or more terms
        # still retain a meaningful lexical anchor; one/two-term fields would
        # become broad hard-negative probes and are left to the full field
        # query.  The global cap keeps latency/query work predictable.
        if len(units) >= 3:
            for index in range(len(units)):
                if len(probes) >= SIMILARITY_MAX_PROBES:
                    break
                add(
                    f"{field_name}-drop-{index}",
                    " ".join(units[:index] + units[index + 1 :]),
                    SIMILARITY_DROP_ONE_WEIGHT,
                )
    return probes


def _merge_similarity_results(
    responses: List[Tuple[str, str, float, Dict[str, Any]]],
    *,
    source_key: Optional[str],
    limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fuse probe responses by issue key, preserving honest lane facts."""
    merged: Dict[str, Dict[str, Any]] = {}
    for label, probe_query, weight, response in responses:
        for rank, result in enumerate(response.get("results", []), start=1):
            issue = result.get("issue") or {}
            key = str(issue.get("key") or "")
            if not key or (source_key and key.casefold() == source_key.casefold()):
                continue
            entry = merged.setdefault(
                key,
                {
                    "result": dict(result),
                    "score": 0.0,
                    "lanes": {},
                    "matched_fields": set(),
                    "snippets": {},
                    "exact_boosts": set(),
                    "neighborhood": {},
                    "duplicate_chain": {},
                    "probe_contributions": {},
                },
            )
            entry["score"] += weight / (SIMILARITY_RRF_K + rank)
            contribution_key = (label, probe_query, rank, result.get("rank_score"))
            entry["probe_contributions"][contribution_key] = {
                "label": label,
                "query": probe_query,
                "weight": weight,
                "original_rank": rank,
                "original_score": result.get("rank_score"),
            }
            for lane in result.get("contributing_lanes", []):
                lane_key = (lane.get("lane"), lane.get("rank"), lane.get("raw_score"))
                entry["lanes"][lane_key] = dict(lane)
            entry["matched_fields"].update(result.get("matched_fields", []))
            for field, snippet in (result.get("snippets") or {}).items():
                # Keep the first (higher-priority) probe's safe snippet.  The
                # probe order is deterministic and no snippet is fabricated.
                entry["snippets"].setdefault(field, snippet)
            entry["exact_boosts"].update(result.get("exact_boosts", []))
            for relation in result.get("neighborhood", []):
                rel_key = (relation.get("from_key"), relation.get("kind"), relation.get("to_key"))
                entry["neighborhood"][rel_key] = dict(relation)
            for chain in result.get("duplicate_chain", []):
                entry["duplicate_chain"][chain.get("canonical_key")] = dict(chain)

    def order(item: Tuple[str, Dict[str, Any]]) -> Tuple[float, str, str]:
        result = item[1]["result"]
        issue = result.get("issue") or {}
        # Similarity's local RRF score is the primary order.  Equal scores
        # follow the same freshness/key contract as ranked search.
        return (
            -item[1]["score"],
            ranked._desc_key(str(issue.get("updated_at") or "")),
            item[0],
        )

    ordered = sorted(merged.items(), key=order)
    output: List[Dict[str, Any]] = []
    for key, entry in ordered[:limit]:
        result = dict(entry["result"])
        result["rank_score"] = entry["score"]
        result["contributing_lanes"] = sorted(
            entry["lanes"].values(),
            key=lambda lane: (
                lane.get("rank", 0),
                lane.get("lane", ""),
                lane.get("raw_score", 0.0),
            ),
        )
        result["matched_fields"] = sorted(entry["matched_fields"])
        result["snippets"] = dict(sorted(entry["snippets"].items()))
        result["exact_boosts"] = sorted(entry["exact_boosts"])
        result["neighborhood"] = [
            entry["neighborhood"][key]
            for key in sorted(
                entry["neighborhood"], key=lambda value: tuple(str(part or "") for part in value)
            )
        ]
        result["duplicate_chain"] = [
            entry["duplicate_chain"][key]
            for key in sorted(entry["duplicate_chain"], key=lambda value: str(value or ""))
        ]
        result["probe_contributions"] = [
            entry["probe_contributions"][key]
            for key in sorted(
                entry["probe_contributions"],
                key=lambda value: (
                    str(value[0]),
                    str(value[1]),
                    int(value[2]),
                    str(value[3]),
                ),
            )
        ]
        output.append(result)
    return output, {"candidate_keys_seen": len(merged)}


def _expand_duplicate_chains(
    hit_keys: List[str],
    *,
    exclude_key: Optional[str],
    conflicts: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Confirmed duplicates of the returned hits, expanded one level.

    An issue whose ``duplicate_of`` names a hit is a confirmed duplicate of
    that hit regardless of its own status. A native ``duplicates`` link from
    the duplicate source to a hit is authoritative over a conflicting legacy
    field, so one issue never appears with two asserted canonicals. Multiple
    distinct native targets are malformed rather than a deterministic choice:
    no expansion is asserted, and callers may receive a typed conflict fact
    through ``conflicts``. Expansion is exactly one level — chains deeper than
    that belong to the issue graph, not to this advisory answer. The source
    issue never appears in its own expansion.
    """
    from cli_agent_orchestrator.services.issue_tracker import _issue_row

    if not hit_keys:
        return []
    canonical_by_folded = {str(key).casefold(): str(key) for key in hit_keys}
    with SessionLocal() as db:
        field_rows = (
            db.query(TrackerIssueModel).filter(TrackerIssueModel.duplicate_of.in_(hit_keys)).all()
        )
        hit_link_rows = (
            db.query(TrackerLinkModel)
            .filter(
                TrackerLinkModel.kind == "duplicates",
                TrackerLinkModel.to_key.in_(hit_keys),
            )
            .all()
        )
        duplicate_keys = {str(row.key) for row in field_rows}
        duplicate_keys.update(str(row.from_key) for row in hit_link_rows)
        issues = (
            db.query(TrackerIssueModel)
            .filter(TrackerIssueModel.key.in_(sorted(duplicate_keys)))
            .all()
        )
        issue_by_key = {str(row.key).casefold(): row for row in issues}
        # The first query finds duplicate sources related to a returned hit;
        # the second observes every native target for those sources.  Limiting
        # the latter to hit targets would hide a conflicting second target in
        # a one-hit query and make the selected canonical query-dependent.
        link_rows = (
            db.query(TrackerLinkModel)
            .filter(
                TrackerLinkModel.kind == "duplicates",
                TrackerLinkModel.from_key.in_(sorted(duplicate_keys)),
            )
            .all()
        )

    # Native directional links are the current relation vocabulary.  The
    # legacy ``duplicate_of`` field remains readable for old rows, but when a
    # duplicate carries both representations the native target is
    # authoritative; emitting both targets would assert two canonicals for
    # one issue. Multiple native targets are malformed, so report a conflict
    # and omit the expansion rather than asserting an arbitrary canonical.
    native_targets: Dict[str, List[str]] = {}
    native_source_names: Dict[str, List[str]] = {}
    for link in link_rows:
        duplicate_key = str(link.from_key)
        target_key = str(link.to_key)
        if duplicate_key.casefold() != target_key.casefold():
            native_targets.setdefault(duplicate_key.casefold(), []).append(target_key)
            native_source_names.setdefault(duplicate_key.casefold(), []).append(duplicate_key)

    pairs: Dict[Tuple[str, str], TrackerIssueModel] = {}
    for field_row in field_rows:
        duplicate_key = str(field_row.key)
        field_canonical = canonical_by_folded.get(str(field_row.duplicate_of or "").casefold())
        if not field_canonical or duplicate_key.casefold() == field_canonical.casefold():
            continue
        if duplicate_key.casefold() in native_targets:
            # See the native-link precedence rule above.
            continue
        pairs[(field_canonical.casefold(), duplicate_key.casefold())] = field_row
    for duplicate_folded in sorted(native_targets):
        canonicals = native_targets[duplicate_folded]
        canonical_by_target_folded: Dict[str, str] = {}
        for target_key in canonicals:
            target_folded = target_key.casefold()
            previous = canonical_by_target_folded.get(target_folded)
            if previous is None or (target_folded, target_key) < (target_folded, previous):
                canonical_by_target_folded[target_folded] = target_key
        canonical_keys = [
            canonical_by_target_folded[key] for key in sorted(canonical_by_target_folded)
        ]
        duplicate_key = sorted(
            set(native_source_names.get(duplicate_folded, [duplicate_folded])),
            key=lambda value: (value.casefold(), value),
        )[0]
        if len(canonical_keys) > 1:
            if conflicts is not None:
                conflicts.append(
                    {
                        "code": "multiple-native-duplicate-targets",
                        "message": "native duplicate source has multiple canonical targets",
                        "duplicate_key": duplicate_key,
                        "canonical_keys": canonical_keys,
                        "hit_canonical_keys": [
                            canonical_by_folded[key]
                            for key in sorted(canonical_by_target_folded)
                            if key in canonical_by_folded
                        ],
                    }
                )
            continue
        native_canonical = canonical_by_folded.get(canonical_keys[0].casefold())
        duplicate_row = issue_by_key.get(duplicate_folded)
        if (
            not native_canonical
            or duplicate_row is None
            or duplicate_key.casefold() == native_canonical.casefold()
        ):
            continue
        # A dual field/link representation is one expansion, never two.
        pairs[(native_canonical.casefold(), duplicate_key.casefold())] = duplicate_row

    expansions: List[Dict[str, Any]] = []
    for (canonical_folded, _), row in sorted(pairs.items()):
        if exclude_key and str(row.key).casefold() == str(exclude_key).casefold():
            continue
        canonical = canonical_by_folded[canonical_folded]
        expansions.append({"duplicate_of": canonical, "issue": _issue_row(row)})
    return expansions


def find_similar_issues(request: SimilarIssuesRequest) -> Dict[str, Any]:
    """Answer "what already exists that looks like this?" with explanations.

    The ranked-search envelope is narrowed, not reshaped: candidates keep the
    full per-result explanation objects, and the wrapper adds only what this
    surface owns — the query source, self-exclusion, and the one-level
    duplicate-chain expansion beside their hits.
    """
    from cli_agent_orchestrator.services.issue_tracker import TrackerError, get_issue

    _validate_request(request)

    source_key: Optional[str] = None
    fields: Dict[str, Any]
    if request.issue_key:
        row = get_issue(str(request.issue_key).strip())
        source_key = str(row["key"])
        fields = {
            "title": row.get("title"),
            "body": row.get("body"),
            **{name: row.get(name) for name in QUERY_PROSE_FIELDS + QUERY_CONTEXT_FIELDS},
        }
        effective_kind = str(row.get("kind") or DEFAULT_DRAFT_KIND)
    else:
        fields = dict(request.draft or {})
        effective_kind = str(fields.get("kind") or DEFAULT_DRAFT_KIND)

    query_text = _compose_query(fields)
    if not query_text:
        raise TrackerError(
            "invalid-query",
            "similarity requires at least one nonempty draft/issue text field",
        )
    # Self-exclusion runs after retrieval, so issue-key mode fetches one
    # extra candidate: when the source itself ranks within ``limit``, a
    # near-duplicate just outside it still survives exclusion.
    fetch_limit = min(request.limit + 1, ranked.MAX_LIMIT) if source_key else request.limit
    probes = _similarity_probes(fields, query_text)
    responses: List[Tuple[str, str, float, Dict[str, Any]]] = []
    failures: List[Dict[str, str]] = []
    for label, probe_query, weight in probes:
        try:
            response = ranked.ranked_search(
                ranked.RankedSearchRequest(
                    query=probe_query,
                    project_ids=request.project_ids,
                    all_projects=request.all_projects,
                    kinds=(effective_kind,),
                    include_comments=request.include_comments,
                    mode=request.mode,
                    limit=fetch_limit,
                )
            )
            responses.append((label, probe_query, weight, response))
        except (TrackerError, ranked.TrackerRankedSearchError) as exc:
            # A probe is advisory.  Keep useful results from sibling probes;
            # only an entirely unavailable search surface is a typed refusal.
            failures.append({"label": label, "code": exc.code, "message": exc.message})
    if not responses:
        if failures:
            failure = failures[0]
            raise TrackerError(failure["code"], failure["message"])
        raise TrackerError("invalid-query", "similarity produced no usable lexical probes")

    candidates, probe_facts = _merge_similarity_results(
        responses, source_key=source_key, limit=request.limit
    )

    primary_response = responses[0][3]
    reasons: List[str] = []
    lane_availability: Dict[str, Dict[str, Any]] = {}
    effective_modes: Set[str] = set()
    for _, _, _, response in responses:
        effective_modes.add(str(response.get("mode_effective") or request.mode))
        degradation = response.get("degradation") or {}
        for reason in degradation.get("reasons", []):
            if reason not in reasons:
                reasons.append(str(reason))
        for lane_name, availability in (degradation.get("lanes") or {}).items():
            existing = lane_availability.setdefault(lane_name, {"available": False})
            existing["available"] = bool(existing.get("available")) or bool(
                availability.get("available")
            )
            if availability.get("reason") and not existing.get("reason"):
                existing["reason"] = str(availability["reason"])

    for failure in failures:
        reason = f"similarity-probe-failed:{failure['label']}:{failure['code']}"
        if reason not in reasons:
            reasons.append(reason)

    semantic_requested = request.mode in ("semantic", "hybrid")
    mode_effective = request.mode if request.mode in effective_modes else "lexical"
    if mode_effective == "hybrid" and "hybrid" not in effective_modes:
        mode_effective = "lexical"
    if mode_effective == "semantic" and "semantic" not in effective_modes:
        mode_effective = "lexical"
    degraded = (
        bool(reasons) or bool(failures) or (semantic_requested and mode_effective == "lexical")
    )
    coverage = {
        "status": (
            "inconclusive"
            if degraded and not candidates
            else ("partial" if failures else ("degraded" if degraded else "complete"))
        ),
        "complete": not degraded,
        "inconclusive": degraded and not candidates,
        "probes_requested": len(probes),
        "probes_completed": len(responses),
        "probes_failed": len(failures),
        "partial": bool(failures),
        **probe_facts,
    }
    # Ranked-search timings and query-time refresh counters are intentionally
    # not copied: they are measurements of each individual request and would
    # make the API/CLI contract differ for otherwise identical probes.  Keep
    # stable semantic capability facts and add the deterministic probe audit.
    primary_diagnostics = primary_response.get("diagnostics") or {}
    diagnostics: Dict[str, Any] = {}
    if isinstance(primary_diagnostics.get("semantic"), dict):
        diagnostics["semantic"] = {
            key: primary_diagnostics["semantic"].get(key)
            for key in (
                "served",
                "generation_id",
                "issue_vectors_returned",
                "comment_issues_returned",
            )
        }
    diagnostics["similarity_probes"] = [
        {"label": label, "weight": weight, "query": probe_query}
        for label, probe_query, weight in probes
    ]
    diagnostics["similarity_probe_failures"] = failures

    hit_keys = [
        str((candidate.get("issue") or {}).get("key"))
        for candidate in candidates
        if (candidate.get("issue") or {}).get("key")
    ]
    duplicate_conflicts: List[Dict[str, Any]] = []
    expansions = _expand_duplicate_chains(
        hit_keys, exclude_key=source_key, conflicts=duplicate_conflicts
    )
    diagnostics["similarity_duplicate_conflicts"] = duplicate_conflicts

    return {
        "query_source": {
            "mode": "issue_key" if source_key else "draft",
            "issue_key": source_key,
            "kind": effective_kind,
        },
        "query": primary_response["query"],
        "scope": primary_response["scope"],
        "mode_requested": request.mode,
        "mode_effective": mode_effective,
        "degradation": {
            "requested_mode": request.mode,
            "effective_mode": mode_effective,
            "reasons": reasons,
            "lanes": lane_availability,
            "coverage": coverage,
        },
        "generations": primary_response.get("generations", {}),
        "diagnostics": diagnostics,
        "coverage": coverage,
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
