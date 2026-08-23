"""Ranked lexical issue search (design §10.3–§10.5, milestone M1.3).

Comments are documents; issues are results. This service retrieves bounded
candidates independently from three lexical lanes — field-weighted issue BM25,
comment BM25, and exact technical-string matching — aggregates comment hits at
issue level (best two retained, contribution capped, ``important`` boosted),
fuses lane ranks with weighted reciprocal-rank fusion (constant 60), and
returns a complete explanation for every result.

Structured filtering and subtree scoping are not reimplemented here: candidate
issues come from :mod:`services.tracker_filters`, the same builder ``issue
list`` uses, so the two surfaces cannot disagree about what a filter means.
Scope rules follow §10.1: exactly one of tracker-project ids or
``all_projects``, cycle-safe transitive ``part-of`` closure, and
cross-project descendants excluded by intersection.

Default input is literal free-form text. Every term is quoted before it
reaches FTS5, so shell commands, stack traces, paths, and stray operator words
(``AND``, ``OR``, ``NOT``, ``NEAR``, ``*``) can never act as query syntax.
Explicitly double-quoted segments are preserved as deliberate phrases.

The starting weights below are hypotheses (§10.3) recorded for tuning from the
relevance fixture only — they are deliberately not policy. Semantic lanes and
the ``hybrid`` default arrive at M2; requests asking for them degrade visibly
to lexical through the response's degradation metadata rather than silently.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import text as _sql_text

from cli_agent_orchestrator.clients import tracker_search_schema
from cli_agent_orchestrator.clients.database import (
    SessionLocal,
    TrackerIssueModel,
    TrackerLinkModel,
)
from cli_agent_orchestrator.services.tracker_filters import (
    ScopeResolution,
    StructuredFilters,
    is_effectively_empty_query,
    resolve_scope,
)


def _exec(executor: Any, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Run one raw SQL statement through SQLAlchemy's text() binding."""
    return executor.execute(_sql_text(sql), params or {})


#: Weighted RRF constant (design §10.4: "a documented initial constant such
#: as 60"). Higher flattens rank differences between lanes.
RRF_K = 60

#: Lane weights for fusion — hypotheses; tune only from the relevance fixture.
LANE_WEIGHT_ISSUE_BM25 = 1.0
LANE_WEIGHT_COMMENT_BM25 = 0.75
LANE_WEIGHT_EXACT = 1.1

#: Bounded additive boost when the whole query equals an issue key or the
#: exact stored failing command (§10.4 "bounded deterministic boost").
EXACT_FINGERPRINT_BOOST = 0.05

#: Issue-document BM25 column weights, hypotheses per §10.3, keyed by FTS
#: column name. Order-insensitive here; the SQL layer applies them in the
#: canonical column order and a test pins that correspondence.
ISSUE_FIELD_WEIGHTS: Dict[str, float] = {
    "key_text": 1.0,
    "title": 5.0,
    "failing_command": 4.0,
    "actual_outcome": 4.0,
    "expected_outcome": 2.0,
    "reproduction_steps": 3.0,
    "component": 2.0,
    "labels_text": 2.0,
    "kind": 0.5,
    "status": 0.5,
    "severity": 0.5,
    "reporter": 0.75,
    "assignee": 0.75,
    "collaborators_text": 0.75,
    "branches_text": 0.5,
    "worktrees_text": 0.5,
    "pull_requests_text": 0.5,
    "session_name": 0.5,
    "terminal_id": 0.5,
    "source_path": 0.5,
    "duplicate_of": 0.5,
    "origin": 0.5,
    "body": 1.0,
    "evidence": 1.0,
    "resolution": 2.0,
    "observed_revision": 2.0,
}

#: Comment-document BM25 column weights (author low, body carries the signal).
COMMENT_FIELD_WEIGHTS: Dict[str, float] = {"author": 0.5, "body": 1.0}

#: Best-N comment hits per issue retained during aggregation (§10.4 cap):
#: many progress comments cannot outweigh one precise document.
MAX_COMMENT_HITS_PER_ISSUE = 2

#: Additive adjustment applied to a comment's BM25 score (lower is better)
#: during aggregation: importance boosts, ordinariness does not.
IMPORTANT_COMMENT_BONUS = -0.25

#: Per-lane retrieval caps — hypotheses like the weights, not policy.
ISSUE_LANE_CANDIDATE_CAP = 250
COMMENT_LANE_CANDIDATE_CAP = 400
EXACT_LANE_ROW_CAP = 500

#: Request bounds (§10.1 request-size bound; §10.5 pagination bounds).
MAX_QUERY_CHARS = 1000
MAX_QUERY_UNITS = 64
MIN_LIMIT = 1
MAX_LIMIT = 100
DEFAULT_LIMIT = 20

#: Priority order for exact-lane matches: identity and reproducers outrank
#: prose. Lower wins.
EXACT_FIELD_PRIORITY: Dict[str, int] = {
    "key": 0,
    "failing_command": 1,
    "actual_outcome": 2,
    "reproduction_steps": 3,
    "expected_outcome": 4,
    "observed_revision": 5,
    "evidence": 6,
    "resolution": 7,
    "component": 8,
    "comment_body": 9,
    "title": 10,
    "body": 11,
}

#: Issue fields the exact lane searches and explanations may name.
EXACT_SEARCH_FIELDS: Tuple[str, ...] = (
    "key",
    "title",
    "failing_command",
    "actual_outcome",
    "expected_outcome",
    "reproduction_steps",
    "evidence",
    "resolution",
    "component",
    "observed_revision",
    "body",
)

_SQLITE_PARAM_CHUNK = 500
_SNIPPET_WINDOW = 60
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_QUOTED_SEGMENT_RE = re.compile(r'"([^"]*)"')
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


class TrackerRankedSearchError(RuntimeError):
    """A ranked-search refusal carrying the tracker's typed error shape.

    Mirrors ``issue_tracker.TrackerError``'s ``(code, message, details)``
    contract so the API/CLI lane (M1.4a) maps both classes through one
    refusal renderer instead of learning a second dialect.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class RankedSearchRequest:
    """One ranked-search request (§10.1/§10.2/§10.5).

    Exactly one of ``project_ids`` / ``all_projects`` must be supplied.
    ``mode`` accepts ``lexical`` (effective today) plus ``semantic``/
    ``hybrid``, which degrade visibly until M2 installs the vector lanes.
    """

    query: str
    project_ids: Tuple[str, ...] = ()
    all_projects: bool = False
    subtree_roots: Tuple[str, ...] = ()
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
    include_comments: bool = True
    mode: str = "lexical"
    limit: int = DEFAULT_LIMIT
    offset: int = 0


# ---------------------------------------------------------------------------
# Literal-safe query handling (§10.5)
# ---------------------------------------------------------------------------


def normalize_query_units(query: str) -> List[str]:
    """Split free-form text into literal units, preserving deliberate phrases.

    Double-quoted segments survive verbatim as single units; the remaining
    text splits on whitespace. Units carrying no alphanumeric character are
    dropped — after tokenization they match nothing, and quoting cannot save
    them. A query reduced to nothing by this normalization is rejected as
    empty by the caller, matching the documented nonempty-normalized-text rule.
    """
    text = (query or "").strip()
    units: List[str] = []

    def push(candidate: str) -> None:
        cleaned = candidate.strip()
        if cleaned and not _PUNCT_ONLY_RE.match(cleaned.replace('"', "")):
            units.append(cleaned)

    pos = 0
    for match in _QUOTED_SEGMENT_RE.finditer(text):
        for token in text[pos : match.start()].split():
            push(token)
        push(match.group(1))
        pos = match.end()
    for token in text[pos:].split():
        push(token)
    if len(units) > MAX_QUERY_UNITS:
        raise TrackerRankedSearchError(
            "invalid", f"query has too many terms (max {MAX_QUERY_UNITS})"
        )
    return units


def build_fts_match_query(query: str) -> str:
    """Render normalized units as an all-literal FTS5 MATCH expression.

    Every unit becomes a double-quoted phrase (implicit AND between units), so
    punctuation-heavy input — flags, paths, stack frames — reaches the index
    as text, never as query grammar (§10.5).
    """
    units = normalize_query_units(query)
    if not units:
        raise TrackerRankedSearchError(
            "invalid-query",
            "ranked search requires nonempty normalized free-form text; "
            "empty-text browsing belongs to issue list",
        )
    return " ".join('"' + unit.replace('"', '""') + '"' for unit in units)


def _snippet(text: str, terms: Sequence[str]) -> str:
    """A safe plain-text window around the first term occurrence."""
    body = text or ""
    lowered = body.lower()
    cut: Optional[int] = None
    for term in terms:
        needle = term.lower()
        if not needle:
            continue
        found = lowered.find(needle)
        if found >= 0 and (cut is None or found < cut):
            cut = found
    if cut is None:
        cut = 0
    start = max(0, cut - 20)
    end = min(len(body), start + _SNIPPET_WINDOW * 2)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(body) else ""
    return f"{prefix}{body[start:end]}{suffix}"


def _query_terms(units: Sequence[str]) -> List[str]:
    """Lowercased word list used for advisory snippet windows."""
    words: List[str] = []
    for unit in units:
        words.extend(word.lower() for word in _WORD_RE.findall(unit))
    return list(dict.fromkeys(words))


def _tracker_error(code: str, message: str):
    from cli_agent_orchestrator.services.issue_tracker import TrackerError

    return TrackerError(code, message)


# ---------------------------------------------------------------------------
# Candidate-key constraints shared by every lane
# ---------------------------------------------------------------------------


def _chunked_in_sql(
    alias_column: str, keys: Sequence[str], prefix: str
) -> Tuple[str, Dict[str, Any]]:
    """``alias.column IN (...)`` chunked under SQLite's host-parameter limit."""
    params: Dict[str, Any] = {}
    if not keys:
        return " AND 1 = 0", params
    groups: List[str] = []
    for i in range(0, len(keys), _SQLITE_PARAM_CHUNK):
        chunk = keys[i : i + _SQLITE_PARAM_CHUNK]
        placeholders = []
        for j, key in enumerate(chunk):
            name = f"{prefix}{i}_{j}"
            placeholders.append(f":{name}")
            params[name] = key
        groups.append(f"{alias_column} IN ({', '.join(placeholders)})")
    return " AND (" + " OR ".join(groups) + ")", params


# ---------------------------------------------------------------------------
# Lane execution — each returns an ordered list of (issue_key, raw_score)
# ---------------------------------------------------------------------------


def _fts_declared_columns(table: str) -> List[Tuple[str, bool]]:
    """All declared FTS columns in order as ``(name, indexed)``.

    bm25() weight arguments map to every declared column left-to-right,
    UNINDEXED ones included (SQLite assigns 1.0 when arguments run out), so
    the weight list must cover the full declaration, never just the indexed
    subset.
    """
    ddl = tracker_search_schema._FTS_TABLES[table]
    body = ddl.split("(", 1)[1].rsplit(")", 1)[0]
    parts: List[str] = []
    depth = 0
    current = ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    columns: List[Tuple[str, bool]] = []
    for part in parts:
        stripped = part.strip()
        if not stripped or stripped.upper().startswith("TOKENIZE"):
            continue
        tokens = stripped.split()
        columns.append((tokens[0], not any(t.upper() == "UNINDEXED" for t in tokens[1:])))
    return columns


def _bm25_weight_args(table: str) -> str:
    """Literal BM25 weight list covering every declared column in order.

    Unlisted columns (the UNINDEXED identity/context ones) take SQLite's
    default of 1.0; an indexed column without a configured weight is a
    configuration error rather than a silent misweighting.
    """
    weights = (
        ISSUE_FIELD_WEIGHTS
        if table == tracker_search_schema.ISSUE_FTS_TABLE
        else COMMENT_FIELD_WEIGHTS
    )
    declared = _fts_declared_columns(table)
    missing = [name for name, indexed in declared if indexed and name not in weights]
    if missing:
        raise TrackerRankedSearchError(
            "configuration",
            f"missing BM25 weight(s) for {table} column(s): {', '.join(missing)}",
        )
    rendered = ", ".join(repr(float(weights.get(name, 1.0))) for name, _ in declared)
    return ", " + rendered


def run_issue_bm25_lane(
    conn: Any, match_expr: str, candidate_sql: str, candidate_params: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """Field-weighted issue BM25 (lane 1): best matches first, lower=better."""
    fts = tracker_search_schema.ISSUE_FTS_TABLE
    sql = (
        f"SELECT c.issue_key, bm25({fts}{_bm25_weight_args(fts)}) AS score\n"
        f"FROM {fts} AS c\n"
        f"WHERE {fts} MATCH :match{candidate_sql}\n"
        "ORDER BY score, c.issue_key\n"
        f"LIMIT {ISSUE_LANE_CANDIDATE_CAP}"
    )
    rows = _exec(conn, sql, {"match": match_expr, **candidate_params}).fetchall()
    return [(str(row[0]), float(row[1])) for row in rows]


def run_comment_bm25_lane(
    conn: Any,
    match_expr: str,
    candidate_sql: str,
    candidate_params: Dict[str, Any],
    *,
    include_comments: bool,
) -> Tuple[List[Tuple[str, float]], Dict[str, Dict[str, Any]]]:
    """Comment BM25 (lane 2) aggregated at issue level (§10.4).

    Comment hits group by issue; at most ``MAX_COMMENT_HITS_PER_ISSUE``
    contribute, importance boosts a hit's score, and the issue-level rank is
    its best retained hit. Returns the ranking plus winning-comment facts for
    explanations.
    """
    if not include_comments:
        return [], {}
    fts = tracker_search_schema.COMMENT_FTS_TABLE
    sql = (
        "SELECT c.comment_id, c.issue_key, c.body, src.important,\n"
        f"       bm25({fts}{_bm25_weight_args(fts)}) AS score\n"
        f"FROM {fts} AS c\n"
        "JOIN tracker_issue_comments AS src ON src.id = c.comment_id\n"
        f"WHERE {fts} MATCH :match{candidate_sql}\n"
        "ORDER BY score, c.comment_id\n"
        f"LIMIT {COMMENT_LANE_CANDIDATE_CAP}"
    )
    best_per_issue: Dict[str, List[Tuple[float, int, str, bool]]] = {}
    total_matching: Dict[str, int] = {}
    for comment_id, issue_key, body, important, score in _exec(
        conn, sql, {"match": match_expr, **candidate_params}
    ).fetchall():
        key = str(issue_key)
        total_matching[key] = total_matching.get(key, 0) + 1
        adjusted = float(score) + (IMPORTANT_COMMENT_BONUS if int(important) else 0.0)
        best_per_issue.setdefault(key, []).append(
            (adjusted, int(comment_id), str(body or ""), bool(int(important)))
        )
    ranked: List[Tuple[str, float]] = []
    winning: Dict[str, Dict[str, Any]] = {}
    for issue_key, hits in best_per_issue.items():
        hits.sort(key=lambda hit: (hit[0], hit[1]))
        retained = hits[:MAX_COMMENT_HITS_PER_ISSUE]
        best = retained[0]
        ranked.append((issue_key, best[0]))
        winning[issue_key] = {
            "comment_id": best[1],
            "body": best[2],
            "important": best[3],
            "retained_hits": len(retained),
            "additional_comment_ids": [hit[1] for hit in retained[1:]],
            # Scoped to this lane's retrieval window: when COMMENT_LANE_CANDIDATE_CAP
            # truncates a very hot corpus, this counts hits seen, not all hits.
            "total_matching_comments": total_matching[issue_key],
        }
    ranked.sort(key=lambda pair: (pair[1], pair[0]))
    return ranked, winning


def run_exact_lane(
    conn: Any, raw_query: str, resolution: ScopeResolution
) -> Tuple[List[Tuple[str, float]], Dict[str, Dict[str, Any]]]:
    """Exact/substring lane (§10.3 lane 3) for strings tokenization mishandles.

    Escaped-LIKE matching across identity, reproducer, and prose fields plus
    comment bodies. Per issue, the representing match is the earliest field
    priority, ties broken by shorter matched text (a tight fingerprint beats a
    passing mention); comment matches use -length so a longer body wins as the
    richer window at equal priority. Issues order by that priority, then
    newest update, then key — matched length never orders issues against each
    other. Raw scores encode negated priority so RRF sees a stable lane
    ordering while diagnostics stay legible.
    """
    needle = "%" + raw_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    conditions = " OR ".join(f"s.{name} LIKE :needle ESCAPE '\\'" for name in EXACT_SEARCH_FIELDS)
    issue_sql_params: Dict[str, Any] = {"needle": needle}
    candidate_sql = ""
    if resolution.allowed_keys is not None:
        fragment, params = _chunked_in_sql("s.key", sorted(resolution.allowed_keys), "ex")
        candidate_sql = fragment
        issue_sql_params.update(params)
    sql = (
        f"SELECT s.key, s.updated_at, {', '.join('s.' + f for f in EXACT_SEARCH_FIELDS)}\n"
        "FROM tracker_issues AS s\n"
        f"WHERE ({conditions}){candidate_sql}\n"
        "ORDER BY s.key\n"
        f"LIMIT {EXACT_LANE_ROW_CAP}"
    )
    # Per issue: best (field priority, tighter matched length) wins; smaller
    # tuples rank better. Comment matches use -len so a longer body — a
    # richer window — beats a shorter one at equal priority.
    best: Dict[str, Tuple[int, int]] = {}
    facts: Dict[str, Dict[str, Any]] = {}
    updated_by_key: Dict[str, Optional[str]] = {}
    for row in _exec(conn, sql, issue_sql_params).fetchall():
        key = str(row[0])
        updated = row[1]
        if updated is None:
            updated_by_key[key] = None
        elif hasattr(updated, "isoformat"):
            updated_by_key[key] = updated.isoformat()
        else:
            updated_by_key[key] = str(updated)
        for offset, field_name in enumerate(EXACT_SEARCH_FIELDS, start=2):
            value = row[offset]
            if value is None:
                continue
            text_value = str(value)
            if raw_query.lower() not in text_value.lower():
                continue
            candidate = (EXACT_FIELD_PRIORITY[field_name], len(text_value))
            if key not in best or candidate < best[key]:
                best[key] = candidate
                facts[key] = {
                    "matched_field": field_name,
                    "snippet": _snippet(text_value, [raw_query]),
                }

    comment_sql_params: Dict[str, Any] = {"needle": needle}
    comment_candidate = ""
    if resolution.allowed_keys is not None:
        fragment, params = _chunked_in_sql("c.issue_key", sorted(resolution.allowed_keys), "ec")
        comment_candidate = fragment
        comment_sql_params.update(params)
    comment_sql = (
        "SELECT c.issue_key, c.body FROM tracker_issue_comments AS c\n"
        f"WHERE c.body LIKE :needle ESCAPE '\\'{comment_candidate}\n"
        "ORDER BY c.issue_key, c.id\n"
        f"LIMIT {EXACT_LANE_ROW_CAP}"
    )
    for issue_key, body in _exec(conn, comment_sql, comment_sql_params).fetchall():
        key = str(issue_key)
        text_value = str(body or "")
        if raw_query.lower() not in text_value.lower():
            continue
        candidate = (EXACT_FIELD_PRIORITY["comment_body"], -len(text_value))
        if key not in best or candidate < best[key]:
            best[key] = candidate
            facts[key] = {
                "matched_field": "comment_body",
                "snippet": _snippet(text_value, [raw_query]),
            }

    ordered_entries = sorted(
        best.items(),
        key=lambda item: (item[1][0], _desc_key(updated_by_key.get(item[0]) or ""), item[0]),
    )
    ranked = [(key, float(-best[key][0])) for key, _ in ordered_entries]
    return ranked, facts


def _desc_key(iso: str) -> str:
    """A key sorting NEWER strings first when the outer sort is ascending.

    Digits keep full timestamp precision (including fractional seconds), so
    sub-second ``updated_at`` differences still order before the key falls
    through, per the §10.5 tie-break.
    """
    if not iso:
        return _DESC_MAX
    digits = re.sub(r"[^0-9]", "", iso)[:20].zfill(20)
    # 10^20-1 minus the digits: bigger timestamps become smaller keys.
    return str(int(_DESC_MAX) - int(digits)).zfill(20)


_DESC_MAX = str(10**20 - 1)


# ---------------------------------------------------------------------------
# Fusion and service entry point
# ---------------------------------------------------------------------------


_LANE_WEIGHTS = {
    "issue-bm25": LANE_WEIGHT_ISSUE_BM25,
    "comment-bm25": LANE_WEIGHT_COMMENT_BM25,
    "exact": LANE_WEIGHT_EXACT,
}


def _fuse_lane_ranks(lanes: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Dict[str, Any]]:
    """Weighted RRF over per-lane issue rankings (§10.4).

    Each lane contributes ``weight / (k + rank)`` for the issues it returned.
    Raw scores ride along as diagnostics only; they are never added together.
    """
    fused: Dict[str, Dict[str, Any]] = {}
    for lane_name, ranking in lanes.items():
        weight = _LANE_WEIGHTS[lane_name]
        for rank_zero_based, (issue_key, raw_score) in enumerate(ranking):
            rank = rank_zero_based + 1
            entry = fused.setdefault(issue_key, {"score": 0.0, "lanes": []})
            entry["score"] += weight / (RRF_K + rank)
            entry["lanes"].append({"lane": lane_name, "rank": rank, "raw_score": raw_score})
    return fused


def _meta_snapshot(conn: Any) -> Dict[str, Any]:
    row = _exec(
        conn,
        f"SELECT schema_version, document_schema_version, content_clock, "
        f"active_vector_generation, rebuilt_at "
        f"FROM {tracker_search_schema.SEARCH_META_TABLE} WHERE singleton = 1",
    ).fetchone()
    if row is None:
        raise TrackerRankedSearchError(
            "search-unavailable",
            "the derived search projection is not installed; run the lexical rebuild verb",
        )
    return {
        "schema_version": int(row[0]),
        "document_schema_version": int(row[1]),
        "content_clock": int(row[2]),
        "active_vector_generation": row[3],
        "rebuilt_at": row[4],
    }


def ranked_search(request: RankedSearchRequest) -> Dict[str, Any]:
    """Execute one ranked lexical search request and return explained results."""
    from cli_agent_orchestrator.services.issue_tracker import TrackerError, _issue_row

    started = time.perf_counter()
    try:
        _validate_request(request)
        raw_query = request.query.strip()
        match_expr = build_fts_match_query(raw_query)
    except TrackerRankedSearchError as exc:
        raise TrackerError(exc.code, exc.message) from exc

    reasons: List[str] = []
    effective_mode = "lexical"
    if request.mode in ("semantic", "hybrid"):
        reasons.append(
            f"requested mode {request.mode!r}: semantic lanes install at milestone M2; "
            "degraded visibly to lexical"
        )

    filters = StructuredFilters(
        kinds=request.kinds,
        statuses=request.statuses,
        severities=request.severities,
        components=request.components,
        observed_revisions=request.observed_revisions,
        labels=request.labels,
        without_labels=request.without_labels,
        assignee=request.assignee,
        reporter=request.reporter,
        open_only=request.open_only,
        unlabeled=request.unlabeled,
    ).validated()

    units = normalize_query_units(raw_query)
    terms = _query_terms(units)

    with SessionLocal() as db:
        try:
            resolution = resolve_scope(
                db,
                project_ids=request.project_ids,
                all_projects=request.all_projects,
                subtree_roots=request.subtree_roots,
            )
        except TrackerError as exc:
            if exc.code == "invalid":
                raise TrackerError("invalid-scope", exc.message) from exc
            raise
        conn = db.connection()
        meta = _meta_snapshot(conn)

        # Authoritative structured filters + scope intersect into one
        # candidate-key set through the shared builder's predicates.
        candidate_q = db.query(TrackerIssueModel.key)
        for condition in filters.orm_conditions():
            candidate_q = candidate_q.filter(condition)
        if resolution.allowed_keys is not None:
            allowed = sorted(resolution.allowed_keys)
            if allowed:
                candidate_q = candidate_q.filter(
                    TrackerIssueModel.key.in_(allowed)
                    if len(allowed) <= _SQLITE_PARAM_CHUNK
                    else _or_chunks(allowed)
                )
            else:
                candidate_q = candidate_q.filter(sqlalchemy_false())
        candidate_keys = frozenset(str(row[0]) for row in candidate_q.all())
        scoped = ScopeResolution(
            project_ids=resolution.project_ids,
            all_projects=resolution.all_projects,
            subtree_roots=resolution.subtree_roots,
            closure_keys=resolution.closure_keys,
            allowed_keys=candidate_keys,
        )

        candidate_sql, candidate_params = (
            _chunked_in_sql("c.issue_key", sorted(candidate_keys), "ck")
            if candidate_keys
            else (" AND 1 = 0", {})
        )

        lane_timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        issue_ranking = run_issue_bm25_lane(conn, match_expr, candidate_sql, candidate_params)
        lane_timings["issue-bm25"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        comment_ranking, winning_comments = run_comment_bm25_lane(
            conn,
            match_expr,
            candidate_sql,
            candidate_params,
            include_comments=request.include_comments,
        )
        lane_timings["comment-bm25"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        exact_ranking, exact_facts = run_exact_lane(conn, raw_query, scoped)
        lane_timings["exact"] = (time.perf_counter() - t0) * 1000.0

        fused = _fuse_lane_ranks(
            {
                "issue-bm25": issue_ranking,
                "comment-bm25": comment_ranking,
                "exact": exact_ranking,
            }
        )

        boosted = _apply_exact_fingerprint_boosts(conn, raw_query, fused)

        rows_by_key: Dict[str, TrackerIssueModel] = {}
        if fused:
            rows = (
                db.query(TrackerIssueModel)
                .filter(
                    TrackerIssueModel.key.in_(list(fused))
                    if len(fused) <= _SQLITE_PARAM_CHUNK
                    else _or_chunks(sorted(fused))
                )
                .all()
            )
            rows_by_key = {str(row.key): row for row in rows}

        def result_order(item: Tuple[str, Dict[str, Any]]) -> Tuple[float, str, str]:
            row = rows_by_key.get(item[0])
            updated = row.updated_at.isoformat() if row is not None and row.updated_at else ""
            return (-item[1]["score"], _desc_key(updated), item[0])

        ordered = sorted(fused.items(), key=result_order)
        total = len(ordered)
        page = ordered[request.offset : request.offset + request.limit]

        results: List[Dict[str, Any]] = [
            _build_explanation(
                db,
                issue_key,
                payload,
                rows_by_key.get(issue_key),
                exact_facts,
                winning_comments,
                boosted,
                terms,
                raw_query,
            )
            for issue_key, payload in page
        ]

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "query": raw_query,
            "scope": {
                "project_ids": list(resolution.project_ids),
                "all_projects": resolution.all_projects,
                "subtree_roots": list(resolution.subtree_roots),
                "subtree_closure_size": len(resolution.closure_keys),
            },
            "mode_requested": request.mode,
            "mode_effective": effective_mode,
            "degradation": {
                "requested_mode": request.mode,
                "effective_mode": effective_mode,
                "reasons": reasons,
                "lanes": {
                    "issue-bm25": {"available": True},
                    "comment-bm25": {"available": True},
                    "exact": {"available": True},
                    "semantic-issue": {"available": False, "reason": "installs at M2"},
                    "semantic-comment": {"available": False, "reason": "installs at M2"},
                },
            },
            "generations": dict(meta),
            "diagnostics": {
                "lane_elapsed_ms": {k: round(v, 3) for k, v in lane_timings.items()},
                "total_elapsed_ms": round(elapsed_ms, 3),
            },
            "total": total,
            "limit": request.limit,
            "offset": request.offset,
            "results": results,
        }


def _validate_request(request: RankedSearchRequest) -> None:
    if request.limit < MIN_LIMIT or request.limit > MAX_LIMIT:
        raise TrackerRankedSearchError(
            "invalid", f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}"
        )
    if request.offset < 0:
        raise TrackerRankedSearchError("invalid", "offset must be nonnegative")
    if request.mode not in ("lexical", "semantic", "hybrid"):
        raise TrackerRankedSearchError("invalid", f"unknown search mode {request.mode!r}")
    raw_query = (request.query or "").strip()
    if len(raw_query) > MAX_QUERY_CHARS:
        raise TrackerRankedSearchError(
            "invalid", f"query too long (max {MAX_QUERY_CHARS} characters)"
        )
    if is_effectively_empty_query(raw_query):
        raise TrackerRankedSearchError(
            "invalid-query",
            "ranked search requires nonempty normalized free-form text; "
            "empty-text browsing belongs to issue list",
        )


def _apply_exact_fingerprint_boosts(
    conn: Any, raw_query: str, fused: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    """Bounded deterministic boosts for whole-query equality fingerprints."""
    boosted: Dict[str, List[str]] = {}
    casefolded = raw_query.casefold()
    for issue_key, payload in fused.items():
        if issue_key.casefold() == casefolded:
            payload["score"] += EXACT_FINGERPRINT_BOOST
            boosted.setdefault(issue_key, []).append("issue-key-equality")
    if fused:
        fragment, params = _chunked_in_sql("key", sorted(fused), "bki")
        rows = _exec(
            conn,
            "SELECT key, failing_command FROM tracker_issues "
            f"WHERE failing_command IS NOT NULL{fragment}",
            params,
        ).fetchall()
        for key, command in rows:
            issue_key = str(key)
            marks = boosted.get(issue_key, [])
            if issue_key in fused and str(command).strip().casefold() == casefolded:
                payload = fused[issue_key]
                payload["score"] += EXACT_FINGERPRINT_BOOST
                if "issue-key-equality" not in marks:
                    boosted.setdefault(issue_key, []).append("failing-command-equality")
    return boosted


def _build_explanation(
    db: Any,
    issue_key: str,
    payload: Dict[str, Any],
    row: Optional[TrackerIssueModel],
    exact_facts: Dict[str, Dict[str, Any]],
    winning_comments: Dict[str, Dict[str, Any]],
    boosted: Dict[str, List[str]],
    terms: Sequence[str],
    raw_query: str,
) -> Dict[str, Any]:
    """Assemble the complete §10.5 explanation for one result."""
    from cli_agent_orchestrator.services.issue_tracker import _issue_row

    explanation: Dict[str, Any] = {
        "issue": _issue_row(row) if row is not None else None,
        "rank_score": round(payload["score"], 9),
        "contributing_lanes": sorted(payload["lanes"], key=lambda l: (l["rank"], l["lane"])),
        "matched_fields": [],
        "snippets": {},
        "winning_comment": None,
        "exact_boosts": boosted.get(issue_key, []),
        "neighborhood": [],
        "duplicate_chain": [],
    }
    matched_fields: Set[str] = set()
    snippets: Dict[str, str] = {}
    fact = exact_facts.get(issue_key)
    if fact:
        matched_fields.add(fact["matched_field"])
        snippets[fact["matched_field"]] = fact["snippet"]
    win = winning_comments.get(issue_key)
    if win:
        matched_fields.add("comments")
        if "comments" not in snippets:
            snippets["comments"] = _snippet(win.get("body", ""), terms)
        explanation["winning_comment"] = {
            "comment_id": win["comment_id"],
            "important": win["important"],
            "retained_hits": win["retained_hits"],
            "additional_comment_ids": win["additional_comment_ids"],
            "total_matching_comments": win["total_matching_comments"],
        }
    if row is not None:
        for field_name in EXACT_SEARCH_FIELDS:
            if field_name == "key" or field_name in matched_fields:
                continue
            value = getattr(row, field_name, None)
            text_value = str(value) if value is not None else ""
            if text_value and all(term in text_value.lower() for term in terms):
                matched_fields.add(field_name)
                snippets[field_name] = _snippet(text_value, terms)
    explanation["matched_fields"] = sorted(matched_fields)
    explanation["snippets"] = snippets
    if row is not None and row.duplicate_of:
        canon = (
            db.query(TrackerIssueModel).filter(TrackerIssueModel.key == row.duplicate_of).first()
        )
        explanation["duplicate_chain"] = [
            {
                "canonical_key": row.duplicate_of,
                "canonical_title": canon.title if canon is not None else None,
                "resolved": canon is not None,
            }
        ]
    links = (
        db.query(TrackerLinkModel)
        .filter((TrackerLinkModel.from_key == issue_key) | (TrackerLinkModel.to_key == issue_key))
        .all()
    )
    explanation["neighborhood"] = [
        {"from_key": link.from_key, "to_key": link.to_key, "kind": link.kind} for link in links
    ]
    return explanation


def sqlalchemy_false() -> Any:
    from sqlalchemy import false

    return false()


def _or_chunks(keys: Sequence[str]) -> Any:
    """An OR of chunked ``key IN (...)`` clauses usable as an ORM filter."""
    from sqlalchemy import or_

    clauses = []
    for i in range(0, len(keys), _SQLITE_PARAM_CHUNK):
        clauses.append(TrackerIssueModel.key.in_(keys[i : i + _SQLITE_PARAM_CHUNK]))
    return or_(*clauses) if clauses else sqlalchemy_false()


def search_status() -> Dict[str, Any]:
    """Lightweight §10.5 diagnostics: availability, generations, backlog.

    Reads through one raw DBAPI connection so the derived-schema helpers run
    against the driver directly, mirroring the rebuild/report verbs.
    """
    with SessionLocal() as db:
        bind: Any = db.get_bind()
        raw = bind.raw_connection()
        try:
            installed = raw.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name IN "
                f"('{tracker_search_schema.SEARCH_META_TABLE}', "
                f"'{tracker_search_schema.ISSUE_FTS_TABLE}', "
                f"'{tracker_search_schema.COMMENT_FTS_TABLE}')"
            ).fetchone()[0]
            if int(installed) < 3:
                return {"installed": False}
            meta_row = raw.execute(
                f"SELECT schema_version, document_schema_version, content_clock, "
                f"active_vector_generation, rebuilt_at "
                f"FROM {tracker_search_schema.SEARCH_META_TABLE} WHERE singleton = 1"
            ).fetchone()
            if meta_row is None:
                return {"installed": False}
            meta = {
                "schema_version": int(meta_row[0]),
                "document_schema_version": int(meta_row[1]),
                "content_clock": int(meta_row[2]),
                "active_vector_generation": meta_row[3],
                "rebuilt_at": meta_row[4],
            }
            dirty_total = raw.execute(
                f"SELECT COUNT(*) FROM {tracker_search_schema.VECTOR_DIRTY_TABLE}"
            ).fetchone()[0]
            dirty_failed = raw.execute(
                f"SELECT COUNT(*) FROM {tracker_search_schema.VECTOR_DIRTY_TABLE} "
                "WHERE last_error IS NOT NULL"
            ).fetchone()[0]
            vectors = raw.execute(
                f"SELECT COUNT(*) FROM {tracker_search_schema.SEARCH_VECTORS_TABLE}"
            ).fetchone()[0]
            return {
                "installed": True,
                **meta,
                "lexical": {
                    "issue_documents_missing": (
                        tracker_search_schema.count_missing_documents(raw, "issue")
                    ),
                    "comment_documents_missing": (
                        tracker_search_schema.count_missing_documents(raw, "comment")
                    ),
                },
                "semantic": {
                    "active_generation": meta["active_vector_generation"],
                    "dirty_documents": int(dirty_total),
                    "failed_documents": int(dirty_failed),
                    "vectors": int(vectors),
                },
            }
        finally:
            raw.close()


__all__ = [
    "RankedSearchRequest",
    "TrackerRankedSearchError",
    "build_fts_match_query",
    "normalize_query_units",
    "ranked_search",
    "search_status",
]
