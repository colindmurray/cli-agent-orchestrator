# ABOUTME: Retrieval lanes for the issue-search harness: legacy substring
# ABOUTME: baseline plus the FTS and semantic interfaces later lanes install.
"""Retrievers measured by the harness.

``LegacySubstringRetriever`` is a faithful mirror of the live tracker's
``list_issues`` free-text path (``services/issue_tracker.py``): a single
``ILIKE '%<query>%'`` needle across title, body, key, failing_command,
reproduction_steps, expected_outcome, actual_outcome, and evidence, ordered by
``created_at DESC, id DESC``. SQL ``LIKE`` wildcard semantics are reproduced
exactly — including ``%``/``_`` inside a query acting as wildcards — because
those quirks are part of what the baseline measures.

``FtsRetriever`` and ``SemanticRetriever`` are the versioned interfaces lanes
E (FTS5 BM25) and M2 (semantic documents) implement; until installed they
report themselves unavailable and the runner stays null-safe.

Scope (tracker projects, subtree roots) is request semantics shared by every
lane, applied identically so lane comparisons isolate ranking quality.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .snapshot import Issue, Snapshot


@dataclass(frozen=True)
class RankedIssue:
    """One retrieval result with explanation fields (design §10.5)."""

    key: str
    rank: int
    lane: str
    matched_fields: tuple[str, ...] = ()
    raw_score: float | None = None


@dataclass(frozen=True)
class Scope:
    """Exactly one scope form per ranked request (design §10.1)."""

    tracker_projects: tuple[str, ...] = ()
    subtree_roots: tuple[str, ...] = ()
    all_projects: bool = False

    def __post_init__(self) -> None:
        if bool(self.tracker_projects) == self.all_projects:
            raise ValueError("exactly one of tracker_projects or all_projects is required")


def _like_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a SQL LIKE pattern (as the live query builds it) to a regex.

    ``%`` matches any sequence including newlines (DOTALL). Case-insensitive
    matching mirrors SQLite's ASCII-only folding; fixture queries are ASCII so
    the two agree on everything the corpus exercises.
    """

    parts: list[str] = []
    for ch in pattern:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    return re.compile("".join(parts), re.DOTALL | re.IGNORECASE)


class Retriever:
    """Base class; subclasses implement :meth:`_rank_unscoped`."""

    name = "base"

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot

    def search(
        self,
        query: str,
        scope: Scope,
        limit: int = 100,
    ) -> list[RankedIssue]:
        """Return ranked issue keys for the query under the request scope."""

        if not query or not query.strip():
            raise ValueError("nonempty normalized free-form text is required")
        candidates = self._scoped_candidates(scope)
        ranked = self._rank_unscoped(query.strip(), candidates)
        return [
            RankedIssue(key=key, rank=i + 1, lane=self.name, **extra)
            for i, (key, extra) in enumerate(ranked[:limit])
        ]

    def _scoped_candidates(self, scope: Scope) -> list[Issue]:
        issues = list(self.snapshot.issues)
        if not scope.all_projects:
            wanted = set(scope.tracker_projects)
            issues = [i for i in issues if i.project_id in wanted]
        if scope.subtree_roots:
            closure = self.snapshot.subtree_closure(list(scope.subtree_roots))
            issues = [i for i in issues if i.key in closure]
        return issues

    def _rank_unscoped(self, query: str, candidates: list[Issue]) -> list[tuple[str, dict]]:
        """Rank candidates for the stripped query; newest-first tie-break."""

        raise NotImplementedError


class LegacySubstringRetriever(Retriever):
    """The live ``%substring%`` ILIKE search, mirrored field-for-field."""

    name = "legacy-substring"

    def _rank_unscoped(self, query: str, candidates: list[Issue]) -> list[tuple[str, dict]]:
        needle = f"%{query}%"
        regex = _like_to_regex(needle)
        hits: list[tuple[Issue, tuple[str, ...]]] = []
        for issue in candidates:
            matched = tuple(
                field
                for field, text in issue.searchable_fields().items()
                if text and regex.fullmatch(text)
            )
            if matched:
                hits.append((issue, matched))
        # Live ordering: created_at DESC, id DESC (issue_tracker._apply_order).
        hits.sort(key=lambda pair: (pair[0].created_at or "", pair[0].seq), reverse=True)
        return [
            (issue.key, {"matched_fields": matched, "raw_score": None}) for issue, matched in hits
        ]


class FtsRetriever(Retriever):
    """Field-weighted FTS5 BM25 lane — installed by lane E (design §10.3)."""

    name = "fts-bm25"

    def __init__(self, snapshot: Snapshot) -> None:
        raise NotImplementedError(
            "FtsRetriever is the lane-E interface; no FTS index exists at " "fixture revision M0.1"
        )


class SemanticRetriever(Retriever):
    """Semantic document lane — installed at M2 (design §9/§10.3)."""

    name = "semantic"

    def __init__(self, snapshot: Snapshot) -> None:
        raise NotImplementedError(
            "SemanticRetriever is the M2 interface; no embeddings exist at " "fixture revision M0.1"
        )


class EmptyLaneRetriever(Retriever):
    """Counterfactual: simulates the wrapped lane returning nothing."""

    def __init__(self, inner: Retriever) -> None:  # noqa: D107
        self.inner = inner
        self.name = inner.name
        self.snapshot = inner.snapshot

    def _rank_unscoped(self, query: str, candidates: list[Issue]) -> list[tuple[str, dict]]:
        return []


class EmptyLaneRetriever(Retriever):
    """Counterfactual: simulates the wrapped lane returning nothing."""

    def __init__(self, inner: Retriever) -> None:  # noqa: D107
        self.inner = inner
        self.name = inner.name
        self.snapshot = inner.snapshot

    def _rank_unscoped(self, query: str, candidates: list[Issue]) -> list[tuple[str, dict]]:
        return []


def build_default_lanes(snapshot: Snapshot) -> list[Retriever]:
    """Lanes available at this fixture revision: the substring baseline only."""

    return [LegacySubstringRetriever(snapshot)]


def available_lane_names() -> list[str]:
    return ["legacy-substring", "fts-bm25", "semantic"]


__all__ = [
    "Scope",
    "RankedIssue",
    "LegacySubstringRetriever",
    "FtsRetriever",
    "SemanticRetriever",
    "ReverseOrderRetriever",
    "EmptyLaneRetriever",
    "build_default_lanes",
    "available_lane_names",
]
