# ABOUTME: Relevance metrics over fixture cases: recall@k, MRR, hard-negative
# ABOUTME: load, plus null-safe semantic coverage and informational timing.
"""Metrics for the issue-search fixture.

Deterministic relevance metrics (recall@k, MRR, hard-negative load, semantic
coverage) are computed from ranks only and are byte-stable across reruns from
a fixed snapshot. Wall-clock performance (latency, cold start) is reported in
a separate ``performance`` block and is explicitly excluded from gate
comparison — a timing number can never make a run RED.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from .retrievers import RankedIssue
from .snapshot import TERMINAL_STATUSES


@dataclass(frozen=True)
class CaseExpectation:
    primary: tuple[str, ...]
    acceptable: tuple[str, ...]
    hard_negatives: tuple[str, ...]

    @property
    def expected_set(self) -> frozenset[str]:
        return frozenset(self.primary) | frozenset(self.acceptable)


@dataclass
class CaseResult:
    case_id: str
    case_class: str
    num_expected: int
    first_hit_rank: int | None  # first primary-or-acceptable hit (1-based)
    first_primary_rank: int | None
    hits_at_5: int
    hits_at_10: int
    hard_negatives_at_5: int
    hard_negatives_at_10: int
    hard_negative_above_first_hit: bool
    open_terminal_mix: tuple[int, int] | None = None  # (open, terminal) in top-10
    latency_ms: float | None = None


@dataclass
class LaneMetrics:
    lane: str
    cases: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    mrr_primary: float
    hard_negative_load_at_5: int
    hard_negative_load_at_10: int
    hard_negative_case_rate: float
    hard_negative_above_first_hit_rate: float
    semantic_coverage: float | None = None
    per_case: list[CaseResult] = field(default_factory=list)
    performance: dict[str, float | str] = field(default_factory=dict)


def _first_rank(ranked: Sequence[RankedIssue], keys: frozenset[str]) -> int | None:
    for item in ranked:
        if item.key in keys:
            return item.rank
    return None


def evaluate_case(
    case_id: str,
    case_class: str,
    expectation: CaseExpectation,
    ranked: Sequence[RankedIssue],
    issue_status: dict[str, str],
    latency_ms: float | None = None,
    wants_status_mix: bool = False,
) -> CaseResult:
    """Score one fixture case against one ranked result list."""

    keys_at = {k: [item.key for item in ranked[:k]] for k in (5, 10)}
    expected = expectation.expected_set
    negatives = frozenset(expectation.hard_negatives)
    first_hit = _first_rank(ranked, expected)
    first_primary = _first_rank(ranked, frozenset(expectation.primary))
    mix: tuple[int, int] | None = None
    if wants_status_mix:
        top10_statuses = [
            issue_status[k] for k in keys_at[10] if k in expected and k in issue_status
        ]
        mix = (
            sum(1 for s in top10_statuses if s not in TERMINAL_STATUSES),
            sum(1 for s in top10_statuses if s in TERMINAL_STATUSES),
        )
    return CaseResult(
        case_id=case_id,
        case_class=case_class,
        num_expected=len(expected),
        first_hit_rank=first_hit,
        first_primary_rank=first_primary,
        hits_at_5=sum(1 for k in keys_at[5] if k in expected),
        hits_at_10=sum(1 for k in keys_at[10] if k in expected),
        hard_negatives_at_5=sum(1 for k in keys_at[5] if k in negatives),
        hard_negatives_at_10=sum(1 for k in keys_at[10] if k in negatives),
        hard_negative_above_first_hit=(
            first_hit is not None
            and any(item.key in negatives and item.rank < first_hit for item in ranked)
        ),
        open_terminal_mix=mix,
        latency_ms=latency_ms,
    )


def aggregate(lane: str, results: Sequence[CaseResult]) -> LaneMetrics:
    """Fold per-case results into lane metrics (micro-averaged)."""

    total_expected = sum(r.num_expected for r in results)
    total_hits_5 = sum(r.hits_at_5 for r in results)
    total_hits_10 = sum(r.hits_at_10 for r in results)
    mrr = sum(1.0 / r.first_hit_rank for r in results if r.first_hit_rank is not None) / len(
        results
    )
    mrr_primary = sum(
        1.0 / r.first_primary_rank for r in results if r.first_primary_rank is not None
    ) / len(results)
    load_5 = sum(r.hard_negatives_at_5 for r in results)
    load_10 = sum(r.hard_negatives_at_10 for r in results)
    polluted = sum(1 for r in results if r.hard_negatives_at_5 > 0 or r.hard_negatives_at_10 > 0)
    above = sum(1 for r in results if r.hard_negative_above_first_hit)
    return LaneMetrics(
        lane=lane,
        cases=len(results),
        recall_at_5=total_hits_5 / total_expected if total_expected else 0.0,
        recall_at_10=total_hits_10 / total_expected if total_expected else 0.0,
        mrr=mrr,
        mrr_primary=mrr_primary,
        hard_negative_load_at_5=load_5,
        hard_negative_load_at_10=load_10,
        hard_negative_case_rate=polluted / len(results) if results else 0.0,
        hard_negative_above_first_hit_rate=above / len(results) if results else 0.0,
        per_case=list(results),
    )


def summarize_performance(latencies_ms: Sequence[float], cold_start_ms: float) -> dict:
    """Wall-clock block: informational only, never gated."""

    if not latencies_ms:
        return {"cold_start_ms": cold_start_ms, "median_ms": None, "p95_ms": None}
    ordered = sorted(latencies_ms)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "cold_start_ms": round(cold_start_ms, 3),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
    }
