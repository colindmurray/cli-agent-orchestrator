# ABOUTME: Metric math unit tests on synthetic ranked lists.
# ABOUTME: Pins recall@k, MRR, and hard-negative load definitions exactly.
"""Metric definition tests."""

from __future__ import annotations

from harness.metrics import CaseExpectation, evaluate_case
from harness.retrievers import RankedIssue


def _ranked(*keys: str) -> list[RankedIssue]:
    return [RankedIssue(key=k, rank=i + 1, lane="test") for i, k in enumerate(keys)]


def test_recall_counts_expected_hits_in_top_k() -> None:
    expectation = CaseExpectation(primary=("a",), acceptable=("b", "c"), hard_negatives=())
    result = evaluate_case("t", "test", expectation, _ranked("x", "a", "b", "c", "d"), {})
    # a at 2, b at 3, c at 4 -> hits@5 = 3, hits@10 = 3 of 3 expected.
    assert result.hits_at_5 == 3
    assert result.hits_at_10 == 3
    assert result.first_hit_rank == 2


def test_results_beyond_top_k_do_not_count() -> None:
    expectation = CaseExpectation(primary=("a", "b"), acceptable=(), hard_negatives=())
    result = evaluate_case(
        "t", "test", expectation, _ranked("x1", "x2", "x3", "x4", "x5", "a", "b"), {}
    )
    assert result.hits_at_5 == 0
    assert result.hits_at_10 == 2
    assert result.first_hit_rank == 6


def test_hard_negative_above_first_hit_detected() -> None:
    expectation = CaseExpectation(primary=("a",), acceptable=(), hard_negatives=("n",))
    result = evaluate_case("t", "test", expectation, _ranked("n", "a"), {})
    assert result.hard_negative_above_first_hit is True
    assert result.hard_negatives_at_5 == 1
    clean = evaluate_case("t", "test", expectation, _ranked("a", "n"), {})
    assert clean.hard_negative_above_first_hit is False
    assert clean.hard_negatives_at_5 == 1


def test_status_mix_counts_only_expected_candidates() -> None:
    expectation = CaseExpectation(primary=("open1", "closed1"), acceptable=(), hard_negatives=())
    statuses = {"open1": "open", "closed1": "closed", "noise": "open"}
    result = evaluate_case(
        "t",
        "status-mix",
        expectation,
        _ranked("open1", "noise", "closed1"),
        statuses,
        wants_status_mix=True,
    )
    assert result.open_terminal_mix == (1, 1)
