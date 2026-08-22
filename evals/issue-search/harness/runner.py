# ABOUTME: Deterministic runner: executes fixture cases through each lane and
# ABOUTME: emits the report the gate checks.
"""Run the fixture corpus through every available lane and report metrics.

The runner validates the fixture against the snapshot before measuring: every
referenced issue key must exist, duplicate-pair labels must be confirmed
relations in the snapshot, and verbatim/comment-derived queries must actually
occur in the fixed corpus. A fixture that references data the snapshot does
not contain is a broken measurement, not a low score.

Output splits into a deterministic ``metrics`` block (ranks only; byte-stable
across reruns) and an informational ``performance`` block (wall clock).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .metrics import CaseResult, aggregate, evaluate_case, summarize_performance
from .retrievers import (
    EmptyLaneRetriever,
    RankedIssue,
    Retriever,
    Scope,
    build_default_lanes,
)
from .snapshot import Snapshot, load_fixture, load_snapshot

REPORT_SCHEMA_VERSION = 1

# Fixture case classes that carry special semantics.
STATUS_MIX_CLASSES = frozenset({"status-mix"})
COMMENT_CLASSES = frozenset({"comment-only"})
PARAPHRASE_CLASSES = frozenset({"nl-paraphrase"})


class FixtureError(ValueError):
    """The fixture references data the snapshot cannot support."""


def _corpus_text(snapshot: Snapshot) -> str:
    parts: list[str] = []
    for issue in snapshot.issues:
        parts.extend(issue.searchable_fields().values())
        parts.append(issue.resolution or "")
    for comment in snapshot.comments:
        parts.append(comment.body)
    return "\n".join(parts).lower()


def validate_fixture(fixture: dict[str, Any], snapshot: Snapshot) -> None:
    """Fail closed on a fixture the snapshot cannot support."""

    if fixture.get("snapshot_id") != snapshot.snapshot_id:
        raise FixtureError(
            f"fixture pins snapshot {fixture.get('snapshot_id')!r} but loaded "
            f"{snapshot.snapshot_id!r}"
        )
    known_keys = {issue.key for issue in snapshot.issues}
    known_projects = {issue.project_id for issue in snapshot.issues}
    duplicate_of = {
        issue.key: issue.duplicate_of for issue in snapshot.issues if issue.duplicate_of
    }
    corpus = _corpus_text(snapshot)
    for case in fixture["cases"]:
        for key in (
            case["expected"]["primary"]
            + case["expected"].get("acceptable", [])
            + case["expected"].get("hard_negatives", [])
        ):
            if key not in known_keys:
                raise FixtureError(f"case {case['id']}: unknown issue key {key!r}")
        scope = case.get("scope", {})
        projects = scope.get("tracker_projects", [])
        has_all = scope.get("all_projects", False)
        if bool(projects) == bool(has_all):
            raise FixtureError(f"case {case['id']}: exactly one scope form required")
        for project in projects:
            if project not in known_projects:
                raise FixtureError(f"case {case['id']}: unknown tracker project {project!r}")
        for root in scope.get("subtree_roots", []):
            if root not in known_keys:
                raise FixtureError(f"case {case['id']}: unknown subtree root {root!r}")
        if case["class"] == "duplicate-pair":
            dup_sources = [
                k
                for k in case["expected"].get("acceptable", []) + case["expected"]["primary"]
                if k in duplicate_of
            ]
            if not dup_sources:
                raise FixtureError(
                    f"case {case['id']}: duplicate-pair case declares no "
                    "confirmed duplicate relation"
                )
            for dup_key in dup_sources:
                target = duplicate_of[dup_key]
                if target not in known_keys:
                    raise FixtureError(
                        f"case {case['id']}: duplicate {dup_key} points at "
                        f"missing canonical {target}"
                    )
        derivation = case.get("provenance", {}).get("derivation", "verbatim")
        if derivation == "verbatim":
            if case["query"].lower() not in corpus:
                raise FixtureError(
                    f"case {case['id']}: verbatim query does not occur in the " "snapshot corpus"
                )
        elif derivation == "comment":
            comment_text = "\n".join(c.body for c in snapshot.comments).lower()
            if case["query"].lower() not in comment_text:
                raise FixtureError(
                    f"case {case['id']}: comment-derived query does not occur "
                    "in snapshot comments"
                )


def _parse_scope(case: dict[str, Any]) -> Scope:
    scope = case.get("scope", {})
    projects = tuple(scope.get("tracker_projects", []))
    roots = tuple(scope.get("subtree_roots", []))
    all_projects = bool(scope.get("all_projects", False))
    if not projects and not all_projects:
        # Default dashboard behavior scopes to the active project; fixture
        # cases declare scope explicitly, so absence here means cao-system.
        projects = ("cao-system",)
    return Scope(
        tracker_projects=projects,
        subtree_roots=roots,
        all_projects=all_projects,
    )


def run_fixture(
    snapshot: Snapshot,
    fixture: dict[str, Any],
    inject: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Execute every case through every lane; returns the full report.

    ``inject`` simulates a relevance regression for counterfactual-sensitivity
    verification:

    - ``"promote-noise"`` floods every result list with deterministic
      irrelevant issues ahead of real matches — the observable symptom of a
      ranking regression (wrong field weights, broken fusion).
    - ``"empty"`` drops the lane's results entirely.

    Both preserve the inner lane's name so the gate compares the degraded run
    against that lane's recorded baseline.
    """

    validate_fixture(fixture, snapshot)
    issue_status = {issue.key: issue.status for issue in snapshot.issues}
    noise_pool = [
        RankedIssue(key=issue.key, rank=0, lane="noise")
        for issue in sorted(snapshot.issues, key=lambda i: i.seq, reverse=True)
    ]
    lanes: list[Retriever] = build_default_lanes(snapshot)
    if inject == "empty":
        lanes = [EmptyLaneRetriever(lane) for lane in lanes]
    elif inject is not None and inject != "promote-noise":
        raise ValueError(f"unknown injection: {inject!r}")

    lane_reports: dict[str, Any] = {}
    semantic_present = any(lane.name.startswith("semantic") for lane in lanes)
    for lane in lanes:
        per_case: list[CaseResult] = []
        latencies: list[float] = []
        cold_start_ms = 0.0
        first_call_seen = False
        for case in fixture["cases"]:
            expectation = case_expectation(case)
            scope = _parse_scope(case)
            started = time.perf_counter()
            ranked = lane.search(case["query"], scope=scope, limit=limit)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not first_call_seen:
                cold_start_ms = elapsed_ms
                first_call_seen = True
            if inject == "promote-noise":
                ranked = _promote_noise(ranked, expectation, noise_pool)
            latencies.append(elapsed_ms)
            per_case.append(
                evaluate_case(
                    case_id=case["id"],
                    case_class=case["class"],
                    expectation=expectation,
                    ranked=ranked,
                    issue_status=issue_status,
                    latency_ms=elapsed_ms,
                    wants_status_mix=case["class"] in STATUS_MIX_CLASSES,
                )
            )
        lane_metrics = aggregate(lane.name, per_case)
        lane_metrics.semantic_coverage = 1.0 if semantic_present else None
        lane_metrics.performance = summarize_performance(latencies, cold_start_ms)
        lane_reports[lane.name] = lane_to_dict(lane_metrics)

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "fixture_version": fixture["fixture_version"],
        "snapshot_id": snapshot.snapshot_id,
        "injection": inject,
        "semantic_lane_installed": semantic_present,
        "lanes": lane_reports,
    }


NOISE_COUNT = 15


def _promote_noise(
    ranked: list[RankedIssue],
    expectation: Any,
    noise_pool: list[RankedIssue],
) -> list[RankedIssue]:
    """Prepend irrelevant issues ahead of the ranked results.

    The case's declared hard negatives come first (the ranker now prefers the
    confusables — the failure mode the hard-negative metric exists to catch),
    then deterministic high-seq issues fill the remaining noise slots.
    """

    noise_keys: list[str] = list(expectation.hard_negatives)
    used: set[str] = set(noise_keys) | {item.key for item in ranked}
    used |= set(expectation.expected_set)
    for item in noise_pool:
        if len(noise_keys) >= NOISE_COUNT:
            break
        if item.key in used:
            continue
        used.add(item.key)
        noise_keys.append(item.key)
    noise = [RankedIssue(key=key, rank=0, lane="noise") for key in noise_keys]
    reordered = noise + list(ranked)
    return [
        RankedIssue(key=item.key, rank=i + 1, lane=item.lane) for i, item in enumerate(reordered)
    ]


def case_expectation(case: dict[str, Any]) -> Any:
    from .metrics import CaseExpectation

    expected = case["expected"]
    return CaseExpectation(
        primary=tuple(expected["primary"]),
        acceptable=tuple(expected.get("acceptable", [])),
        hard_negatives=tuple(expected.get("hard_negatives", [])),
    )


def lane_to_dict(lane_metrics: Any) -> dict[str, Any]:
    """Deterministic serialization of one lane's metrics."""

    return {
        "metrics": {
            "cases": lane_metrics.cases,
            "recall_at_5": round(lane_metrics.recall_at_5, 6),
            "recall_at_10": round(lane_metrics.recall_at_10, 6),
            "mrr": round(lane_metrics.mrr, 6),
            "mrr_primary": round(lane_metrics.mrr_primary, 6),
            "hard_negative_load_at_5": lane_metrics.hard_negative_load_at_5,
            "hard_negative_load_at_10": lane_metrics.hard_negative_load_at_10,
            "hard_negative_case_rate": round(lane_metrics.hard_negative_case_rate, 6),
            "hard_negative_above_first_hit_rate": round(
                lane_metrics.hard_negative_above_first_hit_rate, 6
            ),
            "semantic_coverage": lane_metrics.semantic_coverage,
        },
        "performance": lane_metrics.performance,
        "per_case": [
            {
                "case_id": r.case_id,
                "case_class": r.case_class,
                "num_expected": r.num_expected,
                "first_hit_rank": r.first_hit_rank,
                "first_primary_rank": r.first_primary_rank,
                "hits_at_5": r.hits_at_5,
                "hits_at_10": r.hits_at_10,
                "hard_negatives_at_5": r.hard_negatives_at_5,
                "hard_negatives_at_10": r.hard_negatives_at_10,
                "hard_negative_above_first_hit": r.hard_negative_above_first_hit,
                "open_terminal_mix": r.open_terminal_mix,
            }
            for r in sorted(lane_metrics.per_case, key=lambda r: r.case_id)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent.parent
    parser.add_argument("--snapshots", type=Path, default=here / "snapshots")
    parser.add_argument("--fixture", type=Path, default=here / "fixtures" / "corpus.v1.json")
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--inject", choices=["promote-noise", "empty"], default=None)
    args = parser.parse_args(argv)

    snapshot = load_snapshot(args.snapshots, args.snapshot_id)
    fixture = load_fixture(args.fixture)
    report = run_fixture(snapshot, fixture, inject=args.inject)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
