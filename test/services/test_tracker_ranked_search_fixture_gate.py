"""The M0.1 relevance harness run against the ranked lexical service (§16.1).

The committed fixture corpus is materialized into a real tracker store —
issues, comments, and part-of links inserted in export order, then the M1.2
search-projection migration builds the FTS documents — and every fixture case
runs through :func:`tracker_ranked_search.ranked_search`. Rank-derived metrics
are computed with the harness's own metric code and gated against the
recorded legacy substring baseline with the same tolerances ``harness.gate``
applies: the service must not regress retrieval quality that main already
delivers. Wall-clock latency is collected informationally, never gated.

``evals/**`` is used strictly read-only here, per the lane's working rules.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import (
    _TRACKER_ORM_TABLE_NAMES,
    Base,
    TrackerCommentModel,
    TrackerIssueModel,
    TrackerLinkModel,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.services import tracker_ranked_search as rsearch
from cli_agent_orchestrator.services.tracker_ranked_search import RankedSearchRequest

EVALS_DIR = Path(__file__).resolve().parents[2] / "evals" / "issue-search"
FIXTURE_PATH = EVALS_DIR / "fixtures" / "corpus.v1.json"
BASELINE_PATH = EVALS_DIR / "baselines" / "legacy-substring.json"

SERVICE_LANE = "ranked-lexical"
TOLERANCE = 0.02

if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from harness.metrics import CaseExpectation, aggregate, evaluate_case  # noqa: E402
from harness.retrievers import RankedIssue  # noqa: E402
from harness.snapshot import load_fixture, load_snapshot  # noqa: E402


def _parse_ts(value):
    """Snapshot timestamps are naive ISO strings; the ORM column wants datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _materialize_snapshot(db_path: Path):
    """Rebuild the snapshot corpus as a live store with the projection."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(
        bind=engine,
        tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
    )
    _migrate_tracker_search_projection(engine)

    snapshot = load_snapshot(EVALS_DIR / "snapshots")
    session = sessionmaker(bind=engine)()
    for issue in snapshot.issues:
        session.add(
            TrackerIssueModel(
                id=issue.seq,
                key=issue.key,
                project_id=issue.project_id,
                title=issue.title,
                body=issue.body,
                status=issue.status,
                severity=issue.severity,
                component=issue.component,
                failing_command=issue.failing_command,
                evidence=issue.evidence,
                resolution=issue.resolution,
                reproduction_steps=issue.reproduction_steps,
                expected_outcome=issue.expected_outcome,
                actual_outcome=issue.actual_outcome,
                created_at=_parse_ts(issue.created_at),
                updated_at=_parse_ts(issue.updated_at),
                labels=json.dumps(issue.labels),
                duplicate_of=issue.duplicate_of,
            )
        )
    for comment in snapshot.comments:
        session.add(
            TrackerCommentModel(
                id=comment.id,
                issue_key=comment.issue_key,
                author=comment.author,
                body=comment.body,
                created_at=_parse_ts(comment.created_at),
            )
        )
    seen_pairs = set()
    for parent, children in snapshot.part_of_children.items():
        for child in children:
            pair = (child, parent)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                session.add(TrackerLinkModel(from_key=child, to_key=parent, kind="part-of"))
    session.commit()
    session.close()
    return engine, snapshot


@pytest.fixture(scope="module")
def corpus_store(tmp_path_factory):
    engine, snapshot = _materialize_snapshot(tmp_path_factory.mktemp("corpus") / "corpus.db")
    yield {"engine": engine, "snapshot": snapshot}
    engine.dispose()


def _request_for(case, mode, limit=100):
    """One fixture case as a service request, with the shared scope default."""
    scope = case.get("scope", {})
    projects = tuple(scope.get("tracker_projects") or ())
    all_projects = bool(scope.get("all_projects", False))
    if not projects and not all_projects:
        projects = ("cao-system",)
    return RankedSearchRequest(
        query=case["query"],
        project_ids=projects,
        all_projects=all_projects,
        subtree_roots=tuple(scope.get("subtree_roots") or ()),
        limit=limit,
        mode=mode,
    )


def _run_case(engine, case, mode="lexical"):
    """Run one fixture case through the service; return (RankedIssue list, ms)."""
    with_session = sessionmaker(bind=engine)
    original = rsearch.SessionLocal
    rsearch.SessionLocal = with_session
    try:
        started = time.perf_counter()
        response = rsearch.ranked_search(_request_for(case, mode=mode))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        rsearch.SessionLocal = original
    ranked = [
        RankedIssue(
            key=result["issue"]["key"],
            rank=index + 1,
            lane=SERVICE_LANE,
            matched_fields=tuple(result["matched_fields"]),
        )
        for index, result in enumerate(response["results"])
    ]
    return ranked, elapsed_ms


class TestHarnessGateAgainstRankedService:
    def test_service_does_not_regress_the_legacy_baseline(self, corpus_store):
        snapshot = corpus_store["snapshot"]
        engine = corpus_store["engine"]
        fixture = load_fixture(FIXTURE_PATH)

        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        base_metrics = baseline["lanes"]["legacy-substring"]["metrics"]

        issue_status = {issue.key: issue.status for issue in snapshot.issues}
        results = []
        latencies = []
        for case in fixture["cases"]:
            expected = case["expected"]
            ranked, elapsed_ms = _run_case(engine, case)
            latencies.append(elapsed_ms)
            results.append(
                evaluate_case(
                    case_id=case["id"],
                    case_class=case["class"],
                    expectation=CaseExpectation(
                        primary=tuple(expected["primary"]),
                        acceptable=tuple(expected.get("acceptable", [])),
                        hard_negatives=tuple(expected.get("hard_negatives", [])),
                    ),
                    ranked=ranked,
                    issue_status=issue_status,
                    latency_ms=elapsed_ms,
                )
            )

        lane_metrics = aggregate(SERVICE_LANE, results)
        ran = {
            "recall_at_5": lane_metrics.recall_at_5,
            "recall_at_10": lane_metrics.recall_at_10,
            "mrr": lane_metrics.mrr,
            "mrr_primary": lane_metrics.mrr_primary,
            "hard_negative_load_at_5": lane_metrics.hard_negative_load_at_5,
            "hard_negative_load_at_10": lane_metrics.hard_negative_load_at_10,
            "hard_negative_case_rate": lane_metrics.hard_negative_case_rate,
            "hard_negative_above_first_hit_rate": (lane_metrics.hard_negative_above_first_hit_rate),
        }

        summary = {"metrics": {k: round(v, 6) for k, v in ran.items()}}
        for metric in ("recall_at_5", "recall_at_10", "mrr", "mrr_primary"):
            assert ran[metric] >= base_metrics[metric] - TOLERANCE, (
                f"{metric} regressed against the recorded legacy baseline: "
                f"{base_metrics[metric]} -> {ran[metric]}"
            )
        for metric in ("hard_negative_case_rate", "hard_negative_above_first_hit_rate"):
            assert ran[metric] <= base_metrics[metric] + TOLERANCE, (
                f"{metric} worsened against the recorded legacy baseline: "
                f"{base_metrics[metric]} -> {ran[metric]}"
            )
        for metric in ("hard_negative_load_at_5", "hard_negative_load_at_10"):
            allowed = int(base_metrics[metric]) + TOLERANCE * max(
                1, int(base_metrics.get("cases", 1))
            )
            assert ran[metric] <= allowed, (
                f"{metric} rose above the legacy floor: " f"{base_metrics[metric]} -> {ran[metric]}"
            )

        ordered = sorted(latencies)
        p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
        performance = {
            "median_ms": round(ordered[len(ordered) // 2], 3),
            "p95_ms": round(ordered[p95_index], 3),
        }
        print(f"\n{SERVICE_LANE} metrics:", json.dumps(summary["metrics"], indent=1))
        print("performance_ms:", json.dumps(performance))

    def test_service_beats_the_legacy_baseline_on_recall(self, corpus_store):
        """The gate above only forbids regression; lexical ranking must win."""
        snapshot = corpus_store["snapshot"]
        engine = corpus_store["engine"]
        fixture = load_fixture(FIXTURE_PATH)

        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        base_metrics = baseline["lanes"]["legacy-substring"]["metrics"]

        issue_status = {issue.key: issue.status for issue in snapshot.issues}
        results = []
        for case in fixture["cases"]:
            expected = case["expected"]
            ranked, _ = _run_case(engine, case)
            results.append(
                evaluate_case(
                    case_id=case["id"],
                    case_class=case["class"],
                    expectation=CaseExpectation(
                        primary=tuple(expected["primary"]),
                        acceptable=tuple(expected.get("acceptable", [])),
                        hard_negatives=tuple(expected.get("hard_negatives", [])),
                    ),
                    ranked=ranked,
                    issue_status=issue_status,
                )
            )
        lane_metrics = aggregate(SERVICE_LANE, results)
        assert lane_metrics.mrr > base_metrics["mrr"], (
            "ranked fusion should beat the substring baseline's MRR "
            f"({base_metrics['mrr']}), got {lane_metrics.mrr}"
        )


HYBRID_LANE = "ranked-hybrid"
_EXACT_CLASSES = frozenset(
    {"exact-command", "exact-symbol", "exact-error", "exact-path"}
)


class TestHybridVersusLexicalHarnessComparison:
    """§16.1 gate: the fixture corpus through hybrid mode, recorded against
    lexical-only.

    This class REPORTS; it does not gate the ranking comparison and does not
    flip any default — whether hybrid becomes the default is an operator
    decision taken from the recorded numbers. The assertions are structural:
    both modes ran every case, and the semantic legs genuinely served.

    Requires a prepared embedding model (the suite's scratch state root has
    none): run with ``CAO_SEARCH_MODELS_DIR=<prepared dir>`` pointing at the
    operator's prepared model, following the offline-drill convention.
    """

    @pytest.fixture(scope="class")
    def hybrid_corpus(self, corpus_store):
        """The lexical corpus plus one active generation over the prepared model."""
        pytest.importorskip("sentence_transformers")
        pytest.importorskip("sqlite_vec")
        from cli_agent_orchestrator.services.embedding_adapter import (
            default_models_dir,
            load_embedder,
            read_metadata,
        )
        from cli_agent_orchestrator.services.vector_lifecycle import (
            activate_generation,
            create_generation,
            refresh_generation,
        )

        models_dir = default_models_dir()
        metadata = read_metadata(models_dir)
        if metadata is None:
            pytest.skip(
                f"no prepared generation at {models_dir}; run `cao issue search-index "
                "model prepare` first (or point CAO_SEARCH_MODELS_DIR at one)"
            )
        engine = corpus_store["engine"]
        db_path = str(engine.url.database)
        created = create_generation(metadata=metadata, target_engine=engine)
        refresh_generation(
            generation_id=created["generation_id"],
            embedder=load_embedder(models_dir),
            db_path=db_path,
        )
        activate_generation(created["generation_id"], target_engine=engine)
        return {"engine": engine, "generation_id": created["generation_id"]}

    @staticmethod
    def _run_all(engine, mode):
        snapshot = load_snapshot(EVALS_DIR / "snapshots")
        fixture = load_fixture(FIXTURE_PATH)
        issue_status = {issue.key: issue.status for issue in snapshot.issues}
        results = []
        served = 0
        for case in fixture["cases"]:
            ranked, elapsed_ms = _run_case(engine, case, mode=mode)
            results.append(
                evaluate_case(
                    case_id=case["id"],
                    case_class=case["class"],
                    expectation=CaseExpectation(
                        primary=tuple(case["expected"]["primary"]),
                        acceptable=tuple(case["expected"].get("acceptable", [])),
                        hard_negatives=tuple(case["expected"].get("hard_negatives", [])),
                    ),
                    ranked=ranked,
                    issue_status=issue_status,
                    latency_ms=elapsed_ms,
                )
            )
        return results

    @pytest.fixture(scope="class")
    def comparison(self, hybrid_corpus):
        engine = hybrid_corpus["engine"]
        out: dict = {}
        for mode, lane in (("lexical", SERVICE_LANE), ("hybrid", HYBRID_LANE)):
            results = self._run_all(engine, mode)
            exact = [r for r in results if r.case_class in _EXACT_CLASSES]
            out[mode] = {
                "overall": aggregate(lane, results),
                "exact": aggregate(lane, exact) if exact else None,
            }
        return out

    def test_hybrid_served_every_case_with_semantic_legs(self, hybrid_corpus):
        """Every hybrid request resolved a generation and scanned vectors."""
        engine = hybrid_corpus["engine"]
        fixture = load_fixture(FIXTURE_PATH)
        original = rsearch.SessionLocal
        rsearch.SessionLocal = sessionmaker(bind=engine)
        try:
            for case in fixture["cases"]:
                response = rsearch.ranked_search(_request_for(case, mode="hybrid", limit=10))
                assert response["mode_effective"] == "hybrid", (
                    f"case {case['id']} degraded: {response['degradation']['reasons']}"
                )
                assert response["diagnostics"]["semantic"]["served"] is True
        finally:
            rsearch.SessionLocal = original

    def test_comparison_report_records_both_modes(self, comparison, capsys):
        """Record recall/MRR/hard-negative load per mode, overall and for the
        exact-string classes; printed whole so the ticket comment can carry
        it verbatim. Metric deltas are recorded, never gated here — whether
        hybrid becomes the default is an operator decision (§16.1)."""

        def block(metrics):
            if metrics is None:
                return None
            return {
                "recall_at_5": round(metrics.recall_at_5, 4),
                "recall_at_10": round(metrics.recall_at_10, 4),
                "mrr": round(metrics.mrr, 4),
                "mrr_primary": round(metrics.mrr_primary, 4),
                "hard_negative_load_at_5": metrics.hard_negative_load_at_5,
                "hard_negative_load_at_10": metrics.hard_negative_load_at_10,
                "hard_negative_case_rate": round(metrics.hard_negative_case_rate, 4),
            }

        report = {
            mode: {
                scope: block(values[scope]) for scope in ("overall", "exact")
            }
            for mode, values in comparison.items()
        }
        print("\n=== M2.3 harness comparison (fixture corpus.v1) ===")
        print(json.dumps(report, indent=1))
        for mode in ("lexical", "hybrid"):
            assert comparison[mode]["overall"] is not None
        assert comparison["lexical"]["exact"] is not None
        assert comparison["hybrid"]["exact"] is not None
