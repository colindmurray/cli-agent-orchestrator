"""The ranked lexical search service (design §10.3–§10.5, §19.2 query tests).

The contract under test: literal-safe query handling; field-weighted issue
BM25 and comment BM25 aggregated at issue level with the important-comment
boost and contribution cap; an exact technical-string lane that survives
punctuation; weighted RRF fusion with the documented tie-break
(``rank_score``, then ``updated_at DESC``, then key); scope and structured
filters through the shared builder; complete per-result explanations; visible
degradation for not-yet-installed semantic modes; and typed refusals for empty
queries and out-of-bounds pagination.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import tracker_search_schema
from cli_agent_orchestrator.clients.database import (
    _TRACKER_ORM_TABLE_NAMES,
    Base,
    TrackerCommentModel,
    TrackerIssueModel,
    TrackerLinkModel,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.services import tracker_ranked_search as rsearch
from cli_agent_orchestrator.services.issue_tracker import TrackerError
from cli_agent_orchestrator.services.tracker_ranked_search import RankedSearchRequest


class SearchDb:
    """A file-backed tracker store with the search projection installed."""

    def __init__(self, path):
        self.path = str(path)
        self.engine = create_engine(f"sqlite:///{self.path}")
        Base.metadata.create_all(
            bind=self.engine,
            tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
        )
        _migrate_tracker_search_projection(self.engine)


@pytest.fixture
def sdb(tmp_path):
    db = SearchDb(tmp_path / "ranked.db")
    yield db
    db.engine.dispose()


BASE_TS = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(days_offset: int) -> datetime:
    return BASE_TS + timedelta(days=days_offset)


def seed(db: SearchDb) -> None:
    """A small corpus engineered so each §19.2 row has a discriminating case."""
    session = sessionmaker(bind=db.engine)()
    rows = [
        # key, project, title, body, status, severity, kind, component,
        # observed_revision, labels, failing_command, updated offset days,
        # duplicate_of
        (
            "k-1",
            "p1",
            "Deploy pipeline bounces on dry run",
            "The guarded bounce returned no verified activation receipt.",
            "open",
            "P1",
            "bug",
            "conduct",
            "v1.2.3",
            '["deploy","infra"]',
            "conduct deploy --dry-run --bounce",
            2,
            None,
        ),
        (
            "k-2",
            "p1",
            "Deploy pipeline duplicate record",
            "See canonical deploy report.",
            "closed",
            "P2",
            "bug",
            None,
            None,
            '["deploy"]',
            None,
            5,
            "k-1",
        ),
        (
            "k-3",
            "p1",
            "Unrelated widget tuning",
            "Nothing here mentions the pipeline except once: deploy.",
            "triage",
            "unset",
            "task",
            "ui",
            None,
            "[]",
            None,
            9,
            None,
        ),
        (
            "k-4",
            "p1",
            "Artifact lives in this title",
            "quiet body",
            "blocked",
            "P3",
            "bug",
            None,
            None,
            '["quiet"]',
            None,
            1,
            None,
        ),
        (
            "k-5",
            "p1",
            "Unrelated title here",
            "The artifact word zephyr appears only here.",
            "resolved",
            "unset",
            "bug",
            None,
            None,
            "[]",
            None,
            3,
            None,
        ),
        (
            "k-6",
            "p1",
            "Chatty lease deadlock thread",
            "short body",
            "open",
            "P2",
            "bug",
            None,
            None,
            "[]",
            None,
            4,
            None,
        ),
        (
            "k-7",
            "p1",
            "Concise successor lease deadlock report",
            "short body",
            "open",
            "P2",
            "bug",
            None,
            None,
            "[]",
            None,
            6,
            None,
        ),
        (
            "k-8",
            "p2",
            "Cross-project descendant of k-1",
            "child of the deploy root",
            "open",
            "unset",
            "bug",
            None,
            None,
            "[]",
            None,
            7,
            None,
        ),
        (
            "k-9",
            "p1",
            "Feature request: darker mode",
            "cosmetic wish",
            "open",
            "unset",
            "feature",
            None,
            None,
            "[]",
            None,
            8,
            None,
        ),
        (
            "k-10",
            "p1",
            "Cycle member one",
            "cyc",
            "open",
            "unset",
            "bug",
            None,
            None,
            "[]",
            None,
            1,
            None,
        ),
        (
            "k-11",
            "p1",
            "Cycle member two",
            "cyc",
            "open",
            "unset",
            "bug",
            None,
            None,
            "[]",
            None,
            1,
            None,
        ),
        (
            "k-12",
            "p1",
            "Importance twin probe",
            "twin comments decide the boost",
            "open",
            "unset",
            "bug",
            None,
            None,
            "[]",
            None,
            2,
            None,
        ),
        (
            "k-13",
            "p1",
            "Body symbol probe",
            "the marker zeta_symbol_only_body_marker lives only in this body",
            "open",
            "unset",
            "bug",
            None,
            None,
            "[]",
            None,
            2,
            None,
        ),
    ]
    for (
        key,
        project,
        title,
        body,
        status,
        severity,
        kind,
        component,
        observed_revision,
        labels,
        failing_command,
        updated_offset,
        duplicate_of,
    ) in rows:
        session.add(
            TrackerIssueModel(
                key=key,
                project_id=project,
                title=title,
                body=body,
                status=status,
                severity=severity,
                kind=kind,
                component=component,
                observed_revision=observed_revision,
                labels=labels,
                failing_command=failing_command,
                duplicate_of=duplicate_of,
                created_at=_ts(updated_offset),
                updated_at=_ts(updated_offset),
            )
        )
    session.flush()
    # A chatty thread: many weak ordinary comment hits for the lease query.
    for n in range(6):
        session.add(
            TrackerCommentModel(
                issue_key="k-6",
                author=f"agent-{n}",
                body=(
                    f"Progress note {n}: we looked at the successor lease deadlock "
                    "again today and the investigation continues along broad lines."
                ),
            )
        )
    # The concise competitor: one precise, important comment.
    session.add(
        TrackerCommentModel(
            issue_key="k-7",
            author="operator",
            body="successor lease deadlock fixed by fencing token rotation",
            important=True,
        )
    )
    session.add(
        TrackerCommentModel(
            issue_key="k-1", author="reviewer", body="the bounce receipt was wrapped"
        )
    )
    # Importance twins on one issue: byte-identical bodies, the ordinary one
    # inserted FIRST so its id is lower. Only the importance boost can make
    # the higher-id important comment win — which is exactly what the red
    # proof flips.
    session.add(
        TrackerCommentModel(issue_key="k-12", author="twin-a", body="successor lease deadlock")
    )
    session.add(
        TrackerCommentModel(
            issue_key="k-12",
            author="twin-b",
            body="successor lease deadlock",
            important=True,
        )
    )
    # part-of links: k-2 child of k-1; cross-project k-8 under k-1; a cycle.
    for child, parent in (("k-2", "k-1"), ("k-8", "k-1"), ("k-10", "k-11"), ("k-11", "k-10")):
        session.add(TrackerLinkModel(from_key=child, to_key=parent, kind="part-of"))
    session.commit()
    session.close()


@pytest.fixture
def service(sdb, monkeypatch):
    seed(sdb)
    monkeypatch.setattr(rsearch, "SessionLocal", sessionmaker(bind=sdb.engine))
    return sdb


def search(service, query, **kwargs):
    request = RankedSearchRequest(query=query, **{"project_ids": ("p1",), **kwargs})
    return rsearch.ranked_search(request)


def keys_of(response):
    return [r["issue"]["key"] for r in response["results"]]


# ---------------------------------------------------------------------------
# Query handling (§10.5): literal safety, emptiness, bounds
# ---------------------------------------------------------------------------


class TestLiteralQuerySafety:
    def test_punctuation_heavy_query_survives_and_matches_its_owner(self, service):
        response = search(service, "conduct deploy --dry-run --bounce")
        assert keys_of(response)[0] == "k-1"

    def test_fts_operator_words_never_act_as_syntax(self, service):
        # Raw FTS grammar would parse OR/NOT/NEAR/"*" as operators; the safe
        # quoting treats them as literals and still returns explained results.
        response = search(service, 'NOT OR NEAR "*"', limit=5)
        assert response["mode_effective"] == "lexical"
        for result in response["results"]:
            assert set(result["contributing_lanes"]) <= {"issue-bm25", "comment-bm25", "exact"}

    def test_deliberately_quoted_phrase_is_preserved(self, service):
        phrase_only = search(service, '"activation receipt"')
        assert "k-1" in keys_of(phrase_only)

    def test_empty_and_punctuation_only_queries_are_rejected(self, service):
        for bad in ("", "   ", "!!! ??? --"):
            with pytest.raises(TrackerError) as excinfo:
                search(service, bad)
            assert excinfo.value.code == "invalid-query"

    def test_overlong_query_is_refused(self, service):
        with pytest.raises(TrackerError):
            search(service, "x" * 1001)

    def test_too_many_terms_is_refused(self, service):
        with pytest.raises(TrackerError):
            search(service, " ".join(f"t{n}" for n in range(65)))


# ---------------------------------------------------------------------------
# Lanes, weighting, aggregation (§10.3/§10.4)
# ---------------------------------------------------------------------------


class TestLaneBehavior:
    def test_title_match_outranks_body_match_for_the_same_term(self, service):
        response = search(service, "artifact")
        ordered = keys_of(response)
        assert ordered.index("k-4") < ordered.index("k-5")

    def test_comment_hits_aggregate_at_issue_level_with_cap_and_boost(self, service):
        response = search(service, "successor lease deadlock")
        ordered = keys_of(response)
        # Six weak ordinary hits cannot outweigh one precise important hit.
        assert ordered.index("k-7") < ordered.index("k-6")
        by_key = {r["issue"]["key"]: r for r in response["results"]}
        winning = by_key["k-7"]["winning_comment"]
        assert winning is not None
        assert winning["important"] is True
        assert winning["retained_hits"] == 1
        chatty = by_key["k-6"]["winning_comment"]
        assert chatty["total_matching_comments"] == 6
        assert chatty["retained_hits"] == rsearch.MAX_COMMENT_HITS_PER_ISSUE

    def test_include_comments_false_drops_the_comment_lane(self, service):
        response = search(service, "successor lease deadlock", include_comments=False)
        for result in response["results"]:
            lanes = {entry["lane"] for entry in result["contributing_lanes"]}
            assert "comment-bm25" not in lanes
            assert result["winning_comment"] is None

    def test_exact_lane_pins_symbol_that_tokenization_shreds(self, service):
        response = search(service, "_codex_mcp_pin_preflight")
        # No corpus row carries this symbol; the point is the exact lane runs
        # escaped-LIKE matching without FTS syntax errors and returns nothing.
        assert response["total"] == 0

    def test_exact_lane_finds_a_symbol_that_exists_only_in_the_body(self, service):
        """Red proof for the exact-lane column alignment.

        The marker occurs only in k-13's body — a column the BM25 phrase lane
        tokenizes apart (underscores split) and the only lane that can own the
        hit is exact substring. If the exact lane reads the wrong column this
        returns nothing or attributes the wrong field.
        """
        response = search(service, "zeta_symbol_only_body_marker")
        assert "k-13" in keys_of(response)
        top = next(r for r in response["results"] if r["issue"]["key"] == "k-13")
        lanes = {entry["lane"]: entry for entry in top["contributing_lanes"]}
        assert "exact" in lanes
        # body's EXACT_FIELD_PRIORITY is 11; raw_score encodes -priority.
        assert lanes["exact"]["raw_score"] == pytest.approx(-11.0)
        assert top["matched_fields"]
        assert any("zeta_symbol_only_body_marker" in s for s in top["snippets"].values())

    def test_exact_lane_matches_wrapped_error_fragment_via_bm25_phrase(self, service):
        response = search(service, "guarded bounce returned no verified activation receipt")
        assert keys_of(response)[0] == "k-1"

    def test_exact_issue_key_query_boosts_its_issue(self, service):
        response = search(service, "k-1")
        top = response["results"][0]
        assert top["issue"]["key"] == "k-1"
        assert "issue-key-equality" in top["exact_boosts"]

    def test_exact_failing_command_equality_records_a_fingerprint_boost(self, service):
        response = search(service, "conduct deploy --dry-run --bounce")
        top = response["results"][0]
        assert top["issue"]["failing_command"] == "conduct deploy --dry-run --bounce"
        assert "failing-command-equality" in top["exact_boosts"]

    def test_rrf_fusion_math_wires_weights_and_constant_together(self):
        lanes = {
            "issue-bm25": [("a", -3.2), ("b", -5.0)],
            "comment-bm25": [("a", -1.0)],
            "exact": [],
        }
        fused = rsearch._fuse_lane_ranks(lanes)
        # rank_score is the weighted RRF sum, not a sum of raw scores.
        expected_a = rsearch.LANE_WEIGHT_ISSUE_BM25 / (
            rsearch.RRF_K + 1
        ) + rsearch.LANE_WEIGHT_COMMENT_BM25 / (rsearch.RRF_K + 1)
        expected_b = rsearch.LANE_WEIGHT_ISSUE_BM25 / (rsearch.RRF_K + 2)
        assert fused["a"]["score"] == pytest.approx(expected_a)
        assert fused["b"]["score"] == pytest.approx(expected_b)
        assert {entry["lane"] for entry in fused["a"]["lanes"]} == {
            "issue-bm25",
            "comment-bm25",
        }
        assert fused["a"]["lanes"][0]["raw_score"] == -3.2

    def test_bm25_weight_arguments_align_with_every_declared_column(self):
        """bm25() maps weights to ALL declared columns (UNINDEXED included).

        A list built over only the indexed subset silently shifts every
        weight onto the wrong column; this pin freezes the alignment.
        """
        for table, weights in (
            (tracker_search_schema.ISSUE_FTS_TABLE, rsearch.ISSUE_FIELD_WEIGHTS),
            (tracker_search_schema.COMMENT_FTS_TABLE, rsearch.COMMENT_FIELD_WEIGHTS),
        ):
            declared = rsearch._fts_declared_columns(table)
            args_text = rsearch._bm25_weight_args(table)
            values = [float(part) for part in args_text.strip(", ").split(",")]
            assert len(values) == len(declared), (table, declared, values)
            for value, (name, indexed) in zip(values, declared):
                if indexed:
                    assert value == weights[name], (table, name, value)
                else:
                    assert value == 1.0, (table, name, value)

    def test_title_weight_position_actually_receives_the_title_weight(self, service):
        """End-to-end: the §10.3 title>body hypothesis orders the fixture."""
        response = search(service, "artifact")
        ordered = keys_of(response)
        assert ordered.index("k-4") < ordered.index("k-5")


# ---------------------------------------------------------------------------
# Scope and structured filters (§10.1/§10.2)
# ---------------------------------------------------------------------------


class TestScopeAndFilters:
    def test_repeated_projects_or_and_cross_project_exclusion_by_default(self, service):
        both = rsearch.ranked_search(
            RankedSearchRequest(query="deploy", project_ids=("p1", "p2"), limit=50)
        )
        assert "k-8" in keys_of(both)
        only_p1 = search(service, "deploy", limit=50)
        assert "k-8" not in keys_of(only_p1)

    def test_scope_forms_are_mutually_exclusive(self, service):
        with pytest.raises(TrackerError) as both:
            rsearch.ranked_search(
                RankedSearchRequest(query="deploy", project_ids=("p1",), all_projects=True)
            )
        assert both.value.code == "invalid-scope"
        with pytest.raises(TrackerError) as neither:
            rsearch.ranked_search(RankedSearchRequest(query="deploy"))
        assert neither.value.code == "invalid-scope"

    def test_subtree_closure_includes_transitive_descendants(self, service):
        response = search(service, "deploy", subtree_roots=("k-1",), limit=50)
        keys = keys_of(response)
        assert "k-1" in keys and "k-2" in keys

    def test_subtree_closure_survives_a_membership_cycle(self, service):
        response = search(service, "cycle", subtree_roots=("k-10",), limit=50)
        keys = keys_of(response)
        assert "k-10" in keys and "k-11" in keys

    def test_cross_project_descendant_excluded_from_project_scoped_subtree(self, service):
        response = search(service, "descendant", subtree_roots=("k-1",), limit=50)
        assert response["scope"]["subtree_closure_size"] >= 3
        assert "k-8" not in keys_of(response)
        everything = rsearch.ranked_search(
            RankedSearchRequest(
                query="descendant", all_projects=True, subtree_roots=("k-1",), limit=50
            )
        )
        assert "k-8" in keys_of(everything)

    def test_each_filter_family_works_alone(self, service):
        assert "k-9" in keys_of(search(service, "darker mode"))
        bugs = search(service, "deploy", kinds=("bug",), limit=50)
        assert "k-9" not in keys_of(bugs)
        statuses = search(service, "deploy", statuses=("closed",), limit=50)
        assert keys_of(statuses)[:1] == ["k-2"]
        severities = search(service, "deploy", severities=("P1",), limit=50)
        assert keys_of(severities)[:1] == ["k-1"]
        components = search(service, "deploy", components=("conduct",), limit=50)
        assert keys_of(components)[:1] == ["k-1"]
        revisions = search(service, "deploy", observed_revisions=("v1.2.3",), limit=50)
        assert keys_of(revisions)[:1] == ["k-1"]
        labeled = search(service, "deploy", labels=("infra",), limit=50)
        assert keys_of(labeled)[:1] == ["k-1"]
        without = search(service, "deploy", without_labels=("deploy",), limit=50)
        assert "k-1" not in keys_of(without)
        unlabeled = search(service, "deploy", unlabeled=True, limit=50)
        assert keys_of(unlabeled)[:1] != ["k-1"]
        assignee = search(service, "deploy", assignee="nobody-matches", limit=10)
        assert assignee["total"] == 0
        reporter = search(service, "deploy", reporter="nobody-matches", limit=10)
        assert reporter["total"] == 0
        terminal = search(service, "deploy", open_only=True, limit=50)
        assert "k-2" not in keys_of(terminal)

    def test_observed_revision_family_filters_exactly(self, service):
        miss = search(service, "deploy", observed_revisions=("v0.0.1",), limit=10)
        assert "k-1" not in keys_of(miss)
        hit = search(service, "deploy", observed_revisions=("v1.2.3",), limit=10)
        assert "k-1" in keys_of(hit)

    def test_families_compose_across_boundaries(self, service):
        composed = search(
            service,
            "deploy",
            statuses=("closed", "open"),
            severities=("P1", "P2"),
            labels=("deploy",),
            without_labels=("infra",),
            limit=50,
        )
        keys = keys_of(composed)
        assert "k-1" not in keys  # carries infra label
        assert "k-2" in keys  # closed P2 deploy label

    def test_terminal_issues_remain_similarity_candidates(self, service):
        response = search(service, "duplicate record deploy", limit=20)
        assert "k-2" in keys_of(response)


# ---------------------------------------------------------------------------
# Ordering, pagination, explanations, degradation (§10.4/§10.5)
# ---------------------------------------------------------------------------


class TestOrderingPaginationExplanations:
    def test_tie_break_prefers_newer_updated_at_then_key_order(self, service, monkeypatch):
        # Counterfactual seam: force equal rank scores so the documented
        # tie-break (updated_at DESC, then key) is what the assertion pins.
        constant_scores = {
            key: {"score": 0.42, "lanes": [{"lane": "exact", "rank": 1, "raw_score": 0.0}]}
            for key in ("k-4", "k-5", "k-6")
        }
        monkeypatch.setattr(rsearch, "_fuse_lane_ranks", lambda lanes: dict(constant_scores))
        response = search(service, "deploy", limit=50)
        page_keys = [k for k in keys_of(response) if k in constant_scores]
        # k-6 updated most recently, then k-5, then k-4.
        assert page_keys == ["k-6", "k-5", "k-4"]

    def test_tie_break_handles_mixed_timestamp_precision(self):
        """A fraction-less instant equals .000000 — never loses to an older one."""
        older_with_fraction = rsearch._desc_key("2026-08-01T12:00:00.500000")
        newer_without_fraction = rsearch._desc_key("2026-08-10T12:00:00")
        assert newer_without_fraction < older_with_fraction
        same_second = rsearch._desc_key("2026-08-10T12:00:00")
        assert same_second == newer_without_fraction

    def test_limit_bounds_are_typed_refusals(self, service):
        with pytest.raises(TrackerError):
            search(service, "deploy", limit=0)
        with pytest.raises(TrackerError):
            search(service, "deploy", limit=101)
        with pytest.raises(TrackerError):
            search(service, "deploy", offset=-1)

    def test_pagination_slices_stably_against_total(self, service):
        full = search(service, "deploy", limit=100)
        total = full["total"]
        page_one = search(service, "deploy", limit=1, offset=0)
        page_two = search(service, "deploy", limit=1, offset=1)
        assert page_one["total"] == total
        assert keys_of(page_one)[0] == keys_of(full)[0]
        assert keys_of(page_two)[0] == keys_of(full)[1]
        beyond = search(service, "deploy", limit=5, offset=total + 10)
        assert beyond["results"] == [] and beyond["total"] == total

    REQUIRED_EXPLANATION_FIELDS = (
        "issue",
        "rank_score",
        "contributing_lanes",
        "matched_fields",
        "snippets",
        "winning_comment",
        "exact_boosts",
        "neighborhood",
        "duplicate_chain",
    )

    def test_every_result_carries_a_complete_explanation(self, service):
        response = search(service, "deploy bounce receipt", limit=50)
        assert response["results"], "expected at least one result"
        for result in response["results"]:
            for field_name in self.REQUIRED_EXPLANATION_FIELDS:
                assert field_name in result, field_name
            assert result["issue"] is not None
            lane_names = {entry["lane"] for entry in result["contributing_lanes"]}
            assert lane_names, "at least one contributing lane"
            for entry in result["contributing_lanes"]:
                assert isinstance(entry["rank"], int) and entry["rank"] >= 1
                assert "raw_score" in entry
            if result["winning_comment"] is not None:
                assert result["matched_fields"]
        top = response["results"][0]
        assert top["neighborhood"], "k-1 has part-of children recorded as links"
        assert any(link["kind"] == "part-of" for link in top["neighborhood"])

    def test_duplicate_chain_expands_toward_the_canonical_issue(self, service):
        response = search(service, "pipeline duplicate record", limit=20)
        by_key = {r["issue"]["key"]: r for r in response["results"]}
        chain = by_key["k-2"]["duplicate_chain"]
        assert chain and chain[0]["canonical_key"] == "k-1"
        assert chain[0]["resolved"] is True

    def test_generations_echo_content_clock_and_active_generation(self, service):
        response = search(service, "deploy")
        generations = response["generations"]
        assert isinstance(generations["content_clock"], int)
        assert generations["document_schema_version"] == 1
        assert generations["active_vector_generation"] is None

    def test_semantic_modes_degrade_visibly_to_lexical(self, service):
        hybrid = search(service, "deploy bounce", mode="hybrid")
        assert hybrid["mode_requested"] == "hybrid"
        assert hybrid["mode_effective"] == "lexical"
        assert hybrid["degradation"]["reasons"]
        lexical = search(service, "deploy bounce")
        assert keys_of(hybrid) == keys_of(lexical)

    def test_diagnostics_report_lane_timings(self, service):
        response = search(service, "deploy")
        timings = response["diagnostics"]["lane_elapsed_ms"]
        assert {"issue-bm25", "comment-bm25", "exact"} <= set(timings)


class TestStatusDiagnostics:
    def test_status_reports_installation_and_backlog(self, service, monkeypatch):
        monkeypatch.setattr(rsearch, "SessionLocal", sessionmaker(bind=service.engine))
        status = rsearch.search_status()
        assert status["installed"] is True
        assert status["lexical"] == {
            "issue_documents_missing": 0,
            "comment_documents_missing": 0,
        }
        assert status["semantic"]["dirty_documents"] == 0

    def test_status_on_a_store_without_the_projection_reports_uninstalled(
        self, tmp_path, monkeypatch
    ):
        engine = create_engine(f"sqlite:///{tmp_path / 'bare.db'}")
        Base.metadata.create_all(
            bind=engine,
            tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
        )
        monkeypatch.setattr(rsearch, "SessionLocal", sessionmaker(bind=engine))
        assert rsearch.search_status() == {"installed": False}
        engine.dispose()


# ---------------------------------------------------------------------------
# Mutation red proofs (§19.7): reverting production behavior must break the
# named assertions above, or the suite proves nothing about the wiring.
# ---------------------------------------------------------------------------


class TestMutationRedProofs:
    def test_ignoring_the_important_comment_boost_turns_the_winner_red(self, service, monkeypatch):
        # Byte-identical twin bodies on k-12; the ordinary twin has the lower
        # id, so ONLY the importance boost can make the important twin win.
        normal = search(service, "successor lease deadlock", limit=50)
        winner = next(r for r in normal["results"] if r["issue"]["key"] == "k-12")[
            "winning_comment"
        ]
        assert winner is not None
        assert winner["important"] is True

        monkeypatch.setattr(rsearch, "IMPORTANT_COMMENT_BONUS", 0.0)
        reverted = search(service, "successor lease deadlock", limit=50)
        reverted_winner = next(r for r in reverted["results"] if r["issue"]["key"] == "k-12")[
            "winning_comment"
        ]
        # The central assertion above must NOT hold under the revert; a
        # still-true assertion would mean this test never pinned the boost.
        assert reverted_winner["important"] is False

    def test_bypassing_scope_validation_turns_the_cross_project_exclusion_red(
        self, service, monkeypatch
    ):
        scoped = search(service, "descendant", subtree_roots=("k-1",), limit=50)
        assert "k-8" not in keys_of(scoped)

        def unrestricted(*args, **kwargs):
            from cli_agent_orchestrator.services.tracker_filters import ScopeResolution

            return ScopeResolution(all_projects=True)

        monkeypatch.setattr(rsearch, "resolve_scope", unrestricted)
        bypassed = search(service, "descendant", subtree_roots=("k-1",), limit=50)
        try:
            assert "k-8" not in keys_of(bypassed)
        except AssertionError:
            pass
        else:
            pytest.fail("bypassing resolve_scope did not change results; exclusion unpinned")

    def test_removing_safe_quoting_turns_the_literal_safety_red(self, service, monkeypatch):
        safe = rsearch.build_fts_match_query('NOT OR NEAR "*stuff')
        assert safe == '"NOT" "OR" "NEAR" """*stuff"'

        def unsafe(text):
            return text

        monkeypatch.setattr(rsearch, "build_fts_match_query", unsafe)
        try:
            search(service, 'NOT OR NEAR "*stuff')
        except Exception:
            # Raw FTS grammar reached the index and SQLite refused it: the
            # literal-safety behavior is what stands between this input and
            # that failure.
            return
        pytest.fail("unquoted operator words did not break the query; quoting unpinned")


class TestRawFtsOperatorLeakage:
    def test_default_input_never_leaks_raw_operators_into_results(self, service):
        # A query stuffed with every FTS5 metacharacter returns the same shape
        # of explained payload as ordinary text — no syntax errors leak out.
        nasty = 'a"b (c) [d] {e} ^f -g +h |i| *j? k\\ l: m;n<o>'
        response = search(service, nasty, limit=10)
        assert response["query"] == nasty
        for result in response["results"]:
            assert result["issue"]["key"]
