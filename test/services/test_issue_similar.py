"""The similar-issue service (cond-0645).

The contract under test: strict XOR between ``issue_key`` and ``draft`` and
between the two scope forms; undeclared draft fields refused, never ignored;
the search spans open and terminal statuses, defaults the draft kind to
``bug``, keeps comments out of the match set, and never returns the source
issue itself; confirmed duplicates of hits expand exactly one level beside
their hit.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import (
    Base,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.services import issue_similar as similar
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services import tracker_ranked_search as ranked


@pytest.fixture(autouse=True)
def similar_db(tmp_path, monkeypatch):
    """A file-backed tracker store with the search projection installed."""
    engine = create_engine(f"sqlite:///{tmp_path}/similar-service.db")
    Base.metadata.create_all(bind=engine)
    _migrate_tracker_search_projection(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    monkeypatch.setattr(ranked, "SessionLocal", sessions)
    monkeypatch.setattr(similar, "SessionLocal", sessions)
    yield engine
    engine.dispose()


def seed_corpus():
    """A corpus whose hits discriminate scope, kind, status, and chains."""
    tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
    tracker.create_project(name="Other", project_id="other", issue_prefix="oth")
    open_bug = tracker.create_issue(
        project_id="cao-system",
        key="cond-0001",
        title="deploy pipeline bounces on dry run",
        force=True,
    )
    terminal_bug = tracker.create_issue(
        project_id="cao-system",
        key="cond-0002",
        title="deploy pipeline bounces on dry run during rollback",
        status="closed",
        force=True,
    )
    task = tracker.create_issue(
        project_id="cao-system",
        key="cond-0003",
        title="rerun the deploy pipeline manually",
        kind="task",
        force=True,
    )
    outside = tracker.create_issue(
        project_id="other",
        key="oth-0001",
        title="deploy pipeline bounces on dry run in another repo",
        force=True,
    )
    return open_bug, terminal_bug, task, outside


def _keys(payload):
    return [(c["issue"]["key"]) for c in payload["candidates"]]


class TestStrictXorValidation:
    def test_issue_key_and_draft_together_is_a_typed_refusal(self):
        tracker.create_project(name="P", project_id="p", issue_prefix="p")
        filed = tracker.create_issue(project_id="p", title="deploy pipeline bounces", force=True)
        with pytest.raises(tracker.TrackerError) as excinfo:
            similar.find_similar_issues(
                similar.SimilarIssuesRequest(
                    issue_key=filed["key"],
                    draft={"title": "deploy pipeline bounces"},
                    all_projects=True,
                )
            )
        assert excinfo.value.code == "invalid"
        assert "exactly one of issue_key or draft" in excinfo.value.message

    def test_neither_issue_key_nor_draft_is_a_typed_refusal(self):
        with pytest.raises(tracker.TrackerError) as excinfo:
            similar.find_similar_issues(similar.SimilarIssuesRequest(all_projects=True))
        assert excinfo.value.code == "invalid"
        assert "exactly one of issue_key or draft" in excinfo.value.message

    def test_both_scope_forms_is_a_typed_refusal(self):
        seed_corpus()
        with pytest.raises(tracker.TrackerError) as excinfo:
            similar.find_similar_issues(
                similar.SimilarIssuesRequest(
                    draft={"title": "deploy pipeline bounces"},
                    project_ids=("cao-system",),
                    all_projects=True,
                )
            )
        assert excinfo.value.code == "invalid-scope"

    def test_neither_scope_form_is_a_typed_refusal(self):
        seed_corpus()
        with pytest.raises(tracker.TrackerError) as excinfo:
            similar.find_similar_issues(
                similar.SimilarIssuesRequest(draft={"title": "deploy pipeline bounces"})
            )
        assert excinfo.value.code == "invalid-scope"

    @pytest.mark.parametrize("limit", [0, ranked.MAX_LIMIT + 1])
    def test_limit_outside_bounds_is_a_typed_refusal(self, limit):
        with pytest.raises(tracker.TrackerError) as excinfo:
            similar.find_similar_issues(
                similar.SimilarIssuesRequest(
                    draft={"title": "deploy pipeline bounces"},
                    all_projects=True,
                    limit=limit,
                )
            )
        assert excinfo.value.code == "invalid"

    @pytest.mark.parametrize(
        "field", ["key", "project_id", "status", "duplicate_of", "origin", "force"]
    )
    def test_server_owned_draft_fields_are_refused_not_ignored(self, field):
        with pytest.raises(tracker.TrackerError) as excinfo:
            similar.find_similar_issues(
                similar.SimilarIssuesRequest(
                    draft={"title": "deploy pipeline bounces", field: "whatever"},
                    all_projects=True,
                )
            )
        assert excinfo.value.code == "invalid"
        assert field in excinfo.value.message


class TestCandidateSemantics:
    def test_draft_kind_defaults_to_bug_and_filters_candidates(self):
        open_bug, terminal_bug, task, outside = seed_corpus()
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(draft={"title": "deploy pipeline"}, all_projects=True)
        )
        assert payload["query_source"] == {"mode": "draft", "issue_key": None, "kind": "bug"}
        keys = _keys(payload)
        assert task["key"] not in keys
        assert {open_bug["key"], terminal_bug["key"], outside["key"]} <= set(keys)

    def test_search_spans_open_and_terminal_statuses(self):
        open_bug, terminal_bug, _, _ = seed_corpus()
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(draft={"title": "deploy pipeline"}, all_projects=True)
        )
        keys = _keys(payload)
        assert open_bug["key"] in keys
        assert terminal_bug["key"] in keys

    def test_scope_selects_the_candidate_pool(self):
        _, _, _, outside = seed_corpus()
        scoped = similar.find_similar_issues(
            similar.SimilarIssuesRequest(
                draft={"title": "deploy pipeline"}, project_ids=("cao-system",)
            )
        )
        assert outside["key"] not in _keys(scoped)

        everywhere = similar.find_similar_issues(
            similar.SimilarIssuesRequest(draft={"title": "deploy pipeline"}, all_projects=True)
        )
        assert outside["key"] in _keys(everywhere)

    def test_issue_key_mode_excludes_self_but_finds_peers(self):
        open_bug, terminal_bug, _, outside = seed_corpus()
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(issue_key=open_bug["key"], all_projects=True)
        )
        assert payload["query_source"]["mode"] == "issue_key"
        keys = _keys(payload)
        assert open_bug["key"] not in keys
        assert terminal_bug["key"] in keys
        assert outside["key"] in keys

    def test_limit_one_still_returns_the_near_duplicate_when_self_ranks_first(self):
        tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
        # The near-duplicate is created first, so the source's later
        # updated_at wins the shared score tie: the source is its own top
        # hit and the near-duplicate sits at rank two.
        near_dup = tracker.create_issue(
            project_id="cao-system",
            key="cond-0001",
            title="deploy pipeline bounces on dry run during rollback",
            force=True,
        )
        source = tracker.create_issue(
            project_id="cao-system",
            key="cond-0002",
            title="deploy pipeline bounces on dry run",
            force=True,
        )
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(
                issue_key=source["key"],
                project_ids=("cao-system",),
                limit=1,
            )
        )
        assert payload["limit"] == 1
        assert _keys(payload) == [near_dup["key"]]

    def test_unknown_issue_key_is_a_typed_not_found(self):
        with pytest.raises(tracker.TrackerError) as excinfo:
            similar.find_similar_issues(
                similar.SimilarIssuesRequest(issue_key="cond-9999", all_projects=True)
            )
        assert excinfo.value.code == "not-found"

    def test_comments_are_out_of_the_match_set_by_default(self):
        open_bug, *_ = seed_corpus()
        tracker.add_comment(open_bug["key"], body="the zephyr word lives only here")
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(draft={"title": "zephyr"}, all_projects=True)
        )
        assert payload["include_comments"] is False
        # The exact lane still sees raw text, so the issue can surface, but
        # no comment document may rank as a contributing lane or a winner.
        assert open_bug["key"] in _keys(payload)
        for candidate in payload["candidates"]:
            assert candidate["winning_comment"] is None
            assert "comment-bm25" not in [lane["lane"] for lane in candidate["contributing_lanes"]]

    def test_candidates_carry_the_full_ranked_explanation(self):
        seed_corpus()
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(draft={"title": "deploy pipeline"}, all_projects=True)
        )
        for candidate in payload["candidates"]:
            assert set(candidate) == {
                "issue",
                "rank_score",
                "contributing_lanes",
                "probe_contributions",
                "matched_fields",
                "snippets",
                "winning_comment",
                "exact_boosts",
                "neighborhood",
                "duplicate_chain",
            }

    def test_long_title_reserves_high_value_probes_and_recalls_exact_command(self):
        tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
        exact = tracker.create_issue(
            project_id="cao-system",
            key="cond-0030",
            title="unrelated title",
            failing_command="conduct deploy --dry-run",
            force=True,
        )
        long_title = " ".join(f"titleword{index}" for index in range(100))
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(
                draft={
                    "title": long_title,
                    "failing_command": "conduct deploy --dry-run",
                    "reproduction_steps": "run the command once",
                    "actual_outcome": "the deploy bounces",
                },
                project_ids=("cao-system",),
                mode="lexical",
                limit=1,
            )
        )
        assert payload["candidates"][0]["issue"]["key"] == exact["key"]
        labels = [probe["label"] for probe in payload["diagnostics"]["similarity_probes"]]
        assert {"failing_command", "reproduction_steps", "actual_outcome"} <= set(labels)
        assert "failing_command" in payload["candidates"][0]["matched_fields"]

    def test_failed_punctuation_probe_is_partial_and_preserves_failure_details(self, monkeypatch):
        tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
        calls = []

        def search(request):
            calls.append(request.query)
            if request.query == "!!!":
                raise ranked.TrackerRankedSearchError(
                    "invalid-query", "ranked search requires nonempty normalized text"
                )
            return {
                "query": request.query,
                "scope": {
                    "project_ids": ["cao-system"],
                    "all_projects": False,
                    "subtree_roots": [],
                    "subtree_closure_size": 0,
                },
                "mode_requested": request.mode,
                "mode_effective": "lexical",
                "degradation": {"reasons": [], "lanes": {}},
                "generations": {},
                "diagnostics": {},
                "results": [],
            }

        monkeypatch.setattr(ranked, "ranked_search", search)
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(
                draft={"title": "valid sibling", "failing_command": "!!!"},
                project_ids=("cao-system",),
                mode="lexical",
            )
        )
        assert calls
        assert payload["coverage"]["probes_requested"] == len(
            payload["diagnostics"]["similarity_probes"]
        )
        assert (
            payload["coverage"]["probes_completed"] + payload["coverage"]["probes_failed"]
            == payload["coverage"]["probes_requested"]
        )
        assert payload["coverage"]["probes_failed"] == 1
        assert payload["coverage"]["partial"] is True
        assert payload["coverage"]["status"] == "inconclusive"
        assert payload["coverage"]["inconclusive"] is True
        assert payload["degradation"]["reasons"]
        assert payload["diagnostics"]["similarity_probe_failures"] == [
            {
                "label": "failing_command",
                "code": "invalid-query",
                "message": "ranked search requires nonempty normalized text",
            }
        ]

    def test_a_textless_draft_is_a_typed_invalid_query(self):
        with pytest.raises(tracker.TrackerError) as excinfo:
            similar.find_similar_issues(
                similar.SimilarIssuesRequest(draft={"labels": ["x"]}, all_projects=True)
            )
        assert excinfo.value.code == "invalid-query"


class TestDuplicateChainExpansion:
    def test_confirmed_duplicates_of_hits_expand_one_level(self):
        open_bug, _, _, _ = seed_corpus()
        # The duplicates themselves carry unrelated titles so the query hits
        # only cond-0001 and the chain shape is unambiguous.
        dup = tracker.create_issue(
            project_id="cao-system",
            key="cond-0004",
            title="already tracked elsewhere alpha",
            force=True,
        )
        tracker.update_issue(dup["key"], status="duplicate", duplicate_of=open_bug["key"])
        deeper = tracker.create_issue(
            project_id="cao-system",
            key="cond-0005",
            title="already tracked elsewhere beta",
            force=True,
        )
        tracker.update_issue(deeper["key"], status="duplicate", duplicate_of=dup["key"])

        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(draft={"title": "deploy pipeline"}, all_projects=True)
        )
        assert open_bug["key"] in _keys(payload)
        expanded = {
            (row["duplicate_of"], row["issue"]["key"]) for row in payload["duplicate_expansions"]
        }
        # Exactly one level: the duplicate OF the hit expands; the duplicate
        # of that duplicate belongs to the issue graph, not this answer.
        assert (open_bug["key"], dup["key"]) in expanded
        assert (dup["key"], deeper["key"]) not in expanded

    def test_the_source_issue_never_appears_in_its_own_expansion(self):
        open_bug, _, _, _ = seed_corpus()
        source = tracker.create_issue(
            project_id="cao-system",
            key="cond-0006",
            title="deploy pipeline bounces once more",
            force=True,
        )
        tracker.update_issue(source["key"], status="duplicate", duplicate_of=open_bug["key"])

        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(issue_key=source["key"], all_projects=True)
        )
        keys = _keys(payload)
        assert source["key"] not in keys
        assert source["key"] not in [row["issue"]["key"] for row in payload["duplicate_expansions"]]

    def test_native_link_only_duplicate_expands_in_source_to_canonical_direction(self):
        canonical, *_ = seed_corpus()
        duplicate = tracker.create_issue(
            project_id="cao-system", key="cond-0010", title="link-only duplicate", force=True
        )
        tracker.add_link(duplicate["key"], to_key=canonical["key"], kind="duplicates")
        expansions = similar._expand_duplicate_chains([canonical["key"]], exclude_key=None)
        assert [(row["duplicate_of"], row["issue"]["key"]) for row in expansions] == [
            (canonical["key"], duplicate["key"])
        ]

    def test_field_only_and_dual_duplicate_representations_dedupe(self):
        canonical, *_ = seed_corpus()
        field_only = tracker.create_issue(
            project_id="cao-system", key="cond-0011", title="field-only duplicate", force=True
        )
        tracker.update_issue(field_only["key"], status="duplicate", duplicate_of=canonical["key"])
        dual = tracker.create_issue(
            project_id="cao-system", key="cond-0012", title="dual duplicate", force=True
        )
        tracker.update_issue(dual["key"], status="duplicate", duplicate_of=canonical["key"])
        tracker.add_link(dual["key"], to_key=canonical["key"], kind="duplicates")
        expansions = similar._expand_duplicate_chains([canonical["key"]], exclude_key=None)
        assert [(row["duplicate_of"], row["issue"]["key"]) for row in expansions] == [
            (canonical["key"], field_only["key"]),
            (canonical["key"], dual["key"]),
        ]

    def test_native_duplicate_link_wins_over_conflicting_legacy_canonical(self):
        first, second, *_ = seed_corpus()
        duplicate = tracker.create_issue(
            project_id="cao-system", key="cond-0015", title="conflicting duplicate", force=True
        )
        tracker.update_issue(duplicate["key"], status="duplicate", duplicate_of=first["key"])
        tracker.add_link(duplicate["key"], to_key=second["key"], kind="duplicates")
        expansions = similar._expand_duplicate_chains(
            [first["key"], second["key"]], exclude_key=None
        )
        assert [(row["duplicate_of"], row["issue"]["key"]) for row in expansions] == [
            (second["key"], duplicate["key"])
        ]

    def test_duplicate_cycles_are_one_level_and_source_exclusion_is_preserved(self):
        canonical, *_ = seed_corpus()
        first = tracker.create_issue(
            project_id="cao-system", key="cond-0013", title="cycle first", force=True
        )
        second = tracker.create_issue(
            project_id="cao-system", key="cond-0014", title="cycle second", force=True
        )
        tracker.add_link(first["key"], to_key=canonical["key"], kind="duplicates")
        tracker.add_link(second["key"], to_key=first["key"], kind="duplicates")
        tracker.add_link(canonical["key"], to_key=first["key"], kind="duplicates")
        expansions = similar._expand_duplicate_chains(
            [canonical["key"], first["key"]], exclude_key=first["key"]
        )
        pairs = [(row["duplicate_of"], row["issue"]["key"]) for row in expansions]
        assert (canonical["key"], first["key"]) not in pairs
        assert (first["key"], second["key"]) in pairs
        assert len(pairs) == len(set(pairs))

    def test_multiple_native_targets_are_conflict_inconclusive_when_both_hit(self):
        tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
        first = tracker.create_issue(
            project_id="cao-system", key="cond-0022", title="shared canonical alpha", force=True
        )
        second = tracker.create_issue(
            project_id="cao-system", key="cond-0023", title="shared canonical beta", force=True
        )
        duplicate = tracker.create_issue(
            project_id="cao-system", key="cond-0024", title="unrelated duplicate source", force=True
        )
        tracker.add_link(duplicate["key"], to_key=first["key"], kind="duplicates")
        tracker.add_link(duplicate["key"], to_key=second["key"], kind="duplicates")

        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(
                draft={"title": "shared canonical"}, project_ids=("cao-system",)
            )
        )

        assert {first["key"], second["key"]} <= set(_keys(payload))
        assert not payload["duplicate_expansions"]
        assert payload["diagnostics"]["similarity_duplicate_conflicts"] == [
            {
                "code": "multiple-native-duplicate-targets",
                "message": "native duplicate source has multiple canonical targets",
                "duplicate_key": duplicate["key"],
                "canonical_keys": [first["key"], second["key"]],
                "hit_canonical_keys": [first["key"], second["key"]],
            }
        ]

    def test_multiple_native_targets_are_conflict_inconclusive_when_one_hit(self):
        tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
        first = tracker.create_issue(
            project_id="cao-system", key="cond-0025", title="only alpha canonical", force=True
        )
        second = tracker.create_issue(
            project_id="cao-system", key="cond-0026", title="different beta record", force=True
        )
        duplicate = tracker.create_issue(
            project_id="cao-system", key="cond-0027", title="unrelated duplicate source", force=True
        )
        tracker.add_link(duplicate["key"], to_key=first["key"], kind="duplicates")
        tracker.add_link(duplicate["key"], to_key=second["key"], kind="duplicates")

        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(draft={"title": "only alpha"}, project_ids=("cao-system",))
        )

        assert _keys(payload) == [first["key"]]
        assert not payload["duplicate_expansions"]
        assert payload["diagnostics"]["similarity_duplicate_conflicts"] == [
            {
                "code": "multiple-native-duplicate-targets",
                "message": "native duplicate source has multiple canonical targets",
                "duplicate_key": duplicate["key"],
                "canonical_keys": [first["key"], second["key"]],
                "hit_canonical_keys": [first["key"]],
            }
        ]


class TestSimilarityProbeCoverage:
    def test_one_token_title_drift_is_recalled_with_bounded_probes(self):
        tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
        near = tracker.create_issue(
            project_id="cao-system",
            key="cond-0020",
            title="worker lease renewal deadlocks on restart",
            force=True,
        )
        unrelated = tracker.create_issue(
            project_id="cao-system",
            key="cond-0021",
            title="worker lease status dashboard overview",
            force=True,
        )
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(
                draft={"title": "worker lease renewal stalls on restart"},
                project_ids=("cao-system",),
                limit=1,
                mode="lexical",
            )
        )
        assert payload["candidates"][0]["issue"]["key"] == near["key"]
        assert unrelated["key"] not in [row["issue"]["key"] for row in payload["candidates"]]
        probes = payload["coverage"]["probes_requested"]
        assert probes <= similar.SIMILARITY_MAX_PROBES
        assert payload["coverage"]["probes_completed"] == probes

    def test_degraded_empty_similarity_is_inconclusive_and_non_gating(self, monkeypatch):
        tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")

        def degraded(request):
            return {
                "query": request.query,
                "scope": {
                    "project_ids": ["cao-system"],
                    "all_projects": False,
                    "subtree_roots": [],
                    "subtree_closure_size": 0,
                },
                "mode_requested": request.mode,
                "mode_effective": "lexical",
                "degradation": {"reasons": ["semantic unavailable"], "lanes": {}},
                "generations": {},
                "diagnostics": {},
                "results": [],
            }

        monkeypatch.setattr(ranked, "ranked_search", degraded)
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(
                draft={"title": "unseen issue"}, project_ids=("cao-system",), mode="hybrid"
            )
        )
        assert payload["total"] == 0
        assert payload["mode_effective"] == "lexical"
        assert payload["coverage"]["inconclusive"] is True
        assert payload["degradation"]["reasons"] == ["semantic unavailable"]

    def test_semantic_mode_and_lane_facts_are_propagated(self, monkeypatch):
        tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
        issue = tracker.create_issue(
            project_id="cao-system", key="cond-0022", title="semantic match", force=True
        )

        def semantic(request):
            return {
                "query": request.query,
                "scope": {
                    "project_ids": ["cao-system"],
                    "all_projects": False,
                    "subtree_roots": [],
                    "subtree_closure_size": 0,
                },
                "mode_requested": request.mode,
                "mode_effective": "hybrid",
                "degradation": {"reasons": [], "lanes": {"semantic-issue": {"available": True}}},
                "generations": {"active_vector_generation": "g-1"},
                "diagnostics": {
                    "semantic": {
                        "served": True,
                        "generation_id": "g-1",
                        "issue_vectors_returned": 1,
                        "comment_issues_returned": 0,
                    }
                },
                "results": [
                    {
                        "issue": issue,
                        "rank_score": 0.1,
                        "contributing_lanes": [
                            {"lane": "semantic-issue", "rank": 1, "raw_score": 0.2}
                        ],
                        "matched_fields": ["title"],
                        "snippets": {"title": "semantic match"},
                        "winning_comment": None,
                        "exact_boosts": [],
                        "neighborhood": [],
                        "duplicate_chain": [],
                    }
                ],
            }

        monkeypatch.setattr(ranked, "ranked_search", semantic)
        payload = similar.find_similar_issues(
            similar.SimilarIssuesRequest(
                draft={"title": "semantic match"}, project_ids=("cao-system",)
            )
        )
        assert payload["mode_effective"] == "hybrid"
        assert payload["degradation"]["lanes"]["semantic-issue"]["available"] is True
        assert payload["candidates"][0]["contributing_lanes"][0]["lane"] == "semantic-issue"

    def test_equal_probe_rrf_tie_prefers_newer_issue_then_key(self):
        def result(key, updated_at):
            return {
                "issue": {"key": key, "updated_at": updated_at},
                "rank_score": 0.25,
                "contributing_lanes": [{"lane": "exact", "rank": 1, "raw_score": 0.0}],
                "matched_fields": ["title"],
                "snippets": {"title": key},
                "winning_comment": None,
                "exact_boosts": [],
                "neighborhood": [],
                "duplicate_chain": [],
            }

        candidates, _ = similar._merge_similarity_results(
            [
                (
                    "older-probe",
                    "older query",
                    1.0,
                    {"results": [result("cond-0031", "2026-08-01T00:00:00Z")]},
                ),
                (
                    "newer-probe",
                    "newer query",
                    1.0,
                    {"results": [result("cond-0032", "2026-08-02T00:00:00Z")]},
                ),
            ],
            source_key=None,
            limit=2,
        )
        assert [row["issue"]["key"] for row in candidates] == ["cond-0032", "cond-0031"]
        assert candidates[0]["probe_contributions"] == [
            {
                "label": "newer-probe",
                "query": "newer query",
                "weight": 1.0,
                "original_rank": 1,
                "original_score": 0.25,
            }
        ]
