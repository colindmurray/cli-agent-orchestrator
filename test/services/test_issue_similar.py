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
                "matched_fields",
                "snippets",
                "winning_comment",
                "exact_boosts",
                "neighborhood",
                "duplicate_chain",
            }

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
