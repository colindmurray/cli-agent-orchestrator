"""`cao issue search` — the native ranked-search CLI surface (§12.2).

The contract under test: the flag grammar forwards to the shared ranked-search
service request exactly (repeatable multi-value flags included), ``--json``
emits the full explanation objects the API emits, human mode surfaces lane
badges, snippets, degradation, and pagination; scope is exactly one of
``--tracker-project`` or ``--all-projects``; and every refusal keeps the
service's typed classification rather than dying as a usage error.
"""

import json

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.cli.commands import issue as issue_cli
from cli_agent_orchestrator.clients.database import (
    Base,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services import tracker_ranked_search as ranked


@pytest.fixture(autouse=True)
def search_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/search-cli.db")
    Base.metadata.create_all(bind=engine)
    _migrate_tracker_search_projection(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    monkeypatch.setattr(ranked, "SessionLocal", sessions)
    # The group callbacks create the schema on the real engine. Here the
    # schema already exists on the test engine, so the call is neutralised.
    monkeypatch.setattr(issue_cli, "ensure_tracker_schema", lambda: None)
    yield engine
    engine.dispose()


@pytest.fixture
def runner():
    return CliRunner()


def seed_corpus():
    """Two projects whose corpus discriminates every flag family."""
    tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
    tracker.create_project(name="Other", project_id="other", issue_prefix="oth")
    parent = tracker.create_issue(
        project_id="cao-system",
        key="cond-0001",
        title="deploy pipeline bounces on dry run",
        labels=["deploy", "infra"],
        observed_revision="v1.2.3",
        force=True,
    )
    child = tracker.create_issue(
        project_id="cao-system",
        key="cond-0002",
        title="deploy rollback torn state",
        labels=["deploy"],
        status="closed",
        observed_revision="v4.5.6",
        force=True,
    )
    outside = tracker.create_issue(
        project_id="other",
        key="oth-0001",
        title="deploy dashboard wish",
        kind="feature",
        force=True,
    )
    task = tracker.create_issue(
        project_id="cao-system",
        key="cond-0003",
        title="rerun the deploy pipeline manually",
        kind="task",
        status="blocked",
        force=True,
    )
    tracker.add_link("cond-0002", to_key="cond-0001", kind="part-of")
    tracker.add_comment(
        "cond-0003", body="the zephyr word lives only in this comment", important=True
    )
    return parent, child, outside, task


def search(runner, *args):
    result = runner.invoke(issue_cli.issue, ["search", *args])
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


class TestFlagGrammarForwardsToTheService:
    def test_scope_forms_select_the_candidate_set(self, runner):
        parent, child, outside, task = seed_corpus()
        scoped = json.loads(search(runner, "deploy", "--tracker-project", "cao-system", "--json"))
        keys = [row["issue"]["key"] for row in scoped["results"]]
        assert parent["key"] in keys and child["key"] in keys
        assert outside["key"] not in keys

        repeated = json.loads(
            search(
                runner,
                "deploy",
                "--tracker-project",
                "cao-system",
                "--tracker-project",
                "other",
                "--json",
            )
        )
        assert {row["issue"]["key"] for row in repeated["results"]} >= {
            parent["key"],
            child["key"],
            outside["key"],
        }

        everything = json.loads(search(runner, "deploy", "--all-projects", "--json"))
        assert everything["scope"]["all_projects"] is True
        assert outside["key"] in [row["issue"]["key"] for row in everything["results"]]

    def test_repeated_kind_flags_or_compose(self, runner):
        _, _, _, task = seed_corpus()
        payload = json.loads(
            search(
                runner,
                "deploy pipeline",
                "--all-projects",
                "--kind",
                "bug",
                "--kind",
                "task",
                "--json",
            )
        )
        keys = [row["issue"]["key"] for row in payload["results"]]
        assert task["key"] in keys
        assert all(row["issue"]["kind"] in ("bug", "task") for row in payload["results"])

    def test_repeated_label_flags_and_compose_and_without_label_excludes(self, runner):
        parent, child, _, _ = seed_corpus()
        required = json.loads(
            search(
                runner,
                "deploy",
                "--all-projects",
                "--label",
                "deploy",
                "--label",
                "infra",
                "--json",
            )
        )
        assert [row["issue"]["key"] for row in required["results"]] == [parent["key"]]

        excluded = json.loads(
            search(
                runner,
                "deploy",
                "--all-projects",
                "--without-label",
                "infra",
                "--without-label",
                "nonexistent-label",
                "--json",
            )
        )
        keys = [row["issue"]["key"] for row in excluded["results"]]
        assert child["key"] in keys
        assert parent["key"] not in keys

    def test_under_restricts_to_the_part_of_subtree_closure(self, runner):
        parent, child, outside, task = seed_corpus()
        payload = json.loads(
            search(runner, "deploy", "--all-projects", "--under", "cond-0001", "--json")
        )
        keys = [row["issue"]["key"] for row in payload["results"]]
        assert parent["key"] in keys and child["key"] in keys
        assert outside["key"] not in keys and task["key"] not in keys

    def test_observed_revision_is_exactly_filterable_from_the_cli(self, runner):
        parent, _, _, _ = seed_corpus()
        exact = json.loads(
            search(runner, "deploy", "--all-projects", "--observed-revision", "v1.2.3", "--json")
        )
        assert [row["issue"]["key"] for row in exact["results"]] == [parent["key"]]

        none = json.loads(
            search(runner, "deploy", "--all-projects", "--observed-revision", "v9.9.9", "--json")
        )
        assert none["total"] == 0

    def test_scalar_filters_forward(self, runner):
        _, child, _, _ = seed_corpus()
        payload = json.loads(
            search(
                runner,
                "deploy",
                "--all-projects",
                "--status",
                "closed",
                "--severity",
                "unset",
                "--json",
            )
        )
        assert [row["issue"]["key"] for row in payload["results"]] == [child["key"]]

    def test_open_only_excludes_terminal_statuses(self, runner):
        _, child, _, _ = seed_corpus()
        payload = json.loads(search(runner, "deploy", "--all-projects", "--open-only", "--json"))
        assert child["key"] not in [row["issue"]["key"] for row in payload["results"]]

    def test_no_comments_drops_the_comment_bm25_lane(self, runner):
        seed_corpus()
        full = json.loads(search(runner, "zephyr", "--all-projects", "--json"))
        assert "comment-bm25" in [lane["lane"] for lane in full["results"][0]["contributing_lanes"]]
        trimmed = json.loads(search(runner, "zephyr", "--all-projects", "--no-comments", "--json"))
        assert "comment-bm25" not in [
            lane["lane"] for lane in trimmed["results"][0]["contributing_lanes"]
        ]

    def test_limit_and_offset_window_without_losing_total(self, runner):
        seed_corpus()
        page = json.loads(
            search(runner, "deploy", "--all-projects", "--limit", "1", "--offset", "1", "--json")
        )
        assert len(page["results"]) == 1
        assert page["total"] > 1

    def test_flag_grammar_produces_the_service_request_it_claims(self, runner):
        """Wiring proof: the rendered CLI invocation and an identically
        constructed RankedSearchRequest return byte-identical explanations."""
        seed_corpus()
        from cli_agent_orchestrator.services.tracker_ranked_search import (
            RankedSearchRequest,
        )

        direct = ranked.ranked_search(
            RankedSearchRequest(
                query="deploy",
                project_ids=("cao-system",),
                kinds=("bug", "task"),
                labels=("deploy",),
                without_labels=("infra",),
                open_only=True,
                limit=5,
            )
        )
        through_cli = json.loads(
            search(
                runner,
                "deploy",
                "--tracker-project",
                "cao-system",
                "--kind",
                "bug",
                "--kind",
                "task",
                "--label",
                "deploy",
                "--without-label",
                "infra",
                "--open-only",
                "--limit",
                "5",
                "--json",
            )
        )
        # Wall-clock diagnostics are the only legitimate difference.
        through_cli.pop("diagnostics")
        direct.pop("diagnostics")
        assert through_cli == direct


class TestJsonShapeAndHumanRendering:
    def test_json_emits_full_explanation_objects(self, runner):
        seed_corpus()
        payload = json.loads(search(runner, "deploy pipeline bounces", "--all-projects", "--json"))
        assert set(payload) >= {
            "query",
            "scope",
            "mode_requested",
            "mode_effective",
            "degradation",
            "generations",
            "diagnostics",
            "total",
            "limit",
            "offset",
            "results",
        }
        explanation = payload["results"][0]
        assert set(explanation) == {
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

    def test_human_mode_shows_hits_scores_lane_badges_and_snippets(self, runner):
        seed_corpus()
        out = search(runner, "deploy pipeline bounces", "--all-projects")
        assert 'hit(s) for "deploy pipeline bounces"' in out
        assert "· mode lexical" in out
        assert "lanes " in out
        assert "matched " in out
        assert any(field in out for field in ("title:", "body:", "failing_command:"))
        assert "hit(s)" in out.splitlines()[-1]

    def test_human_mode_shows_degradation_for_uninstalled_modes(self, runner):
        seed_corpus()
        out = search(runner, "deploy", "--all-projects", "--mode", "semantic")
        assert "degraded:" in out
        assert (
            "no active vector generation: build and activate one with the "
            "search-index model verbs before requesting semantic retrieval"
        ) in out

    def test_pagination_footer_appears_when_truncating(self, runner):
        seed_corpus()
        out = search(runner, "deploy", "--all-projects", "--limit", "1")
        assert "-- showing 1-1 of" in out

    def test_empty_result_set_renders_quietly(self, runner):
        seed_corpus()
        out = search(runner, "deploy", "--all-projects", "--observed-revision", "v9.9.9")
        assert out.startswith("0 hit(s)")


class TestRefusalsKeepTheirClassification:
    def test_neither_scope_form_is_a_typed_refusal(self, runner):
        seed_corpus()
        result = runner.invoke(issue_cli.issue, ["search", "deploy"])
        assert result.exit_code == 1
        assert "[invalid-scope]" in result.output

    def test_both_scope_forms_is_a_typed_refusal(self, runner):
        seed_corpus()
        result = runner.invoke(
            issue_cli.issue,
            ["search", "deploy", "--tracker-project", "cao-system", "--all-projects"],
        )
        assert result.exit_code == 1
        assert "[invalid-scope]" in result.output

    def test_a_punctuation_only_query_is_a_typed_refusal(self, runner):
        seed_corpus()
        result = runner.invoke(issue_cli.issue, ["search", "!!! ???", "--all-projects"])
        assert result.exit_code == 1
        assert "[invalid-query]" in result.output

    def test_unknown_vocabulary_is_forwarded_and_classified_by_the_service(self, runner):
        seed_corpus()
        result = runner.invoke(
            issue_cli.issue, ["search", "deploy", "--all-projects", "--kind", "widget"]
        )
        assert result.exit_code == 1
        assert "[invalid]" in result.output
