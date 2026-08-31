"""`cao issue similar` — the native similar-issue CLI surface (cond-0645).

The contract under test: the flag grammar forwards to the shared
similar-issue service exactly, ``--json`` emits byte-identical JSON to the
API route for the same request (parity), human mode surfaces ranked
candidates and confirmed-duplicate expansions, every refusal keeps the
service's typed classification, and the verb stays advisory end to end.
"""

import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.api.main import app
from cli_agent_orchestrator.cli.commands import issue as issue_cli
from cli_agent_orchestrator.clients.database import (
    Base,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.plugins import PluginRegistry
from cli_agent_orchestrator.services import issue_similar as similar_service
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services import tracker_ranked_search as ranked


@pytest.fixture(autouse=True)
def similar_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/similar-cli.db")
    Base.metadata.create_all(bind=engine)
    _migrate_tracker_search_projection(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    monkeypatch.setattr(ranked, "SessionLocal", sessions)
    monkeypatch.setattr(similar_service, "SessionLocal", sessions)
    # The group callbacks create the schema on the real engine. Here the
    # schema already exists on the test engine, so the call is neutralised.
    monkeypatch.setattr(issue_cli, "ensure_tracker_schema", lambda: None)
    yield engine
    engine.dispose()


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def api_client():
    app.state.plugin_registry = PluginRegistry()

    class HostClient(TestClient):
        def request(self, method, url, **kwargs):
            headers = kwargs.pop("headers", None) or {}
            if not any(k.lower() == "host" for k in headers):
                headers["Host"] = "localhost"
            return super().request(method, url, headers=headers, **kwargs)

    return HostClient(app)


def seed_corpus():
    tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
    tracker.create_project(name="Other", project_id="other", issue_prefix="oth")
    hit = tracker.create_issue(
        project_id="cao-system",
        key="cond-0001",
        title="deploy pipeline bounces on dry run",
        failing_command="conduct deploy --dry-run",
        force=True,
    )
    terminal = tracker.create_issue(
        project_id="cao-system",
        key="cond-0002",
        title="deploy pipeline bounces on dry run during rollback",
        status="closed",
        failing_command="conduct deploy --dry-run",
        force=True,
    )
    outside = tracker.create_issue(
        project_id="other",
        key="oth-0001",
        title="deploy pipeline bounces on dry run in another repo",
        failing_command="conduct deploy --dry-run",
        force=True,
    )
    confirmed = tracker.create_issue(
        project_id="cao-system",
        key="cond-0003",
        title="already tracked elsewhere",
        force=True,
    )
    tracker.update_issue(confirmed["key"], status="duplicate", duplicate_of=hit["key"])
    return hit, terminal, outside, confirmed


def similar(runner, *args):
    result = runner.invoke(issue_cli.issue, ["similar", *args])
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


class TestFlagGrammarForwardsToTheService:
    def test_draft_file_probe_finds_candidates_and_expansion(self, runner, tmp_path):
        hit, terminal, _, confirmed = seed_corpus()
        draft = tmp_path / "draft.json"
        draft.write_text(json.dumps({"title": "deploy pipeline bounces"}))
        payload = json.loads(
            similar(runner, "--draft-file", str(draft), "--all-tracker-projects", "--json")
        )
        assert payload["query_source"] == {"mode": "draft", "issue_key": None, "kind": "bug"}
        keys = [row["issue"]["key"] for row in payload["candidates"]]
        assert hit["key"] in keys and terminal["key"] in keys
        expansions = {
            (row["duplicate_of"], row["issue"]["key"]) for row in payload["duplicate_expansions"]
        }
        assert (hit["key"], confirmed["key"]) in expansions

    def test_issue_key_mode_excludes_self(self, runner):
        hit, terminal, outside, _ = seed_corpus()
        payload = json.loads(
            similar(
                runner,
                "--issue-key",
                hit["key"],
                "--all-tracker-projects",
                "--json",
            )
        )
        keys = [row["issue"]["key"] for row in payload["candidates"]]
        assert hit["key"] not in keys
        assert terminal["key"] in keys and outside["key"] in keys
        assert payload["query_source"]["mode"] == "issue_key"

    def test_scope_form_selects_the_pool(self, runner):
        hit, _, outside, _ = seed_corpus()
        scoped = json.loads(
            similar(
                runner,
                "--issue-key",
                hit["key"],
                "--tracker-project",
                "cao-system",
                "--json",
            )
        )
        assert outside["key"] not in [row["issue"]["key"] for row in scoped["candidates"]]
        everywhere = json.loads(
            similar(runner, "--issue-key", hit["key"], "--all-tracker-projects", "--json")
        )
        assert outside["key"] in [row["issue"]["key"] for row in everywhere["candidates"]]

    def test_limit_forwards(self, runner, tmp_path):
        seed_corpus()
        draft_path = tmp_path / "draft.json"
        draft_path.write_text(json.dumps({"title": "deploy pipeline bounces"}))
        page = json.loads(
            similar(
                runner,
                "--draft-file",
                str(draft_path),
                "--all-tracker-projects",
                "--limit",
                "1",
                "--json",
            )
        )
        assert len(page["candidates"]) == 1


class TestParityWithTheApi:
    def test_cli_json_is_byte_identical_to_the_api_route_json(self, runner, api_client, tmp_path):
        """Wiring proof: the rendered CLI invocation and an identically
        shaped POST /tracker/issues/similar return the same JSON document."""
        seed_corpus()
        draft_path = tmp_path / "draft.json"
        draft_path.write_text(json.dumps({"title": "deploy pipeline bounces"}))

        through_cli = json.loads(
            similar(
                runner,
                "--draft-file",
                str(draft_path),
                "--all-tracker-projects",
                "--limit",
                "5",
                "--json",
            )
        )
        through_api = api_client.post(
            "/tracker/issues/similar",
            json={
                "draft": {"title": "deploy pipeline bounces"},
                "all_projects": True,
                "limit": 5,
            },
        )
        assert through_api.status_code == 200, through_api.text
        assert through_cli == through_api.json()

    def test_issue_key_mode_parity(self, runner, api_client):
        hit, *_ = seed_corpus()
        through_cli = json.loads(
            similar(
                runner,
                "--issue-key",
                hit["key"],
                "--tracker-project",
                "cao-system",
                "--json",
            )
        )
        through_api = api_client.post(
            "/tracker/issues/similar",
            json={"issue_key": hit["key"], "project_ids": ["cao-system"]},
        )
        assert through_api.status_code == 200, through_api.text
        assert through_cli == through_api.json()


class TestHumanRenderingAndRefusals:
    def test_human_mode_lists_hits_and_confirmed_duplicates(self, runner, tmp_path):
        hit, _, _, confirmed = seed_corpus()
        draft_path = tmp_path / "draft.json"
        draft_path.write_text(json.dumps({"title": "deploy pipeline bounces"}))
        out = similar(runner, "--draft-file", str(draft_path), "--all-tracker-projects")
        assert f"for the draft · kind bug" in out
        assert hit["key"] in out
        assert "mode hybrid→lexical" in out
        assert "coverage degraded" in out
        assert "probe draft" in out
        assert "confirmed duplicates of hits:" in out
        assert f"{hit['key']} <- {confirmed['key']}" in out

    def test_human_mode_surfaces_probe_audit_and_does_not_call_inconclusive_empty_zero(
        self, runner, tmp_path, monkeypatch
    ):
        draft_path = tmp_path / "draft.json"
        draft_path.write_text(json.dumps({"title": "offline similarity"}))
        monkeypatch.setattr(
            similar_service,
            "find_similar_issues",
            lambda request: {
                "query_source": {"mode": "draft", "issue_key": None, "kind": "bug"},
                "query": "offline similarity",
                "scope": {
                    "project_ids": ["cao-system"],
                    "all_projects": False,
                    "subtree_roots": [],
                    "subtree_closure_size": 0,
                },
                "mode_requested": "hybrid",
                "mode_effective": "lexical",
                "degradation": {
                    "requested_mode": "hybrid",
                    "effective_mode": "lexical",
                    "reasons": ["semantic unavailable"],
                    "lanes": {},
                },
                "coverage": {
                    "status": "inconclusive",
                    "complete": False,
                    "inconclusive": True,
                    "partial": False,
                    "probes_requested": 1,
                    "probes_completed": 1,
                    "probes_failed": 0,
                    "candidate_keys_seen": 0,
                },
                "diagnostics": {
                    "similarity_probe_failures": [],
                    "similarity_probes": [
                        {"label": "draft", "query": "offline similarity", "weight": 2.0}
                    ],
                },
                "generations": {},
                "include_comments": False,
                "limit": 20,
                "total": 0,
                "candidates": [],
                "duplicate_expansions": [],
            },
        )
        out = similar(
            runner,
            "--draft-file",
            str(draft_path),
            "--all-tracker-projects",
        )
        assert "mode hybrid→lexical" in out
        assert "coverage inconclusive" in out
        assert "degraded: semantic unavailable" in out
        assert "retrieval coverage is inconclusive" in out
        assert "0 similar issue(s)" not in out

    def test_neither_input_is_a_typed_refusal(self, runner):
        seed_corpus()
        result = runner.invoke(issue_cli.issue, ["similar", "--all-tracker-projects"])
        assert result.exit_code == 1
        assert "[invalid]" in result.output
        assert "exactly one of issue_key or draft" in result.output

    def test_both_inputs_are_a_typed_refusal(self, runner, tmp_path):
        hit, *_ = seed_corpus()
        draft_path = tmp_path / "draft.json"
        draft_path.write_text(json.dumps({"title": "deploy pipeline bounces"}))
        result = runner.invoke(
            issue_cli.issue,
            [
                "similar",
                "--issue-key",
                hit["key"],
                "--draft-file",
                str(draft_path),
                "--all-tracker-projects",
            ],
        )
        assert result.exit_code == 1
        assert "[invalid]" in result.output

    def test_both_scope_forms_are_a_typed_refusal(self, runner, tmp_path):
        draft_path = tmp_path / "draft.json"
        draft_path.write_text(json.dumps({"title": "deploy pipeline bounces"}))
        result = runner.invoke(
            issue_cli.issue,
            [
                "similar",
                "--draft-file",
                str(draft_path),
                "--tracker-project",
                "cao-system",
                "--all-tracker-projects",
            ],
        )
        assert result.exit_code == 1
        assert "[invalid-scope]" in result.output

    def test_an_invalid_json_draft_file_is_a_typed_refusal(self, runner, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        result = runner.invoke(
            issue_cli.issue,
            ["similar", "--draft-file", str(bad), "--all-tracker-projects"],
        )
        assert result.exit_code == 1
        assert "[invalid]" in result.output
        assert "unreadable draft file" in result.output
