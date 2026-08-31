"""HTTP surface for the issue tracker.

These tests exercise the boundary the dashboard and `conduct` actually talk
to: route ordering (a literal path must beat a parameterised one), the mapping
from a service refusal to an HTTP status a client can branch on, and PATCH
semantics where "field absent" and "field sent empty" mean different things.
"""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import Base, TerminalModel
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services import project_dashboard


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/tracker-api.db")
    Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    monkeypatch.setattr(project_dashboard, "SessionLocal", sessions)
    yield
    engine.dispose()


@pytest.fixture
def project(client, tmp_path):
    repo = tmp_path / "cao-conductor"
    repo.mkdir()
    response = client.post(
        "/tracker/projects",
        json={
            "name": "CAO System",
            "id": "cao-system",
            "issue_prefix": "cond",
            "scopes": [
                {"kind": "path", "value": str(repo)},
                {"kind": "session", "value": "cao-p1-closure"},
            ],
        },
    )
    assert response.status_code == 201, response.text
    return {"body": response.json(), "repo": repo}


def _issue(client, **overrides):
    # Most API tests exercise behavior unrelated to filing policy. Explicitly
    # record that their tiny fixture is an incomplete bug so each test does not
    # obscure its subject with the three diagnostic fields.
    payload = {"project_id": "cao-system", "title": "a defect", "force": True}
    payload.update(overrides)
    response = client.post("/tracker/issues", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


class TestProjectRoutes:
    def test_a_created_project_is_listed(self, client, project):
        rows = client.get("/tracker/projects").json()
        assert [r["id"] for r in rows] == ["cao-system"]

    def test_project_detail_carries_scopes_and_counts(self, client, project):
        _issue(client)
        body = client.get("/tracker/projects/cao-system").json()
        assert body["counts"]["open"] == 1
        assert {s["kind"] for s in body["scopes"]} == {"path", "session"}

    def test_a_missing_project_is_404_not_400(self, client):
        assert client.get("/tracker/projects/nope").status_code == 404

    def test_a_duplicate_project_is_409(self, client, project):
        response = client.post("/tracker/projects", json={"name": "x", "id": "cao-system"})
        assert response.status_code == 409

    def test_an_invalid_slug_is_400(self, client):
        response = client.post("/tracker/projects", json={"name": "x", "id": "Not A Slug"})
        assert response.status_code == 400

    def test_archiving_hides_the_project_from_the_default_list(self, client, project):
        client.patch("/tracker/projects/cao-system", json={"status": "archived"})
        assert client.get("/tracker/projects").json() == []
        assert len(client.get("/tracker/projects?include_archived=true").json()) == 1

    def test_deleting_a_project_with_issues_is_409_without_force(self, client, project):
        _issue(client)
        assert client.delete("/tracker/projects/cao-system").status_code == 409
        assert client.delete("/tracker/projects/cao-system?force=true").status_code == 200


class TestProjectDashboardRoutes:
    def test_home_highlights_favorites_priority_and_session_counts(
        self, client, project, monkeypatch
    ):
        monkeypatch.setattr(
            project_dashboard,
            "get_backend",
            lambda: type("Backend", (), {"list_sessions": lambda self: []})(),
        )
        favorite = _issue(client, title="release project", kind="project", favorite=True)
        urgent = _issue(client, title="critical bug", severity="P0")

        body = client.get("/tracker/projects/cao-system/dashboard").json()
        assert body["issues"]["favorites"][0]["key"] == favorite["key"]
        assert body["issues"]["urgent"][0]["key"] == urgent["key"]
        assert body["sessions"]["total"] == 1
        assert body["sessions"]["active"] == 0
        assert body["sessions"]["historical"] == 1
        assert body["sessions"]["recent"][0]["name"] == "cao-p1-closure"

    def test_sessions_join_worker_history_and_archived_logs(
        self, client, project, monkeypatch, tmp_path
    ):
        log_dir = tmp_path / "terminal-logs"
        log_dir.mkdir()
        monkeypatch.setattr(project_dashboard, "TERMINAL_LOG_DIR", log_dir)
        monkeypatch.setattr(
            project_dashboard,
            "get_backend",
            lambda: type("Backend", (), {"list_sessions": lambda self: []})(),
        )
        with project_dashboard.SessionLocal() as db:
            db.add(
                TerminalModel(
                    id="deadbeef",
                    tmux_session="cao-p1-closure",
                    tmux_window="worker",
                    provider="codex",
                    agent_profile="reviewer",
                    caller_id="feedface",
                    native_session_id="native-1",
                )
            )
            db.commit()
        (log_dir / "deadbeef.snapshot.json").write_text(
            json.dumps(
                {
                    "terminal_id": "deadbeef",
                    "session_name": "cao-p1-closure",
                    "window_name": "worker",
                    "provider": "codex",
                    "agent_profile": "reviewer",
                    "caller_id": "feedface",
                    "working_directory": str(project["repo"]),
                }
            ),
            encoding="utf-8",
        )
        (log_dir / "deadbeef.scrollback").write_text("line one\nline two\n", encoding="utf-8")

        listed = client.get("/tracker/projects/cao-system/sessions").json()
        assert listed["total"] == 1
        assert listed["sessions"][0]["worker_count"] == 1
        assert listed["sessions"][0]["artifact_count"] == 2

        detail = client.get("/tracker/projects/cao-system/sessions/cao-p1-closure").json()
        worker = detail["session"]["terminals"][0]
        assert worker["terminal_id"] == "deadbeef"
        assert worker["native_session_id"] == "native-1"
        assert worker["caller_id"] == "feedface"
        assert worker["log_available"] is True

        log = client.get(
            "/tracker/projects/cao-system/sessions/cao-p1-closure/" "terminals/deadbeef/log"
        ).json()
        assert log["output"] == "line one\nline two"
        assert log["source"] == "archived-scrollback"


class TestResolveRouting:
    def test_resolve_is_not_swallowed_by_the_project_id_route(self, client, project):
        # `resolve` is a legal slug. If the parameterised route were declared
        # first, this would 404 as "no such project: resolve" and the CLI's
        # only way to ask "where would this file?" would silently break.
        body = client.get("/tracker/projects/resolve", params={"session": "cao-p1-closure"}).json()
        assert body == {
            "project_id": "cao-system",
            "matched_by": "session",
            "matched_value": "cao-p1-closure",
        }

    def test_resolve_reports_no_match_as_a_null_project_not_an_error(
        self, client, project, tmp_path
    ):
        body = client.get(
            "/tracker/projects/resolve", params={"cwd": str(tmp_path / "elsewhere")}
        ).json()
        assert body["project_id"] is None

    def test_resolve_with_an_unknown_explicit_project_is_404(self, client):
        assert (
            client.get("/tracker/projects/resolve", params={"project": "nope"}).status_code == 404
        )


class TestScopeRoutes:
    def test_a_scope_claimed_by_another_project_is_409(self, client, project):
        client.post("/tracker/projects", json={"name": "Other", "id": "other"})
        response = client.post(
            "/tracker/projects/other/scopes",
            json={"kind": "path", "value": str(project["repo"])},
        )
        assert response.status_code == 409

    def test_a_scope_can_be_removed(self, client, project):
        detail = client.get("/tracker/projects/cao-system").json()
        scope_id = detail["scopes"][0]["id"]
        assert client.delete(f"/tracker/projects/cao-system/scopes/{scope_id}").status_code == 200
        assert len(client.get("/tracker/projects/cao-system").json()["scopes"]) == 1

    def test_removing_another_projects_scope_is_404(self, client, project):
        client.post("/tracker/projects", json={"name": "Other", "id": "other"})
        scope_id = client.get("/tracker/projects/cao-system").json()["scopes"][0]["id"]
        assert client.delete(f"/tracker/projects/other/scopes/{scope_id}").status_code == 404


class TestIssueRoutes:
    def test_bug_filing_details_are_encouraged_with_an_explicit_override(self, client, project):
        refused = client.post(
            "/tracker/issues",
            json={"project_id": "cao-system", "title": "underspecified bug"},
        )
        assert refused.status_code == 400
        assert "reproduction_steps" in refused.json()["detail"]

        complete = client.post(
            "/tracker/issues",
            json={
                "project_id": "cao-system",
                "title": "complete bug",
                "reproduction_steps": "1. Start the server\n2. Reconnect",
                "expected_outcome": "The dashboard reconnects",
                "actual_outcome": "The dashboard remains disconnected",
                "favorite": True,
            },
        )
        assert complete.status_code == 201, complete.text
        assert complete.json()["kind"] == "bug"
        assert complete.json()["favorite"] is True

        forced = client.post(
            "/tracker/issues",
            json={
                "project_id": "cao-system",
                "title": "explicit exception",
                "force": True,
            },
        )
        assert forced.status_code == 201

    def test_public_item_type_vocabulary_includes_project_planning_levels(self, client):
        body = client.get("/tracker/vocabulary").json()
        assert body["item_kinds"] == [
            "project",
            "bug",
            "feature",
            "milestone",
            "goal",
            "epic",
            "story",
            "task",
        ]

    def test_non_bug_types_reject_bug_only_diagnostics(self, client, project):
        created = client.post(
            "/tracker/issues",
            json={"project_id": "cao-system", "title": "release goal", "kind": "goal"},
        )
        assert created.status_code == 201, created.text
        refused = client.post(
            "/tracker/issues",
            json={
                "project_id": "cao-system",
                "title": "not a bug",
                "kind": "project",
                "reproduction_steps": "this does not apply",
            },
        )
        assert refused.status_code == 400

    def test_filing_by_cwd_resolves_the_project(self, client, project):
        nested = project["repo"] / "conduct"
        nested.mkdir()
        body = _issue(client, project_id=None, cwd=str(nested))
        assert (body["project_id"], body["resolved_by"]) == ("cao-system", "path")

    def test_an_unresolvable_filing_site_is_422(self, client, project, tmp_path):
        response = client.post(
            "/tracker/issues", json={"title": "orphan", "cwd": str(tmp_path / "nowhere")}
        )
        assert response.status_code == 422

    def test_stats_is_not_swallowed_by_the_issue_key_route(self, client, project):
        _issue(client, severity="P1")
        body = client.get("/tracker/issues/stats", params={"project_id": "cao-system"}).json()
        assert body["by_severity"]["P1"] == 1

    def test_an_unknown_issue_is_404(self, client, project):
        assert client.get("/tracker/issues/cond-9999").status_code == 404

    def test_an_invalid_severity_is_400(self, client, project):
        response = client.post(
            "/tracker/issues", json={"project_id": "cao-system", "title": "x", "severity": "P9"}
        )
        assert response.status_code == 400

    def test_listing_filters_compose(self, client, project):
        _issue(client, title="one", severity="P1", component="conduct")
        _issue(client, title="two", severity="P3", component="conduct")
        _issue(client, title="three", severity="P1", component="fork")
        body = client.get(
            "/tracker/issues",
            params={"project_id": "cao-system", "severity": "P1", "component": "conduct"},
        ).json()
        assert [i["title"] for i in body["issues"]] == ["one"]

    def test_repeated_status_params_are_an_or(self, client, project):
        a = _issue(client, title="one")
        _issue(client, title="two")
        client.patch(f"/tracker/issues/{a['key']}", json={"status": "blocked"})
        body = client.get("/tracker/issues?status=blocked&status=open").json()
        assert body["total"] == 2

    def test_repeated_label_params_are_an_and(self, client, project):
        _issue(client, title="both", labels=["wayfinder:task", "initiative:alpha"])
        _issue(client, title="one", labels=["wayfinder:task"])
        body = client.get(
            "/tracker/issues",
            params=[
                ("project_id", "cao-system"),
                ("label", "wayfinder:task"),
                ("label", "initiative:alpha"),
            ],
        ).json()
        assert [issue["title"] for issue in body["issues"]] == ["both"]

    def test_reproduction_steps_round_trip_through_post_and_patch(self, client, project):
        issue = _issue(client, reproduction_steps="1. run the probe")
        assert issue["reproduction_steps"] == "1. run the probe"
        response = client.patch(
            f"/tracker/issues/{issue['key']}",
            json={"reproduction_steps": "1. run twice"},
        )
        assert response.status_code == 200
        assert response.json()["reproduction_steps"] == "1. run twice"

    def test_work_context_round_trips_through_post_and_patch(self, client, project):
        issue = _issue(
            client,
            collaborators=["codex:sess-1"],
            branches=["fix/a"],
            worktrees=["/tmp/wt-a"],
            pull_requests=["o/r#1"],
        )
        assert issue["collaborators"] == ["codex:sess-1"]
        response = client.patch(
            f"/tracker/issues/{issue['key']}",
            json={"collaborators": ["claude_code:sess-2"], "pull_requests": ["o/r#1", "o/r#2"]},
        )
        assert response.status_code == 200
        assert response.json()["collaborators"] == ["claude_code:sess-2"]
        assert response.json()["pull_requests"] == ["o/r#1", "o/r#2"]

    def test_reassignment_retains_the_former_owner_with_a_narrow_override(self, client, project):
        issue = _issue(client, assignee="codex:sess-1", collaborators=["colin"])
        response = client.patch(
            f"/tracker/issues/{issue['key']}",
            json={"assignee": "claude_code:sess-2"},
        )
        assert response.status_code == 200
        assert response.json()["collaborators"] == ["colin", "codex:sess-1"]

        issue = _issue(client, assignee="codex:sess-3", collaborators=["colin"])
        response = client.patch(
            f"/tracker/issues/{issue['key']}",
            json={
                "assignee": "claude_code:sess-4",
                "drop_previous_assignee": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["collaborators"] == ["colin"]

    def test_in_progress_assignment_policy_has_an_explicit_override(self, client, project):
        refused = client.post(
            "/tracker/issues",
            json={"project_id": "cao-system", "title": "active", "status": "in-progress"},
        )
        assert refused.status_code == 400
        forced = client.post(
            "/tracker/issues",
            json={
                "project_id": "cao-system",
                "title": "exception",
                "status": "in-progress",
                "force": True,
            },
        )
        assert forced.status_code == 201
        issue = _issue(client)
        active = client.patch(
            f"/tracker/issues/{issue['key']}",
            json={"status": "in-progress", "assignee": "colin"},
        )
        assert active.status_code == 200
        clear = client.patch(f"/tracker/issues/{issue['key']}", json={"assignee": ""})
        assert clear.status_code == 400
        override = client.patch(
            f"/tracker/issues/{issue['key']}", json={"assignee": "", "force": True}
        )
        assert override.status_code == 200

    def test_project_field_options_are_searchable_and_bounded(self, client, project):
        _issue(client, component="dashboard", labels=["initiative:ux"])
        _issue(client, component="conduct", labels=["other"])
        body = client.get(
            "/tracker/projects/cao-system/options",
            params={"field": "component", "q": "dash", "limit": 1},
        ).json()
        assert body["matching_total"] == 1
        assert body["options"][0]["value"] == "dashboard"

    def test_project_field_options_reject_unknown_fields(self, client, project):
        response = client.get("/tracker/projects/cao-system/options", params={"field": "status"})
        assert response.status_code == 400


class TestObservedRevisionRoutes:
    def test_filing_with_an_observed_revision_round_trips(self, client, project):
        created = client.post(
            "/tracker/issues",
            json={
                "project_id": "cao-system",
                "title": "a defect",
                "observed_revision": "v1.2.3",
                "force": True,
            },
        )
        assert created.status_code == 201, created.text
        assert created.json()["observed_revision"] == "v1.2.3"
        assert (
            client.get(f"/tracker/issues/{created.json()['key']}").json()["observed_revision"]
            == "v1.2.3"
        )

    def test_the_observed_revision_is_optional_and_defaults_to_null(self, client, project):
        issue = _issue(client)
        assert issue["observed_revision"] is None

    def test_the_observed_revision_is_patchable_and_clearable(self, client, project):
        issue = _issue(client)
        moved = client.patch(
            f"/tracker/issues/{issue['key']}", json={"observed_revision": "abc1234"}
        )
        assert moved.status_code == 200
        assert moved.json()["observed_revision"] == "abc1234"
        cleared = client.patch(f"/tracker/issues/{issue['key']}", json={"observed_revision": ""})
        assert cleared.json()["observed_revision"] is None


class TestPatchSemantics:
    def test_an_absent_field_is_untouched(self, client, project):
        issue = _issue(client, assignee="terra", severity="P2")
        body = client.patch(
            f"/tracker/issues/{issue['key']}", json={"status": "in-progress"}
        ).json()
        assert (body["assignee"], body["severity"]) == ("terra", "P2")

    def test_an_empty_string_clears_a_field(self, client, project):
        # This is the distinction that lets the dashboard unassign an issue.
        # Without model_fields_set, "" and "absent" would be indistinguishable.
        issue = _issue(client, assignee="terra")
        body = client.patch(f"/tracker/issues/{issue['key']}", json={"assignee": ""}).json()
        assert body["assignee"] is None

    def test_labels_are_replaced_wholesale(self, client, project):
        issue = _issue(client, labels=["a", "b"])
        body = client.patch(f"/tracker/issues/{issue['key']}", json={"labels": ["c"]}).json()
        assert body["labels"] == ["c"]

    def test_the_actor_is_recorded_and_not_treated_as_a_field(self, client, project):
        issue = _issue(client)
        client.patch(
            f"/tracker/issues/{issue['key']}", json={"status": "blocked", "actor": "colin"}
        )
        events = client.get(f"/tracker/issues/{issue['key']}").json()["events"]
        field_events = [e for e in events if e["kind"] == "field"]
        assert [(e["field"], e["actor"]) for e in field_events] == [("status", "colin")]


class TestCommentsAndLinksRoutes:
    def test_a_comment_round_trips(self, client, project):
        issue = _issue(client)
        assert (
            client.post(
                f"/tracker/issues/{issue['key']}/comments",
                json={"body": "reproduced", "author": "colin"},
            ).status_code
            == 201
        )
        detail = client.get(f"/tracker/issues/{issue['key']}").json()
        assert [c["body"] for c in detail["comments"]] == ["reproduced"]
        assert [c["important"] for c in detail["comments"]] == [False]

    def test_an_empty_comment_is_400(self, client, project):
        issue = _issue(client)
        response = client.post(f"/tracker/issues/{issue['key']}/comments", json={"body": "  "})
        assert response.status_code == 400

    def test_a_comment_can_be_created_important(self, client, project):
        issue = _issue(client)
        created = client.post(
            f"/tracker/issues/{issue['key']}/comments",
            json={"body": "root cause", "important": True},
        )
        assert created.status_code == 201
        detail = client.get(f"/tracker/issues/{issue['key']}").json()
        assert detail["comments"][0]["important"] is True

    def test_comment_forwards_the_parent_clock_fence_and_returns_its_effect(self, client, project):
        issue = _issue(client)
        created = client.post(
            f"/tracker/issues/{issue['key']}/comments",
            json={
                "body": "audited",
                "important": True,
                "expected_updated_at": issue["updated_at"],
            },
        )
        assert created.status_code == 201
        result = created.json()
        assert result["important"] is True
        assert result["effect_id"] > 0
        assert (
            result["updated_at"]
            == client.get(f"/tracker/issues/{issue['key']}").json()["updated_at"]
        )

    def test_stale_comment_clock_is_a_conflict_without_a_comment(self, client, project):
        issue = _issue(client)
        client.patch(f"/tracker/issues/{issue['key']}", json={"body": "changed"})
        response = client.post(
            f"/tracker/issues/{issue['key']}/comments",
            json={"body": "stale", "expected_updated_at": issue["updated_at"]},
        )
        assert response.status_code == 409
        assert client.get(f"/tracker/issues/{issue['key']}").json()["comments"] == []

    def test_comment_importance_set_clear_and_retry(self, client, project):
        issue = _issue(client)
        comment = client.post(
            f"/tracker/issues/{issue['key']}/comments", json={"body": "note"}
        ).json()
        url = f"/tracker/issues/{issue['key']}/comments/{comment['id']}"

        set_response = client.patch(url, json={"important": True, "actor": "colin"})
        assert set_response.status_code == 200
        assert set_response.json()["changed"] is True

        retry = client.patch(url, json={"important": True})
        assert retry.status_code == 200
        assert retry.json()["changed"] is False

        cleared = client.patch(url, json={"important": False, "actor": "colin"})
        assert cleared.json()["changed"] is True

        events = [
            e
            for e in client.get(f"/tracker/issues/{issue['key']}").json()["events"]
            if e["kind"] == "comment-field"
        ]
        assert [(e["field"], e["old_value"], e["new_value"], e["actor"]) for e in events] == [
            ("important", "false", "true", "colin"),
            ("important", "true", "false", "colin"),
        ]

    def test_importance_patch_on_an_unknown_comment_is_404(self, client, project):
        issue = _issue(client)
        response = client.patch(
            f"/tracker/issues/{issue['key']}/comments/9999", json={"important": True}
        )
        assert response.status_code == 404

    def test_deleting_a_comment_records_the_audit_and_bumps_the_parent(self, client, project):
        import time as _time

        issue = _issue(client)
        before = client.get(f"/tracker/issues/{issue['key']}").json()
        comment = client.post(
            f"/tracker/issues/{issue['key']}/comments", json={"body": "soon gone"}
        ).json()
        # A distinct clock tick so the bump is observable even within one second.
        _time.sleep(0.01)

        response = client.delete(
            f"/tracker/issues/{issue['key']}/comments/{comment['id']}?actor=colin"
        )
        assert response.status_code == 200

        after = client.get(f"/tracker/issues/{issue['key']}").json()
        assert after["comments"] == []
        assert after["updated_at"] > before["updated_at"]
        deletions = [e for e in after["events"] if e["kind"] == "comment-deleted"]
        assert len(deletions) == 1
        assert deletions[0]["old_value"] == "soon gone"
        assert deletions[0]["new_value"] == str(comment["id"])
        assert deletions[0]["actor"] == "colin"

    def test_feature_comment_surfaces_share_the_importance_contract(self, client, project):
        created = client.post(
            "/tracker/features",
            json={"project_id": "cao-system", "title": "a feature"},
        )
        assert created.status_code == 201
        feature_key = created.json()["key"]
        comment = client.post(
            f"/tracker/features/{feature_key}/comments",
            json={"body": "note", "important": True},
        ).json()
        assert comment["important"] is True

        patched = client.patch(
            f"/tracker/features/{feature_key}/comments/{comment['id']}",
            json={"important": False},
        )
        assert patched.status_code == 200
        assert patched.json()["changed"] is True
        detail = client.get(f"/tracker/features/{feature_key}").json()
        assert detail["comments"][0]["important"] is False

    def test_a_link_round_trips_and_can_be_removed(self, client, project):
        a, b = _issue(client, title="a"), _issue(client, title="b")
        created = client.post(
            f"/tracker/issues/{a['key']}/links", json={"to_key": b["key"], "kind": "blocks"}
        )
        assert created.status_code == 201
        link_id = created.json()["id"]
        assert client.delete(f"/tracker/issues/{a['key']}/links/{link_id}").status_code == 200
        assert client.get(f"/tracker/issues/{b['key']}").json()["links"] == []

    def test_link_forwards_both_endpoint_clocks_and_returns_both_new_clocks(self, client, project):
        a, b = _issue(client, title="a"), _issue(client, title="b")
        created = client.post(
            f"/tracker/issues/{a['key']}/links",
            json={
                "to_key": b["key"],
                "kind": "blocks",
                "expected_from_updated_at": a["updated_at"],
                "expected_to_updated_at": b["updated_at"],
            },
        )
        assert created.status_code == 201
        result = created.json()
        assert len(result["effect_ids"]) == 2
        assert (
            result["from_updated_at"]
            == client.get(f"/tracker/issues/{a['key']}").json()["updated_at"]
        )
        assert (
            result["to_updated_at"]
            == client.get(f"/tracker/issues/{b['key']}").json()["updated_at"]
        )

    def test_link_forwards_the_second_endpoint_clock(self, client, project):
        a, b = _issue(client, title="a"), _issue(client, title="b")
        client.patch(f"/tracker/issues/{b['key']}", json={"body": "changed"})
        response = client.post(
            f"/tracker/issues/{a['key']}/links",
            json={
                "to_key": b["key"],
                "kind": "blocks",
                "expected_from_updated_at": a["updated_at"],
                "expected_to_updated_at": b["updated_at"],
            },
        )
        assert response.status_code == 409
        assert client.get(f"/tracker/issues/{a['key']}").json()["links"] == []

    def test_an_unknown_link_kind_is_400(self, client, project):
        a, b = _issue(client, title="a"), _issue(client, title="b")
        response = client.post(
            f"/tracker/issues/{a['key']}/links", json={"to_key": b["key"], "kind": "supersedes"}
        )
        assert response.status_code == 400


class TestVocabularyAndExport:
    def test_the_vocabulary_is_served_rather_than_duplicated_client_side(self, client):
        body = client.get("/tracker/vocabulary").json()
        assert body["statuses"] == list(tracker.STATUSES)
        assert body["severities"] == list(tracker.SEVERITIES)

    def test_the_export_returns_markdown(self, client, project):
        _issue(client, title="event-mirror traceback", severity="P2")
        response = client.get("/tracker/projects/cao-system/export")
        assert response.headers["content-type"].startswith("text/markdown")
        assert "## cond-0001 — [P2] event-mirror traceback" in response.text


class TestIssueSnapshotRoute:
    def test_literal_snapshot_route_precedes_issue_key_and_returns_the_service_contract(
        self, client, project
    ):
        issue = _issue(client, title="selected")

        response = client.post(
            "/tracker/issues/snapshot",
            json={"project_id": "cao-system", "keys": [issue["key"]]},
        )

        assert response.status_code == 200, response.text
        assert response.json()["selected_keys"] == [issue["key"]]
        route_paths = [route.path for route in client.app.routes]
        assert route_paths.index("/tracker/issues/snapshot") < route_paths.index(
            "/tracker/issues/{issue_key}"
        )

    def test_snapshot_body_is_strict_and_service_refusals_keep_their_http_type(
        self, client, project
    ):
        unknown = client.post(
            "/tracker/issues/snapshot",
            json={"project_id": "cao-system", "keys": ["cond-0001"], "key": "cond-0001"},
        )
        assert unknown.status_code == 422
        assert "key" in unknown.text

        missing = client.post(
            "/tracker/issues/snapshot",
            json={"project_id": "cao-system", "keys": ["cond-9999"]},
        )
        assert missing.status_code == 404
        assert missing.json()["detail"]["missing_keys"] == ["cond-9999"]


class TestUnknownFieldsAreRefused:
    """An unknown field is a 422 naming it, never a silently ignored 200.

    Pydantic drops what it does not recognise by default, so
    `PATCH {"project_id": "other"}` looked like it moved an issue between
    projects and did nothing at all — a 200 for an operation that never
    happened.
    """

    def test_patching_a_non_editable_field_is_refused(self, client, project):
        issue = _issue(client)
        response = client.patch(f"/tracker/issues/{issue['key']}", json={"project_id": "other"})
        assert response.status_code == 422
        assert "project_id" in response.text

    def test_a_misspelled_field_is_refused(self, client, project):
        issue = _issue(client)
        assert (
            client.patch(f"/tracker/issues/{issue['key']}", json={"assigne": "x"}).status_code
            == 422
        )

    def test_an_unknown_field_on_create_is_refused(self, client, project):
        response = client.post(
            "/tracker/issues", json={"project_id": "cao-system", "title": "x", "sevrity": "P1"}
        )
        assert response.status_code == 422

    def test_an_out_of_range_limit_is_refused_not_truncated(self, client, project):
        # Silently returning 500 of 100000 requested rows reads as "that was
        # everything".
        assert client.get("/tracker/issues?limit=100000").status_code == 422

    def test_a_prefix_already_in_use_is_409(self, client, project):
        response = client.post(
            "/tracker/projects", json={"name": "Other", "id": "other", "issue_prefix": "cond"}
        )
        assert response.status_code == 409
        assert "cao-system" in response.json()["detail"]


class TestMapMembershipRoutes:
    def test_part_of_link_round_trips_and_children_lists_members(self, client, project):
        m = _issue(client, title="map")
        a, b = _issue(client, title="a"), _issue(client, title="b")
        created = client.post(
            f"/tracker/issues/{a['key']}/links",
            json={"to_key": m["key"], "kind": "part-of"},
        )
        assert created.status_code == 201
        client.post(
            f"/tracker/issues/{b['key']}/links", json={"to_key": m["key"], "kind": "part-of"}
        )
        body = client.get(f"/tracker/issues/{m['key']}/children").json()
        assert [c["title"] for c in body["children"]] == ["a", "b"]
        # Raw JSON direction stays explicit: from the child, to the map.
        link = client.get(f"/tracker/issues/{m['key']}").json()["links"][0]
        assert (link["kind"], link["from_key"], link["to_key"]) == ("part-of", a["key"], m["key"])

    def test_children_of_an_unknown_issue_is_404(self, client, project):
        assert client.get("/tracker/issues/cond-9999/children").status_code == 404

    def test_the_vocabulary_offers_part_of(self, client):
        body = client.get("/tracker/vocabulary").json()
        assert "part-of" in body["link_kinds"]


class TestFrontierRoute:
    def test_frontier_returns_only_takeable_children(self, client, project):
        m = _issue(client, title="map")
        a, b, c = (_issue(client, title=t) for t in ("a", "b", "c"))
        for t in (a, b, c):
            client.post(
                f"/tracker/issues/{t['key']}/links", json={"to_key": m["key"], "kind": "part-of"}
            )
        # Claim b (assigned), block c with an open ticket; a stays takeable.
        client.patch(f"/tracker/issues/{b['key']}", json={"assignee": "terra"})
        client.post(
            f"/tracker/issues/{a['key']}/links", json={"to_key": c["key"], "kind": "blocks"}
        )
        body = client.get(f"/tracker/issues/{m['key']}/frontier").json()
        assert [t["title"] for t in body["frontier"]] == ["a"]

    def test_frontier_of_an_unknown_issue_is_404(self, client, project):
        assert client.get("/tracker/issues/cond-9999/frontier").status_code == 404


class TestClaimRoutes:
    def test_claim_assigns_the_issue(self, client, project):
        issue = _issue(client)
        response = client.post(f"/tracker/issues/{issue['key']}/claim", json={"claimant": "terra"})
        assert response.status_code == 200
        body = response.json()
        assert (body["assignee"], body["claimed"], body["already_claimed"]) == (
            "terra",
            True,
            False,
        )

    def test_a_second_claim_is_a_409_reporting_the_observed_claimant(self, client, project):
        issue = _issue(client)
        client.post(f"/tracker/issues/{issue['key']}/claim", json={"claimant": "terra"})
        response = client.post(f"/tracker/issues/{issue['key']}/claim", json={"claimant": "muse"})
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["observed_assignee"] == "terra"
        assert "terra" in detail["message"]

    def test_a_retry_by_the_current_claimant_is_idempotent(self, client, project):
        issue = _issue(client)
        client.post(f"/tracker/issues/{issue['key']}/claim", json={"claimant": "terra"})
        again = client.post(f"/tracker/issues/{issue['key']}/claim", json={"claimant": "terra"})
        assert again.status_code == 200
        assert again.json()["already_claimed"] is True

    def test_unclaim_is_the_exit_that_lets_another_worker_in(self, client, project):
        issue = _issue(client)
        client.post(f"/tracker/issues/{issue['key']}/claim", json={"claimant": "terra"})
        released = client.post(f"/tracker/issues/{issue['key']}/unclaim", json={"actor": "colin"})
        assert released.status_code == 200
        assert released.json()["assignee"] is None
        reclaim = client.post(f"/tracker/issues/{issue['key']}/claim", json={"claimant": "muse"})
        assert reclaim.status_code == 200

    def test_claiming_a_terminal_issue_is_409(self, client, project):
        issue = _issue(client, status="closed")
        response = client.post(f"/tracker/issues/{issue['key']}/claim", json={"claimant": "terra"})
        assert response.status_code == 409

    def test_claiming_an_unknown_issue_is_404(self, client, project):
        response = client.post("/tracker/issues/cond-9999/claim", json={"claimant": "terra"})
        assert response.status_code == 404

    def test_a_claim_without_a_claimant_is_422_or_400(self, client, project):
        issue = _issue(client)
        response = client.post(f"/tracker/issues/{issue['key']}/claim", json={})
        assert response.status_code in (400, 422)


class TestOptimisticUpdateRoute:
    def test_a_matching_expected_updated_at_applies(self, client, project):
        issue = _issue(client, body="v1")
        response = client.patch(
            f"/tracker/issues/{issue['key']}",
            json={"body": "v2", "expected_updated_at": issue["updated_at"]},
        )
        assert response.status_code == 200
        assert response.json()["body"] == "v2"

    def test_a_stale_expected_updated_at_is_409_with_the_current_version(self, client, project):
        issue = _issue(client, body="v1")
        stale = issue["updated_at"]
        current = client.patch(f"/tracker/issues/{issue['key']}", json={"body": "other"}).json()
        response = client.patch(
            f"/tracker/issues/{issue['key']}",
            json={"body": "v2", "expected_updated_at": stale},
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["current_updated_at"] == current["updated_at"]
        # The refused write changed nothing.
        assert client.get(f"/tracker/issues/{issue['key']}").json()["body"] == "other"

    def test_an_edit_without_the_precondition_stays_unconditional(self, client, project):
        issue = _issue(client, body="v1")
        client.patch(f"/tracker/issues/{issue['key']}", json={"body": "v2"})
        response = client.patch(f"/tracker/issues/{issue['key']}", json={"body": "v3"})
        assert response.status_code == 200


class TestAtomicLabelRoutes:
    def test_add_labels_merges_via_patch(self, client, project):
        issue = _issue(client, labels=["bug"])
        body = client.patch(
            f"/tracker/issues/{issue['key']}", json={"add_labels": ["needs-triage"]}
        ).json()
        assert body["labels"] == ["bug", "needs-triage"]

    def test_remove_and_clear_labels_via_patch(self, client, project):
        issue = _issue(client, labels=["a", "b", "c"])
        body = client.patch(f"/tracker/issues/{issue['key']}", json={"remove_labels": ["b"]}).json()
        assert body["labels"] == ["a", "c"]
        body = client.patch(f"/tracker/issues/{issue['key']}", json={"clear_labels": True}).json()
        assert body["labels"] == []

    def test_full_replacement_plus_a_delta_is_400(self, client, project):
        issue = _issue(client, labels=["a"])
        response = client.patch(
            f"/tracker/issues/{issue['key']}", json={"labels": ["b"], "add_labels": ["c"]}
        )
        assert response.status_code == 400
        assert client.get(f"/tracker/issues/{issue['key']}").json()["labels"] == ["a"]

    def test_the_resulting_label_set_is_audited(self, client, project):
        issue = _issue(client, labels=["a"])
        client.patch(
            f"/tracker/issues/{issue['key']}",
            json={"add_labels": ["b"], "remove_labels": ["a"], "actor": "colin"},
        )
        events = client.get(f"/tracker/issues/{issue['key']}").json()["events"]
        label_events = [e for e in events if e["kind"] == "field" and e["field"] == "labels"]
        assert len(label_events) == 1
        assert label_events[0]["actor"] == "colin"


class TestTriageDiscoveryRoutes:
    def test_the_unlabeled_filter(self, client, project):
        _issue(client, title="bare")
        _issue(client, title="tagged", labels=["bug"])
        body = client.get("/tracker/issues", params={"unlabeled": True}).json()
        assert [i["title"] for i in body["issues"]] == ["bare"]

    def test_unlabeled_composes_with_kind_and_the_total_stays_unpaged(self, client, project):
        _issue(client, title="bare issue")
        client.post("/tracker/features", json={"project_id": "cao-system", "title": "bare feature"})
        body = client.get(
            "/tracker/issues", params={"unlabeled": True, "kind": "all", "limit": 1}
        ).json()
        assert body["total"] == 2
        assert len(body["issues"]) == 1
        only_features = client.get(
            "/tracker/issues", params={"unlabeled": True, "kind": "feature"}
        ).json()
        assert [i["title"] for i in only_features["issues"]] == ["bare feature"]
        # The default stays bug-only.
        default = client.get("/tracker/issues").json()
        assert [i["title"] for i in default["issues"]] == ["bare issue"]

    def test_repeated_without_label_excludes_any_exact_match(self, client, project):
        _issue(client, title="ready", labels=["source:wayfinder"])
        _issue(client, title="triaged", labels=["needs-triage"])
        _issue(client, title="waiting", labels=["needs-info"])
        _issue(client, title="similar", labels=["needs-info-extra"])

        body = client.get(
            "/tracker/issues",
            params=[("without_label", "needs-triage"), ("without_label", "needs-info")],
        ).json()

        assert {i["title"] for i in body["issues"]} == {"ready", "similar"}


class TestLabelFacetRoute:
    def test_label_counts_for_a_project(self, client, project):
        _issue(client, title="a", labels=["effort:maps", "wayfinder:map"])
        _issue(client, title="b", labels=["effort:maps"])
        _issue(client, title="bare")
        body = client.get("/tracker/projects/cao-system/labels").json()
        by_label = {f["label"]: f for f in body["labels"]}
        assert by_label["effort:maps"]["total"] == 2
        assert by_label["wayfinder:map"]["open"] == 1
        assert body["unlabeled"] == 1

    def test_labels_of_an_unknown_project_is_404(self, client, project):
        assert client.get("/tracker/projects/nope/labels").status_code == 404


class TestMapProjectionRoute:
    def test_the_projection_round_trips(self, client, project):
        m = _issue(client, title="the map", labels=["wayfinder:map"])
        a, b = _issue(client, title="a"), _issue(client, title="b")
        for t in (a, b):
            client.post(
                f"/tracker/issues/{t['key']}/links", json={"to_key": m["key"], "kind": "part-of"}
            )
        client.post(
            f"/tracker/issues/{a['key']}/links", json={"to_key": b["key"], "kind": "blocks"}
        )
        body = client.get(f"/tracker/issues/{m['key']}/map").json()
        assert body["map"]["key"] == m["key"]
        assert body["progress"]["total"] == 2
        assert body["frontier"] == [a["key"]]
        children = {c["key"]: c for c in body["children"]}
        assert children[b["key"]]["blocked_by"] == [a["key"]]
        assert children[a["key"]]["frontier"] is True

    def test_map_of_an_unknown_issue_is_404(self, client, project):
        assert client.get("/tracker/issues/cond-9999/map").status_code == 404


class TestGraphProjectionRoute:
    def test_graph_returns_transitive_hierarchy_and_relationship_context(self, client, project):
        root = _issue(client, title="root", kind="project")
        child = _issue(client, title="child", kind="milestone")
        leaf = _issue(client, title="leaf", kind="task")
        outside = _issue(client, title="outside")
        client.post(
            f"/tracker/issues/{child['key']}/links",
            json={"to_key": root["key"], "kind": "part-of"},
        )
        client.post(
            f"/tracker/issues/{leaf['key']}/links",
            json={"to_key": child["key"], "kind": "part-of"},
        )
        client.post(
            f"/tracker/issues/{outside['key']}/links",
            json={"to_key": leaf["key"], "kind": "blocks"},
        )
        response = client.get(f"/tracker/issues/{root['key']}/graph")
        assert response.status_code == 200
        body = response.json()
        assert [(row["title"], row["depth"]) for row in body["nodes"]] == [
            ("root", 0),
            ("child", 1),
            ("leaf", 2),
        ]
        assert [row["title"] for row in body["external"]] == ["outside"]

    def test_graph_bounds_are_validated_and_unknown_root_is_404(self, client, project):
        issue = _issue(client)
        assert (
            client.get(f"/tracker/issues/{issue['key']}/graph", params={"max_depth": 0}).status_code
            == 422
        )
        assert client.get("/tracker/issues/cond-9999/graph").status_code == 404

    def test_hierarchy_audit_returns_recursive_frontier(self, client, project):
        root = _issue(client, title="root", kind="project")
        child = _issue(client, title="child", kind="task")
        client.post(
            f"/tracker/issues/{child['key']}/links",
            json={"to_key": root["key"], "kind": "part-of"},
        )

        response = client.get(f"/tracker/issues/{root['key']}/audit")

        assert response.status_code == 200
        payload = response.json()
        assert payload["counts"]["nodes"] == 2
        assert [row["key"] for row in payload["frontier"]] == [child["key"]]


class TestMapProjectionRouteExternalEndpoints:
    def test_every_external_link_endpoint_is_served_with_its_blocking_role(self, client, project):
        m = _issue(client, title="the map", labels=["wayfinder:map"])
        a = _issue(client, title="a")
        client.post(
            f"/tracker/issues/{a['key']}/links", json={"to_key": m["key"], "kind": "part-of"}
        )
        blocker = _issue(client, title="outside blocker")
        neighbour = _issue(client, title="related work")
        client.post(
            f"/tracker/issues/{blocker['key']}/links", json={"to_key": a["key"], "kind": "blocks"}
        )
        client.post(
            f"/tracker/issues/{a['key']}/links",
            json={"to_key": neighbour["key"], "kind": "relates"},
        )
        body = client.get(f"/tracker/issues/{m['key']}/map").json()
        external = {e["key"]: e for e in body["external"]}
        # Both endpoints are materialized — the relates neighbour is visible
        # exactly like the blocker — and `blocking` tells them apart.
        assert external[blocker["key"]]["blocking"] == [a["key"]]
        assert external[neighbour["key"]]["blocking"] == []


class TestFeatureParity:
    """Feature edits share the issue machinery, so the deltas and the CAS
    precondition must work there too — otherwise the dashboard would be lying
    about generic support depending on kind."""

    def test_feature_patch_accepts_label_deltas(self, client, project):
        client.post(
            "/tracker/features", json={"project_id": "cao-system", "title": "wish", "labels": ["a"]}
        )
        body = client.patch(
            "/tracker/features/cond-0001", json={"add_labels": ["b"], "remove_labels": ["a"]}
        ).json()
        assert body["labels"] == ["b"]

    def test_feature_patch_honors_expected_updated_at(self, client, project):
        created = client.post(
            "/tracker/features", json={"project_id": "cao-system", "title": "wish"}
        ).json()
        ok = client.patch(
            "/tracker/features/cond-0001",
            json={"body": "v2", "expected_updated_at": created["updated_at"]},
        )
        assert ok.status_code == 200
        stale = client.patch(
            "/tracker/features/cond-0001",
            json={"body": "v3", "expected_updated_at": created["updated_at"]},
        )
        assert stale.status_code == 409
        assert stale.json()["detail"]["current_updated_at"] == ok.json()["updated_at"]

    def test_the_features_list_accepts_unlabeled(self, client, project):
        client.post("/tracker/features", json={"project_id": "cao-system", "title": "bare"})
        client.post(
            "/tracker/features",
            json={"project_id": "cao-system", "title": "tagged", "labels": ["x"]},
        )
        body = client.get("/tracker/features", params={"unlabeled": True}).json()
        assert [i["title"] for i in body["issues"]] == ["bare"]

    def test_the_features_list_accepts_without_label(self, client, project):
        client.post(
            "/tracker/features",
            json={
                "project_id": "cao-system",
                "title": "ready",
                "labels": ["source:wayfinder"],
            },
        )
        client.post(
            "/tracker/features",
            json={"project_id": "cao-system", "title": "waiting", "labels": ["needs-info"]},
        )
        body = client.get("/tracker/features", params={"without_label": "needs-info"}).json()
        assert [i["title"] for i in body["issues"]] == ["ready"]
