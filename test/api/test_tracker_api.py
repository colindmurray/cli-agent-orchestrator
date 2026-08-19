"""HTTP surface for the issue tracker.

These tests exercise the boundary the dashboard and `conduct` actually talk
to: route ordering (a literal path must beat a parameterised one), the mapping
from a service refusal to an HTTP status a client can branch on, and PATCH
semantics where "field absent" and "field sent empty" mean different things.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import Base
from cli_agent_orchestrator.services import issue_tracker as tracker


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/tracker-api.db")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessionmaker(bind=engine))
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
    payload = {"project_id": "cao-system", "title": "a defect"}
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

    def test_an_empty_comment_is_400(self, client, project):
        issue = _issue(client)
        response = client.post(f"/tracker/issues/{issue['key']}/comments", json={"body": "  "})
        assert response.status_code == 400

    def test_a_link_round_trips_and_can_be_removed(self, client, project):
        a, b = _issue(client, title="a"), _issue(client, title="b")
        created = client.post(
            f"/tracker/issues/{a['key']}/links", json={"to_key": b["key"], "kind": "blocks"}
        )
        assert created.status_code == 201
        link_id = created.json()["id"]
        assert client.delete(f"/tracker/issues/{a['key']}/links/{link_id}").status_code == 200
        assert client.get(f"/tracker/issues/{b['key']}").json()["links"] == []

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
        # The default stays issue-only.
        default = client.get("/tracker/issues").json()
        assert [i["title"] for i in default["issues"]] == ["bare issue"]


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
