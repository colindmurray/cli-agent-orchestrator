"""The ranked-search HTTP surface (design §12.1, §19.2 rows via the API).

The contract under test: the literal ``/tracker/issues/search`` route wins
over ``/tracker/issues/{issue_key}``; repeated query parameters become
repeated filter values; scope is exactly one of ``project_id`` or
``all_projects`` with both/neither refused as a typed invalid request;
uninstalled modes degrade visibly instead of silently; pagination bounds are
enforced at the boundary; and a read-scoped search writes nothing.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import (
    Base,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services import tracker_ranked_search as ranked


@pytest.fixture(autouse=True)
def search_db(tmp_path, monkeypatch):
    """A file-backed tracker store with the search projection installed."""
    engine = create_engine(f"sqlite:///{tmp_path}/search-api.db")
    Base.metadata.create_all(bind=engine)
    _migrate_tracker_search_projection(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    monkeypatch.setattr(ranked, "SessionLocal", sessions)
    yield engine
    engine.dispose()


@pytest.fixture
def project(client):
    response = client.post(
        "/tracker/projects",
        json={"name": "CAO System", "id": "cao-system", "issue_prefix": "cond"},
    )
    assert response.status_code == 201, response.text


def _issue(client, **overrides):
    payload = {"project_id": "cao-system", "title": "a defect", "force": True}
    payload.update(overrides)
    response = client.post("/tracker/issues", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _search(client, q="deploy", **params):
    return client.get("/tracker/issues/search", params={"q": q, **params})


def _keys(response):
    return [row["issue"]["key"] for row in response.json()["results"]]


class TestRouteOrderAndScope:
    def test_search_resolves_before_the_issue_key_route(self, client, project):
        filed = _issue(client, title="deploy pipeline bounces on dry run")
        response = _search(client, q="deploy pipeline", project_id="cao-system")
        assert response.status_code == 200, response.text
        body = response.json()
        # The parameterised route would have answered with the raw issue dict
        # (no results/degradation envelope) or a 404; the literal route won.
        assert body["results"][0]["issue"]["key"] == filed["key"]
        assert body["mode_effective"] == "lexical"
        assert set(body) >= {
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

    def test_neither_scope_form_is_a_typed_invalid_request(self, client, project):
        response = _search(client, q="deploy")
        assert response.status_code == 400
        assert "exactly one scope form" in response.json()["detail"]

    def test_both_scope_forms_is_a_typed_invalid_request(self, client, project):
        response = _search(client, q="deploy", project_id="cao-system", all_projects="true")
        assert response.status_code == 400
        assert "exactly one scope form" in response.json()["detail"]

    def test_an_unknown_subtree_root_is_scoped_to_the_request_not_a_500(self, client, project):
        response = _search(client, q="deploy", all_projects="true", under="cond-9999")
        assert response.status_code == 400
        assert "unknown subtree root" in response.json()["detail"]


class TestRepeatedParameters:
    def test_repeated_project_ids_span_projects(self, client, project):
        other = client.post(
            "/tracker/projects",
            json={"name": "Other", "id": "other", "issue_prefix": "oth"},
        )
        assert other.status_code == 201, other.text
        here = _issue(client, title="deploy pipeline bounces")
        there = _issue(client, project_id="other", title="deploy rollback torn")

        both = _search(client, q="deploy", project_id=["cao-system", "other"])
        assert here["key"] in _keys(both)
        assert there["key"] in _keys(both)

        only_here = _search(client, q="deploy", project_id="cao-system")
        assert here["key"] in _keys(only_here)
        assert there["key"] not in _keys(only_here)

    def test_repeated_kind_params_or_compose(self, client, project):
        bug = _issue(client, title="deploy pipeline bounces", kind="bug")
        task = _issue(client, title="deploy the pipeline again", kind="task")
        _issue(client, title="deploy dashboard wish", kind="feature")

        response = _search(client, q="deploy", all_projects="true", kind=["bug", "task"])
        keys = _keys(response)
        assert bug["key"] in keys
        assert task["key"] in keys
        assert len(keys) == 2

    def test_repeated_label_params_and_compose_and_without_label_excludes(self, client, project):
        both = _issue(client, title="deploy pipeline bounces", labels=["deploy", "infra"])
        only_deploy = _issue(client, title="deploy rollback torn", labels=["deploy"])
        _issue(client, title="deploy dashboard wish", labels=["ui"])

        required = _search(client, q="deploy", all_projects="true", label=["deploy", "infra"])
        assert _keys(required) == [both["key"]]

        excluded = _search(
            client,
            q="deploy",
            all_projects="true",
            label=["deploy"],
            without_label=["infra"],
        )
        assert excluded.json()["total"] == 1
        assert _keys(excluded) == [only_deploy["key"]]

    def test_observed_revision_is_exactly_filterable(self, client, project):
        pinned = _issue(
            client,
            title="deploy pipeline bounces",
            observed_revision="v1.2.3",
        )
        _issue(client, title="deploy rollback torn", observed_revision="v4.5.6")

        exact = _search(client, q="deploy", all_projects="true", observed_revision=["v1.2.3"])
        assert _keys(exact) == [pinned["key"]]

        repeated = _search(
            client,
            q="deploy",
            all_projects=True,
            observed_revision=["v1.2.3", "v4.5.6"],
        )
        assert repeated.json()["total"] == 2

        unknown = _search(client, q="deploy", all_projects="true", observed_revision=["v9.9.9"])
        assert unknown.status_code == 200
        assert unknown.json()["total"] == 0

    def test_no_comments_disables_the_comment_bm25_lane(self, client, project):
        quiet = _issue(client, title="quiet body only")
        client.post(
            f"/tracker/issues/{quiet['key']}/comments",
            json={"body": "the zephyr word lives only in this comment"},
        )

        with_comments = _search(client, q="zephyr", all_projects="true").json()
        assert [row["issue"]["key"] for row in with_comments["results"]] == [quiet["key"]]
        lanes_with = [lane["lane"] for lane in with_comments["results"][0]["contributing_lanes"]]
        assert "comment-bm25" in lanes_with
        assert with_comments["results"][0]["winning_comment"] is not None

        # --include-comments=false removes the comment-bm25 lane only; the
        # exact lane still sees comment bodies, so the issue remains findable
        # while its comment evidence stops being explained.
        without = _search(client, q="zephyr", all_projects="true", include_comments="false")
        assert without.status_code == 200
        result = without.json()["results"][0]
        assert result["issue"]["key"] == quiet["key"]
        assert "comment-bm25" not in [lane["lane"] for lane in result["contributing_lanes"]]
        assert result["winning_comment"] is None


class TestDegradationAndBounds:
    def test_uninstalled_modes_degrade_visibly(self, client, project):
        _issue(client, title="deploy pipeline bounces")
        response = _search(client, q="deploy", all_projects="true", mode="hybrid")
        assert response.status_code == 200
        body = response.json()
        assert body["mode_requested"] == "hybrid"
        assert body["mode_effective"] == "lexical"
        assert body["degradation"]["requested_mode"] == "hybrid"
        assert body["degradation"]["effective_mode"] == "lexical"
        assert body["degradation"]["reasons"]
        lanes = body["degradation"]["lanes"]
        assert lanes["semantic-issue"]["available"] is False
        assert lanes["issue-bm25"]["available"] is True

    def test_default_mode_is_lexical_with_no_degradation(self, client, project):
        _issue(client, title="deploy pipeline bounces")
        body = _search(client, q="deploy", all_projects="true").json()
        assert body["mode_requested"] == "lexical"
        assert body["mode_effective"] == "lexical"
        assert body["degradation"]["reasons"] == []

    def test_pagination_bounds_are_enforced_at_the_boundary(self, client, project):
        _issue(client, title="deploy pipeline bounces")
        for params in ({"limit": 0}, {"limit": 101}, {"offset": -1}):
            response = _search(client, q="deploy", all_projects="true", **params)
            assert response.status_code == 422, params
        ok = _search(client, q="deploy", all_projects="true", limit=100, offset=0)
        assert ok.status_code == 200

    def test_windows_page_without_losing_total(self, client, project):
        first = _issue(client, title="deploy pipeline bounces on dry run")
        second = _issue(client, title="deploy rollback torn state")
        page_one = _search(client, q="deploy", all_projects="true", limit=1, offset=0).json()
        page_two = _search(client, q="deploy", all_projects="true", limit=1, offset=1).json()
        assert page_one["total"] == 2
        assert len(page_one["results"]) == 1
        assert page_two["total"] == 2
        assert {page_one["results"][0]["issue"]["key"], page_two["results"][0]["issue"]["key"]} == {
            first["key"],
            second["key"],
        }


class TestRefusalsAreTyped:
    def test_an_empty_query_is_refused_by_the_service_dialect(self, client, project):
        response = _search(client, q="", all_projects="true")
        assert response.status_code == 400
        assert "nonempty normalized free-form text" in response.json()["detail"]

    def test_a_punctuation_only_query_is_refused_as_empty(self, client, project):
        response = _search(client, q="!!! ???", all_projects="true")
        assert response.status_code == 400
        assert "nonempty normalized free-form text" in response.json()["detail"]

    def test_an_unknown_filter_vocabulary_is_the_service_refusal(self, client, project):
        response = _search(client, q="deploy", all_projects="true", kind="widget")
        assert response.status_code == 400


class TestReadScopedWritesNothing:
    def test_search_leaves_the_record_and_its_trail_untouched(self, client, project):
        filed = _issue(
            client, title="deploy pipeline bounces", failing_command="conduct deploy --dry-run"
        )
        client.post(
            f"/tracker/issues/{filed['key']}/comments",
            json={"body": "the bounce receipt was wrapped", "important": True},
        )
        before = client.get(f"/tracker/issues/{filed['key']}").json()

        searched = _search(client, q="conduct deploy --dry-run", project_id="cao-system")
        assert searched.status_code == 200
        assert _keys(searched) == [filed["key"]]

        after = client.get(f"/tracker/issues/{filed['key']}").json()
        assert after == before
