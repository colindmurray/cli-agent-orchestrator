"""The similar-issue HTTP surface (cond-0645).

The contract under test: the literal ``/tracker/issues/similar`` route wins
over ``/tracker/issues/{issue_key}``; issue_key XOR draft and project_ids XOR
all_projects are typed 400s; undeclared top-level fields are a 422 while
undeclared draft fields are the service's 400 naming the field; an unknown
issue key is a 404; candidates come explained with duplicate-chain
expansions; and a read-scoped probe writes nothing.
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
    engine = create_engine(f"sqlite:///{tmp_path}/similar-api.db")
    Base.metadata.create_all(bind=engine)
    _migrate_tracker_search_projection(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    monkeypatch.setattr(ranked, "SessionLocal", sessions)
    monkeypatch.setattr(similar, "SessionLocal", sessions)
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


def _similar(client, **payload):
    return client.post("/tracker/issues/similar", json=payload)


def _keys(response):
    return [row["issue"]["key"] for row in response.json()["candidates"]]


class TestRouteOrderAndEnvelope:
    def test_similar_resolves_before_the_issue_key_route(self, client, project):
        filed = _issue(client, title="deploy pipeline bounces on dry run")
        response = _similar(client, issue_key=filed["key"], all_projects=True)
        assert response.status_code == 200, response.text
        body = response.json()
        # The parameterised routes would have answered with a raw issue dict,
        # a 405, or a 404; the literal POST route won.
        assert body["query_source"]["mode"] == "issue_key"
        assert set(body) >= {
            "query_source",
            "query",
            "scope",
            "include_comments",
            "limit",
            "total",
            "candidates",
            "duplicate_expansions",
        }

    def test_candidates_come_explained_with_chain_expansion(self, client, project):
        hit = _issue(client, title="deploy pipeline bounces on dry run")
        dup = _issue(client, key="cond-0007", title="already tracked alpha")
        client.patch(
            f"/tracker/issues/{dup['key']}",
            json={"status": "duplicate", "duplicate_of": hit["key"]},
        )
        response = _similar(client, draft={"title": "deploy pipeline bounces"}, all_projects=True)
        assert response.status_code == 200, response.text
        body = response.json()
        assert hit["key"] in _keys(response)
        expansions = {
            (row["duplicate_of"], row["issue"]["key"]) for row in body["duplicate_expansions"]
        }
        assert (hit["key"], dup["key"]) in expansions


class TestStrictXorAtTheBoundary:
    def test_issue_key_and_draft_together_is_a_typed_400(self, client, project):
        filed = _issue(client, title="deploy pipeline bounces")
        response = _similar(
            client,
            issue_key=filed["key"],
            draft={"title": "deploy pipeline bounces"},
            all_projects=True,
        )
        assert response.status_code == 400
        assert "exactly one of issue_key or draft" in response.json()["detail"]

    def test_neither_issue_key_nor_draft_is_a_typed_400(self, client, project):
        response = _similar(client, all_projects=True)
        assert response.status_code == 400
        assert "exactly one of issue_key or draft" in response.json()["detail"]

    def test_both_scope_forms_is_a_typed_400(self, client, project):
        response = _similar(
            client,
            draft={"title": "deploy pipeline bounces"},
            project_ids=["cao-system"],
            all_projects=True,
        )
        assert response.status_code == 400
        assert "exactly one scope form" in response.json()["detail"]

    def test_neither_scope_form_is_a_typed_400(self, client, project):
        response = _similar(client, draft={"title": "deploy pipeline bounces"})
        assert response.status_code == 400
        assert "exactly one scope form" in response.json()["detail"]


class TestUndeclaredFieldsAreRejected:
    def test_an_undeclared_top_level_field_is_a_422(self, client, project):
        response = _similar(client, draft={"title": "x"}, all_projects=True, status="open")
        assert response.status_code == 422

    @pytest.mark.parametrize("field", ["status", "key", "project_id", "duplicate_of"])
    def test_server_owned_draft_fields_are_a_400_naming_the_field(self, client, project, field):
        response = _similar(
            client, draft={"title": "deploy pipeline bounces", field: "x"}, all_projects=True
        )
        assert response.status_code == 400
        assert field in str(response.json()["detail"])

    def test_limit_outside_bounds_is_a_422_at_the_boundary(self, client, project):
        for limit in (0, ranked.MAX_LIMIT + 1):
            response = _similar(client, draft={"title": "x"}, all_projects=True, limit=limit)
            assert response.status_code == 422, limit


class TestRefusalsAndReadPosture:
    def test_an_unknown_issue_key_is_a_404(self, client, project):
        response = _similar(client, issue_key="cond-9999", all_projects=True)
        assert response.status_code == 404

    def test_self_exclusion_holds_over_http(self, client, project):
        filed = _issue(client, title="deploy pipeline bounces on dry run")
        _issue(client, title="deploy pipeline bounces on dry run during rollback")
        response = _similar(client, issue_key=filed["key"], all_projects=True)
        assert filed["key"] not in _keys(response)

    def test_the_probe_writes_nothing(self, client, project):
        filed = _issue(
            client, title="deploy pipeline bounces", failing_command="conduct deploy --dry-run"
        )
        before = client.get(f"/tracker/issues/{filed['key']}").json()

        response = _similar(client, draft={"title": "conduct deploy --dry-run"}, all_projects=True)
        assert response.status_code == 200
        assert _keys(response) == [filed["key"]]
        assert client.get(f"/tracker/issues/{filed['key']}").json() == before


class TestNonGating:
    def test_similarity_failure_cannot_block_issue_creation(self, client, project, monkeypatch):
        """NON-GATING proof (cond-0645): the create path must not depend on
        the similar-issue service. With the service wired to explode on any
        call — a raw runtime bomb here, and typed refusals in the sibling
        cases — filing an issue still succeeds untouched."""

        def explode(*args, **kwargs):
            raise RuntimeError("similarity service is on fire")

        monkeypatch.setattr(similar, "find_similar_issues", explode)

        filed = _issue(client, title="deploy pipeline bounces")
        assert filed["key"].startswith("cond-")

        # The probe surface itself reports the explosion instead of lying.
        with pytest.raises(RuntimeError, match="similarity service is on fire"):
            _similar(client, draft={"title": "deploy pipeline bounces"}, all_projects=True)

    @pytest.mark.parametrize("code", ["invalid", "not-found", "invalid-query"])
    def test_typed_similarity_refusals_never_touch_the_create_path(
        self, client, project, monkeypatch, code
    ):
        def refuse(*args, **kwargs):
            raise tracker.TrackerError(code, f"similarity refused: {code}")

        monkeypatch.setattr(similar, "find_similar_issues", refuse)
        filed = _issue(client, title="deploy pipeline bounces")
        assert filed["key"].startswith("cond-")
