"""The search-index maintenance HTTP surface (cond-0770).

The four routes under test are thin adapters over
:mod:`services.search_index_maintenance` — the same orchestrator the CLI
calls — so what is worth proving here is the HTTP contract itself: the literal
``/tracker/issues/search-index/*`` paths win over
``/tracker/issues/{issue_key}``; reads are read-scoped and writes demand
write; a bare ``POST .../refresh`` is the bounded drain; a refusal keeps the
reason the orchestrator observed plus its operator action; and no maintenance
or search route ever downloads model weights.
"""

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Any, Dict
from pathlib import Path

from cli_agent_orchestrator.api import tracker as tracker_api
from cli_agent_orchestrator.api.main import app
from cli_agent_orchestrator.clients.database import (
    Base,
    _migrate_tracker_search_projection,
    _TRACKER_ORM_TABLE_NAMES,
)
from cli_agent_orchestrator.security import auth
from cli_agent_orchestrator.services import embedding_adapter as adapter
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services import search_index_maintenance as maintenance
from cli_agent_orchestrator.services import tracker_ranked_search as ranked
from cli_agent_orchestrator.services import vector_lifecycle as vlc

DIMENSIONS = 32


@pytest.fixture(autouse=True)
def search_db(tmp_path, monkeypatch):
    """A file-backed tracker store bound to every seam the routes read.

    ``maintenance.engine`` and the ``vector_lifecycle.engine`` its primitives
    use are separate module globals holding the pooled engine; both must point
    at the same file or a generation row lands in operator state while the
    response is computed from the lookalike.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/search-index-api.db")
    Base.metadata.create_all(
        bind=engine,
        tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
    )
    _migrate_tracker_search_projection(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(maintenance, "engine", engine)
    monkeypatch.setattr(vlc, "engine", engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    monkeypatch.setattr(ranked, "SessionLocal", sessions)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def no_model_downloads(monkeypatch):
    """No maintenance route may download weights; preparation is CLI-only."""

    def forbidden(*args, **kwargs):
        raise AssertionError("maintenance routes must not download model weights")

    monkeypatch.setattr(adapter, "prepare_model", forbidden)
    monkeypatch.setattr(adapter, "_default_snapshot_downloader", forbidden)


def _prepared_metadata(snapshot: Path) -> Dict[str, Any]:
    """The metadata record ``prepare_model`` writes, over a local fixture artifact.

    Written by hand rather than by calling ``prepare_model`` because the
    download is exactly what these routes must never do; the digest is
    measured over the fixture snapshot with the adapter's own function so the
    capability report can honestly re-verify it.
    """
    return {
        "schema": adapter.METADATA_SCHEMA,
        "model_id": adapter.MODEL_ID,
        "model_revision": adapter.MODEL_REVISION,
        "runtime_id": adapter.RUNTIME_ID,
        "runtime_versions": {
            package: adapter._read_dist_version(package)
            for package in adapter.REQUIRED_RUNTIME_PACKAGES
        },
        "dimensions": DIMENSIONS,
        "element_type": "float32",
        "normalized": True,
        "distance_metric": "cosine",
        "document_schema_version_id": adapter.DOCUMENT_SCHEMA_VERSION_ID,
        "document_schema_version_name": adapter.DOCUMENT_SCHEMA_VERSION_NAME,
        "artifact_sha256": adapter.dir_sha256(snapshot),
        "artifact_bytes": sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file()),
        "vec_version_pinned": adapter.PINNED_VEC_VERSION,
        "snapshot_path": str(snapshot),
        "snapshot_rel_path": "snapshot",
        "prepared_at": "2026-08-31T00:00:00+00:00",
    }


class StubEmbedder:
    """Same contract as the prepared metadata declares: float32 unit vectors."""

    dimensions = DIMENSIONS

    def _vector(self, text):
        import numpy as np

        vec = np.zeros(self.dimensions, dtype=np.float64)
        vec[hash(str(text)) % self.dimensions] = 1.0
        return np.asarray(vec, dtype="<f4")

    def embed(self, texts, *args, **kwargs):  # noqa: ARG002 - signature parity
        return [self._vector(text).tobytes() for text in texts]


@pytest.fixture
def prepared(search_db, tmp_path, monkeypatch):
    """A prepared model on disk with a bound generation, plus a stub embedder.

    Preparation itself is not an HTTP route by design — downloading weights is
    the operator's CLI decision — so the tests install the on-disk prepared
    state that command leaves behind, then bind the generation through the
    orchestrator exactly as ``cao issue search-index model prepare`` does.
    """
    models_dir = tmp_path / "models"
    snapshot = models_dir / "snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b'{"model_type": "bert"}')
    (snapshot / "model.safetensors").write_bytes(b"measured-weights")
    monkeypatch.setattr(adapter, "SEARCH_MODELS_DIR", models_dir)
    record = _prepared_metadata(snapshot)
    # The identity check compares the record against THIS build's pin, so the
    # fixture pins the same digest the fake artifact measures — the same
    # substitution `model prepare` performs on a real download.
    monkeypatch.setattr(adapter, "MODEL_ARTIFACT_SHA256", record["artifact_sha256"])
    adapter._write_metadata_atomic(adapter.metadata_path(models_dir), record)

    outcome = maintenance.prepare_index(metadata=record)
    monkeypatch.setattr(adapter, "load_embedder", lambda models_dir=None: StubEmbedder())
    return outcome


@pytest.fixture
def project(client):
    response = client.post(
        "/tracker/projects",
        json={"name": "CAO System", "id": "cao-system", "issue_prefix": "cond"},
    )
    assert response.status_code == 201, response.text
    response = client.post(
        "/tracker/issues",
        json={"project_id": "cao-system", "title": "deploy pipeline bounces", "force": True},
    )
    assert response.status_code == 201, response.text


# ---------------------------------------------------------------------------
# route resolution
# ---------------------------------------------------------------------------


def test_status_resolves_before_the_issue_key_route(client):
    """``search-index`` is a legal issue-key string; the literal route wins."""
    response = client.get("/tracker/issues/search-index/status")
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) >= {
        "capability",
        "engine",
        "lexical",
        "semantic",
        "active_generation",
        "next_actions",
    }
    assert body["active_generation"] is None, "nothing is prepared yet"


def test_integrity_check_resolves_before_the_issue_key_route(client):
    response = client.get("/tracker/issues/search-index/integrity-check")
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) >= {"fts_internal", "coverage", "vector_dirty", "generations", "semantic"}


def test_refresh_resolves_before_the_issue_key_route(client, prepared):
    """A bare POST is the bounded, activation-free drain."""
    response = client.post("/tracker/issues/search-index/refresh")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scope"] == "bounded"
    assert body["activations"] == []
    assert body["active_generation"] is None


def test_rebuild_resolves_before_the_issue_key_route(client, prepared, project):
    response = client.post("/tracker/issues/search-index/rebuild", json={"scope": "lexical"})
    assert response.status_code == 200, response.text
    assert response.json()["lexical"]["documents_rebuilt"] == 1


# ---------------------------------------------------------------------------
# scope enforcement
# ---------------------------------------------------------------------------


def _override_scopes(scopes):
    async def _dep():
        return list(scopes)

    return _dep


@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setenv("CAO_AUTH_JWKS_URI", "https://idp.example/jwks")


@pytest.fixture(autouse=True)
def _clear_scope_overrides():
    yield
    app.dependency_overrides.pop(auth.get_current_scopes, None)


def test_read_scope_is_admitted_on_the_read_routes(client, auth_on):
    app.dependency_overrides[auth.get_current_scopes] = _override_scopes([auth.SCOPE_READ])
    assert client.get("/tracker/issues/search-index/status").status_code == 200
    assert client.get("/tracker/issues/search-index/integrity-check").status_code == 200


def test_read_scope_is_refused_on_the_write_routes(client, prepared, auth_on):
    app.dependency_overrides[auth.get_current_scopes] = _override_scopes([auth.SCOPE_READ])
    assert client.post("/tracker/issues/search-index/refresh").status_code == 403
    assert (
        client.post("/tracker/issues/search-index/rebuild", json={"scope": "lexical"}).status_code
        == 403
    )


def test_write_scope_is_admitted_on_the_write_routes(client, prepared, auth_on):
    app.dependency_overrides[auth.get_current_scopes] = _override_scopes([auth.SCOPE_WRITE])
    assert client.post("/tracker/issues/search-index/refresh").status_code == 200
    assert (
        client.post("/tracker/issues/search-index/rebuild", json={"scope": "lexical"}).status_code
        == 200
    )


def test_write_scope_is_refused_on_the_read_routes(client, auth_on):
    app.dependency_overrides[auth.get_current_scopes] = _override_scopes([auth.SCOPE_WRITE])
    assert client.get("/tracker/issues/search-index/status").status_code == 200, (
        "a write token holds read too"
    )


# ---------------------------------------------------------------------------
# the journey over HTTP
# ---------------------------------------------------------------------------


def test_refresh_all_activates_and_status_reports_it(client, prepared, project):
    response = client.post("/tracker/issues/search-index/refresh", json={"all": True})
    assert response.status_code == 200, response.text
    body = response.json()
    generation_id = prepared["generation_id"]
    assert body["activations"] == [{"generation_id": generation_id, "activated": True}]
    assert body["active_generation"] == generation_id

    status_payload = client.get("/tracker/issues/search-index/status").json()
    assert status_payload["semantic"]["state"] == "ready"
    assert status_payload["active_generation"] == generation_id
    assert status_payload["next_actions"] == []


# ---------------------------------------------------------------------------
# typed refusals
# ---------------------------------------------------------------------------


def test_an_unknown_rebuild_scope_is_a_typed_bad_request(client, prepared):
    response = client.post("/tracker/issues/search-index/rebuild", json={"scope": "everything"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["reason"] == "invalid-scope"
    assert "--lexical" in detail["action"]


def test_rebuilding_vectors_without_a_prepared_model_is_a_typed_conflict(client):
    response = client.post("/tracker/issues/search-index/rebuild", json={"scope": "vectors"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "unprepared"
    assert "model prepare" in detail["action"]


def test_an_undrained_build_is_reported_not_failed(client, prepared, project):
    """A partially drained build is a valid answer, never an activation.

    A second issue makes the queue two documents deep, so a bounded batch of
    one leaves the build genuinely unfinished; ``all`` then finishes it.
    """
    assert (
        client.post(
            "/tracker/issues",
            json={"project_id": "cao-system", "title": "widget color tuning", "force": True},
        ).status_code
        == 201
    )

    partial = client.post("/tracker/issues/search-index/refresh", json={"limit": 1})
    assert partial.status_code == 200, partial.text
    body = partial.json()
    assert body["refresh"]["published"] == 1
    assert body["activations"] == [], "a bounded drain never offers activation"
    assert body["active_generation"] is None

    finished = client.post("/tracker/issues/search-index/refresh", json={"all": True})
    assert finished.status_code == 200, finished.text
    assert finished.json()["active_generation"] == prepared["generation_id"]


def test_maintenance_refusals_carry_reason_and_action_over_http():
    """The mapping table: an installation refusal is a 409, a malformed
    request is a 400 — a client can branch on the status code alone."""
    assert tracker_api._maintenance_http(
        maintenance.SearchIndexMaintenanceError(
            "unprepared", "not prepared", action="run prepare"
        )
    ).status_code == 409
    assert tracker_api._maintenance_http(
        maintenance.SearchIndexMaintenanceError("schema-missing", "no derived schema")
    ).status_code == 409
    unknown = HTTPException(status_code=404)
    mapped = tracker_api._maintenance_http(
        maintenance.SearchIndexMaintenanceError("unknown-generation", "no such generation")
    )
    assert mapped.status_code == 404
    assert unknown.status_code == 404  # the boundary keeps FastAPI's own shape


def test_a_store_without_the_derived_schema_is_a_typed_conflict(tmp_path, monkeypatch, client):
    """The projection is installed with the tracker schema; a store that predates
    it is an installation fact with a remedy, not a 500."""
    engine = create_engine(f"sqlite:///{tmp_path}/bare-api.db")
    Base.metadata.create_all(
        bind=engine,
        tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
    )
    try:
        monkeypatch.setattr(maintenance, "engine", engine)
        monkeypatch.setattr(vlc, "engine", engine)
        response = client.get("/tracker/issues/search-index/integrity-check")
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["reason"] == "schema-missing"
        assert "cao issue" in detail["action"]
    finally:
        engine.dispose()
