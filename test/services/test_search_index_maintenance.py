"""The shared search-index maintenance orchestrator (cond-0770).

cond-0643 and cond-0644 proved the vector primitives; cond-0770 files the gap
that nothing called them. These tests therefore exercise the ORCHESTRATOR —
the same module the CLI verbs and the REST routes call — and every journey is
driven through :mod:`search_index_maintenance` rather than by calling
``create_generation`` or ``activate_generation`` directly, because a primitive
test that activates a generation by hand is exactly what concealed the missing
wiring.

Two rows the accepted design pins that no primitive test could catch live here:
the authority boundary (:func:`TestAuthorityBoundary`) diffs every
authoritative table around each journey, so an orchestrator that "reindexed"
by rewriting issues would fail, and the build-completion contract
(:func:`TestRefresh`) proves an incomplete build never activates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import tracker_search_schema as schema
from cli_agent_orchestrator.clients.database import (
    _TRACKER_ORM_TABLE_NAMES,
    Base,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.services import search_index_maintenance as maintenance
from cli_agent_orchestrator.services import tracker_ranked_search as ranked
from cli_agent_orchestrator.services import vector_lifecycle as vlc

DIMENSIONS = 32

#: The authoritative tables maintenance must never touch (§3.2 invariant).
AUTHORITATIVE_TABLES = (
    "tracker_issues",
    "tracker_issue_comments",
    "tracker_issue_links",
    "tracker_issue_events",
)


# ---------------------------------------------------------------------------
# Store fixture and deterministic embedder
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A file-backed tracker store bound to the maintenance module's engine.

    Two module globals hold the pooled engine — ``maintenance.engine`` and the
    ``vector_lifecycle.engine`` its primitives read — and BOTH must be patched,
    or a generation write lands in the operator's real store while the test
    asserts against an empty lookalike. ``ranked.SessionLocal`` is patched
    alongside so the end-to-end hybrid-search assertions read the same file.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/maintenance.db")
    Base.metadata.create_all(
        bind=engine,
        tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
    )
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tracker_projects (id, name) VALUES ('p', 'P')"))
    _migrate_tracker_search_projection(engine)
    monkeypatch.setattr(maintenance, "engine", engine, raising=True)
    monkeypatch.setattr(vlc, "engine", engine, raising=True)
    monkeypatch.setattr(ranked, "SessionLocal", sessionmaker(bind=engine))
    yield engine
    engine.dispose()


def _execute(engine, sql, params=()):
    raw = engine.raw_connection()
    try:
        cursor = raw.execute(sql, params)
        names = [column[0] for column in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        raw.commit()
        return [dict(zip(names, row)) for row in rows]
    finally:
        raw.close()


def _seed_issue(engine, key, title, body="", **extra):
    columns = {"key": key, "project_id": "p", "title": title, "body": body, **extra}
    names = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    _execute(
        engine,
        f"INSERT INTO tracker_issues ({names}) VALUES ({placeholders})",
        tuple(columns.values()),
    )
    return key


def _seed_comment(engine, issue_key, body, author="alice"):
    raw = engine.raw_connection()
    try:
        cursor = raw.execute(
            "INSERT INTO tracker_issue_comments (issue_key, author, body) VALUES (?, ?, ?)",
            (issue_key, author, body),
        )
        comment_id = cursor.lastrowid
        raw.commit()
    finally:
        raw.close()
    return int(comment_id)


def _generation_rows(engine):
    return _execute(
        engine,
        f"SELECT generation_id, state, model_id, dimensions, failure\n"
        f"FROM {schema.VECTOR_GENERATIONS_TABLE} ORDER BY created_at, generation_id",
    )


def _dirty_counts(engine):
    rows = _execute(
        engine,
        f"SELECT attempt_count, last_error, next_attempt_at FROM {schema.VECTOR_DIRTY_TABLE}",
    )
    return {
        "total": len(rows),
        "failed": sum(1 for row in rows if row["last_error"] is not None),
        "backed_off": sum(1 for row in rows if row["next_attempt_at"] is not None),
        "attempts": sorted(int(row["attempt_count"]) for row in rows),
    }


def _active_pointer(engine):
    return _execute(
        engine,
        f"SELECT active_vector_generation FROM {schema.SEARCH_META_TABLE} WHERE singleton = 1",
    )[0]["active_vector_generation"]


def _snapshot_authoritative(engine) -> Dict[str, List[Dict[str, Any]]]:
    return {
        table: _execute(engine, f"SELECT * FROM {table} ORDER BY 1")
        for table in AUTHORITATIVE_TABLES
    }


class DeterministicEmbedder:
    """Token-frequency vectors over an orthonormal basis; exact, explainable.

    Shared tokens give provably smaller cosine distances, so an assertion that
    one issue outranks another is a statement about the embedded text rather
    than about float noise. ``fail_on`` turns this into the failure fixture the
    §9.3 retry rows need; the markers are matched against the document text the
    versioned builder produced, which is what the refresh loop really embeds.
    """

    dimensions = DIMENSIONS

    def __init__(self, fail_on=(), dimensions: int = DIMENSIONS):
        self.fail_on = tuple(fail_on)
        self.dimensions = dimensions
        self.calls: List[List[str]] = []

    def _vector(self, text):
        vec = np.zeros(int(self.dimensions), dtype=np.float64)
        for token in str(text).lower().split():
            index = VOCAB.get(token.strip(".,:;"))
            if index is not None:
                vec[index] += 1.0
        norm = np.linalg.norm(vec)
        return np.asarray(vec / norm if norm else vec, dtype="<f4")

    def embed(self, texts, *args, **kwargs):  # noqa: ARG002 - signature parity
        self.calls.append(list(texts))
        blobs = []
        for text in texts:
            if any(marker in text for marker in self.fail_on):
                raise RuntimeError(f"embedding refused for {text[:32]!r}")
            blobs.append(self._vector(text).tobytes())
        return blobs


VOCAB = {"deploy": 0, "pipeline": 1, "bounce": 2, "widget": 3, "color": 4, "tuning": 5}

#: Issues whose text never collides, so ranks are provable, plus one comment
#: that repeats k-1's vocabulary — the comment lane has something real to win.
DEPLOY_ISSUE = ("deploy-1", "deploy pipeline bounce", "the deploy pipeline bounces on dry run")
WIDGET_ISSUE = ("widget-2", "widget color tuning", "tune the widget color profile")
DEPLOY_COMMENT = "the deploy bounce reproduced twice"


def _prepared_metadata(**overrides):
    """A metadata record shaped exactly like ``embedding_adapter`` output."""
    record = {
        "schema": "cao-search-generation-metadata-v1",
        "model_id": "test/local-model",
        "model_revision": "rev000111222333",
        "runtime_id": "sentence-transformers",
        "runtime_versions": {
            "sentence-transformers": "6.1.0",
            "torch": "2.9.0",
            "transformers": "4.51.0",
        },
        "dimensions": DIMENSIONS,
        "element_type": "float32",
        "normalized": True,
        "distance_metric": "cosine",
        "document_schema_version_id": 1,
        "artifact_sha256": "a" * 64,
    }
    record.update(overrides)
    return record


@pytest.fixture
def prepared(store):
    """Bind a generation through the orchestrator, without touching a model."""
    return maintenance.prepare_index(metadata=_prepared_metadata())


def _populate(store):
    _seed_issue(store, *DEPLOY_ISSUE)
    _seed_issue(store, *WIDGET_ISSUE)
    _seed_comment(store, DEPLOY_ISSUE[0], DEPLOY_COMMENT)
    return DeterministicEmbedder()


def _search_hybrid(store, embedder, query=DEPLOY_ISSUE[1]):
    """One real hybrid query with only the query embedder injected."""
    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(ranked, "_query_embedder", lambda models_dir=None: embedder, raising=True)
    try:
        return ranked.ranked_search(
            ranked.RankedSearchRequest(query=query, project_ids=("p",), mode="hybrid")
        )
    finally:
        monkeypatched.undo()


# ---------------------------------------------------------------------------
# prepare: explicit model preparation binds exactly one generation
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_prepare_creates_one_building_generation_and_queues_every_document(self, store):
        _populate(store)
        outcome = maintenance.prepare_index(metadata=_prepared_metadata())

        assert outcome["action"] == "created"
        assert outcome["generation_state"] == "building"
        rows = _generation_rows(store)
        assert len(rows) == 1 and rows[0]["state"] == "building"
        assert outcome["enqueued_documents"] == 3, "two issues plus one comment"
        assert _active_pointer(store) is None, "preparing activates nothing"

    def test_repeated_prepare_reuses_and_mints_no_series_of_generations(self, store):
        """Acceptance: repeated prepare is idempotent over generation rows."""
        _populate(store)
        first = maintenance.prepare_index(metadata=_prepared_metadata())
        for _ in range(4):
            again = maintenance.prepare_index(metadata=_prepared_metadata())
            assert again["action"] == "reused"
            assert again["generation_id"] == first["generation_id"]
        assert len(_generation_rows(store)) == 1

    def test_repeated_prepare_still_reuses_after_the_generation_went_active(self, store):
        """The idempotency that matters operationally: prepare, build, re-prepare.

        Reuse must be decided by stored identity, not by state, or every
        scheduled prepare after a successful build would start a fresh
        full-corpus build for no reason.
        """
        _populate(store)
        maintenance.prepare_index(metadata=_prepared_metadata())
        maintenance.refresh_index(all=True, embedder=DeterministicEmbedder())
        assert _active_pointer(store) is not None

        again = maintenance.prepare_index(metadata=_prepared_metadata())
        assert again["action"] == "reused"
        assert again["generation_state"] == "active"
        assert len(_generation_rows(store)) == 1

    def test_prepare_resumes_an_interrupted_build_rather_than_starting_over(self, store):
        _populate(store)
        maintenance.prepare_index(metadata=_prepared_metadata())
        maintenance.refresh_index(all=True, embedder=DeterministicEmbedder(fail_on=("deploy",)))
        assert _active_pointer(store) is None
        assert len(_generation_rows(store)) == 1

        resumed = maintenance.prepare_index(metadata=_prepared_metadata())
        assert resumed["action"] == "reused"
        assert resumed["generation_state"] == "building"
        assert len(_generation_rows(store)) == 1

    def test_a_different_model_prepares_a_second_generation(self, store):
        """A real model change IS a new generation (§13.3); only an identical
        identity is reusable."""
        _populate(store)
        maintenance.prepare_index(metadata=_prepared_metadata())
        other = maintenance.prepare_index(metadata=_prepared_metadata(model_id="test/other"))
        assert other["action"] == "created"
        assert len(_generation_rows(store)) == 2

    def test_preparing_without_a_prepared_model_refuses_and_names_the_prepare_command(self, store):
        with pytest.raises(maintenance.SearchIndexMaintenanceError) as excinfo:
            maintenance.ensure_generation()
        assert excinfo.value.reason == "unprepared"
        assert "model prepare" in excinfo.value.action

    def test_prepare_on_a_store_without_the_derived_schema_names_the_install_remedy(
        self, tmp_path, monkeypatch
    ):
        engine = create_engine(f"sqlite:///{tmp_path}/bare.db")
        Base.metadata.create_all(
            bind=engine,
            tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
        )
        monkeypatch.setattr(maintenance, "engine", engine, raising=True)
        try:
            with pytest.raises(maintenance.SearchIndexMaintenanceError) as excinfo:
                maintenance.prepare_index(metadata=_prepared_metadata())
            assert excinfo.value.reason == "schema-missing"
            assert "cao issue" in excinfo.value.action
        finally:
            engine.dispose()

    def test_prepare_binds_the_generation_after_a_real_metadata_prepare(
        self, store, tmp_path, monkeypatch
    ):
        """The production prepare path: real ``prepare_model`` over a fake
        snapshot, then the generation half — the two halves of one command."""
        from cli_agent_orchestrator.services import embedding_adapter as adapter

        _fake_runtime(monkeypatch)
        models_dir = tmp_path / "models"
        outcome = maintenance.prepare_index(
            models_dir=models_dir,
            snapshot_downloader=_fake_downloader,
        )

        assert outcome["model"]["artifact_sha256"] == _fake_digest()
        assert outcome["action"] == "created"
        assert outcome["generation"]["runtime_version"] == "6.1.0"
        assert _generation_rows(store)[0]["model_id"] == adapter.MODEL_ID


def _FAKE_VERSIONS() -> Dict[str, str]:
    return {"sentence-transformers": "6.1.0", "torch": "2.9.0", "transformers": "4.51.0"}


def _fake_runtime(monkeypatch) -> Dict[str, str]:
    """Pin the runtime probe and target digest so prepare needs no network."""
    from cli_agent_orchestrator.services import embedding_adapter as adapter

    versions = _FAKE_VERSIONS()
    monkeypatch.setattr(adapter, "_read_dist_version", lambda name: versions.get(name))
    monkeypatch.setattr(adapter, "MODEL_ARTIFACT_SHA256", _fake_digest())
    return versions


def _fake_digest() -> str:
    import hashlib

    digest = hashlib.sha256()
    for rel, data in sorted(
        (("config.json", b'{"model_type": "bert"}'), ("model.safetensors", b"measured-weights"))
    ):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest()


def _fake_downloader(*, repo_id, revision, cache_dir, ignore_patterns=None):
    """A deterministic stand-in for the HF snapshot download."""
    snapshot = Path(cache_dir) / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_bytes(b'{"model_type": "bert"}')
    (snapshot / "model.safetensors").write_bytes(b"measured-weights")
    return str(snapshot)


# ---------------------------------------------------------------------------
# refresh: drain, coverage proof, activation
# ---------------------------------------------------------------------------


class TestRefresh:
    def test_refresh_all_activates_and_the_active_pointer_becomes_non_null(self, store, prepared):
        _populate(store)
        result = maintenance.refresh_index(all=True, embedder=DeterministicEmbedder())

        assert result["refresh"]["published"] == 3
        assert result["activations"] == [
            {"generation_id": prepared["generation_id"], "activated": True}
        ]
        assert result["active_generation"] == prepared["generation_id"]
        assert _active_pointer(store) == prepared["generation_id"]

    def test_refresh_all_then_explicit_hybrid_search_serves_the_semantic_lane(self, store, prepared):
        """Acceptance: the production journey reaches a semantic contribution.

        ``mode="hybrid"`` through the real ``ranked_search`` entry point with
        only the query embedder injected — this is the wiring cond-0770 files
        as missing, so the assertion is about lanes in the explained response,
        not about a helper's return value.
        """
        _populate(store)
        embedder = DeterministicEmbedder()
        maintenance.refresh_index(all=True, embedder=embedder)

        response = _search_hybrid(store, embedder)

        assert response["mode_effective"] == "hybrid"
        assert response["diagnostics"]["semantic"]["served"] is True
        lanes = {
            lane["lane"] for row in response["results"] for lane in row["contributing_lanes"]
        }
        assert any(lane.startswith("semantic") for lane in lanes), sorted(lanes)
        assert [row["issue"]["key"] for row in response["results"]] == [
            DEPLOY_ISSUE[0],
            WIDGET_ISSUE[0],
        ]

    def test_an_embedder_whose_width_disagrees_with_the_generation_never_activates(
        self, store, prepared
    ):
        """The §13.3 encoding proof through the orchestrator: a build produced
        at the wrong width is refused, reported, and never goes live."""
        _populate(store)
        result = maintenance.refresh_index(
            all=True, embedder=DeterministicEmbedder(dimensions=DIMENSIONS + 8)
        )

        offer = result["activations"][0]
        assert offer["activated"] is False
        assert offer["refused_reason"] == "activation-refused-encoding"
        assert result["active_generation"] is None
        assert _active_pointer(store) is None

    def test_bounded_refresh_drains_one_batch_and_never_activates(self, store, prepared):
        _populate(store)
        result = maintenance.refresh_index(embedder=DeterministicEmbedder())
        assert result["scope"] == "bounded"
        assert result["activations"] == []
        assert _active_pointer(store) is None, "a bounded drain is not a build completion"

    def test_an_incomplete_build_is_never_offered_activation_and_stays_retryable(
        self, store, prepared
    ):
        """Acceptance: failed/incomplete builds never activate (§13.3).

        The embedder refuses every document, so the queue cannot drain; the
        build must simply not be offered, stay ``building`` and resumable, and
        the exit must be the ordinary refresh command with retries reset.
        """
        _populate(store)
        result = maintenance.refresh_index(
            all=True, embedder=DeterministicEmbedder(fail_on=("deploy", "comment"))
        )

        assert result["activations"] == [], "an incomplete build is not a candidate"
        assert _active_pointer(store) is None
        assert {row["state"] for row in _generation_rows(store)} == {"building"}
        counts = _dirty_counts(store)
        assert counts["total"] == counts["failed"] == 3

        recovery = maintenance.refresh_index(
            all=True, retry_failed=True, embedder=DeterministicEmbedder()
        )
        assert recovery["activations"] == [
            {"generation_id": prepared["generation_id"], "activated": True}
        ]
        assert _active_pointer(store) == prepared["generation_id"]

    def test_a_queue_that_drained_incomplete_refuses_activation_and_reports_why(
        self, store, prepared
    ):
        """The other incomplete shape: the queue looks empty but coverage does
        not hold. The orchestrator must surface the typed refusal — not crash,
        not activate — and the exit for a build this damaged is a rebuild."""
        _populate(store)
        # Damage the derived queue the way a dropped/pruned row does: the
        # comment will never be embedded, yet the queue reports drained.
        _execute(
            store,
            f"DELETE FROM {schema.VECTOR_DIRTY_TABLE} WHERE document_kind = 'comment'",
        )

        result = maintenance.refresh_index(all=True, embedder=DeterministicEmbedder())

        offer = result["activations"][0]
        assert offer["generation_id"] == prepared["generation_id"]
        assert offer["activated"] is False
        assert offer["refused_reason"] == "activation-refused-coverage"
        assert _active_pointer(store) is None
        assert {row["state"] for row in _generation_rows(store)} == {"building"}

        repaired = maintenance.rebuild_index(
            scope="vectors", metadata=_prepared_metadata(), embedder=DeterministicEmbedder()
        )
        assert repaired["vectors"]["activation"]["activated"] is True
        assert _active_pointer(store) == repaired["vectors"]["activation"]["generation_id"]

    def test_an_embedding_failure_records_retry_state_and_excludes_the_document(self, store, prepared):
        _populate(store)
        maintenance.refresh_index(embedder=DeterministicEmbedder(fail_on=("deploy",)))

        counts = _dirty_counts(store)
        assert counts["total"] == counts["failed"] == 3
        assert counts["backed_off"] == 3
        assert counts["attempts"] == [1, 1, 1]
        assert _active_pointer(store) is None

    def test_retry_failed_resets_backoff_so_the_documents_become_eligible(self, store, prepared):
        _populate(store)
        maintenance.refresh_index(embedder=DeterministicEmbedder(fail_on=("deploy",)))

        result = maintenance.refresh_index(
            retry_failed=True, embedder=DeterministicEmbedder()
        )

        assert result["retry_failed"]["reset"] == 3
        assert result["refresh"]["published"] == 3, "the reset rows became eligible"
        assert _dirty_counts(store)["total"] == 0

    def test_retry_failed_honours_an_explicit_bound(self, store, prepared):
        """``--retry-failed --limit N`` resets N rows, not the whole backlog."""
        for index in range(5):
            _seed_issue(store, f"bound-{index}", f"deploy pipeline {index}")
        maintenance.refresh_index(embedder=DeterministicEmbedder(fail_on=("deploy",)))
        assert _dirty_counts(store)["failed"] == 5

        result = maintenance.refresh_index(
            retry_failed=True, limit=2, embedder=DeterministicEmbedder(fail_on=("deploy",))
        )

        assert result["retry_failed"]["reset"] == 2
        assert result["retry_failed"]["bounded"] is True
        counts = _dirty_counts(store)
        # Only the two reset rows were retried, so only they carry attempt 2.
        assert counts["attempts"] == [1, 1, 1, 2, 2]

    def test_refresh_report_names_every_leg_an_operator_acts_on(self, store, prepared):
        _populate(store)
        payload = maintenance.refresh_index(
            all=True, retry_failed=True, embedder=DeterministicEmbedder()
        )
        json.dumps(payload)
        assert set(payload) >= {
            "scope",
            "retry_failed",
            "refresh",
            "activations",
            "active_generation",
        }
        assert set(payload["refresh"]) >= {
            "attempted",
            "published",
            "deleted_source_gone",
            "discarded_stale",
            "failed",
            "damaged_skipped",
        }


# ---------------------------------------------------------------------------
# rebuild
# ---------------------------------------------------------------------------


class TestRebuild:
    def test_rebuild_lexical_repopulates_documents_and_requeues_every_generation(
        self, store, prepared
    ):
        _populate(store)
        maintenance.refresh_index(all=True, embedder=DeterministicEmbedder())
        assert _active_pointer(store) is not None
        assert _dirty_counts(store)["total"] == 0

        result = maintenance.rebuild_index(scope="lexical")

        assert result["lexical"] == {"documents_rebuilt": 3, "issues": 2, "comments": 1}
        rebuilt_at = _execute(
            store, "SELECT rebuilt_at FROM tracker_search_meta WHERE singleton = 1"
        )[0]["rebuilt_at"]
        assert rebuilt_at
        # Every live document is queued again, so no pre-rebuild vector can be
        # served against rebuilt text (§13.2 step 4).
        assert _dirty_counts(store)["total"] == 3

    def test_rebuild_vectors_builds_and_activates_a_fresh_generation(self, store, prepared):
        _populate(store)
        maintenance.refresh_index(all=True, embedder=DeterministicEmbedder())
        first = _active_pointer(store)

        result = maintenance.rebuild_index(
            scope="vectors", metadata=_prepared_metadata(), embedder=DeterministicEmbedder()
        )

        assert result["vectors"]["activation"]["activated"] is True
        second = _active_pointer(store)
        assert second != first
        states = {row["generation_id"]: row["state"] for row in _generation_rows(store)}
        assert states[second] == "active"
        assert states[first] == "retired"

    def test_rebuild_all_does_lexical_then_vectors(self, store, prepared):
        _populate(store)
        result = maintenance.rebuild_index(
            scope="all", metadata=_prepared_metadata(), embedder=DeterministicEmbedder()
        )
        assert result["lexical"]["documents_rebuilt"] == 3
        assert result["vectors"]["activation"]["activated"] is True
        assert _active_pointer(store) is not None

    def test_rebuild_refuses_an_unknown_scope(self, store):
        with pytest.raises(maintenance.SearchIndexMaintenanceError) as excinfo:
            maintenance.rebuild_index(scope="everything")
        assert excinfo.value.reason == "invalid-scope"
        assert "--lexical" in excinfo.value.action

    def test_rebuild_vectors_never_downloads_a_model(self, store, prepared, monkeypatch):
        """Vector repair reads prepared metadata; downloading stays with prepare."""
        from cli_agent_orchestrator.services import embedding_adapter as adapter

        def forbidden(*args, **kwargs):
            raise AssertionError("rebuild must not download model weights")

        monkeypatch.setattr(adapter, "prepare_model", forbidden)
        monkeypatch.setattr(adapter, "_default_snapshot_downloader", forbidden)
        _populate(store)
        with pytest.raises(maintenance.SearchIndexMaintenanceError) as excinfo:
            maintenance.rebuild_index(scope="vectors")
        assert excinfo.value.reason == "unprepared"
        assert "model prepare" in excinfo.value.action

    def test_rebuild_with_documents_that_cannot_embed_stays_building_and_retryable(
        self, store, prepared
    ):
        """``rebuild --vectors`` against a failing embedder reports the refusal
        and leaves the replacement build resumable — the same shape as refresh."""
        _populate(store)
        result = maintenance.rebuild_index(
            scope="vectors",
            metadata=_prepared_metadata(),
            embedder=DeterministicEmbedder(fail_on=("deploy",)),
        )

        activation = result["vectors"]["activation"]
        assert activation["activated"] is False
        assert activation["refused_reason"] == "activation-refused-dirty"
        assert _active_pointer(store) is None
        assert {row["state"] for row in _generation_rows(store)} == {"building"}

        # The exit is the ordinary refresh command, not another rebuild.
        recovery = maintenance.refresh_index(
            all=True, retry_failed=True, embedder=DeterministicEmbedder()
        )
        assert recovery["activations"][0]["activated"] is True
        assert _active_pointer(store) is not None

    def test_a_rebuild_whose_refresh_leg_raises_records_the_generation_failed(
        self, store, prepared, monkeypatch
    ):
        """A refresh that cannot run at all is not retry state: the build is
        marked ``failed`` with what was observed, never left looking resumable."""
        from cli_agent_orchestrator.services.search_engine_factory import SearchEngineError

        def absent_runtime(*args, **kwargs):
            raise SearchEngineError(
                "runtime-missing", "the sqlite-vec package is not installed"
            )

        monkeypatch.setattr(vlc, "open_search_connection", absent_runtime)
        _populate(store)
        with pytest.raises(maintenance.SearchIndexMaintenanceError) as excinfo:
            maintenance.rebuild_index(
                scope="vectors", metadata=_prepared_metadata(), embedder=DeterministicEmbedder()
            )

        assert excinfo.value.reason == "runtime-missing"
        assert "[search]" in excinfo.value.action
        rows = _generation_rows(store)
        failed = [row for row in rows if row["state"] == "failed"]
        assert len(failed) == 1, rows
        assert "sqlite-vec" in failed[0]["failure"]
        assert _active_pointer(store) is None


# ---------------------------------------------------------------------------
# status and integrity
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_status_names_both_remedies_when_the_runtime_is_gone_but_the_model_is_prepared(
        self, store, tmp_path, monkeypatch
    ):
        """Acceptance: a base install gets a typed answer plus its remedy.

        The model was prepared while ``[search]`` was installed and the extra
        has since gone — the exact "permanently degraded" state cond-0770
        files. Status must name the install command AND the prepare command,
        because either order repairs the installation.
        """
        from cli_agent_orchestrator.services import embedding_adapter as adapter

        _fake_runtime(monkeypatch)
        models_dir = tmp_path / "models"
        maintenance.prepare_index(models_dir=models_dir, snapshot_downloader=_fake_downloader)
        monkeypatch.setattr(adapter, "_read_dist_version", lambda name: None)

        report = maintenance.index_status(models_dir=models_dir, run_probe=False)

        assert report["capability"]["state"] == "runtime-missing"
        actions = {entry["state"]: entry["action"] for entry in report["next_actions"]}
        assert "[search]" in actions["runtime-missing"]

    def test_status_on_a_fresh_install_reports_unprepared_with_the_prepare_command(self, store):
        """Nothing prepared yet: the degraded answer is still a valid report."""
        report = maintenance.index_status(models_dir=None, run_probe=False)

        assert report["semantic"]["state"] == "unprepared"
        actions = {entry["state"]: entry["action"] for entry in report["next_actions"]}
        assert "model prepare" in actions["unprepared"]

    def test_status_after_a_complete_build_reports_ready_with_no_pending_action(
        self, store, tmp_path, monkeypatch
    ):
        """The whole journey through real commands: prepare, refresh, status.

        Real ``prepare_model`` over a fake snapshot so the capability half
        honestly reports ``prepared`` — a status test that faked the report
        would prove nothing about the payload an operator polls.
        """
        _fake_runtime(monkeypatch)
        _populate(store)

        models_dir = tmp_path / "models"
        prepared = maintenance.prepare_index(models_dir=models_dir, snapshot_downloader=_fake_downloader)
        maintenance.refresh_index(
            all=True,
            embedder=DeterministicEmbedder(dimensions=int(prepared["generation"]["dimensions"])),
        )

        report = maintenance.index_status(models_dir=models_dir, run_probe=False)

        assert prepared["action"] == "created"
        assert report["capability"]["state"] == "prepared"
        assert report["semantic"]["state"] == "ready"
        assert report["active_generation"] == prepared["generation_id"]
        assert report["lexical"]["issues"] == 2
        assert report["lexical"]["comments"] == 1
        assert report["lexical"]["vectors"] == 3
        assert report["next_actions"] == []

    def test_status_names_the_refresh_action_while_a_build_is_incomplete(self, store, prepared):
        _populate(store)
        report = maintenance.index_status()

        assert report["semantic"]["state"] == "building"
        actions = {entry["state"]: entry["action"] for entry in report["next_actions"]}
        assert "refresh --all" in actions["building"]

    def test_integrity_check_reports_coverage_and_generations_without_repairing(
        self, store, prepared
    ):
        _populate(store)
        before = _snapshot_authoritative(store)

        report = maintenance.integrity_check()

        assert report["fts_internal"] == {"issues": "ok", "comments": "ok"}
        assert report["coverage"]["issue"]["source_rows"] == 2
        assert report["coverage"]["issue"]["documents"] == 2
        assert report["coverage"]["comment"]["documents"] == 1
        assert report["vector_dirty"]["total"] == 3
        assert len(report["generations"]) == 1
        assert report["semantic"]["state"] == "building"
        assert _snapshot_authoritative(store) == before, "integrity check repairs nothing"

    def test_integrity_check_on_a_store_without_the_derived_schema_is_typed(
        self, tmp_path, monkeypatch
    ):
        engine = create_engine(f"sqlite:///{tmp_path}/bare-integrity.db")
        Base.metadata.create_all(
            bind=engine,
            tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
        )
        monkeypatch.setattr(maintenance, "engine", engine, raising=True)
        try:
            with pytest.raises(maintenance.SearchIndexMaintenanceError) as excinfo:
                maintenance.integrity_check()
            assert excinfo.value.reason == "schema-missing"
        finally:
            engine.dispose()


# ---------------------------------------------------------------------------
# authority boundary
# ---------------------------------------------------------------------------


class TestAuthorityBoundary:
    def test_no_journey_writes_an_authoritative_tracker_row(self, store, prepared):
        """Acceptance: derived maintenance never mutates tracker truth.

        Every journey runs back to back against one store, which is stricter
        than running them separately: a single journey that wrote a source row
        would surface in the final diff.
        """
        _populate(store)
        before = _snapshot_authoritative(store)
        healthy = DeterministicEmbedder()
        failing = DeterministicEmbedder(fail_on=("deploy",))

        maintenance.prepare_index(metadata=_prepared_metadata())
        maintenance.refresh_index(embedder=failing)
        maintenance.refresh_index(all=True, embedder=failing)
        maintenance.refresh_index(retry_failed=True, all=True, embedder=healthy)
        maintenance.rebuild_index(scope="lexical")
        maintenance.rebuild_index(scope="vectors", metadata=_prepared_metadata(), embedder=healthy)
        maintenance.index_status()
        maintenance.integrity_check()

        assert _snapshot_authoritative(store) == before

    def test_query_time_semantic_search_writes_no_authoritative_row(self, store, prepared):
        """§19.3: the bounded drain a hybrid query performs stays read-scoped."""
        _populate(store)
        embedder = DeterministicEmbedder()
        maintenance.refresh_index(all=True, embedder=embedder)
        before = _snapshot_authoritative(store)

        _search_hybrid(store, embedder)

        assert _snapshot_authoritative(store) == before


# ---------------------------------------------------------------------------
# degraded runtime
# ---------------------------------------------------------------------------


class TestDegradedRuntime:
    def test_refresh_without_the_vector_runtime_is_typed_and_leaves_the_store_consistent(
        self, store, prepared, monkeypatch
    ):
        """A base install can still create a generation and read status; the
        one journey that needs sqlite-vec refuses with the install action
        instead of a traceback, and the build stays resumable."""
        from cli_agent_orchestrator.services.search_engine_factory import SearchEngineError

        def runtime_missing(*args, **kwargs):
            raise SearchEngineError("runtime-missing", "the sqlite-vec package is not installed")

        monkeypatch.setattr(vlc, "open_search_connection", runtime_missing)
        _populate(store)

        with pytest.raises(maintenance.SearchIndexMaintenanceError) as excinfo:
            maintenance.refresh_index(all=True, embedder=DeterministicEmbedder())

        assert excinfo.value.reason == "runtime-missing"
        assert "[search]" in excinfo.value.action
        assert _active_pointer(store) is None
        assert {row["state"] for row in _generation_rows(store)} == {"building"}
        report = maintenance.index_status()
        assert report["semantic"]["state"] == "building"
