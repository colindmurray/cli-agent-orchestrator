"""The M2.3 semantic lanes (cond-0644): eligibility, persisted metric
decoding, issue-level aggregation, mode surfaces, and explanations.

Every §10.3/§19.3 freshness condition is a named test: a semantic row is
eligible only when its source still exists, its stored ``content_version``
equals the current FTS document's version, and no dirty row exists for that
``(generation, document_key)``. The §19.7 mutation proof drops the freshness
joins and demonstrates the named exclusion test would fail.

Embeddings here are deterministic keyword→orthonormal-basis vectors, so
distances — and therefore ranks — are exact and explainable without model
weights. The real-model gate lives in the harness comparison module.
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import (
    _TRACKER_ORM_TABLE_NAMES,
    Base,
    TrackerCommentModel,
    TrackerIssueModel,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.services import tracker_ranked_search as rsearch
from cli_agent_orchestrator.services.embedding_adapter import EmbeddingCapabilityError
from cli_agent_orchestrator.services.search_engine_factory import SearchEngineError
from cli_agent_orchestrator.services.vector_lifecycle import (
    activate_generation,
    create_generation,
    refresh_generation,
)

_PRODUCTION_QUERY_EMBEDDER = rsearch._query_embedder

DIMENSIONS = 32


class _ConcurrentMissCache(dict):
    """Make every worker observe the first cache miss before any can load."""

    def __init__(self, workers):
        super().__init__()
        self._first_lookup = threading.Barrier(workers)
        self._seen_threads = set()
        self._seen_lock = threading.Lock()

    def _is_first_lookup(self):
        thread_id = threading.get_ident()
        with self._seen_lock:
            first_lookup = thread_id not in self._seen_threads
            self._seen_threads.add(thread_id)
        return first_lookup

    def __contains__(self, key):
        if self._is_first_lookup():
            present = super().__contains__(key)
            self._first_lookup.wait(timeout=5)
            return present
        return super().__contains__(key)

    def get(self, key, default=None):
        if self._is_first_lookup():
            value = super().get(key, default)
            self._first_lookup.wait(timeout=5)
            return value
        return super().get(key, default)


# ---------------------------------------------------------------------------
# Store fixtures and deterministic embedders
# ---------------------------------------------------------------------------


class SemanticStore:
    """A file-backed tracker store with the search projection installed."""

    def __init__(self, path):
        self.path = str(path)
        self.engine = create_engine(f"sqlite:///{self.path}")
        Base.metadata.create_all(
            bind=self.engine,
            tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
        )
        _migrate_tracker_search_projection(self.engine)

    def raw(self):
        return self.engine.raw_connection()

    def execute(self, sql, params=()):
        raw = self.raw()
        try:
            cursor = raw.execute(sql, params)
            rows = cursor.fetchall()
            raw.commit()
            return rows
        finally:
            raw.close()


@pytest.fixture
def store(tmp_path):
    db = SemanticStore(tmp_path / "semantic.db")
    yield db
    db.engine.dispose()


BASE_TS = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(days_offset: int) -> datetime:
    return BASE_TS + timedelta(days=days_offset)


class BasisEmbedder:
    """Deterministic token-frequency → orthonormal-basis embedder.

    Tokens map through an optional synonym table first, then to fixed basis
    dimensions; a text's vector is the L2-normalized token-frequency vector.
    Identical inputs always produce identical unit vectors, and shared tokens
    produce provably smaller cosine distances.
    """

    def __init__(self, vocab, synonyms=None, dimensions=DIMENSIONS):
        self.vocab = dict(vocab)
        self.synonyms = dict(synonyms or {})
        self.dimensions = dimensions

    def _vector(self, text_value):
        vec = np.zeros(self.dimensions, dtype=np.float64)
        for token in re.findall(r"[a-z0-9_]+", str(text_value).lower()):
            token = self.synonyms.get(token, token)
            index = self.vocab.get(token)
            if index is not None:
                vec[index] += 1.0
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        return np.asarray(vec, dtype="<f4")

    def embed(self, texts, *, batch_size=32, observer=None):  # noqa: ARG002
        return [self._vector(text).tobytes() for text in texts]


VOCAB = {
    "deploy": 0,
    "pipeline": 1,
    "bounce": 2,
    "receipt": 3,
    "lease": 4,
    "deadlock": 5,
    "fencing": 6,
    "token": 7,
    "rotation": 8,
    "widget": 9,
    "color": 10,
}
SYNONYMS = {
    "shipping": "deploy",
    "infrastructure": "pipeline",
    "stall": "deadlock",
    # The k-4 twins spell their terms with inflections the FTS index does not
    # stem, so only the semantic lanes match them; the §10.4 importance
    # contract below is then observable through the fused explanation.
    "leasing": "lease",
    "deadlocks": "deadlock",
}


def seed(store):
    session = sessionmaker(bind=store.engine)()
    rows = [
        # key, project, title, body, failing_command, important_comment
        (
            "k-1",
            "p1",
            "Deploy pipeline bounces on dry run",
            "the deploy pipeline returned no verified bounce receipt",
            None,
            None,
        ),
        (
            "k-2",
            "p1",
            "Widget color tuning wish",
            "the widget color palette needs softer defaults",
            None,
            None,
        ),
        (
            "k-3",
            "p1",
            "Election misbehaves under load",
            "title says little; the thread knows more",
            None,
            "lease deadlock fixed by fencing token rotation",
        ),
        (
            "k-4",
            "p1",
            "Lease deadlock in successor election",
            "successor election lease deadlock reproduces on restart",
            None,
            None,
        ),
    ]
    for key, project, title, body, command, important_body in rows:
        session.add(
            TrackerIssueModel(
                key=key,
                project_id=project,
                title=title,
                body=body,
                status="open",
                severity="P2",
                kind="bug",
                failing_command=command,
                labels="[]",
                created_at=_ts(1),
                updated_at=_ts(1),
            )
        )
    if rows[2][5]:
        session.add(
            TrackerCommentModel(issue_key="k-3", author="operator", body=rows[2][5], important=True)
        )
    # Importance twins on k-4: byte-different bodies whose basis vectors are
    # nearly parallel; only the importance bonus may let the farther-but-
    # important twin win, while its reported distance stays raw.
    session.flush()
    session.add(TrackerCommentModel(issue_key="k-4", author="twin-a", body="leasing deadlocks"))
    # Nine repetitions per query word: the important twin embeds only barely
    # farther from the query than the ordinary twin (cosine distance ≈0.001),
    # so the production ordering bonus (-0.05) — and nothing else — decides
    # the winner, while its reported distance stays the raw larger value.
    session.add(
        TrackerCommentModel(
            issue_key="k-4",
            author="twin-b",
            body=(
                "leasing leasing leasing leasing leasing leasing leasing "
                "leasing leasing deadlocks deadlocks deadlocks deadlocks "
                "deadlocks deadlocks deadlocks deadlocks deadlocks fencing"
            ),
            important=True,
        )
    )
    session.commit()
    session.close()


def build_active_generation(store, embedder=None, metadata_overrides=None):
    """Create, refresh, and activate one generation over the seeded corpus."""
    metadata = dict(MINIMAL_RECORD)
    if metadata_overrides:
        metadata.update(metadata_overrides)
    created = create_generation(metadata=metadata, target_engine=store.engine)
    refresh_generation(
        generation_id=created["generation_id"],
        embedder=embedder if embedder is not None else BasisEmbedder(VOCAB, SYNONYMS),
        db_path=store.path,
    )
    activate_generation(created["generation_id"], target_engine=store.engine)
    return created["generation_id"]


MINIMAL_RECORD = {
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


@pytest.fixture
def hybrid(store):
    seed(store)
    build_active_generation(store)
    embedder = BasisEmbedder(VOCAB, SYNONYMS)
    original_session = rsearch.SessionLocal
    original_embedder = rsearch._query_embedder
    rsearch.SessionLocal = sessionmaker(bind=store.engine)
    rsearch._query_embedder = lambda models_dir=None: embedder
    yield store
    rsearch.SessionLocal = original_session
    rsearch._query_embedder = original_embedder
    rsearch.reset_query_embedder_cache()


def search(service_store, query, **kwargs):
    request = rsearch.RankedSearchRequest(query=query, **{"project_ids": ("p1",), **kwargs})
    return rsearch.ranked_search(request)


def keys_of(response):
    return [r["issue"]["key"] for r in response["results"]]


def semantic_lane_names(response):
    names = set()
    for result in response["results"]:
        for lane in result["contributing_lanes"]:
            names.add(lane["lane"])
    return names


# ---------------------------------------------------------------------------
# Freshness eligibility (§10.3/§19.3): each condition pinned separately
# ---------------------------------------------------------------------------


class TestSemanticEligibility:
    # Every probe runs in mode="semantic" on purpose: a freshness condition
    # gates the semantic lanes only, and in hybrid mode the lexical lanes
    # would still serve an issue whose vector went stale — masking the
    # exclusion this class pins.
    def test_vector_for_deleted_source_is_excluded_while_the_blob_survives(self, hybrid):
        response = search(hybrid, "widget color tuning", mode="semantic", include_comments=False)
        assert "k-2" in keys_of(response)
        generation = hybrid.execute(
            "SELECT generation_id FROM tracker_vector_generations WHERE state = 'active'"
        )[0][0]
        hybrid.execute("DELETE FROM tracker_issues WHERE key = 'k-2'")
        blob_count = hybrid.execute(
            "SELECT COUNT(*) FROM tracker_search_vectors WHERE issue_key = 'k-2'"
        )[0][0]
        assert blob_count == 1, "fixture requires the orphaned vector blob to survive"

        after = search(hybrid, "widget color tuning", mode="semantic", include_comments=False)
        assert "k-2" not in keys_of(after)

    def test_stale_content_version_is_excluded_even_without_a_dirty_row(self, hybrid):
        response = search(hybrid, "deploy pipeline bounce", mode="semantic", include_comments=False)
        assert keys_of(response)[0] == "k-1"
        # Simulate derived-row drift directly: the stored vector now names an
        # older content_version than the current FTS document, and no dirty
        # row exists. Only the version join can exclude it.
        hybrid.execute("UPDATE tracker_issue_fts SET content_version = content_version + 10")
        stale = search(hybrid, "deploy pipeline bounce", mode="semantic", include_comments=False)
        assert "k-1" not in keys_of(stale)

    def test_dirty_row_excludes_an_otherwise_fresh_vector(self, hybrid, monkeypatch):
        response = search(hybrid, "deploy pipeline bounce", mode="semantic", include_comments=False)
        assert keys_of(response)[0] == "k-1"
        generation = hybrid.execute(
            "SELECT generation_id FROM tracker_vector_generations WHERE state = 'active'"
        )[0][0]
        document_key = hybrid.execute(
            "SELECT document_key FROM tracker_search_vectors "
            "WHERE generation_id = ? AND document_kind = 'issue'",
            (generation,),
        )[0][0]
        hybrid.execute(
            "INSERT INTO tracker_vector_dirty (generation_id, document_key, issue_key,\n"
            "    document_kind, source_id, content_version, document_schema_version, enqueued_at)\n"
            "SELECT generation_id, document_key, issue_key, document_kind, source_id,\n"
            "    content_version, 1, '2026-08-01T00:00:00.000Z'\n"
            "FROM tracker_search_vectors WHERE generation_id = ? AND document_key = ?",
            (generation, document_key),
        )
        # A live query would first drain the dirty row (§9.2) and re-embed the
        # document fresh. Hold the drain so the row is still dirty at scan
        # time: only the NOT EXISTS predicate can now exclude the vector.
        from cli_agent_orchestrator.services import vector_lifecycle as vlc

        monkeypatch.setattr(
            vlc,
            "drain_bounded_batch",
            lambda **kwargs: {"attempted": 0, "published": 0},
        )
        dirty = search(hybrid, "deploy pipeline bounce", mode="semantic", include_comments=False)
        assert "k-1" not in keys_of(dirty)

    def test_dirty_rows_are_generation_scoped(self, hybrid):
        response = search(hybrid, "deploy pipeline bounce", mode="semantic", include_comments=False)
        assert keys_of(response)[0] == "k-1"
        generation = hybrid.execute(
            "SELECT generation_id FROM tracker_vector_generations WHERE state = 'active'"
        )[0][0]
        hybrid.execute(
            "INSERT INTO tracker_vector_dirty (generation_id, document_key, issue_key,\n"
            "    document_kind, source_id, content_version, document_schema_version, enqueued_at)\n"
            "SELECT 'some-other-generation', document_key, issue_key, document_kind, source_id,\n"
            "    content_version, 1, '2026-08-01T00:00:00.000Z'\n"
            "FROM tracker_search_vectors WHERE generation_id = ? AND document_kind = 'issue'",
            (generation,),
        )
        still_served = search(
            hybrid, "deploy pipeline bounce", mode="semantic", include_comments=False
        )
        assert keys_of(still_served)[0] == "k-1"

    def test_mutation_permitting_stale_vectors_turns_the_exclusion_red(self, hybrid, monkeypatch):
        """§19.7 mutation proof: dropping the freshness joins must flip the
        named exclusion tests above."""
        hybrid.execute("UPDATE tracker_issue_fts SET content_version = content_version + 10")
        stale = search(hybrid, "deploy pipeline bounce", mode="semantic", include_comments=False)
        assert "k-1" not in keys_of(stale)

        monkeypatch.setattr(rsearch, "SEMANTIC_FRESHNESS_JOINS", "")
        monkeypatch.setattr(rsearch, "SEMANTIC_FRESHNESS_PREDICATE", "1 = 1\n")
        permitted = search(
            hybrid, "deploy pipeline bounce", mode="semantic", include_comments=False
        )
        try:
            assert "k-1" not in keys_of(permitted)
        except AssertionError:
            pass  # the mutation changed the outcome: the exclusion is pinned
        else:
            pytest.fail(
                "permitting stale semantic vectors did not change results; "
                "the freshness exclusion is unpinned"
            )


# ---------------------------------------------------------------------------
# Persisted metadata decides metric/normalization/element decoding (§10.3)
# ---------------------------------------------------------------------------


class FixedVectorEmbedder:
    """Maps each text to the vector of the first mapping key it contains."""

    def __init__(self, mapping, dimensions=DIMENSIONS):
        self.mapping = {key: np.asarray(vec, dtype="<f4") for key, vec in mapping.items()}
        self.dimensions = dimensions

    def embed(self, texts, *, batch_size=32, observer=None):  # noqa: ARG002
        results = []
        for text_value in texts:
            lowered = str(text_value).lower()
            chosen = None
            for key, vec in self.mapping.items():
                if key.lower() in lowered:
                    chosen = vec
                    break
            if chosen is None:
                chosen = np.zeros(self.dimensions, dtype="<f4")
            results.append(chosen.tobytes())
        return results


def _unit_basis(index, scale=1.0, dimensions=DIMENSIONS):
    vec = np.zeros(dimensions, dtype="<f4")
    vec[index] = scale
    return vec


class TestPersistedMetricDecoding:
    @staticmethod
    def _metric_probe_store(tmp_path_factory, metric, name):
        store = SemanticStore(tmp_path_factory.mktemp(name) / f"{name}.db")
        session = sessionmaker(bind=store.engine)()
        # Two issues whose documents embed to fixed, non-unit vectors; the
        # query sits at half e0, so cosine prefers doc-a and l2 prefers doc-b.
        session.add(
            TrackerIssueModel(
                key="doc-a",
                project_id="p1",
                title="alpha probe",
                body="alpha probe",
                status="open",
                kind="bug",
                labels="[]",
                created_at=_ts(1),
                updated_at=_ts(1),
            )
        )
        session.add(
            TrackerIssueModel(
                key="doc-b",
                project_id="p1",
                title="beta probe",
                body="beta probe",
                status="open",
                kind="bug",
                labels="[]",
                created_at=_ts(1),
                updated_at=_ts(1),
            )
        )
        session.commit()
        session.close()
        record = dict(MINIMAL_RECORD)
        record["distance_metric"] = metric
        created = create_generation(metadata=record, target_engine=store.engine)
        embedder = FixedVectorEmbedder(
            {
                "alpha probe": _unit_basis(0, scale=2.0),
                "beta probe": _unit_basis(1),
                "half probe": _unit_basis(0, scale=0.5),
            }
        )
        refresh_generation(
            generation_id=created["generation_id"],
            embedder=embedder,
            db_path=store.path,
        )
        activate_generation(created["generation_id"], target_engine=store.engine)
        original_session = rsearch.SessionLocal
        original_embedder = rsearch._query_embedder
        rsearch.SessionLocal = sessionmaker(bind=store.engine)
        rsearch._query_embedder = lambda models_dir=None: embedder
        return store, original_session, original_embedder, embedder

    def test_cosine_generation_ranks_by_cosine_distance(self, tmp_path_factory):
        store, session, original_embedder, embedder = self._metric_probe_store(
            tmp_path_factory, "cosine", "cosdb"
        )
        try:
            response = search(store, "half probe", mode="semantic")
            semantic_lanes = [
                entry
                for entry in response["results"][0]["contributing_lanes"]
                if entry["lane"] == "semantic-issue"
            ]
            assert response["mode_effective"] == "semantic"
            assert keys_of(response)[0] == "doc-a"
            assert semantic_lanes[0]["raw_score"] == pytest.approx(0.0, abs=1e-5)
        finally:
            rsearch.SessionLocal = session
            rsearch._query_embedder = original_embedder
            store.engine.dispose()

    def test_l2_generation_ranks_by_l2_not_the_process_default(self, tmp_path_factory):
        # The adapter's process default is cosine; if the scan consulted that
        # default instead of the persisted row, this store would rank doc-a
        # first exactly like the cosine store above. It must not.
        store, session, original_embedder, embedder = self._metric_probe_store(
            tmp_path_factory, "l2", "l2db"
        )
        try:
            response = search(store, "half probe", mode="semantic")
            assert response["mode_effective"] == "semantic"
            assert keys_of(response)[0] == "doc-b"
            assert response["generations"]["vector_generation"]["distance_metric"] == "l2"
        finally:
            rsearch.SessionLocal = session
            rsearch._query_embedder = original_embedder
            store.engine.dispose()

    def test_dimension_mismatch_between_model_and_generation_degrades(self, hybrid, monkeypatch):
        monkeypatch.setattr(
            rsearch,
            "_query_embedder",
            lambda models_dir=None: BasisEmbedder(VOCAB, SYNONYMS, dimensions=8),
        )
        response = search(hybrid, "deploy pipeline", mode="hybrid")
        assert response["mode_effective"] == "lexical"
        assert any("dimension" in reason for reason in response["degradation"]["reasons"])
        assert response["degradation"]["lanes"]["semantic-issue"]["available"] is False


# ---------------------------------------------------------------------------
# Undefined distances: zero embeddings must be skipped, not fatal (§10.3)
# ---------------------------------------------------------------------------


class TestUndefinedDistances:
    def test_zero_embedding_issue_document_is_excluded_from_cosine_lanes(self, hybrid):
        # k-3's issue document carries no vocabulary token, so its stored
        # embedding is the zero vector and its cosine distance is NULL —
        # unrankable, not closest. The scan must skip it instead of crashing
        # the lanes; k-3 stays reachable through its comment document, whose
        # vector is defined.
        response = search(hybrid, "deploy pipeline bounce", mode="semantic")
        assert response["mode_effective"] == "semantic"
        assert keys_of(response), "the scan survived a corpus holding a zero embedding"
        by_key = {r["issue"]["key"]: r for r in response["results"]}
        assert "k-3" in by_key, "fixture requires k-3 served via its comment document"
        for entry in by_key["k-3"]["contributing_lanes"]:
            assert entry["lane"] != "semantic-issue"

    def test_l2_generation_ranks_a_zero_embedding_instead_of_dropping_it(self, tmp_path_factory):
        store = SemanticStore(tmp_path_factory.mktemp("l2zero") / "l2zero.db")
        session = sessionmaker(bind=store.engine)()
        session.add(
            TrackerIssueModel(
                key="doc-near",
                project_id="p1",
                title="alpha probe",
                body="alpha probe",
                status="open",
                kind="bug",
                labels="[]",
                created_at=_ts(1),
                updated_at=_ts(1),
            )
        )
        session.add(
            TrackerIssueModel(
                key="doc-zero",
                project_id="p1",
                title="empty probe",
                body="empty probe",
                status="open",
                kind="bug",
                labels="[]",
                created_at=_ts(1),
                updated_at=_ts(1),
            )
        )
        session.commit()
        session.close()
        record = dict(MINIMAL_RECORD)
        record["distance_metric"] = "l2"
        created = create_generation(metadata=record, target_engine=store.engine)
        embedder = FixedVectorEmbedder(
            {
                "alpha probe": _unit_basis(0),
                "empty probe": np.zeros(DIMENSIONS, dtype="<f4"),
                "alpha query": _unit_basis(0),
            }
        )
        refresh_generation(
            generation_id=created["generation_id"], embedder=embedder, db_path=store.path
        )
        activate_generation(created["generation_id"], target_engine=store.engine)
        original_session = rsearch.SessionLocal
        original_embedder = rsearch._query_embedder
        rsearch.SessionLocal = sessionmaker(bind=store.engine)
        rsearch._query_embedder = lambda models_dir=None: embedder
        try:
            response = search(store, "alpha query", mode="semantic")
            assert response["mode_effective"] == "semantic"
            # L2 defines every distance: the zero vector sits at the origin,
            # one unit from the unit query, and must rank — never vanish.
            ranked = {r["issue"]["key"]: r for r in response["results"]}
            assert set(ranked) == {"doc-near", "doc-zero"}
            zero_lane = [
                entry
                for entry in ranked["doc-zero"]["contributing_lanes"]
                if entry["lane"] == "semantic-issue"
            ]
            assert zero_lane
            assert zero_lane[0]["raw_score"] == pytest.approx(1.0, abs=1e-6)
        finally:
            rsearch.SessionLocal = original_session
            rsearch._query_embedder = original_embedder
            store.engine.dispose()


# ---------------------------------------------------------------------------
# Issue-level aggregation of comment documents (§10.4)
# ---------------------------------------------------------------------------


class TestSemanticAggregation:
    def test_best_two_comments_retained_and_reported(self, hybrid):
        response = search(hybrid, "fencing token rotation deadlock", mode="hybrid")
        by_key = {r["issue"]["key"]: r for r in response["results"]}
        k3 = by_key["k-3"]["winning_comment"]
        assert k3 is not None
        assert k3["important"] is True
        assert k3["retained_hits"] <= 2
        # The lexical comment lane also matches here; either lane may own the
        # navigation target, but the owner is always named and a semantic
        # owner reports its raw distance.
        assert k3["source_lane"] in ("comment-bm25", "semantic-comment")
        if k3["source_lane"] == "semantic-comment":
            assert isinstance(k3["raw_distance"], float)

    def test_important_bonus_flips_rank_while_distance_stays_raw(self, hybrid, monkeypatch):
        # The important twin's body embeds slightly farther from the query
        # than the ordinary twin's. Only the ordering-only bonus can let the
        # important twin win; its REPORTED distance must stay the raw,
        # larger value — never relabeled by importance.
        normal = search(hybrid, "lease deadlock", mode="hybrid", limit=50)
        winner = next(r for r in normal["results"] if r["issue"]["key"] == "k-4")["winning_comment"]
        assert winner is not None
        assert winner["source_lane"] == "semantic-comment"
        assert winner["important"] is True
        boosted_distance = winner["raw_distance"]

        monkeypatch.setattr(rsearch, "SEMANTIC_IMPORTANT_COMMENT_BONUS", 0.0)
        reverted = search(hybrid, "lease deadlock", mode="hybrid", limit=50)
        reverted_winner = next(r for r in reverted["results"] if r["issue"]["key"] == "k-4")[
            "winning_comment"
        ]
        assert reverted_winner["important"] is False
        assert reverted_winner["raw_distance"] == pytest.approx(0.0, abs=1e-6)
        assert boosted_distance > reverted_winner["raw_distance"], (
            "the important twin must win despite the strictly larger raw "
            "distance; otherwise the rank-vs-distance contract is unpinned"
        )


# ---------------------------------------------------------------------------
# Mode surfaces: semantic-only, hybrid fusion, visible degradation (§13.5)
# ---------------------------------------------------------------------------


class TestModeSurfaces:
    def test_query_embedder_cold_load_is_single_flight(self, monkeypatch):
        """Concurrent misses load one model and all callers reuse it."""
        from cli_agent_orchestrator.services import embedding_adapter

        workers = 5
        sentinel = object()
        loads = []
        load_lock = threading.Lock()

        def load(models_dir=None):
            with load_lock:
                loads.append(models_dir)
            return sentinel

        monkeypatch.setattr(rsearch, "_EMBEDDER_CACHE", _ConcurrentMissCache(workers))
        monkeypatch.setattr(embedding_adapter, "load_embedder", load)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(lambda _: _PRODUCTION_QUERY_EMBEDDER("/models"), range(workers))
            )

        assert results == [sentinel] * workers
        assert loads == ["/models"]

    def test_query_embedder_failed_load_is_not_cached(self, monkeypatch):
        """A failed first load leaves the next caller able to retry."""
        from cli_agent_orchestrator.services import embedding_adapter

        sentinel = object()
        loads = []

        def load(models_dir=None):
            loads.append(models_dir)
            if len(loads) == 1:
                raise RuntimeError("cold load failed")
            return sentinel

        monkeypatch.setattr(rsearch, "_EMBEDDER_CACHE", {})
        monkeypatch.setattr(embedding_adapter, "load_embedder", load)

        with pytest.raises(RuntimeError, match="cold load failed"):
            _PRODUCTION_QUERY_EMBEDDER("/models")

        assert _PRODUCTION_QUERY_EMBEDDER("/models") is sentinel
        assert loads == ["/models", "/models"]

    def test_query_embedder_cache_is_evicted_when_activation_changes_identity(
        self, store, monkeypatch
    ):
        """A cached A embedder must not produce a query vector for active B."""
        seed(store)

        class IdentityEmbedder(BasisEmbedder):
            def __init__(self, metadata):
                super().__init__(VOCAB, SYNONYMS)
                self.metadata = dict(metadata)

        model_a = dict(MINIMAL_RECORD)
        model_b = dict(MINIMAL_RECORD)
        model_b.update(
            {"model_id": "test/model-b", "model_revision": "rev-b", "artifact_sha256": "b" * 64}
        )
        embed_a = IdentityEmbedder(model_a)
        embed_b = IdentityEmbedder(model_b)
        first = build_active_generation(store, embedder=embed_a, metadata_overrides=model_a)

        session_factory = sessionmaker(bind=store.engine)
        monkeypatch.setattr(rsearch, "SessionLocal", session_factory)
        from cli_agent_orchestrator.services import embedding_adapter

        current = [embed_a]
        loads = []

        def load(_models_dir=None):
            loads.append(current[0])
            return current[0]

        monkeypatch.setattr(embedding_adapter, "load_embedder", load)
        monkeypatch.setattr(rsearch, "_query_embedder", _PRODUCTION_QUERY_EMBEDDER)
        rsearch.reset_query_embedder_cache()
        db = session_factory()
        try:
            conn = db.connection()
            context = rsearch.resolve_semantic_context(
                conn, rsearch._meta_snapshot(conn), "deploy", []
            )
        finally:
            db.close()
        assert context is not None and context.generation_id == first
        assert len(loads) == 1

        second = build_active_generation(store, embedder=embed_b, metadata_overrides=model_b)
        current[0] = embed_b
        db = session_factory()
        try:
            conn = db.connection()
            reasons = []
            context = rsearch.resolve_semantic_context(
                conn, rsearch._meta_snapshot(conn), "deploy", reasons
            )
        finally:
            db.close()
        assert context is not None and context.generation_id == second
        assert len(loads) == 2, "activation must force a reload of the cached query embedder"
        assert reasons == []

    def test_semantic_mode_serves_semantic_lanes_only(self, hybrid):
        # Paraphrase through the synonym table: no token overlap with any
        # stored field, so both lexical lanes and the exact lane find nothing;
        # only the semantic lanes can retrieve k-1.
        response = search(hybrid, "shipping infrastructure stall", mode="semantic")
        assert response["mode_effective"] == "semantic"
        assert response["degradation"]["reasons"] == []
        lanes = semantic_lane_names(response)
        assert lanes and lanes <= {"semantic-issue", "semantic-comment"}
        assert "k-1" in keys_of(response)
        assert response["diagnostics"]["semantic"]["served"] is True

    def test_hybrid_mode_fuses_lexical_and_semantic_lanes(self, hybrid):
        response = search(hybrid, "deploy pipeline bounce receipt", mode="hybrid")
        assert response["mode_effective"] == "hybrid"
        assert response["degradation"]["reasons"] == []
        top = response["results"][0]
        lane_names = {entry["lane"] for entry in top["contributing_lanes"]}
        assert "issue-bm25" in lane_names
        assert lane_names & {"semantic-issue", "semantic-comment"}
        assert response["degradation"]["lanes"]["semantic-issue"]["available"] is True
        assert response["degradation"]["lanes"]["semantic-comment"]["available"] is True

    def test_no_active_generation_degrades_visibly_to_lexical(self, tmp_path):
        store = SemanticStore(tmp_path / "nogen.db")
        seed(store)
        original_session = rsearch.SessionLocal
        rsearch.SessionLocal = sessionmaker(bind=store.engine)
        try:
            response = search(store, "deploy pipeline", mode="hybrid")
            assert response["mode_effective"] == "lexical"
            assert any(
                "no active vector generation" in reason
                for reason in response["degradation"]["reasons"]
            )
            assert response["degradation"]["lanes"]["semantic-issue"]["available"] is False
            assert keys_of(response), "lexical lanes still serve results"
        finally:
            rsearch.SessionLocal = original_session
            store.engine.dispose()

    def test_unprepared_model_degrades_visibly(self, hybrid, monkeypatch):
        def unprepared(models_dir=None):
            raise EmbeddingCapabilityError(
                "unprepared", "no generation metadata; run the prepare verb first"
            )

        monkeypatch.setattr(rsearch, "_query_embedder", unprepared)
        response = search(hybrid, "deploy pipeline", mode="hybrid")
        assert response["mode_effective"] == "lexical"
        assert any("embedding model unavailable" in r for r in response["degradation"]["reasons"])

    def test_runtime_missing_degrades_visibly(self, hybrid, monkeypatch):
        def runtime_missing(db_path):
            raise SearchEngineError("runtime-missing", "the sqlite-vec package is not installed")

        monkeypatch.setattr(rsearch, "_open_search_engine", runtime_missing)
        response = search(hybrid, "deploy pipeline", mode="hybrid")
        assert response["mode_effective"] == "lexical"
        assert any("vector scan unavailable" in r for r in response["degradation"]["reasons"])

    def test_query_embedding_failure_degrades_visibly(self, hybrid, monkeypatch):
        class FailingEmbedder(BasisEmbedder):
            def embed(self, texts, *, batch_size=32, observer=None):
                raise RuntimeError("simulated inference outage")

        monkeypatch.setattr(
            rsearch, "_query_embedder", lambda models_dir=None: FailingEmbedder(VOCAB)
        )
        response = search(hybrid, "deploy pipeline", mode="hybrid")
        assert response["mode_effective"] == "lexical"
        assert any("query embedding failed" in r for r in response["degradation"]["reasons"])

    def test_in_memory_store_degrades_instead_of_scanning_a_lookalike(self, tmp_path, monkeypatch):
        memory_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(
            bind=memory_engine,
            tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
        )
        _migrate_tracker_search_projection(memory_engine)
        # Activation demands full coverage of live documents, and an
        # in-memory corpus can never be embedded (there is no second-
        # connection path) — so the generation activates empty and the issue
        # lands afterwards, exactly the state a real in-memory session holds.
        created = create_generation(metadata=dict(MINIMAL_RECORD), target_engine=memory_engine)
        activate_generation(created["generation_id"], target_engine=memory_engine)
        seed_memory = sessionmaker(bind=memory_engine)()
        seed_memory.add(
            TrackerIssueModel(
                key="m-1",
                project_id="p1",
                title="deploy pipeline probe",
                body="deploy pipeline",
                status="open",
                kind="bug",
                labels="[]",
                created_at=_ts(1),
                updated_at=_ts(1),
            )
        )
        seed_memory.commit()
        seed_memory.close()
        monkeypatch.setattr(
            rsearch,
            "_query_embedder",
            lambda models_dir=None: BasisEmbedder(VOCAB, SYNONYMS),
        )
        original_session = rsearch.SessionLocal
        rsearch.SessionLocal = sessionmaker(bind=memory_engine)
        try:
            response = search(None, "deploy pipeline", mode="hybrid")
            assert response["mode_effective"] == "lexical"
            assert any("filesystem path" in r for r in response["degradation"]["reasons"])
            assert response["degradation"]["lanes"]["semantic-issue"]["available"] is False
            assert keys_of(response), "lexical lanes still serve results"
        finally:
            rsearch.SessionLocal = original_session
            memory_engine.dispose()

    def test_drain_failure_is_a_visible_partial_not_a_failed_query(self, hybrid, monkeypatch):
        from cli_agent_orchestrator.services import vector_lifecycle as vlc

        def broken_drain(**kwargs):  # noqa: ARG001
            raise RuntimeError("simulated refresh outage")

        monkeypatch.setattr(vlc, "drain_bounded_batch", broken_drain)
        response = search(hybrid, "deploy pipeline bounce", mode="hybrid")
        assert response["mode_effective"] == "hybrid"
        assert any("query-time refresh skipped" in r for r in response["degradation"]["reasons"])
        assert keys_of(response), "semantic lanes still served despite the skipped drain"

    def test_include_comments_false_drops_only_the_comment_side(self, hybrid):
        response = search(hybrid, "deploy pipeline bounce", mode="hybrid", include_comments=False)
        assert response["mode_effective"] == "hybrid"
        for result in response["results"]:
            assert all(
                entry["lane"] != "semantic-comment" for entry in result["contributing_lanes"]
            )


# ---------------------------------------------------------------------------
# Explanations carry per-lane diagnostics and coverage metadata (§10.5)
# ---------------------------------------------------------------------------


class TestSemanticExplanations:
    def test_every_result_carries_complete_semantic_aware_explanation(self, hybrid):
        response = search(hybrid, "lease deadlock fencing", mode="hybrid", limit=10)
        assert response["results"]
        for result in response["results"]:
            assert result["rank_score"] >= 0.0
            for entry in result["contributing_lanes"]:
                assert set(entry) == {"lane", "rank", "raw_score"}
                assert entry["rank"] >= 1
                if entry["lane"].startswith("semantic"):
                    assert isinstance(entry["raw_score"], float)
                    assert entry["raw_score"] >= 0.0
        top = response["results"][0]
        assert top["winning_comment"] is not None
        assert top["winning_comment"]["source_lane"] in ("comment-bm25", "semantic-comment")

    def test_generations_echo_names_the_served_vector_generation(self, hybrid):
        response = search(hybrid, "deploy pipeline", mode="hybrid")
        served = response["generations"]["vector_generation"]
        assert served["state"] == "active"
        assert served["model_id"] == MINIMAL_RECORD["model_id"]
        assert served["distance_metric"] == "cosine"
        assert served["element_type"] == "float32"
        assert served["dimensions"] == DIMENSIONS
        pointer = response["generations"]["active_vector_generation"]
        assert pointer == served["generation_id"]

    def test_diagnostics_report_semantic_coverage_and_refresh(self, hybrid):
        response = search(hybrid, "deploy pipeline bounce", mode="hybrid")
        semantic_diag = response["diagnostics"]["semantic"]
        assert semantic_diag["served"] is True
        assert semantic_diag["generation_id"] == (
            response["generations"]["vector_generation"]["generation_id"]
        )
        assert semantic_diag["issue_vectors_returned"] >= 1
        assert isinstance(semantic_diag["comment_issues_returned"], int)
        assert isinstance(semantic_diag["query_refresh"], dict)
        assert "semantic-scan" in response["diagnostics"]["lane_elapsed_ms"]

    def test_status_reports_active_generation_details(self, hybrid):
        status = rsearch.search_status()
        semantic = status["semantic"]
        assert semantic["active_generation"]
        assert semantic["state"] == "active"
        assert semantic["model_id"] == MINIMAL_RECORD["model_id"]
        assert semantic["distance_metric"] == "cosine"
        assert semantic["dimensions"] == DIMENSIONS
