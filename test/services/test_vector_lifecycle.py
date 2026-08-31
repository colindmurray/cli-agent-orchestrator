"""The M2.2 vector lifecycle (cond-0643) against design §9.3/§13.3/§19.3.

Every concurrency/failure row of §19.3 is a named test below. Races are
simulated deterministically by embedders that perform a concurrent mutation
between the refresh loop's read phase and its publish phase — exactly the
interleaving the compare-and-set contract exists for.
"""

import hashlib
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from cli_agent_orchestrator.clients import tracker_search_schema as schema
from cli_agent_orchestrator.clients.database import (
    _TRACKER_ORM_TABLE_NAMES,
    Base,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.services import vector_lifecycle as vlc
from cli_agent_orchestrator.services.embedding_adapter import EmbeddingCapabilityError

DIMENSIONS = 384


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/vector-lifecycle.db")
    Base.metadata.create_all(
        bind=eng,
        tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
    )
    with eng.begin() as conn:
        conn.execute(text("INSERT INTO tracker_projects (id, name) VALUES ('p', 'P')"))
    _migrate_tracker_search_projection(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def db_file(store):
    return Path(store.url.database)


def _raw(store):
    return store.raw_connection()


def _seed_issue(eng, key="cao-1", title="first", **extra):
    columns = {"key": key, "project_id": "p", "title": title, **extra}
    names = ", ".join(columns)
    params = ", ".join("?" for _ in columns)
    raw = eng.raw_connection()
    try:
        raw.execute(
            f"INSERT INTO tracker_issues ({names}) VALUES ({params})", tuple(columns.values())
        )
        raw.commit()
    finally:
        raw.close()
    return key


def _seed_comment(eng, issue_key="cao-1", body="a comment", author="alice"):
    raw = eng.raw_connection()
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


def _update_issue_title(eng, key, title):
    raw = eng.raw_connection()
    try:
        raw.execute("UPDATE tracker_issues SET title = ? WHERE key = ?", (title, key))
        raw.commit()
    finally:
        raw.close()


def _delete_issue(eng, key):
    raw = eng.raw_connection()
    try:
        raw.execute("DELETE FROM tracker_issues WHERE key = ?", (key,))
        raw.commit()
    finally:
        raw.close()


def _dirty_rows(eng, generation_id=None):
    sql = (
        f"SELECT generation_id, document_key, content_version, document_schema_version,\n"
        f"       attempt_count, next_attempt_at, last_error\n"
        f"FROM {schema.VECTOR_DIRTY_TABLE}"
    )
    params = ()
    if generation_id is not None:
        sql += " WHERE generation_id = ?"
        params = (generation_id,)
    raw = eng.raw_connection()
    try:
        cursor = raw.execute(sql + " ORDER BY document_key", params)
        names = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
    finally:
        raw.close()
    return [dict(zip(names, row)) for row in rows]


def _vectors(eng, generation_id):
    raw = eng.raw_connection()
    try:
        cursor = raw.execute(
            f"SELECT document_key, content_version, length(embedding) AS blob_bytes,\n"
            f"       content_sha256\n"
            f"FROM {schema.SEARCH_VECTORS_TABLE} WHERE generation_id = ?\n"
            f"ORDER BY document_key",
            (generation_id,),
        )
        names = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
    finally:
        raw.close()
    return [dict(zip(names, row)) for row in rows]


def _pointer(eng):
    raw = eng.raw_connection()
    try:
        return raw.execute(
            f"SELECT active_vector_generation FROM {schema.SEARCH_META_TABLE} "
            "WHERE singleton = 1"
        ).fetchone()[0]
    finally:
        raw.close()


def _generation_state(eng, generation_id):
    raw = eng.raw_connection()
    try:
        row = raw.execute(
            f"SELECT state, failure FROM {schema.VECTOR_GENERATIONS_TABLE} "
            "WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()
    finally:
        raw.close()
    return None if row is None else (row[0], row[1])


def semantic_rows(eng, generation_id):
    """The query-time eligibility contract: live source AND exact version."""
    sql = f"""
        SELECT v.document_key
          FROM {schema.SEARCH_VECTORS_TABLE} AS v
          LEFT JOIN tracker_issues AS i ON v.document_kind = 'issue' AND i.id = v.source_id
          LEFT JOIN tracker_issue_comments AS c
                 ON v.document_kind = 'comment' AND c.id = v.source_id
          LEFT JOIN {schema.ISSUE_FTS_TABLE} AS fi ON v.document_kind = 'issue'
                 AND fi.rowid = v.source_id
          LEFT JOIN {schema.COMMENT_FTS_TABLE} AS fc ON v.document_kind = 'comment'
                 AND fc.rowid = v.source_id
         WHERE v.generation_id = ?
           AND ((v.document_kind = 'issue' AND i.id IS NOT NULL
                 AND fi.content_version = v.content_version)
             OR (v.document_kind = 'comment' AND c.id IS NOT NULL
                 AND fc.content_version = v.content_version))
    """
    raw = eng.raw_connection()
    try:
        return sorted(row[0] for row in raw.execute(sql, (generation_id,)).fetchall())
    finally:
        raw.close()


def blob_for(text_value: str, dimensions: int = DIMENSIONS) -> bytes:
    digest = hashlib.sha256(text_value.encode("utf-8")).digest()
    needed = dimensions * 4
    expanded = (digest * (needed // len(digest) + 1))[:needed]
    return bytes(expanded)


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


class FakeEmbedder:
    """Derived from production LoadedEmbedder.embed's exact signature."""

    def __init__(self, metadata=None, fail_predicate=None, dimensions=DIMENSIONS):
        self.metadata = dict(metadata or MINIMAL_RECORD)
        self.fail_predicate = fail_predicate
        self.dimensions = dimensions
        self.batches = []

    def embed(self, texts, *, batch_size=32, observer=None):  # noqa: ARG002
        self.batches.append(list(texts))
        if self.fail_predicate is not None:
            failures = [t for t in texts if self.fail_predicate(t)]
            if failures:
                raise RuntimeError(f"simulated embedding outage for {failures!r}")
        return [blob_for(t, self.dimensions) for t in texts]


class MutatingEmbedder(FakeEmbedder):
    """Performs a concurrent mutation mid-embed, after the read phase."""

    def __init__(self, mutate, **kwargs):
        super().__init__(**kwargs)
        self.mutate = mutate

    def embed(self, texts, *, batch_size=32, observer=None):
        self.batches.append(list(texts))
        self.mutate()
        if self.fail_predicate is not None:
            failures = [t for t in texts if self.fail_predicate(t)]
            if failures:
                raise RuntimeError(f"simulated embedding outage for {failures!r}")
        return [blob_for(t, self.dimensions) for t in texts]


# ---------------------------------------------------------------------------
# Document builder (§9.1)
# ---------------------------------------------------------------------------


class TestDocumentBuilder:
    def test_issue_form_matches_the_design_identity_document(self):
        text_out = vlc.issue_document_text(
            {
                "title": "refresh races",
                "component": "issue-tracker/search",
                "failing_command": "conduct up",
                "actual_outcome": "stale served",
                "expected_outcome": "discard",
                "reproduction_steps": "run twice",
                "observed_revision": "abc123",
                "body": "body text",
                "resolution": None,
            }
        )
        assert text_out == (
            "title: refresh races\n"
            "component: issue-tracker/search\n"
            "command: conduct up\n"
            "actual: stale served\n"
            "expected: discard\n"
            "reproduction: run twice\n"
            "observed revision: abc123\n"
            "body: body text\n"
            "resolution:"
        )

    def test_comment_form_matches_the_design_compact_prefix(self):
        text_out = vlc.comment_document_text(
            {
                "issue_key": "cao-7",
                "issue_title": "the title",
                "component": "c",
                "author": "bob",
                "body": "line one line two",
            }
        )
        assert text_out == (
            "issue: cao-7 — the title\ncomponent: c\ncomment by bob:\nline one line two"
        )

    def test_normalizes_whitespace_and_truncates_long_fields(self):
        long_body = "x" * (vlc.DOCUMENT_FIELD_LIMIT + 500)
        text_out = vlc.issue_document_text({"title": "a\n b   c", "body": long_body})
        lines = {}
        for line in text_out.split("\n"):
            if ": " in line:
                name, value = line.split(": ", 1)
                lines[name] = value
        assert lines["title"] == "a b c"
        assert len(lines["body"]) == vlc.DOCUMENT_FIELD_LIMIT


# ---------------------------------------------------------------------------
# Generation creation + bulk enqueue (§13.3 first half)
# ---------------------------------------------------------------------------


class TestCreateGeneration:
    def test_creation_enqueues_every_live_document_in_one_transaction(self, store):
        _seed_issue(store, "cao-1")
        _seed_issue(store, "cao-2", title="second")
        _seed_comment(store, issue_key="cao-1")

        result = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)

        state, failure = _generation_state(store, result["generation_id"])
        assert (state, failure) == ("building", None)
        assert result["enqueued_documents"] == 3
        keys = {row["document_key"] for row in _dirty_rows(store, result["generation_id"])}
        assert keys == {"issue:cao-1", "issue:cao-2", "comment:1"}
        versions = {
            row["document_key"]: row["content_version"]
            for row in _dirty_rows(store, result["generation_id"])
        }
        raw = _raw(store)
        try:
            fts_versions = {
                f"issue:{key}": version
                for key, version in raw.execute(
                    f"SELECT issue_key, content_version FROM {schema.ISSUE_FTS_TABLE}"
                ).fetchall()
            }
        finally:
            raw.close()
        assert {k: v for k, v in versions.items() if k.startswith("issue:")} == fts_versions

    def test_triggers_take_over_after_commit_for_building_generations(self, store):
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        before = len(_dirty_rows(store, created["generation_id"]))
        _seed_issue(store, "cao-3", title="after create")
        after = _dirty_rows(store, created["generation_id"])
        assert before == 0 and len(after) == 1
        assert after[0]["document_key"] == "issue:cao-3"

    def test_refuses_unprepared_metadata_typed_and_writes_nothing(self, store, tmp_path):
        with pytest.raises(vlc.VectorLifecycleError) as excinfo:
            vlc.create_generation(models_dir=tmp_path / "absent", target_engine=store)
        assert excinfo.value.reason == "unprepared"
        raw = _raw(store)
        try:
            count = raw.execute(
                f"SELECT COUNT(*) FROM {schema.VECTOR_GENERATIONS_TABLE}"
            ).fetchone()[0]
        finally:
            raw.close()
        assert count == 0

    def test_refuses_incomplete_record_with_typed_reason(self, store):
        broken = {k: v for k, v in MINIMAL_RECORD.items() if k != "artifact_sha256"}
        with pytest.raises(vlc.VectorLifecycleError) as excinfo:
            vlc.create_generation(metadata=broken, target_engine=store)
        assert excinfo.value.reason == "metadata-incomplete"

    def test_refuses_missing_schema_atomically(self, tmp_path):
        bare = create_engine(f"sqlite:///{tmp_path}/bare.db")
        try:
            with pytest.raises(vlc.VectorLifecycleError) as excinfo:
                vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=bare)
            assert excinfo.value.reason == "schema-missing"
        finally:
            bare.dispose()


# ---------------------------------------------------------------------------
# §19.3 named matrix tests
# ---------------------------------------------------------------------------


class TestConcurrencyAndFailureMatrix:
    def test_refresh_selects_only_the_generation_matching_the_bound_embedder(self, store, db_file):
        """A model migration builds B without redirtying or re-embedding active A."""
        _seed_issue(store, "cao-1", title="shared width")
        model_a = dict(MINIMAL_RECORD)
        model_b = dict(MINIMAL_RECORD)
        model_b.update(
            {"model_id": "test/model-b", "model_revision": "rev-b", "artifact_sha256": "b" * 64}
        )
        first = vlc.create_generation(metadata=model_a, target_engine=store)
        first_refresh = vlc.refresh_generation(
            generation_id=first["generation_id"],
            db_path=str(db_file),
            embedder=FakeEmbedder(metadata=model_a),
        )
        assert first_refresh["published"] == 1
        vlc.activate_generation(first["generation_id"], target_engine=store)

        second = vlc.create_generation(metadata=model_b, target_engine=store)
        assert _dirty_rows(store, first["generation_id"]) == []
        assert len(_dirty_rows(store, second["generation_id"])) == 1

        migrated = vlc.refresh_generation(
            db_path=str(db_file), embedder=FakeEmbedder(metadata=model_b)
        )

        assert migrated["published"] == 1
        assert len(_vectors(store, first["generation_id"])) == 1
        assert len(_vectors(store, second["generation_id"])) == 1
        assert _dirty_rows(store, first["generation_id"]) == []
        assert _dirty_rows(store, second["generation_id"]) == []

    def test_mutation_during_embedding_cannot_clear_a_newer_dirty_row(self, store, db_file):
        """§19.3: mutation during embedding cannot clear a newer dirty row."""
        _seed_issue(store, "cao-1", title="before")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)

        racing = MutatingEmbedder(lambda: _update_issue_title(store, "cao-1", "after"))
        result = vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=racing
        )

        assert result["discarded_stale"] == 1
        assert result["published"] == 0
        assert _vectors(store, created["generation_id"]) == []
        dirty = _dirty_rows(store, created["generation_id"])
        assert len(dirty) == 1
        # The newer version owns the row with reset retry state.
        assert dirty[0]["attempt_count"] == 0
        assert dirty[0]["last_error"] is None

    def test_deletion_during_embedding_cannot_resurrect_a_semantic_result(self, store, db_file):
        """§19.3: deletion during embedding cannot resurrect a semantic result."""
        _seed_issue(store, "cao-1", title="doomed")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)

        racing = MutatingEmbedder(lambda: _delete_issue(store, "cao-1"))
        result = vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=racing
        )

        assert result["published"] == 0
        assert _vectors(store, created["generation_id"]) == []
        assert _dirty_rows(store, created["generation_id"]) == []
        assert semantic_rows(store, created["generation_id"]) == []

    def test_two_concurrent_refreshers_duplicate_work_but_only_matching_publishes(
        self, store, db_file
    ):
        """§19.3: two refreshers may duplicate work; only matching publishes."""
        _seed_issue(store, "cao-1", title="contested")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)

        def inner_refresher_completes_first():
            vlc.refresh_generation(
                generation_id=created["generation_id"],
                db_path=str(db_file),
                embedder=FakeEmbedder(),
            )

        outer = MutatingEmbedder(inner_refresher_completes_first)
        outer_result = vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=outer
        )

        vectors = _vectors(store, created["generation_id"])
        assert len(vectors) == 1
        assert outer_result["published"] == 0
        assert outer_result["discarded_stale"] == 1
        published_version = vectors[0]["content_version"]
        fts_raw = _raw(store)
        try:
            fts_version = fts_raw.execute(
                f"SELECT content_version FROM {schema.ISSUE_FTS_TABLE} WHERE rowid = 1"
            ).fetchone()[0]
        finally:
            fts_raw.close()
        assert published_version == fts_version
        assert semantic_rows(store, created["generation_id"]) == ["issue:cao-1"]

    def test_embedding_failure_retains_dirty_error_state_and_excludes_stale_semantic_data(
        self, store, db_file
    ):
        """§19.3: embedding failure retains dirty/error; stale data excluded."""
        _seed_issue(store, "cao-1", title="v1 title")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        first = vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        assert first["published"] == 1

        _update_issue_title(store, "cao-1", "v2 title")
        failing = FakeEmbedder(fail_predicate=lambda t: "v2 title" in t)
        second = vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=failing
        )
        assert second["failed"] == 1 and second["published"] == 0

        dirty = _dirty_rows(store, created["generation_id"])
        assert len(dirty) == 1
        assert dirty[0]["attempt_count"] == 1
        assert dirty[0]["last_error"] is not None
        assert dirty[0]["next_attempt_at"] is not None
        vectors = _vectors(store, created["generation_id"])
        assert len(vectors) == 1 and vectors[0]["document_key"] == "issue:cao-1"
        # The proven vector predates the current version: never served fresh.
        assert semantic_rows(store, created["generation_id"]) == []

    def test_stale_failure_cannot_overwrite_a_newer_dirty_rows_retry_state(self, store, db_file):
        """§19.3: stale embedding failure cannot overwrite newer retry state."""
        _seed_issue(store, "cao-1", title="v1 title")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)

        def bump_then_fail():
            _update_issue_title(store, "cao-1", "v2 title")

        racing_failure = MutatingEmbedder(bump_then_fail, fail_predicate=lambda t: True)
        result = vlc.refresh_generation(
            generation_id=created["generation_id"],
            db_path=str(db_file),
            embedder=racing_failure,
        )
        assert result["failed"] == 0 and result["published"] == 0

        dirty = _dirty_rows(store, created["generation_id"])
        assert len(dirty) == 1
        assert dirty[0]["content_version"] >= 2
        assert dirty[0]["attempt_count"] == 0
        assert dirty[0]["last_error"] is None
        assert dirty[0]["next_attempt_at"] is None

    def test_interrupted_generation_never_becomes_active_by_presence_alone(self, store, db_file):
        """§19.3: interrupted builds remain failed/rebuildable, never active."""
        _seed_issue(store, "cao-1")
        _seed_issue(store, "cao-2", title="second")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)

        partial = vlc.refresh_generation(
            generation_id=created["generation_id"],
            limit=1,
            db_path=str(db_file),
            embedder=FakeEmbedder(),
        )
        assert partial["published"] == 1

        with pytest.raises(vlc.VectorLifecycleError) as excinfo:
            vlc.activate_generation(created["generation_id"], target_engine=store)
        assert excinfo.value.reason == "activation-refused-dirty"
        assert _pointer(store) is None
        assert _generation_state(store, created["generation_id"]) == ("building", None)

        assert vlc.mark_generation_failed(
            created["generation_id"], failure="operator stopped the build", target_engine=store
        )
        with pytest.raises(vlc.VectorLifecycleError) as excinfo:
            vlc.activate_generation(created["generation_id"], target_engine=store)
        assert excinfo.value.reason == "activation-refused-state"

        # Recovery is a fresh build, which completes normally.
        rebuilt = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        vlc.refresh_generation(
            generation_id=rebuilt["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        activated = vlc.activate_generation(rebuilt["generation_id"], target_engine=store)
        assert _pointer(store) == rebuilt["generation_id"]
        assert activated["activated_at"] is not None

    def test_encoding_dimension_mismatch_never_activates_a_mixed_generation(self, store, db_file):
        """§19.3: model/dimension mismatch never mixes generations."""
        _seed_issue(store, "cao-1")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )

        raw = _raw(store)
        try:
            raw.execute(
                f"UPDATE {schema.SEARCH_VECTORS_TABLE} SET embedding = ? "
                "WHERE generation_id = ?",
                (blob_for("alien", dimensions=128), created["generation_id"]),
            )
            raw.commit()
        finally:
            raw.close()

        with pytest.raises(vlc.VectorLifecycleError) as excinfo:
            vlc.activate_generation(created["generation_id"], target_engine=store)
        assert excinfo.value.reason == "activation-refused-encoding"
        assert _pointer(store) is None

    def test_coverage_gap_never_activates_a_partial_generation(self, store, db_file):
        _seed_issue(store, "cao-1")
        _seed_issue(store, "cao-2", title="second")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        raw = _raw(store)
        try:
            raw.execute(
                f"DELETE FROM {schema.SEARCH_VECTORS_TABLE} "
                "WHERE generation_id = ? AND document_key = 'issue:cao-2'",
                (created["generation_id"],),
            )
            raw.commit()
        finally:
            raw.close()
        with pytest.raises(vlc.VectorLifecycleError) as excinfo:
            vlc.activate_generation(created["generation_id"], target_engine=store)
        assert excinfo.value.reason == "activation-refused-coverage"
        assert _pointer(store) is None

    def test_default_status_reports_rebuilding_rather_than_mixing_models(self, store, db_file):
        """§19.3: model mismatch never mixes; default reports semantic rebuilding."""
        _seed_issue(store, "cao-1")
        first = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        vlc.refresh_generation(
            generation_id=first["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        vlc.activate_generation(first["generation_id"], target_engine=store)
        assert vlc.semantic_status(target_engine=store)["state"] == "ready"

        challenger_model = dict(MINIMAL_RECORD)
        challenger_model.update({"model_id": "test/challenger", "model_revision": "rev999"})
        second = vlc.create_generation(metadata=challenger_model, target_engine=store)

        status = vlc.semantic_status(target_engine=store)
        assert status["state"] == "rebuilding"
        assert status["active_generation"] == first["generation_id"]
        states = {g["generation_id"]: g["state"] for g in status["generations"]}
        assert states[first["generation_id"]] == "active"
        assert states[second["generation_id"]] == "building"

    def test_lexical_rebuild_requeues_all_live_documents_disqualifying_old_vectors(
        self, store, db_file
    ):
        """§19.3: lexical rebuild queues every active/building generation first."""
        from cli_agent_orchestrator.services.tracker_search import rebuild_lexical

        _seed_issue(store, "cao-1")
        _seed_issue(store, "cao-2", title="second")
        _seed_comment(store, issue_key="cao-1")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        refreshed = vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        assert refreshed["published"] == 3
        assert semantic_rows(store, created["generation_id"]) == [
            "comment:1",
            "issue:cao-1",
            "issue:cao-2",
        ]

        summary = rebuild_lexical(store)

        assert summary["documents_rebuilt"] == 3
        dirty = _dirty_rows(store, created["generation_id"])
        keys = {row["document_key"] for row in dirty}
        issue_keys = {key for key in keys if key.startswith("issue:")}
        comment_keys = {key for key in keys if key.startswith("comment:")}
        assert issue_keys == {"issue:cao-1", "issue:cao-2"}
        assert len(comment_keys) == 1
        # Every pre-rebuild vector is now ineligible BEFORE any refresh runs.
        assert semantic_rows(store, created["generation_id"]) == []

    def test_lexical_search_remains_usable_during_semantic_failure_and_rebuild(
        self, store, db_file
    ):
        """§19.3: lexical search remains usable during semantic failure/rebuild."""
        _seed_issue(store, "cao-1", title="searchable widget")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        failing = FakeEmbedder(fail_predicate=lambda t: True)
        vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=failing
        )
        raw = _raw(store)
        try:
            hits = raw.execute(
                f"SELECT issue_key FROM {schema.ISSUE_FTS_TABLE} "
                "WHERE tracker_issue_fts MATCH ?",
                ("widget",),
            ).fetchall()
            source = raw.execute(
                "SELECT key, title FROM tracker_issues WHERE key = 'cao-1'"
            ).fetchone()
        finally:
            raw.close()
        assert [row[0] for row in hits] == ["cao-1"]
        assert source == ("cao-1", "searchable widget")

    def test_query_time_drain_under_read_semantics_writes_no_authoritative_row_or_event(
        self, store, db_file
    ):
        """§19.3: query-time derived refresh under _READ writes nothing live."""
        _seed_issue(store, "cao-1", body="drain me")
        _seed_comment(store, issue_key="cao-1", body="also drain me")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)

        authoritative_tables = (
            "tracker_issues",
            "tracker_issue_comments",
            "tracker_issue_events",
            "tracker_issue_links",
        )

        def snapshot():
            raw = _raw(store)
            try:
                return {
                    table: raw.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                    for table in authoritative_tables
                }
            finally:
                raw.close()

        before = snapshot()
        drained = vlc.drain_bounded_batch(db_path=str(db_file), embedder=FakeEmbedder())
        after = snapshot()

        assert before == after
        assert drained["published"] == 2
        assert len(_vectors(store, created["generation_id"])) == 2
        assert semantic_rows(store, created["generation_id"]) == ["comment:1", "issue:cao-1"]

    def test_embedding_unavailable_is_typed_and_leaves_authoritative_state_unchanged(
        self, store, db_file, tmp_path
    ):
        """§19.3 primitive: prefiling unavailability never changes filing outcome."""
        _seed_issue(store, "cao-1")
        _seed_comment(store, issue_key="cao-1")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)

        with pytest.raises(EmbeddingCapabilityError) as excinfo:
            vlc.refresh_generation(models_dir=tmp_path / "no-models-here", db_path=str(db_file))
        assert excinfo.value.reason == "unprepared"

        dirty = _dirty_rows(store, created["generation_id"])
        assert len(dirty) == 2
        for row in dirty:
            assert row["attempt_count"] == 0
            assert row["last_error"] is None
        assert _vectors(store, created["generation_id"]) == []
        assert _generation_state(store, created["generation_id"]) == ("building", None)


# ---------------------------------------------------------------------------
# Refresh mechanics beyond the matrix
# ---------------------------------------------------------------------------


class TestRefreshMechanics:
    def test_publish_stores_exact_blob_and_content_hash(self, store, db_file):
        _seed_issue(store, "cao-1", title="hash me", body="hash me")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        expected_text = vlc.issue_document_text({"title": "hash me", "body": "hash me"})
        vectors = _vectors(store, created["generation_id"])
        assert vectors[0]["content_sha256"] == hashlib.sha256(expected_text.encode()).hexdigest()
        assert vectors[0]["blob_bytes"] == DIMENSIONS * 4

    def test_failed_row_backs_off_until_next_attempt_at(self, store, db_file):
        _seed_issue(store, "cao-1", title="retry later")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        failing = FakeEmbedder(fail_predicate=lambda t: True)
        first = vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=failing
        )
        assert first["failed"] == 1
        immediate = vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        assert immediate["attempted"] == 0

        raw = _raw(store)
        try:
            raw.execute(f"UPDATE {schema.VECTOR_DIRTY_TABLE} SET next_attempt_at = NULL")
            raw.commit()
        finally:
            raw.close()
        retried = vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        assert retried["published"] == 1
        assert _dirty_rows(store, created["generation_id"]) == []

    def test_explicit_target_refreshes_only_that_generation(self, store, db_file):
        _seed_issue(store, "cao-1")
        first = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        second = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        vlc.refresh_generation(
            generation_id=first["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        assert _vectors(store, first["generation_id"]) != []
        assert _vectors(store, second["generation_id"]) == []
        assert _dirty_rows(store, second["generation_id"]) != []

    def test_unknown_and_retired_targets_are_typed_refusals(self, store, db_file):
        _seed_issue(store, "cao-1")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        with pytest.raises(vlc.VectorLifecycleError) as excinfo:
            vlc.refresh_generation(generation_id="gen-absent", db_path=str(db_file))
        assert excinfo.value.reason == "unknown-generation"

        raw = _raw(store)
        try:
            raw.execute(
                f"UPDATE {schema.VECTOR_GENERATIONS_TABLE} SET state = 'retired' "
                "WHERE generation_id = ?",
                (created["generation_id"],),
            )
            raw.commit()
        finally:
            raw.close()
        with pytest.raises(vlc.VectorLifecycleError) as excinfo:
            vlc.refresh_generation(generation_id=created["generation_id"], db_path=str(db_file))
        assert excinfo.value.reason == "generation-not-refreshable"


# ---------------------------------------------------------------------------
# Activation and status
# ---------------------------------------------------------------------------


class TestActivationAndStatus:
    def test_activation_switches_pointer_retires_predecessor_clears_retired_dirty(
        self, store, db_file
    ):
        _seed_issue(store, "cao-1")
        first = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        vlc.refresh_generation(
            generation_id=first["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        vlc.activate_generation(first["generation_id"], target_engine=store)
        assert _generation_state(store, first["generation_id"])[0] == "active"

        # A post-activation mutation leaves obsolete dirty work on the old
        # generation; the successor build must clear it at activation.
        _update_issue_title(store, "cao-1", "changed under active")
        assert _dirty_rows(store, first["generation_id"]) != []
        second = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        vlc.refresh_generation(
            generation_id=second["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        vlc.activate_generation(second["generation_id"], target_engine=store)

        assert _pointer(store) == second["generation_id"]
        assert _generation_state(store, first["generation_id"])[0] == "retired"
        assert _generation_state(store, second["generation_id"])[0] == "active"
        assert _dirty_rows(store, first["generation_id"]) == []
        assert _dirty_rows(store, second["generation_id"]) == []
        assert semantic_rows(store, second["generation_id"]) == ["issue:cao-1"]

    def test_activation_prunes_vectors_whose_source_was_deleted(self, store, db_file):
        _seed_issue(store, "cao-1")
        _seed_issue(store, "cao-2", title="vanishing")
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        # Delete bypassing the trigger path is impossible via SQL (triggers are
        # installed); delete through the source so only the vector row lingers
        # — the trigger removes FTS and dirty work but never vectors.
        _delete_issue(store, "cao-2")
        assert len(_vectors(store, created["generation_id"])) == 2

        vlc.activate_generation(created["generation_id"], target_engine=store)
        remaining = {v["document_key"] for v in _vectors(store, created["generation_id"])}
        assert remaining == {"issue:cao-1"}

    def test_mark_generation_failed_moves_only_building_generations(self, store):
        _seed_issue(store, "cao-1")
        building = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        assert vlc.mark_generation_failed(
            building["generation_id"], failure="interrupted", target_engine=store
        )
        assert _generation_state(store, building["generation_id"]) == ("failed", "interrupted")

        # A failed generation cannot be re-marked; unknown ids report absent.
        assert not vlc.mark_generation_failed(
            building["generation_id"], failure="again", target_engine=store
        )
        assert not vlc.mark_generation_failed("gen-absent", failure="x", target_engine=store)

    def test_status_transitions_from_unprepared_through_building_to_ready(self, store, db_file):
        assert vlc.semantic_status(target_engine=store)["state"] == "unprepared"
        created = vlc.create_generation(metadata=MINIMAL_RECORD, target_engine=store)
        building = vlc.semantic_status(target_engine=store)
        assert building["state"] == "building"
        assert building["active_generation"] is None

        _seed_issue(store, "cao-1")
        vlc.refresh_generation(
            generation_id=created["generation_id"], db_path=str(db_file), embedder=FakeEmbedder()
        )
        vlc.activate_generation(created["generation_id"], target_engine=store)
        ready = vlc.semantic_status(target_engine=store)
        assert ready["state"] == "ready"
        assert ready["active_generation"] == created["generation_id"]

        _seed_issue(store, "cao-2", title="pending work")
        stale = vlc.semantic_status(target_engine=store)
        assert stale["state"] == "stale"

    def test_status_reports_unavailable_without_schema(self, tmp_path):
        bare = create_engine(f"sqlite:///{tmp_path}/bare-status.db")
        try:
            report = vlc.semantic_status(target_engine=bare)
            assert report == {"installed": False, "state": "unavailable"}
        finally:
            bare.dispose()
