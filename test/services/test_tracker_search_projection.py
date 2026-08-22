"""The tracker search projection's trigger contract and maintenance verbs.

The contract under test: every direct/bulk SQL write to ``tracker_issues`` and
``tracker_issue_comments`` projects through the seven source triggers — FTS
documents rewritten with exact content versions, one dirty row per active or
building vector generation, nothing for lexical-only installs — while
favorite/link/importance-only changes leave documents untouched; and the
explicit rebuild restores coverage after corruption without ever leaving old
vectors eligible against new text.
"""

import sqlite3
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import (
    _TRACKER_ORM_TABLE_NAMES,
    Base,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.clients.tracker_search_schema import (
    COMMENT_FTS_TABLE,
    ISSUE_FTS_TABLE,
    SEARCH_META_TABLE,
    SEARCH_VECTORS_TABLE,
    TRIGGER_NAMES,
    VECTOR_DIRTY_TABLE,
    VECTOR_GENERATIONS_TABLE,
)
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services.tracker_search import integrity_report, rebuild_lexical


class ProjectionDb:
    """A file-backed tracker store with the search projection installed."""

    def __init__(self, path):
        self.path = str(path)
        self.engine = create_engine(f"sqlite:///{self.path}")
        Base.metadata.create_all(
            bind=self.engine,
            tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
        )
        _migrate_tracker_search_projection(self.engine)

    def raw(self) -> sqlite3.Connection:
        """A plain DBAPI connection proving triggers fire outside app code."""
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def clock(self) -> int:
        with self.engine.begin() as conn:
            return conn.execute(
                text(f"SELECT content_clock FROM {SEARCH_META_TABLE} WHERE singleton = 1")
            ).scalar()

    def fts_rows(self, table):
        with self.engine.begin() as conn:
            return conn.execute(text(f"SELECT * FROM {table}")).fetchall()

    def snapshot_derived(self):
        with self.engine.begin() as conn:
            state = {"clock": self.clock()}
            for name in (
                SEARCH_META_TABLE,
                VECTOR_DIRTY_TABLE,
                VECTOR_GENERATIONS_TABLE,
                SEARCH_VECTORS_TABLE,
                ISSUE_FTS_TABLE,
                COMMENT_FTS_TABLE,
            ):
                state[name] = conn.execute(text(f"SELECT * FROM {name}")).fetchall()
        return state


@pytest.fixture
def proj_db(tmp_path):
    db = ProjectionDb(tmp_path / "projection.db")
    yield db
    db.engine.dispose()


@pytest.fixture
def proj_db_with_generations(proj_db):
    """A lexical install upgraded with one building and one active generation."""
    insert_generation = text(
        f"INSERT INTO {VECTOR_GENERATIONS_TABLE} (generation_id, state, model_id, "
        "model_revision, runtime_id, runtime_version, artifact_sha256, dimensions, "
        "element_type, distance_metric, normalized, document_schema_version, created_at) "
        "VALUES (:gid, :state, 'test-model', 'rev-1', 'test-runtime', '1.0.0', "
        "'deadbeef', 384, 'float32', 'cosine', 1, 1, '2026-08-21T00:00:00Z')"
    )
    with proj_db.engine.begin() as conn:
        conn.execute(insert_generation, {"gid": "gen-building", "state": "building"})
        conn.execute(insert_generation, {"gid": "gen-active", "state": "active"})
        conn.execute(insert_generation, {"gid": "gen-retired", "state": "retired"})
        conn.execute(
            text(f"UPDATE {SEARCH_META_TABLE} SET active_vector_generation = 'gen-active'")
        )
    return proj_db


def _seed_project(db):
    with db.engine.begin() as conn:
        conn.execute(text("INSERT INTO tracker_projects (id, name) VALUES ('p', 'P')"))


def _rowid(db, issue_key):
    with db.engine.begin() as conn:
        return conn.execute(
            text("SELECT id FROM tracker_issues WHERE key = :k"), {"k": issue_key}
        ).scalar()


def _issue_doc_version(db, issue_key):
    with db.engine.begin() as conn:
        return conn.execute(
            text(
                f"SELECT content_version FROM {ISSUE_FTS_TABLE} "
                f"WHERE rowid = {_rowid(db, issue_key)}"
            )
        ).scalar()


# ---------------------------------------------------------------------------
# Direct/bulk SQL writes project through the triggers, outside any app path
# ---------------------------------------------------------------------------


class TestDirectSqlWritesProjectThroughTriggers:
    def test_an_issue_insert_creates_its_document_at_the_current_clock_tick(self, proj_db):
        conn = proj_db.raw()
        try:
            before = proj_db.clock()
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title, body, labels, status) "
                "VALUES ('p-1', 'p', 'Login fails on SSO', 'the body', "
                "'[\"auth\", \"sso\"]', 'open')"
            )
            conn.commit()
            assert proj_db.clock() == before + 1
        finally:
            conn.close()
        with proj_db.engine.begin() as sql:
            doc = sql.execute(
                text(
                    f"SELECT rowid, issue_key, content_version, title, labels_text, body "
                    f"FROM {ISSUE_FTS_TABLE}"
                )
            ).fetchall()
        assert doc == [(1, "p-1", before + 1, "Login fails on SSO", "auth sso", "the body")]

    def test_a_comment_insert_carries_issue_title_and_component_context(self, proj_db):
        _seed_project(proj_db)
        conn = proj_db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title, component) "
                "VALUES ('p-1', 'p', 'Crash on save', 'io')"
            )
            conn.execute(
                "INSERT INTO tracker_issue_comments (issue_key, author, body) "
                "VALUES ('p-1', 'w1', 'stack trace attached')"
            )
            conn.commit()
        finally:
            conn.close()
        with proj_db.engine.begin() as sql:
            doc = sql.execute(
                text(
                    f"SELECT comment_id, issue_key, issue_title, component, author, body "
                    f"FROM {COMMENT_FTS_TABLE}"
                )
            ).fetchall()
        assert doc == [(1, "p-1", "Crash on save", "io", "w1", "stack trace attached")]

    def test_an_indexed_update_rewrites_the_document_with_a_fresh_version(self, proj_db):
        conn = proj_db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title, status) "
                "VALUES ('p-1', 'p', 't', 'open')"
            )
            conn.commit()
            version_before = conn.execute(
                f"SELECT content_version FROM {ISSUE_FTS_TABLE} WHERE rowid = 1"
            ).fetchone()[0]
            conn.execute("UPDATE tracker_issues SET status = 'in-progress' WHERE key = 'p-1'")
            status_now, version_now = conn.execute(
                f"SELECT status, content_version FROM {ISSUE_FTS_TABLE} WHERE rowid = 1"
            ).fetchone()
            assert (status_now, version_now) == ("in-progress", version_before + 1)
        finally:
            conn.close()

    def test_a_delete_removes_documents_and_dirty_work(self, proj_db_with_generations):
        db = proj_db_with_generations
        conn = db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-1', 'p', 't')"
            )
            conn.execute("INSERT INTO tracker_issue_comments (issue_key, body) VALUES ('p-1', 'c')")
            conn.commit()
            assert len(db.fts_rows(ISSUE_FTS_TABLE)) == 1
            assert len(db.fts_rows(COMMENT_FTS_TABLE)) == 1
            assert len(db.fts_rows(VECTOR_DIRTY_TABLE)) > 0
            conn.execute("DELETE FROM tracker_issue_comments WHERE id = 1")
            conn.execute("DELETE FROM tracker_issues WHERE key = 'p-1'")
            conn.commit()
            assert db.fts_rows(ISSUE_FTS_TABLE) == []
            assert db.fts_rows(COMMENT_FTS_TABLE) == []
            assert db.fts_rows(VECTOR_DIRTY_TABLE) == []
        finally:
            conn.close()

    def test_a_rolled_back_write_leaves_zero_projection_side_effects(
        self, proj_db_with_generations
    ):
        db = proj_db_with_generations
        before = db.snapshot_derived()
        conn = db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-x', 'p', 'x')"
            )
            conn.execute("INSERT INTO tracker_issue_comments (issue_key, body) VALUES ('p-x', 'c')")
            conn.execute("UPDATE tracker_issues SET status = 'closed' WHERE key = 'p-x'")
            conn.rollback()
        finally:
            conn.close()
        assert db.snapshot_derived() == before


# ---------------------------------------------------------------------------
# Which mutations reindex, and which must not
# ---------------------------------------------------------------------------


class TestReindexBoundaries:
    @pytest.fixture
    def service_db(self, proj_db, monkeypatch):
        monkeypatch.setattr(tracker, "SessionLocal", sessionmaker(bind=proj_db.engine))
        tracker.create_project(name="P", project_id="p", issue_prefix="p")
        return proj_db

    def test_claiming_reindexes_status_and_bumps_the_clock(self, service_db):
        db = service_db
        issue = tracker.create_issue(project_id="p", title="claim me")
        clock_before = db.clock()
        doc_version_before = _issue_doc_version(db, issue["key"])
        tracker.claim_issue(issue["key"], claimant="worker-a")
        assert db.clock() == clock_before + 1
        with db.engine.begin() as conn:
            status, version = conn.execute(
                text(
                    f"SELECT status, content_version FROM {ISSUE_FTS_TABLE} "
                    f"WHERE rowid = {_rowid(db, issue['key'])}"
                )
            ).fetchone()
        assert status == "in-progress"
        assert version > doc_version_before

    def test_a_favorite_only_change_neither_reindexes_nor_dirties(self, service_db, monkeypatch):
        db = service_db
        issue = tracker.create_issue(project_id="p", title="fav me")
        before = db.snapshot_derived()
        tracker.update_issue(issue["key"], favorite=True, actor="op")
        after = db.snapshot_derived()
        assert after["clock"] == before["clock"]
        assert after[ISSUE_FTS_TABLE] == before[ISSUE_FTS_TABLE]

    def test_link_changes_neither_reindex_nor_dirty(self, service_db):
        db = service_db
        first = tracker.create_issue(project_id="p", title="one")
        second = tracker.create_issue(project_id="p", title="two")
        before = db.snapshot_derived()
        link_id = tracker.add_link(first["key"], to_key=second["key"], kind="blocks")["id"]
        tracker.remove_link(link_id, actor="op")
        after = db.snapshot_derived()
        assert after["clock"] == before["clock"]
        assert after[ISSUE_FTS_TABLE] == before[ISSUE_FTS_TABLE]
        assert after[COMMENT_FTS_TABLE] == before[COMMENT_FTS_TABLE]

    def test_an_importance_only_touch_advances_the_clock_but_not_the_document(
        self, service_db, monkeypatch
    ):
        db = service_db
        issue = tracker.create_issue(project_id="p", title="important things")
        comment_id = tracker.add_comment(issue["key"], body="the signal", author="w")["id"]

        from cli_agent_orchestrator.clients.tracker_search_schema import (
            VECTOR_GENERATIONS_TABLE,
        )

        insert_generation = text(
            f"INSERT INTO {VECTOR_GENERATIONS_TABLE} (generation_id, state, model_id, "
            "model_revision, runtime_id, runtime_version, artifact_sha256, dimensions, "
            "element_type, distance_metric, normalized, document_schema_version, created_at) "
            "VALUES ('gen-active', 'active', 'm', 'r', 'st', '1', 'aa', 384, 'float32', "
            "'cosine', 1, 1, '2026-08-21T00:00:00Z')"
        )
        with db.engine.begin() as conn:
            conn.execute(insert_generation)

        clock_before = db.clock()
        with db.engine.begin() as conn:
            version_before = conn.execute(
                text(f"SELECT content_version FROM {COMMENT_FTS_TABLE} WHERE rowid = {comment_id}")
            ).scalar()
        tracker.set_comment_importance(issue["key"], comment_id, important=True, actor="op")
        assert db.clock() == clock_before + 1
        with db.engine.begin() as conn:
            version_after = conn.execute(
                text(f"SELECT content_version FROM {COMMENT_FTS_TABLE} WHERE rowid = {comment_id}")
            ).scalar()
            dirty = conn.execute(text(f"SELECT COUNT(*) FROM {VECTOR_DIRTY_TABLE}")).scalar()
        # Importance is live ranking data: no rewrite, no embedding work.
        assert version_after == version_before
        assert dirty == 0

    def test_a_title_change_fans_out_to_comment_documents(self, proj_db_with_generations):
        db = proj_db_with_generations
        conn = db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-1', 'p', 'old')"
            )
            conn.execute(
                "INSERT INTO tracker_issue_comments (issue_key, body) VALUES ('p-1', 'note')"
            )
            conn.commit()
            stale_version = conn.execute(
                f"SELECT content_version FROM {COMMENT_FTS_TABLE} WHERE rowid = 1"
            ).fetchone()[0]
            conn.execute("UPDATE tracker_issues SET title = 'new' WHERE key = 'p-1'")
            title, version = conn.execute(
                f"SELECT issue_title, content_version FROM {COMMENT_FTS_TABLE} WHERE rowid = 1"
            ).fetchone()
            assert title == "new"
            assert version > stale_version
        finally:
            conn.close()

    def test_a_status_change_does_not_churn_comment_documents(self, proj_db_with_generations):
        db = proj_db_with_generations
        conn = db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-1', 'p', 't')"
            )
            conn.execute(
                "INSERT INTO tracker_issue_comments (issue_key, body) VALUES ('p-1', 'note')"
            )
            conn.commit()
            before = conn.execute(
                f"SELECT content_version FROM {COMMENT_FTS_TABLE} WHERE rowid = 1"
            ).fetchone()[0]
            conn.execute("UPDATE tracker_issues SET status = 'closed' WHERE key = 'p-1'")
            assert (
                conn.execute(
                    f"SELECT content_version FROM {COMMENT_FTS_TABLE} WHERE rowid = 1"
                ).fetchone()[0]
                == before
            )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# The durable vector outbox
# ---------------------------------------------------------------------------


class TestDirtyOutbox:
    def test_a_lexical_only_install_accumulates_no_dirty_rows(self, proj_db):
        conn = proj_db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-1', 'p', 't')"
            )
            conn.execute("INSERT INTO tracker_issue_comments (issue_key, body) VALUES ('p-1', 'c')")
            conn.execute("UPDATE tracker_issues SET title = 't2' WHERE key = 'p-1'")
            conn.commit()
        finally:
            conn.close()
        assert proj_db.fts_rows(VECTOR_DIRTY_TABLE) == []

    def test_each_write_enqueues_one_row_per_active_and_building_generation(
        self, proj_db_with_generations
    ):
        db = proj_db_with_generations
        conn = db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-1', 'p', 't')"
            )
            conn.execute("INSERT INTO tracker_issue_comments (issue_key, body) VALUES ('p-1', 'c')")
            conn.commit()
        finally:
            conn.close()
        with db.engine.begin() as sql:
            rows = sql.execute(
                text(
                    f"SELECT generation_id, document_key, document_kind FROM {VECTOR_DIRTY_TABLE} "
                    "ORDER BY generation_id, document_key"
                )
            ).fetchall()
        # The retired generation never receives work.
        assert rows == [
            ("gen-active", "comment:1", "comment"),
            ("gen-active", "issue:p-1", "issue"),
            ("gen-building", "comment:1", "comment"),
            ("gen-building", "issue:p-1", "issue"),
        ]

    def test_reenqueueing_resets_backoff_and_error_state(self, proj_db_with_generations):
        db = proj_db_with_generations
        conn = db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-1', 'p', 't')"
            )
            conn.commit()
            conn.execute(
                f"UPDATE {VECTOR_DIRTY_TABLE} SET attempt_count = 5, "
                "next_attempt_at = '2099-01-01T00:00:00Z', last_error = 'boom' "
                "WHERE document_key = 'issue:p-1'"
            )
            conn.execute("UPDATE tracker_issues SET title = 't2' WHERE key = 'p-1'")
            states = conn.execute(
                f"SELECT attempt_count, next_attempt_at, last_error FROM {VECTOR_DIRTY_TABLE} "
                "WHERE document_key = 'issue:p-1'"
            ).fetchall()
            assert states == [(0, None, None)] * len(states)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Forced deletion through the service layer
# ---------------------------------------------------------------------------


class TestForcedDeletionCleansProjections:
    def test_delete_issue_removes_every_derived_row(self, proj_db_with_generations, monkeypatch):
        db = proj_db_with_generations
        monkeypatch.setattr(tracker, "SessionLocal", sessionmaker(bind=db.engine))
        tracker.create_project(name="P", project_id="p", issue_prefix="p")
        issue = tracker.create_issue(project_id="p", title="doomed")
        tracker.add_comment(issue["key"], body="also doomed", author="w")
        tracker.delete_issue(issue["key"])
        assert db.fts_rows(ISSUE_FTS_TABLE) == []
        assert db.fts_rows(COMMENT_FTS_TABLE) == []
        assert db.fts_rows(VECTOR_DIRTY_TABLE) == []

    def test_forced_project_deletion_removes_every_derived_row(
        self, proj_db_with_generations, monkeypatch
    ):
        db = proj_db_with_generations
        monkeypatch.setattr(tracker, "SessionLocal", sessionmaker(bind=db.engine))
        tracker.create_project(name="P", project_id="p", issue_prefix="p")
        first = tracker.create_issue(project_id="p", title="one")
        second = tracker.create_issue(project_id="p", title="two")
        tracker.add_comment(first["key"], body="c1", author="w")
        tracker.add_comment(second["key"], body="c2", author="w")
        tracker.delete_project("p", force=True)
        assert db.fts_rows(ISSUE_FTS_TABLE) == []
        assert db.fts_rows(COMMENT_FTS_TABLE) == []
        assert db.fts_rows(VECTOR_DIRTY_TABLE) == []


# ---------------------------------------------------------------------------
# Rebuild (§13.2)
# ---------------------------------------------------------------------------


class TestRebuild:
    @pytest.fixture
    def corrupted_db(self, proj_db_with_generations, monkeypatch):
        """A store damaged in every way a rebuild must recover from."""
        db = proj_db_with_generations
        monkeypatch.setattr(tracker, "SessionLocal", sessionmaker(bind=db.engine))
        tracker.create_project(name="P", project_id="p", issue_prefix="p")
        kept = tracker.create_issue(project_id="p", title="kept issue", body="kept body")
        doomed = tracker.create_issue(project_id="p", title="doomed issue", body="doomed body")
        tracker.add_comment(kept["key"], body="kept comment", author="w")
        with db.engine.begin() as conn:
            # A pre-rebuild vector carrying the CURRENT document version: it
            # is eligible exactly until the rebuild assigns fresh versions.
            current_version = conn.execute(
                text(
                    f"SELECT content_version FROM {ISSUE_FTS_TABLE} "
                    f"WHERE rowid = {_rowid(db, kept['key'])}"
                )
            ).scalar()
            conn.execute(
                text(
                    f"INSERT INTO {SEARCH_VECTORS_TABLE} (generation_id, document_key, issue_key, "
                    "document_kind, source_id, content_version, content_sha256, embedding, "
                    "indexed_at) VALUES ('gen-active', :dkey, :ikey, 'issue', :sid, :ver, 'h', "
                    "x'00000000000000000000000000000000', '2026-08-21T00:00:00Z')"
                ),
                {
                    "dkey": f"issue:{kept['key']}",
                    "ikey": kept["key"],
                    "sid": _rowid(db, kept["key"]),
                    "ver": current_version,
                },
            )
            # A missing document, an orphan document, and tampered versions.
            conn.execute(
                text(f"DELETE FROM {ISSUE_FTS_TABLE} WHERE rowid = {_rowid(db, kept['key'])}")
            )
            conn.execute(
                text(
                    f"INSERT INTO {ISSUE_FTS_TABLE} (rowid, issue_key, content_version, title) "
                    "VALUES (99999, 'ghost-key', 7, 'ghost')"
                )
            )
            conn.execute(
                text(
                    f"UPDATE {COMMENT_FTS_TABLE} SET content_version = 3 WHERE issue_key = "
                    f"'{kept['key']}'"
                )
            )
            # Dirty work whose source has since disappeared.
            conn.execute(
                text(
                    f"INSERT INTO {VECTOR_DIRTY_TABLE} (generation_id, document_key, issue_key, "
                    "document_kind, source_id, content_version, document_schema_version, "
                    "enqueued_at) VALUES ('gen-active', 'issue:ghost', 'ghost', 'issue', "
                    f"{_rowid(db, doomed['key'])}, 4, 1, '2026-08-21T00:00:00Z')"
                )
            )
            conn.execute(text(f"DELETE FROM tracker_issues WHERE key = '{doomed['key']}'"))
        return db

    def test_rebuild_restores_exact_coverage_and_repairs_versions(self, corrupted_db):
        db = corrupted_db
        summary = rebuild_lexical(db.engine)
        with db.engine.begin() as conn:
            missing_issues = conn.execute(
                text(
                    "SELECT COUNT(*) FROM tracker_issues AS s WHERE NOT EXISTS "
                    f"(SELECT 1 FROM {ISSUE_FTS_TABLE} AS f WHERE f.rowid = s.id)"
                )
            ).scalar()
            orphan_docs = conn.execute(
                text(
                    f"SELECT COUNT(*) FROM {ISSUE_FTS_TABLE} AS f WHERE NOT EXISTS "
                    "(SELECT 1 FROM tracker_issues AS s WHERE s.id = f.rowid)"
                )
            ).scalar()
            unversioned = conn.execute(
                text(f"SELECT COUNT(*) FROM {ISSUE_FTS_TABLE} " "WHERE content_version IS NULL")
            ).scalar()
        assert missing_issues == 0
        assert orphan_docs == 0
        assert unversioned == 0
        assert summary["issues"] == 1
        assert summary["comments"] == 1

    def test_rebuild_queues_every_live_document_for_active_and_building_generations(
        self, corrupted_db
    ):
        db = corrupted_db
        rebuild_lexical(db.engine)
        with db.engine.begin() as conn:
            queued = {
                (row[0], row[1])
                for row in conn.execute(
                    text(
                        f"SELECT generation_id, document_key FROM {VECTOR_DIRTY_TABLE} "
                        "WHERE document_kind = 'issue'"
                    )
                ).fetchall()
            }
            live_keys = [row[0] for row in conn.execute(text("SELECT key FROM tracker_issues"))]
        expected = {
            (generation, f"issue:{key}")
            for generation in ("gen-active", "gen-building")
            for key in live_keys
        }
        assert queued == expected

    def test_pre_rebuild_vectors_are_ineligible_afterwards(self, corrupted_db):
        """A vector matching pre-rebuild text cannot satisfy freshness after it."""
        db = corrupted_db
        with db.engine.begin() as conn:
            stored_version = conn.execute(
                text(f"SELECT content_version FROM {SEARCH_VECTORS_TABLE}")
            ).scalar()
        rebuild_lexical(db.engine)
        with db.engine.begin() as conn:
            fresh_versions = [
                row[0]
                for row in conn.execute(
                    text(
                        f"SELECT f.content_version FROM {ISSUE_FTS_TABLE} AS f "
                        "JOIN tracker_issues AS s ON s.id = f.rowid"
                    )
                ).fetchall()
            ]
            dirty_for_generation = conn.execute(
                text(
                    f"SELECT COUNT(*) FROM {VECTOR_DIRTY_TABLE} "
                    "WHERE generation_id = 'gen-active'"
                )
            ).scalar()
        # Fresh versions strictly exceed every pre-rebuild version, and the
        # generation holds dirty work for every live document.
        assert all(version > stored_version for version in fresh_versions)
        assert dirty_for_generation >= len(fresh_versions)
        report = integrity_report(db.engine)
        assert report["vector_stale"]["total_vectors"] == 1
        assert report["vector_stale"]["stale_vectors"] == 1

    def test_stale_dirty_rows_for_deleted_sources_are_pruned(self, corrupted_db):
        db = corrupted_db
        rebuild_lexical(db.engine)
        with db.engine.begin() as conn:
            ghost = conn.execute(
                text(f"SELECT COUNT(*) FROM {VECTOR_DIRTY_TABLE} WHERE issue_key = 'ghost'")
            ).scalar()
        assert ghost == 0

    def test_rebuild_records_itself_and_leaves_triggers_installed(self, corrupted_db):
        db = corrupted_db
        rebuilt_at = datetime.now(timezone.utc).isoformat()
        result = rebuild_lexical(db.engine)
        assert result["rebuilt_at"] >= rebuilt_at[:19]
        with db.engine.begin() as conn:
            meta_rebuilt_at = conn.execute(
                text(f"SELECT rebuilt_at FROM {SEARCH_META_TABLE}")
            ).scalar()
            triggers = conn.execute(
                text(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name IN "
                    f"({', '.join(repr(name) for name in TRIGGER_NAMES)})"
                )
            ).scalar()
        assert meta_rebuilt_at == result["rebuilt_at"]
        assert triggers == len(TRIGGER_NAMES)

    def test_rebuild_is_safe_on_a_freshly_installed_store(self, proj_db):
        summary = rebuild_lexical(proj_db.engine)
        assert summary == {
            "rebuilt_at": summary["rebuilt_at"],
            "documents_rebuilt": 0,
            "issues": 0,
            "comments": 0,
        }


# ---------------------------------------------------------------------------
# Read-only integrity report (§13.4)
# ---------------------------------------------------------------------------


class TestIntegrityReport:
    def test_an_uninstalled_store_reports_installed_false(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path}/empty.db")
        try:
            assert integrity_report(engine) == {"installed": False}
        finally:
            engine.dispose()

    def test_the_report_covers_every_section_without_mutating_anything(
        self, proj_db_with_generations
    ):
        db = proj_db_with_generations
        before = db.snapshot_derived()
        report = integrity_report(db.engine)
        assert db.snapshot_derived() == before
        assert report["installed"] is True
        for section in (
            "fts_internal",
            "coverage",
            "duplicate_orphan_document_keys",
            "vector_dirty",
            "vector_stale",
            "generations",
            "coverage_by_project",
            "last_failures",
        ):
            assert section in report, f"missing §13.4 field: {section}"
        assert report["fts_internal"] == {"issues": "ok", "comments": "ok"}
        assert report["coverage"]["issue"]["source_rows"] == 0
        assert report["generations"][0]["model_id"] == "test-model"

    def test_the_report_detects_defects_it_exists_to_expose(self, proj_db_with_generations):
        db = proj_db_with_generations
        conn = db.raw()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-1', 'p', 't')"
            )
            conn.execute("INSERT INTO tracker_issue_comments (issue_key, body) VALUES ('p-1', 'c')")
            conn.commit()
            conn.execute(f"DELETE FROM {COMMENT_FTS_TABLE}")
            conn.execute(
                f"INSERT INTO {SEARCH_VECTORS_TABLE} (generation_id, document_key, issue_key, "
                "document_kind, source_id, content_version, content_sha256, embedding, "
                "indexed_at) VALUES ('gen-active', 'issue:p-1', 'p-1', 'issue', 1, 999, 'h', "
                "x'00000000', '2026-08-21T00:00:00Z')"
            )
        finally:
            conn.close()
        report = integrity_report(db.engine)
        assert report["coverage"]["comment"]["missing_documents"] == 1
        assert report["vector_stale"]["stale_vectors"] == 1


# ---------------------------------------------------------------------------
# Service-layer and direct-SQL writers converge on the same projections
# ---------------------------------------------------------------------------


class TestDirectAndServiceWritersConverge:
    def test_identical_sources_produce_identical_documents(self, proj_db, monkeypatch):
        db = proj_db
        monkeypatch.setattr(tracker, "SessionLocal", sessionmaker(bind=db.engine))
        tracker.create_project(name="P", project_id="p", issue_prefix="p")
        service_issue = tracker.create_issue(
            project_id="p",
            title="Shared title",
            body="Shared body",
            labels=["alpha", "beta"],
        )
        tracker.add_comment(service_issue["key"], body="shared note", author="worker")

        conn = db.raw()
        try:
            conn.execute("BEGIN")
            direct_issue_id = 10_000
            conn.execute(
                "INSERT INTO tracker_issues (id, key, project_id, title, body, labels) "
                f"VALUES ({direct_issue_id}, 'p-direct', 'p', 'Shared title', 'Shared body', "
                '\'["alpha", "beta"]\')'
            )
            conn.execute(
                "INSERT INTO tracker_issue_comments (id, issue_key, author, body) "
                f"VALUES ({direct_issue_id}, 'p-direct', 'worker', 'shared note')"
            )
            conn.commit()
        finally:
            conn.close()

        with db.engine.begin() as sql:
            columns = "title, body, labels_text"
            service_doc = sql.execute(
                text(
                    f"SELECT {columns} FROM {ISSUE_FTS_TABLE} "
                    f"WHERE rowid = {_rowid(db, service_issue['key'])}"
                )
            ).fetchone()
            direct_doc = sql.execute(
                text(f"SELECT {columns} FROM {ISSUE_FTS_TABLE} WHERE rowid = {direct_issue_id}")
            ).fetchone()
            service_comment = sql.execute(
                text(
                    f"SELECT f.author, f.body FROM {COMMENT_FTS_TABLE} AS f "
                    "JOIN tracker_issue_comments AS c ON c.id = f.rowid "
                    "WHERE c.issue_key = :k",
                ),
                {"k": service_issue["key"]},
            ).fetchone()
            direct_comment = sql.execute(
                text(
                    f"SELECT author, body FROM {COMMENT_FTS_TABLE} WHERE rowid = {direct_issue_id}"
                )
            ).fetchone()
        assert service_doc == direct_doc
        assert service_comment == direct_comment
