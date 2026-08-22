"""The tracker search projection migration (cond-0637).

The contract under test: ``ensure_tracker_schema`` (CLI entry point) and
``init_db`` (API entry point) both run ONE injectable, idempotent raw
migration that installs the derived search surface — metadata singleton,
vector outbox/generation/vector tables, both FTS5 documents, seven source
triggers, backfill with exact content versions, and a coverage proof — inside
one ``BEGIN IMMEDIATE``, refusing an incompatible prior shape with a typed
error instead of trusting ``IF NOT EXISTS``.
"""

import sqlite3

import pytest
from sqlalchemy import create_engine, text

from cli_agent_orchestrator.clients.database import (
    _TRACKER_ORM_TABLE_NAMES,
    Base,
    TrackerSchemaMigrationError,
    _migrate_tracker_search_projection,
    ensure_tracker_schema,
    init_db,
)
from cli_agent_orchestrator.clients.tracker_search_schema import (
    COMMENT_FTS_TABLE,
    ISSUE_FTS_TABLE,
    SEARCH_META_TABLE,
    SEARCH_VECTORS_TABLE,
    TRIGGER_NAMES,
    VECTOR_DIRTY_TABLE,
    VECTOR_GENERATIONS_TABLE,
    TrackerSearchSchemaError,
)

DERIVED_OBJECTS = (
    SEARCH_META_TABLE,
    VECTOR_DIRTY_TABLE,
    VECTOR_GENERATIONS_TABLE,
    SEARCH_VECTORS_TABLE,
    ISSUE_FTS_TABLE,
    COMMENT_FTS_TABLE,
) + TRIGGER_NAMES


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/search-migration.db")
    yield eng
    eng.dispose()


def _create_tracker_tables(eng):
    Base.metadata.create_all(
        bind=eng,
        tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
    )


def _derived_shape(eng):
    """Every derived object's normalized CREATE text, for parity checks."""
    with eng.begin() as conn:
        rows = conn.execute(
            text("SELECT name, type, sql FROM sqlite_master ORDER BY name")
        ).fetchall()
    return {
        name: (kind, " ".join(str(sql or "").split()).lower())
        for name, kind, sql in rows
        if name in DERIVED_OBJECTS
    }


def _clock(eng):
    with eng.begin() as conn:
        return conn.execute(
            text(f"SELECT content_clock FROM {SEARCH_META_TABLE} WHERE singleton = 1")
        ).scalar()


class TestBothEntryPointsInstallIdenticalSchema:
    def test_ensure_tracker_schema_produces_the_derived_schema(self, tmp_path, monkeypatch):
        from cli_agent_orchestrator.clients import database as db_module

        engine = create_engine(f"sqlite:///{tmp_path}/cli.db")
        try:
            monkeypatch.setattr(db_module, "engine", engine)
            ensure_tracker_schema()
            shape = _derived_shape(engine)
            for name in DERIVED_OBJECTS:
                assert name in shape, f"{name} was not installed by the CLI entry point"
        finally:
            engine.dispose()

    def test_init_db_and_ensure_tracker_schema_agree_on_the_derived_schema(
        self, tmp_path, monkeypatch
    ):
        from cli_agent_orchestrator.clients import database as db_module

        cli_engine = create_engine(f"sqlite:///{tmp_path}/cli.db")
        api_engine = create_engine(f"sqlite:///{tmp_path}/api.db")
        try:
            monkeypatch.setattr(db_module, "engine", cli_engine)
            ensure_tracker_schema()
            monkeypatch.setattr(db_module, "engine", api_engine)
            init_db()
            assert _derived_shape(api_engine) == _derived_shape(cli_engine)
        finally:
            cli_engine.dispose()
            api_engine.dispose()

    def test_repeated_migration_is_idempotent(self, engine):
        _create_tracker_tables(engine)
        _migrate_tracker_search_projection(engine)
        first = _derived_shape(engine)
        _migrate_tracker_search_projection(engine)
        _migrate_tracker_search_projection(engine)
        assert _derived_shape(engine) == first

    def test_repeated_migration_preserves_state_and_never_reseeds(self, engine):
        """A second run must not rewind the clock or duplicate the singleton."""
        _create_tracker_tables(engine)
        _migrate_tracker_search_projection(engine)
        with engine.begin() as conn:
            conn.execute(text("UPDATE tracker_search_meta SET content_clock = 41"))
        _migrate_tracker_search_projection(engine)
        assert _clock(engine) == 41
        with engine.begin() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {SEARCH_META_TABLE}")).scalar()
        assert count == 1


class TestBackfill:
    def test_backfill_assigns_exact_ascending_versions_and_advances_the_clock(self, engine):
        _create_tracker_tables(engine)
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO tracker_projects (id, name) VALUES ('p', 'P')"))
            for index in range(1, 4):
                conn.execute(
                    text(
                        "INSERT INTO tracker_issues (key, project_id, title, labels) "
                        f"VALUES ('p-{index}', 'p', 't{index}', '[\"a\"]')"
                    )
                )
            conn.execute(
                text(
                    "INSERT INTO tracker_issue_comments (issue_key, body) "
                    "VALUES ('p-1', 'note one'), ('p-2', 'note two')"
                )
            )
        _migrate_tracker_search_projection(engine)
        with engine.begin() as conn:
            issue_versions = [
                row[0]
                for row in conn.execute(
                    text(
                        f"SELECT content_version FROM {ISSUE_FTS_TABLE} " "ORDER BY content_version"
                    )
                )
            ]
            comment_versions = [
                row[0]
                for row in conn.execute(
                    text(
                        f"SELECT content_version FROM {COMMENT_FTS_TABLE} "
                        "ORDER BY content_version"
                    )
                )
            ]
        assert issue_versions == [1, 2, 3]
        assert comment_versions == [4, 5]
        assert _clock(engine) == 5

    def test_backfill_decodes_json_arrays_without_serialization_artifacts(self, engine):
        _create_tracker_tables(engine)
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO tracker_projects (id, name) VALUES ('p', 'P')"))
            conn.execute(
                text(
                    "INSERT INTO tracker_issues (key, project_id, title, labels, branches) "
                    "VALUES ('p-1', 'p', 't', '[\"alpha\", \"beta gamma\", \"\"]', "
                    "'[\"refs/heads/main\"]')"
                )
            )
        _migrate_tracker_search_projection(engine)
        with engine.begin() as conn:
            labels_text = conn.execute(
                text(f"SELECT labels_text FROM {ISSUE_FTS_TABLE} WHERE rowid = 1")
            ).scalar()
            branches_text = conn.execute(
                text(f"SELECT branches_text FROM {ISSUE_FTS_TABLE} WHERE rowid = 1")
            ).scalar()
        assert labels_text == "alpha beta gamma"
        assert branches_text == "refs/heads/main"

    def test_a_gap_left_while_triggers_were_absent_is_healed(self, engine):
        _create_tracker_tables(engine)
        _migrate_tracker_search_projection(engine)
        # Simulate a writer that dropped the triggers and then wrote rows.
        with engine.begin() as conn:
            for name in TRIGGER_NAMES:
                conn.execute(text(f"DROP TRIGGER IF EXISTS {name}"))
            conn.execute(
                text(
                    "INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-9', 'p', 'late')"
                )
            )
        before = _clock(engine)
        _migrate_tracker_search_projection(engine)
        with engine.begin() as conn:
            missing = conn.execute(
                text(
                    "SELECT COUNT(*) FROM tracker_issues AS s WHERE NOT EXISTS "
                    f"(SELECT 1 FROM {ISSUE_FTS_TABLE} AS f WHERE f.rowid = s.id)"
                )
            ).scalar()
            versioned = conn.execute(
                text(f"SELECT COUNT(*) FROM {ISSUE_FTS_TABLE} " "WHERE content_version IS NOT NULL")
            ).scalar()
        assert missing == 0
        assert versioned == 1
        assert _clock(engine) == before + 1

    def test_an_already_covered_store_backfills_nothing(self, engine):
        _create_tracker_tables(engine)
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO tracker_projects (id, name) VALUES ('p', 'P')"))
            conn.execute(
                text("INSERT INTO tracker_issues (key, project_id, title) VALUES ('p-1', 'p', 't')")
            )
        _migrate_tracker_search_projection(engine)
        before = (_clock(engine), _derived_shape(engine))
        _migrate_tracker_search_projection(engine)
        assert (_clock(engine), _derived_shape(engine)) == before


class TestIncompatibleShapes:
    def test_a_foreign_meta_table_is_refused(self, engine):
        _create_tracker_tables(engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE TABLE {SEARCH_META_TABLE} ("
                    "singleton INTEGER PRIMARY KEY, schema_version TEXT)"
                )
            )
        with pytest.raises(TrackerSearchSchemaError) as exc:
            _migrate_tracker_search_projection(engine)
        assert exc.value.table == SEARCH_META_TABLE

    def test_a_foreign_fts_definition_is_refused(self, engine):
        _create_tracker_tables(engine)
        with engine.begin() as conn:
            conn.execute(text(f"CREATE VIRTUAL TABLE {ISSUE_FTS_TABLE} USING fts5(title)"))
        with pytest.raises(TrackerSearchSchemaError):
            _migrate_tracker_search_projection(engine)

    def test_a_foreign_vector_table_is_refused(self, engine):
        _create_tracker_tables(engine)
        with engine.begin() as conn:
            conn.execute(
                text(f"CREATE TABLE {VECTOR_DIRTY_TABLE} (generation_id TEXT PRIMARY KEY)")
            )
        with pytest.raises(TrackerSearchSchemaError):
            _migrate_tracker_search_projection(engine)

    def test_a_non_integer_clock_in_the_singleton_is_refused(self, engine):
        _create_tracker_tables(engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE TABLE {SEARCH_META_TABLE} ("
                    "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                    "schema_version INTEGER NOT NULL, "
                    "document_schema_version INTEGER NOT NULL, "
                    "content_clock INTEGER NOT NULL DEFAULT 0, "
                    "active_vector_generation TEXT, rebuilt_at TEXT)"
                )
            )
            conn.execute(
                text(
                    f"INSERT INTO {SEARCH_META_TABLE} (singleton, schema_version, "
                    "document_schema_version, content_clock) VALUES (1, 1, 1, 'many')"
                )
            )
        with pytest.raises(TrackerSearchSchemaError) as exc:
            _migrate_tracker_search_projection(engine)
        assert "content_clock" in str(exc.value)

    def test_a_dangling_active_generation_pointer_is_refused(self, engine):
        _create_tracker_tables(engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE TABLE {SEARCH_META_TABLE} ("
                    "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                    "schema_version INTEGER NOT NULL, "
                    "document_schema_version INTEGER NOT NULL, "
                    "content_clock INTEGER NOT NULL DEFAULT 0, "
                    "active_vector_generation TEXT, rebuilt_at TEXT)"
                )
            )
            conn.execute(
                text(
                    f"INSERT INTO {SEARCH_META_TABLE} (singleton, schema_version, "
                    "document_schema_version, content_clock, active_vector_generation) "
                    "VALUES (1, 1, 1, 0, 'gen-vanished')"
                )
            )
        with pytest.raises(TrackerSearchSchemaError) as exc:
            _migrate_tracker_search_projection(engine)
        assert "gen-vanished" in str(exc.value)

    def test_a_half_migrated_source_table_is_refused(self, engine):
        """Without the observed-columns migration there is no projection."""
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tracker_issues ("
                    "id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, "
                    "project_id TEXT NOT NULL, title TEXT NOT NULL)"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE tracker_issue_comments ("
                    "id INTEGER PRIMARY KEY, issue_key TEXT NOT NULL, body TEXT NOT NULL, "
                    "important BOOLEAN NOT NULL DEFAULT 0 CHECK (important IN (0, 1)))"
                )
            )
        with pytest.raises(TrackerSchemaMigrationError) as exc:
            _migrate_tracker_search_projection(engine)
        assert "observed_revision" in str(exc.value)

    def test_a_refusal_leaves_no_partial_derived_state_behind(self, engine):
        """Tables created earlier in the transaction must roll back too."""
        _create_tracker_tables(engine)
        with engine.begin() as conn:
            conn.execute(
                text(f"CREATE TABLE {VECTOR_DIRTY_TABLE} (generation_id TEXT PRIMARY KEY)")
            )
        with pytest.raises(TrackerSearchSchemaError):
            _migrate_tracker_search_projection(engine)
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name IN "
                    f"('{SEARCH_META_TABLE}', '{ISSUE_FTS_TABLE}', "
                    f"'{VECTOR_GENERATIONS_TABLE}')"
                )
            ).scalar()
        assert rows == 0


class TestFreshStoresWithoutTrackerTables:
    def test_absent_tracker_tables_are_left_to_create_all(self, engine):
        _migrate_tracker_search_projection(engine)
        assert _derived_shape(engine) == {}

    def test_one_tracker_table_without_the_other_is_refused(self, engine):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tracker_issues ("
                    "id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, "
                    "project_id TEXT NOT NULL, title TEXT NOT NULL, "
                    "observed_revision TEXT NULL)"
                )
            )
        with pytest.raises((TrackerSearchSchemaError, TrackerSchemaMigrationError)):
            _migrate_tracker_search_projection(engine)


class TestImmediateLockFirst:
    def test_the_migration_takes_begin_immediate_before_validating(self, engine):
        """The write lock is the migration's FIRST statement.

        Holding BEGIN IMMEDIATE from before shape validation through the
        coverage proof is what keeps a concurrent writer out of the
        backfill/trigger gap.
        """

        class RecordingConnection:
            def __init__(self, inner):
                self._inner = inner
                self.statements = []

            def execute(self, sql, *args, **kwargs):
                self.statements.append(str(sql))
                return self._inner.execute(sql, *args, **kwargs)

            def commit(self):
                self._inner.commit()

            def rollback(self):
                self._inner.rollback()

            def close(self):
                self._inner.close()

        recorded = []

        class RecordingEngine:
            def __init__(self, inner):
                self._inner = inner

            def raw_connection(self):
                wrapper = RecordingConnection(self._inner.raw_connection())
                recorded.append(wrapper)
                return wrapper

        _create_tracker_tables(engine)
        _migrate_tracker_search_projection(RecordingEngine(engine))
        assert recorded, "the migration never opened a raw connection"
        assert recorded[0].statements[0] == "BEGIN IMMEDIATE"

    def test_a_concurrent_writer_cannot_land_inside_the_migration(self, tmp_path):
        """A writer holding the database when the migration starts waits out.

        The migration blocks on ``BEGIN IMMEDIATE`` until the concurrent
        writer commits, then validates and projects the committed state; it
        never validates against or installs over an uncommitted snapshot.
        """
        import threading
        import time

        db_path = tmp_path / "concurrent.db"
        engine = create_engine(f"sqlite:///{db_path}")
        migration_engine = create_engine(f"sqlite:///{db_path}")
        try:
            _create_tracker_tables(engine)
            lock_held = threading.Event()
            release = threading.Event()

            def hold_lock_then_commit():
                writer = sqlite3.connect(str(db_path), timeout=10.0)
                try:
                    writer.execute("BEGIN IMMEDIATE")
                    writer.execute(
                        "INSERT INTO tracker_issues (key, project_id, title) "
                        "VALUES ('p-1', 'p', 'written during contention')"
                    )
                    lock_held.set()
                    release.wait(timeout=10.0)
                    writer.commit()
                finally:
                    writer.close()

            holder = threading.Thread(target=hold_lock_then_commit)
            holder.start()
            try:
                assert lock_held.wait(timeout=10.0)
                migrated = threading.Thread(
                    target=_migrate_tracker_search_projection, args=(migration_engine,)
                )
                migrated.start()
                time.sleep(0.3)
                assert (
                    migrated.is_alive()
                ), "the migration did not block on the concurrent write lock"
            finally:
                release.set()
                holder.join(timeout=10.0)
                migrated.join(timeout=10.0)
            assert not migrated.is_alive()
            with engine.begin() as conn:
                covered = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM tracker_issues AS s WHERE NOT EXISTS "
                        f"(SELECT 1 FROM {ISSUE_FTS_TABLE} AS f WHERE f.rowid = s.id)"
                    )
                ).scalar()
            assert covered == 0
        finally:
            engine.dispose()
            migration_engine.dispose()
