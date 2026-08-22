"""The observed-columns tracker migration (cond-0636).

The contract under test: ``ensure_tracker_schema`` (CLI entry point) and
``init_db`` (API entry point) both run ONE injectable, idempotent raw
migration that takes ``BEGIN IMMEDIATE`` before validating shape, adds
``tracker_issues.observed_revision`` and ``tracker_issue_comments.important``,
and refuses an incompatible prior shape with a typed error instead of
log-and-continuing into a half-migrated store.
"""

import pytest
from sqlalchemy import create_engine, text

from cli_agent_orchestrator.clients.database import (
    _TRACKER_ORM_TABLE_NAMES,
    Base,
    TrackerSchemaMigrationError,
    _migrate_tracker_observed_revision_columns,
    ensure_tracker_schema,
    init_db,
)


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/observed-columns.db")
    yield eng
    eng.dispose()


def _create_legacy_store(eng):
    """Tracker tables as they looked before either column existed."""
    with eng.begin() as conn:
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
                "id INTEGER PRIMARY KEY, issue_key TEXT NOT NULL, "
                "author TEXT, body TEXT NOT NULL)"
            )
        )


def _columns(eng, table):
    with eng.begin() as conn:
        return {row[1]: row for row in conn.execute(text(f"PRAGMA table_info({table})"))}


class TestLegacyMigration:
    def test_both_columns_are_added_to_a_legacy_store(self, engine):
        _create_legacy_store(engine)
        _migrate_tracker_observed_revision_columns(engine)
        assert "observed_revision" in _columns(engine, "tracker_issues")
        important = _columns(engine, "tracker_issue_comments")["important"]
        assert important[2].upper() == "BOOLEAN"
        assert important[3] == 1  # NOT NULL
        assert str(important[4]).strip("'\"") == "0"

    def test_existing_rows_survive_the_migration(self, engine):
        _create_legacy_store(engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO tracker_issues (key, project_id, title) "
                    "VALUES ('cond-0001', 'p', 'held')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO tracker_issue_comments (issue_key, body) VALUES ('cond-0001', 'note')"
                )
            )
        _migrate_tracker_observed_revision_columns(engine)
        with engine.begin() as conn:
            issue = conn.execute(
                text("SELECT key, observed_revision FROM tracker_issues")
            ).fetchall()
            comment = conn.execute(
                text("SELECT body, important FROM tracker_issue_comments")
            ).fetchall()
        assert issue == [("cond-0001", None)]
        assert comment == [("note", 0)]

    def test_repeated_migration_is_idempotent(self, engine):
        _create_legacy_store(engine)
        _migrate_tracker_observed_revision_columns(engine)
        first = _columns(engine, "tracker_issue_comments")
        _migrate_tracker_observed_revision_columns(engine)
        _migrate_tracker_observed_revision_columns(engine)
        assert _columns(engine, "tracker_issue_comments") == first
        assert len(_columns(engine, "tracker_issues")) == 5

    def test_the_migrated_important_column_enforces_its_check(self, engine):
        _create_legacy_store(engine)
        _migrate_tracker_observed_revision_columns(engine)
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO tracker_issues (key, project_id, title) VALUES ('c', 'p', 't')")
            )
            try:
                conn.execute(
                    text(
                        "INSERT INTO tracker_issue_comments (issue_key, body, important) "
                        "VALUES ('c', 'b', 2)"
                    )
                )
            except Exception:
                return
        raise AssertionError("the migrated column accepted a value outside (0, 1)")


class TestIncompatibleShapes:
    def test_a_nullable_important_column_is_refused(self, engine):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tracker_issue_comments ("
                    "id INTEGER PRIMARY KEY, issue_key TEXT NOT NULL, "
                    "body TEXT NOT NULL, important BOOLEAN NULL DEFAULT 0)"
                )
            )
        with pytest.raises(TrackerSchemaMigrationError) as exc:
            _migrate_tracker_observed_revision_columns(engine)
        assert "must be NOT NULL" in str(exc.value)
        assert exc.value.table == "tracker_issue_comments"

    def test_an_incompatible_default_is_refused(self, engine):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tracker_issue_comments ("
                    "id INTEGER PRIMARY KEY, issue_key TEXT NOT NULL, "
                    "body TEXT NOT NULL, important BOOLEAN NOT NULL DEFAULT 1)"
                )
            )
        with pytest.raises(TrackerSchemaMigrationError) as exc:
            _migrate_tracker_observed_revision_columns(engine)
        assert "incompatible default" in str(exc.value)

    def test_a_non_boolean_important_type_is_refused(self, engine):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tracker_issue_comments ("
                    "id INTEGER PRIMARY KEY, issue_key TEXT NOT NULL, "
                    "body TEXT NOT NULL, important TEXT NOT NULL DEFAULT 0)"
                )
            )
        with pytest.raises(TrackerSchemaMigrationError):
            _migrate_tracker_observed_revision_columns(engine)

    def test_a_not_null_observed_revision_is_refused(self, engine):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tracker_issues ("
                    "id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, "
                    "project_id TEXT NOT NULL, title TEXT NOT NULL, "
                    "observed_revision TEXT NOT NULL)"
                )
            )
        with pytest.raises(TrackerSchemaMigrationError) as exc:
            _migrate_tracker_observed_revision_columns(engine)
        assert "must be nullable" in str(exc.value)

    def test_a_non_text_observed_revision_is_refused(self, engine):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tracker_issues ("
                    "id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, "
                    "project_id TEXT NOT NULL, title TEXT NOT NULL, "
                    "observed_revision INTEGER NULL)"
                )
            )
        with pytest.raises(TrackerSchemaMigrationError):
            _migrate_tracker_observed_revision_columns(engine)

    def test_a_malformed_base_table_is_refused(self, engine):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tracker_issue_comments ("
                    "id INTEGER PRIMARY KEY, note TEXT NOT NULL)"
                )
            )
        with pytest.raises(TrackerSchemaMigrationError) as exc:
            _migrate_tracker_observed_revision_columns(engine)
        assert "malformed" in str(exc.value)

    def test_a_refusal_leaves_no_partial_state_behind(self, engine):
        """Both tables are touched inside one transaction: a refusal on one must
        roll back the other's ALTER too."""
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tracker_issues ("
                    "id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, "
                    "project_id TEXT NOT NULL, title TEXT NOT NULL, "
                    "observed_revision INTEGER NULL)"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE tracker_issue_comments ("
                    "id INTEGER PRIMARY KEY, issue_key TEXT NOT NULL, "
                    "author TEXT, body TEXT NOT NULL)"
                )
            )
        with pytest.raises(TrackerSchemaMigrationError):
            _migrate_tracker_observed_revision_columns(engine)
        # The comments ALTER ran after the issues validation failed shape —
        # neither store may carry the change.
        assert "observed_revision" in _columns(engine, "tracker_issues")
        assert "important" not in _columns(engine, "tracker_issue_comments")


class TestFreshAndInjectedEngines:
    def test_the_fresh_create_all_shape_passes_validation_unchanged(self, engine):
        Base.metadata.create_all(
            bind=engine,
            tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
        )
        before_issues = _columns(engine, "tracker_issues")
        _migrate_tracker_observed_revision_columns(engine)
        assert _columns(engine, "tracker_issues") == before_issues
        assert "important" in _columns(engine, "tracker_issue_comments")

    def test_absent_tracker_tables_are_left_to_create_all(self, engine):
        _migrate_tracker_observed_revision_columns(engine)
        assert _columns(engine, "tracker_issues") == {}

    def test_the_migration_takes_begin_immediate_before_validating(self, engine):
        """The write lock is the migration's FIRST statement, before any PRAGMA.

        Holding BEGIN IMMEDIATE from before shape validation through both
        ALTERs is what keeps a concurrent writer out of the validate/alter gap.
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

        _create_legacy_store(engine)
        _migrate_tracker_observed_revision_columns(RecordingEngine(engine))
        assert recorded, "the migration never opened a raw connection"
        assert recorded[0].statements[0] == "BEGIN IMMEDIATE"
        assert recorded[0].statements[1].startswith("PRAGMA table_info")


class TestBothEntryPointsRunIt:
    """ensure_tracker_schema (CLI) and init_db (API) must land the SAME shape.

    Both call the shared migration against their module-global engine, so each
    test points that global at a private store and runs the real entry point.
    """

    @staticmethod
    def _shape(engine):
        with engine.begin() as conn:
            issues = [
                (row[1], row[2], row[3], row[4])
                for row in conn.execute(text("PRAGMA table_info(tracker_issues)"))
            ]
            comments = [
                (row[1], row[2], row[3], row[4])
                for row in conn.execute(text("PRAGMA table_info(tracker_issue_comments)"))
            ]
        return issues, comments

    def test_ensure_tracker_schema_produces_the_migrated_shape(self, tmp_path, monkeypatch):
        from cli_agent_orchestrator.clients import database as db_module

        engine = create_engine(f"sqlite:///{tmp_path}/cli.db")
        try:
            monkeypatch.setattr(db_module, "engine", engine)
            ensure_tracker_schema()
            issues, comments = self._shape(engine)
            assert any(col[0] == "observed_revision" for col in issues)
            assert any(col[0] == "important" for col in comments)
        finally:
            engine.dispose()

    def test_init_db_and_ensure_tracker_schema_agree_on_the_final_shape(
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
            assert self._shape(api_engine) == self._shape(cli_engine)
        finally:
            cli_engine.dispose()
            api_engine.dispose()
