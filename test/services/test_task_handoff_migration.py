"""M3-E ORM/migration parity, column evolution, restart persistence, rollback."""

from __future__ import annotations

import sqlite3
import uuid

from sqlalchemy import Integer, create_engine
from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateColumn

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.clients.database import Base
from cli_agent_orchestrator.services import task_handoff as th
from cli_agent_orchestrator.services import task_occurrence as occ

_HANDOFF_COLUMNS = {
    "handoff_id",
    "schema_version",
    "session_name",
    "task_occurrence_id",
    "from_agent_id",
    "to_agent_id",
    "from_incarnation_id",
    "from_terminal_id",
    "from_generation",
    "donor_revision",
    "packet_digest",
    "packet_control_id",
    "quiescence_json",
    "quiescence_digest",
    "delivery_state",
    "delivery_outcome",
    "delivery_receipt",
    "to_incarnation_id",
    "to_terminal_id",
    "to_generation",
    "successor_occurrence_id",
    "state",
    "receipt_digest",
    "detail",
    "initiated_by",
    "created_at",
    "updated_at",
    "settled_at",
}

_PARTIAL_INDEXES = {
    "ix_task_occurrence_handoffs_pending_occurrence",
    "ix_task_occurrence_handoffs_pending_donor",
    "ix_task_occurrence_handoffs_pending_recipient",
}

#: Frozen. The columns M3-E shipped that SQLite could not ``ALTER TABLE ADD`` —
#: the primary key and every NOT NULL column with no default that the store
#: carrying this table already had. This set is a historical record of the
#: shipped shape; a column introduced later (such as ``donor_revision`` added
#: after stores carrying the table existed) must never be added here — doing
#: so would silently license the one shape the migration cannot repair. It is
#: what makes "added after M3-E" checkable: a later column must be addable so
#: the migration can ALTER it in.
_M3E_UNADDABLE_COLUMNS = frozenset(
    {
        "handoff_id",
        "schema_version",
        "session_name",
        "task_occurrence_id",
        "from_agent_id",
        "to_agent_id",
        "from_incarnation_id",
        "from_terminal_id",
        "packet_digest",
        "packet_control_id",
        "quiescence_json",
        "quiescence_digest",
        "delivery_state",
        "state",
        "initiated_by",
        "created_at",
        "updated_at",
    }
)

_MODEL = database.TaskOccurrenceHandoffModel


def _model_columns():
    return {column.name for column in _MODEL.__table__.columns}


def _columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _indexes(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA index_list({table})")}


def _migrated_db(tmp_path, monkeypatch):
    """A pre-M3-E store brought forward by the additive migration alone."""
    path = tmp_path / "legacy.db"
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE TABLE terminals (id TEXT PRIMARY KEY)")
    from cli_agent_orchestrator import constants
    from cli_agent_orchestrator.clients import database as db_module

    monkeypatch.setattr(constants, "DATABASE_FILE", path)
    monkeypatch.setattr(db_module, "DATABASE_URL", f"sqlite:///{path}")
    db_module._migrate_task_occurrences()
    db_module._migrate_task_occurrence_handoffs()
    return path


def _orm_db(tmp_path):
    orm_path = tmp_path / "orm.db"
    engine = create_engine(f"sqlite:///{orm_path}")
    Base.metadata.create_all(bind=engine)
    engine.dispose()
    return orm_path


def test_migration_matches_the_orm_schema_column_for_column(tmp_path, monkeypatch):
    path = _migrated_db(tmp_path, monkeypatch)
    with sqlite3.connect(str(path)) as conn:
        assert _columns(conn, "task_occurrence_handoffs") == _HANDOFF_COLUMNS
    with sqlite3.connect(str(_orm_db(tmp_path))) as conn:
        assert _columns(conn, "task_occurrence_handoffs") == _HANDOFF_COLUMNS


def test_the_three_pending_indexes_are_partial_in_both_schemas(tmp_path, monkeypatch):
    """Full unique indexes would forbid a second handback for the same pair.

    The constraint is "one handoff in flight", not "one handoff ever". Getting
    it wrong would make a worker that was handed back once ineligible forever,
    which is exactly the reversibility this milestone exists to provide.
    """
    path = _migrated_db(tmp_path, monkeypatch)
    for store in (path, _orm_db(tmp_path)):
        with sqlite3.connect(str(store)) as conn:
            assert _PARTIAL_INDEXES <= _indexes(conn, "task_occurrence_handoffs")
            for name in _PARTIAL_INDEXES:
                sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
                ).fetchone()[0]
                assert "WHERE" in sql.upper()
                assert "'pending'" in sql


def test_migration_is_idempotent(tmp_path, monkeypatch):
    """Re-running preserves rows, not merely the column set.

    A migration that dropped and recreated the table would keep the columns
    identical, so the row is what makes this test about idempotency rather
    than about shape.
    """
    path = _migrated_db(tmp_path, monkeypatch)
    from cli_agent_orchestrator.clients import database as db_module

    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "INSERT INTO task_occurrence_handoffs (handoff_id, schema_version, session_name, "
            "task_occurrence_id, from_agent_id, to_agent_id, from_incarnation_id, "
            "from_terminal_id, donor_revision, packet_digest, packet_control_id, "
            "quiescence_json, quiescence_digest, delivery_state, state, initiated_by, "
            "created_at, updated_at) VALUES "
            "('h1', 'v', 's', 'o', 'a', 'b', 'i', 't', 0, 'd', 'c', '{}', 'q', 'pending', "
            "'pending', 'me', 'now', 'now')"
        )

    db_module._migrate_task_occurrence_handoffs()
    with sqlite3.connect(str(path)) as conn:
        assert _columns(conn, "task_occurrence_handoffs") == _HANDOFF_COLUMNS
        surviving = conn.execute("SELECT handoff_id FROM task_occurrence_handoffs").fetchall()
    assert surviving == [("h1",)]


# ---------------------------------------------------------------------------
# column evolution on a store that already carries the table (cond-0433)
# ---------------------------------------------------------------------------


def _create_at_shape(conn, names):
    """Create the handoff table carrying only ``names``, typed from the model."""
    specs = [
        str(CreateColumn(column).compile(dialect=sqlite_dialect()))
        for column in _MODEL.__table__.columns
        if column.name in names
    ]
    conn.execute(
        f"CREATE TABLE {_MODEL.__tablename__} ({', '.join(specs)}, PRIMARY KEY (handoff_id))"
    )


def _row_at_shape(names):
    """One row filling every column ``names`` requires, typed from the model."""
    return {
        column.name: (7 if isinstance(column.type, Integer) else f"{column.name}-original")
        for column in _MODEL.__table__.columns
        if column.name in names and not column.nullable
    }


def _store_at_older_shape(tmp_path, monkeypatch, names):
    """A store carrying the handoff table at ``names``, with one row in it."""
    path = tmp_path / "older-shape.db"
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE TABLE terminals (id TEXT PRIMARY KEY)")
        _create_at_shape(conn, names)
        row = _row_at_shape(names)
        conn.execute(
            f"INSERT INTO {_MODEL.__tablename__} ({', '.join(row)}) "
            f"VALUES ({', '.join('?' for _ in row)})",
            tuple(row.values()),
        )
    from cli_agent_orchestrator import constants

    monkeypatch.setattr(constants, "DATABASE_FILE", path)
    monkeypatch.setattr(database, "DATABASE_URL", f"sqlite:///{path}")
    return path, row


def test_a_store_that_already_has_the_table_gains_the_columns_it_lacks(tmp_path, monkeypatch):
    """The defect this closes: ``CREATE TABLE IF NOT EXISTS`` is a silent no-op
    against an existing table, so a store created at an older shape kept that
    shape forever — and the hold at ``task_handoff.hold_refusal`` is fail-closed
    on every managed write, so one missing column refuses every steer, tell and
    operator message on that installation, handoff party or not.

    The older shape here is the minimal legal one: the primary key and the NOT
    NULL columns, which are the only ones SQLite could never append later.
    """
    path, row = _store_at_older_shape(tmp_path, monkeypatch, _M3E_UNADDABLE_COLUMNS)
    with sqlite3.connect(str(path)) as conn:
        assert _columns(conn, "task_occurrence_handoffs") == set(_M3E_UNADDABLE_COLUMNS)

    database._migrate_task_occurrence_handoffs()

    with sqlite3.connect(str(path)) as conn:
        present = _columns(conn, "task_occurrence_handoffs")
        missing = _model_columns() - present
        assert not missing, (
            "the migration left an existing store short of "
            f"{sorted(missing)}; every managed write on it would be refused as "
            "handoff-held"
        )
        assert _PARTIAL_INDEXES <= _indexes(conn, "task_occurrence_handoffs")
        stored = dict(
            zip(
                [d[0] for d in conn.execute("SELECT * FROM task_occurrence_handoffs").description],
                conn.execute("SELECT * FROM task_occurrence_handoffs").fetchone(),
            )
        )
    # The pre-existing row keeps its bytes, and every appended nullable column
    # reads NULL rather than a fabricated value.
    assert {name: stored[name] for name in row} == row
    appended = present - set(row)
    assert appended
    assert all(
        stored[column.name] is None
        for column in _MODEL.__table__.columns
        if column.name in appended and column.nullable
    )


def test_reconciling_an_older_shape_store_is_idempotent(tmp_path, monkeypatch):
    """A rerun adds nothing and rewrites nothing.

    Asserted on the row as well as the shape: a migration that dropped and
    rebuilt the table would keep the columns identical and lose the row.
    """
    path, row = _store_at_older_shape(tmp_path, monkeypatch, _M3E_UNADDABLE_COLUMNS)
    database._migrate_task_occurrence_handoffs()
    with sqlite3.connect(str(path)) as conn:
        first = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'task_occurrence_handoffs'"
        ).fetchone()[0]

    database._migrate_task_occurrence_handoffs()

    with sqlite3.connect(str(path)) as conn:
        assert (
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'task_occurrence_handoffs'"
            ).fetchone()[0]
            == first
        )
        surviving = conn.execute("SELECT handoff_id FROM task_occurrence_handoffs").fetchall()
    assert surviving == [(row["handoff_id"],)]


def test_a_blocked_column_is_logged_at_error_with_its_consequence(tmp_path, monkeypatch, caplog):
    """init_db() deliberately continues past an unappendable column.

    A typed refusal beats a dead installation, so the migration swallows the
    reconcile's raise. But that leaves the store a column short, and the
    fail-closed hold then refuses every managed write on it as undecidable —
    a degraded installation, not a routine event. It has to be logged at
    error with the column named and the consequence spelled out; a routine
    warning is how the one installation that hits this gets missed.
    """
    import logging

    # A store whose handoff table lacks ``schema_version`` — NOT NULL with no
    # default, so SQLite cannot ALTER it in and, because the table already has
    # a row, no value can be invented for the missing column. The reconcile
    # must raise and the caller must log at error with the row count observed.
    _store_at_older_shape(tmp_path, monkeypatch, set(_M3E_UNADDABLE_COLUMNS) - {"schema_version"})

    with caplog.at_level(logging.WARNING, logger="cli_agent_orchestrator.clients.database"):
        database._migrate_task_occurrence_handoffs()  # must not raise

    blocked = [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR
        and "schema_version" in record.message
        and "column short" in record.message
    ]
    assert blocked, (
        "an unappendable column left the store a column short without an "
        "error-level trace naming it and its consequence; captured: "
        f"{[(r.levelname, r.message) for r in caplog.records]}"
    )
    # The message must state what it observed — the row count and why no value
    # can be supplied — and must not claim "to a populated table" unconditionally.
    assert any(
        "Row count observed" in r.message for r in blocked
    ), f"blocked message did not report the observed row count: {[r.message for r in blocked]}"
    assert all("to a populated table" not in r.message for r in blocked)


def test_no_column_added_after_m3e_is_beyond_alter_table():
    """The gate on a *future* column, which is what makes the repair a gate.

    A column added to ``TaskOccurrenceHandoffModel`` is absent from every store
    that already carries the table, so the migration has to be able to ALTER it
    in. SQLite only appends a column whose value for the stored rows is
    determined — nullable, or NOT NULL with a constant default — and appends no
    PRIMARY KEY or UNIQUE column at all. A new column outside that set cannot
    reach an upgraded store, and the fail-closed hold turns that into a total
    loss of managed input there, so it must be caught here rather than in
    production.

    The check runs in both directions, because the frozen set is a
    hand-maintained literal. A *relaxing* drift — an existing NOT NULL column
    flipped to nullable, or given a server_default — leaves the name in the
    frozen set while the model's addability moves on, and a name-set
    difference in one direction stays empty and passes. Asserting the
    (name, addability) pairs forces the drift to be named here, whichever
    side of the line it moves.
    """
    unaddable = {
        column.name
        for column in _MODEL.__table__.columns
        if database._sqlite_add_column_spec(column) is None
    }
    added_later = sorted(unaddable - _M3E_UNADDABLE_COLUMNS)
    assert not added_later, (
        f"{added_later} cannot be added to a store that already has "
        "task_occurrence_handoffs. Make the column nullable, give it a "
        "server_default, or rebuild the table explicitly — do not widen "
        "_M3E_UNADDABLE_COLUMNS."
    )
    relaxed = sorted(_M3E_UNADDABLE_COLUMNS - unaddable)
    assert not relaxed, (
        f"{relaxed} shipped with M3-E as unappendable but the model now lets "
        "SQLite ALTER them in — a nullability or server_default change on an "
        "existing column. If that change is deliberate, remove the name from "
        "_M3E_UNADDABLE_COLUMNS; do not leave it frozen over a shape the model "
        "no longer has."
    )


def test_the_raw_ddl_declares_every_column_the_model_does(tmp_path):
    """The migration's own ``CREATE TABLE`` is the shape a store that predates
    the table is created at, so it and the model must not drift apart."""
    path = tmp_path / "raw-ddl.db"
    with sqlite3.connect(str(path)) as conn:
        conn.execute(database._TASK_OCCURRENCE_HANDOFFS_DDL)
        declared = _columns(conn, "task_occurrence_handoffs")
    model = _model_columns()
    assert declared == model, (
        "_TASK_OCCURRENCE_HANDOFFS_DDL and TaskOccurrenceHandoffModel disagree: "
        f"declared only by the model={sorted(model - declared)}, "
        f"declared only by the DDL={sorted(declared - model)}"
    )


def test_records_survive_a_restart(tmp_path, monkeypatch):
    path = tmp_path / "restart.db"
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        database, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    donor_agent = str(uuid.uuid4())
    recipient_agent = str(uuid.uuid4())
    occurrence_id = str(uuid.uuid4())
    handoff_id = str(uuid.uuid4())
    occ.open_occurrence(
        occ.OpenRequest(
            task_occurrence_id=occurrence_id,
            session_name="cao-restart",
            agent_id=donor_agent,
            round_index=0,
            dispatch_digest="a" * 64,
            incarnation=occ.EffectIncarnation(incarnation_id="inc-1", terminal_id="term-1"),
        )
    )
    th.begin_handoff(
        th.BeginRequest(
            handoff_id=handoff_id,
            session_name="cao-restart",
            task_occurrence_id=occurrence_id,
            to_agent_id=recipient_agent,
            packet_digest="d" * 64,
            evidence=th.QuiescenceEvidence(
                incarnation_id="inc-1",
                terminal_id="term-1",
                turn_state=th.TURN_TERMINAL,
                observed_at="2026-08-16T12:00:00Z",
            ),
            initiated_by="supervisor",
            expected_donor_revision=0,
        )
    )
    engine.dispose()

    reopened = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(
        database, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=reopened)
    )
    record = th.get_handoff(handoff_id)
    assert record["state"] == th.STATE_PENDING
    assert record["from_agent_id"] == donor_agent
    assert record["quiescence"]["turn_state"] == th.TURN_TERMINAL
    # The hold survives a restart, which is what makes it a durable suspension
    # rather than one process's in-memory opinion.
    assert th.hold_refusal(donor_agent) is not None
    reopened.dispose()


def test_rollback_to_a_build_without_m3e_leaves_the_occurrence_schema_untouched(
    tmp_path, monkeypatch
):
    """M3-E adds one table and changes nothing M3-D owns.

    This is the whole rollback story. Because no column and no state value was
    added to ``task_occurrences``, an older binary's schema and its reading of
    every existing row are byte-identical to what they were; the handoff rows
    are simply unread.
    """
    path = _migrated_db(tmp_path, monkeypatch)
    with sqlite3.connect(str(path)) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "task_occurrence_handoffs" in tables
        occurrence_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'task_occurrences'"
        ).fetchone()[0]

    assert "handoff" not in occurrence_sql.lower()
    assert set(database.TaskOccurrenceModel.__table__.c.keys()) == {
        "task_occurrence_id",
        "schema_version",
        "session_name",
        "agent_id",
        "round_index",
        "dispatch_digest",
        "dispatch_provenance_json",
        "incarnation_id",
        "terminal_id",
        "generation",
        "lineage_id",
        "native_session_id",
        "state",
        "current_boundary_digest",
        "current_report_digest",
        "current_checkpoint_digest",
        "current_provenance_json",
        "current_summary_seed_digest",
        "current_artifact_seed_digest",
        "current_seed_quality",
        "current_seed_json",
        "final_disposition",
        "finalized_boundary_digest",
        "finalized_report_digest",
        "finalized_checkpoint_digest",
        "finalized_provenance_json",
        "finalized_summary_seed_digest",
        "finalized_artifact_seed_digest",
        "finalized_seed_quality",
        "finalized_seed_json",
        "finalized_by",
        "finalized_at",
        "revision",
        "created_at",
        "updated_at",
    }
    # And no third occurrence state was introduced.
    assert occ.STATES == {occ.STATE_OPEN, occ.STATE_FINALIZED}


def test_schema_versions_are_stamped_on_every_m3e_row(tmp_path, monkeypatch):
    """A row with no schema version is one a later build cannot interpret.

    Asserted on a real row rather than on the constant: NOT NULL catches a
    *missing* stamp, not a wrong one, so a `_begin_once` that wrote the
    occurrence service's version would satisfy the schema and this test alike.
    """
    path = tmp_path / "stamp.db"
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        database, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    donor_agent = str(uuid.uuid4())
    occurrence_id = str(uuid.uuid4())
    occ.open_occurrence(
        occ.OpenRequest(
            task_occurrence_id=occurrence_id,
            session_name="cao-stamp",
            agent_id=donor_agent,
            round_index=0,
            dispatch_digest="a" * 64,
            incarnation=occ.EffectIncarnation(incarnation_id="inc-1", terminal_id="term-1"),
        )
    )
    record = th.begin_handoff(
        th.BeginRequest(
            handoff_id=str(uuid.uuid4()),
            session_name="cao-stamp",
            task_occurrence_id=occurrence_id,
            to_agent_id=str(uuid.uuid4()),
            packet_digest="d" * 64,
            evidence=th.QuiescenceEvidence(
                incarnation_id="inc-1",
                terminal_id="term-1",
                turn_state=th.TURN_TERMINAL,
                observed_at="2026-08-16T12:00:00Z",
            ),
            initiated_by="supervisor",
            expected_donor_revision=0,
        )
    )
    assert record["schema_version"] == th.SCHEMA_VERSION
    assert th.SCHEMA_VERSION.startswith("cao-m3e-")
    assert record["schema_version"] != occ.SCHEMA_VERSION
    engine.dispose()


# ---------------------------------------------------------------------------
# cond-0500 donor-revision rebuild: empty and populated pre-#132 stores
# ---------------------------------------------------------------------------

#: The exact pre-#132 shape as of f8f60f64^ (verify with
#: ``git show f8f60f64^:src/cli_agent_orchestrator/clients/database.py``).
_PRE132_DDL = (
    "CREATE TABLE IF NOT EXISTS task_occurrence_handoffs ("
    "handoff_id TEXT NOT NULL PRIMARY KEY, schema_version TEXT NOT NULL, "
    "session_name TEXT NOT NULL, task_occurrence_id TEXT NOT NULL, "
    "from_agent_id TEXT NOT NULL, to_agent_id TEXT NOT NULL, "
    "from_incarnation_id TEXT NOT NULL, from_terminal_id TEXT NOT NULL, "
    "from_generation TEXT, packet_digest TEXT NOT NULL, "
    "packet_control_id TEXT NOT NULL, quiescence_json TEXT NOT NULL, "
    "quiescence_digest TEXT NOT NULL, rollback_predicate_json TEXT NOT NULL, "
    "delivery_state TEXT NOT NULL, delivery_outcome TEXT, delivery_receipt TEXT, "
    "successor_occurrence_id TEXT, state TEXT NOT NULL, "
    "epoch INTEGER NOT NULL DEFAULT 0, receipt_digest TEXT, detail TEXT, "
    "initiated_by TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
    "settled_at TEXT)"
)


def _pre132_store(tmp_path, monkeypatch, *, with_row: bool = False):
    """A store at the exact pre-#132 DDL, optionally with one legacy row."""
    path = tmp_path / ("pre132-with-row.db" if with_row else "pre132-empty.db")
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS terminals (id TEXT PRIMARY KEY)")
        conn.execute(_PRE132_DDL)
        if with_row:
            # Every NOT NULL column must be filled; rollback_predicate_json is the
            # load-bearing one that makes the old INSERT fail without a rebuild.
            conn.execute(
                "INSERT INTO task_occurrence_handoffs (handoff_id, schema_version, session_name, "
                "task_occurrence_id, from_agent_id, to_agent_id, from_incarnation_id, "
                "from_terminal_id, packet_digest, packet_control_id, quiescence_json, "
                "quiescence_digest, rollback_predicate_json, delivery_state, state, "
                "initiated_by, created_at, updated_at) VALUES "
                "('h-legacy', 'cao-m3e-1', 'sess', 'occ-1', 'agent-a', 'agent-b', "
                "'inc-a', 'term-a', 'd', 'c', '{}', 'q', '{\"pred\": 1}', 'pending', "
                "'pending', 'me', 'now', 'now')"
            )
    from cli_agent_orchestrator import constants

    monkeypatch.setattr(constants, "DATABASE_FILE", path)
    monkeypatch.setattr(database, "DATABASE_URL", f"sqlite:///{path}")
    return path


def test_empty_pre132_store_is_rebuilt_without_error_and_writable(tmp_path, monkeypatch, caplog):
    """Empty pre-#132 store → model's full column set, no ERROR, INSERT succeeds.

    Without the rebuild this fails on ``rollback_predicate_json``: even after
    making ``donor_revision`` addable, the legacy table still carries a NOT NULL
    no-default column the model never writes, so every INSERT raises.
    """
    import logging

    path = _pre132_store(tmp_path, monkeypatch, with_row=False)

    with caplog.at_level(logging.WARNING, logger="cli_agent_orchestrator.clients.database"):
        database._migrate_task_occurrence_handoffs()

    # No ERROR record — an empty store's rebuild is not degraded.
    assert not any(
        r.levelno >= logging.ERROR for r in caplog.records
    ), f"unexpected ERROR: {[(r.levelname, r.message) for r in caplog.records]}"

    with sqlite3.connect(str(path)) as conn:
        present = {
            row[1] for row in conn.execute("PRAGMA table_info(task_occurrence_handoffs)").fetchall()
        }
        assert _model_columns() <= present, f"model columns missing: {_model_columns() - present}"

    # INSERT of the model's columns must succeed (the pin for leg 3).
    engine = create_engine(f"sqlite:///{path}")
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        row = database.TaskOccurrenceHandoffModel(
            handoff_id=str(uuid.uuid4()),
            schema_version="v",
            session_name="s",
            task_occurrence_id=str(uuid.uuid4()),
            from_agent_id=str(uuid.uuid4()),
            to_agent_id=str(uuid.uuid4()),
            from_incarnation_id="inc",
            from_terminal_id="term",
            packet_digest="d" * 64,
            packet_control_id="c",
            quiescence_json="{}",
            quiescence_digest="q" * 64,
            delivery_state="pending",
            state="pending",
            initiated_by="me",
            created_at="now",
            updated_at="now",
            donor_revision=7,
        )
        s.add(row)
        s.commit()
    finally:
        s.close()
        engine.dispose()


def test_legacy_row_survives_rebuild_with_null_donor_revision(tmp_path, monkeypatch, caplog):
    """Populated pre-#132 store → row survives byte-identical, donor_revision NULL."""

    path = _pre132_store(tmp_path, monkeypatch, with_row=True)

    # Capture legacy row before migrate for byte-identity check.
    with sqlite3.connect(str(path)) as conn:
        before = dict(
            zip(
                [d[0] for d in conn.execute("SELECT * FROM task_occurrence_handoffs").description],
                conn.execute("SELECT * FROM task_occurrence_handoffs").fetchone(),
            )
        )

    import logging

    with caplog.at_level(logging.WARNING, logger="cli_agent_orchestrator.clients.database"):
        database._migrate_task_occurrence_handoffs()

    # The legacy row keeps its bytes; rollback_predicate_json is retained.
    with sqlite3.connect(str(path)) as conn:
        after = dict(
            zip(
                [d[0] for d in conn.execute("SELECT * FROM task_occurrence_handoffs").description],
                conn.execute("SELECT * FROM task_occurrence_handoffs").fetchone(),
            )
        )
        # Common columns are byte-identical.
        for col in before:
            if col in after:
                assert (
                    after[col] == before[col]
                ), f"column {col} changed: {before[col]!r} -> {after[col]!r}"
        assert after["rollback_predicate_json"] == '{"pred": 1}'
        # New donor_revision reads NULL (unpinned, not 0).
        assert after["donor_revision"] is None
        # No ERROR; rebuild succeeded.
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    # And a new INSERT still succeeds.
    engine = create_engine(f"sqlite:///{path}")
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        row = database.TaskOccurrenceHandoffModel(
            handoff_id=str(uuid.uuid4()),
            schema_version="v",
            session_name="s",
            task_occurrence_id=str(uuid.uuid4()),
            from_agent_id=str(uuid.uuid4()),
            to_agent_id=str(uuid.uuid4()),
            from_incarnation_id="inc",
            from_terminal_id="term",
            packet_digest="d" * 64,
            packet_control_id="c",
            quiescence_json="{}",
            quiescence_digest="q" * 64,
            delivery_state="pending",
            state="pending",
            initiated_by="me",
            created_at="now",
            updated_at="now",
            donor_revision=0,
        )
        s.add(row)
        s.commit()
    finally:
        s.close()
        engine.dispose()

    # _row_dict surfaces None, not 0.
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        # Use ORM to read the legacy row back via _row_dict.
        engine2 = create_engine(f"sqlite:///{path}")
        Session2 = sessionmaker(bind=engine2)
        s2 = Session2()
        try:
            orm_row = (
                s2.query(database.TaskOccurrenceHandoffModel).filter_by(handoff_id="h-legacy").one()
            )
            d = th._row_dict(orm_row)
            assert d["donor_revision"] is None
        finally:
            s2.close()
            engine2.dispose()


def test_rebuilt_shape_matches_fresh_store_plus_carried_columns(tmp_path, monkeypatch):
    """Byte-equivalence: rebuilt PRAGMA table_info vs fresh create_all, plus indexes."""

    # Rebuilt from pre-132 empty.
    path = _pre132_store(tmp_path, monkeypatch, with_row=False)
    database._migrate_task_occurrence_handoffs()

    with sqlite3.connect(str(path)) as rebuilt_conn:
        rebuilt_info = {
            row[1]: (row[1], row[2], row[3], row[4], row[5])
            for row in rebuilt_conn.execute(
                "PRAGMA table_info(task_occurrence_handoffs)"
            ).fetchall()
        }
        rebuilt_indexes = {
            row[1]: rebuilt_conn.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (row[1],)
            ).fetchone()[0]
            for row in rebuilt_conn.execute(
                "PRAGMA index_list(task_occurrence_handoffs)"
            ).fetchall()
        }

    # Fresh store via ORM.
    fresh_path = tmp_path / "fresh.db"
    fresh_engine = create_engine(f"sqlite:///{fresh_path}")
    Base.metadata.create_all(bind=fresh_engine)
    fresh_engine.dispose()
    with sqlite3.connect(str(fresh_path)) as fresh_conn:
        fresh_info = {
            row[1]: (row[1], row[2], row[3], row[4], row[5])
            for row in fresh_conn.execute("PRAGMA table_info(task_occurrence_handoffs)").fetchall()
        }
        fresh_indexes = {
            row[1]: fresh_conn.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (row[1],)
            ).fetchone()[0]
            for row in fresh_conn.execute("PRAGMA index_list(task_occurrence_handoffs)").fetchall()
        }

    # Fresh columns are all present in rebuilt, with identical type/notnull/dflt/pk.
    for name, tup in fresh_info.items():
        assert name in rebuilt_info, f"fresh column {name} missing in rebuilt"
        assert (
            rebuilt_info[name] == tup
        ), f"column {name} differs: fresh {tup} vs rebuilt {rebuilt_info[name]}"

    # Only extra columns are the carried store-only ones (epoch and rollback_predicate_json).
    extra = set(rebuilt_info.keys()) - set(fresh_info.keys())
    assert extra == {"epoch", "rollback_predicate_json"}, f"unexpected extra columns: {extra}"
    # Relaxed rollback_predicate_json is nullable; epoch retains its default.
    assert rebuilt_info["rollback_predicate_json"][2] == 0  # notnull 0 -> nullable
    assert rebuilt_info["epoch"][2] == 1
    assert rebuilt_info["epoch"][3] == "0"

    # Three partial indexes are present with WHERE pending.
    for idx in _PARTIAL_INDEXES:
        assert idx in rebuilt_indexes, f"missing index {idx}"
        sql = rebuilt_indexes[idx]
        assert "WHERE" in sql.upper()
        assert "'pending'" in sql
        assert idx in fresh_indexes
        # Compare WHERE clause presence, not normalized DDL whitespace.
        assert fresh_indexes[idx].upper().count("WHERE") == sql.upper().count("WHERE")


def _bind_roster_helper(agent_id, *, suffix, generation=None):
    """A real roster incarnation, the way any live worker has one (for null-donor test)."""
    from cli_agent_orchestrator.services import stable_agent_roster as roster

    return roster.bind_generation(
        roster.BindingContract(
            agent_id=agent_id,
            session_name="sess-null",
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="claude_code",
            native_session_id=f"native-{suffix}",
            acquisition_method="chosen_session_id",
            terminal_id=f"term-{suffix}",
            generation=generation or f"gen-{suffix}",
            pane_id=f"%{suffix}",
            pane_pid=6000 + int(suffix),
            process_identity={"pid": 6000 + int(suffix), "start_marker": f"marker-{suffix}"},
            execution_mode="native_tui",
            admitted=True,
        )
    )


def test_rebuild_is_atomic_under_copy_failure(tmp_path, monkeypatch):
    """Interruption during copy leaves original table and rows intact, no orphan."""

    path = _pre132_store(tmp_path, monkeypatch, with_row=True)
    with sqlite3.connect(str(path)) as conn:
        before_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(task_occurrence_handoffs)").fetchall()
        }
        before_row = conn.execute("SELECT handoff_id FROM task_occurrence_handoffs").fetchone()[0]
        before_count = conn.execute("SELECT COUNT(*) FROM task_occurrence_handoffs").fetchone()[0]

    import pytest

    # Use a proxy that forwards isolation_level correctly; Connection.execute is read-only.
    real_conn = sqlite3.connect(str(path))
    orig_isolation = real_conn.isolation_level

    class _Proxy:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __setattr__(self, name, value):
            if name == "_inner":
                object.__setattr__(self, name, value)
            elif name == "isolation_level":
                object.__setattr__(self, name, value)
                setattr(self._inner, name, value)
            else:
                setattr(self._inner, name, value)

        def execute(self, sql, params=()):
            if isinstance(sql, str) and "INSERT INTO" in sql and "__cao_rebuild" in sql:
                raise RuntimeError("injected copy failure")
            if params:
                return self._inner.execute(sql, params)
            return self._inner.execute(sql)

    proxy = _Proxy(real_conn)
    # Ensure proxy mirrors real isolation_level initially.
    proxy.isolation_level = orig_isolation
    try:
        with pytest.raises(RuntimeError, match="injected copy failure"):
            database._reconcile_columns_from_model(proxy, database.TaskOccurrenceHandoffModel)
        orphan_inside = proxy.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%cao_rebuild%'"
        ).fetchall()
        assert orphan_inside == [], f"orphan table left inside transaction: {orphan_inside}"
    finally:
        # Restore real isolation_level (proxy's finally already did, but ensure)
        try:
            real_conn.isolation_level = orig_isolation
        except Exception:
            pass
        real_conn.close()

    # Verify outside with a fresh connection that the original table is intact.
    with sqlite3.connect(str(path)) as conn:
        after_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(task_occurrence_handoffs)").fetchall()
        }
        after_rows = conn.execute("SELECT handoff_id FROM task_occurrence_handoffs").fetchall()
        after_count = conn.execute("SELECT COUNT(*) FROM task_occurrence_handoffs").fetchone()[0]
        assert after_cols == before_cols
        assert after_count == before_count
        assert after_rows[0][0] == before_row
        orphan = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%cao_rebuild%'"
        ).fetchall()
        assert orphan == [], f"orphan table left: {orphan}"


def test_null_donor_revision_cannot_be_adopted_or_transferred(tmp_path, monkeypatch):
    """A NULL donor_revision never compares equal and cannot satisfy the CAS."""

    path = tmp_path / "null-donor.db"
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        database, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    from cli_agent_orchestrator.clients.database import SessionLocal as _SL

    donor_agent = str(uuid.uuid4())
    recipient_agent = str(uuid.uuid4())
    occ_id = str(uuid.uuid4())
    occ.open_occurrence(
        occ.OpenRequest(
            task_occurrence_id=occ_id,
            session_name="sess-null",
            agent_id=donor_agent,
            round_index=0,
            dispatch_digest="a" * 64,
            incarnation=occ.EffectIncarnation(incarnation_id="inc-a", terminal_id="term-a"),
        )
    )
    # Bind roster for transfer helper.
    _bind_roster_helper(donor_agent, suffix="10", generation="gen-a")
    _bind_roster_helper(recipient_agent, suffix="20", generation="gen-b")

    # Insert a handoff with NULL donor_revision directly (simulating legacy row).
    handoff_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO task_occurrence_handoffs (handoff_id, schema_version, session_name, "
            "task_occurrence_id, from_agent_id, to_agent_id, from_incarnation_id, "
            "from_terminal_id, donor_revision, packet_digest, packet_control_id, "
            "quiescence_json, quiescence_digest, delivery_state, state, initiated_by, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                handoff_id,
                th.SCHEMA_VERSION,
                "sess-null",
                occ_id,
                donor_agent,
                recipient_agent,
                "inc-a",
                "term-a",
                None,
                "d" * 64,
                "c",
                "{}",
                "q" * 64,
                "pending",
                "pending",
                "me",
                "now",
                "now",
            ),
        )

    # _begin_once replay with same handoff_id must not adopt — NULL never equals any revision.
    import pytest

    with pytest.raises(th.TaskHandoffConflict) as exc:
        th.begin_handoff(
            th.BeginRequest(
                handoff_id=handoff_id,
                session_name="sess-null",
                task_occurrence_id=occ_id,
                to_agent_id=recipient_agent,
                packet_digest="d" * 64,
                evidence=th.QuiescenceEvidence(
                    incarnation_id="inc-a",
                    terminal_id="term-a",
                    turn_state=th.TURN_TERMINAL,
                    observed_at="2026-08-16T12:00:00Z",
                ),
                initiated_by="me",
                expected_donor_revision=0,
            )
        )
    assert "different immutable content" in str(exc.value)

    # Transfer CAS must refuse with "predates the revision pin" and suggest rollback.
    # First mark delivery as delivered (required for transfer) via raw update.
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE task_occurrence_handoffs SET delivery_state='delivered', "
            "to_incarnation_id='inc-b', to_terminal_id='term-b', to_generation='gen-b' "
            "WHERE handoff_id=?",
            (handoff_id,),
        )

    with pytest.raises(th.TaskHandoffConflict) as exc2:
        th.complete_handoff(
            handoff_id,
            incarnation=occ.EffectIncarnation(
                incarnation_id="inc-b", terminal_id="term-b", generation="gen-b", lineage_id="lin-b"
            ),
            expected_revision=0,
            completed_by="me",
        )
    msg = str(exc2.value)
    assert "predates the revision pin" in msg
    assert "packet cannot be shown to describe the round being transferred" in msg
    assert "roll" in msg.lower()

    # Rollback must still succeed — it does not read donor_revision.
    result = th.rollback_handoff(handoff_id, rolled_back_by="me", reason="test rollback")
    assert result["state"] == th.STATE_ROLLED_BACK

    engine.dispose()


def test_row_count_is_taken_inside_exclusive_transaction(tmp_path, monkeypatch):
    """Pin that SELECT COUNT(*) is taken under BEGIN IMMEDIATE, not before.

    If the count were taken before the lock, a concurrent insert between the
    count and the lock would make new_count != row_count and spuriously
    brick the store until the next init_db(). The count and the blocked
    decision must be inside the same exclusive transaction as the copy.
    """

    path = _pre132_store(tmp_path, monkeypatch, with_row=False)
    # Use a proxy that records the order of execute calls.
    real_conn = sqlite3.connect(str(path))
    order: list[str] = []

    class _OrderProxy:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __setattr__(self, name, value):
            if name == "_inner":
                object.__setattr__(self, name, value)
            elif name == "isolation_level":
                object.__setattr__(self, name, value)
                setattr(self._inner, name, value)
            else:
                setattr(self._inner, name, value)

        def execute(self, sql, params=()):
            # Record the SQL verb for ordering checks.
            if isinstance(sql, str):
                stripped = sql.strip().upper()
                if stripped.startswith("BEGIN"):
                    order.append("BEGIN")
                elif "SELECT COUNT(*)" in sql.upper() and "TASK_OCCURRENCE_HANDOFFS" in sql.upper():
                    order.append("COUNT")
                elif "INSERT INTO" in sql.upper() and "__CAO_REBUILD" in sql.upper():
                    order.append("INSERT")
            if params:
                return self._inner.execute(sql, params)
            return self._inner.execute(sql)

    proxy = _OrderProxy(real_conn)
    # Ensure proxy starts with same isolation_level as real.
    proxy.isolation_level = real_conn.isolation_level
    try:
        database._reconcile_columns_from_model(proxy, database.TaskOccurrenceHandoffModel)
    finally:
        try:
            real_conn.isolation_level = proxy.isolation_level  # type: ignore[attr-defined]
        except Exception:
            pass
        real_conn.close()

    # COUNT must come after BEGIN, not before.
    assert "BEGIN" in order, f"BEGIN not recorded: {order}"
    assert "COUNT" in order, f"COUNT not recorded: {order}"
    assert order.index("BEGIN") < order.index(
        "COUNT"
    ), f"COUNT was taken before BEGIN — ordering is outside the lock: {order}"

    # Also verify that a concurrent insert between outer PRAGMA and BEGIN would
    # be correctly counted if count is inside. Simulate by checking that a
    # rebuild that sees a row inserted just before BEGIN still succeeds
    # (empty store with a row inserted after the initial PRAGMA but before
    # the transaction would have row_count=0 outside and row_count=1 inside;
    # inside correctly rebuilds with 1 row, outside would mismatch).
    # We pin this by direct unit test of the ordering, not by racing threads
    # — the ordering assertion above is the load-bearing check.
    # Mutation: moving the SELECT COUNT(*) back before BEGIN IMMEDIATE makes
    # this test fail at the index assertion.


# ---------------------------------------------------------------------------
# Wiring: real init_db() against a populated pre-#132 store, then health
# ---------------------------------------------------------------------------


def test_legacy_populated_store_boots_through_real_init_db_and_serves_health(tmp_path, monkeypatch):
    """Wiring gap: the migration unit tests prove the reconcile, not the boot.

    ``test/clients/test_database.py::TestInitDb::test_init_db`` mocks ``Base`` so
    the real ``create_all``/migrations never run — a change to ``init_db()`` that
    hangs or raises at server boot would stay green. This test boots the app
    through the *real* ``init_db()`` against a legacy pre-#132 store (the
    populated shape that triggers the rebuild) and proves the server is
    serviceable afterwards.

    Placement: ``test/services/test_task_handoff_migration.py`` rather than
    ``test/api/test_session_env_lifespan.py`` because the only factory for a
    legacy store (``_PRE132_DDL`` / ``_pre132_store``) already lives here; a
    second copy in ``test/api/`` would be the same DDL drifting in two places.
    The ``test_session_env_lifespan`` precedent is the app-boot pattern, not the
    store shape, so co-locating with the shape keeps the wire and its fixture
    together.

    Bound trade: ``init_db()`` is 57 ms on a fresh store and well under a
    second on a legacy store here. A tight bound (e.g. 1 s) would be meaningful
    but flakes on slow CI — the repo already has one wall-clock flake
    (``test_a_native_identity_timeout_is_bounded_and_releases_the_lease``). The
    bound here is 10 s: an order of magnitude above the observed latency, so a
    genuine hang (``BEGIN IMMEDIATE`` blocked, infinite loop) trips it, but a
    slow runner does not. If 10 s still flakes, drop the timing assert and keep
    completion + serviceability — a hanging ``init_db()`` would instead be proved
    by the test timing out.

    Isolation: the suite already injects a per-session ``CAO_STATE_ROOT`` via
    ``test/conftest.py``; this test additionally monkeypatches
    ``constants.DATABASE_FILE`` and the import-time ``database.engine``/
    ``SessionLocal`` so the lifespan's ``init_db()`` hits the tmp store and
    cannot touch the operator's or suite's DB file. No ``Base``/migration mock.
    """

    import time

    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from cli_agent_orchestrator import constants
    from cli_agent_orchestrator.api.main import app as cao_app

    # Populated legacy store — the population that actually triggers the rebuild.
    path = _pre132_store(tmp_path, monkeypatch, with_row=True)

    # Redirect the import-time engine/session to the legacy file. The migration
    # itself uses ``sqlite3.connect(DATABASE_FILE)`` so ``DATABASE_FILE`` alone
    # would be enough for the DDL, but ``init_db`` also runs ``Base.metadata.
    # create_all(bind=engine)`` which is bound at import time.
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(
        database, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    # Keep DATABASE_URL consistent — not read by init_db after import, but
    # re-exported by callers that rebuild an engine from it.
    monkeypatch.setattr(database, "DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.setattr(constants, "DATABASE_FILE", path)

    # The migrations must genuinely execute — do not mock Base/create_all/_migrate_*.
    start = time.monotonic()
    try:
        database.init_db()
    except Exception as exc:  # pragma: no cover - failure path is the assertion
        raise AssertionError(f"init_db() raised on legacy store: {exc}") from exc
    elapsed = time.monotonic() - start
    # Loose but bounded — see docstring trade. Under a competing held write
    # lock ``BEGIN IMMEDIATE`` respects the 5 s busy timeout and returns
    # ``database is locked`` (caught as warning, next boot repairs), so 10 s
    # covers both the happy path and the lock path without hanging the suite.
    assert elapsed < 10, f"init_db() hung or was pathologically slow: {elapsed:.2f}s"

    # Post-migration: the legacy row is still there, donor_revision is NULL, and
    # the new shape is readable through the ORM the server uses (not just via
    # raw sqlite3) — the cheapest proof the store is usable after boot short of
    # a full HTTP serve.
    with database.SessionLocal() as session:
        row = (
            session.query(database.TaskOccurrenceHandoffModel)
            .filter_by(handoff_id="h-legacy")
            .one()
        )
        d = th._row_dict(row)
        assert d["donor_revision"] is None

    # Cheapest honest proof the wiring holds: the app actually serves after the
    # migrated boot. ``TestClient`` drives the real lifespan (which reruns
    # ``init_db()`` idempotently — that second run is also part of the wiring
    # proof; the first timed run above is what carries the hang bound).
    try:
        with TestClient(cao_app, base_url="http://localhost") as client:
            resp = client.get("/health")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body.get("status") == "ok"
            # Health is backend-agnostic; just prove we got *a* health body back,
            # not that a particular provider is installed on this runner.
    finally:
        engine.dispose()
