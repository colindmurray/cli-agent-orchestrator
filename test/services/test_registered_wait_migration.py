"""Registered-wait ORM/migration parity and idempotence."""

from __future__ import annotations

import sqlite3

from sqlalchemy import create_engine

from cli_agent_orchestrator.clients import database

EXPECTED_COLUMNS = {
    "wait_id",
    "operation_id",
    "request_digest",
    "request_json",
    "session_name",
    "owner_agent_id",
    "owner_incarnation_id",
    "owner_terminal_id",
    "owner_generation",
    "state",
    "deadline_at",
    "expiry_operation_id",
    "wake_message_id",
    "wake_pending_since",
    "outcome_json",
    "created_at",
    "updated_at",
}


def _shape(path):
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(registered_waits)")}
        indexes = {
            row[1]: bool(row[2]) for row in conn.execute("PRAGMA index_list(registered_waits)")
        }
    return columns, indexes


def test_registered_wait_migration_matches_orm_and_is_idempotent(tmp_path, monkeypatch):
    migrated = tmp_path / "migrated.db"
    orm = tmp_path / "orm.db"
    migration_engine = create_engine(f"sqlite:///{migrated}")
    orm_engine = create_engine(f"sqlite:///{orm}")
    try:
        monkeypatch.setattr(database, "engine", migration_engine)
        database._migrate_registered_waits()
        database._migrate_registered_waits()
        database.RegisteredWaitModel.__table__.create(bind=orm_engine)
    finally:
        migration_engine.dispose()
        orm_engine.dispose()

    migrated_shape = _shape(migrated)
    assert migrated_shape == _shape(orm)
    assert migrated_shape[0] == EXPECTED_COLUMNS
    assert migrated_shape[1]["ix_registered_waits_operation"] is True
    assert migrated_shape[1]["ix_registered_waits_expiry_operation"] is True
