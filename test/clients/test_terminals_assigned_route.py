"""Persistence contract for a terminal's assigned provider route."""

from __future__ import annotations

import sqlite3

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database


def test_terminal_model_declares_nullable_assigned_route_columns():
    columns = database.TerminalModel.__table__.c

    assert columns.assigned_model.nullable is True
    assert columns.assigned_effort.nullable is True


def test_terminal_migration_adds_assigned_route_without_backfill(tmp_path, monkeypatch):
    db_file = tmp_path / "legacy.db"
    with sqlite3.connect(db_file) as connection:
        connection.execute(
            "CREATE TABLE terminals ("
            "id TEXT PRIMARY KEY, tmux_session TEXT NOT NULL, "
            "tmux_window TEXT NOT NULL, provider TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO terminals (id, tmux_session, tmux_window, provider) "
            "VALUES ('legacy0001', 'cao-s', 'w-0', 'kiro_cli')"
        )

    monkeypatch.setattr("cli_agent_orchestrator.constants.DATABASE_FILE", db_file, raising=False)

    database._migrate_terminals_schema()
    database._migrate_terminals_schema()

    with sqlite3.connect(db_file) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(terminals)")]
        route = connection.execute(
            "SELECT assigned_model, assigned_effort FROM terminals " "WHERE id = 'legacy0001'"
        ).fetchone()

    assert columns.count("assigned_model") == 1
    assert columns.count("assigned_effort") == 1
    assert route == (None, None)


def test_assigned_route_survives_database_restart(tmp_path, monkeypatch):
    db_file = tmp_path / "route.db"
    first_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    database.Base.metadata.create_all(bind=first_engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=first_engine))

    created = database.create_terminal(
        "term0001",
        "cao-s",
        "w-0",
        "codex",
        native_session_id="session-1",
        assigned_model="gpt-5.6-sol",
        assigned_effort="high",
    )
    assert (created["assigned_model"], created["assigned_effort"]) == (
        "gpt-5.6-sol",
        "high",
    )
    first_engine.dispose()

    second_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=second_engine))
    try:
        metadata = database.get_terminal_metadata("term0001")
    finally:
        second_engine.dispose()

    assert metadata is not None
    assert (metadata["assigned_model"], metadata["assigned_effort"]) == (
        "gpt-5.6-sol",
        "high",
    )
