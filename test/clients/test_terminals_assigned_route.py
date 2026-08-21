"""Persistence contract for a terminal's assigned provider route."""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.services import terminal_projection


def test_terminal_model_declares_nullable_assigned_route_columns():
    columns = database.TerminalModel.__table__.c

    assert columns.assigned_model.nullable is True
    assert columns.assigned_effort.nullable is True
    assert columns.assigned_quota_provider.nullable is True
    assert database.ManagedLaunchV2TerminalModel.__table__.c.v2_assigned_quota_provider.nullable


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
            "SELECT assigned_model, assigned_effort, assigned_quota_provider FROM terminals "
            "WHERE id = 'legacy0001'"
        ).fetchone()

    assert columns.count("assigned_model") == 1
    assert columns.count("assigned_effort") == 1
    assert columns.count("assigned_quota_provider") == 1
    assert route == (None, None, None)


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
        assigned_quota_provider="openai",
    )
    assert (created["assigned_model"], created["assigned_effort"]) == (
        "gpt-5.6-sol",
        "high",
    )
    database.create_terminal_v2(
        "term0002",
        "cao-s",
        "w-1",
        "codex",
        generation="gen-2",
        assigned_quota_provider="bytedance",
    )
    first_engine.dispose()

    second_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=second_engine))
    try:
        metadata = database.get_terminal_metadata("term0001")
        metadata_v2 = database.get_terminal_metadata_v2("term0002")
        monkeypatch.setattr(terminal_projection, "_provider_status", lambda _id: "idle")
        projected = terminal_projection.project_row(metadata, None, vintage="v1")
        projected_v2 = terminal_projection.project_row(metadata_v2, None, vintage="v2")
    finally:
        second_engine.dispose()

    assert metadata is not None
    assert (metadata["assigned_model"], metadata["assigned_effort"]) == (
        "gpt-5.6-sol",
        "high",
    )
    assert projected["assigned_quota_provider"] == "openai"
    assert projected_v2["assigned_quota_provider"] == "bytedance"


def test_missing_quota_column_is_unreadable(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'missing.db'}")
    database.Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE terminals DROP COLUMN assigned_quota_provider")
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))
    with pytest.raises(Exception, match="assigned_quota_provider"):
        database.get_terminal_metadata("missing")
    engine.dispose()


def test_v2_missing_quota_column_is_unreadable_to_projection(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'missing-v2.db'}")
    database.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))
    database.create_terminal_v2("missing2", "cao-s", "w-2", "codex", generation="gen-missing-v2")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "ALTER TABLE managed_launch_v2_terminals DROP COLUMN v2_assigned_quota_provider"
        )
    monkeypatch.setattr(terminal_projection, "_observed_panes", lambda: {})
    with pytest.raises(Exception, match="v2_assigned_quota_provider"):
        terminal_projection.project_terminal("missing2")
    engine.dispose()
