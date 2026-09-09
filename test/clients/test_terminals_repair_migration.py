"""Migration coverage for the cond-0817 repair columns on ``terminals``.

The repair added nullable ``provider_readiness`` and
``create_request_fingerprint`` (the issue draft called the latter
``supervisor_request_fingerprint``; the shipped column name is asserted
here). Fresh databases get both from ``Base.metadata.create_all``; this
migration is the only path for a database created before the repair.
Both land NULL on legacy rows and are never backfilled, and a repeated
migration is a no-op that leaves existing row values untouched.
"""

from __future__ import annotations

import sqlite3

from cli_agent_orchestrator.clients import database


def test_model_declares_nullable_repair_columns():
    columns = database.TerminalModel.__table__.c

    assert columns.provider_readiness.nullable is True
    assert columns.create_request_fingerprint.nullable is True


def _legacy_db(path):
    """A terminals table shaped before the cond-0817 repair, with live values."""
    with sqlite3.connect(str(path)) as connection:
        connection.execute(
            "CREATE TABLE terminals ("
            "id TEXT PRIMARY KEY, tmux_session TEXT NOT NULL, "
            "tmux_window TEXT NOT NULL, provider TEXT NOT NULL, "
            "pane_id TEXT, assigned_model TEXT, assigned_effort TEXT)"
        )
        connection.execute(
            "INSERT INTO terminals "
            "(id, tmux_session, tmux_window, provider, pane_id, assigned_model, assigned_effort) "
            "VALUES ('legacy8101', 'cao-s', 'w-0', 'claude_code', '%81', 'gpt-5.6-sol', 'high')"
        )
        connection.execute(
            "INSERT INTO terminals (id, tmux_session, tmux_window, provider) "
            "VALUES ('legacy8102', 'cao-s', 'w-1', 'codex')"
        )


def _columns(path) -> list[str]:
    with sqlite3.connect(str(path)) as connection:
        return [row[1] for row in connection.execute("PRAGMA table_info(terminals)")]


def _legacy_row(path, terminal_id: str):
    with sqlite3.connect(str(path)) as connection:
        return connection.execute(
            "SELECT tmux_session, tmux_window, provider, pane_id, "
            "assigned_model, assigned_effort, provider_readiness, "
            "create_request_fingerprint FROM terminals WHERE id = ?",
            (terminal_id,),
        ).fetchone()


def test_first_apply_adds_both_columns_null_preserving_rows(tmp_path, monkeypatch):
    db_file = tmp_path / "pre-repair.db"
    _legacy_db(db_file)
    assert "provider_readiness" not in _columns(db_file)
    assert "create_request_fingerprint" not in _columns(db_file)

    monkeypatch.setattr("cli_agent_orchestrator.constants.DATABASE_FILE", db_file, raising=False)
    database._migrate_terminals_schema()

    columns = _columns(db_file)
    assert columns.count("provider_readiness") == 1
    assert columns.count("create_request_fingerprint") == 1
    assert _legacy_row(db_file, "legacy8101") == (
        "cao-s",
        "w-0",
        "claude_code",
        "%81",
        "gpt-5.6-sol",
        "high",
        None,
        None,
    )
    assert _legacy_row(db_file, "legacy8102")[6:] == (None, None)


def test_second_apply_is_idempotent_and_keeps_values(tmp_path, monkeypatch):
    db_file = tmp_path / "pre-repair-idem.db"
    _legacy_db(db_file)
    monkeypatch.setattr("cli_agent_orchestrator.constants.DATABASE_FILE", db_file, raising=False)

    database._migrate_terminals_schema()
    database._migrate_terminals_schema()

    columns = _columns(db_file)
    assert columns.count("provider_readiness") == 1
    assert columns.count("create_request_fingerprint") == 1
    assert _legacy_row(db_file, "legacy8101") == (
        "cao-s",
        "w-0",
        "claude_code",
        "%81",
        "gpt-5.6-sol",
        "high",
        None,
        None,
    )


def test_reapply_preserves_populated_repair_values(tmp_path, monkeypatch):
    """Populated repair fields survive a second production migration (cond-0852)."""
    from cli_agent_orchestrator import constants

    db_file = tmp_path / "populated-repair.db"
    _legacy_db(db_file)
    monkeypatch.setattr("cli_agent_orchestrator.constants.DATABASE_FILE", db_file, raising=False)
    assert constants.DATABASE_FILE == db_file
    database._migrate_terminals_schema()

    readiness, fingerprint = "ready", "9f" * 32
    with sqlite3.connect(str(db_file)) as connection:
        connection.execute(
            "UPDATE terminals SET provider_readiness = ?, "
            "create_request_fingerprint = ? WHERE id = ?",
            (readiness, fingerprint, "legacy8101"),
        )

    assert constants.DATABASE_FILE == db_file
    database._migrate_terminals_schema()

    assert _legacy_row(db_file, "legacy8101") == (
        "cao-s",
        "w-0",
        "claude_code",
        "%81",
        "gpt-5.6-sol",
        "high",
        readiness,
        fingerprint,
    )
    assert _legacy_row(db_file, "legacy8102")[6:] == (None, None)
