"""One-transaction issue snapshot behavior."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import (
    Base,
    TrackerCommentModel,
    TrackerIssueModel,
)
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services.issue_tracker import TrackerError


@pytest.fixture
def snapshot_store(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.db"
    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
    )
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
    Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    tracker.create_project(
        name="CAO System",
        project_id="cao-system",
        issue_prefix="cond",
    )
    try:
        yield engine, sessions
    finally:
        engine.dispose()


def test_snapshot_stays_at_one_wal_read_instant(snapshot_store):
    engine, sessions = snapshot_store
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0001",
        title="old title",
    )
    tracker.add_comment("cond-0001", body="old comment", author="before")

    selected_read = threading.Event()
    writer_done = threading.Event()
    writer_errors = []
    main_thread = threading.get_ident()

    def write_new_version():
        assert selected_read.wait(timeout=5)
        try:
            with sessions() as db:
                row = db.query(TrackerIssueModel).filter_by(key="cond-0001").one()
                row.title = "new title"
                row.updated_at = datetime(2026, 8, 31, 5, 0, tzinfo=timezone.utc)
                db.add(
                    TrackerCommentModel(
                        issue_key="cond-0001",
                        author="after",
                        body="new comment",
                    )
                )
                db.commit()
        except BaseException as exc:  # surfaced in the test thread below
            writer_errors.append(exc)
        finally:
            writer_done.set()

    @event.listens_for(engine, "after_cursor_execute")
    def pause_after_selected_issue_read(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ):
        if (
            threading.get_ident() == main_thread
            and not selected_read.is_set()
            and "FROM tracker_issues" in statement
        ):
            selected_read.set()
            assert writer_done.wait(timeout=5)

    writer = threading.Thread(target=write_new_version, daemon=True)
    writer.start()
    snapshot = tracker.snapshot_issues(project_id="cao-system", keys=["cond-0001"])
    writer.join(timeout=5)

    assert not writer.is_alive()
    assert writer_errors == []
    assert snapshot["issues"][0]["title"] == "old title"
    assert [comment["body"] for comment in snapshot["issues"][0]["comments"]] == ["old comment"]

    live = tracker.get_issue("cond-0001")
    assert live["title"] == "new title"
    assert [comment["body"] for comment in live["comments"]] == [
        "old comment",
        "new comment",
    ]


def test_snapshot_closes_over_roots_native_links_text_refs_and_projects(snapshot_store):
    tracker.create_project(
        name="External",
        project_id="external",
        issue_prefix="ext",
    )
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0001",
        title="selected one",
        body="The parent is also named directly as cond-0004; missing cond-9999.",
    )
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0002",
        title="selected two",
    )
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0003",
        title="parent",
    )
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0004",
        title="root",
    )
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0005",
        title="incoming relation endpoint",
    )
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0006",
        title="comment reference",
    )
    tracker.create_issue(
        project_id="external",
        key="ext-0001",
        title="cross-project endpoint",
    )
    tracker.create_issue(
        project_id="external",
        key="ext-0002",
        title="transitive text reference",
    )

    tracker.add_comment(
        "cond-0001",
        author="reviewer",
        body="Selected comment names cond-0006.",
        important=True,
    )
    tracker.add_comment(
        "cond-0006",
        author="context",
        body="Comments on closure records are retained too.",
    )
    tracker.add_comment(
        "ext-0001",
        author="context",
        body="Follow ext-0002; ext-9999 does not exist.",
    )
    tracker.add_link("cond-0001", to_key="cond-0003", kind="part-of")
    tracker.add_link("cond-0003", to_key="cond-0004", kind="part-of")
    tracker.add_link("cond-0002", to_key="ext-0001", kind="relates")
    tracker.add_link("cond-0005", to_key="cond-0001", kind="blocks")

    snapshot = tracker.snapshot_issues(
        project_id="cao-system",
        keys=["cond-0002", "cond-0001"],
    )

    assert snapshot == tracker.snapshot_issues(
        project_id="cao-system",
        keys=["cond-0001", "cond-0002"],
    )
    assert snapshot["schema"] == "cao-tracker-issue-snapshot-v1"
    assert snapshot["selected_keys"] == ["cond-0001", "cond-0002"]
    assert snapshot["selected_keys_digest"] == {
        "algorithm": "sha256-sorted-newline-v1",
        "value": "0ba5ffc42aa997a3fc1596dc64f6b04817ce64ee5ff8b8a73085d4bf0b1570f9",
    }
    assert snapshot["root_keys"] == ["cond-0003", "cond-0004"]
    assert snapshot["reference_keys"] == [
        "cond-0005",
        "cond-0006",
        "ext-0001",
        "ext-0002",
    ]
    assert snapshot["roles"] == [
        {"key": "cond-0001", "role": "selected"},
        {"key": "cond-0002", "role": "selected"},
        {"key": "cond-0003", "role": "root"},
        {"key": "cond-0004", "role": "root"},
        {"key": "cond-0005", "role": "reference"},
        {"key": "cond-0006", "role": "reference"},
        {"key": "ext-0001", "role": "reference"},
        {"key": "ext-0002", "role": "reference"},
    ]
    assert [issue["key"] for issue in snapshot["issues"]] == [
        "cond-0001",
        "cond-0002",
        "cond-0003",
        "cond-0004",
        "cond-0005",
        "cond-0006",
        "ext-0001",
        "ext-0002",
    ]
    assert all("events" not in issue for issue in snapshot["issues"])
    issues = {issue["key"]: issue for issue in snapshot["issues"]}
    assert issues["cond-0001"]["comments"][0]["important"] is True
    assert issues["cond-0006"]["comments"][0]["body"].startswith("Comments on closure")
    assert {
        (link["from_key"], link["to_key"], link["kind"]) for link in issues["cond-0001"]["links"]
    } == {
        ("cond-0001", "cond-0003", "part-of"),
        ("cond-0005", "cond-0001", "blocks"),
    }
    assert {
        (link["from_key"], link["to_key"], link["kind"])
        for issue in snapshot["issues"]
        for link in issue["links"]
    } == {
        ("cond-0001", "cond-0003", "part-of"),
        ("cond-0002", "ext-0001", "relates"),
        ("cond-0003", "cond-0004", "part-of"),
        ("cond-0005", "cond-0001", "blocks"),
    }
    assert [project["id"] for project in snapshot["projects"]] == [
        "cao-system",
        "external",
    ]
    assert [project["name"] for project in snapshot["projects"]] == [
        "CAO System",
        "External",
    ]
    assert snapshot["unresolved_references"] == [
        {"kind": "issue", "key": "cond-9999", "reason": "not-found", "sources": ["text"]},
        {"kind": "issue", "key": "ext-9999", "reason": "not-found", "sources": ["text"]},
    ]
    assert snapshot["consistency"] == {"kind": "sqlite-read-transaction"}
    assert "captured_at" not in snapshot
    assert "transaction_id" not in snapshot


def test_snapshot_rejects_duplicate_missing_and_cross_project_selection(snapshot_store):
    tracker.create_project(
        name="External",
        project_id="external",
        issue_prefix="ext",
    )
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0001",
        title="selected",
    )
    tracker.create_issue(
        project_id="external",
        key="ext-0001",
        title="wrong project",
    )

    with pytest.raises(TrackerError) as duplicate:
        tracker.snapshot_issues(
            project_id="cao-system",
            keys=["cond-0001", "cond-0001"],
        )
    assert duplicate.value.code == "invalid"
    assert duplicate.value.details == {"duplicate_keys": ["cond-0001"]}

    with pytest.raises(TrackerError) as missing:
        tracker.snapshot_issues(
            project_id="cao-system",
            keys=["cond-0001", "cond-9999"],
        )
    assert missing.value.code == "not-found"
    assert missing.value.details == {"missing_keys": ["cond-9999"]}

    with pytest.raises(TrackerError) as mismatch:
        tracker.snapshot_issues(
            project_id="cao-system",
            keys=["cond-0001", "ext-0001"],
        )
    assert mismatch.value.code == "conflict"
    assert mismatch.value.details == {
        "project_mismatches": [{"key": "ext-0001", "project_id": "external"}]
    }


def test_snapshot_uses_exactly_one_session_factory(snapshot_store, monkeypatch):
    _engine, sessions = snapshot_store
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0001",
        title="selected",
    )
    calls = 0

    def counted_session():
        nonlocal calls
        calls += 1
        return sessions()

    monkeypatch.setattr(tracker, "SessionLocal", counted_session)

    tracker.snapshot_issues(project_id="cao-system", keys=["cond-0001"])

    assert calls == 1
