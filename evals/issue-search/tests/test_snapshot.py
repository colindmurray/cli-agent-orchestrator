# ABOUTME: Tests for snapshot loading and validation.
# ABOUTME: A malformed or ambiguous snapshot must fail closed, never load.
"""Snapshot loader tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness.snapshot import SnapshotError, load_snapshot


def test_committed_snapshot_loads(snapshot) -> None:
    assert snapshot.snapshot_id.startswith("snap-")
    assert snapshot.counts["issues"] == len(snapshot.issues)
    assert snapshot.counts["comments"] == len(snapshot.comments)
    # Corpus shape the fixture depends on.
    keys = {issue.key for issue in snapshot.issues}
    assert "cond-0087" in keys
    assert "aegix-0001" in keys and "dnd-0001" in keys


def test_snapshot_id_mismatch_fails(tmp_path: Path) -> None:
    snap_dir = tmp_path / "snap-deadbeefcafe"
    snap_dir.mkdir()
    export = {
        "export_schema_version": 1,
        "counts": {},
        "projects": [],
        "issues": [],
        "comments": [],
        "links": [],
    }
    (snap_dir / "export.json").write_text(json.dumps(export), encoding="utf-8")
    provenance = {"snapshot_id": "snap-somethingelse", "counts": {}}
    (snap_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(SnapshotError, match="does not match"):
        load_snapshot(tmp_path)


def test_count_mismatch_fails(tmp_path: Path) -> None:
    snap_dir = tmp_path / "snap-deadbeefcafe"
    snap_dir.mkdir()
    export = {
        "export_schema_version": 1,
        "counts": {"issues": 1},
        "projects": [],
        "issues": [{"id": 1, "key": "x-0001", "project_id": "p"}],
        "comments": [],
        "links": [],
    }
    (snap_dir / "export.json").write_text(json.dumps(export), encoding="utf-8")
    provenance = {"snapshot_id": "snap-deadbeefcafe", "counts": {"issues": 5}}
    (snap_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(SnapshotError, match="provenance counts"):
        load_snapshot(tmp_path)


def test_ambiguous_snapshots_dir_fails(tmp_path: Path) -> None:
    for name in ("snap-aaaaaaaaaaaa", "snap-bbbbbbbbbbbb"):
        (tmp_path / name).mkdir()
    with pytest.raises(SnapshotError, match="exactly one"):
        load_snapshot(tmp_path)


def test_subtree_closure_is_cycle_safe(snapshot) -> None:
    closure = snapshot.subtree_closure(["cond-0628"])
    assert "cond-0628" in closure
    assert "cond-0633" in closure
    # Closure never invents keys.
    known = {issue.key for issue in snapshot.issues}
    assert closure <= known


def test_terminal_status_classification(snapshot) -> None:
    by_key = {issue.key: issue for issue in snapshot.issues}
    assert by_key["cond-0087"].is_terminal  # closed canonical
    assert not by_key["cond-0376"].is_terminal  # open live report
