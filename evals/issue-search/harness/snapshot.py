# ABOUTME: Loader and validator for fixed issue-search snapshot exports.
# ABOUTME: Exposes the corpus in retrieval-ready, deterministically ordered form.
"""Load a snapshot export and expose it as a deterministic retrieval corpus.

The export is produced once by ``tools/export_snapshot.py`` from the live
tracker database (read-only) and committed. The harness treats it as immutable
input: every retriever sees the same rows in the same order, which is what
makes reruns from a fixed snapshot reproduce identical relevance numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPORT_SCHEMA_VERSION = 1


class SnapshotError(ValueError):
    """The snapshot export is missing, malformed, or mismatched."""


@dataclass(frozen=True)
class Issue:
    """One tracker issue exactly as exported."""

    seq: int  # export row id; reproduces the live tie-break ordering
    key: str
    project_id: str
    title: str
    body: str
    status: str
    severity: str
    component: str | None
    failing_command: str | None
    evidence: str | None
    resolution: str | None
    reproduction_steps: str | None
    expected_outcome: str | None
    actual_outcome: str | None
    created_at: str | None
    updated_at: str | None
    labels: list[str]
    duplicate_of: str | None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def searchable_fields(self) -> dict[str, str]:
        """The fields the legacy substring search covers.

        Mirrors ``list_issues`` in ``services/issue_tracker.py``: title, body,
        key, failing_command, reproduction_steps, expected_outcome,
        actual_outcome, evidence.
        """

        return {
            "title": self.title or "",
            "body": self.body or "",
            "key": self.key or "",
            "failing_command": self.failing_command or "",
            "reproduction_steps": self.reproduction_steps or "",
            "expected_outcome": self.expected_outcome or "",
            "actual_outcome": self.actual_outcome or "",
            "evidence": self.evidence or "",
        }


# Terminal statuses per the tracker's own semantics (issue_tracker.py).
TERMINAL_STATUSES = frozenset({"closed", "resolved", "wontfix", "duplicate"})


@dataclass(frozen=True)
class Comment:
    id: int
    issue_key: str
    author: str | None
    body: str
    created_at: str | None


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    issues: tuple[Issue, ...]
    comments: tuple[Comment, ...]
    part_of_children: dict[str, tuple[str, ...]]  # parent -> direct children
    counts: dict[str, int]

    def issue(self, key: str) -> Issue:
        for issue in self.issues:
            if issue.key == key:
                return issue
        raise SnapshotError(f"unknown issue key in snapshot: {key}")

    def subtree_closure(self, roots: list[str]) -> frozenset[str]:
        """Cycle-safe closure of ``root --part-of--> parent`` descendants.

        Includes each root itself and every transitive child. Mirrors the
        ranked-retrieval scope semantics (design §10.1); uses an explicit
        visited set rather than the bounded graph projection.
        """

        closure: set[str] = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node in closure:
                continue
            closure.add(node)
            stack.extend(self.part_of_children.get(node, ()))
        return frozenset(closure)


def load_snapshot(snapshots_dir: Path, snapshot_id: str | None = None) -> Snapshot:
    """Load and validate one snapshot export.

    With ``snapshot_id=None`` the snapshots directory must contain exactly one
    snapshot; ambiguity is an error rather than a silent pick.
    """

    if snapshot_id is None:
        candidates = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if len(candidates) != 1:
            raise SnapshotError(
                "snapshots directory must contain exactly one snapshot when "
                f"snapshot_id is omitted; found {[p.name for p in candidates]}"
            )
        snapshot_id = candidates[0].name
    snapshot_dir = snapshots_dir / snapshot_id
    export_path = snapshot_dir / "export.json"
    provenance_path = snapshot_dir / "provenance.json"
    if not export_path.is_file() or not provenance_path.is_file():
        raise SnapshotError(f"snapshot {snapshot_id} is incomplete: {snapshot_dir}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("snapshot_id") != snapshot_id:
        raise SnapshotError(
            f"provenance snapshot_id {provenance.get('snapshot_id')!r} does not "
            f"match directory {snapshot_id!r}"
        )

    raw = json.loads(export_path.read_text(encoding="utf-8"))
    if raw.get("export_schema_version") != EXPORT_SCHEMA_VERSION:
        raise SnapshotError(
            f"unsupported export schema version: {raw.get('export_schema_version')}"
        )
    declared = provenance.get("counts", {})
    for name in ("issues", "comments", "links"):
        if name in declared and len(raw[name]) != declared[name]:
            raise SnapshotError(
                f"snapshot {snapshot_id}: provenance counts {name}="
                f"{declared[name]} do not match export rows {len(raw[name])}"
            )

    issues = tuple(
        Issue(
            seq=row["id"],
            key=row["key"],
            project_id=row["project_id"],
            title=row["title"] or "",
            body=row["body"] or "",
            status=row["status"] or "open",
            severity=row["severity"] or "unset",
            component=row.get("component"),
            failing_command=row.get("failing_command"),
            evidence=row.get("evidence"),
            resolution=row.get("resolution"),
            reproduction_steps=row.get("reproduction_steps"),
            expected_outcome=row.get("expected_outcome"),
            actual_outcome=row.get("actual_outcome"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            labels=list(row.get("labels") or []),
            duplicate_of=row.get("duplicate_of"),
        )
        for row in raw["issues"]
    )
    comments = tuple(
        Comment(
            id=row["id"],
            issue_key=row["issue_key"],
            author=row.get("author"),
            body=row["body"] or "",
            created_at=row.get("created_at"),
        )
        for row in raw["comments"]
    )
    part_of_children: dict[str, list[str]] = {}
    for link in raw["links"]:
        if link["kind"] == "part-of":
            part_of_children.setdefault(link["to_key"], []).append(link["from_key"])
    frozen_children = {k: tuple(v) for k, v in part_of_children.items()}
    return Snapshot(
        snapshot_id=snapshot_id,
        issues=issues,
        comments=comments,
        part_of_children=frozen_children,
        counts=dict(raw.get("counts", {})),
    )


def load_fixture(path: Path) -> dict[str, Any]:
    """Load the fixture corpus document with minimal structural validation."""

    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("fixture_schema_version") != 1:
        raise SnapshotError(
            f"unsupported fixture schema version: {fixture.get('fixture_schema_version')}"
        )
    case_ids = [case["id"] for case in fixture["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise SnapshotError("fixture contains duplicate case ids")
    return fixture
