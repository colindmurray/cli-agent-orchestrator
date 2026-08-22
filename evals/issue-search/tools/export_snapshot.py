# ABOUTME: Read-only export of the live CAO tracker SQLite database into a
# ABOUTME: versioned snapshot consumed by the issue-search evaluation harness.
"""Export a fixed, read-only snapshot of the live tracker for evaluation.

The live tracker database is never opened for writing: the exporter connects
with SQLite URI ``mode=ro`` and issues SELECT statements only. The committed
artifact is ``snapshots/<snapshot-id>/export.json`` plus a ``provenance.json``
recording where the data came from and what was redacted.

``export.json`` is a pure function of the database content: no timestamps,
paths, or other run-local metadata are hashed. The snapshot id is therefore
derived from the SHA-256 of the exported bytes — the same database content
always yields the same id, and a mutated export is self-evident.

Personal-provider email addresses are redacted before anything is written
(the live corpus contains at least one maintainer address in a comment body);
the replacement is a fixed token so redaction stays deterministic and the
redaction count is recorded in the provenance block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Personal mail providers, mirroring the repo's fixture-PII guard
# (test/test_fixtures_no_personal_pii.py). Synthetic addresses
# (example.com, example.test, noreply@) and tooling banners (git@github.com)
# are deliberately not redacted.
_PERSONAL_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@(?:gmail|googlemail|yahoo|ymail|hotmail|outlook|live|msn|"
    r"icloud|me|mac|aol|proton|protonmail|pm|gmx|zoho|yandex|mail)\.[A-Za-z]{2,}",
    re.IGNORECASE,
)
_REDACTION_TOKEN = "REDACTED_PERSONAL_EMAIL"

ISSUE_COLUMNS = [
    "id",
    "key",
    "project_id",
    "title",
    "body",
    "status",
    "severity",
    "component",
    "reporter",
    "assignee",
    "labels",
    "failing_command",
    "evidence",
    "resolution",
    "session_name",
    "terminal_id",
    "source_path",
    "duplicate_of",
    "origin",
    "kind",
    "created_at",
    "updated_at",
    "closed_at",
    "reproduction_steps",
    "collaborators",
    "branches",
    "worktrees",
    "pull_requests",
    "expected_outcome",
    "actual_outcome",
    "favorite",
]

JSON_COLUMNS = ("labels", "collaborators", "branches", "worktrees", "pull_requests")


def _redact_text(value: str) -> tuple[str, int]:
    """Redact personal emails from a string; return (value, redaction count)."""

    matches = _PERSONAL_EMAIL_RE.findall(value)
    if not matches:
        return value, 0
    return _PERSONAL_EMAIL_RE.sub(_REDACTION_TOKEN, value), len(matches)


def _load_json_column(raw: Any) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw


def export_snapshot(db_path: Path) -> tuple[dict[str, Any], int]:
    """Read the tracker database read-only; return (export document, redactions)."""

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        projects = [
            {
                "id": row["id"],
                "name": row["name"],
                "status": row["status"],
                "issue_prefix": row["issue_prefix"],
            }
            for row in conn.execute(
                "SELECT id, name, status, issue_prefix FROM tracker_projects ORDER BY id"
            )
        ]

        issues: list[dict[str, Any]] = []
        redactions = 0
        for row in conn.execute(
            f"SELECT {', '.join(ISSUE_COLUMNS)} FROM tracker_issues ORDER BY id"
        ):
            issue = {col: row[col] for col in ISSUE_COLUMNS}
            for col in JSON_COLUMNS:
                issue[col] = _load_json_column(issue[col])
            for field, value in list(issue.items()):
                if isinstance(value, str):
                    issue[field], n = _redact_text(value)
                    redactions += n
                elif isinstance(value, list):
                    cleaned = []
                    for item in value:
                        cleaned_item, n = _redact_text(item)
                        cleaned.append(cleaned_item)
                        redactions += n
                    issue[field] = cleaned
            issues.append(issue)

        comments = []
        for row in conn.execute(
            "SELECT id, issue_key, author, body, created_at "
            "FROM tracker_issue_comments ORDER BY issue_key, id"
        ):
            comment = {
                "id": row["id"],
                "issue_key": row["issue_key"],
                "author": row["author"],
                "created_at": row["created_at"],
            }
            comment["body"], n = _redact_text(row["body"])
            redactions += n
            comments.append(comment)

        links = [
            {"from_key": row["from_key"], "kind": row["kind"], "to_key": row["to_key"]}
            for row in conn.execute(
                "SELECT from_key, kind, to_key FROM tracker_issue_links "
                "ORDER BY from_key, kind, to_key"
            )
        ]
    finally:
        conn.close()

    export = {
        "export_schema_version": 1,
        "counts": {
            "projects": len(projects),
            "issues": len(issues),
            "comments": len(comments),
            "links": len(links),
        },
        "projects": projects,
        "issues": issues,
        "comments": comments,
        "links": links,
    }
    return export, redactions


def write_snapshot(
    export: dict[str, Any], db_path: Path, redactions: int, snapshots_root: Path
) -> Path:
    """Serialize the export, derive the snapshot id from its hash, and write it."""

    payload = json.dumps(export, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    snapshot_id = f"snap-{digest[:12]}"
    snapshot_dir = snapshots_root / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "export.json").write_bytes(payload)
    provenance = {
        "snapshot_id": snapshot_id,
        "export_sha256": digest,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(db_path),
        "source_db_size_bytes": db_path.stat().st_size,
        "redactions": redactions,
        "access_mode": "read-only (sqlite URI mode=ro); live DB never written",
        "counts": export["counts"],
    }
    (snapshot_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return snapshot_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="live tracker SQLite path")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "snapshots",
        help="snapshots root (default: evals/issue-search/snapshots)",
    )
    args = parser.parse_args(argv)
    if not args.db.exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 1
    export, redactions = export_snapshot(args.db)
    snapshot_dir = write_snapshot(export, args.db, redactions, args.out)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot_dir.name,
                "counts": export["counts"],
                "redactions": redactions,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
