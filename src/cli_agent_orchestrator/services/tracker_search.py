"""Explicit maintenance verbs for the derived tracker search projection.

The lexical rebuild (design §13.2) is the repair counterpart of the read-only
integrity report (§13.4): the report never repairs, and the rebuild never
serves a half-rebuilt result set because it holds one immediate transaction
from shape validation through the FTS maintenance to the commit. Ordinary
issue writers either wait for that short transaction or observe the database's
existing typed busy behavior; none of them participate in model work.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from cli_agent_orchestrator.clients import tracker_search_schema
from cli_agent_orchestrator.clients.database import engine

SEARCH_META_TABLE = tracker_search_schema.SEARCH_META_TABLE


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw_connection(target_engine: Optional[Any] = None) -> Any:
    return (target_engine if target_engine is not None else engine).raw_connection()


def rebuild_lexical(target_engine: Optional[Any] = None) -> Dict[str, Any]:
    """Rebuild both FTS documents from authoritative rows (design §13.2).

    The idempotent projection installation runs first inside the same
    immediate transaction, so the verb is safe on a store whose projection
    predates an upgrade or was partially damaged. The rebuild repopulates
    every document with fresh content versions and — while the same
    transaction still holds — queues every live document into
    ``tracker_vector_dirty`` for each active/building generation, making all
    pre-rebuild vectors ineligible before any embedding work can start.

    Returns the per-kind document counts and the rebuild timestamp.
    """
    rebuilt_at = _utcnow_iso()
    raw = _raw_connection(target_engine)
    try:
        raw.execute("BEGIN IMMEDIATE")
        try:
            # The object shape, not the coverage proof: this verb IS the
            # recovery path for stores whose coverage proof refuses.
            tracker_search_schema.ensure_schema_objects(raw)
            summary = tracker_search_schema.rebuild_lexical(raw, rebuilt_at=rebuilt_at)
            tracker_search_schema.verify_coverage(raw)
            raw.commit()
        except Exception:
            raw.rollback()
            raise
    finally:
        raw.close()
    return {"rebuilt_at": rebuilt_at, **summary}


def integrity_report(target_engine: Optional[Any] = None) -> Dict[str, Any]:
    """Read-only §13.4 report over the derived search surface.

    Never repairs anything: a store without the derived schema reports
    ``installed=false`` rather than installing it, because a report must not
    mutate the thing it measures.
    """
    raw = _raw_connection(target_engine)
    try:
        installed = raw.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name IN "
            f"('{tracker_search_schema.SEARCH_META_TABLE}', "
            f"'{tracker_search_schema.ISSUE_FTS_TABLE}', "
            f"'{tracker_search_schema.COMMENT_FTS_TABLE}')"
        ).fetchone()[0]
        if int(installed) < 3:
            return {"installed": False}
        payload = tracker_search_schema.integrity_report(raw)
        return {"installed": True, **payload}
    finally:
        raw.close()
