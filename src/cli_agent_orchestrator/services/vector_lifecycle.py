"""Vector generation lifecycle (design §9.3, §13.3).

Owns everything that happens to a vector generation after the M1.2 schema
and triggers exist: creating a ``building`` generation with its initial bulk
enqueue in one immediate transaction, refreshing dirty documents through the
content-version compare-and-set (embedding outside SQLite), recording typed
failures through the same CAS, and the activation transaction that proves
coverage/encoding/dimension before switching ``active_vector_generation``
and retiring the predecessor.

Transaction boundaries follow design §9.3 exactly. The refresh loop reads
each document's FTS content and retains the dirty row's versions, releases
the database, builds text/hash/embeds with no transaction open, then
publishes in one short immediate transaction that inserts the vector and
deletes the matching dirty row; that delete is the CAS. A zero-row CAS means
the source changed or disappeared during inference and the computed vector
is discarded — concurrent refreshers may duplicate work but cannot publish
stale content, and no lease or fail-closed global lock exists.

Query-time derived maintenance treats :func:`drain_bounded_batch` as a
``_READ`` operation (design §9.2): it may advance disposable vector-cache
rows but never touches an authoritative tracker row or event.

Connection seams. Refresh is indexing work and opens its connection through
:func:`search_engine_factory.open_search_connection` — the dedicated
factory, never the authoritative pooled engine, per §7.2. Generation
creation, activation, failure marking, and status are pure relational
maintenance over already-stored bytes; they use the same injectable
raw-connection seam as ``services/tracker_search.py`` without loading
sqlite-vec, so a store can finish or audit a generation even while the vec
runtime is absent (§13.5). Neither path ever writes through the pooled
``SessionLocal`` engine.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from cli_agent_orchestrator.clients import tracker_search_schema as schema
from cli_agent_orchestrator.clients.database import engine
from cli_agent_orchestrator.services.search_engine_factory import open_search_connection

__all__ = [
    "VectorLifecycleError",
    "BACKOFF_BASE_SECONDS",
    "BACKOFF_CAP_SECONDS",
    "DOCUMENT_FIELD_LIMIT",
    "BOUNDED_REFRESH_BATCH",
    "issue_document_text",
    "comment_document_text",
    "new_generation_id",
    "generation_record_from_metadata",
    "create_generation",
    "refresh_generation",
    "drain_bounded_batch",
    "mark_generation_failed",
    "activate_generation",
    "semantic_status",
]

#: Retry backoff for failed embeddings (design §9.3 "retry after backoff"):
#: exponential from the base, capped. ``attempt_number`` is the count AFTER
#: the failed attempt.
BACKOFF_BASE_SECONDS = 60
BACKOFF_CAP_SECONDS = 3600

#: Per-field character cap applied by the versioned document builder (§9.1:
#: "Long fields are normalized and truncated by a versioned document
#: builder"). Part of document-schema v1 alongside the schema module's
#: ``DOCUMENT_SCHEMA_VERSION``.
DOCUMENT_FIELD_LIMIT = 4000

#: Default batch bound for query-time derived refresh (§9.2: "A semantic/
#: hybrid query drains a bounded eligible batch for the active generation
#: before vector retrieval").
BOUNDED_REFRESH_BATCH = 32

_DIRTY_KEY_COLUMNS = "generation_id, document_key"

# Keep this tuple in lockstep with the generation table.  A dimension-only
# check is insufficient: two model revisions/artifacts can emit same-shaped
# blobs that are numerically incompatible.
GENERATION_IDENTITY_COLUMNS = (
    "model_id",
    "model_revision",
    "runtime_id",
    "runtime_version",
    "artifact_sha256",
    "dimensions",
    "element_type",
    "distance_metric",
    "normalized",
    "document_schema_version",
)


class VectorLifecycleError(RuntimeError):
    """Typed lifecycle refusal; ``reason`` names the observed condition."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlite_moment(moment: datetime) -> str:
    """UTC timestamp in the same ``...T...SS.SSSZ`` shape SQLite writes."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


# ---------------------------------------------------------------------------
# Document builder (§9.1) — versioned text forms over FTS document columns
# ---------------------------------------------------------------------------


def _field(value: Any) -> str:
    """Normalize whitespace and truncate to the v1 per-field cap."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:DOCUMENT_FIELD_LIMIT]


def issue_document_text(row: Mapping[str, Any]) -> str:
    """The issue identity document form from §9.1.

    Reads the FTS document columns (the exact indexed projection), never raw
    source rows, so the embedded text cannot disagree with the content
    version being published. Mechanical claim/work-context text — assignee,
    session ids, worktree paths, PR lists — stays out of the embedding input.
    """
    return "\n".join(
        line.rstrip()
        for line in (
            f"title: {_field(row.get('title'))}",
            f"component: {_field(row.get('component'))}",
            f"command: {_field(row.get('failing_command'))}",
            f"actual: {_field(row.get('actual_outcome'))}",
            f"expected: {_field(row.get('expected_outcome'))}",
            f"reproduction: {_field(row.get('reproduction_steps'))}",
            f"observed revision: {_field(row.get('observed_revision'))}",
            f"body: {_field(row.get('body'))}",
            f"resolution: {_field(row.get('resolution'))}",
        )
    )


def comment_document_text(row: Mapping[str, Any]) -> str:
    """The comment document form from §9.1 with its compact context prefix."""
    return "\n".join(
        line.rstrip()
        for line in (
            f"issue: {_field(row.get('issue_key'))} — {_field(row.get('issue_title'))}",
            f"component: {_field(row.get('component'))}",
            f"comment by {_field(row.get('author'))}:",
            _field(row.get("body")),
        )
    )


_DOCUMENT_BUILDERS = {
    "issue": issue_document_text,
    "comment": comment_document_text,
}

_FTS_READ_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "issue": (
        "title",
        "component",
        "failing_command",
        "actual_outcome",
        "expected_outcome",
        "reproduction_steps",
        "observed_revision",
        "body",
        "resolution",
    ),
    "comment": ("issue_key", "issue_title", "component", "author", "body"),
}


def _source_table_for(document_kind: str) -> str:
    return "tracker_issues" if document_kind == "issue" else "tracker_issue_comments"


def _fts_table_for(document_kind: str) -> str:
    return schema.ISSUE_FTS_TABLE if document_kind == "issue" else schema.COMMENT_FTS_TABLE


# ---------------------------------------------------------------------------
# Generation creation + initial bulk enqueue (§13.3, first half)
# ---------------------------------------------------------------------------


def new_generation_id(now: Optional[datetime] = None) -> str:
    """A fresh generation id: microsecond timestamp plus a random tail."""
    moment = now or datetime.now(timezone.utc)
    return f"{moment.strftime('%Y%m%dT%H%M%S%f')}-{secrets.token_hex(3)}"


def _require_derived_schema(raw: Any) -> None:
    found = raw.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN "
        f"('{schema.SEARCH_META_TABLE}', '{schema.VECTOR_GENERATIONS_TABLE}', "
        f"'{schema.VECTOR_DIRTY_TABLE}', '{schema.SEARCH_VECTORS_TABLE}', "
        f"'{schema.ISSUE_FTS_TABLE}', '{schema.COMMENT_FTS_TABLE}')"
    ).fetchone()[0]
    if int(found) < 6:
        raise VectorLifecycleError(
            "schema-missing",
            "derived search schema is not installed; run tracker schema init "
            "(which installs the M1.2 projection) before vector lifecycle work",
        )


def generation_record_from_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Project prepared-model metadata onto the generation-table columns, typed.

    Public because the maintenance orchestrator needs the SAME projection the
    insert uses: deciding reuse-vs-create means comparing a candidate
    generation's identity against the rows already stored, and a second copy
    of this mapping could quietly disagree with the one that writes the row.
    """

    def required(key: str) -> Any:
        value = metadata.get(key)
        if value is None:
            raise VectorLifecycleError(
                "metadata-incomplete",
                f"prepared-model metadata is missing {key!r}; re-run the explicit "
                "model prepare command to rewrite it",
            )
        return value

    runtime_id = required("runtime_id")
    runtime_versions = metadata.get("runtime_versions")
    runtime_version = (
        runtime_versions.get(runtime_id) if isinstance(runtime_versions, Mapping) else None
    )
    if not runtime_version:
        raise VectorLifecycleError(
            "metadata-incomplete",
            f"prepared-model metadata records no {runtime_id!r} runtime version",
        )
    dimensions = required("dimensions")
    if not isinstance(dimensions, int) or dimensions <= 0:
        raise VectorLifecycleError(
            "metadata-incomplete", f"dimensions {dimensions!r} is not a positive integer"
        )
    element_type = required("element_type")
    if element_type != "float32":
        raise VectorLifecycleError(
            "metadata-incompatible",
            f"element_type {element_type!r} violates the float32 generation contract",
        )
    distance_metric = required("distance_metric")
    if distance_metric not in ("cosine", "l2"):
        raise VectorLifecycleError(
            "metadata-incompatible",
            f"distance_metric {distance_metric!r} is outside the declared domain",
        )
    document_schema_version = required("document_schema_version_id")
    if not isinstance(document_schema_version, int) or document_schema_version <= 0:
        raise VectorLifecycleError(
            "metadata-incomplete",
            f"document_schema_version_id {document_schema_version!r} is not a positive integer",
        )
    return {
        "model_id": required("model_id"),
        "model_revision": required("model_revision"),
        "runtime_id": runtime_id,
        "runtime_version": runtime_version,
        "artifact_sha256": required("artifact_sha256"),
        "dimensions": dimensions,
        "element_type": element_type,
        "distance_metric": distance_metric,
        "normalized": 1 if required("normalized") else 0,
        "document_schema_version": document_schema_version,
    }


def create_generation(
    *,
    models_dir: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    target_engine: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create one ``building`` generation and enqueue every live document.

    The generation row and the bulk enqueue share ONE immediate transaction
    (design §13.3), so no mutation can land between creation and enqueue:
    the seven triggers take over the moment the transaction commits. The
    record comes from the prepared-model metadata (M2.1); loading model
    weights is deliberately not required here. Creating a second building
    generation is allowed — a fresh build is the recovery path after a
    failed or abandoned build — and each gets its own id.
    """
    if metadata is None:
        from cli_agent_orchestrator.services.embedding_adapter import (
            default_models_dir,
            read_metadata,
        )

        metadata = read_metadata(models_dir if models_dir is not None else default_models_dir())
    if not isinstance(metadata, Mapping):
        raise VectorLifecycleError(
            "unprepared",
            "no prepared-model metadata found; run the explicit model prepare "
            "command before creating a vector generation",
        )
    record = generation_record_from_metadata(metadata)
    raw = (target_engine if target_engine is not None else engine).raw_connection()
    try:
        raw.execute("BEGIN IMMEDIATE")
        try:
            _require_derived_schema(raw)
            created = _create_generation_in_transaction(raw, record)
            raw.commit()
        except Exception:
            raw.rollback()
            raise
    finally:
        raw.close()
    return {
        **created,
        **record,
    }


def _create_generation_in_transaction(raw: Any, record: Mapping[str, Any]) -> Dict[str, Any]:
    """Insert and enqueue one generation while the caller owns BEGIN IMMEDIATE.

    The helper is deliberately private: :func:`ensure_generation` uses it to
    keep its identity recheck, insert, and initial queue in one transaction,
    while the public create verb retains its explicit fresh-generation
    semantics for recovery/build repair callers.
    """
    created_at = _utcnow_iso()
    generation_id = new_generation_id()
    raw.execute(
        f"INSERT INTO {schema.VECTOR_GENERATIONS_TABLE} (\n"
        "  generation_id, state, model_id, model_revision, runtime_id,\n"
        "  runtime_version, artifact_sha256, dimensions, element_type,\n"
        "  distance_metric, normalized, document_schema_version,\n"
        "  created_at\n"
        ") VALUES (?, 'building', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            generation_id,
            record["model_id"],
            record["model_revision"],
            record["runtime_id"],
            record["runtime_version"],
            record["artifact_sha256"],
            record["dimensions"],
            record["element_type"],
            record["distance_metric"],
            record["normalized"],
            record["document_schema_version"],
            created_at,
        ),
    )
    enqueued = schema.enqueue_all_live_documents(raw)
    return {
        "generation_id": generation_id,
        "state": "building",
        "created_at": created_at,
        "enqueued_documents": enqueued,
    }


# ---------------------------------------------------------------------------
# Per-document refresh with content-version CAS (§9.3)
# ---------------------------------------------------------------------------


def _backoff_seconds(attempt_number: int) -> int:
    scaled: int = BACKOFF_BASE_SECONDS * (2 ** max(0, attempt_number - 1))
    return min(scaled, BACKOFF_CAP_SECONDS)


def _resolve_target_generations(raw: Any, generation_id: Optional[str]) -> List[str]:
    if generation_id is None:
        rows = raw.execute(
            f"SELECT generation_id FROM {schema.VECTOR_GENERATIONS_TABLE} "
            "WHERE state IN ('building', 'active') ORDER BY created_at, generation_id"
        ).fetchall()
        return [row[0] for row in rows]
    row = raw.execute(
        f"SELECT state FROM {schema.VECTOR_GENERATIONS_TABLE} WHERE generation_id = ?",
        (generation_id,),
    ).fetchone()
    if row is None:
        raise VectorLifecycleError(
            "unknown-generation",
            f"no {schema.VECTOR_GENERATIONS_TABLE} row named {generation_id!r}",
        )
    if row[0] not in ("building", "active"):
        raise VectorLifecycleError(
            "generation-not-refreshable",
            f"generation {generation_id!r} is {row[0]!r}; only building/active "
            "generations accept refresh work",
        )
    return [generation_id]


def _generation_identity(raw: Any, generation_id: str) -> Dict[str, Any]:
    """Read the complete persisted identity for one refresh target."""
    row = raw.execute(
        f"SELECT {', '.join(GENERATION_IDENTITY_COLUMNS)} "
        f"FROM {schema.VECTOR_GENERATIONS_TABLE} WHERE generation_id = ?",
        (generation_id,),
    ).fetchone()
    if row is None:
        raise VectorLifecycleError(
            "unknown-generation", f"no generation row named {generation_id!r}"
        )
    return dict(zip(GENERATION_IDENTITY_COLUMNS, row))


def _embedder_identity(embedder: Any) -> Optional[Dict[str, Any]]:
    """Project an embedder's bound metadata, when it exposes that contract.

    Test/dry-run embedders may intentionally be metadata-free; a single
    explicitly selected generation can still use one of those.  A refresh
    spanning generations with different persisted identities is refused below
    unless the embedder proves which identity it carries.
    """
    metadata = getattr(embedder, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    try:
        return generation_record_from_metadata(metadata)
    except VectorLifecycleError as exc:
        raise VectorLifecycleError(
            "embedder-identity-invalid",
            f"embedder metadata cannot establish a generation identity: {exc}",
        ) from exc


def _same_generation_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(column) == right.get(column) for column in GENERATION_IDENTITY_COLUMNS)


def _select_eligible_batch(
    raw: Any, generation_ids: List[str], limit: Optional[int], now_iso: str
) -> List[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in generation_ids)
    params: List[Any] = [*generation_ids, now_iso]
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(limit)
    rows: List[sqlite3.Row] = raw.execute(
        f"SELECT {_DIRTY_KEY_COLUMNS}, issue_key, document_kind, source_id,\n"
        f"       content_version, document_schema_version\n"
        f"FROM {schema.VECTOR_DIRTY_TABLE}\n"
        f"WHERE generation_id IN ({placeholders})\n"
        f"  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)\n"
        f"ORDER BY enqueued_at, document_key\n{limit_sql}",
        params,
    ).fetchall()
    return rows


def _read_document_snapshot(raw: Any, row: sqlite3.Row) -> Optional[Mapping[str, Any]]:
    """Read the FTS document at the dirty row's version (§9.3 step 1).

    Returns ``None`` when the source row is gone — deletion races resolve at
    publish time through the exact-version CAS, never by guessing. A dirty
    row whose live FTS version differs from its own is damage no trigger
    path can produce; the caller reports and skips it.
    """
    document_kind = row["document_kind"]
    source_exists = raw.execute(
        f"SELECT 1 FROM {_source_table_for(document_kind)} WHERE id = ?",
        (row["source_id"],),
    ).fetchone()
    if source_exists is None:
        return None
    columns = _FTS_READ_COLUMNS[document_kind]
    selected = ", ".join(("content_version",) + columns)
    fts_row = raw.execute(
        f"SELECT {selected} FROM {_fts_table_for(document_kind)} WHERE rowid = ?",
        (row["source_id"],),
    ).fetchone()
    if fts_row is None:
        return None
    snapshot: Dict[str, Any] = {name: fts_row[index] for index, name in enumerate(columns, start=1)}
    snapshot["_fts_content_version"] = int(fts_row[0])
    return snapshot


def _publish_vector(
    raw: Any,
    row: sqlite3.Row,
    content_sha256: str,
    blob: bytes,
) -> Dict[str, int]:
    """Publish one computed vector under the full §9.3 success CAS.

    One short immediate transaction re-checks the source, inserts/replaces
    the generation-scoped vector, and deletes the dirty row ONLY at the same
    content and document-schema version. Zero deleted rows roll the insert
    back: the source changed during inference and the vector is discarded.
    """
    document_kind = row["document_kind"]
    raw.execute("BEGIN IMMEDIATE")
    try:
        source_exists = raw.execute(
            f"SELECT 1 FROM {_source_table_for(document_kind)} WHERE id = ?",
            (row["source_id"],),
        ).fetchone()
        if source_exists is None:
            raw.rollback()
            deleted = _delete_dirty_exact(raw, row)
            return {"published": 0, "discarded_stale": 0, "deleted_source_gone": deleted}
        raw.execute(
            f"INSERT INTO {schema.SEARCH_VECTORS_TABLE} (\n"
            "  generation_id, document_key, issue_key, document_kind, source_id,\n"
            "  content_version, content_sha256, embedding, indexed_at\n"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)\n"
            "ON CONFLICT (generation_id, document_key) DO UPDATE SET\n"
            "  issue_key = excluded.issue_key,\n"
            "  document_kind = excluded.document_kind,\n"
            "  source_id = excluded.source_id,\n"
            "  content_version = excluded.content_version,\n"
            "  content_sha256 = excluded.content_sha256,\n"
            "  embedding = excluded.embedding,\n"
            "  indexed_at = excluded.indexed_at",
            (
                row["generation_id"],
                row["document_key"],
                row["issue_key"],
                document_kind,
                row["source_id"],
                row["content_version"],
                content_sha256,
                blob,
                _utcnow_iso(),
            ),
        )
        cursor = raw.execute(
            f"DELETE FROM {schema.VECTOR_DIRTY_TABLE}\n"
            " WHERE generation_id = ? AND document_key = ?\n"
            "   AND content_version = ? AND document_schema_version = ?",
            (
                row["generation_id"],
                row["document_key"],
                row["content_version"],
                row["document_schema_version"],
            ),
        )
        if cursor.rowcount != 1:
            raw.rollback()
            return {"published": 0, "discarded_stale": 1, "deleted_source_gone": 0}
        raw.commit()
        return {"published": 1, "discarded_stale": 0, "deleted_source_gone": 0}
    except Exception:
        raw.rollback()
        raise


def _delete_dirty_exact(raw: Any, row: sqlite3.Row) -> int:
    """Delete only this exact dirty version; anything newer stays queued."""
    cursor = raw.execute(
        f"DELETE FROM {schema.VECTOR_DIRTY_TABLE}\n"
        " WHERE generation_id = ? AND document_key = ?\n"
        "   AND content_version = ? AND document_schema_version = ?",
        (
            row["generation_id"],
            row["document_key"],
            row["content_version"],
            row["document_schema_version"],
        ),
    )
    raw.commit()
    return int(cursor.rowcount or 0)


def _cas_failure(raw: Any, row: sqlite3.Row, error: str) -> int:
    """Record a failed attempt through the §9.3 failure CAS.

    Updates attempts, next retry time, and the error only where the dirty
    row still carries the content and document-schema version the embed was
    computed against; zero rows means a newer version owns the row and the
    stale failure is discarded so it cannot overwrite the newer version's
    reset retry state.
    """
    attempt_row = raw.execute(
        f"SELECT attempt_count FROM {schema.VECTOR_DIRTY_TABLE}\n"
        " WHERE generation_id = ? AND document_key = ?\n"
        "   AND content_version = ? AND document_schema_version = ?",
        (
            row["generation_id"],
            row["document_key"],
            row["content_version"],
            row["document_schema_version"],
        ),
    ).fetchone()
    if attempt_row is None:
        return 0
    next_attempt_number = int(attempt_row[0]) + 1
    retry_moment = datetime.now(timezone.utc) + timedelta(
        seconds=_backoff_seconds(next_attempt_number)
    )
    next_retry = _sqlite_moment(retry_moment)
    cursor = raw.execute(
        f"UPDATE {schema.VECTOR_DIRTY_TABLE}\n"
        "   SET attempt_count = attempt_count + 1,\n"
        "       next_attempt_at = ?,\n"
        "       last_error = ?\n"
        " WHERE generation_id = ? AND document_key = ?\n"
        "   AND content_version = ? AND document_schema_version = ?",
        (
            next_retry,
            error,
            row["generation_id"],
            row["document_key"],
            row["content_version"],
            row["document_schema_version"],
        ),
    )
    raw.commit()
    return int(cursor.rowcount or 0)


def refresh_generation(
    *,
    generation_id: Optional[str] = None,
    limit: Optional[int] = None,
    embedder: Optional[Any] = None,
    models_dir: Optional[Any] = None,
    db_path: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Refresh eligible dirty documents for the target generation(s).

    Per document (design §9.3): read the FTS content and retain versions;
    release the database; build text, hash it, and embed outside SQLite;
    publish in one short transaction only if the source still exists AND the
    dirty row still carries the same content and document-schema version,
    deleting exactly that dirty version; a zero-row CAS discards the computed
    vector. Failures update attempts/backoff/error through the same CAS, so
    a stale failure can never overwrite a newer version's reset retry state.

    ``embedder`` defaults to :func:`load_embedder` against ``models_dir``;
    tests inject fakes through it. With ``generation_id=None`` every
    active/building generation drains. ``limit`` bounds the batch. Typed
    target refusals resolve before any model work: an unknown or retired
    generation never loads weights.
    """
    result: Dict[str, Any] = {
        "attempted": 0,
        "published": 0,
        "deleted_source_gone": 0,
        "discarded_stale": 0,
        "failed": 0,
        "damaged_skipped": 0,
    }
    with open_search_connection(db_path=db_path, connection_factory=connection_factory) as sc:
        raw = sc.connection
        raw.row_factory = sqlite3.Row
        targets = _resolve_target_generations(raw, generation_id)
        identities = [_generation_identity(raw, target) for target in targets]
        if identities and any(
            not _same_generation_identity(identities[0], identity) for identity in identities[1:]
        ):
            raise VectorLifecycleError(
                "generation-identity-mismatch",
                "one refresh pass selected multiple active/building generations "
                "with different model/runtime/artifact identities; refresh each "
                "generation with its matching embedder",
            )
        batch = _select_eligible_batch(
            raw, targets, limit, _sqlite_moment(datetime.now(timezone.utc))
        )
        result["attempted"] = len(batch)

        rows_by_key: Dict[Tuple[str, str], sqlite3.Row] = {}
        live_keys: List[Tuple[str, str]] = []
        texts: List[str] = []
        source_gone_keys: List[Tuple[str, str]] = []
        for row in batch:
            key = (row["generation_id"], row["document_key"])
            rows_by_key[key] = row
            snapshot = _read_document_snapshot(raw, row)
            if snapshot is None:
                source_gone_keys.append(key)
                continue
            if snapshot["_fts_content_version"] != int(row["content_version"]):
                result["damaged_skipped"] += 1
                continue
            live_keys.append(key)
            texts.append(_DOCUMENT_BUILDERS[row["document_kind"]](snapshot))

        blobs: List[bytes] = []
        if texts:
            # Embedding happens with no transaction open (§9.3 step 3). The
            # model resolves lazily so refusals above never pay for it, and a
            # capability-absent runtime stays a typed refusal: it is not a
            # per-document embedding failure and must not spend dirty rows'
            # retry state on it.
            resolved_embedder = embedder
            if resolved_embedder is None:
                from cli_agent_orchestrator.services.embedding_adapter import load_embedder

                resolved_embedder = load_embedder(models_dir)
            bound_identity = _embedder_identity(resolved_embedder)
            if bound_identity is not None and any(
                not _same_generation_identity(identity, bound_identity) for identity in identities
            ):
                raise VectorLifecycleError(
                    "generation-identity-mismatch",
                    "embedder identity does not match the persisted target generation; "
                    "refusing to publish blobs under the wrong generation",
                )
            try:
                blobs = resolved_embedder.embed(texts)
                if len(blobs) != len(texts):
                    raise VectorLifecycleError(
                        "embedder-contract",
                        f"embedder returned {len(blobs)} blobs for {len(texts)} texts",
                    )
            except Exception as exc:
                for key in live_keys:
                    result["failed"] += _cas_failure(raw, rows_by_key[key], repr(exc))
                return result

        for position, key in enumerate(live_keys):
            content_sha256 = hashlib.sha256(texts[position].encode("utf-8")).hexdigest()
            outcome = _publish_vector(raw, rows_by_key[key], content_sha256, blobs[position])
            result["published"] += outcome["published"]
            result["discarded_stale"] += outcome["discarded_stale"]
            result["deleted_source_gone"] += outcome["deleted_source_gone"]

        for key in source_gone_keys:
            result["deleted_source_gone"] += _delete_dirty_exact(raw, rows_by_key[key])
    return result


def drain_bounded_batch(
    *,
    limit: int = BOUNDED_REFRESH_BATCH,
    embedder: Optional[Any] = None,
    models_dir: Optional[Any] = None,
    db_path: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """The ``_READ``-semantics bounded drain a semantic/hybrid query performs.

    Advances disposable vector-cache rows only (design §9.2); authoritative
    issues/comments/events/links are untouched, the operation is safe to
    repeat, and unavailability must degrade the caller's answer instead of
    changing any filing-relevant state.
    """
    return refresh_generation(
        generation_id=None,
        limit=limit,
        embedder=embedder,
        models_dir=models_dir,
        db_path=db_path,
        connection_factory=connection_factory,
    )


# ---------------------------------------------------------------------------
# Failure marking, activation, status (§13.3 second half, §13.4 diagnostics)
# ---------------------------------------------------------------------------


def mark_generation_failed(
    generation_id: str,
    *,
    failure: str,
    target_engine: Optional[Any] = None,
) -> bool:
    """Move an interrupted/stuck ``building`` generation to ``failed``.

    Recovery is a fresh build (a new generation), never re-activation of the
    failed one: activation requires ``building`` and refuses anything else.
    Active/retired generations cannot be marked failed — their state carries
    provenance the active pointer or receipts may still name.
    """
    raw = (target_engine if target_engine is not None else engine).raw_connection()
    try:
        raw.execute("BEGIN IMMEDIATE")
        try:
            cursor = raw.execute(
                f"UPDATE {schema.VECTOR_GENERATIONS_TABLE}\n"
                "   SET state = 'failed', failure = ?\n"
                " WHERE generation_id = ? AND state = 'building'",
                (failure, generation_id),
            )
            updated = int(cursor.rowcount or 0) == 1
            raw.commit()
        except Exception:
            raw.rollback()
            raise
    finally:
        raw.close()
    return updated


def activate_generation(
    generation_id: str, *, target_engine: Optional[Any] = None
) -> Dict[str, Any]:
    """Prove coverage/encoding/dimension, then switch and retire (§13.3).

    One immediate transaction prunes target-generation vectors whose live
    source no longer exists, proves no generation-scoped dirty rows remain,
    proves every live source document has exactly one current-version vector
    (and no vector exists without its live source at the same version), and
    proves every vector carries the declared float32 encoding and dimension.
    Only then does it switch ``active_vector_generation``, mark this
    generation ``active``, retire the predecessor, and clear obsolete dirty
    work for the retired generation. Interrupted builds never become active
    by presence alone: a refused proof leaves the pointer untouched.
    """
    activated_at = _utcnow_iso()
    raw = (target_engine if target_engine is not None else engine).raw_connection()
    try:
        raw.execute("BEGIN IMMEDIATE")
        try:
            gen_row = raw.execute(
                f"SELECT state, dimensions FROM {schema.VECTOR_GENERATIONS_TABLE}\n"
                " WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()
            if gen_row is None:
                raise VectorLifecycleError(
                    "unknown-generation",
                    f"no {schema.VECTOR_GENERATIONS_TABLE} row named {generation_id!r}",
                )
            state, dimensions = str(gen_row[0]), int(gen_row[1])
            if state != "building":
                raise VectorLifecycleError(
                    "activation-refused-state",
                    f"generation {generation_id!r} is {state!r}, not 'building'; "
                    "only a building generation activates",
                )

            for document_kind in ("issue", "comment"):
                raw.execute(
                    f"DELETE FROM {schema.SEARCH_VECTORS_TABLE}\n"
                    f" WHERE generation_id = ? AND document_kind = '{document_kind}'\n"
                    f"   AND source_id NOT IN "
                    f"(SELECT id FROM {_source_table_for(document_kind)})",
                    (generation_id,),
                )

            remaining_dirty = int(
                raw.execute(
                    f"SELECT COUNT(*) FROM {schema.VECTOR_DIRTY_TABLE} " "WHERE generation_id = ?",
                    (generation_id,),
                ).fetchone()[0]
            )
            if remaining_dirty != 0:
                raise VectorLifecycleError(
                    "activation-refused-dirty",
                    f"generation {generation_id!r} still has {remaining_dirty} dirty "
                    "document(s); refresh to completion before activating",
                )

            for document_kind in ("issue", "comment"):
                source_table = _source_table_for(document_kind)
                fts_table = _fts_table_for(document_kind)
                missing = int(
                    raw.execute(
                        f"SELECT COUNT(*) FROM {source_table} AS s\n"
                        " WHERE NOT EXISTS (\n"
                        f"   SELECT 1 FROM {schema.SEARCH_VECTORS_TABLE} AS v\n"
                        f"   WHERE v.generation_id = ? AND v.document_kind = '{document_kind}'\n"
                        "     AND v.source_id = s.id\n"
                        f"     AND v.content_version = (SELECT f.content_version\n"
                        f"                              FROM {fts_table} AS f"
                        " WHERE f.rowid = s.id))",
                        (generation_id,),
                    ).fetchone()[0]
                )
                if missing != 0:
                    raise VectorLifecycleError(
                        "activation-refused-coverage",
                        f"generation {generation_id!r} lacks current-version vectors "
                        f"for {missing} live {document_kind} document(s)",
                    )
                extra = int(
                    raw.execute(
                        f"SELECT COUNT(*) FROM {schema.SEARCH_VECTORS_TABLE} AS v\n"
                        f" WHERE v.generation_id = ? AND v.document_kind = '{document_kind}'\n"
                        "   AND NOT EXISTS (\n"
                        f"   SELECT 1 FROM {source_table} AS s\n"
                        f"    JOIN {fts_table} AS f ON f.rowid = s.id\n"
                        "    WHERE s.id = v.source_id AND f.content_version = v.content_version)",
                        (generation_id,),
                    ).fetchone()[0]
                )
                if extra != 0:
                    raise VectorLifecycleError(
                        "activation-refused-coverage",
                        f"generation {generation_id!r} holds {extra} {document_kind} "
                        "vector(s) whose live source/version does not match",
                    )

            bad_encoding = int(
                raw.execute(
                    f"SELECT COUNT(*) FROM {schema.SEARCH_VECTORS_TABLE}\n"
                    " WHERE generation_id = ?\n"
                    "   AND (length(embedding) % 4 != 0 OR length(embedding) != ? * 4)",
                    (generation_id, dimensions),
                ).fetchone()[0]
            )
            if bad_encoding != 0:
                raise VectorLifecycleError(
                    "activation-refused-encoding",
                    f"generation {generation_id!r} holds {bad_encoding} vector(s) whose "
                    f"byte length is not float32 x {dimensions} dimensions",
                )

            previous_row = raw.execute(
                "SELECT active_vector_generation FROM "
                f"{schema.SEARCH_META_TABLE} WHERE singleton = 1"
            ).fetchone()
            previous_active = previous_row[0] if previous_row else None

            raw.execute(
                f"UPDATE {schema.VECTOR_GENERATIONS_TABLE}\n"
                "   SET state = 'active', activated_at = ?, failure = NULL\n"
                " WHERE generation_id = ? AND state = 'building'",
                (activated_at, generation_id),
            )
            if previous_active and previous_active != generation_id:
                raw.execute(
                    f"UPDATE {schema.VECTOR_GENERATIONS_TABLE}\n"
                    "   SET state = 'retired'\n"
                    " WHERE generation_id = ? AND state = 'active'",
                    (previous_active,),
                )
                raw.execute(
                    f"DELETE FROM {schema.VECTOR_DIRTY_TABLE} WHERE generation_id = ?",
                    (previous_active,),
                )
            raw.execute(
                f"UPDATE {schema.SEARCH_META_TABLE} SET active_vector_generation = ?\n"
                " WHERE singleton = 1",
                (generation_id,),
            )
            raw.commit()
        except Exception:
            raw.rollback()
            raise
    finally:
        raw.close()
    return {"generation_id": generation_id, "activated_at": activated_at}


def semantic_status(*, target_engine: Optional[Any] = None) -> Dict[str, Any]:
    """Coverage/state diagnostic consumed by ranked retrieval surfaces.

    Reports the installed flag and the derived state — ``unprepared`` (no
    generations), ``building`` (building exists, none active), ``rebuilding``
    (an active generation coexists with a building one; the default hybrid
    surface reports rebuilding rather than mixing models), ``stale`` (active
    generation with eligible dirty work pending), or ``ready`` — plus
    per-generation provenance and counts (§13.4).
    """
    raw = (target_engine if target_engine is not None else engine).raw_connection()
    try:
        installed = raw.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN "
            f"('{schema.SEARCH_META_TABLE}', '{schema.VECTOR_GENERATIONS_TABLE}')"
        ).fetchone()[0]
        if int(installed) < 2:
            return {"installed": False, "state": "unavailable"}
        pointer_row = raw.execute(
            f"SELECT active_vector_generation FROM {schema.SEARCH_META_TABLE} "
            "WHERE singleton = 1"
        ).fetchone()
        active_pointer = pointer_row[0] if pointer_row else None
        generations = [
            {
                "generation_id": row[0],
                "state": row[1],
                "model_id": row[2],
                "model_revision": row[3],
                "dimensions": row[4],
                "distance_metric": row[5],
                "normalized": bool(row[6]),
                "document_schema_version": row[7],
                "created_at": row[8],
                "activated_at": row[9],
                "failure": row[10],
                "vectors": int(row[11]),
                "dirty": int(row[12]),
                "dirty_failed": int(row[13]),
            }
            for row in raw.execute(
                f"SELECT g.generation_id, g.state, g.model_id, g.model_revision,\n"
                f"       g.dimensions, g.distance_metric, g.normalized,\n"
                f"       g.document_schema_version, g.created_at, g.activated_at,\n"
                f"       g.failure,\n"
                f"       (SELECT COUNT(*) FROM {schema.SEARCH_VECTORS_TABLE} v\n"
                f"         WHERE v.generation_id = g.generation_id),\n"
                f"       (SELECT COUNT(*) FROM {schema.VECTOR_DIRTY_TABLE} d\n"
                f"         WHERE d.generation_id = g.generation_id),\n"
                f"       (SELECT COUNT(*) FROM {schema.VECTOR_DIRTY_TABLE} d\n"
                f"         WHERE d.generation_id = g.generation_id\n"
                f"           AND d.last_error IS NOT NULL)\n"
                f"FROM {schema.VECTOR_GENERATIONS_TABLE} g\n"
                f"ORDER BY g.created_at, g.generation_id"
            ).fetchall()
        ]
        has_active = any(g["state"] == "active" for g in generations)
        has_building = any(g["state"] == "building" for g in generations)
        if not generations:
            state = "unprepared"
        elif has_building and not has_active:
            state = "building"
        elif has_building:
            state = "rebuilding"
        elif has_active:
            pending = int(
                raw.execute(
                    f"SELECT COUNT(*) FROM {schema.VECTOR_DIRTY_TABLE} d\n"
                    f"JOIN {schema.VECTOR_GENERATIONS_TABLE} g"
                    " ON g.generation_id = d.generation_id\n"
                    " WHERE g.state = 'active'\n"
                    "   AND (d.next_attempt_at IS NULL OR d.next_attempt_at <= "
                    "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
                ).fetchone()[0]
            )
            state = "stale" if pending > 0 else "ready"
        else:
            state = "unavailable"
        return {
            "installed": True,
            "state": state,
            "active_generation": active_pointer if has_active else None,
            "generations": generations,
        }
    finally:
        raw.close()
