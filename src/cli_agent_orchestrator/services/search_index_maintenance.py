"""Shared search-index maintenance orchestrator (design §9.2, §12, §13).

The M1.2/M2.2 primitives own single steps — one generation insert, one
compare-and-set refresh pass, one activation proof, one lexical rebuild, one
integrity read — and until now nothing called them in sequence, so an
installation stayed lexical-only forever. THIS module is the caller: it
composes those primitives into the four operator journeys the accepted CLI
and REST surfaces expose, and every surface below is a thin adapter over it,
so the CLI and the API cannot drift into two different lifecycles.

Journeys and their write scope:

``prepare_index``
    The explicit model half (the only path that ever downloads weights)
    followed by reuse-or-create of a matching ``building`` generation.
    Downloading is the operator's decision; creating the generation is not,
    because a prepared model with no generation is exactly the permanently
    degraded state cond-0770 files. Repeated preparation reuses the
    existing generation, so re-running the command never mints a series of
    ``building`` rows.

``refresh_index``
    Drains the durable outbox through the §9.3 compare-and-set, optionally
    resetting the backoff of rows that failed earlier. With ``all=True`` a
    ``building`` generation whose queue reached zero is offered for
    activation: the coverage/encoding proof inside the activation
    transaction is what makes an incomplete build refuse, never this
    caller's optimism, and a refused activation leaves the generation
    ``building`` so the next refresh can finish it.

``rebuild_index``
    The §13.2 lexical repair verb and/or a fresh §13.3 vector generation,
    built to completion and activated.

``index_status`` / ``integrity_check``
    Read-only §13.4 diagnostics. ``index_status`` also names the next
    operator action for every degraded state it can observe, because a
    typed refusal that names no remedy just moves the search elsewhere.

Authority boundary. Every write here lands in the derived schema — FTS
documents, the vector outbox, vector rows, generation rows, and the search
metadata singleton. No function in this module inserts, updates, or deletes
``tracker_issues``, ``tracker_issue_comments``, ``tracker_issue_links``, or
``tracker_issue_events``, and none loads model weights except
``prepare_index``'s explicit model half. ``test_search_index_maintenance``
pins that boundary by diffing the authoritative tables around each journey.

Connection seams. Relational maintenance (generation reuse, retry reset,
activation, status, integrity, lexical rebuild) opens the same injectable
raw connection the lifecycle module uses. Refresh is indexing work and goes
through :func:`search_engine_factory.open_search_connection`, so sqlite-vec
is loaded only on a dedicated connection and never on the authoritative
pooled engine (§7.2). Both default to the real store and accept injections,
which is how tests exercise the whole journey without model weights.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

from cli_agent_orchestrator.clients import tracker_search_schema as schema
from cli_agent_orchestrator.clients.database import engine
from cli_agent_orchestrator.services import vector_lifecycle as lifecycle

__all__ = [
    "SearchIndexMaintenanceError",
    "REBUILD_SCOPES",
    "DEFAULT_RETRY_LIMIT",
    "prepare_index",
    "ensure_generation",
    "refresh_index",
    "rebuild_index",
    "index_status",
    "integrity_check",
]

#: Valid ``rebuild`` scopes, in the order ``--all`` applies them.
REBUILD_SCOPES = ("lexical", "vectors", "all")

#: Upper bound on how many failed outbox rows one retry pass resets. The
#: reset is cheap, but an unbounded ``--retry-failed`` on a corpus that has
#: been failing for days would silently turn into a full-corpus re-embed with
#: no operator-visible bound; the count is reported either way.
DEFAULT_RETRY_LIMIT = 10_000

#: The remedy for an absent search runtime, shared by status and by the
#: refresh/refusal paths, so both surfaces name the same install command.
RUNTIME_ACTION = (
    "install the optional search runtime: "
    "`uv pip install 'cli-agent-orchestrator[search]'` (or pip equivalent)"
)

_PREPARE_ACTION = (
    "run `cao issue search-index model prepare` to download and verify the " "pinned local model"
)
_REBUILD_ACTION = "re-run `cao issue search-index rebuild --vectors` for a fresh build"
_SCHEMA_ACTION = "run any `cao issue` command once so the tracker schema migration runs"

#: The operator remedy for every lifecycle refusal this orchestrator lets
#: reach its boundary. Translation happens HERE, once, so the CLI and the API
#: cannot grow two different answers to the same observed condition; a reason
#: missing from the map still translates, with the generic finish-the-build
#: remedy rather than no remedy at all.
_LIFECYCLE_ACTIONS = {
    "unprepared": _PREPARE_ACTION,
    "metadata-incomplete": _PREPARE_ACTION,
    "metadata-incompatible": _PREPARE_ACTION,
    "schema-missing": _SCHEMA_ACTION,
    "embedder-contract": _REBUILD_ACTION,
    "generation-identity-mismatch": _REBUILD_ACTION,
    "embedder-identity-invalid": _REBUILD_ACTION,
}
_DEFAULT_LIFECYCLE_ACTION = (
    "run `cao issue search-index refresh --all` to finish or repair the build"
)


def _as_maintenance_error(exc: "lifecycle.VectorLifecycleError") -> "SearchIndexMaintenanceError":
    """Carry a lifecycle refusal over this module's typed boundary."""
    return SearchIndexMaintenanceError(
        exc.reason,
        str(exc),
        action=_LIFECYCLE_ACTIONS.get(exc.reason, _DEFAULT_LIFECYCLE_ACTION),
    )


class SearchIndexMaintenanceError(RuntimeError):
    """Typed maintenance refusal; ``reason`` names the observed condition.

    ``action`` carries the operator remedy when one exists, so the CLI and
    the API can both surface "what was observed" and "what to do" without
    parsing message text.
    """

    def __init__(self, reason: str, message: str, *, action: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.action = action


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw(target_engine: Optional[Any] = None):
    return (target_engine if target_engine is not None else engine).raw_connection()


def _store_db_path(target_engine: Optional[Any] = None) -> Optional[str]:
    """The filesystem path of the store this module's engine names, if any.

    The refresh leg needs a SECOND sqlite connection (the dedicated sqlite-vec
    factory, §7.2), so it has to know where the store lives. Deriving that
    from the same engine the relational legs use — rather than falling back to
    a global constant — is what keeps one injection seam meaningful: a test or
    an alternate store that replaces the engine gets a search connection to
    the same file, and an in-memory store reports the fact instead of opening
    a lookalike database that could never see its rows.
    """
    target = target_engine if target_engine is not None else engine
    url = getattr(target, "url", None)
    database = getattr(url, "database", None)
    if not database or database == ":memory:":
        return None
    return str(database)


#: Derived tables every maintenance journey presumes. Absence is an
#: installation fact (the tracker schema never ran), never a live hold.
_DERIVED_TABLES = (
    schema.SEARCH_META_TABLE,
    schema.VECTOR_GENERATIONS_TABLE,
    schema.VECTOR_DIRTY_TABLE,
    schema.SEARCH_VECTORS_TABLE,
    schema.ISSUE_FTS_TABLE,
    schema.COMMENT_FTS_TABLE,
)


def _require_derived_schema(raw: Any) -> None:
    """Refuse with the operator remedy when the derived schema is absent.

    ``clients.database`` installs this projection as part of the tracker
    schema itself, so reaching this refusal means the store predates the
    search migration or was never initialized — not that a build is
    incomplete.
    """
    found = raw.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN "
        f"({', '.join(repr(name) for name in _DERIVED_TABLES)})"
    ).fetchone()[0]
    if int(found) < len(_DERIVED_TABLES):
        raise SearchIndexMaintenanceError(
            "schema-missing",
            "the derived tracker search schema is not installed on this store",
            action="run any `cao issue` command once so the tracker schema migration runs",
        )


# ---------------------------------------------------------------------------
# Generation identity and reuse (§13.1, §13.3)
# ---------------------------------------------------------------------------


#: Columns that together name ONE generation identity. Two rows agreeing on
#: all of them are the same model/runtime/artifact/document build, so the
#: second is redundant work rather than a distinct generation.
_IDENTITY_COLUMNS = lifecycle.GENERATION_IDENTITY_COLUMNS


def _matching_generation(raw: Any, identity: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The live generation already built for this exact model/artifact identity.

    An unfinished ``building`` row wins over a finished ``active`` one:
    resuming the interrupted build is the cheaper repair, and activation will
    retire the old active generation when the new one proves complete. A
    ``failed`` or ``retired`` row never matches — a failed build is repaired
    by a fresh generation, not by re-offering the failed one.
    """
    predicates = " AND ".join(f"{column} = ?" for column in _IDENTITY_COLUMNS)
    row = raw.execute(
        f"SELECT generation_id, state FROM {schema.VECTOR_GENERATIONS_TABLE}\n"
        f" WHERE {predicates} AND state IN ('building', 'active')\n"
        " ORDER BY CASE state WHEN 'building' THEN 0 ELSE 1 END,"
        "       created_at, generation_id\n"
        " LIMIT 1",
        tuple(identity[column] for column in _IDENTITY_COLUMNS),
    ).fetchone()
    if row is None:
        return None
    return {"generation_id": row[0], "state": row[1]}


def ensure_generation(
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    models_dir: Optional[Any] = None,
    target_engine: Optional[Any] = None,
) -> Dict[str, Any]:
    """Reuse the generation matching the prepared model, or create one.

    Reads prepared-model metadata only — no download, no model load — so it
    is safe to call from any maintenance path. The refusal when metadata is
    absent names the one command that can produce it.

    ``create_generation`` re-projects the metadata it is handed, so it
    receives the RAW record here rather than :func:`identity`: the projection
    exists to compare generation rows, and feeding a projection back through
    it would look up ``runtime_versions`` on a record that no longer carries
    one.
    """
    if metadata is None:
        from cli_agent_orchestrator.services.embedding_adapter import (
            default_models_dir,
            read_metadata,
        )

        metadata = read_metadata(models_dir if models_dir is not None else default_models_dir())
    if not isinstance(metadata, Mapping):
        raise SearchIndexMaintenanceError(
            "unprepared",
            "no prepared-model metadata is recorded for this installation",
            action=_PREPARE_ACTION,
        )
    try:
        identity = lifecycle.generation_record_from_metadata(metadata)
    except lifecycle.VectorLifecycleError as exc:
        raise _as_maintenance_error(exc) from exc

    raw = _raw(target_engine)
    try:
        # The identity check and the generation insert/initial outbox enqueue
        # must share one write lock. Otherwise two concurrent prepare calls can
        # both observe no match and publish duplicate full queues.
        raw.execute("BEGIN IMMEDIATE")
        try:
            _require_derived_schema(raw)
            existing = _matching_generation(raw, identity)
            if existing is not None:
                raw.commit()
                return {
                    "action": "reused",
                    "generation_id": existing["generation_id"],
                    "generation_state": existing["state"],
                    "enqueued_documents": 0,
                    **identity,
                }
            created = lifecycle._create_generation_in_transaction(raw, identity)
            raw.commit()
        except Exception:
            raw.rollback()
            raise
    except lifecycle.VectorLifecycleError as exc:
        raise _as_maintenance_error(exc) from exc
    finally:
        raw.close()
    return {
        "action": "created",
        "generation_id": created["generation_id"],
        "generation_state": created["state"],
        "enqueued_documents": created["enqueued_documents"],
        **identity,
    }


def prepare_index(
    *,
    models_dir: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    target_engine: Optional[Any] = None,
    snapshot_downloader: Optional[Callable[..., str]] = None,
) -> Dict[str, Any]:
    """The explicit prepare journey: verify the artifact, then bind a generation.

    ``embedding_adapter.prepare_model`` is the only network-touching step and
    is already idempotent; the generation half is idempotent through
    :func:`ensure_generation`. Together they make repeated operator
    preparation a no-op rather than a growing pile of ``building`` rows.
    """
    from cli_agent_orchestrator.services.embedding_adapter import prepare_model

    kwargs: Dict[str, Any] = {}
    if snapshot_downloader is not None:
        kwargs["snapshot_downloader"] = snapshot_downloader
    record = prepare_model(models_dir, **kwargs) if metadata is None else dict(metadata)
    prepared = ensure_generation(metadata=record, target_engine=target_engine)
    return {
        "model": dict(record),
        "generation": prepared,
        "action": prepared["action"],
        "generation_id": prepared["generation_id"],
        "generation_state": prepared["generation_state"],
        "enqueued_documents": prepared["enqueued_documents"],
    }


# ---------------------------------------------------------------------------
# Refresh and retry (§9.2, §9.3, §13.3 activation)
# ---------------------------------------------------------------------------


def _reset_failed_retries(raw: Any, *, limit: Optional[int]) -> Dict[str, Any]:
    """Make failed outbox rows eligible again, up to an explicit bound.

    Only rows that actually recorded a failure are touched, and only their
    ``next_attempt_at`` moves: ``last_error`` stays until a successful
    publish deletes the row, so the diagnostic survives the retry. Rows whose
    backoff has already expired are left alone — resetting them would only
    inflate the reported count. The bound is applied by rowid because SQLite
    carries no ``UPDATE ... LIMIT`` outside a non-default build.
    """
    bound = DEFAULT_RETRY_LIMIT if limit is None else limit
    if bound <= 0:
        return {"reset": 0, "remaining_failed": _count_failed(raw), "bounded": True}
    targets = raw.execute(
        f"SELECT rowid FROM {schema.VECTOR_DIRTY_TABLE}\n"
        " WHERE last_error IS NOT NULL AND next_attempt_at IS NOT NULL\n"
        " ORDER BY enqueued_at, document_key\n"
        f" LIMIT {int(bound)}"
    ).fetchall()
    reset = 0
    for (rowid,) in targets:
        cursor = raw.execute(
            f"UPDATE {schema.VECTOR_DIRTY_TABLE} SET next_attempt_at = NULL WHERE rowid = ?",
            (rowid,),
        )
        reset += int(cursor.rowcount or 0)
    return {"reset": reset, "remaining_failed": _count_failed(raw), "bounded": reset >= bound}


def _count_failed(raw: Any) -> int:
    return int(
        raw.execute(
            f"SELECT COUNT(*) FROM {schema.VECTOR_DIRTY_TABLE} WHERE last_error IS NOT NULL"
        ).fetchone()[0]
    )


def _building_generations(raw: Any) -> List[str]:
    return [
        row[0]
        for row in raw.execute(
            f"SELECT generation_id FROM {schema.VECTOR_GENERATIONS_TABLE} "
            "WHERE state = 'building' ORDER BY created_at, generation_id"
        ).fetchall()
    ]


def _remaining_dirty(raw: Any, generation_id: str) -> int:
    return int(
        raw.execute(
            f"SELECT COUNT(*) FROM {schema.VECTOR_DIRTY_TABLE} WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()[0]
    )


def _refresh_leg(**kwargs: Any) -> Dict[str, Any]:
    """Run one refresh pass, typing every refusal at the boundary.

    An absent sqlite-vec runtime is an installation fact an operator can
    repair, not a crash: it arrives here as a typed refusal carrying the
    install action, so the CLI and the API both answer with a remedy instead
    of a traceback. Lifecycle refusals are translated the same way.
    """
    from cli_agent_orchestrator.services.search_engine_factory import SearchEngineError

    try:
        return lifecycle.refresh_generation(**kwargs)
    except lifecycle.VectorLifecycleError as exc:
        raise _as_maintenance_error(exc) from exc
    except SearchEngineError as exc:
        action = RUNTIME_ACTION if exc.reason == "runtime-missing" else None
        raise SearchIndexMaintenanceError(exc.reason, exc.message, action=action) from exc


def refresh_index(
    *,
    all: bool = False,  # noqa: A002 - the operator flag is spelled --all
    retry_failed: bool = False,
    limit: Optional[int] = None,
    embedder: Optional[Any] = None,
    models_dir: Optional[Any] = None,
    db_path: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    target_engine: Optional[Any] = None,
) -> Dict[str, Any]:
    """Drain the vector outbox; with ``all``, complete and activate a build.

    ``all=False`` is the bounded shape a query-time caller uses: one batch,
    derived rows only, no activation attempt. ``all=True`` is the explicit
    maintenance shape: the whole queue for the prepared embedder's generation,
    then every ``building`` generation whose queue reached zero is offered for
    activation. During a model migration this leaves the old active generation
    untouched while its replacement builds. A refused activation is reported
    per generation with its typed reason and leaves that generation ``building``
    — the refusal is the coverage proof doing its job, not a failure of this
    command, so the remedy is to run the refresh again rather than to start over.
    """
    retry: Dict[str, Any] = {"reset": 0, "remaining_failed": None, "bounded": False}
    if retry_failed:
        raw = _raw(target_engine)
        try:
            raw.execute("BEGIN IMMEDIATE")
            try:
                retry = _reset_failed_retries(raw, limit=limit)
                raw.commit()
            except Exception:
                raw.rollback()
                raise
        finally:
            raw.close()

    refreshed = _refresh_leg(
        generation_id=None,
        limit=None if all else limit,
        embedder=embedder,
        models_dir=models_dir,
        db_path=db_path if db_path is not None else _store_db_path(target_engine),
        connection_factory=connection_factory,
    )

    activations: List[Dict[str, Any]] = []
    if all:
        raw = _raw(target_engine)
        try:
            candidates = [
                generation_id
                for generation_id in _building_generations(raw)
                if _remaining_dirty(raw, generation_id) == 0
            ]
        finally:
            raw.close()
        for generation_id in candidates:
            entry: Dict[str, Any] = {"generation_id": generation_id, "activated": False}
            try:
                lifecycle.activate_generation(generation_id, target_engine=target_engine)
                entry["activated"] = True
            except lifecycle.VectorLifecycleError as exc:
                entry["refused_reason"] = exc.reason
                entry["detail"] = str(exc)
            activations.append(entry)

    return {
        "scope": "all" if all else "bounded",
        "retry_failed": retry,
        "refresh": refreshed,
        "activations": activations,
        "active_generation": _active_pointer(target_engine),
    }


def _active_pointer(target_engine: Optional[Any] = None) -> Optional[str]:
    raw = _raw(target_engine)
    try:
        _require_derived_schema(raw)
        row = raw.execute(
            f"SELECT active_vector_generation FROM {schema.SEARCH_META_TABLE} WHERE singleton = 1"
        ).fetchone()
    finally:
        raw.close()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Rebuild (§13.2, §13.3)
# ---------------------------------------------------------------------------


def _rebuild_lexical(*, target_engine: Optional[Any]) -> Dict[str, Any]:
    raw = _raw(target_engine)
    try:
        raw.execute("BEGIN IMMEDIATE")
        try:
            _require_derived_schema(raw)
            result = schema.rebuild_lexical(raw, rebuilt_at=_utcnow_iso())
            raw.commit()
        except Exception:
            raw.rollback()
            raise
    finally:
        raw.close()
    return result


def _rebuild_vectors(
    *,
    metadata: Optional[Mapping[str, Any]],
    models_dir: Optional[Any],
    embedder: Optional[Any],
    db_path: Optional[Any],
    connection_factory: Optional[Callable[[], Any]],
    target_engine: Optional[Any],
) -> Dict[str, Any]:
    """One complete build-and-activate attempt over a fresh generation.

    Two failure shapes leave different records behind. A document that cannot
    embed is ordinary §9.3 retry state: the refresh pass returns counts, the
    leftover queue refuses the activation, and the generation stays
    ``building`` so ``refresh --all --retry-failed`` finishes the very build
    that was interrupted. A refresh leg that RAISES — the search runtime
    vanished mid-build, the store would not open — is not retry state, so the
    generation is marked ``failed`` with the observed reason instead of being
    left looking resumable; the repair is the same command again, which always
    creates a fresh generation.
    """
    try:
        created = lifecycle.create_generation(
            metadata=metadata, models_dir=models_dir, target_engine=target_engine
        )
    except lifecycle.VectorLifecycleError as exc:
        raise _as_maintenance_error(exc) from exc
    generation_id = created["generation_id"]
    try:
        refreshed = _refresh_leg(
            generation_id=generation_id,
            embedder=embedder,
            models_dir=models_dir,
            db_path=db_path if db_path is not None else _store_db_path(target_engine),
            connection_factory=connection_factory,
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        try:
            lifecycle.mark_generation_failed(
                generation_id, failure=failure, target_engine=target_engine
            )
        except Exception:  # noqa: BLE001 - marking must not mask the original refusal
            pass
        if isinstance(exc, SearchIndexMaintenanceError):
            exc.action = exc.action or _REBUILD_ACTION
            raise
        if isinstance(exc, lifecycle.VectorLifecycleError):
            raise _as_maintenance_error(exc) from exc
        raise SearchIndexMaintenanceError(
            "refresh-failed",
            f"vector rebuild for generation {generation_id!r} failed during refresh: {failure}",
            action=_REBUILD_ACTION,
        ) from exc

    raw = _raw(target_engine)
    try:
        leftover = _remaining_dirty(raw, generation_id)
    finally:
        raw.close()

    activation: Dict[str, Any] = {"generation_id": generation_id, "activated": False}
    if leftover:
        activation["refused_reason"] = "activation-refused-dirty"
        activation["detail"] = f"{leftover} document(s) remain unembedded after the refresh pass"
    else:
        try:
            lifecycle.activate_generation(generation_id, target_engine=target_engine)
            activation["activated"] = True
        except lifecycle.VectorLifecycleError as exc:
            activation["refused_reason"] = exc.reason
            activation["detail"] = str(exc)
    return {
        "generation": created,
        "refresh": refreshed,
        "activation": activation,
        "active_generation": _active_pointer(target_engine),
    }


def rebuild_index(
    *,
    scope: str = "all",
    metadata: Optional[Mapping[str, Any]] = None,
    models_dir: Optional[Any] = None,
    embedder: Optional[Any] = None,
    db_path: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    target_engine: Optional[Any] = None,
) -> Dict[str, Any]:
    """The repair verb: rebuild the lexical projection, the vectors, or both.

    Both legs rebuild derived state only. ``vectors`` needs prepared-model
    metadata and refuses with a typed action when it is absent; it never
    downloads, so preparing the model stays a separate operator decision.
    """
    if scope not in REBUILD_SCOPES:
        raise SearchIndexMaintenanceError(
            "invalid-scope",
            f"rebuild scope {scope!r} is not one of {', '.join(REBUILD_SCOPES)}",
            action="pass --lexical, --vectors, or --all",
        )
    result: Dict[str, Any] = {"scope": scope, "lexical": None, "vectors": None}
    if scope in ("lexical", "all"):
        result["lexical"] = _rebuild_lexical(target_engine=target_engine)
    if scope in ("vectors", "all"):
        result["vectors"] = _rebuild_vectors(
            metadata=metadata,
            models_dir=models_dir,
            embedder=embedder,
            db_path=db_path,
            connection_factory=connection_factory,
            target_engine=target_engine,
        )
    return result


# ---------------------------------------------------------------------------
# Read-only diagnostics (§13.4)
# ---------------------------------------------------------------------------

#: Typed operator action for every degraded state ``index_status`` can name.
#: A refusal without a reachable remedy just relocates the search; each entry
#: here is a command the same CLI exposes.
_ACTIONS = {
    "runtime-missing": RUNTIME_ACTION,
    "version-mismatch": (
        "reinstall the pinned search runtime so the observed versions match "
        "the prepared generation"
    ),
    "unprepared": "run `cao issue search-index model prepare`",
    "unavailable": "run `cao issue` once to install the tracker search schema",
    "building": "run `cao issue search-index refresh --all` to complete the build",
    "stale": "run `cao issue search-index refresh --all` to catch up the changed documents",
}


def _next_actions(capability_state: str, semantic_state: str) -> List[Dict[str, str]]:
    actions: List[Dict[str, str]] = []
    for state in (capability_state, semantic_state):
        action = _ACTIONS.get(state)
        if action and not any(entry["action"] == action for entry in actions):
            actions.append({"state": state, "action": action})
    return actions


def index_status(
    *,
    models_dir: Optional[Any] = None,
    run_probe: bool = False,
    target_engine: Optional[Any] = None,
    engine_describer: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Read-only capability + index report, with the next operator action.

    ``run_probe`` stays off by default so the answer never loads model
    weights: status is a read surface and must stay cheap enough to poll.
    """
    from cli_agent_orchestrator.services import embedding_adapter as adapter
    from cli_agent_orchestrator.services.search_engine_factory import describe_search_engine

    report = adapter.diagnose_embedding(models_dir, run_probe=run_probe)
    semantic = lifecycle.semantic_status(target_engine=target_engine)

    lexical: Dict[str, Any] = {"installed": False}
    raw = _raw(target_engine)
    try:
        installed = raw.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN "
            f"('{schema.SEARCH_META_TABLE}', '{schema.ISSUE_FTS_TABLE}', "
            f"'{schema.COMMENT_FTS_TABLE}')"
        ).fetchone()[0]
        if int(installed) >= 3:
            meta = raw.execute(
                "SELECT schema_version, document_schema_version, content_clock, rebuilt_at "
                f"FROM {schema.SEARCH_META_TABLE} WHERE singleton = 1"
            ).fetchone()
            lexical = {
                "installed": True,
                "schema_version": int(meta[0]) if meta else None,
                "document_schema_version": int(meta[1]) if meta else None,
                "content_clock": int(meta[2]) if meta else None,
                "rebuilt_at": meta[3] if meta else None,
                "issues": int(
                    raw.execute(f"SELECT COUNT(*) FROM {schema.ISSUE_FTS_TABLE}").fetchone()[0]
                ),
                "comments": int(
                    raw.execute(f"SELECT COUNT(*) FROM {schema.COMMENT_FTS_TABLE}").fetchone()[0]
                ),
                "vectors": int(
                    raw.execute(f"SELECT COUNT(*) FROM {schema.SEARCH_VECTORS_TABLE}").fetchone()[0]
                ),
                "dirty": int(
                    raw.execute(f"SELECT COUNT(*) FROM {schema.VECTOR_DIRTY_TABLE}").fetchone()[0]
                ),
                "dirty_failed": _count_failed(raw),
            }
    finally:
        raw.close()

    try:
        if engine_describer is not None:
            engine_report: Dict[str, Any] = engine_describer()
        else:
            engine_report = describe_search_engine()
    except Exception as exc:  # noqa: BLE001 - an unobservable engine is the finding
        engine_report = {"observed": False, "detail": f"{type(exc).__name__}: {exc}"}

    return {
        "capability": report.as_dict(),
        "engine": engine_report,
        "lexical": lexical,
        "semantic": semantic,
        "active_generation": semantic.get("active_generation"),
        "next_actions": _next_actions(report.state.value, str(semantic.get("state"))),
    }


def integrity_check(*, target_engine: Optional[Any] = None) -> Dict[str, Any]:
    """The read-only §13.4 report. Repairs belong exclusively to ``rebuild``."""
    raw = _raw(target_engine)
    try:
        _require_derived_schema(raw)
        report = schema.integrity_report(raw)
    finally:
        raw.close()
    report["semantic"] = lifecycle.semantic_status(target_engine=target_engine)
    return report
