"""`cao issue search-index` — the tracker search index's operator surface.

Every accepted verb lives here: the explicit model `prepare` (the only path
that ever downloads model weights), the capability and index `status` report,
`refresh`, `rebuild`, and `integrity-check`. The commands are thin adapters
over :mod:`services.search_index_maintenance`, which owns the actual
lifecycle — the REST routes call the same orchestrator, so the two surfaces
cannot drift into two different index lifecycles.

Like the rest of `cao issue`, commands call the service directly — preparing
a model must work exactly when nothing else is running — and every command
takes `--json` so an agent gets a parseable answer from the same code path a
human reads. Degraded states are reported with the operator action that
repairs them, never with a bare refusal.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from typing import Any, Optional

import click

from cli_agent_orchestrator.services import embedding_adapter as adapter
from cli_agent_orchestrator.services import search_index_maintenance as maintenance


def _fail(exc: Exception, as_json: bool = False) -> None:
    """Report a typed refusal, parseable under --json, and exit non-zero.

    Every refusal this surface raises carries a stable ``reason`` and, when
    one exists, the operator ``action`` that repairs the observed state — the
    remedy travels with the refusal rather than living in a man page.
    """
    reason = getattr(exc, "reason", type(exc).__name__)
    message = getattr(exc, "message", None) or str(exc)
    action = getattr(exc, "action", None)
    if as_json:
        payload: dict[str, Any] = {"ok": False, "reason": reason, "message": message}
        if action:
            payload["action"] = action
        click.echo(jsonlib.dumps(payload))
    else:
        click.echo(f"error [{reason}]: {message}", err=True)
        if action:
            click.echo(f"action: {action}", err=True)
    raise SystemExit(1)


@click.group(name="search-index")
def search_index():
    """Manage the tracker search index (lexical documents and vectors)."""


@search_index.group(name="model")
def model():
    """Prepare and diagnose the local embedding model generation."""


@model.command(name="prepare")
@click.option("--models-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True)
def model_prepare(models_dir: Optional[Path], as_json: bool):
    """Explicitly download, digest-verify, and record the pinned model.

    The ONLY command that touches the network for model weights. Safe to
    re-run: an already-verified store is returned unchanged, a corrupt
    metadata file is rewritten from the verified artifact, and a generation
    for this exact model is reused rather than minted again.
    """
    # Best-effort read for the idempotency flag ONLY: a corrupt or absent
    # file must never block prepare itself — repairing that state is exactly
    # what this command is for.
    try:
        before = adapter.read_metadata(
            models_dir if models_dir is not None else adapter.default_models_dir()
        )
    except adapter.EmbeddingCapabilityError:
        before = None
    try:
        outcome = maintenance.prepare_index(models_dir=models_dir)
    except (adapter.EmbeddingCapabilityError, maintenance.SearchIndexMaintenanceError) as exc:
        _fail(exc, as_json)
        return
    record = outcome["model"]
    payload: dict[str, Any] = dict(record)
    payload["prepare"] = {
        "ok": True,
        "idempotent": before == record if before is not None else False,
    }
    payload["generation"] = outcome["generation"]

    def render(rec: dict[str, Any]) -> None:
        prep = rec["prepare"]
        click.echo(
            f"prepared {rec['model_id']}@{rec['model_revision'][:12]} "
            f"({rec['dimensions']}d {rec['element_type']}, {rec['distance_metric']}, "
            f"artifact sha256 {rec['artifact_sha256'][:12]}…)"
        )
        if prep["idempotent"]:
            click.echo("already prepared and verified; metadata unchanged")
        generation = rec["generation"]
        verb = "reused" if generation["action"] == "reused" else "created"
        click.echo(
            f"{verb} {generation['generation_state']} generation "
            f"{generation['generation_id']} "
            f"({generation['enqueued_documents']} document(s) queued)"
        )

    _emit(payload, as_json, render)


@model.command(name="status")
@click.option("--models-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option(
    "--no-probe",
    is_flag=True,
    help="report from metadata/runtime/engine observation only (no model load)",
)
@click.option("--json", "as_json", is_flag=True)
def model_status(models_dir: Optional[Path], no_probe: bool, as_json: bool):
    """Report the embedding capability state with positive signals.

    States: prepared, unprepared, runtime-missing, version-mismatch,
    probe-failed. A not-prepared answer is a valid report, not an error —
    the exit code stays 0 so scripts can parse every state.
    """
    report = adapter.diagnose_embedding(models_dir, run_probe=not no_probe)

    def render(rep: dict[str, Any]) -> None:
        click.echo(f"state: {rep['state']}")
        signals = rep.get("signals", {})
        for key in (
            "model_id",
            "artifact_sha256_observed",
            "runtime_versions_observed",
            "engine",
            "probe",
            "detail",
        ):
            if key in signals:
                click.echo(f"{key}: {jsonlib.dumps(signals[key], sort_keys=True)}")

    _emit(report.as_dict(), as_json, render)


def _emit(payload: Any, as_json: bool, renderer=None) -> None:
    if as_json or renderer is None:
        click.echo(jsonlib.dumps(payload, indent=2, sort_keys=True))
    else:
        renderer(payload)


# --------------------------------------------------------------------------
# index-level maintenance verbs
# --------------------------------------------------------------------------


@search_index.command(name="status")
@click.option("--models-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True)
def index_status(models_dir: Optional[Path], as_json: bool):
    """Report the whole search index: capability, engine, lexical, semantic.

    Read-only and cheap enough to poll — it never loads model weights. A
    degraded state is a valid report with exit code 0, and every degraded
    state it can observe names the command that repairs it under
    ``next_actions``.
    """
    try:
        report = maintenance.index_status(models_dir=models_dir)
    except maintenance.SearchIndexMaintenanceError as exc:
        _fail(exc, as_json)
        return

    def render(rep: dict[str, Any]) -> None:
        click.echo(f"capability: {rep['capability']['state']}")
        click.echo(f"engine observed: {rep['engine'].get('observed', False)}")
        lexical = rep["lexical"]
        if lexical.get("installed"):
            click.echo(
                f"lexical: {lexical['issues']} issue + {lexical['comments']} comment "
                f"document(s), clock {lexical['content_clock']}"
            )
        else:
            click.echo("lexical: search schema not installed")
        click.echo(f"semantic: {rep['semantic'].get('state')}")
        click.echo(f"active generation: {rep['active_generation']}")
        for entry in rep["next_actions"]:
            click.echo(f"next action [{entry['state']}]: {entry['action']}")

    _emit(report, as_json, render)


@search_index.command(name="refresh")
@click.option(
    "--all", "drain_all", is_flag=True, help="drain the whole queue, not one bounded batch"
)
@click.option(
    "--retry-failed",
    is_flag=True,
    help="reset the backoff of documents whose embedding failed, then refresh",
)
@click.option("--limit", type=int, default=None, help="bound the batch explicitly")
@click.option("--models-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True)
def refresh(
    drain_all: bool, retry_failed: bool, limit: int, models_dir: Optional[Path], as_json: bool
):
    """Embed queued documents for the prepared model's generation.

    Without --all this drains one bounded batch — the same derived work a
    semantic query performs — and never activates anything. With --all it
    drains completely and offers a finished building generation for
    activation. During a model migration the old active generation remains the
    serving fallback while the replacement builds; the activation proof inside
    that transaction keeps an incomplete build from ever going live.
    """
    try:
        result = maintenance.refresh_index(
            all=drain_all,
            retry_failed=retry_failed,
            limit=limit,
            models_dir=models_dir,
        )
    except (adapter.EmbeddingCapabilityError, maintenance.SearchIndexMaintenanceError) as exc:
        _fail(exc, as_json)
        return

    def render(rep: dict[str, Any]) -> None:
        retry = rep["retry_failed"]
        if retry_failed:
            click.echo(
                f"retry: reset {retry['reset']} failed document(s), "
                f"{retry['remaining_failed']} still failing"
            )
        counts = rep["refresh"]
        click.echo(
            f"refresh ({rep['scope']}): {counts['published']} published, "
            f"{counts['failed']} failed, {counts['discarded_stale']} stale, "
            f"{counts['deleted_source_gone']} source gone, "
            f"{counts['damaged_skipped']} damaged skipped"
        )
        for activation in rep["activations"]:
            if activation["activated"]:
                click.echo(f"activated generation {activation['generation_id']}")
            else:
                click.echo(
                    f"activation refused for {activation['generation_id']}: "
                    f"{activation.get('refused_reason')} — {activation.get('detail')}"
                )
        click.echo(f"active generation: {rep['active_generation']}")

    _emit(result, as_json, render)


@search_index.command(name="rebuild")
@click.option("--lexical", "scope_lexical", is_flag=True, help="rebuild the FTS documents")
@click.option(
    "--vectors", "scope_vectors", is_flag=True, help="build and activate a fresh vector generation"
)
@click.option("--all", "scope_all", is_flag=True, help="rebuild both")
@click.option("--models-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True)
def rebuild(
    scope_lexical: bool,
    scope_vectors: bool,
    scope_all: bool,
    models_dir: Optional[Path],
    as_json: bool,
):
    """Repair the derived index. Never rewrites an issue, comment, link, or event.

    Exactly one of --lexical, --vectors, --all is required. Lexical repair
    repopulates the FTS documents from the authoritative rows with fresh
    content versions and requeues every live document, so no pre-rebuild
    vector can be served against rebuilt text. Vector repair builds a fresh
    generation and activates it only after the coverage proof passes.
    """
    chosen = [
        flag
        for flag, given in (
            ("lexical", scope_lexical),
            ("vectors", scope_vectors),
            ("all", scope_all),
        )
        if given
    ]
    if len(chosen) != 1:
        _fail(
            maintenance.SearchIndexMaintenanceError(
                "invalid-scope",
                "exactly one of --lexical, --vectors, --all is required"
                + (
                    f"; got {', '.join('--' + name for name in chosen)}" if chosen else "; got none"
                ),
                action="pass exactly one scope flag",
            ),
            as_json,
        )
        return
    try:
        result = maintenance.rebuild_index(scope=chosen[0], models_dir=models_dir)
    except (adapter.EmbeddingCapabilityError, maintenance.SearchIndexMaintenanceError) as exc:
        _fail(exc, as_json)
        return

    def render(rep: dict[str, Any]) -> None:
        if rep["lexical"] is not None:
            counts = rep["lexical"]
            click.echo(
                f"lexical rebuilt: {counts['documents_rebuilt']} document(s) "
                f"({counts['issues']} issue, {counts['comments']} comment)"
            )
        vectors = rep["vectors"]
        if vectors is not None:
            activation = vectors["activation"]
            if activation["activated"]:
                click.echo(f"vector generation {activation['generation_id']} rebuilt and activated")
            else:
                click.echo(
                    f"vector generation {activation['generation_id']} rebuilt but not activated: "
                    f"{activation.get('refused_reason')} — {activation.get('detail')}"
                )

    _emit(result, as_json, render)


@search_index.command(name="integrity-check")
@click.option("--json", "as_json", is_flag=True)
def integrity_check(as_json: bool):
    """Read-only §13.4 report on the derived index. Repairs belong to rebuild.

    Reports FTS internal integrity, exact source-to-FTS coverage, duplicate
    and orphan document keys, dirty/failed/ready vector counts, stale
    vectors, active generation provenance, per-project coverage, and the last
    recorded failures. It repairs nothing.
    """
    try:
        report = maintenance.integrity_check()
    except maintenance.SearchIndexMaintenanceError as exc:
        _fail(exc, as_json)
        return

    def render(rep: dict[str, Any]) -> None:
        for kind, status_text in rep["fts_internal"].items():
            click.echo(f"fts {kind}: {status_text}")
        for kind, counts in rep["coverage"].items():
            click.echo(
                f"coverage {kind}: {counts['documents']}/{counts['source_rows']} "
                f"({counts['missing_documents']} missing, {counts['orphan_documents']} orphan)"
            )
        click.echo(
            f"vectors: {rep['vector_stale']['total_vectors']} total, "
            f"{rep['vector_stale']['stale_vectors']} stale; "
            f"dirty {rep['vector_dirty']['total']} "
            f"({rep['vector_dirty']['failed']} failed, {rep['vector_dirty']['ready']} ready)"
        )
        click.echo(f"semantic: {rep['semantic'].get('state')}")

    _emit(report, as_json, render)
