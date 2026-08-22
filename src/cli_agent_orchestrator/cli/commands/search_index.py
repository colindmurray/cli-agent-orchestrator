"""`cao issue search-index` — the tracker search index's operator surface.

V1 carries the model half of the search index: the explicit prepare command
(the only path that ever downloads model weights) and the capability status
report. Lexical rebuild/refresh/integrity verbs join this same group from the
search-schema lane.

Like the rest of `cao issue`, commands call the service directly — preparing
a model must work exactly when nothing else is running — and every command
takes `--json` so an agent gets a parseable answer from the same code path a
human reads.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from typing import Any, Optional

import click

from cli_agent_orchestrator.services import embedding_adapter as adapter


def _fail(exc: adapter.EmbeddingCapabilityError, as_json: bool = False) -> None:
    """Report a typed refusal, parseable under --json, and exit non-zero."""
    if as_json:
        click.echo(jsonlib.dumps({"ok": False, "reason": exc.reason, "message": exc.message}))
    else:
        click.echo(f"error [{exc.reason}]: {exc.message}", err=True)
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
    re-run: an already-verified store is returned unchanged, and a corrupt
    metadata file is rewritten from the verified artifact.
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
        record = adapter.prepare_model(models_dir)
    except adapter.EmbeddingCapabilityError as exc:
        _fail(exc, as_json)
        return
    payload: dict[str, Any] = dict(record)
    payload["prepare"] = {
        "ok": True,
        "idempotent": before == record if before is not None else False,
    }

    def render(rec: dict[str, Any]) -> None:
        prep = rec["prepare"]
        click.echo(
            f"prepared {rec['model_id']}@{rec['model_revision'][:12]} "
            f"({rec['dimensions']}d {rec['element_type']}, {rec['distance_metric']}, "
            f"artifact sha256 {rec['artifact_sha256'][:12]}…)"
        )
        if prep["idempotent"]:
            click.echo("already prepared and verified; metadata unchanged")

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
