"""The production journey cond-0770 files as missing, driven through the CLI.

``model prepare`` → ``refresh --all`` → a non-null active generation →
``cao issue search --mode hybrid`` returning a semantic lane contribution,
every step through the same Click commands an operator types. The maintenance
primitives existed before this lane; nothing called them in sequence, which is
why an installation stayed lexical-only forever.

This module is the named mutation-proof target: silencing the activation call
in :mod:`services.search_index_maintenance` must turn
:func:`test_prepare_refresh_then_explicit_hybrid_search_serves_the_semantic_lane`
red on the active-generation assertion, because the generation is then built
and left ``building`` forever — the permanently degraded state this lane
repairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.cli.commands import issue as issue_cli
from cli_agent_orchestrator.cli.commands.issue import issue_search
from cli_agent_orchestrator.cli.commands.search_index import (
    index_status,
    model_prepare,
    refresh,
)
from cli_agent_orchestrator.clients.database import Base, _migrate_tracker_search_projection
from cli_agent_orchestrator.services import embedding_adapter as adapter
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services import search_index_maintenance as maintenance
from cli_agent_orchestrator.services import tracker_ranked_search as ranked
from cli_agent_orchestrator.services import vector_lifecycle as vlc

QUERY = "deploy pipeline bounce"
DEPLOY_KEY = "cond-0001"
WIDGET_KEY = "cond-0002"


class JourneyEmbedder:
    """The 384-wide stand-in for the pinned model, shared by both legs.

    The refresh leg resolves weights through ``embedding_adapter.load_embedder``
    and the query leg through ``tracker_ranked_search._query_embedder``; both
    seams are patched so one consistent vector space serves the whole journey.
    Token-frequency vectors over a disjoint basis make the ranking a statement
    about the text: the deploy query shares its whole vocabulary with
    ``cond-0001`` and nothing with ``cond-0002``.
    """

    dimensions = 384

    def _vector(self, text):
        import re as _re

        import numpy as _np

        vec = _np.zeros(self.dimensions, dtype=_np.float64)
        for token in _re.findall(r"[a-z0-9_]+", str(text).lower()):
            index = _VOCAB.get(token)
            if index is not None:
                vec[index] += 1.0
        norm = _np.linalg.norm(vec)
        return _np.asarray(vec / norm if norm else vec, dtype="<f4")

    def embed(self, texts, *args, **kwargs):  # noqa: ARG002 - signature parity
        return [self._vector(text).tobytes() for text in texts]


_VOCAB = {
    "deploy": 0,
    "pipeline": 1,
    "bounce": 2,
    "dry": 3,
    "run": 4,
    "widget": 5,
    "color": 6,
    "tuning": 7,
}


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def journey(tmp_path, monkeypatch):
    """A scratch store, a fake runtime, and one embedder across both seams.

    Every global that holds the pooled engine or the search session is bound
    to the same file-backed store, so the CLI's direct-service calls write
    where the assertions read and nothing reaches operator state.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/journey.db")
    Base.metadata.create_all(bind=engine)
    _migrate_tracker_search_projection(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(maintenance, "engine", engine)
    monkeypatch.setattr(vlc, "engine", engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    monkeypatch.setattr(ranked, "SessionLocal", sessions)
    monkeypatch.setattr(issue_cli, "ensure_tracker_schema", lambda: None)

    versions = {"sentence-transformers": "6.0.0", "torch": "2.13.0", "transformers": "5.15.1"}
    monkeypatch.setattr(adapter, "_read_dist_version", lambda name: versions.get(name))
    monkeypatch.setattr(adapter, "MODEL_ARTIFACT_SHA256", _fake_digest())
    embedder = JourneyEmbedder()
    monkeypatch.setattr(adapter, "load_embedder", lambda models_dir=None: embedder)
    monkeypatch.setattr(ranked, "_query_embedder", lambda models_dir=None: embedder)
    try:
        yield {"engine": engine, "embedder": embedder, "tmp_path": tmp_path}
    finally:
        engine.dispose()


def _fake_digest() -> str:
    import hashlib

    digest = hashlib.sha256()
    for rel, data in sorted(
        (("config.json", b'{"model_type": "bert"}'), ("model.safetensors", b"measured-weights"))
    ):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest()


def _fake_downloader(*, repo_id, revision, cache_dir, ignore_patterns=None):
    snapshot = Path(cache_dir) / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_bytes(b'{"model_type": "bert"}')
    (snapshot / "model.safetensors").write_bytes(b"measured-weights")
    return str(snapshot)


def _seed_corpus():
    """The corpus, written through the real tracker writers."""
    tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
    tracker.create_issue(
        project_id="cao-system",
        key=DEPLOY_KEY,
        title="deploy pipeline bounces on dry run",
        body="the deploy pipeline bounces every dry run",
        force=True,
    )
    tracker.create_issue(
        project_id="cao-system",
        key=WIDGET_KEY,
        title="widget color tuning guide",
        body="tune the widget color profile",
        force=True,
    )
    tracker.add_comment(DEPLOY_KEY, body="the deploy bounce reproduced twice")


def _prepare(runner, tmp_path):
    """Step 1: explicit model preparation through the real command."""
    models_dir = tmp_path / "models"
    with patch.object(adapter, "_default_snapshot_downloader", _fake_downloader):
        prepared = runner.invoke(model_prepare, ["--models-dir", str(models_dir), "--json"])
    assert prepared.exit_code == 0, prepared.output
    return models_dir, json.loads(prepared.output)


def test_prepare_refresh_then_explicit_hybrid_search_serves_the_semantic_lane(runner, journey):
    """The acceptance journey, end to end over public commands."""
    _seed_corpus()
    tmp_path = journey["tmp_path"]
    models_dir, prepared = _prepare(runner, tmp_path)
    generation_id = prepared["generation"]["generation_id"]

    # Step 2: the full drain completes the build and activates it.
    refreshed = runner.invoke(refresh, ["--all", "--models-dir", str(models_dir), "--json"])
    assert refreshed.exit_code == 0, refreshed.output
    payload = json.loads(refreshed.output)
    assert payload["refresh"]["published"] == 3, "two issues plus one comment"
    assert payload["activations"] == [{"generation_id": generation_id, "activated": True}]
    assert payload["active_generation"] == generation_id

    # Step 3: status reports the activated generation as the served one.
    reported = runner.invoke(index_status, ["--models-dir", str(models_dir), "--json"])
    assert reported.exit_code == 0, reported.output
    status_payload = json.loads(reported.output)
    assert status_payload["semantic"]["state"] == "ready"
    assert status_payload["active_generation"] == generation_id
    assert status_payload["next_actions"] == []

    # Step 4: an explicit hybrid search actually reaches the semantic lane.
    searched = runner.invoke(
        issue_search,
        [QUERY, "--tracker-project", "cao-system", "--mode", "hybrid", "--json"],
    )
    assert searched.exit_code == 0, searched.output
    hits = json.loads(searched.output)

    assert hits["mode_effective"] == "hybrid", hits["degradation"]
    assert hits["diagnostics"]["semantic"]["served"] is True
    lanes = {lane["lane"] for row in hits["results"] for lane in row["contributing_lanes"]}
    assert any(lane.startswith("semantic") for lane in lanes), sorted(lanes)
    assert [row["issue"]["key"] for row in hits["results"]][:1] == [DEPLOY_KEY], [
        (row["issue"]["key"], row["rank_score"]) for row in hits["results"]
    ]

    human = runner.invoke(
        issue_search, [QUERY, "--tracker-project", "cao-system", "--mode", "hybrid"]
    )
    assert human.exit_code == 0, human.output
    assert "mode hybrid" in human.output
    assert "semantic" in human.output, human.output


def test_the_same_journey_degrades_visibly_when_the_generation_is_never_activated(runner, journey):
    """The contrast case: without activation the same search reports exactly
    what an operator needs to know instead of silently serving lexical only."""
    _seed_corpus()
    tmp_path = journey["tmp_path"]
    models_dir, _prepared = _prepare(runner, tmp_path)

    # Build the vectors but never offer activation.
    bounded = runner.invoke(refresh, ["--models-dir", str(models_dir), "--json"])
    assert bounded.exit_code == 0, bounded.output
    assert json.loads(bounded.output)["active_generation"] is None

    searched = runner.invoke(
        issue_search,
        [QUERY, "--tracker-project", "cao-system", "--mode", "hybrid", "--json"],
    )
    assert searched.exit_code == 0, searched.output
    hits = json.loads(searched.output)
    assert hits["mode_effective"] == "lexical", hits["degradation"]
    assert hits["diagnostics"]["semantic"]["served"] is False
    assert hits["degradation"]["reasons"], "the degradation must be legible"
