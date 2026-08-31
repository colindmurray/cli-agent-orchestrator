"""CLI surface of `cao issue search-index`: prepare/status/maintenance contracts.

The service-level tests prove the lifecycle machinery; these prove the
operator surface wiring — including that a corrupt metadata file is repaired
by `prepare` rather than blocking it (the refusal's own remedy must be
reachable through the command that names it), and that every maintenance verb
answers with the state an operator acts on, not a bare exit code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.cli.commands import issue as issue_cli
from cli_agent_orchestrator.cli.commands.search_index import (
    index_status,
    integrity_check,
    model_prepare,
    model_status,
    rebuild,
    refresh,
)
from cli_agent_orchestrator.clients.database import Base, _migrate_tracker_search_projection
from cli_agent_orchestrator.services import embedding_adapter as adapter
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services import search_index_maintenance as maintenance
from cli_agent_orchestrator.services import tracker_ranked_search as ranked
from cli_agent_orchestrator.services import vector_lifecycle as vlc


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Bind every CLI invocation to a scratch store, never operator state.

    The commands call the service directly, and three module globals hold the
    pooled engine/search session — all three must point at the same file, or
    `model prepare` writes a generation row into whatever store the process
    inherited while the assertions below read an empty lookalike.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/search-index-cli.db")
    Base.metadata.create_all(bind=engine)
    _migrate_tracker_search_projection(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(maintenance, "engine", engine)
    monkeypatch.setattr(vlc, "engine", engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)
    monkeypatch.setattr(ranked, "SessionLocal", sessions)
    monkeypatch.setattr(issue_cli, "ensure_tracker_schema", lambda: None)
    yield engine
    engine.dispose()


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _fake_digest() -> str:
    import hashlib

    digest = hashlib.sha256()
    for rel, data in sorted(
        [("config.json", b'{"model_type": "bert"}'), ("model.safetensors", b"measured-weights")]
    ):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest()


def _seed_corpus():
    """Two issues and a comment through the real tracker writers."""
    tracker.create_project(name="CAO System", project_id="cao-system", issue_prefix="cond")
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0001",
        title="deploy pipeline bounces on dry run",
        force=True,
    )
    tracker.create_issue(
        project_id="cao-system",
        key="cond-0002",
        title="widget color tuning guide",
        force=True,
    )
    tracker.add_comment("cond-0001", body="the deploy bounce reproduced twice")


def _prepare(runner, tmp_path):
    """Prepare through the CLI with the download faked out."""
    from unittest.mock import patch

    def fake_downloader(*, repo_id, revision, cache_dir, ignore_patterns=None):
        snapshot = Path(cache_dir) / "snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_bytes(b'{"model_type": "bert"}')
        (snapshot / "model.safetensors").write_bytes(b"measured-weights")
        return str(snapshot)

    models_dir = tmp_path / "models"
    with patch.object(adapter, "_default_snapshot_downloader", fake_downloader):
        result = runner.invoke(
            model_prepare, ["--models-dir", str(models_dir), "--json"]
        )
    assert result.exit_code == 0, result.output
    return models_dir, json.loads(result.output)


@pytest.fixture()
def fake_runtime(monkeypatch: pytest.MonkeyPatch):
    versions = {"sentence-transformers": "6.0.0", "torch": "2.13.0", "transformers": "5.15.1"}
    monkeypatch.setattr(
        adapter, "_read_dist_version", lambda name: versions.get(name), raising=True
    )
    # Re-pin the target artifact to the deterministic fake content so the CLI
    # exercises the full prepare/verify path without the real 182 MB artifact.
    monkeypatch.setattr(adapter, "MODEL_ARTIFACT_SHA256", _fake_digest())
    # Keep the CLI's own default dir resolution out of the operator state.
    monkeypatch.setattr(adapter, "default_models_dir", lambda: Path("/cli-test-models"))
    return versions


def test_status_unprepared_is_exit_zero_json_state(runner, tmp_path):
    result = runner.invoke(model_status, ["--models-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["state"] == "unprepared"
    assert payload["signals"]["metadata_present"] is False


def test_prepare_repairs_corrupt_metadata_through_the_cli(runner, tmp_path, fake_runtime):
    """The P1 regression: the refusal's remedy must work via the CLI itself."""
    from unittest.mock import patch

    def fake_downloader(*, repo_id, revision, cache_dir, ignore_patterns=None):
        snapshot = Path(cache_dir) / "snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_bytes(b'{"model_type": "bert"}')
        (snapshot / "model.safetensors").write_bytes(b"measured-weights")
        return str(snapshot)

    (tmp_path / "generation-metadata.json").write_text("{corrupt")
    with patch.object(adapter, "_default_snapshot_downloader", fake_downloader):
        result = runner.invoke(model_prepare, ["--models-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["prepare"]["ok"] is True
    assert payload["artifact_sha256"] == _fake_digest()
    assert json.loads((tmp_path / "generation-metadata.json").read_text())["schema"] == (
        adapter.METADATA_SCHEMA
    )


def test_prepare_reports_idempotent_on_verified_rerun(runner, tmp_path, fake_runtime):
    from unittest.mock import patch

    def fake_downloader(*, repo_id, revision, cache_dir, ignore_patterns=None):
        snapshot = Path(cache_dir) / "snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        for rel, data in sorted(
            [("config.json", b'{"model_type": "bert"}'), ("model.safetensors", b"measured-weights")]
        ):
            (snapshot / rel).write_bytes(data)
        return str(snapshot)

    with patch.object(adapter, "_default_snapshot_downloader", fake_downloader):
        first = runner.invoke(model_prepare, ["--models-dir", str(tmp_path), "--json"])
        second = runner.invoke(model_prepare, ["--models-dir", str(tmp_path), "--json"])
    assert first.exit_code == 0 and second.exit_code == 0
    assert json.loads(first.output)["prepare"]["idempotent"] is False
    assert json.loads(second.output)["prepare"]["idempotent"] is True


def test_prepare_failure_is_typed_json_when_json_requested(
    runner, tmp_path, fake_runtime, monkeypatch
):
    from unittest.mock import patch

    def poisoned_downloader(*, repo_id, revision, cache_dir, ignore_patterns=None):
        snapshot = Path(cache_dir) / "snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model.safetensors").write_bytes(b"tampered")
        return str(snapshot)

    with patch.object(adapter, "_default_snapshot_downloader", poisoned_downloader):
        result = runner.invoke(model_prepare, ["--models-dir", str(tmp_path), "--json"])
    # A digest mismatch is the deterministic, hermetic failure here; the
    # typed refusal must arrive as JSON with its classification intact.
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["reason"] == "digest-mismatch"
    assert payload["message"]


def test_status_human_renderer_lists_state(runner, tmp_path):
    result = runner.invoke(model_status, ["--models-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert result.output.startswith("state: unprepared")


# ---------------------------------------------------------------------------
# model prepare reports the generation it bound (cond-0770)
# ---------------------------------------------------------------------------


def test_prepare_reports_the_generation_it_bound(runner, tmp_path, fake_runtime, isolated_store):
    """Prepare is the operator's entry point into the lifecycle: the command
    must say which generation it bound and how much work it queued."""
    _seed_corpus()
    models_dir, payload = _prepare(runner, tmp_path)

    assert payload["prepare"]["ok"] is True
    assert payload["generation"]["action"] == "created"
    assert payload["generation"]["generation_state"] == "building"
    assert payload["generation"]["enqueued_documents"] == 3


def test_prepare_twice_reuses_one_generation(runner, tmp_path, fake_runtime, isolated_store):
    """Acceptance: repeated prepare never mints a series of building rows."""
    _seed_corpus()
    models_dir, _prepared = _prepare(runner, tmp_path)
    models_dir, second = _prepare(runner, tmp_path)

    assert second["generation"]["action"] == "reused"
    rows = isolated_store.raw_connection()
    try:
        found = rows.execute("SELECT COUNT(*) FROM tracker_vector_generations").fetchone()[0]
    finally:
        rows.close()
    assert found == 1


# ---------------------------------------------------------------------------
# status / refresh / rebuild / integrity-check
# ---------------------------------------------------------------------------


def test_status_on_a_fresh_install_names_the_next_actions(runner):
    result = runner.invoke(index_status, ["--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["capability"]["state"] in {"unprepared", "runtime-missing"}
    assert payload["semantic"]["state"] == "unprepared"
    states = {entry["state"] for entry in payload["next_actions"]}
    assert "unprepared" in states


class StubEmbedder:
    """A 384-deterministic stand-in for the real sentence-transformers model.

    The CLI has no embedder injection seam — it resolves weights from disk by
    design — so the tests that drive a refresh patch the adapter's loader and
    hand back vectors that satisfy the prepared generation's declared width,
    encoding, and normalization contract.
    """

    dimensions = 384

    def __init__(self, fail_on=()):
        self.fail_on = tuple(fail_on)

    def _vector(self, text_value):
        import re as _re

        import numpy as _np

        vec = _np.zeros(self.dimensions, dtype=_np.float64)
        for token in _re.findall(r"[a-z0-9_]+", str(text_value).lower()):
            index = _CLI_VOCAB.get(token)
            if index is not None:
                vec[index] += 1.0
        norm = _np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        return _np.asarray(vec, dtype="<f4")

    def embed(self, texts, *args, **kwargs):  # noqa: ARG002 - signature parity
        blobs = []
        for text in texts:
            if any(marker in text for marker in self.fail_on):
                raise RuntimeError(f"embedding refused for {str(text)[:32]!r}")
            blobs.append(self._vector(text).tobytes())
        return blobs


_CLI_VOCAB = {"deploy": 0, "pipeline": 1, "bounce": 2, "dry": 3, "run": 4, "widget": 5, "color": 6, "tuning": 7}


@pytest.fixture()
def stub_embedder(monkeypatch):
    """Route every embedder resolution — refresh and query time — to the stub."""
    embedder = StubEmbedder()
    monkeypatch.setattr(adapter, "load_embedder", lambda models_dir=None: embedder)
    return embedder


def test_status_after_the_journey_reports_ready_and_no_pending_action(
    runner, tmp_path, fake_runtime, stub_embedder
):
    _seed_corpus()
    models_dir, prepared = _prepare(runner, tmp_path)
    assert runner.invoke(refresh, ["--all", "--models-dir", str(models_dir), "--json"]).exit_code == 0

    result = runner.invoke(index_status, ["--models-dir", str(models_dir), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["capability"]["state"] == "prepared"
    assert payload["semantic"]["state"] == "ready"
    assert payload["active_generation"] == prepared["generation"]["generation_id"]
    assert payload["lexical"]["issues"] == 2 and payload["lexical"]["comments"] == 1
    assert payload["next_actions"] == []

    human = runner.invoke(index_status, ["--models-dir", str(models_dir)])
    assert human.exit_code == 0
    assert "capability: prepared" in human.output
    assert "semantic: ready" in human.output


def test_refresh_reports_activation_and_the_active_generation(
    runner, tmp_path, fake_runtime, stub_embedder
):
    _seed_corpus()
    models_dir, prepared = _prepare(runner, tmp_path)

    result = runner.invoke(refresh, ["--all", "--models-dir", str(models_dir), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["refresh"]["published"] == 3
    assert payload["activations"] == [
        {"generation_id": prepared["generation"]["generation_id"], "activated": True}
    ]
    assert payload["active_generation"] == prepared["generation"]["generation_id"]

    # Re-running the same verb is a no-op that still reports state: an active
    # generation is not re-offered and nothing re-embeds.
    human = runner.invoke(refresh, ["--all", "--models-dir", str(models_dir)])
    assert human.exit_code == 0, human.output
    assert "refresh (all): 0 published" in human.output
    assert f"active generation: {prepared['generation']['generation_id']}" in human.output


def test_refresh_reports_a_refused_activation_instead_of_dying(
    runner, tmp_path, fake_runtime, stub_embedder, isolated_store
):
    """An incomplete build reaches the operator as a typed report with the
    repair, not as a traceback and not as a silent success."""
    _seed_corpus()
    models_dir, _prepared = _prepare(runner, tmp_path)

    # Damage the derived queue the way a pruned row does: the comment will
    # never be embedded, so the queue drains while coverage cannot hold.
    raw = isolated_store.raw_connection()
    try:
        raw.execute("DELETE FROM tracker_vector_dirty WHERE document_kind = 'comment'")
        raw.commit()
    finally:
        raw.close()

    # The bounded drain does the embedding work and never offers activation —
    # that is what `--all` is for.
    bounded = runner.invoke(refresh, ["--models-dir", str(models_dir)])
    assert bounded.exit_code == 0, bounded.output
    assert "refresh (bounded): 2 published" in bounded.output
    assert "activated generation" not in bounded.output
    assert "active generation: None" in bounded.output

    refused = runner.invoke(refresh, ["--all", "--models-dir", str(models_dir)])
    assert refused.exit_code == 0, refused.output
    assert "activation refused for" in refused.output
    assert "activation-refused-coverage" in refused.output

    result = runner.invoke(refresh, ["--all", "--models-dir", str(models_dir), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["activations"][0]["activated"] is False
    assert payload["active_generation"] is None, "a refused build never goes live"


def test_refresh_surfaces_a_search_runtime_refusal_with_the_install_action(
    runner, tmp_path, fake_runtime, stub_embedder, monkeypatch
):
    from cli_agent_orchestrator.services.search_engine_factory import SearchEngineError

    def runtime_missing(*args, **kwargs):
        raise SearchEngineError("runtime-missing", "the sqlite-vec package is not installed")

    _seed_corpus()
    models_dir, _prepared = _prepare(runner, tmp_path)
    monkeypatch.setattr(vlc, "open_search_connection", runtime_missing)

    result = runner.invoke(refresh, ["--all", "--models-dir", str(models_dir), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["reason"] == "runtime-missing"
    assert "[search]" in payload["action"]


def test_rebuild_requires_exactly_one_scope(runner):
    for flags in ([], ["--lexical", "--vectors"]):
        result = runner.invoke(rebuild, flags + ["--json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["reason"] == "invalid-scope"
        assert "scope" in payload["action"]


def test_rebuild_lexical_then_vectors_through_the_cli(
    runner, tmp_path, fake_runtime, stub_embedder
):
    _seed_corpus()
    models_dir, _prepared = _prepare(runner, tmp_path)
    assert runner.invoke(refresh, ["--all", "--models-dir", str(models_dir), "--json"]).exit_code == 0

    lexical = runner.invoke(rebuild, ["--lexical", "--models-dir", str(models_dir), "--json"])
    assert lexical.exit_code == 0, lexical.output
    payload = json.loads(lexical.output)
    assert payload["lexical"]["documents_rebuilt"] == 3
    assert payload["vectors"] is None

    vectors = runner.invoke(rebuild, ["--vectors", "--models-dir", str(models_dir), "--json"])
    assert vectors.exit_code == 0, vectors.output
    payload = json.loads(vectors.output)
    assert payload["vectors"]["activation"]["activated"] is True

    human = runner.invoke(rebuild, ["--vectors", "--models-dir", str(models_dir)])
    assert "rebuilt and activated" in human.output


def test_rebuild_vectors_without_a_prepared_model_names_the_prepare_command(runner):
    result = runner.invoke(rebuild, ["--vectors", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["reason"] == "unprepared"
    assert "model prepare" in payload["action"]


def test_integrity_check_reports_the_read_only_report(runner, tmp_path, fake_runtime):
    _seed_corpus()
    models_dir, _prepared = _prepare(runner, tmp_path)

    result = runner.invoke(integrity_check, ["--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["fts_internal"] == {"issues": "ok", "comments": "ok"}
    assert payload["coverage"]["issue"]["documents"] == 2
    assert payload["vector_dirty"]["total"] == 3

    human = runner.invoke(integrity_check, [])
    assert "fts issues: ok" in human.output
    assert "coverage issue:" in human.output
