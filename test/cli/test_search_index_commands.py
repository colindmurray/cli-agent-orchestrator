"""CLI surface of `cao issue search-index model`: prepare/status contracts.

The service-level tests prove the capability machinery; these prove the
operator surface wiring — including that a corrupt metadata file is repaired
by `prepare` rather than blocking it (the refusal's own remedy must be
reachable through the command that names it).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli_agent_orchestrator.cli.commands.search_index import model_prepare, model_status
from cli_agent_orchestrator.services import embedding_adapter as adapter


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
