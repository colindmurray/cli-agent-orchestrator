"""Embedding adapter: explicit prepare, idempotency, typed diagnostics, embed
contract validation, and offline-by-construction loading.

Every heavy dependency is behind injection seams, so the whole capability
state machine is exercised here without torch — the real-runtime path is
proven separately by the offline drill and the dogfood probe.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cli_agent_orchestrator.services import embedding_adapter as adapter
from cli_agent_orchestrator.services.search_engine_factory import (
    PINNED_VEC_VERSION,
    describe_search_engine,
    open_search_connection,
)

_FULL_VERSIONS = {
    "sentence-transformers": "6.0.0",
    "torch": "2.13.0",
    "transformers": "5.15.1",
}


def _ok_versions(name: str) -> str:
    return _FULL_VERSIONS[name]


_FAKE_SNAPSHOT_FILES = [
    ("config.json", b'{"model_type": "bert"}'),
    ("model.safetensors", b"measured-weights"),
]


def _fake_expected_digest() -> str:
    """Digest the deterministic fake snapshot produces under dir_sha256."""
    import hashlib

    digest = hashlib.sha256()
    for rel, data in sorted(_FAKE_SNAPSHOT_FILES):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest()


def _materialize_snapshot(cache_dir: Path) -> Path:
    snapshot = cache_dir / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for rel, data in _FAKE_SNAPSHOT_FILES:
        (snapshot / rel).write_bytes(data)
    return snapshot


class FakeDownloader:
    """Records calls and materializes a deterministic fake snapshot."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, repo_id, revision, cache_dir, ignore_patterns=None) -> str:
        self.calls.append({"repo_id": repo_id, "revision": revision})
        return str(_materialize_snapshot(Path(cache_dir)))


def _diagnose(models_dir, **kwargs):
    kwargs.setdefault("expected_artifact_sha256", _fake_expected_digest())
    return adapter.diagnose_embedding(models_dir, **kwargs)


def _prepare(tmp_path: Path, downloader=None, **kwargs):
    kwargs.setdefault("expected_artifact_sha256", _fake_expected_digest())
    return adapter.prepare_model(
        tmp_path,
        snapshot_downloader=downloader or FakeDownloader(),
        dist_versions=_ok_versions,
        **kwargs,
    )


def _fake_engine_describer(observed: str | None = None):
    return {
        "runtime_present": True,
        "extension_api_available": True,
        "vec_version_observed": observed or PINNED_VEC_VERSION,
        "vec_version_pinned": PINNED_VEC_VERSION,
    }


def _unit_vector(dim: int = 384) -> np.ndarray:
    vec = np.full((dim,), 1.0 / dim**0.5, dtype=np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


class FakeEncoder:
    """Stands in for a loaded SentenceTransformer."""

    def __init__(self, vectors=None, fail_seq_length: bool = False) -> None:
        self.vectors = vectors
        self.max_seq_length: int | None = None
        self.encode_calls: list[dict] = []

    def encode(self, texts, *, batch_size=32, convert_to_numpy=False, show_progress_bar=False):
        self.encode_calls.append({"texts": list(texts), "batch_size": batch_size})
        if self.vectors is not None:
            return self.vectors
        stacked = np.stack([_unit_vector() for _ in texts])
        return stacked


# --- prepare ----------------------------------------------------------------


def test_prepare_writes_generation_ready_metadata(tmp_path):
    downloader = FakeDownloader()
    record = _prepare(tmp_path, downloader)

    assert downloader.calls == [{"repo_id": adapter.MODEL_ID, "revision": adapter.MODEL_REVISION}]
    # Every §9.4 record required before semantic enable:
    assert record["model_id"] == adapter.MODEL_ID
    assert record["model_revision"] == adapter.MODEL_REVISION
    assert record["runtime_id"] == "sentence-transformers"
    assert record["runtime_versions"] == _FULL_VERSIONS
    assert record["dimensions"] == 384
    assert record["element_type"] == "float32"
    assert record["normalized"] is True
    assert record["distance_metric"] == "cosine"
    assert record["max_seq_length"] == 256
    assert record["document_schema_version_id"] == 1
    assert record["document_schema_version_name"] == "m0.3-issue-doc-v0"
    # The fake artifact's digest is whatever the deterministic algorithm says;
    # the record must carry exactly that observed digest:
    assert record["artifact_sha256"] == adapter.dir_sha256(Path(record["snapshot_path"]))
    assert record["artifact_bytes"] == sum(len(data) for _, data in _FAKE_SNAPSHOT_FILES)
    assert record["vec_version_pinned"] == "v0.1.9"
    assert record["schema"] == adapter.METADATA_SCHEMA
    # Persisted on disk:
    on_disk = json.loads((tmp_path / "generation-metadata.json").read_text())
    assert on_disk == record


def test_prepare_verifies_digest_against_recorded_value(tmp_path):
    with pytest.raises(adapter.ArtifactDigestMismatch) as excinfo:
        _prepare(tmp_path, expected_artifact_sha256="0" * 64)
    assert excinfo.value.reason == "digest-mismatch"
    assert excinfo.value.observed != excinfo.value.expected
    # A failed verify must not leave generation-ready metadata behind:
    assert not (tmp_path / "generation-metadata.json").exists()


def test_prepare_requires_runtime_to_record_honest_metadata(tmp_path):
    with pytest.raises(adapter.EmbeddingCapabilityError) as excinfo:
        adapter.prepare_model(
            tmp_path,
            snapshot_downloader=FakeDownloader(),
            dist_versions=lambda name: None if name == "torch" else _FULL_VERSIONS[name],
        )
    assert excinfo.value.reason == "runtime-missing"
    assert "torch" in excinfo.value.message


def test_prepare_is_idempotent_downloader_runs_once(tmp_path):
    downloader = FakeDownloader()
    first = _prepare(tmp_path, downloader)
    before = (tmp_path / "generation-metadata.json").read_bytes()

    second = _prepare(tmp_path, downloader)

    assert len(downloader.calls) == 1, "verified re-prepare must not re-download"
    assert second == first
    assert second["prepared_at"] == first["prepared_at"], "no-op must not move prepared_at"
    assert (tmp_path / "generation-metadata.json").read_bytes() == before


def test_prepare_repairs_corrupt_metadata_from_verified_artifact(tmp_path):
    _prepare(tmp_path)
    (tmp_path / "generation-metadata.json").write_text("{corrupt")

    repaired = _prepare(tmp_path)

    assert repaired["schema"] == adapter.METADATA_SCHEMA
    assert json.loads((tmp_path / "generation-metadata.json").read_text()) == repaired


# --- diagnostics --------------------------------------------------------------


def test_diagnose_unprepared_on_empty_dir(tmp_path):
    report = _diagnose(tmp_path, run_probe=False)
    assert report.state is adapter.DiagnosticState.UNPREPARED
    assert report.signals["metadata_present"] is False


def test_diagnose_unprepared_on_corrupt_metadata(tmp_path):
    _prepare(tmp_path)
    (tmp_path / "generation-metadata.json").write_text("{corrupt")
    report = _diagnose(tmp_path, run_probe=False)
    assert report.state is adapter.DiagnosticState.UNPREPARED
    assert "cannot be parsed" in report.signals["detail"]


def test_diagnose_unprepared_when_artifact_missing(tmp_path):
    record = _prepare(tmp_path)
    # Simulate a lost artifact: point the metadata at a deleted snapshot.
    record["snapshot_path"] = str(tmp_path / "gone")
    record["snapshot_rel_path"] = "gone"
    (tmp_path / "generation-metadata.json").write_text(json.dumps(record))
    report = _diagnose(tmp_path, run_probe=False)
    assert report.state is adapter.DiagnosticState.UNPREPARED
    assert report.signals["metadata_present"] is True
    assert report.signals["artifact_present"] is False


def test_diagnose_runtime_missing(tmp_path):
    _prepare(tmp_path)
    report = _diagnose(
        tmp_path,
        run_probe=False,
        dist_versions=lambda name: (
            None if name == "sentence-transformers" else _FULL_VERSIONS[name]
        ),
        engine_describer=_fake_engine_describer,
    )
    assert report.state is adapter.DiagnosticState.RUNTIME_MISSING
    assert report.signals["runtime_versions_observed"]["sentence-transformers"] is None


def test_diagnose_version_mismatch_on_runtime_drift(tmp_path):
    _prepare(tmp_path)
    drifted = dict(_FULL_VERSIONS, **{"sentence-transformers": "7.7.7"})
    report = _diagnose(
        tmp_path,
        run_probe=False,
        dist_versions=lambda name: drifted[name],
        engine_describer=_fake_engine_describer,
    )
    assert report.state is adapter.DiagnosticState.VERSION_MISMATCH
    assert "7.7.7" in report.signals["detail"]


def test_diagnose_version_mismatch_on_vec_version_drift(tmp_path):
    _prepare(tmp_path)
    report = _diagnose(
        tmp_path,
        run_probe=False,
        dist_versions=_ok_versions,
        engine_describer=lambda: _fake_engine_describer(observed="v0.1.8"),
    )
    assert report.state is adapter.DiagnosticState.VERSION_MISMATCH
    assert "v0.1.8" in report.signals["detail"]


def test_diagnose_prepared_without_probe(tmp_path):
    _prepare(tmp_path)
    report = _diagnose(
        tmp_path,
        run_probe=False,
        dist_versions=_ok_versions,
        engine_describer=_fake_engine_describer,
    )
    assert report.state is adapter.DiagnosticState.PREPARED
    assert report.signals["artifact_sha256_observed"] == report.signals["artifact_sha256_recorded"]
    assert "probe" not in report.signals


def test_diagnose_prepared_with_probe(tmp_path):
    _prepare(tmp_path)
    encoder = FakeEncoder()
    report = _diagnose(
        tmp_path,
        run_probe=True,
        dist_versions=_ok_versions,
        engine_describer=_fake_engine_describer,
        embedder_factory=lambda record, snapshot: encoder,
    )
    assert report.state is adapter.DiagnosticState.PREPARED
    probe = report.signals["probe"]
    assert probe["dimensions"] == 384
    assert probe["blob_bytes"] == 384 * 4
    assert 0.999 <= probe["l2_norm"] <= 1.001


def test_diagnose_probe_failed_on_wrong_dimensions(tmp_path):
    _prepare(tmp_path)
    bad = np.stack([np.ones(7, dtype=np.float32) / 7**0.5])
    report = _diagnose(
        tmp_path,
        run_probe=True,
        dist_versions=_ok_versions,
        engine_describer=_fake_engine_describer,
        embedder_factory=lambda record, snapshot: FakeEncoder(vectors=bad),
    )
    assert report.state is adapter.DiagnosticState.PROBE_FAILED
    assert "dimensions" in report.signals["detail"]


def test_diagnose_probe_failed_on_non_unit_norm(tmp_path):
    _prepare(tmp_path)
    bad = np.stack([np.full(384, 3.0, dtype=np.float32)])
    report = _diagnose(
        tmp_path,
        run_probe=True,
        dist_versions=_ok_versions,
        engine_describer=_fake_engine_describer,
        embedder_factory=lambda record, snapshot: FakeEncoder(vectors=bad),
    )
    assert report.state is adapter.DiagnosticState.PROBE_FAILED
    assert "norm" in report.signals["detail"]


# --- embed contract -------------------------------------------------------------


def _loaded_embedder(vectors) -> adapter.LoadedEmbedder:
    record = {
        "model_id": adapter.MODEL_ID,
        "runtime_versions": dict(_FULL_VERSIONS),
        "dimensions": 384,
    }
    return adapter.LoadedEmbedder(
        model=FakeEncoder(vectors=vectors), metadata=record, snapshot_dir=Path("/x")
    )


def test_embed_produces_float32_little_endian_unit_blobs():
    vec = _unit_vector()
    blobs = _loaded_embedder(np.stack([vec])).embed(["hello"])
    assert len(blobs) == 1
    assert len(blobs[0]) == 384 * 4
    decoded = np.frombuffer(blobs[0], dtype="<f4")
    np.testing.assert_allclose(decoded, vec, atol=1e-6)
    # Little-endian on the wire, independent of host order:
    assert blobs[0][:4] == struct.pack("<f", float(vec[0]))


def test_embed_observer_receives_batch_stats():
    stats = []
    _loaded_embedder(np.stack([_unit_vector()])).embed(
        ["hello"], batch_size=1, observer=stats.append
    )
    assert stats[0].documents == 1
    assert stats[0].batch_size == 1
    assert stats[0].dimensions == 384
    assert stats[0].elapsed_ms >= 0.0


def test_embed_rejects_wrong_dtype():
    bad = np.stack([_unit_vector()]).astype(np.float64)
    with pytest.raises(adapter.EmbeddingValidationError) as excinfo:
        _loaded_embedder(bad).embed(["hello"])
    assert "float32" in excinfo.value.message


def test_embed_rejects_row_count_mismatch():
    vecs = np.stack([_unit_vector(), _unit_vector()])
    with pytest.raises(adapter.EmbeddingValidationError) as excinfo:
        _loaded_embedder(vecs).embed(["only-one"])
    assert "2 vectors for 1 inputs" in excinfo.value.message


def test_embed_empty_input_returns_empty_without_calling_encoder():
    encoder = FakeEncoder()
    embedder = adapter.LoadedEmbedder(
        model=encoder, metadata={"dimensions": 384}, snapshot_dir=Path("/x")
    )
    assert embedder.embed([]) == []
    assert encoder.encode_calls == []


# --- offline-by-construction loading -----------------------------------------------


def _stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    class StubST:
        def __init__(self, path, device=None, local_files_only=False, **kwargs):
            calls.append({"path": path, "device": device, "local_files_only": local_files_only})
            self.max_seq_length = None

        def encode(self, *args, **kwargs):  # pragma: no cover - never reached here
            raise AssertionError("not used in this test")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubST
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


def test_load_embedder_uses_local_files_only_and_pins_seq_length(tmp_path, monkeypatch):
    record = _prepare(tmp_path)
    calls = _stub_sentence_transformers(monkeypatch)

    embedder = adapter.load_embedder(
        tmp_path,
        dist_versions=_ok_versions,
        expected_artifact_sha256=_fake_expected_digest(),
    )

    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True, "loader must never consult the hub"
    assert calls[0]["device"] == "cpu"
    assert calls[0]["path"] == record["snapshot_path"]
    assert embedder.model.max_seq_length == 256


def test_load_embedder_refuses_unprepared(tmp_path):
    with pytest.raises(adapter.EmbeddingCapabilityError) as excinfo:
        adapter.load_embedder(tmp_path, dist_versions=_ok_versions)
    assert excinfo.value.reason == "unprepared"


def test_load_embedder_refuses_runtime_missing_before_any_model_load(tmp_path, monkeypatch):
    _prepare(tmp_path)
    calls = _stub_sentence_transformers(monkeypatch)
    with pytest.raises(adapter.EmbeddingCapabilityError) as excinfo:
        adapter.load_embedder(
            tmp_path,
            dist_versions=lambda name: None if name == "torch" else _FULL_VERSIONS[name],
            expected_artifact_sha256=_fake_expected_digest(),
        )
    assert excinfo.value.reason == "runtime-missing"
    assert calls == [], "runtime check must precede any model construction"


# --- review-round repairs -----------------------------------------------------


def test_diagnose_refuses_foreign_generation_record(tmp_path):
    """A self-consistent record for a DIFFERENT model must not read prepared."""
    record = _prepare(tmp_path)
    foreign = dict(record)
    foreign["model_id"] = "other-org/other-model"
    (tmp_path / "generation-metadata.json").write_text(json.dumps(foreign))
    report = _diagnose(
        tmp_path,
        run_probe=False,
        dist_versions=_ok_versions,
        engine_describer=_fake_engine_describer,
    )
    assert report.state is adapter.DiagnosticState.VERSION_MISMATCH
    assert "pinned generation" in report.signals["detail"]

    calls = []
    with pytest.raises(adapter.EmbeddingCapabilityError) as excinfo:
        adapter.load_embedder(
            tmp_path,
            embedder_factory=lambda rec, snap: calls.append(1),
            dist_versions=_ok_versions,
            expected_artifact_sha256=_fake_expected_digest(),
        )
    assert excinfo.value.reason == "version-mismatch"
    assert calls == [], "identity check must run before any model construction"


def test_prepare_and_diagnose_survive_symlinked_models_dir(tmp_path):
    """macOS /var-style symlinked state roots must not crash rel-path math."""
    real = tmp_path / "real-models"
    link = tmp_path / "linked-models"
    real.mkdir()
    link.symlink_to(real)
    record = _prepare(link)
    assert record["snapshot_rel_path"] is not None
    report = _diagnose(
        link, run_probe=False, dist_versions=_ok_versions, engine_describer=_fake_engine_describer
    )
    assert report.state is adapter.DiagnosticState.PREPARED


def test_unobservable_engine_version_is_runtime_missing_not_mismatch(tmp_path):
    """No observed version means no comparison happened; say so honestly."""
    _prepare(tmp_path)

    def unobservable():
        return {
            "runtime_present": True,
            "extension_api_available": False,
            "vec_version_observed": None,
            "vec_version_pinned": PINNED_VEC_VERSION,
        }

    report = _diagnose(
        tmp_path, run_probe=False, dist_versions=_ok_versions, engine_describer=unobservable
    )
    assert report.state is adapter.DiagnosticState.RUNTIME_MISSING
    assert "no engine version could be observed" in report.signals["detail"]


def test_load_failed_still_reports_extension_api_available(monkeypatch):
    import sys

    class ExplodingVec(types.ModuleType):
        def load(self, conn):  # pragma: no cover - exercised via factory
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "sqlite_vec", ExplodingVec("sqlite_vec"))
    signals = describe_search_engine()
    assert signals["runtime_present"] is True
    assert signals["extension_api_available"] is True
    assert signals["vec_version_observed"] is None


def test_factory_refuses_ambiguous_dual_source(tmp_path):
    from cli_agent_orchestrator.services.search_engine_factory import SearchEngineError

    with pytest.raises(SearchEngineError) as excinfo:
        open_search_connection(
            db_path=tmp_path / "x.db",
            connection_factory=lambda: sqlite3.connect(":memory:"),
        )
    assert excinfo.value.reason == "open-failed"
    assert "not both" in excinfo.value.message


def test_metadata_records_python_version_provenance(tmp_path):
    import platform

    record = _prepare(tmp_path)
    assert record["python_version"] == platform.python_version()
