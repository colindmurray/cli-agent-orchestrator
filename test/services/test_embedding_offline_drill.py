"""Installed-runtime offline drill (hybrid search design §19.4, last row).

Skips cleanly wherever the ``[search]`` extra or a prepared generation is
absent — the base-install contract is a typed capability answer, not a model.
Where the runtime and a prepared artifact exist, this drill PROVES the
offline-after-prepare contract under hostile conditions:

* ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` make any hub call fail loudly;
* every outbound socket creation raises — the network is not merely
  "probably unused", it is UNAVAILABLE;
* the drill still diagnoses PREPARED and produces validated embeddings.

Run it for real after an explicit prepare, e.g.::

    CAO_SEARCH_MODELS_DIR=<prepared dir> .venv/bin/pytest \
        test/services/test_embedding_offline_drill.py -m slow
"""

from __future__ import annotations

import os
import socket

import pytest

from cli_agent_orchestrator.services import embedding_adapter as adapter

pytestmark = pytest.mark.slow

pytest.importorskip("sentence_transformers")
pytest.importorskip("sqlite_vec")


@pytest.fixture()
def network_disabled(monkeypatch: pytest.MonkeyPatch):
    """Make every socket creation a loud failure for the drill's duration."""

    def refused(*args, **kwargs):
        raise OSError("network disabled by offline drill")

    monkeypatch.setattr(socket, "socket", refused)
    monkeypatch.setattr(socket, "create_connection", refused)
    yield


@pytest.fixture()
def hub_offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    yield


@pytest.fixture()
def prepared_models_dir():
    models_dir = adapter.default_models_dir()
    if adapter.read_metadata(models_dir) is None:
        pytest.skip(
            f"no prepared generation at {models_dir}; run `cao issue search-index "
            "model prepare` first (or point CAO_SEARCH_MODELS_DIR at one)"
        )
    return models_dir


def test_offline_drill_diagnoses_prepared_and_embeds(
    prepared_models_dir, hub_offline, network_disabled
):
    report = adapter.diagnose_embedding(prepared_models_dir, run_probe=True)
    assert report.state is adapter.DiagnosticState.PREPARED, report.signals
    assert report.signals["probe"]["dimensions"] == 384
    assert report.signals["engine"]["vec_version_observed"] == "v0.1.9"


def test_offline_drill_embeddings_are_validated_and_separable(
    prepared_models_dir, hub_offline, network_disabled
):
    embedder = adapter.load_embedder(prepared_models_dir)
    blobs = embedder.embed(
        [
            "sqlite-vec loads only through the dedicated search connection factory",
            "the search connection factory loads sqlite-vec on its own connection",
            "the dashboard renders a topology widget of tmux panes",
        ],
        batch_size=2,
    )
    assert len(blobs) == 3
    import numpy as np

    vecs = np.stack([np.frombuffer(blob, dtype="<f4") for blob in blobs])
    assert vecs.shape == (3, 384)
    norms = np.linalg.norm(vecs.astype(np.float64), axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-3)

    def cosine(a, b):
        return float(np.dot(a.astype(np.float64), b.astype(np.float64)))

    related = cosine(vecs[0], vecs[1])
    unrelated = cosine(vecs[0], vecs[2])
    assert related > 0.5, f"paraphrase pair should be close, got {related}"
    assert (
        related > unrelated
    ), f"related pair ({related:.3f}) must outrank unrelated ({unrelated:.3f})"


def test_offline_drill_output_is_byte_stable_within_process(
    prepared_models_dir, hub_offline, network_disabled
):
    embedder = adapter.load_embedder(prepared_models_dir)
    text = ["byte determinism probe"]
    first = embedder.embed(text)[0]
    second = embedder.embed(text)[0]
    assert first == second
