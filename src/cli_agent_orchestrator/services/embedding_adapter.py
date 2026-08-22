"""Local embedding adapter: explicit prepare, offline embed, typed capability.

Implements the measured v1 decision for semantic issue search (hybrid
issue-search design §9.4, records required before semantic enable):

* ONE pinned model generation — ``sentence-transformers/all-MiniLM-L6-v2`` at
  an immutable revision, 384-dimensional float32 little-endian vectors,
  L2-normalized by the model graph itself, cosine distance, 256-token input
  ceiling. Sharing a model id never implies interchangeable outputs, so the
  runtime versions are bound into the generation metadata alongside the
  artifact digest rather than assumed.
* Preparation is an EXPLICIT operator command (:func:`prepare_model`): it
  downloads the pinned snapshot once, verifies the artifact against the
  recorded digest, and writes generation-ready metadata. No issue write and
  no ordinary search path ever downloads anything — after prepare, loading
  reads the local snapshot only (``local_files_only``), so the capability
  runs with the network unavailable.
* Capability diagnostics (:func:`diagnose_embedding`) report exactly five
  states — ``prepared``, ``unprepared``, ``runtime-missing``,
  ``version-mismatch``, ``probe-failed`` — and every state carries positive
  signals (what was actually observed: digests, versions, probe stats), not
  an exit code to reverse-engineer.

Heavy dependencies (sentence-transformers/torch/sqlite-vec) are optional
``[search]`` extra members and are imported lazily; on a base install every
entry point degrades to a typed ``runtime-missing`` answer instead of an
ImportError.
"""

from __future__ import annotations

import enum
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from cli_agent_orchestrator.constants import SEARCH_MODELS_DIR, SEARCH_MODELS_DIR_ENV
from cli_agent_orchestrator.services.search_engine_factory import (
    PINNED_VEC_VERSION,
    describe_search_engine,
)

logger = logging.getLogger(__name__)

# --- The pinned v1 generation (measured, not chosen by popularity) ----------

METADATA_SCHEMA = "cao-search-generation-metadata-v1"
METADATA_FILENAME = "generation-metadata.json"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
#: sha256 over the snapshot's files in sorted-relative-path order (path bytes,
#: NUL separator, file bytes) — the exact digest recorded by the M0.3
#: measurement run. Verifying against it is what makes "the artifact on disk
#: is the measured artifact" a checked fact instead of a directory's opinion.
MODEL_ARTIFACT_SHA256 = "488a3886929a909be2f279ff81048db2c781fd4c87228a74c2430b516d214b14"
MODEL_ARTIFACT_BYTES = 182_519_781

DIMENSIONS = 384
ELEMENT_TYPE = "float32"
#: Normalization is baked into the model graph (outputs are unit-norm without
#: a caller-side flag), so embed() VALIDATES the norm rather than re-normalizing:
#: a re-normalizing wrapper would silently bless a wrong or drifted artifact.
NORMALIZED = True
DISTANCE_METRIC = "cosine"
#: MiniLM truncates inputs at 256 tokens; document builders must stay
#: short-field-first because anything longer is silently cut (design §9.4).
MAX_SEQ_LENGTH = 256

RUNTIME_ID = "sentence-transformers"
#: Runtime distributions whose installed versions are bound into the
#: generation metadata. Sharing a model id across runtimes is NOT numerically
#: interchangeable (the M0.3 interchangeability measurement), so a version
#: drift is a capability refusal, not a footnote.
REQUIRED_RUNTIME_PACKAGES = ("sentence-transformers", "torch", "transformers")

#: Versioned document builder identity recorded in the metadata. The builder
#: itself lands with the vector-refresh lane; the generation must already name
#: which document form its vectors encode.
DOCUMENT_SCHEMA_VERSION_ID = 1
DOCUMENT_SCHEMA_VERSION_NAME = "m0.3-issue-doc-v0"

#: Snapshot filters copied EXACTLY from the measurement run's prepare script:
#: the recorded digest covers the full snapshot MINUS these packaging trees,
#: so a prepare that filtered differently would produce a different — and
#: correctly refused — digest.
SNAPSHOT_IGNORE_PATTERNS = [
    "*.git*",
    "*.msgpack",
    "*.h5",
    "*.tflite",
    "*.ot",
    "onnx/*",
    "onnx*",
    "openvino/*",
    "openvino*",
    "*onnx/model*",
    "**/onnx/**",
]

#: Norm tolerance for the post-encode validation. float32 L2 normalization
#: over 384 dims lands far inside 1e-3; a violation means the loaded artifact
#: is not the graph that was measured.
_NORM_TOLERANCE = 1e-3

EmbedBatchObserver = Callable[["EmbedBatchStats"], None]
SnapshotDownloader = Callable[..., str]
DistVersionReader = Callable[[str], Optional[str]]


# --- Typed failures ----------------------------------------------------------


class EmbeddingCapabilityError(Exception):
    """Typed embedding-capability failure.

    ``reason`` is one of the diagnostic vocabulary values (``unprepared``,
    ``runtime-missing``, ``version-mismatch``, ``probe-failed``) or a
    prepare-time classification (``digest-mismatch``, ``prepare-failed``).
    Callers branch on ``reason``; the message is for humans and carries the
    observed values.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


class ArtifactDigestMismatch(EmbeddingCapabilityError):
    """The downloaded/local artifact does not match the recorded digest."""

    def __init__(self, observed: str, expected: str, artifact_path: Path) -> None:
        super().__init__(
            "digest-mismatch",
            f"artifact digest mismatch at {artifact_path}: observed {observed}, "
            f"recorded {expected}. The local snapshot is not the measured "
            "artifact; delete it and re-run prepare from a trusted network.",
        )
        self.observed = observed
        self.expected = expected
        self.artifact_path = artifact_path


class EmbeddingValidationError(EmbeddingCapabilityError):
    """A produced vector violated the generation contract (dim/dtype/norm)."""


# --- Locations and small helpers ---------------------------------------------


def default_models_dir() -> Path:
    """Resolve the models directory: env override, else the CAO state root."""
    override = os.environ.get(SEARCH_MODELS_DIR_ENV)
    if override:
        return Path(override)
    return SEARCH_MODELS_DIR


def metadata_path(models_dir: Union[str, Path]) -> Path:
    return Path(models_dir) / METADATA_FILENAME


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_dist_version(name: str) -> Optional[str]:
    """Installed distribution version, or None when absent (never imports)."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dir_sha256(root: Path) -> str:
    """Deterministic directory digest: sorted relative paths, then bytes.

    Same algorithm as the measurement run's prepare script so the recorded
    digest reproduces byte-for-byte: for every file in sorted-relative-path
    order, hash the POSIX relative path, a NUL separator, then the file
    content in 1 MiB chunks.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        with open(path, "rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def _default_snapshot_downloader(
    *,
    repo_id: str,
    revision: str,
    cache_dir: Path,
    ignore_patterns: Optional[Sequence[str]] = None,
) -> str:
    """Download the pinned snapshot via huggingface_hub (the only network call
    this module ever makes, and only from :func:`prepare_model`)."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # A machine-readable operator command should not paint progress bars
    # across its stderr.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    return str(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            ignore_patterns=(
                list(ignore_patterns)
                if ignore_patterns is not None
                else list(SNAPSHOT_IGNORE_PATTERNS)
            ),
        )
    )


def _write_metadata_atomic(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(record), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_metadata(models_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read the generation metadata, or None when absent/unreadable.

    Absent and corrupt are different observations, so corrupt metadata raises
    :class:`EmbeddingCapabilityError` (``unprepared`` with the parse failure
    in the message) rather than silently reading as "nothing prepared": a
    half-written or hand-mangled metadata file must not be mistaken for a
    store that was never prepared.
    """
    path = metadata_path(models_dir)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise EmbeddingCapabilityError(
            "unprepared",
            f"generation metadata at {path} exists but cannot be parsed ({exc}); "
            "re-run prepare to rewrite it",
        ) from exc
    if not isinstance(record, dict):
        raise EmbeddingCapabilityError(
            "unprepared",
            f"generation metadata at {path} is not an object; re-run prepare to rewrite it",
        )
    return record


def _snapshot_dir(models_dir: Path, record: Mapping[str, Any]) -> Path:
    """Locate the snapshot: metadata-relative path first, absolute fallback.

    The relative form survives a state root that moved (backup restore,
    CAO_STATE_ROOT change); the absolute form is the recorded fallback for
    snapshots prepared outside their own models_dir.
    """
    rel = record.get("snapshot_rel_path")
    if isinstance(rel, str) and rel:
        candidate = models_dir / rel
        if candidate.exists():
            return candidate
    absolute = record.get("snapshot_path")
    if isinstance(absolute, str) and absolute:
        candidate = Path(absolute)
        if candidate.exists():
            return candidate
    raise EmbeddingCapabilityError(
        "unprepared",
        "generation metadata names a snapshot that does not exist on disk "
        f"(looked at {rel!r} under {models_dir} and {absolute!r}); re-run prepare",
    )


def _check_generation_identity(record: Mapping[str, Any], expected_artifact_sha256: str) -> None:
    """Verify the record names THE pinned generation, not just a consistent one.

    A well-formed self-consistent metadata file for a DIFFERENT model or
    revision (another machine's store restored under CAO_SEARCH_MODELS_DIR,
    say) would otherwise read as prepared and produce embeddings no other
    generation can compare against. The identity triple — model id, immutable
    revision, recorded artifact digest — must equal this build's pins;
    anything else is a version-mismatch naming both sides.
    """
    observed_id = record.get("model_id")
    observed_rev = record.get("model_revision")
    observed_digest = record.get("artifact_sha256")
    if (
        observed_id == MODEL_ID
        and observed_rev == MODEL_REVISION
        and observed_digest == expected_artifact_sha256
    ):
        return
    raise EmbeddingCapabilityError(
        "version-mismatch",
        "prepared generation does not match this build's pinned generation: "
        f"model {observed_id!r}@{observed_rev!r} digest {observed_digest!r} but this "
        f"build pins {MODEL_ID!r}@{MODEL_REVISION!r} digest {expected_artifact_sha256!r}. "
        "Re-run prepare to install the pinned artifact.",
    )


# --- Explicit operator prepare ------------------------------------------------


def prepare_model(
    models_dir: Union[str, Path, None] = None,
    *,
    snapshot_downloader: Optional[SnapshotDownloader] = None,
    expected_artifact_sha256: Optional[str] = None,
    dist_versions: Optional[DistVersionReader] = None,
    hf_cache_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Prepare the pinned embedding generation. Explicit; idempotent.

    Downloads (or reuses) the pinned snapshot, verifies the artifact digest
    against the recorded value, and writes generation-ready metadata binding
    every §9.4 record: model id + immutable revision, runtime id + versions,
    dimensions, encoding/normalization/distance convention, artifact digest,
    and the document-builder schema version. Returns the metadata record.

    Re-running against an already-prepared, digest-verified store returns the
    EXISTING record unchanged — ``prepared_at`` does not move, the downloader
    is not called — so scheduled re-invocations are free. When the artifact is
    intact but the metadata is missing, corrupt, or drifted, prepare rewrites
    it (repair, not blessing): the digest, not the old file, decides truth.

    ``snapshot_downloader`` and ``dist_versions`` are injection seams for
    tests; the defaults are the real network downloader and the installed
    distribution versions.
    """
    # Resolved at CALL time (not as a def-time default) so embedders of the
    # module — including tests — can re-pin the target artifact explicitly.
    if expected_artifact_sha256 is None:
        expected_artifact_sha256 = MODEL_ARTIFACT_SHA256
    resolved_dir = Path(models_dir) if models_dir is not None else default_models_dir()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_cache = Path(hf_cache_dir) if hf_cache_dir is not None else resolved_dir / "_hf-cache"
    read_version = dist_versions if dist_versions is not None else _read_dist_version

    runtime_versions: Dict[str, Optional[str]] = {
        package: read_version(package) for package in REQUIRED_RUNTIME_PACKAGES
    }
    missing = sorted(p for p, v in runtime_versions.items() if v is None)
    if missing:
        raise EmbeddingCapabilityError(
            "runtime-missing",
            "prepare requires the [search] runtime to record honest generation "
            f"metadata, but these distributions are not installed: {', '.join(missing)}",
        )

    downloader = snapshot_downloader or _default_snapshot_downloader

    # Fast path: a well-formed existing record for THIS model/revision whose
    # snapshot is on disk and digest-verifies needs no download at all —
    # recomputing the digest IS the verification. Identical state returns the
    # existing record unchanged (idempotent); a corrupted artifact refuses
    # loudly instead of silently re-downloading over evidence; anything else
    # falls through to the download/repair path.
    existing_path = metadata_path(resolved_dir)
    existing: Optional[Dict[str, Any]] = None
    if existing_path.exists():
        try:
            loaded = json.loads(existing_path.read_text())
            existing = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError):
            existing = None

    snapshot: Optional[Path] = None
    if (
        isinstance(existing, dict)
        and existing.get("model_id") == MODEL_ID
        and existing.get("model_revision") == MODEL_REVISION
    ):
        try:
            candidate = _snapshot_dir(resolved_dir, existing)
        except EmbeddingCapabilityError:
            candidate = None
        if candidate is not None:
            observed_digest = dir_sha256(candidate)
            if observed_digest != expected_artifact_sha256:
                raise ArtifactDigestMismatch(observed_digest, expected_artifact_sha256, candidate)
            snapshot = candidate
            if (
                existing.get("artifact_sha256") == expected_artifact_sha256
                and existing.get("runtime_versions") == runtime_versions
            ):
                logger.info(
                    "embedding generation already prepared and verified at %s "
                    "(digest ok, metadata unchanged)",
                    resolved_dir,
                )
                return existing

    if snapshot is None:
        snapshot = Path(
            downloader(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=resolved_cache,
                ignore_patterns=list(SNAPSHOT_IGNORE_PATTERNS),
            )
        )
        if not snapshot.is_dir():
            raise EmbeddingCapabilityError(
                "prepare-failed",
                f"snapshot downloader returned a non-directory: {snapshot}",
            )
        observed_digest = dir_sha256(snapshot)
        if observed_digest != expected_artifact_sha256:
            raise ArtifactDigestMismatch(observed_digest, expected_artifact_sha256, snapshot)

    observed_bytes = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())

    resolved_snapshot = snapshot.resolve()
    resolved_models = resolved_dir.resolve()
    rel_path: Optional[str] = None
    if resolved_snapshot.is_relative_to(resolved_models):
        # Both sides resolved so a symlinked models_dir (macOS /var, volume
        # aliases) cannot make the containment check and the relativization
        # disagree.
        rel_path = resolved_snapshot.relative_to(resolved_models).as_posix()

    record: Dict[str, Any] = {
        "schema": METADATA_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_id": RUNTIME_ID,
        "runtime_versions": runtime_versions,
        "dimensions": DIMENSIONS,
        "element_type": ELEMENT_TYPE,
        "normalized": NORMALIZED,
        "distance_metric": DISTANCE_METRIC,
        "max_seq_length": MAX_SEQ_LENGTH,
        "document_schema_version_id": DOCUMENT_SCHEMA_VERSION_ID,
        "document_schema_version_name": DOCUMENT_SCHEMA_VERSION_NAME,
        "artifact_sha256": observed_digest,
        "artifact_bytes": observed_bytes,
        "vec_version_pinned": PINNED_VEC_VERSION,
        "python_version": platform.python_version(),
        "snapshot_path": str(snapshot),
        "snapshot_rel_path": rel_path,
        "prepared_at": _utcnow_iso(),
    }

    _write_metadata_atomic(existing_path, record)
    logger.info(
        "prepared embedding generation %s@%s (artifact sha256 ok, %d bytes) -> %s",
        MODEL_ID,
        MODEL_REVISION,
        observed_bytes,
        existing_path,
    )
    return record


# --- Offline-after-prepare load and embed ------------------------------------


def _check_runtime(
    metadata: Mapping[str, Any], read_version: DistVersionReader
) -> Dict[str, Optional[str]]:
    """Verify the installed runtime against the generation's bound versions.

    Returns the observed versions on success; raises typed
    ``runtime-missing`` / ``version-mismatch`` otherwise. The Python version
    is recorded in metadata but never gating: interpreters change for
    unrelated operational reasons and the artifact/runtime binding that
    decides numeric behavior is the three distributions.
    """
    bound = metadata.get("runtime_versions")
    if not isinstance(bound, dict):
        raise EmbeddingCapabilityError(
            "version-mismatch",
            "generation metadata carries no runtime_versions map; re-run prepare",
        )
    observed: Dict[str, Optional[str]] = {}
    missing: List[str] = []
    drifted: Dict[str, Dict[str, Optional[str]]] = {}
    for package in REQUIRED_RUNTIME_PACKAGES:
        version = read_version(package)
        observed[package] = version
        if version is None:
            missing.append(package)
        elif package in bound and bound[package] != version:
            drifted[package] = {"observed": version, "expected": bound[package]}
    if missing:
        raise EmbeddingCapabilityError(
            "runtime-missing",
            "embedding runtime not installed: " + ", ".join(sorted(missing)),
        )
    if drifted:
        detail = "; ".join(
            f"{pkg} observed {v['observed']} but generation bound {v['expected']}"
            for pkg, v in sorted(drifted.items())
        )
        raise EmbeddingCapabilityError(
            "version-mismatch",
            f"installed runtime does not match the prepared generation ({detail}). "
            "Embeddings are only comparable within one runtime; re-prepare and "
            "rebuild vectors as a new generation to switch runtimes.",
        )
    return observed


def _default_embedder_factory(metadata: Mapping[str, Any], snapshot_dir: Path) -> Any:
    """Load the pinned model from the LOCAL snapshot only.

    ``local_files_only=True`` is the offline-after-prepare contract in one
    argument: the loader never consults the hub, so a prepared capability
    works with the network fully unavailable. ``normalize_embeddings`` is
    deliberately left at its default — normalization is baked into this
    model's graph, and a caller-side re-normalization would hide artifact
    drift that the post-encode validation is supposed to catch.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    model = SentenceTransformer(
        str(snapshot_dir),
        device="cpu",
        local_files_only=True,
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


@dataclass
class LoadedEmbedder:
    """A loaded pinned model plus its generation metadata.

    ``embed`` returns one float32 little-endian bytes blob per input, each
    validated against the generation contract (dimension, dtype, unit norm)
    before serialization — the same bytes shape ``tracker_search_vectors``
    stores and sqlite-vec's scalar distance consumes.
    """

    model: Any
    metadata: Dict[str, Any]
    snapshot_dir: Path
    dimensions: int = DIMENSIONS

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        observer: Optional[EmbedBatchObserver] = None,
    ) -> List[bytes]:
        """Embed texts into validated float32-LE bytes blobs.

        ``observer`` (benchmark adapter hook) receives one
        :class:`EmbedBatchStats` per call with elapsed time and batch shape —
        measurement rides the production path instead of a parallel one.
        """
        import numpy as np

        if not texts:
            return []
        started = datetime.now(timezone.utc)
        raw = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = self._validate_batch(raw, np, len(texts))
        blobs = [np.ascontiguousarray(vec, dtype="<f4").tobytes() for vec in vectors]
        if observer is not None:
            elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
            observer(
                EmbedBatchStats(
                    documents=len(texts),
                    batch_size=batch_size,
                    elapsed_ms=round(elapsed_ms, 3),
                    dimensions=self.dimensions,
                )
            )
        return blobs

    def _validate_batch(self, raw: Any, np: Any, expected_count: int) -> Any:
        array = np.asarray(raw)
        if array.ndim != 2:
            raise EmbeddingValidationError(
                "probe-failed",
                f"encoder returned {array.ndim}-dimensional output with shape "
                f"{array.shape}, expected (n, {self.dimensions})",
            )
        if array.shape[0] != expected_count:
            raise EmbeddingValidationError(
                "probe-failed",
                f"encoder returned {array.shape[0]} vectors for " f"{expected_count} inputs",
            )
        if array.shape[1] != self.dimensions:
            raise EmbeddingValidationError(
                "probe-failed",
                f"encoder returned {array.shape[1]} dimensions, generation binds "
                f"{self.dimensions}",
            )
        if array.dtype != np.float32:
            raise EmbeddingValidationError(
                "probe-failed",
                f"encoder returned dtype {array.dtype}, generation binds float32",
            )
        norms = np.linalg.norm(array.astype(np.float64), axis=1)
        bad = [
            (index, float(norm))
            for index, norm in enumerate(norms)
            if abs(norm - 1.0) > _NORM_TOLERANCE
        ]
        if bad:
            index, norm = bad[0]
            raise EmbeddingValidationError(
                "probe-failed",
                f"vector {index} has L2 norm {norm:.6f}, expected unit-norm "
                f"(tolerance {_NORM_TOLERANCE}); the loaded artifact is not "
                "behaving like the measured normalized model graph",
            )
        return array


@dataclass
class EmbedBatchStats:
    """What one embed call cost — the benchmark adapter hook payload."""

    documents: int
    batch_size: int
    elapsed_ms: float
    dimensions: int


def load_embedder(
    models_dir: Union[str, Path, None] = None,
    *,
    embedder_factory: Optional[Callable[[Mapping[str, Any], Path], Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    dist_versions: Optional[DistVersionReader] = None,
    expected_artifact_sha256: Optional[str] = None,
) -> LoadedEmbedder:
    """Load the prepared generation, refusing typed absence/mismatch first.

    Order matters and is observable: metadata absence is ``unprepared``; a
    record naming a different model/revision/digest is ``version-mismatch``
    (the pin check runs before anything heavy); missing distributions are
    ``runtime-missing``; installed-but-different versions are
    ``version-mismatch``; only then does the (possibly injected) factory load
    the local snapshot. A base install therefore answers in milliseconds
    without ever touching torch.
    """
    # Call-time pin resolution, matching prepare_model: re-pinning the module
    # constant (tests, alternate builds) must reach every entry point the same
    # way instead of freezing at import time.
    if expected_artifact_sha256 is None:
        expected_artifact_sha256 = MODEL_ARTIFACT_SHA256
    resolved_dir = Path(models_dir) if models_dir is not None else default_models_dir()
    record: Optional[Dict[str, Any]] = (
        dict(metadata) if metadata is not None else read_metadata(resolved_dir)
    )
    if record is None:
        raise EmbeddingCapabilityError(
            "unprepared",
            f"no generation metadata at {metadata_path(resolved_dir)}; run the "
            "explicit model prepare command first",
        )
    _check_generation_identity(record, expected_artifact_sha256)
    read_version = dist_versions if dist_versions is not None else _read_dist_version
    _check_runtime(record, read_version)
    snapshot = _snapshot_dir(resolved_dir, record)
    factory = embedder_factory or _default_embedder_factory
    model = factory(record, snapshot)
    return LoadedEmbedder(model=model, metadata=record, snapshot_dir=snapshot)


# --- Capability diagnostics ----------------------------------------------------


class DiagnosticState(str, enum.Enum):
    """The five capability states, reported with positive signals."""

    PREPARED = "prepared"
    UNPREPARED = "unprepared"
    RUNTIME_MISSING = "runtime-missing"
    VERSION_MISMATCH = "version-mismatch"
    PROBE_FAILED = "probe-failed"


@dataclass
class CapabilityReport:
    """One diagnostic answer: a state plus what was actually observed."""

    state: DiagnosticState
    signals: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"state": self.state.value, "signals": self.signals}


def diagnose_embedding(
    models_dir: Union[str, Path, None] = None,
    *,
    run_probe: bool = True,
    embedder_factory: Optional[Callable[[Mapping[str, Any], Path], Any]] = None,
    engine_describer: Optional[Callable[[], Dict[str, Any]]] = None,
    dist_versions: Optional[DistVersionReader] = None,
    expected_artifact_sha256: Optional[str] = None,
) -> CapabilityReport:
    """Diagnose the embedding capability end to end, never raising.

    Evaluation order mirrors the dependency chain, and each stage records
    positive signals before possibly stopping:

    1. metadata present and parseable, snapshot on disk, digest re-verified
       — otherwise ``unprepared``;
    2. the record names THIS build's pinned generation (model id, revision,
       artifact digest) — otherwise ``version-mismatch``;
    3. required distributions installed — otherwise ``runtime-missing``;
    4. installed versions equal the generation-bound versions, and the
       loaded engine's ``vec_version()`` equals the pin — otherwise
       ``version-mismatch``;
    5. a real probe embedding passes the dim/dtype/norm contract — otherwise
       ``probe-failed``; all green is ``prepared``.

    An engine whose version could not be OBSERVED at all (extension API
    unavailable or the load itself failing) reports ``runtime-missing`` with
    what was seen — never a mismatch between two versions when only one of
    them exists.

    ``run_probe=False`` answers from metadata/runtime/engine observation only
    (no model load, no encode). Every injection parameter exists so tests can
    exercise all five states without heavy dependencies.
    """
    if expected_artifact_sha256 is None:
        expected_artifact_sha256 = MODEL_ARTIFACT_SHA256
    signals: Dict[str, Any] = {}
    resolved_dir = Path(models_dir) if models_dir is not None else default_models_dir()
    signals["models_dir"] = str(resolved_dir)
    read_version = dist_versions if dist_versions is not None else _read_dist_version

    try:
        record = read_metadata(resolved_dir)
    except EmbeddingCapabilityError as exc:
        return CapabilityReport(DiagnosticState.UNPREPARED, {**signals, "detail": exc.message})
    signals["metadata_present"] = record is not None
    if record is None:
        return CapabilityReport(DiagnosticState.UNPREPARED, signals)

    signals["model_id"] = record.get("model_id")
    signals["model_revision"] = record.get("model_revision")
    signals["document_schema_version_name"] = record.get("document_schema_version_name")

    try:
        snapshot = _snapshot_dir(resolved_dir, record)
    except EmbeddingCapabilityError as exc:
        signals["artifact_present"] = False
        return CapabilityReport(DiagnosticState.UNPREPARED, {**signals, "detail": exc.message})
    signals["artifact_present"] = True
    signals["artifact_path"] = str(snapshot)

    observed_digest = dir_sha256(snapshot)
    signals["artifact_sha256_observed"] = observed_digest
    signals["artifact_sha256_recorded"] = record.get("artifact_sha256")
    if observed_digest != record.get("artifact_sha256"):
        return CapabilityReport(
            DiagnosticState.UNPREPARED,
            {
                **signals,
                "detail": "artifact on disk does not match the recorded digest; "
                "re-run prepare from a trusted source",
            },
        )
    try:
        _check_generation_identity(record, expected_artifact_sha256)
    except EmbeddingCapabilityError as exc:
        return CapabilityReport(
            DiagnosticState.VERSION_MISMATCH, {**signals, "detail": exc.message}
        )

    try:
        observed_versions = _check_runtime(record, read_version)
    except EmbeddingCapabilityError as exc:
        signals["runtime_versions_observed"] = {
            pkg: read_version(pkg) for pkg in REQUIRED_RUNTIME_PACKAGES
        }
        state = (
            DiagnosticState.RUNTIME_MISSING
            if exc.reason == "runtime-missing"
            else DiagnosticState.VERSION_MISMATCH
        )
        return CapabilityReport(state, {**signals, "detail": exc.message})
    signals["runtime_versions_observed"] = observed_versions
    signals["runtime_versions_bound"] = record.get("runtime_versions")

    describer = engine_describer or describe_search_engine
    engine = describer()
    signals["engine"] = engine
    if not engine.get("runtime_present"):
        return CapabilityReport(
            DiagnosticState.RUNTIME_MISSING,
            {**signals, "detail": "sqlite-vec runtime not installed ([search] extra)"},
        )
    observed_vec = engine.get("vec_version_observed")
    if observed_vec is None:
        return CapabilityReport(
            DiagnosticState.RUNTIME_MISSING,
            {
                **signals,
                "detail": "sqlite-vec is installed but no engine version could be "
                "observed (extension API unavailable or load failed); vector search "
                "cannot serve on this sqlite build",
            },
        )
    if observed_vec != engine.get("vec_version_pinned"):
        return CapabilityReport(
            DiagnosticState.VERSION_MISMATCH,
            {
                **signals,
                "detail": f"vec_version() observed {observed_vec!r} but the "
                f"generation pins {engine.get('vec_version_pinned')!r}",
            },
        )

    if not run_probe:
        return CapabilityReport(DiagnosticState.PREPARED, signals)

    try:
        embedder = load_embedder(
            resolved_dir,
            embedder_factory=embedder_factory,
            metadata=record,
            dist_versions=read_version,
            expected_artifact_sha256=expected_artifact_sha256,
        )
        stats: List[EmbedBatchStats] = []
        blobs = embedder.embed(
            ["capability probe: explicit prepare, offline embed"],
            batch_size=1,
            observer=stats.append,
        )
        import numpy as np

        decoded = np.frombuffer(blobs[0], dtype="<f4")
        signals["probe"] = {
            "documents": 1,
            "blob_bytes": len(blobs[0]),
            "dimensions": int(decoded.shape[0]),
            "l2_norm": round(float(np.linalg.norm(decoded.astype(np.float64))), 6),
            "elapsed_ms": stats[0].elapsed_ms if stats else None,
        }
    except EmbeddingCapabilityError as exc:
        return CapabilityReport(DiagnosticState.PROBE_FAILED, {**signals, "detail": exc.message})
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        return CapabilityReport(
            DiagnosticState.PROBE_FAILED,
            {**signals, "detail": f"probe embed failed: {exc}"},
        )
    return CapabilityReport(DiagnosticState.PREPARED, signals)
