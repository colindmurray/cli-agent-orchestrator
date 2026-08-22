"""Closed, secret-free validation for one M10 positive route-observation receipt.

The positive ``cao-route-observation-receipt-v1`` is the only receipt a
canary may mint: it proves a correlated provider-surface observation and a
proven owned close, bound to the exact operation/request, target/provider
tuple, and requester.  It never copies raw screen text — the observation
carries only typed facts plus the evidence digest, and this validator
rejects any multi-line string as a shape that could smuggle pane content.

The field set mirrors ``route_observation._build_positive_receipt`` and the
spec-§7 minimum field set; the terminal result vocabulary is the closed
three-value set from ``route_observation.TERMINAL_RESULTS``.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from test.e2e.route_observation_canary.fixtures import (
    CODEX_EFFORT_VOCABULARY,
    CODEX_PINNED_VERSION,
)
from typing import Any, cast

RECEIPT_SCHEMA = "cao-route-observation-receipt-v1"
KIND_OBSERVED_CLOSED = "observed-closed"
OBSERVATION_KIND = "codex-status-v1"
CLOSE_SURFACE = "non-modal"
CLOSE_ACTION_NONE = "none"
CLOSE_OUTCOME_COMPOSER_RESTORED = "composer-restored"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "operation_id",
        "request_digest",
        "requester_terminal_id",
        "requester_generation",
        "target_terminal_id",
        "target_generation",
        "native_session_id",
        "provider",
        "provider_version",
        "provider_artifact_sha256",
        "observation",
        "close_proof",
        "final_event_digest",
        "committed_at",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {
        "kind",
        "observation_kind",
        "observed_state",
        "reason",
        "observed_at",
        "provider_version",
        "parser_key",
        "session_id",
        "correlated",
        "model",
        "effort",
        "render_floor",
        "evidence_sha256",
    }
)
_RENDER_FLOOR_FIELDS = frozenset({"width"})
_CLOSE_PROOF_FIELDS = frozenset({"kind", "surface", "close_action", "outcome", "closed_at"})

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:%/-]{0,127}$")

_MAX_RECEIPT_BYTES = 16_384


class RouteObservationReceiptInvalid(ValueError):
    """A receipt is malformed, or claims more than a positive canary proved."""


def _closed_mapping(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteObservationReceiptInvalid(f"{label} must be a mapping")
    copied = dict(value)
    unknown = sorted(set(copied) - fields)
    missing = sorted(fields - set(copied))
    if unknown or missing:
        raise RouteObservationReceiptInvalid(
            f"{label} must use the closed field set; unknown={unknown}, missing={missing}"
        )
    return copied


def _text(value: Any, *, label: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise RouteObservationReceiptInvalid(f"{label} must be non-empty bounded text")
    # Raw screen text is multi-line; a receipt fact never carries it.
    if "\n" in value or "\r" in value:
        raise RouteObservationReceiptInvalid(f"{label} must not carry raw screen text")
    return value


def _uuid(value: Any, *, label: str) -> str:
    text = _text(value, label=label, maximum=64)
    try:
        if str(uuid.UUID(text)) != text:
            raise ValueError
    except ValueError as exc:
        raise RouteObservationReceiptInvalid(f"{label} must be a canonical lowercase UUID") from exc
    return text


def _identifier(value: Any, *, label: str) -> str:
    text = _text(value, label=label, maximum=128)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise RouteObservationReceiptInvalid(f"{label} is not a well-formed identifier")
    return text


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RouteObservationReceiptInvalid(f"{label} must be a lowercase sha256 digest")
    return value


def _utc(value: Any, *, label: str) -> str:
    text = _text(value, label=label, maximum=64)
    if _UTC_RE.fullmatch(text) is None:
        raise RouteObservationReceiptInvalid(f"{label} must be an RFC3339 UTC timestamp")
    return text


def _optional_text(value: Any, *, label: str, maximum: int = 256) -> str | None:
    if value is None:
        return None
    return _text(value, label=label, maximum=maximum)


def _validate_observation(value: Any, *, native_session_id: str) -> dict[str, Any]:
    """The positive provider-surface observation fact, bound to the target.

    Only ``observed_state == 'observed'`` and ``correlated is True`` prove a
    positive observation; the observation carries typed facts and the
    evidence digest — never raw pane rows.
    """
    observation = _closed_mapping(value, _OBSERVATION_FIELDS, label="observation")
    if observation["kind"] != "provider-surface":
        raise RouteObservationReceiptInvalid("observation.kind must be 'provider-surface'")
    if observation["observation_kind"] != OBSERVATION_KIND:
        raise RouteObservationReceiptInvalid(
            f"observation.observation_kind must be {OBSERVATION_KIND!r}"
        )
    if observation["parser_key"] != OBSERVATION_KIND:
        raise RouteObservationReceiptInvalid(f"observation.parser_key must be {OBSERVATION_KIND!r}")
    if observation["observed_state"] != "observed":
        raise RouteObservationReceiptInvalid(
            "a positive receipt requires observation.observed_state == 'observed'"
        )
    if observation["reason"] is not None:
        raise RouteObservationReceiptInvalid("a positive observation must carry reason == null")
    if observation["correlated"] is not True:
        raise RouteObservationReceiptInvalid(
            "a positive observation must correlate to the exact target (correlated == true)"
        )
    session_id = _uuid(observation["session_id"], label="observation.session_id")
    if session_id != native_session_id:
        raise RouteObservationReceiptInvalid(
            "observation.session_id must equal the receipt native_session_id (correlation)"
        )
    _utc(observation["observed_at"], label="observation.observed_at")
    _text(
        observation["provider_version"],
        label="observation.provider_version",
        maximum=96,
    )
    model = _optional_text(observation["model"], label="observation.model")
    effort = _optional_text(observation["effort"], label="observation.effort")
    if effort is not None and effort not in CODEX_EFFORT_VOCABULARY:
        raise RouteObservationReceiptInvalid(
            f"observation.effort must be one of {sorted(CODEX_EFFORT_VOCABULARY)}; "
            "a suffix outside the closed vocabulary is never guessed"
        )
    floor = _closed_mapping(observation["render_floor"], _RENDER_FLOOR_FIELDS, label="render_floor")
    width = floor["width"]
    if width is not None and (not isinstance(width, int) or isinstance(width, bool) or width <= 0):
        raise RouteObservationReceiptInvalid(
            "render_floor.width must be a positive integer or null"
        )
    _digest(observation["evidence_sha256"], label="observation.evidence_sha256")
    return {
        **observation,
        "session_id": session_id,
        "model": model,
        "effort": effort,
    }


def _validate_close_proof(value: Any) -> dict[str, Any]:
    """The proven owned close on the non-modal surface.

    For a positive receipt the close must be proven ``composer-restored``;
    the close action is always ``none`` (no modal, no ``Escape`` on the
    Codex non-modal surface).
    """
    proof = _closed_mapping(value, _CLOSE_PROOF_FIELDS, label="close_proof")
    if proof["kind"] != "owned-close":
        raise RouteObservationReceiptInvalid("close_proof.kind must be 'owned-close'")
    if proof["surface"] != CLOSE_SURFACE:
        raise RouteObservationReceiptInvalid(
            f"close_proof.surface must be {CLOSE_SURFACE!r} (the Codex surface is non-modal)"
        )
    if proof["close_action"] != CLOSE_ACTION_NONE:
        raise RouteObservationReceiptInvalid(
            f"close_proof.close_action must be {CLOSE_ACTION_NONE!r}; "
            "no Escape exists on the non-modal surface"
        )
    if proof["outcome"] != CLOSE_OUTCOME_COMPOSER_RESTORED:
        raise RouteObservationReceiptInvalid(
            "a positive receipt requires a proven composer-restored close"
        )
    _utc(proof["closed_at"], label="close_proof.closed_at")
    return proof


def validate_receipt(value: Any) -> dict[str, Any]:
    """Validate and return a detached JSON-safe copy of one positive receipt.

    A positive receipt is deliberately strict: it proves the correlated
    observed-closed result, the exact build binding, and the exact requester,
    and it never copies raw screen text.
    """
    receipt = _closed_mapping(value, _TOP_LEVEL_FIELDS, label="receipt")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise RouteObservationReceiptInvalid(f"schema must be {RECEIPT_SCHEMA!r}")
    if receipt["kind"] != KIND_OBSERVED_CLOSED:
        raise RouteObservationReceiptInvalid(
            f"kind must be {KIND_OBSERVED_CLOSED!r}; a canary never mints a partial receipt"
        )
    operation_id = _uuid(receipt["operation_id"], label="operation_id")
    _digest(receipt["request_digest"], label="request_digest")
    requester_terminal_id = _identifier(
        receipt["requester_terminal_id"], label="requester_terminal_id"
    )
    requester_generation = _identifier(
        receipt["requester_generation"], label="requester_generation"
    )
    target_terminal_id = _identifier(receipt["target_terminal_id"], label="target_terminal_id")
    target_generation = _identifier(receipt["target_generation"], label="target_generation")
    native_session_id = _identifier(receipt["native_session_id"], label="native_session_id")
    _identifier(receipt["provider"], label="provider")
    _text(receipt["provider_version"], label="provider_version", maximum=96)
    _digest(receipt["provider_artifact_sha256"], label="provider_artifact_sha256")
    _digest(receipt["final_event_digest"], label="final_event_digest")
    _utc(receipt["committed_at"], label="committed_at")

    observation = _validate_observation(receipt["observation"], native_session_id=native_session_id)
    close_proof = _validate_close_proof(receipt["close_proof"])

    clean = {
        **receipt,
        "operation_id": operation_id,
        "requester_terminal_id": requester_terminal_id,
        "requester_generation": requester_generation,
        "target_terminal_id": target_terminal_id,
        "target_generation": target_generation,
        "native_session_id": native_session_id,
        "observation": observation,
        "close_proof": close_proof,
    }

    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_RECEIPT_BYTES:
        raise RouteObservationReceiptInvalid("receipt exceeds the 16 KiB evidence bound")
    return cast(dict[str, Any], json.loads(encoded))


def receipt_digest(value: Any) -> str:
    """The canonical SHA-256 digest of one validated positive receipt."""
    receipt = validate_receipt(value)
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_receipt(path: Path, value: Any) -> Path:
    """Validate and write one receipt as sorted, indented JSON."""
    receipt = validate_receipt(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
