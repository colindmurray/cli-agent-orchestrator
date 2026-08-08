"""The canonical gate-2 proof receipt: one byte sequence, three readers.

The receipt state's ``receipt_sha256`` commits to *these bytes*, and the server
refuses to start when the digest does not match. So the bytes are a three-party
contract — the proof runner writes them, the fork server hashes them, a reviewer
reads them — and every choice here is about making those three agree exactly.

**Byte-exactness is the whole point**, which is why this module encodes by hand
instead of reaching for ``json.dumps``. The order is pinned per object, never
lexicographic and never insertion-luck; control characters are always the long
``\\u00xx`` form rather than JSON's short escapes; and there is exactly one
trailing ``0x0a``. A second serialization of the same facts is a second artifact,
so nothing here ever re-serializes a receipt it has already written.

**Redaction happens before the digest**, so only one byte sequence ever exists.
It replaces enumerated free-text values with a fixed placeholder and **never
deletes a key or an object** — which is what lets "archive a redacted receipt"
and "close a gate on that receipt" both be true. Because the key set is closed
and every field required, no proof outcome, step fact, capability-dark value, or
non-effect count can be redacted away.

**Presence is not enough for success.** A receipt recording
``is_disposable_instance: false`` or ``instance_destroyed: false`` would satisfy a
closed-schema check, the digest, and the server's verification, leaving only a
human to notice that the proof ran on a non-disposable instance or never cleaned
up. Those values are therefore part of the schema contract and are refused here,
at write time, exactly as a wrong type is.

The **partial** is a structurally different document, not a flagged receipt. It
has no ``proofs`` key at all, so it cannot supply ``proofs_recorded``, cannot be
pointed at by a receipt state, and cannot be misread as success by anybody.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

RECEIPT_SCHEMA = "cao-gate2-proof-receipt-v1"
PARTIAL_SCHEMA = "cao-gate2-proof-partial-v1"

RECEIPT_BASENAME = "gate2-proof-receipt.json"
PARTIAL_BASENAME = "gate2-proof-partial.json"

#: The fixed placeholder. Redaction substitutes it; it never removes anything.
REDACTED = "[redacted]"

# --- pinned orders -------------------------------------------------------
# Each tuple is both the required key set and the canonical order. Treating the
# two as one thing is deliberate: a key set that drifts from its order silently
# changes the digest.

RECEIPT_ORDER = (
    "schema",
    "target",
    "isolation",
    "designation",
    "capability_dark",
    "ordinary_project_non_effect",
    "proofs",
    "steps",
    "teardown",
    "identities",
)

PARTIAL_ORDER = (
    "schema",
    "target",
    "isolation",
    "observed_fact_names",
    "unobserved_fact_names",
    "refusal_reason",
    "steps",
    "teardown",
    "identities",
)

TARGET_ORDER = ("fork_source_sha256", "deployed_artifact_sha256", "server_start_id")
ISOLATION_ORDER = ("state_root", "proof_project", "is_disposable_instance")
DESIGNATION_ORDER = ("sha256", "project", "opened_at", "closed_at")
CAPABILITY_DARK_ORDER = (
    "advertisement_enabled",
    "provider_tuples_enabled",
    "listener_enabled_for_ordinary_projects",
)
NON_EFFECT_ORDER = (
    "ordinary_projects_observed",
    "authority_rows_created",
    "epoch_allocations_created",
    "grants_minted",
    "project_json_files_touched",
)
PROOFS_ORDER = ("lineage_isolation", "supervisor_creation_discriminator")
PROOF_ENTRY_ORDER = ("outcome", "evidence_refs", "observed_at")
STEP_ORDER = (
    "seq",
    "command",
    "outcome",
    "reason_code",
    "reason_detail",
    "terminal_created",
    "terminal_torn_down",
    "epoch_allocated",
    "epoch_reused",
    "observed_at",
)
TEARDOWN_ORDER = ("instance_destroyed", "state_root_removed", "artifacts_deleted")
IDENTITIES_ORDER = ("operator_ref", "tool_version", "spec_head", "reviewed_source_head")

OUTCOME_PROVEN = "proven"


class ReceiptError(RuntimeError):
    """The receipt or partial does not satisfy its closed contract."""


# --- canonical encoding (§5.4, verbatim) ---------------------------------


def _escape(text: str) -> str:
    """Escape per §5.4: ``"`` and ``\\``, and control characters as ``\\u00xx``.

    The long form is used for *every* control character, including newline and
    tab, rather than JSON's short escapes — two encoders that disagree about
    ``\\n`` versus ``\\u000a`` produce different bytes and therefore different
    digests.
    """
    out = []
    for char in text:
        if char == '"':
            out.append('\\"')
        elif char == "\\":
            out.append("\\\\")
        elif ord(char) < 0x20:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return "".join(out)


def _encode_value(value: Any, order: Optional[Sequence[str]] = None) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if value < 0:
            raise ReceiptError("negative integers are outside the canonical form")
        return str(value)
    if isinstance(value, str):
        return f'"{_escape(value)}"'
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if order is None:
            raise ReceiptError("an object cannot be encoded without its pinned order")
        return _encode_object(value, order)
    raise ReceiptError(f"{type(value).__name__} is not encodable in the canonical form")


def _encode_object(obj: Mapping[str, Any], order: Sequence[str]) -> str:
    parts = []
    for key in order:
        if key not in obj:
            raise ReceiptError(f"missing required key {key!r}")
        parts.append(f'"{_escape(key)}":{_encode_value(obj[key], _nested_order(key))}')
    return "{" + ",".join(parts) + "}"


def _nested_order(key: str) -> Optional[Sequence[str]]:
    return {
        "target": TARGET_ORDER,
        "isolation": ISOLATION_ORDER,
        "designation": DESIGNATION_ORDER,
        "capability_dark": CAPABILITY_DARK_ORDER,
        "ordinary_project_non_effect": NON_EFFECT_ORDER,
        "proofs": PROOFS_ORDER,
        "lineage_isolation": PROOF_ENTRY_ORDER,
        "supervisor_creation_discriminator": PROOF_ENTRY_ORDER,
        "teardown": TEARDOWN_ORDER,
        "identities": IDENTITIES_ORDER,
    }.get(key)


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    """Serialize a receipt or partial to its one canonical byte sequence.

    Exactly one trailing ``0x0a``, and the pinned order throughout — including
    inside ``steps``, whose entries are objects with their own fixed order.
    """
    schema = document.get("schema")
    if schema == RECEIPT_SCHEMA:
        order: Sequence[str] = RECEIPT_ORDER
    elif schema == PARTIAL_SCHEMA:
        order = PARTIAL_ORDER
    else:
        raise ReceiptError(f"unknown schema {schema!r}")

    parts = []
    for key in order:
        if key not in document:
            raise ReceiptError(f"missing required key {key!r}")
        value = document[key]
        if key == "steps":
            parts.append(f'"{key}":' + _encode_steps(value))
        else:
            parts.append(f'"{key}":{_encode_value(value, _nested_order(key))}')
    return ("{" + ",".join(parts) + "}\n").encode("utf-8")


def _encode_steps(steps: Any) -> str:
    if not isinstance(steps, (list, tuple)):
        raise ReceiptError("steps must be an array")
    return "[" + ",".join(_encode_object(step, STEP_ORDER) for step in steps) + "]"


def digest(document_bytes: bytes) -> str:
    return hashlib.sha256(document_bytes).hexdigest()


# --- redaction (closed, non-deleting) ------------------------------------


def redact(document: Mapping[str, Any]) -> dict:
    """Replace only the enumerated free-text values. Never delete a key.

    The enumerated fields are exactly ``steps[].command``,
    ``identities.operator_ref``, and ``proofs.*.evidence_refs[]``. Nothing else
    may be redacted, and because the key set is closed and every field required,
    the result is still a **complete** receipt: no proof outcome, step fact,
    capability-dark value, or non-effect count can be redacted away.
    """
    out = {key: value for key, value in document.items()}

    steps = out.get("steps")
    if isinstance(steps, (list, tuple)):
        out["steps"] = [
            {key: (REDACTED if key == "command" else value) for key, value in step.items()}
            for step in steps
        ]

    identities = out.get("identities")
    if isinstance(identities, Mapping):
        out["identities"] = {
            key: (REDACTED if key == "operator_ref" else value) for key, value in identities.items()
        }

    proofs = out.get("proofs")
    if isinstance(proofs, Mapping):
        redacted_proofs = {}
        for name, entry in proofs.items():
            if isinstance(entry, Mapping):
                redacted_proofs[name] = {
                    key: ([REDACTED for _ in value] if key == "evidence_refs" else value)
                    for key, value in entry.items()
                }
            else:
                redacted_proofs[name] = entry
        out["proofs"] = redacted_proofs

    return out


# --- validation ----------------------------------------------------------


def _require_exact_keys(obj: Any, order: Sequence[str], where: str) -> None:
    if not isinstance(obj, Mapping):
        raise ReceiptError(f"{where} must be an object")
    keys = set(obj)
    expected = set(order)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        raise ReceiptError(f"{where}: missing={missing} unknown={unknown}")
    for key in order:
        if obj[key] is None:
            raise ReceiptError(f"{where}.{key} is null; every field is required and non-null")


def validate_receipt(document: Mapping[str, Any]) -> None:
    """Validate a *successful* receipt against the closed contract.

    Rejects an unknown key, a missing key, a wrong ``schema``, a wrong type, or a
    ``null`` anywhere — and additionally rejects the pinned success values, which
    are part of the schema contract rather than a reviewer's job.
    """
    _require_exact_keys(document, RECEIPT_ORDER, "receipt")
    if document["schema"] != RECEIPT_SCHEMA:
        raise ReceiptError(f"receipt.schema must be {RECEIPT_SCHEMA!r}")

    _require_exact_keys(document["target"], TARGET_ORDER, "target")
    _require_exact_keys(document["isolation"], ISOLATION_ORDER, "isolation")
    _require_exact_keys(document["designation"], DESIGNATION_ORDER, "designation")
    _require_exact_keys(document["capability_dark"], CAPABILITY_DARK_ORDER, "capability_dark")
    _require_exact_keys(
        document["ordinary_project_non_effect"], NON_EFFECT_ORDER, "ordinary_project_non_effect"
    )
    _require_exact_keys(document["proofs"], PROOFS_ORDER, "proofs")
    _require_exact_keys(document["teardown"], TEARDOWN_ORDER, "teardown")
    _require_exact_keys(document["identities"], IDENTITIES_ORDER, "identities")

    for name in PROOFS_ORDER:
        _require_exact_keys(document["proofs"][name], PROOF_ENTRY_ORDER, f"proofs.{name}")
        entry = document["proofs"][name]
        if not isinstance(entry["evidence_refs"], (list, tuple)):
            raise ReceiptError(f"proofs.{name}.evidence_refs must be an array")
        if entry["outcome"] != OUTCOME_PROVEN:
            raise ReceiptError(
                f"proofs.{name}.outcome is {entry['outcome']!r}; a successful receipt "
                f"requires {OUTCOME_PROVEN!r}. A run that cannot say this writes a "
                "partial, not a receipt."
            )

    _validate_steps(document["steps"])

    if document["isolation"]["is_disposable_instance"] is not True:
        raise ReceiptError(
            "isolation.is_disposable_instance must be true; a proof run on a "
            "non-disposable instance is refused here rather than left to a reader"
        )

    for key in TEARDOWN_ORDER:
        if document["teardown"][key] is not True:
            raise ReceiptError(f"teardown.{key} must be true in a successful receipt")

    for key in CAPABILITY_DARK_ORDER:
        if document["capability_dark"][key] is not False:
            raise ReceiptError(f"capability_dark.{key} must be false")

    for key in NON_EFFECT_ORDER:
        value = document["ordinary_project_non_effect"][key]
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise ReceiptError(f"ordinary_project_non_effect.{key} must be zero")


def _validate_steps(steps: Any) -> None:
    if not isinstance(steps, (list, tuple)) or not steps:
        raise ReceiptError("steps must be a non-empty array")
    for index, step in enumerate(steps):
        _require_exact_keys(step, STEP_ORDER, f"steps[{index}]")
        if not isinstance(step["seq"], int) or isinstance(step["seq"], bool):
            raise ReceiptError(f"steps[{index}].seq must be an integer")


def validate_partial(document: Mapping[str, Any]) -> None:
    """Validate a partial — which must **not** carry a ``proofs`` key.

    Its ``teardown`` records what actually happened, so the pinned success values
    are deliberately *not* required here. A partial asserting them is still not a
    proof of anything; what makes it unable to attest is the absent ``proofs``
    key and the different schema tag, not the values it happens to hold.
    """
    if "proofs" in document:
        raise ReceiptError(
            "a partial must have no 'proofs' key; that absence is what makes it "
            "structurally unable to attest"
        )
    _require_exact_keys(document, PARTIAL_ORDER, "partial")
    if document["schema"] != PARTIAL_SCHEMA:
        raise ReceiptError(f"partial.schema must be {PARTIAL_SCHEMA!r}")
    _require_exact_keys(document["target"], TARGET_ORDER, "target")
    _require_exact_keys(document["isolation"], ISOLATION_ORDER, "isolation")
    _require_exact_keys(document["teardown"], TEARDOWN_ORDER, "teardown")
    _require_exact_keys(document["identities"], IDENTITIES_ORDER, "identities")
    for key in ("observed_fact_names", "unobserved_fact_names"):
        if not isinstance(document[key], (list, tuple)):
            raise ReceiptError(f"partial.{key} must be an array")
    _validate_steps(document["steps"])


# --- atomic write --------------------------------------------------------


def write_atomically(path: Path, document_bytes: bytes) -> None:
    """Temp-write, fsync the file, rename, fsync the directory, mode ``0600``.

    A partially written receipt must never be observable at the pinned path,
    because a reader that hashes a truncated file gets a digest that matches
    nothing and a server that refuses to start with no useful reason.
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory / f".{path.name}.tmp"

    descriptor = os.open(temp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, document_bytes)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(temp, 0o600)
    os.replace(temp, path)

    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def emit_receipt(path: Path, document: Mapping[str, Any]) -> tuple[bytes, str]:
    """Redact, validate, serialize, digest, and write. Returns ``(bytes, sha256)``.

    Redaction precedes the digest so there is only ever one byte sequence, and
    validation runs on the redacted document because that is what a reader and
    the server will see.
    """
    redacted = redact(document)
    validate_receipt(redacted)
    payload = canonical_bytes(redacted)
    write_atomically(path, payload)
    return payload, digest(payload)


def emit_partial(path: Path, document: Mapping[str, Any]) -> tuple[bytes, str]:
    redacted = redact(document)
    validate_partial(redacted)
    payload = canonical_bytes(redacted)
    write_atomically(path, payload)
    return payload, digest(payload)
