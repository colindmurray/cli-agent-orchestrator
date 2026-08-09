"""The gate-2 receipt state: what the server reads to know which interval it is in.

Two rules turn on the fork observing *this deployment's gate-2 receipt records
both proofs* — when the ordinary-operation bypass ends, and when a proof
designation stops being honored. The campaign's own gate-2 receipt is record
discipline for humans; it was never an artifact a server consults. This file is.

It is **a pointer and a digest, never the evidence.** Writing it performs no
proof, closes no gate, enables no capability, mints no authority, and advances
the ordered gates by nothing. Its only two effects are to satisfy the first of
the two bypass-end conditions and to close the designation honor-window. Even
forged it leaves the bypass running unless the gate-6 binding is *also*
installed, and it adds **no new path to a `(1, 1)` mint** — it can only change
which interval a deployment believes it is in.

Three properties stop mere presence from standing in for the two deployment
proofs: ``proofs_recorded`` must name **both** of them, so a partial list
satisfies nothing; ``receipt_sha256`` binds the file to the canonical receipt;
and the key set is exact, so a file that means something else cannot be read as
this one.

Read **once, at server start**. Absent is the safe default and keeps the bypass
running. Malformed or digest-mismatched **refuses server start** — the same
server-scoped, singular blast radius a malformed designation has, and
deliberately unlike a malformed per-project binding, which yields ``UNKNOWN``
for that one project.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Beside the channel socket, the gate-2 designation, and the project-state
#: bindings, in the server's own state root.
RECEIPT_STATE_BASENAME = "gate2-proof-receipt-state.json"

#: The canonical receipt this file points at, when it is present beside it.
CANONICAL_RECEIPT_BASENAME = "gate2-proof-receipt.json"

RECEIPT_STATE_SCHEMA = "cao-gate2-proof-receipt-state-v1"

#: Exactly these three keys. Not a minimum — a superset is malformed too.
_REQUIRED_KEYS = frozenset({"schema", "receipt_sha256", "proofs_recorded"})

#: The two deployment proofs §14 gate 2 closes on. Both must be named.
PROOF_LINEAGE_ISOLATION = "lineage-isolation"
PROOF_SUPERVISOR_CREATION_DISCRIMINATOR = "supervisor-creation-discriminator"
REQUIRED_PROOFS = (PROOF_LINEAGE_ISOLATION, PROOF_SUPERVISOR_CREATION_DISCRIMINATOR)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ReceiptStateError(RuntimeError):
    """The receipt state exists but is not usable. Server start fails closed."""


@dataclass(frozen=True)
class Gate2ReceiptState:
    """A recorded gate-2 receipt state, already validated.

    Its existence means only that a deployment *recorded* both proofs. Whether
    the proofs were genuinely performed is §14's process, owned by the fork lane
    and ops, and nothing here can substitute for it.
    """

    receipt_sha256: str
    proofs_recorded: tuple[str, ...]
    sha256: str
    path: str

    @property
    def records_both_proofs(self) -> bool:
        return set(self.proofs_recorded) == set(REQUIRED_PROOFS)


def receipt_state_path() -> Path:
    from cli_agent_orchestrator.constants import CAO_HOME_DIR

    return Path(CAO_HOME_DIR) / RECEIPT_STATE_BASENAME


def canonical_receipt_path() -> Path:
    from cli_agent_orchestrator.constants import CAO_HOME_DIR

    return Path(CAO_HOME_DIR) / CANONICAL_RECEIPT_BASENAME


def load_receipt_state(path: Optional[Path] = None) -> Optional[Gate2ReceiptState]:
    """Read the receipt state at server start, or ``None`` when there is none.

    ``None`` is returned **only** for genuine absence, which leaves the bypass
    running. Everything else raises, because an operator who wrote this file
    believes the deployment has moved past gate 2, and starting as though it
    were absent would silently contradict that.
    """
    target = path if path is not None else receipt_state_path()

    try:
        info = target.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReceiptStateError(f"{target} cannot be inspected: {exc}") from exc

    if not stat.S_ISREG(info.st_mode):
        raise ReceiptStateError(f"{target} is not a regular file")

    mode = stat.S_IMODE(info.st_mode)
    if mode != 0o600:
        raise ReceiptStateError(f"{target} has mode {mode:04o}; the receipt state must be 0600")

    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise ReceiptStateError(f"{target} cannot be read: {exc}") from exc

    own_digest = hashlib.sha256(raw).hexdigest()

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptStateError(f"{target} is not valid UTF-8 JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ReceiptStateError(f"{target} must contain a JSON object")

    keys = set(parsed)
    if keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - keys)
        extra = sorted(keys - _REQUIRED_KEYS)
        raise ReceiptStateError(
            f"{target} must carry exactly {sorted(_REQUIRED_KEYS)}; "
            f"missing={missing} unexpected={extra}"
        )

    if parsed["schema"] != RECEIPT_STATE_SCHEMA:
        raise ReceiptStateError(
            f"{target} declares schema {parsed['schema']!r}; only "
            f"{RECEIPT_STATE_SCHEMA!r} is accepted"
        )

    digest = parsed["receipt_sha256"]
    if not isinstance(digest, str) or not _HEX64.match(digest):
        raise ReceiptStateError(
            f"{target} carries receipt_sha256 {digest!r}, which is not 64 "
            "lowercase hex characters. An uppercase or truncated digest is "
            "refused rather than normalized, because this value is the binding "
            "to the canonical receipt."
        )

    proofs = parsed["proofs_recorded"]
    if not isinstance(proofs, list) or not all(isinstance(item, str) for item in proofs):
        raise ReceiptStateError(f"{target} must carry proofs_recorded as a list of strings")
    if set(proofs) != set(REQUIRED_PROOFS) or len(proofs) != len(REQUIRED_PROOFS):
        raise ReceiptStateError(
            f"{target} records proofs {proofs!r}; it must name exactly "
            f"{list(REQUIRED_PROOFS)}. A partial list satisfies nothing — this "
            "is what stops mere file presence from standing in for the two "
            "deployment proofs."
        )

    _verify_against_canonical_receipt(target, digest)

    logger.info("gate-2 receipt state loaded: receipt_sha256=%s proofs=%s", digest, proofs)
    return Gate2ReceiptState(
        receipt_sha256=digest,
        proofs_recorded=tuple(proofs),
        sha256=own_digest,
        path=str(target),
    )


def _verify_against_canonical_receipt(state_path: Path, declared_digest: str) -> None:
    """Verify the canonical receipt **unconditionally**. No verify-if-present.

    Once the receipt is load-bearing, "check it when it happens to be reachable"
    is not a check: it makes the strength of the binding depend on whether a file
    was copied. So with a receipt state present, the canonical receipt must be
    present, readable, of the right schema, and hash to the declared digest —
    anything else **refuses server start**.

    The earlier best-effort branch is withdrawn deliberately. Its residual was
    that a forged receipt state bearing any well-formed digest was accepted on a
    deployment where the receipt had never been placed; now that state cannot
    start a server at all.
    """
    from cli_agent_orchestrator.services import gate2_proof_receipt

    receipt = canonical_receipt_path()
    try:
        raw = receipt.read_bytes()
    except FileNotFoundError:
        raise ReceiptStateError(
            f"{state_path} is present, so the canonical gate-2 receipt must be at "
            f"{receipt}; it is absent. A receipt state without its receipt names "
            "evidence this deployment does not hold."
        ) from None
    except OSError as exc:
        raise ReceiptStateError(f"{receipt} cannot be read: {exc}") from exc

    # The referent is validated in **full**, not merely by schema tag and digest.
    # A digest only proves the bytes are the ones the pointer named; it says
    # nothing about whether those bytes are a receipt the writer would ever have
    # emitted. Checking the tag alone let a hand-written document that
    # `validate_receipt` refuses — an unproven outcome, a non-disposable
    # instance, an incomplete teardown — be accepted at start, so the strongest
    # checks in the codec applied only to documents this process happened to
    # write itself.
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptStateError(f"{receipt} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ReceiptStateError(f"{receipt} must contain a JSON object")

    schema = parsed.get("schema")
    if schema != gate2_proof_receipt.RECEIPT_SCHEMA:
        raise ReceiptStateError(
            f"{receipt} declares schema {schema!r}; only "
            f"{gate2_proof_receipt.RECEIPT_SCHEMA!r} may be a receipt-state referent. "
            "A partial can never attest."
        )

    try:
        gate2_proof_receipt.validate_receipt(parsed)
    except gate2_proof_receipt.ReceiptError as exc:
        raise ReceiptStateError(
            f"{receipt} is not a valid canonical gate-2 receipt: {exc}"
        ) from exc

    actual = hashlib.sha256(raw).hexdigest()
    if actual != declared_digest:
        raise ReceiptStateError(
            f"{state_path} declares receipt_sha256 {declared_digest} but "
            f"{receipt} hashes to {actual}. The state file must point at the "
            "canonical receipt it names."
        )


def receipt_fields(state: Optional[Gate2ReceiptState]) -> dict:
    """The receipt-state facts an audit record carries, absence stated positively."""
    if state is None:
        return {
            "gate2_receipt_state_present": False,
            "gate2_receipt_state_sha256": None,
            "gate2_receipt_sha256": None,
            "gate2_proofs_recorded": None,
        }
    return {
        "gate2_receipt_state_present": True,
        "gate2_receipt_state_sha256": state.sha256,
        "gate2_receipt_sha256": state.receipt_sha256,
        "gate2_proofs_recorded": list(state.proofs_recorded),
    }


def write_receipt_state_for_proof_run(
    path: Path, receipt_sha256: str, proofs: tuple[str, ...] = REQUIRED_PROOFS
) -> Gate2ReceiptState:
    """Write a receipt state with the mode the loader requires.

    For the operator-side proof tool and for tests. **Not** reachable from any
    API path, channel verb, or request — nothing in the HTTP or channel surface
    imports it, which is what keeps this artifact out of caller control.
    """
    import os

    payload = json.dumps(
        {
            "schema": RECEIPT_STATE_SCHEMA,
            "receipt_sha256": receipt_sha256,
            "proofs_recorded": list(proofs),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)
    return Gate2ReceiptState(
        receipt_sha256=receipt_sha256,
        proofs_recorded=tuple(proofs),
        sha256=hashlib.sha256(payload).hexdigest(),
        path=str(path),
    )
