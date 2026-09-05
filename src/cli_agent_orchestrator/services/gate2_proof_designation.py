"""The gate-2 proof designation: an endpoint-less, operator-written surface.

Gate 2 closes only on deployment proofs that exercise the whole authority
pipeline — allocation, phase-C re-verification, teardown, epoch non-reuse. But
the pre-closure rule says none of that runs until gate 2 is proven. That is a
circularity, and the way out is that the pipeline runs pre-closure for exactly
one **disposable proof project**, and nowhere else.

Something has to say which project that is, and the choice of surface is the
keystone. It cannot be a request field, a channel verb, a flag, an environment
variable, or a caller assertion: each of those is the caller-controlled door the
channel exists to remove, and admitting one here would hand a caller the ability
to select the project whose pipeline runs. So the designation is a durable file
in the server's own state root that **no API path, channel verb, or request can
create, alter, read back, or reference**. There is no endpoint for it, and this
module exposes no reader that takes caller input.

It is **not a credential.** It confers nothing on anybody and cannot be
presented or quoted. It selects *which project is scratch*, never *who may
act* — the operator-origin predicate still gates admission unchanged, so a
designation admits no one.

Read **once, at server start**, so it cannot be flipped mid-run to change a
decision already in flight. **Absent is the default and the safe state**: with
no designation the ordinary bypass applies to every project and the pipeline
runs nowhere.

**Declared limitation.** A same-UID process with write access to the state root
can create or alter this file. That yields no authority — a managed worker still
cannot pass A0 — so the worst case is a *real* project's pipeline running before
its deployment's receipt exists: a state-hygiene fault, not an authority hijack,
and one the exact-match rule and the recorded digest make visible rather than
silent. The recommended practice removes the residual entirely: run the proof on
a disposable instance with its own ``CAO_STATE_ROOT``, so the designation cannot
sit beside any real project's state at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Beside the channel socket and its lock, in the server's own state root.
DESIGNATION_BASENAME = "gate2-proof-designation.json"

#: The only accepted schema tag, and the only accepted key set. A designation is
#: refused rather than interpreted, so an unexpected shape can never be read as
#: naming a project.
DESIGNATION_SCHEMA = "cao-gate2-proof-designation-v1"

_REQUIRED_KEYS = frozenset({"schema", "project"})


class DesignationError(RuntimeError):
    """The designation exists but is not usable. Startup fails closed.

    Distinct from absence: absence is the ordinary safe default, while a
    malformed, ambiguous, or multi-project designation is an operator mistake
    that must be corrected rather than guessed at.
    """


@dataclass(frozen=True)
class Gate2ProofDesignation:
    """One exact project, plus the digest a gate-2 receipt must record."""

    project: str
    sha256: str
    path: str


def designation_path() -> Path:
    """``<state-root>/gate2-proof-designation.json``, by the one state-root rule."""
    from cli_agent_orchestrator.constants import CAO_HOME_DIR

    return Path(CAO_HOME_DIR) / DESIGNATION_BASENAME


def load_designation(path: Optional[Path] = None) -> Optional[Gate2ProofDesignation]:
    """Read the designation at server start, or ``None`` when there is none.

    Returns ``None`` **only** for genuine absence. Every other problem raises,
    because a designation that cannot be read exactly is an operator error and
    silently continuing would mean the operator believes a proof run is scoped
    when it is not.
    """
    target = path if path is not None else designation_path()

    try:
        info = target.lstat()
    except FileNotFoundError:
        # The default, and the safe state: the pipeline runs nowhere.
        return None
    except OSError as exc:
        raise DesignationError(f"{target} cannot be inspected: {exc}") from exc

    if not stat.S_ISREG(info.st_mode):
        raise DesignationError(f"{target} is not a regular file")

    mode = stat.S_IMODE(info.st_mode)
    if mode != 0o600:
        raise DesignationError(
            f"{target} has mode {mode:04o}; the designation must be 0600 so it is "
            "not readable or writable beyond its owner"
        )

    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise DesignationError(f"{target} cannot be read: {exc}") from exc

    digest = hashlib.sha256(raw).hexdigest()

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DesignationError(f"{target} is not valid UTF-8 JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise DesignationError(f"{target} must contain a JSON object naming one project")

    keys = set(parsed)
    if keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - keys)
        extra = sorted(keys - _REQUIRED_KEYS)
        raise DesignationError(
            f"{target} must carry exactly {sorted(_REQUIRED_KEYS)}; "
            f"missing={missing} unexpected={extra}. An unexpected key is refused "
            "rather than ignored, because a designation is never interpreted."
        )

    if parsed["schema"] != DESIGNATION_SCHEMA:
        raise DesignationError(
            f"{target} declares schema {parsed['schema']!r}; only "
            f"{DESIGNATION_SCHEMA!r} is accepted"
        )

    project = parsed["project"]
    # A list, an object, or a comma-joined string are the three shapes an
    # operator might reach for to name more than one project. All are refused:
    # "exact-match to one project" is the property that keeps every other
    # project on the deployment under the ordinary bypass.
    if not isinstance(project, str):
        raise DesignationError(
            f"{target} names a {type(project).__name__} project; exactly one "
            "project string is required, and a multi-project designation is "
            "refused rather than interpreted"
        )
    if not project.strip():
        raise DesignationError(f"{target} names an empty project")
    if project != project.strip():
        raise DesignationError(
            f"{target} names {project!r}, which has surrounding whitespace; an "
            "ambiguous project name is refused rather than trimmed"
        )
    if "," in project or "\n" in project:
        raise DesignationError(
            f"{target} names {project!r}, which looks like several projects; "
            "exactly one project may be designated"
        )

    logger.info("gate-2 proof designation loaded: project=%s sha256=%s", project, digest)
    return Gate2ProofDesignation(project=project, sha256=digest, path=str(target))


def receipt_fields(
    designation: Optional[Gate2ProofDesignation],
    *,
    window_opened_at: Optional[str] = None,
    window_closed_at: Optional[str] = None,
) -> dict:
    """The designation facts a gate-2 proof receipt must record.

    The digest and the project are what let a reviewer confirm the proof ran
    where it was declared to and nowhere else; the window is what bounds it in
    time. Emitted even when there is no designation, so a receipt can state
    that absence positively rather than by omission.
    """
    if designation is None:
        return {
            "gate2_proof_designation_present": False,
            "gate2_proof_designation_sha256": None,
            "gate2_proof_designation_project": None,
            "gate2_proof_designation_window_opened_at": None,
            "gate2_proof_designation_window_closed_at": None,
        }
    return {
        "gate2_proof_designation_present": True,
        "gate2_proof_designation_sha256": designation.sha256,
        "gate2_proof_designation_project": designation.project,
        "gate2_proof_designation_window_opened_at": window_opened_at,
        "gate2_proof_designation_window_closed_at": window_closed_at,
    }


def write_designation_for_proof_run(path: Path, project: str) -> Gate2ProofDesignation:
    """Write a designation with the exact mode the loader requires.

    Provided for the operator-side proof tool and for tests. It is **not**
    reachable from any API path, channel verb, or request — nothing in the HTTP
    or channel surface imports it, which is the property that keeps the
    designation out of caller control. Callers of this function already hold
    filesystem access to the state root, which is the authority it assumes.
    """
    payload = json.dumps(
        {"schema": DESIGNATION_SCHEMA, "project": project}, separators=(",", ":")
    ).encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)
    return Gate2ProofDesignation(
        project=project, sha256=hashlib.sha256(payload).hexdigest(), path=str(path)
    )
