"""The project-state binding: where the fork looks, never what it is told.

The existing-run branch turns on whether a prior run of a project survives, and
that fact lives in the conductor's ``project.json`` — state this server cannot
see. The fix is not to transmit the answer. **The witness value is never
transmitted; only where to look is bound, and the fork derives the trit by its
own observation.** That is what makes it authenticated by construction rather
than asserted: a caller cannot choose the answer, because a caller never supplies
one. A witness value appearing in any request is refused, not ignored.

**`ABSENT` requires positive proof** — a readable, existing project-state
directory that demonstrably lacks ``project.json``. A missing path, an unreadable
one, a non-directory, a malformed binding, or no binding at all is ``UNKNOWN``.
Absence of evidence is never evidence of absence, and ``UNKNOWN`` refuses.

**Blast radius, deliberately smaller than the server-scoped artifacts.** A
malformed designation or receipt state refuses server start: each is singular and
server-wide. A malformed binding yields ``UNKNOWN`` for **that project only** and
does not refuse start, because bindings are per-project and one bad file must not
take every other project's bring-up down with it. Both are fail-closed; they
differ only in how much they stop.

*Ownership.* The conductor lane owns producing ``project.json``; the operator or
installer owns establishing the binding, exactly as it owns the gate-2
designation. This module owns only the observation and the trit.
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

#: One file per project, in the server's own state root.
BINDINGS_DIRNAME = "project-state-bindings"

BINDING_SCHEMA = "cao-project-state-binding-v1"

#: Exactly these three keys.
_REQUIRED_KEYS = frozenset({"schema", "project", "project_state_dir"})

#: The conductor-side artifact that is the authoritative run identity. Nothing
#: else may stand in for it.
PROJECT_JSON_BASENAME = "project.json"


@dataclass(frozen=True)
class WitnessObservation:
    """What phase A observed, and on what basis.

    ``(1, 1)`` is the most consequential value this protocol mints, so an
    auditor must be able to see why. That is the whole reason this record exists
    rather than a bare trit: the digest, or the positively observed absence, is
    the basis.
    """

    #: "absent" | "present" | "unknown" — the trit as observed.
    witness: str
    #: Where the fork looked, or ``None`` when there was nowhere to look.
    source_path: Optional[str]
    #: The observed ``project.json`` digest, when one was there to hash.
    project_json_sha256: Optional[str]
    #: True only when a readable directory was seen to *lack* ``project.json``.
    project_json_observed_absent: bool
    #: Why the observation came out as it did, for the audit trail.
    detail: Optional[str] = None

    def as_provenance(self) -> dict:
        return {
            "witness": self.witness,
            "witness_source_path": self.source_path,
            "witness_project_json_sha256": self.project_json_sha256,
            "witness_project_json_observed_absent": self.project_json_observed_absent,
            "witness_detail": self.detail,
        }


def bindings_dir() -> Path:
    from cli_agent_orchestrator.constants import CAO_HOME_DIR

    return Path(CAO_HOME_DIR) / BINDINGS_DIRNAME


def binding_path(project: str) -> Path:
    return bindings_dir() / f"{project}.json"


def binding_recorded(project: str) -> bool:
    """Whether a binding artifact exists for this project.

    Existence only — this is the second of the two bypass-end conditions, and it
    asks whether the gate-6 mechanism has landed for this project, not whether
    the binding is well-formed. A malformed binding is *recorded* (so the bypass
    ends) and then observes ``UNKNOWN`` (so authority still refuses). Those are
    deliberately separate questions: conflating them would let a bad file
    silently reinstate the bypass and mint nothing while looking healthy.
    """
    if not project:
        return False
    try:
        return binding_path(project).is_file()
    except OSError:
        return False


def observe_witness(project: str) -> WitnessObservation:
    """Derive the trit for a project by looking where its binding points.

    Never raises. Every failure is ``UNKNOWN`` with a detail naming the cause, so
    one project's broken binding cannot affect another's bring-up or the server's
    start.
    """
    if not project:
        return WitnessObservation(
            witness="unknown",
            source_path=None,
            project_json_sha256=None,
            project_json_observed_absent=False,
            detail="binding-absent",
        )

    path = binding_path(project)
    parsed = _load_binding(path)
    if parsed is None:
        return WitnessObservation(
            witness="unknown",
            source_path=None,
            project_json_sha256=None,
            project_json_observed_absent=False,
            detail="binding-absent" if not path.exists() else "binding-malformed",
        )

    state_dir = Path(parsed["project_state_dir"])

    try:
        info = state_dir.lstat()
    except OSError:
        # A bound path that does not exist or cannot be inspected proves
        # nothing about whether a prior run existed.
        return WitnessObservation(
            witness="unknown",
            source_path=str(state_dir),
            project_json_sha256=None,
            project_json_observed_absent=False,
            detail="project-state-dir-unreadable",
        )

    if not stat.S_ISDIR(info.st_mode):
        return WitnessObservation(
            witness="unknown",
            source_path=str(state_dir),
            project_json_sha256=None,
            project_json_observed_absent=False,
            detail="project-state-dir-not-a-directory",
        )

    if not os.access(state_dir, os.R_OK | os.X_OK):
        return WitnessObservation(
            witness="unknown",
            source_path=str(state_dir),
            project_json_sha256=None,
            project_json_observed_absent=False,
            detail="project-state-dir-unreadable",
        )

    project_json = state_dir / PROJECT_JSON_BASENAME
    try:
        raw = project_json.read_bytes()
    except FileNotFoundError:
        # A readable directory demonstrably lacking project.json. This is the
        # only positive proof of a fresh project, and it is what admits (1, 1).
        return WitnessObservation(
            witness="absent",
            source_path=str(project_json),
            project_json_sha256=None,
            project_json_observed_absent=True,
            detail="project-json-observed-absent",
        )
    except OSError:
        return WitnessObservation(
            witness="unknown",
            source_path=str(project_json),
            project_json_sha256=None,
            project_json_observed_absent=False,
            detail="project-json-unreadable",
        )

    return WitnessObservation(
        witness="present",
        source_path=str(project_json),
        project_json_sha256=hashlib.sha256(raw).hexdigest(),
        project_json_observed_absent=False,
        detail="project-json-observed",
    )


def _load_binding(path: Path) -> Optional[dict]:
    """Parse a binding, or ``None`` when it is absent or malformed.

    Malformed is not an exception here on purpose: it must degrade this one
    project to ``UNKNOWN``, not refuse the server.
    """
    try:
        info = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        logger.warning("project-state binding %s is not a regular file", path)
        return None
    if stat.S_IMODE(info.st_mode) != 0o600:
        logger.warning("project-state binding %s is not mode 0600", path)
        return None

    try:
        parsed = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("project-state binding %s is not valid UTF-8 JSON", path)
        return None

    if not isinstance(parsed, dict) or set(parsed) != _REQUIRED_KEYS:
        logger.warning("project-state binding %s does not carry exactly three keys", path)
        return None
    if parsed["schema"] != BINDING_SCHEMA:
        logger.warning("project-state binding %s declares an unknown schema", path)
        return None

    project = parsed["project"]
    state_dir = parsed["project_state_dir"]
    for value in (project, state_dir):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            logger.warning("project-state binding %s has an empty or padded field", path)
            return None
        if "\x00" in value:
            logger.warning("project-state binding %s contains a NUL byte", path)
            return None
    if not os.path.isabs(state_dir):
        logger.warning("project-state binding %s names a relative path", path)
        return None
    if path.stem != project:
        # The filename and the declared project must agree, or one binding could
        # answer for another project.
        logger.warning("project-state binding %s names a different project", path)
        return None

    return parsed


def write_binding_for_project(path: Path, project: str, project_state_dir: str) -> None:
    """Write a binding with the mode the reader requires.

    For the operator/installer and for tests. Not reachable from any API path,
    channel verb, or request.
    """
    payload = json.dumps(
        {
            "schema": BINDING_SCHEMA,
            "project": project,
            "project_state_dir": project_state_dir,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)
