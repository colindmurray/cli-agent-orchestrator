"""``cao-gate2-proof-run`` — the ops-only gate-2 proof runner.

The fork lane authors and tests this; **ops alone executes it**, on the target
deployment's host, against an isolated disposable instance. No implementation
lease authorizes a live run, and nothing here closes a gate: a successful run
produces a receipt, and the gate closes later, on a reviewer reading that receipt.

**It is a local process an operator invokes, and that is the security property.**
It is reachable from no API path, no channel verb, no request field, no flag on a
running server, and no environment variable a worker can set — the same property
the designation has. It also never drives an already-running installed server: it
starts its own against the isolated root and stops it at the end, because a runner
that could attach to a live deployment would be a way to make a real server
perform proof steps.

**Refusals are refusals, not warnings.** A default, relative, or
ordinary-project-bearing state root is rejected before anything starts, and an
existing receipt or partial is a collision that is never overwritten — a second
run must not be able to quietly replace the artifact a reviewer is about to read.

**Exit contract:** ``0`` and a receipt written, or non-zero and a partial written.
Never both artifacts, never neither. A run that cannot say every required fact
writes the partial, which has no ``proofs`` key and can never attest.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from cli_agent_orchestrator.services import gate2_proof_receipt as codec

#: The default root a proof run must never touch. Named here rather than derived
#: so the refusal cannot drift if the default moves.
DEFAULT_ROOT_SUFFIX = os.path.join(".aws", "cli-agent-orchestrator")

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_REFUSED = 2


class ProofRunRefused(RuntimeError):
    """A precondition failed before any proof step ran. Nothing was written."""


@dataclass(frozen=True)
class ProofRunRequest:
    state_root: Path
    project: str
    designation: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cao-gate2-proof-run",
        description=(
            "Execute the gate-2 proof run against an isolated disposable instance "
            "and write a canonical receipt (exit 0) or a truthful partial (non-zero). "
            "Ops-only; closes no gate and enables no capability."
        ),
    )
    parser.add_argument("--state-root", required=True, help="absolute isolated state root")
    parser.add_argument("--project", required=True, help="the exact disposable proof project")
    parser.add_argument("--designation", required=True, help="path to the gate-2 designation")
    return parser


def validate_request(state_root: str, project: str, designation: str) -> ProofRunRequest:
    """Every precondition, evaluated before anything is started or written."""
    if not project or project != project.strip():
        raise ProofRunRefused("--project must be an exact non-empty name with no padding")

    if not state_root:
        raise ProofRunRefused("--state-root is required and must be stated explicitly")
    if not os.path.isabs(state_root):
        raise ProofRunRefused(
            f"--state-root {state_root!r} is relative; a relative root would follow the "
            "working directory of whichever process happened to start"
        )

    resolved = Path(os.path.realpath(state_root))

    default_root = Path(os.path.realpath(Path.home() / DEFAULT_ROOT_SUFFIX))
    if resolved == default_root:
        raise ProofRunRefused(
            f"--state-root resolves to the default root {resolved}; a proof run must "
            "use its own disposable root so its designation and authority rows cannot "
            "sit beside real project state"
        )

    if _contains_ordinary_project_state(resolved, project):
        raise ProofRunRefused(
            f"--state-root {resolved} already holds state for a project other than "
            f"{project!r}; a run against a live root is refused, not warned about"
        )

    designation_path = Path(designation)
    if not designation_path.is_file():
        raise ProofRunRefused(f"--designation {designation} is not a readable file")

    return ProofRunRequest(state_root=resolved, project=project, designation=designation_path)


def _contains_ordinary_project_state(root: Path, proof_project: str) -> bool:
    """Whether this root carries any project's state other than the proof project.

    Conservative: anything it cannot read counts as occupied, because "I could not
    tell" must not admit a run against a live root.
    """
    bindings = root / "project-state-bindings"
    if bindings.exists():
        try:
            for entry in bindings.iterdir():
                if entry.name != f"{proof_project}.json":
                    return True
        except OSError:
            return True

    # A database in the root means a server has already run here with real state.
    for marker in ("cao.db", "terminals.db"):
        if (root / marker).exists():
            return True
    return False


def refuse_on_existing_artifacts(root: Path) -> None:
    """A receipt or partial already there is a collision, never an overwrite."""
    for basename in (codec.RECEIPT_BASENAME, codec.PARTIAL_BASENAME):
        if (root / basename).exists():
            raise ProofRunRefused(
                f"{root / basename} already exists; a proof run never overwrites an "
                "artifact a reviewer may be about to read"
            )


#: Injected so the contract above is testable hermetically. The default
#: implementation is the one ops runs; it starts its own server and drives the
#: proof steps. Tests substitute a fake executor rather than a live server,
#: because a runner whose refusals can only be exercised against a real
#: deployment has untestable refusals.
ProofExecutor = Callable[[ProofRunRequest], dict]


def _unavailable_executor(request: ProofRunRequest) -> dict:
    """The default executor is deliberately not implemented in this slice.

    Authoring and testing the runner's contract is fork-lane work; **executing a
    live proof against a real deployment is ops-only** and no implementation lease
    authorizes it. Rather than ship a half-driver that could be pointed at a
    running server, the executor is a named seam that ops supplies, and this
    default refuses.
    """
    raise ProofRunRefused(
        "no proof executor is configured: live proof execution is an ops action on an "
        "isolated disposable deployment, and this build ships the contract, the codec, "
        "and the refusals — not an unattended driver for a real server"
    )


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    executor: ProofExecutor = _unavailable_executor,
) -> int:
    """Validate, execute, and emit exactly one artifact. Returns the exit code."""
    args = _parser().parse_args(argv)

    try:
        request = validate_request(args.state_root, args.project, args.designation)
        refuse_on_existing_artifacts(request.state_root)
    except ProofRunRefused as exc:
        # Nothing was started, so nothing is written: a precondition refusal is
        # not a proof attempt and must not leave an artifact behind.
        print(f"cao-gate2-proof-run: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    try:
        document = executor(request)
    except ProofRunRefused as exc:
        # The executor refused *before* executing anything — the "no executor
        # configured" case. Nothing began, so this belongs with the other
        # pre-execution refusals: non-zero, and neither artifact. Writing a
        # partial here would claim a proof attempt that never happened.
        print(f"cao-gate2-proof-run: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001 - a begun run always leaves a partial
        # The run has begun. From here every exit writes exactly one artifact.
        # A run that started and then failed is not a precondition refusal, and
        # reporting it as one would lose the record of what was observed before
        # it stopped.
        print(f"cao-gate2-proof-run: proof did not complete: {exc}", file=sys.stderr)
        document = _partial_from_failure(request, exc)

    schema = document.get("schema")
    if schema not in (codec.RECEIPT_SCHEMA, codec.PARTIAL_SCHEMA):
        print(
            f"cao-gate2-proof-run: executor returned unknown schema {schema!r}",
            file=sys.stderr,
        )
        document = _partial_from_failure(
            request, RuntimeError(f"executor returned unknown schema {schema!r}")
        )
        schema = codec.PARTIAL_SCHEMA

    # A returned receipt is validated *before* anything is written, because a
    # document that fails validation is a begun run that failed — not an I/O
    # problem, and certainly not a pre-execution refusal. Discovering that inside
    # the write attempt conflated the two and lost the artifact Gate 2 needs.
    if schema == codec.RECEIPT_SCHEMA:
        try:
            codec.validate_receipt(codec.redact(document))
        except codec.ReceiptError as exc:
            print(
                f"cao-gate2-proof-run: the run produced an invalid receipt: {exc}",
                file=sys.stderr,
            )
            document = _partial_from_failure(request, exc)
            schema = codec.PARTIAL_SCHEMA

    try:
        if schema == codec.RECEIPT_SCHEMA:
            payload, sha = codec.emit_receipt(request.state_root / codec.RECEIPT_BASENAME, document)
            print(f"receipt written: sha256={sha} bytes={len(payload)}")
            return EXIT_OK

        payload, sha = codec.emit_partial(request.state_root / codec.PARTIAL_BASENAME, document)
        print(f"partial written: sha256={sha} bytes={len(payload)}", file=sys.stderr)
        return EXIT_PARTIAL
    except Exception as exc:  # noqa: BLE001 - an emitter failure is still an exit
        # Only a genuine write failure reaches here now — the disk, not the
        # document. There is no artifact to leave when the artifact itself cannot
        # be written, so a no-artifact refusal is unavoidable; it is caught and
        # named distinctly so it is never confused with a validation outcome.
        print(
            f"cao-gate2-proof-run: I/O failure writing the artifact: {exc}",
            file=sys.stderr,
        )
        return EXIT_REFUSED


def _partial_from_failure(request: "ProofRunRequest", exc: BaseException) -> dict:
    """A truthful partial for a run that began and did not finish.

    It records what the run can actually say -- nothing observed, and no teardown
    asserted -- rather than borrowing a receipt's success values.
    """
    return {
        "schema": codec.PARTIAL_SCHEMA,
        "target": {
            "fork_source_sha256": "",
            "deployed_artifact_sha256": "",
            "server_start_id": "",
        },
        "isolation": {
            "state_root": str(request.state_root),
            "proof_project": request.project,
            "is_disposable_instance": True,
        },
        "observed_fact_names": [],
        "unobserved_fact_names": ["lineage_isolation", "supervisor_creation_discriminator"],
        "refusal_reason": f"proof-run-failed: {exc}",
        "steps": [
            {
                "seq": 1,
                "command": "cao-gate2-proof-run",
                "outcome": "failed",
                "reason_code": "",
                "reason_detail": str(exc),
                "terminal_created": False,
                "terminal_torn_down": False,
                "epoch_allocated": False,
                "epoch_reused": False,
                "observed_at": "",
            }
        ],
        "teardown": {
            "instance_destroyed": False,
            "state_root_removed": False,
            "artifacts_deleted": False,
        },
        "identities": {
            "operator_ref": "",
            "tool_version": "",
            "spec_head": "",
            "reviewed_source_head": "",
        },
    }


def main() -> None:
    raise SystemExit(run())
