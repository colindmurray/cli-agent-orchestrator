"""Shared gate-2 artifact builders for the receipt/state/interval suites.

Deliberately not a `test_` module so pytest does not collect it.

Since receipt verification became unconditional, installing a receipt state means
installing the receipt it points at — a state without its receipt now refuses
server start, which is the contract, not an inconvenience. These helpers keep the
suites honest about that by always writing a matching pair.
"""

from __future__ import annotations

from pathlib import Path

from cli_agent_orchestrator.services import gate2_proof_receipt as codec
from cli_agent_orchestrator.services import gate2_proof_receipt_state as rs


def minimal_receipt() -> dict:
    """A complete, valid successful receipt with the pinned success values."""
    step = {
        "seq": 1,
        "command": "proof step",
        "outcome": "bound",
        "reason_code": "",
        "reason_detail": "",
        "terminal_created": True,
        "terminal_torn_down": False,
        "epoch_allocated": True,
        "epoch_reused": False,
        "observed_at": "2026-08-08T00:00:00Z",
    }
    return {
        "schema": codec.RECEIPT_SCHEMA,
        "target": {
            "fork_source_sha256": "a" * 64,
            "deployed_artifact_sha256": "b" * 64,
            "server_start_id": "start-1",
        },
        "isolation": {
            "state_root": "/tmp/cao-proof",
            "proof_project": "cao-gate2-scratch",
            "is_disposable_instance": True,
        },
        "designation": {
            "sha256": "c" * 64,
            "project": "cao-gate2-scratch",
            "opened_at": "2026-08-08T00:00:00Z",
            "closed_at": "2026-08-08T00:10:00Z",
        },
        "capability_dark": {
            "advertisement_enabled": False,
            "provider_tuples_enabled": False,
            "listener_enabled_for_ordinary_projects": False,
        },
        "ordinary_project_non_effect": {
            "ordinary_projects_observed": 0,
            "authority_rows_created": 0,
            "epoch_allocations_created": 0,
            "grants_minted": 0,
            "project_json_files_touched": 0,
        },
        "proofs": {
            "lineage_isolation": {
                "outcome": "proven",
                "evidence_refs": ["ref/1"],
                "observed_at": "2026-08-08T00:01:00Z",
            },
            "supervisor_creation_discriminator": {
                "outcome": "proven",
                "evidence_refs": ["ref/2"],
                "observed_at": "2026-08-08T00:02:00Z",
            },
        },
        "steps": [step],
        "teardown": {
            "instance_destroyed": True,
            "state_root_removed": True,
            "artifacts_deleted": True,
        },
        "identities": {
            "operator_ref": "ops@host",
            "tool_version": "1",
            "spec_head": "8a01802f",
            "reviewed_source_head": "401ef0e",
        },
    }


def install_recorded_gate2_state(root: Path) -> str:
    """Write a canonical receipt and the receipt state that points at it.

    Returns the receipt digest. This is what "gate 2 recorded" means on a
    deployment now: the pointer and the bytes it commits to, together.
    """
    _, sha = codec.emit_receipt(root / codec.RECEIPT_BASENAME, minimal_receipt())
    rs.write_receipt_state_for_proof_run(root / rs.RECEIPT_STATE_BASENAME, sha)
    return sha
