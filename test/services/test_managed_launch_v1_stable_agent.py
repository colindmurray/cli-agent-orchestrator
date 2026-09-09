"""stable_agent_id projection on the v1 (bridged) managed-launch surface.

cond-0842: an ordinary bridged launch must bind the conductor's durable goal
to the fork-authoritative stable agent. The v1 row projects the roster's
deterministic initial id from the immutable terminal+generation allocated at
reserve: the same id on every exact-id replay/get/reconcile read, a distinct
id per reservation, no schema change and no request-shape change.

The pairing tests below prove the projection against the roster itself,
not helper-equals-itself: a fresh reservation has no roster incarnation
(roster authoritative, nothing shadowed), a real ``bind_generation`` with
the projected id records exactly that id (read back through the roster's
own readers), a replay adopts it, and a pre-existing different binding
for the same incarnation is refused rather than minted a second identity.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from cli_agent_orchestrator.models.managed_launch import (
    PROTOCOL_VERSION,
    ManagedLaunchReserveRequest,
)
from cli_agent_orchestrator.services import managed_launch
from cli_agent_orchestrator.services import stable_agent_roster

DELIVERY_ID = "33333333-3333-4333-8333-333333333333"


def _reserve_request(tmp_path, **changes):
    executable = tmp_path / "fake-provider"
    if not executable.exists():
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reservation_id": str(uuid.uuid4()),
        "session_name": "cao-test",
        "provider": "codex",
        "agent_profile": "reviewer-sol-max",
        "caller_id": "deadbeef",
        "project": "test-project",
        "task_id": "test-task",
        "delivery_id": DELIVERY_ID,
        "working_directory": str(tmp_path),
        "trusted_project_root": str(tmp_path),
        "expected_model": "gpt-5.6-sol",
        "expected_effort": "xhigh",
        "provider_executable": str(executable),
        "provider_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }
    payload.update(changes)
    return ManagedLaunchReserveRequest(**payload)


def _recovery_contract(record, agent_id):
    """The bind a recovery seam builds for this reservation's incarnation."""
    return stable_agent_roster.BindingContract(
        agent_id=agent_id,
        session_name=record["session_name"],
        role=stable_agent_roster.ROLE_WORKER,
        profile_family=record["agent_profile"],
        harness=record["provider"],
        terminal_id=record["terminal_id"],
        generation=record["generation"],
    )


def test_reserve_projects_the_roster_derived_stable_agent_id(
    isolated_memory_db, tmp_path
):
    record, created = managed_launch.reserve(_reserve_request(tmp_path))
    assert created is True
    assert record["stable_agent_id"] == (
        stable_agent_roster.derive_initial_agent_id(
            record["terminal_id"], record["generation"]
        )
    )
    # The generation is load-bearing: the generation-less derivation names
    # a different identity (the unmanaged-terminal family) and must never
    # be projected for a generation-bound reservation.
    assert record["stable_agent_id"] != (
        stable_agent_roster.derive_initial_agent_id(record["terminal_id"])
    )


def test_replay_get_and_reconcile_return_the_same_stable_agent_id(
    isolated_memory_db, tmp_path
):
    request = _reserve_request(tmp_path)
    first, _ = managed_launch.reserve(request)
    replay, created_again = managed_launch.reserve(request)
    assert created_again is False
    assert replay["stable_agent_id"] == first["stable_agent_id"]
    assert (
        managed_launch.get(request.reservation_id)["stable_agent_id"]
        == first["stable_agent_id"]
    )
    assert (
        managed_launch.reconcile(request.reservation_id)["stable_agent_id"]
        == first["stable_agent_id"]
    )


def test_distinct_reservations_get_distinct_stable_agent_ids(
    isolated_memory_db, tmp_path
):
    one, _ = managed_launch.reserve(_reserve_request(tmp_path))
    two, _ = managed_launch.reserve(_reserve_request(tmp_path))
    assert one["stable_agent_id"] != two["stable_agent_id"]


def test_projected_id_is_the_roster_recorded_identity_for_the_reservation(
    isolated_memory_db, tmp_path
):
    """The projected id equals the actual roster identity, via the roster."""
    record, _ = managed_launch.reserve(_reserve_request(tmp_path))
    terminal_id = record["terminal_id"]
    generation = record["generation"]
    # Roster authoritative: a fresh reservation shadows no existing worker.
    assert (
        stable_agent_roster.get_incarnation_by_terminal(terminal_id, generation)
        is None
    )
    bound = stable_agent_roster.bind_generation(
        _recovery_contract(record, record["stable_agent_id"])
    )
    assert bound["agent"]["agent_id"] == record["stable_agent_id"]
    # Read back through the roster's own readers, never the helper.
    assert (
        stable_agent_roster.get_agent(record["stable_agent_id"])["agent_id"]
        == record["stable_agent_id"]
    )
    incarnation = stable_agent_roster.get_incarnation_by_terminal(
        terminal_id, generation
    )
    assert incarnation is not None
    assert incarnation["agent_id"] == record["stable_agent_id"]
    # An exact replay adopts the same identity: no duplicate agent.
    again = stable_agent_roster.bind_generation(
        _recovery_contract(record, record["stable_agent_id"])
    )
    assert again["agent"]["agent_id"] == record["stable_agent_id"]
    assert again["adopted"] is True


def test_roster_refuses_a_second_stable_identity_for_the_same_incarnation(
    isolated_memory_db, tmp_path
):
    """A recovered worker keeps its identity; the projection cannot override."""
    record, _ = managed_launch.reserve(_reserve_request(tmp_path))
    other_agent_id = str(uuid.uuid4())
    stable_agent_roster.bind_generation(
        _recovery_contract(record, other_agent_id)
    )
    with pytest.raises(stable_agent_roster.StableAgentConflict):
        stable_agent_roster.bind_generation(
            _recovery_contract(record, record["stable_agent_id"])
        )
    incarnation = stable_agent_roster.get_incarnation_by_terminal(
        record["terminal_id"], record["generation"]
    )
    assert incarnation is not None
    assert incarnation["agent_id"] == other_agent_id
