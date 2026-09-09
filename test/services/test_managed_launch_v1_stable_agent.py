"""stable_agent_id projection AND registration on the v1 managed-launch surface.

cond-0842 R3: the projection alone left the roster empty, so the shipped
hook reader answered dead-incarnation. These tests prove the real v1
lifecycle code (reserve/claim_launch/mark_ready/claim_admission/
complete_admission) registers the projected identity in the roster itself:
no test pre-seeds a roster row and no test calls ``bind_generation`` for
the expected id. The only explicit ``bind_generation`` left is a foreign
identity, proving the real writer's record cannot be overridden.

Pairing under test (frozen conductor ``bc3fa399``,
``_materialize_launch_goal`` stores ``companion.get("stable_agent_id")``
as the goal assignee; the shipped hook reader resolves
``GET /roster/terminals/{id}`` — no generation — to the live incarnation,
fences its generation, then reads that agent's occurrences): every paired
test below asserts the roster's own readers return the projected id in the
exact endpoint payload shapes, so the conductor-stored assignee equals the
roster-registered identity.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from cli_agent_orchestrator.models.managed_launch import (
    PROTOCOL_VERSION,
    ManagedLaunchAdmitRequest,
    ManagedLaunchReserveRequest,
)
from cli_agent_orchestrator.services import managed_launch, stable_agent_roster, task_occurrence
from cli_agent_orchestrator.services.managed_provider_bridge import BRIDGE_VERSION

DELIVERY_ID = "33333333-3333-4333-8333-333333333333"


@pytest.fixture
def effect_tmp(isolated_memory_db, monkeypatch, tmp_path):
    from cli_agent_orchestrator import constants

    monkeypatch.setattr(constants, "COMPANION_DIR", tmp_path / "companion")
    return tmp_path


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


def _ready_receipt_for(record, request, native_session_id=None):
    native_session_id = native_session_id or f"provider-session-{uuid.uuid4()}"
    return {
        "bridge_version": BRIDGE_VERSION,
        "receipt_id": native_session_id,
        "provider_session_id": native_session_id,
        "provider_receipt_kind": "codex-thread-start",
        "provider_transcript_sha256": "a" * 64,
        "provider_version": "0.146.0",
        "model_input_ready": True,
        "reservation_id": request.reservation_id,
        "terminal_id": record["terminal_id"],
        "generation": record["generation"],
        "provider": record["provider"],
        "agent_profile": record["agent_profile"],
        "model": request.expected_model,
        "effort": request.expected_effort,
        "working_directory": request.working_directory,
    }


def _admit_request(message="review the exact head", **changes):
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "delivery_id": DELIVERY_ID,
        "message": message,
        "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
        "sender_id": "deadbeef",
        "orchestration_type": "assign",
        "context": {
            "boot_id": "11111111-1111-4111-8111-111111111111",
            "project": "test-project",
            "task_id": "test-task",
            "run_id": "test-task",
            "task_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "dossier_sha256": "3" * 64,
            "lease_sha256": "4" * 64,
            "command_packet_sha256": "5" * 64,
            "source_chain_sha256": "6" * 64,
        },
    }
    payload.update(changes)
    return ManagedLaunchAdmitRequest(**payload)


def _submission_receipt(record, admission, native_session_id):
    return {
        "bridge_version": BRIDGE_VERSION,
        "receipt_id": "provider-turn-opaque",
        "provider_session_id": native_session_id,
        "provider_turn_id": "provider-turn-opaque",
        "provider_receipt_kind": "codex-turn-start",
        "provider_transcript_sha256": "b" * 64,
        "reservation_id": record["reservation_id"],
        "terminal_id": record["terminal_id"],
        "generation": record["generation"],
        "provider": record["provider"],
        "agent_profile": record["agent_profile"],
        "model": record["request"]["expected_model"],
        "effort": record["request"]["expected_effort"],
        "working_directory": record["working_directory"],
        "delivery_id": admission.delivery_id,
        "receiver_id": record["terminal_id"],
        "message_sha256": admission.message_sha256,
        "sender_id": admission.sender_id,
        "context": admission.context.model_dump(mode="json"),
        "provider_accepted": True,
        "submitted_at": "2026-07-22T00:00:00Z",
    }


def _launch_to_ready(tmp_path):
    """Drive the real pre-provider lifecycle: reserve/claim/mark_ready."""
    request = _reserve_request(tmp_path)
    record, created = managed_launch.reserve(request)
    assert created is True
    record, should_launch = managed_launch.claim_launch(request.reservation_id)
    assert should_launch is True
    native_session_id = f"provider-session-{uuid.uuid4()}"
    receipt = _ready_receipt_for(record, request, native_session_id)
    ready = managed_launch.mark_ready(
        request.reservation_id,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        receipt=receipt,
    )
    assert ready["state"] == "ready"
    return request, ready, native_session_id


def _endpoint_incarnation(terminal_id, generation=None):
    """The exact payload shape of GET /roster/terminals/{id}."""
    return {
        "schema": "cao-m3-roster-incarnation-v1",
        "incarnation": stable_agent_roster.get_incarnation_by_terminal(terminal_id, generation),
    }


def test_reserve_projects_the_roster_derived_stable_agent_id(isolated_memory_db, tmp_path):
    record, created = managed_launch.reserve(_reserve_request(tmp_path))
    assert created is True
    assert record["stable_agent_id"] == (
        stable_agent_roster.derive_initial_agent_id(record["terminal_id"], record["generation"])
    )
    # The generation is load-bearing: the generation-less derivation names
    # a different identity (the unmanaged-terminal family) and must never
    # be projected for a generation-bound reservation.
    assert record["stable_agent_id"] != (
        stable_agent_roster.derive_initial_agent_id(record["terminal_id"])
    )


def test_replay_get_and_reconcile_return_the_same_stable_agent_id(isolated_memory_db, tmp_path):
    request = _reserve_request(tmp_path)
    first, _ = managed_launch.reserve(request)
    replay, created_again = managed_launch.reserve(request)
    assert created_again is False
    assert replay["stable_agent_id"] == first["stable_agent_id"]
    assert managed_launch.get(request.reservation_id)["stable_agent_id"] == first["stable_agent_id"]
    assert (
        managed_launch.reconcile(request.reservation_id)["stable_agent_id"]
        == first["stable_agent_id"]
    )


def test_distinct_reservations_get_distinct_stable_agent_ids(isolated_memory_db, tmp_path):
    one, _ = managed_launch.reserve(_reserve_request(tmp_path))
    two, _ = managed_launch.reserve(_reserve_request(tmp_path))
    assert one["stable_agent_id"] != two["stable_agent_id"]


def test_mark_ready_registers_the_projected_identity_in_the_roster(
    effect_tmp,
):
    """The real writer: mark_ready registers exactly the projected id."""
    tmp_path = effect_tmp
    request, ready, native_session_id = _launch_to_ready(tmp_path)
    projected = ready["stable_agent_id"]
    # The terminal-only read is exactly what the frozen conductor reader
    # calls (GET /roster/terminals/{id}, no generation): it must be live.
    answer = _endpoint_incarnation(ready["terminal_id"])
    assert answer["schema"] == "cao-m3-roster-incarnation-v1"
    incarnation = answer["incarnation"]
    assert isinstance(incarnation, dict)
    assert incarnation["agent_id"] == projected
    assert incarnation["generation"] == ready["generation"]
    assert incarnation["disposition"] == stable_agent_roster.INCARNATION_BOUND
    # The exact-generation read agrees (the hook's generation fence).
    exact = _endpoint_incarnation(ready["terminal_id"], ready["generation"])
    assert exact["incarnation"]["agent_id"] == projected
    # The conductor-stored goal assignee is companion.get("stable_agent_id"):
    # it must name the roster's own agent.
    assignee = ready["stable_agent_id"]
    assert stable_agent_roster.get_agent(assignee)["agent_id"] == projected
    # The lineage carries the real provider-reported native session.
    agent = stable_agent_roster.get_agent(projected)
    lineage = agent.get("current_lineage") or {}
    assert lineage.get("native_session_id") == native_session_id


def test_first_hook_read_before_admission_sees_a_live_incarnation(
    effect_tmp,
):
    """A hook firing after readiness but before admission resolves live."""
    tmp_path = effect_tmp
    request, ready, native_session_id = _launch_to_ready(tmp_path)
    assert ready["state"] == "ready"
    assert ready["admission"] is None
    incarnation = _endpoint_incarnation(ready["terminal_id"])["incarnation"]
    assert isinstance(incarnation, dict)
    assert incarnation["agent_id"] == ready["stable_agent_id"]
    # Generation fence inputs the hook reader needs are all present.
    assert incarnation["generation"] == ready["generation"]


def test_mark_ready_retry_adopts_the_same_identity(effect_tmp):
    """Response loss between commit and caller: the retry adopts, never dups."""
    tmp_path = effect_tmp
    request = _reserve_request(tmp_path)
    record, _ = managed_launch.reserve(request)
    assert (
        stable_agent_roster.get_incarnation_by_terminal(record["terminal_id"], record["generation"])
        is None
    )
    record, _ = managed_launch.claim_launch(request.reservation_id)
    receipt = _ready_receipt_for(record, request)
    first = managed_launch.mark_ready(
        request.reservation_id,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        receipt=receipt,
    )
    again = managed_launch.mark_ready(
        request.reservation_id,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        receipt=receipt,
    )
    assert again["stable_agent_id"] == first["stable_agent_id"]
    assert again["state"] == "ready"
    incarnations = stable_agent_roster.list_incarnations(agent_id=first["stable_agent_id"])
    assert len(incarnations) == 1


def test_roster_bind_failure_keeps_the_row_launching_and_retry_heals(effect_tmp, monkeypatch):
    """Partial failure: a transient roster outage refuses typed, keeps the
    row launching (no orphaned live worker state), and the retry heals."""
    tmp_path = effect_tmp
    request = _reserve_request(tmp_path)
    record, _ = managed_launch.reserve(request)
    record, _ = managed_launch.claim_launch(request.reservation_id)
    receipt = _ready_receipt_for(record, request)
    real_bind = stable_agent_roster.bind_generation

    def _flaky(*args, **kwargs):
        raise stable_agent_roster.StableAgentUnavailable("simulated outage")

    monkeypatch.setattr(stable_agent_roster, "bind_generation", _flaky)
    with pytest.raises(managed_launch.ManagedLaunchUnavailable):
        managed_launch.mark_ready(
            request.reservation_id,
            terminal_id=record["terminal_id"],
            generation=record["generation"],
            receipt=receipt,
        )
    assert managed_launch.get(request.reservation_id)["state"] == "launching"
    assert (
        stable_agent_roster.get_incarnation_by_terminal(record["terminal_id"], record["generation"])
        is None
    )
    monkeypatch.setattr(stable_agent_roster, "bind_generation", real_bind)
    healed = managed_launch.mark_ready(
        request.reservation_id,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        receipt=receipt,
    )
    assert healed["state"] == "ready"
    incarnation = _endpoint_incarnation(record["terminal_id"])["incarnation"]
    assert incarnation["agent_id"] == healed["stable_agent_id"]


def test_claim_heals_a_retained_row_without_a_roster_record(effect_tmp, monkeypatch):
    """A row that reached ready before the writer existed (simulated by
    silencing the mark_ready writer) is healed by the real claim path with
    the identical deterministic identity — never a new one."""
    tmp_path = effect_tmp
    request = _reserve_request(tmp_path)
    record, _ = managed_launch.reserve(request)
    record, _ = managed_launch.claim_launch(request.reservation_id)
    receipt = _ready_receipt_for(record, request)
    real_writer = managed_launch._bind_v1_roster_incarnation
    monkeypatch.setattr(managed_launch, "_bind_v1_roster_incarnation", lambda *a, **k: None)
    ready = managed_launch.mark_ready(
        request.reservation_id,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        receipt=receipt,
    )
    assert ready["state"] == "ready"
    assert (
        stable_agent_roster.get_incarnation_by_terminal(record["terminal_id"], record["generation"])
        is None
    )
    monkeypatch.setattr(managed_launch, "_bind_v1_roster_incarnation", real_writer)
    admission = _admit_request()
    claimed, should_send = managed_launch.claim_admission(request.reservation_id, admission)
    assert should_send is True
    incarnation = _endpoint_incarnation(record["terminal_id"])["incarnation"]
    assert isinstance(incarnation, dict)
    assert incarnation["agent_id"] == ready["stable_agent_id"]
    assert incarnation["generation"] == record["generation"]


def test_complete_admission_marks_the_incarnation_admitted(effect_tmp):
    """End-to-end paired admission through real code: the conductor-stored
    assignee equals the roster agent, and delivery flips bound->admitted."""
    tmp_path = effect_tmp
    request, ready, native_session_id = _launch_to_ready(tmp_path)
    admission = _admit_request()
    claimed, should_send = managed_launch.claim_admission(request.reservation_id, admission)
    assert should_send is True
    done = managed_launch.complete_admission(
        request.reservation_id,
        admission.delivery_id,
        _submission_receipt(claimed, admission, native_session_id),
    )
    assert done["state"] == "admitted"
    assignee = done["stable_agent_id"]
    assert assignee == ready["stable_agent_id"]
    assert stable_agent_roster.get_agent(assignee)["agent_id"] == assignee
    incarnation = _endpoint_incarnation(ready["terminal_id"])["incarnation"]
    assert incarnation["agent_id"] == assignee
    assert incarnation["disposition"] == stable_agent_roster.INCARNATION_ADMITTED
    # The open occurrence the linkage reader needs resolves for this agent.
    occurrence_id = str(uuid.uuid4())
    task_occurrence.open_occurrence(
        task_occurrence.OpenRequest(
            task_occurrence_id=occurrence_id,
            session_name=ready["session_name"],
            agent_id=assignee,
            round_index=0,
            dispatch_digest=task_occurrence.dispatch_digest_for({"task": "test-task"}),
            incarnation=task_occurrence.EffectIncarnation(
                incarnation_id=incarnation["incarnation_id"],
                terminal_id=ready["terminal_id"],
                generation=ready["generation"],
                lineage_id=incarnation.get("lineage_id"),
                native_session_id=native_session_id,
            ),
        )
    )
    history = task_occurrence.occurrence_history(ready["session_name"], assignee)
    assert history["open"] is not None
    assert history["open"]["task_occurrence_id"] == occurrence_id


def test_real_writer_record_cannot_be_overridden_by_a_second_identity(
    effect_tmp,
):
    """The writer's registration is authoritative: a foreign bind of the
    same incarnation is refused and the projected identity stands."""
    tmp_path = effect_tmp
    request, ready, native_session_id = _launch_to_ready(tmp_path)
    with pytest.raises(stable_agent_roster.StableAgentConflict):
        stable_agent_roster.bind_generation(
            stable_agent_roster.BindingContract(
                agent_id=str(uuid.uuid4()),
                session_name=ready["session_name"],
                role=stable_agent_roster.ROLE_WORKER,
                profile_family=ready["agent_profile"],
                harness=ready["provider"],
                terminal_id=ready["terminal_id"],
                generation=ready["generation"],
            )
        )
    incarnation = _endpoint_incarnation(ready["terminal_id"])["incarnation"]
    assert incarnation["agent_id"] == ready["stable_agent_id"]


def test_successor_generations_resolve_independently(effect_tmp):
    """Stale-generation safety: each reservation's terminal resolves its
    own live incarnation; a cross-generation exact read is empty."""
    tmp_path = effect_tmp
    _, first, _ = _launch_to_ready(tmp_path)
    _, second, _ = _launch_to_ready(tmp_path)
    assert first["stable_agent_id"] != second["stable_agent_id"]
    one = _endpoint_incarnation(first["terminal_id"])["incarnation"]
    two = _endpoint_incarnation(second["terminal_id"])["incarnation"]
    assert one["agent_id"] == first["stable_agent_id"]
    assert two["agent_id"] == second["stable_agent_id"]
    assert one["generation"] == first["generation"]
    assert two["generation"] == second["generation"]
    assert _endpoint_incarnation(first["terminal_id"], second["generation"])["incarnation"] is None
