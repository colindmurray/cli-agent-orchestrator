"""Admission opens the conductor-named task occurrence (cond-0842).

Parameterized over the three ordinary entry surfaces — v1 bridged, v2
ACP, v2 native — every test drives real service calls against fresh
stores with fake provider effects only: no test seeds a goal, a roster
row, or an occurrence. The only explicit ``open_occurrence`` calls left
model a *second* worker's occurrence, proving the claim/abandon paths
refuse foreign bindings rather than overriding them.

Chain under test: reserve/claim/bind/readiness (per surface) ->
claim_admission opens the occurrence atomically with ``admitting`` ->
roster incarnation, occurrence record, and occurrence history agree ->
abandonment (definitive zero-byte only) fences and finalizes.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from cli_agent_orchestrator.models.managed_launch import (
    PROTOCOL_VERSION,
    ManagedLaunchAdmitRequest,
    ManagedLaunchObservationRequest,
    ManagedLaunchReserveRequest,
)
from cli_agent_orchestrator.models.managed_launch_v2 import (
    PROTOCOL_VERSION_V2,
    ManagedLaunchV2AdmitRequest,
    ManagedLaunchV2BindRequest,
    ManagedLaunchV2ReserveRequest,
)
from cli_agent_orchestrator.services import managed_launch as v1
from cli_agent_orchestrator.services import managed_launch_v2 as v2
from cli_agent_orchestrator.services import stable_agent_roster as roster
from cli_agent_orchestrator.services import task_occurrence as occurrence
from cli_agent_orchestrator.services.managed_provider_bridge import BRIDGE_VERSION

DELIVERY_ID = "33333333-3333-4333-8333-333333333333"
V2_DELIVERY_ID = "44444444-4444-4444-8444-444444444444"


@pytest.fixture
def effect_tmp(isolated_memory_db, monkeypatch, tmp_path):
    from cli_agent_orchestrator import constants

    monkeypatch.setattr(constants, "COMPANION_DIR", tmp_path / "companion")
    return tmp_path


def _provider_executable(tmp_path):
    executable = tmp_path / "fake-provider"
    if not executable.exists():
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    return executable


def _v1_request(tmp_path, **changes):
    executable = _provider_executable(tmp_path)
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


def _v1_ready_receipt(record, request, native_session_id):
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


def _v1_admit_request(message="review the exact head", occurrence_id=None, **changes):
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
    if occurrence_id is not None:
        payload["task_occurrence_id"] = occurrence_id
    payload.update(changes)
    return ManagedLaunchAdmitRequest(**payload)


def _v1_ready_row(tmp_path):
    """A v1 reservation through real claim_launch/mark_ready (the writer binds)."""
    request = _v1_request(tmp_path)
    record, _ = v1.reserve(request)
    record, _ = v1.claim_launch(request.reservation_id)
    receipt = _v1_ready_receipt(record, request, f"provider-session-{uuid.uuid4()}")
    ready = v1.mark_ready(
        request.reservation_id,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        receipt=receipt,
    )
    assert ready["state"] == "ready"
    return request, ready


def _v1_claim(tmp_path, occurrence_id):
    request, ready = _v1_ready_row(tmp_path)
    admission = _v1_admit_request(occurrence_id=occurrence_id)
    claimed, should_send = v1.claim_admission(request.reservation_id, admission)
    assert should_send is True
    assert claimed["state"] == "admitting"
    return request, claimed


def test_v1_claim_opens_the_conductor_named_occurrence(effect_tmp):
    """Ordinary v1 entrypoint: the claim opens occurrence/round-0/admission atomically."""
    tmp_path = effect_tmp
    occurrence_id = str(uuid.uuid4())
    request, claimed = _v1_claim(tmp_path, occurrence_id)
    stored = occurrence.get_occurrence(occurrence_id)
    assert stored["state"] == occurrence.STATE_OPEN
    assert stored["agent_id"] == claimed["stable_agent_id"]
    assert stored["round_index"] == 0
    assert stored["terminal_id"] == claimed["terminal_id"]
    assert stored["generation"] == claimed["generation"]
    assert stored["session_name"] == claimed["session_name"]
    # The roster half of the pairing names the same identity.
    incarnation = roster.get_incarnation_by_terminal(
        claimed["terminal_id"], claimed["generation"]
    )
    assert incarnation is not None
    assert incarnation["agent_id"] == stored["agent_id"]
    assert incarnation["incarnation_id"] == stored["incarnation_id"]
    # And the occurrence-history read the hook chain serves resolves it open.
    history = occurrence.occurrence_history(claimed["session_name"], stored["agent_id"])
    assert history["open"] is not None
    assert history["open"]["task_occurrence_id"] == occurrence_id


def test_v1_claim_replay_adopts_without_a_second_occurrence(effect_tmp):
    """Response loss after claim: the identical replay adopts everything."""
    tmp_path = effect_tmp
    occurrence_id = str(uuid.uuid4())
    request, claimed = _v1_claim(tmp_path, occurrence_id)
    again, should_send = v1.claim_admission(
        request.reservation_id, _v1_admit_request(occurrence_id=occurrence_id)
    )
    assert should_send is False
    assert again["stable_agent_id"] == claimed["stable_agent_id"]
    assert len(roster.list_incarnations(agent_id=claimed["stable_agent_id"])) == 1
    stored = occurrence.get_occurrence(occurrence_id)
    assert stored["state"] == occurrence.STATE_OPEN
    assert stored["revision"] == 0


def test_v1_claim_replay_with_a_different_occurrence_is_refused(effect_tmp):
    """A retried delivery id carrying another task's occurrence is refused."""
    tmp_path = effect_tmp
    request, claimed = _v1_claim(tmp_path, str(uuid.uuid4()))
    with pytest.raises(v1.ManagedLaunchConflict):
        v1.claim_admission(
            request.reservation_id, _v1_admit_request(occurrence_id=str(uuid.uuid4()))
        )
    # The first delivery's chain stands untouched.
    assert occurrence.get_occurrence(
        claimed["admission"]["task_occurrence_id"]
    )["state"] == occurrence.STATE_OPEN


def test_v1_second_incarnation_cannot_represent_the_same_occurrence(effect_tmp):
    """A fallback on a new incarnation must mint, never re-present.

    The conductor's fallback delivery keeps its journaled occurrence on its
    own reservation; if it ever reached a new incarnation under the old id,
    the occurrence content check (terminal/generation/digest) refuses
    rather than opening another worker's occurrence under a new identity.
    Same-goal fallback association therefore only ever adopts the row the
    occurrence already binds — rebinding is unexpressible, not merely
    refused here.
    """
    tmp_path = effect_tmp
    occurrence_id = str(uuid.uuid4())
    request, claimed = _v1_claim(tmp_path, occurrence_id)
    other_request, other_ready = _v1_ready_row(tmp_path)
    assert other_ready["terminal_id"] != claimed["terminal_id"]
    with pytest.raises(v1.ManagedLaunchConflict):
        v1.claim_admission(
            other_request.reservation_id,
            _v1_admit_request(
                occurrence_id=occurrence_id,
                delivery_id=str(uuid.uuid4()),
            ),
        )
    # The first delivery's chain stands untouched.
    stored = occurrence.get_occurrence(occurrence_id)
    assert stored["state"] == occurrence.STATE_OPEN
    assert stored["agent_id"] == claimed["stable_agent_id"]
    assert stored["revision"] == 0


def test_v1_taskless_claim_opens_nothing(effect_tmp):
    """A taskless admission (no occurrence id) claims without an occurrence."""
    tmp_path = effect_tmp
    request, ready = _v1_ready_row(tmp_path)
    claimed, should_send = v1.claim_admission(
        request.reservation_id, _v1_admit_request()
    )
    assert should_send is True
    assert claimed["admission"]["task_occurrence_id"] is None
    assert roster.get_incarnation_by_terminal(
        claimed["terminal_id"], claimed["generation"]
    )["agent_id"] == claimed["stable_agent_id"]


def test_v1_legacy_claim_heals_with_the_presented_occurrence(effect_tmp):
    """A claim from before the seam (no id) heals on retry with the same delivery."""
    tmp_path = effect_tmp
    request, ready = _v1_ready_row(tmp_path)
    legacy, should_send = v1.claim_admission(
        request.reservation_id, _v1_admit_request()
    )
    assert should_send is True
    assert legacy["admission"]["task_occurrence_id"] is None
    occurrence_id = str(uuid.uuid4())
    healed, should_send_again = v1.claim_admission(
        request.reservation_id, _v1_admit_request(occurrence_id=occurrence_id)
    )
    assert should_send_again is False
    assert healed["admission"]["task_occurrence_id"] == occurrence_id
    stored = occurrence.get_occurrence(occurrence_id)
    assert stored["state"] == occurrence.STATE_OPEN
    assert stored["agent_id"] == healed["stable_agent_id"]


def test_v1_legacy_heal_refuses_a_foreign_occurrence(effect_tmp):
    """The heal opens only this delivery's occurrence, never another worker's."""
    tmp_path = effect_tmp
    # A second worker with its own honestly opened occurrence.
    other_request, other_ready = _v1_ready_row(tmp_path)
    foreign_id = str(uuid.uuid4())
    v1.claim_admission(
        other_request.reservation_id, _v1_admit_request(occurrence_id=foreign_id)
    )
    # A legacy claim for this worker, then a heal naming the foreign id.
    request, ready = _v1_ready_row(tmp_path)
    v1.claim_admission(request.reservation_id, _v1_admit_request())
    with pytest.raises(v1.ManagedLaunchConflict):
        v1.claim_admission(
            request.reservation_id, _v1_admit_request(occurrence_id=foreign_id)
        )
    assert occurrence.get_occurrence(foreign_id)["agent_id"] == other_ready[
        "stable_agent_id"
    ]


# ---------------------------------------------------------------------------
# v2 (ACP and native share claim_admission; the occurrence id rides the row)
# ---------------------------------------------------------------------------

V2_MODEL = "gpt-5.6-sol"
V2_EFFORT = "xhigh"


def _v2_request(worktree, tmp_path, execution_mode=None, occurrence_id=None, **changes):
    executable = _provider_executable(tmp_path)
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "reservation_id": str(uuid.uuid4()),
        "session_name": "cao-test",
        "provider": "kimi_cli",
        "agent_profile": "reviewer",
        "caller_id": "deadbeef",
        "project": "test-project",
        "task_id": "test-task",
        "run_id": "run-0001",
        "delivery_id": V2_DELIVERY_ID,
        "launch_nonce": "n" * 40,
        "obligation_generation": "obgen-7c2e4a1b",
        "working_directory": str(worktree),
        "provider_executable": str(executable),
        "provider_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "expected_model": V2_MODEL,
        "expected_effort": V2_EFFORT,
    }
    if execution_mode is not None:
        payload["execution_mode"] = execution_mode
    if occurrence_id is not None:
        payload["task_occurrence_id"] = occurrence_id
    payload.update(changes)
    return ManagedLaunchV2ReserveRequest(**payload)


def _v2_receipt(record, reservation_id, native_session_id, native=False):
    return {
        "bridge_version": BRIDGE_VERSION,
        "receipt_id": native_session_id,
        "provider_session_id": native_session_id,
        "provider_receipt_kind": (
            "kimi-native-tui-attached" if native else "kimi-acp-session-new"
        ),
        # Native bind additionally requires a proven-capable provider build.
        "provider_version": "kimi 0.29.0" if native else "kimi-cli-stub",
        "provider_transcript_sha256": "a" * 64,
        "model_input_ready": True,
        "reservation_id": reservation_id,
        "terminal_id": record["terminal_id"],
        "generation": record["generation"],
        "provider": record["provider"],
        "agent_profile": record["agent_profile"],
        "model": V2_MODEL,
        "effort": V2_EFFORT,
        "working_directory": record["working_directory"],
    }


def _v2_bind_request(record):
    return ManagedLaunchV2BindRequest(
        protocol_version=PROTOCOL_VERSION_V2,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        attempt_id=str(uuid.uuid4()),
    )


def _v2_admit_request(bound, **changes):
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "delivery_id": V2_DELIVERY_ID,
        "message": "review the exact head",
        "message_sha256": hashlib.sha256(b"review the exact head").hexdigest(),
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
        "native_binding_digest": v2.native_binding_digest(bound),
    }
    payload.update(changes)
    return ManagedLaunchV2AdmitRequest(**payload)


def _v2_worktree(tmp_path, monkeypatch):
    import subprocess

    from cli_agent_orchestrator import constants

    worktree = tmp_path / "worktree"
    worktree.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=worktree, check=True)
    # Unique content per call: a test binding two reservations rebuilds the
    # worktree twice, and an empty second commit would fail.
    (worktree / "f.txt").write_text(f"x-{uuid.uuid4()}")
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=worktree, check=True)
    monkeypatch.setattr(constants, "COMPANION_DIR", tmp_path / "companion")
    return worktree


def _v2_bound_row(tmp_path, monkeypatch, execution_mode, occurrence_id=None):
    """A v2 reservation through real claim_launch/bind (bridge read faked)."""
    from cli_agent_orchestrator.services import managed_provider_bridge as bridge

    worktree = _v2_worktree(tmp_path, monkeypatch)
    occurrence_id = occurrence_id or str(uuid.uuid4())
    record, _ = v2.reserve(
        _v2_request(worktree, tmp_path, execution_mode=execution_mode,
                    occurrence_id=occurrence_id)
    )
    v2.claim_launch(record["reservation_id"])
    native_session_id = f"v2-native-{uuid.uuid4()}"
    receipt = _v2_receipt(
        record, record["reservation_id"], native_session_id,
        native=(execution_mode == "native_tui"),
    )
    monkeypatch.setattr(
        bridge, "read_state",
        lambda rid: {"state": "ready", "readiness": receipt},
    )
    bound = v2.bind_native(record["reservation_id"], _v2_bind_request(record))
    assert bound["state"] == "bound"
    return bound, occurrence_id


@pytest.mark.parametrize("execution_mode", ["acp", "native_tui"])
def test_v2_claim_opens_the_row_named_occurrence(effect_tmp, monkeypatch, execution_mode):
    """Ordinary v2 entrypoint (both modes): claim opens the row's occurrence."""
    tmp_path = effect_tmp
    bound, occurrence_id = _v2_bound_row(tmp_path, monkeypatch, execution_mode)
    claimed, should_send = v2.claim_admission(
        bound["reservation_id"], _v2_admit_request(bound)
    )
    assert should_send is True
    assert claimed["state"] == "admitting"
    stored = occurrence.get_occurrence(occurrence_id)
    assert stored["state"] == occurrence.STATE_OPEN
    assert stored["agent_id"] == bound["stable_agent_id"]
    assert stored["round_index"] == 0
    assert stored["terminal_id"] == bound["terminal_id"]
    assert stored["generation"] == bound["generation"]
    incarnation = roster.get_incarnation_by_terminal(
        bound["terminal_id"], bound["generation"]
    )
    assert incarnation is not None
    assert incarnation["agent_id"] == stored["agent_id"]
    assert incarnation["incarnation_id"] == stored["incarnation_id"]
    history = occurrence.occurrence_history(bound["session_name"], stored["agent_id"])
    assert history["open"] is not None
    assert history["open"]["task_occurrence_id"] == occurrence_id


@pytest.mark.parametrize("execution_mode", ["acp", "native_tui"])
def test_v2_claim_replay_adopts(effect_tmp, monkeypatch, execution_mode):
    """An exact v2 replay adopts the occurrence; no second row appears."""
    tmp_path = effect_tmp
    bound, occurrence_id = _v2_bound_row(tmp_path, monkeypatch, execution_mode)
    v2.claim_admission(bound["reservation_id"], _v2_admit_request(bound))
    again, should_send = v2.claim_admission(
        bound["reservation_id"], _v2_admit_request(bound)
    )
    assert should_send is False
    stored = occurrence.get_occurrence(occurrence_id)
    assert stored["state"] == occurrence.STATE_OPEN
    assert stored["revision"] == 0


@pytest.mark.parametrize("execution_mode", ["acp", "native_tui"])
def test_v2_second_incarnation_cannot_represent_the_same_occurrence(
    effect_tmp, monkeypatch, execution_mode
):
    """Both v2 modes: re-presenting an occurrence on a new incarnation is refused."""
    tmp_path = effect_tmp
    bound, occurrence_id = _v2_bound_row(tmp_path, monkeypatch, execution_mode)
    v2.claim_admission(bound["reservation_id"], _v2_admit_request(bound))
    # A second reservation (new incarnation) carrying the first delivery's
    # occurrence id: presenting it on the new incarnation must refuse at
    # the occurrence content check.
    other_bound, _ = _v2_bound_row(
        tmp_path, monkeypatch, execution_mode, occurrence_id=occurrence_id
    )
    assert other_bound["terminal_id"] != bound["terminal_id"]
    with pytest.raises(v2.ManagedLaunchConflict):
        v2.claim_admission(
            other_bound["reservation_id"],
            _v2_admit_request(other_bound, delivery_id=str(uuid.uuid4())),
        )
    stored = occurrence.get_occurrence(occurrence_id)
    assert stored["state"] == occurrence.STATE_OPEN
    assert stored["agent_id"] == bound["stable_agent_id"]
    assert stored["revision"] == 0


@pytest.mark.parametrize("execution_mode", ["acp", "native_tui"])
def test_v2_taskless_claim_opens_nothing(effect_tmp, monkeypatch, execution_mode):
    """A v2 row reserved without an occurrence id claims occurrence-free."""
    tmp_path = effect_tmp
    from cli_agent_orchestrator.services import managed_provider_bridge as bridge

    worktree = _v2_worktree(tmp_path, monkeypatch)
    record, _ = v2.reserve(_v2_request(worktree, tmp_path, execution_mode=execution_mode))
    assert record.get("task_occurrence_id") is None
    v2.claim_launch(record["reservation_id"])
    receipt = _v2_receipt(
        record, record["reservation_id"], f"v2-native-{uuid.uuid4()}",
        native=(execution_mode == "native_tui"),
    )
    monkeypatch.setattr(
        bridge, "read_state",
        lambda rid: {"state": "ready", "readiness": receipt},
    )
    bound = v2.bind_native(record["reservation_id"], _v2_bind_request(record))
    claimed, should_send = v2.claim_admission(
        bound["reservation_id"], _v2_admit_request(bound)
    )
    assert should_send is True
    assert claimed["state"] == "admitting"


# ---------------------------------------------------------------------------
# Definitive zero-byte abandonment
# ---------------------------------------------------------------------------

def _abandon_observation(claimed, request, kind="cancelled"):
    return ManagedLaunchObservationRequest(
        protocol_version=PROTOCOL_VERSION,
        observation_id=str(uuid.uuid5(uuid.UUID(request.reservation_id), "test-abandon-v1")),
        kind=kind,
        terminal_id=claimed["terminal_id"],
        generation=claimed["generation"],
        provider=claimed["provider"],
        agent_profile=claimed["agent_profile"],
        model=request.expected_model,
        effort=request.expected_effort,
        preflight_class="recovery-abandon-admitting",
        evidence_digest="e" * 64,
        detail="test abandonment: pane absent, no submission recorded",
    )


def _bridge_without_submission(monkeypatch):
    from cli_agent_orchestrator.services import managed_provider_bridge as bridge

    monkeypatch.setattr(bridge, "read_state", lambda rid: {"state": "ready"})


def test_v1_abandon_fences_and_finalizes_the_chain(effect_tmp, monkeypatch):
    """Definitive abandonment: fence + abandoned occurrence + terminal state."""
    tmp_path = effect_tmp
    occurrence_id = str(uuid.uuid4())
    request, claimed = _v1_claim(tmp_path, occurrence_id)
    _bridge_without_submission(monkeypatch)
    abandoned = v1.append_observation(
        request.reservation_id, _abandon_observation(claimed, request)
    )
    assert abandoned["state"] == "cancelled"
    admission = abandoned["admission"]
    assert admission["status"] == "refused"
    assert admission["refusal_reason"] == "abandoned"
    stored = occurrence.get_occurrence(occurrence_id)
    assert stored["state"] == occurrence.STATE_FINALIZED
    assert stored["finalized"]["disposition"] == occurrence.DISPOSITION_ABANDONED
    # The fence holds: no completion can follow, and re-abandoning adopts.
    with pytest.raises(v1.ManagedLaunchConflict):
        v1.complete_admission(
            request.reservation_id, DELIVERY_ID, {"provider": "codex"}
        )
    again = v1.append_observation(
        request.reservation_id, _abandon_observation(claimed, request)
    )
    assert again["state"] == "cancelled"
    assert again["admission"]["status"] == "refused"


def test_v1_abandon_refuses_when_bridge_recorded_a_submission(
    effect_tmp, monkeypatch
):
    """A recorded submission is ambiguity, never abandonment: both preserved."""
    tmp_path = effect_tmp
    from cli_agent_orchestrator.services import managed_provider_bridge as bridge

    occurrence_id = str(uuid.uuid4())
    request, claimed = _v1_claim(tmp_path, occurrence_id)
    monkeypatch.setattr(
        bridge, "read_state",
        lambda rid: {"state": "ready", "submission": {"turn_id": "t-1"}},
    )
    with pytest.raises(v1.ManagedLaunchConflict):
        v1.append_observation(
            request.reservation_id, _abandon_observation(claimed, request)
        )
    assert v1.get(request.reservation_id)["state"] == "admitting"
    assert occurrence.get_occurrence(occurrence_id)["state"] == occurrence.STATE_OPEN


def test_v1_abandon_refuses_a_foreign_occurrence(effect_tmp, monkeypatch):
    """Abandonment never finalizes another worker's occurrence."""
    tmp_path = effect_tmp
    other_request, other_claimed = _v1_claim(tmp_path, str(uuid.uuid4()))
    foreign_id = other_claimed["admission"]["task_occurrence_id"]
    # A fresh claim naming the other worker's occurrence is refused...
    request, ready = _v1_ready_row(tmp_path)
    with pytest.raises(v1.ManagedLaunchConflict):
        v1.claim_admission(
            request.reservation_id, _v1_admit_request(occurrence_id=foreign_id)
        )
    # ...and abandoning the honest row finalizes only its own chain...
    own_id = str(uuid.uuid4())
    _v1_claim(tmp_path, own_id)
    _bridge_without_submission(monkeypatch)
    done = v1.append_observation(
        other_request.reservation_id,
        _abandon_observation(other_claimed, other_request),
    )
    assert done["admission"]["status"] == "refused"
    # ...while the untouched worker's chain stands.
    assert occurrence.get_occurrence(foreign_id)["state"] == occurrence.STATE_FINALIZED
    assert occurrence.get_occurrence(own_id)["state"] == occurrence.STATE_OPEN


@pytest.mark.parametrize("execution_mode", ["acp", "native_tui"])
def test_v2_permanent_refusal_finalizes_the_occurrence(
    effect_tmp, monkeypatch, execution_mode
):
    """A permanent v2 refusal ends the occurrence chain in the same txn."""
    tmp_path = effect_tmp
    bound, occurrence_id = _v2_bound_row(tmp_path, monkeypatch, execution_mode)
    v2.claim_admission(bound["reservation_id"], _v2_admit_request(bound))
    refused = v2.mark_admission_refused(
        bound["reservation_id"],
        V2_DELIVERY_ID,
        "composer_plan_invalid",
        "zero task bytes were written",
    )
    assert refused["admission"]["status"] == "refused"
    stored = occurrence.get_occurrence(occurrence_id)
    assert stored["state"] == occurrence.STATE_FINALIZED
    assert stored["finalized"]["disposition"] == occurrence.DISPOSITION_ABANDONED


@pytest.mark.parametrize("execution_mode", ["acp", "native_tui"])
def test_v2_retryable_refusal_keeps_the_occurrence_open(
    effect_tmp, monkeypatch, execution_mode
):
    """A retryable v2 refusal preserves the occurrence for later completion."""
    tmp_path = effect_tmp
    bound, occurrence_id = _v2_bound_row(tmp_path, monkeypatch, execution_mode)
    v2.claim_admission(bound["reservation_id"], _v2_admit_request(bound))
    v2.mark_admission_refused(
        bound["reservation_id"],
        V2_DELIVERY_ID,
        "provider_not_yet_ready",
        "may complete later",
    )
    assert occurrence.get_occurrence(occurrence_id)["state"] == occurrence.STATE_OPEN
