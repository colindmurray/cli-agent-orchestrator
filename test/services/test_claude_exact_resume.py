"""S8.2 exact resume adapter tests (red-first)."""

from __future__ import annotations

import uuid
import hashlib
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli_agent_orchestrator.services import claude_exact_resume as cer
from cli_agent_orchestrator.services import claude_native_launch
from cli_agent_orchestrator.services import managed_launch_v2 as v2
from cli_agent_orchestrator.models.managed_launch_v2 import (
    ManagedLaunchV2ReserveRequest,
    PROTOCOL_VERSION_V2,
)
from cli_agent_orchestrator.services.claude_exact_resume import (
    ExactResumeRequest,
    TypedIneligibility,
)
from cli_agent_orchestrator.services import native_attachment
from cli_agent_orchestrator.services import execution_mode as em


def _valid_uuid() -> str:
    return str(uuid.uuid4())


def _valid_request(**overrides):
    base = {
        "predecessor_native_session_id": _valid_uuid(),
        "operation_id": _valid_uuid(),
        "provider": "claude_code",
        "model": "sonnet",
        "effort": "high",
        "quota_provider": "anthropic",
        "provider_route": "anthropic",
        "auth_transport": "api_key",
    }
    base.update(overrides)
    return ExactResumeRequest(**base)


# ---------------------------------------------------------------------------
# Typed ACP refusal
# ---------------------------------------------------------------------------


def test_typed_acp_refusal_names_code_and_clearing_path():
    record = {
        "provider": "claude_code",
        "execution_mode": "acp",
        "native_session_id": _valid_uuid(),
        "provider_version": "claude 2.1.220",
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == cer.ACP_RESUME_UNSUPPORTED
    assert "ACP_RESUME_UNSUPPORTED" in str(exc.value)
    # clearing path must be named
    assert "clearing path" in str(exc.value).lower()


def test_acp_refusal_even_when_other_fields_would_also_fail():
    # ACP should win even if native_session_id is invalid
    record = {
        "provider": "wrong",
        "execution_mode": "acp",
        "native_session_id": "not-a-uuid",
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == cer.ACP_RESUME_UNSUPPORTED


# ---------------------------------------------------------------------------
# Required carried fields: absent/None is typed refusal, never silent default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "provider",
        "model",
        "effort",
        "quota_provider",
        "provider_route",
        "auth_transport",
    ],
)
def test_missing_carried_field_refuses_typed(field):
    data = {
        "predecessor_native_session_id": _valid_uuid(),
        "operation_id": _valid_uuid(),
        "provider": "claude_code",
        "model": "sonnet",
        "effort": "high",
        "quota_provider": "anthropic",
        "provider_route": "anthropic",
        "auth_transport": "api_key",
    }
    data[field] = None
    with pytest.raises(TypedIneligibility):
        ExactResumeRequest.from_dict(data)
    # Also via dataclass direct with None should raise
    data2 = dict(data)
    data2[field] = ""
    with pytest.raises((TypedIneligibility, Exception)):
        ExactResumeRequest(**{k: v for k, v in data2.items() if v is not None})  # type: ignore


def test_missing_predecessor_id_refuses():
    data = {
        "operation_id": _valid_uuid(),
        "provider": "claude_code",
        "model": "sonnet",
        "effort": "high",
        "quota_provider": "anthropic",
        "provider_route": "anthropic",
        "auth_transport": "api_key",
    }
    with pytest.raises(TypedIneligibility):
        ExactResumeRequest.from_dict(data)


def test_invalid_predecessor_uuid_refuses():
    with pytest.raises(Exception):
        _valid_request(predecessor_native_session_id="not-a-uuid")
    with pytest.raises(Exception):
        _valid_request(predecessor_native_session_id=_valid_uuid().upper())


def test_invalid_operation_id_refuses():
    with pytest.raises(Exception):
        _valid_request(operation_id="not-a-uuid")


# ---------------------------------------------------------------------------
# Eligibility predicate
# ---------------------------------------------------------------------------


def test_eligible_record_passes():
    record = {
        "provider": "claude_code",
        "execution_mode": "native_tui",
        "native_session_id": _valid_uuid(),
        "provider_version": "claude 2.1.220",
    }
    assert cer.is_eligible(record) is True


def test_ineligible_provider_refuses():
    record = {
        "provider": "kimi_cli",
        "execution_mode": "native_tui",
        "native_session_id": _valid_uuid(),
        "provider_version": "claude 2.1.220",
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == "INELIGIBLE_PROVIDER"


def test_ineligible_execution_mode_refuses():
    record = {
        "provider": "claude_code",
        "execution_mode": "bogus",
        "native_session_id": _valid_uuid(),
        "provider_version": "claude 2.1.220",
    }
    with pytest.raises(TypedIneligibility):
        cer.is_eligible(record)


def test_missing_native_session_id_refuses():
    record = {
        "provider": "claude_code",
        "execution_mode": "native_tui",
        "provider_version": "claude 2.1.220",
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == "MISSING_NATIVE_SESSION_ID"


def test_ambiguous_effect_refuses():
    record = {
        "provider": "claude_code",
        "execution_mode": "native_tui",
        "native_session_id": _valid_uuid(),
        "provider_version": "claude 2.1.220",
        "ambiguous_effect": True,
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == "AMBIGUOUS_EFFECT_BOUNDARY"


def test_ineligible_on_missing_receipt():
    """P1-3: absent capability receipt/version proof is NOT eligible, never vacuously eligible."""
    record = {
        "provider": "claude_code",
        "execution_mode": "native_tui",
        "native_session_id": _valid_uuid(),
        # no provider_version, no capability_receipt, no readiness
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == "MISSING_CAPABILITY_RECEIPT"
    assert "clearing path" in str(exc.value).lower()


def test_absent_execution_mode_refusal():
    """P2-2: absent execution_mode on a resume source is typed refusal, never default."""
    record = {
        "provider": "claude_code",
        # "execution_mode" absent
        "native_session_id": _valid_uuid(),
        "provider_version": "claude 2.1.220",
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == "INELIGIBLE_EXECUTION_MODE"


# ---------------------------------------------------------------------------
# Full route carry-through: each carried field asserted individually
# ---------------------------------------------------------------------------


def test_full_route_carry_through_into_argv_and_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(cer, "COMPANION_DIR", tmp_path / "companion")
    monkeypatch.setattr(v2, "COMPANION_DIR", tmp_path / "companion")
    req = _valid_request(
        model="sonnet",
        effort="high",
        quota_provider="qp-1",
        provider_route="anthropic",
        auth_transport="api_key",
    )
    # Argv must carry predecessor id and model/effort
    argv = cer.build_resume_argv(req)
    assert claude_native_launch.resumes_exactly(argv, req.predecessor_native_session_id)
    argv_with_model = cer.build_resume_argv_with_model(req)
    assert "--model" in argv_with_model
    assert req.model in argv_with_model or req.model.lower() in [a.lower() for a in argv_with_model]
    # Each carried field must be present in bootstrap when prepared via the
    # resume helper (we stub version/digest)
    from cli_agent_orchestrator.services.managed_launch_v2 import _prepare_claude_resume_session

    record = {
        "terminal_id": "abcd1234",
        "generation": _valid_uuid(),
        "working_directory": str(tmp_path),
    }
    bootstrap, hook = _prepare_claude_resume_session(
        record=record,
        pending=req,
        version_output="claude 2.1.220",
        digest="0" * 64,
    )
    assert bootstrap["carried_provider"] == req.provider
    assert bootstrap["carried_quota_provider"] == req.quota_provider
    assert bootstrap["carried_provider_route"] == req.provider_route
    assert bootstrap["carried_auth_transport"] == req.auth_transport
    assert bootstrap["requested_model"] == req.model
    assert bootstrap["requested_effort"] == req.effort
    assert bootstrap["predecessor_native_session_id"] == req.predecessor_native_session_id
    assert bootstrap["operation_id"] == req.operation_id


def test_each_carried_field_individually_reaches_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(cer, "COMPANION_DIR", tmp_path / "companion")
    from cli_agent_orchestrator.services.managed_launch_v2 import _prepare_claude_resume_session

    for field, value in [
        ("provider", "claude_code"),
        ("model", "opus"),
        ("effort", "high"),
        ("quota_provider", "qp-x"),
        ("provider_route", "anthropic"),
        ("auth_transport", "oauth"),
    ]:
        req = _valid_request(**{field: value} if field != "provider" else {})
        record = {
            "terminal_id": "abcd1234",
            "generation": _valid_uuid(),
            "working_directory": str(tmp_path),
        }
        bootstrap, _ = _prepare_claude_resume_session(
            record=record, pending=req, version_output="claude 2.1.220", digest="0" * 64
        )
        carried_map = {
            "provider": "carried_provider",
            "model": "requested_model",
            "effort": "requested_effort",
            "quota_provider": "carried_quota_provider",
            "provider_route": "carried_provider_route",
            "auth_transport": "carried_auth_transport",
        }
        assert bootstrap[carried_map[field]] == value


# ---------------------------------------------------------------------------
# Identity exclusivity: exactly one identity form, dual refuses pre-I/O
# ---------------------------------------------------------------------------


def test_dual_identity_refuses_before_provider_io():
    req = _valid_request()
    provider_called = []

    def fake_provider():
        provider_called.append(True)

    with pytest.raises(TypedIneligibility) as exc:
        cer.assert_single_identity(resume_request=req, mint_intent=True)
    assert exc.value.code == "DUAL_IDENTITY_FORMS"
    assert provider_called == []
    cer.assert_single_identity(resume_request=req, mint_intent=False)
    cer.assert_single_identity(resume_request=None, mint_intent=True)
    cer.assert_single_identity(resume_request=None, mint_intent=False)


def test_resume_uses_build_resume_argv_not_mint(monkeypatch):
    req = _valid_request()
    with (
        patch.object(
            claude_native_launch, "build_resume_argv", wraps=claude_native_launch.build_resume_argv
        ) as mock_resume,
        patch.object(claude_native_launch, "mint_session_id") as mock_mint,
    ):
        argv = cer.build_resume_argv(req)
        mock_resume.assert_called_once()
        mock_mint.assert_not_called()
        assert claude_native_launch.resumes_exactly(argv, req.predecessor_native_session_id)


# ---------------------------------------------------------------------------
# Fence-before-successor ordering + correct generation key + intent-before-fence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fence_before_successor_ordering(tmp_path, monkeypatch, isolated_memory_db):
    """Predecessor fence completes before successor launch, with correct generation key."""
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=worktree, check=True)
    (worktree / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=worktree, check=True)

    monkeypatch.setattr(v2, "COMPANION_DIR", tmp_path / "companion")
    monkeypatch.setattr(cer, "COMPANION_DIR", tmp_path / "companion")

    import hashlib

    executable = tmp_path / "fake-claude"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    req = ManagedLaunchV2ReserveRequest(
        protocol_version=PROTOCOL_VERSION_V2,
        reservation_id=str(uuid.uuid4()),
        session_name="cao-test",
        provider="claude_code",
        agent_profile="reviewer",
        caller_id="deadbeef",
        working_directory=str(worktree),
        trusted_project_root=None,
        expected_model="sonnet",
        expected_effort="high",
        provider_executable=str(executable),
        provider_executable_sha256=digest,
        obligation_generation="obgen-1",
        task_id="t1",
        run_id="r1",
        delivery_id=str(uuid.uuid4()),
        launch_nonce="n" * 40,
        execution_mode="native_tui",
    )
    record, _ = v2.reserve(req)
    reservation_id = record["reservation_id"]
    v2.claim_launch(reservation_id)

    # Create predecessor attachment with known generation
    predecessor_generation = _valid_uuid()
    predecessor_terminal = "11111111"
    predecessor_session_id = _valid_uuid()
    # Create a minimal predecessor terminal record via native_attachment
    # We need to declare the predecessor session so fence can find its generation
    native_attachment.declare(
        provider="claude_code",
        native_session_id=predecessor_session_id,
        terminal_id=predecessor_terminal,
        generation=predecessor_generation,
        execution_mode=em.NATIVE_TUI,
        intent=native_attachment.acquire_intent(
            acquisition_method=native_attachment.ACQUISITION_CHOSEN_SESSION_ID,
            acquisition_receipt={"session_id": predecessor_session_id},
            admits_only_new_instructions=True,
            replays_task_bytes=False,
        ),
    )

    resume_req = _valid_request(
        predecessor_native_session_id=predecessor_session_id,
        model="sonnet",
        effort="high",
        quota_provider="anthropic",
        provider_route="anthropic",
        auth_transport="api_key",
    )
    cer.set_pending_resume(reservation_id, resume_req)

    call_order = []
    captured_fence_args = {}

    orig_fence = cer.fence_predecessor
    orig_record = cer.record_successor_generation

    def tracked_fence(*args, **kwargs):
        call_order.append("fence")
        captured_fence_args.update(kwargs)
        # also track intent-before-fence: intent should already be recorded
        intent_path = Path(
            tmp_path / "companion" / "claude_exact_resume" / f"{resume_req.operation_id}.json"
        )
        assert intent_path.exists(), "intent-before-fence: successor intent must exist before fence"
        return orig_fence(*args, **kwargs)

    def tracked_record(*args, **kwargs):
        call_order.append("intent")
        return orig_record(*args, **kwargs)

    monkeypatch.setattr(cer, "fence_predecessor", tracked_fence)
    monkeypatch.setattr(cer, "record_successor_generation", tracked_record)

    from cli_agent_orchestrator.services import managed_provider_bridge as bridge

    monkeypatch.setattr(bridge, "provider_version_banner", lambda *a, **k: "claude 2.1.220")
    orig_prepare = v2._prepare_claude_resume_session

    def stub_prepare(*args, **kwargs):
        call_order.append("prepare")
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(v2, "_prepare_claude_resume_session", stub_prepare)

    from cli_agent_orchestrator.services import native_tui_launch as ntl

    def fake_start(**kwargs):
        call_order.append("launch")
        raise RuntimeError("stub launch for ordering test")

    monkeypatch.setattr(ntl, "start", fake_start)

    from cli_agent_orchestrator.services.managed_provider_bridge import (
        _profile_material,
        profile_digest,
    )

    pm = _profile_material(record["agent_profile"], record["terminal_id"])
    bridge_request = {
        "bridge_version": "test",
        "controller_pid": 1,
        "reservation_id": reservation_id,
        "terminal_id": record["terminal_id"],
        "generation": record["generation"],
        "provider": record["provider"],
        "agent_profile": record["agent_profile"],
        "profile_sha256": pm["profile_sha256"],
        "model": record["request"]["expected_model"],
        "effort": record["request"]["expected_effort"],
        "working_directory": record["working_directory"],
        "provider_executable": str(executable),
        "provider_executable_sha256": digest,
        "provider_route": record["request"].get("provider_route", "anthropic"),
        "route_envelope": record["request"].get("route_envelope"),
        "project": record["request"].get("project") or record["run_id"],
        "task_id": record["task_id"],
        "delivery_id": record["request"]["delivery_id"],
        "run_id": record["run_id"],
        "obligation_generation": record["obligation_generation"],
        "assigned_policy_sha256": profile_digest(record["agent_profile"]),
        "rendezvous_identity": "test-rendezvous",
    }

    import asyncio

    try:
        await v2._launch_native_tui(reservation_id, v2.get(reservation_id), bridge_request)
    except Exception:
        pass

    assert "intent" in call_order
    assert "fence" in call_order
    assert "launch" in call_order
    assert call_order.index("intent") < call_order.index("fence") < call_order.index("launch")
    # Correct generation key: fence must have been called with predecessor GENERATION, not session id
    assert captured_fence_args.get("generation") == predecessor_generation
    assert captured_fence_args.get("generation") != predecessor_session_id
    cer.clear_pending_resume(reservation_id)


@pytest.mark.asyncio
async def test_refusal_before_side_effects_no_fence_or_intent(
    tmp_path, monkeypatch, isolated_memory_db
):
    """P1-1: TypedIneligibility (e.g., ACP) must complete BEFORE any fence or intent."""
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=worktree, check=True)
    (worktree / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=worktree, check=True)

    monkeypatch.setattr(v2, "COMPANION_DIR", tmp_path / "companion")
    monkeypatch.setattr(cer, "COMPANION_DIR", tmp_path / "companion")

    import hashlib

    executable = tmp_path / "fake-claude"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    # Create reservation with execution_mode acp to trigger ACP refusal via successor record
    # But we need successor to be claude_code; we will force ACP via predecessor eligibility.
    # Instead we create a normal native_tui successor but make pending's source ineligible
    # by having no capability receipt and then trigger is_eligible failure.
    # To trigger P1-1 we will make the successor's execution_mode absent (typed) by patching
    # the record after reserve to have no execution_mode, and ensure pending triggers ACP.
    req = ManagedLaunchV2ReserveRequest(
        protocol_version=PROTOCOL_VERSION_V2,
        reservation_id=str(uuid.uuid4()),
        session_name="cao-test",
        provider="claude_code",
        agent_profile="reviewer",
        caller_id="deadbeef",
        working_directory=str(worktree),
        trusted_project_root=None,
        expected_model="sonnet",
        expected_effort="high",
        provider_executable=str(executable),
        provider_executable_sha256=digest,
        obligation_generation="obgen-1",
        task_id="t1",
        run_id="r1",
        delivery_id=str(uuid.uuid4()),
        launch_nonce="n" * 40,
        execution_mode="native_tui",
    )
    record, _ = v2.reserve(req)
    reservation_id = record["reservation_id"]
    v2.claim_launch(reservation_id)

    # Prepare a resume that will be ineligible due to missing capability and will be checked before fence
    # We will make is_eligible fail by providing no provider_version and no receipt
    # To force this, we will monkeypatch version_output to empty string via bridge stub
    from cli_agent_orchestrator.services import managed_provider_bridge as bridge

    monkeypatch.setattr(bridge, "provider_version_banner", lambda *a, **k: "")

    predecessor_session = _valid_uuid()
    # Create attachment for predecessor so generation check would pass if we got that far
    native_attachment.declare(
        provider="claude_code",
        native_session_id=predecessor_session,
        terminal_id="22222222",
        generation=_valid_uuid(),
        execution_mode=em.NATIVE_TUI,
        intent=native_attachment.acquire_intent(
            acquisition_method=native_attachment.ACQUISITION_CHOSEN_SESSION_ID,
            acquisition_receipt={"session_id": predecessor_session},
            admits_only_new_instructions=True,
            replays_task_bytes=False,
        ),
    )

    resume_req = _valid_request(
        predecessor_native_session_id=predecessor_session,
        model="sonnet",
        effort="high",
        quota_provider="anthropic",
        provider_route="anthropic",
        auth_transport="api_key",
    )
    cer.set_pending_resume(reservation_id, resume_req)

    fence_called = []
    intent_called = []

    orig_fence = cer.fence_predecessor
    orig_intent = cer.record_successor_generation

    def spy_fence(*args, **kwargs):
        fence_called.append(True)
        return orig_fence(*args, **kwargs)

    def spy_intent(*args, **kwargs):
        intent_called.append(True)
        return orig_intent(*args, **kwargs)

    monkeypatch.setattr(cer, "fence_predecessor", spy_fence)
    monkeypatch.setattr(cer, "record_successor_generation", spy_intent)

    from cli_agent_orchestrator.services.managed_provider_bridge import (
        _profile_material,
        profile_digest,
    )

    pm = _profile_material(record["agent_profile"], record["terminal_id"])
    bridge_request = {
        "bridge_version": "test",
        "controller_pid": 1,
        "reservation_id": reservation_id,
        "terminal_id": record["terminal_id"],
        "generation": record["generation"],
        "provider": record["provider"],
        "agent_profile": record["agent_profile"],
        "profile_sha256": pm["profile_sha256"],
        "model": record["request"]["expected_model"],
        "effort": record["request"]["expected_effort"],
        "working_directory": record["working_directory"],
        "provider_executable": str(executable),
        "provider_executable_sha256": digest,
        "provider_route": record["request"].get("provider_route", "anthropic"),
        "route_envelope": record["request"].get("route_envelope"),
        "project": record["request"].get("project") or record["run_id"],
        "task_id": record["task_id"],
        "delivery_id": record["request"]["delivery_id"],
        "run_id": record["run_id"],
        "obligation_generation": record["obligation_generation"],
        "assigned_policy_sha256": profile_digest(record["agent_profile"]),
        "rendezvous_identity": "test-rendezvous",
    }

    # This should return a preflight_blocked with no fence or intent side effects?
    # Our new ordering ensures eligibility is checked before fence/intent, so for
    # empty version_output, is_eligible will see no provider_version and no receipt
    # and thus be ineligible, raising before fence.
    result = await v2._launch_native_tui(reservation_id, v2.get(reservation_id), bridge_request)
    # Should be preflight_blocked due to missing capability / version
    assert result.get("state") == "preflight_blocked"
    assert fence_called == []
    assert intent_called == []
    cer.clear_pending_resume(reservation_id)


@pytest.mark.asyncio
async def test_prod_path_dual_identity_refusal(tmp_path, monkeypatch, isolated_memory_db):
    """P2-1: production selection path must refuse dual identity before side effects."""
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=worktree, check=True)
    (worktree / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=worktree, check=True)

    monkeypatch.setattr(v2, "COMPANION_DIR", tmp_path / "companion")
    monkeypatch.setattr(cer, "COMPANION_DIR", tmp_path / "companion")

    import hashlib

    executable = tmp_path / "fake-claude"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    req = ManagedLaunchV2ReserveRequest(
        protocol_version=PROTOCOL_VERSION_V2,
        reservation_id=str(uuid.uuid4()),
        session_name="cao-test",
        provider="claude_code",
        agent_profile="reviewer",
        caller_id="deadbeef",
        working_directory=str(worktree),
        trusted_project_root=None,
        expected_model="sonnet",
        expected_effort="high",
        provider_executable=str(executable),
        provider_executable_sha256=digest,
        obligation_generation="obgen-1",
        task_id="t1",
        run_id="r1",
        delivery_id=str(uuid.uuid4()),
        launch_nonce="n" * 40,
        execution_mode="native_tui",
    )
    record, _ = v2.reserve(req)
    reservation_id = record["reservation_id"]
    v2.claim_launch(reservation_id)

    predecessor_session = _valid_uuid()
    native_attachment.declare(
        provider="claude_code",
        native_session_id=predecessor_session,
        terminal_id="33333333",
        generation=_valid_uuid(),
        execution_mode=em.NATIVE_TUI,
        intent=native_attachment.acquire_intent(
            acquisition_method=native_attachment.ACQUISITION_CHOSEN_SESSION_ID,
            acquisition_receipt={"session_id": predecessor_session},
            admits_only_new_instructions=True,
            replays_task_bytes=False,
        ),
    )

    resume_req = _valid_request(
        predecessor_native_session_id=predecessor_session,
        model="sonnet",
        effort="high",
        quota_provider="anthropic",
        provider_route="anthropic",
        auth_transport="api_key",
    )
    cer.set_pending_resume(reservation_id, resume_req)
    cer.set_pending_mint_intent(reservation_id, True)

    fence_called = []

    orig_fence = cer.fence_predecessor

    def spy_fence(*args, **kwargs):
        fence_called.append(True)
        return orig_fence(*args, **kwargs)

    monkeypatch.setattr(cer, "fence_predecessor", spy_fence)
    from cli_agent_orchestrator.services import managed_provider_bridge as bridge

    monkeypatch.setattr(bridge, "provider_version_banner", lambda *a, **k: "claude 2.1.220")
    from cli_agent_orchestrator.services.managed_provider_bridge import (
        _profile_material,
        profile_digest,
    )

    pm = _profile_material(record["agent_profile"], record["terminal_id"])
    bridge_request = {
        "bridge_version": "test",
        "controller_pid": 1,
        "reservation_id": reservation_id,
        "terminal_id": record["terminal_id"],
        "generation": record["generation"],
        "provider": record["provider"],
        "agent_profile": record["agent_profile"],
        "profile_sha256": pm["profile_sha256"],
        "model": record["request"]["expected_model"],
        "effort": record["request"]["expected_effort"],
        "working_directory": record["working_directory"],
        "provider_executable": str(executable),
        "provider_executable_sha256": digest,
        "provider_route": record["request"].get("provider_route", "anthropic"),
        "route_envelope": record["request"].get("route_envelope"),
        "project": record["request"].get("project") or record["run_id"],
        "task_id": record["task_id"],
        "delivery_id": record["request"]["delivery_id"],
        "run_id": record["run_id"],
        "obligation_generation": record["obligation_generation"],
        "assigned_policy_sha256": profile_digest(record["agent_profile"]),
        "rendezvous_identity": "test-rendezvous",
    }

    result = await v2._launch_native_tui(reservation_id, v2.get(reservation_id), bridge_request)
    # Dual identity must be refused pre-I/O, so preflight_blocked and no fence
    assert result.get("state") == "preflight_blocked"
    assert "DUAL_IDENTITY_FORMS" in result.get("preflight_failure", {}).get("detail", "")
    assert fence_called == []
    cer.clear_pending_resume(reservation_id)
    cer.clear_pending_mint_intent(reservation_id)


# ---------------------------------------------------------------------------
# Restart readback: successor generation durably recorded
# ---------------------------------------------------------------------------


def test_successor_generation_durably_recorded_and_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(cer, "COMPANION_DIR", tmp_path / "companion")
    monkeypatch.setattr(v2, "COMPANION_DIR", tmp_path / "companion")
    op_id = _valid_uuid()
    tid = "abcd1234"
    gen = _valid_uuid()
    rec = cer.record_successor_generation(
        operation_id=op_id, successor_terminal_id=tid, successor_generation=gen
    )
    assert rec["successor_generation"] == gen
    read = cer.read_successor_generation(op_id)
    assert read is not None
    assert read["successor_generation"] == gen
    assert read["successor_terminal_id"] == tid
    rec2 = cer.record_successor_generation(
        operation_id=op_id, successor_terminal_id=tid, successor_generation=gen
    )
    assert rec2["successor_generation"] == gen
    rec3 = cer.record_successor_generation(
        operation_id=op_id, successor_terminal_id="other123", successor_generation=_valid_uuid()
    )
    assert rec3["successor_generation"] == gen
    assert rec3["successor_terminal_id"] == tid


def test_ambiguous_retry_reuses_operation_id(tmp_path, monkeypatch):
    monkeypatch.setattr(cer, "COMPANION_DIR", tmp_path / "companion")
    op_id = _valid_uuid()
    gen1 = _valid_uuid()
    cer.record_successor_generation(
        operation_id=op_id, successor_terminal_id="tid1", successor_generation=gen1
    )
    rec = cer.read_successor_generation(op_id)
    assert rec["operation_id"] == op_id
    assert rec["successor_generation"] == gen1
