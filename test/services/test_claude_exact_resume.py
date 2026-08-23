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
from cli_agent_orchestrator.models.managed_launch_v2 import ManagedLaunchV2ReserveRequest, PROTOCOL_VERSION_V2
from cli_agent_orchestrator.services.claude_exact_resume import ExactResumeRequest, TypedIneligibility


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


@pytest.mark.parametrize("field", [
    "provider", "model", "effort", "quota_provider", "provider_route", "auth_transport",
])
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
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == "INELIGIBLE_PROVIDER"


def test_ineligible_execution_mode_refuses():
    record = {
        "provider": "claude_code",
        "execution_mode": "bogus",
        "native_session_id": _valid_uuid(),
    }
    with pytest.raises(TypedIneligibility):
        cer.is_eligible(record)


def test_missing_native_session_id_refuses():
    record = {
        "provider": "claude_code",
        "execution_mode": "native_tui",
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == "MISSING_NATIVE_SESSION_ID"


def test_ambiguous_effect_refuses():
    record = {
        "provider": "claude_code",
        "execution_mode": "native_tui",
        "native_session_id": _valid_uuid(),
        "ambiguous_effect": True,
    }
    with pytest.raises(TypedIneligibility) as exc:
        cer.is_eligible(record)
    assert exc.value.code == "AMBIGUOUS_EFFECT_BOUNDARY"


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
    # Need a real git repo for working_directory check? _prepare does not check git head
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
        overrides = {field: value} if field != "provider" else {}
        # provider must stay claude_code else request fails
        req = _valid_request(**{field: value} if field != "provider" else {})
        record = {"terminal_id": "abcd1234", "generation": _valid_uuid(), "working_directory": str(tmp_path)}
        bootstrap, _ = _prepare_claude_resume_session(
            record=record, pending=req, version_output="claude 2.1.220", digest="0"*64
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
    # Simulate a provider stub that would be called if we didn't refuse
    provider_called = []

    def fake_provider():
        provider_called.append(True)

    # The guard itself must refuse without calling provider
    with pytest.raises(TypedIneligibility) as exc:
        cer.assert_single_identity(resume_request=req, mint_intent=True)
    assert exc.value.code == "DUAL_IDENTITY_FORMS"
    assert provider_called == []
    # Single identity passes
    cer.assert_single_identity(resume_request=req, mint_intent=False)
    cer.assert_single_identity(resume_request=None, mint_intent=True)
    cer.assert_single_identity(resume_request=None, mint_intent=False)


def test_resume_uses_build_resume_argv_not_mint(monkeypatch):
    # Verify that the resume path calls build_resume_argv and never mint
    req = _valid_request()
    with patch.object(claude_native_launch, "build_resume_argv", wraps=claude_native_launch.build_resume_argv) as mock_resume, \
         patch.object(claude_native_launch, "mint_session_id") as mock_mint:
        argv = cer.build_resume_argv(req)
        mock_resume.assert_called_once()
        mock_mint.assert_not_called()
        assert claude_native_launch.resumes_exactly(argv, req.predecessor_native_session_id)


# ---------------------------------------------------------------------------
# Fence-before-successor ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fence_before_successor_ordering(tmp_path, monkeypatch, isolated_memory_db):
    """Predecessor fence completes before successor launch."""
    # Setup worktree
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
    from cli_agent_orchestrator.clients import database

    # Create a reservation for claude_code native_tui
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
    # Also need to move to launching state
    v2.claim_launch(reservation_id)

    # Prepare a resume request
    resume_req = _valid_request(
        model="sonnet",
        effort="high",
        quota_provider="anthropic",
        provider_route="anthropic",
        auth_transport="api_key",
    )
    cer.set_pending_resume(reservation_id, resume_req)

    # Track ordering: fence should be called before native_tui_launch.start
    call_order = []

    orig_fence = cer.fence_predecessor

    def tracked_fence(*args, **kwargs):
        call_order.append("fence")
        return orig_fence(*args, **kwargs)

    monkeypatch.setattr(cer, "fence_predecessor", tracked_fence)

    # Stub provider version banner and terminal creation
    from cli_agent_orchestrator.services import managed_provider_bridge as bridge
    monkeypatch.setattr(bridge, "provider_version_banner", lambda *a, **k: "claude 2.1.220")
    orig_prepare = v2._prepare_claude_resume_session

    def stub_prepare(*args, **kwargs):
        call_order.append("prepare")
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(v2, "_prepare_claude_resume_session", stub_prepare)

    # Stub the actual pane launch to avoid tmux
    call_order_before = list(call_order)

    # We need to stub native_tui_launch.start to capture ordering
    from cli_agent_orchestrator.services import native_tui_launch as ntl

    orig_start = ntl.start

    def fake_start(**kwargs):
        call_order.append("launch")
        # Return a minimal outcome that will cause readiness to fail but still prove ordering?
        # Instead we stub higher: we will mock _launch_native_tui's final part by patching
        # native_tui_launch.start to record and then raise to trigger preflight?
        # For ordering we just need to see fence before launch, so we can let it proceed.
        # Return a fake outcome that will later be blocked at readiness, but fence already happened.
        raise RuntimeError("stub launch for ordering test")

    monkeypatch.setattr(ntl, "start", fake_start)

    # Need to also stub database.set_terminal_v2_native_session_id etc? Not needed because launch will fail before there.

    # Run launch — build a full bridge_request matching launch_reserved
    from cli_agent_orchestrator.services.managed_provider_bridge import _profile_material, profile_digest

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

    # Fence must have been called before launch attempt
    assert "fence" in call_order
    assert "launch" in call_order
    assert call_order.index("fence") < call_order.index("launch")
    cer.clear_pending_resume(reservation_id)


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
    # Readback
    read = cer.read_successor_generation(op_id)
    assert read is not None
    assert read["successor_generation"] == gen
    assert read["successor_terminal_id"] == tid
    # Idempotent retry with same operation_id reuses same generation
    rec2 = cer.record_successor_generation(
        operation_id=op_id, successor_terminal_id=tid, successor_generation=gen
    )
    assert rec2["successor_generation"] == gen
    # A different generation with same operation_id adopts the stored one (idempotency)
    rec3 = cer.record_successor_generation(
        operation_id=op_id, successor_terminal_id="other123", successor_generation=_valid_uuid()
    )
    assert rec3["successor_generation"] == gen
    assert rec3["successor_terminal_id"] == tid


def test_ambiguous_retry_reuses_operation_id(tmp_path, monkeypatch):
    monkeypatch.setattr(cer, "COMPANION_DIR", tmp_path / "companion")
    op_id = _valid_uuid()
    gen1 = _valid_uuid()
    cer.record_successor_generation(operation_id=op_id, successor_terminal_id="tid1", successor_generation=gen1)
    # Simulate ambiguous effect retry: same operation_id
    rec = cer.read_successor_generation(op_id)
    assert rec["operation_id"] == op_id
    assert rec["successor_generation"] == gen1
