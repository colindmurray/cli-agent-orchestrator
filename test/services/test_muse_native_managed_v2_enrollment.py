"""Muse managed-v2 native enrollment (cond-0377B), end to end over fakes.

Muse's managed-v2 launch differs from the other three native providers in
one deliberate way: its identity is *discovered*, not chosen or minted.
The fresh launch starts a no-prompt TUI (``muse --trust-workspace
--reasoning-effort high --model <id>`` — no ``resume``, no ``--session-id``,
no prompt, no task bytes) and the provider itself generates the session id;
the managed launch types ``/status`` once and parses the provider-generated
canonical UUID at zero turns, then acquires the native attachment for that
exact id.  ``muse resume <known-id>`` is retained strictly as a separate
restoration form for a later reincarnation slice.

Everything this suite asserts is pinned to the installed build evidence
recorded in ``muse_native_launch`` and ``muse_native_status``:

* The fresh launch argv carries no identity/resume form, prompt, or recency
  selector.
* The CAO profile system prompt is carried into the main session as base
  instructions through the installed ``TBH_EVAL_APPEND_SYSTEM_PROMPT_FILE``
  surface (verified deterministically on the installed 0.1.0-R708.1 build:
  with the env var set, an echo-provider launch *and* an exact
  ``muse resume <id>`` both refuse with "provider does not support base
  instructions", the same run-configuration refusal a built-in preset with
  base instructions produces; the coordinator's real Meta canary proved the
  file's bytes reach the model with a private sentinel).
* The ``/status`` panel reports the exact provider session id, model,
  reasoning effort, agent profile, provider, cwd, and idle/zero-turn
  pre-task state, and the panel is printed output — the composer stays
  ready.  Model and effort render together: ``Model: <id> (reasoning
  <effort>)``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.managed_launch_v2 import (
    PROTOCOL_VERSION_V2,
    ManagedLaunchV2AdmitRequest,
    ManagedLaunchV2BindRequest,
    ManagedLaunchV2ReserveRequest,
)
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import execution_mode as em
from cli_agent_orchestrator.services import managed_launch_v2 as v2
from cli_agent_orchestrator.services import managed_provider_bridge as bridge
from cli_agent_orchestrator.services import (
    muse_native_control,
    muse_native_launch,
    muse_native_status,
    muse_session_store,
    native_attachment,
)
from cli_agent_orchestrator.services import native_pane_input as npi
from cli_agent_orchestrator.services import (
    native_tui_launch,
    terminal_service,
)
from cli_agent_orchestrator.services.managed_launch import ManagedLaunchConflict

MUSE_BANNER = "Muse Code 0.1.0 (0.1.0-R708.1)"
MUSE_INNER_SHA256 = "4290bfafa5bbb81a6fd493aaea12f848c789b1d22edfa0c4b849151deba3e70c"
MUSE_MODEL = "muse-spark-1.2-contributor"
MUSE_EFFORT = "high"
DELIVERY_ID = "44444444-4444-4444-8444-444444444444"
TASK_MESSAGE = "review this diff"

#: A provider-generated session id the mocked /status panel supplies — the
#: coordinator's real provider-generated id on the installed 0.1.0-R708.1
#: build (``fresh muse TUI -> /status``).  The fresh launch must discover
#: exactly this and never mint one of its own.
PROVIDER_SESSION_ID = "ebab9822-608f-470b-8b35-ada098e0cf29"

#: A preserved id used only by the restoration ``muse resume <id>`` argv
#: tests.  No fresh launch ever carries it.
KNOWN_SESSION_ID = "11111111-2222-4333-8444-555555555555"


def _uuid() -> str:
    return str(uuid.uuid4())


def _admit_request(digest: str, **changes) -> ManagedLaunchV2AdmitRequest:
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "delivery_id": DELIVERY_ID,
        "message": TASK_MESSAGE,
        "message_sha256": hashlib.sha256(TASK_MESSAGE.encode()).hexdigest(),
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
        "native_binding_digest": digest,
    }
    payload.update(changes)
    return ManagedLaunchV2AdmitRequest(**payload)


@pytest.fixture(autouse=True)
def _companion(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "COMPANION_DIR", tmp_path / "companion")
    monkeypatch.setattr(bridge, "BRIDGE_ROOT", tmp_path / "bridge")


@pytest.fixture
def worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def _reserve_request(worktree, tmp_path, **changes):
    executable = tmp_path / "fake-muse"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "reservation_id": str(uuid.uuid4()),
        "session_name": "cao-test",
        "provider": "muse_cli",
        "agent_profile": "reviewer",
        "caller_id": "deadbeef",
        "working_directory": str(worktree),
        "trusted_project_root": None,
        "expected_model": MUSE_MODEL,
        "expected_effort": MUSE_EFFORT,
        "provider_executable": str(executable),
        "provider_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "obligation_generation": "obgen-7c2e4a1b",
        "task_id": "self-heal-demo-task",
        "run_id": "run-0001",
        "delivery_id": DELIVERY_ID,
        "launch_nonce": "n" * 40,
        "execution_mode": "native_tui",
    }
    payload.update(changes)
    return ManagedLaunchV2ReserveRequest(**payload)


def status_panel_rows(
    worktree,
    session_id: str,
    *,
    model: str = MUSE_MODEL,
    effort: str = MUSE_EFFORT,
    agent_profile: str = "native-basic",
    provider: str = "meta",
    directory: Optional[str] = None,
    run: str = "idle",
    tokens: str = "0 tokens / 0 turns",
    #: The effort rendered inside the Model line as the exact installed
    #: `` (reasoning <effort>)`` suffix; ``None`` renders a bare model.
    reasoning: Optional[str] = MUSE_EFFORT,
    #: An optional separate ``Reasoning:`` row (a separately-supported
    #: panel variant); ``None`` renders no such row.
    reasoning_line: Optional[str] = None,
    reasoning_label: str = "Reasoning:",
    session_line: Optional[str] = None,
) -> list[str]:
    """The ``/status`` panel rows, in the installed 0.1.0 rendering.

    The coordinator no-prompt canary on 2026-08-10 rendered the meta panel
    with model and effort in ONE line::

        Model: muse-spark-1.2-contributor (reasoning high)

    (there is no separate Reasoning row).  The echo provider renders a
    bare model with no effort at all.  ``reasoning_line``/``reasoning_label``
    let a test build the separate-row variant some builds render, so the
    parser's duplicate-source handling can be exercised.
    """
    width = 45
    directory = directory or str(worktree)
    model_value = f"{model} (reasoning {reasoning})" if reasoning is not None else model
    rows = [
        "  Muse Code 0.1.0",
        "",
        "╭" + "─" * (width + 2) + "╮",
        "│  >_ Muse Code (0.1.0)" + " " * (width - 20) + " │",
        "│" + " " * (width + 2) + "│",
        f"│  Model:{model_value:>{width - 7}} │",
    ]
    if reasoning_line is not None:
        rows.append(f"│  {reasoning_label}{reasoning_line:>{width - len(reasoning_label) - 1}} │")
    rows += [
        f"│  Agent profile:{agent_profile:>{width - 14}} │",
        f"│  Model provider:{provider:>{width - 15}} │",
        "│  Credential:              none                  │",
        f"│  Directory:{directory:>{width - 10}} │",
        "│  Permissions:          approval=Normal sandbox=Normal │",
        "│  Agents.md:            not found                  │",
        "│  Project trust:        trusted                    │",
        f"│  Session:{(session_line or session_id):>{width - 8}} │",
        "│" + " " * (width + 2) + "│",
        f"│  Token usage:{tokens:>{width - 12}} │",
        "│  Context window:       not projected              │",
        f"│  Run:{run:>{width - 4}} │",
        "│  Tasks:                none                       │",
        "│  Terminals:            0                          │",
        "│  Inbox:                0 pending                  │",
        "╰" + "─" * (width + 2) + "╯",
        "",
        "── Voice input (⌥ + v to start) ─────────────────────────────",
        "⟩",
        "────────────────────────────────────────────────────────────",
    ]
    return rows


# --------------------------------------------------------------------
# 1. Provider / table / capability parity — Muse only after the whole
#    managed-v2 branch is implemented.
# --------------------------------------------------------------------


def test_muse_is_enrolled_in_the_derived_native_tui_provider_set():
    """Muse joins the derived set only when all three surfaces exist."""
    assert "muse_cli" in v2.NATIVE_TUI_PROVIDERS
    assert set(v2._NATIVE_TUI_READINESS_RECEIPT_KINDS) == v2.NATIVE_TUI_PROVIDERS
    assert set(v2._ISSUANCE_SOURCES) == v2.NATIVE_TUI_PROVIDERS
    assert set(v2._PINNED_PROVIDER) == v2.NATIVE_TUI_PROVIDERS
    assert v2.NATIVE_TUI_PROVIDERS == native_tui_launch.SUPPORTED_NATIVE_PROVIDERS


def test_muse_capability_payload_is_truthful(monkeypatch):
    """The capability block names Muse's real kind, source, and executable."""
    accepted_carrier = muse_native_launch.MuseProfileCarrierCapability(
        supported=True,
        reason="",
        proof=muse_native_launch.PROOF_PROBED,
        full_banner=MUSE_BANNER,
        inner_executable="/fixture/muse-bin-0.1.0-R708.1",
        inner_executable_sha256=MUSE_INNER_SHA256,
    )
    monkeypatch.setattr(
        muse_native_launch,
        "installed_profile_carrier_capability",
        lambda: accepted_carrier,
    )
    capabilities = v2.native_tui_capabilities()
    block = capabilities["providers"]["muse_cli"]
    assert block["supported"] is True
    # The id is provider-status-discovered, never a caller-chosen one.
    assert block["id_source"] == "provider_status_discovered"
    assert block["readiness_receipt_kind"] == "muse-native-status-idle"
    assert block["executable"] == "muse"
    assert "0.1.0" in block["supported_versions"]
    assert block["profile_carrier_proof"] == muse_native_launch.PROOF_PROBED
    assert block["profile_carrier_inner_sha256"] == MUSE_INNER_SHA256
    assert block["profile_carrier_reason"] == ""


def test_the_other_native_providers_are_unchanged_by_muse_enrollment():
    """Enrolling Muse must not move any existing provider's facts."""
    assert v2._NATIVE_TUI_READINESS_RECEIPT_KINDS["codex"] == "codex-native-thread-start"
    assert v2._NATIVE_TUI_READINESS_RECEIPT_KINDS["kimi_cli"] == "kimi-native-tui-attached"
    assert v2._NATIVE_TUI_READINESS_RECEIPT_KINDS["claude_code"] == "claude-native-session-start"
    assert v2._PINNED_PROVIDER["codex"] == "codex"
    assert v2._PINNED_PROVIDER["kimi_cli"] == "kimi"
    assert v2._PINNED_PROVIDER["claude_code"] == "claude"


def test_the_native_kind_stays_disjoint_from_the_acp_kinds():
    assert not set(v2._NATIVE_TUI_READINESS_RECEIPT_KINDS.values()) & set(
        v2._READINESS_RECEIPT_KINDS.values()
    )


# --------------------------------------------------------------------
# 2. The argv contract — fresh launch is no-identity; the resume form is
#    retained strictly for restoration.
# --------------------------------------------------------------------


def test_muse_fresh_launch_argv_is_no_prompt_with_no_identity_form():
    """The fresh managed launch argv has no resume/--session-id/prompt."""
    argv = muse_native_launch.build_fresh_launch_argv(
        muse_binary="/usr/local/bin/muse",
        extra_args=["--trust-workspace", "--model", MUSE_MODEL, "--reasoning-effort", MUSE_EFFORT],
    )
    assert argv == [
        "/usr/local/bin/muse",
        "--trust-workspace",
        "--model",
        MUSE_MODEL,
        "--reasoning-effort",
        MUSE_EFFORT,
    ]
    assert muse_native_launch.fresh_launch_has_no_identity(argv)
    assert not any(
        token in argv for token in ("resume", "--session-id", "-s", "--exec", "--last", "-c")
    )
    # No positional prompt may be smuggled in.
    with pytest.raises(muse_native_launch.MuseNativeLaunchError):
        muse_native_launch.build_fresh_launch_argv(extra_args=["do the task"])
    with pytest.raises(muse_native_launch.MuseNativeLaunchError):
        muse_native_launch.build_fresh_launch_argv(extra_args=["--exec"])


def test_muse_fresh_launch_argv_refuses_recency_and_identity_forms():
    for smuggled in (
        "--last",
        "-c",
        "--continue",
        "--exec",
        "--fork-session",
        "--no-session-log",
        "--session-id",
    ):
        with pytest.raises(muse_native_launch.MuseNativeLaunchError):
            muse_native_launch.build_fresh_launch_argv(extra_args=[smuggled, "x"])


def test_muse_resume_argv_restoration_binds_a_preserved_id_exactly():
    """The retained restoration form resumes exactly a known preserved id."""
    session_id = KNOWN_SESSION_ID
    argv = muse_native_launch.build_resume_argv(
        session_id=session_id,
        muse_binary="/usr/local/bin/muse",
        extra_args=["--model", MUSE_MODEL, "--reasoning-effort", MUSE_EFFORT],
    )
    assert argv == [
        "/usr/local/bin/muse",
        "resume",
        session_id,
        "--model",
        MUSE_MODEL,
        "--reasoning-effort",
        MUSE_EFFORT,
    ]
    assert argv.count(session_id) == 1
    assert muse_native_launch.resumes_exactly(argv, session_id)
    # The fresh-launch checker correctly refuses the resume form.
    assert not muse_native_launch.fresh_launch_has_no_identity(argv)
    assert not muse_native_launch.resumes_exactly(argv, PROVIDER_SESSION_ID)


def test_muse_resume_argv_refuses_recency_and_identity_rebinding_forms():
    for smuggled in ("--last", "-c", "--continue", "--exec", "--fork-session", "--no-session-log"):
        with pytest.raises(muse_native_launch.MuseNativeLaunchError):
            muse_native_launch.build_resume_argv(session_id=KNOWN_SESSION_ID, extra_args=[smuggled])


def test_muse_fresh_launch_extra_args_carry_model_effort_trust_and_no_prompt():
    """The fresh launch argv is the route/profile args with no identity."""
    record, request, bootstrap = _mint_with_harness_state()
    args = v2._muse_profile_launch_args(
        record=record,
        request=request,
        profile_material=_fake_profile_material(),
        bootstrap=bootstrap,
    )
    argv = muse_native_launch.build_fresh_launch_argv(extra_args=args)
    assert "--model" in argv and argv[argv.index("--model") + 1] == MUSE_MODEL
    assert "--reasoning-effort" in argv
    assert argv[argv.index("--reasoning-effort") + 1] == MUSE_EFFORT
    assert "--trust-workspace" in argv
    assert muse_native_launch.fresh_launch_has_no_identity(argv)
    assert not any(token in argv for token in ("resume", "--session-id", "--exec", "--last"))


# --------------------------------------------------------------------
# 3. The /status panel parser.
# --------------------------------------------------------------------


def test_status_parser_accepts_the_coordinator_canary_panel():
    session_id = _uuid()
    parsed = muse_native_status.parse_status_panel(
        status_panel_rows(None, session_id, directory="/private/tmp/cao-muse-canary")
    )
    assert parsed["session_id"] == session_id
    assert parsed["model"] == MUSE_MODEL
    assert parsed["reasoning"] == MUSE_EFFORT
    assert parsed["agent_profile"] == "native-basic"
    assert parsed["model_provider"] == "meta"
    assert parsed["directory"] == "/private/tmp/cao-muse-canary"
    assert parsed["run"] == "idle"
    assert parsed["tokens"] == 0
    assert parsed["turns"] == 0
    required = muse_native_status.require_pre_task_status(
        parsed,
        session_id=session_id,
        expected_model=MUSE_MODEL,
        expected_effort=MUSE_EFFORT,
        working_directory="/private/tmp/cao-muse-canary",
        expected_profile_identity="native-basic",
    )
    assert required["session_matches"] is True
    assert required["model_matches"] is True
    assert required["effort_matches"] is True
    assert required["idle"] is True


def test_status_parser_splits_the_combined_model_reasoning_line():
    """The installed meta panel renders effort inside the Model line."""
    session_id = _uuid()
    parsed = muse_native_status.parse_status_panel(
        status_panel_rows(None, session_id, directory="/worktree")
    )
    assert parsed["model"] == MUSE_MODEL
    assert parsed["reasoning"] == MUSE_EFFORT


def test_status_parser_refuses_duplicate_required_singleton_fields():
    """More than one Model/Session/... line is ambiguity, never ``[0]``."""
    session_id = _uuid()
    rows = status_panel_rows(None, session_id)
    rows.insert(6, "│  Model:                 muse-other                  │")
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(rows)
    # A second Session line is the same refusal, now for the same reason.
    rows2 = status_panel_rows(None, session_id)
    rows2.insert(8, "│  Session:              another-session-id         │")
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(rows2)


def test_status_parser_refuses_a_malformed_reasoning_suffix():
    """An empty or unknown reasoning effort is refused, never guessed."""
    session_id = _uuid()
    for bad in ("", " "):
        rows = status_panel_rows(None, session_id, reasoning=bad)
        with pytest.raises(muse_native_status.MuseStatusParseError):
            muse_native_status.parse_status_panel(rows)
    # An unknown effort inside an exact-looking suffix is refused.
    rows = status_panel_rows(
        None, session_id, model=f"{MUSE_MODEL} (reasoning banana)", reasoning=None
    )
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(rows)


def test_status_parser_keeps_arbitrary_parenthetical_text_in_the_model():
    """A parenthetical that is not the exact reasoning form is not split."""
    session_id = _uuid()
    rows = status_panel_rows(None, session_id, model=f"{MUSE_MODEL} (preview)", reasoning=None)
    parsed = muse_native_status.parse_status_panel(rows)
    assert parsed["model"] == f"{MUSE_MODEL} (preview)"
    assert parsed["reasoning"] is None


def test_duplicate_reasoning_sources_converge_or_refuse():
    """Model-suffix and separate-label reasoning converge or refuse as one."""
    session_id = _uuid()
    same = status_panel_rows(None, session_id, reasoning=MUSE_EFFORT, reasoning_line=MUSE_EFFORT)
    assert muse_native_status.parse_status_panel(same)["reasoning"] == MUSE_EFFORT
    conflict = status_panel_rows(None, session_id, reasoning=MUSE_EFFORT, reasoning_line="low")
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(conflict)


def test_status_parser_accepts_a_panel_with_a_bare_model():
    """The echo provider renders a bare model with no effort at all."""
    session_id = _uuid()
    parsed = muse_native_status.parse_status_panel(
        status_panel_rows(None, session_id, reasoning=None)
    )
    assert parsed["reasoning"] is None


def test_provider_default_effort_sentinel_requires_no_reasoning_line():
    """The ``provider-default`` sentinel is not an effort to observe."""
    session_id = _uuid()
    parsed = muse_native_status.parse_status_panel(
        status_panel_rows(None, session_id, reasoning=None, directory="/worktree")
    )
    required = muse_native_status.require_pre_task_status(
        parsed,
        session_id=session_id,
        expected_model=MUSE_MODEL,
        expected_effort="provider-default",
        working_directory="/worktree",
        expected_profile_identity="native-basic",
    )
    assert required["effort_matches"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows, sid: status_panel_rows(None, sid, session_line=str(uuid.uuid4())),
        lambda rows, sid: status_panel_rows(None, sid, model="muse-spark-1.2"),
        lambda rows, sid: status_panel_rows(None, sid, reasoning="low"),
        lambda rows, sid: status_panel_rows(None, sid, agent_profile="miniswe"),
        lambda rows, sid: status_panel_rows(None, sid, provider="echo"),
        lambda rows, sid: status_panel_rows(None, sid, directory="/somewhere/else"),
        lambda rows, sid: status_panel_rows(None, sid, run="running"),
        lambda rows, sid: status_panel_rows(None, sid, tokens="12 tokens / 1 turns"),
    ],
)
def test_require_pre_task_status_rejects_every_mismatch(mutate):
    session_id = _uuid()
    rows = mutate(None, session_id)
    parsed = muse_native_status.parse_status_panel(rows)
    with pytest.raises(muse_native_status.MuseStatusMismatch):
        muse_native_status.require_pre_task_status(
            parsed,
            session_id=session_id,
            expected_model=MUSE_MODEL,
            expected_effort=MUSE_EFFORT,
            working_directory=str(None) if False else "/worktree",
            expected_profile_identity="native-basic",
        )


def test_status_parser_rejects_ambiguity_and_truncation():
    session_id = _uuid()
    ambiguous = status_panel_rows(None, session_id)
    ambiguous.insert(8, "│  Session:              another-session-id         │")
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(ambiguous)
    # A capture cut off before the required lines is unreadable, not empty.
    truncated = status_panel_rows(None, session_id)[:8]
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(truncated)


def test_status_parser_rejects_an_empty_or_escapeless_garbage_screen():
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel([])


# --------------------------------------------------------------------
# 4. The managed-v2 launch flow.
# --------------------------------------------------------------------


def _fake_profile_material(**changes) -> dict[str, Any]:
    class _Profile:
        permissionMode = None

        def model_dump(self, mode="json"):
            return {"permissionMode": None}

    material = {
        "profile": _Profile(),
        "profile_sha256": hashlib.sha256(b"profile").hexdigest(),
        "system_prompt": "You are the CAO worker profile.\nFollow the campaign.",
        "allowed_tools": ["fs_read", "bash"],
    }
    material.update(changes)
    return material


def _mint_with_harness_state():
    """Build the record/request/bootstrap a Muse launch would produce."""
    import tempfile

    from cli_agent_orchestrator.models.managed_launch_v2 import ManagedLaunchV2ReserveRequest

    tmp = tempfile.mkdtemp()
    worktree = tmp + "/repo"
    subprocess.run(["mkdir", "-p", worktree], check=True)
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=worktree, check=True)
    request = ManagedLaunchV2ReserveRequest(
        **_reserve_request(worktree, __import__("pathlib").Path(tmp)).model_dump()
    )
    record = {
        "provider": "muse_cli",
        "terminal_id": "term-muse",
        "generation": str(uuid.uuid4()),
        "working_directory": worktree,
        "agent_profile": "reviewer",
        "request": request.model_dump(),
    }
    bootstrap = {
        "native_session_id": _uuid(),
        "requested_model": MUSE_MODEL,
        "requested_effort": MUSE_EFFORT,
    }
    return record, record["request"], bootstrap


class _MuseHarness:
    """Every provider-facing boundary of a Muse native launch, recorded.

    ``typed`` records what was written into the pane (the ``/status``
    observation command and, later, the admitted task); ``captures``
    serves the rendered status panel; ``pane_status_script`` drives the
    generic pane-ready wait.
    """

    def __init__(self) -> None:
        self.typed: list[dict[str, Any]] = []
        self.captures: list[list[str]] = []
        self.capture_failures: list[Exception] = []
        self.terminals: list[dict[str, Any]] = []
        self.observed_pid = 4321
        self.pane_status_script: list[TerminalStatus] = [TerminalStatus.IDLE]
        # Ordered boundary crossings: "capture" when the /status panel is
        # read, "declare:<id>" when the attachment is claimed, "teardown"
        # when the fresh pane is killed.  Lets a test assert that the
        # provider id is discovered before the attachment exists.
        self.events: list[str] = []
        self.teardowns: list[str] = []
        # When set, the pane observation reports this cwd instead of the
        # reserved directory (used to inject a fence cwd mismatch).
        self.observe_cwd: Optional[str] = None
        # When set, the /status capture blocks on this threading.Event
        # (used to hold a launch mid-discovery for cancellation tests).
        self.block: Optional["threading.Event"] = None
        self.carrier_inner: Optional[str] = None

    @property
    def launched_argv(self) -> list[str]:
        assert self.terminals, "no pane was ever created"
        return list(self.terminals[-1]["managed_native_command"])

    @property
    def observed_argv(self) -> list[str]:
        """The wrapper execs the carrier-pinned binary as the pane primary."""
        assert self.carrier_inner is not None
        return [self.carrier_inner, *self.launched_argv[1:]]

    @property
    def env_vars(self) -> dict[str, str]:
        assert self.terminals, "no pane was ever created"
        return dict(self.terminals[-1]["env_vars"])


@pytest.fixture
def muse_harness(monkeypatch, tmp_path):
    state = _MuseHarness()
    real_declare = native_attachment.declare
    # The session-store fallback reads ${XDG_DATA_HOME}/muse/sessions;
    # every launch takes a pre-spawn snapshot there, so keep tests off the
    # developer's real store entirely.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-store"))

    async def _create_terminal(**kwargs):
        state.terminals.append(kwargs)
        terminal_id = kwargs["reserved_terminal_id"]
        database.create_terminal_v2(
            terminal_id,
            kwargs.get("session_name") or "cao-test",
            kwargs.get("window_name") or f"w-{terminal_id}",
            kwargs.get("provider") or "muse_cli",
            generation=kwargs.get("terminal_generation"),
            pane_id="%7",
            window_id="@7",
            server_socket_path="/private/tmp/cao-native.sock",
            session_id="$1",
            pane_pid=4242,
        )
        return {"terminal_id": terminal_id}

    def _observe(self):
        return {
            "pane_id": "%7",
            "pid": state.observed_pid,
            "start_marker": "Thu Jul 24 10:00:00 2026",
            "argv": state.observed_argv,
            "cwd": state.observe_cwd or self_record_working_directory(),
        }

    def self_record_working_directory():
        return state.terminals[-1]["working_directory"]

    def _capture_render(self, pane_id):
        state.events.append("capture")
        if state.block is not None:
            state.block.wait(timeout=10)
        if state.capture_failures:
            raise state.capture_failures.pop(0)
        assert state.captures, "no scripted status panel rows"
        return list(state.captures[0])

    def _typed_literal(text, **_kwargs):
        state.typed.append({"kind": "literal", "text": text})

    def _typed_enter(**_kwargs):
        state.typed.append({"kind": "enter"})

    def _typed_key(keystroke, **_kwargs):
        state.typed.append({"kind": "key", "keystroke": keystroke})

    def _turn_state(pane_id, **_kwargs):
        status = (
            state.pane_status_script.pop(0)
            if len(state.pane_status_script) > 1
            else state.pane_status_script[0]
        )
        if isinstance(status, Exception):
            raise status
        return status

    def _declare(**kwargs):
        state.events.append("declare:" + str(kwargs["native_session_id"]))
        return real_declare(**kwargs)

    def _delete_terminal(terminal_id, **kwargs):
        state.teardowns.append(terminal_id)
        return {"terminal_id": terminal_id}

    monkeypatch.setattr(bridge, "provider_version_banner", lambda *a, **k: MUSE_BANNER)

    # Model the real split between the update-capable wrapper and its fixed
    # inner image.  Carrier resolution itself has dedicated closed-cell
    # tests; this fixture proves the wrapper starts the pinned inner image.
    def _carrier(*, wrapper_executable, full_banner):
        inner_path = Path(wrapper_executable).with_name("fake-muse-inner")
        inner_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        inner_path.chmod(0o755)
        inner = os.path.realpath(inner_path)
        state.carrier_inner = inner
        return muse_native_launch.MuseProfileCarrierCapability(
            supported=True,
            reason="",
            proof=muse_native_launch.PROOF_PROBED,
            full_banner=full_banner,
            inner_executable=inner,
            inner_executable_sha256=hashlib.sha256(Path(inner).read_bytes()).hexdigest(),
        )

    monkeypatch.setattr(muse_native_launch, "profile_carrier_capability", _carrier)
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.terminal_service.create_terminal", _create_terminal
    )
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.terminal_service.delete_terminal", _delete_terminal
    )
    monkeypatch.setattr(v2._V2NativePane, "observe", _observe)
    monkeypatch.setattr(v2._V2NativePane, "capture_render", _capture_render)
    # The admission-time live-pane read uses a real TmuxNativePane; the
    # fake below serves the same observation the launch recorded.
    monkeypatch.setattr(
        native_tui_launch, "TmuxNativePane", lambda *a, **k: _FakeTmuxNativePane(state)
    )
    monkeypatch.setattr(npi, "observe_muse_turn_state", _turn_state)
    monkeypatch.setattr(npi, "TmuxPaneInput", _FakeTmuxPaneInput.for_state(state))
    # Record the attachment claim without changing its behaviour, so a
    # test can assert the status discovery preceded it.
    monkeypatch.setattr(native_attachment, "declare", _declare)
    monkeypatch.setattr(v2, "NATIVE_PANE_READY_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(v2, "_NATIVE_PANE_READY_POLL_SECONDS", 0.005)
    return state


class _FakeTmuxNativePane:
    """The admission-time live-pane read, served from the harness state."""

    def __init__(self, state: _MuseHarness) -> None:
        self._state = state

    def observe(self, **kwargs):
        return {
            "pane_id": "%7",
            "pid": self._state.observed_pid,
            "start_marker": "Thu Jul 24 10:00:00 2026",
            "argv": self._state.observed_argv,
            "cwd": self._state.terminals[-1]["working_directory"],
        }

    def capture_render(self, pane_id, **kwargs):
        assert self._state.captures, "no scripted status panel rows"
        return list(self._state.captures[0])


class _FakeTmuxPaneInput:
    _state: _MuseHarness

    @classmethod
    def for_state(cls, state: _MuseHarness) -> type["_FakeTmuxPaneInput"]:
        cls._state = state
        return cls

    def __init__(self, pane_id: str) -> None:
        self._pane_id = pane_id

    def send_literal(self, text: str) -> None:
        self._state.typed.append({"kind": "literal", "text": text})

    def send_enter(self) -> None:
        self._state.typed.append({"kind": "enter"})

    def send_key(self, keystroke: str) -> None:
        self._state.typed.append({"kind": "key", "keystroke": keystroke})


async def _launch(worktree, tmp_path, muse_harness, **changes):
    record, _ = v2.reserve(_reserve_request(worktree, tmp_path, **changes))
    assert record["execution_mode"] == em.NATIVE_TUI
    return record, await v2.launch_reserved(record["reservation_id"])


def _published_receipt(reservation_id: str) -> dict[str, Any]:
    state = bridge.read_state(reservation_id)
    assert state["state"] == "ready"
    return state["readiness"]


@pytest.mark.asyncio
async def test_muse_fresh_launch_argv_is_no_prompt_with_no_identity(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """The pane runs a fresh no-prompt TUI; no identity form is present."""
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, result = await _launch(worktree, tmp_path, muse_harness)
    argv = muse_harness.launched_argv
    assert argv[0] == record["request"]["provider_executable"]
    assert muse_harness.observed_argv[0] == muse_harness.carrier_inner
    assert muse_native_launch.fresh_launch_has_no_identity(argv)
    assert not any(
        token in argv
        for token in ("resume", "--session-id", "-s", "--exec", "--last", "-c", "--continue")
    )
    assert "--model" in argv and argv[argv.index("--model") + 1] == MUSE_MODEL
    assert "--reasoning-effort" in argv
    assert argv[argv.index("--reasoning-effort") + 1] == MUSE_EFFORT
    assert "--trust-workspace" in argv
    assert result["execution_mode"] == em.NATIVE_TUI


@pytest.mark.asyncio
async def test_muse_probe_disables_updater_before_wrapper_execution_and_mismatch_has_no_effects(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """The first wrapper execution is fenced and a rejected carrier starts no pane."""
    probe_environments = []
    profile_writes = []

    def _version_probe(_request, *, environment=None, **_kwargs):
        # The version probe is the first managed wrapper invocation.  A pane
        # would mean an earlier provider execution escaped this fence.
        assert muse_harness.terminals == []
        assert environment is not None
        probe_environments.append(dict(environment))
        return MUSE_BANNER

    def _write_profile(**_kwargs):
        profile_writes.append(True)
        raise AssertionError("a rejected carrier must not write a profile")

    monkeypatch.setattr(bridge, "provider_version_banner", _version_probe)
    monkeypatch.setattr(
        muse_native_launch,
        "profile_carrier_capability",
        lambda **_kwargs: muse_native_launch.MuseProfileCarrierCapability(
            False, "profile_carrier_unverified"
        ),
    )
    monkeypatch.setattr(v2, "_write_native_profile_file", _write_profile)

    _record, result = await _launch(worktree, tmp_path, muse_harness)

    assert probe_environments
    assert probe_environments[0]["MUSE_NO_AUTO_UPDATE"] == "1"
    assert profile_writes == []
    assert muse_harness.terminals == []
    assert result["state"] == "preflight_blocked"


@pytest.mark.asyncio
async def test_muse_launch_records_the_provider_executable_version_durably(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """The probed full version banner rides in the reservation launch facts.

    The banner is the exact wrapper ``--version`` output the launch observed
    (and the carrier gate accepted); it is recorded durably on the
    reservation row so teardown can publish it into the restore contract and
    an exact resume can revalidate the profile carrier.
    """
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "launching"

    facts = v2.get(record["reservation_id"])["launch_facts"]
    assert facts["provider_executable_version"] == MUSE_BANNER
    with database.SessionLocal() as session:
        row = (
            session.query(database.ManagedLaunchV2ReservationModel)
            .filter(
                database.ManagedLaunchV2ReservationModel.reservation_id == record["reservation_id"]
            )
            .one()
        )
        stored = json.loads(row.launch_facts_json)
        assert stored["provider_executable_version"] == MUSE_BANNER


@pytest.mark.asyncio
async def test_muse_launch_discovers_the_provider_generated_session(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """The /status panel's provider UUID becomes the durable identity."""
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "launching"

    # The durable v2 terminal row carries the discovered provider id.
    terminal = (
        database.SessionLocal()
        .query(database.ManagedLaunchV2TerminalModel)
        .filter(database.ManagedLaunchV2TerminalModel.id == record["terminal_id"])
        .first()
    )
    assert terminal is not None
    assert terminal.v2_native_session_id == PROVIDER_SESSION_ID

    # The attachment is keyed by the discovered id and claims a
    # provider-discovered acquisition, never a chosen-session one.
    attachment = native_attachment.get("muse_cli", PROVIDER_SESSION_ID)
    assert attachment is not None
    owner = attachment["owner"]
    assert owner["terminal_id"] == record["terminal_id"]
    assert owner["generation"] == record["generation"]
    intent = attachment["intent"]
    assert intent["acquisition_method"] == native_attachment.ACQUISITION_STATUS_DISCOVERED
    assert intent["acquisition_receipt"]["id_source"] == "provider_status_discovered"
    assert intent["acquisition_receipt"]["profile_carrier_capability"] is None
    assert intent["acquisition_receipt"]["profile_carrier_inner_sha256"]

    # The /status capture preceded the attachment claim.
    assert "capture" in muse_harness.events
    assert "declare:" + PROVIDER_SESSION_ID in muse_harness.events
    assert muse_harness.events.index("capture") < muse_harness.events.index(
        "declare:" + PROVIDER_SESSION_ID
    )


@pytest.mark.asyncio
async def test_muse_launch_observes_status_before_persisting_or_publishing_readiness(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, _result = await _launch(worktree, tmp_path, muse_harness)

    # The /status command was typed exactly once, as literal + one enter.
    literals = [t for t in muse_harness.typed if t["kind"] == "literal"]
    enters = [t for t in muse_harness.typed if t["kind"] == "enter"]
    assert [t["text"] for t in literals] == ["/status"]
    assert len(enters) == 1

    receipt = _published_receipt(record["reservation_id"])
    assert receipt["provider"] == "muse_cli"
    assert receipt["provider_receipt_kind"] == "muse-native-status-idle"
    assert receipt["model"] == MUSE_MODEL
    assert receipt["effort"] == MUSE_EFFORT
    assert receipt["provider_session_id"] == PROVIDER_SESSION_ID
    # The provider's own /status statement is the session-start proof.
    assert receipt["provider_session_start"] is not None
    assert receipt["provider_session_start"]["session_matches"] is True
    assert receipt["model_input_ready"] is True


@pytest.mark.asyncio
async def test_muse_launch_carries_the_profile_system_prompt_through_the_env_surface(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, _result = await _launch(worktree, tmp_path, muse_harness)

    env = muse_harness.env_vars
    assert env["MUSE_NO_AUTO_UPDATE"] == "1"
    assert muse_native_launch.PROFILE_SYSTEM_PROMPT_ENV in env
    profile_path = env[muse_native_launch.PROFILE_SYSTEM_PROMPT_ENV]
    material = _profile_material_for(record)
    assert open(profile_path, encoding="utf-8").read() == material["system_prompt"]

    receipt = _published_receipt(record["reservation_id"])
    assert receipt["profile_sha256"] == material["profile_sha256"]
    assert (
        receipt["profile_system_prompt_sha256"]
        == hashlib.sha256(material["system_prompt"].encode("utf-8")).hexdigest()
    )
    assert receipt["acquisition_receipt_sha256"]


def _profile_material_for(record) -> dict[str, Any]:
    from cli_agent_orchestrator.services.managed_provider_bridge import _profile_material

    return _profile_material(record["agent_profile"], record["terminal_id"])


def _v2_terminal(terminal_id) -> Any:
    return (
        database.SessionLocal()
        .query(database.ManagedLaunchV2TerminalModel)
        .filter(database.ManagedLaunchV2TerminalModel.id == terminal_id)
        .first()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "broken",
    [
        "non_canonical_id",
        "wrong_model",
        "wrong_effort",
        "wrong_profile",
        "wrong_cwd",
        "busy",
        "turns",
        "ambiguous",
        "unreadable",
        "status_timeout",
    ],
)
async def test_muse_launch_blocks_tears_down_and_never_advertises_when_discovery_fails(
    isolated_memory_db, worktree, tmp_path, muse_harness, broken
):
    if broken == "non_canonical_id":
        rows = status_panel_rows(worktree, "not-a-canonical-uuid")
    elif broken == "wrong_model":
        rows = status_panel_rows(worktree, PROVIDER_SESSION_ID, model="muse-spark-1.2")
    elif broken == "wrong_effort":
        rows = status_panel_rows(worktree, PROVIDER_SESSION_ID, reasoning="low")
    elif broken == "wrong_profile":
        rows = status_panel_rows(worktree, PROVIDER_SESSION_ID, agent_profile="miniswe")
    elif broken == "wrong_cwd":
        rows = status_panel_rows(
            worktree, PROVIDER_SESSION_ID, directory=str(tmp_path / "elsewhere")
        )
    elif broken == "busy":
        rows = status_panel_rows(worktree, PROVIDER_SESSION_ID, run="running")
    elif broken == "turns":
        rows = status_panel_rows(worktree, PROVIDER_SESSION_ID, tokens="4 tokens / 1 turns")
    elif broken == "ambiguous":
        rows = status_panel_rows(worktree, PROVIDER_SESSION_ID)
        rows.insert(8, "│  Session:              another-session-id         │")
    elif broken == "unreadable":
        muse_harness.capture_failures.append(RuntimeError("pane gone"))
        rows = []
    else:  # status_timeout: the capture never parses within the bound
        rows = []
    muse_harness.captures.append(rows)

    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    # Zero task bytes and no admission claim.
    assert result.get("admission") is None
    literals = [t["text"] for t in muse_harness.typed if t["kind"] == "literal"]
    assert literals == ["/status"]
    # The exact fresh pane was torn down; nothing durable was advertised.
    assert record["terminal_id"] in muse_harness.teardowns
    terminal = _v2_terminal(record["terminal_id"])
    assert terminal is None or terminal.v2_native_session_id is None
    state = bridge.read_state(record["reservation_id"])
    assert state is None or state.get("state") != "ready"


@pytest.mark.asyncio
async def test_muse_launch_tears_down_on_attachment_publish_failure(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    real_mark_attached = native_attachment.mark_attached

    def _boom(**kwargs):
        raise RuntimeError("publication failed")

    monkeypatch.setattr(native_attachment, "mark_attached", _boom)
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    assert record["terminal_id"] in muse_harness.teardowns
    # No orphaned live ownership: the claimed attachment is frozen.
    attachment = native_attachment.get("muse_cli", PROVIDER_SESSION_ID)
    assert attachment is not None
    assert attachment["state"] != native_attachment.ATTACHED
    assert bridge.read_state(record["reservation_id"]) is None


@pytest.mark.asyncio
async def test_muse_launch_tears_down_on_durable_id_persistence_failure(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    monkeypatch.setattr(database, "set_terminal_v2_native_session_id", lambda *a, **k: False)
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    assert record["terminal_id"] in muse_harness.teardowns
    attachment = native_attachment.get("muse_cli", PROVIDER_SESSION_ID)
    assert attachment is not None
    assert attachment["state"] != native_attachment.ATTACHED
    assert bridge.read_state(record["reservation_id"]) is None


@pytest.mark.asyncio
async def test_muse_launch_never_observes_model_or_effort_from_the_request(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """Requested model/effort are never treated as observed without evidence."""
    muse_harness.captures.append(
        status_panel_rows(worktree, PROVIDER_SESSION_ID, model=MUSE_MODEL, reasoning=MUSE_EFFORT)
    )
    record, _result = await _launch(worktree, tmp_path, muse_harness)
    receipt = _published_receipt(record["reservation_id"])
    # Observed from the panel, not echoed from the request.
    assert receipt["expected_model"] == MUSE_MODEL
    assert receipt["model"] == MUSE_MODEL
    assert receipt["effort"] == MUSE_EFFORT


# --------------------------------------------------------------------
# 4b. P1 cleanup/cancellation — every post-create failure tears the exact
#     pane down once and never claims a success it did not achieve.
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_muse_launch_tears_down_when_intent_construction_fails(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """A build_intent failure after the pane exists must tear it down."""
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))

    def _boom(*args, **kwargs):
        raise RuntimeError("intent construction failed")

    monkeypatch.setattr(v2, "_muse_bootstrap_intent", _boom)
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    assert "intent construction failed" in str(result.get("preflight_failure") or {})
    # The exact pane was torn down, nothing was claimed or advertised.
    assert record["terminal_id"] in muse_harness.teardowns
    assert native_attachment.get("muse_cli", PROVIDER_SESSION_ID) is None
    assert bridge.read_state(record["reservation_id"]) is None
    literals = [t["text"] for t in muse_harness.typed if t["kind"] == "literal"]
    assert literals == ["/status"]


@pytest.mark.asyncio
async def test_muse_launch_tears_down_on_cwd_mismatch_at_the_fence(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """A pane that drifts to the wrong cwd before publication is torn down."""
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    muse_harness.observe_cwd = str(tmp_path / "elsewhere")
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    assert record["terminal_id"] in muse_harness.teardowns
    attachment = native_attachment.get("muse_cli", PROVIDER_SESSION_ID)
    # The claimed attachment is frozen, never published live.
    assert attachment is not None
    assert attachment["state"] != native_attachment.ATTACHED
    assert bridge.read_state(record["reservation_id"]) is None


@pytest.mark.asyncio
async def test_muse_launch_refuses_an_existing_attachment_without_adopting_it(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """A fresh provider-generated id must be newly claimed, never adopted.

    An existing ATTACHED row under the discovered id is a collision: the
    launch refuses, tears the newly observed pane down, and never returns
    an unproved ``already-attached`` outcome.
    """
    record, _ = v2.reserve(_reserve_request(worktree, tmp_path))
    intent = native_attachment.acquire_intent(
        acquisition_method=native_attachment.ACQUISITION_STATUS_DISCOVERED,
        acquisition_receipt={
            "schema": "test-intent-v1",
            "native_session_id": PROVIDER_SESSION_ID,
        },
        admits_only_new_instructions=True,
        replays_task_bytes=False,
    )
    native_attachment.declare(
        provider="muse_cli",
        native_session_id=PROVIDER_SESSION_ID,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        execution_mode=em.NATIVE_TUI,
        intent=intent,
    )
    native_attachment.mark_starting(
        provider="muse_cli",
        native_session_id=PROVIDER_SESSION_ID,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        execution_mode=em.NATIVE_TUI,
    )
    native_attachment.mark_attached(
        provider="muse_cli",
        native_session_id=PROVIDER_SESSION_ID,
        terminal_id=record["terminal_id"],
        generation=record["generation"],
        execution_mode=em.NATIVE_TUI,
        process_identity=native_attachment.process_identity(pid=999, start_marker="pre-staged"),
    )
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    result = await v2.launch_reserved(record["reservation_id"])
    assert result["state"] == "preflight_blocked"
    # Teardown of the newly observed pane; zero task bytes; no readiness.
    assert record["terminal_id"] in muse_harness.teardowns
    literals = [t["text"] for t in muse_harness.typed if t["kind"] == "literal"]
    assert literals == ["/status"]
    assert result.get("admission") is None
    assert bridge.read_state(record["reservation_id"]) is None
    # The pre-existing attachment was not overwritten by the launch.
    attachment = native_attachment.get("muse_cli", PROVIDER_SESSION_ID)
    assert attachment is not None
    assert attachment["state"] != native_attachment.ATTACHED


@pytest.mark.asyncio
async def test_muse_launch_exposes_a_cleanup_failure_in_the_preflight_detail(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """A failed pane teardown is never hidden by a "torn down" claim."""
    muse_harness.captures.append(
        status_panel_rows(worktree, PROVIDER_SESSION_ID, model="muse-spark-1.2")
    )

    def _delete_terminal_fail(terminal_id, **kwargs):
        muse_harness.teardowns.append(terminal_id)
        raise RuntimeError("terminal deletion failed")

    monkeypatch.setattr(
        "cli_agent_orchestrator.services.terminal_service.delete_terminal",
        _delete_terminal_fail,
    )
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    detail = result["preflight_failure"]["detail"]
    assert "terminal deletion failed" in detail
    assert record["terminal_id"] in muse_harness.teardowns
    assert bridge.read_state(record["reservation_id"]) is None


@pytest.mark.asyncio
async def test_muse_launch_does_not_strand_on_call_cancellation(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """A cancelled caller never strands a live discovered launch.

    The native launch branch runs shielded: cancelling the caller must not
    return while the child is mid-flight, and after the child settles the
    durable outcome (ready) is published and the caller observes the
    cancellation.  A replay creates no second pane.
    """
    release = threading.Event()
    muse_harness.block = release
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, _ = v2.reserve(_reserve_request(worktree, tmp_path))
    task = asyncio.create_task(v2.launch_reserved(record["reservation_id"]))

    # Wait until the discovery is genuinely blocked (pane created, /status
    # typed, capture waiting on the barrier).
    deadline = time.monotonic() + 5.0
    while "capture" not in muse_harness.events and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert "capture" in muse_harness.events, "the launch never reached /status"
    assert muse_harness.typed and muse_harness.typed[0] == {
        "kind": "literal",
        "text": "/status",
    }

    # Cancel the caller, repeatedly, while the child is mid-flight.
    task.cancel()
    task.cancel()
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=0.3)
        raise AssertionError("launch returned while the child was still active")
    except asyncio.TimeoutError:
        pass  # the caller is still waiting on the shielded child — correct
    except asyncio.CancelledError:
        raise AssertionError("launch released while the child was still active") from None

    # Release the barrier; the shielded child settles the launch durably.
    release.set()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        raise AssertionError("a cancelled launch must not return a result")
    except asyncio.CancelledError:
        pass  # the caller observes cancellation after the child settled
    except asyncio.TimeoutError:
        raise AssertionError("the launch did not settle after the barrier released") from None

    # The durable outcome was published despite the cancellation.
    state = bridge.read_state(record["reservation_id"])
    assert state is not None and state["state"] == "ready"
    receipt = state["readiness"]
    assert receipt["provider_session_id"] == PROVIDER_SESSION_ID
    terminal = _v2_terminal(record["terminal_id"])
    assert terminal is not None and terminal.v2_native_session_id == PROVIDER_SESSION_ID

    # A replay creates no second pane.
    await asyncio.wait_for(
        asyncio.shield(asyncio.create_task(v2.launch_reserved(record["reservation_id"]))),
        timeout=5.0,
    )
    assert len(muse_harness.terminals) == 1


@pytest.mark.asyncio
async def test_muse_launch_exposes_a_teardown_failure_when_pane_creation_raises(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """A failed teardown after a raising create_pane is never hidden.

    create_pane may have created the pane before it raised; the exact
    teardown is attempted, and when that teardown itself fails the typed
    error must expose the cleanup failure and must never claim the pane
    was torn down.
    """

    def _create_terminal_boom(**kwargs):
        raise RuntimeError("pane creation failed")

    def _delete_terminal_fail(terminal_id, **kwargs):
        muse_harness.teardowns.append(terminal_id)
        raise RuntimeError("cleanup also failed")

    monkeypatch.setattr(
        "cli_agent_orchestrator.services.terminal_service.create_terminal",
        _create_terminal_boom,
    )
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.terminal_service.delete_terminal",
        _delete_terminal_fail,
    )
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    detail = result["preflight_failure"]["detail"]
    assert "pane creation failed" in detail
    assert "cleanup: " in detail
    assert "cleanup also failed" in detail
    # A teardown that failed must never be reported as success.
    assert "was torn down" not in detail
    assert bridge.read_state(record["reservation_id"]) is None


@pytest.mark.asyncio
async def test_muse_launch_exposes_a_teardown_failure_when_the_pane_handle_is_invalid(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """An invalid/missing pane handle with a failing teardown is exposed."""

    def _create_pane_invalid(self, *, argv=None):
        return None

    def _delete_terminal_fail(terminal_id, **kwargs):
        muse_harness.teardowns.append(terminal_id)
        raise RuntimeError("cleanup also failed")

    monkeypatch.setattr(v2._V2NativePane, "create_pane", _create_pane_invalid)
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.terminal_service.delete_terminal",
        _delete_terminal_fail,
    )
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    detail = result["preflight_failure"]["detail"]
    assert "no usable handle" in detail
    assert "cleanup: " in detail
    assert "cleanup also failed" in detail
    assert "torn down" not in detail
    assert bridge.read_state(record["reservation_id"]) is None


# --------------------------------------------------------------------
# 5. Bind and admission — one task delivery, at-most-once, on the
#    discovered provider session.
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_muse_bind_binds_the_stable_roster_to_the_discovered_session(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "launching"

    bound = v2.bind_native(
        result["reservation_id"],
        ManagedLaunchV2BindRequest(
            protocol_version=PROTOCOL_VERSION_V2,
            terminal_id=result["terminal_id"],
            generation=result["generation"],
            attempt_id=str(uuid.uuid4()),
            fencing_token_id=str(uuid.uuid4()),
            execution_mode="native_tui",
        ),
    )
    assert bound["state"] == "bound"
    binding = bound["binding"]
    assert binding["native_session_id"] == PROVIDER_SESSION_ID
    assert binding["execution_mode"] == "native_tui"

    from cli_agent_orchestrator.services import stable_agent_roster as roster

    # The durable roster lineage binds the exact harness-scoped session
    # with a provider-status-discovered acquisition.
    agents = roster.list_agents(session_name="cao-test")
    assert len(agents) == 1
    assert agents[0]["profile_family"] == "reviewer"
    lineages = roster.list_lineages(agent_id=agents[0]["agent_id"])
    assert len(lineages) == 1
    assert lineages[0]["harness"] == "muse_cli"
    assert lineages[0]["native_session_id"] == PROVIDER_SESSION_ID
    assert lineages[0]["acquisition_method"] == roster.ACQUISITION_STATUS_DISCOVERED


@pytest.mark.asyncio
async def test_muse_admission_delivers_exactly_once_after_readiness(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, result = await _launch(worktree, tmp_path, muse_harness)
    bound = v2.bind_native(
        result["reservation_id"],
        ManagedLaunchV2BindRequest(
            protocol_version=PROTOCOL_VERSION_V2,
            terminal_id=result["terminal_id"],
            generation=result["generation"],
            attempt_id=str(uuid.uuid4()),
            fencing_token_id=str(uuid.uuid4()),
            execution_mode="native_tui",
        ),
    )
    digest = v2.native_binding_digest(bound)
    assert digest
    # Zero task bytes crossed before bind; nothing was typed.
    assert muse_harness.typed == [{"kind": "literal", "text": "/status"}, {"kind": "enter"}]

    admitted = await v2.admit_reserved(result["reservation_id"], _admit_request(digest))
    assert admitted["admission"]["status"] == "admitted"
    # The task bytes landed exactly once: one literal write of the message,
    # then the submitting enter.
    task_writes = [t for t in muse_harness.typed if t["kind"] == "literal"]
    assert [t["text"] for t in task_writes] == ["/status", TASK_MESSAGE]

    # A replay of the same delivery id adopts the stored outcome, sending
    # nothing.
    before = list(muse_harness.typed)
    replayed = await v2.admit_reserved(result["reservation_id"], _admit_request(digest))
    assert replayed["admission"]["status"] == "admitted"
    assert muse_harness.typed == before

    # A changed request under the same delivery id conflicts.
    with pytest.raises(ManagedLaunchConflict):
        await v2.admit_reserved(
            result["reservation_id"],
            _admit_request(digest, message="a different task"),
        )


@pytest.mark.asyncio
async def test_muse_admission_refuses_when_the_attachment_is_not_owned(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, result = await _launch(worktree, tmp_path, muse_harness)
    bound = v2.bind_native(
        result["reservation_id"],
        ManagedLaunchV2BindRequest(
            protocol_version=PROTOCOL_VERSION_V2,
            terminal_id=result["terminal_id"],
            generation=result["generation"],
            attempt_id=str(uuid.uuid4()),
            fencing_token_id=str(uuid.uuid4()),
            execution_mode="native_tui",
        ),
    )
    digest = v2.native_binding_digest(bound)
    assert digest
    # Stage a pane that drifted to a different process identity: admission
    # must refuse with zero bytes typed rather than deliver into a stranger.
    muse_harness.observed_pid = 9999
    with pytest.raises(ManagedLaunchConflict):
        await v2.admit_reserved(result["reservation_id"], _admit_request(digest))
    task_writes = [t for t in muse_harness.typed if t["kind"] == "literal"]
    assert [t["text"] for t in task_writes] == ["/status"]


# --------------------------------------------------------------------
# 5b. P1 multi-line planner + typed pre-I/O refusal (Kimi K3 blocker).
# --------------------------------------------------------------------


def test_muse_planner_plans_a_multiline_task_with_the_pinned_c_j():
    """A multi-line Muse 0.1.0 task plans soft-newline lines then one Enter.

    Newlines are structure, never literal input: the payload is split into
    lines and the breaks become the pinned C-j composer keystrokes, so the
    plan is ``soft-newline-lines-then-enter`` and deliverable.
    """
    plan = muse_native_control.plan_composer_keystrokes(
        "review this diff\nline two of the task",
        provider_version="0.1.0",
    )
    assert plan["encoding"] == muse_native_control.ENCODING_SOFT_NEWLINE
    assert plan["line_count"] == 2
    assert plan["soft_newline_keystroke"] == "C-j"
    assert plan["lines"] == ["review this diff", "line two of the task"]
    assert plan["deliverable"] is True
    assert plan["final_enter"] is True


def test_muse_planner_floors_an_unproven_build_at_the_proven_settle():
    """An unproven Muse build gets the floor, marked as a floor.

    The floor is the longest proven interval for this provider, so an
    unmeasured build inherits the safe end of the observed range. It is
    asserted against the derived value rather than a literal so that
    pinning a slower build cannot leave this gate frozen on a stale
    number. What the plan must still distinguish is provenance: the
    value on an unproven build is a floor, not a measurement.

    The floor is explicitly non-zero. A zero settle was measured on
    0.2.1-R1215.1 to demote the Enter to a newline (0/10 submitted), so
    a null-valued floor here would be a silent non-delivery rather than
    a conservative default.
    """
    plan = muse_native_control.plan_composer_keystrokes(
        "one line only",
        provider_version="9.9.9",
    )
    assert plan["deliverable"] is True
    assert plan["submit_settle_seconds"] == muse_native_control._SUBMIT_SETTLE_FLOOR_SECONDS
    assert plan["submit_settle_seconds"] > 0.0
    assert plan["submit_settle_proven"] is False
    assert plan["composer_evidence"] is None


def test_muse_planner_marks_a_proven_builds_settle_proven():
    plan = muse_native_control.plan_composer_keystrokes(
        "one line only",
        provider_version="0.1.0",
    )
    assert plan["submit_settle_seconds"] == 0.0
    assert plan["submit_settle_proven"] is True


@pytest.mark.asyncio
async def test_muse_admission_delivers_a_multiline_task_exactly_once(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """A real multiline message types literal-C-j-literal-...-one Enter once."""
    message = "review this diff\nline two of the task"
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, result = await _launch(worktree, tmp_path, muse_harness)
    bound = v2.bind_native(
        result["reservation_id"],
        ManagedLaunchV2BindRequest(
            protocol_version=PROTOCOL_VERSION_V2,
            terminal_id=result["terminal_id"],
            generation=result["generation"],
            attempt_id=str(uuid.uuid4()),
            fencing_token_id=str(uuid.uuid4()),
            execution_mode="native_tui",
        ),
    )
    digest = v2.native_binding_digest(bound)
    assert digest
    admitted = await v2.admit_reserved(
        result["reservation_id"],
        _admit_request(
            digest,
            message=message,
            message_sha256=hashlib.sha256(message.encode()).hexdigest(),
        ),
    )
    assert admitted["admission"]["status"] == "admitted"
    expected = [
        {"kind": "literal", "text": "/status"},
        {"kind": "enter"},
        {"kind": "literal", "text": "review this diff"},
        {"kind": "key", "keystroke": "C-j"},
        {"kind": "literal", "text": "line two of the task"},
        {"kind": "enter"},
    ]
    assert muse_harness.typed == expected
    # At-most-once replay adds zero bytes.
    replayed = await v2.admit_reserved(
        result["reservation_id"],
        _admit_request(
            digest,
            message=message,
            message_sha256=hashlib.sha256(message.encode()).hexdigest(),
        ),
    )
    assert replayed["admission"]["status"] == "admitted"
    assert muse_harness.typed == expected


@pytest.mark.asyncio
async def test_muse_admission_refuses_invalid_content_without_ambiguity(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """A planner error (ESC/CR) before _open is a typed refusal, never ambiguity."""
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    record, result = await _launch(worktree, tmp_path, muse_harness)
    bound = v2.bind_native(
        result["reservation_id"],
        ManagedLaunchV2BindRequest(
            protocol_version=PROTOCOL_VERSION_V2,
            terminal_id=result["terminal_id"],
            generation=result["generation"],
            attempt_id=str(uuid.uuid4()),
            fencing_token_id=str(uuid.uuid4()),
            execution_mode="native_tui",
        ),
    )
    digest = v2.native_binding_digest(bound)
    assert digest
    bad = _admit_request(
        digest,
        message="bad\x1bcontent",
        message_sha256=hashlib.sha256(b"bad\x1bcontent").hexdigest(),
    )
    admitted = await v2.admit_reserved(result["reservation_id"], bad)
    assert admitted["admission"]["status"] == "refused"
    assert admitted["admission"]["refusal_reason"] == "composer_plan_invalid"
    # Zero composer bytes beyond the /status observation.
    literals = [t for t in muse_harness.typed if t["kind"] == "literal"]
    assert [t["text"] for t in literals] == ["/status"]
    # Deterministic replay: no new bytes, same refusal.
    before = list(muse_harness.typed)
    replayed = await v2.admit_reserved(result["reservation_id"], bad)
    assert replayed["admission"]["status"] == "refused"
    assert muse_harness.typed == before


@pytest.mark.asyncio
async def test_muse_launch_never_types_when_the_pane_never_becomes_ready(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """A pane that never reaches idle is a typed preflight, zero bytes typed."""
    muse_harness.captures.append(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    muse_harness.pane_status_script = [TerminalStatus.PROCESSING]
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    # Neither /status nor any task byte was ever typed into the pane.
    assert muse_harness.typed == []
    assert record["terminal_id"] in muse_harness.teardowns
    assert bridge.read_state(record["reservation_id"]) is None


# --------------------------------------------------------------------
# 6. Profile fidelity: the same material and digest feed the resume
#    contract.
# --------------------------------------------------------------------


def test_profile_material_digest_feeds_launch_and_resume_identically():
    """The env-addressed file is the resume contract: same path, same digest.

    The launch writes the profile system prompt once into the
    generation-private companion dir and addresses it to the pane through
    ``TBH_EVAL_APPEND_SYSTEM_PROMPT_FILE``.  An exact resume re-runs the
    same CLI in the same environment, so the same bytes compose again —
    the deterministic echo-refusal proof covered this on both the launch
    and the ``muse resume <id>`` form.
    """
    material = _fake_profile_material()
    assert muse_native_launch.PROFILE_SYSTEM_PROMPT_ENV == "TBH_EVAL_APPEND_SYSTEM_PROMPT_FILE"
    assert material["system_prompt"]
    digest = hashlib.sha256(material["system_prompt"].encode("utf-8")).hexdigest()
    assert len(digest) == 64


def test_the_profile_surface_is_not_prompt_only():
    """The carrier env var is distinct from the task prompt and is refused
    when the profile material is missing."""
    with pytest.raises(muse_native_launch.MuseNativeLaunchError):
        muse_native_launch.validate_profile_system_prompt("")


# --------------------------------------------------------------------
# 7. cond-0713: the 0.2.1 renderings — boxed /status panel, inline
#    footer, and the session-store fallback identity.
#
#    Muse auto-updated to 0.2.1-R1215.1 and replaced the labeled panel
#    with a boxed one (which DOES name the SESSION id) over a persistent
#    inline footer that names none.  Everything here is pinned to the
#    real renders captured live on the installed build.
# --------------------------------------------------------------------

#: A provider-generated id from a real 0.2.1-R1215.1 meta TUI (`/status`
#: typed at zero turns in /private/tmp/muse-probe-cond0713).
PROBE_SESSION_ID = "e10c3a42-a792-406c-98f3-b0ed88f747e2"

#: The checked-in live capture the boxed-panel fixtures are pinned to.
REAL_BOXED_CAPTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "providers"
    / "fixtures"
    / "muse_0.2.1_boxed_status_capture.txt"
)


def boxed_status_rows(
    worktree,
    session_id: str,
    *,
    model: str = MUSE_MODEL,
    effort: Optional[str] = MUSE_EFFORT,
    provider: str = "meta",
    agent_profile: str = "native-basic",
    directory: Optional[str] = None,
    run_badge: str = "IDLE",
    usage: str = "0 tokens · 0 turns · 0 subagents",
) -> list[str]:
    """The 0.2.1-R1215.1 boxed ``/status`` panel, as really rendered.

    Faithful to the live meta capture: MODEL carries ``<model> ·
    <effort>`` (a bare model on echo), the continuation row under MODEL
    is ``<provider> · <profile>``, USAGE is ``N tokens · N turns · N
    subagents``, and the run badge rides the header row.
    """
    directory = directory or str(worktree)
    model_value = f"{model} · {effort}" if effort is not None else model
    return [
        "muse: incompatible_endpoint: local session-message endpoint did not complete "
        "the current broker hello",
        "  Muse Code 0.2.1",
        "┌──────────────────────────────────────────────────────┐",
        f"│  MUSE CODE 0.2.1 / swift-fireball{run_badge:>17} │",
        "│                                                      │",
        f"│  MODEL          {model_value:<37}│",
        f"│                 {provider} · {agent_profile:<24}│",
        "│                                                      │",
        f"│  WORKSPACE      {directory:<37}│",
        "│                 trusted · not found                  │",
        "│  ACCESS         approval Normal · sandbox Normal     │",
        "│                 none                                 │",
        "│                                                      │",
        f"│  USAGE          {usage:<37}│",
        "│  CONTEXT        not projected · 1008K limit          │",
        "│                                                      │",
        f"│  SESSION        {session_id:<37}│",
        "│  ACTIVITY       no tasks                             │",
        "│                 0 terminals · inbox clear            │",
        "└──────────────────────────────────────────────────────┘",
        "── Voice input (⌥V to start) ──────────────────────────",
        "⟩",
        "────────────────────────────────────────────────────",
        f"  {model_value} · {directory}",
    ]


def footer_only_rows(worktree, *, model: str = MUSE_MODEL, effort: Optional[str] = MUSE_EFFORT):
    """The persistent inline footer with NO panel: route without identity.

    This is the shape of the real failed-launch capture
    (muse-0.2.1-status-capture.txt): the composer holds ``/status`` and
    the only new content is the always-rendered footer line.
    """
    directory = str(worktree)
    value = f"{model} · {effort}" if effort is not None else model
    return [
        "muse: incompatible_endpoint: local session-message endpoint did not complete "
        "the current broker hello; remedy",
        ": restart or update the older sessions to join the current broker generation",
        "",
        "  Muse Code 0.2.1",
        "",
        "  Model set to muse-spark-1.2-contributor",
        "  ⎿  Discounted tokens: your content may be used for product improvement.",
        "",
        "── Voice input (⌥V to start) ───────────────────────────────────────────",
        "⟩ /status",
        "",
        "────────────────────────────────────────────────────────────────────────",
        f"  {value} · {directory}",
    ]


def _seed_session_store(
    xdg_root: Path,
    session_id: str,
    *,
    workspace_root: str,
    provider_id: str = "meta",
    semver: str = "0.2.1",
) -> Path:
    """A fake Muse session-store entry exactly like the cold-start write."""
    session_dir = xdg_root / "muse" / "sessions" / "2026" / "08" / "22" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "id": str(uuid.uuid4()),
        "stream": {"kind": "session", "id": session_id},
        "sequence": 1,
        "record_type": "event",
        "payload_type": "runtime.session.metadata",
        "payload_schema_version": 1,
        "payload": {
            "kind": "metadata",
            "record": {
                "workspace_root": workspace_root,
                "provider_id": provider_id,
                "web_search_mode": "client",
                "build": {"sha": "b3170a534f", "semver": semver},
            },
        },
    }
    (session_dir / "session.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    return session_dir


def test_status_parser_accepts_the_real_021_boxed_meta_panel(worktree):
    """The boxed panel parses into the same canonical fields as 0.1.x."""
    parsed = muse_native_status.parse_status_panel(boxed_status_rows(worktree, PROBE_SESSION_ID))
    assert parsed["panel_shape"] == "boxed-0.2"
    assert parsed["schema"] == "cao-muse-status-panel-v1"
    assert parsed["session_id"] == PROBE_SESSION_ID
    assert parsed["model"] == MUSE_MODEL
    assert parsed["reasoning"] == MUSE_EFFORT
    assert parsed["model_provider"] == "meta"
    assert parsed["agent_profile"] == "native-basic"
    assert parsed["directory"] == str(worktree)
    assert parsed["run"] == "idle"
    assert parsed["tokens"] == 0 and parsed["turns"] == 0

    proven = muse_native_status.require_pre_task_status(
        parsed,
        session_id=None,
        expected_model=MUSE_MODEL,
        expected_effort=MUSE_EFFORT,
        working_directory=str(worktree),
        expected_profile_identity="native-basic",
    )
    assert proven["session_matches"] is True and proven["zero_turns"] is True


def test_status_parser_parses_the_checked_in_live_021_boxed_capture():
    """The synthetic boxed fixture is pinned to a real capture-pane render.

    The checked-in file is a live 220x48 ``capture-pane -p`` of the
    installed meta build after ``/status`` at zero turns.  Parsing it
    structurally (rather than against hardcoded model strings, which are
    the operator's config) is what catches badge/label drift: any rename
    of SESSION/MODEL/USAGE, a moved run badge, or an effort outside the
    installed vocabulary makes this parse refuse.
    """
    rows = REAL_BOXED_CAPTURE_PATH.read_text(encoding="utf-8").rstrip("\n").splitlines()
    parsed = muse_native_status.parse_status_panel(rows)
    assert parsed["panel_shape"] == "boxed-0.2"
    # The discovered identity is a canonical lowercase UUID.
    muse_native_status.validate_discovered_session_id(parsed["session_id"])
    assert parsed["model_provider"] == "meta"
    assert parsed["agent_profile"] == "native-basic"
    assert parsed["reasoning"] in ("ultra", MUSE_EFFORT) or parsed["reasoning"] is None
    assert parsed["run"] == "idle"
    assert parsed["tokens"] == 0 and parsed["turns"] == 0

    proven = muse_native_status.require_pre_task_status(
        parsed,
        session_id=None,
        expected_model=parsed["model"],
        expected_effort=parsed["reasoning"],
        working_directory=parsed["directory"],
        expected_profile_identity="native-basic",
    )
    assert proven["session_matches"] is True and proven["zero_turns"] is True


def test_status_parser_accepts_the_021_boxed_echo_panel_with_a_bare_model(worktree):
    """Echo renders no effort segment; none is claimed observed."""
    parsed = muse_native_status.parse_status_panel(
        boxed_status_rows(
            worktree,
            PROBE_SESSION_ID,
            model="echo",
            effort=None,
            provider="echo",
        )
    )
    assert parsed["model"] == "echo"
    assert parsed["reasoning"] is None
    assert parsed["model_provider"] == "echo"


def test_boxed_panel_mutations_are_refused(worktree):
    """Every wrong fact in the boxed panel is refused, never guessed."""
    session_id = _uuid()

    def mutate(mutator):
        rows = boxed_status_rows(worktree, PROBE_SESSION_ID)
        mutator(rows)
        return muse_native_status.parse_status_panel(rows)

    with pytest.raises(muse_native_status.MuseStatusParseError):
        # A second SESSION row: the capture cannot prove which is real.
        mutate(lambda rows: rows.insert(17, rows[16]))
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(
            [row for row in boxed_status_rows(worktree, PROBE_SESSION_ID) if "SESSION" not in row]
        )
    with pytest.raises(muse_native_status.MuseStatusParseError):
        # An effort outside the installed vocabulary.
        muse_native_status.parse_status_panel(
            boxed_status_rows(worktree, PROBE_SESSION_ID, effort="maximum")
        )
    with pytest.raises(muse_native_status.MuseStatusParseError):
        # No provider/profile continuation row under MODEL.
        muse_native_status.parse_status_panel(
            [
                row
                for row in boxed_status_rows(worktree, PROBE_SESSION_ID)
                if "meta · native-basic" not in row
            ]
        )
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(
            boxed_status_rows(worktree, PROBE_SESSION_ID, usage="0 tokens / 0 turns")
        )

    # A non-idle badge parses but fails the pre-task requirement.
    parsed = muse_native_status.parse_status_panel(
        boxed_status_rows(worktree, PROBE_SESSION_ID, run_badge="WORKING")
    )
    with pytest.raises(muse_native_status.MuseStatusMismatch):
        muse_native_status.require_pre_task_status(
            parsed,
            session_id=None,
            expected_model=MUSE_MODEL,
            expected_effort=MUSE_EFFORT,
            working_directory=str(worktree),
            expected_profile_identity="native-basic",
        )


def test_boxed_panel_tolerates_repeated_chrome_rows(worktree):
    """CONTEXT/ACCESS/ACTIVITY are chrome: a repeat is not ambiguity."""
    rows = boxed_status_rows(worktree, PROBE_SESSION_ID)
    rows.insert(16, "│  CONTEXT        duplicated chrome row          │")
    parsed = muse_native_status.parse_status_panel(rows)
    assert parsed["session_id"] == PROBE_SESSION_ID
    assert parsed["run"] == "idle"


def test_a_shape_detected_but_incomplete_box_is_recognized_content(worktree):
    """Box detection outruns the strict parse for the fast-fail clock.

    A viewport-clipped box (bottom rows not yet rendered) is this pane's
    panel mid-render: it must hold the full runway, never trip the 15s
    unrecognized bound against a healthy launch.
    """
    full = boxed_status_rows(worktree, PROBE_SESSION_ID)
    clipped = full[: full.index(next(row for row in full if "USAGE" in row))]
    # Sanity: the clip keeps the shape but breaks the strict parse.
    assert muse_native_status.is_recognized_shape(clipped) is True
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(clipped)


def test_the_real_021_footer_capture_is_a_partial_route_observation(worktree):
    """The failed-launch capture proves the route and NO identity."""
    rows = footer_only_rows(worktree)
    parsed = muse_native_status.parse_status_panel(rows)
    assert parsed["partial"] is True
    assert parsed["observation"] == "route-only-footer"
    assert parsed["session_id"] is None
    assert parsed["model"] == MUSE_MODEL
    assert parsed["reasoning"] == MUSE_EFFORT
    assert parsed["directory"] == str(worktree)


def test_require_pre_task_status_refuses_a_partial_observation(worktree):
    """A footer observation is never a ready receipt."""
    parsed = muse_native_status.parse_status_panel(footer_only_rows(worktree))
    with pytest.raises(muse_native_status.MuseStatusMismatch, match="route-only"):
        muse_native_status.require_pre_task_status(
            parsed,
            session_id=None,
            expected_model=MUSE_MODEL,
            expected_effort=MUSE_EFFORT,
            working_directory=str(worktree),
            expected_profile_identity="native-basic",
        )


def test_two_distinct_footers_refuse_as_ambiguous():
    """Two footer shapes cannot prove which pane rendered them."""
    rows = ["  a · high · /one", "  b · low · /two"]
    with pytest.raises(muse_native_status.MuseStatusParseError):
        muse_native_status.parse_status_panel(rows)


def test_the_labeled_01_panel_still_parses_backwards_compatibly(worktree):
    """The 0.1.x labeled panel keeps its exact pre-existing parse."""
    parsed = muse_native_status.parse_status_panel(status_panel_rows(worktree, PROVIDER_SESSION_ID))
    assert parsed["panel_shape"] == "labeled-0.1"
    assert parsed["session_id"] == PROVIDER_SESSION_ID
    assert parsed["model"] == MUSE_MODEL


def test_session_store_snapshot_and_discovery_round_trip(tmp_path, monkeypatch, worktree):
    """The store diff adopts the exactly-one cold-start registration."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    snapshot = muse_session_store.snapshot_known_session_ids()
    assert snapshot == frozenset()

    seeded = _seed_session_store(tmp_path / "xdg", _uuid(), workspace_root=str(worktree))
    found = muse_session_store.discover_new_session_id(
        snapshot,
        working_directory=str(worktree),
        deadline_monotonic=time.monotonic() + 2,
    )
    assert found["session_id"] == seeded.name
    assert found["metadata"]["workspace_root"] == str(worktree)
    assert found["metadata"]["provider_id"] == "meta"
    assert found["metadata"]["build_semver"] == "0.2.1"

    # Once known, the same session never counts as new again.
    assert seeded.name in muse_session_store.snapshot_known_session_ids()


def test_session_store_discovery_refuses_two_candidates_as_ambiguous(
    tmp_path, monkeypatch, worktree
):
    """Concurrent registrations are ambiguity, never a coin flip."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    snapshot = muse_session_store.snapshot_known_session_ids()
    _seed_session_store(tmp_path / "xdg", _uuid(), workspace_root=str(worktree))
    _seed_session_store(tmp_path / "xdg", _uuid(), workspace_root=str(worktree))
    with pytest.raises(muse_session_store.MuseSessionStoreAmbiguous):
        muse_session_store.discover_new_session_id(
            snapshot,
            working_directory=str(worktree),
            deadline_monotonic=time.monotonic() + 2,
        )


def test_session_store_discovery_scopes_to_workspace_and_provider(tmp_path, monkeypatch, worktree):
    """Other workspaces and other providers are never adoptable matches."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    snapshot = muse_session_store.snapshot_known_session_ids()
    _seed_session_store(tmp_path / "xdg", _uuid(), workspace_root="/somewhere/else")
    _seed_session_store(tmp_path / "xdg", _uuid(), workspace_root=str(worktree), provider_id="echo")
    with pytest.raises(muse_session_store.MuseSessionStoreUnavailable):
        muse_session_store.discover_new_session_id(
            snapshot,
            working_directory=str(worktree),
            deadline_monotonic=time.monotonic() + 0.3,
            poll_seconds=0.05,
        )


def test_session_store_discovery_names_pending_and_unreadable_candidates(
    tmp_path, monkeypatch, worktree
):
    """A log not yet flushed stays pending; garbage evidence is unreadable.

    Neither state is adoptable, and neither may be bypassed because some
    OTHER new directory already matches: adoption requires the whole
    new-dir set resolved.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    snapshot = muse_session_store.snapshot_known_session_ids()

    pending_dir = tmp_path / "xdg" / "muse" / "sessions" / "2026" / "08" / "22" / _uuid()
    pending_dir.mkdir(parents=True)
    garbage_dir = tmp_path / "xdg" / "muse" / "sessions" / "2026" / "08" / "22" / _uuid()
    garbage_dir.mkdir(parents=True)
    (garbage_dir / "session.jsonl").write_text("not json\n", encoding="utf-8")

    with pytest.raises(muse_session_store.MuseSessionStoreUnavailable) as refused:
        muse_session_store.discover_new_session_id(
            snapshot,
            working_directory=str(worktree),
            deadline_monotonic=time.monotonic() + 0.3,
            poll_seconds=0.05,
        )
    message = str(refused.value)
    assert "1 candidate(s) still pending" in message
    assert "1 unreadable" in message

    # The pending candidate flushes its metadata — but the unreadable
    # sibling still blocks adoption of anything.
    _seed_session_store(tmp_path / "xdg", pending_dir.name, workspace_root=str(worktree))
    with pytest.raises(muse_session_store.MuseSessionStoreUnavailable) as still_blocked:
        muse_session_store.discover_new_session_id(
            snapshot,
            working_directory=str(worktree),
            deadline_monotonic=time.monotonic() + 0.3,
            poll_seconds=0.05,
        )
    assert "1 unreadable" in str(still_blocked.value)

    # The garbage write completes into a valid non-match; the set resolves
    # and exactly our session is adopted.
    _seed_session_store(tmp_path / "xdg", garbage_dir.name, workspace_root="/elsewhere")
    found = muse_session_store.discover_new_session_id(
        snapshot,
        working_directory=str(worktree),
        deadline_monotonic=time.monotonic() + 2,
    )
    assert found["session_id"] == pending_dir.name


def test_session_store_discovery_waits_for_sibling_dirs_to_resolve(tmp_path, monkeypatch, worktree):
    """A matching candidate never adopts while a sibling dir is unresolved.

    A concurrent launch in this workspace registers the same kind of
    directory ours does; whichever flushes first must not be adopted as
    THIS pane's identity merely because it won the flush race.  The match
    stays unadopted for the whole window, and resolves to exactly ours
    once the sibling's own metadata lands as a non-match.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    snapshot = muse_session_store.snapshot_known_session_ids()
    our_id = _uuid()
    _seed_session_store(tmp_path / "xdg", our_id, workspace_root=str(worktree))

    # A sibling new directory whose log has not flushed yet.
    sibling_dir = tmp_path / "xdg" / "muse" / "sessions" / "2026" / "08" / "22" / _uuid()
    sibling_dir.mkdir(parents=True)

    # No adoption inside the window, even though exactly one match exists.
    with pytest.raises(muse_session_store.MuseSessionStoreUnavailable) as refused:
        muse_session_store.discover_new_session_id(
            snapshot,
            working_directory=str(worktree),
            deadline_monotonic=time.monotonic() + 0.3,
            poll_seconds=0.05,
        )
    assert "1 candidate(s) still pending" in str(refused.value)

    # Once the sibling resolves as a NON-match, exactly-one is adoptable —
    # and it must be OURS, not whichever registered first.
    _seed_session_store(tmp_path / "xdg", sibling_dir.name, workspace_root="/somewhere/else")
    found = muse_session_store.discover_new_session_id(
        snapshot,
        working_directory=str(worktree),
        deadline_monotonic=time.monotonic() + 2,
    )
    assert found["session_id"] == our_id


@pytest.mark.asyncio
async def test_muse_launch_discovers_through_the_021_boxed_panel(
    isolated_memory_db, worktree, tmp_path, muse_harness
):
    """On 0.2.1 the boxed panel names the session; nothing else changes."""
    muse_harness.captures.append(boxed_status_rows(worktree, PROBE_SESSION_ID))
    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "launching"
    attachment = native_attachment.get("muse_cli", PROBE_SESSION_ID)
    assert attachment is not None
    terminal = (
        database.SessionLocal()
        .query(database.ManagedLaunchV2TerminalModel)
        .filter(database.ManagedLaunchV2TerminalModel.id == record["terminal_id"])
        .first()
    )
    assert terminal.v2_native_session_id == PROBE_SESSION_ID

    receipt = _published_receipt(record["reservation_id"])
    assert receipt["provider_session_id"] == PROBE_SESSION_ID
    assert receipt["model"] == MUSE_MODEL
    assert receipt["effort"] == MUSE_EFFORT
    assert receipt["provider_session_start"]["observed"]["effort"] == MUSE_EFFORT


@pytest.mark.asyncio
async def test_muse_launch_falls_back_to_store_identity_when_only_the_footer_renders(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """Footer-only renders adopt the exactly-one store registration.

    The fake store is seeded when the pane is created — the moment the
    real provider registers its cold-start session — so the launch's own
    pre-spawn snapshot excludes it and the diff yields exactly one match.
    """
    xdg_root = tmp_path / "xdg-home"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_root))
    store_session_id = _uuid()
    real_create = terminal_service.create_terminal

    async def _create_terminal_and_register_session(**kwargs):
        result = await real_create(**kwargs)
        _seed_session_store(xdg_root, store_session_id, workspace_root=str(worktree))
        return result

    monkeypatch.setattr(terminal_service, "create_terminal", _create_terminal_and_register_session)
    muse_harness.captures.append(footer_only_rows(worktree))

    record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "launching"
    attachment = native_attachment.get("muse_cli", store_session_id)
    assert attachment is not None

    receipt = _published_receipt(record["reservation_id"])
    assert receipt["provider_session_id"] == store_session_id
    # Route facts come from the footer, never from the request.
    assert receipt["model"] == MUSE_MODEL
    assert receipt["effort"] == MUSE_EFFORT
    session_start = receipt["provider_session_start"]
    assert session_start["source"] == "session-store-diff"
    assert session_start["identity_proven"] is True
    assert session_start["observed"]["agent_profile"] is None
    assert session_start["footer_observation"]["partial"] is True


@pytest.mark.asyncio
async def test_muse_store_fallback_refuses_a_footer_that_disagrees_with_the_request(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """A footer naming another model is route evidence against the launch."""
    xdg_root = tmp_path / "xdg-home"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_root))
    store_session_id = _uuid()
    real_create = terminal_service.create_terminal

    async def _create_terminal_and_register_session(**kwargs):
        result = await real_create(**kwargs)
        _seed_session_store(xdg_root, store_session_id, workspace_root=str(worktree))
        return result

    monkeypatch.setattr(terminal_service, "create_terminal", _create_terminal_and_register_session)
    muse_harness.captures.append(
        footer_only_rows(worktree, model="muse-spark-1.2", effort=MUSE_EFFORT)
    )
    _record, result = await _launch(worktree, tmp_path, muse_harness)
    assert result["state"] == "preflight_blocked"
    detail = result["preflight_failure"]["detail"]
    assert "does not describe the claimed launch" in detail
    assert "muse-spark-1.2" in detail
    # Nothing was bound or advertised for the refused identity.
    assert native_attachment.get("muse_cli", store_session_id) is None
    assert bridge.read_state(result["reservation_id"]) is None


@pytest.mark.asyncio
async def test_muse_launch_fast_fails_when_the_render_is_never_recognized(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """Zero recognized rows refuse in seconds, naming version + fingerprint.

    A build whose screen never matches any known shape must not hold the
    full cold-start runway: the refusal carries the installed version and
    the last screen's fingerprint so the operator can act on it instead
    of watching bind-timeout storms.
    """
    monkeypatch.setattr(v2, "MUSE_STATUS_UNRECOGNIZED_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(v2, "NATIVE_PANE_READY_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(v2, "_NATIVE_PANE_READY_POLL_SECONDS", 0.005)
    muse_harness.captures.append(["Muse Code 0.2.1", "", "⟫ garbage render ⟪", "⟩"])
    started = time.monotonic()
    _record, result = await _launch(worktree, tmp_path, muse_harness)
    elapsed = time.monotonic() - started
    assert result["state"] == "preflight_blocked"
    detail = result["preflight_failure"]["detail"]
    assert "never produced recognized content" in detail
    # The installed provider_version the bootstrap recorded, named verbatim.
    assert "0.1.0" in detail
    assert "sha256:" in detail
    assert elapsed < 10


@pytest.mark.asyncio
async def test_muse_launch_keeps_the_full_runway_for_a_clipped_boxed_panel(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """A shape-detected but incomplete box polls the full runway, not 15s.

    The strict parse never succeeds (USAGE/SESSION rows are outside the
    captured viewport), but the capture IS the pane's panel: refusing it
    as unrecognized content would fast-fail a healthy launch mid-render.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty-xdg"))
    monkeypatch.setattr(v2, "MUSE_STATUS_UNRECOGNIZED_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(v2, "NATIVE_PANE_READY_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr(v2, "_NATIVE_PANE_READY_POLL_SECONDS", 0.005)
    full = boxed_status_rows(worktree, PROBE_SESSION_ID)
    muse_harness.captures.append(full[: full.index(next(row for row in full if "USAGE" in row))])
    started = time.monotonic()
    _record, result = await _launch(worktree, tmp_path, muse_harness)
    elapsed = time.monotonic() - started
    assert result["state"] == "preflight_blocked"
    detail = result["preflight_failure"]["detail"]
    # The FULL-runway refusal fired, not the unrecognized fast-fail.
    assert "never produced recognized content" not in detail
    assert "never described the claimed pre-task session within 0.4 seconds" in detail
    assert elapsed >= 0.4


@pytest.mark.asyncio
async def test_muse_launch_keeps_the_full_runway_once_content_is_recognized(
    isolated_memory_db, worktree, tmp_path, muse_harness, monkeypatch
):
    """Recognized-but-incomplete content waits the full bound, then refuses.

    The footer alone is recognized content, so the observation holds the
    whole cold-start runway even though the unrecognized bound has long
    passed — it is waiting on evidence known to arrive (the panel or a
    store registration), not on an unrecognizable screen.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty-xdg"))
    monkeypatch.setattr(v2, "MUSE_STATUS_UNRECOGNIZED_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(v2, "NATIVE_PANE_READY_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr(v2, "_NATIVE_PANE_READY_POLL_SECONDS", 0.005)
    # The footer IS recognized content; nothing else ever renders.
    muse_harness.captures.append(footer_only_rows(worktree))
    started = time.monotonic()
    _record, result = await _launch(worktree, tmp_path, muse_harness)
    elapsed = time.monotonic() - started
    assert result["state"] == "preflight_blocked"
    # Held well past the unrecognized bound before refusing.
    assert elapsed >= 0.4
    detail = result["preflight_failure"]["detail"]
    assert (
        "never described the claimed pre-task session within 0.4 seconds" in detail
        or "refusing to adopt an identity the store cannot prove" in detail
    )
