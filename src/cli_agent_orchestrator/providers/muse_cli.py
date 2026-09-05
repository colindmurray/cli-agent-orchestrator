"""Muse Code CLI provider — Meta's native terminal agent on Muse Spark 1.3.

Muse Code (https://developer.meta.com/ai/lp/start-building/) is Meta's
terminal coding agent. CAO launches it as a native-CLI worker (the same shape as
``kimi_cli``) so the Muse Spark 1.3 model runs through its own harness rather
than a Claude Code gateway: the Meta Model API rejects several Anthropic-API
fields Claude Code 2.x always sends (``context_management``, ``output_config``),
so a claude_code-on-Muse route cannot work directly.

TUI characteristics (Muse Code 0.1.x):
  - Command: ``muse``
  - Idle prompt: a ``⟩`` input line at the bottom when ready for input
  - Processing: no idle ``⟩`` prompt (turn-in-flight spinner)
  - Response format: assistant reply lines prefixed with ``◆``
  - Auto-approve: ``--yolo`` disables approval and sandbox for a workspace run
  - Model selection: ``--model <id>``; ``--reasoning-effort`` maps CAO effort
  - Exit: ``/exit``

Status detection: the ``⟩`` prompt stays rendered through a whole turn, so the
in-flight signal is the spinner text ("esc to interrupt"); otherwise the bare
``⟩`` prompt means ready (IDLE before a task, COMPLETED after one).

Profile material: the managed-v2 native launch composes the CAO profile
system prompt into the session as base instructions through the
``TBH_EVAL_APPEND_SYSTEM_PROMPT_FILE`` env surface (verified via runtime
probe, see ``muse_native_launch.probe_profile_carrier``).  The
``--agents <JSON>`` overlay does NOT compose into the main session
agent — it registers session agent definitions for the workflow/subagent
``agentType`` path only — so the managed launch never relies on it for the
CAO role/profile.  The reviewer's declared read-only tools are enforced by
prompt + policy only (no SECURITY_PROMPT injection into the model), so keep
review tasks read-only by instruction.

Identity: a fresh managed launch starts a no-prompt TUI (``muse
--trust-workspace ... --model <id>``) and *discovers* the provider-generated
session id from the provider's own ``/status`` panel at zero turns;
``muse resume <id>`` is the restoration form for a later reincarnation and
never a caller-chosen creation.
"""

from __future__ import annotations

import logging
import re
import shlex
from typing import TYPE_CHECKING, List, Optional

from cli_agent_orchestrator.backends.registry import get_backend
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.base import (
    BaseProvider,
    PreparedSealedLaunch,
    SealedLaunchMaterial,
    SealedPreparationUnsupported,
    SealedProfileSupport,
    container_maps_set,
    custom_permission_mode_set,
    custom_timeout_set,
    dropped_q_fields,
    foreign_native_fields,
    policy_restricted,
)
from cli_agent_orchestrator.utils.agent_profiles import load_agent_profile
from cli_agent_orchestrator.utils.terminal import wait_for_shell, wait_until_status
from cli_agent_orchestrator.utils.text import strip_terminal_escapes

if TYPE_CHECKING:
    from cli_agent_orchestrator.models.agent_profile import AgentProfile

logger = logging.getLogger(__name__)

# Muse Code 0.1.x TUI markers (empirically verified: the ⟩ prompt stays
# rendered through a whole turn, the spinner alternates ◇/◆ and reads
# "Thinking (Ns · esc to interrupt)", and reply continuations are 2-space
# indented bare lines under a single ◆ lead).
IDLE_PROMPT_PATTERN = r"⟩\s*$"  # input prompt line (rendered all turn)
IDLE_PROMPT_PATTERN_LOG = r"⟩"
SPINNER_PATTERN = r"esc to interrupt"  # in-flight marker (◇/◆ Thinking…)
ERROR_PATTERN = r"(crash report written|Traceback \(most recent call last\))"
ANSI_CODE_PATTERN = r"\x1b\[[0-9;]*[A-Za-z]"
# The lead of a real reply bullet: "◆ text", but never a spinner line.
REPLY_LEAD_PATTERN = r"^◆[ \t]+(.+)$"


class MuseCliProvider(BaseProvider):
    """Native Muse Code worker provider (Muse Spark 1.3 standard or Contributor).

    The model is resolved in this order: ``expected_model`` (CAO's managed-launch
    override, set by ``conduct spawn`` from the route model, so a single profile
    can target either ``muse-spark-1.3`` or ``muse-spark-1.3-contributor`` at
    launch), then the profile's ``model`` field, then Muse Code's default.
    """

    BINARY_NAME = "muse"

    def __init__(
        self,
        terminal_id: str,
        session_name: str,
        window_name: str,
        agent_profile: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        skill_prompt: Optional[str] = None,
        expected_model: Optional[str] = None,
        expected_effort: Optional[str] = None,
        launch_profile: Optional["AgentProfile"] = None,
    ) -> None:
        super().__init__(terminal_id, session_name, window_name, allowed_tools, skill_prompt)
        self._agent_profile = agent_profile
        self._expected_model = expected_model
        self._expected_effort = expected_effort
        # The launch's already-loaded profile (cond-0817): when set, the
        # launch argv consumes this exact object and never reloads the
        # profile by name. None keeps the legacy load.
        self._launch_profile = launch_profile
        self._initialized = False
        self._has_received_input = False
        # Shell process running in the pane before muse launches; used to detect
        # a crashed/exited muse pane that has reverted to a bare shell.
        self.shell_baseline: Optional[str] = None

    @property
    def paste_enter_count(self) -> int:
        return 1

    def mark_input_received(self) -> None:
        super().mark_input_received()
        self._has_received_input = True

    def _resolve_model(self) -> Optional[str]:
        if self._expected_model:
            return self._expected_model
        if self._launch_profile is not None:
            return self._launch_profile.model
        if self._agent_profile:
            try:
                return load_agent_profile(self._agent_profile).model
            except Exception:  # pragma: no cover - profile resolution is best-effort
                return None
        return None

    def _build_command(self) -> str:
        """Build the Muse Code launch command (escaped for tmux send_keys)."""
        parts = [self.BINARY_NAME, "--yolo"]
        model = self._resolve_model()
        if model:
            parts.extend(["--model", model])
        if self._expected_effort:
            parts.extend(["--reasoning-effort", self._expected_effort])
        return shlex.join(parts)

    @classmethod
    def supports_sealed_launch(
        cls, material: Optional[SealedLaunchMaterial]
    ) -> SealedProfileSupport:
        """Sealed support covers model plus effort only, for content-free
        wildcard profiles.

        The v1 argv pins the frozen model (and expected effort) with
        ``--yolo`` — it carries no prompt, no skills, no MCP material, and
        no policy. Support therefore requires every dropped field to be
        empty or default: a content-free profile under a wildcard policy.
        Anything else is refused rather than recorded as launched.
        """
        if material is None or material.profile is None:
            return SealedProfileSupport(False, "no frozen profile was supplied")
        profile = material.profile
        dropped = []
        if material.system_prompt:
            dropped.append("system_prompt")
        if material.skill_text:
            dropped.append("skills")
        if policy_restricted(material.allowed_tools):
            dropped.append("allowedTools")
        if getattr(profile, "mcpServers", None):
            dropped.append("mcpServers")
        dropped.extend(dropped_q_fields(profile))
        dropped.extend(foreign_native_fields(profile, own=""))
        for extra in ("codexConfig",):
            if getattr(profile, extra, None):
                dropped.append(extra)
        if custom_permission_mode_set(profile):
            dropped.append("permissionMode")
        if custom_timeout_set(profile):
            dropped.append("provider_init_timeout")
        if container_maps_set(profile):
            dropped.append("container")
        if dropped:
            return SealedProfileSupport(
                False,
                "Muse launch argv carries only model and effort; "
                f"{', '.join(sorted(dropped))} would be silently dropped from "
                "the launch",
            )
        return SealedProfileSupport(
            True,
            "Muse launch argv (model, effort) uses only the frozen material; "
            "the profile is content-free under a wildcard policy",
        )

    @classmethod
    def prepare_sealed_launch(
        cls, material: Optional[SealedLaunchMaterial]
    ) -> PreparedSealedLaunch:
        """Carry the content-free wildcard shape, pre-effect.

        The argv pins only the frozen model and effort — already-typed
        material fields needing no serialization — so there is nothing
        to validate and no payload to carry: preparation is the choke
        point, not a second gate (the capability decision stays in
        :meth:`supports_sealed_launch`). A missing frozen profile is
        still refused outright.
        """
        if material is None or material.profile is None:
            raise SealedPreparationUnsupported("no frozen profile was supplied")
        return PreparedSealedLaunch(provider="muse_cli")

    async def initialize(self) -> bool:
        """Launch ``muse --yolo`` inside the tmux window (cwd = the worktree)."""
        if not await wait_for_shell(self.terminal_id, timeout=10.0):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        # Capture the shell process before launching muse — used later to detect
        # when muse has exited and the pane is back to a bare shell, so a queued
        # message can never be typed into zsh as arbitrary commands.
        self.shell_baseline = get_backend().get_pane_current_command(
            self.session_name, self.window_name
        )

        # Arm the StatusMonitor stickiness gate before the launch keystrokes.
        from cli_agent_orchestrator.services.status_monitor import status_monitor

        status_monitor.notify_input_sent(self.terminal_id)

        command = self._build_command()
        get_backend().send_keys(self.session_name, self.window_name, command)

        if not await wait_until_status(
            self.terminal_id, {TerminalStatus.IDLE, TerminalStatus.COMPLETED}, timeout=45.0
        ):
            raise TimeoutError("Muse Code initialization timed out after 45 seconds")

        self._initialized = True
        return True

    def get_status_from_screen(self, screen_lines: List[str]) -> TerminalStatus:
        """Detect Muse Code's state from the rendered screen rows.

        Mirrors the Kimi observer contract: the native pane turn-state
        observers delegate to the provider's own detector so there is
        exactly one description of what a Muse screen means.  A fresh
        instance is used (the v2 pane has no shell baseline), so the
        shell-revert check is skipped and the ``⟩`` composer line decides
        idle.
        """
        return self.get_status("\n".join(screen_lines))

    def get_status(self, output: str) -> TerminalStatus:
        """Detect Muse Code's state from the tmux capture buffer.

        Strategy (mirrors kimi_cli): bottom-anchored. The bare ``⟩`` input line
        means ready; once any ``◆`` response has appeared the turn is COMPLETED
        (latched via ``_has_received_input`` for long responses that scroll the
        marker out of the rolling buffer); no idle prompt means a turn is in
        flight; crash/error markers are ERROR.
        """
        native = self._resolve_native_status(output)
        if native is not None:
            return native

        output = self._resolve_buffer(output)
        if not output:
            return TerminalStatus.UNKNOWN

        clean = strip_terminal_escapes(output)

        # A pane that reverted to the shell means muse crashed/exited. Returning
        # ERROR prevents the inbox service from typing a queued message into the
        # shell (which would execute it as arbitrary commands).
        if self._initialized and self.shell_baseline:
            current_cmd = get_backend().get_pane_current_command(
                self.session_name, self.window_name
            )
            if current_cmd == self.shell_baseline:
                return TerminalStatus.ERROR

        if re.search(ERROR_PATTERN, clean):
            return TerminalStatus.ERROR

        # The ⟩ prompt is rendered through the whole turn, so it cannot
        # distinguish idle from in-flight. The spinner text ("esc to
        # interrupt") is the reliable in-flight signal; check it first.
        if re.search(SPINNER_PATTERN, clean):
            return TerminalStatus.PROCESSING

        if re.search(IDLE_PROMPT_PATTERN, clean, re.MULTILINE):
            if self._has_received_input:
                return TerminalStatus.COMPLETED
            return TerminalStatus.IDLE
        return TerminalStatus.PROCESSING

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Return Muse Code's last reply, skipping the spinner and spanning
        the 2-space-indented continuation lines under the ◆ lead."""
        clean = strip_terminal_escapes(script_output)
        lines = clean.splitlines()
        last_idx = -1
        for i, line in enumerate(lines):
            if re.match(REPLY_LEAD_PATTERN, line) and "esc to interrupt" not in line:
                last_idx = i
        if last_idx == -1:
            raise ValueError("No Muse Code response found in script output")
        parts = [re.sub(r"^◆[ \t]+", "", lines[last_idx]).strip()]
        for line in lines[last_idx + 1 :]:
            if re.match(r"^[ \t]{2}\S", line) and "Voice input" not in line:
                parts.append(line.strip())
            else:
                break
        return "\n".join(parts).strip()

    def get_idle_pattern_for_log(self) -> str:
        return IDLE_PROMPT_PATTERN_LOG

    def exit_cli(self) -> str:
        return "/exit"

    def cleanup(self) -> None:
        return None
