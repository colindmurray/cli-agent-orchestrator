"""The cond-0713 /status resend contract for Muse managed-launch discovery.

Muse 0.2.1 silently swallows a ``/status`` submitted during cold start:
the composer consumes the input near input-ready and no panel ever
renders, while an identical resubmit once idle renders the full boxed
panel within seconds (live-proven on the installed build).  Discovery
therefore resends the literal command while no panel-shaped content has
been seen, bounded by the cold-start runway, and stops the moment a
panel renders because a second landed panel would be duplicate evidence
the parser refuses.

These tests drive :func:`_observe_muse_status_panel` directly through a
recording pane-input fake and a scripted capture, so each scenario pins
exactly what was typed against exactly what was observed.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from cli_agent_orchestrator.services import managed_launch_v2 as v2
from cli_agent_orchestrator.services import native_pane_input

#: A provider-generated id from a real 0.2.1-R1215.1 meta TUI.
PROBE_SESSION_ID = "e10c3a42-a792-406c-98f3-b0ed88f747e2"
MUSE_MODEL = "muse-spark-1.2-contributor"
MUSE_EFFORT = "high"


class RecordingPaneInput:
    """Stands in for TmuxPaneInput, recording every literal and Enter."""

    instances: list["RecordingPaneInput"] = []

    def __init__(self, pane_id: str, **_kwargs: Any) -> None:
        assert pane_id.startswith("%"), pane_id
        self.literals: list[str] = []
        self.enters = 0
        RecordingPaneInput.instances.append(self)

    def send_literal(self, text: str) -> None:
        self.literals.append(text)

    def send_enter(self) -> None:
        self.enters += 1


class ScriptedScreen:
    """Serve one screen until a condition holds, then another forever.

    The provider truth being modeled: the panel renders as a consequence
    of a *landed* submit, so the gate reads what was actually typed into
    the pane rather than the clock.
    """

    def __init__(
        self,
        silent_rows: list[str],
        panel_rows: list[str],
        *,
        render_when: Any,
    ) -> None:
        self._silent = list(silent_rows)
        self._panel = list(panel_rows)
        self._render_when = render_when
        self.calls = 0

    def __call__(self, _pane_id: str) -> list[str]:
        self.calls += 1
        if self._render_when():
            return list(self._panel)
        return list(self._silent)


def boxed_status_rows(directory: str) -> list[str]:
    """The 0.2.1-R1215.1 boxed ``/status`` panel, faithful to the live render."""
    model_value = f"{MUSE_MODEL} · {MUSE_EFFORT}"
    return [
        "  Muse Code 0.2.1",
        "┌──────────────────────────────────────────────────────┐",
        f"│  MUSE CODE 0.2.1 / swift-fireball{'IDLE':>17} │",
        "│                                                      │",
        f"│  MODEL          {model_value:<37}│",
        f"│                 {'meta · native-basic':<37}│",
        "│                                                      │",
        f"│  WORKSPACE      {directory:<37}│",
        "│                 trusted · not found                  │",
        "│  ACCESS         approval Normal · sandbox Normal     │",
        "│                 none                                 │",
        "│                                                      │",
        "│  USAGE          0 tokens · 0 turns · 0 subagents     │",
        "│  CONTEXT        not projected · 1008K limit          │",
        "│                                                      │",
        f"│  SESSION        {PROBE_SESSION_ID:<37}│",
        "│  ACTIVITY       no tasks                             │",
        "│                 0 terminals · inbox clear            │",
        "└──────────────────────────────────────────────────────┘",
        "⟩",
        f"  {model_value} · {directory}",
    ]


GARBAGE_ROWS = ["Muse Code 0.2.1", "", "⟫ garbage render ⟪", "⟩"]


def clipped_boxed_rows(directory: str) -> list[str]:
    """A boxed panel the viewport clipped mid-render: shape without parse.

    Everything from the USAGE row down is outside the captured viewport,
    so SESSION/USAGE are missing, but the top border and the surviving
    labels still make this the pane's own panel by shape detection.
    """
    full = boxed_status_rows(directory)
    return full[: full.index(next(row for row in full if "USAGE" in row))]


def footer_only_rows(directory: str) -> list[str]:
    """The persistent inline footer with no panel: route without identity.

    Modeled on the real failed-launch capture: the composer holds
    ``/status`` and the only recognizable content is the always-rendered
    footer line above the composer.
    """
    return [
        "  Muse Code 0.2.1",
        "",
        "── Voice input (⌥V to start) ───────────────────────────────────────────",
        "⟩ /status",
        f"  {MUSE_MODEL} · {MUSE_EFFORT} · {directory}",
    ]


def _observe(screen: ScriptedScreen, worktree: str):
    """Drive one observation with fast test timings and recorded input."""
    return v2._observe_muse_status_panel(
        {},
        "%7",
        capture=screen,
        session_id=None,
        expected_model=MUSE_MODEL,
        expected_effort=MUSE_EFFORT,
        working_directory=worktree,
        expected_profile_identity="native-basic",
        muse_version="0.2.1-R1215.1",
    )


@pytest.fixture
def recording_input(monkeypatch):
    RecordingPaneInput.instances = []
    monkeypatch.setattr(native_pane_input, "TmuxPaneInput", RecordingPaneInput)
    return RecordingPaneInput.instances


@pytest.fixture
def fast_timings(monkeypatch):
    monkeypatch.setattr(v2, "NATIVE_PANE_READY_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(v2, "_NATIVE_PANE_READY_POLL_SECONDS", 0.005)
    monkeypatch.setattr(v2, "MUSE_STATUS_RETYPE_INTERVAL_SECONDS", 0.05)


def test_second_submit_renders_the_panel_after_the_first_was_swallowed(
    recording_input, fast_timings, tmp_path
):
    """A swallowed first submit is retried, and the retry's panel admits.

    The screen stays silent until a second Enter has actually been typed,
    modeling the live-proven 0.2.1 swallow: the first submit renders
    nothing, an identical resubmit renders the full boxed panel.
    """
    screen = ScriptedScreen(
        GARBAGE_ROWS,
        boxed_status_rows(str(tmp_path)),
        render_when=lambda: recording_input[-1].enters >= 2,
    )
    observation = _observe(screen, str(tmp_path))
    assert observation["observed"]["session_id"] == PROBE_SESSION_ID
    typed = recording_input[-1]
    assert typed.literals == ["/status", "/status"]
    assert typed.enters == 2


def test_every_submit_swallowed_refuses_naming_attempts_and_captures(
    recording_input, monkeypatch, tmp_path
):
    """Submits that are all swallowed exhaust the runway into one refusal.

    The refusal states what was observed — how many times ``/status`` was
    submitted, how many captures were read, the installed version, and the
    fingerprint of the last screen — and the counted submits equal the
    Enters actually typed.
    """
    monkeypatch.setattr(v2, "NATIVE_PANE_READY_TIMEOUT_SECONDS", 0.25)
    monkeypatch.setattr(v2, "_NATIVE_PANE_READY_POLL_SECONDS", 0.005)
    monkeypatch.setattr(v2, "MUSE_STATUS_RETYPE_INTERVAL_SECONDS", 0.05)
    started = time.monotonic()
    with pytest.raises(v2.ManagedLaunchConflict) as excinfo:
        _observe(
            ScriptedScreen(
                GARBAGE_ROWS, boxed_status_rows(str(tmp_path)), render_when=lambda: False
            ),
            str(tmp_path),
        )
    assert time.monotonic() - started >= 0.25
    detail = str(excinfo.value)
    assert "never described the claimed pre-task session within 0.25 seconds" in detail
    attempts = int(detail.split("status_submit_attempts=")[1].split(",")[0])
    assert attempts > 1, detail
    assert "installed 0.2.1-R1215.1" in detail
    assert "captures=" in detail
    assert "sha256:" in detail
    typed = recording_input[-1]
    assert attempts == typed.enters


def test_immediate_recognition_never_resends(recording_input, fast_timings, tmp_path):
    """A panel that renders on the first capture is never re-driven."""
    screen = ScriptedScreen(
        GARBAGE_ROWS,
        boxed_status_rows(str(tmp_path)),
        render_when=lambda: True,
    )
    observation = _observe(screen, str(tmp_path))
    assert observation["observed"]["session_id"] == PROBE_SESSION_ID
    typed = recording_input[-1]
    assert typed.literals == ["/status"]
    assert typed.enters == 1


def test_a_clipped_panel_suppresses_every_resend(recording_input, monkeypatch, tmp_path):
    """A shape-detected but incomplete box stops the resend gate.

    The strict parse never succeeds on a viewport-clipped box, but the
    box IS the pane's panel already rendering: resending would stack a
    second panel onto the first and make the capture ambiguous.  With
    the retype interval long spent, exactly one Enter was ever typed.
    """
    monkeypatch.setattr(v2, "NATIVE_PANE_READY_TIMEOUT_SECONDS", 0.25)
    monkeypatch.setattr(v2, "_NATIVE_PANE_READY_POLL_SECONDS", 0.005)
    monkeypatch.setattr(v2, "MUSE_STATUS_RETYPE_INTERVAL_SECONDS", 0.05)
    screen = ScriptedScreen(
        GARBAGE_ROWS,
        clipped_boxed_rows(str(tmp_path)),
        render_when=lambda: True,
    )
    with pytest.raises(v2.ManagedLaunchConflict) as excinfo:
        _observe(screen, str(tmp_path))
    assert "status_submit_attempts=1," in str(excinfo.value)
    typed = recording_input[-1]
    assert typed.literals == ["/status"]
    assert typed.enters == 1


def test_a_footer_only_capture_never_suppresses_the_resend(recording_input, monkeypatch, tmp_path):
    """Resends continue while only the inline footer renders.

    The footer renders whether or not a submit landed, so it is never
    evidence the command was taken: across footer-only captures the gate
    keeps firing until the runway is spent, one literal ``/status`` per
    Enter.
    """
    monkeypatch.setattr(v2, "NATIVE_PANE_READY_TIMEOUT_SECONDS", 0.25)
    monkeypatch.setattr(v2, "_NATIVE_PANE_READY_POLL_SECONDS", 0.005)
    monkeypatch.setattr(v2, "MUSE_STATUS_RETYPE_INTERVAL_SECONDS", 0.05)
    rows = footer_only_rows(str(tmp_path))
    screen = ScriptedScreen(rows, rows, render_when=lambda: False)
    with pytest.raises(v2.ManagedLaunchConflict) as excinfo:
        _observe(screen, str(tmp_path))
    detail = str(excinfo.value)
    attempts = int(detail.split("status_submit_attempts=")[1].split(",")[0])
    assert attempts > 1, detail
    typed = recording_input[-1]
    assert typed.literals == ["/status"] * attempts
    assert typed.enters == attempts
