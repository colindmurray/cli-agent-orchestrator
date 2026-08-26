"""Fixture surfaces for the M10 Codex route-observation canary cases.

Two kinds of surface, both fake — nothing here touches a live pane:

- **Captured-render fixtures** — the faithful 80x30 and 100x30 rows of the
  cond-0230 M10-D0 exact-build capture
  (``~/Projects/cao-conductor-worktrees/cond0230-m10d0-codex-exact-build-capture-20260816/codex-status-{80,100}x30.txt``),
  attestation inputs only.  The retained artifacts redact the session UUID to
  ``<UUID>``; here it is substituted by one concrete canonical UUID so the
  branded ``codex-status-v1`` parser can validate it.  These are the
  render-floor fixtures: at 80 columns (below the 87-column Model floor) the
  truncated ``Model:`` value is ``not-rendered`` and never asserted; at 100
  columns the captured ``(reasoning medium, summaries auto)`` suffix is
  parsed as effort ``medium`` and its trailing display annotation is ignored.
  At 80 columns the Model row remains not-rendered and the model is never
  guessed.
- **Synthetic pinned panels** — a full-width panel whose Model row carries an
  exactly-parseable reasoning suffix, used by the positive-path case.  It
  mirrors the captured build's layout (``>_ OpenAI Codex (v0.147.0)`` brand
  header, ``Session:`` row, value column at index 39) but with a Model value
  inside the closed effort vocabulary, because the captured full-width render
  deliberately is NOT a positive observation.
"""

from __future__ import annotations

from typing import Optional, Sequence

from cli_agent_orchestrator.services import route_observation_codex as roc

CODEX_PINNED_VERSION = "0.147.0"

#: One concrete canonical session UUID substituting the ``<UUID>`` redaction
#: the retained capture artifacts carry.  The branded parser rejects any
#: non-UUID session value, so a fixture surface needs a concrete identity to
#: be parseable at all.
SESSION_ID = "4f5f46c7-b660-4f6f-a144-d2c6dceccf95"

#: Captured ``/status`` render at 100 columns (30 rows), faithful to
#: ``codex-status-100x30.txt``.  The Model row is fully rendered and carries
#: ``(reasoning medium, summaries auto)`` — effort ``medium`` followed by a
#: non-authoritative display annotation that the adapter ignores completely.
CAPTURED_STATUS_100X30_ROWS: tuple[str, ...] = (
    "╰─────────────────────────────────────────────────────╯",
    "",
    "  Tip: Try the Desktop app. Run 'codex app' or visit https://chatgpt.com/codex?app-landing-page=true",
    "",
    "⚠ Heads up, you have less than 25% of your weekly limit left. Run /status for a breakdown.",
    "",
    "/status",
    "",
    "╭────────────────────────────────────────────────────────────────────────────────────────────────╮",
    "│  >_ OpenAI Codex (v0.147.0)                                                                    │",
    "│                                                                                                │",
    "│ Visit https://chatgpt.com/codex/settings/usage for up-to-date                                  │",
    "│ information on rate limits and credits                                                         │",
    "│                                                                                                │",
    "│  Model:                              gpt-5.6-luna (reasoning medium, summaries auto)           │",
    "│  Directory:                          <SCRATCH>/worktree            │",
    "│  Permissions:                        Full Access                                               │",
    "│  Agents.md:                          AGENTS.md                                                 │",
    "│  Account:                            <ACCOUNT> (Pro)                           │",
    "│  Collaboration mode:                 Default                                                   │",
    "│  Session:                            4f5f46c7-b660-4f6f-a144-d2c6dceccf95                      │",
    "│                                                                                                │",
    "│  Weekly limit:                       [████░░░░░░░░░░░░░░░░] 19% left (resets 01:09 on 20 Aug)  │",
    "│  GPT-5.3-Codex-Spark Weekly limit:   [████████████████████] 100% left (resets 17:06 on 23 Aug) │",
    "╰────────────────────────────────────────────────────────────────────────────────────────────────╯",
    "",
    "",
    "› Find and fix a bug in @filename",
    "",
    "  gpt-5.6-luna medium · <SCRATCH>/worktree",
)

#: Captured ``/status`` render at 80 columns (30 rows), faithful to
#: ``codex-status-80x30.txt``.  80 columns is at/above the 76-column Session
#: floor but below the 87-column Model floor, so the Session UUID is asserted
#: while the Model value (cut off at ``gpt-5.6-luna (reasoning medium,
#: summari``) is ``not-rendered`` — the adapter never guesses the half value.
CAPTURED_STATUS_80X30_ROWS: tuple[str, ...] = (
    "",
    "⚠ Heads up, you have less than 25% of your weekly limit left. Run /status for a",
    "  breakdown.",
    "",
    "/status",
    "",
    "╭──────────────────────────────────────────────────────────────────────────────╮",
    "│  >_ OpenAI Codex (v0.147.0)                                                  │",
    "│                                                                              │",
    "│ Visit https://chatgpt.com/codex/settings/usage for up-to-date                │",
    "│ information on rate limits and credits                                       │",
    "│                                                                              │",
    "│  Model:                              gpt-5.6-luna (reasoning medium, summari │",
    "│  Directory:                          /private<TEMP_PATH>                 │",
    "│  Permissions:                        Full Access                             │",
    "│  Agents.md:                          AGENTS.md                               │",
    "│  Account:                            <ACCOUNT> (Pro)         │",
    "│  Collaboration mode:                 Default                                 │",
    "│  Session:                            4f5f46c7-b660-4f6f-a144-d2c6dceccf95    │",
    "│                                                                              │",
    "│  Weekly limit:                       [████░░░░░░░░░░░░░░░░] 19% left         │",
    "│                                      (resets 01:09 on 20 Aug)                │",
    "│  GPT-5.3-Codex-Spark Weekly limit:   [████████████████████] 100% left        │",
    "│                                      (resets 17:06 on 23 Aug)                │",
    "╰──────────────────────────────────────────────────────────────────────────────╯",
    "",
    "",
    "› Find and fix a bug in @filename",
    "",
    "  gpt-5.6-luna medium · <SCRATCH>/worktree",
)

#: The closed reasoning-effort vocabulary the installed 0.147.0 build accepts.
#: Mirrors ``route_observation_codex._CODEX_EFFORT_VOCABULARY``; a suffix
#: outside this set is malformed evidence and is refused, never guessed.
CODEX_EFFORT_VOCABULARY = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "ultra"})


def codex_route_panel_rows(
    session_id: str = SESSION_ID,
    *,
    model: str = "gpt-5.6-luna",
    effort: str | None = "medium",
    version: str = CODEX_PINNED_VERSION,
) -> list[str]:
    """A synthetic pinned Codex status panel with a parseable Model row.

    Mirrors the captured build's layout — brand header ``>_ OpenAI Codex
    (v0.147.0)``, a ``Session:`` row, value column at index 39 — with a Model
    value whose reasoning suffix is exactly in the closed effort vocabulary.
    The captured full-width render carries
    ``(reasoning medium, summaries auto)``; the adapter extracts ``medium``
    and ignores the trailing annotation.
    ``effort=None`` omits the suffix; ``model=None`` omits the Model row
    entirely (a full-width panel that still lacks it is truncated/different,
    never a positive observation).
    """
    rows = [f">_ OpenAI Codex (v{version})", f"Session: {session_id}"]
    if model is not None:
        value = model + (f" (reasoning {effort})" if effort else "")
        rows.append(f"Model: {value}")
    rows.append("cwd: /Users/x/repo")
    return rows


class FakeCodexPaneSurface:
    """A fake Codex pane surface for one canary case.

    Records every status command and every key event so a case can prove
    at-most-once ``/status`` and the absence of any ``Escape`` on the
    non-modal surface.  ``composer_restored`` is the close-proof verdict:
    ``True`` (proven), ``False`` (proven not restored), or ``None``
    (unprovable -> ``indeterminate``).  ``send_key`` exists only as the
    recording seam for a hypothetical Escape; the adapter never calls it.
    """

    def __init__(
        self,
        *,
        rows: Sequence[str],
        pane_width: int | None = 100,
        submission_proven: bool = True,
        composer_restored: bool | None = True,
    ) -> None:
        self._rows = list(rows)
        self._pane_width = pane_width
        self._submission_proven = submission_proven
        self._composer_restored = composer_restored
        self.status_commands_sent = 0
        self.key_events: list[str] = []

    @property
    def pane_id(self) -> str:
        return "%9"

    def capture_screen(self) -> list[str]:
        return list(self._rows)

    def pane_width(self) -> Optional[int]:
        return self._pane_width

    def await_input_ready(self) -> roc.PrewriteReadiness:
        return roc.PrewriteReadiness(roc.PREWRITE_READY, "idle")

    def send_status_command(self) -> bool:
        self.status_commands_sent += 1
        return self._submission_proven

    def composer_restored(self) -> Optional[bool]:
        return self._composer_restored

    def send_key(self, keystroke: str) -> None:
        self.key_events.append(keystroke)
