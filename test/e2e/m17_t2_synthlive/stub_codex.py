#!/usr/bin/env python3
"""M17 T2 synthetic-live stub ``codex`` binary (self-contained template).

This file is the committed SOURCE of the stub ``codex`` executable the T2
synthetic-live lane installs at the attested
``~/.local/share/uv/tools/cli-agent-orchestrator/bin/codex`` layout.  It is a
**stub of the provider binary only**: every pane byte it renders is REAL tmux
content, and nothing in it calls a model, an API, or spends quota.

The stub refuses to run unless ``T2_SYNTHLIVE=1`` is set: a leaked copy must
never be a working ``codex`` that fabricates provider output outside the T2
harness.  The harness sets the guard for every pane and every launch it makes.

The captured-render fixtures (``CAPTURED_STATUS_80X30_ROWS`` /
``CAPTURED_STATUS_100X30_ROWS``) and the synthetic positive panel are embedded
by the test at install time by replacing the four ``_STUB_*_ROWS_`` markers
with Python list literals.  The committed file therefore reads with those
markers as bare names; only the installed copy is executable.

Modes (argv):

- ``codex status`` — render the captured status fixture whose width matches
  the pane, as exact pane bytes (no trailing newline), then stay alive.
- ``codex`` (no args) — interactive: render the full 100-column captured
  context with a parseable ``Model:`` row plus the ``› `` composer prompt,
  and redraw on every submitted line so a submitted ``/status`` leaves the
  composer region (the submission-observation seam the real adapter relies
  on).  ``--redraw-delay <seconds>`` inserts that pause between a submitted
  line and its redraw, which the pane-death mid-observation case uses to
  open a wide, deterministic window in which the pane's shell is really
  killed after the submission barrier has genuinely passed.
- ``codex garbage`` — first render the ordinary writable Codex surface, then
  replace it with 30 rows that are NOT a Codex status panel only after the
  adapter submits literal ``/status``.  This exercises an inconclusive
  post-effect observation without making unreadable startup content stand in
  for prewrite readiness.
"""

from __future__ import annotations

import os
import subprocess
import sys
import termios
import time

# Replaced by the test with ``repr(...)`` of the real fixture content.  The
# committed names are intentionally undefined: they are markers, and only the
# installed copy carries the fixture literals.
_POSITIVE_ROWS: list[str] = _STUB_POSITIVE_ROWS_
_FIX80_ROWS: list[str] = _STUB_FIX80_ROWS_
_FIX100_ROWS: list[str] = _STUB_FIX100_ROWS_
_GARBAGE_ROWS: list[str] = _STUB_GARBAGE_ROWS_

#: The only environment that may run the stub.  Anything else is a leaked
#: copy and must fail closed rather than fabricate provider output.
GUARD_ENV = "T2_SYNTHLIVE"
GUARD_VALUE = "1"

#: The narrowest width that picks the 100-column capture over the 80-column
#: one.  Between the 76-column Session floor and the 87-column Model floor,
#: which fixture renders is the pane's business; the adapter re-derives the
#: floors itself from ``#{pane_width}``.
_FIX100_AT_LEAST = 90


def guard_ok() -> bool:
    return os.environ.get(GUARD_ENV) == GUARD_VALUE


def pane_width() -> int:
    """The pane's real column width, read through tmux when available."""
    pane = os.environ.get("TMUX_PANE", "")
    if pane:
        try:
            result = subprocess.run(
                ["tmux", "display", "-p", "-t", pane, "#{pane_width}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            width = int((result.stdout or "").strip())
            if width > 0:
                return width
        except Exception:
            pass
    columns = os.environ.get("COLUMNS", "")
    if columns.isdigit() and int(columns) > 0:
        return int(columns)
    return 100


def render_status() -> None:
    """Print the captured fixture for the pane's width, exactly, then idle."""
    rows = _FIX100_ROWS if pane_width() >= _FIX100_AT_LEAST else _FIX80_ROWS
    sys.stdout.write("\n".join(rows))
    sys.stdout.flush()
    try:
        sys.stdin.read()
    except Exception:
        pass


def _draw(rows: list[str]) -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.write("\n".join(rows))
    sys.stdout.flush()


def _draw_composer(rows: list[str], text: str) -> None:
    composer = ["", f"› {text}", ""]
    if text.startswith("/"):
        composer.extend(
            [
                "  /status      show current session configuration and token usage",
                "  /statusline  configure which items appear in the status line",
            ]
        )
    else:
        composer.append("  gpt-5.6-luna high · ~/project")
    _draw([*rows, *composer])


def _interactive_terminal_settings(prior_terminal: list[object]) -> list[object]:
    """Return raw-input settings without mutating the restore snapshot."""
    interactive_terminal = list(prior_terminal)
    interactive_terminal[6] = list(prior_terminal[6])  # type: ignore[arg-type]
    interactive_terminal[3] = int(interactive_terminal[3]) & ~(termios.ECHO | termios.ICANON)
    interactive_terminal[6][termios.VMIN] = 1  # type: ignore[index]
    interactive_terminal[6][termios.VTIME] = 0  # type: ignore[index]
    return interactive_terminal


def _redraw_on_submit(
    rows: list[str], *, redraw_delay: float, after_status_rows: list[str] | None = None
) -> None:
    """Draw ``rows`` and redraw after each submitted line.

    When ``after_status_rows`` is provided, an exact submitted ``/status``
    selects those rows; every other line redraws the initial surface.
    ``redraw_delay`` gives the pane-death case a deterministic window after
    the barrier passes and before the composer is cleared.
    """
    _draw(rows)
    line = b""
    stdin_fd = sys.stdin.fileno()
    prior_terminal = termios.tcgetattr(stdin_fd)
    interactive_terminal = _interactive_terminal_settings(prior_terminal)
    termios.tcsetattr(stdin_fd, termios.TCSANOW, interactive_terminal)
    try:
        while True:
            try:
                chunk = sys.stdin.buffer.read(1)
            except Exception:
                break
            if not chunk:
                break
            if chunk in {b"\r", b"\n"}:
                submitted = line
                line = b""
                if redraw_delay > 0:
                    time.sleep(redraw_delay)
                if submitted == b"/status" and after_status_rows is not None:
                    _draw_composer(after_status_rows, "")
                else:
                    _draw_composer(rows, "")
                continue
            line += chunk
            try:
                composed = line.decode("utf-8")
            except UnicodeDecodeError:
                continue
            _draw_composer(rows, composed)
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSANOW, prior_terminal)


def interactive(*, redraw_delay: float) -> None:
    """Render the positive panel plus composer; redraw on each submitted line."""
    _redraw_on_submit(_POSITIVE_ROWS, redraw_delay=redraw_delay)


def render_garbage() -> None:
    """Start writable, then redraw unparseable rows after submitted /status."""
    _redraw_on_submit(
        _POSITIVE_ROWS,
        redraw_delay=0.0,
        after_status_rows=_GARBAGE_ROWS,
    )


def _redraw_delay_argv(argv: list[str]) -> float:
    if "--redraw-delay" in argv:
        try:
            index = argv.index("--redraw-delay")
            return float(argv[index + 1])
        except (IndexError, ValueError):
            pass
    return 0.0


def main() -> None:
    if not guard_ok():
        sys.stderr.write(
            f"stub codex: {GUARD_ENV}={GUARD_VALUE} is required; refusing to "
            "fabricate provider output outside the T2 synthetic-live harness\n"
        )
        sys.exit(3)
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        render_status()
    elif len(sys.argv) > 1 and sys.argv[1] == "garbage":
        render_garbage()
    else:
        interactive(redraw_delay=_redraw_delay_argv(sys.argv[1:]))


if __name__ == "__main__":
    main()
