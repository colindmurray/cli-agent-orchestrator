#!/usr/bin/env python3
"""M17 T2 synthetic-live stub ``codex`` binary (self-contained template).

This file is the committed SOURCE of the stub ``codex`` executable the T2
synthetic-live lane installs at the attested
``~/.local/share/uv/tools/cli-agent-orchestrator/bin/codex`` layout.  It is a
**stub of the provider binary only**: every pane byte it renders is REAL tmux
content, and nothing in it calls a model, an API, or spends quota.

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
  on).  ``--raw`` runs the same renderer with the terminal in raw mode (no
  echo), which the pane-death mid-observation case uses to hold the
  submission barrier open while the pane's shell is really killed.
- ``codex garbage`` — render 30 rows of text that is NOT a codex status
  panel (the negative render case), then stay alive.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Replaced by the test with ``repr(...)`` of the real fixture content.  The
# committed names are intentionally undefined: they are markers, and only the
# installed copy carries the fixture literals.
_POSITIVE_ROWS: list[str] = _STUB_POSITIVE_ROWS_
_FIX80_ROWS: list[str] = _STUB_FIX80_ROWS_
_FIX100_ROWS: list[str] = _STUB_FIX100_ROWS_
_GARBAGE_ROWS: list[str] = _STUB_GARBAGE_ROWS_

#: The narrowest width that picks the 100-column capture over the 80-column
#: one.  Between the 76-column Session floor and the 87-column Model floor,
#: which fixture renders is the pane's business; the adapter re-derives the
#: floors itself from ``#{pane_width}``.
_FIX100_AT_LEAST = 90


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


def _redraw_on_submit(rows: list[str], *, raw: bool) -> None:
    """Draw ``rows`` and redraw on every submitted line so a submitted
    ``/status`` leaves the composer region (the submission-observation seam
    the adapter's barrier relies on).  ``raw`` disables terminal echo so the
    pane-death case can hold the barrier open while the shell is killed."""
    _draw(rows)
    if not raw:
        line = b""
        while True:
            try:
                chunk = sys.stdin.buffer.read(1)
            except Exception:
                break
            if not chunk:
                break
            line += chunk
            if chunk == b"\n":
                line = b""
                _draw(rows)
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        line = b""
        while True:
            chunk = sys.stdin.buffer.read(1)
            if not chunk:
                break
            line += chunk
            if chunk == b"\n":
                line = b""
                _draw(rows)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        _draw(rows)


def interactive(*, raw: bool) -> None:
    """Render the positive panel plus composer; redraw on each submitted line."""
    _redraw_on_submit(_POSITIVE_ROWS, raw=raw)


def render_garbage() -> None:
    """Render rows that cannot parse as a codex status panel; redraw on
    submit so the adapter's submission barrier still resolves and reaches
    the (failed) parse — the negative-render observation."""
    _redraw_on_submit(_GARBAGE_ROWS, raw=False)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        render_status()
    elif len(sys.argv) > 1 and sys.argv[1] == "garbage":
        render_garbage()
    else:
        interactive(raw="--raw" in sys.argv[1:])


if __name__ == "__main__":
    main()
