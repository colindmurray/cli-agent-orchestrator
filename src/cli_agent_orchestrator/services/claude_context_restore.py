"""Claude direct passive CAO goal restoration (cond-0845 slice 2, Claude first).

A managed Claude worker loses its CAO assignment on compaction: the native
session survives, but the model-facing goal capsule it was launched with is
compacted away. This module restores it passively through Claude's own
``SessionStart`` hook, which is the one hook event that fires post-compaction
(matcher ``compact``) as well as on admitted startup/resume, and whose
``hookSpecificOutput.additionalContext`` is inserted as a system reminder
read on the next model request.

Official contract (verified 2026-09-09 against
``https://code.claude.com/docs/en/hooks`` with installed claude 2.1.233):

* ``SessionStart`` matchers filter on ``source``: ``startup``, ``resume``,
  ``clear``, ``compact``, ``fork``. This adapter installs ``compact`` plus
  ``startup|resume``. ``clear`` is excluded: a cleared session is a fresh
  conversation by operator intent, not a compaction to recover from.
* stdin carries ``session_id`` (plus ``cwd``, ``transcript_path``,
  ``hook_event_name``, ``source``, optional ``model``). No generation or
  terminal id — the managed generation is baked into the hook command at
  install time from the authoritative launch record, never read from stdin.
* stdout ``{"hookSpecificOutput": {"hookEventName": "SessionStart",
  "additionalContext": "..."}}`` is wrapped in a system reminder. Every
  hook's value is received (multi-hook values concatenate). Strings are
  capped at 10,000 chars (spill to file + preview), so this adapter stays
  well under that and truncates explicitly.
* Hook entries merge across settings levels rather than replacing each
  other, so a managed ``--settings`` payload is additive over user/project
  hooks. Within this adapter's own payload, :func:`with_context_restore`
  preserves every pre-existing entry (notably the readiness hook, which
  carries no matcher) while appending the restore entries.

What this adapter does NOT do, by construction:

* It never submits a turn, calls steer/send, releases a hold, or bypasses
  a resume-paused worker. The wrapper's only subprocess is ``conduct goal
  hook-context`` — the read-only slice-1 projection (conductor PR #349,
  ``1f1e415f``) — which starts zero turns by contract. The argv is pinned
  in :func:`build_conduct_argv` and asserted in tests; there is no code
  path that could utter any other verb.
* It never invents a goal. When the projection answers anything but ``ok``
  (startup before admission, stale generation after rotation, unknown
  worker, unreadable store), the wrapper restores nothing: startup-timing
  unavailability renders a short truthful note so the next eligible hook
  (resume/compact) can recover the latest goal, while stale/ambiguous
  callbacks render empty context so a former assignment is never
  re-injected. Lookup failure (conduct missing, slow, or malformed) is
  scoped to restoration only: exit 0 with empty context, never a blocked
  session start.
* No timers, no coalescing, no event steer, no physical resume behaviour.
  Those are later slices, not this candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The claiming harness as the conductor's canonical provider id. Checked
#: against the live provider fork-side; never trusted blind (slice-1 seam).
HARNESS = "claude_code"

#: The hook event that restores context. SessionStart, and only
#: SessionStart: PostCompact has no decision control and its stdout is
#: user-only, so it cannot inject (verified in the official reference).
RESTORE_HOOK_EVENT = "SessionStart"

#: Post-compaction recovery. ``compact`` fires after auto or manual
#: compaction — the event this slice exists for.
COMPACT_MATCHER = "compact"

#: Admitted startup/resume recovery. A fresh managed launch (or an exact
#: resume) replays the current goal into context at session start; when
#: startup precedes goal/admission availability the wrapper answers a
#: truthful unavailable note and the next eligible hook recovers.
STARTUP_RESUME_MATCHER = "startup|resume"

#: The official per-string cap is 10,000 chars (spill to file + preview).
#: This adapter stays under it with room for sibling hooks' values, and
#: truncates explicitly rather than letting the provider spill silently.
MAX_ADDITIONAL_CONTEXT_CHARS = 8000

#: How long the wrapper waits for the read-only projection before
#: degrading to empty context. SessionStart hooks must stay fast (they
#: run on every session), and a slow read must never stall a start.
CONDUCT_TIMEOUT_SECONDS = 20.0

#: Label marking every injected string as restoration. The model must be
#: able to tell restored assignment context from live conversation.
RESTORATION_LABEL = "CAO goal restoration"

#: Rendered when the projection is reachable but holds no projectable
#: goal yet (typically startup before task admission). Truthful and
#: short: it names what is missing and which event recovers it, and it
#: carries no objective text that could stale-instruct.
UNAVAILABLE_PREFIX = "CAO context restoration unavailable"

#: Marker proving an injected body was cut to the bound rather than
#: silently spilled by the provider.
TRUNCATION_MARKER = "[truncated here; full goal via `conduct goal current`]"

#: Installed console entry for the wrapper (see ``[project.scripts]``).
WRAPPER_ENTRY_POINT = "cao-claude-hook-context"

#: Slice-1 projection envelope this wrapper understands. Unknown schemas
#: are uninterpretable, so they degrade to empty context, never to a
#: guess about what their fields mean.
HOOK_CONTEXT_SCHEMA = "cao-hook-context-v1"

#: Projection answers that carry a projectable goal.
_OK_RESULT = "ok"

#: Projection answers that name a missing (not former) assignment: the
#: worker exists but nothing projectable is stored yet.
_MISSING_ASSIGNMENT_RESULTS = frozenset({"no-assignment", "unavailable"})

#: Projection answers that name a former or unresolvable binding: the
#: callback predates a rotation or names another worker. Restoring here
#: would project another assignment's context, so these stay silent.
_DISCARD_RESULTS = frozenset({"stale-generation", "dead-incarnation", "ambiguous", "no-worker"})

#: Result types the slice-1 seam documents. Anything else is a contract
#: this reader does not speak.
_KNOWN_RESULTS = frozenset({_OK_RESULT} | _MISSING_ASSIGNMENT_RESULTS | _DISCARD_RESULTS)


def restore_hook_entries(command: str) -> List[Dict[str, Any]]:
    """The additive ``SessionStart`` entries installing restoration.

    Two entries, one command: ``compact`` recovers post-compaction and
    ``startup|resume`` replays the current goal on admitted start. Both
    invoke the same wrapper, which re-reads the *current* projection on
    every event — so a repeated compact or a version change between
    events always renders the latest goal, never a baked copy.
    """
    return [
        {
            "matcher": COMPACT_MATCHER,
            "hooks": [{"type": "command", "command": command}],
        },
        {
            "matcher": STARTUP_RESUME_MATCHER,
            "hooks": [{"type": "command", "command": command}],
        },
    ]


def with_context_restore(settings: Dict[str, Any], *, command: str) -> Dict[str, Any]:
    """Return ``settings`` plus the restore entries, preserving all else.

    The input is not mutated: the caller keeps its readiness-only payload
    and this returns the readiness+restore composition. Every pre-existing
    entry — the readiness hook (which carries no matcher), user/project
    entries merged from files, any other event — is carried over
    byte-identical. Only ``hooks.SessionStart`` grows, by exactly the two
    entries from :func:`restore_hook_entries`.
    """
    composed = copy.deepcopy(settings)
    hooks = composed.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise TypeError("cannot attach context restoration: 'hooks' is not an object")
    existing = hooks.setdefault(RESTORE_HOOK_EVENT, [])
    if not isinstance(existing, list):
        raise TypeError("cannot attach context restoration: 'hooks.SessionStart' is not a list")
    existing.extend(restore_hook_entries(command))
    return composed


def _executable(path: str) -> Optional[str]:
    """``path`` when it names something this host could execute, else None.

    An install-time baked path whose binary was later uninstalled must
    degrade like an unresolvable one — installing a dead hook command
    would fail every session start with a hook error notice.
    """
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


def resolve_wrapper_executable(explicit: Optional[str] = None) -> Optional[str]:
    """Absolute wrapper path, or None when it cannot be resolved.

    None is a normal answer, not an error: a launch whose environment
    cannot resolve the wrapper degrades to readiness-only settings (logged
    by the caller) rather than failing a launch over a restoration hook.
    """
    if explicit:
        if os.path.dirname(explicit):
            return _executable(explicit)
        found = shutil.which(explicit)
        return _executable(found) if found else None
    found = shutil.which(WRAPPER_ENTRY_POINT)
    return _executable(found) if found else None


def resolve_conduct_binary(explicit: Optional[str] = None) -> Optional[str]:
    """Absolute ``conduct`` path, or None when it cannot be resolved.

    Same degradation contract as :func:`resolve_wrapper_executable`.
    """
    if explicit:
        if os.path.dirname(explicit):
            return _executable(explicit)
        found = shutil.which(explicit)
        return _executable(found) if found else None
    found = shutil.which("conduct")
    return _executable(found) if found else None


def restore_command(
    *,
    wrapper_executable: str,
    terminal_id: str,
    generation: str,
    conduct_binary: str,
    timeout_seconds: float = CONDUCT_TIMEOUT_SECONDS,
) -> str:
    """The shell command baked into the managed ``--settings`` payload.

    The terminal id and generation come from the authoritative launch
    record at install time — they are what make the projection's
    generation fence provable (``verified``) instead of merely current.
    The hook's own stdin contributes only the native session id, because
    that is the one identity the provider actually observes.
    """
    parts = [
        wrapper_executable,
        "--terminal",
        terminal_id,
        "--terminal-generation",
        generation,
        "--conduct-bin",
        conduct_binary,
        "--timeout",
        str(timeout_seconds),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def attach_to_launch_settings(
    settings: Dict[str, Any],
    *,
    terminal_id: str,
    generation: str,
    wrapper_executable: Optional[str] = None,
    conduct_binary: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Additively compose restoration onto launch ``--settings``.

    Returns ``(composed_settings, degraded_reason)``. On success the
    settings carry readiness *and* restoration and the reason is None.
    When either executable is unresolvable the input settings are
    returned unchanged with a reason naming what is missing — a launch
    is never failed over a restoration hook, and the caller logs the
    reason so the degradation is observable rather than silent.
    """
    wrapper = resolve_wrapper_executable(wrapper_executable)
    conduct = resolve_conduct_binary(conduct_binary)
    if wrapper is None or conduct is None:
        missing = ", ".join(
            name
            for name, value in (
                ("wrapper", wrapper),
                ("conduct", conduct),
            )
            if value is None
        )
        return settings, (
            f"context restoration not installed ({missing} unresolvable); "
            "launching readiness-only"
        )
    command = restore_command(
        wrapper_executable=wrapper,
        terminal_id=terminal_id,
        generation=generation,
        conduct_binary=conduct,
    )
    return with_context_restore(settings, command=command), None


def build_conduct_argv(
    *,
    conduct_binary: str,
    native_session_id: str,
    terminal_id: Optional[str] = None,
    terminal_generation: Optional[str] = None,
) -> List[str]:
    """The wrapper's one permitted subprocess: the read-only projection.

    Pinned here so there is exactly one place that decides what the
    wrapper may invoke. ``goal hook-context`` performs only GETs and
    store reads (slice-1 contract): it starts zero turns and grants no
    resume, hold release, or continuation. No other verb may be added
    without changing this function — and its tests.
    """
    argv = [
        conduct_binary,
        "goal",
        "hook-context",
        "--harness",
        HARNESS,
        "--native-session-id",
        native_session_id,
    ]
    if terminal_generation is not None:
        argv += ["--terminal-generation", terminal_generation]
    if terminal_id is not None:
        argv += ["--terminal", terminal_id]
    return argv


def parse_hook_input(raw: bytes) -> Optional[str]:
    """The claimed native session id from hook stdin, or None.

    None covers every unparseable shape — empty stdin, non-JSON bytes, a
    JSON non-object, a missing/non-string/blank ``session_id`` — because
    none of them names a worker to project, and the wrapper answers all
    of them the same silent-empty way.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    return session_id.strip()


def _empty_output() -> Dict[str, Any]:
    """Valid no-op output: restores nothing, blocks nothing.

    An empty ``additionalContext`` string (rather than empty stdout or a
    nonzero exit) keeps the hook schema-valid so session start proceeds
    with no error notice and no injected text.
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": RESTORE_HOOK_EVENT,
            "additionalContext": "",
        }
    }


def _restore_output(context: str) -> Dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": RESTORE_HOOK_EVENT,
            "additionalContext": context,
        }
    }


def _summarize(value: Any, *, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def render_restoration(answer: Dict[str, Any]) -> Optional[str]:
    """The model-facing restoration string for a projection answer.

    Returns the bounded labeled string for ``ok`` answers, a short
    truthful unavailable note for missing-assignment answers, and None
    for every answer that must restore nothing (stale/ambiguous/wrong
    worker, unknown result types, unreadable shapes). Never raises for
    resolution outcomes: an uninterpretable answer degrades to None, the
    same as a discard.
    """
    if not isinstance(answer, dict):
        return None
    if answer.get("schema") != HOOK_CONTEXT_SCHEMA:
        return None
    result_type = answer.get("result_type")
    if result_type == _OK_RESULT:
        rendered = _render_goal(answer.get("goal"), answer.get("identity"))
        if rendered is None:
            return None
        # The bound lives here, not in the field renderers: a long goal
        # is cut once, with the explicit marker, rather than pre-trimmed
        # field by field into a body that merely looks complete.
        return apply_output_bound(rendered)
    if result_type in _MISSING_ASSIGNMENT_RESULTS:
        detail = answer.get("detail")
        reason = _summarize(detail, limit=200) if detail else "no goal is stored yet"
        return (
            f"{UNAVAILABLE_PREFIX}: {reason}. "
            "Ordinary task admission stores the goal first; "
            "the next SessionStart (resume/compact) recovers it."
        )
    return None


def _render_goal(goal: Any, identity: Any) -> Optional[str]:
    if not isinstance(goal, dict):
        return None
    lines = [f"{RESTORATION_LABEL} (read-only; starts no turn, authorizes nothing):"]
    goal_id = goal.get("goal_id")
    state = goal.get("state")
    version = goal.get("goal_version")
    header = " ".join(
        part
        for part in (
            f"goal {goal_id!r}" if goal_id is not None else None,
            f"state {state!r}" if state is not None else None,
            f"version {version!r}" if version is not None else None,
        )
        if part
    )
    if header:
        lines.append(header)
    objective = goal.get("objective")
    if isinstance(objective, str) and objective.strip():
        # Whitespace-collapsed but never length-cut here: the output
        # bound in render_restoration governs, with its explicit marker.
        lines.append(f"objective: {' '.join(objective.split())}")
    outstanding = goal.get("requirements_outstanding")
    if isinstance(outstanding, list) and outstanding:
        shown = ", ".join(str(item) for item in outstanding[:20])
        lines.append(f"outstanding requirements: {shown}")
        count = goal.get("requirements_outstanding_count")
        if isinstance(count, int) and count > len(outstanding[:20]):
            lines.append(f"(+{count - len(outstanding[:20])} more)")
    if goal.get("completion_requirements_truncated"):
        count = goal.get("completion_requirements_count")
        lines.append(f"(requirement contract truncated; {count} total)")
    active_hold = goal.get("active_hold")
    if isinstance(active_hold, dict):
        reason = active_hold.get("reason_kind")
        release = f"{active_hold.get('release_kind')}:{active_hold.get('release_id') or '—'}"
        lines.append(f"waiting: {reason} until {release}")
    next_action = goal.get("next_action")
    if isinstance(next_action, str) and next_action.strip():
        lines.append(_summarize(next_action, limit=400))
    if isinstance(identity, dict):
        fence = identity.get("generation_fence")
        if fence is not None:
            lines.append(f"generation fence: {fence}")
    return "\n".join(lines)


def apply_output_bound(context: str, *, limit: int = MAX_ADDITIONAL_CONTEXT_CHARS) -> str:
    """Enforce the output bound with an explicit truncation marker.

    Bodies within the bound pass through untouched (no marker). Longer
    bodies are cut so the marker still fits, so the model always sees
    that — and where — content was cut, rather than meeting a provider
    spill file mid-assignment.
    """
    if len(context) <= limit:
        return context
    marker = "\n" + TRUNCATION_MARKER
    return context[: max(0, limit - len(marker))] + marker


def run_wrapper(
    *,
    stdin_bytes: bytes,
    argv: Sequence[str],
    run_conduct=None,
) -> Tuple[int, Dict[str, Any], str]:
    """Execute one hook invocation. Always exits 0.

    Returns ``(exit_code, output_json, stderr_note)``. ``run_conduct``
    is the seam tests substitute for the ``conduct`` subprocess; it
    receives the pinned argv and returns the parsed projection answer.
    Production passes None and runs the real binary with the configured
    timeout. Every failure — unparseable input, missing conduct, slow
    conduct, malformed answer, discard verdict — restores nothing and
    still exits 0: a restoration hook must never block session start.
    """
    args = parse_wrapper_argv(list(argv))
    session_id = parse_hook_input(stdin_bytes)
    if session_id is None:
        return 0, _empty_output(), "hook input names no native session; restoring nothing"
    conduct_argv = build_conduct_argv(
        conduct_binary=args.conduct_bin,
        native_session_id=session_id,
        terminal_id=args.terminal,
        terminal_generation=args.terminal_generation,
    )
    try:
        if run_conduct is not None:
            answer = run_conduct(conduct_argv)
        else:
            answer = _run_conduct(conduct_argv, timeout_seconds=args.timeout)
    except Exception as exc:  # noqa: BLE001 - lookup failure is scoped to restoration
        return (
            0,
            _empty_output(),
            f"goal projection lookup failed ({exc}); restoration only, session start unaffected",
        )
    context = render_restoration(answer)
    if context is None:
        return 0, _empty_output(), "projection names no restorable goal; restoring nothing"
    return (
        0,
        _restore_output(apply_output_bound(context, limit=args.max_chars)),
        "",
    )


def _run_conduct(conduct_argv: List[str], *, timeout_seconds: float) -> Dict[str, Any]:
    """Run the real projection binary and parse its answer.

    Raises on every failure mode (missing binary, timeout, nonzero exit,
    malformed JSON, JSON non-object): the caller scopes all of them to
    restoration-only degradation.
    """
    try:
        proc = subprocess.run(
            conduct_argv,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"conduct binary not found: {conduct_argv[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"conduct goal hook-context timed out after {timeout_seconds:.0f}s"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"conduct goal hook-context exited {proc.returncode}: "
            f"{(proc.stderr or b'').decode('utf-8', 'replace').strip()[:200]}"
        )
    try:
        answer = json.loads((proc.stdout or b"").decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError("conduct goal hook-context printed malformed JSON") from exc
    if not isinstance(answer, dict):
        raise RuntimeError("conduct goal hook-context answer is not a JSON object")
    return answer


def parse_wrapper_argv(argv: List[str]) -> argparse.Namespace:
    """Wrapper flags. All binding comes from install-time baking except
    the native session id, which arrives on stdin."""
    parser = argparse.ArgumentParser(
        prog=WRAPPER_ENTRY_POINT,
        description="Claude SessionStart CAO goal restoration (read-only; starts no turn)",
    )
    parser.add_argument(
        "--terminal",
        default=None,
        help="search hint: the terminal this hook serves (from the launch record)",
    )
    parser.add_argument(
        "--terminal-generation",
        default=None,
        help="managed generation baked from the launch record (clears the fence)",
    )
    parser.add_argument(
        "--conduct-bin",
        default="conduct",
        help="conductor CLI binary (absolute path baked at install)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=CONDUCT_TIMEOUT_SECONDS,
        help="seconds to wait for the projection before degrading",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_ADDITIONAL_CONTEXT_CHARS,
        help="bound on the injected additionalContext string",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Process entry: stdin in, hook JSON out, exit 0 always."""
    exit_code, output, stderr_note = run_wrapper(
        stdin_bytes=sys.stdin.buffer.read(),
        argv=list(argv) if argv is not None else sys.argv[1:],
    )
    if stderr_note:
        print(stderr_note, file=sys.stderr)
    # Stdout carries ONLY the JSON object: a shell profile or stray print
    # would break the hook's schema validation.
    sys.stdout.write(json.dumps(output))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
