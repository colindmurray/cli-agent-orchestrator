"""Codex direct passive CAO goal restoration (cond-0845 slice 2, Codex).

A managed Codex worker loses its CAO assignment on compaction: the native
thread survives, but the model-facing goal capsule it was launched with is
compacted away. This module restores it passively through Codex's own
``SessionStart`` hook with matcher ``^compact$``, which the official hooks
reference verifies fires after root-session compaction — including
automatic mid-turn compaction, where the hook's additional context goes to
the immediate continuation instead of waiting for a later user turn.

Official contract (verified 2026-09-09 against
``https://learn.chatgpt.com/docs/hooks`` with installed codex-cli 0.153.4;
config acceptance additionally verified offline via
``CODEX_HOME=<scratch> codex doctor``: ``config.toml parse ok``,
``config loaded``, 0 fail):

* ``SessionStart`` matchers filter on ``source``: ``startup``, ``resume``,
  ``clear``, ``compact``. This adapter installs **``^compact$`` only**.
* stdin carries ``session_id`` (plus ``transcript_path``, ``cwd``,
  ``hook_event_name``, ``model``, ``permission_mode``). No generation or
  terminal id — the managed generation is baked into the hook command at
  install time from the authoritative launch record, never read from stdin.
* stdout ``{"hookSpecificOutput": {"hookEventName": "SessionStart",
  "additionalContext": "..."}}`` is added as extra developer context.
  Oversized output spills (≈2,500 tokens) to
  ``<tmp>/hook_outputs/<session_id>/<uuid>.txt`` with a preview, so this
  adapter stays well under that and truncates explicitly. The per-handler
  ``additionalContextLimit`` tunes the threshold.
* Hook commands run with the session cwd. Default hook timeout is 600s;
  this adapter bounds itself far below that (see
  :data:`CONDUCT_TIMEOUT_SECONDS`).
* If more than one hook source exists, Codex loads all matching hooks, so
  a worker-scoped ``config.toml`` is additive over the user's own hooks —
  which this adapter never writes (see :func:`compose_codex_home`).

Child/parent identity (the Codex-critical rule): the reference states
"Subagent hooks use the parent session id", and a subagent start fires
``SessionStart`` (indistinguishable payload: same ``session_id``, no
subagent marker) alongside ``SubagentStart``. Restoring on ``startup``
would therefore inject the parent's goal into every child at spawn. This
adapter installs no ``startup``/``resume`` matcher and no
``SubagentStart``/``SubagentStop`` entries, so a subagent start never
triggers restoration — structurally, not by payload inspection, because
there is nothing truthful to inspect. Residual: a subagent's own mid-run
compaction would still present the parent id under ``^compact$``; that
behavior is UNPROVEN on the installed rev (no live model launch in this
lane) and is a deployed-QA validation case, not a guessed mapping — the
wrapper never invents a child identity. ``PostCompact`` is deliberately
not a second mechanism: same subagent exposure with unclear ordering
against ``SessionStart^compact``, and §10.1 selects one primary mechanism
per worker.

What this adapter does NOT do, by construction:

* It never submits a turn, calls steer/send, releases a hold, or bypasses
  a resume-paused worker. The wrapper's only subprocess is ``conduct goal
  hook-context`` — the read-only slice-1 projection (conductor PR #349,
  ``1f1e415f``) — which starts zero turns by contract. The argv is pinned
  in :func:`build_conduct_argv` and asserted in tests; there is no code
  path that could utter any other verb.
* It never invents a goal. When the projection answers anything but ``ok``
  (startup before admission, stale generation after rotation, unknown
  worker, unreadable store), the wrapper restores nothing: missing
  assignments render a short truthful note so the next compact recovers
  the latest goal, while stale/ambiguous callbacks render empty context
  so a former assignment is never re-injected. Lookup failure (conduct
  missing, slow, or malformed) is scoped to restoration only: exit 0 with
  empty context, never a blocked session start.
* No timers, no coalescing, no event steer, no physical resume behaviour.
  Those are later slices, not this candidate.
"""

from __future__ import annotations

import argparse
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
HARNESS = "codex"

#: The only hook event that restores context, and the only matcher. A
#: subagent start fires SessionStart with the parent session id and no
#: distinguishing field, so ``startup``/``resume`` matchers would inject
#: the parent goal into every child at spawn. ``compact`` is the event
#: this slice exists for; rotation/resume replay is out of scope until a
#: proven child distinguisher exists (deployed QA, not a guess).
RESTORE_HOOK_EVENT = "SessionStart"
COMPACT_MATCHER = "^compact$"

#: The official spill threshold is ≈2,500 tokens (≈10k chars); this adapter
#: stays under it with room for sibling hooks' values, and truncates
#: explicitly rather than letting the provider spill silently.
MAX_ADDITIONAL_CONTEXT_CHARS = 6000

#: Per-handler spill threshold for the installed entry. Under the wrapper
#: bound above, so the provider limit is never what cuts the text.
ADDITIONAL_CONTEXT_LIMIT = 2000

#: How long the wrapper waits for the read-only projection before
#: degrading to empty context. Compaction hooks run before the next model
#: request, so this stays small — but it is a bound, not a non-block
#: guarantee: the wrapper waits synchronously, so a wedged server can
#: delay the continuation by up to this long before the hook gives up and
#: restores nothing.
CONDUCT_TIMEOUT_SECONDS = 10.0

#: Label marking every injected string as restoration. The model must be
#: able to tell restored assignment context from live conversation.
RESTORATION_LABEL = "CAO goal restoration"

#: Rendered when the projection is reachable but holds no projectable
#: goal yet (typically a compact before task admission). Truthful and
#: short: it names what is missing and which event recovers it, and it
#: carries no objective text that could stale-instruct.
UNAVAILABLE_PREFIX = "CAO context restoration unavailable"

#: Marker proving an injected body was cut to the bound rather than
#: silently spilled by the provider.
TRUNCATION_MARKER = "[truncated here; full goal via `conduct goal current`]"

#: Installed console entry for the wrapper (see ``[project.scripts]``).
WRAPPER_ENTRY_POINT = "cao-codex-hook-context"

#: Slice-1 projection envelope this wrapper understands. Unknown schemas
#: are uninterpretable, so they degrade to empty context, never to a
#: guess about what their fields mean.
HOOK_CONTEXT_SCHEMA = "cao-hook-context-v1"

#: Goal states whose assignment is over (§10.2): restoring their
#: objective text would re-inject a finished assignment as if it were
#: current work. An ``ok`` projection carrying one of these renders
#: nothing.
TERMINAL_GOAL_STATES = frozenset({"satisfied", "cancelled"})

#: The selected restoration mechanism, recorded in the existing launch
#: facts so diagnostics can show what a generation installed (§10.1).
MECHANISM = "codex-SessionStart:^compact$"

#: Directory name of the generation-private Codex home under the
#: per-terminal/generation companion dir (Kimi ``kimi-home`` precedent).
PRIVATE_HOME_DIRNAME = "codex-home"

#: Managed config filename inside the private home. The only file the
#: launcher owns there; everything else links back to the provider home.
MANAGED_CONFIG_FILENAME = "config.toml"

#: Provider-home entries that must NOT link back into the private home.
#: ``config.toml`` is the managed file (user hooks/config are never
#: written, only read as the append base). Auth-adjacent and
#: generation-scoped provider state stays private per generation: sharing
#: threads, memory, goals, or queue rows across generations would let a
#: successor generation resume or observe its predecessor's state — the
#: exact staleness class this increment fights. ``auth.json`` links back
#: (identity, not state); the doctor probe confirms it is home-scoped.
_LINK_BACK_DENY = frozenset({
    "config.toml",
    "state_5.sqlite",
    "logs_2.sqlite",
    "goals_1.sqlite",
    "memories_1.sqlite",
    "queue_1.sqlite",
    "thread_history_1.sqlite",
})

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


def restore_hook_toml(*, command: str) -> str:
    """The managed ``[[hooks.SessionStart]]`` TOML block for one command.

    Pure text: the caller appends it to the worker's config base (see
    :func:`compose_managed_config`). Anchored ``^compact$`` matcher — the
    only event this adapter installs — with the per-handler spill limit.
    """
    quoted = json.dumps(command)
    return (
        "\n"
        "# Managed by CAO context restoration (cond-0845, codex). Do not\n"
        "# hand-edit: this file is composed per managed generation.\n"
        "[[hooks.SessionStart]]\n"
        f"matcher = {json.dumps(COMPACT_MATCHER)}\n"
        "[[hooks.SessionStart.hooks]]\n"
        'type = "command"\n'
        f"command = {quoted}\n"
        f"additionalContextLimit = {ADDITIONAL_CONTEXT_LIMIT}\n"
    )


def compose_managed_config(base_text: str, *, command: str) -> str:
    """Worker config text: user base preserved byte-identical plus one block.

    Textual append, never a TOML round-trip: ``[[hooks.SessionStart]]`` is
    an array of tables, so a repeated block is valid TOML and Codex loads
    hooks from every source that defines them. A malformed user base is
    not repaired here — Codex rejects the whole file loudly, which is the
    honest failure, and the launch degrades before the mint (see the
    caller). The base is never written back to the user's home.
    """
    block = restore_hook_toml(command=command)
    if base_text and not base_text.endswith("\n"):
        base_text += "\n"
    return base_text + block


def _executable(path: str) -> Optional[str]:
    """``path`` when it names something this host could execute, else None."""
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


def resolve_wrapper_executable(explicit: Optional[str] = None) -> Optional[str]:
    """Absolute wrapper path, or None when it cannot be resolved.

    None is a normal answer, not an error: a launch whose environment
    cannot resolve the wrapper degrades to restoration-uninstalled
    settings (logged by the caller) rather than failing a launch over a
    restoration hook.
    """
    if explicit:
        if os.path.dirname(explicit):
            return _executable(explicit)
        found = shutil.which(explicit)
        return _executable(found) if found else None
    found = shutil.which(WRAPPER_ENTRY_POINT)
    return _executable(found) if found else None


def resolve_conduct_binary(explicit: Optional[str] = None) -> Optional[str]:
    """Absolute ``conduct`` path, or None when it cannot be resolved."""
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
    """The shell command baked into the managed hooks config.

    The terminal id and generation come from the authoritative launch
    record at install time — they are what make the projection's
    generation fence provable (``verified``) instead of merely current.
    The hook's own stdin contributes only the native session id, because
    that is the one identity the provider actually observes. The command
    runs with the session cwd; absolute baked paths keep it independent
    of it.
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


def compose_codex_home(
    *,
    companion_dir: str,
    terminal_id: str,
    generation: str,
    provider_home: str,
    base_config_text: str,
    command: str,
) -> Tuple[str, Optional[str]]:
    """Build the generation-private Codex home. Returns ``(home, degraded)``.

    Layout (Kimi ``kimi-home`` precedent): everything in the provider home
    links back except the deny list — ``config.toml`` is composed fresh
    from the user's base text plus exactly one managed block, and
    ``*.sqlite*`` provider-state databases stay per generation so a
    successor can never resume or observe its predecessor's threads,
    memory, goals, or queue rows. ``auth.json`` links back (identity, not
    state). ``degraded`` is None on success; otherwise names what failed
    and the caller launches without restoration rather than refusing.
    """
    private_home = Path(companion_dir) / terminal_id / generation / PRIVATE_HOME_DIRNAME
    try:
        private_home.mkdir(mode=0o700, parents=True, exist_ok=True)
        provider = Path(provider_home).expanduser()
        if provider.is_dir():
            for source in sorted(provider.iterdir()):
                if source.name == MANAGED_CONFIG_FILENAME or (
                    source.suffix == ".sqlite"
                    or source.suffixes[-2:] == [".sqlite", "-journal"]
                    or source.suffixes[-2:] == [".sqlite", "-wal"]
                ):
                    continue
                # _LINK_BACK_DENY names the exact state files observed on
                # the pinned rev; the suffix rules above carry the same
                # policy across provider renames.
                if source.name in _LINK_BACK_DENY:
                    continue
                destination = private_home / source.name
                if destination.exists() or destination.is_symlink():
                    if not destination.is_symlink() or destination.resolve() != source.resolve():
                        return str(private_home), (
                            f"generation-private Codex home entry drifted: {destination}"
                        )
                    continue
                destination.symlink_to(source, target_is_directory=source.is_dir())
        config_path = private_home / MANAGED_CONFIG_FILENAME
        config_path.write_text(
            compose_managed_config(base_config_text, command=command),
            encoding="utf-8",
        )
        config_path.chmod(0o600)
    except OSError as exc:
        return str(private_home), f"could not compose generation-private Codex home: {exc}"
    return str(private_home), None


def describe_installation(
    *,
    terminal_id: str,
    generation: str,
    codex_home: Optional[str],
    degraded_reason: Optional[str],
) -> Dict[str, Any]:
    """The launch-facts record of what restoration this generation got.

    One small dict, written to the existing launch facts by the caller:
    the selected mechanism when installed, or the reason the generation
    launched without restoration. A later diagnostic reads this instead
    of re-deriving install state from logs.
    """
    return {
        "mechanism": MECHANISM if degraded_reason is None else None,
        "terminal_id": terminal_id,
        "terminal_generation": generation,
        "codex_home": codex_home,
        "degraded_reason": degraded_reason,
    }


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
    of them the same silent-empty way. Note what is NOT inspected here:
    Codex gives a subagent's SessionStart the parent session id with no
    distinguishing field, so no payload inspection could separate child
    from parent — that separation lives in the installed matcher
    (``^compact$`` only), never in a guessed mapping.
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

    An empty ``additionalContext`` string keeps the hook schema-valid so
    the session continues with no error notice and no injected text.
    (Exit 0 with no output would also succeed; the explicit empty object
    keeps every wrapper outcome in one observable shape.)
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
        goal = answer.get("goal")
        if isinstance(goal, dict) and goal.get("state") in TERMINAL_GOAL_STATES:
            # §10.2: a satisfied or cancelled assignment is over. The
            # projection says "no restoration" for it; this consumer
            # enforces that by restoring nothing, rather than carrying
            # the verdict prose alongside the finished objective.
            return None
        rendered = _render_goal(goal, answer.get("identity"))
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
            "the next compact recovers it."
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
    still exits 0: a restoration hook must never block the continuation.
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
            f"goal projection lookup failed ({exc}); restoration only, session unaffected",
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
        description="Codex SessionStart CAO goal restoration (read-only; starts no turn)",
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
    # Stdout carries ONLY the JSON object: any other text would break the
    # hook's schema validation (and Codex adds plain stdout as developer
    # context, so stray prints would leak into the model).
    sys.stdout.write(json.dumps(output))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


def _provider_home(base_environment: Dict[str, str]) -> Path:
    """The Codex home the worker would inherit without restoration."""
    configured = base_environment.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def _read_base_config(provider_home: Path) -> str:
    """User config text, or empty when the provider home has none.

    Read, never written: the user's ``config.toml`` is the append base
    only. A missing file is normal (fresh provider home); an unreadable
    one degrades the whole installation (fail visibly at prepare time,
    not as a half-composed home).
    """
    path = provider_home / MANAGED_CONFIG_FILENAME
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise RuntimeError(f"could not read Codex user config {path}: {exc}") from exc


def prepare_codex_restoration(
    *,
    record: Dict[str, Any],
    base_environment: Dict[str, str],
    companion_dir: str,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Give a native Codex generation its restoration binding.

    Mirrors ``_kimi_profile_environment``: returns ``(environment,
    installation)`` where the environment points ``CODEX_HOME`` at a
    generation-private home (composed config + linked-back provider
    state) and the installation dict feeds the existing launch-facts
    record. Never raises for restoration causes — every failure degrades
    to an unmodified environment with a reason, so a launch is never
    refused over a restoration hook. Programmer errors (missing record
    keys) raise KeyError loudly instead of installing a wrong binding.
    """
    terminal_id = record["terminal_id"]
    generation = record["generation"]
    try:
        wrapper = resolve_wrapper_executable()
        conduct = resolve_conduct_binary()
        if wrapper is None or conduct is None:
            missing = ", ".join(
                name
                for name, value in (("wrapper", wrapper), ("conduct", conduct))
                if value is None
            )
            reason = (
                f"context restoration not installed ({missing} unresolvable); "
                "launching without restoration hooks"
            )
            return dict(base_environment), describe_installation(
                terminal_id=terminal_id,
                generation=generation,
                codex_home=None,
                degraded_reason=reason,
            )
        command = restore_command(
            wrapper_executable=wrapper,
            terminal_id=terminal_id,
            generation=generation,
            conduct_binary=conduct,
        )
        provider_home = _provider_home(base_environment)
        home, degraded = compose_codex_home(
            companion_dir=companion_dir,
            terminal_id=terminal_id,
            generation=generation,
            provider_home=str(provider_home),
            base_config_text=_read_base_config(provider_home),
            command=command,
        )
        if degraded is not None:
            return dict(base_environment), describe_installation(
                terminal_id=terminal_id,
                generation=generation,
                codex_home=None,
                degraded_reason=degraded,
            )
        environment = dict(base_environment)
        environment["CODEX_HOME"] = home
        return environment, describe_installation(
            terminal_id=terminal_id,
            generation=generation,
            codex_home=home,
            degraded_reason=None,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return dict(base_environment), describe_installation(
            terminal_id=terminal_id,
            generation=generation,
            codex_home=None,
            degraded_reason=f"context restoration not installed ({exc}); launching without it",
        )
