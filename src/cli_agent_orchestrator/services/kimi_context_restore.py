"""Kimi PostCompact-notify + CAO delivery restoration (cond-0845 Kimi lane).

A managed Kimi worker loses its CAO assignment when its context
compacts. Kimi 0.36.1 offers no context-injecting post-compaction hook:
``PostCompact`` fires through ``fireAndForgetTrigger`` (installed
``main.mjs``), so hook stdout can never enter model context — the hook
is NOTIFY-only by vendor construction. Delivery therefore rides CAO's
own machinery: this adapter installs one ``PostCompact`` entry whose
wrapper calls read-only ``conduct goal hook-context`` and, on an ``ok``
answer with a verified fence, asks the fork context-restore boundary
(``POST /terminals/{id}/context-restore``) to admit-or-refuse exactly
one ``KIND_REMIND``. The boundary re-verifies goal, holds, waits,
lifecycle, occurrence, and turn state under the project flock (shared),
the session fence, and byte admission before the first provider byte.

What this adapter does NOT do, by construction:

* It never submits a turn, calls steer/send, releases a hold, or
  bypasses a resume-paused worker from the hook itself. The wrapper's
  subprocesses are ``conduct goal hook-context`` (read-only, zero
  turns) and one loopback POST to the admitting boundary. The argv is
  pinned in :func:`build_conduct_argv` and asserted in tests.
* It never invents a goal. Anything but ``ok`` + verified fence
  delivers nothing; stale/ambiguous/terminal callbacks exit 0 silently
  so a former assignment is never re-injected.
* No timers here: the periodic path lives in the conductor sentinel
  tick and rendezvouses on the same boundary (one pending row per
  occurrence either way).

Model-entry proof (unique marker in actual model context, active→idle
boundary behavior) belongs to a separately preflighted live driver;
this candidate wires the mechanism and pins every fence around it.
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
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The claiming harness as the conductor's canonical provider id.
HARNESS = "kimi_cli"

#: The notifying event. PostCompact, and only PostCompact: it is the one
#: event that fires after compaction; its output is ignored
#: provider-side (fire-and-forget), so it notifies and never injects.
RESTORE_HOOK_EVENT = "PostCompact"

#: Selected mechanism string for launch facts / diagnostics (§10.1).
MECHANISM = "postcompact-notify-then-cao-delivery"

#: Marker comment identifying the managed block inside config.toml.
MANAGED_BLOCK_BEGIN = "# BEGIN cao-kimi-context-restore (managed; do not hand-edit)"
MANAGED_BLOCK_END = "# END cao-kimi-context-restore"

#: Bound on rendered restoration text handed to delivery.
MAX_CONTEXT_CHARS = 6000

#: Hook subprocess timeout, inside Kimi's own hook timeout ceiling.
HOOK_TIMEOUT_SECONDS = 25

#: Per-leg bounds inside one wrapper run (conduct re-read, fence hold,
#: boundary POST). Each leg is small; together they stay inside the
#: hook ceiling above.
CONDUCT_TIMEOUT_SECONDS = 10.0
FENCE_FLOCK_TIMEOUT_SECONDS = 5.0
BOUNDARY_TIMEOUT_SECONDS = 10.0


def restore_hook_toml(*, command: str) -> str:
    """The one managed ``[[hooks]]`` block: PostCompact notify only."""
    quoted = command.replace('"', '\\"')
    return (
        f"{MANAGED_BLOCK_BEGIN}\n"
        f'[[hooks]]\nevent = "{RESTORE_HOOK_EVENT}"\ncommand = "{quoted}"\n'
        f"timeout = {HOOK_TIMEOUT_SECONDS}\n"
        f"{MANAGED_BLOCK_END}\n"
    )


def _strip_managed_block(base_text: str) -> str:
    """Remove a previously installed managed block (teardown/refresh)."""
    if MANAGED_BLOCK_BEGIN not in base_text:
        return base_text
    before, _, rest = base_text.partition(MANAGED_BLOCK_BEGIN)
    _, _, after = rest.partition(MANAGED_BLOCK_END)
    out = before.rstrip("\n")
    if after.strip():
        out += "\n" + after.lstrip("\n")
    elif out:
        out += "\n"
    return out


def compose_managed_config(base_text: str, *, command: str) -> str:
    """Worker config text: user base preserved plus one managed block.

    Textual append, never a TOML round-trip (mirrors the Codex
    adapter): the base keeps every user hook byte-identical, a refresh
    replaces only our own block, and teardown removes exactly it. A
    malformed user base is not repaired here.
    """
    clean = _strip_managed_block(base_text or "")
    block = restore_hook_toml(command=command)
    if clean and not clean.endswith("\n"):
        clean += "\n"
    return clean + block


def teardown_managed_config(config_text: str) -> str:
    """Remove the managed block, preserving everything else byte-identical."""
    return _strip_managed_block(config_text or "")


def _executable(path: str) -> Optional[str]:
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


def resolve_wrapper_executable(
    explicit: Optional[str] = None,
    *,
    search_path: Optional[str] = None,
) -> Optional[str]:
    """Absolute wrapper path, or None (degrade, never refuse a launch)."""
    if explicit:
        return _executable(explicit)
    env = {"PATH": search_path} if search_path is not None else None
    found = shutil.which("cao-kimi-hook-context", path=search_path)
    if found:
        return _executable(found)
    _ = env
    return None


def resolve_conduct_binary(
    explicit: Optional[str] = None,
    *,
    search_path: Optional[str] = None,
) -> Optional[str]:
    """Absolute conduct path, or None (degrade, never refuse a launch)."""
    if explicit:
        return _executable(explicit)
    found = shutil.which("conduct", path=search_path)
    if found:
        return _executable(found)
    return None


def default_fork_base() -> str:
    """The fork loopback base, same rule as the conductor client.

    Mirrors ``conduct.lib.caoapi.default_base_url`` exactly (same env
    names, same default) so the hook and the CLI never disagree about
    where the server lives. No new authority invented.
    """
    host = os.environ.get("CAO_API_HOST", "127.0.0.1")
    port = os.environ.get("CAO_API_PORT", "9889")
    return f"http://{host}:{port}"


def restore_command(
    *,
    wrapper_executable: str,
    terminal_id: str,
    generation: str,
    conduct_binary: str,
    fork_base: Optional[str] = None,
    native_session_id: Optional[str] = None,
) -> str:
    """The baked hook command: every binding from the launch record.

    Nothing is read from hook stdin for identity — stdin's session id
    is cross-checked, never trusted (slice-1 seam). The fork base is
    baked when the launcher knows it; otherwise the wrapper applies
    :func:`default_fork_base` at runtime (worker inherits the env).
    """
    parts = [
        shlex.quote(wrapper_executable),
        "--terminal", shlex.quote(terminal_id),
        "--terminal-generation", shlex.quote(generation),
        "--conduct", shlex.quote(conduct_binary),
    ]
    if fork_base:
        parts += ["--fork-base", shlex.quote(fork_base)]
    if native_session_id:
        parts += ["--native-session-id", shlex.quote(native_session_id)]
    return " ".join(parts)


def describe_installation(
    *,
    terminal_id: str,
    generation: str,
    kimi_home: Optional[str],
    degraded_reason: Optional[str],
) -> Dict[str, Any]:
    """The launch-facts record of what restoration this generation got."""
    return {
        "mechanism": MECHANISM if degraded_reason is None else None,
        "terminal_id": terminal_id,
        "terminal_generation": generation,
        "kimi_home": kimi_home,
        "degraded_reason": degraded_reason,
    }


def build_conduct_argv(
    *,
    conduct_binary: str,
    native_session_id: str,
    terminal_id: Optional[str] = None,
    terminal_generation: Optional[str] = None,
) -> List[str]:
    """The wrapper's one permitted conductor subprocess: read-only."""
    argv = [
        conduct_binary,
        "goal",
        "hook-context",
        "--harness", HARNESS,
        "--native-session-id", native_session_id,
    ]
    if terminal_id:
        argv += ["--terminal", terminal_id]
    if terminal_generation:
        argv += ["--terminal-generation", terminal_generation]
    return argv


def parse_hook_input(raw: bytes) -> Dict[str, Any]:
    """Parse Kimi hook stdin (snake_case JSON); never raises on shape."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _empty_result() -> Dict[str, Any]:
    # PostCompact is fire-and-forget: stdout is ignored provider-side.
    # Emit the empty object so logs stay quiet either way.
    return {}


def render_restoration(answer: Dict[str, Any]) -> Optional[str]:
    """Bounded restoration-labeled context, or None when undeliverable.

    Only an ``ok`` answer with a verified generation fence proceeds;
    everything else restores nothing (stale/ambiguous/terminal exit 0).
    """
    if not isinstance(answer, dict):
        return None
    if answer.get("result_type") != "ok":
        return None
    identity = answer.get("identity") or {}
    if identity.get("generation_fence") != "verified":
        return None
    goal = answer.get("goal") or {}
    objective = goal.get("objective") or goal.get("summary") or ""
    version = (goal.get("goal_version", goal.get("version", "?")))
    requirement_lines = []
    for req in (goal.get("requirements") or [])[:20]:
        if isinstance(req, dict):
            requirement_lines.append(
                f"- {req.get('id', '?')}: {req.get('summary', req.get('kind', ''))}")
    context = (
        "[cao-context-restoration] The worker context compacted; this is "
        "the current CAO goal, not a new assignment. Continue the open "
        "work below under existing continuation policy.\n"
        f"objective (goal version {version}): {objective}"
    )
    if requirement_lines:
        context += "\nrequirements:\n" + "\n".join(requirement_lines)
    next_action = answer.get("next_action") or goal.get("next_action")
    if next_action:
        context += f"\nnext permitted action: {next_action}"
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n[truncated]"
    return context


def _run_conduct(conduct_argv: List[str], *, timeout_seconds: float) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            conduct_argv, capture_output=True, text=True,
            timeout=timeout_seconds, stdin=subprocess.DEVNULL)
    except Exception as exc:
        return {"transport_error": f"conduct spawn failed: {exc}"}
    if proc.returncode != 0:
        return {"transport_error": f"conduct exited {proc.returncode}: {proc.stderr[-500:]}"}
    try:
        answer = json.loads(proc.stdout)
    except Exception as exc:
        return {"transport_error": f"conduct output is not JSON: {exc}"}
    return answer if isinstance(answer, dict) else {"transport_error": "conduct answer is not an object"}


def _valid_flock_path(flock_path: object) -> bool:
    """The fence's flock pointer names the real exclusion file, or nothing."""
    return (isinstance(flock_path, str)
            and os.path.basename(flock_path) == "goal-effect.lock"
            and os.path.isabs(flock_path))


def managed_wire_roots(*, terminal_id: str,
                       generation: Optional[str]) -> List[str]:
    """Where this generation's Kimi session wire files may live.

    The generation-private managed homes first (the launcher's
    ``COMPANION_DIR`` layouts), then the provider default.  Only
    existing directories are returned.  Markers are unique per
    operation id, so a hit in any root is unambiguous; a missing root
    is no evidence, never an error.
    """
    from cli_agent_orchestrator.constants import COMPANION_DIR
    candidates = []
    if terminal_id and generation:
        candidates.append(str(
            Path(COMPANION_DIR) / terminal_id / generation / "kimi-home"))
        candidates.append(str(
            Path(COMPANION_DIR) / "kimi-homes" / f"{terminal_id}-{generation}"))
    try:
        candidates.append(os.path.expanduser("~/.kimi-code"))
    except Exception:  # noqa: BLE001 - home lookup failure scans nothing
        pass
    return [c for c in candidates if c and os.path.isdir(c)]


def _post_boundary(*, fork_base: str, terminal_id: str, fence: Dict[str, Any],
                   context: str, timeout_seconds: float) -> Dict[str, Any]:
    """Ask the fork boundary to admit-or-refuse one reminder delivery.

    Every POST mints its own operation id: a repeated compaction
    rendezvouses with the live pending row as ``already-pending``
    (zero new bytes) instead of re-delivering, and the boundary's
    wire-evidence check completes the row when the marker landed.
    """
    payload = json.dumps({
        "operation_id": str(uuid.uuid4()),
        "occurrence_id": fence.get("occurrence_id"),
        "generation": fence.get("terminal_generation"),
        "native_session_id": fence.get("native_session_id"),
        "goal_version": fence.get("goal_version"),
        "hold_high_water": fence.get("hold_high_water"),
        "flock_path": fence.get("flock_path"),
        "context": context,
    }).encode("utf-8")
    url = (fork_base.rstrip("/") + "/terminals/"
           + urllib.parse.quote(str(terminal_id), safe="")
           + "/context-restore")
    req = urllib.request.Request(url, data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # nosec - loopback only
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"transport_error": f"boundary POST failed: {exc}"}
    return body if isinstance(body, dict) else {"transport_error": "boundary answer is not an object"}


def _fresh_fence_problem(*, fresh: object, held_flock_path: str,
                         terminal_id: Optional[str],
                         terminal_generation: Optional[str],
                         native_session_id: str) -> Optional[str]:
    """Why a fresh projection must not be delivered, or None when current.

    Mirrors the boundary's own structural preconditions — plus the two
    facts only the wrapper can check: the fresh fence still names the
    lock path this hold covers, and it still names this wrapper's
    baked identity. Anything else defers with zero POST.
    """
    if not isinstance(fresh, dict) or "transport_error" in fresh:
        detail = fresh.get("transport_error", fresh) if isinstance(
            fresh, dict) else fresh
        return f"fresh projection unreadable ({detail})"
    fence = fresh.get("delivery_fence")
    if not isinstance(fence, dict):
        return "fresh projection carries no delivery fence"
    if not fence.get("occurrence_id"):
        return "fresh fence names no occurrence"
    if not fence.get("terminal_generation"):
        return "fresh fence names no terminal generation"
    if not _valid_flock_path(fence.get("flock_path")):
        return "fresh fence names no project lock"
    if str(fence["flock_path"]) != held_flock_path:
        return (f"fresh fence moved to a different project lock "
                f"({fence['flock_path']!r}); the held lock no longer "
                f"covers it")
    if (terminal_generation and fence.get("terminal_generation")
            and str(fence["terminal_generation"]) != str(
                terminal_generation)):
        return "fresh fence generation moved under this wrapper"
    if (terminal_id and fence.get("terminal_id")
            and str(fence["terminal_id"]) != str(terminal_id)):
        return "fresh fence names a different terminal"
    if not fence.get("native_session_id"):
        return "fresh fence names no native session"
    if (native_session_id and str(fence["native_session_id"])
            != str(native_session_id)):
        return "native session rotated under this fence"
    return None


def run_wrapper(
    *,
    hook_input: Dict[str, Any],
    terminal_id: str,
    terminal_generation: Optional[str],
    native_session_id: Optional[str],
    conduct_binary: str,
    fork_base: str,
    out,
    err,
) -> int:
    """PostCompact notify path. Always exits 0: notification delivery is
    best-effort and must never block or fail a compaction."""
    stdin_session = hook_input.get("session_id")
    if native_session_id and stdin_session and stdin_session != native_session_id:
        # The stdin session id disagrees with the baked binding: say so
        # on stderr and restore nothing (never trust stdin for identity).
        print(f"cao-kimi-hook-context: stdin session {stdin_session!r} does not "
              f"match baked {native_session_id!r}; restoring nothing", file=err)
        return 0
    conduct_argv = build_conduct_argv(
        conduct_binary=conduct_binary,
        native_session_id=native_session_id or str(stdin_session or ""),
        terminal_id=terminal_id,
        terminal_generation=terminal_generation)
    answer = _run_conduct(conduct_argv,
                          timeout_seconds=CONDUCT_TIMEOUT_SECONDS)
    if not isinstance(answer, dict) or "transport_error" in answer:
        print(f"cao-kimi-hook-context: projection unavailable: "
              f"{answer.get('transport_error', answer)}", file=err)
        return 0
    # F5 (repair): the locator projection above was read with no hold,
    # so NOTHING is ever delivered on it. A hold activation or claim
    # could have committed between that read and now, and the boundary
    # — fork-side — cannot re-read conductor legs. Delivery requires
    # all three under one shared hold of the project lock: the hold
    # acquired, a fresh conductor projection re-read inside it, and
    # that fresh fence matching the held lock path and this wrapper's
    # baked identity. Anything else defers with zero POST: the due
    # clock and any pending row are untouched, and the next PostCompact
    # notification or the sentinel periodic rendezvous retries. A stale
    # projection is never safer than a missing restoration — it can
    # authorize a turn the goal no longer permits.
    locator_fence = answer.get("delivery_fence") or {}
    held_flock = locator_fence.get("flock_path")
    if not _valid_flock_path(held_flock):
        print(f"cao-kimi-hook-context: deferred (locator fence names no "
              f"project lock; the boundary would refuse unfenced delivery; "
              f"retry on the next PostCompact or periodic rendezvous; "
              f"no bytes sent)", file=err)
        return 0
    held_flock = str(held_flock)
    bound_native = native_session_id or str(stdin_session or "")
    from cli_agent_orchestrator.services import (
        goal_effect_flock as _flock)
    try:
        with _flock.hold_path(
                held_flock, shared=True,
                timeout_seconds=FENCE_FLOCK_TIMEOUT_SECONDS):
            fresh = _run_conduct(
                conduct_argv, timeout_seconds=CONDUCT_TIMEOUT_SECONDS)
            problem = _fresh_fence_problem(
                fresh=fresh, held_flock_path=held_flock,
                terminal_id=terminal_id,
                terminal_generation=terminal_generation,
                native_session_id=bound_native)
            if problem is not None:
                print(f"cao-kimi-hook-context: deferred ({problem}; retry "
                      f"on the next PostCompact or periodic rendezvous; "
                      f"no bytes sent)", file=err)
                return 0
            assert isinstance(fresh, dict)
            context = render_restoration(fresh)
            if context is None:
                print(f"cao-kimi-hook-context: deferred (fresh projection "
                      f"renders nothing deliverable; retry on the next "
                      f"PostCompact or periodic rendezvous; no bytes sent)",
                      file=err)
                return 0
            boundary = _post_boundary(
                fork_base=fork_base, terminal_id=terminal_id,
                fence=fresh["delivery_fence"], context=context,
                timeout_seconds=BOUNDARY_TIMEOUT_SECONDS)
    except _flock.GoalEffectBusy as exc:
        print(f"cao-kimi-hook-context: deferred (project lock held: {exc}; "
              f"the next PostCompact notification or the sentinel periodic "
              f"rendezvous retries; the due clock and any pending row are "
              f"untouched; no bytes sent)", file=err)
        return 0
    if "transport_error" in boundary:
        print(f"cao-kimi-hook-context: delivery boundary unreachable: "
              f"{boundary['transport_error']}", file=err)
        return 0
    print(f"cao-kimi-hook-context: boundary answered "
          f"{boundary.get('status', boundary)}", file=err)
    return 0


def parse_wrapper_argv(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Wrapper flags. All binding comes from install-time baking."""
    parser = argparse.ArgumentParser(prog="cao-kimi-hook-context")
    parser.add_argument("--terminal", required=True)
    parser.add_argument("--terminal-generation", default=None)
    parser.add_argument("--native-session-id", default=None)
    parser.add_argument("--conduct", required=True,
                        help="conductor CLI binary (absolute path baked at install)")
    parser.add_argument("--fork-base", default=None,
                        help="fork loopback base URL (baked at install; else "
                             "CAO_API_HOST/CAO_API_PORT at runtime)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Hook entrypoint: read stdin, notify, exit 0 always."""
    args = parse_wrapper_argv(argv)
    raw = sys.stdin.buffer.read()
    return run_wrapper(
        hook_input=parse_hook_input(raw),
        terminal_id=args.terminal,
        terminal_generation=args.terminal_generation,
        native_session_id=args.native_session_id,
        conduct_binary=args.conduct,
        fork_base=args.fork_base or default_fork_base(),
        out=sys.stdout, err=sys.stderr)


def _provider_home(base_environment: Dict[str, str]) -> Path:
    home = base_environment.get("KIMI_CODE_HOME") or os.path.expanduser("~/.kimi-code")
    return Path(home)


# --- Delivery boundary (KIND_REMIND effect path) ---------------------------
#
# Lock order, fixed: project goal-effect flock (shared, named by the
# conductor fence) -> session_effect_admission (shared) ->
# provider_byte_admission -> pane lease -> first provider byte. Writers
# take the project flock exclusively (conductor semantic writers) or
# the session write claim exclusively (fork wait/occurrence writers);
# nothing nested here takes another lock, so no upgrade and no ABBA.

#: Bounded provider write, mirroring the control-input deadline.
WRITE_DEADLINE_SECONDS = 45.0

#: Delivery outcomes. ``deferred`` is the flip case (turn state moved
#: between admission and first byte): back off and re-rendezvous later,
#: never a blind retry, never a clock discard.
OUTCOME_POSTED = "posted"
OUTCOME_PENDING = "pending"
OUTCOME_REFUSED = "refused"
OUTCOME_DEFERRED = "deferred"
OUTCOME_UNKNOWN = "unknown"
OUTCOME_COMPLETED = "completed"


def _fork_wait_cover(*, session_name: str, terminal_id: str,
                     generation: Optional[str]) -> Optional[Tuple[str, str]]:
    """Re-read fork wait legs: cover/refuse/defer per the Sol-2 table.

    Returns None when no wait constrains this terminal generation, else
    ``(reason, detail)``. Pending counts as covering (undecided defers);
    expiry-intent/wake-pending defers competing delivery; invalid defers
    for §7 recovery (terminal state alone never grants permission);
    terminal rows release. Any unreadable leg defers — never runnable.
    """
    from cli_agent_orchestrator.services import registered_waits
    try:
        listing = registered_waits.list_waits(
            session_name=session_name, terminal_id=terminal_id)
    except Exception as exc:
        return ("wait_unreadable",
                f"fork wait list unreadable, deferring: {exc}")
    rows = listing.get("waits", listing) if isinstance(listing, dict) else listing
    if not isinstance(rows, list):
        return ("wait_unreadable", "fork wait list unreadable, deferring")
    from cli_agent_orchestrator.services import kimi_native_control as adapter
    states = set()
    for w in rows:
        if not isinstance(w, dict):
            return ("wait_unreadable", "fork wait row unreadable, deferring")
        owner = w.get("owner") or {}
        owner_term = owner.get("terminal_id", owner.get("terminal"))
        if owner_term is not None and str(owner_term) != str(terminal_id):
            continue
        owner_gen = owner.get("generation", owner.get("terminal_generation"))
        if owner_gen is not None and generation is not None and str(
                owner_gen) != str(generation):
            continue
        states.add(str(w.get("state", "?")))
    if not states:
        return None
    terminal = {"resolved", "cancelled", "invalid"}
    if states <= terminal:
        if "invalid" in states:
            # Owner loss needs §7 recovery; the terminal row alone does
            # not release the constraint. Ordinary release (a later
            # acknowledged/resolved row) is observed here, not inferred.
            return ("wait_recovery",
                    "invalid wait row without ordinary release; §7 recovery "
                    "owns this worker until it re-registers or resolves")
        return None
    deferring = {"expiry-intent", "expiry-wake-pending", "interrupted-by-stop"}
    if states <= (terminal | deferring):
        return ("wait_settling",
                f"expiry machinery owns the worker (states {sorted(states)}); "
                "deferring competing delivery until it settles")
    return ("wait_cover",
            f"covering wait states {sorted(states)} constrain this worker")


def _fork_occurrence_current(*, occurrence_id: str, terminal_id: str,
                             generation: Optional[str]) -> Optional[Tuple[str, str]]:
    """Re-read fork occurrence currency (beyond terminal/generation tuples)."""
    from cli_agent_orchestrator.services import task_occurrence
    try:
        occ = task_occurrence.get_occurrence(occurrence_id)
    except Exception as exc:
        return ("occurrence_unreadable",
                f"fork occurrence unreadable, deferring: {exc}")
    if not isinstance(occ, dict) or occ.get("state") != "open":
        return ("occurrence_closed",
                f"occurrence {occurrence_id!r} is not open; discarding")
    bound = occ.get("bound_terminal_id", occ.get("terminal_id"))
    bound_gen = occ.get("bound_generation", occ.get("generation"))
    if bound != terminal_id or (bound_gen is not None
                                and generation is not None
                                and bound_gen != generation):
        return ("occurrence_moved",
                f"occurrence {occurrence_id!r} is no longer bound to terminal "
                f"{terminal_id!r} generation {generation!r}; discarding")
    return None


def _fork_lifecycle_working(*, session_name: str) -> Optional[Tuple[str, str]]:
    """Refuse unless the session lifecycle is working (Pause-safe)."""
    from cli_agent_orchestrator.services import session_lifecycle as sl
    try:
        lifecycle = sl.describe(session_name)
    except Exception as exc:
        return ("lifecycle_unreadable",
                f"session lifecycle unreadable, deferring: {exc}")
    state = lifecycle.get("lifecycle") if isinstance(lifecycle, dict) else None
    if state != "working":
        return ("lifecycle_not_working",
                f"session {session_name!r} lifecycle is {state!r}, not "
                "working; no reminder-driven continuation")
    return None


def _observe_branch(*, pane_id: str, terminal_id: str, session_name: str,
                    window_name: str) -> Tuple[str, Optional[str]]:
    """Decide submit vs steer from the provider's own detector.

    Returns ``(turn_state, detail)`` with turn_state in {"idle","active"}.
    IDLE and COMPLETED both mean an input-ready composer; every other
    observed state — and any observation failure — selects the steer
    branch only when a proven chord exists, else the adapter refuses
    (missing capability, never a blind submit into a live turn).
    """
    from cli_agent_orchestrator.services import managed_launch_v2
    from cli_agent_orchestrator.models.terminal import TerminalStatus
    try:
        status = managed_launch_v2._observe_turn_state(
            "kimi_cli", pane_id=pane_id, terminal_id=terminal_id,
            session_name=session_name, window_name=window_name)
    except Exception as exc:
        return "active", f"turn state unobservable ({exc}); steer-or-refuse"
    if status in (TerminalStatus.IDLE, TerminalStatus.COMPLETED):
        return "idle", f"provider reports {status.value}"
    return "active", f"provider reports {status.value if hasattr(status, 'value') else status}"


def submit_context_reminder(
    *,
    terminal_id: str,
    operation_id: str,
    occurrence_id: str,
    context: str,
    fence: Dict[str, Any],
    lease_timeout: float = 0.0,
) -> Dict[str, Any]:
    """Admit-or-refuse one context reminder and deliver it exactly once.

    Fence fields (all re-verified, none trusted): ``generation``,
    ``native_session_id``, ``goal_version``, ``hold_high_water``,
    ``flock_path`` (absolute ``goal-effect.lock`` path — fail closed
    when absent). The marker is the operation id: unique per delivery,
    echoed by provider evidence on acceptance.

    Returns ``{"status", "detail", "record"?}`` with status in
    posted/pending/refused/deferred/unknown/completed.
    """
    from cli_agent_orchestrator.services import cohort_journal
    from cli_agent_orchestrator.services import control_input_service
    from cli_agent_orchestrator.services import goal_effect_flock
    from cli_agent_orchestrator.services import kimi_native_control as adapter
    from cli_agent_orchestrator.services.pane_input_arbiter import (
        PaneBusyError, pane_input_lease)
    from cli_agent_orchestrator.utils.terminal import managed_window_name

    if not operation_id or not occurrence_id or not terminal_id:
        return {"status": "refused",
                "detail": "operation_id, occurrence_id, and terminal_id are required"}
    if not (context or "").strip():
        return {"status": "refused", "detail": "empty reminder context; nothing to send"}
    generation = fence.get("generation")
    flock_path = fence.get("flock_path")
    if not generation or not flock_path:
        return {"status": "refused",
                "detail": "fence must carry generation and flock_path; refusing unfenced delivery"}
    if not _valid_flock_path(flock_path):
        return {"status": "refused",
                "detail": "flock_path must be an absolute goal-effect.lock path"}

    resolved = control_input_service.resolve_control_identity(terminal_id)
    if resolved is None:
        return {"status": "refused",
                "detail": f"no terminal {terminal_id!r} is known to this server"}
    if resolved.provider != "kimi_cli":
        return {"status": "refused",
                "detail": f"terminal {terminal_id!r} is provider {resolved.provider!r}, not kimi_cli"}
    if resolved.terminal_generation != generation:
        return {"status": "refused",
                "detail": f"terminal generation is {resolved.terminal_generation!r}, fence says "
                          f"{generation!r}; the generation moved, discarding"}
    fence_native = fence.get("native_session_id")
    if not fence_native:
        return {"status": "refused",
                "detail": "fence must carry native_session_id; refusing unfenced delivery"}
    if resolved.native_session_id != fence_native:
        return {"status": "refused",
                "detail": "native session rotated under this fence; discarding"}
    if resolved.pane_id is None or resolved.pane_dead:
        return {"status": "refused", "detail": "pane is gone or dead; nothing was typed"}
    if resolved.window_id is None or resolved.pane_pid is None:
        return {"status": "refused",
                "detail": "the pane's window and root process could not both be "
                          "observed; nothing was typed"}
    if resolved.native_session_id is None:
        return {"status": "refused",
                "detail": "no native session is bound; nothing was typed"}
    if resolved.session_name is None:
        return {"status": "deferred",
                "detail": "session name unresolvable; deferring"}

    def _journal_fence_refusal(reason: str, detail: str) -> Dict[str, Any]:
        """Journal a typed zero-byte refusal for a failed fence leg.

        The canonical writer path for ``REFUSED_WAIT_COVER`` /
        ``REFUSED_LIFECYCLE`` / ``REFUSED_OCCURRENCE``: the refusal is
        recorded against this dispatch's operation id instead of
        answered off-record.  Journaling itself is best-effort — a
        store failure still returns the refusal, never an exception.
        """
        from datetime import datetime, timezone
        record: Dict[str, Any] = {}
        try:
            record = adapter.refuse_reminder(
                operation_id=operation_id,
                native_session_id=resolved.native_session_id,
                terminal_id=terminal_id,
                generation=resolved.terminal_generation,
                execution_mode=resolved.execution_mode,
                occurrence_id=occurrence_id,
                text=context,
                marker=operation_id,
                observation=adapter.turn_observation(
                    active_turn_id=None,
                    observed_at=datetime.now(timezone.utc).isoformat(),
                    observer="kimi_context_restore",
                ),
                reason=reason,
                detail=detail)
        except adapter.NativeControlConflict:
            try:
                record = adapter.get(operation_id) or {}
            except Exception:  # noqa: BLE001 - keep the refusal answer
                record = {}
        except Exception:  # noqa: BLE001 - keep the refusal answer
            record = {}
        return {"status": "refused", "detail": detail, "record": record}

    import time as _time
    deadline = _time.monotonic() + WRITE_DEADLINE_SECONDS
    try:
        with goal_effect_flock.hold_path(flock_path, shared=True,
                                         timeout_seconds=10.0):
            with cohort_journal.session_effect_admission(resolved.session_name):
                problem = _fork_lifecycle_working(session_name=resolved.session_name)
                if problem is not None:
                    if problem[0] == "lifecycle_not_working":
                        return _journal_fence_refusal(
                            adapter.REFUSED_LIFECYCLE, problem[1])
                    return {"status": "deferred", "detail": problem[1]}
                problem = _fork_occurrence_current(
                    occurrence_id=occurrence_id, terminal_id=terminal_id,
                    generation=resolved.terminal_generation)
                if problem is not None:
                    reason = problem[0]
                    if reason in ("occurrence_closed", "occurrence_moved"):
                        return _journal_fence_refusal(
                            adapter.REFUSED_OCCURRENCE, problem[1])
                    return {"status": "deferred", "detail": problem[1]}
                problem = _fork_wait_cover(
                    session_name=resolved.session_name, terminal_id=terminal_id,
                    generation=resolved.terminal_generation)
                if problem is not None:
                    reason = problem[0]
                    if reason in ("wait_cover", "wait_recovery"):
                        return _journal_fence_refusal(
                            adapter.REFUSED_WAIT_COVER, problem[1])
                    return {"status": "deferred", "detail": problem[1]}
                turn_state, turn_detail = _observe_branch(
                    pane_id=resolved.pane_id, terminal_id=terminal_id,
                    session_name=resolved.session_name,
                    window_name=managed_window_name(
                        terminal_id, str(resolved.terminal_generation)))
                chord = None
                if turn_state == "active":
                    proven = adapter.steer_chords(resolved.provider_version)
                    if not proven:
                        return {"status": "deferred",
                                "detail": f"turn is active but no steer chord is proven for build "
                                          f"{resolved.provider_version!r} (missing capability, not "
                                          "failure); backing off to idle-submit or degraded routes"}
                    chord = sorted(proven)[0]
                from cli_agent_orchestrator.services.control_input_journal import (
                    ControlInputBinding)
                binding = ControlInputBinding(
                    request_id=operation_id,
                    terminal_id=terminal_id,
                    pane_id=resolved.pane_id,
                    window_id=resolved.window_id,
                    pane_pid=resolved.pane_pid,
                    request_sha256="",
                    generation=resolved.terminal_generation,
                    server_socket_path=resolved.bound_server_socket_path,
                )
                client = control_input_service._tmux_client()
                with control_input_service.provider_byte_admission(
                        resolved, terminal_id, binding.generation):
                    with pane_input_lease(
                            resolved.pane_id,
                            holder=f"context-reminder:{operation_id}",
                            timeout=lease_timeout):
                        live = client.pane_control_identity(
                            pane_id=binding.pane_id, deadline_monotonic=deadline)
                        if live is None or live.dead:
                            return {"status": "refused",
                                    "detail": "pane is gone or dead as of the write lease"}
                        if (live.window_id != binding.window_id
                                or live.pane_pid != binding.pane_pid):
                            return {"status": "refused",
                                    "detail": "pane identity moved under the lease; discarding"}
                        from datetime import datetime, timezone
                        observation = adapter.turn_observation(
                            active_turn_id=None,
                            observed_at=datetime.now(timezone.utc).isoformat(),
                            observer="kimi_context_restore",
                        )
                        transport = control_input_service._NativeComposerTransport(
                            client, binding.pane_id,
                            binding.server_socket_path,
                            deadline_monotonic=deadline)

                        def pre_write():
                            fresh, _ = _observe_branch(
                                pane_id=binding.pane_id, terminal_id=terminal_id,
                                session_name=resolved.session_name,
                                window_name=managed_window_name(
                                    terminal_id, str(resolved.terminal_generation)))
                            if fresh != turn_state:
                                return (adapter.REFUSED_TURN_MISMATCH,
                                        f"turn state flipped {turn_state!r}->{fresh!r} between "
                                        "admission and first byte; deferring to re-rendezvous")
                            return None

                        from cli_agent_orchestrator.services.canonical_json import (
                            canonical_sha256 as _content_sha)
                        incoming_sha = _content_sha({"context": context})
                        roots = managed_wire_roots(
                            terminal_id=terminal_id,
                            generation=resolved.terminal_generation)

                        def _attempt(op_id, supersede):
                            """One remind() attempt with submit's error mapping.

                            Returns (True, record) or (False, terminal
                            response). Terminal responses carry
                            new_bytes False: nothing was typed.
                            """
                            try:
                                return (True, adapter.remind(
                                    operation_id=op_id,
                                    native_session_id=resolved.native_session_id,
                                    terminal_id=terminal_id,
                                    generation=resolved.terminal_generation,
                                    execution_mode=resolved.execution_mode,
                                    occurrence_id=occurrence_id,
                                    text=context,
                                    marker=op_id,
                                    fence_snapshot={
                                        "goal_version": fence.get("goal_version"),
                                        "hold_high_water": fence.get("hold_high_water"),
                                    },
                                    observation=observation,
                                    transport=transport,
                                    turn_state=turn_state,
                                    provider_version=resolved.provider_version,
                                    steer_chord=chord,
                                    pre_write=pre_write,
                                    deadline_monotonic=deadline,
                                    supersede_ids=supersede))
                            except adapter.NativeControlConflict as exc:
                                return (False, {"status": "refused",
                                                "detail": str(exc),
                                                "new_bytes": False})
                            except adapter.NativeControlInvalid as exc:
                                return (False, {"status": "refused",
                                                "detail": str(exc),
                                                "new_bytes": False})
                            except Exception as exc:
                                try:
                                    adapter.mark_ambiguous(
                                        operation_id=op_id,
                                        reason=f"reminder submit raised mid-effect: {exc}")
                                except Exception:
                                    pass
                                return (False, {"status": "unknown",
                                                "detail": f"submit raised before its outcome was "
                                                          f"journaled: {exc}; reconcile by exact "
                                                          f"operation id",
                                                "new_bytes": False})

                        def _settle(record):
                            """Wire-settle one adopted row; never types bytes."""
                            try:
                                settlement = adapter.reconcile_reminder_from_wire(
                                    operation_id=record.get("operation_id") or operation_id,
                                    marker=record.get("operation_id") or operation_id,
                                    session_home=roots)
                                if settlement.get("reconciled") and isinstance(
                                        settlement.get("record"), dict):
                                    record = settlement["record"]
                            except Exception:  # noqa: BLE001 - reconcile never breaks adopt
                                pass
                            return record

                        def _deliver_new(supersede):
                            """Deliver this POST's context as a new row.

                            Only the posted outcome carries new bytes (and
                            a clock reset downstream); every other outcome
                            carries new_bytes False.
                            """
                            ok, result = _attempt(operation_id, supersede)
                            if not ok:
                                return result
                            out = result.get("reminder_outcome")
                            if out == "posted":
                                return {"status": "posted", "detail": turn_detail,
                                        "record": result, "new_bytes": True}
                            if out == "refused":
                                reason = result.get("refusal_reason", "")
                                if reason == adapter.REFUSED_TURN_MISMATCH:
                                    return {"status": "deferred", "new_bytes": False,
                                            "detail": "turn flipped at the last safe point",
                                            "record": result}
                                return {"status": "refused", "new_bytes": False,
                                        "detail": reason or "refused",
                                        "record": result}
                            # Raced adoption (or a mid-effect ambiguity):
                            # settle read-only, never new bytes.
                            record = _settle(result)
                            if record.get("state") == "completed":
                                return {"status": "completed", "new_bytes": False,
                                        "detail": "rendezvous settled the adopted row",
                                        "record": record}
                            return {"status": "pending", "new_bytes": False,
                                    "detail": f"reminder {record.get('reminder_outcome')}; "
                                              f"backing off",
                                    "record": record}

                        def _marker_of(row):
                            transport = row.get("transport") or {}
                            return transport.get("marker") or row.get("operation_id")

                        def _sha_of(row):
                            transport = row.get("transport") or {}
                            return transport.get("context_sha256")

                        rows = adapter.unresolved_reminders_for(
                            terminal_id=terminal_id,
                            generation=resolved.terminal_generation)
                        live = [r for r in rows
                                if r.get("state") in ("posted", "accepted")]
                        owned = [r for r in rows
                                 if r.get("state") in ("intended", "writing")]
                        ambiguous = [r for r in rows
                                     if r.get("state") == "ambiguous"]
                        if owned:
                            # An effect may still be typing: rendezvous
                            # later, never compound now.
                            return {"status": "deferred", "new_bytes": False,
                                    "detail": f"an effect may still own reminder "
                                              f"{owned[0].get('operation_id')}; will not compound",
                                    "record": owned[0]}
                        same = [r for r in live + ambiguous
                                if _sha_of(r) is not None
                                and _sha_of(r) == incoming_sha]
                        if same:
                            # Re-POST of the same compaction: rendezvous
                            # with that row and settle it from the wire.
                            # Zero new bytes by construction.
                            elected = same[-1]
                            ok, result = _attempt(elected.get("operation_id"),
                                                  frozenset())
                            if not ok:
                                return result
                            record = _settle(result)
                            if record.get("state") == "completed":
                                return {"status": "completed", "new_bytes": False,
                                        "detail": "wire evidence shows the marker reached "
                                                  "the model",
                                        "record": record}
                            return {"status": "pending", "new_bytes": False,
                                    "detail": f"reminder {record.get('reminder_outcome')}; "
                                              f"backing off",
                                    "record": record}
                        if len(live) > 1:
                            return {"status": "deferred", "new_bytes": False,
                                    "detail": "multiple live reminder rows; refusing to guess "
                                              "which compaction this continues",
                                    "record": live[-1]}
                        if live:
                            # New compaction while one row is live: settle
                            # the old receipt, then deliver the new
                            # context — never swallow it by adopting.
                            target = live[0]
                            settled = _settle(target)
                            if settled is not target:
                                target = settled
                            if target.get("state") in ("completed", "refused"):
                                return _deliver_new(
                                    {r.get("operation_id") for r in ambiguous})
                            try:
                                composition = adapter.reconcile_reminder_composer(
                                    operation_id=target.get("operation_id"),
                                    marker=_marker_of(target),
                                    pane_id=resolved.pane_id,
                                    session_home=roots)
                            except Exception:
                                composition = {"composer_holds_marker": None,
                                             "reason": "composer check raised",
                                             "record": target}
                            holds = composition.get("composer_holds_marker")
                            if holds is not False:
                                return {"status": "deferred", "new_bytes": False,
                                        "detail": f"prior reminder "
                                                  f"{target.get('operation_id')} unproven "
                                                  f"({composition.get('reason')}); will not "
                                                  f"compound bytes; retry on a later compaction",
                                        "record": composition.get("record") or target}
                            return _deliver_new(
                                {r.get("operation_id") for r in ambiguous}
                                | {target.get("operation_id")})
                        if ambiguous:
                            # No live rows, only superseded history: deliver
                            # only when no history row still holds partial
                            # bytes in the composer.
                            for prior in ambiguous:
                                try:
                                    composition = adapter.reconcile_reminder_composer(
                                        operation_id=prior.get("operation_id"),
                                        marker=_marker_of(prior),
                                        pane_id=resolved.pane_id,
                                        session_home=roots)
                                except Exception:
                                    composition = {"composer_holds_marker": None,
                                                 "reason": "composer check raised",
                                                 "record": prior}
                                if composition.get("composer_holds_marker") is not False:
                                    return {"status": "deferred", "new_bytes": False,
                                            "detail": f"superseded reminder "
                                                      f"{prior.get('operation_id')} still constrains "
                                                      f"delivery ({composition.get('reason')}); will "
                                                      f"not compound bytes",
                                            "record": composition.get("record") or prior}
                            return _deliver_new(
                                {r.get("operation_id") for r in ambiguous})
                        return _deliver_new(frozenset())
    except goal_effect_flock.GoalEffectBusy as exc:
        return {"status": "deferred", "detail": str(exc)}
    except cohort_journal.SessionEffectRefused as exc:
        return {"status": "refused", "detail": str(exc)}
    except PaneBusyError as exc:
        return {"status": "deferred",
                "detail": f"another writer holds the pane: {exc}"}
    except Exception as exc:
        return {"status": "unknown", "detail": f"delivery failed before journaling: {exc}"}
    # All delivery paths above return directly; nothing falls through.


def compose_kimi_home(
    *,
    companion_dir: str,
    terminal_id: str,
    generation: str,
    provider_home: str,
    base_config_text: str,
    command: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Generation-private KIMI_CODE_HOME with the managed PostCompact entry.

    Returns ``(kimi_home, degraded_reason)``. Never raises for
    restoration causes: any failure degrades to ``(None, reason)`` so a
    launch is never refused over a restoration hook.
    """
    try:
        home = Path(companion_dir) / "kimi-homes" / f"{terminal_id}-{generation}"
        home.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(home, 0o700)
        except OSError:
            pass
        composed = compose_managed_config(base_config_text or "", command=command)
        (home / "config.toml").write_text(composed, encoding="utf-8")
        return str(home), None
    except Exception as exc:
        return None, f"kimi home composition failed: {exc}; launching without restoration hooks"


def prepare_kimi_restoration(
    *,
    record: Dict[str, Any],
    base_environment: Dict[str, str],
    companion_dir: str,
    fork_base: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Give a native Kimi generation its PostCompact notify binding.

    Mirrors ``prepare_codex_restoration``: returns ``(environment,
    installation)`` with ``KIMI_CODE_HOME`` pointed at the
    generation-private home. Never raises for restoration causes.
    ``fork_base`` is baked when the launcher knows it; otherwise the
    wrapper applies the CAO_API_HOST/PORT rule at runtime.
    """
    terminal_id = record["terminal_id"]
    generation = record["generation"]
    worker_path = base_environment.get("PATH")
    wrapper = resolve_wrapper_executable(search_path=worker_path)
    conduct = resolve_conduct_binary(search_path=worker_path)
    if wrapper is None or conduct is None:
        missing = ", ".join(
            name for name, value in (("wrapper", wrapper), ("conduct", conduct))
            if value is None)
        return dict(base_environment), describe_installation(
            terminal_id=terminal_id, generation=generation, kimi_home=None,
            degraded_reason=(f"context restoration not installed ({missing} "
                             "unresolvable); launching without restoration hooks"))
    native_session = record.get("native_session_id")
    command = restore_command(
        wrapper_executable=wrapper, terminal_id=terminal_id,
        generation=generation, conduct_binary=conduct, fork_base=fork_base,
        native_session_id=native_session)
    provider_home = _provider_home(base_environment)
    try:
        base_config_text = (provider_home / "config.toml").read_text(encoding="utf-8")
    except OSError:
        base_config_text = ""
    home, degraded = compose_kimi_home(
        companion_dir=companion_dir, terminal_id=terminal_id,
        generation=generation, provider_home=str(provider_home),
        base_config_text=base_config_text, command=command)
    environment = dict(base_environment)
    if home is not None:
        environment["KIMI_CODE_HOME"] = home
    return environment, describe_installation(
        terminal_id=terminal_id, generation=generation, kimi_home=home,
        degraded_reason=degraded)
