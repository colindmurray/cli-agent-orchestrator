"""Muse pre-model passive CAO goal restoration (cond-0845 slice, Muse lane).

A managed Muse worker loses its CAO assignment when its context compacts:
the native session survives, but the model-facing goal capsule it was
launched with is compacted away. This module restores it passively through
Muse's own ``PreLLMCall`` hook, which fires before every model call and
whose ``additionalContext`` output is installed as a developer-role
request block — so every call re-renders the *current* projection instead
of replaying a baked copy.

Proven contract (fixture-observed on installed
``/Users/colin/.local/bin/muse-bin-1.0.3-R2198.1``, non-billable echo
provider; evidence under
``/private/var/folders/1p/q35nt92x2_s69k1bvg1dcw2m0000gn/T/muserev2/``
and ``/private/tmp/muserev/stdin.log``):

* Registration shape ``{"hooks": {"PreLLMCall": [{"matcher": "*",
  "hooks": [{"type": "command", "command": "..."}]}]}}`` in the workspace
  ``.muse/hooks.json`` fires (stdin observed verbatim). Hook keys are
  derived (``<file>:pre_llm_call:def-<hash>``), so isolation comes from
  distinct baked commands per terminal, not from key names.
* stdin is snake_case and carries the native session identity
  (``session_id``), plus ``turn_id``, ``cwd``, ``request_id``,
  ``messages``/``tools`` with counts. No generation or terminal id —
  the managed terminal is baked into the hook command at install time
  from the launch path, never read from stdin.
* stdout ``{"hookSpecificOutput": {"hookEventName": "PreLLMCall",
  "additionalContext": "..."}}`` completes with effect ``context`` and
  installs a developer-role block (observed
  ``context_block_updated``). Omitting ``hookEventName`` fails
  explicitly. Bare ``{}`` is the no-op.
* Trust: project hooks load under the fork's existing ``--yolo``
  run-flag trust (the provider launches ``muse --yolo``). If a launch
  ever drops that flag without ``--trust-workspace``, project hooks
  silently do not load — a concrete limitation recorded here, not a
  second trust mechanism: this adapter never weakens global trust.

What this adapter does NOT do, by construction (same posture as the
Claude/AGY siblings):

* It never submits a turn, calls steer/send, releases a hold, or bypasses
  a resume-paused worker. The wrapper's only subprocess is ``conduct goal
  hook-context`` — the read-only slice-1 projection (conductor PR #349,
  ``1f1e415f``) — which starts zero turns by contract. The argv is pinned
  in :func:`build_conduct_argv` and asserted in tests.
* It never invents a goal. Anything but ``ok`` restores nothing; a missing
  assignment renders a short truthful note naming the next model call as
  the recovery event, while stale/ambiguous/terminal callbacks render an
  empty object so a former assignment is never re-injected.
* No timers, no coalescing, no event steer, no physical resume behaviour.

Renderer reuse: goal rendering, the output bound, the projection schema,
and terminal-state discard all come from
:mod:`cli_agent_orchestrator.services.claude_context_restore` — one
renderer, not a third. Muse-specific are only the hook I/O
(``session_id`` in, ``hookSpecificOutput.additionalContext`` out), the
harness-claimed argv pin, the hooks-file composition, and the
unavailable-note wording (next model call, not next SessionStart).

Duplicate-binder defence (AGY R3 lesson, applied here from the start):
the wrapper queries the projection globally — no ``--terminal`` hint,
which would narrow the reader first and let two live binders each
resolve ``ok`` — and injects only when the single ``ok`` answer's
live-matched ``identity.terminal_id`` exactly equals its baked
``--terminal``. Two binders answer ``ambiguous`` (empty both sides); a
unique resolution for another terminal is discarded (stale keys, shared
workspaces). One query, one current answer, no second authority.

Named gaps this slice does not close:

* Model-entry proof: effect-accepted plus a structurally valid
  ``additionalContext`` payload is not delivery proof. The marker-echo
  validation that the block actually reaches the model belongs to a
  later lane, as does any live-model launch: no live provider is
  launched here.
* Generation fence: this path carries no managed generation, so the
  wrapper sends no generation and the projection fence stays
  ``unverified`` by documented omission rather than by an invented
  value.
* TUI firing: the fixture proves the ``exec`` shape; the interactive
  TUI shares the same binary and hook engine but its firing was not
  observed here (no tmux route in this sandbox). Later validation must
  confirm it rather than assume it.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import json
import logging
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from cli_agent_orchestrator.services import claude_context_restore as claude

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms only
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

#: The claiming harness as the conductor's canonical provider id
#: (``ProviderType.MUSE_CLI``). Checked against the live provider
#: fork-side; never trusted blind (slice-1 seam).
HARNESS = "muse_cli"

#: The hook event that restores context. PreLLMCall, and only
#: PreLLMCall: it is the one event whose output installs request
#: context ahead of the model call, on every call.
RESTORE_HOOK_EVENT = "PreLLMCall"

#: Installed console entry for the wrapper (see ``[project.scripts]``).
WRAPPER_ENTRY_POINT = "cao-muse-hook-context"

#: The selected restoration mechanism, recorded in the launch-facts
#: record so diagnostics can show what a terminal installed (§10.1).
MECHANISM = "muse-PreLLMCall:additionalContext"

#: Prefix for matching this lane's baked commands inside a shared
#: hooks file. Muse derives hook keys itself, so entries are
#: recognised by their baked command, never by key name.
COMMAND_MARKER = "cao-muse-hook-context"

#: The workspace hooks file this adapter owns entries in.
HOOKS_DIRNAME = ".muse"
HOOKS_FILENAME = "hooks.json"

#: Label marking every injected string as restoration.
RESTORATION_LABEL = "CAO goal restoration"

#: Rendered when the projection holds no projectable goal yet. Same
#: truthfulness contract as the siblings, reworded: recovery arrives on
#: the next model call, which re-fires this hook.
UNAVAILABLE_PREFIX = "CAO context restoration unavailable"


def _terminal_flag(terminal_id: str) -> str:
    """The baked self-identity fragment install/uninstall match on."""
    return "--terminal " + shlex.quote(terminal_id)


def hooks_file(workspace: Path) -> Path:
    """Where the managed hook entries live for this workspace."""
    return Path(workspace) / HOOKS_DIRNAME / HOOKS_FILENAME


def restore_hook_handler(*, command: str) -> Dict[str, Any]:
    """The additive ``PreLLMCall`` handler installing restoration."""
    return {"type": "command", "command": command}


def restore_hook_entry(*, command: str) -> Dict[str, Any]:
    """One ``PreLLMCall`` matcher entry. Minimal proven shape: the
    fixture fires with exactly this; no timeout knob is baked because
    none is proven on the installed build."""
    return {"matcher": "*", "hooks": [restore_hook_handler(command=command)]}


#: How long install/uninstall waits for the hooks-file lock before
#: degrading (seconds). The lock is held across one read-modify-write
#: (milliseconds); waiting longer would stall a launch behind a wedged
#: holder, so contention degrades instead. Tests shorten it.
LOCK_TIMEOUT_SECONDS = 5.0


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[bool]:
    """Hold an exclusive advisory lock across one read-modify-write.

    Yields True holding the sidecar lock, False when the lock is
    contended or unusable (the caller degrades instead of racing).
    Without ``fcntl`` (non-POSIX) there is no lock to hold: yields True
    and the atomic replace below is the only protection.
    """
    if fcntl is None:
        yield True
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(_lock_path(path), "a+b")
    except OSError:
        yield False
        return
    try:
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    yield False
                    return
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(0.05)
        yield True
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON so readers see only the old or the new revision.

    Temp file in the same directory plus ``os.replace``: a crashed
    writer leaves a stray ``.tmp`` file, never a torn ``hooks.json``.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, indent=2) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_hooks_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Read ``hooks.json``; ``(None, reason)`` refuses to clobber it.

    Missing file is not an error — ``({}, None)`` means "nothing to
    preserve". A present-but-unreadable file degrades the whole install
    instead of replacing user config with a guess.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, None
    except OSError as exc:
        return None, f"could not read {path} ({exc}); leaving user hooks untouched"
    try:
        data = json.loads(raw)
    except ValueError:
        return None, f"{path} is not valid JSON; leaving user hooks untouched"
    if not isinstance(data, dict):
        return None, f"{path} root is not an object; leaving user hooks untouched"
    return data, None


def _is_own_entry(entry: Any, *, terminal_id: str) -> bool:
    """Whether a ``PreLLMCall`` entry is this terminal's baked hook."""
    if not isinstance(entry, dict):
        return False
    handlers = entry.get("hooks")
    if not isinstance(handlers, list):
        return False
    marker = _terminal_flag(terminal_id)
    for handler in handlers:
        if not isinstance(handler, dict):
            continue
        command = handler.get("command")
        if isinstance(command, str) and COMMAND_MARKER in command and marker in command:
            return True
    return False


def _composable_error(hooks_config: Any) -> Optional[str]:
    """Why ``hooks_config`` cannot take our entry, or None when it can.

    Only the paths this adapter writes are inspected: a present
    ``"hooks"`` key must be an object and a present
    ``"hooks.PreLLMCall"`` must be a list. Anything else — absent keys,
    other events in any shape — is none of this adapter's business and
    passes through untouched.
    """
    if not isinstance(hooks_config, dict):
        return "hooks config root is not an object"
    entries = hooks_config.get("hooks")
    if entries is not None and not isinstance(entries, dict):
        return '"hooks" key is not an object'
    if isinstance(entries, dict):
        pre = entries.get(RESTORE_HOOK_EVENT)
        if pre is not None and not isinstance(pre, list):
            return f'"hooks.{RESTORE_HOOK_EVENT}" is not a list'
    return None


def with_context_restore(
    hooks_config: Dict[str, Any], *, terminal_id: str, command: str
) -> Dict[str, Any]:
    """Return ``hooks_config`` plus this terminal's restore entry.

    The input is not mutated. Every pre-existing key — user hooks for
    any event — is carried over untouched; this terminal's own older
    entries are replaced (reinstall is idempotent), everything else is
    preserved byte-for-value. A config whose ``"hooks"`` or
    ``"hooks.PreLLMCall"`` shape cannot take our entry raises
    ``ValueError`` with the reason instead of replacing user config —
    callers that cannot raise (install) degrade with the same text.
    """
    problem = _composable_error(hooks_config)
    if problem is not None:
        raise ValueError(f"{problem}; leaving user hooks untouched")
    composed = copy.deepcopy(hooks_config)
    entries = composed.setdefault("hooks", {})
    pre = entries.setdefault(RESTORE_HOOK_EVENT, [])
    kept = [entry for entry in pre if not _is_own_entry(entry, terminal_id=terminal_id)]
    kept.append(restore_hook_entry(command=command))
    entries[RESTORE_HOOK_EVENT] = kept
    return composed


def without_context_restore(hooks_config: Dict[str, Any], *, terminal_id: str) -> Dict[str, Any]:
    """Return ``hooks_config`` minus this terminal's restore entries.

    Every other entry — sibling workers, user hooks, other events —
    passes through untouched: uninstalling one worker never removes
    anything it did not bake.
    """
    composed = copy.deepcopy(hooks_config)
    entries = composed.get("hooks")
    if not isinstance(entries, dict):
        return composed
    pre = entries.get(RESTORE_HOOK_EVENT)
    if isinstance(pre, list):
        entries[RESTORE_HOOK_EVENT] = [
            entry for entry in pre if not _is_own_entry(entry, terminal_id=terminal_id)
        ]
    return composed


def install(
    workspace: Path,
    *,
    terminal_id: str,
    command: str,
) -> Tuple[Optional[Path], bool, Optional[str]]:
    """Install this terminal's restore entry into the workspace hooks file.

    Returns ``(path, created_file, degraded_reason)``. ``created_file``
    tells the uninstaller whether removing the file afterwards is safe.
    Any failure — unreadable config, an unmergeable shape, unwritable
    dir, a lock held by a concurrent writer — degrades with a reason
    instead of failing the launch that called it. The read-modify-write runs under an exclusive
    sidecar lock with an atomic replace, so concurrent installs cannot
    lose each other's entries or tear the file.
    """
    path = hooks_file(Path(workspace))
    with _locked(path) as locked:
        if not locked:
            return (
                None,
                False,
                f"hooks file is locked by another writer ({path}); "
                "launching without restoration",
            )
        data, reason = _read_hooks_file(path)
        if data is None:
            return None, False, reason
        problem = _composable_error(data)
        if problem is not None:
            return None, False, f"{path}: {problem}; leaving user hooks untouched"
        created_file = not path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(
                path, with_context_restore(data, terminal_id=terminal_id, command=command)
            )
        except OSError as exc:
            return None, False, f"could not write {path} ({exc}); launching without restoration"
    return path, created_file, None


def uninstall(workspace: Path, *, terminal_id: str, created_file: bool) -> bool:
    """Remove this terminal's restore entries; never touches anything else.

    The file itself is removed only when this install created it *and*
    nothing remains after our entries are taken out (no ``hooks`` keys
    with any entries left). A pre-existing file is kept, since its
    presence was the operator's choice. Best-effort: every failure
    returns False rather than raising into teardown. Runs under the same
    exclusive lock as :func:`install`, so an uninstall racing an install
    cannot silently drop the install's entry.
    """
    path = hooks_file(Path(workspace))
    with _locked(path) as locked:
        if not locked:
            return False
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return False
        try:
            data = json.loads(raw)
        except ValueError:
            return False
        if not isinstance(data, dict):
            return False
        remaining = without_context_restore(data, terminal_id=terminal_id)
        if _is_empty_hooks(remaining) and created_file:
            try:
                path.unlink()
            except OSError:
                return False
            return True
        if remaining == data:
            return False
        try:
            _atomic_write_json(path, remaining)
        except OSError:
            return False
    return True


def _is_empty_hooks(data: Dict[str, Any]) -> bool:
    """Whether no hook entries remain anywhere in the config."""
    entries = data.get("hooks")
    if not isinstance(entries, dict):
        return True
    for value in entries.values():
        if isinstance(value, list) and value:
            return False
        if isinstance(value, dict) and value:
            return False
    return True


def build_conduct_argv(
    *,
    conduct_binary: str,
    native_session_id: str,
) -> List[str]:
    """The wrapper's one permitted subprocess: the read-only projection.

    Pinned here so there is exactly one place that decides what the
    wrapper may invoke. ``goal hook-context`` performs only GETs and
    store reads: it starts zero turns and grants no resume, hold
    release, or continuation — with this lane's harness claim. No other
    verb may be added without changing this function — and its tests.
    Deliberately NO ``--terminal`` hint: a hint would narrow the
    reader first and hide duplicate binders. The global query lets the
    reader see every binder and answer ``ambiguous``; the wrapper then
    injects only when the resolved terminal exactly equals its baked
    self (:func:`_owns_answer`). No ``--terminal-generation`` either:
    this path carries no managed generation, so the fence stays
    ``unverified`` by documented omission rather than by an invented
    value.
    """
    return [
        conduct_binary,
        "goal",
        "hook-context",
        "--harness",
        HARNESS,
        "--native-session-id",
        native_session_id,
    ]


def _owns_answer(answer: Any, terminal_id: Optional[str]) -> bool:
    """True when an ``ok`` answer resolves exactly this baked terminal.

    The reader answers ``ambiguous`` for duplicate live binders, and
    every non-``ok`` answer is already unrenderable here; the remaining
    hole is a unique resolution for a DIFFERENT terminal — a lingering
    entry from a dead terminal, or a sibling worker's hook observing
    this session in a shared workspace. Closed by comparing the
    reader's live-matched ``identity.terminal_id`` (frozen ``1f1e415f``
    ``conduct/lib/hook_context.py`` ``_identity``: it comes from the
    live match, never an echo of any hint) against the terminal baked
    at install. Unbaked (None) proves nothing: discard.
    """
    if not isinstance(answer, dict) or answer.get("result_type") != "ok":
        return False
    identity = answer.get("identity")
    if not isinstance(identity, dict):
        return False
    resolved = identity.get("terminal_id")
    return (
        isinstance(resolved, str)
        and bool(resolved)
        and terminal_id is not None
        and resolved == terminal_id
    )


def parse_hook_input(raw: bytes) -> Optional[str]:
    """The claimed native session id from hook stdin, or None.

    Strictly the observed snake_case contract: ``hook_event_name`` must
    be ``PreLLMCall`` and ``session_id`` a non-blank string. A sibling
    harness's ``sessionId``/``conversationId`` shape names nothing on
    this path and is refused the same silent-empty way as every other
    unparseable shape (empty stdin, non-JSON bytes, JSON non-object,
    wrong event, missing/non-string/blank id) — none of them names a
    session to project.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("hook_event_name") != RESTORE_HOOK_EVENT:
        return None
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    return session_id.strip()


def _empty_output() -> Dict[str, Any]:
    """Valid no-op output: restores nothing, breaks nothing.

    A bare ``{}`` is allow-and-ignore on the installed build, so the
    invocation proceeds with no injected text on every failure path.
    """
    return {}


def _restore_output(context: str) -> Dict[str, Any]:
    """The fixture-proven injection shape: ``hookEventName`` echo plus
    ``additionalContext``. The echo is required — without it the run
    fails validation and restores nothing."""
    return {
        "hookSpecificOutput": {
            "hookEventName": RESTORE_HOOK_EVENT,
            "additionalContext": context,
        }
    }


def _summarize_detail(detail: Any, *, limit: int = 200) -> str:
    text = detail if isinstance(detail, str) else json.dumps(detail, default=str)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def render_restoration(answer: Dict[str, Any]) -> Optional[str]:
    """The model-facing restoration string for a projection answer.

    Goal rendering, the output bound, the schema check, and
    terminal-state discard are the Claude sibling's renderer — one
    renderer for all harnesses. Only the missing-assignment note is
    reworded here: on this path recovery arrives on the next model
    call (this hook re-fires before every call), not on a SessionStart
    event this lane never uses for restoration.
    """
    if not isinstance(answer, dict):
        return None
    if answer.get("schema") != claude.HOOK_CONTEXT_SCHEMA:
        return None
    text = claude.render_restoration(answer)
    if text is None:
        return None
    if text.startswith(claude.UNAVAILABLE_PREFIX):
        detail = answer.get("detail")
        reason = _summarize_detail(detail) if detail else "no goal is stored yet"
        return (
            f"{UNAVAILABLE_PREFIX}: {reason}. "
            "Ordinary task admission stores the goal first; "
            "the next model call recovers it."
        )
    return text


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
    timeout. Every failure — unparseable input, wrong event, missing
    conduct, slow conduct, malformed answer, discard verdict, foreign
    resolution — injects nothing and still exits 0: a restoration hook
    must never break the invocation.
    """
    args = parse_wrapper_argv(list(argv))
    session_id = parse_hook_input(stdin_bytes)
    if session_id is None:
        return 0, _empty_output(), "hook input names no session; injecting nothing"
    # One global query per invocation: no --terminal hint (it would
    # narrow the reader first and hide duplicate binders), no second
    # query afterwards — uniqueness and ownership are both decided from
    # this single current answer below.
    conduct_argv = build_conduct_argv(
        conduct_binary=args.conduct_bin,
        native_session_id=session_id,
    )
    try:
        if run_conduct is not None:
            answer = run_conduct(conduct_argv)
        else:
            # Same-package reuse of the sibling's projection runner: one
            # subprocess implementation, not a third.
            answer = claude._run_conduct(conduct_argv, timeout_seconds=args.timeout)
    except Exception as exc:  # noqa: BLE001 - lookup failure is scoped to restoration
        return (
            0,
            _empty_output(),
            f"goal projection lookup failed ({exc}); restoration only, invocation unaffected",
        )
    context = render_restoration(answer)
    if context is None:
        return 0, _empty_output(), "projection names no restorable goal; injecting nothing"
    if (
        isinstance(answer, dict)
        and answer.get("result_type") == "ok"
        and not _owns_answer(answer, args.terminal)
    ):
        return (
            0,
            _empty_output(),
            "projection resolved another terminal or none; discarding rather than "
            "restoring a foreign assignment",
        )
    return (
        0,
        _restore_output(claude.apply_output_bound(context, limit=args.max_chars)),
        "",
    )


def parse_wrapper_argv(argv: List[str]) -> argparse.Namespace:
    """Wrapper flags. All binding comes from install-time baking except
    the native session id, which arrives on stdin."""
    parser = argparse.ArgumentParser(
        prog=WRAPPER_ENTRY_POINT,
        description="Muse PreLLMCall CAO goal restoration (read-only; starts no turn)",
    )
    parser.add_argument(
        "--terminal",
        default=None,
        help=(
            "baked self identity: the terminal this hook serves (from the launch path). "
            "Compared against the reader's live-matched terminal before anything "
            "injects; never sent as a search hint, so duplicate live binders stay "
            "visible to the reader instead of resolving ok under separate hints."
        ),
    )
    parser.add_argument(
        "--conduct-bin",
        default="conduct",
        help="conductor CLI binary (absolute path baked at install)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=claude.CONDUCT_TIMEOUT_SECONDS,
        help="seconds to wait for the projection before degrading",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=claude.MAX_ADDITIONAL_CONTEXT_CHARS,
        help="bound on the injected additionalContext string",
    )
    return parser.parse_args(argv)


def attach(
    workspace: Path,
    *,
    terminal_id: str,
    wrapper_executable: Optional[str] = None,
    conduct_binary: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Install restoration for one Muse terminal; never fail the launch.

    Returns ``(installation, degraded_reason)``. On success the
    workspace hooks file carries this terminal's ``PreLLMCall`` entry
    and the installation dict records the mechanism for diagnostics.
    When an executable is unresolvable — or the hooks file refuses the
    merge — nothing is installed and the reason names what is missing.
    """
    # The default resolves THIS lane's entry point, not a sibling's:
    # resolving None would find another harness's wrapper and bake a
    # hook speaking the wrong claim (AGY R1 lesson).
    wrapper = claude.resolve_wrapper_executable(wrapper_executable or WRAPPER_ENTRY_POINT)
    conduct = claude.resolve_conduct_binary(conduct_binary)
    if wrapper is None or conduct is None:
        missing = ", ".join(
            name for name, value in (("wrapper", wrapper), ("conduct", conduct)) if value is None
        )
        return None, (
            f"context restoration not installed ({missing} unresolvable); "
            "launching without PreLLMCall restoration"
        )
    # Baked --terminal is the hook's self identity for the ownership
    # check, not a reader hint (hints are never sent: see
    # build_conduct_argv). No --terminal-generation: this wrapper has no
    # such flag and this path carries no managed generation (unverified
    # fence by documented omission, never an invented value).
    command = " ".join(
        shlex.quote(part)
        for part in (
            wrapper,
            "--terminal",
            terminal_id,
            "--conduct-bin",
            conduct,
            "--timeout",
            str(claude.CONDUCT_TIMEOUT_SECONDS),
        )
    )
    _, created_file, install_reason = install(workspace, terminal_id=terminal_id, command=command)
    if install_reason is not None:
        return None, install_reason
    return (
        describe_installation(terminal_id=terminal_id, created_file=created_file),
        None,
    )


def describe_installation(*, terminal_id: str, created_file: bool) -> Dict[str, Any]:
    """The installation record of what restoration this terminal got.

    One small dict for the caller to log and surface: the selected
    mechanism, the terminal it serves, and whether the hooks file is
    ours to remove at teardown. ``terminal_generation`` is None by
    construction — this path carries no managed generation (see the
    module docstring), so there is no fence value to record.
    """
    return {
        "mechanism": MECHANISM,
        "terminal_id": terminal_id,
        "terminal_generation": None,
        "hooks_file_created": created_file,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Process entry: stdin in, hook JSON out, exit 0 always."""
    exit_code, output, stderr_note = run_wrapper(
        stdin_bytes=sys.stdin.buffer.read(),
        argv=list(argv) if argv is not None else sys.argv[1:],
    )
    if stderr_note:
        print(stderr_note, file=sys.stderr)
    # Stdout carries ONLY the JSON object: anything else would break
    # the hook's output validation.
    sys.stdout.write(json.dumps(output))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
