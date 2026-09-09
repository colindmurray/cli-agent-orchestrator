"""AGY pre-model passive CAO goal restoration (cond-0845 slice 2, AGY lane).

A managed AGY worker loses its CAO assignment when its conversation compacts:
the native conversation survives, but the model-facing goal capsule it was
launched with is compacted away. This module restores it passively through
AGY's own ``PreInvocation`` hook, which fires before every model call and
whose ``injectSteps`` output is injected into the trajectory ahead of that
call — so every invocation re-renders the *current* projection instead of
replaying a baked copy.

Official contract (verified 2026-09-09 against
``https://antigravity.google/docs/hooks`` markdown with installed agy
1.1.28):

* Hooks live in a ``hooks.json`` file in the customization directory (the
  workspace ``.agents/`` dir or ``~/.gemini/config/``). The file maps hook
  names to event configs: ``{"<name>": {"PreInvocation":
  [{"type": "command", "command": "...", "timeout": 30}]}}``. This adapter
  installs one key per terminal in the *workspace* file — never the shared
  global one — and merges additively, preserving every pre-existing entry.
  Concurrent writers serialize on a sidecar lock and every write is
  temp-file plus atomic replace, so racing installs cannot lose keys or
  tear the file; a contended lock degrades the launch, never blocks it.
* ``PreInvocation`` takes no matcher. stdin carries ``invocationNum``,
  ``initialNumSteps`` plus ``conversationId`` (UUID), ``workspacePaths``,
  ``transcriptPath``, ``artifactDirectoryPath``, ``modelName`` (camelCase).
  No generation or terminal id — the managed terminal is baked into the
  hook command at install time from the launch path, never read from stdin.
* stdout ``{"injectSteps": [{"ephemeralMessage": "..."}]}`` injects a
  transient system message ahead of the model call. ``ephemeralMessage``
  is transient by name, so per-invocation re-emission is what makes
  presence hold across a compaction: there is nothing durable to go stale.
* The doc states no command-failure semantics (nonzero exit, malformed
  stdout, timeout) for ``PreInvocation``, so the wrapper always exits 0
  with schema-valid JSON — empty ``injectSteps`` on every failure — and
  never blocks or breaks the invocation it rode in on.

What this adapter does NOT do, by construction (same posture as the Claude
sibling it reuses):

* It never submits a turn, calls steer/send, releases a hold, or bypasses
  a resume-paused worker. The wrapper's only subprocess is ``conduct goal
  hook-context`` — the read-only slice-1 projection (conductor PR #349,
  ``1f1e415f``) — which starts zero turns by contract. The argv is pinned
  in :func:`build_conduct_argv` and asserted in tests.
* It never invents a goal. Anything but ``ok`` restores nothing; a missing
  assignment renders a short truthful note naming the next invocation as
  the recovery event, while stale/ambiguous/terminal callbacks render
  empty steps so a former assignment is never re-injected.
* No timers, no coalescing, no event steer, no physical resume behaviour.

Renderer reuse: goal rendering, the output bound, the projection schema,
and terminal-state discard all come from
:mod:`cli_agent_orchestrator.services.claude_context_restore` — one
renderer, not a second. AGY-specific are only the hook I/O
(``conversationId`` in, ``injectSteps`` out), the harness-claimed argv
pin (the sibling's builder hardcodes its own harness), the hook-file
composition, and the unavailable-note wording (next *invocation*, not
next SessionStart).

Named gaps this slice does not close (see also the module tests, which
pin the degraded behaviour, not the proof):

* Model-entry proof: parse-success plus a structurally valid
  ``injectSteps`` payload is not delivery proof. The marker-echo
  validation (slice-5 style) that an ``ephemeralMessage`` actually
  reaches the model still belongs to a later lane, as does any live-AGY
  launch: no live provider is launched here.
* Generation fence: the AGY provider path carries terminal identity but
  no managed generation, so the projection fence stays ``unverified`` (an
  omission the slice-1 seam explicitly supports). Terminal scope plus
  terminal-state discard is the staleness defence on this path — not a
  second store.
* Duplicate live binders: the wrapper queries globally (no ``--terminal``
  hint, which would narrow the reader first and let two binders each
  resolve ``ok``) and injects only when the single ``ok`` answer's
  live-matched ``identity.terminal_id`` exactly equals its baked
  ``--terminal``. Two binders answer ``ambiguous`` (empty both sides);
  a unique resolution for another terminal is discarded (stale keys,
  shared workspaces). One query, one current answer, no second authority.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import json
import logging
import os
import re
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
#: (``ProviderType.ANTIGRAVITY_CLI``). Checked against the live provider
#: fork-side; never trusted blind (slice-1 seam).
HARNESS = "antigravity_cli"

#: The projection result type carrying a projectable goal (the literal
#: protocol value in the ``cao-hook-context-v1`` envelope — conductor
#: ``1f1e415f`` ``conduct/lib/hook_context.py`` ``_read_goal`` answers
#: ``"ok"`` with the resolved identity plus the projected goal).
_OK_RESULT = "ok"

#: The hook event that restores context. PreInvocation, and only
#: PreInvocation: it is the one event whose output injects into the
#: trajectory ahead of the model call, on every call.
RESTORE_HOOK_EVENT = "PreInvocation"

#: Installed console entry for the wrapper (see ``[project.scripts]``).
WRAPPER_ENTRY_POINT = "cao-agy-hook-context"

#: The selected restoration mechanism, recorded in the launch-facts
#: record so diagnostics can show what a terminal installed (§10.1).
MECHANISM = "agy-PreInvocation:ephemeralMessage"

#: Handler timeout baked into the installed hook entry (seconds). The
#: official default is 30; stated explicitly so the install is
#: self-describing. The wrapper's own conduct timeout stays well under
#: it (reused from the Claude sibling).
HOOK_TIMEOUT_SECONDS = 30

#: Prefix for the installed hook key: one key per terminal, so two
#: workers sharing a workspace never clobber each other and uninstall
#: removes exactly the entry its own install added.
HOOK_KEY_PREFIX = "cao-goal-restore"

#: The workspace customization file this adapter owns one key in
#: (official doc: workspace ``.agents/`` dir).
HOOKS_DIRNAME = ".agents"
HOOKS_FILENAME = "hooks.json"

#: Label marking every injected string as restoration.
RESTORATION_LABEL = "CAO goal restoration"

#: Rendered when the projection holds no projectable goal yet. Same
#: truthfulness contract as the Claude sibling, reworded: recovery
#: arrives on the next model invocation, not a SessionStart event.
UNAVAILABLE_PREFIX = "CAO context restoration unavailable"


def hook_key(terminal_id: str) -> str:
    """The ``hooks.json`` key this terminal's install owns.

    Sanitized so an unusual terminal id cannot break out of a key
    position or collide with another terminal's key after sanitizing.
    """
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", terminal_id).strip("_") or "terminal"
    return f"{HOOK_KEY_PREFIX}-{safe}"


def hooks_file(workspace: Path) -> Path:
    """Where the managed hook entry lives for this workspace."""
    return Path(workspace) / HOOKS_DIRNAME / HOOKS_FILENAME


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


def restore_hook_entry(
    *, command: str, timeout_seconds: int = HOOK_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    """The additive ``PreInvocation`` value installing restoration."""
    return {
        RESTORE_HOOK_EVENT: [{"type": "command", "command": command, "timeout": timeout_seconds}]
    }


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


def with_context_restore(
    hooks_config: Dict[str, Any], *, terminal_id: str, command: str
) -> Dict[str, Any]:
    """Return ``hooks_config`` plus this terminal's restore key.

    The input is not mutated. Every pre-existing key — user hooks for
    any event — is carried over untouched; only this terminal's own key
    is (re)set, so reinstalling is idempotent.
    """
    composed = copy.deepcopy(hooks_config)
    composed[hook_key(terminal_id)] = restore_hook_entry(command=command)
    return composed


def without_context_restore(hooks_config: Dict[str, Any], *, terminal_id: str) -> Dict[str, Any]:
    """Return ``hooks_config`` minus this terminal's restore key.

    Every other key passes through untouched: uninstalling one worker
    never removes a sibling worker's entry or any user hook.
    """
    composed = copy.deepcopy(hooks_config)
    composed.pop(hook_key(terminal_id), None)
    return composed


def install(
    workspace: Path,
    *,
    terminal_id: str,
    command: str,
) -> Tuple[Optional[Path], bool, Optional[str]]:
    """Install this terminal's restore key into the workspace hooks file.

    Returns ``(path, created_file, degraded_reason)``. ``created_file``
    tells the uninstaller whether removing the file afterwards is safe.
    Any failure — unreadable config, unwritable dir, a lock held by a
    concurrent writer — degrades with a reason instead of failing the
    launch that called it. The read-modify-write runs under an exclusive
    sidecar lock with an atomic replace, so concurrent installs cannot
    lose each other's keys or tear the file.
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
    """Remove this terminal's restore key; never touches anything else.

    The file itself is removed only when this install created it *and*
    nothing remains after our key is taken out. A pre-existing file —
    even one left with just ``{}`` — is kept, since its presence was
    the operator's choice. Best-effort: every failure returns False
    rather than raising into teardown. Runs under the same exclusive
    lock as :func:`install`, so an uninstall racing an install cannot
    silently drop the install's key.
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
        if hook_key(terminal_id) not in data:
            return False
        remaining = without_context_restore(data, terminal_id=terminal_id)
        try:
            if created_file and not remaining:
                path.unlink()
            else:
                _atomic_write_json(path, remaining)
        except OSError:
            return False
    return True


def build_conduct_argv(
    *,
    conduct_binary: str,
    native_session_id: str,
) -> List[str]:
    """The wrapper's one permitted subprocess: the read-only projection.

    Pinned here so there is exactly one place that decides what the
    wrapper may invoke. Same shape as the Claude sibling's pinned argv
    (``goal hook-context`` performs only GETs and store reads: it starts
    zero turns and grants no resume, hold release, or continuation) with
    this lane's harness claim — the sibling's builder hardcodes its own
    harness, so this lane pins its own list rather than inheriting the
    wrong identity. No other verb may be added without changing this
    function — and its tests. Deliberately NO ``--terminal`` hint: a
    hint narrows the reader's candidate set first, so two live terminals
    binding one conversation would each resolve ``ok`` under their own
    hint and inject twice. The global query lets the reader see every
    binder and answer ``ambiguous``; the wrapper then injects only when
    the resolved terminal exactly equals its baked self
    (:func:`_owns_answer`). No ``--terminal-generation`` either: the AGY
    provider path carries no managed generation, so the fence stays
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
    key from a dead terminal, or a sibling worker's hook observing this
    conversation in a shared workspace. Closed by comparing the
    reader's live-matched ``identity.terminal_id`` (frozen ``1f1e415f``
    ``conduct/lib/hook_context.py`` ``_identity``: it comes from the
    live match, never an echo of any hint) against the terminal baked
    at install. Unbaked (None) proves nothing: discard.
    """
    if not isinstance(answer, dict) or answer.get("result_type") != _OK_RESULT:
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
    """The claimed conversation id from hook stdin, or None.

    Strictly the documented camelCase ``conversationId``: a sibling
    harness's ``session_id`` shape names nothing on this path and is
    refused the same silent-empty way as every other unparseable shape
    (empty stdin, non-JSON bytes, JSON non-object, missing/non-string/
    blank id) — none of them names a conversation to project.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    conversation_id = payload.get("conversationId")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        return None
    return conversation_id.strip()


def _empty_output() -> Dict[str, Any]:
    """Valid no-op output: restores nothing, breaks nothing.

    An explicitly empty ``injectSteps`` list (rather than empty stdout
    or a nonzero exit, whose handling is undocumented on the installed
    build) keeps the hook response schema-valid so the invocation
    proceeds with no injected text.
    """
    return {"injectSteps": []}


def _restore_output(context: str) -> Dict[str, Any]:
    return {"injectSteps": [{"ephemeralMessage": context}]}


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
    renderer for both harnesses. Only the missing-assignment note is
    reworded here: on this path recovery arrives on the next model
    invocation (this hook re-fires before every call), not on a
    SessionStart event this harness never emits for restoration.
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
            "the next model invocation recovers it."
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
    timeout. Every failure — unparseable input, missing conduct, slow
    conduct, malformed answer, discard verdict — injects nothing and
    still exits 0: a restoration hook must never break the invocation.
    """
    args = parse_wrapper_argv(list(argv))
    conversation_id = parse_hook_input(stdin_bytes)
    if conversation_id is None:
        return 0, _empty_output(), "hook input names no conversation; injecting nothing"
    # One global query per invocation: no --terminal hint (it would
    # narrow the reader first and hide duplicate binders), no second
    # query afterwards — uniqueness and ownership are both decided from
    # this single current answer below.
    conduct_argv = build_conduct_argv(
        conduct_binary=args.conduct_bin,
        native_session_id=conversation_id,
    )
    try:
        if run_conduct is not None:
            answer = run_conduct(conduct_argv)
        else:
            # Same-package reuse of the sibling's projection runner: one
            # subprocess implementation, not a second.
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
    if answer.get("result_type") == _OK_RESULT and not _owns_answer(answer, args.terminal):
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
    the conversation id, which arrives on stdin."""
    parser = argparse.ArgumentParser(
        prog=WRAPPER_ENTRY_POINT,
        description="AGY PreInvocation CAO goal restoration (read-only; starts no turn)",
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
        help="bound on the injected ephemeralMessage string",
    )
    return parser.parse_args(argv)


def attach(
    workspace: Path,
    *,
    terminal_id: str,
    wrapper_executable: Optional[str] = None,
    conduct_binary: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Install restoration for one AGY terminal; never fail the launch.

    Returns ``(installation, degraded_reason)``. On success the
    workspace hooks file carries this terminal's ``PreInvocation`` entry
    and the installation dict records the mechanism for diagnostics.
    When an executable is unresolvable — or the hooks file refuses the
    merge — nothing is installed and the reason names what is missing.
    """
    # The default resolves THIS lane's entry point, not the sibling's:
    # ``resolve_wrapper_executable(None)`` would find
    # ``cao-claude-hook-context`` and bake a hook that speaks the wrong
    # harness claim, so the default names ``cao-agy-hook-context``
    # explicitly (R1 review finding).
    wrapper = claude.resolve_wrapper_executable(wrapper_executable or WRAPPER_ENTRY_POINT)
    conduct = claude.resolve_conduct_binary(conduct_binary)
    if wrapper is None or conduct is None:
        missing = ", ".join(
            name for name, value in (("wrapper", wrapper), ("conduct", conduct)) if value is None
        )
        return None, (
            f"context restoration not installed ({missing} unresolvable); "
            "launching without PreInvocation restoration"
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
        "hook_key": hook_key(terminal_id),
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
    # the hook's schema validation.
    sys.stdout.write(json.dumps(output))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
