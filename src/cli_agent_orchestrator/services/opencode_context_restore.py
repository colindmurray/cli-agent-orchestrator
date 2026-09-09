"""OpenCode per-request passive CAO goal restoration (cond-0845 slice 2, OpenCode).

A managed OpenCode worker loses its CAO assignment on compaction: the native
session survives, but the model-facing goal capsule it was launched with is
compacted away. This module restores it passively through OpenCode's own
``experimental.chat.system.transform`` plugin hook, which runs on **every**
LLM request preparation and lets the plugin push one bounded string onto the
mutable ``system`` array.

Official contract (verified 2026-09-09 against installed opencode 1.18.23,
the plan-pinned source rev):

* Source ``packages/opencode/src/session/llm/request.ts`` (function
  ``LLMRequestPrep.prepare``): ``plugin.trigger(
  "experimental.chat.system.transform", { sessionID, model }, { system })``
  runs per request; the plugin mutates ``system`` in place (extra entries are
  post-merge joined). OpenAI-OAuth providers take the same array as
  ``instructions`` instead — a pushed entry reaches the model either way.
* Plugin docs (``https://opencode.ai/docs/plugins/``): a plugin is a
  JS/TS module exporting a function of ``{ project, client, $, directory,
  worktree }`` returning a hooks object; ``$`` is Bun's shell API.
  Project plugins load from ``.opencode/plugins/`` (additive: every file
  loads, user files untouched); global installs are out of scope for this
  slice. ``experimental.session.compacting`` enriches the compaction
  *summary* only — this adapter never subscribes to it, so restoration can
  never double-inject alongside the summary path.
* ``session.compacted`` is an event-tier notification, not a context path.

How the pieces fit (mirrors the accepted Claude adapter, fork PR #247):

* :func:`render_plugin_source` generates the managed TS plugin with the
  terminal id, managed generation, conduct binary, and timeout **baked in**
  from the authoritative launch record — the same binding that makes the
  projection's generation fence provable (``verified``) instead of merely
  current. The only per-request input is ``input.sessionID``, which the
  provider observes; it travels to the helper in an environment variable,
  never through shell interpolation.
* The helper (console entry :data:`WRAPPER_ENTRY_POINT`) runs the read-only
  slice-1 projection (conductor PR #349, ``1f1e415f``) and prints the one
  bounded string, or nothing. The shim pushes stdout iff non-empty.
* :func:`install_plugin` places the file additively under
  ``<working_directory>/.opencode/plugins/`` with a per-(terminal,
  generation) name. Stale files are safe by construction: a rotated
  generation makes the projection answer ``stale-generation`` and the helper
  prints nothing.

What this adapter does NOT do, by construction:

* It never submits a turn, calls steer/send, releases a hold, or bypasses
  a resume-paused worker. The helper's only subprocess is ``conduct goal
  hook-context`` — the read-only slice-1 projection — which starts zero
  turns by contract. There is no code path that could utter any other verb,
  and the TS shim never touches ``client``, tools, or events.
* It never invents a goal. Anything but a projectable ``ok`` answer prints
  nothing: stale/foreign callbacks, satisfied/cancelled assignments (§10.2),
  and lookup failures restore nothing with exit 0. Unlike the Claude
  SessionStart adapter, a missing assignment also stays silent: the
  transform runs on *every* request, so an "unavailable" note would repeat
  into context on each one; the next request re-reads anyway, and recovery
  is automatic at admission.
* No compacting hook, no ``session.compacted`` handler, no goal store, no
  scheduler, no timers. Those are other slices, not this candidate.

Limits (honest, not waived):

* Each request forks one bounded ``conduct`` subprocess (localhost GETs,
  normally milliseconds). A wedged server can delay a request by up to
  :data:`CONDUCT_TIMEOUT_SECONDS` before the shim degrades to no push.
* Fork-side recording of the opencode native session id (the live binding
  the conductor checks) is owned outside this slice; until it exists, live
  resolution answers ``no-worker`` and the shim pushes nothing — the same
  degraded-safe shape as a stale callback. Unseeded normal-launch
  end-to-end therefore remains dependent on cond-0842.
* ``opencode export`` shows stored system messages, not proof the model
  read them; export output alone is never claimed here as model-entry
  evidence.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cli_agent_orchestrator.services.claude_context_restore import (
    CONDUCT_TIMEOUT_SECONDS,
    HOOK_CONTEXT_SCHEMA,
    MAX_ADDITIONAL_CONTEXT_CHARS,
    apply_output_bound,
    render_restoration,
)

logger = logging.getLogger(__name__)

#: The claiming harness as the conductor's canonical provider id. Must equal
#: the fork's own provider string (``ProviderType.OPENCODE_CLI``); checked
#: against the live provider fork-side by the projection, never trusted
#: blind (slice-1 seam).
HARNESS = "opencode_cli"

#: The per-request projection hook. Runs on every LLM request preparation
#: with ``{ sessionID, model }`` and a mutable ``{ system }`` array.
TRANSFORM_HOOK = "experimental.chat.system.transform"

#: Compaction-summary and event hooks this adapter must never subscribe to.
#: The shim template is asserted to contain none of these strings.
FORBIDDEN_HOOK_FRAGMENTS = (
    "compacting",
    "compacted",
    "session.created",
    "session.idle",
    "event:",
)

#: The selected restoration mechanism, recorded for diagnostics (§10.1).
MECHANISM = "opencode-experimental.chat.system.transform"

#: Installed console entry for the helper (see ``[project.scripts]``).
WRAPPER_ENTRY_POINT = "cao-opencode-hook-context"

#: Env var carrying the provider-observed session id from the TS shim to
#: the helper. Env passing (not shell interpolation) keeps arbitrary
#: session ids out of any shell grammar.
SESSION_ID_ENV_VAR = "CAO_OPENCODE_SESSION_ID"

#: Managed plugin filename prefix. The full name carries the terminal id
#: and generation so concurrent workers never share a file.
PLUGIN_FILENAME_PREFIX = "cao-goal-restoration-"

#: Project plugin directory name, relative to the worker's working
#: directory (official plugin contract).
PLUGIN_DIRNAME = ".opencode/plugins"

__all__ = [
    "CONDUCT_TIMEOUT_SECONDS",
    "FORBIDDEN_HOOK_FRAGMENTS",
    "HARNESS",
    "HOOK_CONTEXT_SCHEMA",
    "MECHANISM",
    "PLUGIN_DIRNAME",
    "PLUGIN_FILENAME_PREFIX",
    "SESSION_ID_ENV_VAR",
    "TRANSFORM_HOOK",
    "WRAPPER_ENTRY_POINT",
    "RestoreBinding",
    "apply_output_bound",
    "build_conduct_argv",
    "build_helper_command",
    "describe_installation",
    "install_plugin",
    "main",
    "parse_helper_argv",
    "plugin_filename",
    "project_plugin_dir",
    "remove_plugin",
    "render_plugin_source",
    "render_restoration",
    "render_transform_text",
    "resolve_session_id",
    "run_helper",
]


@dataclass(frozen=True)
class RestoreBinding:
    """Everything the launch record bakes into one worker's plugin."""

    working_directory: str
    terminal_id: str
    generation: str
    helper_executable: Optional[str] = None
    conduct_binary: Optional[str] = None
    timeout_seconds: float = CONDUCT_TIMEOUT_SECONDS


def _executable(path: str) -> Optional[str]:
    """``path`` when it names something this host could execute, else None."""
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


def resolve_helper_executable(explicit: Optional[str] = None) -> Optional[str]:
    """Absolute helper path, or None when it cannot be resolved.

    None is a normal answer, not an error: a launch whose environment
    cannot resolve the helper degrades to no plugin (logged by the
    caller) rather than failing a launch over a restoration hook.
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


def build_conduct_argv(
    *,
    conduct_binary: str,
    native_session_id: str,
    terminal_id: Optional[str] = None,
    terminal_generation: Optional[str] = None,
) -> List[str]:
    """The helper's one permitted subprocess: the read-only projection.

    Pinned here so there is exactly one place that decides what the
    helper may invoke. ``goal hook-context`` performs only GETs and
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


def _check_shell_safe(value: str, *, field: str) -> str:
    """Reject values that cannot be embedded in the generated plugin.

    The baked command is embedded in a TS template literal and run by
    Bun's shell: backticks, ``${``, backslashes, quotes, and newlines
    must never reach it. Absolute install-time paths never legitimately
    contain these; rejecting them fails the install, never the request.
    """
    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"cannot bake {field}: blank or multi-line")
    for fragment in ("`", "${", "\\", '"'):
        if fragment in value:
            raise ValueError(f"cannot bake {field}: contains {fragment!r}")
    return value


def build_helper_command(
    *,
    helper_executable: str,
    terminal_id: str,
    generation: str,
    conduct_binary: str,
    timeout_seconds: float = CONDUCT_TIMEOUT_SECONDS,
) -> str:
    """The shell command baked into the managed plugin file.

    The terminal id and generation come from the authoritative launch
    record at install time — they are what make the projection's
    generation fence provable (``verified``) instead of merely current.
    The provider-observed session id arrives per request via
    :data:`SESSION_ID_ENV_VAR`, never on this command line.
    """
    for field, value in (
        ("helper", helper_executable),
        ("conduct", conduct_binary),
        ("terminal", terminal_id),
        ("generation", generation),
    ):
        _check_shell_safe(value, field=field)
    parts = [
        helper_executable,
        "--terminal",
        terminal_id,
        "--terminal-generation",
        generation,
        "--conduct-bin",
        conduct_binary,
        "--timeout",
        str(timeout_seconds),
    ]
    return shlex.join(parts)


def render_plugin_source(*, helper_command: str) -> str:
    """The managed TypeScript plugin: one hook, one bounded push.

    Subscribes to :data:`TRANSFORM_HOOK` only. Per request it execs the
    baked helper with the observed session id in the environment and
    pushes stdout iff non-empty. It never throws into the request path,
    never subscribes to compaction/event hooks, and never touches
    ``client``, tools, or settings.
    """
    _check_shell_safe(helper_command, field="helper command")
    return f"""// CAO goal restoration for one managed OpenCode worker (cond-0845).
// Managed file: do not edit. Regenerated per (terminal, generation).
// One bounded read-only goal string per LLM request; starts no turn,
// sends nothing, and never touches compaction, events, or user plugins.
export const CaoGoalRestoration = async ({{ $ }}) => {{
  return {{
    "{TRANSFORM_HOOK}": async (input, output) => {{
      const sessionID = input?.sessionID;
      if (typeof sessionID !== "string" || sessionID.length === 0) return;
      let text = "";
      try {{
        // Spread first: Bun's .env() *replaces* the child environment
        // (verified on bun 1.3.14), so inherit the server env and add
        // only the session id — never interpolate it into shell grammar.
        text = (
          await $`{helper_command}`.env({{ ...process.env, [{json.dumps(SESSION_ID_ENV_VAR)}]: sessionID }}).nothrow().text()
        ).trim();
      }} catch {{
        return;
      }}
      if (text) output.system.push(text);
    }},
  }};
}};
"""


def _check_filename_safe(value: str, *, field: str) -> str:
    if not value or any(
        ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
        for ch in value
    ):
        raise ValueError(f"cannot use {field} in a plugin filename: {value!r}")
    if value.startswith((".", "-")):
        raise ValueError(f"cannot use {field} in a plugin filename: {value!r}")
    return value


def plugin_filename(*, terminal_id: str, generation: str) -> str:
    """Per-(terminal, generation) plugin filename (our prefix, always)."""
    _check_filename_safe(terminal_id, field="terminal id")
    _check_filename_safe(generation, field="generation")
    return f"{PLUGIN_FILENAME_PREFIX}{terminal_id}-{generation}.ts"


def project_plugin_dir(working_directory: str) -> Path:
    """The worker's project plugin directory (official contract)."""
    return Path(working_directory) / ".opencode" / "plugins"


def describe_installation(
    *,
    terminal_id: str,
    generation: str,
    degraded_reason: Optional[str],
) -> Dict[str, Any]:
    """The launch-facts record of what restoration this generation got."""
    return {
        "mechanism": MECHANISM if degraded_reason is None else None,
        "terminal_id": terminal_id,
        "terminal_generation": generation,
        "degraded_reason": degraded_reason,
    }


def install_plugin(binding: RestoreBinding) -> Tuple[Optional[Path], Optional[str]]:
    """Write the worker's managed plugin file, additively.

    Returns ``(path, degraded_reason)``. On success the file exists with
    exactly :func:`render_plugin_source` content and the reason is None.
    When the helper or conduct is unresolvable, nothing is written and
    the reason names what is missing — a launch is never failed over a
    restoration plugin. Pre-existing files (notably user plugins) are
    never modified or removed; our own filename with identical content
    is a benign reinstall, while differing content refuses loudly.
    """
    workdir = Path(binding.working_directory)
    if not workdir.is_dir():
        return None, (
            "context restoration not installed "
            f"(working directory {binding.working_directory!r} is not a directory); "
            "launching without restoration"
        )
    try:
        filename = plugin_filename(terminal_id=binding.terminal_id, generation=binding.generation)
    except ValueError as exc:
        return None, f"context restoration not installed ({exc}); launching without restoration"
    helper = resolve_helper_executable(binding.helper_executable)
    conduct = resolve_conduct_binary(binding.conduct_binary)
    if helper is None or conduct is None:
        missing = ", ".join(
            name for name, value in (("helper", helper), ("conduct", conduct)) if value is None
        )
        return None, (
            f"context restoration not installed ({missing} unresolvable); "
            "launching without restoration"
        )
    try:
        command = build_helper_command(
            helper_executable=helper,
            terminal_id=binding.terminal_id,
            generation=binding.generation,
            conduct_binary=conduct,
            timeout_seconds=binding.timeout_seconds,
        )
        source = render_plugin_source(helper_command=command)
    except ValueError as exc:
        return None, f"context restoration not installed ({exc}); launching without restoration"
    target = project_plugin_dir(binding.working_directory) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_text(encoding="utf-8") == source:
            return target, None
        raise FileExistsError(f"refusing to overwrite differing managed plugin file: {target}")
    target.write_text(source, encoding="utf-8")
    return target, None


def remove_plugin(path: str | Path) -> bool:
    """Remove one managed plugin file. Never touches user files.

    Returns True when a file was removed, False when absent. Refuses
    (ValueError) any path whose filename lacks our managed prefix, so a
    caller bug cannot delete a user plugin.
    """
    candidate = Path(path)
    if candidate.name.startswith((".", "-")) or not candidate.name.startswith(
        PLUGIN_FILENAME_PREFIX
    ):
        raise ValueError(f"refusing to remove non-managed plugin file: {candidate}")
    if not candidate.exists():
        return False
    candidate.unlink()
    return True


def resolve_session_id(
    explicit: Optional[str],
    *,
    env: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """The provider-observed session id, or None when unusable.

    The explicit CLI flag wins (tests, manual probes); otherwise the TS
    shim's :data:`SESSION_ID_ENV_VAR`. Blank/non-string values name no
    worker and degrade to silence, never to a guess.
    """
    if explicit is not None and isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    environ = env if env is not None else os.environ
    candidate = environ.get(SESSION_ID_ENV_VAR)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return None


def render_transform_text(answer: Dict[str, Any]) -> str:
    """The single request-scoped string for a projection answer, or "".

    A projectable ``ok`` renders the shared bounded labeled string (the
    same bytes the Claude adapter injects — reuse, pinned by test).
    Everything else restores nothing: stale/ambiguous/wrong-worker
    answers (re-injecting them would project another assignment's
    context), terminal states (satisfied/cancelled, §10.2), unknown
    schemas, and — deliberately diverging from the SessionStart adapter —
    missing assignments, which stay silent because this hook runs on
    every request and the next request re-reads after admission anyway.
    Never raises for resolution outcomes.
    """
    if not isinstance(answer, dict):
        return ""
    if answer.get("schema") != HOOK_CONTEXT_SCHEMA:
        return ""
    if answer.get("result_type") != "ok":
        return ""
    rendered = render_restoration(answer)
    if rendered is None:
        return ""
    return apply_output_bound(rendered)


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


def run_helper(
    *,
    session_id: Optional[str],
    terminal_id: Optional[str],
    terminal_generation: Optional[str],
    conduct_binary: str,
    timeout_seconds: float,
    max_chars: int,
    run_conduct=None,
) -> Tuple[int, str, str]:
    """Execute one helper invocation. Always exits 0.

    Returns ``(exit_code, stdout_text, stderr_note)``. ``stdout_text``
    is the one bounded string or "". ``run_conduct`` is the seam tests
    substitute for the ``conduct`` subprocess. Every failure — missing
    session id, missing conduct, slow conduct, malformed answer, discard
    verdict — restores nothing and still exits 0: a restoration hook
    must never break the request path.
    """
    if session_id is None:
        return 0, "", "no native session observed; restoring nothing"
    conduct_argv = build_conduct_argv(
        conduct_binary=conduct_binary,
        native_session_id=session_id,
        terminal_id=terminal_id,
        terminal_generation=terminal_generation,
    )
    try:
        if run_conduct is not None:
            answer = run_conduct(conduct_argv)
        else:
            answer = _run_conduct(conduct_argv, timeout_seconds=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - lookup failure is scoped to restoration
        return (
            0,
            "",
            f"goal projection lookup failed ({exc}); restoration only, request unaffected",
        )
    text = render_transform_text(answer)
    if not text:
        return 0, "", "projection names no restorable goal; restoring nothing"
    if len(text) > max_chars:
        text = apply_output_bound(text, limit=max_chars)
    return 0, text, ""


def parse_helper_argv(argv: List[str]) -> argparse.Namespace:
    """Helper flags. All binding comes from install-time baking except
    the native session id, which arrives per request via env."""
    parser = argparse.ArgumentParser(
        prog=WRAPPER_ENTRY_POINT,
        description="OpenCode transform CAO goal restoration (read-only; starts no turn)",
    )
    parser.add_argument(
        "--terminal",
        default=None,
        help="search hint: the terminal this plugin serves (from the launch record)",
    )
    parser.add_argument(
        "--terminal-generation",
        default=None,
        help="managed generation baked from the launch record (clears the fence)",
    )
    parser.add_argument(
        "--native-session-id",
        default=None,
        help="provider-observed session id (defaults to the shim's env var)",
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
        help="bound on the printed restoration string",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Process entry: env/argv in, one string out, exit 0 always."""
    args = parse_helper_argv(list(argv) if argv is not None else sys.argv[1:])
    exit_code, text, stderr_note = run_helper(
        session_id=resolve_session_id(args.native_session_id),
        terminal_id=args.terminal,
        terminal_generation=args.terminal_generation,
        conduct_binary=args.conduct_bin,
        timeout_seconds=args.timeout,
        max_chars=args.max_chars,
    )
    if stderr_note:
        print(stderr_note, file=sys.stderr)
    # Stdout carries ONLY the restoration string (possibly empty): the
    # shim pushes it iff non-empty, so any extra byte would land in
    # model context.
    sys.stdout.write(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
