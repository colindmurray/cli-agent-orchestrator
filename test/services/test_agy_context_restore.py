"""AGY pre-model passive goal restoration (cond-0845 slice 2, AGY lane).

Two reproduced-failure classes, one theme: a hook that restores the wrong
text is worse than a hook that restores nothing.

*The clobbered config.* Registration that rewrites ``hooks.json`` instead
of merging into it silently drops the operator's own hooks — and a
teardown that removes the file unconditionally deletes config it never
owned. Every composition test below drives the real ``install`` /
``uninstall`` against a scratch workspace with pre-existing user entries,
and pins those entries byte-identical afterwards.

*The wrong text.* A wrapper that echoes its stdin, guesses a goal, or
re-injects a stale callback restores another assignment's context into a
live invocation. These tests drive the real wrapper process (stdin in,
stdout out) against a fake ``conduct`` honouring the exact slice-1 CLI
contract, and pin: only ``goal hook-context`` with the ``antigravity_cli``
claim is ever invoked (exactly once — the zero-turn pin), stale and
terminal callbacks inject nothing, and unavailable reads say so without
goal text.

Fixture boundary: the fake ``conduct`` stands in for the live server
reads behind ``conduct goal hook-context`` (conductor PR #349,
``1f1e415f``). Its canned answers use the real envelope
(``cao-hook-context-v1``) and the real result types, but no test here
proves the conductor side — that proof belongs to the conductor lane.
The binding-aware legs (global candidate scan, live-identity
agreement, ambiguous-on-two-binders, live-matched ``identity.terminal_id``)
mirror the frozen projection (``conduct/lib/hook_context.py`` at
``1f1e415f``: ``_terminal_candidates`` with no hint, ``_match_terminals``,
``resolve_hook_caller``, ``_identity``) so the no-double-inject tests
assert the documented contract, not a guess. The fake additionally
refuses any ``--terminal`` hint the way the real reader would hide
duplicates behind one — a regression to hinting fails loudly here
instead of replaying a goal.
Likewise no test launches a live ``agy``: model-entry proof (the
marker-echo validation) belongs to a later lane, and these tests pin the
degraded-but-honest behaviour instead.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from cli_agent_orchestrator.services import agy_context_restore as restore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"

CONVERSATION = "ec33ebf9-0cba-4100-8142-c61503f6c587"
OTHER_CONVERSATION = "aaaaaaaa-0000-4111-8111-aaaaaaaaaaaa"

FAKE_CONDUCT = (
    """\
#!"""
    + sys.executable
    + """
import json, os, sys, time

log_path = os.environ["CONDUCT_ARGV_LOG"]
with open(log_path, "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")

argv = sys.argv[1:]
# The fork client contract (conductor goal_hook_context.py): the wrapper
# must claim this harness with the observed conversation. A wrong
# identity here resolves another worker or none — so the fake refuses it
# loudly rather than replaying a goal onto it.
if "--harness" not in argv or argv[argv.index("--harness") + 1] != "antigravity_cli":
    sys.stderr.write("wrong harness claim\\n")
    sys.exit(2)
if "--native-session-id" not in argv:
    sys.stderr.write("missing native session claim\\n")
    sys.exit(2)
if "--terminal-generation" in argv:
    sys.stderr.write("agy path must not bake a generation\\n")
    sys.exit(2)
# The wrapper must query globally: a --terminal hint would narrow the
# real reader first and hide duplicate binders (R2 finding), so the
# fake refuses it loudly rather than replaying a goal onto it.
if "--terminal" in argv:
    sys.stderr.write("wrapper must query globally, never hint\\n")
    sys.exit(2)


def _flag(name):
    return argv[argv.index(name) + 1] if name in argv else None


def _typed(result_type, detail):
    sys.stdout.write(json.dumps({
        "ok": True,
        "schema": "cao-hook-context-v1",
        "result_type": result_type,
        "detail": detail,
        "recovery": None,
        "identity": None,
        "goal": None,
        "bounds": {},
    }))


# Global-resolution legs mirror the frozen projection (conductor 1f1e415f
# conduct/lib/hook_context.py _terminal_candidates with no hint +
# _match_terminals + resolve_hook_caller): every live terminal is
# examined, a match needs a LIVE binding agreeing with the claimed
# native session, and two live binders refuse as ambiguous.
# CONDUCT_BINDINGS maps terminal -> live native session. The ok leg
# reports the live-matched terminal in identity.terminal_id, exactly
# like the real _identity constructor (live match, never a hint echo).
bindings_raw = os.environ.get("CONDUCT_BINDINGS")
harness_ok = argv[argv.index("--harness") + 1] == "antigravity_cli"
native = _flag("--native-session-id")
if bindings_raw and harness_ok and native:
    bindings = json.loads(bindings_raw)
    binders = sorted(t for t, n in bindings.items() if n == native)
    if len(binders) > 1:
        _typed("ambiguous", "%d live terminals bind %r" % (len(binders), native))
        sys.exit(0)
    if not binders:
        _typed("no-worker", "no live terminal binds %r" % native)
        sys.exit(0)
    with open(os.environ["CONDUCT_CANNED"]) as handle:
        canned = json.load(handle)
    canned["identity"]["terminal_id"] = binders[0]
    canned["identity"]["native_session_id"] = native
    sys.stdout.write(json.dumps(canned))
    sys.exit(0)
mode = os.environ.get("CONDUCT_MODE", "ok")
if mode == "sleep":
    time.sleep(float(os.environ.get("CONDUCT_SLEEP", "5")))
    mode = "ok"
if mode == "fail":
    sys.stderr.write("boom\\n")
    sys.exit(1)
if mode == "garbage":
    sys.stdout.write("not json\\n")
    sys.exit(0)
with open(os.environ["CONDUCT_CANNED"]) as handle:
    sys.stdout.write(handle.read())
"""
)


def _ok_answer(*, objective="Ship the thing", state="open"):
    next_action = (
        "ordinary work may continue; this read starts no turn"
        if state == "open"
        else f"no restoration: occurrence 'occ-1' is {state}; start no turn"
    )
    return {
        "ok": True,
        "schema": "cao-hook-context-v1",
        "result_type": "ok",
        "detail": None,
        "recovery": None,
        "identity": {
            "harness": "antigravity_cli",
            "native_session_id": CONVERSATION,
            "terminal_id": "t1",
            "terminal_generation": None,
            "generation_fence": "unverified",
        },
        "goal": {
            "goal_id": "g-1",
            "state": state,
            "goal_version": "v3",
            "objective": objective,
            "requirements_outstanding": ["r1"],
            "requirements_outstanding_count": 1,
            "completion_requirements_truncated": False,
            "active_hold": None,
            "next_action": next_action,
        },
        "bounds": {},
    }


def _typed_answer(result_type, *, detail="some detail"):
    answer = _ok_answer()
    answer["result_type"] = result_type
    answer["detail"] = detail
    answer["goal"] = None
    return answer


@pytest.fixture()
def fake(tmp_path):
    """An executable fake ``conduct`` honouring the slice-1 CLI contract."""
    script = tmp_path / "conduct"
    script.write_text(FAKE_CONDUCT)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    canned = tmp_path / "canned.json"
    canned.write_text(json.dumps(_ok_answer()))
    log = tmp_path / "argv.log"
    log.write_text("")
    env = dict(
        os.environ,
        CONDUCT_CANNED=str(canned),
        CONDUCT_ARGV_LOG=str(log),
        CONDUCT_MODE="ok",
    )
    return {"script": script, "canned": canned, "log": log, "env": env}


def _hook_stdin(*, conversation_id=CONVERSATION):
    return json.dumps(
        {
            "invocationNum": 3,
            "initialNumSteps": 10,
            "conversationId": conversation_id,
            "workspacePaths": ["/tmp/wt"],
            "transcriptPath": "/tmp/t.jsonl",
            "artifactDirectoryPath": "/tmp/a",
            "modelName": "gemini-3.6-flash-medium",
        }
    ).encode()


def _run_wrapper_process(stdin_bytes, *, fake, extra_args=(), extra_env=None):
    """The real wrapper process: stdin in, hook JSON out."""
    env = dict(fake["env"], PYTHONPATH=str(SRC_DIR))
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.agy_context_restore",
            "--conduct-bin",
            str(fake["script"]),
            "--timeout",
            "5",
            *extra_args,
        ],
        input=stdin_bytes,
        capture_output=True,
        env=env,
        timeout=60,
    )
    return proc


def _logged_argvs(fake):
    return [json.loads(line) for line in Path(fake["log"]).read_text().splitlines() if line.strip()]


def _user_hooks_config():
    return {
        "my-linter-hook": {
            "PostToolUse": [
                {
                    "matcher": "run_command",
                    "hooks": [{"type": "command", "command": "./scripts/lint.sh", "timeout": 10}],
                }
            ]
        },
        "reminder": {"PreInvocation": [{"type": "command", "command": "./scripts/reminder.sh"}]},
    }


class TestHookInputParsing:
    def test_conversation_id_parsed(self):
        assert restore.parse_hook_input(_hook_stdin()) == CONVERSATION

    def test_claude_shaped_session_id_is_refused(self):
        """A sibling harness's ``session_id`` names nothing on this path."""
        raw = json.dumps({"session_id": CONVERSATION, "hook_event_name": "SessionStart"}).encode()
        assert restore.parse_hook_input(raw) is None

    @pytest.mark.parametrize(
        "raw",
        [
            b"",
            b"not json",
            b"[1, 2]",
            b'"str"',
            json.dumps({"conversationId": ""}).encode(),
            json.dumps({"conversationId": "   "}).encode(),
            json.dumps({"conversationId": 42}).encode(),
            json.dumps({}).encode(),
        ],
    )
    def test_unparseable_shapes_are_silent(self, raw):
        assert restore.parse_hook_input(raw) is None


class TestHookFileComposition:
    def test_install_creates_exact_preinvocation_shape(self, tmp_path):
        path, created, degraded = restore.install(
            tmp_path, terminal_id="t1", command="/w wrapper --terminal t1"
        )
        assert degraded is None
        assert created is True
        assert path == tmp_path / ".agents" / "hooks.json"
        data = json.loads(path.read_text())
        assert data == {
            "cao-goal-restore-t1": {
                "PreInvocation": [
                    {
                        "type": "command",
                        "command": "/w wrapper --terminal t1",
                        "timeout": 30,
                    }
                ]
            }
        }

    def test_install_preserves_user_entries_byte_identical(self, tmp_path):
        user = _user_hooks_config()
        hooks = tmp_path / ".agents" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(json.dumps(user, indent=2))
        before = json.loads(hooks.read_text())

        _, _, degraded = restore.install(tmp_path, terminal_id="t1", command="cmd")

        assert degraded is None
        after = json.loads(hooks.read_text())
        assert after["my-linter-hook"] == before["my-linter-hook"]
        assert after["reminder"] == before["reminder"]
        assert set(after) == {"my-linter-hook", "reminder", "cao-goal-restore-t1"}

    def test_install_is_idempotent_and_scoped_per_terminal(self, tmp_path):
        restore.install(tmp_path, terminal_id="t1", command="cmd-one")
        restore.install(tmp_path, terminal_id="t1", command="cmd-two")
        restore.install(tmp_path, terminal_id="t2", command="cmd-two")
        data = json.loads((tmp_path / ".agents" / "hooks.json").read_text())
        assert set(data) == {"cao-goal-restore-t1", "cao-goal-restore-t2"}
        assert data["cao-goal-restore-t1"]["PreInvocation"][0]["command"] == "cmd-two"

    def test_uninstall_removes_only_its_own_key(self, tmp_path):
        hooks = tmp_path / ".agents" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(json.dumps(_user_hooks_config(), indent=2))
        restore.install(tmp_path, terminal_id="t1", command="cmd")
        restore.install(tmp_path, terminal_id="t2", command="cmd")

        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=False) is True

        data = json.loads(hooks.read_text())
        assert set(data) == {"my-linter-hook", "reminder", "cao-goal-restore-t2"}

    def test_uninstall_removes_file_only_when_we_created_it(self, tmp_path):
        path, created, _ = restore.install(tmp_path, terminal_id="t1", command="cmd")
        assert created is True
        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=True) is True
        assert not path.exists()

    def test_uninstall_keeps_preexisting_file(self, tmp_path):
        hooks = tmp_path / ".agents" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(json.dumps(_user_hooks_config(), indent=2))
        restore.install(tmp_path, terminal_id="t1", command="cmd")
        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=False) is True
        assert hooks.exists()
        assert json.loads(hooks.read_text()) == _user_hooks_config()

    def test_uninstall_missing_everything_is_quiet(self, tmp_path):
        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=False) is False
        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=True) is False

    def test_malformed_config_refuses_without_clobbering(self, tmp_path):
        hooks = tmp_path / ".agents" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text("{ not json")
        before = hooks.read_text()

        path, created, degraded = restore.install(tmp_path, terminal_id="t1", command="cmd")

        assert path is None and created is False and degraded is not None
        assert hooks.read_text() == before

    def test_non_object_config_refuses_without_clobbering(self, tmp_path):
        hooks = tmp_path / ".agents" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text('["a list, not hooks"]')

        path, _, degraded = restore.install(tmp_path, terminal_id="t1", command="cmd")

        assert path is None and degraded is not None
        assert hooks.read_text() == '["a list, not hooks"]'

    def test_composition_does_not_mutate_its_input(self):
        user = _user_hooks_config()
        snapshot = json.loads(json.dumps(user))
        restore.with_context_restore(user, terminal_id="t1", command="cmd")
        restore.without_context_restore(user, terminal_id="t1")
        assert user == snapshot

    def test_hook_key_sanitizes_hostile_terminal_ids(self):
        assert restore.hook_key("t1") == "cao-goal-restore-t1"
        key = restore.hook_key("../../etc/x")
        assert key.startswith("cao-goal-restore-")
        assert "/" not in key and key != restore.hook_key("t1")


class TestWrapperProcess:
    def test_ok_answer_injects_ephemeral_message(self, fake):
        proc = _run_wrapper_process(_hook_stdin(), fake=fake, extra_args=["--terminal", "t1"])
        assert proc.returncode == 0
        out = json.loads(proc.stdout)
        steps = out["injectSteps"]
        assert len(steps) == 1
        body = steps[0]["ephemeralMessage"]
        assert "CAO goal restoration" in body
        assert "Ship the thing" in body

    def test_projection_called_exactly_once_globally_without_hints(self, fake):
        """One global query per invocation: no hint, no generation.

        A ``--terminal`` hint would narrow the real reader first and
        hide duplicate binders (each hinted call resolving ``ok`` under
        its own hint); the wrapper must never send one, so the reader
        sees every binder and answers ``ambiguous`` for duplicates.
        """
        _run_wrapper_process(_hook_stdin(), fake=fake, extra_args=["--terminal", "t1"])
        argvs = _logged_argvs(fake)
        assert len(argvs) == 1
        argv = argvs[0]
        assert argv[:4] == ["goal", "hook-context", "--harness", "antigravity_cli"]
        assert "--native-session-id" in argv
        assert argv[argv.index("--native-session-id") + 1] == CONVERSATION
        assert "--terminal" not in argv
        assert "--terminal-generation" not in argv

    def test_stale_and_terminal_callbacks_inject_nothing(self, fake):
        for result_type in ("stale-generation", "dead-incarnation", "ambiguous", "no-worker"):
            fake["canned"].write_text(json.dumps(_typed_answer(result_type)))
            proc = _run_wrapper_process(_hook_stdin(), fake=fake)
            assert proc.returncode == 0
            assert json.loads(proc.stdout) == {"injectSteps": []}
        for state in ("satisfied", "cancelled"):
            fake["canned"].write_text(json.dumps(_ok_answer(state=state)))
            proc = _run_wrapper_process(_hook_stdin(), fake=fake)
            assert proc.returncode == 0
            assert json.loads(proc.stdout) == {"injectSteps": []}

    def test_missing_assignment_names_next_invocation_without_goal_text(self, fake):
        fake["canned"].write_text(json.dumps(_typed_answer("no-assignment")))
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert proc.returncode == 0
        body = json.loads(proc.stdout)["injectSteps"][0]["ephemeralMessage"]
        assert body.startswith("CAO context restoration unavailable")
        assert "next model invocation" in body
        assert "Ship the thing" not in body
        assert "SessionStart" not in body

    def test_conduct_failure_degrades_to_empty_steps(self, fake):
        fake["env"]["CONDUCT_MODE"] = "fail"
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {"injectSteps": []}

    def test_conduct_garbage_degrades_to_empty_steps(self, fake):
        fake["env"]["CONDUCT_MODE"] = "garbage"
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {"injectSteps": []}

    def test_unparseable_input_never_calls_conduct(self, fake):
        proc = _run_wrapper_process(b"not json", fake=fake)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {"injectSteps": []}
        assert _logged_argvs(fake) == []

    def test_long_goal_is_cut_with_marker_inside_the_bound(self, fake):
        fake["canned"].write_text(json.dumps(_ok_answer(objective="x" * 20000)))
        proc = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t1", "--max-chars", "8000"]
        )
        body = json.loads(proc.stdout)["injectSteps"][0]["ephemeralMessage"]
        assert len(body) <= 8000
        assert "truncated here" in body


class TestAttach:
    def test_attach_installs_and_records_mechanism(self, tmp_path):
        installation, degraded = restore.attach(
            tmp_path,
            terminal_id="t1",
            wrapper_executable=sys.executable,
            conduct_binary=sys.executable,
        )
        assert degraded is None
        assert installation["mechanism"] == "agy-PreInvocation:ephemeralMessage"
        assert installation["terminal_id"] == "t1"
        assert installation["terminal_generation"] is None
        assert installation["hook_key"] == "cao-goal-restore-t1"
        data = json.loads((tmp_path / ".agents" / "hooks.json").read_text())
        command = data["cao-goal-restore-t1"]["PreInvocation"][0]["command"]
        assert "--terminal t1" in command
        assert "--terminal-generation" not in command

    def test_attach_degrades_when_executables_unresolvable(self, tmp_path):
        installation, degraded = restore.attach(
            tmp_path,
            terminal_id="t1",
            wrapper_executable="/nonexistent/wrapper",
            conduct_binary="/nonexistent/conduct",
        )
        assert installation is None
        assert degraded is not None
        assert not (tmp_path / ".agents" / "hooks.json").exists()

    def test_attach_degrades_without_clobbering_bad_config(self, tmp_path):
        hooks = tmp_path / ".agents" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text("{ nope")
        installation, degraded = restore.attach(
            tmp_path,
            terminal_id="t1",
            wrapper_executable=sys.executable,
            conduct_binary=sys.executable,
        )
        assert installation is None and degraded is not None
        assert hooks.read_text() == "{ nope"


class TestProviderWiring:
    def _make_provider(self, **kwargs):
        from cli_agent_orchestrator.providers.antigravity_cli import AntigravityCliProvider

        return AntigravityCliProvider(
            terminal_id="t1",
            session_name="test-session",
            window_name="window-0",
            **kwargs,
        )

    def test_no_workspace_leaves_restoration_uninstalled(self, tmp_path):
        provider = self._make_provider()
        provider._install_context_restore()
        assert provider.context_restoration is None

    def test_install_registers_terminal_binding(self, tmp_path, monkeypatch):
        from cli_agent_orchestrator.services import agy_context_restore as agy

        monkeypatch.setattr(
            agy.claude, "resolve_wrapper_executable", lambda explicit=None: sys.executable
        )
        monkeypatch.setattr(
            agy.claude, "resolve_conduct_binary", lambda explicit=None: sys.executable
        )
        provider = self._make_provider(hooks_workspace=str(tmp_path))
        provider._install_context_restore()
        installation = provider.context_restoration
        assert installation is not None
        assert installation["mechanism"] == "agy-PreInvocation:ephemeralMessage"
        assert installation["terminal_id"] == "t1"
        data = json.loads((tmp_path / ".agents" / "hooks.json").read_text())
        assert set(data) == {"cao-goal-restore-t1"}

    def test_cleanup_uninstalls_only_its_own_key(self, tmp_path, monkeypatch):
        from cli_agent_orchestrator.services import agy_context_restore as agy

        monkeypatch.setattr(
            agy.claude, "resolve_wrapper_executable", lambda explicit=None: sys.executable
        )
        monkeypatch.setattr(
            agy.claude, "resolve_conduct_binary", lambda explicit=None: sys.executable
        )
        hooks = tmp_path / ".agents" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(json.dumps(_user_hooks_config(), indent=2))
        first = self._make_provider(hooks_workspace=str(tmp_path))
        first._install_context_restore()
        other = self._make_provider(hooks_workspace=str(tmp_path))
        other.terminal_id = "t2"
        other._install_context_restore()

        first.cleanup()

        data = json.loads(hooks.read_text())
        assert set(data) == {"my-linter-hook", "reminder", "cao-goal-restore-t2"}
        assert first.context_restoration is None

    def test_attach_default_resolves_agy_wrapper_not_claude(self, tmp_path, monkeypatch):
        """The production default (no explicit wrapper) bakes THIS lane.

        R1 finding: ``attach`` defaulted to the sibling resolver, which
        would bake ``cao-claude-hook-context`` — a hook speaking the
        wrong harness claim. The PATH here carries only the AGY entry,
        so a Claude default degrades instead of installing.
        """
        bindir = tmp_path / "bin"
        bindir.mkdir()
        wrapper = bindir / "cao-agy-hook-context"
        wrapper.write_text("#!/bin/sh\nexit 0\n")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", str(bindir))
        installation, degraded = restore.attach(
            tmp_path / "ws",
            terminal_id="t1",
            conduct_binary=sys.executable,
        )
        assert degraded is None
        assert installation is not None
        command = json.loads((tmp_path / "ws" / ".agents" / "hooks.json").read_text())[
            "cao-goal-restore-t1"
        ]["PreInvocation"][0]["command"]
        assert str(wrapper) in command
        assert "cao-claude-hook-context" not in command

    def test_install_failure_never_raises(self, tmp_path, monkeypatch):
        from cli_agent_orchestrator.services import agy_context_restore as agy

        monkeypatch.setattr(
            agy.claude, "resolve_wrapper_executable", lambda explicit=None: sys.executable
        )
        monkeypatch.setattr(
            agy.claude, "resolve_conduct_binary", lambda explicit=None: sys.executable
        )
        hooks = tmp_path / ".agents" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text("{ broken")
        provider = self._make_provider(hooks_workspace=str(tmp_path))
        provider._install_context_restore()  # must not raise
        assert provider.context_restoration is None
        assert hooks.read_text() == "{ broken"
        provider.cleanup()  # must not raise either


class TestHookFileConcurrency:
    """Additive read-modify-write alone is not concurrency-proof (R1).

    Two racing installs could each read the pre-other base and the last
    writer would drop the other's key; a torn write could strand invalid
    JSON. Install/uninstall serialize on a sidecar lock and write
    temp-plus-atomic-replace; these tests prove the serialization is
    real, not the absence of a race in a lucky interleaving.
    """

    def _seed_user_config(self, workspace):
        hooks = workspace / ".agents" / "hooks.json"
        hooks.parent.mkdir(parents=True, exist_ok=True)
        hooks.write_text(json.dumps(_user_hooks_config(), indent=2))
        return hooks

    def test_contended_lock_degrades_install(self, tmp_path, monkeypatch):
        import fcntl

        monkeypatch.setattr(restore, "LOCK_TIMEOUT_SECONDS", 0.15)
        workspace = tmp_path / "ws"
        self._seed_user_config(workspace)
        path = restore.hooks_file(workspace)
        holder = open(restore._lock_path(path), "a+b")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        try:
            installed, created, degraded = restore.install(
                workspace, terminal_id="t1", command="cmd"
            )
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()
        assert installed is None and created is False
        assert degraded is not None and "locked" in degraded

    def test_contended_lock_skips_uninstall(self, tmp_path, monkeypatch):
        import fcntl

        monkeypatch.setattr(restore, "LOCK_TIMEOUT_SECONDS", 0.15)
        workspace = tmp_path / "ws"
        self._seed_user_config(workspace)
        restore.install(workspace, terminal_id="t1", command="cmd")
        path = restore.hooks_file(workspace)
        holder = open(restore._lock_path(path), "a+b")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        try:
            assert restore.uninstall(workspace, terminal_id="t1", created_file=False) is False
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()
        # Skipped teardown leaves the key for the next install to
        # overwrite — never a half-removed file.
        data = json.loads(path.read_text())
        assert "cao-goal-restore-t1" in data

    def test_concurrent_installs_lose_no_keys(self, tmp_path):
        import concurrent.futures

        workspace = tmp_path / "ws"
        self._seed_user_config(workspace)

        def work(i):
            terminal = f"t{i % 4}"
            if i % 7 == 6:
                restore.uninstall(workspace, terminal_id=terminal, created_file=False)
            else:
                restore.install(workspace, terminal_id=terminal, command=f"cmd-{terminal}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(96)))

        data = json.loads((workspace / ".agents" / "hooks.json").read_text())
        # User entries survive every interleaving, byte-identical.
        assert data["my-linter-hook"] == _user_hooks_config()["my-linter-hook"]
        assert data["reminder"] == _user_hooks_config()["reminder"]
        # Every surviving managed key is well-formed; no torn JSON ever
        # parses, so reaching here already proves atomicity.
        for key, value in data.items():
            if not key.startswith("cao-goal-restore-"):
                continue
            assert set(value) == {"PreInvocation"}
            (handler,) = value["PreInvocation"]
            assert handler["type"] == "command"
            assert handler["command"] in {f"cmd-t{i}" for i in range(4)}
            assert handler["timeout"] == 30


class TestOwnershipGate:
    """Direct unit pins for the ``_owns_answer`` comparison."""

    def test_matching_live_terminal_owns(self):
        assert restore._owns_answer(_ok_answer(), "t1") is True

    def test_other_terminal_does_not_own(self):
        assert restore._owns_answer(_ok_answer(), "t2") is False

    def test_unbaked_wrapper_owns_nothing(self):
        assert restore._owns_answer(_ok_answer(), None) is False

    def test_missing_or_foreign_identity_owns_nothing(self):
        answer = _ok_answer()
        del answer["identity"]
        assert restore._owns_answer(answer, "t1") is False
        answer = _ok_answer()
        answer["identity"] = {"terminal_id": None}
        assert restore._owns_answer(answer, "t1") is False

    def test_non_ok_answers_are_not_owned(self):
        for result_type in ("no-assignment", "ambiguous", "stale-generation"):
            answer = _typed_answer(result_type)
            assert restore._owns_answer(answer, "t1") is False


class TestSharedWorkspaceBinding:
    """Two hooks, one workspace, no generation fence — still one truth.

    Every hook queries globally (the fake refuses any ``--terminal``
    hint, exactly because a hint would narrow the real reader first and
    hide duplicate binders). Global legs mirror the frozen projection
    (conductor ``1f1e415f`` ``conduct/lib/hook_context.py``
    ``_terminal_candidates`` with no hint + ``_match_terminals`` +
    ``resolve_hook_caller``): every live terminal is examined, a match
    needs a LIVE binding agreeing with the claimed conversation, two
    live binders answer ``ambiguous``, and the ok leg reports the
    live-matched terminal. The wrapper injects only when that terminal
    exactly equals its baked self — so at most the unique owner
    injects, from one query and one current answer.
    """

    def _bound(self, fake, bindings):
        return dict(fake["env"], CONDUCT_BINDINGS=json.dumps(bindings))

    def test_two_live_binders_inject_nothing_either_side(self, fake):
        """The same conversation live under two terminals: ambiguous.

        Both hooks fire, both ask globally exactly once, both render
        empty — no double injection into either trajectory.
        """
        env = self._bound(fake, {"t1": CONVERSATION, "t2": CONVERSATION})
        first = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t1"], extra_env=env
        )
        second = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t2"], extra_env=env
        )
        assert first.returncode == 0 and second.returncode == 0
        assert json.loads(first.stdout) == {"injectSteps": []}
        assert json.loads(second.stdout) == {"injectSteps": []}
        argvs = _logged_argvs(fake)
        assert len(argvs) == 2
        assert all("--terminal" not in argv for argv in argvs)

    def test_stale_key_and_live_key_inject_exactly_once(self, fake):
        """A lingering key from a dead terminal stays silent.

        The shared workspace file still names t1, but only t2 live-binds
        the conversation: the t1 hook's answer resolves t2, which is not
        its baked self, so it discards; the t2 hook projects — exactly
        one injection across both firings.
        """
        env = self._bound(fake, {"t2": CONVERSATION})
        stale = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t1"], extra_env=env
        )
        live = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t2"], extra_env=env
        )
        assert stale.returncode == 0 and live.returncode == 0
        assert json.loads(stale.stdout) == {"injectSteps": []}
        body = json.loads(live.stdout)["injectSteps"][0]["ephemeralMessage"]
        assert "Ship the thing" in body

    def test_foreign_conversation_never_projects(self, fake):
        """An unbound conversation resolves no-worker: empty, exit 0."""
        env = self._bound(fake, {"t1": CONVERSATION})
        proc = _run_wrapper_process(
            _hook_stdin(conversation_id=OTHER_CONVERSATION),
            fake=fake,
            extra_args=["--terminal", "t1"],
            extra_env=env,
        )
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {"injectSteps": []}

    def test_unbaked_wrapper_discards_even_unique_owner(self, fake):
        """No baked self means no provable ownership: discard.

        The global answer resolves t1 uniquely, but a hook that cannot
        name itself cannot prove it IS t1 — so it injects nothing.
        """
        env = self._bound(fake, {"t1": CONVERSATION})
        proc = _run_wrapper_process(_hook_stdin(), fake=fake, extra_env=env)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {"injectSteps": []}
