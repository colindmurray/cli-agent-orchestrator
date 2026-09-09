"""Muse pre-model passive goal restoration (cond-0845 slice, Muse lane).

Two reproduced-failure classes, one theme: a hook that restores the wrong
text is worse than a hook that restores nothing.

*The clobbered config.* Registration that rewrites ``.muse/hooks.json``
instead of merging into it silently drops the operator's own hooks — and
a teardown that removes the file unconditionally deletes config it never
owned. Every composition test below drives the real ``install`` /
``uninstall`` against a scratch workspace with pre-existing user entries,
and pins those entries intact afterwards.

*The wrong text.* A wrapper that echoes its stdin, guesses a goal, or
re-injects a stale callback restores another assignment's context into a
live model call. These tests drive the real wrapper process (stdin in,
stdout out) against a fake ``conduct`` honouring the exact slice-1 CLI
contract, and pin: only ``goal hook-context`` with the ``muse_cli``
claim is ever invoked (exactly once per invocation — the zero-turn pin),
stale and terminal callbacks inject nothing, and unavailable reads say
so without goal text.

Proven fixture contract (installed
``muse-bin-1.0.3-R2198.1``, non-billable echo provider): ``PreLLMCall``
stdin carries ``session_id`` (snake_case); stdout
``{"hookSpecificOutput": {"hookEventName": "PreLLMCall",
"additionalContext": "..."}}`` completes with the ``context`` effect
and installs a developer-role request block; bare ``{}`` is the no-op.
The binding legs (global candidate scan, live-identity agreement,
ambiguous-on-two-binders, live-matched ``identity.terminal_id``) mirror
the frozen projection (``conduct/lib/hook_context.py`` at ``1f1e415f``).

Fixture boundary: the fake ``conduct`` stands in for the live server
reads behind ``conduct goal hook-context`` (conductor PR #349,
``1f1e415f``). Its canned answers use the real envelope
(``cao-hook-context-v1``) and the real result types, but no test here
proves the conductor side — that proof belongs to the conductor lane.
Likewise no test launches a live ``muse`` with a model: effect
acceptance plus a structurally valid ``additionalContext`` payload is
not model-entry proof, and the marker-echo validation belongs to a
later lane.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from cli_agent_orchestrator.services import muse_context_restore as restore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"

SESSION = "01a08580-f317-7bd1-a92f-6c76a861c43e"
OTHER_SESSION = "aaaaaaaa-0000-4111-8111-aaaaaaaaaaaa"

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
# must claim this harness with the observed native session. A wrong
# identity here resolves another worker or none — so the fake refuses it
# loudly rather than replaying a goal onto it. The wrapper must also
# query globally: a --terminal hint would narrow the real reader first
# and hide duplicate binders.
if "--harness" not in argv or argv[argv.index("--harness") + 1] != "muse_cli":
    sys.stderr.write("wrong harness claim\\n")
    sys.exit(2)
if "--native-session-id" not in argv:
    sys.stderr.write("missing native session claim\\n")
    sys.exit(2)
if "--terminal" in argv:
    sys.stderr.write("wrapper must query globally, never hint\\n")
    sys.exit(2)
if "--terminal-generation" in argv:
    sys.stderr.write("muse path must not bake a generation\\n")
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
harness_ok = argv[argv.index("--harness") + 1] == "muse_cli"
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
            "harness": "muse_cli",
            "native_session_id": SESSION,
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


def _hook_stdin(*, session_id=SESSION, event="PreLLMCall"):
    return json.dumps(
        {
            "hook_event_name": event,
            "request_id": "req-1:0:1",
            "attempt": 1,
            "step": 0,
            "session_id": session_id,
            "turn_id": "turn-1",
            "cwd": "/tmp/wt",
            "message_count": 1,
            "tool_count": 0,
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
            "cli_agent_orchestrator.services.muse_context_restore",
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
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Edit",
                    "hooks": [{"type": "command", "command": "./scripts/check.sh"}],
                }
            ],
            "PreLLMCall": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "./scripts/note.sh"}],
                }
            ],
        }
    }


def _cmd(terminal_id, tag="cmd"):
    return f"cao-muse-hook-context --terminal {terminal_id} --conduct-bin /bin/conduct #{tag}"


class TestHookInputParsing:
    def test_session_id_parsed(self):
        assert restore.parse_hook_input(_hook_stdin()) == SESSION

    def test_wrong_event_is_refused(self):
        raw = _hook_stdin(event="SessionStart")
        assert restore.parse_hook_input(raw) is None

    def test_sibling_shaped_ids_are_refused(self):
        """Another harness's id fields name nothing on this path."""
        assert (
            restore.parse_hook_input(
                json.dumps({"hook_event_name": "PreLLMCall", "sessionId": SESSION}).encode()
            )
            is None
        )
        assert (
            restore.parse_hook_input(
                json.dumps({"hook_event_name": "PreLLMCall", "conversationId": SESSION}).encode()
            )
            is None
        )

    @pytest.mark.parametrize(
        "raw",
        [
            b"",
            b"not json",
            b"[1, 2]",
            b'"str"',
            json.dumps({"hook_event_name": "PreLLMCall", "session_id": ""}).encode(),
            json.dumps({"hook_event_name": "PreLLMCall", "session_id": "   "}).encode(),
            json.dumps({"hook_event_name": "PreLLMCall", "session_id": 42}).encode(),
            json.dumps({"hook_event_name": "PreLLMCall"}).encode(),
            json.dumps({"session_id": SESSION}).encode(),
        ],
    )
    def test_unparseable_shapes_are_silent(self, raw):
        assert restore.parse_hook_input(raw) is None


class TestHookFileComposition:
    def test_install_creates_exact_prellmcall_shape(self, tmp_path):
        """Golden pin of the registration shape proven to fire.

        The probe workspace that fired this shape was overwritten by
        later probes, so its exact bytes are unrecoverable and are NOT
        asserted here; what the preserved evidence pins (observed stdin
        at /private/tmp/muserev/stdin.log, hook terminal + context
        effect at .../T/muserev2/export2.json) is this matcher/handler
        shape, asserted exactly below.
        """
        path, created, degraded = restore.install(
            tmp_path, terminal_id="t1", command="/w wrapper --terminal t1"
        )
        assert degraded is None
        assert created is True
        assert path == tmp_path / ".muse" / "hooks.json"
        data = json.loads(path.read_text())
        assert data == {
            "hooks": {
                "PreLLMCall": [
                    {
                        "matcher": "*",
                        "hooks": [{"type": "command", "command": "/w wrapper --terminal t1"}],
                    }
                ]
            }
        }

    def test_install_preserves_user_entries(self, tmp_path):
        user = _user_hooks_config()
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(json.dumps(user, indent=2))

        _, _, degraded = restore.install(tmp_path, terminal_id="t1", command=_cmd("t1"))

        assert degraded is None
        after = json.loads(hooks.read_text())
        assert after["hooks"]["PreToolUse"] == user["hooks"]["PreToolUse"]
        assert after["hooks"]["PreLLMCall"][0] == user["hooks"]["PreLLMCall"][0]
        assert len(after["hooks"]["PreLLMCall"]) == 2

    def test_install_is_idempotent_and_scoped_per_terminal(self, tmp_path):
        restore.install(tmp_path, terminal_id="t1", command=_cmd("t1", "one"))
        restore.install(tmp_path, terminal_id="t1", command=_cmd("t1", "two"))
        restore.install(tmp_path, terminal_id="t2", command=_cmd("t2", "two"))
        entries = json.loads((tmp_path / ".muse" / "hooks.json").read_text())["hooks"]["PreLLMCall"]
        t1 = [e for e in entries if "--terminal t1" in json.dumps(e)]
        t2 = [e for e in entries if "--terminal t2" in json.dumps(e)]
        assert len(t1) == 1 and len(t2) == 1
        assert "#two" in json.dumps(t1[0])

    def test_uninstall_removes_only_its_own_entries(self, tmp_path):
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(json.dumps(_user_hooks_config(), indent=2))
        restore.install(tmp_path, terminal_id="t1", command=_cmd("t1"))
        restore.install(tmp_path, terminal_id="t2", command=_cmd("t2"))

        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=False) is True

        data = json.loads(hooks.read_text())
        blob = json.dumps(data)
        assert "--terminal t1" not in blob
        assert "--terminal t2" in blob
        assert data["hooks"]["PreToolUse"] == _user_hooks_config()["hooks"]["PreToolUse"]
        assert "./scripts/note.sh" in blob

    def test_uninstall_removes_file_only_when_we_created_it(self, tmp_path):
        path, created, _ = restore.install(tmp_path, terminal_id="t1", command=_cmd("t1"))
        assert created is True
        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=True) is True
        assert not path.exists()

    def test_uninstall_keeps_preexisting_file(self, tmp_path):
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(json.dumps(_user_hooks_config(), indent=2))
        restore.install(tmp_path, terminal_id="t1", command=_cmd("t1"))
        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=False) is True
        assert hooks.exists()
        assert (
            json.loads(hooks.read_text())["hooks"]["PreToolUse"]
            == _user_hooks_config()["hooks"]["PreToolUse"]
        )

    def test_uninstall_missing_everything_is_quiet(self, tmp_path):
        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=False) is False
        assert restore.uninstall(tmp_path, terminal_id="t1", created_file=True) is False

    def test_malformed_config_refuses_without_clobbering(self, tmp_path):
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text("{ not json")

        path, created, degraded = restore.install(tmp_path, terminal_id="t1", command=_cmd("t1"))

        assert path is None and created is False and degraded is not None
        assert hooks.read_text() == "{ not json"

    def test_non_object_config_refuses_without_clobbering(self, tmp_path):
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text('["a list, not hooks"]')

        path, _, degraded = restore.install(tmp_path, terminal_id="t1", command=_cmd("t1"))

        assert path is None and degraded is not None
        assert hooks.read_text() == '["a list, not hooks"]'

    def test_composition_does_not_mutate_its_input(self):
        user = _user_hooks_config()
        snapshot = json.loads(json.dumps(user))
        restore.with_context_restore(user, terminal_id="t1", command="cmd")
        restore.without_context_restore(user, terminal_id="t1")
        assert user == snapshot

    def test_non_dict_hooks_key_degrades_preserving_bytes(self, tmp_path):
        """A present-but-unmergeable "hooks" key refuses, never replaces."""
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text('{"hooks": ["not", "an", "object"], "other": 1}')

        path, created, degraded = restore.install(tmp_path, terminal_id="t1", command=_cmd("t1"))

        assert path is None and created is False
        assert degraded is not None and '"hooks" key is not an object' in degraded
        assert hooks.read_text() == '{"hooks": ["not", "an", "object"], "other": 1}'

    def test_non_list_prellmcall_degrades_preserving_bytes(self, tmp_path):
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        before = '{"hooks": {"PreLLMCall": {"matcher": "*"}, "Stop": []}}'
        hooks.write_text(before)

        path, created, degraded = restore.install(tmp_path, terminal_id="t1", command=_cmd("t1"))

        assert path is None and created is False
        assert degraded is not None and '"hooks.PreLLMCall" is not a list' in degraded
        assert hooks.read_text() == before

    def test_unrelated_event_shapes_pass_through(self, tmp_path):
        """Shapes this adapter never writes are none of its business."""
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text('{"hooks": {"Stop": "legacy-string", "other_top": {"x": 1}}}')

        path, created, degraded = restore.install(tmp_path, terminal_id="t1", command=_cmd("t1"))

        assert degraded is None and created is False
        data = json.loads(hooks.read_text())
        assert data["hooks"]["Stop"] == "legacy-string"
        assert data["hooks"]["other_top"] == {"x": 1}
        assert len(data["hooks"]["PreLLMCall"]) == 1

    def test_pure_composer_raises_typed_error_instead_of_replacing(self):
        with pytest.raises(ValueError, match='"hooks" key is not an object'):
            restore.with_context_restore({"hooks": []}, terminal_id="t1", command=_cmd("t1"))
        with pytest.raises(ValueError, match='"hooks.PreLLMCall" is not a list'):
            restore.with_context_restore(
                {"hooks": {"PreLLMCall": {}}}, terminal_id="t1", command=_cmd("t1")
            )


class TestWrapperProcess:
    def test_ok_answer_injects_additional_context(self, fake):
        proc = _run_wrapper_process(_hook_stdin(), fake=fake, extra_args=["--terminal", "t1"])
        assert proc.returncode == 0
        out = json.loads(proc.stdout)
        assert set(out) == {"hookSpecificOutput"}
        specific = out["hookSpecificOutput"]
        assert specific["hookEventName"] == "PreLLMCall"
        assert "CAO goal restoration" in specific["additionalContext"]
        assert "Ship the thing" in specific["additionalContext"]

    def test_projection_called_exactly_once_globally_without_hints(self, fake):
        """One global query per invocation: no hint, no generation."""
        _run_wrapper_process(_hook_stdin(), fake=fake, extra_args=["--terminal", "t1"])
        argvs = _logged_argvs(fake)
        assert len(argvs) == 1
        argv = argvs[0]
        assert argv[:4] == ["goal", "hook-context", "--harness", "muse_cli"]
        assert "--native-session-id" in argv
        assert argv[argv.index("--native-session-id") + 1] == SESSION
        assert "--terminal" not in argv
        assert "--terminal-generation" not in argv

    def test_stale_and_terminal_callbacks_inject_nothing(self, fake):
        for result_type in ("stale-generation", "dead-incarnation", "ambiguous", "no-worker"):
            fake["canned"].write_text(json.dumps(_typed_answer(result_type)))
            proc = _run_wrapper_process(_hook_stdin(), fake=fake)
            assert proc.returncode == 0
            assert json.loads(proc.stdout) == {}
        for state in ("satisfied", "cancelled"):
            fake["canned"].write_text(json.dumps(_ok_answer(state=state)))
            proc = _run_wrapper_process(_hook_stdin(), fake=fake)
            assert proc.returncode == 0
            assert json.loads(proc.stdout) == {}

    def test_missing_assignment_names_next_call_without_goal_text(self, fake):
        fake["canned"].write_text(json.dumps(_typed_answer("no-assignment")))
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert proc.returncode == 0
        body = json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
        assert body.startswith("CAO context restoration unavailable")
        assert "next model call" in body
        assert "Ship the thing" not in body
        assert "SessionStart" not in body

    def test_conduct_failure_degrades_to_empty_object(self, fake):
        fake["env"]["CONDUCT_MODE"] = "fail"
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {}

    def test_conduct_garbage_degrades_to_empty_object(self, fake):
        fake["env"]["CONDUCT_MODE"] = "garbage"
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {}

    def test_unparseable_input_never_calls_conduct(self, fake):
        proc = _run_wrapper_process(b"not json", fake=fake)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {}
        assert _logged_argvs(fake) == []

    def test_long_goal_is_cut_with_marker_inside_the_bound(self, fake):
        fake["canned"].write_text(json.dumps(_ok_answer(objective="x" * 20000)))
        proc = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t1", "--max-chars", "8000"]
        )
        body = json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
        assert len(body) <= 8000
        assert "truncated here" in body


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
    needs a LIVE binding agreeing with the claimed session, two live
    binders answer ``ambiguous``, and the ok leg reports the
    live-matched terminal. The wrapper injects only when that terminal
    exactly equals its baked self — so at most the unique owner
    injects, from one query and one current answer.
    """

    def _bound(self, fake, bindings):
        return dict(fake["env"], CONDUCT_BINDINGS=json.dumps(bindings))

    def test_two_live_binders_inject_nothing_either_side(self, fake):
        """The same session live under two terminals: ambiguous."""
        env = self._bound(fake, {"t1": SESSION, "t2": SESSION})
        first = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t1"], extra_env=env
        )
        second = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t2"], extra_env=env
        )
        assert first.returncode == 0 and second.returncode == 0
        assert json.loads(first.stdout) == {}
        assert json.loads(second.stdout) == {}
        argvs = _logged_argvs(fake)
        assert len(argvs) == 2
        assert all("--terminal" not in argv for argv in argvs)

    def test_stale_entry_and_live_entry_inject_exactly_once(self, fake):
        """A lingering entry from a dead terminal stays silent."""
        env = self._bound(fake, {"t2": SESSION})
        stale = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t1"], extra_env=env
        )
        live = _run_wrapper_process(
            _hook_stdin(), fake=fake, extra_args=["--terminal", "t2"], extra_env=env
        )
        assert stale.returncode == 0 and live.returncode == 0
        assert json.loads(stale.stdout) == {}
        body = json.loads(live.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "Ship the thing" in body

    def test_foreign_session_never_projects(self, fake):
        env = self._bound(fake, {"t1": SESSION})
        proc = _run_wrapper_process(
            _hook_stdin(session_id=OTHER_SESSION),
            fake=fake,
            extra_args=["--terminal", "t1"],
            extra_env=env,
        )
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {}

    def test_unbaked_wrapper_discards_even_unique_owner(self, fake):
        env = self._bound(fake, {"t1": SESSION})
        proc = _run_wrapper_process(_hook_stdin(), fake=fake, extra_env=env)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {}


class TestHookFileConcurrency:
    """Additive read-modify-write alone is not concurrency-proof.

    Install/uninstall serialize on a sidecar lock and write
    temp-plus-atomic-replace; these tests prove the serialization is
    real, not the absence of a race in a lucky interleaving.
    """

    def _seed_user_config(self, workspace):
        hooks = workspace / ".muse" / "hooks.json"
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
        restore.install(workspace, terminal_id="t1", command=_cmd("t1"))
        path = restore.hooks_file(workspace)
        holder = open(restore._lock_path(path), "a+b")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        try:
            assert restore.uninstall(workspace, terminal_id="t1", created_file=False) is False
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()
        data = json.loads(path.read_text())
        assert "--terminal t1" in json.dumps(data)

    def test_concurrent_installs_lose_no_entries(self, tmp_path):
        import concurrent.futures

        workspace = tmp_path / "ws"
        self._seed_user_config(workspace)

        def work(i):
            terminal = f"t{i % 4}"
            if i % 7 == 6:
                restore.uninstall(workspace, terminal_id=terminal, created_file=False)
            else:
                restore.install(
                    workspace, terminal_id=terminal, command=_cmd(terminal, tag=f"n{i}")
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(96)))

        data = json.loads((workspace / ".muse" / "hooks.json").read_text())
        assert data["hooks"]["PreToolUse"] == _user_hooks_config()["hooks"]["PreToolUse"]
        assert "./scripts/note.sh" in json.dumps(data)


class TestAttach:
    def test_attach_installs_and_records_mechanism(self, tmp_path):
        installation, degraded = restore.attach(
            tmp_path,
            terminal_id="t1",
            wrapper_executable=sys.executable,
            conduct_binary=sys.executable,
        )
        assert degraded is None
        assert installation["mechanism"] == "muse-PreLLMCall:additionalContext"
        assert installation["terminal_id"] == "t1"
        assert installation["terminal_generation"] is None
        data = json.loads((tmp_path / ".muse" / "hooks.json").read_text())
        command = data["hooks"]["PreLLMCall"][0]["hooks"][0]["command"]
        assert "--terminal t1" in command
        assert "--terminal-generation" not in command

    def test_attach_default_resolves_muse_wrapper_not_sibling(self, tmp_path, monkeypatch):
        """The production default (no explicit wrapper) bakes THIS lane."""
        bindir = tmp_path / "bin"
        bindir.mkdir()
        wrapper = bindir / "cao-muse-hook-context"
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
        command = json.loads((tmp_path / "ws" / ".muse" / "hooks.json").read_text())["hooks"][
            "PreLLMCall"
        ][0]["hooks"][0]["command"]
        assert str(wrapper) in command
        assert "cao-claude-hook-context" not in command
        assert "cao-agy-hook-context" not in command

    def test_attach_degrades_when_executables_unresolvable(self, tmp_path):
        installation, degraded = restore.attach(
            tmp_path,
            terminal_id="t1",
            wrapper_executable="/nonexistent/wrapper",
            conduct_binary="/nonexistent/conduct",
        )
        assert installation is None
        assert degraded is not None
        assert not (tmp_path / ".muse" / "hooks.json").exists()

    def test_attach_degrades_without_clobbering_bad_config(self, tmp_path):
        hooks = tmp_path / ".muse" / "hooks.json"
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
        from cli_agent_orchestrator.providers.muse_cli import MuseCliProvider

        return MuseCliProvider(
            terminal_id="t1",
            session_name="test-session",
            window_name="window-0",
            **kwargs,
        )

    def test_no_workspace_leaves_restoration_uninstalled(self):
        provider = self._make_provider()
        provider._install_context_restore()
        assert provider.context_restoration is None

    def test_install_registers_terminal_binding(self, tmp_path, monkeypatch):
        from cli_agent_orchestrator.services import muse_context_restore as muse

        fake_wrapper = str(tmp_path / "cao-muse-hook-context")
        monkeypatch.setattr(
            muse.claude,
            "resolve_wrapper_executable",
            lambda explicit=None: fake_wrapper,
        )
        monkeypatch.setattr(
            muse.claude, "resolve_conduct_binary", lambda explicit=None: sys.executable
        )
        provider = self._make_provider(hooks_workspace=str(tmp_path))
        provider._install_context_restore()
        installation = provider.context_restoration
        assert installation is not None
        assert installation["mechanism"] == "muse-PreLLMCall:additionalContext"
        assert installation["terminal_id"] == "t1"
        data = json.loads((tmp_path / ".muse" / "hooks.json").read_text())
        assert "--terminal t1" in json.dumps(data)

    def test_cleanup_uninstalls_only_its_own_entries(self, tmp_path, monkeypatch):
        from cli_agent_orchestrator.services import muse_context_restore as muse

        fake_wrapper = str(tmp_path / "cao-muse-hook-context")
        monkeypatch.setattr(
            muse.claude,
            "resolve_wrapper_executable",
            lambda explicit=None: fake_wrapper,
        )
        monkeypatch.setattr(
            muse.claude, "resolve_conduct_binary", lambda explicit=None: sys.executable
        )
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(json.dumps(_user_hooks_config(), indent=2))
        first = self._make_provider(hooks_workspace=str(tmp_path))
        first._install_context_restore()
        other = self._make_provider(hooks_workspace=str(tmp_path))
        other.terminal_id = "t2"
        other._install_context_restore()

        first.cleanup()

        data = json.loads(hooks.read_text())
        blob = json.dumps(data)
        assert "--terminal t1" not in blob
        assert "--terminal t2" in blob
        assert "./scripts/note.sh" in blob
        assert first.context_restoration is None

    def test_install_failure_never_raises(self, tmp_path, monkeypatch):
        from cli_agent_orchestrator.services import muse_context_restore as muse

        fake_wrapper = str(tmp_path / "cao-muse-hook-context")
        monkeypatch.setattr(
            muse.claude,
            "resolve_wrapper_executable",
            lambda explicit=None: fake_wrapper,
        )
        monkeypatch.setattr(
            muse.claude, "resolve_conduct_binary", lambda explicit=None: sys.executable
        )
        hooks = tmp_path / ".muse" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text("{ broken")
        provider = self._make_provider(hooks_workspace=str(tmp_path))
        provider._install_context_restore()  # must not raise
        assert provider.context_restoration is None
        assert hooks.read_text() == "{ broken"
        provider.cleanup()  # must not raise either
