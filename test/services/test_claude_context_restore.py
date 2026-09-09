"""Claude direct passive goal restoration (cond-0845 slice 2, Claude first).

Two reproduced-failure classes, one theme: a hook that restores the wrong
text is worse than a hook that restores nothing.

*The dropped hook.* Composition that rebuilds the ``--settings`` payload
instead of appending to it silently drops the readiness SessionStart entry
— and the launch then never proves its identity. Every composition test
below asserts on the real ``claude_native_readiness.prepare`` payload and
pins the readiness entry byte-identical, so a rebuild fails loudly.

*The wrong text.* A wrapper that echoes its stdin, guesses a goal, or
re-injects a stale callback restores another assignment's context into a
live session. These tests drive the real wrapper process (stdin in,
stdout out) against a fake ``conduct`` honouring the exact slice-1 CLI
contract, and pin: only ``goal hook-context`` is ever invoked, stale
callbacks restore nothing, and unavailable reads say so without goal
text.

Fixture boundary: the fake ``conduct`` stands in for the live server
reads behind ``conduct goal hook-context`` (conductor PR #349,
``1f1e415f``). Its canned answers use the real envelope
(``cao-hook-context-v1``) and the real result types, but no test here
proves the conductor side — that proof belongs to the conductor lane.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from cli_agent_orchestrator.services import claude_context_restore as restore
from cli_agent_orchestrator.services import claude_native_readiness as readiness

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"

SESSION = "11111111-1111-4111-8111-111111111111"

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
# loudly rather than replaying a goal onto it.
if "--harness" not in argv or argv[argv.index("--harness") + 1] != "claude_code":
    sys.stderr.write("wrong harness claim\\n")
    sys.exit(2)
if "--native-session-id" not in argv:
    sys.stderr.write("missing native session claim\\n")
    sys.exit(2)
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


def _ok_answer(*, version="v3", objective="Ship the thing"):
    return {
        "ok": True,
        "schema": "cao-hook-context-v1",
        "result_type": "ok",
        "detail": None,
        "recovery": None,
        "identity": {
            "harness": "claude_code",
            "native_session_id": SESSION,
            "terminal_id": "t1",
            "terminal_generation": "g1",
            "generation_fence": "verified",
        },
        "goal": {
            "goal_id": "g-1",
            "state": "open",
            "goal_version": version,
            "objective": objective,
            "requirements_outstanding": ["r1"],
            "requirements_outstanding_count": 1,
            "completion_requirements_truncated": False,
            "active_hold": None,
            "next_action": "ordinary work may continue; this read starts no turn",
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
    """An executable fake ``conduct`` honouring the slice-1 CLI contract.

    It asserts nothing itself: it logs every argv it receives (for the
    zero-turn pin) and replays the canned answer file (for version-change
    and verdict coverage). Behaviour modes via ``CONDUCT_MODE``.
    """
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


def _hook_stdin(*, session_id=SESSION, source="compact"):
    return json.dumps(
        {
            "session_id": session_id,
            "cwd": "/tmp/wt",
            "transcript_path": "/tmp/t.jsonl",
            "hook_event_name": "SessionStart",
            "source": source,
        }
    ).encode()


def _run_wrapper_process(stdin_bytes, *, fake, extra_args=()):
    """The real wrapper process: stdin in, hook JSON out."""
    env = dict(fake["env"], PYTHONPATH=str(SRC_DIR))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.claude_context_restore",
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


class TestLaunchSettingsComposition:
    def test_restore_entries_append_to_the_real_readiness_payload(self, tmp_path):
        """Composition is additive: the readiness entry survives byte-identical."""
        prepared = readiness.prepare(tmp_path, "t1", "g1")
        before = json.loads(json.dumps(prepared["settings"]))

        composed, degraded = restore.attach_to_launch_settings(
            prepared["settings"],
            terminal_id="t1",
            generation="g1",
            wrapper_executable=sys.executable,
            conduct_binary=sys.executable,
        )

        assert degraded is None
        entries = composed["hooks"]["SessionStart"]
        assert entries[0] == before["hooks"]["SessionStart"][0]
        # The readiness entry carries no matcher; the two restore entries do.
        assert "matcher" not in entries[0]
        assert [entry.get("matcher") for entry in entries] == [
            None,
            "compact",
            "startup|resume",
        ]
        for entry in entries[1:]:
            assert entry["hooks"][0]["type"] == "command"
            assert "--terminal t1" in entry["hooks"][0]["command"]
            assert "--terminal-generation g1" in entry["hooks"][0]["command"]

    def test_composition_does_not_mutate_its_input(self, tmp_path):
        prepared = readiness.prepare(tmp_path, "t1", "g1")
        snapshot = json.loads(json.dumps(prepared["settings"]))
        restore.attach_to_launch_settings(
            prepared["settings"],
            terminal_id="t1",
            generation="g1",
            wrapper_executable=sys.executable,
            conduct_binary=sys.executable,
        )
        assert prepared["settings"] == snapshot

    def test_user_and_project_entries_are_preserved(self, tmp_path):
        """A pre-existing user-style entry is carried over byte-identical."""
        prepared = readiness.prepare(tmp_path, "t1", "g1")
        user_entry = {
            "matcher": "Write|Edit",
            "hooks": [{"type": "command", "command": "/home/u/hooks/check.sh"}],
        }
        prepared["settings"]["hooks"].setdefault("PostToolUse", []).append(user_entry)
        user_session_entry = {"hooks": [{"type": "command", "command": "echo hi"}]}
        prepared["settings"]["hooks"]["SessionStart"].append(user_session_entry)

        composed, _ = restore.attach_to_launch_settings(
            prepared["settings"],
            terminal_id="t1",
            generation="g1",
            wrapper_executable=sys.executable,
            conduct_binary=sys.executable,
        )

        assert composed["hooks"]["PostToolUse"] == [user_entry]
        assert composed["hooks"]["SessionStart"][1] == user_session_entry
        assert [e.get("matcher") for e in composed["hooks"]["SessionStart"][2:]] == [
            "compact",
            "startup|resume",
        ]

    def test_unresolvable_executables_degrade_to_readiness_only(self, tmp_path):
        prepared = readiness.prepare(tmp_path, "t1", "g1")
        composed, degraded = restore.attach_to_launch_settings(
            prepared["settings"],
            terminal_id="t1",
            generation="g1",
            wrapper_executable="/nonexistent/wrapper",
            conduct_binary="/nonexistent/conduct",
        )
        # Degradation returns the input unchanged with a reason: the launch
        # is never failed over a restoration hook.
        assert composed == prepared["settings"]
        assert degraded is not None and "readiness-only" in degraded

    def test_mint_and_resume_paths_compose_the_same_binding(self, tmp_path, monkeypatch):
        """Both launch forms route through one helper with their generation."""
        from cli_agent_orchestrator.services import managed_launch_v2 as v2

        monkeypatch.setattr(
            restore, "resolve_wrapper_executable", lambda explicit=None: sys.executable
        )
        monkeypatch.setattr(restore, "resolve_conduct_binary", lambda explicit=None: sys.executable)
        prepared = readiness.prepare(tmp_path, "t9", "gen-7")
        record = {"terminal_id": "t9", "generation": "gen-7"}
        out = v2._with_claude_context_restore(
            {"readiness_path": prepared["readiness_path"], "settings": prepared["settings"]},
            record=record,
        )
        entries = out["settings"]["hooks"]["SessionStart"]
        assert [e.get("matcher") for e in entries] == [None, "compact", "startup|resume"]
        assert "--terminal-generation gen-7" in entries[1]["hooks"][0]["command"]
        assert out["readiness_path"] == prepared["readiness_path"]


class TestWrapperProcessContract:
    def test_compact_restores_the_current_goal(self, fake):
        proc = _run_wrapper_process(
            _hook_stdin(),
            fake=fake,
            extra_args=["--terminal", "t1", "--terminal-generation", "g1"],
        )
        assert proc.returncode == 0
        output = json.loads(proc.stdout.decode())
        assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        context = output["hookSpecificOutput"]["additionalContext"]
        assert "CAO goal restoration" in context
        assert "Ship the thing" in context
        assert "version 'v3'" in context

    def test_repeated_compact_tracks_goal_version_changes(self, fake):
        first = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert (
            "version 'v3'"
            in json.loads(first.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        )

        fake["canned"].write_text(json.dumps(_ok_answer(version="v5")))
        second = _run_wrapper_process(_hook_stdin(), fake=fake)
        context = json.loads(second.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        assert "version 'v5'" in context
        assert "version 'v3'" not in context

    def test_startup_before_admission_is_truthfully_unavailable(self, fake):
        """Startup may precede goal availability: a note, never a goal."""
        fake["canned"].write_text(json.dumps(_typed_answer("no-assignment")))
        proc = _run_wrapper_process(_hook_stdin(source="startup"), fake=fake)
        assert proc.returncode == 0
        context = json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        assert "CAO context restoration unavailable" in context
        assert "Ship the thing" not in context

    def test_the_next_eligible_hook_recovers_the_latest_goal(self, fake):
        """Unavailable now, restored later: the hook re-reads every event."""
        fake["canned"].write_text(json.dumps(_typed_answer("no-assignment")))
        before = _run_wrapper_process(_hook_stdin(source="startup"), fake=fake)
        assert (
            "unavailable"
            in json.loads(before.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        )

        fake["canned"].write_text(json.dumps(_ok_answer(version="v9")))
        after = _run_wrapper_process(_hook_stdin(source="resume"), fake=fake)
        context = json.loads(after.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        assert "version 'v9'" in context
        assert "unavailable" not in context

    @pytest.mark.parametrize(
        "result_type", ["stale-generation", "ambiguous", "no-worker", "dead-incarnation"]
    )
    def test_discard_verdicts_restore_nothing(self, fake, result_type):
        """A former or foreign binding must never re-enter context."""
        fake["canned"].write_text(json.dumps(_typed_answer(result_type)))
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert proc.returncode == 0
        assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""

    def test_session_rotation_is_discarded_then_recovered(self, fake):
        """Old callback after rotation: silent; live hook: latest goal."""
        fake["canned"].write_text(json.dumps(_typed_answer("stale-generation", detail="rotated")))
        stale = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert json.loads(stale.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""
        fake["canned"].write_text(json.dumps(_ok_answer(version="v11")))
        live = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert (
            "version 'v11'"
            in json.loads(live.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        )


class TestWrapperInputFailures:
    @pytest.mark.parametrize(
        "stdin_bytes",
        [
            b"",
            b"not json",
            json.dumps([1, 2]).encode(),
            json.dumps({"hook_event_name": "SessionStart"}).encode(),
            json.dumps({"session_id": 42}).encode(),
            json.dumps({"session_id": "   "}).encode(),
        ],
    )
    def test_unparseable_input_restores_nothing_and_exits_zero(self, fake, stdin_bytes):
        proc = _run_wrapper_process(stdin_bytes, fake=fake)
        assert proc.returncode == 0
        output = json.loads(proc.stdout.decode())
        assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert output["hookSpecificOutput"]["additionalContext"] == ""
        assert _logged_argvs(fake) == []

    def test_stdout_carries_only_the_json_object(self, fake):
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        # A shell profile or stray print would break schema validation.
        json.loads(proc.stdout.decode())


class TestLookupFailureIsScopedToRestoration:
    def test_conduct_failure_exits_zero_with_empty_context(self, fake):
        fake["env"]["CONDUCT_MODE"] = "fail"
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert proc.returncode == 0
        assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""
        assert proc.stderr.decode() != ""

    def test_malformed_answer_exits_zero_with_empty_context(self, fake):
        fake["env"]["CONDUCT_MODE"] = "garbage"
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        assert proc.returncode == 0
        assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""

    def test_missing_conduct_binary_exits_zero_with_empty_context(self, fake):
        env = dict(fake["env"], PYTHONPATH=str(SRC_DIR))
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "cli_agent_orchestrator.services.claude_context_restore",
                "--conduct-bin",
                str(fake["script"]) + "-missing",
            ],
            input=_hook_stdin(),
            capture_output=True,
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0
        assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""

    def test_slow_conduct_degrades_within_the_configured_timeout(self, fake):
        fake["env"]["CONDUCT_MODE"] = "sleep"
        fake["env"]["CONDUCT_SLEEP"] = "30"
        proc = _run_wrapper_process(_hook_stdin(), fake=fake, extra_args=["--timeout", "1"])
        assert proc.returncode == 0
        assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""


class TestZeroTurnsAndNoForeignVerbs:
    def test_only_goal_hook_context_is_ever_invoked(self, fake):
        _run_wrapper_process(
            _hook_stdin(),
            fake=fake,
            extra_args=["--terminal", "t1", "--terminal-generation", "g1"],
        )
        argvs = _logged_argvs(fake)
        assert len(argvs) == 1
        assert argvs[0][:3] == ["goal", "hook-context", "--harness"]
        assert argvs[0][argvs[0].index("--harness") + 1] == "claude_code"
        assert "--native-session-id" in argvs[0]
        assert argvs[0][argvs[0].index("--native-session-id") + 1] == SESSION
        assert argvs[0][argvs[0].index("--terminal-generation") + 1] == "g1"
        assert argvs[0][argvs[0].index("--terminal") + 1] == "t1"

    def test_the_wrapper_has_no_delivery_or_resume_vocabulary(self):
        """Steer/send/submit/turn-start verbs cannot survive here as code.

        The wrapper's only subprocess is the read-only projection; any
        future edit introducing a delivery verb fails this test. The scan
        runs on the AST (prose in docstrings/comments is documentation,
        not a code path), and the pinned argv is asserted token by token.
        """
        import ast as _ast

        tree = _ast.parse(Path(restore.__file__).read_text())
        code_tokens: list[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Constant) and isinstance(node.value, str):
                code_tokens.append(node.value)
            elif isinstance(node, _ast.Name):
                code_tokens.append(node.id)
            elif isinstance(node, _ast.Attribute):
                code_tokens.append(node.attr)
        for verb in ("steer", "send", "submit_turn", "release_hold", "resume_paused"):
            assert verb not in code_tokens, f"forbidden verb {verb!r} in wrapper code"
        # Exactly one subprocess call site exists in the module (the
        # projection read): a second spawn cannot hide beside it.
        spawns = [
            node
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Call)
            and isinstance(node.func, _ast.Attribute)
            and isinstance(node.func.value, _ast.Name)
            and node.func.value.id == "subprocess"
        ]
        assert len(spawns) == 1
        assert spawns[0].func.attr == "run"
        argv = restore.build_conduct_argv(
            conduct_binary="conduct",
            native_session_id=SESSION,
            terminal_id="t1",
            terminal_generation="g1",
        )
        assert argv[1:3] == ["goal", "hook-context"]
        assert not any(token in ("steer", "send") for token in argv)


class TestOutputBounds:
    def test_long_goal_text_is_truncated_with_an_explicit_marker(self, fake):
        fake["canned"].write_text(json.dumps(_ok_answer(objective="word " * 5000)))
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        context = json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        assert len(context) <= restore.MAX_ADDITIONAL_CONTEXT_CHARS
        assert restore.TRUNCATION_MARKER in context
        # The provider cap is never approached silently: the marker proves
        # the cut happened here, not in a provider spill file.
        assert len(context) <= 10_000

    def test_short_goal_text_passes_through_unmarked(self, fake):
        proc = _run_wrapper_process(_hook_stdin(), fake=fake)
        context = json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        assert restore.TRUNCATION_MARKER not in context
        assert "Ship the thing" in context

    def test_bound_is_exact_at_the_boundary(self):
        body = "x" * restore.MAX_ADDITIONAL_CONTEXT_CHARS
        assert restore.apply_output_bound(body) == body
        over = "x" * (restore.MAX_ADDITIONAL_CONTEXT_CHARS + 1)
        cut = restore.apply_output_bound(over)
        assert len(cut) <= restore.MAX_ADDITIONAL_CONTEXT_CHARS
        assert cut.endswith(restore.TRUNCATION_MARKER)

    def test_render_never_exceeds_the_bound_without_a_marker(self):
        for version in ("v1", "v2"):
            answer = _ok_answer(version=version, objective="y " * 9000)
            context = restore.render_restoration(answer)
            assert context is not None
            bounded = restore.apply_output_bound(context)
            assert len(bounded) <= restore.MAX_ADDITIONAL_CONTEXT_CHARS
            assert restore.TRUNCATION_MARKER in bounded
