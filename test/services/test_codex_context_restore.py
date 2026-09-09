"""Codex direct passive goal restoration (cond-0845 slice 2, Codex).

Two reproduced-failure classes, one theme: a hook that restores the wrong
text is worse than a hook that restores nothing.

*The parent in the child.* Codex gives a subagent's SessionStart the
parent session id with no distinguishing field, so a ``startup`` matcher
would inject the parent's goal into every child at spawn. The tests pin
the structural defense: the installed config matches ``^compact$`` only
(parsed TOML, not substring search), no ``SubagentStart``/``SubagentStop``
entries exist, and the wrapper performs no payload inspection that could
amount to a guessed child mapping.

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
import tomllib
from pathlib import Path

import pytest

from cli_agent_orchestrator.services import codex_context_restore as restore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"

SESSION = "thr_abc123def456"

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
if "--harness" not in argv or argv[argv.index("--harness") + 1] != "codex":
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


def _ok_answer(**overrides):
    answer = {
        "ok": True,
        "schema": "cao-hook-context-v1",
        "result_type": "ok",
        "detail": None,
        "identity": {
            "harness": "codex",
            "harness_fence": "verified",
            "native_session_id": SESSION,
            "terminal_id": "term-1",
            "terminal_generation": "gen-1",
            "generation_fence": "verified",
            "agent_id": "agent-1",
            "task_occurrence_id": "occ-1",
        },
        "goal": {
            "goal_id": "goal-1",
            "state": "open",
            "goal_version": 3,
            "objective": "implement the thing",
            "requirements_outstanding": ["final-report"],
            "requirements_outstanding_count": 1,
            "report_state": "outstanding",
            "evidence_count": 0,
            "evidence_refs": [],
            "active_hold": None,
            "next_action": "ordinary work may proceed through policy",
        },
        "authority": "read-only",
    }
    answer.update(overrides)
    return answer


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    """A fake ``conduct`` binary plus canned answers and an argv log."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    conduct = bindir / "conduct"
    conduct.write_text(FAKE_CONDUCT, encoding="utf-8")
    conduct.chmod(0o755)
    wrapper = bindir / "cao-codex-hook-context"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import sys; sys.path.insert(0, " + repr(str(SRC_DIR)) + ")\n"
        "from cli_agent_orchestrator.services.codex_context_restore import main\n"
        "raise SystemExit(main())\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    log = tmp_path / "argv.log"
    log.write_text("", encoding="utf-8")
    canned = tmp_path / "canned.json"
    canned.write_text(json.dumps(_ok_answer()), encoding="utf-8")
    monkeypatch.setenv("CONDUCT_ARGV_LOG", str(log))
    monkeypatch.setenv("CONDUCT_CANNED", str(canned))
    monkeypatch.setenv("CONDUCT_MODE", "ok")
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    return {
        "conduct": str(conduct),
        "wrapper": str(wrapper),
        "log": log,
        "canned": canned,
        "tmp": tmp_path,
    }


def _run_wrapper(fake, stdin_bytes, *argv):
    proc = subprocess.run(
        [fake["wrapper"], *argv],
        input=stdin_bytes,
        capture_output=True,
        timeout=30,
    )
    return proc


def _hook_input(session_id=SESSION, **extra):
    payload = {
        "session_id": session_id,
        "hook_event_name": "SessionStart",
        "source": "compact",
        "cwd": "/work",
        "model": "gpt-5",
    }
    payload.update(extra)
    return json.dumps(payload).encode("utf-8")


class TestConfigComposition:
    def test_base_config_preserved_byte_identical(self, tmp_path):
        base = 'model = "gpt-5"\n[profiles.main]\napproval_policy = "never"\n'
        command = "/bin/cao-codex-hook-context --terminal t"
        composed = restore.compose_managed_config(base, command=command)
        assert composed.startswith(base)
        # Parses as one TOML document with both the user table and ours.
        doc = tomllib.loads(composed)
        assert doc["profiles"]["main"]["approval_policy"] == "never"
        assert len(doc["hooks"]["SessionStart"]) == 1

    def test_installed_entry_matches_compact_only(self, tmp_path):
        composed = restore.compose_managed_config("", command="/bin/w --terminal t")
        doc = tomllib.loads(composed)
        entries = doc["hooks"]["SessionStart"]
        assert [e["matcher"] for e in entries] == ["^compact$"]
        handler = entries[0]["hooks"][0]
        assert handler["type"] == "command"
        assert handler["additionalContextLimit"] == restore.ADDITIONAL_CONTEXT_LIMIT

    def test_no_startup_resume_or_subagent_entries(self, tmp_path):
        """The child/parent defense, structural: subagent starts (source
        startup, parent session id, no marker) must never trigger
        restoration, so those matchers and events are absent by
        construction — not filtered at runtime from an indistinguishable
        payload."""
        composed = restore.compose_managed_config("", command="/bin/w")
        doc = tomllib.loads(composed)
        assert set(doc["hooks"].keys()) == {"SessionStart"}
        for entry in doc["hooks"]["SessionStart"]:
            assert entry["matcher"] == "^compact$"

    def test_missing_base_still_composes(self):
        composed = restore.compose_managed_config("", command="/bin/w")
        assert tomllib.loads(composed)["hooks"]["SessionStart"][0]["matcher"] == "^compact$"

    def test_command_with_spaces_is_quoted(self):
        composed = restore.compose_managed_config("", command="/x/y z --terminal t")
        doc = tomllib.loads(composed)
        assert doc["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "/x/y z --terminal t"


class TestPrivateHome:
    def _provider_home(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.toml").write_text('model = "gpt-5"\n', encoding="utf-8")
        (path / "auth.json").write_text('{"tok": "x"}\n', encoding="utf-8")
        (path / "thread_history_1.sqlite").write_text("threads", encoding="utf-8")
        sub = path / "sessions"
        sub.mkdir()
        (sub / "a.json").write_text("{}\n", encoding="utf-8")
        return path

    def test_layout_links_state_and_owns_config(self, tmp_path):
        provider = self._provider_home(tmp_path / "provider")
        home, degraded = restore.compose_codex_home(
            companion_dir=str(tmp_path / "companion"),
            terminal_id="term-1",
            generation="gen-1",
            provider_home=str(provider),
            base_config_text=(provider / "config.toml").read_text(encoding="utf-8"),
            command="/bin/w --terminal term-1",
        )
        assert degraded is None
        home = Path(home)
        assert (home / "config.toml").stat().st_mode & 0o777 == 0o600
        text = (home / "config.toml").read_text(encoding="utf-8")
        assert text.startswith('model = "gpt-5"\n')
        assert "^compact$" in text
        # Identity links back; generation-scoped state does not.
        assert (home / "auth.json").is_symlink()
        assert (home / "sessions").is_symlink()
        assert not (home / "thread_history_1.sqlite").exists()
        assert not (home / "thread_history_1.sqlite").is_symlink()

    def test_user_home_untouched(self, tmp_path):
        provider = self._provider_home(tmp_path / "provider")
        before = (provider / "config.toml").read_bytes()
        restore.compose_codex_home(
            companion_dir=str(tmp_path / "companion"),
            terminal_id="term-1",
            generation="gen-1",
            provider_home=str(provider),
            base_config_text=before.decode(),
            command="/bin/w",
        )
        assert (provider / "config.toml").read_bytes() == before
        assert len(list((tmp_path / "provider").iterdir())) == 4

    def test_missing_provider_home_still_composes(self, tmp_path):
        home, degraded = restore.compose_codex_home(
            companion_dir=str(tmp_path / "companion"),
            terminal_id="term-1",
            generation="gen-1",
            provider_home=str(tmp_path / "no-such-home"),
            base_config_text="",
            command="/bin/w",
        )
        assert degraded is None
        assert (
            tomllib.loads((Path(home) / "config.toml").read_text())["hooks"]["SessionStart"][0][
                "matcher"
            ]
            == "^compact$"
        )

    def test_drifted_entry_degrades_loudly(self, tmp_path):
        provider = self._provider_home(tmp_path / "provider")
        kwargs = dict(
            companion_dir=str(tmp_path / "companion"),
            terminal_id="term-1",
            generation="gen-1",
            provider_home=str(provider),
            base_config_text="",
            command="/bin/w",
        )
        home, degraded = restore.compose_codex_home(**kwargs)
        assert degraded is None
        # A successor that finds a foreign file where its link belongs
        # refuses to compose over it rather than adopting it silently.
        (Path(home) / "auth.json").unlink()
        (Path(home) / "auth.json").write_text("foreign", encoding="utf-8")
        _, degraded = restore.compose_codex_home(**kwargs)
        assert degraded is not None and "drifted" in degraded


class TestWrapperProcess:
    def test_compact_restores_the_current_goal(self, fake):
        proc = _run_wrapper(
            fake,
            _hook_input(),
            "--terminal",
            "term-1",
            "--terminal-generation",
            "gen-1",
            "--conduct-bin",
            fake["conduct"],
        )
        assert proc.returncode == 0
        output = json.loads(proc.stdout.decode("utf-8"))
        context = output["hookSpecificOutput"]["additionalContext"]
        assert "implement the thing" in context
        assert "version 3" in context
        assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        # The pinned argv carries the baked binding, never stdin extras.
        logged = [json.loads(line) for line in fake["log"].read_text().splitlines()]
        assert logged and logged[0][0:3] == ["goal", "hook-context", "--harness"]
        assert "--terminal-generation" in logged[0]

    def test_repeated_compact_tracks_goal_version_changes(self, fake):
        first = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        assert (
            "version 3"
            in json.loads(first.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        )
        answer = _ok_answer()
        answer["goal"]["goal_version"] = 4
        answer["goal"]["objective"] = "implement the other thing"
        fake["canned"].write_text(json.dumps(answer), encoding="utf-8")
        second = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        context = json.loads(second.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        assert "version 4" in context and "other thing" in context

    def test_claimed_id_forwarded_verbatim_without_mapping(self, fake):
        """The wrapper resolves exactly what the hook claims — no child/
        parent mapping, because no payload field could ground one. Safety
        against subagent injection lives one layer up and is proven, not
        assumed: installed 0.153.4 dispatches subagent starts to
        ``SubagentStart`` handlers (or runs nothing), and this install
        contains no such entries (see the config test above) — so a
        subagent start never reaches this wrapper. What this test pins is
        the wrapper's half of the contract: the claimed id is forwarded
        byte-identical to the read-only projection, never rewritten,
        never defaulted, never matched against a second source."""
        # Decoy identity fields a guessed mapping might consult: the
        # wrapper must ignore all of them and forward session_id alone.
        proc = _run_wrapper(
            fake,
            _hook_input(
                session_id="thr_parent",
                agent_id="child-1",
                transcript_path="/other/rollout.jsonl",
                agent_transcript_path="/other/child.jsonl",
            ),
            "--conduct-bin",
            fake["conduct"],
        )
        logged = [json.loads(line) for line in fake["log"].read_text().splitlines()]
        assert logged and "--native-session-id" in logged[0]
        assert logged[0][logged[0].index("--native-session-id") + 1] == "thr_parent"
        assert proc.returncode == 0

    def test_stale_rotation_restores_nothing(self, fake, monkeypatch):
        answer = _ok_answer(result_type="stale-generation", goal=None)
        fake["canned"].write_text(json.dumps(answer), encoding="utf-8")
        proc = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        assert proc.returncode == 0
        assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""

    def test_missing_assignment_is_truthful_without_goal_text(self, fake):
        answer = _ok_answer(
            result_type="no-assignment",
            goal=None,
            detail="no goal row is bound to occurrence 'occ-9'",
        )
        fake["canned"].write_text(json.dumps(answer), encoding="utf-8")
        proc = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        context = json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        assert "unavailable" in context and "occ-9" in context
        assert "implement the thing" not in context

    @pytest.mark.parametrize(
        "result_type", ["stale-generation", "dead-incarnation", "ambiguous", "no-worker"]
    )
    def test_discard_verdicts_restore_nothing(self, fake, result_type):
        answer = _ok_answer(result_type=result_type, goal=None)
        fake["canned"].write_text(json.dumps(answer), encoding="utf-8")
        proc = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""

    @pytest.mark.parametrize("state", ["satisfied", "cancelled"])
    def test_finished_assignments_restore_nothing(self, fake, state):
        answer = _ok_answer()
        answer["goal"]["state"] = state
        fake["canned"].write_text(json.dumps(answer), encoding="utf-8")
        proc = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""

    def test_unparseable_input_restores_nothing_and_exits_zero(self, fake):
        for raw in (b"", b"not json", b"[1,2]", b'{"cwd": "/x"}', b'{"session_id": "  "}'):
            proc = _run_wrapper(fake, raw, "--conduct-bin", fake["conduct"])
            assert proc.returncode == 0
            assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""

    def test_stdout_carries_only_the_json_object(self, fake):
        proc = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        json.loads(proc.stdout.decode("utf-8"))
        assert set(json.loads(proc.stdout.decode()).keys()) == {"hookSpecificOutput"}

    def test_output_has_no_turn_or_delivery_vocabulary(self, fake):
        """No-spurious-turn pin: the only legal output keys are the hook
        envelope's. No ``continue: false`` (which would end turns), no
        steer/send/resume fields, no second mechanism."""
        proc = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        text = proc.stdout.decode("utf-8")
        for token in (
            'continue":',
            "steer",
            "send",
            "resume",
            "SubagentStart",
            "PostCompact",
            "SubagentStop",
            "UserPromptSubmit",
        ):
            assert token not in text

    def test_conduct_failure_exits_zero_with_empty_context(self, fake, monkeypatch):
        monkeypatch.setenv("CONDUCT_MODE", "fail")
        proc = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        assert proc.returncode == 0
        assert json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"] == ""

    def test_only_goal_hook_context_is_ever_invoked(self, fake, monkeypatch):
        monkeypatch.setenv("CONDUCT_MODE", "fail")
        _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        _run_wrapper(fake, b"junk", "--conduct-bin", fake["conduct"])
        logged = [json.loads(line) for line in fake["log"].read_text().splitlines()]
        assert logged and all(argv[0:2] == ["goal", "hook-context"] for argv in logged)

    def test_long_goal_text_is_truncated_with_an_explicit_marker(self, fake):
        answer = _ok_answer()
        answer["goal"]["objective"] = "x" * 20000
        fake["canned"].write_text(json.dumps(answer), encoding="utf-8")
        proc = _run_wrapper(fake, _hook_input(), "--conduct-bin", fake["conduct"])
        context = json.loads(proc.stdout.decode())["hookSpecificOutput"]["additionalContext"]
        assert len(context) <= restore.MAX_ADDITIONAL_CONTEXT_CHARS
        assert "truncated here" in context

    def test_wrapper_bound_sits_under_the_provider_spill(self):
        # ≈2,500-token spill; our bound must never be what the provider cuts.
        assert restore.MAX_ADDITIONAL_CONTEXT_CHARS < 10000
        assert restore.ADDITIONAL_CONTEXT_LIMIT * 4 >= restore.MAX_ADDITIONAL_CONTEXT_CHARS


class TestPrepare:
    def _env(self, tmp_path, monkeypatch):
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        for name in ("cao-codex-hook-context", "conduct"):
            path = bindir / name
            path.write_text("#!/bin/sh\n", encoding="utf-8")
            path.chmod(0o755)
        monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
        return {"PATH": os.environ["PATH"]}

    def test_prepare_points_home_and_records_mechanism(self, tmp_path, monkeypatch):
        base_env = self._env(tmp_path, monkeypatch)
        provider = tmp_path / "provider"
        provider.mkdir()
        (provider / "config.toml").write_text('model = "gpt-5"\n', encoding="utf-8")
        base_env["CODEX_HOME"] = str(provider)
        record = {"terminal_id": "term-9", "generation": "gen-4", "reservation_id": "res-1"}
        env, installation = restore.prepare_codex_restoration(
            record=record,
            base_environment=base_env,
            companion_dir=str(tmp_path / "companion"),
        )
        assert installation["mechanism"] == restore.MECHANISM
        assert installation["degraded_reason"] is None
        assert env["CODEX_HOME"].endswith("term-9/gen-4/codex-home")
        assert env["CODEX_HOME"] != str(provider)

    def test_prepare_degrades_without_refusing(self, tmp_path, monkeypatch):
        empty = tmp_path / "emptybin"
        empty.mkdir()
        base_env = {"PATH": str(empty)}
        record = {"terminal_id": "term-9", "generation": "gen-4"}
        env, installation = restore.prepare_codex_restoration(
            record=record,
            base_environment=base_env,
            companion_dir=str(tmp_path / "companion"),
        )
        assert installation["mechanism"] is None
        assert installation["degraded_reason"] is not None
        assert "CODEX_HOME" not in env

    def test_prepare_requires_record_binding(self, tmp_path):
        with pytest.raises(KeyError):
            restore.prepare_codex_restoration(
                record={},
                base_environment={},
                companion_dir=str(tmp_path / "companion"),
            )
