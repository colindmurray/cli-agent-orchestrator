"""OpenCode per-request passive goal restoration (cond-0845 slice 2, OpenCode).

Three reproduced-failure classes, one theme: a hook that restores the wrong
text — or breaks the request path — is worse than a hook that restores
nothing.

*The wrong text.* A transform that echoes its input, guesses a goal, or
re-injects a stale/foreign/satisfied callback restores another assignment's
context into a live session. These tests drive the real helper process
(env/argv in, one string out) against a fake ``conduct`` honouring the
exact slice-1 CLI contract, and pin: only ``goal hook-context`` with
``--harness opencode_cli`` is ever invoked, and every non-projectable
answer prints nothing with exit 0.

*The broken request.* The transform runs before *every* LLM request, so a
shim that throws, blocks, or subscribes to compaction/event hooks would
degrade or double-inject normal traffic. The generated plugin source is
asserted hook-for-hook, and the real TypeScript runs under the installed
``bun`` against a fake helper: empty/nonzero/missing helper output pushes
nothing and never throws.

*The damaged project.* Registration writes exactly one additive file under
the worker's ``.opencode/plugins/`` and never modifies, replaces, or
removes anything else; the provider callsite without a binding is
byte-identical to before.

Fixture boundary: the fake ``conduct`` stands in for the live server
reads behind ``conduct goal hook-context`` (conductor PR #349,
``1f1e415f``). Its canned answers use the real envelope
(``cao-hook-context-v1``) and the real result types, but no test here
proves the conductor side — that proof belongs to the conductor lane.
Likewise, ``opencode export`` is never treated as model-entry evidence:
these tests prove registration plus transformation, not model reads.
"""

from __future__ import annotations

import ast
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from cli_agent_orchestrator.models.provider import ProviderType
from cli_agent_orchestrator.providers.opencode_cli import OpenCodeCliProvider
from cli_agent_orchestrator.services import claude_context_restore as claude
from cli_agent_orchestrator.services import opencode_context_restore as restore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"

SESSION = "ses_11111111111141118811111111111111"
TERMINAL = "11111111-1111-4111-8111-111111111111"
GENERATION = "22222222-2222-4222-8222-222222222222"

BUN = shutil.which("bun")
needs_bun = pytest.mark.skipif(BUN is None, reason="bun is required for TS shim tests")

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
# The fork client contract (conductor goal_hook_context.py): the helper
# must claim this harness with the observed native session. A wrong
# identity here resolves another worker or none — so the fake refuses it
# loudly rather than replaying a goal onto it.
if "--harness" not in argv or argv[argv.index("--harness") + 1] != "opencode_cli":
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


def _ok_answer(*, version="v3", objective="Ship the thing", state="open", hold=None):
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
            "harness": "opencode_cli",
            "native_session_id": SESSION,
            "terminal_id": TERMINAL,
            "terminal_generation": GENERATION,
            "generation_fence": "verified",
        },
        "goal": {
            "goal_id": "g-1",
            "state": state,
            "goal_version": version,
            "objective": objective,
            "requirements_outstanding": ["r1"],
            "requirements_outstanding_count": 1,
            "completion_requirements_truncated": False,
            "active_hold": hold,
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


def _binding(tmp_path, *, fake=None, **overrides):
    kwargs = dict(
        working_directory=str(tmp_path / "work"),
        terminal_id=TERMINAL,
        generation=GENERATION,
        helper_executable=sys.executable,
        conduct_binary=str(fake["script"]) if fake else "conduct",
        timeout_seconds=5.0,
    )
    kwargs.update(overrides)
    Path(kwargs["working_directory"]).mkdir(parents=True, exist_ok=True)
    return restore.RestoreBinding(**kwargs)


def _run_helper_process(*, fake, extra_args=(), session_id=SESSION):
    """The real helper process: env/argv in, one string out."""
    env = dict(fake["env"], PYTHONPATH=str(SRC_DIR))
    if session_id is not None:
        env[restore.SESSION_ID_ENV_VAR] = session_id
    else:
        env.pop(restore.SESSION_ID_ENV_VAR, None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.opencode_context_restore",
            "--terminal",
            TERMINAL,
            "--terminal-generation",
            GENERATION,
            "--conduct-bin",
            str(fake["script"]),
            "--timeout",
            "5",
            *extra_args,
        ],
        capture_output=True,
        env=env,
        timeout=60,
    )
    return proc


def _logged_argvs(fake):
    return [json.loads(line) for line in Path(fake["log"]).read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# Provider string and mechanism identity (no invented aliases)
# ---------------------------------------------------------------------------


class TestProviderIdentity:
    def test_harness_is_the_fork_canonical_opencode_provider(self):
        assert restore.HARNESS == "opencode_cli"
        assert restore.HARNESS == ProviderType.OPENCODE_CLI.value

    def test_transform_hook_name_is_pinned(self):
        assert restore.TRANSFORM_HOOK == "experimental.chat.system.transform"

    def test_mechanism_names_the_transform(self):
        assert "experimental.chat.system.transform" in restore.MECHANISM


# ---------------------------------------------------------------------------
# Plugin source: one hook, one bounded push, never throws, never compacts
# ---------------------------------------------------------------------------


def _plugin_source(fake):
    binding = _binding(fake["script"].parent, fake=fake)
    helper = restore.resolve_helper_executable(binding.helper_executable)
    conduct = restore.resolve_conduct_binary(binding.conduct_binary)
    command = restore.build_helper_command(
        helper_executable=helper,
        terminal_id=binding.terminal_id,
        generation=binding.generation,
        conduct_binary=conduct,
        timeout_seconds=binding.timeout_seconds,
    )
    return restore.render_plugin_source(helper_command=command), command


class TestPluginSource:
    def test_exactly_one_transform_hook(self, fake):
        source, _ = _plugin_source(fake)
        assert source.count(restore.TRANSFORM_HOOK) == 1

    def test_no_compaction_or_event_hooks(self, fake):
        source, _ = _plugin_source(fake)
        for fragment in restore.FORBIDDEN_HOOK_FRAGMENTS:
            assert fragment not in source, fragment

    def test_baked_command_is_quoted_verbatim(self, fake):
        source, command = _plugin_source(fake)
        assert command in source
        assert command == shlex.join(shlex.split(command))

    def test_push_only_nonempty_guarded(self, fake):
        source, _ = _plugin_source(fake)
        assert "if (text) output.system.push(text)" in source
        assert source.count("output.system.push") == 1

    def test_never_throws_never_delivers(self, fake):
        source, _ = _plugin_source(fake)
        assert ".nothrow()" in source
        assert "catch" in source
        for verb in ("client.", "tool.execute", "steer", "send(", "$`conduct goal send"):
            assert verb not in source, verb

    def test_session_travels_by_env_not_interpolation(self, fake):
        source, _ = _plugin_source(fake)
        assert restore.SESSION_ID_ENV_VAR in source
        assert ".env(" in source
        assert "${sessionID}" not in source
        assert "+ sessionID" not in source

    def test_rejects_shell_unsafe_bakes(self):
        with pytest.raises(ValueError):
            restore.build_helper_command(
                helper_executable="/bin/helper`id`",
                terminal_id=TERMINAL,
                generation=GENERATION,
                conduct_binary="/usr/bin/conduct",
            )
        with pytest.raises(ValueError):
            restore.render_plugin_source(helper_command="helper --x '${EVIL}'")


# ---------------------------------------------------------------------------
# Helper process: one string out, exit 0 always, only the read projection
# ---------------------------------------------------------------------------


class TestHelperProcess:
    def test_ok_prints_the_bounded_goal_string(self, fake):
        proc = _run_helper_process(fake=fake)
        assert proc.returncode == 0
        text = proc.stdout.decode()
        assert "CAO goal restoration" in text
        assert "Ship the thing" in text
        assert proc.stderr.decode() == ""

    def test_only_the_read_projection_is_invoked(self, fake):
        _run_helper_process(fake=fake)
        argvs = _logged_argvs(fake)
        assert len(argvs) == 1
        assert argvs[0][:2] == ["goal", "hook-context"]
        assert "--harness" in argvs[0]
        assert argvs[0][argvs[0].index("--harness") + 1] == "opencode_cli"
        assert "--native-session-id" in argvs[0]
        assert argvs[0][argvs[0].index("--native-session-id") + 1] == SESSION
        assert "--terminal-generation" in argvs[0]
        assert argvs[0][argvs[0].index("--terminal-generation") + 1] == GENERATION

    @pytest.mark.parametrize(
        "result_type", ["stale-generation", "dead-incarnation", "ambiguous", "no-worker"]
    )
    def test_discard_verdicts_restore_nothing(self, fake, result_type):
        fake["canned"].write_text(json.dumps(_typed_answer(result_type)))
        proc = _run_helper_process(fake=fake)
        assert proc.returncode == 0
        assert proc.stdout.decode() == ""

    @pytest.mark.parametrize("state", ["satisfied", "cancelled"])
    def test_terminal_states_restore_nothing(self, fake, state):
        fake["canned"].write_text(json.dumps(_ok_answer(state=state)))
        proc = _run_helper_process(fake=fake)
        assert proc.returncode == 0
        assert proc.stdout.decode() == ""

    def test_missing_assignment_stays_silent_per_request(self, fake):
        # Deliberate divergence from the SessionStart adapter: an
        # "unavailable" note on every request would repeat into context
        # until admission; silence recovers automatically instead.
        for result_type in ("no-assignment", "unavailable"):
            fake["canned"].write_text(json.dumps(_typed_answer(result_type)))
            proc = _run_helper_process(fake=fake)
            assert proc.returncode == 0
            assert proc.stdout.decode() == "", result_type

    def test_unknown_schema_restores_nothing(self, fake):
        answer = _ok_answer()
        answer["schema"] = "cao-hook-context-v99"
        fake["canned"].write_text(json.dumps(answer))
        proc = _run_helper_process(fake=fake)
        assert proc.returncode == 0
        assert proc.stdout.decode() == ""

    def test_missing_session_id_restores_nothing_without_conduct(self, fake):
        proc = _run_helper_process(fake=fake, session_id=None)
        assert proc.returncode == 0
        assert proc.stdout.decode() == ""
        assert _logged_argvs(fake) == []

    def test_explicit_flag_beats_env(self, fake):
        other = "ses_other41111111111111111111111111111"
        proc = _run_helper_process(
            fake=fake,
            session_id=SESSION,
            extra_args=("--native-session-id", other),
        )
        assert proc.returncode == 0
        argvs = _logged_argvs(fake)
        assert argvs[0][argvs[0].index("--native-session-id") + 1] == other

    @pytest.mark.parametrize("mode", ["fail", "garbage"])
    def test_conduct_failure_restores_nothing(self, fake, mode):
        fake["env"]["CONDUCT_MODE"] = mode
        proc = _run_helper_process(fake=fake)
        assert proc.returncode == 0
        assert proc.stdout.decode() == ""

    def test_conduct_timeout_restores_nothing(self, fake):
        fake["env"]["CONDUCT_MODE"] = "sleep"
        fake["env"]["CONDUCT_SLEEP"] = "30"
        proc = _run_helper_process(fake=fake, extra_args=("--timeout", "1"))
        assert proc.returncode == 0
        assert proc.stdout.decode() == ""

    def test_stdout_carries_only_the_string(self, fake):
        # Notes go to stderr; any extra stdout byte would land in model
        # context, so stdout must be exactly the restoration text.
        proc = _run_helper_process(fake=fake)
        assert proc.stdout.decode() == restore.render_transform_text(_ok_answer())

    def test_single_subprocess_callsite_pinned(self):
        tree = ast.parse(Path(restore.__file__).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ]
        assert len(calls) == 1
        assert calls[0].func.attr == "run"

    def test_no_delivery_verbs_in_argv_builder(self):
        source = Path(restore.__file__).read_text()
        builder = source.split("def build_conduct_argv")[1].split("\ndef ")[0]
        for verb in ("steer", "send", "resume", "release", "start"):
            assert f'"{verb}"' not in builder, verb


# ---------------------------------------------------------------------------
# Render mapping: shared bytes with the accepted Claude adapter
# ---------------------------------------------------------------------------


class TestRenderMapping:
    def test_ok_bytes_equal_the_claude_adapter(self):
        # Reuse proof: the same projection answer renders the same
        # model-facing bytes here as in the accepted Claude path.
        answer = _ok_answer()
        assert restore.render_transform_text(answer) == claude.render_restoration(answer)

    def test_ok_bytes_equal_with_waiting_hold(self):
        hold = {
            "hold_id": "h-1",
            "state": "active",
            "resolution": None,
            "reason_kind": "review",
            "release_kind": "review-decision",
            "release_id": "rd-1",
        }
        answer = _ok_answer(hold=hold)
        assert restore.render_transform_text(answer) == claude.render_restoration(answer)

    def test_non_dict_and_unknown_results_restore_nothing(self):
        assert restore.render_transform_text([]) == ""
        assert restore.render_transform_text({"schema": "x"}) == ""
        for result_type in (
            "stale-generation",
            "dead-incarnation",
            "ambiguous",
            "no-worker",
            "no-assignment",
            "unavailable",
            "something-new",
        ):
            assert restore.render_transform_text(_typed_answer(result_type)) == ""

    def test_long_objective_truncates_with_marker(self):
        answer = _ok_answer(objective="word " * 5000)
        text = restore.render_transform_text(answer)
        assert len(text) <= claude.MAX_ADDITIONAL_CONTEXT_CHARS
        assert claude.TRUNCATION_MARKER in text

    def test_short_objective_passes_through_untouched(self):
        answer = _ok_answer()
        text = restore.render_transform_text(answer)
        assert claude.TRUNCATION_MARKER not in text
        assert "Ship the thing" in text


# ---------------------------------------------------------------------------
# Registration: one additive file, user plugins never touched
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_install_writes_exact_plugin_source(self, tmp_path, fake):
        binding = _binding(tmp_path, fake=fake)
        path, degraded = restore.install_plugin(binding)
        assert degraded is None
        assert path is not None and path.exists()
        assert path.parent == restore.project_plugin_dir(binding.working_directory)
        assert path.name == restore.plugin_filename(terminal_id=TERMINAL, generation=GENERATION)
        helper = restore.resolve_helper_executable(binding.helper_executable)
        conduct = restore.resolve_conduct_binary(binding.conduct_binary)
        expected = restore.render_plugin_source(
            helper_command=restore.build_helper_command(
                helper_executable=helper,
                terminal_id=TERMINAL,
                generation=GENERATION,
                conduct_binary=conduct,
                timeout_seconds=5.0,
            )
        )
        assert path.read_text(encoding="utf-8") == expected

    def test_install_is_additive_over_user_plugins(self, tmp_path, fake):
        binding = _binding(tmp_path, fake=fake)
        plugdir = restore.project_plugin_dir(binding.working_directory)
        plugdir.mkdir(parents=True, exist_ok=True)
        user_plugin = plugdir / "my-plugin.ts"
        user_plugin.write_text("export const Mine = async () => ({});\n")
        path, degraded = restore.install_plugin(binding)
        assert degraded is None
        assert user_plugin.read_text() == "export const Mine = async () => ({});\n"
        assert path is not None and path.name != user_plugin.name

    def test_reinstall_identical_is_benign(self, tmp_path, fake):
        binding = _binding(tmp_path, fake=fake)
        first, _ = restore.install_plugin(binding)
        second, degraded = restore.install_plugin(binding)
        assert degraded is None
        assert second == first

    def test_differing_managed_file_refuses(self, tmp_path, fake):
        binding = _binding(tmp_path, fake=fake)
        path, _ = restore.install_plugin(binding)
        path.write_text("tampered\n")
        with pytest.raises(FileExistsError):
            restore.install_plugin(binding)

    def test_unresolvable_helper_degrades_without_writing(self, tmp_path):
        binding = _binding(tmp_path, helper_executable="/nonexistent/cao-opencode-hook-context")
        path, degraded = restore.install_plugin(binding)
        assert path is None
        assert degraded is not None and "helper" in degraded
        assert list(restore.project_plugin_dir(binding.working_directory).glob("*")) == []

    def test_missing_workdir_degrades(self, tmp_path):
        binding = restore.RestoreBinding(
            working_directory=str(tmp_path / "absent"),
            terminal_id=TERMINAL,
            generation=GENERATION,
        )
        path, degraded = restore.install_plugin(binding)
        assert path is None
        assert degraded is not None

    def test_unsafe_ids_degrade(self, tmp_path):
        binding = _binding(tmp_path, terminal_id="../evil")
        path, degraded = restore.install_plugin(binding)
        assert path is None
        assert degraded is not None

    def test_remove_deletes_only_managed_files(self, tmp_path, fake):
        binding = _binding(tmp_path, fake=fake)
        path, _ = restore.install_plugin(binding)
        assert restore.remove_plugin(path) is True
        assert not path.exists()
        assert restore.remove_plugin(path) is False
        user_plugin = path.parent / "user-plugin.ts"
        user_plugin.write_text("x\n")
        with pytest.raises(ValueError):
            restore.remove_plugin(user_plugin)
        assert user_plugin.exists()

    def test_describe_installation_shape(self):
        installed = restore.describe_installation(
            terminal_id=TERMINAL, generation=GENERATION, degraded_reason=None
        )
        assert installed == {
            "mechanism": restore.MECHANISM,
            "terminal_id": TERMINAL,
            "terminal_generation": GENERATION,
            "degraded_reason": None,
        }
        degraded = restore.describe_installation(
            terminal_id=TERMINAL, generation=GENERATION, degraded_reason="no helper"
        )
        assert degraded["mechanism"] is None
        assert degraded["degraded_reason"] == "no helper"


# ---------------------------------------------------------------------------
# Provider callsite: real initialize() with and without a binding
# ---------------------------------------------------------------------------


def _provider(**overrides):
    kwargs = dict(
        terminal_id="test-tid",
        session_name="test-session",
        window_name="window-0",
        agent_profile="developer",
    )
    kwargs.update(overrides)
    return OpenCodeCliProvider(**kwargs)


class TestProviderCallsite:
    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.opencode_cli.get_backend")
    async def test_initialize_without_binding_never_registers(
        self, mock_backend, mock_shell, mock_wait
    ):
        mock_shell.return_value = True
        mock_wait.return_value = True
        provider = _provider()
        with patch(
            "cli_agent_orchestrator.services.opencode_context_restore.install_plugin"
        ) as install:
            assert await provider.initialize() is True
        install.assert_not_called()

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.opencode_cli.get_backend")
    async def test_initialize_with_binding_installs_plugin(
        self, mock_backend, mock_shell, mock_wait, tmp_path, fake
    ):
        mock_shell.return_value = True
        mock_wait.return_value = True
        binding = _binding(tmp_path, fake=fake)
        provider = _provider(context_restore=binding)
        assert await provider.initialize() is True
        plugdir = restore.project_plugin_dir(binding.working_directory)
        files = list(plugdir.glob("cao-goal-restoration-*.ts"))
        assert len(files) == 1
        sent_cmd = mock_backend.return_value.send_keys.call_args[0][2]
        assert "opencode" in sent_cmd and "--agent developer" in sent_cmd

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.opencode_cli.get_backend")
    async def test_launch_command_identical_with_and_without_binding(
        self, mock_backend, mock_shell, mock_wait, tmp_path, fake
    ):
        mock_shell.return_value = True
        mock_wait.return_value = True
        await _provider().initialize()
        plain_cmd = mock_backend.return_value.send_keys.call_args[0][2]
        await _provider(context_restore=_binding(tmp_path, fake=fake)).initialize()
        bound_cmd = mock_backend.return_value.send_keys.call_args[0][2]
        assert bound_cmd == plain_cmd

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.opencode_cli.get_backend")
    async def test_failed_install_never_fails_launch(
        self, mock_backend, mock_shell, mock_wait, tmp_path
    ):
        mock_shell.return_value = True
        mock_wait.return_value = True
        binding = _binding(tmp_path, helper_executable="/nonexistent/cao-opencode-hook-context")
        provider = _provider(context_restore=binding)
        assert await provider.initialize() is True
        assert mock_backend.return_value.send_keys.called


# ---------------------------------------------------------------------------
# Real TypeScript: generated plugin under installed bun, fake helper
# ---------------------------------------------------------------------------

BUN_DRIVER = """\
import { CaoGoalRestoration } from PROCESS_PLUGIN_PATH;
import { $ } from "bun";

const plugin = await CaoGoalRestoration({ $ });
const hook = plugin[HOOK_NAME];
if (typeof hook !== "function") {
  console.log(JSON.stringify({ error: "missing-hook" }));
  process.exit(0);
}
const output = { system: [] };
try {
  await hook({ sessionID: SESSION_ID_JSON }, output);
  console.log(JSON.stringify({ system: output.system }));
} catch (err) {
  console.log(JSON.stringify({ error: String(err && err.message || err) }));
}
"""


def _write_bun_case(tmp_path, *, plugin_source, helper_script, session_id):
    plugfile = tmp_path / "cao-goal-restoration-test.ts"
    plugfile.write_text(plugin_source)
    driver = tmp_path / "driver.ts"
    driver.write_text(
        BUN_DRIVER.replace("PROCESS_PLUGIN_PATH", json.dumps(str(plugfile)))
        .replace("HOOK_NAME", json.dumps(restore.TRANSFORM_HOOK))
        .replace("SESSION_ID_JSON", json.dumps(session_id))
    )
    return driver


def _run_bun(driver, *, env=None):
    merged = dict(os.environ, PYTHONPATH=str(SRC_DIR))
    if env:
        merged.update(env)
    proc = subprocess.run(
        [BUN, str(driver)],
        capture_output=True,
        timeout=60,
        env=merged,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return json.loads(proc.stdout.decode().splitlines()[-1])


def _helper_shim(path):
    """Executable shim standing in for the installed console entry."""
    path.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + ' -m cli_agent_orchestrator.services.opencode_context_restore "$@"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _fake_helper_sh(path, *, body):
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@needs_bun
class TestBunTransform:
    def test_pushes_helper_stdout(self, tmp_path):
        helper = tmp_path / "helper.sh"
        _fake_helper_sh(
            helper,
            body='if [ "$CAO_OPENCODE_SESSION_ID" != "ses-live" ]; then exit 9; fi\n'
            # Bun .env() replaces the child env (verified): the shim must
            # inherit the server env, or the helper loses PATH/HOME/etc.
            'if [ "$CAO_INHERITED" != "yes" ]; then exit 10; fi\n' "printf 'goal-string'",
        )
        source = restore.render_plugin_source(helper_command=shlex.join([str(helper)]))
        result = _run_bun(
            _write_bun_case(
                tmp_path,
                plugin_source=source,
                helper_script=helper,
                session_id="ses-live",
            ),
            env={"CAO_INHERITED": "yes"},
        )
        assert result == {"system": ["goal-string"]}

    def test_empty_stdout_pushes_nothing(self, tmp_path):
        helper = tmp_path / "helper.sh"
        _fake_helper_sh(helper, body="printf ''")
        source = restore.render_plugin_source(helper_command=shlex.join([str(helper)]))
        result = _run_bun(
            _write_bun_case(
                tmp_path,
                plugin_source=source,
                helper_script=helper,
                session_id="ses-live",
            )
        )
        assert result == {"system": []}

    def test_failing_helper_pushes_nothing_without_throwing(self, tmp_path):
        helper = tmp_path / "helper.sh"
        _fake_helper_sh(helper, body="echo boom >&2\nexit 3")
        source = restore.render_plugin_source(helper_command=shlex.join([str(helper)]))
        result = _run_bun(
            _write_bun_case(
                tmp_path,
                plugin_source=source,
                helper_script=helper,
                session_id="ses-live",
            )
        )
        assert result == {"system": []}

    def test_missing_session_never_execs_helper(self, tmp_path):
        marker = tmp_path / "invoked"
        helper = tmp_path / "helper.sh"
        _fake_helper_sh(helper, body=f"touch {shlex.quote(str(marker))}\nprintf x")
        source = restore.render_plugin_source(helper_command=shlex.join([str(helper)]))
        plugfile = tmp_path / "cao-goal-restoration-test.ts"
        plugfile.write_text(source)
        driver = tmp_path / "driver-nosession.ts"
        driver.write_text(
            'import { CaoGoalRestoration } from %s;\nimport { $ } from "bun";\n'
            "const plugin = await CaoGoalRestoration({ $ });\n"
            "const hook = plugin[%s];\n"
            "const output = { system: [] };\n"
            "await hook({}, output);\n"
            "console.log(JSON.stringify({ system: output.system }));\n"
            % (json.dumps(str(plugfile)), json.dumps(restore.TRANSFORM_HOOK))
        )
        result = _run_bun(driver)
        assert result == {"system": []}
        assert not marker.exists()

    def test_installed_file_end_to_end(self, tmp_path, fake):
        # The install path's exact bytes, driven under bun through the
        # real helper process against the fake conduct: install -> TS
        # shim -> helper -> projection -> one pushed string.
        shim = tmp_path / "cao-opencode-hook-context"
        _helper_shim(shim)
        binding = _binding(
            tmp_path,
            fake=fake,
            helper_executable=str(shim),
        )
        path, degraded = restore.install_plugin(binding)
        assert degraded is None
        content = path.read_text()
        assert "export const CaoGoalRestoration" in content
        result = _run_bun(
            _write_bun_case(
                tmp_path,
                plugin_source=content,
                helper_script=shim,
                session_id=SESSION,
            ),
            env=fake["env"],
        )
        assert result == {"system": [restore.render_transform_text(_ok_answer())]}

    def test_installed_file_with_stale_binding_pushes_nothing(self, tmp_path, fake):
        shim = tmp_path / "cao-opencode-hook-context"
        _helper_shim(shim)
        fake["canned"].write_text(json.dumps(_typed_answer("stale-generation")))
        binding = _binding(
            tmp_path,
            fake=fake,
            helper_executable=str(shim),
        )
        path, degraded = restore.install_plugin(binding)
        assert degraded is None
        result = _run_bun(
            _write_bun_case(
                tmp_path,
                plugin_source=path.read_text(),
                helper_script=shim,
                session_id=SESSION,
            ),
            env=fake["env"],
        )
        assert result == {"system": []}
