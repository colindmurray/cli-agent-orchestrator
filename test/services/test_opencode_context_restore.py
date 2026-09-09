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
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


def _run_helper_process(
    *, fake, extra_args=(), session_id=SESSION, terminal_id=TERMINAL, generation=GENERATION
):
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
            terminal_id,
            "--terminal-generation",
            generation,
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


# ---------------------------------------------------------------------------
# R2 P1: the real construction path carries the binding (never injected)
# ---------------------------------------------------------------------------


class TestManagerBinding:
    def test_create_provider_binds_managed_opencode(self, tmp_path):
        from cli_agent_orchestrator.providers.manager import ProviderManager

        workdir = tmp_path / "work"
        workdir.mkdir()
        provider = ProviderManager().create_provider(
            "opencode_cli",
            "tid-p1",
            "sess-p1",
            "win-p1",
            "developer",
            None,
            terminal_working_directory=str(workdir),
            terminal_generation="gen-p1",
        )
        binding = provider._context_restore
        assert binding is not None
        assert binding.working_directory == str(workdir)
        assert binding.terminal_id == "tid-p1"
        assert binding.generation == "gen-p1"

    def test_create_provider_without_generation_stays_unbound(self, tmp_path):
        from cli_agent_orchestrator.providers.manager import ProviderManager

        workdir = tmp_path / "work"
        workdir.mkdir()
        provider = ProviderManager().create_provider(
            "opencode_cli",
            "tid-p1",
            "sess-p1",
            "win-p1",
            "developer",
            None,
            terminal_working_directory=str(workdir),
        )
        assert provider._context_restore is None

    def test_create_provider_without_root_stays_unbound(self):
        from cli_agent_orchestrator.providers.manager import ProviderManager

        provider = ProviderManager().create_provider(
            "opencode_cli",
            "tid-p1",
            "sess-p1",
            "win-p1",
            "developer",
            None,
            terminal_generation="gen-p1",
        )
        assert provider._context_restore is None

    def test_binding_builder_unit(self):
        from cli_agent_orchestrator.providers.manager import _opencode_restore_binding

        assert (
            _opencode_restore_binding(
                terminal_id="t", terminal_generation=None, working_directory="/w"
            )
            is None
        )
        assert (
            _opencode_restore_binding(
                terminal_id="t", terminal_generation="g", working_directory=None
            )
            is None
        )
        binding = _opencode_restore_binding(
            terminal_id="t", terminal_generation="g", working_directory="/w"
        )
        assert (binding.terminal_id, binding.generation) == ("t", "g")

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.services.terminal_service.status_monitor")
    @patch("cli_agent_orchestrator.services.terminal_service.fifo_manager")
    @patch("cli_agent_orchestrator.services.terminal_service.FIFO_DIR")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.services.terminal_service.db_create_terminal")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.generate_window_name")
    @patch("cli_agent_orchestrator.services.terminal_service.generate_session_name")
    @patch("cli_agent_orchestrator.services.terminal_service.generate_terminal_id")
    @patch("cli_agent_orchestrator.services.terminal_service.load_agent_profile")
    @patch("cli_agent_orchestrator.services.terminal_service.clear_session_env")
    async def test_create_terminal_forwards_generation(
        self,
        mock_clear_env,
        mock_load_profile,
        mock_gen_id,
        mock_gen_session,
        mock_gen_window,
        mock_tmux,
        mock_db_create,
        mock_provider_manager,
        mock_fifo_dir,
        mock_fifo_manager,
        mock_status_monitor,
        tmp_path,
    ):
        """The managed-launch callsite forwards the generation it owns."""
        from cli_agent_orchestrator.models.agent_profile import AgentProfile
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        mock_gen_id.return_value = "test1234"
        mock_gen_session.return_value = "cao-session"
        mock_gen_window.return_value = "developer-abcd"
        mock_tmux.session_exists.return_value = False
        mock_load_profile.return_value = AgentProfile(name="developer", description="Developer")
        mock_provider = AsyncMock()
        mock_provider.initialize.return_value = True
        mock_provider_manager.create_provider.return_value = mock_provider
        mock_fifo_dir.__truediv__ = MagicMock(return_value="fake.fifo")

        workdir = tmp_path / "work"
        workdir.mkdir()
        await create_terminal(
            "opencode_cli",
            "developer",
            new_session=True,
            reserved_terminal_id="abcdef12",
            terminal_generation="gen-fw",
            working_directory=str(workdir),
        )
        _args, kwargs = mock_provider_manager.create_provider.call_args
        assert kwargs.get("terminal_generation") == "gen-fw"
        assert kwargs.get("terminal_working_directory") == str(workdir)


# ---------------------------------------------------------------------------
# R2 P2: install outcome lands on the existing launch facts
# ---------------------------------------------------------------------------


def _v1_reservation_row(*, terminal_id, generation, facts=None):
    from cli_agent_orchestrator.clients import database

    with database.SessionLocal() as db:
        db.add(
            database.ManagedLaunchReservationModel(
                reservation_id=f"res-{terminal_id}",
                terminal_id=terminal_id,
                generation=generation,
                session_name="sess",
                provider="opencode_cli",
                agent_profile="developer",
                caller_id="test",
                working_directory="/tmp/wt",
                state="admitted",
                request_json="{}",
                observations_json="[]",
                launch_facts_json=facts,
                created_at="2026-09-09T00:00:00Z",
                updated_at="2026-09-09T00:00:00Z",
            )
        )
        db.commit()


def _read_facts(terminal_id):
    import json as _json

    from cli_agent_orchestrator.clients import database

    with database.SessionLocal() as db:
        row = (
            db.query(database.ManagedLaunchReservationModel)
            .filter(database.ManagedLaunchReservationModel.terminal_id == terminal_id)
            .one()
        )
        return _json.loads(row.launch_facts_json)


class TestRecordInstallation:
    def test_records_mechanism_additively(self):
        tid, gen = f"tid-{uuid.uuid4().hex[:8]}", f"gen-{uuid.uuid4().hex[:8]}"
        _v1_reservation_row(terminal_id=tid, generation=gen, facts='{"model":"m1"}')
        installation = restore.describe_installation(
            terminal_id=tid, generation=gen, degraded_reason=None
        )
        assert restore.record_installation(terminal_id=tid, installation=installation) is True
        facts = _read_facts(tid)
        assert facts["model"] == "m1"
        assert facts[restore.CONTEXT_RESTORATION_FACTS_KEY] == installation
        assert facts[restore.CONTEXT_RESTORATION_FACTS_KEY]["mechanism"] == restore.MECHANISM

    def test_records_degraded_reason(self):
        tid, gen = f"tid-{uuid.uuid4().hex[:8]}", f"gen-{uuid.uuid4().hex[:8]}"
        _v1_reservation_row(terminal_id=tid, generation=gen, facts="{}")
        installation = restore.describe_installation(
            terminal_id=tid, generation=gen, degraded_reason="no helper"
        )
        assert restore.record_installation(terminal_id=tid, installation=installation) is True
        facts = _read_facts(tid)
        assert facts[restore.CONTEXT_RESTORATION_FACTS_KEY]["mechanism"] is None
        assert facts[restore.CONTEXT_RESTORATION_FACTS_KEY]["degraded_reason"] == "no helper"

    def test_unknown_terminal_returns_false(self):
        assert (
            restore.record_installation(
                terminal_id=f"tid-absent-{uuid.uuid4().hex[:8]}",
                installation={"mechanism": None},
            )
            is False
        )

    def test_corrupt_facts_never_overwritten(self):
        tid, gen = f"tid-{uuid.uuid4().hex[:8]}", f"gen-{uuid.uuid4().hex[:8]}"
        _v1_reservation_row(terminal_id=tid, generation=gen, facts="not-json{")
        assert restore.record_installation(terminal_id=tid, installation={"a": 1}) is False
        from cli_agent_orchestrator.clients import database

        with database.SessionLocal() as db:
            row = (
                db.query(database.ManagedLaunchReservationModel)
                .filter(database.ManagedLaunchReservationModel.terminal_id == tid)
                .one()
            )
            assert row.launch_facts_json == "not-json{"

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.opencode_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.opencode_cli.get_backend")
    async def test_initialize_records_facts(
        self, mock_backend, mock_shell, mock_wait, tmp_path, fake
    ):
        mock_shell.return_value = True
        mock_wait.return_value = True
        tid, gen = f"tid-{uuid.uuid4().hex[:8]}", f"gen-{uuid.uuid4().hex[:8]}"
        _v1_reservation_row(terminal_id=tid, generation=gen, facts="{}")
        binding = _binding(tmp_path, fake=fake, terminal_id=tid, generation=gen)
        provider = _provider(
            terminal_id=tid, session_name="s", window_name="w", context_restore=binding
        )
        assert await provider.initialize() is True
        facts = _read_facts(tid)
        assert facts[restore.CONTEXT_RESTORATION_FACTS_KEY]["mechanism"] == restore.MECHANISM


# ---------------------------------------------------------------------------
# R2 native writer: observe + record through the existing contract
# ---------------------------------------------------------------------------


class TestObserveNativeSession:
    def test_recorded_then_cached(self, tmp_path):
        calls = []

        def record(tid, sid):
            calls.append((tid, sid))
            return True

        outcome = restore.observe_native_session(
            terminal_id="t1",
            generation="g1",
            session_id="ses-1",
            read_generation=lambda tid: "g1",
            record_native=record,
            companion_root=tmp_path,
        )
        assert outcome == "recorded"
        assert calls == [("t1", "ses-1")]
        marker = tmp_path / "t1" / "g1" / restore.NATIVE_SESSION_FILENAME
        assert json.loads(marker.read_text())["native_session_id"] == "ses-1"
        outcome2 = restore.observe_native_session(
            terminal_id="t1",
            generation="g1",
            session_id="ses-1",
            read_generation=lambda tid: (_ for _ in ()).throw(AssertionError("must not read")),
            record_native=lambda tid, sid: (_ for _ in ()).throw(AssertionError("must not record")),
            companion_root=tmp_path,
        )
        assert outcome2 == "already-recorded"
        assert calls == [("t1", "ses-1")]

    def test_refused_is_stable(self, tmp_path):
        calls = []

        def record(tid, sid):
            calls.append((tid, sid))
            return False

        kwargs = dict(
            terminal_id="t1",
            generation="g1",
            session_id="ses-1",
            read_generation=lambda tid: "g1",
            record_native=record,
            companion_root=tmp_path,
        )
        assert restore.observe_native_session(**kwargs) == "refused"
        assert restore.observe_native_session(**kwargs) == "refused"
        assert calls == [("t1", "ses-1")]

    def test_stale_generation_never_records(self, tmp_path):
        calls = []
        outcome = restore.observe_native_session(
            terminal_id="t1",
            generation="g1",
            session_id="ses-1",
            read_generation=lambda tid: "g2",
            record_native=lambda tid, sid: calls.append((tid, sid)),
            companion_root=tmp_path,
        )
        assert outcome == "stale-generation"
        assert calls == []
        assert not (tmp_path / "t1" / "g1" / restore.NATIVE_SESSION_FILENAME).exists()

    def test_unknown_generation_skips(self, tmp_path):
        calls = []
        outcome = restore.observe_native_session(
            terminal_id="t1",
            generation="g1",
            session_id="ses-1",
            read_generation=lambda tid: None,
            record_native=lambda tid, sid: calls.append((tid, sid)),
            companion_root=tmp_path,
        )
        assert outcome == "generation-unknown"
        assert calls == []

    def test_unbound_and_blank(self, tmp_path):
        assert (
            restore.observe_native_session(
                terminal_id=None,
                generation="g1",
                session_id="ses-1",
                companion_root=tmp_path,
            )
            == "unbound"
        )
        assert (
            restore.observe_native_session(
                terminal_id="t1",
                generation="g1",
                session_id="  ",
                companion_root=tmp_path,
            )
            == "no-session"
        )

    def test_record_error_never_raises(self, tmp_path):
        def boom(tid, sid):
            raise RuntimeError("db gone")

        outcome = restore.observe_native_session(
            terminal_id="t1",
            generation="g1",
            session_id="ses-1",
            read_generation=lambda tid: "g1",
            record_native=boom,
            companion_root=tmp_path,
        )
        assert outcome == "error"

    def test_run_helper_threading_keeps_text_on_record_error(self, tmp_path):
        def boom(tid, sid):
            raise RuntimeError("db gone")

        exit_code, text, _note = restore.run_helper(
            session_id=SESSION,
            terminal_id=TERMINAL,
            terminal_generation=GENERATION,
            conduct_binary="conduct",
            timeout_seconds=5,
            max_chars=8000,
            run_conduct=lambda argv: _ok_answer(),
            read_generation=lambda tid: GENERATION,
            record_native=boom,
            companion_root=tmp_path,
        )
        assert exit_code == 0
        assert "Ship the thing" in text

    def test_run_helper_observes(self, tmp_path):
        seen = []
        exit_code, text, _note = restore.run_helper(
            session_id=SESSION,
            terminal_id=TERMINAL,
            terminal_generation=GENERATION,
            conduct_binary="conduct",
            timeout_seconds=5,
            max_chars=8000,
            run_conduct=lambda argv: _ok_answer(),
            read_generation=lambda tid: GENERATION,
            record_native=lambda tid, sid: seen.append((tid, sid)) or True,
            companion_root=tmp_path,
        )
        assert exit_code == 0 and text
        assert seen == [(TERMINAL, SESSION)]

    def test_real_row_write_and_refuse_to_repoint(self, tmp_path, fake):
        """End to end through the real store: first sighting wins."""
        from cli_agent_orchestrator.clients import database

        tid, gen = f"tid-{uuid.uuid4().hex[:8]}", f"gen-{uuid.uuid4().hex[:8]}"
        _v1_reservation_row(terminal_id=tid, generation=gen, facts="{}")
        with database.SessionLocal() as db:
            db.add(
                database.TerminalModel(
                    id=tid, tmux_session="sess", tmux_window="w", provider="opencode_cli"
                )
            )
            db.commit()

        def row_sid():
            with database.SessionLocal() as db:
                row = (
                    db.query(database.TerminalModel).filter(database.TerminalModel.id == tid).one()
                )
                return row.native_session_id

        proc = _run_helper_process(
            fake=fake, session_id="ses-first", terminal_id=tid, generation=gen
        )
        assert proc.returncode == 0
        assert row_sid() == "ses-first"
        proc = _run_helper_process(
            fake=fake, session_id="ses-second", terminal_id=tid, generation=gen
        )
        assert proc.returncode == 0
        # The projection still answers (canned ok); only the row keeps the
        # first sighting: a supersession never becomes an update.
        assert row_sid() == "ses-first"


# ---------------------------------------------------------------------------
# R2 P3: stale-file disposition (unit)
# ---------------------------------------------------------------------------


class TestStaleCleanup:
    def _seed(self, plugdir, *names):
        plugdir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (plugdir / name).write_text("x\n")
        return plugdir

    def test_removes_only_same_terminal_other_generations(self, tmp_path):
        workdir = tmp_path / "work"
        plugdir = self._seed(
            restore.project_plugin_dir(str(workdir)),
            f"{restore.PLUGIN_FILENAME_PREFIX}{TERMINAL}-gen-old.ts",
            f"{restore.PLUGIN_FILENAME_PREFIX}{TERMINAL}-{GENERATION}.ts",
            f"{restore.PLUGIN_FILENAME_PREFIX}other-tid-gen-old.ts",
            "my-plugin.ts",
        )
        removed = restore.remove_stale_plugins(str(workdir), TERMINAL, GENERATION)
        assert [p.name for p in removed] == [
            f"{restore.PLUGIN_FILENAME_PREFIX}{TERMINAL}-gen-old.ts"
        ]
        remaining = sorted(p.name for p in plugdir.iterdir())
        assert remaining == sorted(
            [
                f"{restore.PLUGIN_FILENAME_PREFIX}{TERMINAL}-{GENERATION}.ts",
                f"{restore.PLUGIN_FILENAME_PREFIX}other-tid-gen-old.ts",
                "my-plugin.ts",
            ]
        )

    def test_absent_dir_is_noop(self, tmp_path):
        assert restore.remove_stale_plugins(str(tmp_path / "absent"), TERMINAL, GENERATION) == []

    def test_symlink_and_subdir_never_followed(self, tmp_path):
        workdir = tmp_path / "work"
        plugdir = self._seed(
            restore.project_plugin_dir(str(workdir)),
            f"{restore.PLUGIN_FILENAME_PREFIX}{TERMINAL}-gen-old.ts",
        )
        subdir = plugdir / f"{restore.PLUGIN_FILENAME_PREFIX}{TERMINAL}-gen-dir.ts"
        subdir.mkdir()
        (subdir / "inner.ts").write_text("x\n")
        link = plugdir / f"{restore.PLUGIN_FILENAME_PREFIX}{TERMINAL}-gen-link.ts"
        link.symlink_to(plugdir / f"{restore.PLUGIN_FILENAME_PREFIX}{TERMINAL}-gen-old.ts")
        removed = restore.remove_stale_plugins(str(workdir), TERMINAL, GENERATION)
        assert [p.name for p in removed] == [
            f"{restore.PLUGIN_FILENAME_PREFIX}{TERMINAL}-gen-old.ts"
        ]
        assert subdir.is_dir() and link.is_symlink()

    def test_unsafe_ids_raise(self, tmp_path):
        with pytest.raises(ValueError):
            restore.remove_stale_plugins(str(tmp_path), "../evil", GENERATION)


# ---------------------------------------------------------------------------
# R2 P3: shared workdir — two workers, no cross-inject, no double-inject
# ---------------------------------------------------------------------------

FAKE_BINDING_CONDUCT = (
    """\
#!"""
    + sys.executable
    + """
import json, os, sys

argv = sys.argv[1:]
if "--harness" not in argv or argv[argv.index("--harness") + 1] != "opencode_cli":
    sys.stderr.write("wrong harness claim\\n")
    sys.exit(2)
with open(os.environ["CONDUCT_BINDINGS"]) as handle:
    table = json.load(handle)
session = argv[argv.index("--native-session-id") + 1]
terminal = argv[argv.index("--terminal") + 1] if "--terminal" in argv else None
generation = (
    argv[argv.index("--terminal-generation") + 1] if "--terminal-generation" in argv else None
)
entry = table.get(session)
if (
    entry is not None
    and entry["terminal"] == terminal
    and entry["generation"] == generation
):
    answer = {
        "ok": True,
        "schema": "cao-hook-context-v1",
        "result_type": "ok",
        "detail": None,
        "recovery": None,
        "identity": {
            "harness": "opencode_cli",
            "native_session_id": session,
            "terminal_id": terminal,
            "terminal_generation": generation,
            "generation_fence": "verified",
        },
        "goal": {
            "goal_id": "g-" + session,
            "state": "open",
            "goal_version": "v1",
            "objective": "objective-for-" + session,
            "requirements_outstanding": [],
            "requirements_outstanding_count": 0,
            "completion_requirements_truncated": False,
            "active_hold": None,
            "next_action": "ordinary work may continue; this read starts no turn",
        },
        "bounds": {},
    }
else:
    answer = {
        "ok": True,
        "schema": "cao-hook-context-v1",
        "result_type": "no-worker",
        "detail": "no live terminal binds native session",
        "recovery": None,
        "identity": None,
        "goal": None,
        "bounds": {},
    }
sys.stdout.write(json.dumps(answer))
"""
)


@pytest.fixture()
def binding_conduct(tmp_path):
    script = tmp_path / "binding-conduct"
    script.write_text(FAKE_BINDING_CONDUCT)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    bindings = tmp_path / "bindings.json"
    env = dict(os.environ, CONDUCT_BINDINGS=str(bindings))
    return {"script": script, "bindings": bindings, "env": env}


def _install_bound_plugin(tmp_path, *, workdir, terminal_id, generation, conduct):
    shim = tmp_path / f"cao-opencode-hook-context-{terminal_id[:8]}"
    _helper_shim(shim)
    binding = restore.RestoreBinding(
        working_directory=str(workdir),
        terminal_id=terminal_id,
        generation=generation,
        helper_executable=str(shim),
        conduct_binary=str(conduct["script"]),
        timeout_seconds=5.0,
    )
    path, degraded = restore.install_plugin(binding)
    assert degraded is None
    return path


@needs_bun
class TestSharedWorkdir:
    BINDINGS = {
        "ses-A": {"terminal": "tid-A", "generation": "gen-A"},
        "ses-B": {"terminal": "tid-B", "generation": "gen-B"},
    }

    def _two_workers(self, tmp_path, binding_conduct):
        workdir = tmp_path / "shared"
        workdir.mkdir()
        binding_conduct["bindings"].write_text(json.dumps(self.BINDINGS))
        path_a = _install_bound_plugin(
            tmp_path,
            workdir=workdir,
            terminal_id="tid-A",
            generation="gen-A",
            conduct=binding_conduct,
        )
        path_b = _install_bound_plugin(
            tmp_path,
            workdir=workdir,
            terminal_id="tid-B",
            generation="gen-B",
            conduct=binding_conduct,
        )
        files = sorted(p.name for p in restore.project_plugin_dir(str(workdir)).iterdir())
        assert files == [path_a.name, path_b.name]
        return path_a, path_b

    def _drive(self, tmp_path, plugin_path, session_id, env):
        return _run_bun(
            _write_bun_case(
                tmp_path,
                plugin_source=plugin_path.read_text(),
                helper_script=None,
                session_id=session_id,
            ),
            env=env,
        )

    def test_each_worker_restores_only_its_own_session(self, tmp_path, binding_conduct):
        path_a, path_b = self._two_workers(tmp_path, binding_conduct)
        env = binding_conduct["env"]
        result = self._drive(tmp_path, path_a, "ses-A", env)
        assert result == {"system": [restore.render_transform_text(self._ok("ses-A"))]}
        result = self._drive(tmp_path, path_b, "ses-A", env)
        assert result == {"system": []}
        result = self._drive(tmp_path, path_b, "ses-B", env)
        assert result == {"system": [restore.render_transform_text(self._ok("ses-B"))]}
        result = self._drive(tmp_path, path_a, "ses-B", env)
        assert result == {"system": []}

    def _ok(self, session):
        entry = self.BINDINGS[session]
        return {
            "ok": True,
            "schema": "cao-hook-context-v1",
            "result_type": "ok",
            "detail": None,
            "recovery": None,
            "identity": {
                "harness": "opencode_cli",
                "native_session_id": session,
                "terminal_id": entry["terminal"],
                "terminal_generation": entry["generation"],
                "generation_fence": "verified",
            },
            "goal": {
                "goal_id": "g-" + session,
                "state": "open",
                "goal_version": "v1",
                "objective": "objective-for-" + session,
                "requirements_outstanding": [],
                "requirements_outstanding_count": 0,
                "completion_requirements_truncated": False,
                "active_hold": None,
                "next_action": "ordinary work may continue; this read starts no turn",
            },
            "bounds": {},
        }

    def test_one_request_pushes_at_most_once(self, tmp_path, binding_conduct):
        path_a, _path_b = self._two_workers(tmp_path, binding_conduct)
        result = self._drive(tmp_path, path_a, "ses-A", binding_conduct["env"])
        assert len(result["system"]) == 1

    def test_successor_launch_cleans_predecessor_file(self, tmp_path, binding_conduct):
        workdir = tmp_path / "shared"
        workdir.mkdir()
        binding_conduct["bindings"].write_text(json.dumps(self.BINDINGS))
        old = _install_bound_plugin(
            tmp_path,
            workdir=workdir,
            terminal_id="tid-A",
            generation="gen-old",
            conduct=binding_conduct,
        )
        assert old.exists()
        other = _install_bound_plugin(
            tmp_path,
            workdir=workdir,
            terminal_id="tid-B",
            generation="gen-B",
            conduct=binding_conduct,
        )
        user_plugin = restore.project_plugin_dir(str(workdir)) / "user-plugin.ts"
        user_plugin.write_text("export const U = 1;\n")
        binding = restore.RestoreBinding(
            working_directory=str(workdir),
            terminal_id="tid-A",
            generation="gen-A",
            helper_executable=sys.executable,
            conduct_binary=str(binding_conduct["script"]),
            timeout_seconds=5.0,
        )
        removed = restore.remove_stale_plugins(str(workdir), "tid-A", "gen-A")
        assert [p.name for p in removed] == [old.name]
        new, degraded = restore.install_plugin(binding)
        assert degraded is None and new.exists()
        assert not old.exists()
        assert other.exists() and user_plugin.exists()


# ---------------------------------------------------------------------------
# R2 reader fallback: v1 opencode rows surface recorded sessions
# ---------------------------------------------------------------------------


class TestControlIdentityFallback:
    def _fallback(self, managed, metadata):
        from cli_agent_orchestrator.services import control_input_service

        return control_input_service._managed_native_session_id(managed, metadata)

    def test_managed_value_wins_including_explicit_null(self):
        assert (
            self._fallback(
                {"native_session_id": "live-s"},
                {"provider": "opencode_cli", "native_session_id": "row-s"},
            )
            == "live-s"
        )
        # The v2 refusal shape (explicit null) is never papered over.
        assert (
            self._fallback(
                {"native_session_id": None},
                {"provider": "opencode_cli", "native_session_id": "row-s"},
            )
            is None
        )

    def test_absent_key_falls_back_for_opencode_only(self):
        assert (
            self._fallback(
                {"generation": "g1"},
                {"provider": "opencode_cli", "native_session_id": "row-s"},
            )
            == "row-s"
        )
        assert (
            self._fallback(
                {"generation": "g1"},
                {"provider": "codex", "native_session_id": "row-s"},
            )
            is None
        )
        assert (
            self._fallback(None, {"provider": "opencode_cli", "native_session_id": "row-s"})
            == "row-s"
        )

    def test_resolve_control_identity_surfaces_recorded_opencode_session(self):
        from cli_agent_orchestrator.services import control_input_service

        metadata = {
            "provider": "opencode_cli",
            "native_session_id": "ses-live",
            "tmux_session": "sess",
            "generation": "gen-1",
        }
        managed = {"generation": "gen-1", "provider": "opencode_cli"}
        with (
            patch.object(control_input_service, "_terminal_metadata", return_value=metadata),
            patch.object(control_input_service, "_managed_identity", return_value=managed),
            patch.object(control_input_service, "_tmux_client", return_value=None),
        ):
            resolved = control_input_service.resolve_control_identity("tid-1")
        assert resolved is not None
        assert resolved.native_session_id == "ses-live"

    def test_resolve_control_identity_keeps_v2_null(self):
        from cli_agent_orchestrator.services import control_input_service

        metadata = {
            "provider": "opencode_cli",
            "native_session_id": "ses-live",
            "tmux_session": "sess",
        }
        managed = {"generation": "gen-1", "native_session_id": None}
        with (
            patch.object(control_input_service, "_terminal_metadata", return_value=metadata),
            patch.object(control_input_service, "_managed_identity", return_value=managed),
            patch.object(control_input_service, "_tmux_client", return_value=None),
        ):
            resolved = control_input_service.resolve_control_identity("tid-1")
        assert resolved is not None
        assert resolved.native_session_id is None

    def test_blank_and_missing_stay_none(self):
        assert (
            self._fallback(
                {"generation": "g1"},
                {"provider": "opencode_cli", "native_session_id": "  "},
            )
            is None
        )
        assert self._fallback({"generation": "g1"}, {"provider": "opencode_cli"}) is None
        assert self._fallback(None, None) is None
