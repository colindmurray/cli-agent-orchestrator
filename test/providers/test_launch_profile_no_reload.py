"""Adapter no-reload: launch argv/config consumes the launch profile (cond-0817).

Each adapter that loads a CAO profile by name at launch accepts the
already-loaded ``launch_profile`` from ``ProviderManager.create_provider``
and prefers it over ``load_agent_profile``. With the store loader rigged to
raise, every launch-time consumption below must still succeed from the
supplied object — and constructing through the manager must install that
object. Without ``launch_profile`` the legacy by-name load still runs
(mutation guard: deleting the prefer-branch breaks the positive tests).
"""

from unittest.mock import patch

import pytest

from cli_agent_orchestrator.models.agent_profile import AgentProfile
from cli_agent_orchestrator.providers.manager import ProviderManager

_SENTINEL_MODEL = "sentinel-model-9"


def _sentinel(**overrides):
    fields = {
        "name": "sup",
        "description": "supervisor",
        "provider": "mock_cli",
        "model": _SENTINEL_MODEL,
        "system_prompt": "",
        "mcpServers": None,
    }
    fields.update(overrides)
    return AgentProfile(**fields)


def _raises_loader(module):
    return patch.object(module, "load_agent_profile", side_effect=AssertionError("reload"))


class TestManagerForwardsLaunchProfile:
    @pytest.mark.parametrize(
        "provider_type",
        [
            "kiro_cli",
            "claude_code",
            "codex",
            "kimi_cli",
            "muse_cli",
            "cursor_cli",
            "hermes",
            "antigravity_cli",
        ],
    )
    def test_create_provider_installs_launch_profile(self, provider_type):
        """The construction call-site threads the exact context object."""
        sentinel = _sentinel()
        manager = ProviderManager()
        provider = manager.create_provider(
            provider_type,
            "abcd1234",
            "cao-sup",
            "w-sup",
            "sup",
            launch_profile=sentinel,
        )
        assert provider._launch_profile is sentinel

    def test_absent_launch_profile_keeps_legacy_none(self):
        manager = ProviderManager()
        provider = manager.create_provider("hermes", "abcd1234", "cao-sup", "w-sup", "sup")
        assert provider._launch_profile is None


class TestKiro:
    def test_model_from_launch_profile_without_store_read(self):
        from cli_agent_orchestrator.providers import kiro_cli

        provider = kiro_cli.KiroCliProvider("t1", "cao-s", "w", "sup", launch_profile=_sentinel())
        with _raises_loader(kiro_cli):
            assert provider._get_profile_model() == _SENTINEL_MODEL


class TestClaude:
    def test_load_and_argv_from_launch_profile_without_store_read(self):
        import shlex
        from pathlib import Path

        from cli_agent_orchestrator.providers import claude_code

        sentinel = _sentinel(system_prompt="FROZEN-CLAUDE")
        provider = claude_code.ClaudeCodeProvider(
            "t1", "cao-s", "w", "sup", launch_profile=sentinel
        )
        with _raises_loader(claude_code):
            assert provider._load_profile() is sentinel
            command = provider._build_claude_command()
        # Not model-only: the frozen prompt is composed into the prompt file.
        assert "--model" in command and _SENTINEL_MODEL in command
        parts = shlex.split(command)
        prompt_path = parts[parts.index("--append-system-prompt-file") + 1]
        assert "FROZEN-CLAUDE" in Path(prompt_path).read_text(encoding="utf-8")

    def test_legacy_path_still_loads_by_name(self):
        from cli_agent_orchestrator.providers import claude_code

        provider = claude_code.ClaudeCodeProvider("t1", "cao-s", "w", "sup")
        with _raises_loader(claude_code):
            with pytest.raises(Exception):
                provider._load_profile()


class TestCodex:
    def test_material_from_launch_profile_without_store_read(self):
        from cli_agent_orchestrator.providers import codex

        sentinel = _sentinel(system_prompt="FROZEN-CODEX")
        provider = codex.CodexProvider("t1", "cao-s", "w", "sup", launch_profile=sentinel)
        with _raises_loader(codex):
            material = provider._resolve_codex_profile_material()
        assert material["profile"] is sentinel
        # Not model-only: the frozen system prompt composes the material.
        assert "FROZEN-CODEX" in material["system_prompt"]

    def test_legacy_path_still_loads_by_name(self):
        from cli_agent_orchestrator.providers import codex

        provider = codex.CodexProvider("t1", "cao-s", "w", "sup")
        with _raises_loader(codex):
            with pytest.raises(Exception):
                provider._resolve_codex_profile_material()


class TestKimi:
    def test_command_from_launch_profile_without_store_read(self):
        from pathlib import Path

        from cli_agent_orchestrator.providers import kimi_cli

        sentinel = _sentinel(system_prompt="FROZEN-KIMI")
        provider = kimi_cli.KimiCliProvider("t1", "cao-s", "w", "sup", launch_profile=sentinel)
        with _raises_loader(kimi_cli):
            assert provider._try_load_profile() is sentinel
            command = provider._build_kimi_command()
        # Not model-only: the frozen prompt is composed into the agent file.
        assert "--model" in command and _SENTINEL_MODEL in command
        system_md = Path(provider._temp_dir) / "system.md"
        assert "FROZEN-KIMI" in system_md.read_text(encoding="utf-8")

    def test_legacy_path_still_loads_by_name(self):
        from cli_agent_orchestrator.providers import kimi_cli

        provider = kimi_cli.KimiCliProvider("t1", "cao-s", "w", "sup")
        with _raises_loader(kimi_cli):
            with pytest.raises(Exception):
                provider._build_kimi_command()


class TestMuse:
    def test_model_from_launch_profile_without_store_read(self):
        from cli_agent_orchestrator.providers import muse_cli

        provider = muse_cli.MuseCliProvider("t1", "cao-s", "w", "sup", launch_profile=_sentinel())
        with _raises_loader(muse_cli):
            assert provider._resolve_model() == _SENTINEL_MODEL

    def test_argv_pins_frozen_model_and_effort_without_store_read(self):
        from cli_agent_orchestrator.providers import muse_cli

        provider = muse_cli.MuseCliProvider(
            "t1",
            "cao-s",
            "w",
            "sup",
            expected_model="override-model",
            expected_effort="xhigh",
            launch_profile=_sentinel(),
        )
        with _raises_loader(muse_cli):
            command = provider._build_command()
        # Not model-only: the pinned route carries the frozen effort flag too.
        assert "--model" in command and "override-model" in command
        assert "--reasoning-effort" in command and "xhigh" in command

    def test_argv_drops_frozen_prompt_without_store_read(self):
        """The v1 argv has no prompt channel: a nonempty frozen prompt is
        dropped, which is exactly what the sealed gate refuses."""
        from cli_agent_orchestrator.providers import muse_cli

        provider = muse_cli.MuseCliProvider(
            "t1", "cao-s", "w", "sup", launch_profile=_sentinel(system_prompt="FROZEN-PROMPT")
        )
        with _raises_loader(muse_cli):
            command = provider._build_command()
        assert "--model" in command and _SENTINEL_MODEL in command
        assert "FROZEN-PROMPT" not in command


class TestCursor:
    def test_command_from_launch_profile_without_store_read(self, tmp_path, monkeypatch):
        import json

        from cli_agent_orchestrator.providers import cursor_cli

        # Redirect the CAO tmp dir into scratch (see test_cursor_cli_unit).
        monkeypatch.setenv("CAO_TMP_DIR", str(tmp_path))
        provider = cursor_cli.CursorCliProvider(
            "t1",
            "cao-s",
            "w",
            "sup",
            launch_profile=_sentinel(
                mcpServers={"frozen-srv": {"command": "frozen-srv", "args": []}}
            ),
        )
        with (
            _raises_loader(cursor_cli),
            patch.object(cursor_cli.shutil, "which", return_value="/usr/local/bin/cursor-agent"),
        ):
            command = provider._build_cursor_command()
        # Not model-only: the frozen MCP map is synthesized into --plugin-dir.
        assert "--model" in command and _SENTINEL_MODEL in command
        assert "--plugin-dir" in command and "--approve-mcps" in command
        # The sentinel server reaches the per-session manifest on disk.
        manifests = list(tmp_path.rglob("plugin.json"))
        assert len(manifests) == 1
        assert (
            json.loads(manifests[0].read_text(encoding="utf-8"))["mcpServers"]["frozen-srv"][
                "command"
            ]
            == "frozen-srv"
        )

    def test_command_drops_frozen_prompt_without_store_read(self, tmp_path, monkeypatch):
        """Cursor has no prompt channel (backend rejects --system-prompt):
        a nonempty frozen prompt is dropped, which the sealed gate refuses."""
        from cli_agent_orchestrator.providers import cursor_cli

        monkeypatch.setenv("CAO_TMP_DIR", str(tmp_path))
        provider = cursor_cli.CursorCliProvider(
            "t1", "cao-s", "w", "sup", launch_profile=_sentinel(system_prompt="FROZEN-PROMPT")
        )
        with (
            _raises_loader(cursor_cli),
            patch.object(cursor_cli.shutil, "which", return_value="/usr/local/bin/cursor-agent"),
        ):
            command = provider._build_cursor_command()
        assert "--model" in command and _SENTINEL_MODEL in command
        assert "FROZEN-PROMPT" not in command

    def test_legacy_path_still_loads_by_name(self):
        from cli_agent_orchestrator.providers import cursor_cli

        provider = cursor_cli.CursorCliProvider("t1", "cao-s", "w", "sup")
        with (
            _raises_loader(cursor_cli),
            patch.object(cursor_cli.shutil, "which", return_value="/usr/local/bin/cursor-agent"),
        ):
            with pytest.raises(Exception):
                provider._build_cursor_command()


class TestHermes:
    def test_command_from_launch_profile_without_store_read(self):
        from cli_agent_orchestrator.providers import hermes

        provider = hermes.HermesProvider("t1", "cao-s", "w", "sup", launch_profile=_sentinel())
        with _raises_loader(hermes):
            command = provider._build_hermes_command()
        assert "--model" in command and _SENTINEL_MODEL in command

    def test_command_drops_frozen_prompt_and_policy_without_store_read(self):
        """The default Hermes argv has no prompt/policy channel: both are
        merely logged, so the sealed gate refuses them."""
        from cli_agent_orchestrator.providers import hermes

        provider = hermes.HermesProvider(
            "t1",
            "cao-s",
            "w",
            "sup",
            allowed_tools=["fs_read"],
            launch_profile=_sentinel(system_prompt="FROZEN-PROMPT"),
        )
        with _raises_loader(hermes):
            command = provider._build_hermes_command()
        assert "--model" in command and _SENTINEL_MODEL in command
        assert "FROZEN-PROMPT" not in command
        assert "fs_read" not in command

    def test_legacy_path_still_loads_by_name(self):
        from cli_agent_orchestrator.providers import hermes

        provider = hermes.HermesProvider("t1", "cao-s", "w", "sup")
        with _raises_loader(hermes):
            with pytest.raises(Exception):
                provider._build_hermes_command()


class TestAntigravity:
    def test_timeout_probe_from_launch_profile_without_store_read(self):
        from cli_agent_orchestrator.providers import antigravity_cli

        sentinel = _sentinel()
        provider = antigravity_cli.AntigravityCliProvider(
            "t1", "cao-s", "w", "sup", launch_profile=sentinel
        )
        with _raises_loader(antigravity_cli):
            assert provider._try_load_profile() is sentinel

    def test_argv_renders_frozen_prompt_and_model_without_store_read(self):
        from cli_agent_orchestrator.providers import antigravity_cli

        provider = antigravity_cli.AntigravityCliProvider(
            "t1", "cao-s", "w", "sup", launch_profile=_sentinel(system_prompt="FROZEN-AGY")
        )
        with (
            _raises_loader(antigravity_cli),
            # Hermetic binary probe: the real shutil.which("agy") is None
            # where agy is not installed, which must not fail the argv
            # assertions. The stub proves the probe passed (the command —
            # not the not-found error — was reached) and the launch path
            # below is the frozen one.
            patch("shutil.which", return_value="/usr/local/bin/agy"),
        ):
            command = provider._build_agy_command()
        # Not model-only: the frozen system prompt rides inline via -i.
        assert command.split()[0] == "agy"
        assert "--model" in command and _SENTINEL_MODEL in command
        assert "FROZEN-AGY" in command
