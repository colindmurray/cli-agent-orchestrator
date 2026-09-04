"""Sealed-profile capability matrix (cond-0817 round-4).

Each adapter owns its frozen-profile launch decision; the base default is
unsupported so future or unmarked adapters never silently become supported.
The manager routes the query to the adapter class without constructing a
provider or causing any effect. These tests pin every decision in the
coordinator's matrix, the construction parity of the routing table, and
the conservative defaults — including the mutations that must fail (gate
removed, base flipped, named-agent path allowed).
"""

from unittest.mock import patch

import pytest

from cli_agent_orchestrator.models.agent_profile import AgentProfile
from cli_agent_orchestrator.models.provider import ProviderType
from cli_agent_orchestrator.providers import manager as manager_module
from cli_agent_orchestrator.providers.base import BaseProvider, SealedProfileSupport
from cli_agent_orchestrator.providers.manager import ProviderManager


def _profile(**overrides):
    fields = {
        "name": "sup",
        "description": "supervisor",
        "provider": "mock_cli",
        "model": "frozen-model",
        "system_prompt": "frozen prompt",
    }
    fields.update(overrides)
    return AgentProfile(**fields)


class TestBaseDefault:
    def test_unmarked_adapter_is_unsupported(self):
        class FutureAdapter(BaseProvider):
            async def initialize(self) -> bool:  # pragma: no cover
                return True

            def get_status(self, output):  # pragma: no cover
                raise NotImplementedError

        decision = FutureAdapter.supports_sealed_profile(_profile())
        assert decision.supported is False
        assert decision.reason

    def test_none_profile_is_unsupported_even_for_supporters(self):
        from cli_agent_orchestrator.providers.kimi_cli import KimiCliProvider

        assert KimiCliProvider.supports_sealed_profile(None).supported is False

    def test_mock_inherits_conservative_default(self):
        from cli_agent_orchestrator.providers.mock_cli import MockCliProvider

        assert "supports_sealed_profile" not in MockCliProvider.__dict__
        assert MockCliProvider.supports_sealed_profile(_profile()).supported is False


class TestSupportDecisions:
    @pytest.mark.parametrize(
        "provider_type,adapter_module",
        [
            ("kimi_cli", "kimi_cli"),
            ("codex", "codex"),
            ("cursor_cli", "cursor_cli"),
            ("antigravity_cli", "antigravity_cli"),
            ("muse_cli", "muse_cli"),
        ],
    )
    def test_whole_profile_drivers_support_sealed(self, provider_type, adapter_module):
        decision = ProviderManager().sealed_profile_support(provider_type, _profile())
        assert decision.supported is True
        assert decision.reason

    def test_claude_full_decomposition_supports_sealed(self):
        from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider

        decision = ClaudeCodeProvider.supports_sealed_profile(_profile())
        assert decision.supported is True

    def test_hermes_without_wrapper_supports_sealed(self):
        from cli_agent_orchestrator.providers.hermes import HermesProvider

        decision = HermesProvider.supports_sealed_profile(_profile())
        assert decision.supported is True


class TestRejectDecisions:
    @pytest.mark.parametrize(
        "provider_type",
        ["kiro_cli", "opencode_cli", "copilot_cli"],
    )
    def test_named_agent_adapters_reject_sealed(self, provider_type):
        decision = ProviderManager().sealed_profile_support(provider_type, _profile())
        assert decision.supported is False
        assert "--agent" in decision.reason or "native" in decision.reason

    def test_claude_native_wrapper_rejects_sealed(self):
        from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider

        decision = ClaudeCodeProvider.supports_sealed_profile(
            _profile(native_agent="native-profile-B")
        )
        assert decision.supported is False
        assert "native-profile-B" in decision.reason

    def test_hermes_wrapper_rejects_sealed(self):
        from cli_agent_orchestrator.providers.hermes import HermesProvider

        decision = HermesProvider.supports_sealed_profile(
            _profile(hermesProfile="native-wrapper-B")
        )
        assert decision.supported is False
        assert "native-wrapper-B" in decision.reason

    def test_codex_named_profile_rejects_sealed(self):
        """A set codexProfile forwards mutable native --profile <name>.

        The composer emits ``--profile native-B`` from the same object the
        capability query sees, so sealed support must refuse and name the
        mutable ``~/.codex/config.toml`` block.
        """
        from cli_agent_orchestrator.providers.codex import CodexProvider

        decision = CodexProvider.supports_sealed_profile(_profile(codexProfile="native-B"))
        assert decision.supported is False
        assert "native-B" in decision.reason
        assert "config.toml" in decision.reason

    def test_antigravity_mcp_servers_reject_sealed(self):
        """Nonempty mcpServers merge into the shared mcp_config.json.

        There is no per-launch config path, so a sealed launch cannot pin
        what the supervisor consumes: refuse, naming the shared file.
        """
        from cli_agent_orchestrator.providers.antigravity_cli import AntigravityCliProvider

        decision = AntigravityCliProvider.supports_sealed_profile(
            _profile(mcpServers={"cao-mcp-server": {"command": "cao-mcp-server"}})
        )
        assert decision.supported is False
        assert "mcp_config.json" in decision.reason
        assert "cao-mcp-server" in decision.reason

    def test_antigravity_without_mcp_supports_sealed_with_ambient_scope(self):
        """No MCP material: supported, with the ambient scope stated.

        The reason pins the narrowed contract — CAO contributes no MCP
        material while ambient Antigravity configuration stays outside
        the receipt — so a future full-sealing policy flips this test.
        """
        from cli_agent_orchestrator.providers.antigravity_cli import AntigravityCliProvider

        decision = AntigravityCliProvider.supports_sealed_profile(_profile())
        assert decision.supported is True
        assert "ambient" in decision.reason

    def test_codex_without_named_profile_supports_sealed_without_native_flag(self):
        """No codexProfile: sealed support holds and the shared composer
        emits no native --profile (yolo/inline-material path)."""
        from cli_agent_orchestrator.providers.codex import (
            CodexProvider,
            compose_codex_core_args,
        )

        decision = CodexProvider.supports_sealed_profile(_profile(codexProfile=None))
        assert decision.supported is True
        core = compose_codex_core_args(
            codex_profile=None,
            codex_config=None,
            system_prompt="frozen prompt",
            mcp_servers=[],
            allowed_tools=[],
            trusted_project_root=None,
        )
        assert "--profile" not in core
        assert core[0] == "--yolo"


class TestRouting:
    def test_unknown_provider_type_is_unsupported(self):
        decision = ProviderManager().sealed_profile_support("bogus", _profile())
        assert decision.supported is False
        assert "bogus" in decision.reason

    def test_routing_covers_every_constructed_provider_type(self):
        """The class lookup mirrors the construction chain: a new manager
        branch without a routing entry fails here instead of drifting."""
        assert set(manager_module._ADAPTER_CLASS_BY_TYPE) == {
            member.value for member in ProviderType
        }

    def test_every_routed_class_answers_with_reasons(self):
        manager = ProviderManager()
        for provider_type in manager_module._ADAPTER_CLASS_BY_TYPE:
            decision = manager.sealed_profile_support(provider_type, _profile())
            assert isinstance(decision, SealedProfileSupport)
            assert isinstance(decision.supported, bool)
            assert decision.reason

    def test_capability_query_constructs_nothing(self):
        """The pre-effect gate must not build providers as a side effect."""
        with patch.object(
            manager_module.KiroCliProvider,
            "__init__",
            side_effect=AssertionError("constructed"),
        ):
            decision = ProviderManager().sealed_profile_support("kiro_cli", _profile())
        assert decision.supported is False
