"""Sealed-launch capability matrix (cond-0817 round-7).

Each adapter owns its frozen-material launch decision over one immutable
``SealedLaunchMaterial``: the base default is unsupported so future or
unmarked adapters never silently become supported. The manager only
dispatches to the adapter class without constructing a provider or
causing any effect.

The matrix pins, for every ProviderType and every nonempty
behavior-bearing field, whether the adapter consumes that field from
immutable per-launch material (supported) or would silently drop it
(refused). ``name``/``description``/``capabilities``/``tags`` are
discovery metadata and never trigger a refusal; ``provider`` was consumed
by routing and ``role`` by policy resolution.
"""

from unittest.mock import patch

import pytest

from cli_agent_orchestrator.models.agent_profile import AgentProfile
from cli_agent_orchestrator.models.provider import ProviderType
from cli_agent_orchestrator.providers import manager as manager_module
from cli_agent_orchestrator.providers.base import (
    BaseProvider,
    SealedLaunchMaterial,
    SealedProfileSupport,
)
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


def _material(sys_prompt="frozen prompt", skill="", policy=("*",), **overrides):
    """Material consistent with its profile, like the real builder emits.

    ``skill`` stands in for the composed catalog text for the profile's
    skill scope (``[]`` scopes to nothing, so the stand-in is empty);
    ``tools`` is the effective policy. ``system_prompt`` and
    ``allowedTools`` are mirrored onto the profile so the object could
    have come from ``build_sealed_launch_material``.
    """
    fields = {"system_prompt": sys_prompt, "allowedTools": list(policy)}
    fields["skills"] = [] if not skill else ["skill-a"]
    fields.update(overrides)
    profile = _profile(**fields)
    return SealedLaunchMaterial(
        profile=profile,
        model=profile.model,
        effort=None,
        system_prompt=profile.system_prompt or "",
        skill_text=skill,
        allowed_tools=tuple(policy),
    )


# Sentinel per behavior-bearing field: setting exactly this must flip the
# decision for every adapter that does not consume the field.
_FIELD_SENTINELS = {
    "system_prompt": {"sys_prompt": "FROZEN-PROMPT"},
    "skills": {"skill": "SKILL-CATALOG"},
    "allowedTools": {"policy": ("fs_read",)},
    "prompt": {"prompt": "q-prompt"},
    # ``tools`` below is the Q-CLI passthrough list (distinct from the
    # effective-policy ``policy`` helper kwarg); its value is assigned
    # after the literal so the two meanings never share one expression.
    "tools": {"tools_field": None},  # placeholder, replaced below
    "toolAliases": {"toolAliases": {"a": "b"}},
    "toolsSettings": {"toolsSettings": {"s": 1}},
    "resources": {"resources": ["r"]},
    "hooks": {"hooks": {"h": 1}},
    "useLegacyMcpJson": {"useLegacyMcpJson": True},
    "mcpServers": {"mcpServers": {"srv": {"command": "x"}}},
    "permissionMode": {"permissionMode": "bypassPermissions"},
    "provider_init_timeout": {"provider_init_timeout": 180},
    "container": {"container": {"path_maps": [{"host": "/h", "guest": "/g"}]}},
    "codexConfig": {"codexConfig": {"model_reasoning_effort": "xhigh"}},
    "native_agent": {"native_agent": "native-B"},
    "codexProfile": {"codexProfile": "native-B"},
    "hermesProfile": {"hermesProfile": "native-B"},
}
# Assigned here (not in the literal above) so the Q-CLI ``tools`` list and
# the ``policy`` helper kwarg never share one expression.
_FIELD_SENTINELS["tools"] = {"tools": ["q-tool"]}

_Q_FIELDS = {
    "prompt",
    "tools",
    "toolAliases",
    "toolsSettings",
    "resources",
    "hooks",
    "useLegacyMcpJson",
}
_NATIVE_FIELDS = {"native_agent", "codexProfile", "hermesProfile"}

# Fields each adapter refuses when nonempty: everything it would silently
# drop from the launch. Absence from the set means consumed (supported).
_REFUSE_FIELDS = {
    "claude_code": _Q_FIELDS | {"codexConfig", "codexProfile", "hermesProfile", "native_agent"},
    "codex": _Q_FIELDS
    | {
        "permissionMode",
        "provider_init_timeout",
        "container",
        "native_agent",
        "hermesProfile",
        "codexProfile",
    },
    "kimi_cli": _Q_FIELDS
    | {
        "permissionMode",
        "container",
        "codexConfig",
        "native_agent",
        "codexProfile",
        "hermesProfile",
    },
    "antigravity_cli": {"mcpServers"}
    | _Q_FIELDS
    | {
        "permissionMode",
        "container",
        "codexConfig",
        "native_agent",
        "codexProfile",
        "hermesProfile",
    },
    "cursor_cli": {"system_prompt", "skills", "allowedTools"}
    | _Q_FIELDS
    | {
        "permissionMode",
        "provider_init_timeout",
        "container",
        "codexConfig",
    }
    | _NATIVE_FIELDS,
    "muse_cli": {"system_prompt", "skills", "allowedTools", "mcpServers"}
    | _Q_FIELDS
    | {
        "permissionMode",
        "provider_init_timeout",
        "container",
        "codexConfig",
    }
    | _NATIVE_FIELDS,
    "hermes": {"system_prompt", "skills", "allowedTools", "mcpServers"}
    | _Q_FIELDS
    | {
        "permissionMode",
        "provider_init_timeout",
        "container",
        "codexConfig",
    }
    | _NATIVE_FIELDS,
}

_ALL_BEHAVIOR_FIELDS = sorted(_FIELD_SENTINELS)


def _material_with_field(field):
    """Baseline-empty material with exactly one behavior-bearing field set.

    Every cell reads its sentinel from ``_FIELD_SENTINELS`` — emptying a
    sentinel flips its refuse cells, so no cell can pass on vacuous input.
    """
    kwargs = {"sys_prompt": ""}
    kwargs.update(_FIELD_SENTINELS[field])
    return _material(**kwargs)


class TestBaseDefault:
    def test_unmarked_adapter_is_unsupported(self):
        class FutureAdapter(BaseProvider):
            async def initialize(self) -> bool:  # pragma: no cover
                return True

            def get_status(self, output):  # pragma: no cover
                raise NotImplementedError

        decision = FutureAdapter.supports_sealed_launch(_material())
        assert decision.supported is False
        assert decision.reason

    def test_none_material_is_unsupported_even_for_supporters(self):
        from cli_agent_orchestrator.providers.kimi_cli import KimiCliProvider

        assert KimiCliProvider.supports_sealed_launch(None).supported is False

    def test_mock_inherits_conservative_default(self):
        from cli_agent_orchestrator.providers.mock_cli import MockCliProvider

        assert "supports_sealed_launch" not in MockCliProvider.__dict__
        assert MockCliProvider.supports_sealed_launch(_material()).supported is False


class TestFieldConsumptionMatrix:
    @pytest.mark.parametrize(
        "provider,field",
        [
            (provider, field)
            for provider, fields in sorted(_REFUSE_FIELDS.items())
            for field in sorted(fields)
        ],
    )
    def test_dropped_field_refuses(self, provider, field):
        """Every nonempty behavior-bearing field the adapter does not
        consume refuses — the matrix fails if a new field or adapter
        forgets its gate."""
        decision = ProviderManager().sealed_launch_support(provider, _material_with_field(field))
        assert decision.supported is False
        assert decision.reason

    @pytest.mark.parametrize(
        "provider,field",
        [
            (provider, field)
            for provider in sorted(_REFUSE_FIELDS)
            for field in _ALL_BEHAVIOR_FIELDS
            if field not in _REFUSE_FIELDS[provider]
        ],
    )
    def test_consumed_field_stays_supported(self, provider, field):
        """Fields the adapter genuinely consumes never flip the decision."""
        decision = ProviderManager().sealed_launch_support(provider, _material_with_field(field))
        assert decision.supported is True

    def test_always_refuse_adapters_reject_bare_material(self):
        for provider in ("kiro_cli", "opencode_cli", "copilot_cli", "mock_cli"):
            decision = ProviderManager().sealed_launch_support(provider, _material())
            assert decision.supported is False

    def test_metadata_fields_never_refuse(self):
        """Discovery metadata is explicitly out of sealed launch authority."""
        from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider

        material = _material(
            sys_prompt="P",
            skill="S",
            policy=("fs_read",),
            mcpServers={"srv": {"command": "x"}},
            name="named",
            description="described",
            capabilities=["c"],
            tags=["t"],
        )
        assert ClaudeCodeProvider.supports_sealed_launch(material).supported is True


class TestNamedNativeRefusals:
    def test_claude_native_wrapper_rejects_sealed(self):
        from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider

        decision = ClaudeCodeProvider.supports_sealed_launch(
            _material(native_agent="native-profile-B")
        )
        assert decision.supported is False
        assert "native-profile-B" in decision.reason

    def test_hermes_wrapper_rejects_sealed(self):
        from cli_agent_orchestrator.providers.hermes import HermesProvider

        decision = HermesProvider.supports_sealed_launch(
            _material(sys_prompt="", hermesProfile="native-wrapper-B")
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

        decision = CodexProvider.supports_sealed_launch(_material(codexProfile="native-B"))
        assert decision.supported is False
        assert "native-B" in decision.reason
        assert "config.toml" in decision.reason

    def test_codex_without_named_profile_supports_sealed_without_native_flag(self):
        """No codexProfile: sealed support holds and the shared composer
        emits no native --profile (yolo/inline-material path)."""
        from cli_agent_orchestrator.providers.codex import (
            CodexProvider,
            compose_codex_core_args,
        )

        decision = CodexProvider.supports_sealed_launch(_material(codexProfile=None))
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

    def test_antigravity_mcp_servers_reject_sealed(self):
        """Nonempty mcpServers merge into the shared mcp_config.json.

        There is no per-launch config path, so a sealed launch cannot pin
        what the supervisor consumes: refuse, naming the shared file.
        """
        from cli_agent_orchestrator.providers.antigravity_cli import AntigravityCliProvider

        decision = AntigravityCliProvider.supports_sealed_launch(
            _material(mcpServers={"cao-mcp-server": {"command": "cao-mcp-server"}})
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

        decision = AntigravityCliProvider.supports_sealed_launch(_material())
        assert decision.supported is True
        assert "ambient" in decision.reason


class TestRouting:
    def test_unknown_provider_type_is_unsupported(self):
        decision = ProviderManager().sealed_launch_support("bogus", _material())
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
            decision = manager.sealed_launch_support(provider_type, _material())
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
            decision = ProviderManager().sealed_launch_support("kiro_cli", _material())
        assert decision.supported is False
