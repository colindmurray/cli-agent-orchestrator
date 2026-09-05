"""Sealed-launch preparation (cond-0817 repair).

Preparation validates and serializes every provider-consumed field
exactly once in ``session_service`` — after contract validation and
material construction, before ANY effect. These tests pin the two
rejected-head defects:

1. Malformed provider material (e.g. Codex ``mcpServers: {broken: {}}``)
   must refuse as HTTP 422 with zero mutation — previously the gate
   admitted it and ``clear_session_env`` deleted the persisted session
   env before terminal composition failed late with HTTP 500.
2. The strict contract parser's required set must include ``schema`` —
   previously a missing schema raised ``KeyError``/HTTP 500 instead of
   typed malformed HTTP 400.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.providers.base import (
    SEALED_TERMINAL_ID_PLACEHOLDER,
    SealedPreparationUnsupported,
    bind_codex_material,
    bind_sealed_mcp_document,
)
from cli_agent_orchestrator.providers.codex import bind_codex_material_json
from cli_agent_orchestrator.providers.manager import ProviderManager
from cli_agent_orchestrator.services import session_service
from cli_agent_orchestrator.services import supervisor_profile_receipt as spr
from cli_agent_orchestrator.services.supervisor_profile_receipt import (
    PROFILE_LAUNCH_CONTRACT_SCHEMA,
    ProfileLaunchConflict,
    build_profile_receipt,
    load_supervisor_launch_context,
    validate_profile_contract,
)
from cli_agent_orchestrator.utils import agent_profiles

_SERVICE = "cli_agent_orchestrator.services.terminal_service"


def _write_profile(
    store: Path,
    name: str,
    *,
    provider: str = "mock_cli",
    model: str | None = "test-model-1",
    effort: str | None = None,
    role: str = "supervisor",
    body: str = "Do supervision.",
    extra: tuple = (),
) -> Path:
    lines = ["---", f"name: {name}", f"description: {name} profile"]
    if provider is not None:
        lines.append(f"provider: {provider}")
    if role is not None:
        lines.append(f"role: {role}")
    if model is not None:
        lines.append(f"model: {model}")
    if effort is not None:
        lines.extend(["codexConfig:", f"  model_reasoning_effort: {effort}"])
    lines.extend(extra)
    lines.append("---")
    lines.append(body)
    path = store / f"{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_broken_mcp_profile(store: Path, name: str, provider: str, **kwargs) -> Path:
    """A profile whose MCP entry has no usable transport (no command/url)."""
    return _write_profile(
        store,
        name,
        provider=provider,
        extra=(
            "mcpServers:",
            "  broken: {}",
        ),
        **kwargs,
    )


@pytest.fixture
def profile_store(tmp_path, monkeypatch):
    """Isolate profile resolution to one scratch local store."""
    from cli_agent_orchestrator.services import settings_service

    store = tmp_path / "agent-store"
    store.mkdir()
    monkeypatch.setattr(agent_profiles, "LOCAL_AGENT_STORE_DIR", store)
    monkeypatch.setattr(settings_service, "get_agent_dirs", lambda: {})
    monkeypatch.setattr(settings_service, "get_extra_agent_dirs", lambda: [])
    monkeypatch.setattr(settings_service, "get_disabled_agent_dirs", lambda: [])
    return store


@pytest.fixture
def launched_provider():
    provider = AsyncMock()
    provider.initialize.return_value = True
    provider.shell_baseline = None
    return provider


def _hermetic_backend():
    backend = MagicMock()
    backend.session_exists.return_value = False
    backend.window_identity.return_value = {
        "pane_id": "%1",
        "window_id": "@1",
        "server_socket_path": "/tmp/tmux",
        "session_id": "$1",
        "pane_pid": 12345,
    }
    backend.supports_event_inbox.return_value = False
    backend.supports_pane_identity.return_value = False
    return backend


def _contract_for(context) -> dict:
    return {
        "schema": PROFILE_LAUNCH_CONTRACT_SCHEMA,
        "profile": context.profile_name,
        "role": "supervisor",
        "provider": context.provider,
        "model": context.model,
        "effort": context.effort,
        "provenance": context.provenance,
        "source_path": context.source_path,
        "sha256": context.sha256,
    }


class TestParserRequiredSet:
    @pytest.fixture
    def context(self, profile_store):
        _write_profile(profile_store, "sup")
        return load_supervisor_launch_context("sup")

    @pytest.mark.parametrize(
        "field",
        [
            "schema",
            "profile",
            "role",
            "provider",
            "model",
            "effort",
            "provenance",
            "source_path",
            "sha256",
        ],
    )
    def test_every_required_field_removed_individually_is_malformed(self, context, field):
        """The required set is exactly the nine contract fields.

        Dropping any one — including ``schema``, which once raised
        ``KeyError``/HTTP 500 because the required set omitted it — is
        a typed malformed 400, never an internal error.
        """
        raw = _contract_for(context)
        del raw[field]
        with pytest.raises(spr.ProfileContractMalformed):
            spr.parse_profile_contract(raw)

    def test_missing_schema_is_malformed_not_keyerror(self, context):
        """The round-10 regression pin: no KeyError escapes the parser."""
        raw = _contract_for(context)
        del raw["schema"]
        with pytest.raises(spr.ProfileContractMalformed) as exc_info:
            spr.parse_profile_contract(raw)
        assert "schema" in str(exc_info.value)

    def test_extra_field_is_malformed(self, context):
        raw = _contract_for(context)
        raw["unexpected"] = 1
        with pytest.raises(spr.ProfileContractMalformed):
            spr.parse_profile_contract(raw)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("schema", "wrong-schema"),
            ("role", "worker"),
            ("profile", 42),
            ("provider", ""),
            ("provenance", ""),
            ("model", 42),
            ("effort", 42),
            ("source_path", "relative/path.md"),
            ("sha256", "xyz"),
        ],
    )
    def test_wrong_values_keep_malformed_classification(self, context, field, value):
        """Wrong schema/role/types/path/SHA stay 400 territory."""
        raw = _contract_for(context)
        raw[field] = value
        with pytest.raises(spr.ProfileContractMalformed):
            spr.parse_profile_contract(raw)

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("model", "other-model"),
            ("provenance", "built-in"),
            ("sha256", "0" * 64),
        ],
    )
    def test_well_formed_drift_keeps_conflict_classification(self, context, field, bad):
        """Well-formed values that differ stay 409 territory, not 400."""
        raw = _contract_for(context)
        raw[field] = bad
        parsed = spr.parse_profile_contract(raw)
        with pytest.raises(ProfileLaunchConflict) as exc_info:
            validate_profile_contract(parsed, context)
        assert field in [entry["field"] for entry in exc_info.value.divergent_fields]

    def test_uppercase_sha_still_normalizes(self, context):
        raw = _contract_for(context)
        raw["sha256"] = raw["sha256"].upper()
        parsed = spr.parse_profile_contract(raw)
        assert parsed["sha256"] == context.sha256
        validate_profile_contract(parsed, context)
        assert build_profile_receipt(context)["sha256"] == context.sha256


class TestBrokenCodexMcpZeroMutation:
    @pytest.mark.asyncio
    async def test_broken_codex_mcp_refuses_422_with_zero_mutation(
        self, profile_store, isolated_memory_db
    ):
        """The accepted-head repro, repaired: broken Codex MCP + persisted env.

        ``mcpServers: {broken: {}}`` passes the capability gate (the
        predicate cannot see the transport-less shape) but preparation
        refuses it before any effect: the persisted session env for the
        launch name survives, and clear_session_env, tmux, the DB row,
        the roster bind, and provider construction are all uncalled.
        Previously this deleted the env row and failed late with 500.
        """
        from cli_agent_orchestrator.clients.database import SessionLocal, TerminalModel
        from cli_agent_orchestrator.providers.manager import ProviderManager
        from cli_agent_orchestrator.services import session_env, terminal_service

        session_name = "cao-broken-codex"
        session_env.set_session_env(session_name, {"KEEP": "1"})
        try:
            _write_broken_mcp_profile(profile_store, "sup", "codex")
            context = load_supervisor_launch_context("sup")
            assert context.provider == "codex"
            # The gate admits: the predicate sees no dropped field.
            material = spr.build_sealed_launch_material(context)
            assert ProviderManager().sealed_launch_support("codex", material).supported is True
            with (
                patch(
                    "cli_agent_orchestrator.services.session_service.create_terminal",
                    new=AsyncMock(),
                ) as mock_create,
                patch.object(terminal_service, "clear_session_env") as mock_clear,
                patch.object(
                    ProviderManager,
                    "create_provider",
                    side_effect=AssertionError("provider constructed"),
                ),
                patch.object(
                    terminal_service,
                    "_roster_bind_unmanaged",
                    side_effect=AssertionError("roster bound"),
                ),
                patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
            ):
                with pytest.raises(spr.ProfileLaunchUnsupported) as exc_info:
                    (
                        await session_service.create_session(
                            provider=None,
                            agent_profile="sup",
                            session_name="broken-codex",
                            profile_contract=_contract_for(context),
                        )
                    ).terminal
            assert exc_info.value.provider == "codex"
            assert "broken" in exc_info.value.reason
            mock_create.assert_not_called()
            mock_clear.assert_not_called()
            # The persisted env survives byte-for-byte...
            assert session_env.get_session_env(session_name) == {"KEEP": "1"}
            # ...and no terminal row exists to carry a receipt.
            with SessionLocal() as db:
                assert db.query(TerminalModel).count() == 0
        finally:
            session_env._session_forwarded_env.pop(session_name, None)
            try:
                session_env._delete_row(session_name)
            except Exception:
                pass


class TestPreparationOrdering:
    @pytest.mark.asyncio
    async def test_prepare_runs_before_clear_tmux_db_provider(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """The effect order is prepare < clear < tmux < DB < provider.

        A valid sealed Kimi launch through the real session boundary
        records each stage exactly once, in order: preparation (pure),
        the stale-env pre-clear, tmux window creation, the durable row,
        and provider construction.
        """
        from cli_agent_orchestrator.providers.manager import ProviderManager
        from cli_agent_orchestrator.services import terminal_service

        events: list[str] = []
        _write_profile(profile_store, "sup", provider="kimi_cli")
        context = load_supervisor_launch_context("sup")

        real_prepare = ProviderManager.prepare_sealed_launch

        def _rec_prepare(self, provider_type, material):
            events.append("prepare")
            return real_prepare(self, provider_type, material)

        real_db_create = terminal_service.db_create_terminal

        def _rec_db(*args, **kwargs):
            events.append("db")
            return real_db_create(*args, **kwargs)

        backend = _hermetic_backend()
        backend.create_session.side_effect = lambda *a, **k: events.append("tmux")

        # The gate and preparation run on the real manager singleton
        # (session_service holds its own reference); only provider
        # construction is stubbed, with its order recorded.
        service_provider_manager = MagicMock()

        def _rec_create_provider(*args, **kwargs):
            events.append("provider")
            return launched_provider

        service_provider_manager.create_provider.side_effect = _rec_create_provider

        with ExitStack() as stack:
            stack.enter_context(patch("cli_agent_orchestrator.backends.registry._backend", backend))
            stack.enter_context(patch(f"{_SERVICE}.provider_manager", service_provider_manager))
            stack.enter_context(patch(f"{_SERVICE}.fifo_manager"))
            stack.enter_context(patch(f"{_SERVICE}.status_monitor"))
            clear_mock = stack.enter_context(patch(f"{_SERVICE}.clear_session_env"))
            clear_mock.side_effect = lambda *a, **k: events.append("clear")
            stack.enter_context(patch(f"{_SERVICE}._register_incarnation"))
            stack.enter_context(patch(f"{_SERVICE}.generate_terminal_id", return_value="abcd4240"))
            stack.enter_context(
                patch(f"{_SERVICE}.generate_session_name", return_value="cao-prep-order")
            )
            stack.enter_context(patch(f"{_SERVICE}.generate_window_name", return_value="w-sup"))
            stack.enter_context(
                patch.object(terminal_service, "db_create_terminal", side_effect=_rec_db)
            )
            stack.enter_context(
                patch.object(ProviderManager, "prepare_sealed_launch", _rec_prepare)
            )
            stack.enter_context(
                patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event")
            )
            terminal = (
                await session_service.create_session(
                    provider=None,
                    agent_profile="sup",
                    profile_contract=_contract_for(context),
                )
            ).terminal

        assert terminal.profile_receipt == build_profile_receipt(context)
        assert events == ["prepare", "clear", "tmux", "db", "provider"], events


class TestPreparedArtifactIdentity:
    @pytest.mark.asyncio
    async def test_one_composer_call_and_exact_identity_through_bootstrap_and_resume(
        self, profile_store, isolated_memory_db
    ):
        """One Codex composer call; the same artifact reaches bootstrap+resume.

        The session boundary prepares exactly once (the managed material
        builder runs once); ``create_terminal`` binds the terminal id by
        pure substitution; the bootstrap and the resumed provider
        consume that bound object by identity — no second composition,
        no name-based reload. The prepared value itself keeps the
        placeholder (immutability pin).
        """
        import cli_agent_orchestrator.services.managed_provider_bridge as bridge
        from cli_agent_orchestrator.providers import codex as codex_module
        from cli_agent_orchestrator.providers.base import SEALED_TERMINAL_ID_PLACEHOLDER
        from cli_agent_orchestrator.providers.codex import CodexProvider
        from cli_agent_orchestrator.providers.manager import ProviderManager
        from cli_agent_orchestrator.services import native_attachment, unmanaged_native_identity

        _write_profile(
            profile_store,
            "sup",
            provider="codex",
            model="ident-model-3",
            extra=(
                "mcpServers:",
                "  srv:",
                "    command: /bin/true",
            ),
        )
        context = load_supervisor_launch_context("sup")
        assert context.provider == "codex"

        composer_calls: list = []
        real_compose = bridge._profile_material_from_profile

        def _counting_compose(*args, **kwargs):
            composer_calls.append(1)
            return real_compose(*args, **kwargs)

        prepare_calls: list = []
        prepared_values: list = []
        real_prepare = ProviderManager.prepare_sealed_launch

        def _counting_prepare(self, provider_type, material):
            prepare_calls.append(1)
            prepared = real_prepare(self, provider_type, material)
            prepared_values.append(prepared)
            return prepared

        seen: dict = {}

        def _capture_bootstrap(**kwargs):
            seen.update(kwargs)
            return {
                "native_session_id": "codex-ident-native",
                "acquisition_method": (native_attachment.ACQUISITION_ZERO_TURN_BOOTSTRAP),
                "working_directory": "/tmp",
                "model": "ident-model-3",
                "effort": None,
                "binary_path": None,
            }

        built: list = []
        real_manager = ProviderManager()
        real_create = real_manager.create_provider

        def _capture_create(*args, **kwargs):
            instance = real_create(*args, **kwargs)
            built.append((instance, kwargs))
            return instance

        real_manager.create_provider = MagicMock(side_effect=_capture_create)
        backend = _hermetic_backend()

        with ExitStack() as stack:
            stack.enter_context(patch("cli_agent_orchestrator.backends.registry._backend", backend))
            stack.enter_context(patch(f"{_SERVICE}.provider_manager", real_manager))
            stack.enter_context(patch(f"{_SERVICE}.fifo_manager"))
            stack.enter_context(patch(f"{_SERVICE}.status_monitor"))
            stack.enter_context(patch(f"{_SERVICE}.clear_session_env"))
            stack.enter_context(patch(f"{_SERVICE}._register_incarnation"))
            stack.enter_context(patch(f"{_SERVICE}.generate_terminal_id", return_value="abcd4242"))
            stack.enter_context(
                patch(f"{_SERVICE}.generate_session_name", return_value="cao-ident")
            )
            stack.enter_context(patch(f"{_SERVICE}.generate_window_name", return_value="w-sup"))
            stack.enter_context(
                patch.object(
                    bridge, "_profile_material_from_profile", side_effect=_counting_compose
                )
            )
            stack.enter_context(
                patch.object(ProviderManager, "prepare_sealed_launch", _counting_prepare)
            )
            stack.enter_context(
                patch.object(
                    codex_module.CodexProvider,
                    "initialize",
                    new=AsyncMock(return_value=True),
                )
            )
            stack.enter_context(
                patch.object(
                    unmanaged_native_identity,
                    "resolve_pre_task_identity",
                    side_effect=_capture_bootstrap,
                )
            )
            stack.enter_context(
                patch(
                    f"{_SERVICE}.load_agent_profile",
                    side_effect=AssertionError("terminal reloaded by name"),
                )
            )
            stack.enter_context(
                patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event")
            )
            terminal = (
                await session_service.create_session(
                    provider=None,
                    agent_profile="sup",
                    profile_contract=_contract_for(context),
                )
            ).terminal

        assert terminal.profile_receipt == build_profile_receipt(context)
        # Exactly one preparation and one composer call across the flow.
        assert prepare_calls == [1]
        assert composer_calls == [1]
        # Bootstrap and provider construction share one bound object...
        bootstrap_material = seen["codex_profile_material"]
        assert len(built) == 1
        instance, create_kwargs = built[0]
        assert isinstance(instance, CodexProvider)
        assert create_kwargs["codex_profile_material"] is bootstrap_material
        # ...which is the prepared value bound structurally at the
        # recorded injection sites. (The test context supplies the
        # contract *values*; the session performs its own single
        # boundary read, so profile equality — not identity with the
        # test object — is the correct pin here. Object identity is
        # pinned between bootstrap and provider above.)
        assert "profile" not in bootstrap_material
        assert set(bootstrap_material) == {
            "system_prompt",
            "allowed_tools",
            "mcp_servers",
            "codex_config",
        }
        env_values = [
            item["value"] for entry in bootstrap_material["mcp_servers"] for item in entry["env"]
        ]
        assert "abcd4242" in env_values
        assert SEALED_TERMINAL_ID_PLACEHOLDER not in env_values
        # The prepared value itself keeps the placeholder (immutability
        # pin): binding produced a new bound object, not a mutation.
        assert prepared_values[0].codex_material_json is not None
        assert SEALED_TERMINAL_ID_PLACEHOLDER.encode("ascii") in (
            prepared_values[0].codex_material_json
        )
        # The resumed TUI consumes the same object, never a reload: the
        # provider's material resolver returns it by identity.
        assert instance._resolve_codex_profile_material() is bootstrap_material


class TestMalformedMcpMatrix:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["claude_code", "kimi_cli", "cursor_cli"])
    async def test_malformed_mcp_returns_422_pre_effect(self, profile_store, provider):
        """A transport-less MCP entry refuses pre-effect on every emitter.

        The gate admits (shape is not a dropped field) but preparation
        enforces the one-transport rule, so the refusal names the entry
        and ``create_terminal`` — with its clear, tmux, DB, file, and
        provider effects — is never entered.
        """
        from cli_agent_orchestrator.services import terminal_service

        broken_extra = (
            "mcpServers:",
            "  broken: {}",
        )
        if provider == "cursor_cli":
            # Cursor supports only model+MCP: empty prompt/skills/policy.
            _write_profile(
                profile_store,
                "sup",
                provider=provider,
                body="",
                extra=broken_extra + ('allowedTools: ["*"]',),
            )
        else:
            _write_broken_mcp_profile(profile_store, "sup", provider)
        context = load_supervisor_launch_context("sup")
        assert context.provider == provider
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch.object(terminal_service, "clear_session_env") as mock_clear,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported) as exc_info:
                (
                    await session_service.create_session(
                        provider=None,
                        agent_profile="sup",
                        profile_contract=_contract_for(context),
                    )
                ).terminal
        assert exc_info.value.provider == provider
        assert "broken" in exc_info.value.reason
        assert "exactly one usable transport" in exc_info.value.reason
        mock_create.assert_not_called()
        mock_clear.assert_not_called()


class TestAntigravityMalformedMcp:
    @pytest.mark.asyncio
    async def test_malformed_mcp_refuses_before_shared_file_access(
        self, profile_store, tmp_path, monkeypatch
    ):
        """Any nonempty mcpServers — even malformed — refuses pre-effect.

        The gate and preparation both refuse before the shared
        ``mcp_config.json`` is read or written: the ambient content
        survives and the register/unregister paths are never entered.
        """
        from cli_agent_orchestrator.providers.antigravity_cli import AntigravityCliProvider
        from cli_agent_orchestrator.providers.manager import ProviderManager

        fake_home = tmp_path / "home"
        shared = fake_home / ".gemini" / "config" / "mcp_config.json"
        shared.parent.mkdir(parents=True)
        ambient = {"mcpServers": {"ambient": {"command": "ambient"}}}
        shared.write_text(json.dumps(ambient), encoding="utf-8")
        monkeypatch.setenv("HOME", str(fake_home))
        _write_broken_mcp_profile(profile_store, "sup", "antigravity_cli")
        context = load_supervisor_launch_context("sup")
        assert context.provider == "antigravity_cli"
        # Preparation refuses directly too, naming the shared file.
        material = spr.build_sealed_launch_material(context)
        with pytest.raises(Exception) as prep_exc:
            ProviderManager().prepare_sealed_launch("antigravity_cli", material)
        assert "mcp_config.json" in str(prep_exc.value)
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch.object(
                AntigravityCliProvider,
                "_register_mcp_servers",
                side_effect=AssertionError("shared-file write"),
            ),
            patch.object(
                AntigravityCliProvider,
                "_unregister_mcp_servers",
                side_effect=AssertionError("shared-file read"),
            ),
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported) as exc_info:
                (
                    await session_service.create_session(
                        provider=None,
                        agent_profile="sup",
                        profile_contract=_contract_for(context),
                    )
                ).terminal
        assert "mcp_config.json" in exc_info.value.reason
        mock_create.assert_not_called()
        assert json.loads(shared.read_text(encoding="utf-8")) == ambient


class TestLegacyBehavior:
    @pytest.mark.asyncio
    async def test_unsupported_no_contract_launches_without_preparation(self, profile_store):
        """Kiro without a contract keeps the ordinary legacy launch.

        No sealed kwargs thread through and preparation never runs —
        the repair adds nothing to the path it does not own.
        """
        from cli_agent_orchestrator.providers.manager import ProviderManager

        _write_profile(profile_store, "sup", provider="kiro_cli")
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch.object(
                ProviderManager,
                "prepare_sealed_launch",
                side_effect=AssertionError("prepared on legacy path"),
            ),
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-legacy")
            await session_service.create_session(provider=None, agent_profile="sup")
        kwargs = mock_create.call_args.kwargs
        assert "profile_launch_context" not in kwargs
        assert "sealed_launch_material" not in kwargs
        assert "prepared_sealed_launch" not in kwargs

    @pytest.mark.asyncio
    async def test_supported_no_contract_still_threads_prepared(self, profile_store):
        """Kimi without a contract keeps threading the frozen material.

        The supported no-contract path records its exact receipt as
        before; preparation now runs once and its value threads through
        carrying no payload beyond the validated (empty) MCP map.
        """
        _write_profile(profile_store, "sup", provider="kimi_cli")
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-legacy-sealed")
            await session_service.create_session(provider=None, agent_profile="sup")
        kwargs = mock_create.call_args.kwargs
        assert kwargs["provider"] == "kimi_cli"
        prepared = kwargs["prepared_sealed_launch"]
        assert prepared.provider == "kimi_cli"
        assert prepared.codex_material_json is None
        assert prepared.mcp_servers_json == b"{}"
        assert prepared.mcp_document_json == b'{"mcpServers":{}}'
        assert kwargs["sealed_launch_material"].profile is kwargs["profile_launch_context"].profile


class TestNonJsonMcpShapes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["claude_code", "kimi_cli", "cursor_cli"])
    @pytest.mark.parametrize(
        "case,extra",
        [
            (
                "unquoted-date-env",
                (
                    "mcpServers:",
                    "  srv:",
                    "    command: /bin/true",
                    "    env:",
                    "      START: 2024-01-01",
                ),
            ),
            (
                "nested-list-env",
                (
                    "mcpServers:",
                    "  srv:",
                    "    command: /bin/true",
                    "    env:",
                    "      LIMITS: [1, 2]",
                ),
            ),
            (
                "int-args",
                (
                    "mcpServers:",
                    "  srv:",
                    "    command: /bin/true",
                    "    args: [8080]",
                ),
            ),
            (
                "nan-timeout",
                (
                    "mcpServers:",
                    "  srv:",
                    "    command: /bin/true",
                    "    tool_timeout_sec: .nan",
                ),
            ),
            (
                "string-timeout",
                (
                    "mcpServers:",
                    "  srv:",
                    "    command: /bin/true",
                    "    tool_timeout_sec: fast",
                ),
            ),
            (
                "date-headers",
                (
                    "mcpServers:",
                    "  srv:",
                    "    url: https://example.invalid/mcp",
                    "    headers:",
                    "      X-On: 2024-01-01",
                ),
            ),
        ],
    )
    async def test_non_json_mcp_shapes_refuse_pre_effect(
        self, profile_store, isolated_memory_db, tmp_path, monkeypatch, provider, case, extra
    ):
        """Unquoted YAML dates and non-JSON nested values yield 422.

        The refusal happens in preparation — before create_terminal,
        so the Claude tmp dir and the Cursor plugin dir stay untouched
        (no strict file, no manifest, no shared-file access).
        """
        from cli_agent_orchestrator.services import terminal_service

        monkeypatch.setenv("CAO_TMP_DIR", str(tmp_path / "cao-tmp"))
        if provider == "cursor_cli":
            _write_profile(
                profile_store,
                "sup",
                provider=provider,
                body="",
                extra=extra + ('allowedTools: ["*"]',),
            )
        else:
            _write_profile(profile_store, "sup", provider=provider, extra=extra)
        context = load_supervisor_launch_context("sup")

        manager = ProviderManager()
        material = spr.build_sealed_launch_material(context)
        assert manager.sealed_launch_support(provider, material).supported is True
        with pytest.raises(SealedPreparationUnsupported):
            manager.prepare_sealed_launch(provider, material)

        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch.object(terminal_service, "clear_session_env") as mock_clear,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported):
                (
                    await session_service.create_session(
                        provider=None,
                        agent_profile="sup",
                        profile_contract=_contract_for(context),
                    )
                ).terminal
        mock_create.assert_not_called()
        mock_clear.assert_not_called()
        assert (
            list((tmp_path / "cao-tmp").glob("**/*")) == []
            if (tmp_path / "cao-tmp").exists()
            else True
        )

    def test_never_str_coerces(self):
        """The sealed path has no str() fallback: audit the seam source.

        Docstrings may name the forbidden pattern; the executable body
        may not contain it.
        """
        import ast
        import inspect

        from cli_agent_orchestrator.providers import base as base_module

        for name in (
            "require_json_safe",
            "_require_string_map",
            "sealed_mcp_server_config",
            "prepare_sealed_mcp_documents",
            "dump_sealed_json",
            "bind_sealed_mcp_document",
            "bind_codex_material",
        ):
            (node,) = ast.parse(inspect.getsource(getattr(base_module, name))).body
            assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)), name
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            executable = "\n".join(ast.unparse(part) for part in body)
            assert "str(value)" not in executable, name
            assert "str(item)" not in executable, name


class TestFinalBytesIdentity:
    @pytest.fixture
    def mcp_profile(self, profile_store):
        _write_profile(
            profile_store,
            "sup",
            provider="claude_code",
            extra=(
                "mcpServers:",
                "  srv:",
                "    command: /bin/true",
                "    args: [--flag]",
                "    env:",
                "      FOO: bar",
            ),
        )
        return load_supervisor_launch_context("sup")

    def test_claude_file_bytes_equal_prepared(self, mcp_profile, tmp_path, monkeypatch):
        """The strict file is the prepared document, bound at its sites."""
        import json as _json

        import cli_agent_orchestrator.providers.claude_code as claude_module
        from cli_agent_orchestrator.providers import base as base_module
        from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider

        monkeypatch.setattr(claude_module, "CAO_HOME_DIR", tmp_path)
        dumps_calls: list = []
        real_dumps = base_module.dump_sealed_json

        def _counting_dumps(value, *, source):
            dumps_calls.append(source)
            return real_dumps(value, source=source)

        monkeypatch.setattr(base_module, "dump_sealed_json", _counting_dumps)
        prepared = ProviderManager().prepare_sealed_launch(
            "claude_code", spr.build_sealed_launch_material(mcp_profile)
        )
        assert dumps_calls == ["sealed MCP servers"]
        assert prepared.terminal_id_binding_sites == ("srv",)
        bound = bind_sealed_mcp_document(
            prepared.mcp_document_json, "tid9", sites=prepared.terminal_id_binding_sites
        )
        provider = ClaudeCodeProvider(
            "tid9",
            "s",
            "w",
            "sup",
            ["*"],
            launch_profile=mcp_profile.profile,
            prepared_sealed_launch=prepared,
        )
        command = provider._build_claude_command(mcp_profile.profile)
        assert "--strict-mcp-config" in command
        written = (tmp_path / "tmp" / "tid9.mcp.json").read_bytes()
        assert written == bound
        assert _json.loads(written)["mcpServers"]["srv"]["env"]["CAO_TERMINAL_ID"] == "tid9"

    def test_kimi_argv_text_equals_prepared(self, mcp_profile, tmp_path, monkeypatch):
        """The --mcp-config text is the prepared text, bound at its sites."""
        import shlex

        from cli_agent_orchestrator.providers import base as base_module
        from cli_agent_orchestrator.providers.kimi_cli import KimiCliProvider

        monkeypatch.setenv("CAO_TMP_DIR", str(tmp_path / "cao-tmp"))
        monkeypatch.setattr(KimiCliProvider, "_ensure_mcp_timeout", classmethod(lambda cls: None))
        dumps_calls: list = []
        real_dumps = base_module.dump_sealed_json

        def _counting_dumps(value, *, source):
            dumps_calls.append(source)
            return real_dumps(value, source=source)

        monkeypatch.setattr(base_module, "dump_sealed_json", _counting_dumps)
        material = spr.build_sealed_launch_material(mcp_profile)
        prepared = ProviderManager().prepare_sealed_launch("kimi_cli", material)
        assert dumps_calls == ["sealed MCP servers"]
        assert prepared.terminal_id_binding_sites == ("srv",)
        expected_text = bind_sealed_mcp_document(
            prepared.mcp_servers_json,
            "tid9",
            sites=prepared.terminal_id_binding_sites,
            wrapped=False,
        ).decode("utf-8")
        provider = KimiCliProvider(
            "tid9",
            "s",
            "w",
            "sup",
            ["*"],
            expected_model="m",
            launch_profile=mcp_profile.profile,
            prepared_sealed_launch=prepared,
        )
        command = provider._build_kimi_command()
        parts = shlex.split(command)
        assert expected_text in parts
        assert parts[parts.index("--mcp-config") + 1] == expected_text

    def test_cursor_manifest_bytes_equal_prepared(self, profile_store, tmp_path, monkeypatch):
        """The plugin manifest is the prepared document, bound at its sites."""
        import json as _json

        _write_profile(
            profile_store,
            "cur",
            provider="cursor_cli",
            body="",
            extra=(
                "mcpServers:",
                "  srv:",
                "    command: /bin/true",
                "    env:",
                "      FOO: bar",
                'allowedTools: ["*"]',
            ),
        )
        context = load_supervisor_launch_context("cur")
        from cli_agent_orchestrator.providers import base as base_module
        from cli_agent_orchestrator.providers import cursor_cli as cursor_module
        from cli_agent_orchestrator.providers.cursor_cli import CursorCliProvider

        monkeypatch.setenv("CAO_TMP_DIR", str(tmp_path / "cao-tmp"))
        monkeypatch.setattr(cursor_module.shutil, "which", lambda name: "/usr/bin/agent")
        dumps_calls: list = []
        real_dumps = base_module.dump_sealed_json

        def _counting_dumps(value, *, source):
            dumps_calls.append(source)
            return real_dumps(value, source=source)

        monkeypatch.setattr(base_module, "dump_sealed_json", _counting_dumps)
        prepared = ProviderManager().prepare_sealed_launch(
            "cursor_cli", spr.build_sealed_launch_material(context)
        )
        assert dumps_calls == ["sealed MCP servers"]
        assert prepared.terminal_id_binding_sites == ("srv",)
        bound = bind_sealed_mcp_document(
            prepared.mcp_document_json, "tid9", sites=prepared.terminal_id_binding_sites
        )
        provider = CursorCliProvider(
            "tid9",
            "s",
            "w",
            "cur",
            ["*"],
            launch_profile=context.profile,
            prepared_sealed_launch=prepared,
        )
        command = provider._build_cursor_command()
        assert "--approve-mcps" in command
        written = (tmp_path / "cao-tmp" / "tid9-cursor-plugins" / "plugin.json").read_bytes()
        assert written == bound
        assert _json.loads(written)["mcpServers"]["srv"]["env"]["CAO_TERMINAL_ID"] == "tid9"

    def test_codex_material_final_and_counted_once(self, profile_store):
        """Codex preparation serializes the final material exactly once."""
        import json as _json

        _write_profile(
            profile_store,
            "sup",
            provider="codex",
            extra=(
                "mcpServers:",
                "  srv:",
                "    command: /bin/true",
            ),
        )
        context = load_supervisor_launch_context("sup")
        from cli_agent_orchestrator.providers import codex as codex_module

        dumps_calls: list = []
        real_dumps = codex_module.dump_sealed_json

        def _counting_dumps(value, *, source):
            dumps_calls.append(source)
            return real_dumps(value, source=source)

        codex_module.dump_sealed_json = _counting_dumps
        try:
            prepared = ProviderManager().prepare_sealed_launch(
                "codex", spr.build_sealed_launch_material(context)
            )
        finally:
            codex_module.dump_sealed_json = real_dumps
        assert dumps_calls == ["sealed Codex material"]
        assert prepared.terminal_id_binding_sites == ("srv",)
        final = _json.loads(prepared.codex_material_json)
        assert set(final) == {"system_prompt", "allowed_tools", "mcp_servers", "codex_config"}
        assert "profile" not in final
        # The emitter input is the prepared content bound structurally at
        # the recorded sites — every other value byte-identical.
        bound = bind_codex_material_json(
            prepared.codex_material_json, "tid9", sites=prepared.terminal_id_binding_sites
        )
        expected = _json.loads(prepared.codex_material_json)
        for entry in expected["mcp_servers"]:
            for item in entry["env"]:
                if item["name"] == "CAO_TERMINAL_ID":
                    item["value"] = "tid9"
        assert bound == expected


class TestSentinelLiteralPreservation:
    @pytest.fixture
    def sentinel_profile(self, profile_store):
        from cli_agent_orchestrator.providers.base import SEALED_TERMINAL_ID_PLACEHOLDER as sentinel

        _write_profile(
            profile_store,
            "sent",
            provider="codex",
            body=f"Direct the {sentinel} run.",
            extra=(
                "mcpServers:",
                "  lit:",
                "    command: /bin/true",
                f"    args: [--id, {sentinel}]",
                "    env:",
                f"      NOTE: tok-{sentinel}-end",
                "  own:",
                "    command: /bin/true",
                "    env:",
                f"      CAO_TERMINAL_ID: {sentinel}",
                "  web:",
                f"    url: https://example.invalid/{sentinel}/hook",
                "    headers:",
                f"      X-Trace: {sentinel}",
            ),
        )
        return load_supervisor_launch_context("sent")

    def test_codex_material_keeps_user_literals(self, sentinel_profile):
        """Only the injected env entries bind; user literals stay verbatim."""
        import json as _json

        from cli_agent_orchestrator.providers.base import SEALED_TERMINAL_ID_PLACEHOLDER as sentinel

        prepared = ProviderManager().prepare_sealed_launch(
            "codex", spr.build_sealed_launch_material(sentinel_profile)
        )
        assert prepared.terminal_id_binding_sites == ("lit", "web")
        raw = prepared.codex_material_json.decode("utf-8")
        # Six placeholders prepared: prompt, args, NOTE, own, url, and
        # the injected lit env default (web is a URL entry: no env).
        assert raw.count(sentinel) == 6
        bound = bind_codex_material_json(
            prepared.codex_material_json, "tid9", sites=prepared.terminal_id_binding_sites
        )
        assert f"Direct the {sentinel} run." in bound["system_prompt"]
        lit = next(entry for entry in bound["mcp_servers"] if entry["name"] == "lit")
        env = {item["name"]: item["value"] for item in lit["env"]}
        assert env["CAO_TERMINAL_ID"] == "tid9"
        assert env["NOTE"] == f"tok-{sentinel}-end"
        assert lit["args"] == ["--id", sentinel]
        own = next(entry for entry in bound["mcp_servers"] if entry["name"] == "own")
        own_env = {item["name"]: item["value"] for item in own["env"]}
        # Explicitly user-set, so not a site: the literal survives.
        assert own_env["CAO_TERMINAL_ID"] == sentinel
        web = next(entry for entry in bound["mcp_servers"] if entry["name"] == "web")
        assert web["url"] == f"https://example.invalid/{sentinel}/hook"
        rebound = _json.dumps(bound, sort_keys=True)
        assert rebound.count(sentinel) == 5
        assert rebound.count("tid9") == 1

    def test_mcp_document_keeps_user_literals(self, sentinel_profile):
        """The MCP document binds sites only; literals stay verbatim."""
        import json as _json

        from cli_agent_orchestrator.providers.base import SEALED_TERMINAL_ID_PLACEHOLDER as sentinel
        from cli_agent_orchestrator.providers.base import (
            prepare_sealed_mcp_documents,
        )

        servers_json, document_json, sites = prepare_sealed_mcp_documents(sentinel_profile.profile)
        assert sites == ("lit", "web")
        bound = bind_sealed_mcp_document(document_json, "tid9", sites=sites)
        document = _json.loads(bound)
        servers = document["mcpServers"]
        assert servers["lit"]["env"]["CAO_TERMINAL_ID"] == "tid9"
        assert servers["lit"]["env"]["NOTE"] == f"tok-{sentinel}-end"
        assert servers["lit"]["args"] == ["--id", sentinel]
        assert servers["own"]["env"]["CAO_TERMINAL_ID"] == sentinel
        assert servers["web"]["env"]["CAO_TERMINAL_ID"] == "tid9"
        assert servers["web"]["url"] == f"https://example.invalid/{sentinel}/hook"
        assert servers["web"]["headers"] == {"X-Trace": sentinel}
        assert bound.count(sentinel.encode()) == 5
        assert bound.count(b"tid9") == 2


class TestHttpPreparationMapping:
    @pytest.mark.asyncio
    async def test_broken_codex_mcp_maps_to_422_end_to_end(self, profile_store):
        """The original repro at HTTP level: broken Codex MCP answers 422.

        The real session boundary runs (only the terminal effect is
        stubbed, and it is never reached): malformed provider material
        maps to the operation-scoped 422, never the late 500.
        """
        from fastapi import BackgroundTasks, HTTPException, Response

        from cli_agent_orchestrator.api import main

        _write_broken_mcp_profile(profile_store, "sup", "codex")
        context = load_supervisor_launch_context("sup")
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    response=Response(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                    profile_contract=_contract_for(context),
                )
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["provider"] == "codex"
        mock_create.assert_not_called()
