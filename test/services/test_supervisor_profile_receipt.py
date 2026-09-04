"""Supervisor launch-resolved profile receipts (cond-0817).

Covers the runtime half of the launch-resolved profile contract: the single
launch-boundary read with installed-store precedence, contract validation
(match / divergence / malformed), the threaded no-reload wiring through
session and terminal creation into provider construction, durable receipt
persistence and projection (including legacy absence and post-launch drift),
and the HTTP conflict/retry mapping.
"""

import hashlib
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.agent_profile import AgentProfile
from cli_agent_orchestrator.services import session_service
from cli_agent_orchestrator.services import supervisor_profile_receipt as spr
from cli_agent_orchestrator.services.supervisor_profile_receipt import (
    PROFILE_LAUNCH_CONTRACT_SCHEMA,
    PROFILE_RECEIPT_SCHEMA,
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
) -> Path:
    """Write one flat ``<name>.md`` profile into a store directory."""
    lines = ["---", f"name: {name}", f"description: {name} profile"]
    if provider is not None:
        lines.append(f"provider: {provider}")
    if role is not None:
        lines.append(f"role: {role}")
    if model is not None:
        lines.append(f"model: {model}")
    if effort is not None:
        lines.extend(["codexConfig:", f"  model_reasoning_effort: {effort}"])
    lines.append("---")
    lines.append(body)
    path = store / f"{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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


class _HermeticLaunch(ExitStack):
    """Backend + FIFO + status patches for a hermetic create_terminal run."""

    def __init__(self, *, provider_manager, launched_provider, terminal_id, session):
        super().__init__()
        backend = _hermetic_backend()
        self.enter_context(patch("cli_agent_orchestrator.backends.registry._backend", backend))
        self.enter_context(patch(f"{_SERVICE}.provider_manager", provider_manager))
        self.enter_context(patch(f"{_SERVICE}.fifo_manager"))
        self.enter_context(patch(f"{_SERVICE}.status_monitor"))
        self.enter_context(patch(f"{_SERVICE}.clear_session_env"))
        self.enter_context(patch(f"{_SERVICE}._register_incarnation"))
        self.enter_context(patch(f"{_SERVICE}.generate_terminal_id", return_value=terminal_id))
        self.enter_context(patch(f"{_SERVICE}.generate_session_name", return_value=session))
        self.enter_context(patch(f"{_SERVICE}.generate_window_name", return_value="w-sup"))
        provider_manager.create_provider.return_value = launched_provider


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


class TestLaunchContext:
    def test_single_read_supplies_profile_source_and_digest(self, profile_store):
        path = _write_profile(profile_store, "sup")
        reads = []
        real_read_text = Path.read_text

        def _counting(self, *args, **kwargs):
            if self == path:
                reads.append(self)
            return real_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", _counting):
            context = load_supervisor_launch_context("sup")

        assert reads == [path]
        assert context.profile_name == "sup"
        assert context.source_path == str(path)
        assert context.provenance == "local"
        assert context.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert isinstance(context.profile, AgentProfile)
        assert context.profile.system_prompt == "Do supervision."
        assert context.provider == "mock_cli"
        assert context.model == "test-model-1"
        assert context.effort is None

    def test_installed_store_takes_precedence_over_builtin(self, profile_store):
        _write_profile(profile_store, "developer", model="shadow-model")
        context = load_supervisor_launch_context("developer")
        assert context.provenance == "local"
        assert context.model == "shadow-model"

    def test_builtin_resolves_with_builtin_provenance(self, profile_store):
        context = load_supervisor_launch_context("developer")
        assert context.provenance == "built-in"
        assert context.source_path == "built-in:developer.md"

    def test_codex_route_applies_config_seam(self, profile_store):
        _write_profile(
            profile_store,
            "codex-sup",
            provider="codex",
            model="profile-model",
            effort="xhigh",
        )
        context = load_supervisor_launch_context("codex-sup")
        assert context.provider == "codex"
        assert context.model == "profile-model"
        assert context.effort == "xhigh"

    def test_codex_config_model_wins_over_bare_model(self, profile_store, tmp_path):
        path = tmp_path / "agent-store" / "codex-cfg.md"
        path.write_text(
            "---\nname: codex-cfg\ndescription: x\nprovider: codex\n"
            "model: bare-model\ncodexConfig:\n  model: cfg-model\n"
            "  model_reasoning_effort: xhigh\n---\nbody\n",
            encoding="utf-8",
        )
        context = load_supervisor_launch_context("codex-cfg")
        assert context.model == "cfg-model"
        assert context.effort == "xhigh"

    def test_explicit_provider_wins_and_invalid_profile_provider_falls_back(self, profile_store):
        _write_profile(profile_store, "odd", provider="not-a-provider", model="m")
        assert (
            load_supervisor_launch_context("odd", explicit_provider="kimi_cli").provider
            == "kimi_cli"
        )
        assert load_supervisor_launch_context("odd").provider == "kiro_cli"

    def test_missing_profile_raises_before_any_effect(self, profile_store):
        with pytest.raises(FileNotFoundError):
            load_supervisor_launch_context("ghost")

    def test_receipt_is_runtime_authored_not_request_echo(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        receipt = build_profile_receipt(context)
        assert receipt == {
            "schema": PROFILE_RECEIPT_SCHEMA,
            "profile": "sup",
            "role": "supervisor",
            "provider": "mock_cli",
            "model": "test-model-1",
            "effort": None,
            "provenance": "local",
            "source_path": context.source_path,
            "sha256": context.sha256,
        }


class TestContractValidation:
    @pytest.fixture
    def context(self, profile_store):
        _write_profile(profile_store, "sup")
        return load_supervisor_launch_context("sup")

    def test_matching_contract_validates(self, context):
        validate_profile_contract(_contract_for(context), context)

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("profile", "other"),
            ("role", "worker"),
            ("provider", "kimi_cli"),
            ("model", "other-model"),
            ("effort", "xhigh"),
            ("provenance", "built-in"),
            ("source_path", "/elsewhere/sup.md"),
            ("sha256", "0" * 64),
        ],
    )
    def test_every_compared_field_divergence_conflicts(self, context, field, bad):
        """Each compared field is pinned: dropping any one comparison breaks this."""
        contract = _contract_for(context)
        if field == "role":
            with pytest.raises(ValueError):
                validate_profile_contract({**contract, field: bad}, context)
            return
        if field == "effort":
            # The fixture profile declares no effort; a null->value flip
            # must still conflict rather than pass as "undeclared".
            assert context.effort is None
        with pytest.raises(ProfileLaunchConflict) as exc_info:
            validate_profile_contract({**contract, field: bad}, context)
        assert field in [entry["field"] for entry in exc_info.value.divergent_fields]
        assert exc_info.value.retry

    @pytest.mark.parametrize(
        "contract",
        [
            "not-a-mapping",
            {"schema": "wrong-schema"},
            {"schema": PROFILE_LAUNCH_CONTRACT_SCHEMA, "role": "worker"},
            {"schema": PROFILE_LAUNCH_CONTRACT_SCHEMA, "role": "supervisor"},
            {
                "schema": PROFILE_LAUNCH_CONTRACT_SCHEMA,
                "role": "supervisor",
                "profile": "sup",
                "provider": "mock_cli",
                "model": 42,
                "effort": None,
                "provenance": "local",
                "source_path": "x",
                "sha256": "y",
            },
        ],
    )
    def test_malformed_contracts_are_refused(self, context, contract):
        with pytest.raises(ValueError):
            validate_profile_contract(contract, context)


class TestSessionWiring:
    @pytest.mark.asyncio
    async def test_context_threaded_and_no_second_read(self, profile_store):
        """Real boundary load; everything below consumes it without reloading."""
        _write_profile(profile_store, "sup")
        with (
            patch.object(
                agent_profiles, "load_agent_profile", side_effect=AssertionError("reload")
            ),
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-sup")
            await session_service.create_session(provider=None, agent_profile="sup")

        kwargs = mock_create.call_args.kwargs
        assert kwargs["provider"] == "mock_cli"
        assert kwargs["expected_model"] == "test-model-1"
        assert kwargs["expected_effort"] is None
        context = kwargs["profile_launch_context"]
        assert isinstance(context, spr.ProfileLaunchContext)
        assert context.profile.system_prompt == "Do supervision."

    @pytest.mark.asyncio
    async def test_conflict_has_zero_effects(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        contract["sha256"] = "f" * 64  # bytes changed after the preflight
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(ProfileLaunchConflict):
                await session_service.create_session(
                    provider=None, agent_profile="sup", profile_contract=contract
                )
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_contract_refused_before_effects(self, profile_store):
        _write_profile(profile_store, "sup")
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(ValueError):
                await session_service.create_session(
                    provider=None,
                    agent_profile="sup",
                    profile_contract={"schema": "nope"},
                )
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_with_fresh_contract_launches(self, profile_store):
        _write_profile(profile_store, "sup", model="model-one")
        stale = _contract_for(load_supervisor_launch_context("sup"))
        # The profile bytes change between preflight and retry...
        _write_profile(profile_store, "sup", model="model-two")
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-sup")
            with pytest.raises(ProfileLaunchConflict):
                await session_service.create_session(
                    provider=None, agent_profile="sup", profile_contract=stale
                )
            fresh = _contract_for(load_supervisor_launch_context("sup"))
            await session_service.create_session(
                provider=None, agent_profile="sup", profile_contract=fresh
            )
        assert mock_create.call_count == 1
        assert mock_create.call_args.kwargs["expected_model"] == "model-two"


class TestTerminalReceiptWiring:
    @pytest.mark.asyncio
    async def test_receipt_persisted_projected_and_exact(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """Real create_terminal run: no reload, durable receipt, exact route."""
        from cli_agent_orchestrator.services import terminal_projection
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        service_provider_manager = MagicMock()
        with (
            _HermeticLaunch(
                provider_manager=service_provider_manager,
                launched_provider=launched_provider,
                terminal_id="abcd1234",
                session="cao-sup",
            ),
            # The launch-boundary read already happened: any by-name reload
            # below (Step 3, bootstrap, adapter) fails the test outright.
            patch(
                f"{_SERVICE}.load_agent_profile",
                side_effect=AssertionError("profile reloaded by name"),
            ),
        ):
            terminal = await create_terminal(
                provider="mock_cli",
                agent_profile="sup",
                new_session=True,
                profile_launch_context=context,
                expected_model=context.model,
                expected_effort=context.effort,
            )

        expected_receipt = build_profile_receipt(context)
        # POST /sessions shape: the same receipt rides the response, and it
        # survives the response_model round-trip the route applies.
        assert terminal.profile_receipt == expected_receipt
        from cli_agent_orchestrator.models.terminal import Terminal

        assert (
            Terminal.model_validate(terminal.model_dump()).profile_receipt
            == expected_receipt
        )
        # Provider construction consumed the exact context object.
        create_kwargs = service_provider_manager.create_provider.call_args.kwargs
        assert create_kwargs["launch_profile"] is context.profile
        assert create_kwargs["expected_model"] == "test-model-1"
        assert create_kwargs["expected_effort"] is None
        # Durable row carries the receipt...
        stored = database.get_terminal_metadata("abcd1234")
        assert stored is not None
        assert stored["profile_receipt"] == expected_receipt
        # ...and every read surface agrees: session listing, single
        # terminal projection, and the legacy get_terminal fallback.
        from cli_agent_orchestrator.services.terminal_service import get_terminal

        with (
            patch("cli_agent_orchestrator.services.session_service.get_backend") as mock_backend,
            patch(
                "cli_agent_orchestrator.services.terminal_projection.get_backend",
                return_value=_hermetic_backend(),
            ),
        ):
            mock_backend.return_value.session_exists.return_value = True
            mock_backend.return_value.list_sessions.return_value = [{"id": "cao-sup"}]
            session_view = session_service.get_session("cao-sup")
            assert session_view["terminals"][0]["profile_receipt"] == expected_receipt
            assert (
                terminal_projection.project_terminal("abcd1234")["profile_receipt"]
                == expected_receipt
            )
        with patch(f"{_SERVICE}.status_monitor") as mock_status:
            mock_status.get_status.return_value = MagicMock(value="idle")
            assert get_terminal("abcd1234")["profile_receipt"] == expected_receipt

    @pytest.mark.asyncio
    async def test_post_launch_drift_does_not_change_reads(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        from cli_agent_orchestrator.services import terminal_projection
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup", model="launch-model")
        context = load_supervisor_launch_context("sup")
        service_provider_manager = MagicMock()
        with _HermeticLaunch(
            provider_manager=service_provider_manager,
            launched_provider=launched_provider,
            terminal_id="abcd1235",
            session="cao-drift",
        ):
            terminal = await create_terminal(
                provider="mock_cli",
                agent_profile="sup",
                new_session=True,
                profile_launch_context=context,
            )

        launched_receipt = dict(terminal.profile_receipt)
        assert launched_receipt["model"] == "launch-model"
        # The profile changes after the launch...
        _write_profile(profile_store, "sup", model="drifted-model")
        assert load_supervisor_launch_context("sup").model == "drifted-model"
        # ...but every durable surface still reports the launch truth.
        assert database.get_terminal_metadata("abcd1235")["profile_receipt"] == launched_receipt
        with patch(
            "cli_agent_orchestrator.services.terminal_projection.get_backend",
            return_value=_hermetic_backend(),
        ):
            assert (
                terminal_projection.project_terminal("abcd1235")["profile_receipt"]
                == launched_receipt
            )

    @pytest.mark.asyncio
    async def test_legacy_row_stays_missing(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """A launch without a context persists no receipt: reads stay absent."""
        from cli_agent_orchestrator.services import terminal_projection
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup")
        service_provider_manager = MagicMock()
        with _HermeticLaunch(
            provider_manager=service_provider_manager,
            launched_provider=launched_provider,
            terminal_id="abcd1236",
            session="cao-legacy",
        ):
            terminal = await create_terminal(
                provider="mock_cli",
                agent_profile="sup",
                new_session=True,
            )

        assert terminal.profile_receipt is None
        assert database.get_terminal_metadata("abcd1236")["profile_receipt"] is None
        with patch(
            "cli_agent_orchestrator.services.terminal_projection.get_backend",
            return_value=_hermetic_backend(),
        ):
            assert terminal_projection.project_terminal("abcd1236")["profile_receipt"] is None

    @pytest.mark.asyncio
    async def test_persistence_failure_leaves_no_successful_launch(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """A row write that cannot persist the receipt fails the launch."""
        from cli_agent_orchestrator.services import terminal_service
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        service_provider_manager = MagicMock()
        with (
            _HermeticLaunch(
                provider_manager=service_provider_manager,
                launched_provider=launched_provider,
                terminal_id="abcd1237",
                session="cao-noreceipt",
            ),
            # terminal_service holds its own reference to the db writer.
            patch.object(
                terminal_service,
                "db_create_terminal",
                side_effect=RuntimeError("store down"),
            ),
        ):
            with pytest.raises(RuntimeError, match="store down"):
                await create_terminal(
                    provider="mock_cli",
                    agent_profile="sup",
                    new_session=True,
                    profile_launch_context=context,
                )
        assert database.get_terminal_metadata("abcd1237", warn_if_missing=False) is None

    def test_migration_adds_nullable_receipt_to_legacy_table(self, tmp_path):
        """A pre-receipt terminals table gains the column; old rows read NULL."""
        import sqlite3

        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE terminals ("
            "id TEXT PRIMARY KEY, tmux_session TEXT, tmux_window TEXT, "
            "provider TEXT, agent_profile TEXT)"
        )
        conn.execute(
            "INSERT INTO terminals (id, tmux_session, tmux_window, provider, agent_profile)"
            " VALUES ('deadbeef', 'cao-old', 'w', 'mock_cli', 'sup')"
        )
        conn.commit()
        conn.close()

        from cli_agent_orchestrator import constants

        with patch.object(constants, "DATABASE_FILE", db_path):
            database._migrate_terminals_schema()

        conn = sqlite3.connect(str(db_path))
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(terminals)")}
            assert "profile_receipt" in columns
            row = conn.execute(
                "SELECT profile_receipt FROM terminals WHERE id = 'deadbeef'"
            ).fetchone()
            assert row[0] is None
        finally:
            conn.close()


class TestHttpMapping:
    @pytest.mark.asyncio
    async def test_conflict_maps_to_409_with_retry(self, profile_store):
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        contract["provider"] = "kimi_cli"
        with (
            patch.object(
                main.session_service,
                "create_session",
                side_effect=ProfileLaunchConflict(
                    "diverged", divergent_fields=["provider"], retry="re-preflight"
                ),
            ),
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                    profile_contract=contract,
                )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["retry"] == "re-preflight"
        assert exc_info.value.detail["divergent_fields"] == ["provider"]

    @pytest.mark.asyncio
    async def test_malformed_contract_maps_to_400(self, profile_store):
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup")
        with (
            patch.object(
                main.session_service,
                "create_session",
                side_effect=ValueError("profile_contract schema must be 'cao-...'"),
            ),
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                    profile_contract={"schema": "nope"},
                )
        assert exc_info.value.status_code == 400
