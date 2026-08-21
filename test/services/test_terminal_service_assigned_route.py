"""End-to-end assigned-route wiring through terminal creation and reconstruction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.agent_profile import AgentProfile
from cli_agent_orchestrator.providers.manager import ProviderManager

_SERVICE = "cli_agent_orchestrator.services.terminal_service"


@pytest.fixture(autouse=True)
def _clear_session_env():
    with patch(f"{_SERVICE}.clear_session_env"):
        yield


@pytest.mark.asyncio
@patch(f"{_SERVICE}.status_monitor")
@patch(f"{_SERVICE}.fifo_manager")
@patch(f"{_SERVICE}.FIFO_DIR")
@patch(f"{_SERVICE}.provider_manager")
@patch(f"{_SERVICE}.generate_terminal_id", return_value="route001")
@patch(f"{_SERVICE}.generate_session_name", return_value="cao-route")
@patch(f"{_SERVICE}.generate_window_name", return_value="w-route")
@patch(f"{_SERVICE}.load_agent_profile")
@patch("cli_agent_orchestrator.backends.registry._backend")
async def test_assigned_route_survives_service_write_restart_and_reconstruction(
    backend,
    load_profile,
    _window_name,
    _session_name,
    _terminal_id,
    service_provider_manager,
    fifo_dir,
    _fifo_manager,
    _status_monitor,
    tmp_path,
    monkeypatch,
):
    db_file = tmp_path / "service-route.db"
    first_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    database.Base.metadata.create_all(bind=first_engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=first_engine))

    backend.session_exists.return_value = False
    backend.window_identity.return_value = {
        "pane_id": "%1",
        "window_id": "@1",
        "server_socket_path": "/tmp/tmux",
        "session_id": "$1",
        "pane_pid": "12345",
    }
    backend.supports_event_inbox.return_value = False
    backend.supports_pane_identity.return_value = True
    load_profile.return_value = AgentProfile(name="developer", description="dev")
    fifo_dir.__truediv__ = MagicMock(return_value=tmp_path / "route.fifo")
    launched_provider = AsyncMock()
    launched_provider.initialize.return_value = True
    service_provider_manager.create_provider.return_value = launched_provider

    from cli_agent_orchestrator.services.terminal_service import create_terminal

    try:
        with patch(f"{_SERVICE}._register_incarnation"):
            terminal = await create_terminal(
                provider="kiro_cli",
                agent_profile="developer",
                new_session=True,
                expected_model="gpt-5.6-sol",
                expected_effort="high",
            )
        assert database.set_terminal_native_session_id(terminal.id, "session-1") is True
    finally:
        first_engine.dispose()

    second_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=second_engine))
    try:
        manager = ProviderManager()
        manager.create_provider = MagicMock(return_value=MagicMock(shell_baseline=None))

        provider = manager.get_provider(terminal.id)

        assert manager.create_provider.call_args.kwargs["expected_model"] == "gpt-5.6-sol"
        assert manager.create_provider.call_args.kwargs["expected_effort"] == "high"
        assert manager.create_provider.call_args.kwargs["native_session_id"] == "session-1"
        assert provider is manager.create_provider.return_value
    finally:
        second_engine.dispose()
