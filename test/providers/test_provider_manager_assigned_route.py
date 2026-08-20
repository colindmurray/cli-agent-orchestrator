"""Provider reconstruction contract for persisted assigned routes."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.provider import ProviderType
from cli_agent_orchestrator.providers.manager import ProviderManager


@pytest.fixture
def terminal_database(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'provider-manager.db'}",
        connect_args={"check_same_thread": False},
    )
    database.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))
    yield
    engine.dispose()


@pytest.mark.parametrize(
    ("assigned_model", "assigned_effort", "missing"),
    [
        (None, "high", "assigned_model"),
        ("gpt-5.6-sol", None, "assigned_effort"),
        (None, None, r"assigned_model, assigned_effort"),
    ],
)
def test_managed_terminal_refuses_incomplete_assigned_route(
    terminal_database, assigned_model, assigned_effort, missing
):
    database.create_terminal(
        "managed1",
        "cao-s",
        "w-0",
        ProviderType.CODEX.value,
        native_session_id="session-1",
        assigned_model=assigned_model,
        assigned_effort=assigned_effort,
    )

    with pytest.raises(ValueError, match=missing):
        ProviderManager().get_provider("managed1")


def test_legacy_terminal_without_native_session_keeps_nullable_route(terminal_database):
    database.create_terminal(
        "legacy01",
        "cao-s",
        "w-0",
        ProviderType.CODEX.value,
    )
    manager = ProviderManager()
    manager.create_provider = MagicMock(return_value=MagicMock(shell_baseline=None))

    provider = manager.get_provider("legacy01")

    assert manager.create_provider.call_args.kwargs["expected_model"] is None
    assert manager.create_provider.call_args.kwargs["expected_effort"] is None
    assert provider is manager.create_provider.return_value


def test_absent_terminal_is_not_found(terminal_database):
    with pytest.raises(ValueError, match="not found"):
        ProviderManager().get_provider("absent01")


def test_unreadable_metadata_is_not_treated_as_absent_or_legacy():
    unreadable = OperationalError("SELECT terminals", {}, Exception("database is locked"))

    with patch(
        "cli_agent_orchestrator.providers.manager.get_terminal_metadata",
        side_effect=unreadable,
    ):
        with pytest.raises(OperationalError, match="database is locked"):
            ProviderManager().get_provider("unread01")
