"""Provider reconstruction contract for persisted assigned routes."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.provider import ProviderType
from cli_agent_orchestrator.providers.manager import (
    ProviderManager,
    TerminalAssignedRouteIncompleteError,
)


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
        generation="gen-managed-1",
        native_session_id="session-1",
        assigned_model=assigned_model,
        assigned_effort=assigned_effort,
    )

    with pytest.raises(TerminalAssignedRouteIncompleteError, match=missing) as raised:
        ProviderManager().get_provider("managed1")
    msg = str(raised.value)
    assert "generation" in msg
    assert "native_session_id" in msg
    assert missing.split(",")[0].strip() in msg
    assert "managed terminal with native_session_id" not in msg


def test_managed_incomplete_with_no_native_still_refuses(terminal_database):
    """Generation alone classifies; missing pin with native=None still refuses."""
    database.create_terminal(
        "managed-no-native",
        "cao-s",
        "w-0",
        ProviderType.CODEX.value,
        generation="gen-no-native",
        native_session_id=None,
        assigned_model=None,
        assigned_effort=None,
    )
    with pytest.raises(TerminalAssignedRouteIncompleteError, match="assigned_model") as raised:
        ProviderManager().get_provider("managed-no-native")
    assert "generation" in str(raised.value)
    assert "native_session_id" in str(raised.value)


@pytest.mark.parametrize(
    "provider_type",
    [
        ProviderType.CLAUDE_CODE.value,
        ProviderType.CODEX.value,
        ProviderType.ANTIGRAVITY_CLI.value,
    ],
)
def test_ordinary_pre_task_row_survives_restart_with_native_and_null_assigned(
    tmp_path, monkeypatch, provider_type
):
    """Ordinary pre-task supervisors legitimately acquire native IDs while assigned NULL.

    Generation-NULL rows with native_session_id and NULL assigned_model/effort
    must reconstruct on their existing profile/ambient route after restart,
    not be refused as managed incomplete.
    """
    db_file = tmp_path / f"ordinary-{provider_type}.db"
    first_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    database.Base.metadata.create_all(bind=first_engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=first_engine))

    terminal_id = f"ordinary-{provider_type}"
    # Real production writers: generation NULL, assigned NULL, no native yet.
    database.create_terminal(
        terminal_id,
        "cao-s",
        "w-0",
        provider_type,
        generation=None,
        assigned_model=None,
        assigned_effort=None,
    )
    # Bind native ID through real production writer.
    assert database.set_terminal_native_session_id(terminal_id, f"native-{provider_type}") is True
    first_engine.dispose()

    second_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=second_engine))
    try:
        manager = ProviderManager()
        manager.create_provider = MagicMock(return_value=MagicMock(shell_baseline=None))
        provider = manager.get_provider(terminal_id)
        # Must reconstruct without refusal, using NULL expected route (profile/ambient).
        assert manager.create_provider.call_args.kwargs["expected_model"] is None
        assert manager.create_provider.call_args.kwargs["expected_effort"] is None
        assert (
            manager.create_provider.call_args.kwargs["native_session_id"]
            == f"native-{provider_type}"
        )
        assert provider is manager.create_provider.return_value
    finally:
        second_engine.dispose()


def test_generation_null_legacy_row_remains_reconstructable_after_restart(tmp_path, monkeypatch):
    """A generation-NULL legacy/repaired row with native ID stays reconstructable."""
    db_file = tmp_path / "legacy-repaired.db"
    first_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    database.Base.metadata.create_all(bind=first_engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=first_engine))

    terminal_id = "legacy-repaired-1"
    database.create_terminal(
        terminal_id,
        "cao-s",
        "w-0",
        ProviderType.CLAUDE_CODE.value,
        generation=None,
        assigned_model=None,
        assigned_effort=None,
    )
    assert database.set_terminal_native_session_id(terminal_id, "legacy-native-1") is True
    first_engine.dispose()

    second_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=second_engine))
    try:
        manager = ProviderManager()
        manager.create_provider = MagicMock(return_value=MagicMock(shell_baseline=None))
        provider = manager.get_provider(terminal_id)
        assert provider is manager.create_provider.return_value
        # Claude Code uses profile frontmatter as model channel; assigned pin is proof.
        assert manager.create_provider.call_args.kwargs["expected_model"] is None
    finally:
        second_engine.dispose()


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
