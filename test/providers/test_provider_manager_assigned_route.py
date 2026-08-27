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
    TerminalMetadataCollisionError,
)


@pytest.fixture
def terminal_database(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'provider-manager.db'}",
        connect_args={"check_same_thread": False},
    )
    database.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))
    yield engine
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


def test_v2_only_terminal_reconstructs_from_its_native_session(terminal_database):
    """A fresh manager must recover a terminal that exists only in the v2 store."""
    database.create_terminal_v2(
        "v2-only",
        "cao-v2",
        "w-v2",
        ProviderType.CODEX.value,
        generation="gen-v2-only",
        session_id="v2-session",
        native_session_id="native-v2-only",
    )
    manager = ProviderManager()
    manager.create_provider = MagicMock(return_value=MagicMock(shell_baseline=None))

    with patch(
        "cli_agent_orchestrator.providers.manager.get_terminal_metadata",
        wraps=database.get_terminal_metadata,
    ) as v1_lookup:
        provider = manager.get_provider("v2-only", include_managed_v2=True)

    assert provider is manager.create_provider.return_value
    v1_lookup.assert_called_once_with("v2-only", warn_if_missing=False)
    manager.create_provider.assert_called_once_with(
        ProviderType.CODEX.value,
        "v2-only",
        "cao-v2",
        "w-v2",
        None,
        native_session_id="native-v2-only",
    )


def test_opt_in_v1_terminal_reconstructs_when_v2_table_is_absent(terminal_database):
    """Opt-in probing keeps existing v1 rows usable on pre-migration DBs."""
    database.create_terminal(
        "v1-without-v2-table",
        "cao-v1",
        "w-v1",
        ProviderType.CODEX.value,
        generation="gen-v1",
        native_session_id="native-v1",
        assigned_model="gpt-5.6-sol",
        assigned_effort="high",
    )
    database.ManagedLaunchV2TerminalModel.__table__.drop(bind=terminal_database)
    manager = ProviderManager()
    manager.create_provider = MagicMock(return_value=MagicMock(shell_baseline=None))

    provider = manager.get_provider("v1-without-v2-table", include_managed_v2=True)

    assert provider is manager.create_provider.return_value
    manager.create_provider.assert_called_once_with(
        ProviderType.CODEX.value,
        "v1-without-v2-table",
        "cao-v1",
        "w-v1",
        None,
        native_session_id="native-v1",
        expected_model="gpt-5.6-sol",
        expected_effort="high",
    )


def test_same_id_in_both_vintages_refuses_without_creating_provider(terminal_database):
    """A cross-vintage ID collision must never silently choose one row."""
    database.create_terminal(
        "same-id",
        "cao-v1",
        "w-v1",
        ProviderType.CODEX.value,
        generation="gen-v1",
        assigned_model="gpt-5.6-sol",
        assigned_effort="high",
    )
    database.create_terminal_v2(
        "same-id",
        "cao-v2",
        "w-v2",
        ProviderType.CODEX.value,
        generation="gen-v2",
        native_session_id="native-v2",
    )
    manager = ProviderManager()
    manager.create_provider = MagicMock()
    manager._providers["same-id"] = MagicMock(name="stale-cached-provider")

    with pytest.raises(TerminalMetadataCollisionError, match="both v1 and v2"):
        manager.get_provider("same-id", include_managed_v2=True)

    manager.create_provider.assert_not_called()


def test_untagged_cached_provider_does_not_hide_v2_only_row(terminal_database):
    database.create_terminal_v2(
        "v2-untagged-cache",
        "cao-v2",
        "w-v2",
        ProviderType.CODEX.value,
        generation="gen-v2-cache",
        native_session_id="native-v2-cache",
    )
    manager = ProviderManager()
    stale = MagicMock(name="stale-provider")
    rebuilt = MagicMock(name="rebuilt-provider", shell_baseline=None)
    manager._providers["v2-untagged-cache"] = stale
    manager.create_provider = MagicMock(return_value=rebuilt)

    provider = manager.get_provider("v2-untagged-cache", include_managed_v2=True)

    assert provider is rebuilt
    assert provider is not stale
    manager.create_provider.assert_called_once_with(
        ProviderType.CODEX.value,
        "v2-untagged-cache",
        "cao-v2",
        "w-v2",
        None,
        native_session_id="native-v2-cache",
    )


def test_exact_v2_cache_identity_reuses_provider_without_recreating(terminal_database):
    database.create_terminal_v2(
        "v2-tagged-cache",
        "cao-v2",
        "w-v2",
        ProviderType.CODEX.value,
        generation="gen-v2-tagged",
        native_session_id="native-v2-tagged",
    )
    manager = ProviderManager()
    created = MagicMock(name="created-provider", shell_baseline=None)

    def create_and_cache(*_args, **_kwargs):
        manager._providers["v2-tagged-cache"] = created
        return created

    manager.create_provider = MagicMock(side_effect=create_and_cache)

    first = manager.get_provider("v2-tagged-cache", include_managed_v2=True)
    second = manager.get_provider("v2-tagged-cache", include_managed_v2=True)

    assert first is created
    assert second is created
    manager.create_provider.assert_called_once()


def test_stale_v2_cache_rebuilds_for_new_durable_incarnation(terminal_database):
    terminal_id = "v2-reincarnated-cache"
    database.create_terminal_v2(
        terminal_id,
        "cao-v2-a",
        "w-v2-a",
        ProviderType.CODEX.value,
        generation="gen-v2-a",
        native_session_id="native-v2-a",
    )
    manager = ProviderManager()
    provider_a = MagicMock(name="provider-a", shell_baseline=None)
    provider_b = MagicMock(name="provider-b", shell_baseline=None)
    created = iter((provider_a, provider_b))

    def create_and_cache(*_args, **_kwargs):
        provider = next(created)
        manager._providers[terminal_id] = provider
        return provider

    manager.create_provider = MagicMock(side_effect=create_and_cache)

    assert manager.get_provider(terminal_id, include_managed_v2=True) is provider_a
    assert database.delete_terminal_v2(terminal_id) is True
    database.create_terminal_v2(
        terminal_id,
        "cao-v2-b",
        "w-v2-b",
        ProviderType.CODEX.value,
        generation="gen-v2-b",
        native_session_id="native-v2-b",
    )

    assert manager.get_provider(terminal_id, include_managed_v2=True) is provider_b
    assert manager.get_provider(terminal_id, include_managed_v2=True) is provider_b
    assert manager.create_provider.call_count == 2
    assert manager.create_provider.call_args_list[1] == (
        (
            ProviderType.CODEX.value,
            terminal_id,
            "cao-v2-b",
            "w-v2-b",
            None,
        ),
        {"native_session_id": "native-v2-b"},
    )
    assert manager._provider_identities[terminal_id] == (
        "v2",
        "gen-v2-b",
        "native-v2-b",
        ProviderType.CODEX.value,
    )


def test_missing_v2_table_is_treated_as_absence(monkeypatch):
    """An un-migrated installation has no v2 table, which is still no row."""
    missing_table = OperationalError(
        "SELECT managed_launch_v2_terminals",
        {},
        Exception("no such table: managed_launch_v2_terminals"),
    )
    with (
        patch(
            "cli_agent_orchestrator.providers.manager.get_terminal_metadata",
            return_value=None,
        ),
        patch(
            "cli_agent_orchestrator.providers.manager.get_terminal_metadata_v2",
            side_effect=missing_table,
        ),
    ):
        with pytest.raises(ValueError, match="not found"):
            ProviderManager().get_provider("pre-migration", include_managed_v2=True)


def test_unreadable_v2_metadata_is_not_treated_as_absent():
    unreadable = OperationalError(
        "SELECT managed_launch_v2_terminals",
        {},
        Exception("database is locked"),
    )
    with (
        patch(
            "cli_agent_orchestrator.providers.manager.get_terminal_metadata",
            return_value=None,
        ),
        patch(
            "cli_agent_orchestrator.providers.manager.get_terminal_metadata_v2",
            side_effect=unreadable,
        ),
    ):
        with pytest.raises(OperationalError, match="database is locked"):
            ProviderManager().get_provider("v2-unreadable", include_managed_v2=True)


def test_unrelated_missing_table_is_not_treated_as_missing_v2_surface():
    unreadable = OperationalError(
        "SELECT managed_launch_v2_terminals",
        {},
        Exception("no such table: unrelated_table"),
    )
    with (
        patch(
            "cli_agent_orchestrator.providers.manager.get_terminal_metadata",
            return_value=None,
        ),
        patch(
            "cli_agent_orchestrator.providers.manager.get_terminal_metadata_v2",
            side_effect=unreadable,
        ),
    ):
        with pytest.raises(OperationalError, match="unrelated_table"):
            ProviderManager().get_provider("v2-unreadable", include_managed_v2=True)


def test_v2_only_default_lookup_remains_legacy_invisible_after_opt_in(terminal_database):
    database.create_terminal_v2(
        "v2-default-hidden",
        "cao-v2",
        "w-v2",
        ProviderType.CODEX.value,
        generation="gen-v2-hidden",
        native_session_id="native-v2-hidden",
    )
    manager = ProviderManager()
    provider = MagicMock(name="v2-provider", shell_baseline=None)
    manager.create_provider = MagicMock(
        side_effect=lambda *_args, **_kwargs: manager._providers.setdefault(
            "v2-default-hidden", provider
        )
    )

    assert manager.get_provider("v2-default-hidden", include_managed_v2=True) is provider

    with pytest.raises(ValueError, match="not found"):
        manager.get_provider("v2-default-hidden")
