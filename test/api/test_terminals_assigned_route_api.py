"""API-level contract for assigned-route incomplete vs absent terminal."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.provider import ProviderType
from cli_agent_orchestrator.models.terminal import Terminal


def test_managed_incomplete_row_returns_409_and_absent_remains_404(client, tmp_path, monkeypatch):
    db_file = tmp_path / "api-409.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    database.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))
    try:
        managed_id = "ab12cd34"
        database.create_terminal(
            managed_id,
            "cao-s",
            "w-0",
            ProviderType.CODEX.value,
            generation="gen-api-1",
            native_session_id="native-api-1",
            assigned_model=None,
            assigned_effort="high",
        )

        resp = client.post(f"/terminals/{managed_id}/input", params={"message": "hello"})
        assert resp.status_code == 409, resp.text
        detail = resp.json()["detail"]
        assert "generation" in detail
        assert "native_session_id" in detail
        assert "assigned_model" in detail
        assert "managed terminal with native_session_id" not in detail

        absent_id = "deadbeef"
        resp2 = client.post(f"/terminals/{absent_id}/input", params={"message": "hello"})
        assert resp2.status_code == 404
        assert "not found" in resp2.json()["detail"].lower()
    finally:
        engine.dispose()


def test_create_terminal_validates_and_forwards_quota_provider(client, monkeypatch):
    seen = {}

    async def fake_create(**kwargs):
        seen.update(kwargs)
        return Terminal(
            id="abcd1234",
            name="w",
            session_name="cao-test",
            provider="kiro_cli",
            agent_profile="developer",
            status="idle",
            assigned_quota_provider=kwargs["assigned_quota_provider"],
        )

    monkeypatch.setattr(
        "cli_agent_orchestrator.api.main.terminal_service.create_terminal", fake_create
    )
    monkeypatch.setattr("cli_agent_orchestrator.api.main.resolve_provider", lambda _, fb: fb)
    monkeypatch.setattr(
        "cli_agent_orchestrator.api.main.get_plugin_registry", lambda _request: object()
    )
    url = "/sessions/cao-test/terminals"
    params = {"provider": "kiro_cli", "agent_profile": "developer"}
    response = client.post(url, params=params, json={"quota_provider": "zai"})
    assert response.status_code == 201
    assert seen["assigned_quota_provider"] == "zai"
    assert response.json()["assigned_quota_provider"] == "zai"

    seen.clear()
    response = client.post(url, params=params, json={"quota_provider": ""})
    assert response.status_code == 422
    assert seen == {}
