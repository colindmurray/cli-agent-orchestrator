import uuid

from cli_agent_orchestrator.services import installation_identity


def test_installation_id_is_minted_once_and_persisted(tmp_path, monkeypatch):
    path = tmp_path / "installation-id"
    monkeypatch.setattr(installation_identity, "INSTALLATION_ID_FILE", path)

    first = installation_identity.get_installation_id()
    second = installation_identity.get_installation_id()

    assert second == first
    assert str(uuid.UUID(first)) == first
    assert path.read_text(encoding="utf-8").strip() == first
    assert path.stat().st_mode & 0o777 == 0o600


def test_malformed_installation_id_is_not_silently_replaced(tmp_path, monkeypatch):
    path = tmp_path / "installation-id"
    path.write_text("not-an-installation-id\n", encoding="utf-8")
    monkeypatch.setattr(installation_identity, "INSTALLATION_ID_FILE", path)

    try:
        installation_identity.get_installation_id()
    except installation_identity.InstallationIdentityError as exc:
        assert "malformed" in str(exc)
    else:
        raise AssertionError("malformed durable identity was silently accepted")
