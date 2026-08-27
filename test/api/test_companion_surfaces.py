"""End-to-end API/service tests for the structured companion surfaces
(final conformance §20.2f P1-7/P1-10): exact terminal/process generation,
provider version/session/turn, per-turn route identity, redaction, response
loss, and stale/wrong-generation rejection — through the real HTTP surface,
never unit-injected sentinel boundaries.
"""

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.services import companion_receipts, terminal_service

GEN = "11111111-1111-4111-8111-111111111111"
OTHER_GEN = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def companion(tmp_path, monkeypatch):
    store = tmp_path / "companion"
    store.mkdir()
    monkeypatch.setattr(companion_receipts, "COMPANION_DIR", store)
    monkeypatch.setattr(
        terminal_service,
        "get_terminal_metadata",
        lambda tid, *, warn_if_missing=True: {"generation": GEN} if tid == "deadbeef" else None,
    )
    return store


def test_route_surface_exact_generation(client, companion):
    assert client.get("/terminals/deadbeef/route").status_code == 204
    companion_receipts.record_route_receipt(
        "deadbeef",
        GEN,
        provider="codex",
        model="gpt-5.6-sol",
        effort="max",
        receipt_id="thread-1",
        turn_id="turn-1",
        provider_version="0.146.0",
    )
    response = client.get("/terminals/deadbeef/route")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "codex"
    assert body["model"] == "gpt-5.6-sol"
    assert body["effort"] == "max"
    assert body["receipt_id"] == "thread-1"
    assert body["turn_id"] == "turn-1"
    assert body["provider_version"] == "0.146.0"
    assert body["generation"] == GEN
    # unknown terminal: no observation, never an error
    assert client.get("/terminals/cafebabe/route").status_code == 204


def test_route_surface_rejects_stale_generation(client, companion, monkeypatch):
    companion_receipts.record_route_receipt(
        "deadbeef",
        GEN,
        provider="codex",
        model="m",
        effort="e",
        receipt_id="r",
        turn_id="t",
    )
    # the terminal id now names a replacement incarnation
    monkeypatch.setattr(
        terminal_service,
        "get_terminal_metadata",
        lambda tid, *, warn_if_missing=True: {"generation": OTHER_GEN},
    )
    assert client.get("/terminals/deadbeef/route").status_code == 204


def test_user_prompt_surface_lifecycle(client, companion):
    assert client.get("/terminals/deadbeef/user-prompt").status_code == 204
    companion_receipts.record_prompt(
        "deadbeef",
        GEN,
        prompt_id="p1",
        text="Trust this project?",
        choices=["Yes", "No"],
    )
    response = client.get("/terminals/deadbeef/user-prompt")
    assert response.status_code == 200
    assert response.json()["prompt_id"] == "p1"
    assert response.json()["choices"] == ["Yes", "No"]
    companion_receipts.clear_prompt("deadbeef", GEN, prompt_id="p1")
    assert client.get("/terminals/deadbeef/user-prompt").status_code == 204


def test_refusal_surface(client, companion):
    assert client.get("/terminals/deadbeef/refusal").status_code == 204
    companion_receipts.record_refusal(
        "deadbeef",
        GEN,
        refusal_id="r1",
        identity="This content cannot be shown",
        turn_id="turn-9",
    )
    response = client.get("/terminals/deadbeef/refusal")
    assert response.status_code == 200
    body = response.json()
    assert body["refusal_id"] == "r1"
    assert body["identity"] == "This content cannot be shown"
    assert body["turn_id"] == "turn-9"
    assert body["generation"] == GEN


def test_message_turn_receipt_surface_and_redaction(client, companion):
    url = "/terminals/deadbeef/inbox/messages/msg-1/turn-receipt"
    assert client.get(url).status_code == 204
    companion_receipts.record_message_ack(
        "deadbeef",
        GEN,
        message_id="msg-1",
        ack={
            "kind": "submitted",
            "message_id": "msg-1",
            "message_sha256": "a" * 64,
            "sender_id": "beeffeed",
            "receiver_id": "deadbeef",
            "receiver_generation": GEN,
            "provider": "codex",
            "provider_session_id": "thread-1",
            "provider_turn_id": "turn-1",
            "submitted_at": "2026-01-01T00:00:00Z",
        },
    )
    response = client.get(url)
    assert response.status_code == 200
    body = response.json()
    assert body["message_id"] == "msg-1"
    assert body["provider_turn_id"] == "turn-1"
    assert body["provider_session_id"] == "thread-1"
    assert body["receiver_generation"] == GEN
    # redaction: the message body is never carried by the ack surface
    assert "message" not in body
    assert client.get("/terminals/deadbeef/inbox/messages/msg-2/turn-receipt").status_code == 204


def test_turn_receipt_serves_a_terminal_wake_receipt_when_no_managed_ack(
    client, companion, tmp_path, monkeypatch
):
    from cli_agent_orchestrator.services import wake_receipts

    monkeypatch.setattr(wake_receipts, "WAKE_RECEIPT_DIR", tmp_path / "wake")
    url = "/terminals/deadbeef/inbox/messages/msg-w/turn-receipt"
    # No record yet: 204 (no false close).
    assert client.get(url).status_code == 204
    # watching: still 204 — an open obligation stays observable, not terminal.
    wake_receipts.ensure_watching(
        "deadbeef",
        "msg-w",
        native_session_id=None,
        delivered_at="2026-07-26T12:00:00+00:00",
        deadline_at="2026-07-26T12:00:45+00:00",
    )
    assert client.get(url).status_code == 204
    # terminal wake_confirmed: served, with status-transition provenance and
    # never a provider-native claim.
    wake_receipts.record_wake_confirmed("deadbeef", "msg-w", observed={"to_status": "processing"})
    body = client.get(url).json()
    assert body["message_id"] == "msg-w"
    assert body["state"] == "wake_confirmed"
    assert body["source"] == "status-transition"


def test_managed_ack_is_preferred_over_the_wake_sidecar(client, companion, tmp_path, monkeypatch):
    from cli_agent_orchestrator.services import wake_receipts

    monkeypatch.setattr(wake_receipts, "WAKE_RECEIPT_DIR", tmp_path / "wake")
    # Both a wake sidecar and a managed provider-native ack exist for msg-1.
    wake_receipts.ensure_watching(
        "deadbeef",
        "msg-1",
        native_session_id=None,
        delivered_at="2026-07-26T12:00:00+00:00",
        deadline_at="2026-07-26T12:00:45+00:00",
    )
    wake_receipts.record_wake_confirmed("deadbeef", "msg-1", observed={"to_status": "processing"})
    companion_receipts.record_message_ack(
        "deadbeef",
        GEN,
        message_id="msg-1",
        ack={"kind": "submitted", "message_id": "msg-1", "provider_turn_id": "turn-1"},
    )
    body = client.get("/terminals/deadbeef/inbox/messages/msg-1/turn-receipt").json()
    # The managed provider-native acknowledgement outranks the wake sidecar.
    assert body.get("kind") == "submitted"
    assert body.get("provider_turn_id") == "turn-1"
    assert body.get("state") is None  # the wake record's field is absent


def test_corrupt_record_is_response_loss_fail_closed(client, companion):
    companion_receipts.record_route_receipt(
        "deadbeef",
        GEN,
        provider="p",
        model="m",
        effort="e",
        receipt_id="r",
        turn_id="t",
    )
    path = companion_receipts._record_path("deadbeef", GEN)
    path.write_text("{ corrupt")
    assert client.get("/terminals/deadbeef/route").status_code == 204


def test_v2_only_terminal_serves_all_companion_surfaces_by_exact_generation(
    client, companion, isolated_memory_db
):
    """A v2-only managed terminal is a live companion-surface subject.

    v2 launch metadata intentionally has no row in the legacy ``terminals``
    table.  The read-only HTTP surfaces must still resolve its exact live
    generation and reject both stale records and an absent terminal.
    """
    terminal_id = "c0ffee42"
    generation = "33333333-3333-4333-8333-333333333333"
    database.create_terminal_v2(
        terminal_id,
        tmux_session="cao-v2-only",
        tmux_window="agent",
        provider="codex",
        agent_profile="developer",
        generation=generation,
        session_id="session-v2",
        native_session_id="native-v2",
    )
    # Prove the fixture models the isolated vintage rather than silently
    # finding a legacy row with the same terminal id.
    assert database.get_terminal_metadata(terminal_id, warn_if_missing=False) is None
    assert database.get_terminal_metadata_v2(terminal_id)["generation"] == generation

    companion_receipts.record_prompt(
        terminal_id,
        generation,
        prompt_id="prompt-v2",
        text="Trust this v2-only project?",
        choices=["Yes", "No"],
    )
    companion_receipts.record_refusal(
        terminal_id,
        generation,
        refusal_id="refusal-v2",
        identity="v2-only refusal",
        turn_id="turn-v2",
    )
    companion_receipts.record_route_receipt(
        terminal_id,
        generation,
        provider="codex",
        model="gpt-5.6-sol",
        effort="high",
        receipt_id="route-v2",
        turn_id="turn-v2",
        provider_version="0.146.0",
    )
    companion_receipts.record_message_ack(
        terminal_id,
        generation,
        message_id="message-v2",
        ack={
            "kind": "submitted",
            "message_id": "message-v2",
            "message_sha256": "b" * 64,
            "sender_id": "sender-v2",
            "receiver_id": terminal_id,
            "receiver_generation": generation,
            "provider": "codex",
            "provider_session_id": "session-v2",
            "provider_turn_id": "turn-v2",
            "submitted_at": "2026-08-27T00:00:00Z",
        },
    )

    prompt = client.get(f"/terminals/{terminal_id}/user-prompt")
    assert prompt.status_code == 200
    prompt_body = prompt.json()
    assert prompt_body == {
        "prompt_id": "prompt-v2",
        "text": "Trust this v2-only project?",
        "choices": ["Yes", "No"],
        "generation": generation,
        "recorded_at": prompt_body["recorded_at"],
    }
    assert isinstance(prompt_body["recorded_at"], str)

    refusal = client.get(f"/terminals/{terminal_id}/refusal")
    assert refusal.status_code == 200
    refusal_body = refusal.json()
    assert refusal_body == {
        "refusal_id": "refusal-v2",
        "identity": "v2-only refusal",
        "turn_id": "turn-v2",
        "generation": generation,
        "recorded_at": refusal_body["recorded_at"],
    }
    assert isinstance(refusal_body["recorded_at"], str)

    route = client.get(f"/terminals/{terminal_id}/route")
    assert route.status_code == 200
    route_body = route.json()
    assert route_body == {
        "provider": "codex",
        "model": "gpt-5.6-sol",
        "effort": "high",
        "generation": generation,
        "receipt_id": "route-v2",
        "turn_id": "turn-v2",
        "provider_version": "0.146.0",
        "recorded_at": route_body["recorded_at"],
    }
    assert isinstance(route_body["recorded_at"], str)

    receipt = client.get(f"/terminals/{terminal_id}/inbox/messages/message-v2/turn-receipt")
    assert receipt.status_code == 200
    receipt_body = receipt.json()
    assert receipt_body == {
        "kind": "submitted",
        "message_id": "message-v2",
        "message_sha256": "b" * 64,
        "sender_id": "sender-v2",
        "receiver_id": terminal_id,
        "receiver_generation": generation,
        "provider": "codex",
        "provider_session_id": "session-v2",
        "provider_turn_id": "turn-v2",
        "submitted_at": "2026-08-27T00:00:00Z",
        "recorded_at": receipt_body["recorded_at"],
    }
    assert isinstance(receipt_body["recorded_at"], str)

    # A different incarnation cannot read this generation's records.
    stale_terminal_id = "c0ffee43"
    database.create_terminal_v2(
        stale_terminal_id,
        tmux_session="cao-v2-only-stale",
        tmux_window="agent",
        provider="codex",
        generation=OTHER_GEN,
    )
    companion_receipts.record_prompt(
        stale_terminal_id,
        generation,
        prompt_id="stale-prompt",
        text="stale",
        choices=["No"],
    )
    companion_receipts.record_refusal(
        stale_terminal_id,
        generation,
        refusal_id="stale-refusal",
        identity="stale",
        turn_id="stale-turn",
    )
    companion_receipts.record_route_receipt(
        stale_terminal_id,
        generation,
        provider="codex",
        model="stale",
        effort="low",
        receipt_id="stale-route",
        turn_id="stale-turn",
    )
    companion_receipts.record_message_ack(
        stale_terminal_id,
        generation,
        message_id="stale-message",
        ack={"kind": "submitted", "message_id": "stale-message"},
    )
    for suffix in (
        "user-prompt",
        "refusal",
        "route",
        "inbox/messages/stale-message/turn-receipt",
    ):
        assert client.get(f"/terminals/{stale_terminal_id}/{suffix}").status_code == 204
    assert client.get("/terminals/deadbabe/route").status_code == 204


def test_same_terminal_id_in_both_vintages_fails_closed_for_all_companion_surfaces(
    client, companion, isolated_memory_db, monkeypatch
):
    """A cross-vintage ID collision is ambiguous, so no record is served."""
    terminal_id = "bada55e1"
    v1_generation = "44444444-4444-4444-8444-444444444444"
    v2_generation = "55555555-5555-4555-8555-555555555555"
    database.create_terminal(
        terminal_id,
        tmux_session="cao-v1-collision",
        tmux_window="agent",
        provider="codex",
        generation=v1_generation,
    )
    database.create_terminal_v2(
        terminal_id,
        tmux_session="cao-v2-collision",
        tmux_window="agent",
        provider="codex",
        generation=v2_generation,
    )
    assert database.get_terminal_metadata(terminal_id, warn_if_missing=False)["generation"] == (
        v1_generation
    )
    assert database.get_terminal_metadata_v2(terminal_id)["generation"] == v2_generation

    # The fixture normally replaces the v1 lookup with a single-row sentinel.
    # Use the real public v1 lookup here so the test reaches the cross-vintage
    # ambiguity rather than testing the fixture.
    monkeypatch.setattr(terminal_service, "get_terminal_metadata", database.get_terminal_metadata)

    def record_set(generation: str, label: str) -> None:
        companion_receipts.record_prompt(
            terminal_id,
            generation,
            prompt_id=f"{label}-prompt",
            text=f"{label} prompt",
            choices=["Yes", "No"],
        )
        companion_receipts.record_refusal(
            terminal_id,
            generation,
            refusal_id=f"{label}-refusal",
            identity=f"{label} refusal",
            turn_id=f"{label}-turn",
        )
        companion_receipts.record_route_receipt(
            terminal_id,
            generation,
            provider="codex",
            model=f"{label}-model",
            effort="high",
            receipt_id=f"{label}-route",
            turn_id=f"{label}-turn",
        )
        companion_receipts.record_message_ack(
            terminal_id,
            generation,
            message_id=f"{label}-message",
            ack={
                "kind": "submitted",
                "message_id": f"{label}-message",
                "provider": "codex",
                "provider_session_id": f"{label}-session",
                "provider_turn_id": f"{label}-turn",
                "receiver_id": terminal_id,
                "receiver_generation": generation,
            },
        )

    # Both generations have complete, otherwise valid records.  Neither is
    # safe to expose while the ID resolves to two different live rows.
    record_set(v1_generation, "v1")
    record_set(v2_generation, "v2")
    assert terminal_service.get_terminal_generation_any(terminal_id) is None
    for suffix in (
        "user-prompt",
        "refusal",
        "route",
        "inbox/messages/v1-message/turn-receipt",
        "inbox/messages/v2-message/turn-receipt",
    ):
        assert client.get(f"/terminals/{terminal_id}/{suffix}").status_code == 204
