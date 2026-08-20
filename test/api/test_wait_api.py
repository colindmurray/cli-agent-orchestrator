"""HTTP wiring for the public M7 timer verbs."""

from __future__ import annotations

import asyncio
import uuid

from cli_agent_orchestrator.api import wait as wait_api
from cli_agent_orchestrator.api.main import app
from cli_agent_orchestrator.services import registered_waits, wait_admission

OWNER = {
    "agent_id": "11111111-1111-4111-8111-111111111111",
    "incarnation_id": "inc-1",
    "terminal_id": "term-1",
    "generation": "gen-1",
    "lineage_id": None,
    "native_session_id": None,
    "restore_contract_id": None,
    "restore_contract_digest": None,
}


def test_router_mounts_every_public_timer_verb():
    routes = {
        (route.path, ",".join(sorted(getattr(route, "methods", ()) or ()))) for route in app.routes
    }
    assert ("/wait/capabilities", "GET") in routes
    assert ("/wait/registrations", "POST") in routes
    assert ("/wait/registrations", "GET") in routes
    assert ("/wait/registrations/{wait_id}", "GET") in routes
    assert ("/wait/registrations/{wait_id}/cancel", "POST") in routes
    assert ("/wait/operations/{operation_id}", "GET") in routes


def test_register_route_builds_the_exact_service_request(monkeypatch):
    seen = []

    def register(request):
        seen.append(request)
        return {"wait_id": "wait-1", "state": registered_waits.STATE_ACKNOWLEDGED}

    monkeypatch.setattr(registered_waits, "register", register)
    body = wait_api.RegisterBody(
        operation_id=str(uuid.uuid4()),
        session_name="cao-proj",
        project="proj",
        task_id="cond-0534",
        name="coffee",
        description="resume after coffee",
        duration_seconds=30,
        estimated_seconds=20,
        owner=wait_api.OwnerBody(**OWNER),
    )
    result = asyncio.run(wait_api.register_wait(body, None))
    assert result["state"] == registered_waits.STATE_ACKNOWLEDGED
    assert isinstance(seen[0].owner, wait_admission.WaitOwner)
    assert seen[0].owner.generation == "gen-1"
    assert seen[0].description == "resume after coffee"


def test_cancel_and_list_routes_preserve_query_and_operation(monkeypatch):
    monkeypatch.setattr(
        registered_waits,
        "list_waits",
        lambda **query: [query],
    )
    listed = asyncio.run(wait_api.list_registered_waits("cao-proj", "term-1", None))
    assert listed["waits"] == [{"session_name": "cao-proj", "terminal_id": "term-1"}]

    seen = []
    monkeypatch.setattr(
        registered_waits,
        "cancel",
        lambda wait_id, **request: seen.append((wait_id, request))
        or {"wait_id": wait_id, "state": registered_waits.STATE_CANCELLED},
    )
    operation_id = str(uuid.uuid4())
    result = asyncio.run(
        wait_api.cancel_wait(
            "wait-1", wait_api.CancelBody(operation_id=operation_id, actor="codex:worker"), None
        )
    )
    assert result["state"] == registered_waits.STATE_CANCELLED
    assert seen == [("wait-1", {"operation_id": operation_id, "actor": "codex:worker"})]
