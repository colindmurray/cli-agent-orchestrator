"""Public scheduled-wait API backed by the fork-owned M7 lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from cli_agent_orchestrator.security.auth import (
    SCOPE_ADMIN,
    SCOPE_READ,
    SCOPE_WRITE,
    require_any_scope,
)
from cli_agent_orchestrator.services import registered_waits, wait_admission

router = APIRouter(prefix="/wait", tags=["wait"])
_READ = Depends(require_any_scope(SCOPE_READ, SCOPE_WRITE, SCOPE_ADMIN))
_WRITE = Depends(require_any_scope(SCOPE_WRITE, SCOPE_ADMIN))


class OwnerBody(BaseModel):
    agent_id: str
    incarnation_id: str
    terminal_id: str
    generation: str
    lineage_id: Optional[str] = None
    native_session_id: Optional[str] = None
    restore_contract_id: Optional[str] = None
    restore_contract_digest: Optional[str] = None


class AdapterBody(BaseModel):
    kind: str
    executable: Optional[str] = None
    executable_sha256: Optional[str] = None
    cwd: Optional[str] = None
    argv: Optional[list[str]] = None
    repository: Optional[str] = None
    run_id: Optional[int] = None
    run_attempt: Optional[int] = None
    workflow_id: Optional[int] = None
    head_sha: Optional[str] = None
    ref: Optional[str] = None

    model_config = {"extra": "forbid"}


class RegisterBody(BaseModel):
    operation_id: str
    session_name: str
    project: str
    task_id: str
    name: str
    description: str
    duration_seconds: int = Field(gt=0, le=registered_waits.MAX_ROUND_SECONDS)
    estimated_seconds: Optional[int] = Field(default=None, gt=0)
    owner: OwnerBody
    adapter: Optional[AdapterBody] = None


class CancelBody(BaseModel):
    operation_id: str
    actor: str


def _http(exc: Exception) -> HTTPException:
    if isinstance(
        exc,
        (
            registered_waits.RegisteredWaitInvalid,
            wait_admission.WaitAdmissionInvalid,
        ),
    ):
        code = status.HTTP_400_BAD_REQUEST
    elif isinstance(
        exc,
        (
            registered_waits.RegisteredWaitConflict,
            wait_admission.WaitAdmissionConflict,
        ),
    ):
        code = status.HTTP_409_CONFLICT
    elif isinstance(
        exc,
        (
            registered_waits.RegisteredWaitUnavailable,
            wait_admission.WaitAdmissionUnavailable,
        ),
    ):
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return HTTPException(status_code=code, detail=str(exc).splitlines()[0])


@router.get("/capabilities")
async def wait_capabilities(_: Any = _READ) -> dict[str, Any]:
    return registered_waits.capability()


@router.post("/registrations")
async def register_wait(body: RegisterBody, _: Any = _WRITE) -> dict[str, Any]:
    def _register() -> dict[str, Any]:
        owner = wait_admission.WaitOwner(**body.owner.model_dump())
        adapter = body.adapter.model_dump(exclude_none=True) if body.adapter else None
        return registered_waits.register(
            registered_waits.RegistrationRequest(
                operation_id=body.operation_id,
                session_name=body.session_name,
                project=body.project,
                task_id=body.task_id,
                name=body.name,
                description=body.description,
                duration_seconds=body.duration_seconds,
                estimated_seconds=body.estimated_seconds,
                owner=owner,
                adapter=adapter,
            )
        )

    try:
        return await asyncio.to_thread(_register)
    except (registered_waits.RegisteredWaitError, wait_admission.WaitAdmissionError) as exc:
        raise _http(exc) from exc


@router.get("/operations/{operation_id}")
async def get_wait_operation(operation_id: str, _: Any = _READ) -> dict[str, Any]:
    try:
        record = await asyncio.to_thread(registered_waits.get_by_operation, operation_id)
    except registered_waits.RegisteredWaitError as exc:
        raise _http(exc) from exc
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="wait operation not found"
        )
    return record


@router.get("/registrations/{wait_id}")
async def get_wait(wait_id: str, _: Any = _READ) -> dict[str, Any]:
    try:
        record = await asyncio.to_thread(registered_waits.get, wait_id)
    except registered_waits.RegisteredWaitError as exc:
        raise _http(exc) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="wait not found")
    return record


@router.get("/registrations")
async def list_registered_waits(
    session_name: Optional[str] = Query(default=None),
    terminal_id: Optional[str] = Query(default=None),
    _: Any = _READ,
) -> dict[str, Any]:
    try:
        waits = await asyncio.to_thread(
            registered_waits.list_waits, session_name=session_name, terminal_id=terminal_id
        )
    except registered_waits.RegisteredWaitError as exc:
        raise _http(exc) from exc
    return {"schema": registered_waits.SCHEMA_VERSION, "waits": waits}


@router.post("/registrations/{wait_id}/cancel")
async def cancel_wait(wait_id: str, body: CancelBody, _: Any = _WRITE) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(
            registered_waits.cancel,
            wait_id,
            operation_id=body.operation_id,
            actor=body.actor,
        )
    except registered_waits.RegisteredWaitError as exc:
        raise _http(exc) from exc
