"""Event-loop liveness for the fenced comment and link write surface.

cond-0779 put compare-and-swap retries behind these routes: the tracker service
now sleeps and re-reads under SQLite lock contention. A route declared
``async def`` runs that sleeping retry on Uvicorn's event loop, so one busy
tracker write freezes every concurrent request on the server — including reads
that never touch the tracker store. A plain ``def`` route runs on the worker
threadpool instead and the loop keeps serving.

Two guards. The first pins the shape of each route callable: cheap, and it
names the offender directly. The second boots a real Uvicorn server, parks a
comment write inside the service's own retry path, and proves an unrelated
route still answers over TCP while it is parked. That is the wiring proof, and
it fails the moment ``async def`` comes back.
"""

import inspect
import socket
import sqlite3
import threading
import time as _time_module

import httpx
import pytest
import uvicorn
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.api import tracker as tracker_api
from cli_agent_orchestrator.api.main import app
from cli_agent_orchestrator.clients.database import Base
from cli_agent_orchestrator.services import issue_tracker as tracker

# Every route whose handler reaches a CAS retry loop in issue_tracker. The
# issue-scoped and feature-scoped spellings differ only in how they resolve the
# parent record; both call the same sleeping service functions.
FENCED_COMMENT_AND_LINK_ROUTES = (
    "add_comment",
    "set_comment_importance",
    "delete_comment",
    "add_link",
    "remove_link",
    "add_feature_comment",
    "set_feature_comment_importance",
    "delete_feature_comment",
    "add_feature_link",
    "remove_feature_link",
)

_BOOT_TIMEOUT = 30.0
_PARK_TIMEOUT = 30.0


def test_fenced_comment_and_link_routes_use_the_worker_threadpool():
    """Each fenced write route is a sync callable, so Uvicorn offloads it."""
    for name in FENCED_COMMENT_AND_LINK_ROUTES:
        handler = getattr(tracker_api, name)
        assert not inspect.iscoroutinefunction(handler), (
            f"{name} is a coroutine: its CAS retry loop would block the "
            "event loop and freeze every concurrent request"
        )


class _BusySession:
    """A session context that refuses to open, the way a locked store does."""

    def __enter__(self) -> None:
        raise OperationalError(
            "BEGIN IMMEDIATE", None, sqlite3.OperationalError("database is locked")
        )

    def __exit__(self, *_args) -> bool:
        return False


class _GatedTime:
    """Everything from :mod:`time` except ``sleep``, which the test gates."""

    def __init__(self, real, on_sleep):
        self._real = real
        self._on_sleep = on_sleep

    def sleep(self, seconds):
        self._on_sleep(seconds)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A reachable Uvicorn server over the real app, on a real socket.

    Lifespan is off: the tracker routes need no plugin registry, and starting
    the tmux/event-bus daemons would only add unrelated moving parts to a test
    about one route's threading.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/tracker-liveness.db")
    Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessions)

    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="tracker-liveness-uvicorn", daemon=True)
    thread.start()
    try:
        deadline = _time_module.monotonic() + _BOOT_TIMEOUT
        while not server.started:
            if not thread.is_alive():
                pytest.fail("uvicorn exited before it started serving")
            if _time_module.monotonic() > deadline:
                pytest.fail("uvicorn never started serving within the boot bound")
            _time_module.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=_PARK_TIMEOUT)


@pytest.fixture
def seeded_issue(server):
    """A project and one issue, filed over HTTP through the live server."""
    with httpx.Client(base_url=server, timeout=_BOOT_TIMEOUT) as http:
        created = http.post(
            "/tracker/projects",
            json={"name": "CAO System", "id": "cao-system", "issue_prefix": "cond"},
        )
        assert created.status_code == 201, created.text
        filed = http.post(
            "/tracker/issues",
            json={"project_id": "cao-system", "title": "a defect", "force": True},
        )
        assert filed.status_code == 201, filed.text
        return filed.json()["key"]


def test_an_unrelated_route_stays_reachable_while_a_fenced_write_is_parked(
    server, seeded_issue, monkeypatch
):
    """A blocked tracker write must not take the event loop down with it.

    The first store access is made to report the store busy, which drives the
    real ``add_comment`` into its real retry branch. The gate parks it on the
    module-level ``time.sleep`` the retry loop actually calls, and only an
    event loop that is free to schedule other work can answer the probe.
    """
    entered = threading.Event()
    release = threading.Event()

    def park_on_retry(seconds):
        entered.set()
        assert release.wait(timeout=_PARK_TIMEOUT), "the parked write was never released"

    monkeypatch.setattr(tracker, "time", _GatedTime(_time_module, park_on_retry))

    attempts = {"count": 0}
    real_sessions = tracker.SessionLocal

    def busy_first():
        attempts["count"] += 1
        if attempts["count"] == 1:
            return _BusySession()
        return real_sessions()

    monkeypatch.setattr(tracker, "SessionLocal", busy_first)

    outcome = {}

    def submit():
        with httpx.Client(base_url=server, timeout=_PARK_TIMEOUT) as http:
            outcome["response"] = http.post(
                f"/tracker/issues/{seeded_issue}/comments",
                json={"body": "audited", "author": "colin"},
            )

    writer = threading.Thread(target=submit, daemon=True)
    writer.start()
    try:
        assert entered.wait(timeout=_PARK_TIMEOUT), (
            "the comment write never reached the service retry path, so this "
            "test proved nothing about liveness"
        )

        # The probe is itself an async route: it can only be served by an event
        # loop with a free turn, which is exactly what the parked write steals
        # when the route is a coroutine.
        probe_deadline = _time_module.monotonic() + 5.0
        try:
            with httpx.Client(base_url=server, timeout=2.0) as http:
                probed = http.get("/tracker/vocabulary")
        except httpx.HTTPError as exc:
            pytest.fail(
                f"/tracker/vocabulary went unreachable while a tracker write was "
                f"parked in the service retry path: {exc!r} — the write is "
                "running on the event loop"
            )
        assert (
            _time_module.monotonic() <= probe_deadline
        ), "/tracker/vocabulary answered only after the parked write finished"
        assert probed.status_code == 200, probed.text
    finally:
        release.set()
        writer.join(timeout=_PARK_TIMEOUT)

    assert not writer.is_alive(), "the parked write never completed"
    response = outcome["response"]
    assert response.status_code == 201, response.text
    assert response.json()["body"] == "audited"
    assert attempts["count"] >= 2, "the busy first attempt never came back around"
