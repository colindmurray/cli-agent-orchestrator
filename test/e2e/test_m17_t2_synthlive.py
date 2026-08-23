"""M17 T2 synthetic-live: the route-observation loop on REAL terminals.

This lane is the only autonomous gate that can prove the dark M10 route-
observation capability true without spending CP1/CP2 quota.  It drives real
tmux panes, the real ``fire-marshal.sh`` launch path, the real
``route_observation_codex`` adapter, and the real conductor
``route_completion_consumer`` — the only fake is the ``codex`` provider
binary, which renders captured fixture bytes as real pane content (see
``test/e2e/m17_t2_synthlive/``).  ``route_observation.enabled()`` is never
flipped; the adapter is driven directly exactly as the future driver loop
would, because the dark gate is a marker, not an enforcement point.

Each test maps to one T2 sub-acceptance in
``briefs/lane-T2-synthetic-live.md``:

1. ``test_stub_binary_renders_captured_fixtures_as_real_pane_bytes`` — stub
   binary at the attested layout renders the 80x30/100x30 captures as real
   pane bytes, width provable via ``tmux display -p "#{pane_width}"``.
2. ``test_positive_loop_end_to_end`` — fire-marshal launch -> /status via the
   real pane-input seam -> observation of REAL pane bytes -> consumer wakes.
3. ``test_garbage_render_is_ambiguous_with_proven_bytes`` — garbage render is
   ambiguous, never observed-closed, with the bytes that reached the observer
   proven by tmux capture.
4. ``test_pane_death_mid_observation_recovers_on_restart`` — restart recovery
   on REAL pane death (``tmux kill-pane``), never a monkeypatched kill.
5. ``test_second_launcher_is_single_root_held`` — two launchers on one
   incident hold single-root; the second is refused, never interleaved.
6. ``test_positive_loop_end_to_end`` runs under BOTH ``/bin/bash`` 3.2 and
   Homebrew bash 5.3 (``fire-marshal.sh``'s two supported shells).

Run with ``uv run pytest -m e2e test/e2e/test_m17_t2_synthlive.py -v``.  The
consumer/launch legs skip when the conductor checkout is unreachable
(``T2_CONDUCTOR_REPO`` or ``~/Projects/cao-conductor``); the fork-side pane
tests still run.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from test.e2e.m17_t2_synthlive import harness as th
from test.e2e.route_observation_canary.fixtures import (
    CAPTURED_STATUS_80X30_ROWS,
    CAPTURED_STATUS_100X30_ROWS,
    CODEX_PINNED_VERSION,
    SESSION_ID,
)
from typing import Any, Optional

import pytest

from cli_agent_orchestrator.services import native_status_repair as nsr
from cli_agent_orchestrator.services import route_observation as ro
from cli_agent_orchestrator.services import route_observation_codex as roc

pytestmark = [pytest.mark.e2e, pytest.mark.requires_tmux]

TARGET_TID = "worker-term-1"
TARGET_GEN = "gen-42"
REQ_TID = "sup-term-1"
REQ_GEN = "gen-100"
ARTIFACT = "b" * 64

BASHS = th.supported_bashes()


@pytest.fixture(scope="module", autouse=True)
def _require_tmux() -> None:
    """Skip the whole T2 module when tmux is absent (CI-safe)."""
    if shutil.which("tmux") is None:
        pytest.skip("tmux not installed; the T2 pane loop cannot run")


@pytest.fixture(scope="module")
def h(tmp_path_factory) -> th.T2Harness:
    """One isolated harness (tmux server, state root) for the module."""
    workdir = tmp_path_factory.mktemp("m17-t2")
    with th.T2Harness(workdir) as harness:
        yield harness


@pytest.fixture(scope="module")
def installed_stub(h: th.T2Harness) -> Path:
    """The stub ``codex`` binary installed at the attested layout."""
    if not h.attested_writable():
        pytest.skip(
            "the attested ~/.local/share/uv/tools/cli-agent-orchestrator/bin "
            "layout is not writable here; the stub-binary acceptance cannot run"
        )
    h.install_stub()
    return h.stub_bin()


@pytest.fixture()
def conductor(h: th.T2Harness) -> Path:
    """The conductor checkout; skips when it is unreachable."""
    if h.conductor is None:
        pytest.skip(
            "conductor checkout unreachable (set T2_CONDUCTOR_REPO or use "
            "~/Projects/cao-conductor); the consumer/launch legs skip"
        )
    root = str(h.conductor)
    if root not in sys.path:
        sys.path.insert(0, root)
    return h.conductor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _request(*, operation_id: Optional[str] = None) -> ro.RouteObservationRequest:
    return ro.RouteObservationRequest(
        operation_id=operation_id or str(uuid.uuid4()),
        target_terminal_id=TARGET_TID,
        target_generation=TARGET_GEN,
        native_session_id=SESSION_ID,
        provider="codex",
        provider_version=CODEX_PINNED_VERSION,
        provider_artifact_sha256=ARTIFACT,
        requester_terminal_id=REQ_TID,
        requester_generation=REQ_GEN,
    )


def _observe(h: th.T2Harness, pane_id: str, request: ro.RouteObservationRequest) -> dict:
    """Drive the real adapter against one REAL pane; returns the outcome."""
    surface = roc.RealCodexPaneSurface(pane_id, timeout=10.0)
    return roc.CodexRouteObserver(surface=surface).observe(request)


def _wait_capture(
    h: th.T2Harness, pane_id: str, expected: Optional[list[str]] = None, *, timeout: float = 5.0
) -> list[str]:
    """Poll the REAL pane until it renders ``expected`` (or any content when
    ``expected`` is None), bounded; returns the last capture."""
    deadline = time.monotonic() + timeout
    last: list[str] = []
    while time.monotonic() < deadline:
        last = h.capture(pane_id)
        if expected is not None:
            if last == expected:
                return last
        elif any(row.strip() for row in last):
            return last
        time.sleep(0.1)
    return last


def _incident_evidence(incident_id: str) -> dict:
    return {
        "trigger": "B",
        "scope": "campaign",
        "incident_id": incident_id,
        "evidence_quality": {"session_state_unreadable": False},
        "prior_incidents": [],
        "terminals": [
            {
                "terminal_id": TARGET_TID,
                "provider": "codex",
                "status": "not_fifo_monitored",
                "last_active": "2026-08-20T00:00:00Z",
            }
        ],
    }


def _new_consumer_store(tmp_path: Path):
    from conduct.lib import route_completion_consumer as rc

    store_path = tmp_path / "rc-consumer.sqlite3"
    rc.create(
        str(store_path),
        db_uuid="22222222-3333-4444-5555-666666666666",
        created_at="2026-08-22T09:00:00Z",
    )
    conn = rc.connect(str(store_path))
    return rc.RouteCompletionStore(conn), conn


def _write_posted_control_input(psd: str) -> None:
    from conduct.lib import control_input as journal

    os.makedirs(os.path.join(psd, journal.JOURNAL_DIRNAME), exist_ok=True)
    identity = {
        "terminal_id": TARGET_TID,
        "terminal_generation": TARGET_GEN,
        "native_session_id": SESSION_ID,
        "provider": "codex",
    }
    record = journal.new_record(
        control_id="observe-abc",
        terminal_id=TARGET_TID,
        task_id="task-1",
        kind="command",
        command="/observe",
        text="/observe",
        enter=True,
        expected_identity=identity,
    )
    journal.mark_attempt(record)
    journal.mark_transport(record, posted=True)
    journal.apply_transition(record, "posted", evidence="server accepted transport")
    journal.write_record(psd, record)


def _consume_real_wake(outcome: dict, psd: str, store: Any):
    """Feed the REAL wake + route receipt to the conductor consumer with an
    in-memory model-turn receipt (the synthetic driver-loop half)."""
    from conduct.lib import route_completion_consumer as rc

    wake = outcome["wake"]
    route_receipt = outcome["receipt"]
    contract = rc.turn_receipt_contract()
    digest = rc.wake_payload_digest(wake)
    utc = timezone.utc
    turn_receipt = contract.build_receipt(
        message_id="1001",
        message_sha256=digest,
        message_created_at=datetime(2026, 8, 22, 10, 0, 0, tzinfo=utc),
        sender_id=wake["target_terminal_id"],
        sender_generation=wake["target_generation"],
        receiver_id=wake["requester_terminal_id"],
        receiver_generation=wake["requester_generation"],
        provider=wake["provider"],
        provider_session_id=wake["native_session_id"],
        provider_turn_id="turn-1",
        submitted_at=datetime(2026, 8, 22, 10, 0, 1, tzinfo=utc),
    )
    claim = {"wake": wake, "turn_receipt": turn_receipt, "route_receipt": route_receipt}
    wakes: list[tuple[str, str, str, dict]] = []

    def wake_supervisor(
        receiver_id: str, receiver_generation: str, disposition: str, detail: dict
    ) -> None:
        wakes.append((receiver_id, receiver_generation, disposition, detail))

    resp = rc.consume_wake(
        claim,
        psd=psd,
        store=store,
        wake_supervisor=wake_supervisor,
        env={rc.ENABLED_ENV: "true"},
    )
    return resp, wakes


# ---------------------------------------------------------------------------
# acceptance 1 — the stub binary renders captured fixtures as real pane bytes
# ---------------------------------------------------------------------------


class TestStubBinaryRendersFixtures:
    def test_80_and_100_fixtures_render_as_real_pane_bytes(
        self, h: th.T2Harness, installed_stub: Path
    ) -> None:
        for width, fixture in (
            (80, CAPTURED_STATUS_80X30_ROWS),
            (100, CAPTURED_STATUS_100X30_ROWS),
        ):
            pane = h.new_pane(width=width, height=30, command=f"exec {installed_stub} status")
            try:
                assert (
                    h.pane_width(pane) == width
                ), f"tmux display -p '#{{pane_width}}' must prove {width}"
                # tmux strips each captured line's trailing whitespace, so the
                # REAL rendered bytes are the fixture rows with trailing spaces
                # removed — never a fixture read that bypasses tmux capture.
                expected = [row.rstrip() for row in fixture]
                rows = _wait_capture(h, pane, expected)
                assert rows == expected, (
                    f"the pane rendered at {width} columns must be the captured "
                    f"fixture bytes as REAL tmux content, not a fixture read"
                )
            finally:
                h.kill_pane(pane)


# ---------------------------------------------------------------------------
# acceptance 2 + 6 — the full loop on one real pane, under both supported bash
# ---------------------------------------------------------------------------


class TestPositiveLoop:
    @pytest.mark.parametrize("bash", BASHS, ids=lambda path: path.name)
    def test_positive_loop_end_to_end(
        self,
        bash: Path,
        h: th.T2Harness,
        installed_stub: Path,
        conductor: Path,
        isolated_memory_db,
        tmp_path: Path,
    ) -> None:
        if not BASHS:
            pytest.skip("neither supported bash is present here")
        incident_id = f"t2-{uuid.uuid4().hex[:8]}"
        h.write_incident(incident_id, _incident_evidence(incident_id))
        env = h.setup_launch_surface(stub_bin=installed_stub)
        launch = h.launch_marshal(bash=bash, incident_id=incident_id, extra_env=env)
        try:
            assert launch.exit_code == 0, launch.output
            assert launch.pane_id, launch.output
            assert h.pane_width(launch.pane_id) == 100
            _wait_capture(h, launch.pane_id)

            # /status delivered via the real pane-input seam; the observer
            # reads the REAL pane bytes.
            request = _request()
            outcome = _observe(h, launch.pane_id, request)

            assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
            assert outcome["terminal"] is True
            assert outcome["replayed"] is False
            assert outcome["disposition"] == roc.DISPOSITION_DELIVERED
            assert outcome["observation"]["observed_state"] == "observed"
            assert outcome["observation"]["session_id"] == SESSION_ID
            assert outcome["observation"]["model"] == "gpt-5.6-luna"
            assert outcome["observation"]["effort"] == "high"
            assert outcome["close_proof"]["outcome"] == "composer-restored"
            assert outcome["receipt_digest"]
            assert outcome["wake"]["result_kind"] == ro.RESULT_OBSERVED_CLOSED

            # the conductor consumer consumes the REAL wake.
            psd = str(tmp_path / "project-state")
            _write_posted_control_input(psd)
            store, conn = _new_consumer_store(tmp_path)
            try:
                resp, wakes = _consume_real_wake(outcome, psd, store)
            finally:
                conn.close()
            assert resp["outcome"] == "completed", resp
            assert resp["result_kind"] == ro.RESULT_OBSERVED_CLOSED
            assert len(wakes) == 1
            assert wakes[0][:3] == (REQ_TID, REQ_GEN, "completed")
        finally:
            # Kill the whole isolated server so the launched agent dies and
            # its root holder is cleared before the next bash variant runs.
            h.kill_tmux_server()


# ---------------------------------------------------------------------------
# acceptance 3 — garbage render is ambiguous, with the bytes proven
# ---------------------------------------------------------------------------


class TestGarbageRenderAmbiguous:
    def test_garbage_render_is_ambiguous_with_proven_bytes(
        self, h: th.T2Harness, installed_stub: Path, isolated_memory_db
    ) -> None:
        pane = h.new_pane(width=100, height=31, command=f"exec {installed_stub} garbage")
        try:
            _wait_capture(h, pane)
            before = "\n".join(h.capture(pane))
            assert "garbage-line" in before
            assert "OpenAI Codex" not in before

            request = _request()
            outcome = _observe(h, pane, request)

            assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
            assert outcome["terminal"] is True
            assert outcome["receipt_digest"] is None
            assert outcome["observation"]["observed_state"] == "inconclusive"
            assert outcome["observation"]["reason"] == "panel-unparsed"

            after = h.capture(pane)
            after_joined = "\n".join(after)
            assert "garbage-line" in after_joined
            assert "OpenAI Codex" not in after_joined
            # prove the text that reached the observer WAS the garbage, by
            # capturing what tmux actually rendered and binding the digest.
            assert nsr.evidence_digest(after) == outcome["observation"]["evidence_sha256"]
        finally:
            h.kill_pane(pane)


# ---------------------------------------------------------------------------
# acceptance 4 — pane death mid-observation, restart recovery on REAL death
# ---------------------------------------------------------------------------


class TestPaneDeathMidObservation:
    def test_restart_recovery_on_real_pane_death(
        self, h: th.T2Harness, installed_stub: Path, isolated_memory_db
    ) -> None:
        pane = h.new_pane(width=100, height=31, command=f"exec {installed_stub} --raw")
        _wait_capture(h, pane)
        request = _request()
        holder: dict[str, Any] = {}

        def drive() -> None:
            holder["outcome"] = _observe(h, pane, request)

        thread = threading.Thread(target=drive)
        thread.start()
        # The raw-mode stub holds the submission barrier open; kill the pane's
        # REAL shell mid-observation.
        time.sleep(0.8)
        assert h.pane_alive(pane), "the pane must be alive before the kill"
        h.kill_pane(pane)
        thread.join(timeout=30)
        assert not thread.is_alive(), "the observer must terminate after pane death"

        first = holder["outcome"]
        assert first["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert first["terminal"] is True
        assert first["receipt_digest"] is None
        assert first["observation"]["observed_state"] == "inconclusive"

        # Restart on a FRESH real pane with the SAME operation id: the durable
        # facts reconcile, the terminal result replays, and no second /status
        # ever reaches a pane.
        pane2 = h.new_pane(width=100, height=31, command=f"exec {installed_stub}")
        try:
            _wait_capture(h, pane2)
            second = _observe(h, pane2, request)
            assert second["replayed"] is True
            assert second["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
            record = ro.get(request.operation_id)
            assert record["observation_json"] is not None
            assert record["close_proof_json"] is not None
            pane2_rows = "\n".join(h.capture(pane2))
            assert "› /status" not in pane2_rows, (
                "the restarted operation must not type a second /status into " "the composer"
            )
        finally:
            h.kill_pane(pane2)


# ---------------------------------------------------------------------------
# acceptance 5 — a concurrent second launcher holds single-root
# ---------------------------------------------------------------------------


class TestConcurrentSecondLauncher:
    def test_second_launcher_is_single_root_held(
        self, h: th.T2Harness, installed_stub: Path, conductor: Path
    ) -> None:
        incident_id = f"t2-{uuid.uuid4().hex[:8]}"
        h.write_incident(incident_id, _incident_evidence(incident_id))
        env = h.setup_launch_surface(stub_bin=installed_stub)
        first = h.launch_marshal(bash=th.HOMEBREW_BASH, incident_id=incident_id, extra_env=env)
        try:
            assert first.exit_code == 0, first.output
            assert first.pane_id, first.output
            assert first.launch_dir is not None
            second = h.launch_marshal(bash=th.HOMEBREW_BASH, incident_id=incident_id, extra_env=env)
            assert second.exit_code == 2
            assert "already running on the root session" in second.output
            assert "Two agents resuming one session interleave" in second.output
        finally:
            h.kill_tmux_server()
