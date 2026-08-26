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
3. ``test_garbage_render_is_ambiguous_with_proven_bytes`` — a stable writable
   surface accepts ``/status`` and only then redraws garbage; the post-effect
   observation is ambiguous, never observed-closed, with the bytes that
   reached the observer proven by tmux capture.
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
import runpy
import shutil
import sys
import termios
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
    """The conductor checkout; skips when it is unreachable.  Adds it to
    ``sys.path`` for the import of ``conduct.lib.*`` and removes it on
    teardown so no later test module inherits the cross-repo import path."""
    if h.conductor is None:
        pytest.skip(
            "conductor checkout unreachable (set T2_CONDUCTOR_REPO or use "
            "~/Projects/cao-conductor); the consumer/launch legs skip"
        )
    root = str(h.conductor)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield h.conductor
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


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
    identity = h._run(
        [
            "tmux",
            "display-message",
            "-p",
            "-t",
            pane_id,
            "#{session_name}\t#{window_name}",
        ]
    )
    assert identity.returncode == 0, identity.stderr
    session_name, window_name = identity.stdout.strip().split("\t", 1)
    surface = roc.RealCodexPaneSurface(
        pane_id,
        terminal_id=request.target_terminal_id,
        session_name=session_name,
        window_name=window_name,
        timeout=10.0,
    )
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


def _wait_for_composer_entered(h: th.T2Harness, pane_id: str, *, timeout: float = 5.0) -> None:
    """Wait until the REAL pane shows a typed ``/status`` in the composer
    followed by the Enter echo (a blank line below) — i.e. the submission
    barrier has genuinely passed compose-visible AND the Enter was sent,
    while the stub's delayed redraw still holds the pane content."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = h.capture(pane_id)
        for index, row in enumerate(rows):
            if "› /status" in row and index + 1 < len(rows) and not rows[index + 1].strip():
                return
        time.sleep(0.1)
    raise AssertionError(
        f"the pane never showed '› /status' followed by the Enter echo within "
        f"{timeout}s; last capture rows={rows!r}"
    )


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
    @pytest.mark.parametrize("bash", BASHS, ids=str)
    def test_positive_loop_end_to_end(
        self,
        bash: Path,
        h: th.T2Harness,
        installed_stub: Path,
        conductor: Path,
        isolated_memory_db,
        tmp_path: Path,
    ) -> None:
        if len(BASHS) < 2:
            pytest.skip(
                "acceptance 6 needs BOTH /bin/bash (3.2, macOS-sealed) and "
                "Homebrew bash (5.3); one is missing here"
            )
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
# acceptance 3 — writable first, then post-/status garbage is ambiguous
# ---------------------------------------------------------------------------


class TestGarbageRenderAmbiguous:
    def test_garbage_render_is_ambiguous_with_proven_bytes(
        self, h: th.T2Harness, installed_stub: Path, isolated_memory_db
    ) -> None:
        pane = h.new_pane(width=100, height=31, command=f"exec {installed_stub} garbage")
        try:
            _wait_capture(h, pane)
            before = "\n".join(h.capture(pane))
            assert "OpenAI Codex" in before
            assert "› " in before
            assert "garbage-line" not in before

            request = _request()
            outcome = _observe(h, pane, request)

            assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
            assert outcome["terminal"] is True
            assert outcome["receipt_digest"] is None
            assert outcome["observation"]["observed_state"] == "inconclusive"
            assert outcome["observation"]["reason"] == "panel-unparsed"
            assert ro.get(request.operation_id)["pre_probe_intent_json"] is not None

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
        # DEATH RUN.  The echo stub with a 3.0s delayed redraw opens a wide,
        # deterministic window: the barrier genuinely passes compose-visible
        # (the typed /status echoes) and Enter, then the pane's REAL shell is
        # killed before the delayed redraw clears the composer — so the death,
        # not a settle timeout, is what interrupts the observation.
        pane = h.new_pane(width=100, height=31, command=f"exec {installed_stub} --redraw-delay 3.0")
        _wait_capture(h, pane)
        request = _request()
        holder: dict[str, Any] = {}

        def drive() -> None:
            holder["outcome"] = _observe(h, pane, request)

        thread = threading.Thread(target=drive)
        thread.start()
        _wait_for_composer_entered(h, pane)
        assert h.pane_alive(pane), "the pane must be alive before the kill"
        h.kill_pane(pane)  # REAL process death, mid-observation
        thread.join(timeout=30)
        assert not thread.is_alive(), "the observer must terminate after pane death"

        first = holder["outcome"]
        # (a) the pane really died: it is not alive at join time.
        assert not h.pane_alive(pane)
        # The interrupted observation is honest: ambiguous, never observed-
        # closed, never a fabricated receipt.
        assert first["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert first["terminal"] is True
        assert first["receipt_digest"] is None
        assert first["observation"]["observed_state"] == "inconclusive"
        # The one /status WAS genuinely probed before the pane died.
        record = ro.get(request.operation_id)
        assert record["pre_probe_intent_json"] is not None

        # CONTROL RUN.  The same echo stub, NOT killed, must reach
        # observed-closed — proving the death-run outcome differs only because
        # the pane really died (a plain settle timeout could never produce
        # observed-closed here).  A short 1.5s redraw is enough: the barrier's
        # post-Enter bound is 5.0s, so the redraw lands with ~3.5s slack.
        control_pane = h.new_pane(
            width=100, height=31, command=f"exec {installed_stub} --redraw-delay 1.5"
        )
        try:
            _wait_capture(h, control_pane)
            control = _observe(h, control_pane, _request())
            assert control["result"] == ro.RESULT_OBSERVED_CLOSED
            assert control["receipt_digest"]
        finally:
            h.kill_pane(control_pane)

        # RESTART.  The same operation as the death run replays the durable
        # result on a fresh real pane, without a second /status.
        pane2 = h.new_pane(width=100, height=31, command=f"exec {installed_stub}")
        try:
            _wait_capture(h, pane2)
            second = _observe(h, pane2, request)
            assert second["replayed"] is True
            assert second["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
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
        if not BASHS:
            pytest.skip("no supported bash is present here")
        bash = BASHS[0]
        incident_id = f"t2-{uuid.uuid4().hex[:8]}"
        h.write_incident(incident_id, _incident_evidence(incident_id))
        env = h.setup_launch_surface(stub_bin=installed_stub)
        first = h.launch_marshal(bash=bash, incident_id=incident_id, extra_env=env)
        try:
            assert first.exit_code == 0, first.output
            assert first.pane_id, first.output
            assert first.launch_dir is not None
            second = h.launch_marshal(bash=bash, incident_id=incident_id, extra_env=env)
            assert second.exit_code == 2
            assert "already running on the root session" in second.output
            assert "Two agents resuming one session interleave" in second.output
        finally:
            h.kill_tmux_server()


# ---------------------------------------------------------------------------
# harness unit checks — stub install must not restore a leaked copy forever
# ---------------------------------------------------------------------------


class TestHarnessStubInstallLifecycle:
    """Both branches of ``install_stub``'s pre-existing-target handling: a
    REAL pre-existing binary is backed up and restored exactly; a leftover
    copy of our own stub (carrying the ``T2_SYNTHLIVE`` guard marker) is
    recognised by content, never backed up, and simply removed on teardown —
    so a crashed run's leaked stub cannot become the next run's 'real'
    binary and be restored forever."""

    def _scratch(self, tmp_path: Path, name: str) -> th.T2Harness:
        h = th.T2Harness(tmp_path / name)
        h.attested = tmp_path / f"{name}-attested"
        h.attested.mkdir(parents=True, exist_ok=True)
        return h

    def test_interactive_terminal_settings_preserve_the_restore_snapshot(
        self, installed_stub: Path
    ) -> None:
        stub = runpy.run_path(str(installed_stub), run_name="t2_stub_module")
        raw_settings = stub["_interactive_terminal_settings"]
        prior = [0, 0, 0, termios.ECHO | termios.ICANON, 0, 0, [9] * termios.NCCS]
        original_cc = list(prior[6])

        interactive = raw_settings(prior)

        assert interactive[6] is not prior[6]
        assert prior[6] == original_cc
        assert interactive[6][termios.VMIN] == 1
        assert interactive[6][termios.VTIME] == 0

    def test_real_pre_existing_binary_is_backed_up_and_restored(self, tmp_path: Path) -> None:
        h = self._scratch(tmp_path, "real")
        try:
            real = h.stub_bin()
            real.write_text("#!/bin/sh\necho real-codex\n", encoding="utf-8")
            real.chmod(0o755)
            h.install_stub()
            assert h._stub_backup is not None
            assert h._stub_backup_was_leftover is False
            assert h._stub_backup.data == b"#!/bin/sh\necho real-codex\n"
            h.uninstall_stub()
            assert real.read_text(encoding="utf-8") == "#!/bin/sh\necho real-codex\n"
            assert (real.stat().st_mode & 0o777) == 0o755
        finally:
            shutil.rmtree(h.tmux_tmp, ignore_errors=True)

    def test_leftover_stub_is_removed_not_backed_up(self, tmp_path: Path) -> None:
        h = self._scratch(tmp_path, "leftover")
        try:
            leftover = h.stub_bin()
            leftover.write_text(
                "#!/usr/bin/env python3\n# T2_SYNTHLIVE=1 stub codex; refusing to "
                "fabricate provider output\n",
                encoding="utf-8",
            )
            leftover.chmod(0o755)
            h.install_stub()
            assert h._stub_backup is None
            assert h._stub_backup_was_leftover is True
            h.uninstall_stub()
            assert not leftover.exists()
        finally:
            shutil.rmtree(h.tmux_tmp, ignore_errors=True)
