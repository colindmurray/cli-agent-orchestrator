"""Dark COND-0230 M10-C Codex route-observation adapter tests (capability only).

``CodexRouteObserver`` drives the merged ``route_observation`` stage machine
end to end for one identity-bound ``/status`` control against a fake pane
surface — no real tmux, no real provider.  The M10 capability stays dark: the
stage-machine ``enabled()`` gate is untouched (still ``False``) and nothing
here observes a live provider surface, issues pane input against a live pane,
or delivers a wake.

The four required effect stages stay distinct in the journal exactly as the
stage machine orders them: pre-probe intent (first-CAS authorizes the one
``/status``), provider-surface observation, pre-close intent, close proof.
The Codex surface is non-modal: no ``Escape`` is ever issued and the close
proof encodes ``composer-restored`` / ``not-restored`` / ``indeterminate``
honestly, never a fabricated second ``Escape``.
"""

from __future__ import annotations

import json
import subprocess
import uuid

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.inbox import MessageStatus
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import route_observation as ro
from cli_agent_orchestrator.services import route_observation_codex as roc

ARTIFACT = "a" * 64
SESSION_ID = "4f5f46c7-b660-4f6f-a144-d2c6dceccf95"
CODEX_PINNED_VERSION = "0.147.0"


def _request(
    operation_id=None,
    *,
    target_terminal_id="term-target",
    target_generation="gen-target",
    native_session_id=SESSION_ID,
    provider="codex",
    provider_version=CODEX_PINNED_VERSION,
    provider_artifact_sha256=ARTIFACT,
    requester_terminal_id="term-requester",
    requester_generation="gen-requester",
):
    return ro.RouteObservationRequest(
        operation_id=operation_id or str(uuid.uuid4()),
        target_terminal_id=target_terminal_id,
        target_generation=target_generation,
        native_session_id=native_session_id,
        provider=provider,
        provider_version=provider_version,
        provider_artifact_sha256=provider_artifact_sha256,
        requester_terminal_id=requester_terminal_id,
        requester_generation=requester_generation,
    )


def codex_panel_rows(
    session_id: str = SESSION_ID,
    *,
    model: str = "gpt-5.4-codex",
    effort: str | None = "high",
    version: str = CODEX_PINNED_VERSION,
) -> list[str]:
    """The pinned Codex status panel: brand header, Session row, Model row.

    ``model=None`` omits the Model row entirely (a full-width panel that
    still lacks it is truncated/different, never a positive observation).
    """
    rows = [f">_ OpenAI Codex (v{version})", f"Session: {session_id}"]
    if model is not None:
        value = model + (f" (reasoning {effort})" if effort else "")
        rows.append(f"Model: {value}")
    rows.append("cwd: /Users/x/repo")
    return rows


class FakeCodexPaneSurface:
    """A fake Codex pane surface: canned screen, configured width/verdicts.

    Records every status command and every key event so the tests can prove
    at-most-once ``/status`` and the absence of any ``Escape`` on the
    non-modal surface.
    """

    def __init__(
        self,
        *,
        rows: list[str],
        pane_width: int | None = 100,
        submission_proven: bool = True,
        composer_restored: bool | None = True,
        prewrite_readiness: roc.PrewriteReadiness | None = None,
        readiness_hook=None,
    ) -> None:
        self._rows = list(rows)
        self._pane_width = pane_width
        self._submission_proven = submission_proven
        self._composer_restored = composer_restored
        self._prewrite_readiness = prewrite_readiness or roc.PrewriteReadiness(
            roc.PREWRITE_READY, TerminalStatus.IDLE.value
        )
        self._readiness_hook = readiness_hook
        self.readiness_checks = 0
        self.status_commands_sent = 0
        self.key_events: list[str] = []

    @property
    def pane_id(self) -> str:
        return "%7"

    def capture_screen(self) -> list[str]:
        return list(self._rows)

    def pane_width(self) -> int | None:
        return self._pane_width

    def await_input_ready(self) -> roc.PrewriteReadiness:
        self.readiness_checks += 1
        if self._readiness_hook is not None:
            self._readiness_hook()
        return self._prewrite_readiness

    def send_status_command(self) -> bool:
        self.status_commands_sent += 1
        return self._submission_proven

    def composer_restored(self) -> bool | None:
        return self._composer_restored

    def send_key(self, keystroke: str) -> None:
        self.key_events.append(keystroke)


@pytest.fixture(autouse=True)
def _db(isolated_memory_db):
    return isolated_memory_db


# ---------------------------------------------------------------------------
# the capability stays dark
# ---------------------------------------------------------------------------


class TestDisabled:
    def test_the_stage_machine_capability_is_statelessly_disabled(self):
        assert ro.enabled() is False

    def test_the_adapter_delegates_the_dark_gate_and_does_not_flip_it(self):
        assert roc.enabled() is ro.enabled()
        assert ro.enabled() is False


# ---------------------------------------------------------------------------
# positive path end to end
# ---------------------------------------------------------------------------


class TestPositivePath:
    def test_end_to_end_observed_closed_mints_the_receipt_and_wake(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        observer = roc.CodexRouteObserver(surface=surface)
        outcome = observer.observe(request)

        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert outcome["terminal"] is True
        assert outcome["replayed"] is False
        assert outcome["disposition"] == roc.DISPOSITION_DELIVERED
        assert outcome["receipt_digest"]
        # the one /status was issued exactly once and no Escape ever was.
        assert surface.status_commands_sent == 1
        assert surface.key_events == []

        record = ro.get(request.operation_id)
        assert record["state"] == ro.RESULT_OBSERVED_CLOSED
        for field in ro.STAGE_FACT_FIELDS:
            assert record[field] is not None, field

        # the observed route facts with provider-native provenance
        observation = outcome["observation"]
        assert observation["observed_state"] == "observed"
        assert observation["session_id"] == request.native_session_id
        assert observation["correlated"] is True
        assert observation["model"] == "gpt-5.4-codex"
        assert observation["effort"] == "high"
        assert observation["evidence_sha256"]

        # the close proof is honest about the non-modal surface
        close_proof = outcome["close_proof"]
        assert close_proof["close_action"] == "none"
        assert close_proof["outcome"] == "composer-restored"

        # the positive receipt is derived from the persisted facts
        receipt = outcome["receipt"]
        assert receipt["schema"] == ro.RECEIPT_SCHEMA
        assert receipt["kind"] == ro.RESULT_OBSERVED_CLOSED
        assert receipt["operation_id"] == request.operation_id
        assert receipt["request_digest"] == request.request_digest()
        assert receipt["observation"] == observation
        assert receipt["close_proof"] == close_proof

        # the atomic exact-requester wake claim
        assert outcome["wake"]["operation_id"] == request.operation_id
        assert outcome["wake"]["result_kind"] == ro.RESULT_OBSERVED_CLOSED
        with database.SessionLocal() as session:
            inbox = (
                session.query(database.InboxModel)
                .filter(database.InboxModel.id == outcome["inbox_message_id"])
                .one()
            )
        assert inbox.receiver_id == request.requester_terminal_id
        assert inbox.expected_receiver_generation == request.requester_generation
        assert inbox.sender_id == request.target_terminal_id
        assert inbox.status == MessageStatus.PENDING.value

    def test_a_bare_model_row_records_no_effort(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(effort=None))
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert outcome["observation"]["model"] == "gpt-5.4-codex"
        assert outcome["observation"]["effort"] is None


# ---------------------------------------------------------------------------
# zero-effect refusal and response-loss replay
# ---------------------------------------------------------------------------


class TestZeroEffectAndResponseLossReplay:
    def test_an_exact_retry_replays_without_a_second_status_or_wake(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        observer = roc.CodexRouteObserver(surface=surface)
        first = observer.observe(request)
        second = observer.observe(request)

        assert surface.status_commands_sent == 1
        assert second["replayed"] is True
        assert second["result"] == ro.RESULT_OBSERVED_CLOSED
        assert second["receipt_digest"] == first["receipt_digest"]
        assert second["inbox_message_id"] == first["inbox_message_id"]
        with database.SessionLocal() as session:
            assert session.query(database.InboxModel).count() == 1

    def test_response_loss_replay_via_get_returns_the_stored_result(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        observer = roc.CodexRouteObserver(surface=surface)
        outcome = observer.observe(request)

        stored = observer.read_result(request.operation_id)
        assert stored["result"] == ro.RESULT_OBSERVED_CLOSED
        assert stored["receipt_digest"] == outcome["receipt_digest"]
        assert stored["inbox_message_id"] == outcome["inbox_message_id"]
        # the machine's own read seam agrees
        assert ro.get(request.operation_id)["receipt_digest"] == outcome["receipt_digest"]

    def test_a_losing_operation_replays_the_zero_effect_refusal(self, _db):
        winner = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        observer = roc.CodexRouteObserver(surface=surface)
        # the winner holds the exact tuple, still requested.
        ro.claim(winner)
        loser = _request(target_terminal_id=winner.target_terminal_id)
        first = observer.observe(loser)
        second = observer.observe(loser)

        assert first["result"] == ro.RESULT_ZERO_EFFECT_REFUSAL
        assert first["terminal"] is True
        assert first["replayed"] is False
        assert first["receipt_digest"] is None
        # an exact retry replays the immutable refusal, never a re-decision.
        assert second["replayed"] is True
        assert second["inbox_message_id"] == first["inbox_message_id"]
        # the loser never typed /status and its requester still gets its own
        # deterministic zero-effect wake.
        assert surface.status_commands_sent == 0
        assert first["wake"]["result_kind"] == ro.RESULT_ZERO_EFFECT_REFUSAL


class TestPrewriteReadiness:
    def test_rendered_current_mcp_frame_resets_the_real_surface_ready_streak(self, monkeypatch):
        ready_frame = [
            ">_ OpenAI Codex (v0.149.0)",
            "",
            "› Ask Codex to do anything",
            "",
            "  gpt-5.6-luna high · ~/project",
        ]
        current_mcp_frame = [
            "› Inspect the route",
            "",
            '• Called cao-mcp-server.read_session_output({"terminal_id":"worker"})',
            "  └ Waiting for MCP response",
            "• Working (9s • esc to interrupt)",
            "",
            "› Ask Codex to do anything",
            "",
            "  gpt-5.6-luna high · ~/project",
        ]
        # The real regression showed a ready redraw before current MCP work.
        # Put that work after three ready samples so restoring the old
        # three-poll threshold returns before the detector can see it.
        frames = iter([ready_frame] * 3 + [current_mcp_frame] + [ready_frame] * 11)
        captures = []

        def capture(pane_id, *, timeout):
            frame = next(frames)
            captures.append((pane_id, timeout, frame))
            return frame

        monkeypatch.setattr(roc.npi, "capture_pane_screen", capture)
        monkeypatch.setattr(roc.time, "sleep", lambda _seconds: None)
        surface = roc.RealCodexPaneSurface(
            "%7",
            terminal_id="term-target",
            session_name="cao-target",
            window_name="managed-target",
            timeout=1.0,
        )

        readiness = surface.await_input_ready()

        assert readiness == roc.PrewriteReadiness(roc.PREWRITE_READY, "idle")
        assert len(captures) == 15
        assert captures[3][2] == current_mcp_frame
        assert all(capture[0] == "%7" for capture in captures)

    def test_incomplete_ready_streak_times_out_with_zero_effect(self, _db, monkeypatch):
        monkeypatch.setattr(
            roc.npi,
            "observe_codex_turn_state",
            lambda *args, **kwargs: TerminalStatus.IDLE,
        )

        class IncompleteReadySurface(roc.RealCodexPaneSurface):
            def __init__(self):
                super().__init__(
                    "%7",
                    terminal_id="term-target",
                    session_name="cao-target",
                    window_name="managed-target",
                    timeout=0.0,
                )
                self.status_commands_sent = 0

            def send_status_command(self):
                self.status_commands_sent += 1
                return True

        request = _request()
        surface = IncompleteReadySurface()

        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert outcome["result"] == ro.RESULT_ZERO_EFFECT_REFUSAL
        assert outcome["disposition"] == roc.DISPOSITION_PROVIDER_NOT_READY
        assert outcome["prewrite_readiness"] == {
            "reason": roc.PREWRITE_PROVIDER_NOT_READY,
            "provider_status": TerminalStatus.IDLE.value,
            "detail": "the bound Codex pane did not remain ready for 11 consecutive observations",
        }
        assert surface.status_commands_sent == 0
        record = ro.get(request.operation_id)
        assert all(record[field] is None for field in ro.STAGE_FACT_FIELDS)

    def test_real_surface_unreadable_sample_resets_the_ready_streak(self, monkeypatch):
        observations = iter(
            [
                TerminalStatus.IDLE,
                roc.npi.NativePaneInputUnavailable("tmux capture failed"),
                *([TerminalStatus.IDLE, TerminalStatus.COMPLETED] * 5),
                TerminalStatus.IDLE,
            ]
        )
        calls = 0

        def observe(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = next(observations)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(roc.npi, "observe_codex_turn_state", observe)
        monkeypatch.setattr(roc.time, "sleep", lambda _seconds: None)
        surface = roc.RealCodexPaneSurface(
            "%7",
            terminal_id="term-target",
            session_name="cao-target",
            window_name="managed-target",
            timeout=1.0,
        )

        readiness = surface.await_input_ready()

        assert readiness == roc.PrewriteReadiness(roc.PREWRITE_READY, "idle")
        assert calls == 13

    def test_real_surface_observer_uses_one_literal_and_one_enter_through_the_barrier(
        self, _db, monkeypatch
    ):
        ready_frame = ["› Ask Codex to do anything", "", "  gpt-5.6-luna high · ~/project"]
        empty_frame = ["› ", "", "  gpt-5.6-luna high · ~/project"]
        composed_frame = [
            "transcript",
            "› /status",
            "",
            "  /status      show current session configuration and token usage",
            "  /statusline  configure which items appear in the status line",
            *([""] * 12),
        ]
        plain_frames = iter([ready_frame] * 11 + [codex_panel_rows(), ready_frame])
        styled_frames = iter([composed_frame, empty_frame])
        captures = []
        tmux_calls = []

        def capture(pane_id, *, timeout):
            frame = next(plain_frames)
            captures.append(("plain", pane_id, timeout, frame))
            return frame

        def capture_styled(pane_id, *, timeout):
            frame = next(styled_frames)
            captures.append(("styled", pane_id, timeout, frame))
            return frame

        def run(argv, *, timeout):
            tmux_calls.append((list(argv), timeout))
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(roc.npi, "capture_pane_screen", capture)
        monkeypatch.setattr(roc.npi, "capture_pane_screen_styled", capture_styled)
        monkeypatch.setattr(roc.npi, "_tmux_binary", lambda: "tmux")
        monkeypatch.setattr(roc.npi, "_run", run)
        monkeypatch.setattr(roc, "_pane_width", lambda pane_id, *, timeout: 100)
        monkeypatch.setattr(roc.time, "sleep", lambda _seconds: None)
        surface = roc.RealCodexPaneSurface(
            "%7",
            terminal_id="term-target",
            session_name="cao-target",
            window_name="managed-target",
            timeout=1.0,
        )

        outcome = roc.CodexRouteObserver(surface=surface).observe(_request())

        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert len(captures) == 15
        assert [capture[0] for capture in captures] == [
            *(["plain"] * 11),
            "styled",
            "styled",
            "plain",
            "plain",
        ]
        assert [call[0] for call in tmux_calls] == [
            ["tmux", "send-keys", "-t", "%7", "-l", "--", "/status"],
            ["tmux", "send-keys", "-t", "%7", "Enter"],
        ]

    def test_real_surface_unreadable_observer_writes_no_literal_or_enter(self, _db, monkeypatch):
        tmux_calls = []

        def unreadable(*args, **kwargs):
            raise roc.npi.NativePaneInputUnavailable("tmux capture failed")

        def run(argv, *, timeout):
            tmux_calls.append((list(argv), timeout))
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(roc.npi, "capture_pane_screen", unreadable)
        monkeypatch.setattr(roc.npi, "_run", run)
        surface = roc.RealCodexPaneSurface(
            "%7",
            terminal_id="term-target",
            session_name="cao-target",
            window_name="managed-target",
            timeout=0.0,
        )

        outcome = roc.CodexRouteObserver(surface=surface).observe(_request())

        assert outcome["result"] == ro.RESULT_ZERO_EFFECT_REFUSAL
        assert outcome["disposition"] == roc.DISPOSITION_PANE_UNREADABLE
        assert tmux_calls == []

    def test_real_surface_busy_timeout_is_not_an_unreadable_pane(self, monkeypatch):
        monkeypatch.setattr(
            roc.npi,
            "observe_codex_turn_state",
            lambda *args, **kwargs: TerminalStatus.PROCESSING,
        )
        surface = roc.RealCodexPaneSurface(
            "%7",
            terminal_id="term-target",
            session_name="cao-target",
            window_name="managed-target",
            timeout=0.0,
        )

        assert surface.await_input_ready() == roc.PrewriteReadiness(
            roc.PREWRITE_PROVIDER_NOT_READY,
            TerminalStatus.PROCESSING.value,
        )

    def test_real_surface_unreadable_timeout_is_not_observed_busy(self, monkeypatch):
        def unreadable(*args, **kwargs):
            raise roc.npi.NativePaneInputUnavailable("tmux capture failed")

        monkeypatch.setattr(roc.npi, "observe_codex_turn_state", unreadable)
        surface = roc.RealCodexPaneSurface(
            "%7",
            terminal_id="term-target",
            session_name="cao-target",
            window_name="managed-target",
            timeout=0.0,
        )

        assert surface.await_input_ready() == roc.PrewriteReadiness(
            roc.PREWRITE_PANE_UNREADABLE,
            None,
            "tmux capture failed",
        )

    @pytest.mark.parametrize(
        ("readiness", "disposition"),
        [
            (
                roc.PrewriteReadiness(
                    roc.PREWRITE_PROVIDER_NOT_READY,
                    TerminalStatus.PROCESSING.value,
                ),
                roc.DISPOSITION_PROVIDER_NOT_READY,
            ),
            (
                roc.PrewriteReadiness(
                    roc.PREWRITE_PANE_UNREADABLE,
                    None,
                    "tmux capture failed",
                ),
                roc.DISPOSITION_PANE_UNREADABLE,
            ),
        ],
    )
    def test_prewrite_refusal_is_typed_and_has_no_effect_fact(self, _db, readiness, disposition):
        request = _request()

        def assert_pre_probe_absent():
            record = ro.get(request.operation_id)
            assert record is not None
            assert all(record[field] is None for field in ro.STAGE_FACT_FIELDS)

        surface = FakeCodexPaneSurface(
            rows=codex_panel_rows(),
            prewrite_readiness=readiness,
            readiness_hook=assert_pre_probe_absent,
        )

        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert outcome["result"] == ro.RESULT_ZERO_EFFECT_REFUSAL
        assert outcome["disposition"] == disposition
        assert outcome["prewrite_readiness"] == readiness.fact()
        assert surface.status_commands_sent == 0
        record = ro.get(request.operation_id)
        assert all(record[field] is None for field in ro.STAGE_FACT_FIELDS)
        event = json.loads(record["final_event_json"])
        assert event["prewrite_readiness"] == readiness.fact()
        assert event["disposition"] == disposition

    def test_requester_is_revalidated_after_the_readiness_wait(self, _db):
        request = _request()
        generations = iter([request.requester_generation, "gen-drifted"])
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        observer = roc.CodexRouteObserver(
            surface=surface,
            requester_generation_probe=lambda _terminal_id: next(generations),
        )

        outcome = observer.observe(request)

        assert outcome["result"] == ro.RESULT_ZERO_EFFECT_REFUSAL
        assert outcome["disposition"] == roc.DISPOSITION_REQUESTER_STALE
        assert surface.readiness_checks == 1
        assert surface.status_commands_sent == 0
        record = ro.get(request.operation_id)
        assert all(record[field] is None for field in ro.STAGE_FACT_FIELDS)

    def test_prewrite_refusal_replays_without_rechecking_or_writing(self, _db):
        request = _request()
        readiness = roc.PrewriteReadiness(
            roc.PREWRITE_PROVIDER_NOT_READY,
            TerminalStatus.PROCESSING.value,
        )
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(), prewrite_readiness=readiness)
        observer = roc.CodexRouteObserver(surface=surface)

        first = observer.observe(request)
        second = observer.observe(request)
        reread = observer.read_result(request.operation_id)

        assert first["result"] == ro.RESULT_ZERO_EFFECT_REFUSAL
        assert second["replayed"] is True
        assert second["prewrite_readiness"] == readiness.fact()
        assert reread["prewrite_readiness"] == readiness.fact()
        assert surface.readiness_checks == 1
        assert surface.status_commands_sent == 0
        assert second["inbox_message_id"] == first["inbox_message_id"]
        with database.SessionLocal() as session:
            assert session.query(database.InboxModel).count() == 1

    def test_a_post_authorization_compose_race_stays_ambiguous_and_is_not_retried(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(
            rows=codex_panel_rows(),
            submission_proven=False,
        )
        observer = roc.CodexRouteObserver(surface=surface)

        first = observer.observe(request)
        second = observer.observe(request)

        assert first["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert first["observation"]["reason"] == "submission-unproven"
        assert surface.readiness_checks == 1
        assert surface.status_commands_sent == 1
        assert second["replayed"] is True
        record = ro.get(request.operation_id)
        assert record["pre_probe_intent_json"] is not None


# ---------------------------------------------------------------------------
# stale-requester disposition on generation drift
# ---------------------------------------------------------------------------


class TestStaleRequester:
    def test_generation_drift_records_requester_stale_with_zero_input(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        observer = roc.CodexRouteObserver(
            surface=surface,
            requester_generation_probe=lambda terminal_id: "gen-drifted",
        )
        outcome = observer.observe(request)

        assert surface.status_commands_sent == 0
        assert surface.key_events == []
        assert outcome["result"] == ro.RESULT_ZERO_EFFECT_REFUSAL
        assert outcome["disposition"] == roc.DISPOSITION_REQUESTER_STALE
        assert outcome["receipt_digest"] is None

        record = ro.get(request.operation_id)
        event = json.loads(record["final_event_json"])
        assert event["disposition"] == roc.DISPOSITION_REQUESTER_STALE
        assert event["result"] == ro.RESULT_ZERO_EFFECT_REFUSAL

        # the immutable wake is still claimed with the exact captured requester
        assert outcome["wake"]["result_kind"] == ro.RESULT_ZERO_EFFECT_REFUSAL
        with database.SessionLocal() as session:
            inbox = (
                session.query(database.InboxModel)
                .filter(database.InboxModel.id == outcome["inbox_message_id"])
                .one()
            )
        assert inbox.receiver_id == request.requester_terminal_id
        assert inbox.expected_receiver_generation == request.requester_generation

    def test_stale_requester_on_a_partially_journaled_operation_terminates_ambiguous(self, _db):
        """P2: a stale requester on an operation that already committed effect
        facts must not raise a later-stage conflict.  The requester-stale
        disposition wins and the operation terminates
        ambiguous-after-possible-effect with zero input."""
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        # a prior run journaled the pre-probe intent and the observation (a
        # possible effect) before the requester generation drifted.
        ro.claim(request)
        ro.pre_probe(request, intent={"kind": "pre-probe-intent", "surface": "codex-status-v1"})
        ro.record_observation(
            request,
            observation={
                "kind": "provider-surface",
                "observed_state": "observed",
                "observed_at": "2026-08-16T00:00:00Z",
            },
        )
        observer = roc.CodexRouteObserver(
            surface=surface,
            requester_generation_probe=lambda terminal_id: "gen-drifted",
        )
        outcome = observer.observe(request)

        assert surface.status_commands_sent == 0
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["terminal"] is True
        assert outcome["disposition"] == roc.DISPOSITION_REQUESTER_STALE
        assert outcome["receipt_digest"] is None
        record = ro.get(request.operation_id)
        event = json.loads(record["final_event_json"])
        assert event["disposition"] == roc.DISPOSITION_REQUESTER_STALE
        assert event["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT

    def test_stale_requester_on_a_fully_journaled_operation_seals_the_observed_receipt(self, _db):
        """P1 (round 2): a stale requester must never discard a provable
        observed-closed receipt.  When all four stage facts are durable and
        positively resolved, the fact-derived result is sealed (receipt
        minted) and requester-stale is carried as the disposition — ambiguous
        is only truthful when the evidence is genuinely indeterminate."""
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        # a prior run journaled the complete positive proof chain: pre-probe,
        # a correlated positive observation, pre-close, and a composer-restored
        # close proof — all four facts durable, no terminal commit yet.
        ro.claim(request)
        ro.pre_probe(request, intent={"kind": "pre-probe-intent", "surface": "codex-status-v1"})
        ro.record_observation(
            request,
            observation={
                "kind": "provider-surface",
                "observation_kind": "codex-status-v1",
                "observed_state": "observed",
                "session_id": request.native_session_id,
                "correlated": True,
                "model": "gpt-5.4-codex",
                "effort": "high",
                "observed_at": "2026-08-16T00:00:00Z",
            },
        )
        ro.pre_close(request, intent={"kind": "pre-close-intent", "modal": False, "close": "none"})
        ro.record_close_proof(
            request,
            proof={
                "kind": "owned-close",
                "surface": "non-modal",
                "close_action": "none",
                "outcome": "composer-restored",
                "closed_at": "2026-08-16T00:00:01Z",
            },
        )
        stored = ro.get(request.operation_id)
        assert stored["state"] == ro.STATE_REQUESTED
        for field in ro.STAGE_FACT_FIELDS:
            assert stored[field] is not None, field

        observer = roc.CodexRouteObserver(
            surface=surface,
            requester_generation_probe=lambda terminal_id: "gen-drifted",
        )
        outcome = observer.observe(request)

        assert surface.status_commands_sent == 0
        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert outcome["terminal"] is True
        assert outcome["disposition"] == roc.DISPOSITION_REQUESTER_STALE
        assert outcome["receipt_digest"]
        record = ro.get(request.operation_id)
        assert record["state"] == ro.RESULT_OBSERVED_CLOSED
        event = json.loads(record["final_event_json"])
        assert event["disposition"] == roc.DISPOSITION_REQUESTER_STALE
        assert event["result"] == ro.RESULT_OBSERVED_CLOSED
        receipt = json.loads(record["receipt_json"])
        assert receipt["kind"] == ro.RESULT_OBSERVED_CLOSED
        assert receipt["observation"]["observed_state"] == "observed"
        assert receipt["close_proof"]["outcome"] == "composer-restored"
        assert outcome["wake"]["result_kind"] == ro.RESULT_OBSERVED_CLOSED


# ---------------------------------------------------------------------------
# ambiguous-after-possible-effect handling on the non-modal surface
# ---------------------------------------------------------------------------


class TestAmbiguousAfterPossibleEffect:
    def test_an_unparseable_panel_terminates_ambiguous(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=["> not a status panel", "garbage"])
        observer = roc.CodexRouteObserver(surface=surface)
        outcome = observer.observe(request)

        assert surface.status_commands_sent == 1
        assert surface.key_events == []
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["terminal"] is True
        assert outcome["receipt_digest"] is None
        assert outcome["observation"]["observed_state"] == "inconclusive"
        # the composer did return; the panel itself was unparseable, which is
        # what makes the observation inconclusive and the result ambiguous.
        assert outcome["close_proof"]["outcome"] == "composer-restored"

    def test_an_unproven_submission_is_ambiguous(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(), submission_proven=False)
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "submission-unproven"

    def test_a_positive_observation_with_an_indeterminate_close_is_ambiguous(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(), composer_restored=None)
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "observed"
        assert outcome["close_proof"]["outcome"] == "indeterminate"
        assert outcome["receipt_digest"] is None
        assert surface.key_events == []

    def test_a_positive_observation_with_a_not_restored_composer_is_ambiguous(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(), composer_restored=False)
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["close_proof"]["outcome"] == "not-restored"
        assert outcome["receipt_digest"] is None

    def test_a_transport_failure_is_a_possible_effect_and_terminates_ambiguous(self, _db):
        request = _request()
        surface = _RaisingSendSurface(rows=codex_panel_rows())
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "send-failed"
        assert surface.status_commands_sent == 1


class _RaisingSendSurface(FakeCodexPaneSurface):
    def send_status_command(self) -> bool:
        self.status_commands_sent += 1
        raise RuntimeError("tmux refused the write")


# ---------------------------------------------------------------------------
# the pinned render floors are respected when asserting observed state
# ---------------------------------------------------------------------------


class TestRenderFloor:
    def test_width_below_the_session_floor_cannot_assert_identity(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(), pane_width=70)
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "render-floor-session"

    def test_width_between_the_floors_asserts_session_but_not_model(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(), pane_width=80)
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "render-floor-model"
        assert outcome["observation"]["session_id"] == request.native_session_id
        assert outcome["observation"]["model"] is None
        assert outcome["observation"]["effort"] is None

    def test_a_malformed_model_reasoning_suffix_is_not_positive(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(
            rows=codex_panel_rows(model="gpt-5.4-codex (reasoning turbo)", effort=None)
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["reason"] == "model-row-unparsed"
        # unknown effort remains inconclusive and no non-authoritative detail
        # is exposed as a parsed fact.
        assert outcome["observation"]["effort"] is None

    def test_an_empty_reasoning_parenthetical_is_not_a_bare_model(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(
            rows=codex_panel_rows(model="gpt-5.4-codex (reasoning )", effort=None)
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "model-row-unparsed"
        assert outcome["observation"]["model"] is None
        assert outcome["observation"]["effort"] is None

    def test_a_truncated_model_row_at_full_width_is_not_reported_observed(self, _db):
        """P2 (round 2): a known width is not enough — the Model row's own
        content must be complete.  A value cut mid-parenthetical at a width
        that should have rendered it fully is a truncated capture, and the
        closed effort parse fails closed rather than reporting a half token."""
        request = _request()
        surface = FakeCodexPaneSurface(
            rows=codex_panel_rows(model="gpt-5.4-codex (reasoning h", effort=None),
            pane_width=roc.MODEL_RENDER_FLOOR_COLUMNS,
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "model-row-unparsed"
        assert outcome["observation"]["model"] is None
        assert outcome["observation"]["effort"] is None

    def test_a_truncated_model_row_with_unknown_width_is_not_reported_observed(self, _db):
        """P3: the Model row is asserted only when the pane width proves it
        rendered.  With an unknown/stale width, a truncated Model value must
        not be reported observed — row presence alone never proves the row."""
        request = _request()
        rows = [
            ">_ OpenAI Codex (v0.147.0)",
            f"Session: {SESSION_ID}",
            "Model: gpt-5.4-codex (reasoning h",  # truncated at the pane edge
            "cwd: /Users/x/repo",
        ]
        surface = FakeCodexPaneSurface(rows=rows, pane_width=None)
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "render-floor-model"
        assert outcome["observation"]["model"] is None
        assert outcome["observation"]["effort"] is None


# ---------------------------------------------------------------------------
# reasoning effort extraction ignores non-authoritative trailing annotations
# ---------------------------------------------------------------------------


class TestReasoningEffortExtraction:
    def test_the_captured_annotation_is_ignored_and_authority_is_observed(self, _db):
        request = _request()
        surface = FakeCodexPaneSurface(
            rows=codex_panel_rows(
                model="gpt-5.6-luna (reasoning medium, summaries auto)", effort=None
            )
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        observation = outcome["observation"]
        assert observation["session_id"] == SESSION_ID
        assert observation["model"] == "gpt-5.6-luna"
        assert observation["effort"] == "medium"
        assert "summaries" not in observation
        stored = json.loads(ro.get(request.operation_id)["observation_json"])
        assert stored == observation

    @pytest.mark.parametrize(
        "annotation",
        ["", ", summaries auto", ", arbitrary display text", ", one, two"],
    )
    def test_trailing_annotation_does_not_change_authority(self, annotation):
        parsed = roc.parse_codex_route_panel(
            [
                f">_ OpenAI Codex (v{CODEX_PINNED_VERSION})",
                f"Session: {SESSION_ID}",
                f"Model: gpt-5.6-luna (reasoning medium{annotation})",
            ],
            pane_width=100,
        )
        assert parsed["kind"] == "observed"
        assert parsed["model"] == "gpt-5.6-luna"
        assert parsed["effort"] == "medium"
        assert set(parsed) == {
            "kind",
            "session_id",
            "provider_version",
            "parser_key",
            "model",
            "effort",
            "pane_width",
            "evidence_sha256",
        }

    def test_trailing_annotation_does_not_change_evidence_digest(self):
        prefix = [
            f">_ OpenAI Codex (v{CODEX_PINNED_VERSION})",
            f"Session: {SESSION_ID}",
        ]
        bare = roc.parse_codex_route_panel(
            prefix + ["Model: gpt-5.6-luna (reasoning medium)"], pane_width=100
        )
        decorated = roc.parse_codex_route_panel(
            prefix + ["Model: gpt-5.6-luna (reasoning medium, summaries auto)"],
            pane_width=100,
        )
        assert decorated["evidence_sha256"] == bare["evidence_sha256"]

    @pytest.mark.parametrize("effort", sorted(roc._CODEX_EFFORT_VOCABULARY))
    def test_every_vocabulary_member_observes(self, _db, effort):
        """Every member of the closed effort vocabulary is authoritative."""
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(effort=effort))
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        observation = outcome["observation"]
        assert observation["observed_state"] == "observed"
        assert observation["effort"] == effort
        stored = json.loads(ro.get(request.operation_id)["observation_json"])
        assert stored["effort"] == effort

    def test_a_truncated_parenthetical_remains_inconclusive(self, _db):
        """A value cut mid-parenthetical remains inconclusive."""
        request = _request()
        surface = FakeCodexPaneSurface(
            rows=codex_panel_rows(model="gpt-5.4-codex (reasoning h", effort=None),
            pane_width=roc.MODEL_RENDER_FLOOR_COLUMNS,
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        observation = outcome["observation"]
        assert observation["reason"] == "model-row-unparsed"
        assert observation["model"] is None
        assert observation["effort"] is None
        stored = json.loads(ro.get(request.operation_id)["observation_json"])
        assert stored["reason"] == "model-row-unparsed"

    def test_two_model_rows_remain_inconclusive(self, _db):
        """A duplicated authoritative Model row remains inconclusive."""
        request = _request()
        rows = codex_panel_rows() + ["Model: gpt-5.4-codex (reasoning low)"]
        surface = FakeCodexPaneSurface(rows=rows)
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        observation = outcome["observation"]
        assert observation["observed_state"] == "inconclusive"
        assert observation["reason"] == "model-row-unparsed"
        assert observation["model"] is None
        assert observation["effort"] is None
        stored = json.loads(ro.get(request.operation_id)["observation_json"])
        assert stored["reason"] == "model-row-unparsed"

    def test_send_failure_and_unproven_submission_have_no_effort(self, _db):
        """The no-panel inconclusive path has no parsed effort."""
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(), submission_proven=False)
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["observation"]["reason"] == "submission-unproven"
        assert outcome["observation"]["effort"] is None


# ---------------------------------------------------------------------------
# the observation is correlated to the exact target
# ---------------------------------------------------------------------------


class TestTargetCorrelation:
    def test_a_mismatched_session_is_not_a_positive_observation(self, _db):
        request = _request(native_session_id="3f3f9a1c-0000-4000-8000-0000000000aa")
        surface = FakeCodexPaneSurface(rows=codex_panel_rows(session_id=SESSION_ID))
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "target-mismatch"
        assert outcome["observation"]["correlated"] is False


# ---------------------------------------------------------------------------
# crash-retry recovery: an exact retry reconciles with durable stage facts
# ---------------------------------------------------------------------------


class TestCrashRetryRecovery:
    @pytest.mark.parametrize(
        "readiness",
        [
            roc.PrewriteReadiness(
                roc.PREWRITE_PROVIDER_NOT_READY,
                TerminalStatus.PROCESSING.value,
            ),
            roc.PrewriteReadiness(
                roc.PREWRITE_PANE_UNREADABLE,
                None,
                "tmux capture failed",
            ),
        ],
    )
    def test_retry_after_pre_probe_skips_the_zero_effect_readiness_gate(self, _db, readiness):
        request = _request()
        ro.claim(request)
        ro.pre_probe(request, intent=roc._pre_probe_intent(request, pane_id="%7"))
        surface = FakeCodexPaneSurface(
            rows=["Codex is still starting"],
            prewrite_readiness=readiness,
        )

        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert surface.readiness_checks == 0
        assert surface.status_commands_sent == 0
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["reason"] == "panel-unparsed"
        record = ro.get(request.operation_id)
        assert record["pre_probe_intent_json"] is not None
        assert record["observation_json"] is not None

    def test_retry_with_a_stored_observation_skips_readiness_and_reuses_it(self, _db):
        request = _request()
        ro.claim(request)
        ro.pre_probe(request, intent=roc._pre_probe_intent(request, pane_id="%7"))
        stored_observation = roc._observation_from_parse(
            request,
            roc.parse_codex_route_panel(
                codex_panel_rows(),
                pinned_version=request.provider_version,
                pane_width=100,
            ),
        )
        ro.record_observation(request, observation=stored_observation)
        surface = FakeCodexPaneSurface(
            rows=["not the stored panel"],
            prewrite_readiness=roc.PrewriteReadiness(
                roc.PREWRITE_PANE_UNREADABLE,
                None,
                "tmux capture failed",
            ),
        )

        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert surface.readiness_checks == 0
        assert surface.status_commands_sent == 0
        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert outcome["observation"] == stored_observation

    def test_crash_after_observation_commit_recovers_on_exact_retry(self, _db, monkeypatch):
        """P1: a crash after the observation committed must not strand the
        operation nonterminal.  The retry reconciles with the durable
        observation bytes instead of re-deriving fresh ones (which the
        machine's identical-bytes replay CAS would refuse)."""
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        observer = roc.CodexRouteObserver(surface=surface)
        calls = {"count": 0}
        original = roc.ro.record_close_proof

        def crashing_close_proof(req, *, proof, db=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("simulated crash after the observation commit")
            return original(req, proof=proof, db=db)

        monkeypatch.setattr(roc.ro, "record_close_proof", crashing_close_proof)

        with pytest.raises(RuntimeError, match="simulated crash"):
            observer.observe(request)

        # the observation is durably committed and the operation is nonterminal
        stored = ro.get(request.operation_id)
        assert stored["state"] == ro.STATE_REQUESTED
        assert stored["observation_json"] is not None
        assert stored["close_proof_json"] is None

        # an exact retry reconciles with the durable observation instead of
        # re-deriving fresh bytes and conflicting on the changed-fact CAS.
        outcome = observer.observe(request)
        assert surface.status_commands_sent == 1
        assert surface.readiness_checks == 1
        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert outcome["terminal"] is True
        assert outcome["replayed"] is False
        assert outcome["receipt_digest"]
        # the receipt embeds the durable (run-1) observation, not a re-derivation
        assert outcome["observation"]["observed_state"] == "observed"
        assert outcome["observation"]["session_id"] == request.native_session_id

    def test_crash_after_close_proof_commit_recovers_on_exact_retry(self, _db, monkeypatch):
        """P1: a crash after the close proof committed (before the terminal
        commit) must recover on an exact retry by reusing both durable facts."""
        request = _request()
        surface = FakeCodexPaneSurface(rows=codex_panel_rows())
        observer = roc.CodexRouteObserver(surface=surface)
        calls = {"count": 0}
        original = roc.ro.complete

        def crashing_complete(req, *, result, final_event, db=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("simulated crash after the close proof commit")
            return original(req, result=result, final_event=final_event, db=db)

        monkeypatch.setattr(roc.ro, "complete", crashing_complete)

        with pytest.raises(RuntimeError, match="simulated crash"):
            observer.observe(request)

        stored = ro.get(request.operation_id)
        assert stored["state"] == ro.STATE_REQUESTED
        for field in ro.STAGE_FACT_FIELDS:
            assert stored[field] is not None, field

        outcome = observer.observe(request)
        assert surface.status_commands_sent == 1
        assert surface.readiness_checks == 1
        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert outcome["terminal"] is True
        assert outcome["receipt_digest"]
        assert outcome["inbox_message_id"] is not None
