"""Deterministic unit mirror for the M10 Codex route-observation canary cases.

Authoring lane C3.  Every one of the five spec-§7 cases is driven here against
a fake pane surface — no real codex binary, no tmux, no paid turns — and each
case carries the ``pending_live_execution`` marker (see
``test/e2e/route_observation_canary/cases.py``) with its runner entry point
for the M17 activation lane documented in the installed tests.  The M10
capability stays dark: ``route_observation.enabled()`` and
``route_observation_codex.enabled()`` remain ``False``.

The render-floor fixtures come from the retained cond-0230 M10-D0 exact-build
capture (``codex-status-{80,100}x30.txt``): at 80 columns the truncated Model
row is ``not-rendered`` and at 100 columns the captured ``(reasoning medium,
summaries auto)`` suffix is ``model-unparsed`` — the model is never guessed.
"""

from __future__ import annotations

import json
import sys
import uuid
from test.e2e.route_observation_canary import cases, fixtures
from test.e2e.route_observation_canary import receipt as m10_receipt
from test.e2e.route_observation_canary import runner as canary_runner

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.inbox import MessageStatus
from cli_agent_orchestrator.services import native_status_repair as nsr
from cli_agent_orchestrator.services import route_observation as ro
from cli_agent_orchestrator.services import route_observation_codex as roc

ARTIFACT = "a" * 64


@pytest.fixture(autouse=True)
def _db(isolated_memory_db):
    return isolated_memory_db


def _request(
    operation_id=None,
    *,
    target_terminal_id="term-target",
    target_generation="gen-target",
    native_session_id=fixtures.SESSION_ID,
    provider="codex",
    provider_version=fixtures.CODEX_PINNED_VERSION,
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


# ---------------------------------------------------------------------------
# the capability stays dark and the case registry is closed at five
# ---------------------------------------------------------------------------


class TestDarknessAndRegistry:
    def test_the_stage_machine_capability_is_statelessly_disabled(self):
        assert ro.enabled() is False

    def test_the_adapter_delegates_the_dark_gate_and_does_not_flip_it(self):
        assert roc.enabled() is ro.enabled()
        assert ro.enabled() is False

    def test_the_case_registry_is_closed_at_five_and_every_case_is_pending_live(self):
        assert len(cases.CANARY_CASES) == 5
        assert len(cases.CASE_INDEX) == 5
        assert len(cases.RUNNER_KEYS) == 5
        case_ids = [case.case_id for case in cases.CANARY_CASES]
        assert len(set(case_ids)) == 5
        for case in cases.CANARY_CASES:
            assert cases.get_case(case.case_id) is case
            assert cases.get_case_by_runner_key(case.runner_key) is case
            assert case.pending_live_execution is True
            assert case.marker == "pending_live_execution"
            assert case.runner_entry_point == (
                f"test.e2e.route_observation_canary.runner {case.runner_key} execute"
            )
            assert cases.RUNNER_KEYS[case.runner_key].case_id == case.case_id

    def test_every_case_names_a_real_runner_subcommand(self, monkeypatch):
        for case in cases.CANARY_CASES:
            monkeypatch.setattr(sys, "argv", ["runner", case.runner_key, "--help"])
            with pytest.raises(SystemExit) as excinfo:
                canary_runner.main()
            assert excinfo.value.code == 0


# ---------------------------------------------------------------------------
# spec-§7 case 1 — positive path
# ---------------------------------------------------------------------------


class TestPositivePath:
    def test_identity_bound_status_delivered_observed_closed_mints_receipt_and_wake(self, _db):
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(
            rows=fixtures.codex_route_panel_rows(model="gpt-5.6-luna", effort="medium")
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert outcome["terminal"] is True
        assert outcome["replayed"] is False
        assert outcome["disposition"] == roc.DISPOSITION_DELIVERED
        assert outcome["receipt_digest"]
        # the one /status was issued exactly once; no Escape ever was.
        assert surface.status_commands_sent == 1
        assert surface.key_events == []

        record = ro.get(request.operation_id)
        assert record["state"] == ro.RESULT_OBSERVED_CLOSED
        for field in ro.STAGE_FACT_FIELDS:
            assert record[field] is not None, field

        # the observed route facts carry provider-native provenance and the
        # evidence digest — never raw pane text
        observation = outcome["observation"]
        assert observation["observed_state"] == "observed"
        assert observation["session_id"] == request.native_session_id
        assert observation["correlated"] is True
        assert observation["model"] == "gpt-5.6-luna"
        assert observation["effort"] == "medium"
        assert observation["evidence_sha256"]

        # the close proof is honest about the non-modal surface
        close_proof = outcome["close_proof"]
        assert close_proof["close_action"] == "none"
        assert close_proof["outcome"] == "composer-restored"

        # the positive receipt validates under the M10 canary validator
        receipt = m10_receipt.validate_receipt(outcome["receipt"])
        assert receipt["schema"] == m10_receipt.RECEIPT_SCHEMA
        assert receipt["kind"] == m10_receipt.KIND_OBSERVED_CLOSED
        assert receipt["operation_id"] == request.operation_id
        assert receipt["request_digest"] == request.request_digest()
        assert receipt["observation"] == observation
        assert receipt["close_proof"] == close_proof
        assert receipt["committed_at"]

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
        surface = fixtures.FakeCodexPaneSurface(rows=fixtures.codex_route_panel_rows(effort=None))
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert outcome["observation"]["model"] == "gpt-5.6-luna"
        assert outcome["observation"]["effort"] is None


# ---------------------------------------------------------------------------
# spec-§7 case 2 — stale requester
# ---------------------------------------------------------------------------


class TestStaleRequester:
    def test_generation_drift_records_requester_stale_with_zero_input(self, _db):
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(rows=fixtures.codex_route_panel_rows())
        observer = roc.CodexRouteObserver(
            surface=surface,
            requester_generation_probe=lambda terminal_id: "gen-drifted",
        )
        outcome = observer.observe(request)

        assert surface.status_commands_sent == 0
        assert surface.key_events == []
        assert outcome["result"] == ro.RESULT_ZERO_EFFECT_REFUSAL
        assert outcome["terminal"] is True
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

    def test_stale_requester_on_a_fully_journaled_operation_seals_the_receipt(self, _db):
        """P1 (round 2): a stale requester must never discard a provable
        observed-closed receipt.  When all four stage facts are durable and
        positively resolved, the fact-derived result is sealed (receipt
        minted) and requester-stale is carried as the disposition — ambiguous
        is only truthful when the evidence is genuinely indeterminate."""
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(rows=fixtures.codex_route_panel_rows())
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
                "model": "gpt-5.6-luna",
                "effort": "medium",
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
        receipt = json.loads(ro.get(request.operation_id)["receipt_json"])
        assert receipt["kind"] == ro.RESULT_OBSERVED_CLOSED
        assert receipt["observation"]["observed_state"] == "observed"
        assert receipt["close_proof"]["outcome"] == "composer-restored"
        assert outcome["wake"]["result_kind"] == ro.RESULT_OBSERVED_CLOSED


# ---------------------------------------------------------------------------
# spec-§7 case 3 — response loss / no replay
# ---------------------------------------------------------------------------


class TestReplayNoDuplicate:
    def test_response_loss_replays_the_stored_result_via_operation_get(self, _db):
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(rows=fixtures.codex_route_panel_rows())
        observer = roc.CodexRouteObserver(surface=surface)
        outcome = observer.observe(request)

        stored = observer.read_result(request.operation_id)
        assert stored["result"] == ro.RESULT_OBSERVED_CLOSED
        assert stored["replayed"] is True
        assert stored["disposition"] == roc.DISPOSITION_REPLAYED
        assert stored["receipt_digest"] == outcome["receipt_digest"]
        assert stored["inbox_message_id"] == outcome["inbox_message_id"]
        assert stored["receipt"] == outcome["receipt"]
        # a query never authorizes a second /status, close, or wake
        assert surface.status_commands_sent == 1
        with database.SessionLocal() as session:
            assert session.query(database.InboxModel).count() == 1

    def test_an_exact_retry_replays_without_a_second_status_or_wake(self, _db):
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(rows=fixtures.codex_route_panel_rows())
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


# ---------------------------------------------------------------------------
# spec-§7 case 4 — ambiguous close / no second Escape on the non-modal surface
# ---------------------------------------------------------------------------


class TestAmbiguousCloseNoSecondEscape:
    def test_captured_80x30_render_is_render_floor_model_and_ambiguous(self, _db):
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(
            rows=list(fixtures.CAPTURED_STATUS_80X30_ROWS), pane_width=80
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert surface.status_commands_sent == 1
        assert surface.key_events == []
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["terminal"] is True
        assert outcome["receipt_digest"] is None
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "render-floor-model"
        # the session identity is still asserted at 80 columns; the Model
        # value is not-rendered and never guessed.
        assert outcome["observation"]["session_id"] == request.native_session_id
        assert outcome["observation"]["model"] is None
        assert outcome["observation"]["effort"] is None
        assert outcome["close_proof"]["outcome"] == "composer-restored"

    def test_captured_100x30_render_is_model_unparsed_and_ambiguous(self, _db):
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(
            rows=list(fixtures.CAPTURED_STATUS_100X30_ROWS), pane_width=100
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert surface.status_commands_sent == 1
        assert surface.key_events == []
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["terminal"] is True
        assert outcome["receipt_digest"] is None
        assert outcome["observation"]["observed_state"] == "inconclusive"
        assert outcome["observation"]["reason"] == "model-row-unparsed"
        assert outcome["observation"]["session_id"] == request.native_session_id
        assert outcome["observation"]["model"] is None
        assert outcome["observation"]["effort"] is None
        assert outcome["close_proof"]["outcome"] == "composer-restored"

    def test_an_indeterminate_close_never_issues_a_second_escape(self, _db):
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(
            rows=fixtures.codex_route_panel_rows(), composer_restored=None
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["observation"]["observed_state"] == "observed"
        assert outcome["close_proof"]["outcome"] == "indeterminate"
        assert outcome["receipt_digest"] is None
        assert surface.key_events == []

    def test_an_unprovable_close_after_a_dispatched_close_is_ambiguous(self, _db):
        """A positive observation whose close cannot be proven is woken
        ambiguous and never followed by a second close action."""
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(
            rows=fixtures.codex_route_panel_rows(), composer_restored=False
        )
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)

        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["close_proof"]["outcome"] == "not-restored"
        assert outcome["close_proof"]["close_action"] == "none"
        assert surface.key_events == []

    def test_crash_after_close_commit_with_an_unprovable_close_stays_ambiguous(
        self, _db, monkeypatch
    ):
        """Spec-§7: a crash after a dispatched owned close but before the
        terminal commit must terminate ambiguous-after-possible-effect on
        retry and never issue a second close action."""
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(
            rows=fixtures.codex_route_panel_rows(), composer_restored=None
        )
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

        # the retry reconciles with the durable indeterminate close proof and
        # terminates ambiguous; no second /status and no Escape ever.
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert surface.status_commands_sent == 1
        assert outcome["result"] == ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        assert outcome["terminal"] is True
        assert outcome["close_proof"]["outcome"] == "indeterminate"
        assert outcome["receipt_digest"] is None
        assert surface.key_events == []


# ---------------------------------------------------------------------------
# spec-§7 case 5 — restart recovery via durable stage facts
# ---------------------------------------------------------------------------


class TestRestartRecovery:
    def test_durable_stage_facts_survive_a_crash_without_duplicating_an_effect(
        self, _db, monkeypatch
    ):
        """A crash after the observation committed must not strand the
        operation nonterminal, and a "restarted" observer must reconcile with
        the durable observation bytes instead of re-deriving fresh ones (which
        the machine's identical-bytes replay CAS would refuse)."""
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(rows=fixtures.codex_route_panel_rows())
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

        # a fresh observer (the restarted process) reconciles and completes
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        assert surface.status_commands_sent == 1
        assert outcome["result"] == ro.RESULT_OBSERVED_CLOSED
        assert outcome["terminal"] is True
        assert outcome["replayed"] is False
        assert outcome["receipt_digest"]
        # the receipt embeds the durable (run-1) observation, not a re-derivation
        assert outcome["observation"]["observed_state"] == "observed"
        assert outcome["observation"]["session_id"] == request.native_session_id


# ---------------------------------------------------------------------------
# the captured render-floor fixtures are never guessed
# ---------------------------------------------------------------------------


class TestCapturedRenderFloorFixtures:
    def test_80x30_render_is_below_the_model_floor_and_never_guessed(self):
        parsed = roc.parse_codex_route_panel(
            list(fixtures.CAPTURED_STATUS_80X30_ROWS), pane_width=80
        )
        assert parsed["kind"] == "partial"
        assert parsed["reason"] == "render-floor-model"
        assert parsed["session_id"] == fixtures.SESSION_ID
        assert parsed["model"] is None
        assert parsed["effort"] is None
        assert parsed["provider_version"] == fixtures.CODEX_PINNED_VERSION
        assert parsed["parser_key"] == nsr.PARSER_CODEX_STATUS

    def test_100x30_render_is_model_unparsed_and_never_guessed(self):
        parsed = roc.parse_codex_route_panel(
            list(fixtures.CAPTURED_STATUS_100X30_ROWS), pane_width=100
        )
        assert parsed["kind"] == "inconclusive"
        assert parsed["reason"] == "model-row-unparsed"
        assert parsed["session_id"] == fixtures.SESSION_ID
        assert parsed["provider_version"] == fixtures.CODEX_PINNED_VERSION
        assert parsed["parser_key"] == nsr.PARSER_CODEX_STATUS

    def test_the_fixtures_are_faithful_to_the_captured_build(self):
        # both retained captures carry exactly one branded 0.147.0 header and
        # exactly one Session row naming the substituted concrete UUID.
        for rows in (
            fixtures.CAPTURED_STATUS_80X30_ROWS,
            fixtures.CAPTURED_STATUS_100X30_ROWS,
        ):
            headers = [row for row in rows if ">_ OpenAI Codex (v0.147.0)" in row]
            sessions = [row for row in rows if row.lstrip().startswith("│  Session:")]
            assert len(headers) == 1
            assert len(sessions) == 1
            session_value = (
                sessions[0].split(":", 1)[1].translate(str.maketrans("", "", "│")).strip()
            )
            assert session_value == fixtures.SESSION_ID
            models = [row for row in rows if row.lstrip().startswith("│  Model:")]
            assert len(models) == 1
            assert "gpt-5.6-luna" in models[0]


# ---------------------------------------------------------------------------
# the M10 positive-receipt validator
# ---------------------------------------------------------------------------


class TestReceiptValidator:
    def _positive_receipt(self) -> dict:
        return {
            "schema": m10_receipt.RECEIPT_SCHEMA,
            "kind": m10_receipt.KIND_OBSERVED_CLOSED,
            "operation_id": str(uuid.uuid4()),
            "request_digest": "1" * 64,
            "requester_terminal_id": "term-requester",
            "requester_generation": "gen-requester",
            "target_terminal_id": "term-target",
            "target_generation": "gen-target",
            "native_session_id": fixtures.SESSION_ID,
            "provider": "codex",
            "provider_version": fixtures.CODEX_PINNED_VERSION,
            "provider_artifact_sha256": "2" * 64,
            "observation": {
                "kind": "provider-surface",
                "observation_kind": "codex-status-v1",
                "observed_state": "observed",
                "reason": None,
                "observed_at": "2026-08-16T00:00:00Z",
                "provider_version": fixtures.CODEX_PINNED_VERSION,
                "parser_key": "codex-status-v1",
                "session_id": fixtures.SESSION_ID,
                "correlated": True,
                "model": "gpt-5.6-luna",
                "effort": "medium",
                "render_floor": {"width": 100},
                "evidence_sha256": "3" * 64,
            },
            "close_proof": {
                "kind": "owned-close",
                "surface": "non-modal",
                "close_action": "none",
                "outcome": "composer-restored",
                "closed_at": "2026-08-16T00:00:01Z",
            },
            "final_event_digest": "4" * 64,
            "committed_at": "2026-08-16T00:00:02Z",
        }

    def test_a_positive_receipt_round_trips_with_a_deterministic_digest(self):
        payload = self._positive_receipt()

        assert m10_receipt.validate_receipt(json.loads(json.dumps(payload))) == payload
        assert m10_receipt.receipt_digest(payload) == m10_receipt.receipt_digest(
            dict(reversed(list(payload.items())))
        )

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda value: value.update(schema="cao-route-observation-receipt-v0"),
            lambda value: value.update(kind="ambiguous-after-possible-effect"),
            lambda value: value["observation"].update(observed_state="inconclusive"),
            lambda value: value["observation"].update(correlated=False),
            lambda value: value["observation"].update(effort="medium, summaries auto"),
            lambda value: value["close_proof"].update(outcome="indeterminate"),
            lambda value: value["close_proof"].update(close_action="Escape"),
            lambda value: value.update(unexpected="not part of the closed receipt"),
        ],
    )
    def test_a_receipt_rejects_drift_and_false_green_claims(self, mutate):
        payload = self._positive_receipt()
        mutate(payload)

        with pytest.raises(m10_receipt.RouteObservationReceiptInvalid):
            m10_receipt.validate_receipt(payload)

    def test_a_receipt_never_copies_raw_screen_text(self):
        payload = self._positive_receipt()
        receipt = m10_receipt.validate_receipt(payload)

        def walk(value):
            if isinstance(value, str):
                assert "\n" not in value and "\r" not in value
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)

        walk(receipt)
        # the observation carries the evidence digest, not a raw-pane key
        assert "rows" not in receipt["observation"]
        assert "screen" not in receipt["observation"]
        assert "transcript" not in receipt["observation"]
        assert receipt["observation"]["evidence_sha256"]

    def test_the_canary_writes_a_validated_receipt_file(self, _db, tmp_path):
        request = _request()
        surface = fixtures.FakeCodexPaneSurface(rows=fixtures.codex_route_panel_rows())
        outcome = roc.CodexRouteObserver(surface=surface).observe(request)
        path = tmp_path / "receipt.json"

        written = m10_receipt.write_receipt(path, outcome["receipt"])

        written_value = json.loads(path.read_text(encoding="utf-8"))
        assert written == path
        assert m10_receipt.validate_receipt(written_value) == outcome["receipt"]
        # the validator's deterministic digest is stable over the written bytes
        assert m10_receipt.receipt_digest(written_value) == m10_receipt.receipt_digest(
            outcome["receipt"]
        )


# ---------------------------------------------------------------------------
# the runner entry points the M17 activation lane wires
# ---------------------------------------------------------------------------


def _runner_spec(operation_id: str) -> dict:
    """One installed-source spec for the M17 runner prepare command."""
    return {
        "operation_id": operation_id,
        "target_terminal_id": "term-target",
        "target_generation": "gen-target",
        "native_session_id": fixtures.SESSION_ID,
        "provider": "codex",
        "provider_version": fixtures.CODEX_PINNED_VERSION,
        "provider_artifact_sha256": ARTIFACT,
        "requester_terminal_id": "term-requester",
        "requester_generation": "gen-requester",
    }


class TestRunnerEntryPoints:
    def test_prepare_builds_the_exact_operation_request(self, tmp_path):
        case = cases.POSITIVE_PATH
        spec_path = tmp_path / "spec.json"
        output_path = tmp_path / "prepared.json"
        operation_id = str(uuid.uuid4())
        spec_path.write_text(
            json.dumps(_runner_spec(operation_id), sort_keys=True),
            encoding="utf-8",
        )

        canary_runner._prepare(case.runner_key, spec_path, output_path)

        record = json.loads(output_path.read_text(encoding="utf-8"))
        assert record["case"] == case.runner_key
        assert record["request"]["operation_id"] == operation_id
        assert record["request"]["native_session_id"] == fixtures.SESSION_ID
        assert (
            record["request_digest"]
            == ro.RouteObservationRequest(**record["request"]).request_digest()
        )

    def test_execute_terminates_pending_live_consuming_the_prepared_output(self, tmp_path):
        """The documented M17 flow — ``prepare`` writes the prepared record,
        ``execute`` consumes that exact output and must reach the typed
        ``PendingLiveExecution`` seam for the prepared case (never fabricating
        a live result)."""
        case = cases.POSITIVE_PATH
        spec_path = tmp_path / "spec.json"
        prepared_path = tmp_path / "prepared.json"
        spec_path.write_text(
            json.dumps(_runner_spec(str(uuid.uuid4())), sort_keys=True),
            encoding="utf-8",
        )

        canary_runner._prepare(case.runner_key, spec_path, prepared_path)

        with pytest.raises(canary_runner.PendingLiveExecution, match=case.case_id):
            canary_runner._execute(case.runner_key, prepared_path, tmp_path / "evidence.json")

    def test_execute_refuses_a_prepared_record_for_a_different_case(self, tmp_path):
        case = cases.POSITIVE_PATH
        spec_path = tmp_path / "spec.json"
        prepared_path = tmp_path / "prepared.json"
        spec_path.write_text(
            json.dumps(_runner_spec(str(uuid.uuid4())), sort_keys=True),
            encoding="utf-8",
        )
        canary_runner._prepare(case.runner_key, spec_path, prepared_path)

        with pytest.raises(ValueError, match="names case"):
            canary_runner._execute("ambiguous-close", prepared_path, tmp_path / "evidence.json")
