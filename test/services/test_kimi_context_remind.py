"""KIND_REMIND delivery: one pending per occurrence, frozen payload, marker ack.

Covers the cond-0845 reminder journal on top of the adapter's existing
guarantees (zero bytes on refusal; posted is not accepted). The
positives are the two branches — observed idle submits an ordinary
turn, an observed active turn steers the proven chord into it. The
negatives are what make a reminder safe: a second operation id while
one is unresolved types zero bytes; ambiguity freezes without
resending; acceptance requires provider evidence echoing the frozen
marker; unproven chords and missing chord primitives refuse rather
than guessing at a live turn.
"""

from __future__ import annotations

import json
import os

import pytest

from cli_agent_orchestrator.services import execution_mode as em
from cli_agent_orchestrator.services import kimi_native_control as knc
from cli_agent_orchestrator.services import native_attachment as na

SESSION = "session_9f21ac30"
TERMINAL = "terminal_4d7b"
GENERATION = "gen_1c0e"
OCCURRENCE = "occ_b2c4d6e8"
#: A build with a proven steer pin in the adapter table.
PROVEN_VERSION = "0.33.0"
#: The installed build, which no pin in this candidate covers: active
#: turns must refuse (missing capability), never guess.
UNPROVEN_VERSION = "0.36.1"


@pytest.fixture(autouse=True)
def _db(isolated_memory_db):
    return isolated_memory_db


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def send_literal(self, text: str) -> None:
        self.calls.append(f"literal:{text}")

    def send_enter(self) -> None:
        self.calls.append("enter")

    def send_key(self, keystroke: str) -> None:
        self.calls.append(f"key:{keystroke}")

    def send_chord(self, chord: str) -> None:
        self.calls.append(f"chord:{chord}")


class NoChord(Recorder):
    send_chord = None  # type: ignore[assignment]


class FailsOnEnter(Recorder):
    def send_enter(self) -> None:
        self.calls.append("enter")
        raise OSError("pane vanished before submit")


def _attach(**over) -> dict:
    kw = dict(
        provider=knc.PROVIDER,
        native_session_id=over.get("session", SESSION),
        terminal_id=over.get("terminal_id", TERMINAL),
        generation=over.get("generation", GENERATION),
        execution_mode=em.NATIVE_TUI,
    )
    intent = na.acquire_intent(
        acquisition_method=na.ACQUISITION_ACP_BOOTSTRAP,
        acquisition_receipt={"kind": "kimi-acp-session-new", "session_id": kw["native_session_id"]},
        admits_only_new_instructions=True,
        replays_task_bytes=False,
        bootstrap_sent_no_turn=True,
        bootstrap_detached_before_launch=True,
    )
    na.declare(**kw, intent=intent, pane_id="%7")
    na.mark_starting(**kw, pane_id="%7")
    return na.mark_attached(
        **kw, pane_id="%7", process_identity=na.process_identity(pid=4242, start_marker="88213")
    )


def _idle() -> dict:
    return knc.turn_observation(
        active_turn_id=None, observed_at="2026-09-09T00:00:00Z", observer="status_monitor"
    )


def _remind(transport, **over) -> dict:
    kw = dict(
        operation_id="op_remind_1",
        native_session_id=SESSION,
        terminal_id=TERMINAL,
        generation=GENERATION,
        execution_mode=em.NATIVE_TUI,
        occurrence_id=OCCURRENCE,
        text="current goal: ship the report",
        marker="mk-1",
        observation=_idle(),
        transport=transport,
        turn_state="idle",
        provider_version="0.29.0",
    )
    kw.update(over)
    return knc.remind(**kw)


def _posted_row(operation_id: str) -> dict:
    return knc.get(operation_id)


def test_idle_submit_posts_with_marker_in_payload():
    _attach()
    transport = Recorder()
    record = _remind(transport)
    assert record["state"] == "posted"
    assert record["reminder_outcome"] == "posted"
    assert any("enter" == call for call in transport.calls)
    literals = " ".join(c for c in transport.calls if c.startswith("literal:"))
    assert "[cao-context-restoration marker:mk-1]" in literals
    assert "current goal: ship the report" in literals


def test_frozen_payload_matches_typed_bytes():
    _attach()
    transport = Recorder()
    record = _remind(transport, operation_id="op_freeze_1", marker="mk-f")
    frozen = record["transport"]["frozen_payload_sha256"]
    assert frozen == record["payload_sha256"]
    assert record["transport"]["marker"] == "mk-f"
    assert record["transport"]["branch"] == "submit"


def test_active_turn_steers_proven_chord_without_enter():
    _attach()
    transport = Recorder()
    chords = sorted(knc.steer_chords(PROVEN_VERSION))
    assert chords, "test needs a proven chord set"
    record = _remind(
        transport,
        operation_id="op_steer_1",
        turn_state="active",
        steer_chord=chords[0],
        provider_version=PROVEN_VERSION,
    )
    assert record["state"] == "posted"
    assert f"chord:{chords[0]}" in transport.calls
    assert "enter" not in transport.calls
    assert record["transport"]["branch"] == "steer"


def test_active_turn_without_chord_refuses_with_zero_bytes():
    _attach()
    transport = Recorder()
    record = _remind(transport, operation_id="op_nc_1", turn_state="active")
    assert record["state"] == "refused"
    assert record["refusal_reason"] == knc.REFUSED_UNPROVEN_STEER
    assert transport.calls == []


def test_active_turn_on_unproven_build_refuses_missing_capability():
    _attach()
    transport = Recorder()
    record = _remind(
        transport,
        operation_id="op_0361_1",
        turn_state="active",
        steer_chord="C-s",
        provider_version=UNPROVEN_VERSION,
    )
    assert record["state"] == "refused"
    assert record["refusal_reason"] == knc.REFUSED_UNPROVEN_COMPOSER_NEWLINE
    assert transport.calls == []


def test_active_turn_with_typing_but_no_chord_pin_refuses(monkeypatch):
    _attach()
    monkeypatch.setattr(knc, "steer_chords", lambda _v: frozenset())
    transport = Recorder()
    record = _remind(
        transport,
        operation_id="op_nopin_1",
        turn_state="active",
        steer_chord="C-s",
        provider_version="0.29.0",
    )
    assert record["state"] == "refused"
    assert record["refusal_reason"] == knc.REFUSED_UNPROVEN_STEER
    assert transport.calls == []


def test_active_turn_without_chord_primitive_refuses():
    _attach()
    transport = NoChord()
    chords = sorted(knc.steer_chords(PROVEN_VERSION))
    record = _remind(
        transport,
        operation_id="op_np_1",
        turn_state="active",
        steer_chord=chords[0],
        provider_version=PROVEN_VERSION,
    )
    assert record["state"] == "refused"
    assert record["refusal_reason"] == knc.REFUSED_UNSUPPORTED_CONTROL
    assert transport.calls == []


def test_second_id_while_posted_types_zero_bytes():
    _attach()
    first = _remind(Recorder(), operation_id="op_dup_1")
    assert first["reminder_outcome"] == "posted"
    transport = Recorder()
    second = _remind(transport, operation_id="op_dup_2")
    assert second["reminder_outcome"] == "already-pending"
    assert transport.calls == []


def test_same_id_replay_adopts_without_new_bytes():
    _attach()
    first = _remind(Recorder(), operation_id="op_ad_1")
    transport = Recorder()
    second = _remind(transport, operation_id="op_ad_1")
    assert second["reminder_outcome"] == "adopted"
    assert transport.calls == []


def test_ambiguity_freezes_and_second_id_backs_off():
    _attach()
    broken = _remind(FailsOnEnter(), operation_id="op_amb_1")
    assert broken["reminder_outcome"] == "ambiguous"
    assert knc.get("op_amb_1")["state"] == "ambiguous"
    transport = Recorder()
    second = _remind(transport, operation_id="op_amb_2")
    assert second["reminder_outcome"] == "already-pending"
    assert transport.calls == []


def test_occurrence_mismatch_is_a_conflict_not_an_adopt():
    _attach()
    _remind(Recorder(), operation_id="op_occ_1", occurrence_id="occ-aaa")
    with pytest.raises(knc.NativeControlConflict):
        _remind(Recorder(), operation_id="op_occ_1", occurrence_id="occ-bbb")


def test_idle_claim_naming_an_active_turn_is_invalid():
    _attach()
    busy = knc.turn_observation(
        active_turn_id="turn_live", observed_at="2026-09-09T00:00:00Z", observer="status_monitor"
    )
    with pytest.raises(knc.NativeControlInvalid):
        _remind(Recorder(), operation_id="op_bad_1", observation=busy, turn_state="idle")


def test_acceptance_requires_marker_echo():
    _attach()
    _remind(Recorder(), operation_id="op_mk_1", marker="mk-echo")
    evidence = knc.provider_observation(
        operation_id="op_mk_1",
        observed_at="2026-09-09T00:00:02Z",
        observer="status_monitor",
        entered_turn_id="turn_9",
        evidence={"marker_echo": "wrong"},
    )
    with pytest.raises(knc.NativeControlInvalid):
        knc.record_reminder_acceptance(
            operation_id="op_mk_1", observation=evidence, expected_marker="mk-echo"
        )
    assert knc.get("op_mk_1")["state"] == "posted"
    good = knc.provider_observation(
        operation_id="op_mk_1",
        observed_at="2026-09-09T00:00:03Z",
        observer="status_monitor",
        entered_turn_id="turn_9",
        evidence={"marker_echo": "mk-echo"},
    )
    done = knc.record_reminder_acceptance(
        operation_id="op_mk_1", observation=good, expected_marker="mk-echo"
    )
    assert done["state"] == "completed"


def test_pre_write_refusal_types_zero_bytes():
    _attach()
    transport = Recorder()
    record = _remind(
        transport,
        operation_id="op_pw_1",
        pre_write=lambda: (knc.REFUSED_WAIT_COVER, "a wait landed first"),
    )
    assert record["state"] == "refused"
    assert record["reminder_outcome"] == "refused"
    assert transport.calls == []


def _wire_home(tmp_path, marker, *, accept_only=False):
    """A fake Kimi home whose session wire carries one marker."""
    main = tmp_path / "kh" / "sessions" / "wd_x_1" / "session_abc" / "agents" / "main"
    main.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "type": "turn.prompt",
                "agentId": "main",
                "input": [
                    {
                        "type": "text",
                        "text": f"goal: x\n\n[cao-context-restoration marker:{marker}]",
                    }
                ],
                "origin": {"kind": "user"},
                "time": 1,
            }
        )
    ]
    if not accept_only:
        lines.append(
            json.dumps(
                {
                    "type": "context.append_message",
                    "agentId": "main",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"goal: x\n\n[cao-context-restoration marker:{marker}]",
                            }
                        ],
                        "origin": {"kind": "user"},
                    },
                    "time": 2,
                }
            )
        )
    (main / "wire.jsonl").write_text("\n".join(lines) + "\n")
    return str(tmp_path / "kh")


def test_wire_scan_finds_model_context_entry_and_accept(tmp_path):
    home = _wire_home(tmp_path, "op_scan_1")
    scan = knc.scan_wire_for_marker(session_home=home, marker="op_scan_1")
    assert scan["files_scanned"] == 1
    assert scan["model_context_entry"]["wire_type"] == "context.append_message"
    assert scan["prompt_accepted"]["wire_type"] == "turn.prompt"


def test_wire_scan_missing_home_is_no_evidence(tmp_path):
    scan = knc.scan_wire_for_marker(session_home=str(tmp_path / "absent"), marker="op_scan_9")
    assert scan == {"model_context_entry": None, "prompt_accepted": None, "files_scanned": 0}


def test_reconcile_completes_posted_row_on_marker_echo(tmp_path):
    _attach()
    _remind(Recorder(), operation_id="op_wire_1", marker="op_wire_1")
    home = _wire_home(tmp_path, "op_wire_1")
    result = knc.reconcile_reminder_from_wire(
        operation_id="op_wire_1", marker="op_wire_1", session_home=home
    )
    assert result["reconciled"] is True
    assert result["record"]["state"] == "completed"


def test_reconcile_leaves_posted_row_without_evidence(tmp_path):
    _attach()
    _remind(Recorder(), operation_id="op_wire_2", marker="op_wire_2")
    result = knc.reconcile_reminder_from_wire(
        operation_id="op_wire_2", marker="op_wire_2", session_home=str(tmp_path / "empty")
    )
    assert result["reconciled"] is False
    assert result["reason"] == "no-wire-evidence"
    assert knc.get("op_wire_2")["state"] == "posted"


def test_reconcile_clears_ambiguous_reminder_on_echo(tmp_path):
    _attach()
    _remind(Recorder(), operation_id="op_wire_3", marker="op_wire_3")
    knc.mark_ambiguous(operation_id="op_wire_3", reason="submit raised")
    home = _wire_home(tmp_path, "op_wire_3")
    result = knc.reconcile_reminder_from_wire(
        operation_id="op_wire_3", marker="op_wire_3", session_home=home
    )
    assert result["reconciled"] is True
    assert result["record"]["state"] == "completed"


def test_reconcile_never_moves_completed_row(tmp_path):
    _attach()
    _remind(Recorder(), operation_id="op_wire_4", marker="op_wire_4")
    home = _wire_home(tmp_path, "op_wire_4")
    first = knc.reconcile_reminder_from_wire(
        operation_id="op_wire_4", marker="op_wire_4", session_home=home
    )
    assert first["record"]["state"] == "completed"
    second = knc.reconcile_reminder_from_wire(
        operation_id="op_wire_4", marker="op_wire_4", session_home=home
    )
    assert second["reconciled"] is False
    assert second["reason"] == "already-completed"


def test_ambiguous_remind_does_not_block_other_kinds():
    _attach()
    _remind(Recorder(), operation_id="op_amb_r", marker="op_amb_r")
    knc.mark_ambiguous(operation_id="op_amb_r", reason="submit raised")
    # The production gate every lane calls: an ambiguous reminder is
    # inert labeled context, so other kinds proceed.
    knc._assert_session_unblocked(native_session_id=SESSION, operation_id="op_queue_other")
    assert knc.unresolved_ambiguity(SESSION) is None


def test_ambiguous_steer_still_blocks_everything():
    _attach()
    from cli_agent_orchestrator.clients import database

    with database.SessionLocal() as db:
        db.add(
            database.KimiNativeControlOperationModel(
                operation_id="op_amb_s",
                kind="steer",
                state="ambiguous",
                provider=knc.PROVIDER,
                native_session_id=SESSION,
                terminal_id=TERMINAL,
                generation=GENERATION,
                execution_mode=em.NATIVE_TUI,
                payload_sha256="x",
                intent_json="{}",
                epoch=0,
                created_at="t",
                updated_at="t",
            )
        )
        db.commit()
    with pytest.raises(Exception) as excinfo:
        knc._assert_session_unblocked(native_session_id=SESSION, operation_id="op_queue_blocked")
    assert getattr(excinfo.value, "reason", "") == knc.REFUSED_UNRESOLVED_AMBIGUITY


def test_refuse_reminder_journals_typed_zero_byte_refusal():
    _attach()
    record = knc.refuse_reminder(
        operation_id="op_ref_1",
        native_session_id=SESSION,
        terminal_id=TERMINAL,
        generation=GENERATION,
        execution_mode=em.NATIVE_TUI,
        occurrence_id=OCCURRENCE,
        text="current goal: ship the report",
        marker="op_ref_1",
        observation=_idle(),
        reason=knc.REFUSED_WAIT_COVER,
        detail="a wait landed first",
    )
    assert record["state"] == "refused"
    assert record["refusal_reason"] == knc.REFUSED_WAIT_COVER
    # A replayed id returns its row, never a conflict.
    again = knc.refuse_reminder(
        operation_id="op_ref_1",
        native_session_id=SESSION,
        terminal_id=TERMINAL,
        generation=GENERATION,
        execution_mode=em.NATIVE_TUI,
        occurrence_id=OCCURRENCE,
        text="current goal: ship the report",
        marker="op_ref_1",
        observation=_idle(),
        reason=knc.REFUSED_LIFECYCLE,
        detail="changed mind too late",
    )
    assert again["state"] == "refused"
    assert again["refusal_reason"] == knc.REFUSED_WAIT_COVER


def _stub_viewport(monkeypatch, rows=None, exc=None):
    import cli_agent_orchestrator.services.native_pane_input as pane

    def fake(pane_id, timeout=None):
        if exc is not None:
            raise exc
        return list(rows or [])

    monkeypatch.setattr(pane, "capture_pane_screen", fake)


def test_composer_holding_marker_goes_ambiguous_never_erases(tmp_path, monkeypatch):
    _attach()
    _remind(Recorder(), operation_id="op_cmp_1", marker="op_cmp_1")
    _stub_viewport(
        monkeypatch, rows=["k> partial bytes here", "[cao-context-restoration marker:op_cmp_1]"]
    )
    result = knc.reconcile_reminder_composer(
        operation_id="op_cmp_1", marker="op_cmp_1", pane_id="%1", session_home=str(tmp_path)
    )
    assert result["reconciled"] is True
    assert result["reason"] == "ambiguous-partial-composer"
    assert result["composer_holds_marker"] is True
    assert result["record"]["state"] == "ambiguous"


def test_composer_clear_with_wire_echo_completes(tmp_path, monkeypatch):
    _attach()
    _remind(Recorder(), operation_id="op_cmp_2", marker="op_cmp_2")
    home = _wire_home(tmp_path, "op_cmp_2")
    _stub_viewport(monkeypatch, rows=["k> clean composer"])
    result = knc.reconcile_reminder_composer(
        operation_id="op_cmp_2", marker="op_cmp_2", pane_id="%1", session_home=home
    )
    assert result["reconciled"] is True
    assert result["record"]["state"] == "completed"
    assert result["composer_holds_marker"] is False


def test_composer_clear_without_echo_goes_ambiguous(tmp_path, monkeypatch):
    _attach()
    _remind(Recorder(), operation_id="op_cmp_3", marker="op_cmp_3")
    _stub_viewport(monkeypatch, rows=["k> user typed something else"])
    result = knc.reconcile_reminder_composer(
        operation_id="op_cmp_3",
        marker="op_cmp_3",
        pane_id="%1",
        session_home=str(tmp_path / "empty"),
    )
    assert result["reconciled"] is True
    assert result["reason"] == "ambiguous-unproven-clear"
    assert result["composer_holds_marker"] is False
    assert result["record"]["state"] == "ambiguous"


def test_composer_unreadable_leaves_row_untouched(tmp_path, monkeypatch):
    _attach()
    _remind(Recorder(), operation_id="op_cmp_4", marker="op_cmp_4")
    _stub_viewport(monkeypatch, exc=OSError("tmux down"))
    result = knc.reconcile_reminder_composer(
        operation_id="op_cmp_4", marker="op_cmp_4", pane_id="%1", session_home=str(tmp_path)
    )
    assert result["reconciled"] is False
    assert result["reason"] == "composer-unreadable"
    assert result["composer_holds_marker"] is None
    assert knc.get("op_cmp_4")["state"] == "posted"


def test_loser_id_adopts_without_new_row_or_bytes():
    _attach()
    transport = Recorder()
    first = _remind(transport, operation_id="op_win_1", marker="op_win_1")
    assert first["reminder_outcome"] == "posted"
    before = list(transport.calls)
    second = _remind(transport, operation_id="op_lose_2", marker="op_lose_2")
    assert second["reminder_outcome"] == "already-pending"
    assert second["operation_id"] == "op_win_1"
    assert transport.calls == before
    assert knc.get("op_lose_2") is None
    assert not hasattr(knc, "REFUSED_REMINDER_PENDING")


def test_posted_row_freezes_origin_and_hook_evidence():
    # Audit evidence frozen at POSTED: origin + native invocation
    # fields, retrievable for diagnosis. Identity is the request id.
    _attach()
    transport = Recorder()
    evidence = {
        "session_id": SESSION,
        "trigger": "auto",
        "estimated_token_count": 77,
        "observed_at": "2026-09-09T00:00:00Z",
    }
    record = _remind(
        transport,
        operation_id="op_origin_1",
        marker="op_origin_1",
        origin="event",
        hook_evidence=evidence,
    )
    assert record["reminder_outcome"] == "posted"
    row = knc.get("op_origin_1")
    assert row["transport"]["origin"] == "event"
    assert row["transport"]["hook_evidence"] == evidence
    assert row["transport"]["marker"] == "op_origin_1"


def test_unknown_origin_is_rejected():
    _attach()
    transport = Recorder()
    with pytest.raises(knc.NativeControlInvalid):
        _remind(transport, operation_id="op_origin_x", origin="sidecar")


def test_terminal_receipt_query_serves_health_metadata():
    # The conductor's pending-receipts read: unresolved rows plus the
    # newest terminal one, with origin/evidence for health — and no
    # reminder text anywhere.
    _attach()
    transport = Recorder()
    evidence = {
        "session_id": SESSION,
        "trigger": "auto",
        "estimated_token_count": 5,
        "observed_at": "2026-09-09T00:00:00Z",
    }
    _remind(
        transport,
        operation_id="op_health_1",
        marker="op_health_1",
        origin="event",
        hook_evidence=evidence,
    )
    open_rows = knc.unresolved_reminders_for(terminal_id=TERMINAL, generation=GENERATION)
    assert [r["operation_id"] for r in open_rows] == ["op_health_1"]
    assert open_rows[0]["transport"]["origin"] == "event"
    assert knc.latest_terminal_reminder_for(terminal_id=TERMINAL, generation=GENERATION) is None
    knc.record_reminder_acceptance(
        operation_id="op_health_1",
        observation=knc.provider_observation(
            operation_id="op_health_1",
            observed_at="2026-09-09T00:00:01Z",
            observer="test",
            evidence={
                "marker_echo": "op_health_1",
                "wire_path": "w",
                "wire_time": 1,
                "wire_type": "context.append_message",
            },
        ),
        expected_marker="op_health_1",
    )
    assert knc.unresolved_reminders_for(terminal_id=TERMINAL, generation=GENERATION) == []
    last = knc.latest_terminal_reminder_for(terminal_id=TERMINAL, generation=GENERATION)
    assert last["operation_id"] == "op_health_1"
    assert last["state"] == "completed"
    assert last["transport"]["hook_evidence"] == evidence
    assert "current goal" not in json.dumps(last)
