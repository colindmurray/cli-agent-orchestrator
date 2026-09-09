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
        **kw, pane_id="%7",
        process_identity=na.process_identity(pid=4242, start_marker="88213"))


def _idle() -> dict:
    return knc.turn_observation(
        active_turn_id=None, observed_at="2026-09-09T00:00:00Z",
        observer="status_monitor")


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
        transport, operation_id="op_steer_1", turn_state="active",
        steer_chord=chords[0], provider_version=PROVEN_VERSION)
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
        transport, operation_id="op_0361_1", turn_state="active",
        steer_chord="C-s", provider_version=UNPROVEN_VERSION)
    assert record["state"] == "refused"
    assert record["refusal_reason"] == knc.REFUSED_UNPROVEN_COMPOSER_NEWLINE
    assert transport.calls == []


def test_active_turn_with_typing_but_no_chord_pin_refuses(monkeypatch):
    _attach()
    monkeypatch.setattr(knc, "steer_chords", lambda _v: frozenset())
    transport = Recorder()
    record = _remind(
        transport, operation_id="op_nopin_1", turn_state="active",
        steer_chord="C-s", provider_version="0.29.0")
    assert record["state"] == "refused"
    assert record["refusal_reason"] == knc.REFUSED_UNPROVEN_STEER
    assert transport.calls == []


def test_active_turn_without_chord_primitive_refuses():
    _attach()
    transport = NoChord()
    chords = sorted(knc.steer_chords(PROVEN_VERSION))
    record = _remind(
        transport, operation_id="op_np_1", turn_state="active",
        steer_chord=chords[0], provider_version=PROVEN_VERSION)
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
        active_turn_id="turn_live", observed_at="2026-09-09T00:00:00Z",
        observer="status_monitor")
    with pytest.raises(knc.NativeControlInvalid):
        _remind(Recorder(), operation_id="op_bad_1", observation=busy,
                turn_state="idle")


def test_acceptance_requires_marker_echo():
    _attach()
    _remind(Recorder(), operation_id="op_mk_1", marker="mk-echo")
    evidence = knc.provider_observation(
        operation_id="op_mk_1", observed_at="2026-09-09T00:00:02Z",
        observer="status_monitor", entered_turn_id="turn_9",
        evidence={"marker_echo": "wrong"})
    with pytest.raises(knc.NativeControlInvalid):
        knc.record_reminder_acceptance(
            operation_id="op_mk_1", observation=evidence,
            expected_marker="mk-echo")
    assert knc.get("op_mk_1")["state"] == "posted"
    good = knc.provider_observation(
        operation_id="op_mk_1", observed_at="2026-09-09T00:00:03Z",
        observer="status_monitor", entered_turn_id="turn_9",
        evidence={"marker_echo": "mk-echo"})
    done = knc.record_reminder_acceptance(
        operation_id="op_mk_1", observation=good, expected_marker="mk-echo")
    assert done["state"] == "completed"


def test_pre_write_refusal_types_zero_bytes():
    _attach()
    transport = Recorder()
    record = _remind(
        transport, operation_id="op_pw_1",
        pre_write=lambda: (knc.REFUSED_WAIT_COVER, "a wait landed first"))
    assert record["state"] == "refused"
    assert record["reminder_outcome"] == "refused"
    assert transport.calls == []
