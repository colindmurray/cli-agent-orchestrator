"""Dark COND-0230 M10-C Codex route-observation adapter (capability only).

``CodexRouteObserver`` drives the merged ``route_observation`` stage machine
end to end for one identity-bound ``/status`` control against a Codex native
TUI pane.  The M10 capability stays dark: ``enabled()`` here delegates to the
stage machine's gate, which is always ``False``, and nothing in this slice
observes a live provider surface, issues pane input against a live pane, or
delivers a wake.

The stage machine is the authority.  There is no adapter ABC; the contract is
the ordered stage-call sequence with first-CAS-wins authorizations and the
atomic ``complete(...)``, and this orchestrator follows it exactly — one
operation id, the exact target/requester binding, and never two effects
journaled in one stage.

The Codex ``/status`` surface is **non-modal**: the panel is printed output in
the transcript and the composer stays rendered at the bottom, so there is no
modal to dismiss and no ``Escape`` to issue.  The pre-close intent and close
proof therefore encode the ambiguous-close handling honestly: the close action
is ``"none"`` and the close proof outcome is ``composer-restored``,
``not-restored``, or ``indeterminate`` — an unprovable close is never
fabricated into a second ``Escape``, and the terminal result is
``ambiguous-after-possible-effect`` when the close is unproven.

Observed state is asserted only when the pane is wide enough to have rendered
it.  The installed build's status panel obeys two render floors — the
``Session:`` row renders only at >= 76 columns and the ``Model:`` row (which
carries the reasoning effort) only at >= 87 columns.  A row below its floor is
``not-rendered``, never guessed at, and the observation records the floor
facts so a later reader can distinguish "the pane is too narrow" from "the
session is missing".

The wake path is the stage machine's: ``complete()`` writes the deterministic
exact-requester inbox claim atomically.  This adapter does not invent a
transport; it sequences the stages, records honest observations, and replays a
stored result under a lost response without a second ``/status``, second
close, or second wake.  The requester generation is revalidated immediately
before provider input; a drifted requester is recorded as the normative
``requester-stale`` disposition with zero input.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, Sequence

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.clients.tmux import tmux_binary
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import native_pane_input as npi
from cli_agent_orchestrator.services import native_status_repair as nsr
from cli_agent_orchestrator.services import route_observation as ro

#: The pinned Codex build this adapter's status panel was proven against.
CODEX_PINNED_VERSION = "0.147.0"
#: The provider key used for the pinned cond-0026 submission barrier.
CODE_PROVIDER = "codex"
#: The pinned status-panel parser identity (shared with native_status_repair).
PARSER_KEY = nsr.PARSER_CODEX_STATUS

#: The exact command typed into the pane, at most once, with at most one Enter.
STATUS_COMMAND = "/status"

#: Outcome schema for the orchestrator's own result envelope.
OBSERVER_SCHEMA = "cao-codex-route-observation-v1"

#: Render floors of the installed Codex status panel.  A row whose floor the
#: current pane width does not meet is ``not-rendered`` and is never asserted.
MODEL_RENDER_FLOOR_COLUMNS = 87
SESSION_RENDER_FLOOR_COLUMNS = 76

#: Wake delivery dispositions.  ``requester-stale`` is the normative
#: zero-input disposition when the exact requester generation drifted.
DISPOSITION_DELIVERED = "delivered"
DISPOSITION_REQUESTER_STALE = "requester-stale"
DISPOSITION_REPLAYED = "replayed"
DISPOSITION_PROVIDER_NOT_READY = "provider-not-ready"
DISPOSITION_PANE_UNREADABLE = "pane-unreadable"

PREWRITE_READY = "ready"
PREWRITE_PROVIDER_NOT_READY = "provider-not-ready"
PREWRITE_PANE_UNREADABLE = "pane-unreadable"
_PREWRITE_READINESS_POLL_SECONDS = 0.1
# Codex can briefly redraw an idle composer between asynchronous MCP-startup
# updates.  Restart-5 observed MCP work about 0.4 seconds after the first ready
# frame, so ten full poll gaps hold readiness for at least one second before
# authorizing input rather than merely sampling the same transient redraw.
_PREWRITE_READY_STABLE_POLLS = 11

#: Close-proof outcome vocabulary for the non-modal surface.  A proven
#: composer return is the only positive close; everything else is unproven
#: and terminates ambiguous-after-possible-effect.
CLOSE_COMPOSER_RESTORED = "composer-restored"
CLOSE_NOT_RESTORED = "not-restored"
CLOSE_INDETERMINATE = "indeterminate"

#: The exact reasoning-effort vocabulary the installed build accepts for the
#: ``model_reasoning_effort`` route.  A token outside this set is malformed
#: evidence and is refused, never guessed.
_CODEX_EFFORT_VOCABULARY = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "ultra"})

#: The reasoning effort starts the parenthetical detail.  Any comma-delimited
#: display annotations after it are non-authoritative and intentionally ignored.
_REASONING_SUFFIX = re.compile(r"^(.*?)\s+\(reasoning\s+([^\s(),]+)(?:\s*,[^()]*)?\)$")
# A reasoning parenthetical with no effort token is malformed authoritative
# structure, not a model name with an incidental parenthetical.
_EMPTY_REASONING_SUFFIX = re.compile(r"^(.*?)\s+\(reasoning\s*\)$")

#: A Codex composer prompt row (the live composer, not the printed panel).
_CODEX_COMPOSER_PROMPT = re.compile(r"^\s*(?:›|❯|codex>)(?:\s|$)")


@dataclass(frozen=True)
class PrewriteReadiness:
    """The exact-pane observation made before the probe intent exists."""

    reason: str
    provider_status: Optional[str]
    detail: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.reason == PREWRITE_READY

    def fact(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "provider_status": self.provider_status,
            "detail": self.detail,
        }


class CodexPaneSurface(Protocol):
    """The pane transport the orchestrator needs (fake in tests, real via npi).

    ``send_status_command`` writes literal ``/status`` and at most one Enter
    and returns whether the composer provably gave the control up; a raise
    means the write was refused and is a possible effect.  ``pane_width``
    reports the current column width or None (render floor unknown).
    ``composer_restored`` returns True/False when the close is provable and
    None when it is not.
    """

    pane_id: str

    def capture_screen(self) -> list[str]: ...
    def pane_width(self) -> Optional[int]: ...
    def await_input_ready(self) -> PrewriteReadiness: ...
    def send_status_command(self) -> bool: ...
    def composer_restored(self) -> Optional[bool]: ...


def enabled() -> bool:
    """Whether this build may exercise the M10 Codex route-observation adapter.

    Delegates to the stage machine's dark gate; the adapter itself never flips
    it.  Always ``False`` in this slice.
    """
    return ro.enabled()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# the pane surface (fake in tests; the pinned transport below)
# ---------------------------------------------------------------------------


def _pane_width(pane_id: str, *, timeout: float) -> Optional[int]:
    """The pane's current column width, or None when it cannot be read.

    None is the conservative "render floor unknown" answer: the observer then
    asserts only the rows the capture literally renders, never a value a
    narrow pane could have hidden.
    """
    try:
        result = subprocess.run(
            [tmux_binary(), "display", "-p", "-t", pane_id, "#{pane_width}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        width = int((result.stdout or "").strip())
    except ValueError:
        return None
    return width if width > 0 else None


def _composer_prompt_present(rows: Sequence[str]) -> bool:
    normalized = nsr.normalize_capture_rows(rows)
    return any(_CODEX_COMPOSER_PROMPT.match(row) for row in normalized)


class RealCodexPaneSurface:
    """The live Codex pane transport: ``npi`` captures, ``TmuxPaneInput`` writes.

    Exercised against REAL panes by the M17 T2 synthetic-live suite
    (``test/e2e/test_m17_t2_synthlive.py``); the fake surface drives the
    observer in the unit mirrors.  This is the pinned delivery seam: the
    literal ``/status`` write and its single Enter through ``TmuxPaneInput``,
    the cond-0026 submission barrier, the escape-free capture, and the Codex
    turn-state observer for the composer-restored close proof.
    """

    def __init__(
        self,
        pane_id: str,
        *,
        terminal_id: Optional[str] = None,
        session_name: Optional[str] = None,
        window_name: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.pane_id = pane_id
        self._terminal_id = terminal_id
        self._session_name = session_name
        self._window_name = window_name
        self._timeout = timeout

    def capture_screen(self) -> list[str]:
        return list(npi.capture_pane_screen(self.pane_id, timeout=self._timeout))

    def pane_width(self) -> Optional[int]:
        return _pane_width(self.pane_id, timeout=self._timeout)

    def await_input_ready(self) -> PrewriteReadiness:
        """Wait boundedly for this exact Codex pane to remain ready.

        This is a read-only last-safe-point gate.  The caller invokes it before
        committing the pre-probe intent, so a timeout can truthfully close as a
        zero-effect refusal.  Observed busy and unreadable remain distinct, and
        either resets the ready streak.
        """
        if self._terminal_id is None or self._session_name is None or self._window_name is None:
            return PrewriteReadiness(
                PREWRITE_PANE_UNREADABLE,
                None,
                "the exact terminal, session, and window identity is required",
            )
        deadline = time.monotonic() + self._timeout
        latest = PrewriteReadiness(
            PREWRITE_PANE_UNREADABLE,
            None,
            "the bound Codex pane has not been read",
        )
        consecutive_ready = 0
        while True:
            try:
                status = npi.observe_codex_turn_state(
                    self.pane_id,
                    terminal_id=self._terminal_id,
                    session_name=self._session_name,
                    window_name=self._window_name,
                    timeout=min(self._timeout, max(0.2, deadline - time.monotonic())),
                )
            except Exception as exc:  # noqa: BLE001 - typed unreadable, not observed busy
                consecutive_ready = 0
                latest = PrewriteReadiness(PREWRITE_PANE_UNREADABLE, None, str(exc))
            else:
                # Both states render the writable composer.  COMPLETED is the
                # ordinary settled state of a resumed thread with prior turns;
                # requiring only IDLE would refuse that healthy session forever.
                if status in {TerminalStatus.IDLE, TerminalStatus.COMPLETED}:
                    consecutive_ready += 1
                    if consecutive_ready >= _PREWRITE_READY_STABLE_POLLS:
                        return PrewriteReadiness(PREWRITE_READY, status.value)
                    latest = PrewriteReadiness(
                        PREWRITE_PROVIDER_NOT_READY,
                        status.value,
                        "the bound Codex pane did not remain ready for "
                        f"{_PREWRITE_READY_STABLE_POLLS} consecutive observations",
                    )
                else:
                    consecutive_ready = 0
                    latest = PrewriteReadiness(PREWRITE_PROVIDER_NOT_READY, status.value)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return latest
            time.sleep(min(_PREWRITE_READINESS_POLL_SECONDS, remaining))

    def send_status_command(self) -> bool:
        """Type literal ``/status`` and at most one Enter; return whether the
        composer provably gave the control up (submission proven)."""
        typed = npi.TmuxPaneInput(self.pane_id, timeout=self._timeout)
        barrier = npi.submission_barrier_for(CODE_PROVIDER)
        typed.send_literal(STATUS_COMMAND)
        if barrier is None:
            typed.send_enter()
            return True
        if not npi.await_compose_visible(self.pane_id, STATUS_COMMAND, barrier=barrier):
            return False
        typed.send_enter()
        observed, _ = npi.observe_submission(self.pane_id, STATUS_COMMAND, barrier=barrier)
        return observed == npi.SUBMISSION_SUBMITTED

    def composer_restored(self) -> Optional[bool]:
        if self._terminal_id is None or self._session_name is None or self._window_name is None:
            try:
                return _composer_prompt_present(self.capture_screen())
            except Exception:  # noqa: BLE001 - an unreadable pane is an unproven close
                return None
        try:
            status = npi.observe_codex_turn_state(
                self.pane_id,
                terminal_id=self._terminal_id,
                session_name=self._session_name,
                window_name=self._window_name,
                timeout=self._timeout,
            )
        except Exception:  # noqa: BLE001 - an unreadable pane is an unproven close
            return None
        return status == TerminalStatus.IDLE


# ---------------------------------------------------------------------------
# the pinned status-panel observation parse
# ---------------------------------------------------------------------------


def _render_floor(pane_width: Optional[int]) -> dict[str, Any]:
    """Which rows a pane of this width can have rendered.

    The Model value is never asserted from row presence alone: the Model row
    (which carries the reasoning effort) renders only at or above the 87-column
    floor, so an unknown/stale width proves nothing about a truncated Model
    value and the model is not asserted.  The Session row keeps its literal
    capture because a truncated session fails the canonical-UUID validation.
    """
    if pane_width is None:
        return {"width": None, "session_assertable": True, "model_assertable": False}
    return {
        "width": pane_width,
        "session_assertable": pane_width >= SESSION_RENDER_FLOOR_COLUMNS,
        "model_assertable": pane_width >= MODEL_RENDER_FLOOR_COLUMNS,
    }


def _parse_model_row(normalized: Sequence[str]) -> tuple[Optional[str], Optional[str]]:
    """``(model, effort)`` from the exact ``Model:`` row, or ``(None, None)``.

    An optional exact `` (reasoning <effort>)`` suffix is split off into the
    effort; any other complete parenthetical stays part of the model.  A second
    Model row, a suffix carrying an unknown effort, or a value cut off
    mid-parenthetical at the pane edge is refused rather than guessed: a
    truncated value is a truncated capture, never a bare model.  Once the
    effort token is validated, any comma-delimited display annotation is
    ignored completely.
    """
    model_rows = [row for row in normalized if row.lstrip().startswith("Model:")]
    if not model_rows:
        return None, None
    if len(model_rows) > 1:
        raise nsr.PanelParseError("the Codex status panel renders more than one 'Model:' row")
    value = model_rows[0].split(":", 1)[1].strip()
    match = _REASONING_SUFFIX.fullmatch(value)
    if match is None:
        if _EMPTY_REASONING_SUFFIX.fullmatch(value):
            raise nsr.PanelParseError(
                "the Codex status Model row has an empty reasoning effort; refusing"
            )
        if value.count("(") > value.count(")"):
            raise nsr.PanelParseError(
                "the Codex status Model row is truncated mid-parenthetical; refusing "
                "to report a half-rendered value as the model"
            )
        return value, None
    model, effort = match.group(1).strip(), match.group(2).strip()
    if not effort or effort not in _CODEX_EFFORT_VOCABULARY:
        raise nsr.PanelParseError(
            f"the Codex status Model row carries a malformed reasoning suffix; refusing "
            "rather than guessing an effort from arbitrary parenthetical text",
        )
    return model, effort


def _evidence_rows(normalized: Sequence[str]) -> list[str]:
    """Canonicalize non-authoritative Model-row decoration out of evidence.

    The existing evidence digest remains useful for the authoritative panel
    facts, but changing ignored display text must not change its input.  Only a
    structurally complete reasoning suffix is canonicalized; malformed or
    truncated rows stay untouched and therefore retain refusal evidence.
    """
    rows = list(normalized)
    model_rows = [index for index, row in enumerate(rows) if row.lstrip().startswith("Model:")]
    if len(model_rows) != 1:
        return rows
    index = model_rows[0]
    value = rows[index].split(":", 1)[1].strip()
    match = _REASONING_SUFFIX.fullmatch(value)
    if match is not None:
        model, effort = match.group(1).strip(), match.group(2).strip()
        rows[index] = f"Model: {model} (reasoning {effort})"
    return rows


def parse_codex_route_panel(
    rows: Sequence[str],
    *,
    pinned_version: str = CODEX_PINNED_VERSION,
    pane_width: Optional[int] = None,
) -> dict[str, Any]:
    """Parse one Codex ``/status`` capture into typed observation facts.

    The panel is the pinned ``>_ OpenAI Codex (v<pinned_version>)`` header, a
    ``Session:`` row, and a ``Model:`` row.  A row below its render floor is
    ``not-rendered`` and is never asserted; a full-width panel that still
    lacks the Model row is a truncated/different panel, not an observation.
    The session identity is taken from the same branded parser
    (``codex-status-v1``) the identity repair uses, so there is exactly one
    description of what a Codex status panel means.  Only the session ID,
    model, and reasoning effort become observation facts.

    Returns ``kind`` in {``observed``, ``partial``, ``inconclusive``}.
    """
    floor = _render_floor(pane_width)
    normalized = nsr.normalize_capture_rows(rows)
    evidence = nsr.evidence_digest(_evidence_rows(normalized))
    if not floor["session_assertable"]:
        return {
            "kind": "inconclusive",
            "reason": "render-floor-session",
            "pane_width": floor["width"],
            "evidence_sha256": evidence,
        }
    try:
        status = nsr.parse_codex_status(rows, pinned_version=pinned_version)
    except nsr.PanelParseError:
        return {
            "kind": "inconclusive",
            "reason": "panel-unparsed",
            "pane_width": floor["width"],
            "evidence_sha256": evidence,
        }
    if not floor["model_assertable"]:
        return {
            "kind": "partial",
            "reason": "render-floor-model",
            "session_id": status["session_id"],
            "provider_version": status["provider_version"],
            "parser_key": status["parser_key"],
            "model": None,
            "effort": None,
            "pane_width": floor["width"],
            "evidence_sha256": evidence,
        }
    try:
        model, effort = _parse_model_row(normalized)
    except nsr.PanelParseError:
        return {
            "kind": "inconclusive",
            "reason": "model-row-unparsed",
            "session_id": status["session_id"],
            "provider_version": status["provider_version"],
            "parser_key": status["parser_key"],
            "pane_width": floor["width"],
            "evidence_sha256": evidence,
        }
    if model is None:
        return {
            "kind": "inconclusive",
            "reason": "model-row-absent",
            "session_id": status["session_id"],
            "provider_version": status["provider_version"],
            "parser_key": status["parser_key"],
            "pane_width": floor["width"],
            "evidence_sha256": evidence,
        }
    return {
        "kind": "observed",
        "session_id": status["session_id"],
        "provider_version": status["provider_version"],
        "parser_key": status["parser_key"],
        "model": model,
        "effort": effort,
        "pane_width": floor["width"],
        "evidence_sha256": evidence,
    }


# ---------------------------------------------------------------------------
# bounded stage-fact builders (never raw pane text)
# ---------------------------------------------------------------------------


def _pre_probe_intent(request: ro.RouteObservationRequest, *, pane_id: str) -> dict[str, Any]:
    return {
        "kind": "pre-probe-intent",
        "command": STATUS_COMMAND,
        "surface": PARSER_KEY,
        "pane_id": pane_id,
        "provider_version": request.provider_version,
        "native_session_id": request.native_session_id,
    }


def _pre_close_intent() -> dict[str, Any]:
    return {
        "kind": "pre-close-intent",
        "modal": False,
        "close": "none",
        "verify": "composer-restored",
    }


def _close_proof(outcome: str) -> dict[str, Any]:
    return {
        "kind": "owned-close",
        "surface": "non-modal",
        "close_action": "none",
        "outcome": outcome,
        "closed_at": _now(),
    }


def _inconclusive_observation(
    request: ro.RouteObservationRequest,
    *,
    reason: str,
    evidence_sha256: Optional[str] = None,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "kind": "provider-surface",
        "observation_kind": PARSER_KEY,
        "observed_state": "inconclusive",
        "reason": reason,
        "observed_at": _now(),
        "provider_version": request.provider_version,
        "parser_key": PARSER_KEY,
        "session_id": None,
        "correlated": False,
        "model": None,
        "effort": None,
        "render_floor": None,
        "evidence_sha256": evidence_sha256,
    }
    return observation


def _observation_from_parse(
    request: ro.RouteObservationRequest, parsed: dict[str, Any]
) -> dict[str, Any]:
    """The observation stage fact from one parsed panel, correlated to the target."""
    if parsed["kind"] == "observed":
        correlated = parsed["session_id"] == request.native_session_id
        return {
            "kind": "provider-surface",
            "observation_kind": PARSER_KEY,
            "observed_state": "observed" if correlated else "inconclusive",
            "reason": None if correlated else "target-mismatch",
            "observed_at": _now(),
            "provider_version": parsed["provider_version"],
            "parser_key": parsed["parser_key"],
            "session_id": parsed["session_id"],
            "correlated": correlated,
            "model": parsed["model"],
            "effort": parsed["effort"],
            "render_floor": {"width": parsed["pane_width"]},
            "evidence_sha256": parsed["evidence_sha256"],
        }
    session_id = parsed.get("session_id")
    observation: dict[str, Any] = {
        "kind": "provider-surface",
        "observation_kind": PARSER_KEY,
        "observed_state": "inconclusive",
        "reason": parsed["reason"],
        "observed_at": _now(),
        "provider_version": parsed.get("provider_version", request.provider_version),
        "parser_key": parsed.get("parser_key", PARSER_KEY),
        "session_id": session_id,
        "correlated": session_id == request.native_session_id,
        "model": None,
        "effort": None,
        "render_floor": {"width": parsed.get("pane_width")},
        "evidence_sha256": parsed.get("evidence_sha256"),
    }
    return observation


def _final_event(
    request: ro.RouteObservationRequest,
    *,
    result: str,
    disposition: str,
    observation: Optional[dict[str, Any]],
    close_proof: Optional[dict[str, Any]],
    prewrite_readiness: Optional[PrewriteReadiness] = None,
) -> dict[str, Any]:
    event = {
        "schema_version": ro.SCHEMA_VERSION,
        "result": result,
        "disposition": disposition,
        "operation_id": request.operation_id,
        "request_digest": request.request_digest(),
        "observed_state": observation["observed_state"] if observation else None,
        "close_outcome": close_proof["outcome"] if close_proof else None,
        "committed_at": _now(),
    }
    if prewrite_readiness is not None:
        event["prewrite_readiness"] = prewrite_readiness.fact()
    return event


def _read_wake(inbox_message_id: Optional[int], *, db: Any = None) -> Optional[dict[str, Any]]:
    """The wake claim message for one inbox row, or None (never delivered)."""
    if inbox_message_id is None:
        return None

    def _one(session: Any) -> Optional[dict[str, Any]]:
        row = (
            session.query(database.InboxModel)
            .filter(database.InboxModel.id == inbox_message_id)
            .one_or_none()
        )
        if row is None:
            return None
        parsed: dict[str, Any] = json.loads(row.message)
        return parsed

    if db is not None:
        return _one(db)
    with database.SessionLocal() as session:
        return _one(session)


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------


class CodexRouteObserver:
    """Dark M10-C Codex route-observation orchestrator (duck-typed, no registry).

    Walks the merged ``route_observation`` stage machine in its exact order for
    one identity-bound ``/status`` control: claim, requester revalidation,
    pre-probe (first-CAS authorizes the one ``/status``), the provider-surface
    observation, pre-close, the close proof, and the atomic terminal commit.

    Idempotent under a lost response: an exact retry replays the stored
    terminal result with no second ``/status``, no second close, and no second
    wake; a partially-journaled retry continues from the durable stage facts,
    never re-issuing an already-authorized probe.

    The wake is not delivered here: ``route_observation.complete`` writes the
    exact-requester inbox claim atomically.  This orchestrator's responsibility
    ends at correct stage sequencing and honest observations.
    """

    def __init__(
        self,
        *,
        surface: CodexPaneSurface,
        requester_generation_probe: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self._surface = surface
        self._requester_generation_probe = requester_generation_probe

    def observe(self, request: ro.RouteObservationRequest, *, db: Any = None) -> dict[str, Any]:
        """Drive one identity-bound ``/status`` control to a terminal result."""
        claimed = ro.claim(request, db=db)
        if claimed["terminal"]:
            return self._replay_outcome(request, claimed, db=db)

        disposition = DISPOSITION_DELIVERED
        if not self._requester_is_current(request):
            return self._stale_requester_outcome(request, db=db)

        if claimed["pre_probe_intent_json"] is None:
            readiness = self._surface.await_input_ready()
            if not readiness.ready:
                return self._prewrite_refusal_outcome(request, readiness=readiness, db=db)
            # Readiness is a bounded wait.  Revalidate after it so a requester that
            # was superseded during that window still gets the normative zero-input
            # disposition immediately before provider input is authorized.
            if not self._requester_is_current(request):
                return self._stale_requester_outcome(request, db=db)

        probe = ro.pre_probe(
            request,
            intent=_pre_probe_intent(request, pane_id=self._surface.pane_id),
            db=db,
        )
        newly_authorized = probe.get("authorized") is True
        stored = ro.get(request.operation_id, db=db)
        observation = self._reconcile_observation(
            request, stored=stored, newly_authorized=newly_authorized, db=db
        )
        close_proof = self._reconcile_close_proof(request, stored=stored, db=db)

        if (
            observation["observed_state"] == "observed"
            and close_proof["outcome"] == CLOSE_COMPOSER_RESTORED
        ):
            result = ro.RESULT_OBSERVED_CLOSED
        else:
            result = ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        final_event = _final_event(
            request,
            result=result,
            disposition=disposition,
            observation=observation,
            close_proof=close_proof,
        )
        terminal = ro.complete(request, result=result, final_event=final_event, db=db)
        return self._build_outcome(
            request,
            terminal,
            disposition=disposition,
            db=db,
            observation=observation,
            close_proof=close_proof,
        )

    def _prewrite_refusal_outcome(
        self,
        request: ro.RouteObservationRequest,
        *,
        readiness: PrewriteReadiness,
        db: Any = None,
    ) -> dict[str, Any]:
        """Seal a bounded readiness refusal while every effect fact is null."""
        if readiness.reason == PREWRITE_PROVIDER_NOT_READY:
            disposition = DISPOSITION_PROVIDER_NOT_READY
        elif readiness.reason == PREWRITE_PANE_UNREADABLE:
            disposition = DISPOSITION_PANE_UNREADABLE
        else:
            raise ro.RouteObservationInvalid(
                f"prewrite refusal requires a non-ready reason; got {readiness.reason!r}"
            )
        final_event = _final_event(
            request,
            result=ro.RESULT_ZERO_EFFECT_REFUSAL,
            disposition=disposition,
            observation=None,
            close_proof=None,
            prewrite_readiness=readiness,
        )
        terminal = ro.complete(
            request,
            result=ro.RESULT_ZERO_EFFECT_REFUSAL,
            final_event=final_event,
            db=db,
        )
        return self._build_outcome(
            request,
            terminal,
            disposition=disposition,
            db=db,
            prewrite_readiness=readiness.fact(),
        )

    def _stale_requester_outcome(
        self, request: ro.RouteObservationRequest, *, db: Any = None
    ) -> dict[str, Any]:
        """The zero-input terminal for a drifted requester, sealing the facts.

        The result is always the one the durable facts already prove —
        requester-stale is only the disposition, carried in the final event,
        never a reason to misreport the evidence:

        - no effect fact: ``zero-effect-refusal`` (zero input, nothing happened);
        - all four stage facts durable and positively resolved: the fact-derived
          ``observed-closed`` is sealed and the receipt is minted;
        - otherwise (a genuinely partial journal, or positive facts but an
          unproven close): ``ambiguous-after-possible-effect``.

        The disposition wins over any stage-conflict the later stages would
        raise: this terminal is completed directly, never by walking further
        stages.
        """
        stored = ro.get(request.operation_id, db=db)
        observation: Optional[dict[str, Any]] = None
        close_proof: Optional[dict[str, Any]] = None
        if stored is None or stored["pre_probe_intent_json"] is None:
            result = ro.RESULT_ZERO_EFFECT_REFUSAL
        else:
            if stored["observation_json"] is not None:
                observation = json.loads(stored["observation_json"])
            if stored["close_proof_json"] is not None:
                close_proof = json.loads(stored["close_proof_json"])
            # the machine's ordering invariant makes a present close proof imply
            # all four ordered stage facts.
            if (
                close_proof is not None
                and observation is not None
                and observation.get("observed_state") == "observed"
                and close_proof.get("outcome") == CLOSE_COMPOSER_RESTORED
            ):
                result = ro.RESULT_OBSERVED_CLOSED
            else:
                result = ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT
        final_event = _final_event(
            request,
            result=result,
            disposition=DISPOSITION_REQUESTER_STALE,
            observation=observation,
            close_proof=close_proof,
        )
        terminal = ro.complete(request, result=result, final_event=final_event, db=db)
        return self._build_outcome(
            request,
            terminal,
            disposition=DISPOSITION_REQUESTER_STALE,
            db=db,
        )

    def _reconcile_observation(
        self,
        request: ro.RouteObservationRequest,
        *,
        stored: Optional[dict[str, Any]],
        newly_authorized: bool,
        db: Any = None,
    ) -> dict[str, Any]:
        """The observation stage fact, reconciled with the durable journal.

        A prior run that durably committed the observation (a crash between
        stages) is continued by reusing its exact stored bytes — the machine's
        identical-bytes replay CAS never sees self-manufactured fresh bytes.
        Only an uncommitted observation is derived and recorded.
        """
        if stored is not None and stored["observation_json"] is not None:
            observation: dict[str, Any] = json.loads(stored["observation_json"])
            return observation
        observation = self._derive_observation(request, newly_authorized=newly_authorized)
        ro.record_observation(request, observation=observation, db=db)
        return observation

    def _derive_observation(
        self, request: ro.RouteObservationRequest, *, newly_authorized: bool
    ) -> dict[str, Any]:
        """Capture the pane and build the observation fact (or a possible-effect
        inconclusive when the one authorized probe did not produce a panel)."""
        send_failed = False
        submission_proven = True
        if newly_authorized:
            try:
                submission_proven = bool(self._surface.send_status_command())
            except Exception:  # noqa: BLE001 - a refused write is a possible effect
                send_failed = True
        if send_failed:
            return _inconclusive_observation(request, reason="send-failed")
        if not submission_proven:
            return _inconclusive_observation(request, reason="submission-unproven")
        try:
            rows = self._surface.capture_screen()
        except Exception:  # noqa: BLE001 - an unreadable pane is not a panel
            rows = []
        parsed = parse_codex_route_panel(
            rows,
            pinned_version=request.provider_version,
            pane_width=self._surface.pane_width(),
        )
        return _observation_from_parse(request, parsed)

    def _reconcile_close_proof(
        self,
        request: ro.RouteObservationRequest,
        *,
        stored: Optional[dict[str, Any]],
        db: Any = None,
    ) -> dict[str, Any]:
        """The close-proof stage fact, reconciled with the durable journal.

        A crash after the close proof committed (before the terminal commit) is
        continued by reusing its exact stored bytes; only an uncommitted proof
        is derived and recorded.  The non-modal surface never issues a second
        ``Escape``.
        """
        if stored is not None and stored["close_proof_json"] is not None:
            close_proof: dict[str, Any] = json.loads(stored["close_proof_json"])
            return close_proof
        ro.pre_close(request, intent=_pre_close_intent(), db=db)
        try:
            restored = self._surface.composer_restored()
        except Exception:  # noqa: BLE001 - an unprovable close is indeterminate
            restored = None
        if restored is True:
            close_outcome = CLOSE_COMPOSER_RESTORED
        elif restored is False:
            close_outcome = CLOSE_NOT_RESTORED
        else:
            close_outcome = CLOSE_INDETERMINATE
        close_proof = _close_proof(close_outcome)
        ro.record_close_proof(request, proof=close_proof, db=db)
        return close_proof

    def read_result(self, operation_id: str, *, db: Any = None) -> Optional[dict[str, Any]]:
        """The stored terminal result for one operation, or None.

        Response-loss seam: a caller that lost the response queries the same
        operation by id and replays the stored result — a query never
        authorizes a second status probe, a second close, or any new input.
        """
        record = ro.get(operation_id, db=db)
        if record is None:
            return None
        outcome = {
            "schema": OBSERVER_SCHEMA,
            "operation_id": operation_id,
            "request_digest": record["request_digest"],
            "result": record["state"],
            "terminal": record["terminal"],
            "replayed": True,
            "disposition": DISPOSITION_REPLAYED,
            "receipt_digest": record["receipt_digest"],
            "inbox_message_id": record["inbox_message_id"],
            "observation": None,
            "close_proof": None,
            "prewrite_readiness": None,
            "record": record,
        }
        if record.get("final_event_json"):
            outcome["prewrite_readiness"] = json.loads(record["final_event_json"]).get(
                "prewrite_readiness"
            )
        if record.get("receipt_json"):
            outcome["receipt"] = json.loads(record["receipt_json"])
        wake = _read_wake(record["inbox_message_id"], db=db)
        if wake is not None:
            outcome["wake"] = wake
        return outcome

    def _requester_is_current(self, request: ro.RouteObservationRequest) -> bool:
        if self._requester_generation_probe is None:
            return True
        current = self._requester_generation_probe(request.requester_terminal_id)
        return current == request.requester_generation

    def _replay_outcome(
        self, request: ro.RouteObservationRequest, claimed: dict[str, Any], *, db: Any = None
    ) -> dict[str, Any]:
        stored = ro.get(request.operation_id, db=db)
        record = stored if stored is not None else claimed
        # ``ro.get`` records carry no ``replayed`` key; the claim replay is the
        # authoritative answer here.
        replayed = bool(claimed.get("replayed", True))
        return self._build_outcome(
            request, record, disposition=DISPOSITION_REPLAYED, replayed=replayed, db=db
        )

    def _build_outcome(
        self,
        request: ro.RouteObservationRequest,
        record: dict[str, Any],
        *,
        disposition: str,
        db: Any = None,
        observation: Optional[dict[str, Any]] = None,
        close_proof: Optional[dict[str, Any]] = None,
        prewrite_readiness: Optional[dict[str, Any]] = None,
        replayed: Optional[bool] = None,
    ) -> dict[str, Any]:
        if prewrite_readiness is None and record.get("final_event_json"):
            prewrite_readiness = json.loads(record["final_event_json"]).get("prewrite_readiness")
        outcome: dict[str, Any] = {
            "schema": OBSERVER_SCHEMA,
            "operation_id": request.operation_id,
            "request_digest": request.request_digest(),
            "result": record["state"],
            "terminal": record["terminal"],
            "replayed": bool(record.get("replayed", False)) if replayed is None else replayed,
            "disposition": disposition,
            "receipt_digest": record["receipt_digest"],
            "inbox_message_id": record["inbox_message_id"],
            "observation": observation,
            "close_proof": close_proof,
            "prewrite_readiness": prewrite_readiness,
            "record": record,
        }
        if record.get("receipt_json"):
            outcome["receipt"] = json.loads(record["receipt_json"])
        wake = _read_wake(record["inbox_message_id"], db=db)
        if wake is not None:
            outcome["wake"] = wake
        return outcome
