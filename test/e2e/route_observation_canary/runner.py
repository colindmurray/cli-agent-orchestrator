"""M17 activation-lane runner entry points for the M10 route-observation canaries.

``prepare`` binds one exact operation request to the installed target and
requester identities.  ``execute`` drives that request through the real Codex
pane surface while the production capability remains dark.  The surrounding
M17 harness owns provider launch, exact-build attestation, wake delivery, and
conductor consumption; this runner owns only the fork-side operation and its
process-boundary evidence.

Every provider input is counted in an append-only JSONL trace.  That trace is
shared by fresh runner processes, so response-loss and restart checks prove
that the durable operation did not authorize a second ``/status``.  The
ambiguous-close case kills the real pane only after its status panel was
captured, making the close proof unreadable without replacing the provider
surface with a fake.  The restart case deliberately stops after the durable
observation, then resumes in a separate invocation.

Invocation (one per canary case, named by ``cases.CanaryCase.runner_key``)::

    python -m test.e2e.route_observation_canary.runner positive-path prepare \\
        --spec /absolute/spec.json --output /absolute/prepared.json
    python -m test.e2e.route_observation_canary.runner positive-path execute \\
        --prepared /absolute/prepared.json --output /absolute/evidence.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
from pathlib import Path
from test.e2e.route_observation_canary.cases import RUNNER_KEYS, get_case_by_runner_key
from typing import Any, Callable, Mapping, Optional

import requests

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.clients.tmux import tmux_binary
from cli_agent_orchestrator.services import route_observation as ro
from cli_agent_orchestrator.services import route_observation_codex as roc

#: The case keys the runner accepts, closed at the five spec-§7 cases.
_CASE_CHOICES = sorted(RUNNER_KEYS)


class LiveCanaryInvalid(RuntimeError):
    """The installed runtime cannot honestly execute or validate the case."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True, separators=(",", ":")) + "\n")


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line)
            if not isinstance(event, dict):
                raise LiveCanaryInvalid(f"event trace {path} contains a non-object entry")
            events.append(event)
    return events


class _TracedRealSurface:
    """The real pane surface plus durable effect-count evidence.

    ``kill_after_capture`` is the one case-specific fault.  It acts only after
    the real panel bytes were returned, so observation remains real while the
    later close proof becomes unprovable because the real pane is gone.
    """

    def __init__(
        self,
        *,
        pane_id: str,
        event_log: Path,
        terminal_id: Optional[str],
        session_name: Optional[str],
        window_name: Optional[str],
        kill_after_capture: bool,
    ) -> None:
        self.pane_id = pane_id
        self._event_log = event_log
        self._kill_after_capture = kill_after_capture
        self._killed = False
        self._captured_width: Optional[int] = None
        self._inner = roc.RealCodexPaneSurface(
            pane_id,
            terminal_id=terminal_id,
            session_name=session_name,
            window_name=window_name,
            timeout=20.0,
        )

    def capture_screen(self) -> list[str]:
        self._captured_width = self._inner.pane_width()
        rows = self._inner.capture_screen()
        _append_event(
            self._event_log,
            {
                "kind": "pane-capture",
                "at": _now(),
                "row_count": len(rows),
                "evidence_sha256": roc.nsr.evidence_digest(rows),
            },
        )
        if self._kill_after_capture and not self._killed:
            killed = subprocess.run(
                [tmux_binary(), "kill-pane", "-t", self.pane_id],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if killed.returncode != 0:
                raise LiveCanaryInvalid(
                    f"could not kill the isolated canary pane after capture: "
                    f"{killed.stderr.strip()}"
                )
            self._killed = True
            _append_event(self._event_log, {"kind": "pane-killed", "at": _now()})
        return rows

    def pane_width(self) -> Optional[int]:
        if self._captured_width is not None:
            return self._captured_width
        return self._inner.pane_width()

    def await_input_ready(self) -> roc.PrewriteReadiness:
        readiness = self._inner.await_input_ready()
        _append_event(
            self._event_log,
            {
                "kind": "prewrite-readiness",
                "at": _now(),
                "reason": readiness.reason,
                "provider_status": readiness.provider_status,
            },
        )
        return readiness

    def prove_composer_empty(self, provider_version: str) -> roc.PrewriteReadiness:
        proof = self._inner.prove_composer_empty(provider_version)
        _append_event(
            self._event_log,
            {
                "kind": "composer-emptiness-proof",
                "at": _now(),
                "provider_version": provider_version,
                "reason": proof.reason,
            },
        )
        return proof

    def send_status_command(self) -> bool:
        _append_event(self._event_log, {"kind": "status-authorized", "at": _now()})
        submitted = self._inner.send_status_command()
        _append_event(
            self._event_log,
            {"kind": "status-submission", "at": _now(), "submitted": bool(submitted)},
        )
        return submitted

    def composer_restored(self) -> Optional[bool]:
        return self._inner.composer_restored()


def _generation_probe(runtime: Mapping[str, Any]) -> Optional[Callable[[str], Optional[str]]]:
    url = runtime.get("requester_probe_url")
    if url is None:
        return None
    if not isinstance(url, str) or not url.startswith("http://127.0.0.1:"):
        raise LiveCanaryInvalid("requester_probe_url must name the isolated localhost server")

    def _probe(terminal_id: str) -> Optional[str]:
        response = requests.get(url.format(terminal_id=terminal_id), timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get("id") != terminal_id:
            raise LiveCanaryInvalid("requester generation probe returned a foreign terminal")
        generation = body.get("generation")
        return generation if isinstance(generation, str) else None

    return _probe


def _surface(runtime: Mapping[str, Any], *, ambiguous: bool) -> _TracedRealSurface:
    pane_id = runtime.get("pane_id")
    event_log = runtime.get("event_log")
    if not isinstance(pane_id, str) or not pane_id:
        raise LiveCanaryInvalid("runtime.pane_id is required")
    if not isinstance(event_log, str) or not Path(event_log).is_absolute():
        raise LiveCanaryInvalid("runtime.event_log must be an absolute path")
    return _TracedRealSurface(
        pane_id=pane_id,
        event_log=Path(event_log),
        terminal_id=runtime.get("target_terminal_id"),
        session_name=runtime.get("target_session_name"),
        window_name=runtime.get("target_window_name"),
        kill_after_capture=ambiguous,
    )


def _status_count(runtime: Mapping[str, Any]) -> int:
    return sum(
        event.get("kind") == "status-authorized"
        for event in _read_events(Path(str(runtime["event_log"])))
    )


def _inbox_count() -> int:
    with database.SessionLocal() as session:
        return int(session.query(database.InboxModel).count())


def _inbox_status(message_id: Any) -> str:
    if not isinstance(message_id, int):
        raise LiveCanaryInvalid("terminal outcome did not name an inbox row")
    with database.SessionLocal() as session:
        row = session.get(database.InboxModel, message_id)
        if row is None:
            raise LiveCanaryInvalid(f"terminal outcome named absent inbox row {message_id}")
        return str(row.status)


def _restart_interrupt(
    request: ro.RouteObservationRequest,
    observer: roc.CodexRouteObserver,
    surface: _TracedRealSurface,
    requester_generation_probe: Optional[Callable[[str], Optional[str]]],
) -> dict[str, Any]:
    """Stop after the real observation is durable, before close/result.

    This deliberately uses the stage primitives to create the otherwise tiny
    process boundary after the durable observation commit.  Before entering
    that boundary it repeats the production requester's exact-generation gate.
    The later ``resume`` invocation uses the ordinary observer, which must
    adopt those bytes and finish without another input.
    """
    claimed = ro.claim(request)
    if claimed["terminal"]:
        raise LiveCanaryInvalid("restart interrupt found an already-terminal operation")
    if requester_generation_probe is None:
        raise LiveCanaryInvalid("restart interrupt requires a requester generation probe")
    current_requester_generation = requester_generation_probe(request.requester_terminal_id)
    if current_requester_generation != request.requester_generation:
        raise LiveCanaryInvalid("restart interrupt found a stale requester generation")
    readiness = surface.await_input_ready()
    if not readiness.ready:
        observer._prewrite_refusal_outcome(request, readiness=readiness)
        raise LiveCanaryInvalid(
            f"restart interrupt refused before provider input: {readiness.reason}"
        )
    composer = surface.prove_composer_empty(request.provider_version)
    if not composer.ready:
        observer._prewrite_refusal_outcome(request, readiness=composer)
        raise LiveCanaryInvalid(
            f"restart interrupt refused before provider input: {composer.reason}"
        )
    current_requester_generation = requester_generation_probe(request.requester_terminal_id)
    if current_requester_generation != request.requester_generation:
        observer._stale_requester_outcome(request)
        raise LiveCanaryInvalid(
            "restart interrupt found a stale requester generation after composer proof"
        )
    probe = ro.pre_probe(
        request,
        intent=roc._pre_probe_intent(request, pane_id=surface.pane_id),
    )
    observation = observer._derive_observation(
        request, newly_authorized=probe.get("authorized") is True
    )
    ro.record_observation(request, observation=observation)
    stored = ro.get(request.operation_id)
    if (
        stored is None
        or stored["observation_json"] is None
        or stored["close_proof_json"] is not None
    ):
        raise LiveCanaryInvalid("restart interrupt did not stop at the durable observation stage")
    return {
        "phase": "interrupt",
        "terminal": False,
        "observation": observation,
        "record": stored,
    }


def _validate_case(
    case_key: str,
    outcome: Mapping[str, Any],
    *,
    status_count: int,
    inbox_before: int,
    inbox_after: int,
    expected_inbox_delta: int,
    inbox_message_status: str,
) -> None:
    expected = {
        "positive-path": ro.RESULT_OBSERVED_CLOSED,
        "stale-requester": ro.RESULT_ZERO_EFFECT_REFUSAL,
        "replay-no-duplicate": ro.RESULT_OBSERVED_CLOSED,
        "ambiguous-close": ro.RESULT_AMBIGUOUS_AFTER_POSSIBLE_EFFECT,
        "restart-recovery": ro.RESULT_OBSERVED_CLOSED,
    }[case_key]
    if outcome.get("result") != expected or outcome.get("terminal") is not True:
        raise LiveCanaryInvalid(f"{case_key} produced an unexpected outcome: {outcome}")
    expected_status = 0 if case_key == "stale-requester" else 1
    if status_count != expected_status:
        raise LiveCanaryInvalid(
            f"{case_key} authorized {status_count} status commands; expected {expected_status}"
        )
    if inbox_after - inbox_before != expected_inbox_delta:
        raise LiveCanaryInvalid(
            f"{case_key} changed the inbox by {inbox_after - inbox_before} rows; "
            f"expected {expected_inbox_delta}"
        )
    if inbox_message_status != "pending":
        raise LiveCanaryInvalid(
            f"{case_key} terminal wake had status {inbox_message_status!r}; expected pending"
        )
    if (
        case_key == "stale-requester"
        and outcome.get("disposition") != roc.DISPOSITION_REQUESTER_STALE
    ):
        raise LiveCanaryInvalid("stale-requester did not record requester-stale")
    if case_key == "ambiguous-close":
        observation = outcome.get("observation") or {}
        proof = outcome.get("close_proof") or {}
        if (
            observation.get("observed_state") != "observed"
            or outcome.get("receipt_digest") is not None
            or proof.get("outcome")
            not in {
                roc.CLOSE_INDETERMINATE,
                roc.CLOSE_NOT_RESTORED,
            }
        ):
            raise LiveCanaryInvalid("ambiguous-close fabricated a positive close or receipt")
    elif case_key != "stale-requester" and not outcome.get("receipt_digest"):
        raise LiveCanaryInvalid(f"{case_key} did not mint its positive route receipt")


def build_request_from_spec(spec: Mapping[str, Any]) -> ro.RouteObservationRequest:
    """Build the exact operation request from an installed-source spec.

    The request binds the exact target tuple (target terminal id + generation,
    native session id, provider, provider version, provider artifact SHA-256)
    and the exact requester (requester terminal id + generation), exactly as
    ``route_observation.RouteObservationRequest`` validates it.
    """
    return ro.RouteObservationRequest(
        operation_id=str(spec["operation_id"]),
        target_terminal_id=spec["target_terminal_id"],
        target_generation=spec["target_generation"],
        native_session_id=spec["native_session_id"],
        provider=spec["provider"],
        provider_version=spec["provider_version"],
        provider_artifact_sha256=spec["provider_artifact_sha256"],
        requester_terminal_id=spec["requester_terminal_id"],
        requester_generation=spec["requester_generation"],
    )


def _prepare(case_key: str, spec_path: Path, output_path: Path) -> None:
    """Resolve the exact operation request and write the prepared record.

    The prepared record is self-describing: it carries the ``case`` key it
    was prepared for, so a later ``execute`` consuming it knows the case
    without guessing.
    """
    case = get_case_by_runner_key(case_key)
    spec = _read(spec_path)
    request = build_request_from_spec(spec)
    _write(
        output_path,
        {
            "case": case.runner_key,
            "spec": spec,
            "request": dataclasses.asdict(request),
            "request_digest": request.request_digest(),
        },
    )


def _execute(
    case_key: str,
    prepared_path: Path,
    output_path: Path,
    *,
    restart_phase: Optional[str] = None,
    replay_phase: Optional[str] = None,
) -> None:
    """Drive one installed case and write fork-side evidence."""
    database.init_db()
    prepared = _read(prepared_path)
    if prepared.get("case") != case_key:
        raise ValueError(
            f"prepared record {prepared_path} names case {prepared.get('case')!r}, "
            f"but the CLI requested {case_key!r}"
        )
    case = get_case_by_runner_key(case_key)
    spec = prepared.get("spec")
    runtime = spec.get("runtime") if isinstance(spec, dict) else None
    if not isinstance(runtime, dict):
        raise LiveCanaryInvalid("prepared spec has no runtime object")
    request = ro.RouteObservationRequest(**prepared["request"])
    surface = _surface(runtime, ambiguous=case_key == "ambiguous-close")
    requester_generation_probe = _generation_probe(runtime)
    observer = roc.CodexRouteObserver(
        surface=surface,
        requester_generation_probe=requester_generation_probe,
    )
    inbox_before = _inbox_count()

    if case_key == "restart-recovery" and restart_phase == "interrupt":
        outcome = _restart_interrupt(
            request,
            observer,
            surface,
            requester_generation_probe,
        )
        if _status_count(runtime) != 1:
            raise LiveCanaryInvalid("restart interrupt did not authorize exactly one status")
        _write(
            output_path,
            {
                "schema": "cao-m17-route-observation-canary-evidence-v1",
                "case_id": case.case_id,
                "case": case_key,
                "restart_phase": "interrupt",
                "status_command_count": _status_count(runtime),
                "inbox_count": _inbox_count(),
                "outcome": outcome,
                "recorded_at": _now(),
            },
        )
        return
    if case_key == "restart-recovery" and restart_phase != "resume":
        raise LiveCanaryInvalid("restart-recovery requires --restart-phase interrupt or resume")
    if case_key != "restart-recovery" and restart_phase is not None:
        raise LiveCanaryInvalid("--restart-phase is only valid for restart-recovery")
    if case_key == "replay-no-duplicate" and replay_phase not in {"initial", "retry"}:
        raise LiveCanaryInvalid("replay-no-duplicate requires --replay-phase initial or retry")
    if case_key != "replay-no-duplicate" and replay_phase is not None:
        raise LiveCanaryInvalid("--replay-phase is only valid for replay-no-duplicate")

    outcome = observer.observe(request)
    expected_inbox_delta = 1
    if case_key == "replay-no-duplicate":
        if replay_phase == "initial" and outcome.get("replayed") is not False:
            raise LiveCanaryInvalid("initial response-loss attempt unexpectedly replayed")
        if replay_phase == "retry":
            expected_inbox_delta = 0
            if outcome.get("replayed") is not True:
                raise LiveCanaryInvalid("terminal retry did not replay the stored result")

    status_count = _status_count(runtime)
    inbox_after = _inbox_count()
    inbox_message_status = _inbox_status(outcome.get("inbox_message_id"))
    # Restart resume begins after the interrupt already wrote zero wakes; all
    # other cases begin in this process.  In either form exactly one wake must
    # exist when the terminal result commits.
    if case_key == "restart-recovery":
        inbox_before = 0
    _validate_case(
        case_key,
        outcome,
        status_count=status_count,
        inbox_before=inbox_before,
        inbox_after=inbox_after,
        expected_inbox_delta=expected_inbox_delta,
        inbox_message_status=inbox_message_status,
    )
    _write(
        output_path,
        {
            "schema": "cao-m17-route-observation-canary-evidence-v1",
            "case_id": case.case_id,
            "case": case_key,
            "restart_phase": restart_phase,
            "replay_phase": replay_phase,
            "status_command_count": status_count,
            "inbox_count": inbox_after,
            "inbox_message_status": inbox_message_status,
            "events": _read_events(Path(str(runtime["event_log"]))),
            "outcome": outcome,
            "recorded_at": _now(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=_CASE_CHOICES, help="one spec-§7 canary case")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="build the exact operation request")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("execute", help="drive the installed pane")
    execute.add_argument("--prepared", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--restart-phase", choices=("interrupt", "resume"))
    execute.add_argument("--replay-phase", choices=("initial", "retry"))
    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args.case, args.spec, args.output)
    else:
        _execute(
            args.case,
            args.prepared,
            args.output,
            restart_phase=args.restart_phase,
            replay_phase=args.replay_phase,
        )


if __name__ == "__main__":
    main()
