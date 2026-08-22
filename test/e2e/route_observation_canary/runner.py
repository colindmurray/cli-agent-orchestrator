"""M17 activation-lane runner entry points for the M10 route-observation canaries.

This module is the seam the M17 activation lane wires.  ``prepare`` builds the
exact operation request from an installed-source spec and writes the prepared
record; ``execute`` is where the live installed run happens (real codex
binary, tmux, paid turns).  LIVE execution is deliberately out of scope for
the C3 authoring lane, so ``execute`` raises :class:`PendingLiveExecution` —
a typed "not implemented in this build" fact, never a fabricated result.  The
unit mirror in ``test/services/test_route_observation_canary_unit.py`` drives
the same five cases against fake panes, so the deterministic assertion work
already exists; M17 supplies the real pane surface.

Invocation (one per canary case, named by ``cases.CanaryCase.runner_key``)::

    python -m test.e2e.route_observation_canary.runner positive-path prepare \\
        --spec /absolute/spec.json --output /absolute/prepared.json
    python -m test.e2e.route_observation_canary.runner positive-path execute \\
        --prepared /absolute/prepared.json --output /absolute/evidence.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from test.e2e.route_observation_canary.cases import RUNNER_KEYS, get_case_by_runner_key
from typing import Any, Mapping

from cli_agent_orchestrator.services import route_observation as ro

#: The case keys the runner accepts, closed at the five spec-§7 cases.
_CASE_CHOICES = sorted(RUNNER_KEYS)


class PendingLiveExecution(RuntimeError):
    """The live installed run belongs to the M17 activation lane, not this build."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _prepare(spec_path: Path, output_path: Path) -> None:
    """Resolve the exact operation request and write the prepared record."""
    spec = _read(spec_path)
    request = build_request_from_spec(spec)
    _write(
        output_path,
        {
            "spec": spec,
            "request": dataclasses.asdict(request),
            "request_digest": request.request_digest(),
        },
    )


def _execute(prepared_path: Path, output_path: Path) -> None:
    """Drive the installed Codex pane for one case and write evidence.

    M17 wires the real ``RealCodexPaneSurface`` from the prepared pane facts
    here and validates the terminal outcome against the case's expected
    result.  This build deliberately has no live provider surface, so the
    entry point terminates as pending rather than fabricating a receipt.
    """
    prepared = _read(prepared_path)
    case = get_case_by_runner_key(prepared.get("case", ""))
    raise PendingLiveExecution(
        f"canary {case.case_id} ({case.name}) is pending_live_execution; the M17 "
        "activation lane owns its live installed run against the real Codex pane "
        "and validates the result in isolation"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=_CASE_CHOICES, help="one spec-§7 canary case")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="build the exact operation request")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser(
        "execute", help="drive the installed pane (pending until M17 wires it)"
    )
    execute.add_argument("--prepared", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args.spec, args.output)
    else:
        _execute(args.prepared, args.output)


if __name__ == "__main__":
    main()
