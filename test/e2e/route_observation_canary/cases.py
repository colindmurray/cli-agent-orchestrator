"""The spec-§7 exact-build canary case definitions for the Codex route-observer.

Authoring lane C3.  Each case is authored as code: a definition here, a
fixture surface in :mod:`fixtures`, and a unit-mirror test in
``test/services/test_route_observation_canary_unit.py`` that drives the
``route_observation_codex`` adapter in isolation against a fake pane.

LIVE execution (real codex binary, tmux, paid turns) is deliberately out of
scope for this lane.  Every case therefore carries the durable
``pending_live_execution`` marker (``True``) and names the runner entry
point the M17 activation lane wires before any installed run; the installed
tests in ``test/e2e/test_route_observation_canary.py`` are marked
``pending_live_execution`` and never execute in this build.

The five cases are exactly the spec-§7 canary set, mapped onto the dark
Codex route-observation adapter:

1. positive path — identity-bound ``/status`` delivered -> observed -> closed
   -> the positive ``cao-route-observation-receipt-v1`` + exact-requester wake;
2. stale requester — a superseded requester generation gets zero input and
   the ``requester-stale`` disposition;
3. response loss / no replay — a lost response replays the stored result with
   no duplicate ``/status``, close, or wake;
4. ambiguous close / no second Escape — an unprovable close is woken
   ambiguous and never followed by a second ``Escape`` on the non-modal
   surface;
5. restart recovery — the operation's durable stage facts and result survive
   a restart without duplicating an effect.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The M10 canary case registry is closed at five cases (spec §7).
_MARKER = "pending_live_execution"


def _entry_point(runner_key: str) -> str:
    """The ``runner.py`` command the M17 activation lane runs for one case."""
    return f"test.e2e.route_observation_canary.runner {runner_key} execute"


@dataclass(frozen=True)
class CanaryCase:
    """One spec-§7 installed canary, authored as a definition.

    ``pending_live_execution`` is the durable marker carried by every case:
    the adapter capability stays dark and no live provider surface is touched
    by this lane.  ``runner_entry_point`` names the seam the M17 activation
    lane wires before any installed run.
    """

    case_id: str
    name: str
    spec_section: str
    summary: str
    runner_key: str
    runner_entry_point: str
    pending_live_execution: bool = True

    @property
    def marker(self) -> str:
        return _MARKER


POSITIVE_PATH = CanaryCase(
    case_id="m10-codex-01-positive-path",
    name="positive path",
    spec_section="spec-§7 positive path",
    summary=(
        "identity-bound /status delivered -> observed -> closed -> positive "
        "cao-route-observation-receipt-v1 + exact-requester wake"
    ),
    runner_key="positive-path",
    runner_entry_point=_entry_point("positive-path"),
)

STALE_REQUESTER = CanaryCase(
    case_id="m10-codex-02-stale-requester",
    name="stale requester",
    spec_section="spec-§7 stale requester",
    summary=(
        "a superseded requester generation gets zero input and the " "requester-stale disposition"
    ),
    runner_key="stale-requester",
    runner_entry_point=_entry_point("stale-requester"),
)

REPLAY_NO_DUPLICATE = CanaryCase(
    case_id="m10-codex-03-replay-no-duplicate",
    name="response loss / no replay",
    spec_section="spec-§7 response loss / no replay",
    summary=("a lost response replays the stored result; no duplicate /status, " "close, or wake"),
    runner_key="replay-no-duplicate",
    runner_entry_point=_entry_point("replay-no-duplicate"),
)

AMBIGUOUS_CLOSE = CanaryCase(
    case_id="m10-codex-04-ambiguous-close",
    name="ambiguous close / no second Escape",
    spec_section="spec-§7 ambiguous close / no second Escape",
    summary=(
        "an unprovable close is woken ambiguous and never followed by a "
        "second Escape on the non-modal surface"
    ),
    runner_key="ambiguous-close",
    runner_entry_point=_entry_point("ambiguous-close"),
)

RESTART_RECOVERY = CanaryCase(
    case_id="m10-codex-05-restart-recovery",
    name="restart recovery",
    spec_section="spec-§7 restart recovery",
    summary=(
        "the operation's durable record and result survive a restart without "
        "duplicating an effect"
    ),
    runner_key="restart-recovery",
    runner_entry_point=_entry_point("restart-recovery"),
)

#: The closed five-case registry, in spec order.
CANARY_CASES: tuple[CanaryCase, ...] = (
    POSITIVE_PATH,
    STALE_REQUESTER,
    REPLAY_NO_DUPLICATE,
    AMBIGUOUS_CLOSE,
    RESTART_RECOVERY,
)

#: Lookup by ``case_id`` (``m10-codex-01-positive-path``).
CASE_INDEX: dict[str, CanaryCase] = {case.case_id: case for case in CANARY_CASES}

#: Lookup by ``runner_key`` (``positive-path``).
RUNNER_KEYS: dict[str, CanaryCase] = {case.runner_key: case for case in CANARY_CASES}


def get_case(case_id: str) -> CanaryCase:
    """One case by ``case_id``, or a typed ``KeyError``-equivalent."""
    try:
        return CASE_INDEX[case_id]
    except KeyError:
        known = ", ".join(sorted(CASE_INDEX))
        raise ValueError(f"unknown canary case {case_id!r}; known: {known}") from None


def get_case_by_runner_key(runner_key: str) -> CanaryCase:
    """One case by its ``runner.py`` subcommand key, or a typed error."""
    try:
        return RUNNER_KEYS[runner_key]
    except KeyError:
        known = ", ".join(sorted(RUNNER_KEYS))
        raise ValueError(f"unknown canary runner key {runner_key!r}; known: {known}") from None
