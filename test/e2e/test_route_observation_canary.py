"""M10 Codex route-observation exact-build canary case index.

These are the five spec-§7 canary cases for the dark ``route_observation_codex``
adapter, one named declaration per case.  M17 executes them once, in order, in
``test_m17_t3_live_canary.py`` so the stale-requester case can prove zero input
and the four deliverable cases can each spend exactly one turn without these
declarations duplicating provider effects.  The deterministic assertion work
for each case is also exercised in isolation by
``test/services/test_route_observation_canary_unit.py`` against the fake panes
in ``test/e2e/route_observation_canary/fixtures.py``.

The capability stays dark: ``route_observation.enabled()`` and
``route_observation_codex.enabled()`` remain ``False`` in this build, and
nothing here observes a live provider surface or issues pane input.

M17 wires each runner entry point, e.g.::

    python -m test.e2e.route_observation_canary.runner positive-path prepare \\
        --spec /absolute/spec.json --output /absolute/prepared.json
    python -m test.e2e.route_observation_canary.runner positive-path execute \\
        --prepared /absolute/prepared.json --output /absolute/evidence.json
"""

from __future__ import annotations

from test.e2e.route_observation_canary import cases

import pytest

pytestmark = pytest.mark.e2e


def _grouped_live_runner(case: cases.CanaryCase) -> None:
    pytest.skip(
        f"{case.case_id} ({case.name}) executes in the ordered grouped M17 T3 live "
        f"ladder ({case.runner_entry_point}); this declaration must not duplicate "
        "its provider effect"
    )


def test_installed_m10_codex_positive_path():
    """Spec-§7 positive path — the installed canary mints the positive receipt.

    Runner entry point: test.e2e.route_observation_canary.runner positive-path execute
    The grouped live ladder wires the pane; the unit mirror drives this case in isolation.
    """
    _grouped_live_runner(cases.POSITIVE_PATH)


def test_installed_m10_codex_stale_requester():
    """Spec-§7 stale requester — zero input, requester-stale disposition.

    Runner entry point: test.e2e.route_observation_canary.runner stale-requester execute
    The grouped live ladder wires the pane; the unit mirror drives this case in isolation.
    """
    _grouped_live_runner(cases.STALE_REQUESTER)


def test_installed_m10_codex_replay_no_duplicate():
    """Spec-§7 response loss / no replay — stored result, no duplicate effect.

    Runner entry point: test.e2e.route_observation_canary.runner replay-no-duplicate execute
    The grouped live ladder wires the pane; the unit mirror drives this case in isolation.
    """
    _grouped_live_runner(cases.REPLAY_NO_DUPLICATE)


def test_installed_m10_codex_ambiguous_close():
    """Spec-§7 ambiguous close / no second Escape — woken ambiguous, zero Escapes.

    Runner entry point: test.e2e.route_observation_canary.runner ambiguous-close execute
    The grouped live ladder wires the pane; the unit mirror drives this case in isolation.
    """
    _grouped_live_runner(cases.AMBIGUOUS_CLOSE)


def test_installed_m10_codex_restart_recovery():
    """Spec-§7 restart recovery — durable stage facts survive a restart.

    Runner entry point: test.e2e.route_observation_canary.runner restart-recovery execute
    The grouped live ladder wires the pane; the unit mirror drives this case in isolation.
    """
    _grouped_live_runner(cases.RESTART_RECOVERY)
