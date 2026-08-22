"""M10 Codex route-observation exact-build canaries — installed, pending live.

These are the five spec-§7 canary cases for the dark ``route_observation_codex``
adapter, one named test per case.  LIVE execution (real codex binary, tmux,
paid turns) is out of scope for the C3 authoring lane, so every test carries
the ``pending_live_execution`` marker and names its runner entry point for the
M17 activation lane in its docstring; the deterministic assertion work for
each case is exercised in isolation by
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

pytestmark = [pytest.mark.e2e, pytest.mark.pending_live_execution]


def _pending(case: cases.CanaryCase) -> None:
    pytest.skip(
        f"{case.case_id} ({case.name}) is pending_live_execution: the M17 activation "
        f"lane owns its live installed run ({case.runner_entry_point}) against the "
        "real Codex pane; this build never observes a live surface"
    )


def test_installed_m10_codex_positive_path():
    """Spec-§7 positive path — the installed canary mints the positive receipt.

    Runner entry point: test.e2e.route_observation_canary.runner positive-path execute
    M17 wires the live pane; the unit mirror drives this case in isolation.
    """
    _pending(cases.POSITIVE_PATH)


def test_installed_m10_codex_stale_requester():
    """Spec-§7 stale requester — zero input, requester-stale disposition.

    Runner entry point: test.e2e.route_observation_canary.runner stale-requester execute
    M17 wires the live pane; the unit mirror drives this case in isolation.
    """
    _pending(cases.STALE_REQUESTER)


def test_installed_m10_codex_replay_no_duplicate():
    """Spec-§7 response loss / no replay — stored result, no duplicate effect.

    Runner entry point: test.e2e.route_observation_canary.runner replay-no-duplicate execute
    M17 wires the live pane; the unit mirror drives this case in isolation.
    """
    _pending(cases.REPLAY_NO_DUPLICATE)


def test_installed_m10_codex_ambiguous_close():
    """Spec-§7 ambiguous close / no second Escape — woken ambiguous, zero Escapes.

    Runner entry point: test.e2e.route_observation_canary.runner ambiguous-close execute
    M17 wires the live pane; the unit mirror drives this case in isolation.
    """
    _pending(cases.AMBIGUOUS_CLOSE)


def test_installed_m10_codex_restart_recovery():
    """Spec-§7 restart recovery — durable stage facts survive a restart.

    Runner entry point: test.e2e.route_observation_canary.runner restart-recovery execute
    M17 wires the live pane; the unit mirror drives this case in isolation.
    """
    _pending(cases.RESTART_RECOVERY)
