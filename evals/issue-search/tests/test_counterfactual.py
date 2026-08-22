# ABOUTME: Counterfactual-sensitivity tests (acceptance check 4): injected
# ABOUTME: relevance regressions must drive the harness RED against baseline.
"""Counterfactual sensitivity tests.

A harness that cannot fail is not a harness. These tests inject two concrete
relevance regressions and prove each drives the gated metrics below the
recorded baseline — a RED verdict:

- ``promote-noise``: every result list is flooded with irrelevant issues
  ahead of real matches, the observable symptom of a ranking regression.
- ``empty``: the only live lane returns nothing at all.

The uninjected run must stay GREEN, and the degradation must be visible in
the per-case block, not just in aggregates.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.gate import check_gate
from harness.runner import run_fixture


def _baseline_file(baseline_doc: dict, tmp_path: Path) -> Path:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline_doc), encoding="utf-8")
    return path


def test_uninjected_run_passes_gate(snapshot, fixture_doc, baseline_doc, tmp_path: Path) -> None:
    report = run_fixture(snapshot, fixture_doc)
    violations = check_gate(report, _baseline_file(baseline_doc, tmp_path))
    assert violations == [], f"clean run must be GREEN; violations: {violations}"


def test_promoted_noise_goes_red(snapshot, fixture_doc, baseline_doc, tmp_path: Path) -> None:
    """Flood result lists with irrelevant issues -> metrics collapse."""

    report = run_fixture(snapshot, fixture_doc, inject="promote-noise")
    assert report["injection"] == "promote-noise"
    lane = report["lanes"]["legacy-substring"]["metrics"]
    base = baseline_doc["lanes"]["legacy-substring"]["metrics"]
    assert lane["recall_at_5"] < base["recall_at_5"], "noise flood left recall@5 intact"
    assert lane["mrr"] < base["mrr"], "noise flood did not reduce MRR"
    assert (
        lane["hard_negative_above_first_hit_rate"] > base["hard_negative_above_first_hit_rate"]
    ), "noise flood did not raise hard-negative inversions"
    violations = check_gate(report, _baseline_file(baseline_doc, tmp_path))
    assert violations, "gate stayed GREEN under promote-noise injection"


def test_dropped_lane_goes_red(snapshot, fixture_doc, baseline_doc, tmp_path: Path) -> None:
    """Drop the only lane -> zero recall everywhere -> gate fails loudly."""

    report = run_fixture(snapshot, fixture_doc, inject="empty")
    assert report["injection"] == "empty"
    lane = report["lanes"]["legacy-substring"]["metrics"]
    assert lane["recall_at_5"] == 0.0
    assert lane["recall_at_10"] == 0.0
    assert lane["mrr"] == 0.0
    violations = check_gate(report, _baseline_file(baseline_doc, tmp_path))
    assert any(
        "recall_at_5" in v or "recall_at_10" in v for v in violations
    ), f"gate did not flag dropped-lane recall collapse: {violations}"


def test_degradation_is_visible_per_case(snapshot, fixture_doc) -> None:
    """The per-case block shows the regression, not just the aggregate."""

    clean = run_fixture(snapshot, fixture_doc)
    injected = run_fixture(snapshot, fixture_doc, inject="promote-noise")
    clean_cases = {c["case_id"]: c for c in clean["lanes"]["legacy-substring"]["per_case"]}
    noisy_cases = {c["case_id"]: c for c in injected["lanes"]["legacy-substring"]["per_case"]}
    differing = [
        cid
        for cid in clean_cases
        if noisy_cases[cid]["first_hit_rank"] != clean_cases[cid]["first_hit_rank"]
    ]
    assert len(differing) >= 10, (
        "promote-noise should shift first-hit ranks across many cases; "
        f"only {len(differing)} changed"
    )
