# ABOUTME: Pass/fail gate comparing a run's deterministic metrics against the
# ABOUTME: recorded baseline within a small tolerance.
"""Gate a harness report against its recorded baseline.

Only rank-derived metrics are gated; the wall-clock performance block never
affects the verdict. A metric fails when it is worse than baseline by more
than ``tolerance`` (absolute for 0..1 rates, count-scaled for loads), which
keeps float noise from flapping the gate while catching real regressions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GATED_FLOAT_METRICS = (
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "mrr_primary",
)
GATED_RATE_METRICS = (
    "hard_negative_case_rate",
    "hard_negative_above_first_hit_rate",
)
DEFAULT_TOLERANCE = 0.02


def check_gate(
    report: dict[str, Any],
    baseline_path: Path,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[str]:
    """Return the list of violations; empty means GREEN.

    Reports carrying an injected regression are compared like any other run:
    counterfactual verification depends on the gate judging them and failing
    them. The ``injection`` field is carried in the report so a downstream
    recorder can refuse to persist an injected run as a baseline; refusing to
    compare it here would make injected regressions unverifiable.
    """

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    if report.get("snapshot_id") != baseline.get("snapshot_id"):
        violations.append(
            f"snapshot mismatch: report {report.get('snapshot_id')!r} vs "
            f"baseline {baseline.get('snapshot_id')!r}"
        )
    for lane_name, baseline_lane in baseline["lanes"].items():
        run_lane = report.get("lanes", {}).get(lane_name)
        if run_lane is None:
            violations.append(f"lane {lane_name}: missing from report")
            continue
        base = baseline_lane["metrics"]
        run = run_lane["metrics"]
        for metric in GATED_FLOAT_METRICS:
            if run[metric] < base[metric] - tolerance:
                violations.append(
                    f"lane {lane_name}: {metric} regressed "
                    f"{base[metric]} -> {run[metric]} (tolerance {tolerance})"
                )
        for metric in GATED_RATE_METRICS:
            if run[metric] > base[metric] + tolerance:
                violations.append(
                    f"lane {lane_name}: {metric} worsened "
                    f"{base[metric]} -> {run[metric]} (tolerance {tolerance})"
                )
        for metric in ("hard_negative_load_at_5", "hard_negative_load_at_10"):
            allowed = int(base[metric]) + tolerance * max(1, base.get("cases", 1))
            if run[metric] > allowed:
                violations.append(
                    f"lane {lane_name}: {metric} rose " f"{base[metric]} -> {run[metric]}"
                )
        # Semantic coverage is null-safe on both sides pre-M2.
        base_cov = base.get("semantic_coverage")
        run_cov = run.get("semantic_coverage")
        if base_cov is not None and run_cov is not None and run_cov < base_cov:
            violations.append(
                f"lane {lane_name}: semantic_coverage dropped " f"{base_cov} -> {run_cov}"
            )
    return violations
