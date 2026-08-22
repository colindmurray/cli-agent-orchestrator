"""Baseline-runner slice: end-to-end metric computation on the synthetic repo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))

import run_baseline  # noqa: E402


def _args(mini_fixture: Path, out: Path):
    return run_baseline.main.__wrapped__ if False else None  # placeholder, replaced below


class _Args:
    fixture = None
    out = None
    repo_cao_conductor = None
    repo_cao = None
    cases = None
    keep_trees = None


def _run(tmp_path: Path, mini_fixture: Path) -> dict:
    args = _Args()
    args.fixture = str(mini_fixture)
    args.out = str(tmp_path / "report.json")
    args.cases = None
    report = run_baseline.run(args)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return json.loads(Path(args.out).read_text())


def test_runner_computes_expected_metrics_end_to_end(toy_repo, tmp_path):
    from conftest import mini_fixture as build_mini

    fixture_path = build_mini(toy_repo, tmp_path)
    report = _run(tmp_path, fixture_path)
    row = report["cases"][0]

    # farewell.py is absent from the search tree -> excluded from denominators
    assert row["expected_file_count"] == 2
    assert row["expected_symbol_count"] == 1  # farewell_user is introduced

    assert row["file_recall"] == {"5": 0.5, "10": 0.5, "20": 0.5}
    assert row["symbol_recall"] == {"5": 1.0, "10": 1.0, "20": 1.0}

    assert row["skipped_files"] == [{"path": ".hidden/hook.sh", "reason": "hidden"}]
    assert row["missed_files"] == [".hidden/hook.sh"]
    assert row["missed_symbols"] == []

    escapes = {e["target"]: e for e in row["fallback_escapes"]}
    assert escapes[".hidden/hook.sh"]["fallback_rescued"] is True
    assert row["fallback_escape_rate"] == 1.0

    agg = report["aggregate"]
    assert agg["case_repo_count"] == 1
    assert agg["fallback_escape_rate_micro"] == 1.0


def test_runner_is_byte_deterministic_across_runs(toy_repo, tmp_path):
    from conftest import mini_fixture as build_mini

    fixture_path = build_mini(toy_repo, tmp_path)
    first = _run(tmp_path, fixture_path)
    second = _run(tmp_path, fixture_path)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_runner_respects_case_filter(toy_repo, tmp_path):
    from conftest import mini_fixture as build_mini

    fixture_path = build_mini(toy_repo, tmp_path)
    args = _Args()
    args.fixture = str(fixture_path)
    args.out = str(tmp_path / "filtered.json")
    args.cases = ["toy-0001"]
    report = run_baseline.run(args)
    assert len(report["cases"]) == 1
