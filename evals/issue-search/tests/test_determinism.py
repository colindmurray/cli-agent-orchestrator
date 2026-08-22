# ABOUTME: Determinism tests (acceptance check 5): reruns from the fixed
# ABOUTME: snapshot reproduce identical relevance numbers.
"""Determinism tests.

Rank-derived metrics are a pure function of (snapshot bytes, fixture bytes,
lane code): two runs must produce byte-identical metrics blocks. Wall-clock
performance is excluded by construction — it lives in a separate block and is
never gated. The committed baseline file must also match a fresh run exactly,
pinning it against drift.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from harness.runner import main as runner_main


def _metrics_bytes(snapshot_dir: Path, fixture_path: Path) -> str:
    report_path = Path("/tmp/opencode") / "determinism-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    runner_main(
        [
            "--snapshots",
            str(snapshot_dir),
            "--fixture",
            str(fixture_path),
            "--report",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Strip the performance block: that is wall clock, not relevance.
    for lane in report["lanes"].values():
        lane.pop("performance", None)
    return json.dumps(report, indent=2, sort_keys=True)


def test_reruns_are_byte_identical(issue_search_root: Path, tmp_path: Path, monkeypatch) -> None:
    """Two full runs from the fixed snapshot agree byte for byte."""

    snapshots = issue_search_root / "snapshots"
    fixture = issue_search_root / "fixtures" / "corpus.v1.json"
    monkeypatch.setattr(sys, "argv", ["runner"])
    first = _metrics_bytes(snapshots, fixture)
    second = _metrics_bytes(snapshots, fixture)
    assert first == second


def test_committed_baseline_matches_fresh_run(
    issue_search_root: Path, baseline_doc: dict, tmp_path: Path, monkeypatch
) -> None:
    """The recorded baseline reproduces exactly from committed artifacts."""

    snapshots = issue_search_root / "snapshots"
    fixture = issue_search_root / "fixtures" / "corpus.v1.json"
    monkeypatch.setattr(sys, "argv", ["runner"])
    report_path = tmp_path / "report.json"
    runner_main(
        [
            "--snapshots",
            str(snapshots),
            "--fixture",
            str(fixture),
            "--report",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["snapshot_id"] == baseline_doc["snapshot_id"]
    for lane_name, recorded in baseline_doc["lanes"].items():
        fresh = report["lanes"][lane_name]["metrics"]
        assert fresh == recorded["metrics"], f"lane {lane_name} drifted from the committed baseline"


def test_subprocess_run_is_stable(issue_search_root: Path, tmp_path: Path) -> None:
    """A cold subprocess run agrees with an in-process run's metrics."""

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    common = [
        sys.executable,
        "-m",
        "harness.runner",
        "--snapshots",
        str(issue_search_root / "snapshots"),
        "--fixture",
        str(issue_search_root / "fixtures" / "corpus.v1.json"),
    ]
    env_root = str(issue_search_root)
    for out in (out_a, out_b):
        subprocess.run(
            common + ["--report", str(out)],
            check=True,
            cwd=env_root,
            env={"PYTHONPATH": env_root, "PATH": "/usr/bin:/bin"},
        )
    a = json.loads(out_a.read_text(encoding="utf-8"))
    b = json.loads(out_b.read_text(encoding="utf-8"))
    for lane in (*a["lanes"].values(), *b["lanes"].values()):
        lane.pop("performance", None)
    assert a == b
