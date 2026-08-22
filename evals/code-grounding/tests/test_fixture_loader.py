"""Fixture-loader slice: schema validation and committed-artifact consistency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))

from run_baseline import load_fixture  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "evals/code-grounding/fixtures/cases.json"
VERIFICATION = REPO_ROOT / "evals/code-grounding/reports/fixture-verification.json"

REQUIRED_TYPE_TAGS = {
    "exact-technical",
    "vague-prose",
    "stack-trace",
    "history-dependent",
    "cross-repository",
    "data-flow",
}


def test_load_fixture_accepts_current_schema():
    data = load_fixture(FIXTURE)
    assert data["schema_version"] == 1
    assert len(data["cases"]) == 25


def test_load_fixture_rejects_unknown_schema_version(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 99, "cases": []}))
    with pytest.raises(SystemExit):
        load_fixture(bad)


def test_committed_fixture_covers_required_case_types():
    data = json.loads(FIXTURE.read_text())
    seen = {t for c in data["cases"] for t in c["case_types"]}
    assert REQUIRED_TYPE_TAGS <= seen
    assert 20 <= len(data["cases"]) <= 30


def test_every_case_binds_expected_targets_to_fix_commits():
    data = json.loads(FIXTURE.read_text())
    for case in data["cases"]:
        assert case["issue"]["key"] == case["id"]
        assert case["issue"]["title"]
        assert case["issue"]["narrative"].startswith(case["issue"]["title"])
        assert case["case_types"], case["id"]
        # tool lanes stay optional/null for later pilot lanes (acceptance 5)
        assert set(case["tool_lanes"]) >= {"codebase_memory", "qmd"}
        assert all(v is None for v in case["tool_lanes"].values())
        for repo, blob in case["repos"].items():
            assert blob["fix_commits"], f"{case['id']}/{repo} has no fix commits"
            assert all(len(f["sha"]) == 40 for f in blob["fix_commits"])
            assert blob["search_sha"]
            assert blob["expected_files"], f"{case['id']}/{repo} has no expected files"


def test_cross_repository_cases_span_both_repos():
    data = json.loads(FIXTURE.read_text())
    cross = [c for c in data["cases"] if "cross-repository" in c["case_types"]]
    assert cross, "fixture must contain cross-repository cases"
    for case in cross:
        assert set(case["repos"]) == {"cao-conductor", "cao"}, case["id"]


def test_verification_report_records_zero_failures():
    report = json.loads(VERIFICATION.read_text())
    assert report["failures"] == []
    checks = report["checks"]
    assert len(checks) >= 100
    files = [c for c in checks if c["kind"] == "file"]
    symbols = [c for c in checks if c["kind"] == "symbol"]
    assert files and symbols
    introduced = [c for c in symbols if c.get("origin") == "introduced"]
    outside = [c for c in files if not c["exists_at_search_sha"]]
    assert introduced and outside, (
        "the fixture should knowingly carry fix-introduced targets; "
        "their absence from the search tree drives the recall denominators"
    )
