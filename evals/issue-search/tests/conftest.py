# ABOUTME: Shared fixtures for the issue-search harness tests.
# ABOUTME: Loads the committed snapshot and fixture corpus once per session.
"""Shared fixtures for evals/issue-search tests.

``evals/issue-search/`` is a hyphenated directory outside the repo package, so
the tests put it directly on ``sys.path`` and import the ``harness`` package
by name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ISSUE_SEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(ISSUE_SEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ISSUE_SEARCH_ROOT))

from harness.snapshot import Snapshot, load_fixture, load_snapshot  # noqa: E402


@pytest.fixture(scope="session")
def issue_search_root() -> Path:
    return ISSUE_SEARCH_ROOT


@pytest.fixture(scope="session")
def snapshot() -> Snapshot:
    return load_snapshot(ISSUE_SEARCH_ROOT / "snapshots")


@pytest.fixture(scope="session")
def fixture_doc() -> dict:
    return load_fixture(ISSUE_SEARCH_ROOT / "fixtures" / "corpus.v1.json")


@pytest.fixture(scope="session")
def baseline_doc() -> dict:
    path = ISSUE_SEARCH_ROOT / "baselines" / "legacy-substring.json"
    return json.loads(path.read_text(encoding="utf-8"))
