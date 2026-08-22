# ABOUTME: Fixture validity tests: the committed corpus must reference only
# ABOUTME: data the snapshot supports, and tampering must fail validation.
"""Fixture-validity tests.

The fixture is a measurement instrument; these tests prove the instrument
still fits the corpus it claims to measure (every key exists, duplicate-pair
labels are confirmed relations in the snapshot, verbatim/comment queries
actually occur in the fixed text).
"""

from __future__ import annotations

import copy

import pytest
from harness.runner import FixtureError, validate_fixture


def test_committed_fixture_validates(snapshot, fixture_doc) -> None:
    validate_fixture(fixture_doc, snapshot)


def test_fixture_pins_its_snapshot(snapshot, fixture_doc) -> None:
    tampered = copy.deepcopy(fixture_doc)
    tampered["snapshot_id"] = "snap-000000000000"
    with pytest.raises(FixtureError, match="pins snapshot"):
        validate_fixture(tampered, snapshot)


def test_unknown_expected_key_fails(snapshot, fixture_doc) -> None:
    tampered = copy.deepcopy(fixture_doc)
    tampered["cases"][0]["expected"]["primary"].append("cond-9999")
    with pytest.raises(FixtureError, match="unknown issue key"):
        validate_fixture(tampered, snapshot)


def test_duplicate_label_requires_confirmed_relation(snapshot, fixture_doc) -> None:
    """A duplicate-pair case without a recorded duplicate_of edge is invalid."""

    tampered = copy.deepcopy(fixture_doc)
    case = next(c for c in tampered["cases"] if c["id"] == "dup-pair-bounce-family")
    # Strip every confirmed relation from the case's keys: point it at issues
    # that exist but carry no duplicate_of edge.
    case["expected"]["primary"] = ["cond-0045"]
    case["expected"]["acceptable"] = []
    with pytest.raises(FixtureError, match="confirmed duplicate relation"):
        validate_fixture(tampered, snapshot)


def test_verbatim_query_must_occur_in_corpus(snapshot, fixture_doc) -> None:
    tampered = copy.deepcopy(fixture_doc)
    case = next(c for c in tampered["cases"] if c["provenance"]["derivation"] == "verbatim")
    case["query"] = "a phrase that occurs nowhere in the tracker corpus at all"
    with pytest.raises(FixtureError, match="does not occur"):
        validate_fixture(tampered, snapshot)


def test_comment_derivation_must_occur_in_comments(snapshot, fixture_doc) -> None:
    """Comment-derived queries are checked against comments specifically."""

    tampered = copy.deepcopy(fixture_doc)
    case = next(c for c in tampered["cases"] if c["class"] == "comment-only")
    case["query"] = "zz-nowhere-in-comments-zz"
    with pytest.raises(FixtureError, match="comments"):
        validate_fixture(tampered, snapshot)


def test_every_case_declares_exactly_one_scope(snapshot, fixture_doc) -> None:
    tampered = copy.deepcopy(fixture_doc)
    tampered["cases"][0]["scope"] = {}
    with pytest.raises(FixtureError, match="exactly one scope form"):
        validate_fixture(tampered, snapshot)


def test_paraphrase_cases_exist_and_are_documented(fixture_doc) -> None:
    paraphrases = [c for c in fixture_doc["cases"] if c["provenance"]["derivation"] == "paraphrase"]
    assert paraphrases, "fixture must contain natural-language paraphrase cases"
    for case in paraphrases:
        assert case.get("notes"), f"paraphrase case {case['id']} lacks rationale"


def test_required_case_classes_covered(fixture_doc) -> None:
    classes = {c["class"] for c in fixture_doc["cases"]}
    required = {
        "exact-command",
        "exact-error",
        "exact-symbol",
        "exact-path",
        "nl-paraphrase",
        "duplicate-pair",
        "comment-only",
        "status-mix",
        "project-scope",
        "subtree-scope",
        "stale-revision",
    }
    missing = required - classes
    assert not missing, f"fixture missing required case classes: {sorted(missing)}"
