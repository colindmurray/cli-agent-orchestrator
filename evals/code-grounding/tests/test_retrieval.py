"""Retrieval-primitive slice: query derivation, ranking, visibility rules."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))

from retrieval import (  # noqa: E402
    derive_queries,
    rank_files,
    skip_reason,
    top_k,
)


def test_derive_queries_extracts_exact_spans_and_error_fragments():
    q = derive_queries(
        "greeting output lacks emphasis",
        "Users report `hello bob` renders flat.\nError: emphasis missing in output\nplain line\n",
    )
    assert "hello bob" in q.exact
    assert any(frag.startswith("Error:") for frag in q.exact)
    assert "emphasis" in q.prose


def test_derive_queries_identifiers_exclude_plain_words_and_are_deterministic():
    body = "The deadman_minutes projection misses spec_writer_terra; see deadman_minutes and routing_overrides."
    first = derive_queries("t", body)
    second = derive_queries("t", body)
    assert first.as_dict() == second.as_dict()
    assert "deadman_minutes" in first.identifiers
    assert "projection" not in first.identifiers  # single plain word never qualifies


def test_derive_queries_caps_query_volume():
    body = "\n".join(f"`exact span number {i} here` token_{i}" for i in range(30))
    q = derive_queries("title", body)
    assert len(q.exact) <= 8
    assert len(q.identifiers) <= 10


def test_rank_files_prefers_matching_file_and_top_k_is_stable(toy_repo):
    tree = toy_repo["path"]
    q = derive_queries(
        "greeting emphasis",
        "Users see flat output; expected `hello bob`. Formatting lives in _format_message.\n",
    )
    scores = rank_files(q, tree)
    ranked = top_k(scores, 5)
    assert ranked[0] == "svc/greeter.py"
    assert top_k(scores, 5) == ranked  # deterministic tie-break ordering


def test_skip_reason_classifies_hidden_ignored_and_visible(toy_repo):
    tree = toy_repo["path"]
    assert skip_reason(tree, ".hidden/hook.sh") == "hidden"
    assert skip_reason(tree, "secrets/keys.txt") == "ignored"
    assert skip_reason(tree, "svc/greeter.py") is None
