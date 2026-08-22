# ABOUTME: Retriever behavior tests: LIKE semantics, scope filtering, and the
# ABOUTME: unimplemented lane interfaces.
"""Retriever tests."""

from __future__ import annotations

import pytest
from harness.retrievers import (
    EmptyLaneRetriever,
    FtsRetriever,
    LegacySubstringRetriever,
    Scope,
    SemanticRetriever,
)


def test_scope_requires_exactly_one_form(snapshot) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Scope(tracker_projects=("cao-system",), all_projects=True)
    with pytest.raises(ValueError, match="exactly one"):
        Scope()


def test_empty_query_rejected(snapshot) -> None:
    retriever = LegacySubstringRetriever(snapshot)
    with pytest.raises(ValueError, match="nonempty"):
        retriever.search("   ", scope=Scope(tracker_projects=("cao-system",)))


def test_like_wildcards_are_faithful(snapshot) -> None:
    """Legacy search treats % and _ as SQL wildcards; the mirror must too."""

    retriever = LegacySubstringRetriever(snapshot)
    scope = Scope(tracker_projects=("cao-system",))
    # '%' matches any sequence (here: the letters elided from the symbol), so
    # a literal-substring engine would find nothing.
    percent = retriever.search("TerminalInputBlocked%rror", scope=scope)
    assert any(item.key == "cond-0035" for item in percent)
    # '_' matches exactly one character ('cond_0036' vs key 'cond-0036');
    # a literal reading of that query would match nothing.
    underscore = retriever.search("cond_0036", scope=scope)
    assert any(item.key == "cond-0036" for item in underscore)


def test_substring_matches_verbatim_phrase(snapshot) -> None:
    retriever = LegacySubstringRetriever(snapshot)
    scope = Scope(tracker_projects=("cao-system",))
    hits = retriever.search("activation receipt", scope=scope)
    keys = [item.key for item in hits]
    assert "cond-0087" in keys and "cond-0376" in keys


def test_line_wrap_defeats_literal_substring(snapshot) -> None:
    """The wrapped error fragment cannot match cond-0087's stored body."""

    retriever = LegacySubstringRetriever(snapshot)
    scope = Scope(tracker_projects=("cao-system",))
    hits = retriever.search("guarded bounce returned no verified activation receipt", scope=scope)
    keys = [item.key for item in hits]
    assert "cond-0087" not in keys  # body wraps 'activation\nreceipt'
    assert "cond-0376" in keys  # unwrapped occurrences still match


def test_project_scope_filters_corpus(snapshot) -> None:
    retriever = LegacySubstringRetriever(snapshot)
    scoped = retriever.search(
        "code intelligence",
        scope=Scope(tracker_projects=("aegix",)),
    )
    assert {item.key for item in scoped} <= {"aegix-0001"}


def test_subtree_scope_constrains_results(snapshot) -> None:
    retriever = LegacySubstringRetriever(snapshot)
    closure = snapshot.subtree_closure(["cond-0628"])
    hits = retriever.search(
        "relevance fixture and metrics harness",
        scope=Scope(tracker_projects=("cao-system",), subtree_roots=("cond-0628",)),
    )
    assert hits
    assert {item.key for item in hits} <= closure


def test_fts_and_semantic_lanes_not_installed(snapshot) -> None:
    with pytest.raises(NotImplementedError, match="lane-E"):
        FtsRetriever(snapshot)
    with pytest.raises(NotImplementedError, match="M2"):
        SemanticRetriever(snapshot)


def test_counterfactual_wrapper_behaves(snapshot) -> None:
    scope = Scope(tracker_projects=("cao-system",))
    inner = LegacySubstringRetriever(snapshot)
    ranked = inner.search("activation receipt", scope=scope)
    assert ranked
    assert EmptyLaneRetriever(inner).search("activation receipt", scope=scope) == []
