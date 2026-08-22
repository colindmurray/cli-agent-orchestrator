"""The shared structured-filter/subtree-scope builder (design §10.1–§10.2).

The contract under test: one predicate vocabulary serves ``issue list`` and
ranked search alike; the subtree closure is complete and cycle-safe over
``child --part-of--> parent`` links; cross-project descendants are excluded by
tracker-project intersection; and every refusal is typed and scoped to the
request that caused it.
"""

import sqlite3

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import (
    _TRACKER_ORM_TABLE_NAMES,
    Base,
    TrackerIssueModel,
    TrackerLinkModel,
    _migrate_tracker_search_projection,
)
from cli_agent_orchestrator.services.issue_tracker import TrackerError, list_issues
from cli_agent_orchestrator.services.tracker_filters import (
    StructuredFilters,
    is_effectively_empty_query,
    resolve_scope,
    subtree_closure,
)


class FilterDb:
    """A file-backed tracker store with the search projection installed."""

    def __init__(self, path):
        self.path = str(path)
        self.engine = create_engine(f"sqlite:///{self.path}")
        Base.metadata.create_all(
            bind=self.engine,
            tables=[t for t in Base.metadata.sorted_tables if t.name in _TRACKER_ORM_TABLE_NAMES],
        )
        _migrate_tracker_search_projection(self.engine)

    def raw(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None)
        return conn


@pytest.fixture
def fdb(tmp_path):
    db = FilterDb(tmp_path / "filters.db")
    yield db
    db.engine.dispose()


def _issue(db, key, project_id="p1", **overrides):
    payload = {
        "key": key,
        "project_id": project_id,
        "title": overrides.pop("title", f"issue {key}"),
        "body": overrides.pop("body", ""),
        "status": overrides.pop("status", "open"),
        "severity": overrides.pop("severity", "P2"),
        "labels": overrides.pop("labels", "[]"),
        "kind": overrides.pop("kind", "bug"),
        "created_at": overrides.pop("created_at", None),
        "updated_at": overrides.pop("updated_at", None),
    }
    payload.update(overrides)
    session = sessionmaker(bind=db.engine)()
    session.add(TrackerIssueModel(**payload))
    session.commit()
    session.close()


def _link(db, child, parent, kind="part-of"):
    session = sessionmaker(bind=db.engine)()
    session.add(TrackerLinkModel(from_key=child, to_key=parent, kind=kind))
    session.commit()
    session.close()


# ---------------------------------------------------------------------------
# StructuredFilters: validation and ORM predicates
# ---------------------------------------------------------------------------


class TestStructuredFilterValidation:
    def test_invalid_status_raises_the_tracker_refusal(self):
        with pytest.raises(TrackerError) as excinfo:
            StructuredFilters(statuses=("nope",)).validated()
        assert "invalid status 'nope'" in str(excinfo.value)

    def test_invalid_severity_and_kind_raise(self):
        with pytest.raises(TrackerError):
            StructuredFilters(severities=("P9",)).validated()
        with pytest.raises(TrackerError):
            StructuredFilters(kinds=("widget",)).validated()

    def test_labels_normalise_identically_to_the_list_path(self):
        validated = StructuredFilters(labels=("a", "a", "b")).validated()
        assert validated.labels == ("a", "b")

    def test_scalar_families_pass_through_unmodified_for_list_parity(self):
        validated = StructuredFilters(
            components=(" conduct ",),
            observed_revisions=(" v1 ",),
            assignee=" agent ",
        ).validated()
        assert validated.components == (" conduct ",)
        assert validated.observed_revisions == (" v1 ",)
        assert validated.assignee == " agent "

    def test_orm_conditions_compose_every_family_with_and(self, fdb):
        _issue(
            fdb,
            "f-1",
            labels='["x","y"]',
            status="open",
            severity="P1",
            component="c",
            observed_revision="v1",
        )
        _issue(
            fdb,
            "f-2",
            labels='["x"]',
            status="open",
            severity="P2",
            component="c",
            observed_revision="v2",
        )
        conditions = (
            StructuredFilters(
                statuses=("open",),
                severities=("P1",),
                components=("c",),
                observed_revisions=("v1",),
                labels=("x", "y"),
            )
            .validated()
            .orm_conditions()
        )
        session = sessionmaker(bind=fdb.engine)()
        keys = sorted(k for (k,) in session.query(TrackerIssueModel.key).filter(*conditions).all())
        session.close()
        # AND across families: only the row satisfying every family survives.
        assert keys == ["f-1"]

    def test_without_label_and_unlabeled_conditions(self, fdb):
        _issue(fdb, "g-1", labels='["noise"]')
        _issue(fdb, "g-2", labels="[]")
        conditions = StructuredFilters(without_labels=("noise",)).validated().orm_conditions()
        unlabeled = StructuredFilters(unlabeled=True).validated().orm_conditions()
        session = sessionmaker(bind=fdb.engine)()
        assert [k for (k,) in session.query(TrackerIssueModel.key).filter(*conditions)] == ["g-2"]
        assert [k for (k,) in session.query(TrackerIssueModel.key).filter(*unlabeled)] == ["g-2"]
        session.close()


# ---------------------------------------------------------------------------
# Subtree closure: cycle safety and cross-project intersection
# ---------------------------------------------------------------------------


class TestSubtreeClosure:
    def test_closure_includes_roots_and_transitive_children(self, fdb):
        for key in ("r-0", "r-1", "r-2"):
            _issue(fdb, key)
        _link(fdb, "r-1", "r-0")
        _link(fdb, "r-2", "r-1")
        session = sessionmaker(bind=fdb.engine)()
        try:
            closure = subtree_closure(session, ["r-0"])
        finally:
            session.close()
        assert closure["keys"] == frozenset({"r-0", "r-1", "r-2"})

    def test_a_membership_cycle_terminates_and_stays_complete(self, fdb):
        for key in ("e-1", "e-2", "e-3"):
            _issue(fdb, key)
        _link(fdb, "e-1", "e-2")
        _link(fdb, "e-2", "e-3")
        _link(fdb, "e-3", "e-1")
        session = sessionmaker(bind=fdb.engine)()
        try:
            closure = subtree_closure(session, ["e-1"])
        finally:
            session.close()
        # UNION dedupes the revisited root; every member is still present.
        assert closure["keys"] == frozenset({"e-1", "e-2", "e-3"})

    def test_unknown_root_is_a_typed_scoped_refusal(self, fdb):
        _issue(fdb, "ok-1")
        session = sessionmaker(bind=fdb.engine)()
        try:
            with pytest.raises(TrackerError) as excinfo:
                subtree_closure(session, ["ok-1", "ghost-404"])
        finally:
            session.close()
        assert "ghost-404" in str(excinfo.value)

    def test_resolve_scope_intersects_projects_with_the_closure(self, fdb):
        _issue(fdb, "s-root", project_id="p1")
        _issue(fdb, "s-child", project_id="p1")
        _issue(fdb, "s-cross", project_id="p2")
        _issue(fdb, "outside", project_id="p2")
        _link(fdb, "s-child", "s-root")
        _link(fdb, "s-cross", "s-root")
        session = sessionmaker(bind=fdb.engine)()
        try:
            scoped = resolve_scope(
                session,
                project_ids=("p1",),
                all_projects=False,
                subtree_roots=("s-root",),
            )
            assert scoped.allowed_keys is not None
            # The p2 descendant of a p1 root is excluded by intersection.
            assert scoped.allowed_keys == frozenset({"s-root", "s-child"})
            unscoped = resolve_scope(
                session,
                project_ids=("p1",),
                all_projects=False,
                subtree_roots=(),
            )
            assert unscoped.allowed_keys == frozenset({"s-root", "s-child"})
            everything = resolve_scope(
                session,
                project_ids=(),
                all_projects=True,
                subtree_roots=("s-root",),
            )
            assert everything.allowed_keys is not None
            assert "s-cross" in everything.allowed_keys
        finally:
            session.close()

    def test_scope_forms_are_mutually_exclusive(self, fdb):
        session = sessionmaker(bind=fdb.engine)()
        try:
            with pytest.raises(TrackerError):
                resolve_scope(
                    session,
                    project_ids=("p1",),
                    all_projects=True,
                    subtree_roots=(),
                )
            with pytest.raises(TrackerError):
                resolve_scope(
                    session,
                    project_ids=(),
                    all_projects=False,
                    subtree_roots=(),
                )
        finally:
            session.close()


# ---------------------------------------------------------------------------
# issue list parity through the shared builder
# ---------------------------------------------------------------------------


class TestListParityThroughSharedBuilder:
    @pytest.fixture
    def service_db(self, fdb, monkeypatch):
        from cli_agent_orchestrator.services import issue_tracker as tracker

        monkeypatch.setattr(tracker, "SessionLocal", sessionmaker(bind=fdb.engine))
        return fdb

    def _seed(self, db):
        _issue(
            db,
            "l-1",
            labels='["keep"]',
            status="open",
            severity="P1",
            component="core",
            observed_revision="v9",
        )
        _issue(
            db,
            "l-2",
            labels='["keep","drop"]',
            status="closed",
            severity="P2",
            kind="feature",
            observed_revision="v9",
        )
        _issue(db, "l-3")

    def test_observed_revision_exact_filter_is_available_on_list(self, service_db, monkeypatch):
        self._seed(service_db)
        result = list_issues(project_id="p1", kind="all", observed_revision="v9")
        assert sorted(i["key"] for i in result["issues"]) == ["l-1", "l-2"]
        repeated = list_issues(project_id="p1", kind="all", observed_revision=["v9"])
        assert repeated["total"] == 2

    def test_list_filtering_results_are_unchanged_by_builder_delegation(
        self, service_db, monkeypatch
    ):
        self._seed(service_db)
        combined = list_issues(
            project_id="p1",
            status=["open", "closed"],
            label=["keep"],
            without_label=["drop"],
        )
        assert [i["key"] for i in combined["issues"]] == ["l-1"]
        features_only = list_issues(project_id="p1", kind="feature")
        assert [i["key"] for i in features_only["issues"]] == ["l-2"]
        open_only = list_issues(project_id="p1", open_only=True)
        assert [i["key"] for i in open_only["issues"]] == ["l-3", "l-1"]
        unlabeled = list_issues(project_id="p1", unlabeled=True)
        assert [i["key"] for i in unlabeled["issues"]] == ["l-3"]
        # Default ordering (created_desc, id desc) is untouched, and the
        # default kind=bug isolation still hides the feature row.
        ordered = list_issues(project_id="p1")
        assert [i["key"] for i in ordered["issues"]] == ["l-3", "l-1"]

    def test_list_kind_all_and_none_remain_supported(self, service_db):
        self._seed(service_db)
        assert list_issues(project_id="p1", kind="all")["total"] == 3
        assert list_issues(project_id="p1", kind=None)["total"] == 3


class TestQueryEmptinessRule:
    def test_blank_and_punctuation_only_queries_count_as_empty(self):
        assert is_effectively_empty_query("")
        assert is_effectively_empty_query("   ")
        assert is_effectively_empty_query("!!! ??? --")
        assert not is_effectively_empty_query("deploy --dry-run")
