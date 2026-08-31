"""Project-scoped issue tracking.

The properties under test are the ones that make a shared issue log
trustworthy rather than merely present: an identifier resolves to exactly one
project, a key is never reused, every mutation leaves a trace, and a filing
that cannot be placed fails loudly instead of landing somewhere plausible.
"""

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import Base
from cli_agent_orchestrator.services import issue_tracker as tracker
from cli_agent_orchestrator.services.issue_tracker import TrackerError


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/tracker.db")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessionmaker(bind=engine))
    yield
    engine.dispose()


@pytest.fixture
def cao_system(tmp_path):
    conductor = tmp_path / "cao-conductor"
    fork = tmp_path / "cli-agent-orchestrator"
    conductor.mkdir()
    fork.mkdir()
    project = tracker.create_project(
        name="CAO System",
        project_id="cao-system",
        issue_prefix="cond",
        scopes=[
            {"kind": "path", "value": str(conductor)},
            {"kind": "path", "value": str(fork)},
            {"kind": "session", "value": "cao-p1-closure"},
        ],
    )
    return {"project": project, "conductor": conductor, "fork": fork}


class TestProjectCreation:
    def test_a_project_spans_many_paths_and_sessions(self, cao_system):
        detail = tracker.get_project("cao-system")
        kinds = sorted((s["kind"], s["value"]) for s in detail["scopes"])
        assert [k for k, _ in kinds] == ["path", "path", "session"]

    def test_the_slug_is_derived_from_the_name_when_not_given(self):
        created = tracker.create_project(name="Quest Scheduler")
        assert created["id"] == "quest-scheduler"

    def test_a_name_with_no_usable_characters_is_refused_not_invented(self):
        # Generating `project-1` here would be a naming decision the caller
        # never made, and it is the kind of id nobody can find later.
        with pytest.raises(TrackerError) as exc:
            tracker.create_project(name="???")
        assert exc.value.code == "invalid"

    def test_a_duplicate_project_id_is_refused(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.create_project(name="Something else", project_id="cao-system")
        assert exc.value.code == "conflict"

    def test_a_rejected_scope_leaves_no_half_built_project(self, tmp_path):
        tracker.create_project(name="First", scopes=[{"kind": "path", "value": str(tmp_path)}])
        with pytest.raises(TrackerError):
            tracker.create_project(name="Second", scopes=[{"kind": "path", "value": str(tmp_path)}])
        assert [p["id"] for p in tracker.list_projects()] == ["first"]


class TestScopeUniqueness:
    def test_one_path_cannot_belong_to_two_projects(self, cao_system):
        tracker.create_project(name="Other", project_id="other")
        with pytest.raises(TrackerError) as exc:
            tracker.add_scope("other", kind="path", value=str(cao_system["conductor"]))
        assert exc.value.code == "conflict"
        assert "cao-system" in exc.value.message

    def test_re_registering_the_same_scope_on_the_same_project_is_idempotent(self, cao_system):
        again = tracker.add_scope("cao-system", kind="path", value=str(cao_system["fork"]))
        assert again["created"] is False

    def test_scope_values_are_unique_across_kinds(self, cao_system):
        # A session literally named like a path is pathological, but the
        # uniqueness rule is what stops resolution order from deciding.
        tracker.create_project(name="Other", project_id="other")
        with pytest.raises(TrackerError):
            tracker.add_scope("other", kind="session", value=str(cao_system["conductor"]))


class TestScopeNormalisation:
    def test_a_trailing_separator_is_the_same_scope(self, tmp_path):
        tracker.create_project(name="P", project_id="p")
        tracker.add_scope("p", kind="path", value=str(tmp_path))
        again = tracker.add_scope("p", kind="path", value=str(tmp_path) + "/")
        assert again["created"] is False

    def test_ssh_and_https_remotes_are_one_scope(self):
        tracker.create_project(name="P", project_id="p")
        tracker.add_scope("p", kind="git_remote", value="git@github.com:colindmurray/cao.git")
        again = tracker.add_scope(
            "p", kind="git_remote", value="https://github.com/colindmurray/cao"
        )
        assert again["created"] is False

    def test_remote_credentials_are_never_stored(self):
        tracker.create_project(name="P", project_id="p")
        row = tracker.add_scope(
            "p", kind="git_remote", value="https://user:ghp_secret@github.com/o/r.git"
        )
        assert "ghp_secret" not in row["value"]
        assert row["value"] == "github.com/o/r"

    def test_a_relative_path_scope_is_refused(self):
        tracker.create_project(name="P", project_id="p")
        with pytest.raises(TrackerError):
            tracker.add_scope("p", kind="path", value="relative/dir")

    def test_an_unknown_scope_kind_is_refused(self):
        tracker.create_project(name="P", project_id="p")
        with pytest.raises(TrackerError):
            tracker.add_scope("p", kind="hostname", value="foundry")


class TestResolution:
    def test_an_explicit_project_wins(self, cao_system):
        got = tracker.resolve_project(project="cao-system", session="unknown-session")
        assert (got.project_id, got.matched_by) == ("cao-system", "explicit")

    def test_a_session_resolves_across_directories(self, cao_system, tmp_path):
        got = tracker.resolve_project(session="cao-p1-closure", cwd=str(tmp_path / "elsewhere"))
        assert (got.project_id, got.matched_by) == ("cao-system", "session")

    def test_a_subdirectory_resolves_to_its_project(self, cao_system):
        nested = cao_system["conductor"] / "conduct" / "lib"
        nested.mkdir(parents=True)
        got = tracker.resolve_project(cwd=str(nested))
        assert (got.project_id, got.matched_by) == ("cao-system", "path")

    def test_a_sibling_directory_sharing_a_name_prefix_does_not_match(self, cao_system, tmp_path):
        # `cao-conductor-worktrees` is a REAL sibling of `cao-conductor` on
        # this machine. String-prefix matching would file every worktree
        # issue into the wrong project and nobody would notice for weeks.
        sibling = tmp_path / "cao-conductor-worktrees"
        sibling.mkdir()
        got = tracker.resolve_project(cwd=str(sibling))
        assert got.project_id is None

    def test_the_most_specific_path_scope_wins(self, cao_system):
        gateway = cao_system["conductor"] / "gateway"
        gateway.mkdir()
        tracker.create_project(name="Gateway", project_id="gateway")
        # Same value is refused, so the specific project registers the deeper dir.
        tracker.add_scope("gateway", kind="path", value=str(gateway))
        got = tracker.resolve_project(cwd=str(gateway / "src"))
        assert got.project_id == "gateway"

    def test_session_beats_path(self, cao_system, tmp_path):
        other = tmp_path / "other-repo"
        other.mkdir()
        tracker.create_project(
            name="Other", project_id="other", scopes=[{"kind": "path", "value": str(other)}]
        )
        got = tracker.resolve_project(session="cao-p1-closure", cwd=str(other))
        assert got.project_id == "cao-system"

    def test_a_git_remote_resolves_when_no_path_matches(self, tmp_path):
        tracker.create_project(
            name="Aegix",
            project_id="aegix",
            scopes=[{"kind": "git_remote", "value": "git@github.com:g/aegix.git"}],
        )
        got = tracker.resolve_project(git_remote="https://github.com/g/aegix")
        assert (got.project_id, got.matched_by) == ("aegix", "git_remote")

    def test_an_unregistered_site_resolves_to_nothing_rather_than_raising(self, tmp_path):
        got = tracker.resolve_project(cwd=str(tmp_path / "nowhere"))
        assert got.as_dict() == {"project_id": None, "matched_by": None, "matched_value": None}

    def test_an_explicit_unknown_project_raises(self):
        with pytest.raises(TrackerError) as exc:
            tracker.resolve_project(project="no-such-project")
        assert exc.value.code == "not-found"


class TestIssueKeys:
    def test_keys_use_the_project_prefix_and_increment(self, cao_system):
        a = tracker.create_issue(project_id="cao-system", title="first")
        b = tracker.create_issue(project_id="cao-system", title="second")
        assert (a["key"], b["key"]) == ("cond-0001", "cond-0002")

    def test_a_deleted_issue_never_returns_its_key(self, cao_system):
        first = tracker.create_issue(project_id="cao-system", title="first")
        tracker.delete_issue(first["key"])
        second = tracker.create_issue(project_id="cao-system", title="second")
        # Recycling cond-0001 would silently repoint every commit message,
        # report and evidence path that already quotes it.
        assert second["key"] == "cond-0002"

    def test_an_explicit_key_is_preserved_and_advances_the_counter(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="migrated", key="cond-0242")
        nxt = tracker.create_issue(project_id="cao-system", title="new")
        assert nxt["key"] == "cond-0243"

    def test_an_explicit_key_below_the_counter_does_not_rewind_it(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="migrated", key="cond-0100")
        tracker.create_issue(project_id="cao-system", title="older", key="cond-0005")
        nxt = tracker.create_issue(project_id="cao-system", title="new")
        assert nxt["key"] == "cond-0101"

    def test_a_duplicate_explicit_key_is_refused(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a", key="cond-0007")
        with pytest.raises(TrackerError) as exc:
            tracker.create_issue(project_id="cao-system", title="b", key="cond-0007")
        assert exc.value.code == "conflict"

    def test_two_projects_keep_separate_numbering(self, cao_system):
        tracker.create_project(name="Quest Scheduler", project_id="qs", issue_prefix="qs")
        a = tracker.create_issue(project_id="cao-system", title="a")
        b = tracker.create_issue(project_id="qs", title="b")
        assert (a["key"], b["key"]) == ("cond-0001", "qs-0001")

    def test_renaming_the_prefix_leaves_existing_keys_alone(self, cao_system):
        old = tracker.create_issue(project_id="cao-system", title="a")
        tracker.update_project("cao-system", issue_prefix="cao")
        new = tracker.create_issue(project_id="cao-system", title="b")
        assert tracker.get_issue(old["key"])["key"] == "cond-0001"
        assert new["key"] == "cao-0002"


class TestFiling:
    def test_an_issue_filed_from_a_scoped_directory_lands_in_that_project(self, cao_system):
        issue = tracker.create_issue(
            title="conduct spawn warns about SKILL.md",
            cwd=str(cao_system["fork"] / "src"),
        )
        assert issue["project_id"] == "cao-system"
        assert issue["resolved_by"] == "path"

    def test_an_unresolvable_filing_site_is_refused(self, tmp_path):
        with pytest.raises(TrackerError) as exc:
            tracker.create_issue(title="orphan", cwd=str(tmp_path / "unknown"))
        assert exc.value.code == "unresolved"

    def test_an_empty_title_is_refused(self, cao_system):
        with pytest.raises(TrackerError):
            tracker.create_issue(project_id="cao-system", title="   ")

    def test_an_unknown_status_is_refused(self, cao_system):
        with pytest.raises(TrackerError):
            tracker.create_issue(project_id="cao-system", title="a", status="pending")

    def test_filing_records_the_creation_event(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", reporter="13e6fe47")
        events = tracker.get_issue(issue["key"])["events"]
        assert [(e["kind"], e["actor"]) for e in events] == [("created", "13e6fe47")]

    def test_a_terminal_status_at_filing_time_sets_closed_at(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", status="closed")
        assert issue["closed_at"] is not None


class TestEditing:
    def test_every_changed_field_leaves_an_audit_event(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.update_issue(
            issue["key"], actor="colin", status="in-progress", severity="P2", assignee="terra"
        )
        events = [e for e in tracker.get_issue(issue["key"])["events"] if e["kind"] == "field"]
        assert sorted(e["field"] for e in events) == ["assignee", "severity", "status"]
        assert all(e["actor"] == "colin" for e in events)

    def test_a_no_op_write_records_nothing(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", status="open")
        tracker.update_issue(issue["key"], status="open")
        assert [e["kind"] for e in tracker.get_issue(issue["key"])["events"]] == ["created"]

    def test_closing_stamps_closed_at_and_reopening_clears_it(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        closed = tracker.update_issue(issue["key"], status="closed")
        assert closed["closed_at"] is not None
        reopened = tracker.update_issue(issue["key"], status="open")
        assert reopened["closed_at"] is None

    def test_resolved_is_not_treated_as_closed(self, cao_system):
        # "a fix landed" and "somebody verified it" are different claims, and
        # collapsing them is how fixes get shipped unverified.
        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.update_issue(issue["key"], status="resolved")
        assert tracker.list_issues(project_id="cao-system", open_only=True)["total"] == 1

    def test_an_empty_string_clears_a_free_text_field(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", assignee="terra")
        assert tracker.update_issue(issue["key"], assignee="")["assignee"] is None

    def test_a_non_editable_field_is_refused(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError) as exc:
            tracker.update_issue(issue["key"], key="cond-9999")
        assert exc.value.code == "invalid"

    def test_project_id_cannot_be_edited_through_the_field_path(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError):
            tracker.update_issue(issue["key"], project_id="other")

    def test_an_issue_cannot_duplicate_itself(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError):
            tracker.update_issue(issue["key"], duplicate_of=issue["key"])

    def test_duplicate_of_must_name_a_real_issue(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError) as exc:
            tracker.update_issue(issue["key"], duplicate_of="cond-9999")
        assert exc.value.code == "not-found"


class TestLabels:
    def test_labels_are_deduplicated_and_order_preserved(self, cao_system):
        issue = tracker.create_issue(
            project_id="cao-system", title="a", labels=["quota", "flaky", "quota"]
        )
        assert issue["labels"] == ["quota", "flaky"]

    def test_a_comma_string_is_accepted(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", labels="quota, flaky")
        assert issue["labels"] == ["quota", "flaky"]

    def test_label_filtering_does_not_match_a_longer_label(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a", labels=["ui"])
        tracker.create_issue(project_id="cao-system", title="b", labels=["ui-polish"])
        got = tracker.list_issues(project_id="cao-system", label="ui")
        assert [i["title"] for i in got["issues"]] == ["a"]

    def test_too_many_labels_is_refused(self, cao_system):
        with pytest.raises(TrackerError):
            tracker.create_issue(
                project_id="cao-system", title="a", labels=[f"l{i}" for i in range(40)]
            )


class TestListingAndSearch:
    def test_total_is_the_unpaged_count(self, cao_system):
        for i in range(5):
            tracker.create_issue(project_id="cao-system", title=f"issue {i}")
        page = tracker.list_issues(project_id="cao-system", limit=2)
        assert page["total"] == 5
        assert len(page["issues"]) == 2

    def test_search_covers_title_body_key_and_failing_command(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="alpha", body="nothing")
        tracker.create_issue(
            project_id="cao-system", title="beta", failing_command="conduct spawn --lane x"
        )
        got = tracker.list_issues(project_id="cao-system", query="conduct spawn")
        assert [i["title"] for i in got["issues"]] == ["beta"]

    def test_search_covers_first_class_reproduction_steps(self, cao_system):
        tracker.create_issue(
            project_id="cao-system",
            title="intermittent reconnect",
            reproduction_steps="1. suspend the laptop\n2. resume while the pane is busy",
        )
        got = tracker.list_issues(project_id="cao-system", query="resume while the pane")
        assert [i["title"] for i in got["issues"]] == ["intermittent reconnect"]

    def test_repeated_label_filters_require_every_selected_label(self, cao_system):
        tracker.create_issue(
            project_id="cao-system", title="both", labels=["wayfinder:task", "initiative:alpha"]
        )
        tracker.create_issue(project_id="cao-system", title="one", labels=["wayfinder:task"])
        got = tracker.list_issues(
            project_id="cao-system", label=["wayfinder:task", "initiative:alpha"]
        )
        assert [i["title"] for i in got["issues"]] == ["both"]

    def test_open_only_excludes_terminal_statuses(self, cao_system):
        a = tracker.create_issue(project_id="cao-system", title="a")
        tracker.create_issue(project_id="cao-system", title="b")
        tracker.update_issue(a["key"], status="wontfix")
        assert tracker.list_issues(project_id="cao-system", open_only=True)["total"] == 1

    def test_listing_is_scoped_to_one_project(self, cao_system):
        tracker.create_project(name="Other", project_id="other")
        tracker.create_issue(project_id="cao-system", title="mine")
        tracker.create_issue(project_id="other", title="theirs")
        got = tracker.list_issues(project_id="cao-system")
        assert [i["title"] for i in got["issues"]] == ["mine"]

    def test_severity_order_puts_p1_first(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="low", severity="P4")
        tracker.create_issue(project_id="cao-system", title="high", severity="P1")
        got = tracker.list_issues(project_id="cao-system", order="severity")
        assert [i["title"] for i in got["issues"]] == ["high", "low"]

    def test_limit_is_bounded(self, cao_system):
        assert tracker.list_issues(project_id="cao-system", limit=100000)["limit"] == 500


class TestFirstClassReproduction:
    def test_create_read_update_and_clear_round_trip(self, cao_system):
        created = tracker.create_issue(
            project_id="cao-system", title="a", reproduction_steps="1. run it"
        )
        assert created["reproduction_steps"] == "1. run it"
        updated = tracker.update_issue(created["key"], reproduction_steps="1. run it twice")
        assert updated["reproduction_steps"] == "1. run it twice"
        with pytest.raises(TrackerError):
            tracker.update_issue(created["key"], reproduction_steps="")
        cleared = tracker.update_issue(created["key"], reproduction_steps="", force=True)
        assert cleared["reproduction_steps"] is None
        detail = tracker.get_issue(created["key"])
        events = [e for e in detail["events"] if e["field"] == "reproduction_steps"]
        assert len(events) == 2


class TestAssignmentAndWorkContext:
    def test_reassignment_preserves_the_previous_owner_unless_overridden(self, cao_system):
        issue = tracker.create_issue(
            project_id="cao-system",
            title="handoff",
            assignee="codex:sess-1",
            collaborators=["colin"],
        )
        handed_off = tracker.update_issue(issue["key"], assignee="claude_code:sess-2")
        assert handed_off["assignee"] == "claude_code:sess-2"
        assert handed_off["collaborators"] == ["colin", "codex:sess-1"]

        separate = tracker.create_issue(
            project_id="cao-system",
            title="takeover",
            assignee="codex:sess-3",
            collaborators=["colin"],
        )
        taken_over = tracker.update_issue(
            separate["key"],
            assignee="claude_code:sess-4",
            drop_previous_assignee=True,
        )
        assert taken_over["assignee"] == "claude_code:sess-4"
        assert taken_over["collaborators"] == ["colin"]

    def test_repeatable_context_round_trips_and_is_audited(self, cao_system):
        created = tracker.create_issue(
            project_id="cao-system",
            title="a",
            collaborators=["codex:sess-1", "claude_code:sess-2"],
            branches=["fix/a", "review/a"],
            worktrees=["/tmp/wt-a", "/tmp/wt-review"],
            pull_requests=["https://github.com/o/r/pull/1", "o/r#2"],
        )
        assert created["collaborators"] == ["codex:sess-1", "claude_code:sess-2"]
        assert created["branches"] == ["fix/a", "review/a"]
        assert created["worktrees"] == ["/tmp/wt-a", "/tmp/wt-review"]
        assert created["pull_requests"] == ["https://github.com/o/r/pull/1", "o/r#2"]

        updated = tracker.update_issue(created["key"], collaborators=["codex:sess-1"], branches=[])
        assert updated["collaborators"] == ["codex:sess-1"]
        assert updated["branches"] == []
        events = tracker.get_issue(created["key"])["events"]
        assert {e["field"] for e in events if e["kind"] == "field"} >= {
            "collaborators",
            "branches",
        }

    def test_in_progress_requires_an_assignee_unless_explicitly_forced(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.create_issue(project_id="cao-system", title="a", status="in-progress")
        assert exc.value.code == "invalid"

        forced = tracker.create_issue(
            project_id="cao-system", title="forced", status="in-progress", force=True
        )
        assert forced["assignee"] is None

        issue = tracker.create_issue(project_id="cao-system", title="normal")
        with pytest.raises(TrackerError):
            tracker.update_issue(issue["key"], status="in-progress")
        active = tracker.update_issue(issue["key"], status="in-progress", assignee="colin")
        assert (active["status"], active["assignee"]) == ("in-progress", "colin")
        with pytest.raises(TrackerError):
            tracker.update_issue(issue["key"], assignee="")
        assert tracker.update_issue(issue["key"], assignee="", force=True)["assignee"] is None


class TestFieldOptions:
    def test_search_is_bounded_and_reports_counts(self, cao_system):
        tracker.create_issue(
            project_id="cao-system", title="a", component="dashboard", labels=["initiative:ux"]
        )
        closed = tracker.create_issue(
            project_id="cao-system", title="b", component="dashboard", labels=["initiative:ux"]
        )
        tracker.update_issue(closed["key"], status="closed")
        tracker.create_issue(
            project_id="cao-system", title="c", component="conduct", labels=["unrelated"]
        )

        components = tracker.field_options("cao-system", field="component", query="dash", limit=1)
        assert components["matching_total"] == 1
        assert components["options"] == [{"value": "dashboard", "total": 2, "open": 1}]
        labels = tracker.field_options("cao-system", field="label", query="initiative")
        assert labels["options"] == [{"value": "initiative:ux", "total": 2, "open": 1}]

    def test_repeatable_work_context_has_searchable_options(self, cao_system):
        tracker.create_issue(
            project_id="cao-system",
            title="a",
            collaborators=["codex:sess-a"],
            branches=["fix/searchable-context"],
            worktrees=["/tmp/context-a"],
            pull_requests=["o/r#42"],
        )
        assert (
            tracker.field_options("cao-system", field="collaborator", query="sess")["options"][0][
                "value"
            ]
            == "codex:sess-a"
        )
        assert (
            tracker.field_options("cao-system", field="pull_request", query="#42")["options"][0][
                "value"
            ]
            == "o/r#42"
        )

    def test_unknown_option_field_is_refused(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.field_options("cao-system", field="status")
        assert exc.value.code == "invalid"


class TestCommentsAndLinks:
    def test_a_comment_is_stored_and_audited(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.add_comment(issue["key"], body="reproduced on main", author="colin")
        detail = tracker.get_issue(issue["key"])
        assert [c["body"] for c in detail["comments"]] == ["reproduced on main"]
        assert any(e["kind"] == "comment" for e in detail["events"])

    def test_an_empty_comment_is_refused(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError):
            tracker.add_comment(issue["key"], body="  ")

    def test_links_appear_on_both_issues(self, cao_system):
        a = tracker.create_issue(project_id="cao-system", title="a")
        b = tracker.create_issue(project_id="cao-system", title="b")
        tracker.add_link(a["key"], to_key=b["key"], kind="blocks")
        assert len(tracker.get_issue(a["key"])["links"]) == 1
        assert len(tracker.get_issue(b["key"])["links"]) == 1

    def test_a_duplicate_link_is_idempotent(self, cao_system):
        a = tracker.create_issue(project_id="cao-system", title="a")
        b = tracker.create_issue(project_id="cao-system", title="b")
        first = tracker.add_link(a["key"], to_key=b["key"], kind="blocks")
        again = tracker.add_link(a["key"], to_key=b["key"], kind="blocks")
        assert again["created"] is False
        assert again["id"] == first["id"]
        assert again["from_updated_at"] == tracker.get_issue(a["key"])["updated_at"]
        assert again["to_updated_at"] == tracker.get_issue(b["key"])["updated_at"]

    def test_a_link_to_a_missing_issue_is_refused(self, cao_system):
        a = tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError):
            tracker.add_link(a["key"], to_key="cond-9999", kind="relates")

    def test_deleting_an_issue_removes_its_comments_events_and_links(self, cao_system):
        a = tracker.create_issue(project_id="cao-system", title="a")
        b = tracker.create_issue(project_id="cao-system", title="b")
        tracker.add_comment(a["key"], body="note")
        tracker.add_link(a["key"], to_key=b["key"], kind="relates")
        tracker.delete_issue(a["key"])
        assert tracker.get_issue(b["key"])["links"] == []


class TestAuditPublishFences:
    """Fences are at the write seam: a mutation after review cannot publish a
    comment, relation, or status effect against the stale review."""

    def test_fenced_important_comment_returns_exact_effect_and_parent_clock(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        result = tracker.add_comment(
            issue["key"],
            body="reviewed finding",
            important=True,
            expected_updated_at=issue["updated_at"],
        )
        detail = tracker.get_issue(issue["key"])
        assert result["id"] == detail["comments"][0]["id"]
        assert result["important"] is True
        assert result["updated_at"] == detail["updated_at"]
        event = next(e for e in detail["events"] if e["id"] == result["effect_id"])
        assert event["kind"] == "comment"

    def test_comment_mutation_after_precheck_refuses_without_partial_comment_or_event(
        self, cao_system, monkeypatch
    ):
        import threading

        issue = tracker.create_issue(project_id="cao-system", title="a")
        at_write, release = threading.Event(), threading.Event()
        real_now = tracker._utcnow
        paused = False

        def pause_after_precheck():
            nonlocal paused
            if not paused:
                paused = True
                at_write.set()
                assert release.wait(timeout=5)
            return real_now()

        monkeypatch.setattr(tracker, "_utcnow", pause_after_precheck)
        outcomes = []

        def publish_comment():
            try:
                tracker.add_comment(
                    issue["key"], body="must not land", expected_updated_at=issue["updated_at"]
                )
            except TrackerError as exc:
                outcomes.append(exc)

        worker = threading.Thread(target=publish_comment)
        worker.start()
        assert at_write.wait(timeout=5)
        tracker.update_issue(issue["key"], body="concurrent review note")
        release.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert [exc.code for exc in outcomes] == ["conflict"]
        detail = tracker.get_issue(issue["key"])
        assert detail["comments"] == []
        assert [e["kind"] for e in detail["events"]] == ["created", "field"]

    def test_unfenced_comment_path_remains_compatible(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        result = tracker.add_comment(issue["key"], body="ordinary caller")
        assert result["id"] > 0
        assert result["updated_at"] == tracker.get_issue(issue["key"])["updated_at"]

    def test_fenced_link_returns_both_clocks_and_endpoint_effects(self, cao_system):
        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        result = tracker.add_link(
            source["key"],
            to_key=target["key"],
            kind="relates",
            expected_from_updated_at=source["updated_at"],
            expected_to_updated_at=target["updated_at"],
            action_key="fenced-link-success",
        )
        assert result["id"] > 0
        assert len(result["effect_ids"]) == 2
        assert result["from_updated_at"] == tracker.get_issue(source["key"])["updated_at"]
        assert result["to_updated_at"] == tracker.get_issue(target["key"])["updated_at"]
        assert all(
            any(e["id"] == effect_id for e in tracker.get_issue(key)["events"])
            for key, effect_id in zip((source["key"], target["key"]), result["effect_ids"])
        )

    @pytest.mark.parametrize("mutated_side", ["from", "to"])
    def test_link_mutation_after_both_prechecks_refuses_without_partial_effect(
        self, cao_system, monkeypatch, mutated_side
    ):
        import threading

        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        at_write, release = threading.Event(), threading.Event()
        real_now = tracker._utcnow
        paused = False

        def pause_after_prechecks():
            nonlocal paused
            if not paused:
                paused = True
                at_write.set()
                assert release.wait(timeout=5)
            return real_now()

        monkeypatch.setattr(tracker, "_utcnow", pause_after_prechecks)
        outcomes = []

        def publish_link():
            try:
                tracker.add_link(
                    source["key"],
                    to_key=target["key"],
                    kind="relates",
                    expected_from_updated_at=source["updated_at"],
                    expected_to_updated_at=target["updated_at"],
                    action_key=f"fenced-link-race-{mutated_side}",
                )
            except TrackerError as exc:
                outcomes.append(exc)

        worker = threading.Thread(target=publish_link)
        worker.start()
        assert at_write.wait(timeout=5)
        mutated = source if mutated_side == "from" else target
        tracker.update_issue(mutated["key"], body=f"{mutated_side} changed")
        release.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert [exc.code for exc in outcomes] == ["conflict"]
        assert tracker.get_issue(source["key"])["links"] == []
        assert tracker.get_issue(target["key"])["links"] == []
        assert not [
            event
            for key in (source["key"], target["key"])
            for event in tracker.get_issue(key)["events"]
            if event["kind"] == "link"
        ]

    def test_stale_second_endpoint_clock_refuses_the_link(self, cao_system):
        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        tracker.update_issue(target["key"], body="target changed")
        with pytest.raises(TrackerError) as exc:
            tracker.add_link(
                source["key"],
                to_key=target["key"],
                kind="relates",
                expected_from_updated_at=source["updated_at"],
                expected_to_updated_at=target["updated_at"],
                action_key="stale-second-endpoint",
            )
        assert exc.value.code == "conflict"
        assert exc.value.details["endpoint"] == "to"
        assert tracker.get_issue(source["key"])["links"] == []


class TestTrackerWriteReceiptsAndAvailability:
    def test_fenced_status_retries_an_unrelated_busy_writer_with_the_original_clock(
        self, cao_system, monkeypatch
    ):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        real_apply = tracker._apply_issue_update
        attempts = 0

        def locked_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OperationalError(
                    "UPDATE tracker_issues", {}, RuntimeError("database is locked")
                )
            return real_apply(*args, **kwargs)

        monkeypatch.setattr(tracker, "_apply_issue_update", locked_once)
        monkeypatch.setattr(tracker.time, "sleep", lambda _delay: None)
        result = tracker.update_issue(
            issue["key"], status="resolved", expected_updated_at=issue["updated_at"]
        )
        assert attempts == 2
        assert result["status"] == "resolved"

    def test_fenced_busy_exhaustion_is_distinct_from_a_stale_clock(self, cao_system, monkeypatch):
        issue = tracker.create_issue(project_id="cao-system", title="a")

        def always_locked(*_args, **_kwargs):
            raise OperationalError("UPDATE tracker_issues", {}, RuntimeError("database is locked"))

        monkeypatch.setattr(tracker, "_apply_issue_update", always_locked)
        monkeypatch.setattr(tracker.time, "sleep", lambda _delay: None)
        with pytest.raises(TrackerError) as exc:
            tracker.update_issue(
                issue["key"], status="resolved", expected_updated_at=issue["updated_at"]
            )
        assert exc.value.code == "busy"
        assert exc.value.details == {
            "retryable": True,
            "observed_updated_at": issue["updated_at"],
        }

    def test_fenced_link_replay_returns_its_exact_committed_receipt(self, cao_system):
        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        kwargs = {
            "to_key": target["key"],
            "kind": "relates",
            "expected_from_updated_at": source["updated_at"],
            "expected_to_updated_at": target["updated_at"],
            "action_key": "audit-publish-41",
        }
        committed = tracker.add_link(source["key"], **kwargs)
        # The visible clocks may move long after the first response was lost;
        # replay is bound to the durable action receipt, not the live edge.
        tracker.update_issue(source["key"], body="later source edit")
        tracker.update_issue(target["key"], body="later target edit")
        replayed = tracker.add_link(source["key"], **kwargs)
        assert replayed["replayed"] is True
        assert replayed["created"] is True
        assert replayed["id"] == committed["id"]
        assert replayed["from_updated_at"] == committed["from_updated_at"]
        assert replayed["to_updated_at"] == committed["to_updated_at"]
        assert replayed["effect_ids"] == committed["effect_ids"]
        assert (
            len(
                [
                    event
                    for issue_key in (source["key"], target["key"])
                    for event in tracker.get_issue(issue_key)["events"]
                    if event["kind"] == "link"
                ]
            )
            == 2
        )

    @pytest.mark.parametrize("removal", ["link", "endpoint", "project"])
    def test_fenced_link_receipt_replays_after_its_live_graph_is_removed(self, cao_system, removal):
        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        request = {
            "to_key": target["key"],
            "kind": "relates",
            "expected_from_updated_at": source["updated_at"],
            "expected_to_updated_at": target["updated_at"],
            "action_key": f"deleted-graph-replay-{removal}",
        }
        committed = tracker.add_link(source["key"], **request)

        if removal == "link":
            tracker.remove_link(committed["id"])
        elif removal == "endpoint":
            tracker.delete_issue(target["key"])
        else:
            tracker.delete_project("cao-system", force=True)

        replayed = tracker.add_link(source["key"], **request)
        assert replayed == {**committed, "replayed": True}

    def test_action_key_replay_refuses_a_different_fenced_request_identity(self, cao_system):
        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        request = {
            "to_key": target["key"],
            "kind": "relates",
            "expected_from_updated_at": source["updated_at"],
            "expected_to_updated_at": target["updated_at"],
            "action_key": "exact-request-identity",
        }
        tracker.add_link(source["key"], **request)

        with pytest.raises(TrackerError) as exc:
            tracker.add_link(
                source["key"],
                to_key=target["key"],
                kind="relates",
                expected_from_updated_at=tracker.get_issue(source["key"])["updated_at"],
                expected_to_updated_at=tracker.get_issue(target["key"])["updated_at"],
                action_key="exact-request-identity",
            )
        assert exc.value.code == "conflict"
        assert "different request" in exc.value.message

    def test_fenced_link_requires_an_action_key_for_loss_safe_replay(self, cao_system):
        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        with pytest.raises(TrackerError) as exc:
            tracker.add_link(
                source["key"],
                to_key=target["key"],
                kind="relates",
                expected_from_updated_at=source["updated_at"],
                expected_to_updated_at=target["updated_at"],
            )
        assert exc.value.code == "invalid"
        assert "action_key" in exc.value.message

    def test_action_key_never_adopts_another_workers_existing_link(self, cao_system):
        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        tracker.add_link(source["key"], to_key=target["key"], kind="relates")
        source = tracker.get_issue(source["key"])
        target = tracker.get_issue(target["key"])
        with pytest.raises(TrackerError) as exc:
            tracker.add_link(
                source["key"],
                to_key=target["key"],
                kind="relates",
                expected_from_updated_at=source["updated_at"],
                expected_to_updated_at=target["updated_at"],
                action_key="audit-publish-42",
            )
        assert exc.value.code == "conflict"
        assert "do not adopt" in exc.value.message

    def test_unlink_bumps_both_endpoints_and_mirrors_the_audit_effect(self, cao_system):
        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        link = tracker.add_link(source["key"], to_key=target["key"], kind="blocks")
        source = tracker.get_issue(source["key"])
        target = tracker.get_issue(target["key"])
        result = tracker.remove_link(
            link["id"],
            expected_from_updated_at=source["updated_at"],
            expected_to_updated_at=target["updated_at"],
        )
        assert len(result["effect_ids"]) == 2
        assert result["from_updated_at"] == tracker.get_issue(source["key"])["updated_at"]
        assert result["to_updated_at"] == tracker.get_issue(target["key"])["updated_at"]
        for issue_key, effect_id, other_key in (
            (source["key"], result["effect_ids"][0], target["key"]),
            (target["key"], result["effect_ids"][1], source["key"]),
        ):
            event = next(
                event
                for event in tracker.get_issue(issue_key)["events"]
                if event["id"] == effect_id
            )
            assert (event["kind"], event["old_value"]) == ("unlink", other_key)

    @pytest.mark.parametrize("mutated_side", ["from", "to"])
    def test_unlink_refuses_a_late_endpoint_mutation_without_partial_effect(
        self, cao_system, monkeypatch, mutated_side
    ):
        import threading

        source = tracker.create_issue(project_id="cao-system", title="source")
        target = tracker.create_issue(project_id="cao-system", title="target")
        link = tracker.add_link(source["key"], to_key=target["key"], kind="relates")
        source = tracker.get_issue(source["key"])
        target = tracker.get_issue(target["key"])
        at_write, release = threading.Event(), threading.Event()
        real_now = tracker._utcnow
        paused = False

        def pause_after_precheck():
            nonlocal paused
            if not paused:
                paused = True
                at_write.set()
                assert release.wait(timeout=5)
            return real_now()

        monkeypatch.setattr(tracker, "_utcnow", pause_after_precheck)
        outcomes = []

        def unlink():
            try:
                tracker.remove_link(
                    link["id"],
                    expected_from_updated_at=source["updated_at"],
                    expected_to_updated_at=target["updated_at"],
                )
            except TrackerError as exc:
                outcomes.append(exc)

        worker = threading.Thread(target=unlink)
        worker.start()
        assert at_write.wait(timeout=5)
        changed = source if mutated_side == "from" else target
        tracker.update_issue(changed["key"], body="late endpoint update")
        release.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert [exc.code for exc in outcomes] == ["conflict"]
        assert tracker.get_issue(source["key"])["links"]
        assert tracker.get_issue(target["key"])["links"]
        assert not [
            event
            for issue_key in (source["key"], target["key"])
            for event in tracker.get_issue(issue_key)["events"]
            if event["kind"] == "unlink"
        ]

    def test_deleting_an_issue_bumps_and_audits_each_surviving_peer(self, cao_system):
        removed = tracker.create_issue(project_id="cao-system", title="removed")
        peer = tracker.create_issue(project_id="cao-system", title="peer")
        tracker.add_link(removed["key"], to_key=peer["key"], kind="relates")
        before = tracker.get_issue(peer["key"])
        result = tracker.delete_issue(removed["key"])
        after = tracker.get_issue(peer["key"])
        assert result["peer_updated_at"] == {peer["key"]: after["updated_at"]}
        assert after["updated_at"] >= before["updated_at"]
        unlink = [event for event in after["events"] if event["kind"] == "unlink"]
        assert len(unlink) == 1
        assert unlink[0]["old_value"] == removed["key"]
        assert result["effect_ids"] == [unlink[0]["id"]]

    def test_multifield_close_returns_the_status_effect_as_effect_id(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        result = tracker.update_issue(
            issue["key"],
            status="closed",
            resolution="fixed",
            expected_updated_at=issue["updated_at"],
        )
        assert len(result["effect_ids"]) == 2
        event = next(
            event
            for event in tracker.get_issue(issue["key"])["events"]
            if event["id"] == result["effect_id"]
        )
        assert event["field"] == "status"


class TestProjectDeletion:
    def test_a_project_holding_issues_is_not_deleted_by_accident(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError) as exc:
            tracker.delete_project("cao-system")
        assert exc.value.code == "conflict"

    def test_force_deletes_the_issues_too(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a")
        got = tracker.delete_project("cao-system", force=True)
        assert got["issues_deleted"] == 1
        assert tracker.list_projects() == []

    def test_force_delete_bumps_and_audits_a_peer_outside_the_project(self, cao_system):
        tracker.create_project(name="Peer", project_id="peer", issue_prefix="peer")
        removed = tracker.create_issue(project_id="cao-system", title="removed")
        peer = tracker.create_issue(project_id="peer", title="peer")
        tracker.add_link(removed["key"], to_key=peer["key"], kind="relates")
        before = tracker.get_issue(peer["key"])

        result = tracker.delete_project("cao-system", force=True)

        after = tracker.get_issue(peer["key"])
        assert result["peer_updated_at"] == {peer["key"]: after["updated_at"]}
        assert after["updated_at"] >= before["updated_at"]
        event = next(event for event in after["events"] if event["kind"] == "unlink")
        assert (event["old_value"], result["effect_ids"]) == (removed["key"], [event["id"]])

    def test_archiving_hides_a_project_without_touching_its_issues(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a")
        tracker.update_project("cao-system", status="archived")
        assert tracker.list_projects() == []
        assert len(tracker.list_projects(include_archived=True)) == 1
        assert tracker.list_issues(project_id="cao-system")["total"] == 1


class TestTimestamps:
    def test_stored_timestamps_serialize_as_explicit_utc(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        assert issue["created_at"].endswith("Z")

    def test_a_supplied_creation_stamp_is_preserved(self, cao_system):
        when = datetime(2026, 7, 21, 11, 44, 29, tzinfo=timezone.utc)
        issue = tracker.create_issue(
            project_id="cao-system", title="a", key="cond-0025", created_at=when
        )
        assert issue["created_at"] == "2026-07-21T11:44:29Z"


class TestMarkdownExport:
    def test_the_export_carries_key_severity_and_metadata(self, cao_system):
        tracker.create_issue(
            project_id="cao-system",
            title="event-mirror lock contention logs full traceback",
            severity="P2",
            reporter="13e6fe47",
            failing_command="python3 -B probes.py",
            reproduction_steps="1. run the probe\n2. observe the traceback",
            body="Independent validation confirmed the mirror stays bounded.",
        )
        rendered = tracker.render_markdown("cao-system")
        assert "## cond-0001 — [P2] event-mirror lock contention" in rendered
        assert "- **reporter:** 13e6fe47" in rendered
        assert "python3 -B probes.py" in rendered
        assert "1. run the probe\n2. observe the traceback" in rendered

    def test_the_export_defaults_to_open_issues(self, cao_system):
        a = tracker.create_issue(project_id="cao-system", title="done")
        tracker.create_issue(project_id="cao-system", title="pending")
        tracker.update_issue(a["key"], status="closed")
        rendered = tracker.render_markdown("cao-system")
        assert "pending" in rendered
        assert "done" not in rendered


class TestStats:
    def test_stats_break_down_by_status_severity_and_component(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a", severity="P1", component="conduct")
        tracker.create_issue(project_id="cao-system", title="b", severity="P1", component="fork")
        got = tracker.stats("cao-system")
        assert got["total"] == 2
        assert got["by_severity"]["P1"] == 2
        assert sorted(got["by_component"]) == ["conduct", "fork"]


class TestPrefixUniqueness:
    """Issue keys are unique across the installation, not per project.

    `cond-0242` appears in commit messages, reports and evidence paths, so it
    has to mean one thing. Two projects sharing a prefix would collide at
    key-allocation time with a conflict naming a project the caller never
    mentioned — found by importing a 208-entry ledger into a second project
    that also used `cond`.
    """

    def test_a_second_project_cannot_claim_a_used_prefix(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.create_project(name="Other", project_id="other", issue_prefix="cond")
        assert exc.value.code == "conflict"
        assert "cao-system" in exc.value.message

    def test_a_project_cannot_be_renamed_onto_a_used_prefix(self, cao_system):
        tracker.create_project(name="Other", project_id="other", issue_prefix="oth")
        with pytest.raises(TrackerError):
            tracker.update_project("other", issue_prefix="cond")

    def test_a_project_may_keep_its_own_prefix_through_an_unrelated_edit(self, cao_system):
        # The uniqueness check must exclude the project being edited, or every
        # rename would refuse itself.
        updated = tracker.update_project("cao-system", issue_prefix="cond", name="Renamed")
        assert updated["issue_prefix"] == "cond"

    def test_a_freed_prefix_can_be_reclaimed(self, cao_system):
        tracker.delete_project("cao-system", force=True)
        reclaimed = tracker.create_project(name="Successor", project_id="succ", issue_prefix="cond")
        assert reclaimed["issue_prefix"] == "cond"


class TestExportCompleteness:
    def test_the_export_pages_past_the_list_page_size(self, cao_system):
        # `list_issues` caps a page at 500. This export replaces a file that
        # held every entry, so a cap here would silently drop issue 501 onward
        # and the rendered log would look complete.
        for i in range(505):
            tracker.create_issue(project_id="cao-system", title=f"issue {i}")
        rendered = tracker.render_markdown("cao-system")
        assert rendered.count("\n## cond-") == 505
        assert "505 issue(s)" in rendered


class TestPrefixNamespaceIsTheKeyTable:
    """A vacated prefix is not a free prefix.

    Found by independent review. Ownership was checked against the PROJECT
    table, so a project that renamed its prefix left its issues holding those
    keys while the prefix read as unowned. Giving it to a new project made every
    filing collide — and because the counter bump shares a transaction with the
    insert, the rollback took the bump too, so the counter never advanced and
    the wedge was permanent: HTTP 500 on every filing, forever.
    """

    def test_a_vacated_prefix_is_still_refused_while_keys_hold_it(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a")
        tracker.update_project("cao-system", issue_prefix="cs")
        with pytest.raises(TrackerError) as exc:
            tracker.create_project(name="Beta", project_id="beta", issue_prefix="cond")
        assert exc.value.code == "conflict"
        assert "cond-0001" in exc.value.message

    def test_the_refusal_names_a_key_and_its_project(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a")
        tracker.update_project("cao-system", issue_prefix="cs")
        with pytest.raises(TrackerError) as exc:
            tracker.create_project(name="Beta", project_id="beta", issue_prefix="cond")
        assert "cao-system" in exc.value.message

    def test_a_prefix_with_no_keys_left_is_reclaimable(self, cao_system):
        # Vacated AND unused must stay available, or a typo'd prefix would be
        # burned permanently.
        tracker.update_project("cao-system", issue_prefix="cs")
        got = tracker.create_project(name="Beta", project_id="beta", issue_prefix="cond")
        assert got["issue_prefix"] == "cond"

    def test_force_deleting_a_project_frees_its_prefix_with_its_keys(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a")
        tracker.delete_project("cao-system", force=True)
        got = tracker.create_project(name="Successor", project_id="succ", issue_prefix="cond")
        assert got["issue_prefix"] == "cond"

    def test_a_project_keeps_its_own_prefix_through_an_unrelated_edit(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a")
        updated = tracker.update_project("cao-system", name="Renamed", issue_prefix="cond")
        assert updated["issue_prefix"] == "cond"


class TestDuplicateScopesInOneRequest:
    def test_the_same_value_twice_is_a_conflict_not_a_500(self, tmp_path):
        with pytest.raises(TrackerError) as exc:
            tracker.create_project(
                name="P",
                project_id="p",
                scopes=[
                    {"kind": "path", "value": str(tmp_path)},
                    {"kind": "path", "value": str(tmp_path) + "/"},
                ],
            )
        assert exc.value.code == "conflict"
        assert "twice" in exc.value.message


class TestLabelFilterWildcards:
    """`%` and `_` are LIKE wildcards and legal label characters."""

    def test_an_underscore_label_does_not_match_every_single_character_label(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a", labels=["_"])
        tracker.create_issue(project_id="cao-system", title="b", labels=["x"])
        got = tracker.list_issues(project_id="cao-system", label="_")
        assert [i["title"] for i in got["issues"]] == ["a"]

    def test_a_percent_label_does_not_match_everything(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a", labels=["100%"])
        tracker.create_issue(project_id="cao-system", title="b", labels=["other"])
        got = tracker.list_issues(project_id="cao-system", label="%")
        assert got["total"] == 0


class TestMapMembership:
    """`part-of`: directed child -> parent/map membership (cond-0394).

    The wayfinder skill's map is one issue; its tickets are child issues. The
    membership edge is a first-class directed link so the tracker itself can
    answer "what belongs to this map" instead of every agent reconstructing it
    from bodies.
    """

    def test_a_ticket_joins_its_map_with_part_of(self, cao_system):
        map_issue = tracker.create_issue(
            project_id="cao-system", title="map", labels=["wayfinder:map"]
        )
        ticket = tracker.create_issue(project_id="cao-system", title="ticket")
        link = tracker.add_link(ticket["key"], to_key=map_issue["key"], kind="part-of")
        assert link["created"] is True
        assert (link["from_key"], link["to_key"], link["kind"]) == (
            ticket["key"],
            map_issue["key"],
            "part-of",
        )
        assert tracker.get_issue(map_issue["key"])["links"][0]["kind"] == "part-of"

    def test_children_returns_direct_members_in_creation_order(self, cao_system):
        m = tracker.create_issue(project_id="cao-system", title="map")
        a = tracker.create_issue(project_id="cao-system", title="a")
        b = tracker.create_issue(project_id="cao-system", title="b")
        # Linked in reverse creation order: the ordering must come from the
        # issue records, not from when the edge was wired.
        tracker.add_link(b["key"], to_key=m["key"], kind="part-of")
        tracker.add_link(a["key"], to_key=m["key"], kind="part-of")
        got = tracker.list_children(m["key"])
        assert [c["title"] for c in got["children"]] == ["a", "b"]

    def test_children_excludes_non_members_and_grandchildren(self, cao_system):
        m = tracker.create_issue(project_id="cao-system", title="map")
        child = tracker.create_issue(project_id="cao-system", title="child")
        grand = tracker.create_issue(project_id="cao-system", title="grandchild")
        tracker.create_issue(project_id="cao-system", title="unrelated")
        tracker.add_link(child["key"], to_key=m["key"], kind="part-of")
        tracker.add_link(grand["key"], to_key=child["key"], kind="part-of")
        got = tracker.list_children(m["key"])
        # Direct membership only: no transitive closure.
        assert [c["title"] for c in got["children"]] == ["child"]

    def test_a_blocks_link_is_not_membership(self, cao_system):
        m = tracker.create_issue(project_id="cao-system", title="map")
        t = tracker.create_issue(project_id="cao-system", title="t")
        tracker.add_link(t["key"], to_key=m["key"], kind="blocks")
        assert tracker.list_children(m["key"])["children"] == []

    def test_children_of_a_missing_map_is_not_found(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.list_children("cond-9999")
        assert exc.value.code == "not-found"


def _map_with_tickets(cao_system, titles=("a", "b", "c")):
    m = tracker.create_issue(project_id="cao-system", title="map")
    tickets = [tracker.create_issue(project_id="cao-system", title=t) for t in titles]
    for t in tickets:
        tracker.add_link(t["key"], to_key=m["key"], kind="part-of")
    return m, tickets


class TestFrontier:
    """The frontier: direct children that are nonterminal, unassigned, and
    have no nonterminal incoming blocker (cond-0394).

    Every case constructs the state from canonical issue/link records — there
    is no derived "blocked" flag anywhere to get out of sync.
    """

    def test_all_fresh_children_are_takeable_oldest_first(self, cao_system):
        m, _ = _map_with_tickets(cao_system)
        got = tracker.frontier(m["key"])
        assert [t["title"] for t in got["frontier"]] == ["a", "b", "c"]

    def test_a_claimed_ticket_leaves_the_frontier(self, cao_system):
        m, tickets = _map_with_tickets(cao_system)
        tracker.update_issue(tickets[1]["key"], assignee="terra")
        got = tracker.frontier(m["key"])
        assert [t["title"] for t in got["frontier"]] == ["a", "c"]

    def test_a_terminal_ticket_leaves_the_frontier(self, cao_system):
        m, tickets = _map_with_tickets(cao_system)
        tracker.update_issue(tickets[0]["key"], status="closed")
        got = tracker.frontier(m["key"])
        assert [t["title"] for t in got["frontier"]] == ["b", "c"]

    def test_a_resolved_ticket_is_still_on_the_frontier(self, cao_system):
        # `resolved` is deliberately NOT terminal: a fix landed but nobody
        # verified it, so the ticket still needs a session's attention.
        m, tickets = _map_with_tickets(cao_system)
        tracker.update_issue(tickets[0]["key"], status="resolved")
        got = tracker.frontier(m["key"])
        assert [t["title"] for t in got["frontier"]] == ["a", "b", "c"]

    def test_a_ticket_with_an_open_blocker_is_not_frontier(self, cao_system):
        m, tickets = _map_with_tickets(cao_system)
        blocker = tracker.create_issue(project_id="cao-system", title="blocker")
        tracker.add_link(blocker["key"], to_key=tickets[0]["key"], kind="blocks")
        got = tracker.frontier(m["key"])
        assert [t["title"] for t in got["frontier"]] == ["b", "c"]

    def test_closing_the_blocker_returns_the_ticket_to_the_frontier(self, cao_system):
        m, tickets = _map_with_tickets(cao_system)
        blocker = tracker.create_issue(project_id="cao-system", title="blocker")
        tracker.add_link(blocker["key"], to_key=tickets[0]["key"], kind="blocks")
        tracker.update_issue(blocker["key"], status="closed")
        got = tracker.frontier(m["key"])
        assert [t["title"] for t in got["frontier"]] == ["a", "b", "c"]

    def test_a_resolved_blocker_still_blocks(self, cao_system):
        # Consistent with the status vocabulary: resolved is not closed.
        m, tickets = _map_with_tickets(cao_system)
        blocker = tracker.create_issue(project_id="cao-system", title="blocker")
        tracker.add_link(blocker["key"], to_key=tickets[0]["key"], kind="blocks")
        tracker.update_issue(blocker["key"], status="resolved")
        got = tracker.frontier(m["key"])
        assert [t["title"] for t in got["frontier"]] == ["b", "c"]

    def test_an_outgoing_blocks_edge_does_not_bench_the_blocker(self, cao_system):
        # `a blocks x` says x waits on a — it says nothing about a itself.
        m, tickets = _map_with_tickets(cao_system)
        other = tracker.create_issue(project_id="cao-system", title="other")
        tracker.add_link(tickets[0]["key"], to_key=other["key"], kind="blocks")
        got = tracker.frontier(m["key"])
        assert [t["title"] for t in got["frontier"]] == ["a", "b", "c"]

    def test_tickets_of_another_map_and_non_members_are_not_frontier(self, cao_system):
        m, _ = _map_with_tickets(cao_system)
        tracker.create_issue(project_id="cao-system", title="unrelated")
        other_map = tracker.create_issue(project_id="cao-system", title="other map")
        stray = tracker.create_issue(project_id="cao-system", title="stray")
        tracker.add_link(stray["key"], to_key=other_map["key"], kind="part-of")
        got = tracker.frontier(m["key"])
        assert [t["title"] for t in got["frontier"]] == ["a", "b", "c"]

    def test_frontier_of_a_missing_map_is_not_found(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.frontier("cond-9999")
        assert exc.value.code == "not-found"


class TestClaimLifecycle:
    """Atomic claim/unclaim (cond-0394).

    The claim is one conditional UPDATE, so two cooperative workers cannot both
    win; the loser gets a typed conflict that reports the observed owner, and
    unclaim is the ordinary exit that makes a retry possible.
    """

    def test_claiming_an_open_issue_assigns_it(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        got = tracker.claim_issue(issue["key"], claimant="terra")
        assert got["assignee"] == "terra"
        assert got["status"] == "in-progress"
        assert (got["claimed"], got["already_claimed"]) == (True, False)

    def test_a_second_worker_gets_a_typed_conflict_naming_the_claimant(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.claim_issue(issue["key"], claimant="terra")
        with pytest.raises(TrackerError) as exc:
            tracker.claim_issue(issue["key"], claimant="muse")
        assert exc.value.code == "conflict"
        assert "terra" in exc.value.message
        assert exc.value.details["observed_assignee"] == "terra"

    def test_a_retry_by_the_current_claimant_is_idempotent(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.claim_issue(issue["key"], claimant="terra")
        again = tracker.claim_issue(issue["key"], claimant="terra")
        assert again["already_claimed"] is True
        claims = [e for e in tracker.get_issue(issue["key"])["events"] if e["kind"] == "claim"]
        # A retry writes nothing: the audit trail records the one real claim.
        assert len(claims) == 1

    def test_claiming_a_terminal_issue_is_refused_with_the_observed_status(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", status="closed")
        with pytest.raises(TrackerError) as exc:
            tracker.claim_issue(issue["key"], claimant="terra")
        assert exc.value.code == "conflict"
        assert exc.value.details["observed_status"] == "closed"

    def test_claiming_a_missing_issue_is_not_found(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.claim_issue("cond-9999", claimant="terra")
        assert exc.value.code == "not-found"

    def test_an_empty_claimant_is_invalid(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError) as exc:
            tracker.claim_issue(issue["key"], claimant="  ")
        assert exc.value.code == "invalid"

    def test_unclaim_is_the_exit_that_lets_another_worker_in(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.claim_issue(issue["key"], claimant="terra")
        tracker.unclaim_issue(issue["key"], actor="colin")
        got = tracker.claim_issue(issue["key"], claimant="muse")
        assert (got["assignee"], got["already_claimed"]) == ("muse", False)
        assert got["status"] == "in-progress"

    def test_unclaiming_an_unclaimed_issue_is_idempotent(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        got = tracker.unclaim_issue(issue["key"])
        assert (got["unclaimed"], got["was_claimed"]) == (True, False)
        assert [e["kind"] for e in tracker.get_issue(issue["key"])["events"]] == ["created"]

    def test_claim_and_unclaim_are_audited(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.claim_issue(issue["key"], claimant="terra")
        tracker.unclaim_issue(issue["key"], actor="colin")
        events = [
            (e["kind"], e["field"], e["old_value"], e["new_value"], e["actor"])
            for e in tracker.get_issue(issue["key"])["events"]
        ]
        assert events == [
            ("created", None, None, "a", None),
            ("claim", "assignee", None, "terra", "terra"),
            ("field", "status", "open", "in-progress", "terra"),
            ("unclaim", "assignee", "terra", None, "colin"),
            ("field", "status", "in-progress", "open", "colin"),
        ]


class TestExpectedUpdatedAt:
    """Optimistic precondition on issue edits (cond-0394).

    A wayfinder session edits the map body after reading it; the precondition
    turns "somebody else edited first" into a typed conflict carrying the
    current observable version, instead of a silent overwrite.
    """

    def test_a_matching_precondition_applies_the_edit(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", body="v1")
        got = tracker.update_issue(issue["key"], body="v2", expected_updated_at=issue["updated_at"])
        assert got["body"] == "v2"

    def test_a_stale_precondition_conflicts_and_reports_the_current_version(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", body="v1")
        stale = issue["updated_at"]
        current = tracker.update_issue(issue["key"], body="somebody else")
        with pytest.raises(TrackerError) as exc:
            tracker.update_issue(issue["key"], body="v2", expected_updated_at=stale)
        assert exc.value.code == "conflict"
        assert exc.value.details["current_updated_at"] == current["updated_at"]
        assert current["updated_at"] in exc.value.message

    def test_the_refused_edit_changes_nothing(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", body="v1")
        stale = issue["updated_at"]
        tracker.update_issue(issue["key"], body="somebody else")
        with pytest.raises(TrackerError):
            tracker.update_issue(issue["key"], body="v2", expected_updated_at=stale)
        after = tracker.get_issue(issue["key"])
        assert after["body"] == "somebody else"
        assert [e["kind"] for e in after["events"]] == ["created", "field"]

    def test_without_a_precondition_the_edit_is_unconditional(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", body="v1")
        tracker.update_issue(issue["key"], body="v2")
        got = tracker.update_issue(issue["key"], body="v3")
        assert got["body"] == "v3"

    def test_an_unparseable_precondition_is_invalid(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError) as exc:
            tracker.update_issue(issue["key"], body="v2", expected_updated_at="yesterday")
        assert exc.value.code == "invalid"

    def test_a_status_cas_returns_the_committed_clock_and_audit_effect(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        result = tracker.update_issue(
            issue["key"], status="blocked", expected_updated_at=issue["updated_at"]
        )
        assert result["updated_at"] == tracker.get_issue(issue["key"])["updated_at"]
        assert result["effect_id"] in result["effect_ids"]
        event = next(
            e for e in tracker.get_issue(issue["key"])["events"] if e["id"] == result["effect_id"]
        )
        assert (event["kind"], event["field"], event["new_value"]) == ("field", "status", "blocked")

    def test_a_status_mutation_after_precheck_is_refused_without_a_second_effect(
        self, cao_system, monkeypatch
    ):
        import threading

        issue = tracker.create_issue(project_id="cao-system", title="a")
        at_write, release = threading.Event(), threading.Event()
        real_now = tracker._utcnow
        paused = False

        def pause_after_precheck():
            nonlocal paused
            if not paused:
                paused = True
                at_write.set()
                assert release.wait(timeout=5)
            return real_now()

        monkeypatch.setattr(tracker, "_utcnow", pause_after_precheck)
        outcomes = []

        def stale_status_write():
            try:
                tracker.update_issue(
                    issue["key"], status="closed", expected_updated_at=issue["updated_at"]
                )
            except TrackerError as exc:
                outcomes.append(exc)

        worker = threading.Thread(target=stale_status_write)
        worker.start()
        assert at_write.wait(timeout=5)
        tracker.update_issue(issue["key"], status="blocked")
        release.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert [exc.code for exc in outcomes] == ["conflict"]
        detail = tracker.get_issue(issue["key"])
        assert detail["status"] == "blocked"
        assert [(e["kind"], e["field"]) for e in detail["events"]] == [
            ("created", None),
            ("field", "status"),
        ]


class TestAtomicLabelUpdates:
    """add/remove/clear label deltas in one update (cond-0394).

    Triage moves an issue between role labels; that must not require a stale
    read-modify-write of the whole set, and must not drop labels another actor
    added concurrently. One strategy per update: full replacement (`labels`)
    never combines with the deltas, and `clear_labels` — itself a full
    replacement to the empty set — combines with nothing.
    """

    def test_add_labels_merges_without_touching_unrelated_labels(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", labels=["bug", "ui"])
        got = tracker.update_issue(issue["key"], add_labels=["needs-triage"])
        assert got["labels"] == ["bug", "ui", "needs-triage"]

    def test_remove_labels_drops_only_the_named_ones(self, cao_system):
        issue = tracker.create_issue(
            project_id="cao-system", title="a", labels=["needs-triage", "bug"]
        )
        got = tracker.update_issue(issue["key"], remove_labels=["needs-triage"])
        assert got["labels"] == ["bug"]

    def test_add_and_remove_compose_in_one_update(self, cao_system):
        issue = tracker.create_issue(
            project_id="cao-system", title="a", labels=["needs-triage", "bug"]
        )
        got = tracker.update_issue(
            issue["key"], add_labels=["ready-for-agent"], remove_labels=["needs-triage"]
        )
        assert got["labels"] == ["bug", "ready-for-agent"]

    def test_a_label_in_both_add_and_remove_ends_up_added(self, cao_system):
        # Defined order: removals first, additions after — so additions win.
        issue = tracker.create_issue(project_id="cao-system", title="a", labels=["a"])
        got = tracker.update_issue(issue["key"], add_labels=["a"], remove_labels=["a"])
        assert got["labels"] == ["a"]

    def test_clear_labels_empties_the_set(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", labels=["a", "b"])
        got = tracker.update_issue(issue["key"], clear_labels=True)
        assert got["labels"] == []

    def test_full_replacement_cannot_combine_with_deltas(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", labels=["a"])
        with pytest.raises(TrackerError) as exc:
            tracker.update_issue(issue["key"], labels=["b"], add_labels=["c"])
        assert exc.value.code == "invalid"
        assert tracker.get_issue(issue["key"])["labels"] == ["a"]

    def test_clear_cannot_combine_with_add_or_remove(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", labels=["a"])
        with pytest.raises(TrackerError) as exc:
            tracker.update_issue(issue["key"], clear_labels=True, add_labels=["b"])
        assert exc.value.code == "invalid"

    def test_bounds_apply_to_the_merged_result(self, cao_system):
        labels = [f"l{i}" for i in range(tracker.MAX_LABELS)]
        issue = tracker.create_issue(project_id="cao-system", title="a", labels=labels)
        with pytest.raises(TrackerError) as exc:
            tracker.update_issue(issue["key"], add_labels=["one-too-many"])
        assert exc.value.code == "invalid"
        assert tracker.get_issue(issue["key"])["labels"] == labels

    def test_the_resulting_set_is_audited_once(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", labels=["a", "b"])
        tracker.update_issue(issue["key"], actor="colin", add_labels=["c"], remove_labels=["a"])
        events = [
            e
            for e in tracker.get_issue(issue["key"])["events"]
            if e["kind"] == "field" and e["field"] == "labels"
        ]
        assert len(events) == 1
        assert json.loads(events[0]["new_value"]) == ["b", "c"]
        assert events[0]["actor"] == "colin"

    def test_a_noop_delta_records_nothing(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", labels=["a"])
        tracker.update_issue(issue["key"], add_labels=["a"], remove_labels=["nope"])
        assert [e["kind"] for e in tracker.get_issue(issue["key"])["events"]] == ["created"]


class TestUnlabeledFilter:
    """First-class unlabeled discovery for triage (cond-0394)."""

    def test_unlabeled_returns_only_labelless_issues(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="bare")
        tracker.create_issue(project_id="cao-system", title="tagged", labels=["bug"])
        got = tracker.list_issues(project_id="cao-system", unlabeled=True)
        assert [i["title"] for i in got["issues"]] == ["bare"]

    def test_unlabeled_composes_and_the_total_stays_honest(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="bare open")
        tracker.create_issue(project_id="cao-system", title="bare open 2")
        closed = tracker.create_issue(project_id="cao-system", title="bare closed")
        tracker.create_issue(project_id="cao-system", title="tagged", labels=["x"])
        tracker.update_issue(closed["key"], status="closed")
        page = tracker.list_issues(project_id="cao-system", unlabeled=True, open_only=True, limit=1)
        assert page["total"] == 2
        assert [i["title"] for i in page["issues"]] == ["bare open 2"]

    def test_unlabeled_composes_with_kind(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="bare issue")
        tracker.create_feature(project_id="cao-system", title="bare feature")
        got = tracker.list_issues(project_id="cao-system", unlabeled=True, kind="feature")
        assert [i["title"] for i in got["issues"]] == ["bare feature"]
        got_all = tracker.list_issues(project_id="cao-system", unlabeled=True, kind="all")
        assert sorted(i["title"] for i in got_all["issues"]) == ["bare feature", "bare issue"]


class TestWithoutLabelFilter:
    def test_repeated_exclusions_remove_any_exact_match(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="ready", labels=["source:wayfinder"])
        tracker.create_issue(project_id="cao-system", title="triaged", labels=["needs-triage"])
        tracker.create_issue(project_id="cao-system", title="waiting", labels=["needs-info"])
        tracker.create_issue(project_id="cao-system", title="similar", labels=["needs-info-extra"])

        got = tracker.list_issues(
            project_id="cao-system", without_label=["needs-triage", "needs-info"]
        )

        assert {i["title"] for i in got["issues"]} == {"ready", "similar"}

    def test_exclusions_compose_with_inclusion_and_keep_unpaged_total(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a", labels=["wayfinder:task"])
        tracker.create_issue(
            project_id="cao-system", title="b", labels=["wayfinder:task", "needs-info"]
        )
        tracker.create_issue(project_id="cao-system", title="c", labels=["other"])

        page = tracker.list_issues(
            project_id="cao-system",
            label="wayfinder:task",
            without_label="needs-info",
            limit=1,
        )

        assert page["total"] == 1
        assert [i["title"] for i in page["issues"]] == ["a"]


class TestConcurrentUpdates:
    """The write seam itself must be atomic, not the Python before it.

    These tests park two threads inside the vulnerable window (both rows read,
    both preconditions passed) and only then release them at the write. Under
    the pre-fix check-then-act implementation both commits succeeded and one
    writer's change was silently lost — or recorded twice.
    """

    def test_only_one_of_two_concurrent_cas_writes_wins(self, cao_system, monkeypatch):
        import threading

        issue = tracker.create_issue(project_id="cao-system", title="map", body="v1")
        version = issue["updated_at"]
        barrier = threading.Barrier(2)
        real_parse = tracker._parse_timestamp

        def synced_parse(text, *, field):
            result = real_parse(text, field=field)
            # Both threads have now read the row AND validated the precondition.
            barrier.wait(timeout=5)
            return result

        monkeypatch.setattr(tracker, "_parse_timestamp", synced_parse)
        outcomes = []

        def write(body):
            try:
                tracker.update_issue(issue["key"], body=body, expected_updated_at=version)
                outcomes.append(("ok", body, None))
            except TrackerError as exc:
                outcomes.append((exc.code, body, exc))

        threads = [threading.Thread(target=write, args=(b,)) for b in ("from-a", "from-b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), "a writer deadlocked"

        statuses = sorted(o[0] for o in outcomes)
        assert statuses == ["conflict", "ok"], outcomes
        loser = next(o for o in outcomes if o[0] == "conflict")[2]
        assert loser.details["current_updated_at"] is not None
        assert loser.details["current_updated_at"] != version
        winner_body = next(o[1] for o in outcomes if o[0] == "ok")
        after = tracker.get_issue(issue["key"])
        # The winner's write is the one that survived, and the audit trail
        # records exactly one body transition — the one that happened.
        assert after["body"] == winner_body
        body_events = [e for e in after["events"] if e["kind"] == "field" and e["field"] == "body"]
        assert len(body_events) == 1

    def test_concurrent_label_deltas_both_land(self, cao_system, monkeypatch):
        import threading

        issue = tracker.create_issue(project_id="cao-system", title="a", labels=["base"])
        barrier = threading.Barrier(2)
        synced = set()
        real_normalise = tracker.normalise_labels

        def synced_normalise(labels):
            marker = tuple(labels) if isinstance(labels, list) else None
            if marker in (("worker-a",), ("worker-b",)) and marker not in synced:
                # Each worker syncs once, after its read, before its write; the
                # loser's retry must not re-enter the barrier.
                synced.add(marker)
                barrier.wait(timeout=5)
            return real_normalise(labels)

        monkeypatch.setattr(tracker, "normalise_labels", synced_normalise)
        outcomes = []

        def add(label):
            try:
                tracker.update_issue(issue["key"], add_labels=[label])
                outcomes.append(("ok", label))
            except TrackerError as exc:
                outcomes.append((exc.code, label))

        threads = [threading.Thread(target=add, args=(l,)) for l in ("worker-a", "worker-b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), "a writer deadlocked"

        assert sorted(outcomes) == [("ok", "worker-a"), ("ok", "worker-b")], outcomes
        after = tracker.get_issue(issue["key"])
        assert sorted(after["labels"]) == ["base", "worker-a", "worker-b"]
        # Two real transitions happened, so two label events exist — and each
        # records the transition it actually applied, not its first attempt.
        label_events = [
            e for e in after["events"] if e["kind"] == "field" and e["field"] == "labels"
        ]
        assert len(label_events) == 2
        for e in label_events:
            old, new = json.loads(e["old_value"]), json.loads(e["new_value"])
            assert len(new) == len(old) + 1
            assert set(new) - set(old) in ({"worker-a"}, {"worker-b"})

    def test_concurrent_unclaims_establish_one_release(self, cao_system, monkeypatch):
        import threading

        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.claim_issue(issue["key"], claimant="worker-a")
        barrier = threading.Barrier(2)
        synced = set()
        real_now = tracker._utcnow

        def synced_now():
            ident = threading.get_ident()
            if ident not in synced:
                # Both releasers have now read the same owner and park just
                # before the write; only the write seam may decide the winner.
                synced.add(ident)
                barrier.wait(timeout=5)
            return real_now()

        monkeypatch.setattr(tracker, "_utcnow", synced_now)
        outcomes = []

        def release(actor):
            try:
                got = tracker.unclaim_issue(issue["key"], actor=actor)
                outcomes.append(("ok", actor, got["was_claimed"]))
            except Exception as exc:  # noqa: BLE001 - a raw lock error is a failure mode too
                outcomes.append(("error", actor, exc))

        threads = [
            threading.Thread(target=release, args=(a,)) for a in ("supervisor-a", "supervisor-b")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), "a releaser deadlocked"

        # Exactly one releaser established the transition; the other resolved
        # idempotently — no exception, no duplicate claim of the release.
        assert sorted((o[0], o[2]) for o in outcomes) == [("ok", False), ("ok", True)], outcomes
        after = tracker.get_issue(issue["key"])
        assert after["assignee"] is None
        unclaim_events = [
            (e["actor"], e["old_value"], e["new_value"])
            for e in after["events"]
            if e["kind"] == "unclaim"
        ]
        winner = next(o[1] for o in outcomes if o[2] is True)
        assert unclaim_events == [(winner, "worker-a", None)]

    def test_a_stale_unclaim_cannot_clear_a_successor_claim(self, cao_system, monkeypatch):
        import threading

        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.claim_issue(issue["key"], claimant="worker-a")
        parked = threading.Event()
        proceed = threading.Event()
        real_now = tracker._utcnow

        def parked_now():
            if not parked.is_set():
                # The stale releaser has read worker-a's claim and parks just
                # before its write. The main thread then releases worker-a and
                # worker-c claims — before the stale write lands.
                parked.set()
                proceed.wait(timeout=5)
            return real_now()

        monkeypatch.setattr(tracker, "_utcnow", parked_now)
        outcomes = []

        def release():
            got = tracker.unclaim_issue(issue["key"], actor="supervisor-a")
            outcomes.append(got)

        thread = threading.Thread(target=release)
        thread.start()
        assert parked.wait(timeout=5)
        tracker.unclaim_issue(issue["key"], actor="supervisor-b")
        tracker.claim_issue(issue["key"], claimant="worker-c")
        proceed.set()
        thread.join(timeout=15)
        assert not thread.is_alive(), "the stale releaser deadlocked"

        # The stale call observed worker-a's claim; that claim is gone, so the
        # call established nothing — and worker-c's successor claim survives.
        (got,) = outcomes
        assert got["was_claimed"] is False
        assert got["assignee"] == "worker-c"
        assert tracker.get_issue(issue["key"])["assignee"] == "worker-c"
        kinds = [e["kind"] for e in tracker.get_issue(issue["key"])["events"]]
        assert kinds == [
            "created",
            "claim",
            "field",
            "unclaim",
            "field",
            "claim",
            "field",
        ]


class TestLabelFacets:
    """Label discovery for the dashboard filter bar (cond-0394)."""

    def test_counts_per_label_with_the_open_split(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a", labels=["effort:maps", "bug"])
        b = tracker.create_issue(project_id="cao-system", title="b", labels=["effort:maps"])
        tracker.create_issue(project_id="cao-system", title="c", labels=["bug"])
        tracker.update_issue(b["key"], status="closed")
        got = tracker.label_facets("cao-system")
        by_label = {f["label"]: f for f in got["labels"]}
        assert by_label["effort:maps"] == {"label": "effort:maps", "total": 2, "open": 1}
        assert by_label["bug"] == {"label": "bug", "total": 2, "open": 2}

    def test_the_unlabeled_bucket_matches_the_list_filter(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="bare")
        tracker.create_issue(project_id="cao-system", title="tagged", labels=["x"])
        got = tracker.label_facets("cao-system")
        assert (got["unlabeled"], got["unlabeled_open"]) == (1, 1)
        listed = tracker.list_issues(project_id="cao-system", unlabeled=True, kind="all")
        assert listed["total"] == got["unlabeled"]

    def test_labels_span_kinds(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="bug", labels=["wayfinder:map"])
        tracker.create_feature(project_id="cao-system", title="wish", labels=["wayfinder:map"])
        got = tracker.label_facets("cao-system")
        assert got["labels"][0]["label"] == "wayfinder:map"
        assert got["labels"][0]["total"] == 2

    def test_ordering_is_most_alive_first(self, cao_system):
        tracker.create_issue(project_id="cao-system", title="a", labels=["zzz"])
        tracker.create_issue(project_id="cao-system", title="b", labels=["aaa"])
        tracker.create_issue(project_id="cao-system", title="c", labels=["aaa"])
        got = tracker.label_facets("cao-system")
        assert [f["label"] for f in got["labels"]] == ["aaa", "zzz"]

    def test_scoped_to_one_project(self, cao_system):
        tracker.create_project(name="Other", project_id="other")
        tracker.create_issue(project_id="other", title="theirs", labels=["foreign"])
        tracker.create_issue(project_id="cao-system", title="mine", labels=["mine"])
        got = tracker.label_facets("cao-system")
        assert [f["label"] for f in got["labels"]] == ["mine"]

    def test_an_unknown_project_is_not_found(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.label_facets("nope")
        assert exc.value.code == "not-found"


class TestMapProjection:
    """The one-request map projection behind the dashboard map view (cond-0394)."""

    def test_children_carry_their_classification(self, cao_system):
        m, tickets = _map_with_tickets(cao_system)
        blocker = tracker.create_issue(project_id="cao-system", title="blocker")
        tracker.add_link(blocker["key"], to_key=tickets[0]["key"], kind="blocks")
        tracker.update_issue(tickets[1]["key"], assignee="terra")
        tracker.update_issue(tickets[2]["key"], status="closed")
        got = tracker.map_projection(m["key"])
        by_title = {c["title"]: c for c in got["children"]}
        assert by_title["a"]["blocked_by"] == [blocker["key"]]
        assert by_title["a"]["frontier"] is False
        assert by_title["b"]["frontier"] is False  # claimed
        assert by_title["c"]["frontier"] is False  # terminal
        assert got["frontier"] == []

    def test_frontier_keys_match_the_frontier_interface_exactly(self, cao_system):
        m, tickets = _map_with_tickets(cao_system)
        tracker.update_issue(tickets[2]["key"], assignee="terra")
        got = tracker.map_projection(m["key"])
        assert got["frontier"] == [tickets[0]["key"], tickets[1]["key"]]
        direct = tracker.frontier(m["key"])
        assert [t["key"] for t in direct["frontier"]] == got["frontier"]

    def test_external_blockers_are_included_so_a_benched_child_explains_itself(self, cao_system):
        m, tickets = _map_with_tickets(cao_system, titles=("a",))
        outside = tracker.create_issue(project_id="cao-system", title="outside blocker")
        tracker.add_link(outside["key"], to_key=tickets[0]["key"], kind="blocks")
        got = tracker.map_projection(m["key"])
        assert [e["title"] for e in got["external"]] == ["outside blocker"]
        # The one fact that makes it an external blocker: it benches a member.
        assert got["external"][0]["blocking"] == [tickets[0]["key"]]

    def test_every_link_endpoint_is_materialized_not_just_blockers(self, cao_system):
        """A relates/duplicates/caused-by/outgoing link to a non-member puts
        that issue in `external` too — no returned link may point at an issue
        the caller cannot see."""
        m, tickets = _map_with_tickets(cao_system, titles=("a", "b"))
        related = tracker.create_issue(project_id="cao-system", title="related work")
        duplicate = tracker.create_issue(project_id="cao-system", title="filed twice")
        cause = tracker.create_issue(project_id="cao-system", title="root cause")
        downstream = tracker.create_issue(project_id="cao-system", title="waits on us")
        tracker.add_link(tickets[0]["key"], to_key=related["key"], kind="relates")
        tracker.add_link(tickets[0]["key"], to_key=duplicate["key"], kind="duplicates")
        tracker.add_link(tickets[1]["key"], to_key=cause["key"], kind="caused-by")
        # A child blocking an outsider is an endpoint too — direction and kind
        # make it context, not a blocker OF the map.
        tracker.add_link(tickets[0]["key"], to_key=downstream["key"], kind="blocks")
        got = tracker.map_projection(m["key"])
        by_key = {e["key"]: e for e in got["external"]}
        assert set(by_key) == {related["key"], duplicate["key"], cause["key"], downstream["key"]}
        # None of them benches a member — the blocker marker stays empty.
        assert all(e["blocking"] == [] for e in by_key.values())

    def test_blocking_marks_only_blockers_that_still_bench_a_child(self, cao_system):
        """`blocking` inverts the children's blocked_by lists: a nonterminal
        blocker benches; a terminal one has landed and benches nobody, though
        its link still earns it a row."""
        m, tickets = _map_with_tickets(cao_system, titles=("a",))
        live = tracker.create_issue(project_id="cao-system", title="live blocker")
        landed = tracker.create_issue(project_id="cao-system", title="landed blocker")
        tracker.add_link(live["key"], to_key=tickets[0]["key"], kind="blocks")
        tracker.add_link(landed["key"], to_key=tickets[0]["key"], kind="blocks")
        tracker.update_issue(landed["key"], status="closed")
        got = tracker.map_projection(m["key"])
        by_key = {e["key"]: e for e in got["external"]}
        assert by_key[live["key"]]["blocking"] == [tickets[0]["key"]]
        assert by_key[landed["key"]]["blocking"] == []
        # The child's own blocked_by agrees: benched by the live one only.
        assert got["children"][0]["blocked_by"] == [live["key"]]

    def test_links_cover_the_map_and_children_in_both_directions(self, cao_system):
        m, tickets = _map_with_tickets(cao_system, titles=("a", "b"))
        tracker.add_link(tickets[0]["key"], to_key=tickets[1]["key"], kind="relates")
        got = tracker.map_projection(m["key"])
        shapes = {(l["from_key"], l["to_key"], l["kind"]) for l in got["links"]}
        assert (tickets[0]["key"], m["key"], "part-of") in shapes
        assert (tickets[0]["key"], tickets[1]["key"], "relates") in shapes

    def test_progress_counts_the_direct_children(self, cao_system):
        m, tickets = _map_with_tickets(cao_system)
        tracker.update_issue(tickets[0]["key"], status="closed")
        tracker.update_issue(tickets[1]["key"], status="resolved")
        tracker.update_issue(tickets[2]["key"], assignee="terra")
        got = tracker.map_projection(m["key"])
        # The resolved ticket is nonterminal, unassigned and unblocked — it is
        # still takeable (landed ≠ verified), so it sits on the frontier.
        assert got["progress"] == {
            "total": 3,
            "open": 2,
            "terminal": 1,
            "resolved": 1,
            "claimed": 1,
            "frontier": 1,
        }
        assert got["map"]["key"] == m["key"]

    def test_a_plain_issue_projects_an_empty_map(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="not a map")
        got = tracker.map_projection(issue["key"])
        assert got["children"] == []
        assert got["progress"]["total"] == 0

    def test_an_unknown_map_is_not_found(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.map_projection("cond-9999")
        assert exc.value.code == "not-found"


class TestGraphProjection:
    def test_transitive_part_of_hierarchy_starts_from_any_issue(self, cao_system):
        root = tracker.create_issue(project_id="cao-system", title="product", kind="project")
        milestone = tracker.create_issue(
            project_id="cao-system", title="milestone", kind="milestone"
        )
        story = tracker.create_issue(project_id="cao-system", title="story", kind="story")
        task = tracker.create_issue(project_id="cao-system", title="task", kind="task")
        tracker.add_link(milestone["key"], to_key=root["key"], kind="part-of")
        tracker.add_link(story["key"], to_key=milestone["key"], kind="part-of")
        tracker.add_link(task["key"], to_key=story["key"], kind="part-of")

        got = tracker.graph_projection(root["key"])
        by_key = {row["key"]: row for row in got["nodes"]}
        assert [
            by_key[key]["depth"]
            for key in (root["key"], milestone["key"], story["key"], task["key"])
        ] == [0, 1, 2, 3]
        assert by_key[story["key"]]["parent_keys"] == [milestone["key"]]
        assert by_key[milestone["key"]]["child_count"] == 1
        assert got["stats"] == {"nodes": 4, "descendants": 3, "external": 0, "links": 3, "depth": 3}

    def test_relationship_endpoints_are_materialized_without_becoming_children(self, cao_system):
        root, children = _map_with_tickets(cao_system, titles=("child",))
        blocker = tracker.create_issue(project_id="cao-system", title="blocker")
        related = tracker.create_issue(project_id="cao-system", title="related")
        tracker.add_link(blocker["key"], to_key=children[0]["key"], kind="blocks")
        tracker.add_link(children[0]["key"], to_key=related["key"], kind="relates")

        got = tracker.graph_projection(root["key"])
        assert {row["key"] for row in got["external"]} == {blocker["key"], related["key"]}
        visible = {row["key"] for row in got["nodes"] + got["external"]}
        assert all(
            link["from_key"] in visible and link["to_key"] in visible for link in got["links"]
        )

    def test_cycles_do_not_loop_or_duplicate_nodes(self, cao_system):
        a = tracker.create_issue(project_id="cao-system", title="a")
        b = tracker.create_issue(project_id="cao-system", title="b")
        tracker.add_link(b["key"], to_key=a["key"], kind="part-of")
        tracker.add_link(a["key"], to_key=b["key"], kind="part-of")
        got = tracker.graph_projection(a["key"])
        assert [row["key"] for row in got["nodes"]] == [a["key"], b["key"]]
        assert len(got["links"]) == 2

    def test_depth_bound_is_reported_and_omitted_children_are_not_external(self, cao_system):
        root = tracker.create_issue(project_id="cao-system", title="root")
        child = tracker.create_issue(project_id="cao-system", title="child")
        grandchild = tracker.create_issue(project_id="cao-system", title="grandchild")
        tracker.add_link(child["key"], to_key=root["key"], kind="part-of")
        tracker.add_link(grandchild["key"], to_key=child["key"], kind="part-of")
        # A second relationship touching the included child must not smuggle
        # the depth-omitted grandchild back as generic external context.
        tracker.add_link(grandchild["key"], to_key=child["key"], kind="blocks")
        got = tracker.graph_projection(root["key"], max_depth=1)
        assert [row["key"] for row in got["nodes"]] == [root["key"], child["key"]]
        assert grandchild["key"] not in {row["key"] for row in got["external"]}
        assert got["bounds"]["truncated"] is True
        assert got["bounds"]["reasons"] == ["depth-limit"]

    def test_node_bound_is_reported_without_leaking_omitted_siblings(self, cao_system):
        root = tracker.create_issue(project_id="cao-system", title="root")
        children = [
            tracker.create_issue(project_id="cao-system", title=f"child-{index}")
            for index in range(3)
        ]
        for child in children:
            tracker.add_link(child["key"], to_key=root["key"], kind="part-of")

        got = tracker.graph_projection(root["key"], max_nodes=2)
        assert len(got["nodes"]) == 2
        visible = {row["key"] for row in got["nodes"] + got["external"]}
        assert len(visible & {child["key"] for child in children}) == 1
        assert got["bounds"]["truncated"] is True
        assert "node-limit" in got["bounds"]["reasons"]

    def test_unknown_root_is_not_found(self, cao_system):
        with pytest.raises(TrackerError) as exc:
            tracker.graph_projection("cond-9999")
        assert exc.value.code == "not-found"


class TestHierarchyAudit:
    def test_recursive_counts_findings_blockers_and_leaf_frontier(self, cao_system):
        root = tracker.create_issue(project_id="cao-system", title="root", kind="project")
        story = tracker.create_issue(project_id="cao-system", title="story", kind="story")
        ready = tracker.create_issue(project_id="cao-system", title="ready", kind="task")
        held = tracker.create_issue(project_id="cao-system", title="held", kind="task")
        blocker = tracker.create_issue(project_id="cao-system", title="blocker", kind="bug")
        tracker.add_link(story["key"], to_key=root["key"], kind="part-of")
        tracker.add_link(ready["key"], to_key=story["key"], kind="part-of")
        tracker.add_link(held["key"], to_key=story["key"], kind="part-of")
        tracker.add_link(ready["key"], to_key=root["key"], kind="part-of")
        tracker.add_link(blocker["key"], to_key=held["key"], kind="blocks")

        got = tracker.hierarchy_audit(root["key"])

        assert got["counts"]["nodes"] == 4
        assert got["counts"]["part_of"] == 4
        assert got["counts"]["blocks"] == 1
        assert got["findings"]["multiple_parents"] == [
            {"key": ready["key"], "parents": [root["key"], story["key"]]}
        ]
        assert got["findings"]["hierarchy_cycles"] == []
        assert got["unresolved_blockers"] == [{"blocker": blocker["key"], "blocked": held["key"]}]
        assert [row["key"] for row in got["frontier"]] == [ready["key"]]

    def test_hierarchy_and_blocker_cycles_are_reported(self, cao_system):
        root = tracker.create_issue(project_id="cao-system", title="root")
        child = tracker.create_issue(project_id="cao-system", title="child")
        tracker.add_link(child["key"], to_key=root["key"], kind="part-of")
        tracker.add_link(root["key"], to_key=child["key"], kind="part-of")
        tracker.add_link(root["key"], to_key=child["key"], kind="blocks")
        tracker.add_link(child["key"], to_key=root["key"], kind="blocks")

        got = tracker.hierarchy_audit(root["key"])

        expected_cycle = [sorted([child["key"], root["key"]])]
        assert got["findings"]["hierarchy_cycles"] == expected_cycle
        assert got["findings"]["blocker_cycles"] == expected_cycle

    def test_selected_root_reports_a_parent_outside_the_audited_subtree(self, cao_system):
        outer = tracker.create_issue(project_id="cao-system", title="outer", kind="project")
        root = tracker.create_issue(project_id="cao-system", title="root", kind="project")
        child = tracker.create_issue(project_id="cao-system", title="child", kind="task")
        tracker.add_link(root["key"], to_key=outer["key"], kind="part-of")
        tracker.add_link(child["key"], to_key=root["key"], kind="part-of")

        got = tracker.hierarchy_audit(root["key"])

        assert got["counts"]["nodes"] == 2
        assert got["counts"]["part_of"] == 1
        assert got["counts"]["relationships"] == 0
        assert got["findings"]["root_parents"] == [outer["key"]]

    def test_depth_bound_keeps_a_parent_with_an_external_child_off_the_frontier(self, cao_system):
        root = tracker.create_issue(project_id="cao-system", title="root", kind="project")
        parent = tracker.create_issue(project_id="cao-system", title="parent", kind="story")
        child = tracker.create_issue(project_id="cao-system", title="child", kind="task")
        tracker.add_link(parent["key"], to_key=root["key"], kind="part-of")
        tracker.add_link(child["key"], to_key=parent["key"], kind="part-of")

        got = tracker.hierarchy_audit(root["key"], max_depth=1)

        assert got["bounds"]["truncated"] is True
        assert got["bounds"]["live_children_beyond_bound"] == [parent["key"]]
        assert got["frontier"] == []


class TestObservedRevision:
    """observed_revision records the revision a reporter actually observed.

    It is optional, opaque, caller-supplied text — never inferred from the
    filing site, a worktree, or a branch name, because an invented revision
    looks authoritative while being fiction.
    """

    def test_filing_records_the_observed_revision(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", observed_revision="v1.2.3")
        assert issue["observed_revision"] == "v1.2.3"

    def test_a_revision_is_optional(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        assert issue["observed_revision"] is None

    def test_an_empty_revision_stores_null_not_empty_string(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", observed_revision="  ")
        assert issue["observed_revision"] is None

    def test_the_revision_is_never_inferred_from_the_filing_site(self, cao_system):
        issue = tracker.create_issue(
            project_id="cao-system",
            title="a",
            session_name="cao-p1-closure",
            source_path=str(cao_system["fork"]),
        )
        assert issue["observed_revision"] is None

    def test_a_revision_is_editable_and_clearable(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", observed_revision="v1.2.3")
        moved = tracker.update_issue(issue["key"], observed_revision="abc1234")
        assert moved["observed_revision"] == "abc1234"
        cleared = tracker.update_issue(issue["key"], observed_revision="")
        assert cleared["observed_revision"] is None

    def test_a_same_value_revision_edit_writes_no_event(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a", observed_revision="v1.2.3")
        tracker.update_issue(issue["key"], observed_revision="v1.2.3")
        assert not [
            e
            for e in tracker.get_issue(issue["key"])["events"]
            if e["field"] == "observed_revision"
        ]

    def test_a_changed_revision_leaves_one_audit_event(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        tracker.update_issue(issue["key"], observed_revision="v1.2.3")
        events = [
            e
            for e in tracker.get_issue(issue["key"])["events"]
            if e["kind"] == "field" and e["field"] == "observed_revision"
        ]
        assert len(events) == 1
        assert (events[0]["old_value"], events[0]["new_value"]) == (None, "v1.2.3")


class TestCommentImportance:
    """The important flag is reversible Boolean weight on one comment.

    Every actual change writes exactly one append-only ``comment-field`` event
    and bumps the parent's ``updated_at`` in the same transaction; a retry that
    would not change anything writes nothing at all.
    """

    def test_comments_default_to_routine_weight(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="note")
        assert comment["important"] is False
        assert tracker.get_issue(issue["key"])["comments"][0]["important"] is False

    def test_creation_accepts_important_true(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="root cause", important=True)
        assert comment["important"] is True

    def test_setting_importance_changes_the_row_and_audits_it(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="note", author="colin")
        result = tracker.set_comment_importance(
            issue["key"], comment["id"], important=True, actor="colin"
        )
        assert result["changed"] is True
        detail = tracker.get_issue(issue["key"])
        assert detail["comments"][0]["important"] is True
        events = [e for e in detail["events"] if e["kind"] == "comment-field"]
        assert len(events) == 1
        assert events[0]["actor"] == "colin"
        assert events[0]["field"] == "important"
        assert (events[0]["old_value"], events[0]["new_value"]) == ("false", "true")

    def test_a_change_bumps_the_parent_timestamp(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="note")
        before = tracker.get_issue(issue["key"])["updated_at"]
        result = tracker.set_comment_importance(issue["key"], comment["id"], important=True)
        assert result["updated_at"] is not None
        assert result["updated_at"] != before
        assert tracker.get_issue(issue["key"])["updated_at"] == result["updated_at"]

    def test_a_same_value_retry_writes_nothing(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="note")
        tracker.set_comment_importance(issue["key"], comment["id"], important=True)
        before = tracker.get_issue(issue["key"])
        result = tracker.set_comment_importance(issue["key"], comment["id"], important=True)
        after = tracker.get_issue(issue["key"])
        assert result["changed"] is False
        assert len([e for e in after["events"] if e["kind"] == "comment-field"]) == 1
        assert after["updated_at"] == before["updated_at"]

    def test_importance_is_reversible_both_ways_twice(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="note")
        cid = comment["id"]
        for final, expected_events in ((True, 1), (False, 2), (True, 3), (False, 4)):
            tracker.set_comment_importance(issue["key"], cid, important=final)
            detail = tracker.get_issue(issue["key"])
            assert detail["comments"][0]["important"] is final
            events = [e for e in detail["events"] if e["kind"] == "comment-field"]
            assert len(events) == expected_events
            assert (events[-1]["old_value"], events[-1]["new_value"]) == (
                "false" if final else "true",
                "true" if final else "false",
            )

    def test_an_unknown_comment_is_refused_without_writing(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        with pytest.raises(TrackerError) as exc:
            tracker.set_comment_importance(issue["key"], 9999, important=True)
        assert exc.value.code == "not-found"

    def test_a_comment_id_from_another_issue_is_refused(self, cao_system):
        a = tracker.create_issue(project_id="cao-system", title="a")
        b = tracker.create_issue(project_id="cao-system", title="b")
        comment = tracker.add_comment(a["key"], body="on a")
        with pytest.raises(TrackerError) as exc:
            tracker.set_comment_importance(b["key"], comment["id"], important=True)
        assert exc.value.code == "not-found"


class TestCommentImportanceConcurrency:
    """Only the setter whose guarded UPDATE changes the row may record it."""

    def test_interleaved_transactions_produce_exactly_one_transition(self, cao_system, monkeypatch):
        import threading

        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="note")
        barrier = threading.Barrier(2)
        real_utcnow = tracker._utcnow

        def synced_utcnow():
            # Both setters have read the flag and passed the same-value check;
            # release them into the write together.
            result = real_utcnow()
            barrier.wait(timeout=5)
            return result

        monkeypatch.setattr(tracker, "_utcnow", synced_utcnow)
        outcomes = []

        def setter():
            outcomes.append(
                tracker.set_comment_importance(issue["key"], comment["id"], important=True)
            )

        threads = [threading.Thread(target=setter) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), "an importance setter deadlocked"

        changed = [o for o in outcomes if o["changed"]]
        assert len(changed) == 1
        detail = tracker.get_issue(issue["key"])
        assert detail["comments"][0]["important"] is True
        events = [e for e in detail["events"] if e["kind"] == "comment-field"]
        assert len(events) == 1
        assert (events[0]["old_value"], events[0]["new_value"]) == ("false", "true")

    def test_a_stale_premise_write_matches_zero_rows(self, cao_system):
        """The conditional guard turns a lost race into a no-match, not a second event.

        Setter A reads ``important=false`` and is suspended; setter B completes
        the flip; A resumes and applies the same guarded UPDATE still believing
        the stored value differs from its target. The guard must match zero
        rows so A cannot record the transition B already recorded.
        """
        from sqlalchemy.orm import sessionmaker as _sessionmaker

        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="note")

        stale_session = _sessionmaker(bind=tracker.SessionLocal.kw["bind"])()
        stale_read = stale_session.get(tracker.TrackerCommentModel, comment["id"])
        assert bool(stale_read.important) is False

        won = tracker.set_comment_importance(issue["key"], comment["id"], important=True)
        assert won["changed"] is True

        written = (
            stale_session.query(tracker.TrackerCommentModel)
            .filter(
                tracker.TrackerCommentModel.id == comment["id"],
                tracker.TrackerCommentModel.important != True,  # noqa: E712 - A's stale premise
            )
            .update({"important": True}, synchronize_session=False)
        )
        assert written == 0
        stale_session.rollback()
        stale_session.close()

        detail = tracker.get_issue(issue["key"])
        events = [e for e in detail["events"] if e["kind"] == "comment-field"]
        assert len(events) == 1


class TestCommentDeletionAudit:
    """Deleting a comment leaves the audit trail and timestamp consistent."""

    def test_deletion_writes_an_event_and_bumps_the_parent(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="soon gone", author="colin")
        before = tracker.get_issue(issue["key"])["updated_at"]
        result = tracker.delete_comment(issue["key"], comment["id"], actor="colin")
        assert result["deleted"] is True
        detail = tracker.get_issue(issue["key"])
        assert detail["comments"] == []
        assert detail["updated_at"] != before
        deletions = [e for e in detail["events"] if e["kind"] == "comment-deleted"]
        assert len(deletions) == 1
        assert deletions[0]["actor"] == "colin"
        assert deletions[0]["old_value"] == "soon gone"
        assert deletions[0]["new_value"] == str(comment["id"])

    def test_deletion_is_atomic_with_its_audit_record(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        comment = tracker.add_comment(issue["key"], body="note")
        tracker.delete_comment(issue["key"], comment["id"])
        events = tracker.get_issue(issue["key"])["events"]
        kinds = [e["kind"] for e in events]
        # The deletion event exists even though the commented-on row is gone.
        assert kinds.count("comment-deleted") == 1
        assert kinds.count("comment") == 1

    def test_deleting_an_unknown_comment_writes_nothing(self, cao_system):
        issue = tracker.create_issue(project_id="cao-system", title="a")
        before = tracker.get_issue(issue["key"])
        with pytest.raises(TrackerError) as exc:
            tracker.delete_comment(issue["key"], 9999)
        assert exc.value.code == "not-found"
        after = tracker.get_issue(issue["key"])
        assert after["updated_at"] == before["updated_at"]
        assert not [e for e in after["events"] if e["kind"] == "comment-deleted"]
