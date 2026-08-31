"""`cao project` and `cao issue`.

This surface had no coverage at all until a manual smoke test found that every
command died with a raw SQLAlchemy traceback on a fresh state root: the CLI has
no lifespan, so nothing had created the tables. That is the failure these tests
exist to prevent recurring — the CLI is specifically the path for filing an
issue when the server is *not* running, which makes "works only after the
server has run once" the wrong dependency to have.
"""

import json

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.cli.commands import issue as issue_cli
from cli_agent_orchestrator.clients.database import Base
from cli_agent_orchestrator.services import issue_tracker as tracker


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/cli.db")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(tracker, "SessionLocal", sessionmaker(bind=engine))
    # The group callbacks create the schema on the real engine. Here the schema
    # already exists on the test engine, so the call is neutralised rather than
    # allowed to touch the operator's database.
    monkeypatch.setattr(issue_cli, "ensure_tracker_schema", lambda: None)
    yield
    engine.dispose()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "cao-conductor"
    (path / "conduct").mkdir(parents=True)
    return path


@pytest.fixture
def runner():
    return CliRunner()


def run(runner, group, *args):
    argv = list(args)
    # Most historical tests intentionally use terse bug fixtures because the
    # behavior under test is unrelated to filing hygiene. Record that policy
    # departure explicitly; the dedicated filing-policy test below invokes the
    # command directly so this helper cannot mask a regression.
    if group is issue_cli.issue and argv[:1] == ["file"] and "--force" not in argv:
        try:
            kind = argv[argv.index("--kind") + 1]
        except ValueError:
            kind = "bug"
        if kind == "bug":
            required = (
                ("--reproduction", "--reproduction-file"),
                ("--expected-outcome",),
                ("--actual-outcome",),
            )
            if not all(any(option in argv for option in alternatives) for alternatives in required):
                argv.append("--force")
    result = runner.invoke(group, argv)
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


class TestProjectCommands:
    def test_create_then_show(self, runner, repo):
        run(
            runner,
            issue_cli.project,
            "create",
            "CAO System",
            "--id",
            "cao-system",
            "--prefix",
            "cond",
            "--path",
            str(repo),
            "--session",
            "cao-p1-closure",
        )
        out = run(runner, issue_cli.project, "show", "cao-system")
        assert "cond-NNNN" in out
        assert str(repo) in out
        assert "cao-p1-closure" in out

    def test_resolve_reports_what_matched(self, runner, repo):
        run(runner, issue_cli.project, "create", "P", "--id", "p", "--path", str(repo))
        out = run(runner, issue_cli.project, "resolve", "--cwd", str(repo / "conduct"))
        assert "matched by path" in out

    def test_resolve_says_so_when_nothing_is_registered(self, runner, tmp_path):
        out = run(runner, issue_cli.project, "resolve", "--cwd", str(tmp_path / "unknown"))
        assert "no project registered" in out

    def test_json_output_is_parseable(self, runner, repo):
        run(runner, issue_cli.project, "create", "P", "--id", "p")
        payload = json.loads(run(runner, issue_cli.project, "list", "--json"))
        assert [row["id"] for row in payload] == ["p"]

    def test_a_refusal_exits_non_zero_with_its_classification(self, runner):
        run(runner, issue_cli.project, "create", "P", "--id", "p")
        result = runner.invoke(issue_cli.project, ["create", "Q", "--id", "p"])
        assert result.exit_code == 1
        assert "[conflict]" in result.output

    def test_export_renders_markdown(self, runner, repo, tmp_path):
        run(runner, issue_cli.project, "create", "P", "--id", "p", "--prefix", "pp")
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "a defect",
            "--project",
            "p",
            "--severity",
            "P2",
        )
        out = run(runner, issue_cli.project, "export", "p")
        assert "## pp-0001 — [P2] a defect" in out


class TestIssueCommands:
    @pytest.fixture(autouse=True)
    def project(self, runner, repo):
        run(
            runner,
            issue_cli.project,
            "create",
            "CAO System",
            "--id",
            "cao-system",
            "--prefix",
            "cond",
            "--path",
            str(repo),
        )

    def test_file_resolves_from_cwd(self, runner, repo):
        out = run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "spawn crashed",
            "--cwd",
            str(repo / "conduct"),
        )
        assert "cond-0001" in out
        assert "cao-system" in out

    def test_file_into_an_unregistered_directory_fails_loudly(self, runner, tmp_path):
        result = runner.invoke(
            issue_cli.issue, ["file", "--title", "orphan", "--cwd", str(tmp_path / "nowhere")]
        )
        assert result.exit_code == 1
        assert "[unresolved]" in result.output

    def test_list_defaults_to_all_item_kinds(self, runner):
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "project root",
            "--project",
            "cao-system",
            "--kind",
            "project",
        )
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "a bug",
            "--project",
            "cao-system",
        )
        payload = json.loads(run(runner, issue_cli.issue, "list", "--json"))
        assert {row["kind"] for row in payload["issues"]} == {"project", "bug"}

    def test_audit_renders_the_recursive_projection(self, runner):
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "root",
            "--project",
            "cao-system",
            "--kind",
            "project",
        )
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "task",
            "--project",
            "cao-system",
            "--kind",
            "task",
        )
        run(
            runner,
            issue_cli.issue,
            "link",
            "cond-0002",
            "--to",
            "cond-0001",
            "--kind",
            "part-of",
        )

        payload = json.loads(run(runner, issue_cli.issue, "audit", "cond-0001", "--json"))

        assert payload["counts"]["nodes"] == 2
        assert [row["key"] for row in payload["frontier"]] == ["cond-0002"]

    def test_bug_filing_requires_diagnostics_unless_explicitly_overridden(self, runner):
        refused = runner.invoke(
            issue_cli.issue,
            ["file", "--title", "underspecified", "--project", "cao-system"],
        )
        assert refused.exit_code == 1
        assert "reproduction_steps" in refused.output

        complete = runner.invoke(
            issue_cli.issue,
            [
                "file",
                "--title",
                "complete",
                "--project",
                "cao-system",
                "--reproduction",
                "1. start the server",
                "--expected-outcome",
                "the server starts",
                "--actual-outcome",
                "the process exits",
            ],
        )
        assert complete.exit_code == 0, complete.output + str(complete.exception)

        forced = runner.invoke(
            issue_cli.issue,
            ["file", "--title", "explicit exception", "--project", "cao-system", "--force"],
        )
        assert forced.exit_code == 0, forced.output + str(forced.exception)

    def test_the_full_lifecycle(self, runner):
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "a defect",
            "--project",
            "cao-system",
            "--severity",
            "P2",
            "--component",
            "conduct",
            "--label",
            "noisy",
            "--reproduction",
            "1. run the probe",
        )
        run(runner, issue_cli.issue, "edit", "cond-0001", "--assignee", "terra", "--actor", "colin")
        run(
            runner,
            issue_cli.issue,
            "comment",
            "cond-0001",
            "--body",
            "reproduced",
            "--author",
            "colin",
        )
        run(runner, issue_cli.issue, "close", "cond-0001", "--resolution", "fixed in #12")
        out = run(runner, issue_cli.issue, "show", "cond-0001")
        assert "status:    closed" in out
        assert "resolution: fixed in #12" in out
        assert "reproduction_steps:1. run the probe" in out
        assert "reproduced" in out

    def test_an_edit_with_nothing_to_change_is_refused(self, runner):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        result = runner.invoke(issue_cli.issue, ["edit", "cond-0001"])
        assert result.exit_code == 1
        assert "nothing to change" in result.output

    def test_list_filters(self, runner):
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "high",
            "--project",
            "cao-system",
            "--severity",
            "P1",
        )
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "low",
            "--project",
            "cao-system",
            "--severity",
            "P4",
        )
        out = run(runner, issue_cli.issue, "list", "--project", "cao-system", "--severity", "P1")
        assert "high" in out
        assert "low" not in out

    def test_an_invalid_severity_is_rejected_by_the_parser(self, runner):
        result = runner.invoke(
            issue_cli.issue, ["file", "--title", "x", "--project", "cao-system", "--severity", "P9"]
        )
        assert result.exit_code != 0
        assert "P9" in result.output

    def test_stats_break_down_the_project(self, runner):
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "a",
            "--project",
            "cao-system",
            "--severity",
            "P1",
        )
        payload = json.loads(
            run(runner, issue_cli.issue, "stats", "--project", "cao-system", "--json")
        )
        assert payload["by_severity"]["P1"] == 1

    def test_rm_requires_confirmation(self, runner):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        assert runner.invoke(issue_cli.issue, ["rm", "cond-0001"], input="n\n").exit_code != 0
        assert tracker.get_issue("cond-0001")["key"] == "cond-0001"
        run(runner, issue_cli.issue, "rm", "cond-0001", "--yes")


class TestLedgerImportCommand:
    LEDGER = (
        "# Open issues\n\n---\n\n"
        "## cond-0025 — [P2] a migrated defect\n\n"
        "- **filed:** 2026-07-21T11:44:29Z\n"
        "- **reporter:** human\n"
        "- **status:** open\n\n"
        "body text\n\n---\n"
    )

    @pytest.fixture
    def ledger(self, tmp_path):
        path = tmp_path / "OPEN_ISSUES.md"
        path.write_text(self.LEDGER, encoding="utf-8")
        return path

    def test_dry_run_writes_nothing(self, runner, ledger):
        run(runner, issue_cli.project, "create", "P", "--id", "p", "--prefix", "cond")
        out = run(
            runner, issue_cli.issue, "import-ledger", str(ledger), "--project", "p", "--dry-run"
        )
        assert "would import 1" in out
        assert tracker.list_issues(project_id="p")["total"] == 0

    def test_import_preserves_the_id_and_filing_date(self, runner, ledger):
        run(runner, issue_cli.project, "create", "P", "--id", "p", "--prefix", "cond")
        run(runner, issue_cli.issue, "import-ledger", str(ledger), "--project", "p")
        row = tracker.get_issue("cond-0025")
        assert row["created_at"] == "2026-07-21T11:44:29Z"
        assert row["severity"] == "P2"

    def test_re_running_skips_rather_than_duplicating(self, runner, ledger):
        run(runner, issue_cli.project, "create", "P", "--id", "p", "--prefix", "cond")
        run(runner, issue_cli.issue, "import-ledger", str(ledger), "--project", "p")
        out = run(runner, issue_cli.issue, "import-ledger", str(ledger), "--project", "p")
        assert "1 skipped" in out or "already present" in out
        assert tracker.list_issues(project_id="p")["total"] == 1


class TestMapCommands:
    """Map membership over the CLI: part-of links, children, and directional
    rendering in `show` (cond-0394)."""

    @pytest.fixture(autouse=True)
    def project(self, runner, repo):
        run(
            runner,
            issue_cli.project,
            "create",
            "CAO System",
            "--id",
            "cao-system",
            "--prefix",
            "cond",
            "--path",
            str(repo),
        )
        run(runner, issue_cli.issue, "file", "--title", "the map", "--project", "cao-system")
        run(runner, issue_cli.issue, "file", "--title", "a ticket", "--project", "cao-system")

    def test_part_of_link_and_children(self, runner):
        run(
            runner,
            issue_cli.issue,
            "link",
            "cond-0002",
            "--to",
            "cond-0001",
            "--kind",
            "part-of",
        )
        out = run(runner, issue_cli.issue, "children", "cond-0001")
        assert "cond-0002" in out
        assert "a ticket" in out

    def test_children_json_is_parseable(self, runner):
        run(runner, issue_cli.issue, "link", "cond-0002", "--to", "cond-0001", "--kind", "part-of")
        payload = json.loads(run(runner, issue_cli.issue, "children", "cond-0001", "--json"))
        assert [c["key"] for c in payload["children"]] == ["cond-0002"]

    def test_show_distinguishes_part_of_from_contains(self, runner):
        run(runner, issue_cli.issue, "link", "cond-0002", "--to", "cond-0001", "--kind", "part-of")
        child_view = run(runner, issue_cli.issue, "show", "cond-0002")
        parent_view = run(runner, issue_cli.issue, "show", "cond-0001")
        assert "part of cond-0001" in child_view
        assert "contains cond-0002" in parent_view

    def test_show_distinguishes_blocks_from_blocked_by(self, runner):
        run(runner, issue_cli.issue, "link", "cond-0001", "--to", "cond-0002", "--kind", "blocks")
        blocker_view = run(runner, issue_cli.issue, "show", "cond-0001")
        blocked_view = run(runner, issue_cli.issue, "show", "cond-0002")
        assert "blocks cond-0002" in blocker_view
        assert "blocked by cond-0001" in blocked_view


class TestFrontierCommand:
    @pytest.fixture(autouse=True)
    def project(self, runner, repo):
        run(
            runner,
            issue_cli.project,
            "create",
            "CAO System",
            "--id",
            "cao-system",
            "--prefix",
            "cond",
            "--path",
            str(repo),
        )

    def _map_with_two_tickets(self, runner):
        run(runner, issue_cli.issue, "file", "--title", "the map", "--project", "cao-system")
        run(runner, issue_cli.issue, "file", "--title", "first ticket", "--project", "cao-system")
        run(runner, issue_cli.issue, "file", "--title", "second ticket", "--project", "cao-system")
        for ticket in ("cond-0002", "cond-0003"):
            run(runner, issue_cli.issue, "link", ticket, "--to", "cond-0001", "--kind", "part-of")

    def test_frontier_lists_takeable_tickets_oldest_first(self, runner):
        self._map_with_two_tickets(runner)
        payload = json.loads(run(runner, issue_cli.issue, "frontier", "cond-0001", "--json"))
        assert [t["key"] for t in payload["frontier"]] == ["cond-0002", "cond-0003"]

    def test_frontier_skips_claimed_and_blocked_tickets(self, runner):
        self._map_with_two_tickets(runner)
        run(runner, issue_cli.issue, "claim", "cond-0002", "--as", "terra")
        run(
            runner,
            issue_cli.issue,
            "link",
            "cond-0002",
            "--to",
            "cond-0003",
            "--kind",
            "blocks",
        )
        payload = json.loads(run(runner, issue_cli.issue, "frontier", "cond-0001", "--json"))
        # cond-0002 is claimed; cond-0003 is blocked by the (claimed, open)
        # cond-0002. The frontier is empty and says so truthfully.
        assert payload["frontier"] == []
        out = run(runner, issue_cli.issue, "frontier", "cond-0001")
        assert "nothing takeable" in out

    def test_frontier_of_an_unknown_map_refuses(self, runner):
        result = runner.invoke(issue_cli.issue, ["frontier", "cond-9999"])
        assert result.exit_code == 1
        assert "[not-found]" in result.output


class TestClaimCommands:
    @pytest.fixture(autouse=True)
    def project(self, runner, repo):
        run(
            runner,
            issue_cli.project,
            "create",
            "CAO System",
            "--id",
            "cao-system",
            "--prefix",
            "cond",
            "--path",
            str(repo),
        )
        run(runner, issue_cli.issue, "file", "--title", "a ticket", "--project", "cao-system")

    def test_claim_then_unclaim_round_trip(self, runner):
        out = run(runner, issue_cli.issue, "claim", "cond-0001", "--as", "terra")
        assert "claimed by terra" in out
        assert tracker.get_issue("cond-0001")["assignee"] == "terra"
        out = run(runner, issue_cli.issue, "unclaim", "cond-0001", "--actor", "colin")
        assert "released" in out
        assert tracker.get_issue("cond-0001")["assignee"] is None

    def test_a_conflicting_claim_fails_and_names_the_claimant(self, runner):
        run(runner, issue_cli.issue, "claim", "cond-0001", "--as", "terra")
        result = runner.invoke(issue_cli.issue, ["claim", "cond-0001", "--as", "muse"])
        assert result.exit_code == 1
        assert "[conflict]" in result.output
        assert "terra" in result.output

    def test_a_retry_by_the_same_claimant_is_idempotent(self, runner):
        run(runner, issue_cli.issue, "claim", "cond-0001", "--as", "terra")
        out = run(runner, issue_cli.issue, "claim", "cond-0001", "--as", "terra")
        assert "already claimed" in out

    def test_claim_json_is_parseable(self, runner):
        payload = json.loads(
            run(runner, issue_cli.issue, "claim", "cond-0001", "--as", "terra", "--json")
        )
        assert (payload["assignee"], payload["claimed"]) == ("terra", True)


class TestEditConcurrencyAndLabelFlags:
    @pytest.fixture(autouse=True)
    def project(self, runner, repo):
        run(
            runner,
            issue_cli.project,
            "create",
            "CAO System",
            "--id",
            "cao-system",
            "--prefix",
            "cond",
            "--path",
            str(repo),
        )
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "a ticket",
            "--project",
            "cao-system",
            "--label",
            "needs-triage",
        )

    def test_edit_with_a_matching_expect_updated_at_applies(self, runner):
        current = tracker.get_issue("cond-0001")["updated_at"]
        run(
            runner,
            issue_cli.issue,
            "edit",
            "cond-0001",
            "--body",
            "v2",
            "--expect-updated-at",
            current,
        )
        assert tracker.get_issue("cond-0001")["body"] == "v2"

    def test_edit_with_a_stale_expect_updated_at_is_a_conflict(self, runner):
        stale = tracker.get_issue("cond-0001")["updated_at"]
        run(runner, issue_cli.issue, "edit", "cond-0001", "--body", "somebody else")
        result = runner.invoke(
            issue_cli.issue,
            ["edit", "cond-0001", "--body", "v2", "--expect-updated-at", stale],
        )
        assert result.exit_code == 1
        assert "[conflict]" in result.output
        assert tracker.get_issue("cond-0001")["body"] == "somebody else"

    def test_close_forwards_the_clock_fence_and_reports_the_status_effect(self, runner):
        current = tracker.get_issue("cond-0001")["updated_at"]
        payload = json.loads(
            run(
                runner,
                issue_cli.issue,
                "close",
                "cond-0001",
                "--as",
                "resolved",
                "--expect-updated-at",
                current,
                "--json",
            )
        )
        assert payload["status"] == "resolved"
        assert payload["effect_id"] in payload["effect_ids"]
        assert payload["updated_at"] == tracker.get_issue("cond-0001")["updated_at"]

    def test_comment_and_link_flags_forward_reviewed_clocks(self, runner):
        run(runner, issue_cli.issue, "file", "--title", "other", "--project", "cao-system")
        source = tracker.get_issue("cond-0001")
        target = tracker.get_issue("cond-0002")
        comment = json.loads(
            run(
                runner,
                issue_cli.issue,
                "comment",
                "cond-0001",
                "--body",
                "audited",
                "--expect-updated-at",
                source["updated_at"],
                "--json",
            )
        )
        assert comment["effect_id"] > 0
        source_after_comment = tracker.get_issue("cond-0001")
        link = json.loads(
            run(
                runner,
                issue_cli.issue,
                "link",
                "cond-0001",
                "--to",
                "cond-0002",
                "--expect-from-updated-at",
                source_after_comment["updated_at"],
                "--expect-to-updated-at",
                target["updated_at"],
                "--json",
            )
        )
        assert len(link["effect_ids"]) == 2

    def test_add_and_remove_label_flags_merge(self, runner):
        run(
            runner,
            issue_cli.issue,
            "edit",
            "cond-0001",
            "--add-label",
            "ready-for-agent",
            "--remove-label",
            "needs-triage",
        )
        assert tracker.get_issue("cond-0001")["labels"] == ["ready-for-agent"]

    def test_clear_labels_empties_the_set(self, runner):
        run(runner, issue_cli.issue, "edit", "cond-0001", "--clear-labels")
        assert tracker.get_issue("cond-0001")["labels"] == []

    def test_full_replacement_cannot_mix_with_a_delta(self, runner):
        result = runner.invoke(
            issue_cli.issue,
            ["edit", "cond-0001", "--label", "x", "--add-label", "y"],
        )
        assert result.exit_code == 1
        assert "[invalid]" in result.output


class TestListDiscoveryFlags:
    @pytest.fixture(autouse=True)
    def project(self, runner, repo):
        run(
            runner,
            issue_cli.project,
            "create",
            "CAO System",
            "--id",
            "cao-system",
            "--prefix",
            "cond",
            "--path",
            str(repo),
        )

    def test_unlabeled_lists_only_labelless_issues(self, runner):
        run(runner, issue_cli.issue, "file", "--title", "bare", "--project", "cao-system")
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "tagged",
            "--project",
            "cao-system",
            "--label",
            "bug",
        )
        payload = json.loads(
            run(runner, issue_cli.issue, "list", "--project", "cao-system", "--unlabeled", "--json")
        )
        assert [i["title"] for i in payload["issues"]] == ["bare"]

    def test_without_label_is_repeatable_and_exact(self, runner):
        for title, label in [
            ("ready", "source:wayfinder"),
            ("triaged", "needs-triage"),
            ("waiting", "needs-info"),
            ("similar", "needs-info-extra"),
        ]:
            run(
                runner,
                issue_cli.issue,
                "file",
                "--title",
                title,
                "--project",
                "cao-system",
                "--label",
                label,
            )
        payload = json.loads(
            run(
                runner,
                issue_cli.issue,
                "list",
                "--project",
                "cao-system",
                "--without-label",
                "needs-triage",
                "--without-label",
                "needs-info",
                "--json",
            )
        )
        assert {i["title"] for i in payload["issues"]} == {"ready", "similar"}

    def test_kind_selects_bug_feature_or_all_with_all_the_default(self, runner):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        run(
            runner,
            issue_cli.feature,
            "file",
            "--title",
            "a wish",
            "--project",
            "cao-system",
        )
        default = json.loads(run(runner, issue_cli.issue, "list", "--json"))
        assert sorted(i["title"] for i in default["issues"]) == ["a defect", "a wish"]
        everything = json.loads(run(runner, issue_cli.issue, "list", "--kind", "all", "--json"))
        assert sorted(i["title"] for i in everything["issues"]) == ["a defect", "a wish"]
        features = json.loads(run(runner, issue_cli.issue, "list", "--kind", "feature", "--json"))
        assert [i["title"] for i in features["issues"]] == ["a wish"]


class TestObservedRevisionAndImportance:
    """cond-0636 CLI surfaces: observed_revision on file/edit and the
    reversible comment-importance command."""

    @pytest.fixture(autouse=True)
    def project(self, runner, repo):
        run(
            runner,
            issue_cli.project,
            "create",
            "CAO System",
            "--id",
            "cao-system",
            "--prefix",
            "cond",
            "--path",
            str(repo),
        )

    def test_file_records_an_observed_revision(self, runner, repo):
        run(
            runner,
            issue_cli.issue,
            "file",
            "--title",
            "a defect",
            "--project",
            "cao-system",
            "--observed-revision",
            "v1.2.3",
        )
        out = run(runner, issue_cli.issue, "show", "cond-0001")
        assert "v1.2.3" in out

    def test_edit_updates_and_clears_the_observed_revision(self, runner, repo):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        run(
            runner,
            issue_cli.issue,
            "edit",
            "cond-0001",
            "--observed-revision",
            "abc1234",
            "--actor",
            "colin",
        )
        detail = json.loads(run(runner, issue_cli.issue, "show", "cond-0001", "--json"))
        assert detail["observed_revision"] == "abc1234"
        run(runner, issue_cli.issue, "edit", "cond-0001", "--observed-revision", "")
        detail = json.loads(run(runner, issue_cli.issue, "show", "cond-0001", "--json"))
        assert detail["observed_revision"] is None

    def test_comment_accepts_important_at_creation(self, runner, repo):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        run(
            runner,
            issue_cli.issue,
            "comment",
            "cond-0001",
            "--body",
            "root cause",
            "--important",
        )
        detail = json.loads(run(runner, issue_cli.issue, "show", "cond-0001", "--json"))
        assert [c["important"] for c in detail["comments"]] == [True]

    def test_show_marks_important_comments_in_prose_output(self, runner, repo):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        run(runner, issue_cli.issue, "comment", "cond-0001", "--body", "root cause", "--important")
        run(runner, issue_cli.issue, "comment", "cond-0001", "--body", "ordinary chatter")
        out = run(runner, issue_cli.issue, "show", "cond-0001")
        assert out.count("[important]") == 1

    def test_importance_set_and_clear_round_trip(self, runner, repo):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        run(runner, issue_cli.issue, "comment", "cond-0001", "--body", "note")

        out = run(runner, issue_cli.issue, "comment-importance", "cond-0001", "1", "important")
        assert "-> important" in out
        out = run(runner, issue_cli.issue, "comment-importance", "cond-0001", "1", "routine")
        assert "-> routine" in out

    def test_a_same_value_retry_reports_it_changed_nothing(self, runner, repo):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        run(runner, issue_cli.issue, "comment", "cond-0001", "--body", "note")
        run(runner, issue_cli.issue, "comment-importance", "cond-0001", "1", "important")
        out = run(runner, issue_cli.issue, "comment-importance", "cond-0001", "1", "important")
        assert "already important" in out

    def test_importance_json_is_parseable(self, runner, repo):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        run(runner, issue_cli.issue, "comment", "cond-0001", "--body", "note")
        payload = json.loads(
            run(
                runner,
                issue_cli.issue,
                "comment-importance",
                "cond-0001",
                "1",
                "important",
                "--json",
            )
        )
        assert payload["changed"] is True
        assert payload["important"] is True

    def test_importance_on_an_unknown_comment_refuses(self, runner, repo):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        result = runner.invoke(
            issue_cli.issue, ["comment-importance", "cond-0001", "99", "important"]
        )
        assert result.exit_code != 0

    def test_an_invalid_weight_word_is_rejected_by_the_parser(self, runner, repo):
        run(runner, issue_cli.issue, "file", "--title", "a defect", "--project", "cao-system")
        result = runner.invoke(
            issue_cli.issue, ["comment-importance", "cond-0001", "1", "critical"]
        )
        assert result.exit_code != 0
