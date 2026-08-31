"""Contract tests for the fork side of the CAO development map."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENTS = REPO / "AGENTS.md"
LOCAL_MANIFEST = REPO / ".agent-tools" / "code-intelligence.yml"


def guidance() -> str:
    return " ".join(AGENTS.read_text(encoding="utf-8").split())


def development_map() -> str:
    text = guidance()
    start = text.index("## CAO development map")
    end = text.index("## Agent skills")
    return text[start:end]


def test_fork_start_selects_the_conductor_manifest_and_fork_lane() -> None:
    text = guidance()
    section = development_map()

    assert text.index("## CAO development map") < text.index("## Agent skills")
    assert "tracker project is `cao-system`" in section
    assert "logical repository `cao-conductor`" in section
    assert "`.agent-tools/code-intelligence.yml`" in section
    assert "This fork carries no second manifest" in section
    assert not LOCAL_MANIFEST.exists()
    assert "package root is `src/cli_agent_orchestrator/`" in section
    assert "owns CAO runtime, provider integration, server/API, and dashboard" in section
    assert "Fork-only changes stay in this repository" in section


def test_cross_boundary_change_routes_to_a_conductor_companion_lane() -> None:
    section = development_map()

    assert "campaign policy, managed execution, skills, project configuration" in section
    assert "tracker workflow" in section
    assert "deployment coordination" in section
    assert "inspect both repositories and record both revisions" in section
    assert "cross-repository writes use an explicit companion lane/worktree/PR" in section
    assert "`/tracker/issues/similar`" in section
    assert "the conductor CLI is a thin client" in section


def test_branch_local_source_wins_when_index_and_worktree_diverge() -> None:
    section = development_map()

    assert "index and worktree may diverge in either direction" in section
    assert "stable/main index omits branch-local and uncommitted changes" in section
    assert "inspect the exact worktree and its diff" in section
    assert "Direct source at the resolved branch/worktree revision is authoritative" in section
    assert "`repository | resolved target SHA | path`" in section


def test_optional_search_degrades_to_direct_tools_without_copying_manuals() -> None:
    section = development_map()

    assert "QMD and codebase-memory are optional" in section
    assert "continue with direct source, `rg`, and Git" in section
    assert "Worker ownership is currently one scalar worktree path" in section
    assert "issue's `worktrees` list is provenance, not worker ownership" in section
    assert "never one synthetic repository index" in section

    for generic_command in (
        "qmd update",
        "qmd embed",
        "auto_watch",
        "index_repository",
        "check_index_coverage",
    ):
        assert generic_command not in section
