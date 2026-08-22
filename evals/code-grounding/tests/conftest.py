"""Shared synthetic-repository fixture for the code-grounding eval tests.

Builds a tiny deterministic git repository (two commits) so the fixture
loader, retrieval primitives, and baseline runner can be exercised offline
without touching either real source checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

INIT_FILES = {
    "svc/greeter.py": (
        "def greet_user(name):\n"
        "    return _format_message(f'hello {name}')\n"
        "\n"
        "\n"
        "def _format_message(text):\n"
        "    # adds emphasis to greeting text\n"
        "    return text.upper()\n"
    ),
    "docs/readme.md": "greeter docs\n",
    ".gitignore": "secrets/\n",
    "secrets/keys.txt": "tracked-but-ignored\ndata: deadman_minutes=30\n",
    ".hidden/hook.sh": "#!/bin/sh\n# stale greeting pipeline hook\necho stale\n",
}
FIX_FILES = {
    "svc/greeter.py": (
        "def greet_user(name):\n"
        "    return _format_message(f'hello {name} !!')\n"
        "\n"
        "\n"
        "def _format_message(text):\n"
        "    # adds emphasis to greeting text\n"
        "    return text.upper()\n"
    ),
    "svc/farewell.py": "def farewell_user(name):\n    return f'bye {name}'\n",
    "tests/test_greeter.py": "def test_greet():\n    assert greet_user('a')\n",
    ".hidden/hook.sh": "#!/bin/sh\nfor v in $(git rev-parse --local-env-vars); do unset $v; done\necho ok\n",
}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout


@pytest.fixture()
def toy_repo(tmp_path: Path) -> dict:
    """Two-commit repo: 'init' then a fix commit touching greeter.py, adding
    farewell.py + tests, and rewriting the hidden hook."""
    repo = tmp_path / "toy"
    (repo / "svc").mkdir(parents=True)
    for rel, text in INIT_FILES.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _git(repo, "init", "-q", "-b", "main")
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-qm", "init")
    init_sha = _git(repo, "rev-parse", "HEAD").strip()

    for rel, text in FIX_FILES.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _git(repo, "commit", "-aqm", "fix(toy): polish greeting, add farewell, clean hook env")
    fix_sha = _git(repo, "rev-parse", "HEAD").strip()

    return {
        "path": repo,
        "init_sha": init_sha,
        "fix_sha": fix_sha,
        # parent of the fix = pre-fix search tree
        "search_sha": _git(repo, "rev-parse", f"{fix_sha}^").strip(),
    }


def mini_fixture(toy_repo: dict, tmp_path: Path) -> Path:
    """A one-case fixture JSON wired to the synthetic repo."""
    case = {
        "id": "toy-0001",
        "issue": {
            "key": "toy-0001",
            "tracker_project": "toy-system",
            "status": "closed",
            "component": "toy",
            "title": "greeting output lacks emphasis punctuation",
            "narrative": (
                "greeting output lacks emphasis punctuation\n\n"
                "Users report `hello bob` renders without emphasis. "
                "The `greeting pipeline` should route through message formatting. "
                "Error: emphasis missing in rendered output\n"
            ),
        },
        "case_types": ["vague-prose"],
        "notes": None,
        "shared_fix_commit": None,
        "test_only_fix": False,
        "repos": {
            "toy": {
                "search_sha": toy_repo["search_sha"],
                "fix_commits": [
                    {"sha": toy_repo["fix_sha"], "short": toy_repo["fix_sha"][:8], "subject": "fix"}
                ],
                "pull_request": None,
                "expected_files": [
                    {
                        "path": "svc/greeter.py",
                        "diff_status": "M",
                        "in_search_tree": True,
                        "verified_against_diff": True,
                    },
                    {
                        "path": "svc/farewell.py",
                        "diff_status": "A",
                        "in_search_tree": False,
                        "verified_against_diff": True,
                    },
                    {
                        "path": ".hidden/hook.sh",
                        "diff_status": "M",
                        "in_search_tree": True,
                        "verified_against_diff": True,
                    },
                ],
                "expected_symbols": [
                    {
                        "name": "_format_message",
                        "origin": "preexisting",
                        "pre_fix_grep_hits": 2,
                        "diff_files": ["svc/greeter.py"],
                    },
                    {
                        "name": "farewell_user",
                        "origin": "introduced",
                        "pre_fix_grep_hits": 0,
                        "diff_files": ["svc/farewell.py"],
                    },
                ],
                "symbol_note": None,
            }
        },
        "tool_lanes": {"codebase_memory": None, "qmd": None, "serena": None},
    }
    fixture = {
        "schema_version": 1,
        "description": "synthetic mini fixture",
        "meta": {
            "generated_by": "test",
            "tracker_project": "toy-system",
            "repos": {
                "toy": {"local_path": str(toy_repo["path"]), "env_override": "CAO_EVAL_REPO_TOY"}
            },
            "authoring_base": {},
            "case_count": 1,
            "case_type_coverage": ["vague-prose"],
        },
        "cases": [case],
    }
    out = tmp_path / "mini-cases.json"
    out.write_text(json.dumps(fixture, indent=2))
    return out


import json  # noqa: E402  (kept late so fixtures above stay readable)
