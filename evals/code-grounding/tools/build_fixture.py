"""Build evals/code-grounding/fixtures/cases.json from hand-sampled selections.

Every expected file/symbol below was selected by reading the actual fix diff
at the pinned SHA (manual sampling), then mechanically verified here:
existence of each file at the pre-fix search tree, and pre-fix presence of
each symbol via `git grep` over the whole repository. Sampling decisions and
verification evidence land in reports/fixture-verification.json.

Read-only w.r.t. both source repositories and the tracker: issue bodies are
fetched through `conduct issue show`.

Usage:
    uv run python evals/code-grounding/tools/build_fixture.py \
        --bodies-cache /path/to/full-bodies.json   # optional offline cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOS = {
    "cao-conductor": Path("/Users/colin/Projects/cao-conductor"),
    "cao": Path("/Users/colin/Projects/cli-agent-orchestrator"),
}

# Hand-sampled selections: for every case-repo, the fix commits carrying the
# resolution, the expected files (path + expected diff status), and candidate
# symbols read off the changed hunks. The verifier classifies each candidate
# as preexisting or introduced against the search tree.
SELECTIONS: dict[str, dict] = {
    "cond-0209": {
        "case_types": ["cross-repository", "data-flow"],
        "notes": "Attachment release spans ledger semantics in conductor and the CAO native-attachment lifecycle.",
        "repos": {
            "cao-conductor": {
                "fixes": ["d2f1dc6d1685"],
                "files": [
                    ("conduct/commands/issue.py", "M"),
                    ("conduct/lib/issues.py", "M"),
                    ("conduct/lib/caoapi.py", "M"),
                ],
                "symbols": ["_file_to_tracker", "resolve_tracker_project", "file_issue"],
            },
            "cao": {
                "fixes": ["5d4c4b0b0437"],
                "pr": "colindmurray/cli-agent-orchestrator#76",
                "files": [
                    ("src/cli_agent_orchestrator/services/native_attachment.py", "M"),
                    ("src/cli_agent_orchestrator/services/terminal_service.py", "M"),
                    ("src/cli_agent_orchestrator/api/attachments.py", "A"),
                    ("src/cli_agent_orchestrator/services/native_attachment_recovery.py", "A"),
                ],
                "symbols": [],
                "symbol_note": "Merge PR adds module-level surfaces; expectations carried at file level.",
            },
        },
    },
    "cond-0072": {
        "case_types": ["cross-repository", "data-flow"],
        "notes": "Delivered inbox rows must be reconciled against model-turn receipts in both repos.",
        "repos": {
            "cao-conductor": {
                "fixes": ["81d8f556f0b7", "6431c41861b9", "3d26d5f20140"],
                "files": [
                    ("plugin/conductor_sentinel/model_turn_receipt_contract.py", "A"),
                    ("plugin/conductor_sentinel/model_turn_lookup.py", "A"),
                    ("plugin/conductor_sentinel/refusal_recovery.py", "M"),
                    ("conduct/commands/report.py", "M"),
                    ("conduct/commands/retire.py", "M"),
                ],
                "symbols": [
                    "ModelTurnVerifier",
                    "_reconcile_processing",
                    "_finalization_turn_finality",
                ],
            },
            "cao": {
                "fixes": ["d7b22eeca13d"],
                "files": [
                    ("src/cli_agent_orchestrator/services/control_input_journal.py", "M"),
                    ("src/cli_agent_orchestrator/services/control_input_service.py", "M"),
                    ("src/cli_agent_orchestrator/services/inbox_service.py", "M"),
                    ("src/cli_agent_orchestrator/services/wake_receipts.py", "A"),
                ],
                "symbols": [
                    "deliver_control_input",
                    "_native_composer_preflight",
                    "control_input_request_digest",
                ],
            },
        },
    },
    "cond-0107": {
        "case_types": ["cross-repository"],
        "notes": "v2 preflight failure recovery must use v2 verbs on both sides of the API.",
        "repos": {
            "cao-conductor": {
                "fixes": ["16b6fc83b4d3"],
                "files": [
                    ("conduct/commands/spawn.py", "M"),
                    ("conduct/lib/spawn_delivery.py", "M"),
                    ("conduct/lib/caoapi.py", "M"),
                ],
                "symbols": [
                    "_complete_launch_failed_bridge",
                    "_forfeit_stale_breaker",
                    "cleanup_managed_generation_v2",
                ],
            },
            "cao": {
                "fixes": ["108755642cc7"],
                "files": [
                    ("src/cli_agent_orchestrator/services/managed_launch_v2.py", "M"),
                    ("src/cli_agent_orchestrator/models/managed_launch_v2.py", "M"),
                    ("src/cli_agent_orchestrator/api/main.py", "M"),
                    ("src/cli_agent_orchestrator/services/vintage_migration.py", "M"),
                ],
                "symbols": [
                    "_validate_readiness_for_bind",
                    "_mark_preflight_blocked",
                    "ManagedLaunchV2CleanupRequest",
                ],
            },
        },
    },
    "cond-0173": {
        "case_types": ["cross-repository"],
        "notes": "Abandoned journals alerted forever: dedupe/finality in conductor, truthful legacy vocabulary in CAO.",
        "repos": {
            "cao-conductor": {
                "fixes": ["9b3453a6c007"],
                "pr": "colindmurray/cao-conductor#26",
                "files": [
                    ("plugin/conductor_sentinel/_dedupe.py", "A"),
                    ("plugin/conductor_sentinel/sentinel.py", "M"),
                    ("conduct/lib/spawn_delivery.py", "M"),
                    ("conduct/commands/spawn.py", "M"),
                ],
                "symbols": ["_alert_recovery_failure", "_finalize_recovered_run"],
            },
            "cao": {
                "fixes": ["a5f6bae75487"],
                "files": [("src/cli_agent_orchestrator/models/terminal.py", "M")],
                "symbols": [],
                "symbol_note": "Fix admits lifecycle vocabulary at response boundary; TerminalLifecycleState is introduced by the fix itself.",
            },
        },
    },
    "cond-0462": {
        "case_types": ["stack-trace", "vague-prose"],
        "notes": "KeyError on missing source occurrence digest during successor dispose commit.",
        "repos": {
            "cao-conductor": {
                "fixes": ["015f62699a26"],
                "pr": "colindmurray/cao-conductor#172",
                "files": [
                    ("conduct/commands/_successor.py", "M"),
                    ("tests/test_dispose_commit.py", "A"),
                ],
                "symbols": ["dispose_claim", "_journal_binding"],
            },
        },
    },
    "cond-0549": {
        "case_types": ["stack-trace", "data-flow"],
        "notes": "report-on-behalf teardown must release leases/claims through retained-track custody flow.",
        "repos": {
            "cao-conductor": {
                "fixes": ["a1e4f3f8b185"],
                "files": [
                    ("conduct/commands/report_on_behalf.py", "M"),
                    ("conduct/lib/retained_track.py", "M"),
                    ("conduct/lib/workstate.py", "M"),
                    ("conduct/lib/state.py", "M"),
                ],
                "symbols": ["_pr_custody_equivalent", "run_retirement"],
            },
        },
    },
    "cond-0055": {
        "case_types": ["stack-trace", "vague-prose"],
        "notes": "Codex safety interstitial wedged unattended terminals; fixed inside the shared postfix adjudication commit that also carries sibling work (sampling notes the sharing).",
        "shared_fix_commit": {
            "sha": "9eb0b208c956",
            "also_carries": "cond-0076, W8/W9/W10 hardening",
        },
        "repos": {
            "cao-conductor": {
                "fixes": ["9eb0b208c956"],
                "files": [
                    ("plugin/conductor_sentinel/interstitial_recovery.py", "M"),
                    ("plugin/conductor_sentinel/refusal_recovery.py", "A"),
                    ("plugin/conductor_sentinel/sentinel.py", "M"),
                ],
                "symbols": ["RefusalBoundary", "RefusalBoundaryUnavailable"],
            },
        },
    },
    "cond-0464": {
        "case_types": ["stack-trace"],
        "notes": "Local suites inherited operator settings.json; isolation lives in test fixtures, not src/.",
        "repos": {
            "cao": {
                "fixes": ["04adf8764e62"],
                "pr": "colindmurray/cli-agent-orchestrator#130",
                "files": [
                    ("test/conftest.py", "M"),
                    ("test/fixtures/cao_server.py", "M"),
                ],
                "symbols": ["CaoServer", "_subprocess_env"],
            },
        },
    },
    "cond-0093": {
        "case_types": ["exact-technical"],
        "notes": "Exact pin string 'codex-cli 0.144.6' gates managed launches; bump to 0.145.0.",
        "repos": {
            "cao": {
                "fixes": ["655bcfbcc711"],
                "files": [
                    ("src/cli_agent_orchestrator/services/codex_trust.py", "M"),
                    ("docs/managed-launch-protocol.md", "M"),
                ],
                "symbols": [
                    "SUPPORTED_CODEX_VERSION",
                    "attest_trusted_project",
                    "CodexTrustProbeError",
                ],
            },
        },
    },
    "cond-0312": {
        "case_types": ["exact-technical"],
        "notes": "Kimi 0.31 process-title rewrite defeats exact-session admission proof; rendered-header proof replaces deleted verifier.",
        "repos": {
            "cao": {
                "fixes": ["06c756ecfbc4"],
                "pr": "colindmurray/cli-agent-orchestrator#66",
                "files": [
                    ("src/cli_agent_orchestrator/services/kimi_native_launch.py", "M"),
                    ("src/cli_agent_orchestrator/services/native_tui_launch.py", "M"),
                    ("src/cli_agent_orchestrator/services/managed_launch_v2.py", "M"),
                ],
                "symbols": ["_verify_bound_session_and_cwd", "RenderedSessionProof"],
            },
        },
    },
    "cond-0343": {
        "case_types": ["exact-technical"],
        "notes": "fire-marshal incident-store lock stale recovery across marshal_incidents and launcher scripts.",
        "repos": {
            "cao-conductor": {
                "fixes": ["1d56d496a844"],
                "pr": "colindmurray/cao-conductor#112",
                "files": [
                    ("conduct/lib/marshal_incidents.py", "M"),
                    ("conduct/commands/marshal.py", "M"),
                    ("scripts/fire-marshal.sh", "M"),
                ],
                "symbols": ["IncidentStoreUnwritable", "_now_utc"],
            },
        },
    },
    "cond-0097": {
        "case_types": ["exact-technical"],
        "notes": "Test-only regression fix: held events lock must retain queued recovery events across ticks; refusal surface must be injected, not live urllib.",
        "test_only_fix": True,
        "repos": {
            "cao-conductor": {
                "fixes": ["21db674ed6a0"],
                "files": [("plugin/tests/test_interstitial_recovery.py", "M")],
                "symbols": ["refusal_get_fn", "EVENTS_FILENAME"],
            },
        },
    },
    "cond-0386": {
        "case_types": ["exact-technical", "data-flow"],
        "notes": "Pre-push hook leaks GIT_DIR/GIT_WORK_TREE into nested test repos; fix clears git local env vars.",
        "repos": {
            "cao": {
                "fixes": ["2c00cc069ec0"],
                "files": [
                    ("cao_mcp_apps/.husky/pre-push", "M"),
                    ("test/ext_apps/test_pre_push_env_isolation.py", "A"),
                ],
                "symbols": [],
                "symbol_note": "Shell hook target; string-level expectation (--local-env-vars) recorded as introduced symbol.",
                "extra_symbols_introduced": ["--local-env-vars"],
            },
        },
    },
    "cond-0012": {
        "case_types": ["vague-prose"],
        "notes": "Role-keyed deadman timeouts missed concrete profile names; projection onto routing profiles.",
        "repos": {
            "cao-conductor": {
                "fixes": ["1067d6480cbb"],
                "files": [("conduct/lib/sentinelcfg.py", "M")],
                "symbols": ["build", "deadman_minutes", "task_classes"],
            },
        },
    },
    "cond-0253": {
        "case_types": ["vague-prose"],
        "notes": "Final collection retired a worker before callback route finality; replay must preserve seals and custody.",
        "repos": {
            "cao-conductor": {
                "fixes": ["24e6ba736b4d"],
                "files": [
                    ("conduct/lib/retirement.py", "M"),
                    ("conduct/commands/retire.py", "M"),
                    ("conduct/commands/status.py", "M"),
                ],
                "symbols": ["run_retirement", "seals_equivalent", "observe_delete"],
            },
        },
    },
    "cond-0071": {
        "case_types": ["vague-prose"],
        "notes": "Active Kimi terminal stayed falsely idle; retire reconciliation now weighs turn-terminal evidence. Shares its fix commit with cond-0293.",
        "shared_fix_commit": {"sha": "3d26d5f20140", "also_resolves": "cond-0293"},
        "repos": {
            "cao-conductor": {
                "fixes": ["3d26d5f20140"],
                "files": [("conduct/commands/retire.py", "M")],
                "symbols": ["mark_turn_quiesced", "_install_terminal_heartbeat"],
            },
        },
    },
    "cond-0131": {
        "case_types": ["data-flow", "exact-technical"],
        "notes": "Mouse-wheel scroll put panes into tmux copy mode so control input was discarded; guard flows from tmux client through delivery service.",
        "repos": {
            "cao": {
                "fixes": ["53b29fd26782"],
                "files": [
                    ("src/cli_agent_orchestrator/clients/tmux.py", "M"),
                    ("src/cli_agent_orchestrator/services/control_input_contract.py", "M"),
                    ("src/cli_agent_orchestrator/services/control_input_service.py", "M"),
                ],
                "symbols": ["_deliver_under_lease", "send_literal"],
            },
        },
    },
    "cond-0270": {
        "case_types": ["history-dependent"],
        "notes": "Follow-up to cond-0266: barrier fix did not migrate registries whose epoch gaps predate plan declarations.",
        "repos": {
            "cao-conductor": {
                "fixes": ["69bf0d5b75af"],
                "files": [
                    ("conduct/lib/campaign_registry.py", "M"),
                    ("skills/plan-epoch-synchronization/SKILL.md", "M"),
                ],
                "symbols": ["legacy_migration_assessment", "_apply_legacy_migration"],
            },
        },
    },
    "cond-0414": {
        "case_types": ["history-dependent"],
        "notes": "Deployed reconciler could not consume PR121-preserved alternate callback evidence; extends COND-0392 command.",
        "repos": {
            "cao-conductor": {
                "fixes": ["ca598999d81d", "d08900e088de"],
                "files": [("conduct/commands/reconcile_report_identity.py", "M")],
                "symbols": ["_extract_alternate_identity", "_read_reconciliation_record"],
            },
        },
    },
    "cond-0050": {
        "case_types": ["history-dependent"],
        "notes": "session-env clear failed open and resurrected stale routing; strict fail-closed clear.",
        "repos": {
            "cao": {
                "fixes": ["8e46d8912c00"],
                "files": [
                    ("src/cli_agent_orchestrator/services/session_env.py", "M"),
                    ("src/cli_agent_orchestrator/services/session_service.py", "M"),
                    ("src/cli_agent_orchestrator/services/terminal_service.py", "M"),
                ],
                "symbols": ["clear_session_env", "delete_session"],
            },
        },
    },
    "cond-0067": {
        "case_types": ["history-dependent"],
        "notes": "Failed preclear cleanup destroyed a colliding live terminal; pre-clear became a true preflight after cond-0050 introduced strictness.",
        "repos": {
            "cao": {
                "fixes": ["deda4ab52afa"],
                "files": [
                    ("src/cli_agent_orchestrator/services/terminal_service.py", "M"),
                    ("test/services/test_terminal_service_full.py", "M"),
                ],
                "symbols": ["create_terminal", "clear_session_env", "generate_terminal_id"],
            },
        },
    },
    "cond-0550": {
        "case_types": ["data-flow"],
        "notes": "Rebuilt on-demand provider lost its pinned route; assigned_route must persist and reconstruct.",
        "repos": {
            "cao": {
                "fixes": ["7401a0820013", "5bbdcc701144"],
                "files": [
                    ("src/cli_agent_orchestrator/providers/manager.py", "M"),
                    ("src/cli_agent_orchestrator/clients/database.py", "M"),
                    ("src/cli_agent_orchestrator/services/terminal_service.py", "M"),
                ],
                "symbols": [
                    "ProviderManager",
                    "assigned_route",
                    "TerminalAssignedRouteIncompleteError",
                ],
            },
        },
    },
    "cond-0422": {
        "case_types": ["data-flow"],
        "notes": "continue must prove the retained round became a model turn; submission proof threads through the receipt client.",
        "repos": {
            "cao-conductor": {
                "fixes": ["5477b9c2775b"],
                "pr": "colindmurray/cao-conductor#156",
                "files": [
                    ("conduct/commands/continue_round.py", "M"),
                    ("tests/fake_cao.py", "M"),
                    ("tests/test_continue_round.py", "M"),
                ],
                "symbols": [
                    "get_message_turn_receipt",
                    "_idle_round_pointer",
                    "SUBMISSION_CONFIRM_SECONDS",
                ],
            },
        },
    },
    "cond-0303": {
        "case_types": ["data-flow", "history-dependent"],
        "notes": "Fresh succession bound routing over historical worktree policy; round number carried on prepared slots, generation bindings appended.",
        "repos": {
            "cao-conductor": {
                "fixes": ["e34c9b7e88b9", "54ceca22bd20", "59ef2d387223"],
                "files": [
                    ("conduct/lib/successor_policy_registry.py", "M"),
                    ("conduct/commands/_successor.py", "M"),
                    ("conduct/commands/spawn.py", "M"),
                    ("conduct/lib/ledger.py", "M"),
                    ("conduct/commands/continue_round.py", "M"),
                ],
                "symbols": ["claim_source", "create_locked", "_attach_generation_binding"],
            },
        },
    },
    "cond-0242": {
        "case_types": ["exact-technical", "data-flow"],
        "notes": "libtmux FIFO liveness parse storm starved the event loop; bounded observation across fifo_reader/tmux client/constants.",
        "repos": {
            "cao": {
                "fixes": ["ce4b986e7de0"],
                "pr": "colindmurray/cli-agent-orchestrator#57",
                "files": [
                    ("src/cli_agent_orchestrator/services/fifo_reader.py", "M"),
                    ("src/cli_agent_orchestrator/clients/tmux.py", "M"),
                    ("src/cli_agent_orchestrator/constants.py", "M"),
                    ("docs/configuration.md", "M"),
                ],
                "symbols": ["FifoManager", "_check_pipe_liveness", "get_history"],
            },
        },
    },
}


def sh(repo: str, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(REPOS[repo]), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout


def full_sha(repo: str, short: str) -> str:
    return sh(repo, "rev-parse", short).strip()


def subject(repo: str, sha: str) -> str:
    return sh(repo, "log", "-1", "--format=%s", sha).strip()


def name_status(repo: str, sha: str) -> dict[str, str]:
    parents = sh(repo, "log", "-1", "--format=%P", sha).split()
    rng = f"{parents[0]}..{sha}" if len(parents) > 1 else f"{sha}^..{sha}"
    out: dict[str, str] = {}
    for line in sh(repo, "diff", "--name-status", rng).splitlines():
        parts = line.split("\t")
        out[parts[-1]] = parts[0][0]
    return out


def per_fix_statuses(repo: str, fixes: list[str]) -> dict[str, list[str]]:
    """Path -> statuses across each fix commit individually, so a file added
    by one fix and edited by a later one records both."""
    seen: dict[str, list[str]] = {}
    for f in fixes:
        for path, status in name_status(repo, f).items():
            seen.setdefault(path, []).append(status)
    return seen


def path_exists_at(repo: str, sha: str, path: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(REPOS[repo]), "cat-file", "-e", f"{sha}:{path}"],
        capture_output=True,
    )
    return proc.returncode == 0


def grep_count_at(repo: str, sha: str, term: str) -> int:
    proc = subprocess.run(
        ["git", "-C", str(REPOS[repo]), "grep", "-c", "-F", term, sha],
        capture_output=True,
        text=True,
    )
    return sum(int(line.rsplit(":", 1)[1]) for line in proc.stdout.splitlines() if ":" in line)


def diff_files_mentioning(repo: str, fix_shas: list[str], term: str) -> list[str]:
    """Expected-file attribution: which fix-diff files' hunks mention *term*."""
    found: list[str] = []
    for sha in fix_shas:
        parents = sh(repo, "log", "-1", "--format=%P", sha).split()
        rng = f"{parents[0]}..{sha}" if len(parents) > 1 else f"{sha}^..{sha}"
        current = None
        for line in sh(repo, "diff", "-U0", rng).splitlines():
            if line.startswith("diff --git a/"):
                current = line[len("diff --git a/") :].split(" b/", 1)[0]
            elif line.startswith("+") and term in line and current and current not in found:
                found.append(current)
    return found


def fetch_issue(key: str, cache: dict | None) -> dict:
    if cache and key in cache:
        return cache[key]
    proc = subprocess.run(
        ["conduct", "issue", "show", "--id", key],
        capture_output=True,
        text=True,
        cwd=str(REPOS["cao-conductor"]),
        check=True,
    )
    d = json.loads(proc.stdout)["issue"]
    return {
        "title": d["title"],
        "body": d["body"] or "",
        "status": d["status"],
        "component": d["component"] or "",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="evals/code-grounding/fixtures/cases.json")
    parser.add_argument(
        "--verification-out", default="evals/code-grounding/reports/fixture-verification.json"
    )
    parser.add_argument("--bodies-cache", default=None)
    args = parser.parse_args(argv)

    cache = json.loads(Path(args.bodies_cache).read_text()) if args.bodies_cache else None

    cases: list[dict] = []
    verification: list[dict] = []
    for key, sel in SELECTIONS.items():
        issue = fetch_issue(key, cache)
        case_repos: dict[str, dict] = {}
        for repo_name, blob in sel["repos"].items():
            fixes = []
            for short in blob["fixes"]:
                fixes.append(
                    {
                        "sha": full_sha(repo_name, short),
                        "short": short,
                        "subject": subject(repo_name, short),
                    }
                )
            dated = sorted(
                fixes, key=lambda f: sh(repo_name, "log", "-1", "--format=%ct", f["sha"])
            )
            search_sha = sh(repo_name, "rev-parse", f"{dated[0]['sha']}^").strip()

            ns = per_fix_statuses(repo_name, [f["sha"] for f in fixes])

            files_out = []
            for path, expected_status in blob["files"]:
                actual_list = ns.get(path, [])
                actual = actual_list[0] if actual_list else None
                exists = path_exists_at(repo_name, search_sha, path)
                # A file sampled as Added may be edited again by a later fix
                # of the same case; either status confirms the sampling.
                status_ok = expected_status in actual_list or (not exists and "A" in actual_list)
                files_out.append(
                    {
                        "path": path,
                        "diff_status": "/".join(actual_list) or expected_status,
                        "in_search_tree": exists,
                        "verified_against_diff": status_ok,
                    }
                )
                verification.append(
                    {
                        "case": key,
                        "repo": repo_name,
                        "kind": "file",
                        "target": path,
                        "expected_status": expected_status,
                        "actual_statuses": actual_list,
                        "exists_at_search_sha": exists,
                        "ok": status_ok if actual_list else exists,
                    }
                )

            syms_out = []
            candidates = [(name, "preexisting-or-introduced") for name in blob.get("symbols", [])]
            candidates += [
                (name, "introduced") for name in blob.get("extra_symbols_introduced", [])
            ]
            for name, _hint in candidates:
                hits = grep_count_at(repo_name, search_sha, name)
                origin = "preexisting" if hits > 0 else "introduced"
                syms_out.append(
                    {
                        "name": name,
                        "origin": origin,
                        "pre_fix_grep_hits": hits,
                        "diff_files": diff_files_mentioning(
                            repo_name, [f["sha"] for f in fixes], name
                        ),
                    }
                )
                verification.append(
                    {
                        "case": key,
                        "repo": repo_name,
                        "kind": "symbol",
                        "target": name,
                        "origin": origin,
                        "pre_fix_grep_hits": hits,
                        "ok": True if origin == "preexisting" else None,
                        "note": (
                            None
                            if origin == "preexisting"
                            else "introduced by the fix; excluded from baseline recall denominators"
                        ),
                    }
                )

            case_repos[repo_name] = {
                "search_sha": search_sha,
                "fix_commits": fixes,
                "pull_request": blob.get("pr"),
                "expected_files": files_out,
                "expected_symbols": syms_out,
                "symbol_note": blob.get("symbol_note"),
            }

        cases.append(
            {
                "id": key,
                "issue": {
                    "key": key,
                    "tracker_project": "cao-system",
                    "status": issue["status"],
                    "component": issue["component"],
                    "title": issue["title"],
                    "narrative": f"{issue['title']}\n\n{issue['body']}",
                },
                "case_types": sel["case_types"],
                "notes": sel.get("notes"),
                "shared_fix_commit": sel.get("shared_fix_commit"),
                "test_only_fix": sel.get("test_only_fix", False),
                "repos": case_repos,
                "tool_lanes": {"codebase_memory": None, "qmd": None, "serena": None},
            }
        )

    fixture = {
        "schema_version": 1,
        "description": "Historical-bug code-grounding benchmark: 25 resolved cao-system bugs with fix-commit-bound expected targets. Search runs at each repo's pre-fix search_sha; recall denominators exclude targets absent from the search tree and symbols introduced by the fix.",
        "meta": {
            "generated_by": "evals/code-grounding/tools/build_fixture.py (hand-sampled diffs, mechanical verification)",
            "tracker_project": "cao-system",
            "repos": {
                "cao-conductor": {
                    "local_path": str(REPOS["cao-conductor"]),
                    "env_override": "CAO_EVAL_REPO_CAO_CONDUCTOR",
                },
                "cao": {"local_path": str(REPOS["cao"]), "env_override": "CAO_EVAL_REPO_CAO"},
            },
            "authoring_base": {
                "cli-agent-orchestrator": "539796799ae56f7694f2d201763212a1ccb10e67",
                "cao_conductor_head_at_authoring": sh("cao-conductor", "rev-parse", "HEAD").strip(),
            },
            "case_count": len(cases),
            "case_type_coverage": sorted({t for c in cases for t in c["case_types"]}),
        },
        "cases": cases,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    ver_path = Path(args.verification_out)
    ver_path.parent.mkdir(parents=True, exist_ok=True)
    failures = [v for v in verification if v["ok"] is False]
    ver_path.write_text(
        json.dumps(
            {
                "method": "Manual diff inspection at pinned SHAs (sampling) plus mechanical confirmation: file status via git diff --name-status on the fix range, existence via cat-file at search_sha, symbol pre-fix presence via git grep -c -F at search_sha.",
                "sampled_by": "lane D (opencode:laneD-cond0634)",
                "sampled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "fixture_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                "checks": verification,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"cases written: {len(cases)} -> {out_path}")
    print(f"verification checks: {len(verification)} ({len(failures)} failures) -> {ver_path}")
    for f in failures:
        print("FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
