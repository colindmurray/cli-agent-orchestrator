"""Direct rg/Git baseline runner for the code-grounding fixture.

For every case the runner materializes the pre-fix repository state at the
pinned search SHA in a throwaway shared clone, derives the deterministic
query set from the issue narrative, ranks files with plain ripgrep, and
reports file/symbol recall@5/10/20, skipped-file rate, and fallback escape
rate. Re-runs from the same fixture revision are byte-deterministic.

Usage:
    uv run python evals/code-grounding/baseline/run_baseline.py \
        --fixture evals/code-grounding/fixtures/cases.json \
        --out evals/code-grounding/reports/baseline-run.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval import (  # noqa: E402
    Queries,
    derive_queries,
    git_grep_fallback,
    rank_files,
    skip_reason,
    top_k,
)

RECALL_KS = (5, 10, 20)


def rg_version() -> str:
    proc = subprocess.run(["rg", "--version"], capture_output=True, text=True)
    return proc.stdout.splitlines()[0] if proc.returncode == 0 else "rg-unavailable"


def load_fixture(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise SystemExit(f"unsupported fixture schema_version: {data.get('schema_version')}")
    return data


def resolve_repo_paths(fixture: dict, args) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, meta in fixture["meta"]["repos"].items():
        override = getattr(args, f"repo_{name.replace('-', '_')}", None)
        env_name = f"CAO_EVAL_REPO_{name.upper().replace('-', '_')}"
        chosen = override or os.environ.get(env_name) or meta["local_path"]
        resolved[name] = Path(chosen).expanduser().resolve()
    return resolved


def prepare_clone(source: Path, work_root: Path) -> Path:
    dst = work_root / f"clone-{source.name}"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", "--quiet", str(source), str(dst)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(dst), "config", "core.fsmonitor", "false"],
        check=True,
        capture_output=True,
    )
    return dst


def checkout(clone: Path, sha: str) -> None:
    subprocess.run(
        ["git", "-C", str(clone), "checkout", "--detach", "--quiet", sha],
        check=True,
        capture_output=True,
    )


def evaluate_case_repo(case: dict, repo: str, blob: dict, clone: Path) -> dict:
    checkout(clone, blob["search_sha"])
    narrative = case["issue"]["narrative"]
    queries = derive_queries(case["issue"]["title"], narrative)

    scores = rank_files(queries, clone)
    ranked = top_k(scores, max(RECALL_KS))
    ranked_set = set(ranked)

    expected_files = [f for f in blob["expected_files"] if f.get("in_search_tree", True)]
    expected_symbols = [s for s in blob["expected_symbols"] if s.get("origin") == "preexisting"]

    file_hits = {
        k: len([f for f in expected_files if f["path"] in set(ranked[:k])]) for k in RECALL_KS
    }

    contents: dict[str, str] = {}
    for rel in ranked:
        try:
            contents[rel] = (clone / rel).read_text(errors="replace")
        except OSError:
            contents[rel] = ""
    symbol_hits = {
        k: len(
            [
                s
                for s in expected_symbols
                if any(s["name"] in contents.get(rel, "") for rel in ranked[:k])
            ]
        )
        for k in RECALL_KS
    }

    skipped = []
    for f in expected_files:
        reason = skip_reason(clone, f["path"])
        if reason:
            skipped.append({"path": f["path"], "reason": reason})

    missed_files = [f["path"] for f in expected_files if f["path"] not in ranked_set]
    missed_symbols = [
        s for s in expected_symbols if not any(s["name"] in contents.get(rel, "") for rel in ranked)
    ]

    escapes = []
    expected_paths = {f["path"] for f in expected_files}
    for path in missed_files:
        rescued = _fallback_finds(path.split("/")[-1], expected_paths, clone)
        escapes.append({"target": path, "kind": "file", "fallback_rescued": rescued})
    for sym in missed_symbols:
        rescued = _fallback_finds(sym["name"], expected_paths, clone)
        escapes.append({"target": sym["name"], "kind": "symbol", "fallback_rescued": rescued})

    return {
        "case_id": case["id"],
        "case_types": list(case["case_types"]),
        "issue_title": case["issue"]["title"],
        "repo": repo,
        "search_sha": blob["search_sha"],
        "queries": queries.as_dict(),
        "expected_file_count": len(expected_files),
        "expected_symbol_count": len(expected_symbols),
        "file_recall": {
            str(k): file_hits[k] / len(expected_files) if expected_files else None
            for k in RECALL_KS
        },
        "symbol_recall": {
            str(k): symbol_hits[k] / len(expected_symbols) if expected_symbols else None
            for k in RECALL_KS
        },
        "top_files": [{"path": rel, "score": round(scores[rel], 6)} for rel in ranked],
        "skipped_files": skipped,
        "missed_files": missed_files,
        "missed_symbols": [s["name"] for s in missed_symbols],
        "fallback_escapes": escapes,
        "fallback_escape_rate": (
            (lambda esc: sum(e["fallback_rescued"] for e in esc) / len(esc) if esc else None)(
                escapes
            )
        ),
    }


def _fallback_finds(term: str, expected_paths: set[str], clone: Path) -> bool:
    """True when the exhaustive fallback surfaces evidence inside the case's
    expected files: a missed target counts as escaped only if the fallback
    would have led the searcher to fix-relevant code. File misses try both
    content search for the basename and path-aware `git ls-files` lookup."""
    if len(term) < 4:
        return False
    hits = set(git_grep_fallback(term, clone))
    proc = subprocess.run(
        ["git", "ls-files", f"*{term}*"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(clone),
    )
    hits.update(proc.stdout.splitlines())
    return bool(expected_paths.intersection(hits))


def run(args: argparse.Namespace) -> dict:
    fixture_path = Path(args.fixture).resolve()
    fixture = load_fixture(fixture_path)
    repo_paths = resolve_repo_paths(fixture, args)

    work_root = (
        Path(args.keep_trees)
        if args.keep_trees
        else Path(tempfile.mkdtemp(prefix="cao-code-grounding-"))
    )
    work_root.mkdir(parents=True, exist_ok=True)
    clones: dict[str, Path] = {}
    try:
        for name, path in repo_paths.items():
            clones[name] = prepare_clone(path, work_root)

        rows = []
        for case in fixture["cases"]:
            if args.cases and case["id"] not in args.cases:
                continue
            for repo, blob in case["repos"].items():
                rows.append(evaluate_case_repo(case, repo, blob, clones[repo]))
    finally:
        if not args.keep_trees:
            shutil.rmtree(work_root, ignore_errors=True)

    def macro(key: str, k: str) -> float | None:
        vals = [r[key][k] for r in rows if r[key][k] is not None]
        return sum(vals) / len(vals) if vals else None

    def micro_escape(rows: list[dict]) -> float | None:
        esc = [e for r in rows for e in r["fallback_escapes"]]
        return sum(e["fallback_rescued"] for e in esc) / len(esc) if esc else None

    report = {
        "report_version": 1,
        "fixture": str(fixture_path),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "rg_version": rg_version(),
        "repo_sources": {name: str(p) for name, p in repo_paths.items()},
        "aggregate": {
            "case_repo_count": len(rows),
            "file_recall_macro": {k: macro("file_recall", k) for k in map(str, RECALL_KS)},
            "symbol_recall_macro": {k: macro("symbol_recall", k) for k in map(str, RECALL_KS)},
            "skipped_target_rate": (
                sum(len(r["skipped_files"]) for r in rows)
                / max(sum(r["expected_file_count"] for r in rows), 1)
            ),
            "fallback_escape_rate_micro": micro_escape(rows),
        },
        "cases": rows,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo-cao-conductor", dest="repo_cao_conductor", default=None)
    parser.add_argument("--repo-cao", dest="repo_cao", default=None)
    parser.add_argument("--cases", nargs="*", default=None, help="case ids to run (default all)")
    parser.add_argument("--keep-trees", default=None, help="keep the prepared clone under this dir")
    args = parser.parse_args(argv)

    report = run(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    agg = report["aggregate"]
    print(f"cases: {agg['case_repo_count']}")
    print(
        f"file recall@5/10/20: {agg['file_recall_macro']['5']:.3f} / {agg['file_recall_macro']['10']:.3f} / {agg['file_recall_macro']['20']:.3f}"
    )
    print(
        f"symbol recall@5/10/20: {agg['symbol_recall_macro']['5']:.3f} / {agg['symbol_recall_macro']['10']:.3f} / {agg['symbol_recall_macro']['20']:.3f}"
    )
    print(f"skipped target rate: {agg['skipped_target_rate']:.3f}")
    fer = agg["fallback_escape_rate_micro"]
    print(f"fallback escape rate: {fer if fer is None else f'{fer:.3f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
