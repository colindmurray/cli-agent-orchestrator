# Code-grounding benchmark (M0.2, cond-0634)

A reproducible benchmark of 25 resolved `cao-system` bugs whose fix commits
identify known files and symbols across both repositories (`cao-conductor`,
`cli-agent-orchestrator`). It scores how well retrieval strategies ground an
issue narrative back onto the code that actually had to change, starting from
the direct `rg`/Git baseline that later pilot lanes must beat.

Design anchor: `docs/superpowers/specs/2026-08-21-hybrid-issue-search-and-code-grounded-triage-design.md`
§16.2 (in cao-conductor), including the required case mix: exact technical
strings, vague prose, stack traces, history-dependent questions,
cross-repository cases, and data-flow cases.

## Layout

| Path | Purpose |
| --- | --- |
| `fixtures/cases.json` | The dataset. Schema v1; per-case issue key/narrative/status, case-type tags, per-repo `search_sha` (pre-fix tree), fix commits, PR when known, expected files/symbols, and null `tool_lanes` columns reserved for later tool pilots. |
| `tools/build_fixture.py` | Builder/provenance script encoding the hand-sampled selections; re-verifies every target against the fix diffs and writes both artifacts below. |
| `baseline/retrieval.py` | Deterministic query derivation + rg ranking + visibility rules. |
| `baseline/run_baseline.py` | Baseline runner CLI (see below). |
| `reports/fixture-verification.json` | Per-target verification evidence: diff-status confirmation, existence at the search SHA, pre-fix symbol grep counts, origin classification. |
| `reports/baseline-run.json` | One full deterministic run of the baseline over the fixture revision. |
| `reports/dogfood-probe.md` | Two cases hand-solved with plain rg/Git; surprises recorded for later fusion lanes. |
| `tests/` | Focused pytest slices for the loader, retrieval primitives, and runner metrics (synthetic repo; offline). |

## Ground truth semantics

For each case-repo the runner searches the **pre-fix** state (`search_sha` =
parent of the earliest fix commit): the honest simulation of "issue just
filed". Consequently:

- Files **created by** the fix are recorded (`in_search_tree: false`) but
  excluded from recall denominators — they cannot be found before they exist.
- Symbols are classified by mechanical pre-fix `git grep`: `origin:
  "preexisting"` counts toward symbol recall; `"introduced"` does not.
- Every expected target was sampled by reading the actual fix diff, then
  mechanically verified (diff status via `git diff --name-status` per fix
  commit, existence via `cat-file`, symbols via repo-wide `git grep -F`).
  See `reports/fixture-verification.json` for the full record.

## Metrics

Per case-repo row and macro-aggregated:

- **file recall@5/10/20** — expected existing files present in the top-k
  ranked files.
- **symbol recall@5/10/20** — expected pre-existing symbols present (as
  substring) in the top-k retrieved files' content.
- **skipped-file rate** — expected targets invisible to default-filter rg
  (hidden paths, tracked-but-gitignored paths, binary), with reasons.
- **fallback escape rate** — of the missed targets, the fraction the
  exhaustive direct fallback would have surfaced (`git grep -I -F` over all
  tracked paths plus `git ls-files` path lookup), checked against the case's
  expected files.

Ranking is deterministic: weighted matches (exact spans 3×, identifiers 2×,
title prose 1×, log-scaled term frequency, path-name bonus) with
lexicographic tie-breaks. Query derivation is a pure function of the issue
narrative, so the same fixture revision always yields identical output.

## Running

```sh
# regenerate fixture + verification evidence (read-only vs tracker and repos)
uv run python evals/code-grounding/tools/build_fixture.py

# full baseline run (~20 s)
uv run python evals/code-grounding/baseline/run_baseline.py \
    --fixture evals/code-grounding/fixtures/cases.json \
    --out evals/code-grounding/reports/baseline-run.json

# focused tests
uv run pytest evals/code-grounding/tests -q
```

Repository locations resolve from `meta.repos[*].local_path`, overridable via
`CAO_EVAL_REPO_CAO_CONDUCTOR` / `CAO_EVAL_REPO_CAO` or `--repo-*` flags. Runs
are offline: the runner clones from the local checkouts only
(`git clone --shared` into a temp dir, detached checkout per pinned SHA).
Re-runs are byte-deterministic apart from nothing — the report contains no
wall-clock fields.

## Current baseline (fixture revision as committed)

29 case-repo rows over 25 cases:

| measure | @5 | @10 | @20 |
| --- | --- | --- | --- |
| file recall | 0.314 | 0.470 | 0.567 |
| symbol recall | 0.500 | 0.765 | 0.848 |

Skipped-target rate 0.014; fallback escape rate 0.974. Reading: plain rg's
default filters hide almost nothing permanently because the exhaustive Git
fallback nearly always rescues misses — but primary ranking itself leaves
half the expected files unfound even at k=20. Consumer modules outrank the
modules fixes touch (dogfood probe, cond-0012), and hidden-path targets score
zero without a declared visibility policy (cond-0386). Those two gaps are the
standing candidates for the fused tool lanes.
