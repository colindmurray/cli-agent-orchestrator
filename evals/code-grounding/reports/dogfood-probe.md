# Dogfood probe — hand-solving two fixture cases with plain rg/Git

Probe rule from the lane brief: pick two cases, solve them by hand with only
`rg` and Git, and record where the baseline runner's ranking surprised me.
Both hand-solves were performed at the pinned pre-fix search SHAs before
consulting the runner output; the runner rows were compared afterwards.

## Case 1 — cond-0012 (vague prose, cao-conductor @ d2c9075533)

Narrative: "Sentinel ignores spec-writer idle timeout and raises false
deadman" — no identifiers in the title.

Hand-solve commands:

```
rg -l deadman                       # in /Users/colin/Projects/cao-conductor
git grep -l deadman_minutes
```

Hand result: `plugin/conductor_sentinel/sentinel.py`, several `profiles/*.md`,
and operating docs dominate the hit list; `conduct/lib/sentinelcfg.py` appears
further down via `deadman_minutes`. Following relevance order, a searcher
studies the sentinel's *matching* logic first, while the fix landed in the
*projection* (`sentinelcfg.build` projecting role keys onto concrete profile
names such as `spec-writer-terra`).

Runner outcome: file recall 1.0@5 with `sentinelcfg.py` ranked 4th;
symbol `task_classes` (called by the fix, defined in `conduct/lib/routing.py`)
missed at every k and not fallback-rescued.

Surprises harvested for later fusion lanes:

1. Consumer/outweighs-producer: the module that *consumes* the config ranks
   above the module the fix had to change. Lexical overlap alone cannot
   separate "where the symptom lives" from "where the correction goes".
2. Cross-module call targets (`routing.task_classes`) are invisible unless a
   strategy follows call edges; neither plain rg ranking nor the exhaustive
   fallback surfaces them here.

## Case 2 — cond-0386 (exact technical string, cao @ 68f3efa511)

Narrative: "pre-push hook leaks Git environment and commits fixtures onto
source branch".

Hand-solve commands:

```
rg -l "pre-push"                    # default filters
rg -l --hidden "pre-push"
git grep -l "pre-push"
```

Hand result: default `rg` never sees the fix target
`cao_mcp_apps/.husky/pre-push` at all — it hides inside a dot-directory.
`--hidden` and `git grep` both find it immediately.

Runner outcome: file recall 0.0 at every k, `skipped_files` records the target
with reason `hidden`, and the exhaustive fallback escapes it (`true`) — the
runner reproduces the hand-solved failure mode exactly, which is precisely
what the benchmark should demonstrate for later tool lanes.

Surprise harvested:

3. Fix targets can live entirely outside rg's default visibility (shell hooks
   under `.husky/`). A retrieval strategy that does not state its ignore/hidden
   policy will silently score 0 on such cases while looking healthy elsewhere;
   the fixture's skipped-file and fallback-escape columns make that visible.

## Where this leaves the baseline

The two probes behaved as the metrics predicted after the fact, and each
yielded one fusion candidate: symptom-vs-correction disambiguation (case 1)
and declared visibility policy over hidden paths (case 2).
