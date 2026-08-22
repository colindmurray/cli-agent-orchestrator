# Legacy `%substring%` baseline — fixture v1, snapshot snap-40511acc9849

Measured 2026-08-21 by the lane-B harness (`evals/issue-search/`). Corpus:
685 issues, 955 comments, 524 links across 3 tracker projects, exported
read-only from the live tracker database. Fixture: 25 cases, 17 expected
issue targets, 34 hard-negative constraints.

## Headline metrics (legacy-substring lane)

| Metric | Value |
| --- | ---: |
| recall@5 | 0.2588 |
| recall@10 | 0.3059 |
| MRR (first primary-or-acceptable hit) | 0.5000 |
| MRR (primary/canonical target only) | 0.2933 |
| hard-negative load @5 / @10 | 1 / 1 |
| cases with any hard negative in top-k | 0.04 |
| cases where a hard negative outranks the first hit | 0.04 |
| semantic coverage | null (lane not installed pre-M2) |

Wall-clock performance (informational, never gated): cold start ~5 ms,
median ~5 ms/query on the full corpus.

## Counterfactual sensitivity

`harness.runner --inject promote-noise|empty` simulates ranking regressions;
the committed test suite proves both drive every gated metric RED against
this baseline (`tests/test_counterfactual.py`), and determinism tests prove
reruns from the fixed snapshot reproduce these numbers byte-for-byte.

## Harvest candidates — what embarrassingly fails today

These are lane E's (FTS5 BM25) first targets, in impact order:

1. **Comments are invisible (3/3 comment-only cases fail).** The live search
   covers eight issue fields and zero comments; 955 comments of evidence,
   including decision-bearing dispositions ("Patch or pin compatible pyte
   behavior…") and captured CLI errors ("argument command: invalid choice:
   review"), are unreachable by search. Comment FTS is the single highest-value
   lane (design §10.3 lane 2).
2. **Multi-token natural-language queries return nothing (9/9 fail).** Any
   paraphrase or multi-word query becomes one literal `%needle%`; unless the
   exact token sequence occurs contiguously, recall is zero. All six NL
   paraphrase cases and all scope/onboarding queries fail outright. Field-
   weighted BM25 over tokenized documents fixes this class wholesale.
3. **Duplicate chains dead-end at the duplicate (3/3 duplicate-pair cases miss
   the canonical).** Querying with a duplicate's own wording retrieves the
   duplicate — never the canonical issue it was merged into. Lexical matching
   alone cannot fix this; it needs the confirmed-duplicate-chain expansion of
   design §10.4 as an explanation/navigation signal.
4. **Stored line breaks defeat literal substring matching.** In cond-0087 the
   error fragment `guarded bounce returned no verified activation receipt` is
   wrapped mid-phrase in the stored body, so even the exact-error query cannot
   match the canonical report (it matches only unwrapped siblings).
   Tokenization removes this failure class.
5. **SQL LIKE wildcard leakage.** `_codex_mcp_pin_preflight` treats each
   underscore as a single-char wildcard, so symbol queries match loosely
   rather than pinning their owning issue. An exact/quoted lane (§10.3
   lane 3) should treat identifiers as fingerprints.
6. **Path precision has no protection.** For `conduct/commands/deploy.py`,
   unrelated cond-0469 (which merely cites the path) outranks the deploy
   issues; recency ordering is the only tie-break today.

## What already works

Single-token exact queries against distinctive symbols (`TerminalInputBlockedError`),
revision fingerprints (`d7d871ff…`, `3fcde77a…`), and short contiguous phrases
(`activation receipt`) retrieve their targets at rank 1, and the status-mix
case confirms open and terminal issues are both reachable (top-10 contained
1 open + 6 terminal relevant candidates).

## Reproducing

```
uv run pytest evals/issue-search/tests/ -q          # full self-test
uv run python -m harness.runner                      # fresh report (from evals/issue-search/)
```
