# Issue-search relevance fixture and metrics harness

Measures exact/lexical (and, once installed, semantic/hybrid) issue retrieval
against a fixed read-only snapshot of the live tracker. Scope: cond-0633
(M0.1) of the hybrid-issue-search initiative; design §16.1 defines the fixture
requirements.

## Layout

| Path | Purpose |
| --- | --- |
| `snapshots/<id>/export.json` | Fixed read-only export of the live tracker DB. The id is the SHA-256 prefix of the export bytes: same content, same id; a mutated export is self-evident. |
| `snapshots/<id>/provenance.json` | Source DB path, export time, redaction count, row counts. The live DB is never written (`mode=ro` only). |
| `fixtures/corpus.v1.json` | Versioned fixture cases with stable ids, per-case provenance, verified duplicate-pair labels, hard negatives, and scope forms. |
| `harness/snapshot.py` | Snapshot loader + validators (fail closed on malformed snapshots). |
| `harness/retrievers.py` | Legacy substring baseline (faithful `%ILIKE%` mirror incl. LIKE wildcard quirks) and the lane-E FTS / M2 semantic interfaces. |
| `harness/metrics.py` | recall@5/10, MRR (all-expected and canonical-only), hard-negative load, status mix, null-safe semantic coverage. |
| `harness/runner.py` | Deterministic runner; `--inject promote-noise\|empty` for counterfactual sensitivity. |
| `harness/gate.py` | Compares a run's rank-derived metrics against the recorded baseline (tolerance 0.02). Wall-clock never gates. |
| `baselines/legacy-substring.json` | Recorded legacy baseline a run must not regress against. |
| `reports/baseline-legacy-substring.md` | Human-readable baseline report + harvest candidates for lane E. |
| `tools/export_snapshot.py` | Regenerates a snapshot from the live DB (read-only; redacts personal emails). |

## Commands

```bash
# from this directory
uv run python -m harness.runner                     # fresh report to stdout
uv run python -m harness.runner --inject empty      # drop-lane regression demo

# focused tests (from repo root)
uv run pytest evals/issue-search/tests/ -q
```

## Determinism contract

Rank-derived metrics are a pure function of (snapshot bytes, fixture bytes,
lane code): two runs produce byte-identical metrics blocks, enforced by
`tests/test_determinism.py`. Wall-clock latency lives in a separate
`performance` block and is informational only — it can never flip the gate.

## Adding a case or refreshing the snapshot

- New cases go into a **new fixture version** (`corpus.v2.json`, bumping
  `fixture_version`) with provenance pointing at the pinned snapshot; every
  expected key must exist in that snapshot and duplicate-pair labels must be
  confirmed `duplicate_of` relations in it (the runner validates this and
  fails closed).
- A new snapshot means re-measuring and re-recording the baselines plus a
  reviewer-visible note that the corpus moved; never overwrite an existing
  snapshot directory.
