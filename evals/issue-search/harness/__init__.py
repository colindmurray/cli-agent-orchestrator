# ABOUTME: Issue-search evaluation harness for the CAO tracker.
# ABOUTME: Measures exact/lexical retrieval against a fixed read-only snapshot.
"""Issue-search relevance fixture and metrics harness (cond-0633, M0.1).

Layout:

- ``snapshots/<snapshot-id>/export.json`` — fixed read-only export of the live
  tracker (see ``tools/export_snapshot.py``); the harness never touches the
  live database.
- ``fixtures/corpus.v1.json`` — versioned fixture cases with stable ids and
  per-case provenance.
- ``harness/`` — snapshot loader, retrievers (legacy substring baseline plus
  the FTS/semantic lane interfaces), metrics, runner, and pass/fail gate.
- ``baselines/`` — recorded baseline metrics a run is gated against.
- ``reports/`` — human-readable baseline reports and harvest candidates.

Everything in the harness is deterministic except the explicitly informational
wall-clock performance block (latency, cold start), which is excluded from the
gate comparison.
"""
