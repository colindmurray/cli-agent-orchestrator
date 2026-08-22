# conductor-catalog-golden

Golden catalog fixture for this fork's installed-pair API tests
(`test/api/test_communications_catalog_installed_pair.py`, cond-0610
PR 9C). Produced by cao-conductor's real capture+publish path —
`conduct/lib/catalog_fixture.py` through the REAL store capture
(`communication_store.record_communication`) and projection publish
(`communication_projection.publish_project_catalog`) for one project,
slug `golden-project` — never hand-written. Consumed here as the
installed-pair contract between that producer and this fork's bounded
catalog reader.

Generating head: cao-conductor `5895e6c` (lane worktree
`~/Projects/cao-conductor-worktrees/lane-w2-0610`), regenerated with
the command documented in that repository's own copy of this README:

    PYTHONPATH=. python3 -m conduct.lib.catalog_fixture tests/fixtures/conductor-catalog-golden

## Vendored bytes

The directory is vendored byte-identical from the generating
repository; `test_vendored_index_is_unmodified` pins
`communications.json` against a checksum constant so any
re-vendoring is a visible, deliberate change.

- `communications.json` — the published `cao-communications-index-v1`
  index: four communications bound to task occurrence
  `occ-golden-0001` (an inline assignment whose body carries a script
  tag, a `javascript:` link, and a remote image reference; an
  intermediate checkpoint; a file-snapshot final report; a
  message-only final report) plus their content object metadata.
- `communications/content/<sha256>` — the immutable content objects
  the index addresses; each filename is the blob digest.

Everything else the generator materialises in a full project state
directory (the semantic SQLite store, the `runs/` directory holding
the file-backed report source) is scaffolding and is deliberately not
vendored.

## Determinism

Fixed across regenerations: communication ids, request keys, titles,
authors, `authored_at`, kinds/scopes/capture kinds, every content byte
(hence every sha256, blob_id and byte_size), and the insertion order.

Wall-clock on every regeneration: the envelope's `produced_at` /
`valid_until`; each communication's `recorded_at` (store-stamped by
design — the catalog recency axis is never caller-supplied); each
body's `attachment_id` / `document_id` (uuid4 minted at capture); and
the file-backed body's `provenance.resolved_path`. The installed-pair
tests therefore assert on kinds, scopes, digests, and bytes, and
derive attachment identifiers from the vendored index rather than
pinning them.
