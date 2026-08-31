# Tracker search: hybrid issue search and index maintenance

`cao issue search` ranks issues and their comments across two lanes. The
lexical lane is an FTS5 projection of the authoritative issue and comment
rows and is always available. The semantic lane embeds the same documents
into versioned vector generations and is opt-in: it exists only after the
operator installs the search runtime, prepares the pinned model, and builds a
generation.

Every maintenance verb works on the derived index only — the FTS documents,
the vector outbox, vector rows, generation rows, and the search metadata
singleton. No verb rewrites an issue, comment, link, or event. `cao issue
search-index status` and `integrity-check` are read-only.

## Install and prepare

```console
$ uv pip install 'cli-agent-orchestrator[search]'   # or pip equivalent
$ cao issue search-index model prepare
```

`model prepare` is the only command that touches the network for model
weights. It downloads the pinned `sentence-transformers/all-MiniLM-L6-v2`
snapshot, verifies its digest against the digest recorded in this build, and
writes generation metadata under the state root (`CAO_SEARCH_MODELS_DIR`
overrides the location). It is safe to re-run: an already-verified install is
returned unchanged, a corrupt metadata file is rewritten from the verified
artifact, and the matching generation is reused instead of minted again. The
generation identity includes model id and revision, runtime id and version,
artifact digest, dimensions, encoding, metric, normalization, and document
schema. When active and building generations have different identities,
refresh selects only the generation matching the prepared embedder, so a
same-width replacement cannot publish blobs under the wrong generation.

Without the `[search]` extra every surface degrades with a typed answer and
the install command rather than a traceback: search falls back to lexical and
reports the degradation, and the maintenance verbs name the state they
observed. Search and issue writes never download a model; preparation is
always an explicit operator command.

## Build and activate the index

```console
$ cao issue search-index refresh --all
refresh (all): 3 published, 0 failed, 0 stale, 0 source gone, 0 damaged skipped
activated generation 20260831T051646488428-a16747
active generation: 20260831T051646488428-a16747
```

`refresh` embeds the documents queued in the durable outbox for the prepared
model's generation. Without `--all` it drains one bounded batch — the same
derived work a semantic query performs — and never activates anything. With
`--all` it drains that generation completely and offers every finished
building generation for activation. During a model replacement, the previous
active generation remains untouched and continues serving until the new
generation passes activation.

Activation is where an incomplete build is stopped. The proof inside the
activation transaction requires that no queued work remains, that every live
issue and comment has a current-version vector, and that every vector carries
the declared float32 width. A refused activation leaves the generation
`building` and reports its reason, so the ordinary next step is to run the
refresh again:

```console
$ cao issue search-index refresh --all --retry-failed
```

`--retry-failed` resets the backoff of documents whose embedding failed, up
to `--limit N` rows per pass. Vectors are only ever served from the one
`active` generation, so a stuck build costs nothing at query time.

## Repair

```console
$ cao issue search-index rebuild --lexical    # repopulate the FTS documents
$ cao issue search-index rebuild --vectors    # fresh generation, built and activated
$ cao issue search-index rebuild --all        # both, in that order
```

Exactly one scope flag is required. Lexical repair rebuilds the documents
from the authoritative rows and requeues every live document, so no vector
can be served against rebuilt text. Vector repair builds a fresh generation
and activates it only when the coverage proof passes; a build whose documents
cannot embed stays `building` and resumable through `refresh --all
--retry-failed`.

## Observe

```console
$ cao issue search-index status            # capability, engine, lexical, semantic, next action
$ cao issue search-index integrity-check   # read-only report; repairs belong to rebuild
$ cao issue search-index model status      # embedding capability with positive signals
```

`status` never loads model weights and names the operator action for every
degraded state it can observe. `integrity-check` reports FTS internal
integrity, source-to-FTS coverage, orphan and duplicate document keys,
dirty/failed/stale vector counts, and generation provenance, and repairs
nothing.

## REST surface

The API exposes the same orchestrator over four routes, scoped like the rest
of the tracker API (reads accept `cao:read`; writes demand `cao:write`):

| Route | Scope | Purpose |
|---|---|---|
| `GET /tracker/issues/search-index/status` | read | the status report |
| `GET /tracker/issues/search-index/integrity-check` | read | the integrity report |
| `POST /tracker/issues/search-index/refresh` | write | `{"all": true, "retry_failed": true, "limit": n}`, all optional |
| `POST /tracker/issues/search-index/rebuild` | write | `{"scope": "lexical" \| "vectors" \| "all"}` |

A refusal keeps the reason the orchestrator observed and its operator action
in the detail: an unknown scope is 400, an installation refusal is 409. There
is no model-prepare route; preparing downloads weights and stays a CLI
decision. Ranked search and index-maintenance routes use FastAPI's synchronous
worker execution for blocking SQLite, model-load, and CPU embedding work, so
the Uvicorn event loop remains responsive while a build runs.
