# Issue tracker — manual test matrix

Executed against an **isolated** CAO instance: its own `CAO_STATE_ROOT`, its own
`TMUX_TMPDIR`, and a port nothing else uses. All three are required. A state
root alone is not enough — session discovery is tmux-based, so an instance
sharing the operator's tmux socket will list and act on the live fleet.

```bash
export CAO_STATE_ROOT=/tmp/cao-tracker-verify
export TMUX_TMPDIR=/tmp/cao-tv           # short: the socket path has a length limit
cao-server --port 9971
```

Each row states what is being checked and why it can fail silently. A row whose
failure mode is "you would not notice" is the reason it is in a *manual* matrix
rather than only in the unit suite: several of these are shapes the automated
tests cannot see, because the automated tests supply their own fixtures.

Legend: **E** = exercised end-to-end against the isolated instance (see
`scripts/verify-issue-tracker.sh`), **U** = covered by unit/API tests as well.

---

## A. Project identity

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| A1 | Create a project spanning two repo paths and one session | 201, all three scopes present | The whole premise: one log across many roots | E U |
| A2 | Resolve from a subdirectory of a scoped path | matched_by `path` | Issues are filed from anywhere in a tree | E U |
| A3 | Resolve from a *sibling* directory sharing a name prefix (`cao-conductor-worktrees` vs `cao-conductor`) | no match | A string-prefix match files every worktree issue into the wrong project, and nothing surfaces the error | E U |
| A4 | Resolve by session name from an unrelated directory | matched_by `session` | Supervisors and workers share a session across many directories | E U |
| A5 | Resolve by campaign alias | matched_by `alias` | `conduct issue file --project <campaign>` must land somewhere | E U |
| A6 | Explicit project beats a conflicting session | matched_by `explicit` | A caller that named a project already answered the question | E U |
| A7 | Register a path already owned by another project | 409 naming the owning project | Two owners means filings go to an arbitrary one, silently | E U |
| A8 | Re-register the same scope on the same project | 200, `created: false` | Idempotent setup scripts must not fail on the second run | E U |
| A9 | Register `~/x` and `~/x/` | one scope | Trailing separators are the same directory | E U |
| A10 | Register an ssh remote and its https twin | one scope | `git@github.com:o/r.git` and `https://github.com/o/r` are one repo | E U |
| A11 | Register a remote carrying credentials | stored value has no token | Scope rows are readable by every dashboard client | E U |
| A12 | Register a relative path | 400 | `realpath` would silently anchor it to the server's cwd | E U |
| A13 | Resolve an unregistered directory | `project_id: null`, no error | "None registered" is a legitimate answer | E U |
| A14 | Two path scopes, one nested inside the other | the deeper one wins | The specific answer is the one the operator meant | E U |

## B. Issue keys

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| B1 | File three issues | `cond-0001`…`0003` | | E U |
| B2 | Delete the newest, file again | next key, not the deleted one | A recycled key repoints every commit message and evidence path that quotes it | E U |
| B3 | Import an explicit key above the counter | counter advances past it | Otherwise the next filing collides | E U |
| B4 | Import an explicit key below the counter | counter does not rewind | | E U |
| B5 | File the same explicit key twice | 409 | | E U |
| B6 | Two projects file concurrently | independent sequences | | E U |
| B7 | Change a project's prefix | old keys unchanged, new keys use it | Existing keys are quoted where this database cannot reach | E U |
| B8 | Create a second project with a prefix already in use | 409 naming the owner | Keys are unique installation-wide, so a shared prefix collides later with a conflict naming a project the caller never mentioned | U |
| B9 | Rename a project onto a used prefix | 409 | | U |
| B10 | Re-set a project's own prefix | allowed | The check must exclude the project being edited, or every rename refuses itself | U |
| B11 | Reclaim a deleted project's prefix | allowed | | U |

## C. Filing and editing

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| C1 | File from a scoped cwd with no explicit project | resolves and reports `resolved_by` | The caller needs to know *why* it landed there | E U |
| C2 | File from an unregistered cwd | 422, nothing written | A filing that lands in the wrong log is worse than one that fails | E U |
| C3 | PATCH one field | only that field changes | | E U |
| C4 | PATCH with a field equal to its current value | no audit event | An audit trail full of no-ops is one nobody reads | E U |
| C5 | PATCH an empty string into a free-text field | field cleared | This is how the dashboard unassigns | E U |
| C6 | PATCH `key` | 400 "not editable", not a TypeError | | U |
| C7 | PATCH `project_id` | 422 naming the field | Pydantic drops unknown fields by default, so this returned 200 for an operation that never happened | E U |
| C7b | PATCH a misspelled field (`assigne`) | 422 | Same failure, and the one a client hits by accident | E U |
| C8 | Set status to `closed` | `closed_at` stamped | | E U |
| C9 | Reopen | `closed_at` cleared | | E U |
| C10 | Set status to `resolved` | still counted as open | "a fix landed" ≠ "somebody verified it" | E U |
| C11 | `duplicate_of` itself | 400 | | U |
| C12 | `duplicate_of` an unknown key | 404 | | U |
| C13 | Invalid status / severity | 400 listing the accepted values | | E U |
| C14 | Title over 300 chars | 400 | | U |

## D. Search, filters, listing

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| D1 | `q=` matches title, body, key and failing command | all four | Searching by the command you ran is the natural reflex | E U |
| D2 | Repeated `status=` params | OR, not an error | A comma list would be one unknown status | E U |
| D3 | `label=ui` with issues labelled `ui` and `ui-polish` | only `ui` | Substring matching would silently over-match | E U |
| D4 | `open_only` | excludes closed/wontfix/duplicate, keeps resolved | | E U |
| D5 | `total` with `limit` smaller than the result set | unpaged total | A short page must not be mistaken for the end | E U |
| D6 | `limit=100000` | 422 at the route | Silently returning 500 of 100000 requested rows reads as "that was everything" | E U |
| D6b | `limit=500` | the applied limit is echoed back | A direct service caller is clamped, so the response has to say what it used | E U |
| D7 | `order=severity` | P0 first | | E U |
| D8 | Filters compose (severity AND component) | intersection | | E U |
| D9 | Listing scoped to one project | no cross-project leakage | | E U |

## E. Comments, links, audit

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| E1 | Add a comment | stored, and an audit event | | E U |
| E2 | Empty comment | 400 | | E U |
| E3 | Link two issues | visible from both | | E U |
| E4 | Duplicate link | idempotent | | E U |
| E5 | Link to a missing issue | 404 | | E U |
| E6 | Link an issue to itself | 400 | | U |
| E7 | Delete an issue | comments, events and links go with it | An orphan link renders as a dangling reference on the other issue | E U |
| E8 | Audit trail names an actor | actor recorded per change | A mutable row is only acceptable if every mutation is recorded | E U |

## F. Project lifecycle

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| F1 | Delete a project holding issues | 409 | Deleting an issue log to tidy a list is not recoverable | E U |
| F2 | Delete with `force` | project and issues gone | | E U |
| F3 | Archive | hidden from the default list, issues intact and searchable | Archiving is the non-destructive answer | E U |
| F4 | Rename | id and keys unchanged | | E U |

## G. Ledger migration

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| G1 | Import `CLOSED_ISSUES.md` + `OPEN_ISSUES.md` | 208 of 208 | | E U |
| G2 | Heading count matches imported count per file | equal | 207 of 208 looks like success | E |
| G3 | `## cond-0200 [P2] — …` (severity before the dash) | severity read, entry imported | 1 entry of 208 | E U |
| G4 | `P1: title` and `P1 title` (unbracketed) | severity read, title cleaned | 42 entries | E U |
| G5 | `[P0]` | stays P0 | Flattening it to P1 erases the author's one distinction | E U |
| G6 | `filed: 2026-07-21 (external adjudicator)` | date parsed, note preserved | Those entries have no `reporter:` — the note is the only record of who filed them | E U |
| G7 | An entry with no `filed:` line | body says the date is the migration's | Otherwise a year-old defect claims it was filed today | E U |
| G8 | Unrecognised fields (`affected heads:`, `preserved run:`) | preserved verbatim in the body | ~20 one-off names, each written because it mattered once | E U |
| G9 | A `- **bold:**` line inside a prose body | stays in the body | Hoisting it deletes the sentence it belonged to | E U |
| G10 | `status: deferred` | open, labelled `deferred` | Mapping it to wontfix retires 3 live defects by transcription | E U |
| G11 | Free-text closed status | preserved as `resolution` | It cannot be parsed into a state machine, so it is not | E U |
| G12 | `evidence: (none given)` | null, not the literal string | | E U |
| G13 | Re-run the import | skipped, not duplicated | A partial import must be repeatable | E U |
| G14 | `--dry-run` | parses, writes nothing | | E U |
| G15 | Every imported filing date within the ledger's real range | no entry stamped with today's date except the one that has none | | E |

## H. `conduct issue` routing

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| H1 | Tracker reachable | `store: tracker`, nothing appended to the ledger | | U |
| H2 | Tracker unreachable | `store: ledger`, `tracker_unavailable` set, entry in the ledger | An issue is filed exactly when the server is least safe to assume | U |
| H3 | Tracker refuses (422, unresolved project) | falls back rather than dropping the filing | | U |
| H4 | `--ledger-only` | tracker never called | | U |
| H5 | Pending queue fed from either store | queued | The sentinel re-alert must not depend on which store won | U |
| H6 | `open_issues` in the result | present only on the ledger path | Its presence *is* the signal the tracker declined | U |
| H7 | Severity and component reach the tracker | passed through | | U |
| H8 | `--project <campaign>` offered as a resolution alias | sent as `alias` | | U |

## I. Dashboard

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| I1 | Projects tab lists projects with counts | | | U |
| I2 | Severity filters include P0 | Fetched from `/tracker/vocabulary` | A hard-coded list omits the most severe class in the corpus | U |
| I3 | Scope list visible in one click, removable in place | | Getting a scope wrong is how issues land in the wrong log | U |
| I4 | Editing sends only changed fields | | 16 fields per visit would drown the audit trail | U |
| I5 | Status/severity apply immediately | | The two an operator changes while triaging | U |
| I6 | Each PATCH carries an actor | | | U |
| I7 | Scope conflict shows the server's explanation verbatim | names the owning project | The one fact that says what to do next | U |
| I8 | Repeated status params, not a comma list | | | U |
| I9 | Migrated issues marked as such | | Their dates come from a ledger, not this system | U |
| I10 | Empty project list reads as empty, not as failure | | | U |
| I11a | Export served as `text/markdown` | | | E |
| I11b | One heading per open issue | 80 of 80 | | E |
| I11c | Severities rendered in the heading | | | E |

## J. Isolation

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| J1 | The verification instance writes only under its own `CAO_STATE_ROOT` | live DB untouched | | E |
| J2 | No tracker **rows** in the live install | 0 rows | Empty tracker *tables* do appear there — `init_db`'s `create_all` creates every registered model whenever the CAO suite runs without `CAO_STATE_ROOT`. That is pre-existing suite behaviour and harmless; leaked data would not be | E |
| J3 | Conductor test suite performs no live tracker writes | `CAO_API_PORT` pinned closed in `tests/test_issue.py` | Otherwise running the suite files junk into the live tracker | E |

## K. Map membership and frontier (cond-0394)

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| K1 | Link a child to a map with `part-of` (child → parent) | 201, visible from both issues | A wayfinder map's tickets are first-class members, not a body convention | U |
| K2 | `GET …/children` / `cao issue children` | direct members only, creation order | Grandchildren and non-members leaking in misstates the map | U |
| K3 | `GET …/frontier` / `cao issue frontier` | nonterminal, unassigned, no nonterminal incoming `blocks`; oldest first | A wayfinder session takes the first row; a wrong row means double work or skipped work | U |
| K4 | Frontier with a claimed, closed, or blocked child | excluded | Computed from canonical records — no derived blocked flag to drift | U |
| K5 | Blocker in `resolved` (not a terminal status) | still blocks | Landed ≠ verified; reopening vocabulary would split the two stores' meanings | U |
| K6 | `cao issue show` link direction | "blocks/blocked by", "part of/contains" | A directionless "blocks X" on the blocked issue reads as its opposite; JSON keeps from/to/kind explicit | U |
| K7 | children/frontier of an unknown key | 404 | | U |

## L. Claim lifecycle (cond-0394)

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| L1 | Claim an open issue | assignee set atomically, `claim` event | Two cooperative workers must not both win the same ticket | U |
| L2 | Second claim by another worker | 409 naming the observed claimant | The conflict must state what was observed, or the loser cannot act on it | U |
| L3 | Retry by the current claimant | 200, `already_claimed`, no second event | Retries happen; idempotency keeps the audit trail honest | U |
| L4 | Claim a terminal issue | 409 naming the observed status | Claiming closed work is a stale observation, not ownership | U |
| L5 | Unclaim, then another worker claims | succeeds | Unclaim is the ordinary exit; it ships with claim, not later | U |
| L6 | Unclaim an unclaimed issue | 200 idempotent, no event | The exit must always work | U |

## M. Optimistic map edits (cond-0394)

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| M1 | PATCH with matching `expected_updated_at` | applies | The wayfinder read-then-write loop works when nobody raced it | U |
| M2 | PATCH with a stale value | 409 carrying `current_updated_at`; nothing written | A silent overwrite of a concurrent map edit is the failure this exists to prevent | U |
| M3 | PATCH without the precondition | unconditional | Legacy callers keep their semantics; protection is opt-in | U |
| M4 | An unparseable value | 400 | | U |

## N. Atomic label deltas (cond-0394)

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| N1 | `add_labels` | merged; unrelated labels kept | Triage adds a role without dropping labels another actor set | U |
| N2 | `remove_labels` | only the named labels dropped | | U |
| N3 | `clear_labels` | the set is emptied | The CLI previously had no way to clear all labels | U |
| N4 | `labels` combined with any delta | 400, nothing written | The combination's meaning would depend on a read nobody performed; refuse it | U |
| N5 | A delta that would exceed 32 labels | 400, set unchanged | Bounds apply to the merged result | U |
| N6 | The resulting set | one `labels` audit event, old → new | The audit trail records the outcome, not the arithmetic | U |

## O. Triage discovery (cond-0394)

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| O1 | `unlabeled=true` | only label-less issues | "Never triaged" is the first bucket a triage pass works from | U |
| O2 | `unlabeled` composed with status/kind/limit | intersection; `total` unpaged | A wrong total reads as "that was everything" | U |
| O3 | `cao issue list --kind issue\|feature\|all` | default stays `issue` | Triage spans features; existing scripts must not change meaning | U |
| O4 | `label=ui` still exact | no substring matches | Exactness must survive the new filters | U |

## W. Wayfinder dashboard (cond-0394)

Covered by vitest (`web/src/test/wayfinder.test.tsx`) and Playwright at
1280×800 + 390×844 (`web/e2e/wayfinder.spec.ts`, screenshots under
`web/e2e/__screenshots__/wayfinder/`).

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| W1 | List ⇄ Wayfinder view switch | URL carries `view=wayfinder` | Maps are first-class, and the view is shareable | U |
| W2 | Label chips with counts (`GET /tracker/projects/{id}/labels`) | exact-label filter; `unlabeled` chip with its count | Labels are discovered from the project, never hard-coded | U |
| W3 | Map browser → map view (`GET /tracker/issues/{key}/map`) | destination/body, progress, ordered frontier, classified children, external blockers, links — one request | N detail fetches would let widgets derive state inconsistently | U |
| W4 | Graph | arrow edges colored per kind; legend names every state/kind as text; node click selects; adjacent list carries the same state and is keyboard-operable | A merely-decorative graph is a failure; direction must be readable | U |
| W5 | Claim from frontier/detail | atomic endpoint; 409 surfaces the observed owner | The generic assignee PATCH cannot refuse a second claimant | U |
| W6 | Unclaim | releases; row re-reads | The recovery exit ships with the guard | U |
| W7 | Map-body edit with `expected_updated_at` | stale 409 preserves the draft, banner names the current version, re-read & retry applies the draft | A concurrent session's map edit is never silently overwritten | U |
| W8 | Label edits from the detail | `add_labels`/`remove_labels` deltas, never full replacement | A concurrent tag survives | U |
| W9 | Link rendering in detail | "blocks/blocked by", "part of/contains", etc., key navigates | Directionless "blocks X" reads as its opposite on the wrong issue | U |
| W10 | Empty/loading/error states; no horizontal overflow at 390px | truthful empty state naming the `wayfinder:map` convention | | U |
| W11 | External endpoints in the projection | every link endpoint that is not a member materializes in `external` with a `blocking` list naming the children it actually benches; graph and list render relates/duplicates/caused-by to non-members in both directions, the actual blocker marked | A link whose endpoint is invisible renders as nothing — the map silently loses relationships | U |
| W12 | Back/Forward across the tracker URL state | view/map/key/label/unlabeled/project all restore, absent params clear, and no duplicate entry is pushed (the forward stack survives) | A popstate-triggered duplicate push truncates Forward and strands the operator mid-history | U |

## X. Generic issue graph (cond-0521)

Covered by service/API pytest and vitest
(`web/src/test/issueGraph.test.ts`, `web/src/test/wayfinder.test.tsx`, and
`web/src/test/appDeepLink.test.tsx`).

| # | Check | Expected | Why it matters | |
|---|---|---|---|---|
| X1 | `GET /tracker/issues/{key}/graph` from any issue | bounded transitive `part-of` descendants, depth and parent metadata | Planning hierarchy is not restricted to Wayfinder maps or issue kinds | U |
| X2 | Relationship context | every returned edge has a materialized endpoint; hierarchy children omitted by a bound do not reappear as external context | The graph neither drops meaningful links nor disguises a partial hierarchy | U |
| X3 | Cycle and depth/node bounds | traversal terminates; response names every truncation reason | A malformed hierarchy cannot hang the dashboard or look complete when partial | U |
| X4 | Projects → Issues → Graph | searchable root, quick project roots, Hierarchy/Relationships modes, static type/status filters | Large open vocabularies stay searchable while enums remain quick toggles | U |
| X5 | Collapse, select, and focus | descendants collapse in graph and accessible list; selection opens shared detail; focus reroots | The visualization remains operable without pixel hunting and supports progressive exploration | U |
| X6 | Graph deep link | URL carries `view=graph&root=<key>` and restores Projects/Issues/Graph | A planning view can be shared and refreshed without losing context | U |



---

## Run log

`scripts/verify-issue-tracker.sh` — **77 of 77 passed** on an isolated instance
(own `CAO_STATE_ROOT`, own `TMUX_TMPDIR`, port 9975), including a full import of
the real 208-entry conductor ledger.

The first run found five things the unit tests could not, because the unit
tests supply their own fixtures and the harness does not:

1. **PATCH ignored unknown fields and answered 200.** `{"project_id": "other"}`
   looked like it moved an issue between projects and did nothing. Fixed by
   making every tracker request body reject extra fields.
2. **Issue prefixes were not unique.** Importing a `cond-NNNN` ledger into a
   second project that also used `cond` collided key by key, with conflicts
   naming a project the caller never mentioned. Keys are unique
   installation-wide, so a shared prefix is now refused at the moment somebody
   chooses it.
3. **`limit` was clamped in the service but range-checked at the route**, so
   the two disagreed about what an oversized limit means. The route's 422 wins;
   the service keeps its clamp for direct callers and echoes what it applied.
4. **The live database had six empty tracker tables** — created by `init_db`'s
   `create_all` when the CAO suite runs without `CAO_STATE_ROOT`. Zero rows, no
   data leaked, and pre-existing behaviour rather than something this work
   introduced. The isolation check now asserts on rows, which is the invariant
   that actually matters.
5. Two of the matrix's own expectations were wrong (a one-word project's
   default prefix is the slug, not its initials; the real `cond-0039` carries
   no one-off ledger field — that was fixture invention). Corrected here.
