# Issue tracker: CAO

Issues for the `cli-agent-orchestrator` fork and `cao-conductor` live in the
installation-wide CAO tracker project `cao-system`. Use `conduct issue`; do not
create a parallel GitHub or Markdown issue ledger. GitHub pull requests remain
the publication and optional external-review surface.

The primary Mac mini serves the dashboard on `http://127.0.0.1:9889`. Select
the `cao-system` project there for its Home, Issues, Wayfinder, Graph, and
Sessions views.

The Mac mini installation and database are authoritative. From the MacBook,
run tracker commands through `ssh mac-mini` from the matching repository path;
do not use a MacBook loopback server or its legacy `cond-*` records.

## Resolve and inspect

`conduct issue where` resolves this repository and its worktrees and reports
the installation ID. Use `--tracker-project cao-system` when operating outside
a resolved path. Before a write, pass the reported ID as
`--expect-installation <uuid>` or set `CAO_EXPECTED_INSTALLATION_ID`.

```sh
conduct issue show --id <key>
conduct issue list --tracker-project cao-system --kind all --query "<query>"
```

## Publish

Always pass an explicit kind when filing. Available kinds are `project`,
`milestone`, `goal`, `epic`, `feature`, `story`, `task`, and `bug`.

```sh
conduct issue file \
  --tracker-project cao-system \
  --kind <kind> \
  --title "<title>" \
  --body-file <path>
```

Bugs should include reproduction steps, expected outcome, and actual outcome.
Use `--force` only for a documented exception.

## Relate work

Use native relationships:

```sh
conduct issue link --id <child> --to <parent> --kind part-of
conduct issue link --id <blocker> --to <dependent> --kind blocks
conduct issue link --id <a> --to <b> --kind relates
```

When a publisher reviewed the endpoint clocks before writing, fence the write
at commit time instead of trusting that earlier read:

```sh
conduct issue comment --id <key> --body-file <path> --expect-updated-at <clock> --json
conduct issue link --id <from> --to <to> \
  --expect-from-updated-at <from-clock> --expect-to-updated-at <to-clock> --json
conduct issue close --id <key> --as resolved --expect-updated-at <clock> --json
```

The comment response carries its `id`, new parent `updated_at`, and audit
`effect_id`; a new link carries its `id`, both new endpoint clocks, and both
audit `effect_ids`; a status/close response carries its committed `updated_at`
and status `effect_id`. A stale supplied clock is a typed conflict and writes
no partial comment, relation, audit event, or status update. Omitting clocks
preserves ordinary unfenced work.

Labels represent workflow state, initiatives, sessions, and other
cross-cutting cohorts. Relationships represent containment and execution
order. A cross-repository outcome may use one issue with multiple recorded
branches, worktrees, and pull requests. Split independently deliverable
outcomes into separate issues and connect them.

## Own and finish

Claim work before implementation:

```sh
conduct issue claim --id <key>
```

Managed CAO workers use their detected identity. Outside CAO, pass `--as
<actor-id>`. Record collaborators, branches, worktrees, pull requests,
contributions, and verification evidence. Mark implementation complete with
`resolved`; mark the issue `closed` only after verifying its expected outcome.

```sh
conduct issue close --id <key> --as resolved --resolution "<implementation>"
conduct issue close --id <key> --resolution "verified by <probe> at <revision>"
```

## Matt Pocock workflows

Load `$cao-matt-pocock-skills` alongside any Matt Pocock engineering skill that
uses the tracker.

- `/to-spec` publishes a `feature` or `epic`.
- `/to-tickets` publishes `story` or `task` children.
- `/implement` claims the item and records execution artifacts.
- `/code-review` resolves CAO keys from commits and reads the canonical parent.
- `/triage` uses the configured CAO labels and corresponding statuses.

Read `docs/agents/triage-labels.md` before applying a triage role.

## Wayfinding

A Wayfinder map is a favorite `project` labelled `wayfinder:map`. Decision
tickets are `task` children labelled `wayfinder:research`,
`wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`.

Create every ticket before adding `part-of` and `blocks` relationships. Use:

```sh
conduct issue map --id <map>
conduct issue frontier --id <map>
conduct issue audit --id <map>
conduct issue claim --id <ticket>
```

Resolve a decision with a comment, close its ticket, then update the map using
`--expect-updated-at`.
