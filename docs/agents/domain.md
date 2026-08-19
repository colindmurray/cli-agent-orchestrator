# Domain docs

This repository uses a single domain context.

## Before exploring, read these

- `CONTEXT.md` at the repository root, when present.
- ADRs in `docs/adr/` that touch the area being changed, when present.

Missing domain files are not a setup failure. Proceed silently; the
`domain-modeling` skill creates or extends them when the work actually resolves
terminology or a durable decision.

## Cross-repository boundary

The `cli-agent-orchestrator` fork and `cao-conductor` form one product but
retain separate source contexts. The fork owns the underlying CAO runtime,
provider integration, server/API, and dashboard. Conductor owns campaign
policy, managed execution, skills, project configuration, and deployment
coordination.

For changes crossing that boundary, inspect both repositories, record both
revisions, and verify the paired integration. Keep an ADR in the repository
that owns the coordinating contract and link to it from the sibling rather than
duplicating the decision.

## Use the glossary's vocabulary

When output names a domain concept—in an issue title, refactor proposal,
hypothesis, or test—use the term defined in `CONTEXT.md`. Do not drift to a
synonym the glossary explicitly avoids. If the concept is absent, reconsider
whether it is project language or note the genuine gap for domain modeling.

## Flag ADR conflicts

Surface a contradiction with an existing ADR explicitly rather than silently
overriding it. Name the ADR and explain why the decision may need to be
reopened.
