# Triage roles: CAO

| Matt role | CAO label | CAO status |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | `triage` |
| `needs-info` | `needs-info` | `blocked` |
| `ready-for-agent` | `ready-for-agent` | `open` |
| `ready-for-human` | `ready-for-human` | `open` |
| `wontfix` | `wontfix` | `wontfix` |

Maintain exactly one triage-state label. CAO provenance, initiative, and session
labels do not count as triage state.

Matt's **Unlabeled** bucket means no label from this table. Find it using one
repeatable `conduct issue list --without-label <state>` argument for every row;
`--unlabeled` instead requires the complete CAO label set to be empty.

For intake, Matt category `bug` maps to CAO kind `bug`, and `enhancement` maps
to `feature`. Preserve structural kinds such as `project`, `epic`, `story`, and
`task`.
