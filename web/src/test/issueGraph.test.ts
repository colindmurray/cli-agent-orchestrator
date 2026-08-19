import { describe, expect, it } from 'vitest'
import type { TrackerGraphNode, TrackerGraphProjection, TrackerIssue } from '../api'
import { buildIssueGraph, visibleIssueGraphKeys } from '../lib/issueGraph'

function issue(overrides: Partial<TrackerIssue> & Pick<TrackerIssue, 'key' | 'title'>): TrackerIssue {
  return {
    project_id: 'cao-system',
    kind: 'task',
    body: '',
    status: 'open',
    severity: 'unset',
    component: null,
    reporter: null,
    assignee: null,
    labels: [],
    collaborators: [],
    branches: [],
    worktrees: [],
    pull_requests: [],
    failing_command: null,
    reproduction_steps: null,
    expected_outcome: null,
    actual_outcome: null,
    evidence: null,
    resolution: null,
    session_name: null,
    terminal_id: null,
    source_path: null,
    duplicate_of: null,
    origin: 'api',
    favorite: false,
    created_at: '2026-08-19T00:00:00Z',
    updated_at: '2026-08-19T00:00:00Z',
    closed_at: null,
    comments: [],
    events: [],
    links: [],
    ...overrides,
  }
}

function node(
  key: string,
  title: string,
  depth: number,
  parentKeys: string[],
  childCount: number,
  overrides: Partial<TrackerIssue> = {},
): TrackerGraphNode {
  return {
    ...issue({ key, title, ...overrides }),
    depth,
    parent_keys: parentKeys,
    child_count: childCount,
  }
}

const ROOT = node('cond-0500', 'Project Apollo', 0, [], 1, { kind: 'project' })
const MILESTONE = node('cond-0501', 'Ship beta', 1, ['cond-0500'], 1, { kind: 'milestone' })
const TASK = node('cond-0502', 'Implement graph', 2, ['cond-0501'], 0, { status: 'in-progress' })
const BLOCKER = issue({ key: 'cond-0400', title: 'Repair database', kind: 'bug', status: 'blocked' })

const PROJECTION: TrackerGraphProjection = {
  root: ROOT,
  nodes: [ROOT, MILESTONE, TASK],
  external: [BLOCKER],
  links: [
    { id: 1, kind: 'part-of', from_key: MILESTONE.key, to_key: ROOT.key },
    { id: 2, kind: 'part-of', from_key: TASK.key, to_key: MILESTONE.key },
    { id: 3, kind: 'blocks', from_key: BLOCKER.key, to_key: TASK.key },
    { id: 4, kind: 'relates', from_key: BLOCKER.key, to_key: TASK.key },
  ],
  bounds: { max_depth: 8, max_nodes: 300, truncated: false, reasons: [] },
  stats: { nodes: 3, descendants: 2, external: 1, links: 4, depth: 2 },
}

const EMPTY_FILTERS = { query: '', kinds: [], statuses: [], collapsed: new Set<string>() }

describe('generic issue graph', () => {
  it('lays out the full transitive hierarchy and keeps only hierarchy edges', () => {
    const graph = buildIssueGraph(PROJECTION, 'hierarchy', EMPTY_FILTERS)
    expect(graph.nodes()).toEqual([ROOT.key, MILESTONE.key, TASK.key])
    expect(graph.size).toBe(2)
    expect(graph.getEdgeAttributes(TASK.key, MILESTONE.key)).toMatchObject({
      kind: 'part-of',
      type: 'arrow',
    })
    expect(graph.getNodeAttribute(TASK.key, 'y')).toBeLessThan(
      graph.getNodeAttribute(MILESTONE.key, 'y'),
    )
  })

  it('preserves ancestors of a matching descendant and hides collapsed descendants', () => {
    const matching = visibleIssueGraphKeys(PROJECTION, 'hierarchy', {
      ...EMPTY_FILTERS,
      query: 'Implement',
    })
    expect([...matching]).toEqual(expect.arrayContaining([ROOT.key, MILESTONE.key, TASK.key]))

    const collapsed = visibleIssueGraphKeys(PROJECTION, 'hierarchy', {
      ...EMPTY_FILTERS,
      collapsed: new Set([MILESTONE.key]),
    })
    expect(collapsed.has(ROOT.key)).toBe(true)
    expect(collapsed.has(MILESTONE.key)).toBe(true)
    expect(collapsed.has(TASK.key)).toBe(false)
  })

  it('adds external context in relationship mode and prefers the stronger parallel link', () => {
    const graph = buildIssueGraph(PROJECTION, 'relationships', EMPTY_FILTERS)
    expect(graph.hasNode(BLOCKER.key)).toBe(true)
    expect(graph.getNodeAttribute(BLOCKER.key, 'external')).toBe(true)
    expect(graph.getEdgeAttributes(BLOCKER.key, TASK.key).kind).toBe('blocks')
    expect(graph.size).toBe(3)
  })

  it('filters relationship nodes by enum fields without losing the root', () => {
    const visible = visibleIssueGraphKeys(PROJECTION, 'relationships', {
      ...EMPTY_FILTERS,
      kinds: ['bug'],
      statuses: ['blocked'],
    })
    expect([...visible]).toEqual([ROOT.key, BLOCKER.key])
  })
})
