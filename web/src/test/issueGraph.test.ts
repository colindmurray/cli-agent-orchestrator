import { describe, expect, it } from 'vitest'
import type { TrackerGraphNode, TrackerGraphProjection, TrackerIssue } from '../api'
import {
  buildIssueGraph,
  buildIssueDependencyPlan,
  orderIssueHierarchyNodes,
  visibleIssueGraphKeys,
} from '../lib/issueGraph'

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
    expect(graph.getNodeAttribute(TASK.key, 'label')).toBe(TASK.key)
    expect(graph.getNodeAttribute(TASK.key, 'displayLabel')).toContain(TASK.title)
  })

  it('allocates horizontal space by subtree leaves instead of row index alone', () => {
    const LEFT = node('cond-0503', 'Left branch', 1, [ROOT.key], 2)
    const RIGHT = node('cond-0504', 'Right branch', 1, [ROOT.key], 1)
    const LEFT_A = node('cond-0505', 'Left A', 2, [LEFT.key], 0)
    const LEFT_B = node('cond-0506', 'Left B', 2, [LEFT.key], 0)
    const RIGHT_A = node('cond-0507', 'Right A', 2, [RIGHT.key], 0)
    const projection: TrackerGraphProjection = {
      ...PROJECTION,
      root: ROOT,
      nodes: [{ ...ROOT, child_count: 2 }, LEFT, RIGHT, LEFT_A, LEFT_B, RIGHT_A],
      external: [],
      links: [
        { id: 10, kind: 'part-of', from_key: LEFT.key, to_key: ROOT.key },
        { id: 11, kind: 'part-of', from_key: RIGHT.key, to_key: ROOT.key },
        { id: 12, kind: 'part-of', from_key: LEFT_A.key, to_key: LEFT.key },
        { id: 13, kind: 'part-of', from_key: LEFT_B.key, to_key: LEFT.key },
        { id: 14, kind: 'part-of', from_key: RIGHT_A.key, to_key: RIGHT.key },
      ],
    }
    const graph = buildIssueGraph(projection, 'hierarchy', EMPTY_FILTERS)
    const leftMidpoint = (
      graph.getNodeAttribute(LEFT_A.key, 'x') + graph.getNodeAttribute(LEFT_B.key, 'x')
    ) / 2
    expect(graph.getNodeAttribute(LEFT.key, 'x')).toBe(leftMidpoint)
    expect(graph.getNodeAttribute(RIGHT.key, 'x')).toBe(
      graph.getNodeAttribute(RIGHT_A.key, 'x'),
    )
    expect(graph.getNodeAttribute(LEFT_B.key, 'x')).toBeLessThan(
      graph.getNodeAttribute(RIGHT_A.key, 'x'),
    )
  })

  it('orders breadth-first API nodes as a subtree-contiguous hierarchy', () => {
    const LEFT = node('cond-0503', 'Left branch', 1, [ROOT.key], 1)
    const RIGHT = node('cond-0504', 'Right branch', 1, [ROOT.key], 0)
    const LEFT_CHILD = node('cond-0505', 'Left child', 2, [LEFT.key], 0)
    const ROOT_ROW: TrackerGraphNode = { ...ROOT, child_count: 2 }
    const projection: TrackerGraphProjection = {
      ...PROJECTION,
      root: ROOT,
      // The API returns breadth-first order: both siblings precede the child.
      nodes: [ROOT_ROW, LEFT, RIGHT, LEFT_CHILD],
      external: [],
      links: [
        { id: 10, kind: 'part-of', from_key: LEFT.key, to_key: ROOT.key },
        { id: 11, kind: 'part-of', from_key: RIGHT.key, to_key: ROOT.key },
        { id: 12, kind: 'part-of', from_key: LEFT_CHILD.key, to_key: LEFT.key },
      ],
    }

    expect(orderIssueHierarchyNodes(projection, new Set(projection.nodes.map(row => row.key))))
      .toEqual([ROOT_ROW, LEFT, LEFT_CHILD, RIGHT])
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

  it('projects blocker fan-out and joins into deterministic execution stages', () => {
    const DISCOVER = node('cond-0600', 'Discover constraints', 1, [ROOT.key], 0)
    const API = node('cond-0601', 'Build API', 1, [ROOT.key], 0)
    const UI = node('cond-0602', 'Build UI', 1, [ROOT.key], 0)
    const SHIP = node('cond-0603', 'Ship integration', 1, [ROOT.key], 0)
    const DOCS = node('cond-0604', 'Write independent docs', 1, [ROOT.key], 0)
    const projection: TrackerGraphProjection = {
      ...PROJECTION,
      nodes: [ROOT, DISCOVER, API, UI, SHIP, DOCS],
      external: [],
      links: [
        { id: 20, kind: 'blocks', from_key: DISCOVER.key, to_key: API.key },
        { id: 21, kind: 'blocks', from_key: DISCOVER.key, to_key: UI.key },
        { id: 22, kind: 'blocks', from_key: API.key, to_key: SHIP.key },
        { id: 23, kind: 'blocks', from_key: UI.key, to_key: SHIP.key },
      ],
    }

    const plan = buildIssueDependencyPlan(projection, EMPTY_FILTERS)
    expect(plan.nodes.map(row => [row.key, row.stage])).toEqual([
      [DISCOVER.key, 0],
      [API.key, 1],
      [UI.key, 1],
      [SHIP.key, 2],
      [DOCS.key, 0],
    ])
    expect(plan.tracks).toHaveLength(2)
    expect(plan.tracks[0].stages.map(stage => stage.nodes.map(row => row.key))).toEqual([
      [DISCOVER.key],
      [API.key, UI.key],
      [SHIP.key],
    ])
    expect(plan.tracks[1].independent).toBe(true)
    expect(plan.cycles).toEqual([])
  })

  it('keeps sequencing visible when a container hides nested scope', () => {
    const STORY = node('cond-0610', 'Deliver recovery', 1, [ROOT.key], 1, { kind: 'story' })
    const TASK = node('cond-0611', 'Implement recovery', 2, [STORY.key], 0)
    const VERIFY = node('cond-0612', 'Verify recovery', 1, [ROOT.key], 0)
    const PREP = node('cond-0613', 'Prepare recovery', 2, [STORY.key], 0)
    const projection: TrackerGraphProjection = {
      ...PROJECTION,
      nodes: [ROOT, { ...STORY, child_count: 2 }, VERIFY, TASK, PREP],
      external: [BLOCKER],
      links: [
        { id: 30, kind: 'part-of', from_key: STORY.key, to_key: ROOT.key },
        { id: 31, kind: 'part-of', from_key: TASK.key, to_key: STORY.key },
        { id: 32, kind: 'part-of', from_key: VERIFY.key, to_key: ROOT.key },
        { id: 33, kind: 'blocks', from_key: BLOCKER.key, to_key: TASK.key },
        { id: 34, kind: 'blocks', from_key: TASK.key, to_key: VERIFY.key },
        { id: 35, kind: 'part-of', from_key: PREP.key, to_key: STORY.key },
        { id: 36, kind: 'blocks', from_key: TASK.key, to_key: PREP.key },
      ],
    }

    const plan = buildIssueDependencyPlan(projection, {
      ...EMPTY_FILTERS,
      collapsed: new Set([STORY.key]),
    })
    expect(plan.nodes.map(row => row.key)).toEqual([BLOCKER.key, STORY.key, VERIFY.key])
    expect(plan.edges).toEqual(expect.arrayContaining([
      expect.objectContaining({ from: BLOCKER.key, to: STORY.key, cleared: false }),
      expect.objectContaining({ from: STORY.key, to: VERIFY.key, cleared: false }),
    ]))
    expect(plan.edges).toHaveLength(2)
    expect(plan.nodes.find(row => row.key === STORY.key)?.hiddenScopeCount).toBe(2)
    expect(plan.nodes.find(row => row.key === STORY.key)?.hiddenDependencyCount).toBe(1)
    expect(plan.totalDependencyCount).toBe(3)
    expect(plan.visibleDependencyCount).toBe(2)
    expect(plan.openDependencyCount).toBe(3)
    expect(plan.clearedDependencyCount).toBe(0)
    expect(plan.hiddenDependencyCount).toBe(1)
    expect(plan.tracks[0].total).toBe(5)

    const graph = buildIssueGraph(projection, 'dependencies', {
      ...EMPTY_FILTERS,
      collapsed: new Set([STORY.key]),
    })
    expect(graph.edges().map(edge => graph.getEdgeAttributes(edge).kind)).toEqual(['blocks', 'blocks'])
    expect(graph.getNodeAttribute(BLOCKER.key, 'x')).toBeLessThan(graph.getNodeAttribute(STORY.key, 'x'))
    expect(graph.getNodeAttribute(STORY.key, 'x')).toBeLessThan(graph.getNodeAttribute(VERIFY.key, 'x'))
  })

  it('condenses blocker cycles without inventing a false order', () => {
    const LEFT = node('cond-0620', 'Left side', 1, [ROOT.key], 0)
    const RIGHT = node('cond-0621', 'Right side', 1, [ROOT.key], 0)
    const AFTER = node('cond-0622', 'After cycle', 1, [ROOT.key], 0)
    const projection: TrackerGraphProjection = {
      ...PROJECTION,
      nodes: [ROOT, LEFT, RIGHT, AFTER],
      external: [],
      links: [
        { id: 40, kind: 'blocks', from_key: LEFT.key, to_key: RIGHT.key },
        { id: 41, kind: 'blocks', from_key: RIGHT.key, to_key: LEFT.key },
        { id: 42, kind: 'blocks', from_key: RIGHT.key, to_key: AFTER.key },
      ],
    }

    const plan = buildIssueDependencyPlan(projection, EMPTY_FILTERS)
    expect(plan.cycles).toEqual([[LEFT.key, RIGHT.key]])
    expect(plan.nodes.find(row => row.key === LEFT.key)?.stage).toBe(0)
    expect(plan.nodes.find(row => row.key === RIGHT.key)?.stage).toBe(0)
    expect(plan.nodes.find(row => row.key === AFTER.key)?.stage).toBe(1)
  })
})
