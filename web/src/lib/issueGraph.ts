import Graph from 'graphology'
import { circular } from 'graphology-layout'
import type {
  TrackerGraphNode,
  TrackerGraphProjection,
  TrackerIssue,
  TrackerLink,
} from '../api'
import { MAP_EDGE_COLORS } from './issueMap'

export type IssueGraphMode = 'hierarchy' | 'relationships'

export interface IssueGraphFilters {
  query: string
  kinds: string[]
  statuses: string[]
  collapsed: Set<string>
}

export const ISSUE_GRAPH_NODE_COLORS = {
  root: '#f9fafb',
  open: '#38bdf8',
  active: '#60a5fa',
  blocked: '#f87171',
  resolved: '#2dd4bf',
  terminal: '#6b7280',
  external: '#fb923c',
} as const

function nodeMatches(issue: TrackerIssue, filters: IssueGraphFilters): boolean {
  const needle = filters.query.trim().toLocaleLowerCase()
  if (needle && !`${issue.key} ${issue.title}`.toLocaleLowerCase().includes(needle)) return false
  if (filters.kinds.length && !filters.kinds.includes(issue.kind)) return false
  if (filters.statuses.length && !filters.statuses.includes(issue.status)) return false
  return true
}

function hierarchyVisibleKeys(
  projection: TrackerGraphProjection,
  filters: IssueGraphFilters,
): Set<string> {
  const byKey = new Map(projection.nodes.map(node => [node.key, node]))
  const visible = new Set<string>([projection.root.key])
  const matches = projection.nodes.filter(node => nodeMatches(node, filters))
  for (const node of matches) {
    visible.add(node.key)
    const pending = [...node.parent_keys]
    while (pending.length) {
      const parent = pending.pop()!
      if (visible.has(parent)) continue
      visible.add(parent)
      pending.push(...(byKey.get(parent)?.parent_keys ?? []))
    }
  }

  // Collapsing a node hides every descendant, even if a descendant matched a
  // filter. The row remains visible and names how many direct children it has.
  const children = new Map<string, string[]>()
  for (const link of projection.links) {
    if (link.kind !== 'part-of' || !byKey.has(link.from_key) || !byKey.has(link.to_key)) continue
    const rows = children.get(link.to_key) ?? []
    rows.push(link.from_key)
    children.set(link.to_key, rows)
  }
  for (const collapsed of filters.collapsed) {
    const pending = [...(children.get(collapsed) ?? [])]
    while (pending.length) {
      const child = pending.pop()!
      visible.delete(child)
      pending.push(...(children.get(child) ?? []))
    }
  }
  return visible
}

export function visibleIssueGraphKeys(
  projection: TrackerGraphProjection,
  mode: IssueGraphMode,
  filters: IssueGraphFilters,
): Set<string> {
  if (mode === 'hierarchy') return hierarchyVisibleKeys(projection, filters)
  const visible = new Set<string>([projection.root.key])
  for (const issue of [...projection.nodes, ...projection.external]) {
    if (nodeMatches(issue, filters)) visible.add(issue.key)
  }
  return visible
}

function nodeColor(
  issue: TrackerIssue,
  projection: TrackerGraphProjection,
  external: Set<string>,
): string {
  if (issue.key === projection.root.key) return ISSUE_GRAPH_NODE_COLORS.root
  if (external.has(issue.key)) return ISSUE_GRAPH_NODE_COLORS.external
  if (issue.status === 'in-progress') return ISSUE_GRAPH_NODE_COLORS.active
  if (issue.status === 'blocked') return ISSUE_GRAPH_NODE_COLORS.blocked
  if (issue.status === 'resolved') return ISSUE_GRAPH_NODE_COLORS.resolved
  if (['closed', 'wontfix', 'duplicate'].includes(issue.status)) {
    return ISSUE_GRAPH_NODE_COLORS.terminal
  }
  return ISSUE_GRAPH_NODE_COLORS.open
}

const EDGE_PRIORITY: Record<string, number> = {
  blocks: 5,
  'part-of': 4,
  'caused-by': 3,
  duplicates: 2,
  relates: 1,
}

export function buildIssueGraph(
  projection: TrackerGraphProjection,
  mode: IssueGraphMode,
  filters: IssueGraphFilters,
): Graph {
  const graph = new Graph()
  const visible = visibleIssueGraphKeys(projection, mode, filters)
  const external = new Set(projection.external.map(issue => issue.key))
  const allNodes: TrackerIssue[] = mode === 'hierarchy'
    ? projection.nodes
    : [...projection.nodes, ...projection.external]

  for (const issue of allNodes) {
    if (!visible.has(issue.key)) continue
    graph.addNode(issue.key, {
      label: `${issue.key} · ${issue.title}`,
      title: issue.title,
      kind: issue.kind,
      status: issue.status,
      depth: (issue as TrackerGraphNode).depth ?? null,
      external: external.has(issue.key),
      size: issue.key === projection.root.key ? 13 : external.has(issue.key) ? 5 : 7,
      color: nodeColor(issue, projection, external),
    })
  }

  const best = new Map<string, TrackerLink>()
  for (const link of projection.links) {
    if (mode === 'hierarchy' && link.kind !== 'part-of') continue
    if (!graph.hasNode(link.from_key) || !graph.hasNode(link.to_key)) continue
    const pair = JSON.stringify([link.from_key, link.to_key])
    const existing = best.get(pair)
    if (!existing || (EDGE_PRIORITY[link.kind] ?? 0) > (EDGE_PRIORITY[existing.kind] ?? 0)) {
      best.set(pair, link)
    }
  }
  for (const link of best.values()) {
    graph.addEdge(link.from_key, link.to_key, {
      kind: link.kind,
      label: `${link.kind}: ${link.from_key} → ${link.to_key}`,
      color: MAP_EDGE_COLORS[link.kind] ?? MAP_EDGE_COLORS.relates,
      size: link.kind === 'blocks' || link.kind === 'part-of' ? 2 : 1,
      type: 'arrow',
    })
  }

  if (mode === 'relationships') {
    circular.assign(graph)
  } else {
    const levels = new Map<number, string[]>()
    graph.forEachNode((key, attributes) => {
      const depth = Number(attributes.depth ?? 0)
      const rows = levels.get(depth) ?? []
      rows.push(key)
      levels.set(depth, rows)
    })
    for (const [depth, keys] of levels) {
      keys.sort()
      keys.forEach((key, index) => {
        graph.setNodeAttribute(key, 'x', index - (keys.length - 1) / 2)
        graph.setNodeAttribute(key, 'y', -depth * 1.5)
      })
    }
  }
  return graph
}
