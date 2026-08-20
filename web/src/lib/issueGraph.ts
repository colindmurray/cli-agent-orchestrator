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

/**
 * Return hierarchy rows in deterministic preorder, independently of the order
 * used by the graph API. The API currently walks breadth-first, which is useful
 * for bounded discovery but makes an expanded tree look as though children
 * belong to the final sibling above them.
 */
export function orderIssueHierarchyNodes(
  projection: TrackerGraphProjection,
  visible: Set<string>,
): TrackerGraphNode[] {
  const byKey = new Map(projection.nodes.map(node => [node.key, node]))
  const apiOrder = new Map(projection.nodes.map((node, index) => [node.key, index]))
  const parentCandidates = new Map<string, string[]>()

  for (const link of projection.links) {
    if (link.kind !== 'part-of' || !byKey.has(link.from_key) || !byKey.has(link.to_key)) continue
    const candidates = parentCandidates.get(link.from_key) ?? []
    candidates.push(link.to_key)
    parentCandidates.set(link.from_key, candidates)
  }
  for (const node of projection.nodes) {
    if (parentCandidates.has(node.key)) continue
    const candidates = node.parent_keys.filter(key => byKey.has(key))
    if (candidates.length) parentCandidates.set(node.key, candidates)
  }

  const primaryParent = new Map<string, string>()
  for (const [child, candidates] of parentCandidates) {
    const parent = [...new Set(candidates)].sort()[0]
    if (parent && child !== projection.root.key) primaryParent.set(child, parent)
  }
  const children = new Map<string, string[]>()
  for (const [child, parent] of primaryParent) {
    const branch = children.get(parent) ?? []
    branch.push(child)
    children.set(parent, branch)
  }
  for (const branch of children.values()) {
    branch.sort((left, right) =>
      (apiOrder.get(left) ?? Number.MAX_SAFE_INTEGER) -
        (apiOrder.get(right) ?? Number.MAX_SAFE_INTEGER) || left.localeCompare(right),
    )
  }

  const ordered: TrackerGraphNode[] = []
  const visited = new Set<string>()
  const visit = (key: string): void => {
    if (visited.has(key)) return
    visited.add(key)
    const node = byKey.get(key)
    if (!node) return
    if (visible.has(key)) ordered.push(node)
    for (const child of children.get(key) ?? []) visit(child)
  }

  visit(projection.root.key)
  for (const node of projection.nodes) visit(node.key)
  return ordered
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
      label: issue.key,
      displayLabel: `${issue.key} · ${issue.title}`,
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
    assignHierarchyLayout(graph, projection.root.key)
  }
  return graph
}

function assignHierarchyLayout(graph: Graph, preferredRoot: string): void {
  const parents = new Map<string, string[]>()
  graph.forEachEdge((_edge, attributes, source, target) => {
    if (attributes.kind !== 'part-of') return
    const values = parents.get(source) ?? []
    values.push(target)
    parents.set(source, values)
  })
  for (const values of parents.values()) values.sort()

  // Layout one canonical tree while retaining every graph edge. Multiple
  // parents remain visible and are reported by the audit; choosing one
  // deterministic parent here prevents a node receiving incompatible
  // coordinates.
  const primaryParent = new Map<string, string>()
  graph.forEachNode(key => {
    const candidates = parents.get(key) ?? []
    if (candidates.length) primaryParent.set(key, candidates[0])
  })
  const children = new Map<string, string[]>()
  for (const [child, parent] of primaryParent) {
    const values = children.get(parent) ?? []
    values.push(child)
    children.set(parent, values)
  }
  for (const values of children.values()) values.sort()

  const assigned = new Map<string, number>()
  const visiting = new Set<string>()
  let nextLeaf = 0
  const assign = (key: string): number => {
    const known = assigned.get(key)
    if (known !== undefined) return known
    if (visiting.has(key)) {
      const x = nextLeaf++ * 3
      assigned.set(key, x)
      return x
    }
    visiting.add(key)
    const branch = (children.get(key) ?? []).filter(child => graph.hasNode(child))
    const childXs = branch.map(assign)
    const x = childXs.length
      ? childXs.reduce((sum, value) => sum + value, 0) / childXs.length
      : nextLeaf++ * 3
    visiting.delete(key)
    assigned.set(key, x)
    return x
  }

  const roots = [
    ...(graph.hasNode(preferredRoot) ? [preferredRoot] : []),
    ...graph.nodes().filter(key => key !== preferredRoot && !primaryParent.has(key)).sort(),
    ...graph.nodes().filter(key => key !== preferredRoot).sort(),
  ]
  for (const key of roots) assign(key)

  const values = [...assigned.values()]
  const centre = values.length ? (Math.min(...values) + Math.max(...values)) / 2 : 0
  graph.forEachNode((key, attributes) => {
    graph.setNodeAttribute(key, 'x', (assigned.get(key) ?? 0) - centre)
    graph.setNodeAttribute(key, 'y', -Number(attributes.depth ?? 0) * 3)
  })
}
