import Graph from 'graphology'
import { circular } from 'graphology-layout'
import type {
  TrackerGraphNode,
  TrackerGraphProjection,
  TrackerIssue,
  TrackerLink,
} from '../api'
import { MAP_EDGE_COLORS } from './issueMap'

export type IssueGraphMode = 'hierarchy' | 'dependencies' | 'relationships'

export interface IssueGraphFilters {
  query: string
  kinds: string[]
  statuses: string[]
  collapsed: Set<string>
}

export interface IssueDependencyEdge {
  from: string
  to: string
  cleared: boolean
  sourceLinkIds: number[]
}

export interface IssueDependencyNode {
  key: string
  issue: TrackerIssue
  external: boolean
  stage: number
  track: number
  cyclic: boolean
  hiddenScopeCount: number
  hiddenDependencyCount: number
  scopeTotal: number
  scopeTerminal: number
  scopeActive: number
  scopeBlocked: number
  unresolvedBlockers: string[]
  clearedBlockers: string[]
  blocking: string[]
}

export interface IssueDependencyStage {
  index: number
  nodes: IssueDependencyNode[]
}

export interface IssueDependencyTrack {
  id: number
  independent: boolean
  stages: IssueDependencyStage[]
  total: number
  terminal: number
  active: number
  blocked: number
}

export interface IssueDependencyPlan {
  nodes: IssueDependencyNode[]
  edges: IssueDependencyEdge[]
  tracks: IssueDependencyTrack[]
  cycles: string[][]
  totalDependencyCount: number
  visibleDependencyCount: number
  openDependencyCount: number
  clearedDependencyCount: number
  hiddenDependencyCount: number
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
  if (mode === 'dependencies') {
    return new Set(buildIssueDependencyPlan(projection, filters).nodes.map(node => node.key))
  }
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

const TERMINAL_ISSUE_STATUSES = new Set(['closed', 'wontfix', 'duplicate'])

function hierarchyParents(projection: TrackerGraphProjection): Map<string, string> {
  const members = new Set(projection.nodes.map(node => node.key))
  const candidates = new Map<string, string[]>()
  for (const link of projection.links) {
    if (link.kind !== 'part-of' || !members.has(link.from_key) || !members.has(link.to_key)) continue
    const values = candidates.get(link.from_key) ?? []
    values.push(link.to_key)
    candidates.set(link.from_key, values)
  }
  for (const node of projection.nodes) {
    if (candidates.has(node.key)) continue
    const values = node.parent_keys.filter(key => members.has(key))
    if (values.length) candidates.set(node.key, values)
  }
  return new Map(
    [...candidates.entries()]
      .map(([child, values]) => [child, [...new Set(values)].sort()[0]] as const)
      .filter((entry): entry is readonly [string, string] => Boolean(entry[1])),
  )
}

function stronglyConnectedComponents(
  keys: string[],
  edges: IssueDependencyEdge[],
  order: Map<string, number>,
): string[][] {
  const adjacency = new Map(keys.map(key => [key, [] as string[]]))
  for (const edge of edges) adjacency.get(edge.from)?.push(edge.to)
  for (const values of adjacency.values()) {
    values.sort((left, right) => (order.get(left) ?? 0) - (order.get(right) ?? 0) || left.localeCompare(right))
  }

  let nextIndex = 0
  const indexes = new Map<string, number>()
  const lowLinks = new Map<string, number>()
  const stack: string[] = []
  const onStack = new Set<string>()
  const components: string[][] = []
  const visit = (key: string): void => {
    indexes.set(key, nextIndex)
    lowLinks.set(key, nextIndex)
    nextIndex += 1
    stack.push(key)
    onStack.add(key)
    for (const target of adjacency.get(key) ?? []) {
      if (!indexes.has(target)) {
        visit(target)
        lowLinks.set(key, Math.min(lowLinks.get(key)!, lowLinks.get(target)!))
      } else if (onStack.has(target)) {
        lowLinks.set(key, Math.min(lowLinks.get(key)!, indexes.get(target)!))
      }
    }
    if (lowLinks.get(key) !== indexes.get(key)) return
    const component: string[] = []
    while (stack.length) {
      const member = stack.pop()!
      onStack.delete(member)
      component.push(member)
      if (member === key) break
    }
    component.sort((left, right) => (order.get(left) ?? 0) - (order.get(right) ?? 0) || left.localeCompare(right))
    components.push(component)
  }
  for (const key of keys) if (!indexes.has(key)) visit(key)
  return components.sort((left, right) =>
    (order.get(left[0]) ?? 0) - (order.get(right[0]) ?? 0) || left[0].localeCompare(right[0]),
  )
}

/**
 * Project the scope hierarchy onto its execution order. `part-of` decides
 * which descendants a collapsed container represents; only `blocks` creates
 * sequence. Disconnected blocker components are independent work tracks.
 */
export function buildIssueDependencyPlan(
  projection: TrackerGraphProjection,
  filters: IssueGraphFilters,
): IssueDependencyPlan {
  const memberKeys = new Set(projection.nodes.map(node => node.key))
  const externalKeys = new Set(projection.external.map(node => node.key))
  const allIssues = [...projection.nodes, ...projection.external]
  const byKey = new Map(allIssues.map(issue => [issue.key, issue]))
  const order = new Map(allIssues.map((issue, index) => [issue.key, index]))
  const parents = hierarchyParents(projection)
  const scopeVisible = hierarchyVisibleKeys(projection, {
    query: '',
    kinds: [],
    statuses: [],
    collapsed: filters.collapsed,
  })
  const filteredVisible = hierarchyVisibleKeys(projection, filters)
  const activeFilter = Boolean(filters.query.trim() || filters.kinds.length || filters.statuses.length)

  const representativeCache = new Map<string, string | null>()
  const representative = (key: string): string | null => {
    if (!memberKeys.has(key)) return byKey.has(key) ? key : null
    const cached = representativeCache.get(key)
    if (cached !== undefined) return cached
    let current: string | undefined = key
    const seen = new Set<string>()
    while (current && !scopeVisible.has(current) && !seen.has(current)) {
      seen.add(current)
      current = parents.get(current)
    }
    const result = current && scopeVisible.has(current) ? current : null
    representativeCache.set(key, result)
    return result
  }

  const candidates = new Set<string>()
  for (const key of filteredVisible) {
    const mapped = representative(key)
    if (mapped && mapped !== projection.root.key) candidates.add(mapped)
  }

  const edgeRows = new Map<string, IssueDependencyEdge>()
  const hiddenDependencyCount = new Map<string, number>()
  let totalDependencyCount = 0
  let openDependencyCount = 0
  let clearedDependencyCount = 0
  for (const link of projection.links) {
    if (link.kind !== 'blocks') continue
    const from = representative(link.from_key)
    const to = representative(link.to_key)
    if (!from || !to) continue
    const relevant = !activeFilter
      || filteredVisible.has(link.from_key)
      || filteredVisible.has(link.to_key)
      || (byKey.get(link.from_key) ? nodeMatches(byKey.get(link.from_key)!, filters) : false)
      || (byKey.get(link.to_key) ? nodeMatches(byKey.get(link.to_key)!, filters) : false)
    if (!relevant) continue
    const cleared = TERMINAL_ISSUE_STATUSES.has(byKey.get(link.from_key)?.status ?? '')
    totalDependencyCount += 1
    if (cleared) clearedDependencyCount += 1
    else openDependencyCount += 1
    // A dependency internal to a collapsed scope is summarized by the scope
    // count, not misrepresented as a container blocking itself.
    if (from === to && link.from_key !== link.to_key) {
      hiddenDependencyCount.set(from, (hiddenDependencyCount.get(from) ?? 0) + 1)
      continue
    }
    candidates.add(from)
    candidates.add(to)
    const pair = JSON.stringify([from, to])
    const existing = edgeRows.get(pair)
    if (existing) {
      existing.cleared = existing.cleared && cleared
      existing.sourceLinkIds.push(link.id)
    } else {
      edgeRows.set(pair, { from, to, cleared, sourceLinkIds: [link.id] })
    }
  }

  if (!candidates.size) candidates.add(projection.root.key)
  const hiddenScopeCount = new Map<string, number>()
  const scopeStats = new Map<string, { total: number; terminal: number; active: number; blocked: number }>()
  for (const key of memberKeys) {
    const mapped = representative(key)
    if (mapped && mapped !== key) hiddenScopeCount.set(mapped, (hiddenScopeCount.get(mapped) ?? 0) + 1)
    if (!mapped) continue
    const issue = byKey.get(key)!
    const stats = scopeStats.get(mapped) ?? { total: 0, terminal: 0, active: 0, blocked: 0 }
    stats.total += 1
    if (TERMINAL_ISSUE_STATUSES.has(issue.status)) stats.terminal += 1
    if (issue.status === 'in-progress') stats.active += 1
    if (issue.status === 'blocked') stats.blocked += 1
    scopeStats.set(mapped, stats)
  }

  const keys = [...candidates].filter(key => byKey.has(key)).sort((left, right) =>
    (order.get(left) ?? Number.MAX_SAFE_INTEGER) - (order.get(right) ?? Number.MAX_SAFE_INTEGER)
      || left.localeCompare(right),
  )
  const keySet = new Set(keys)
  const edges = [...edgeRows.values()]
    .filter(edge => keySet.has(edge.from) && keySet.has(edge.to))
    .sort((left, right) =>
      (order.get(left.from) ?? 0) - (order.get(right.from) ?? 0)
        || (order.get(left.to) ?? 0) - (order.get(right.to) ?? 0)
        || left.from.localeCompare(right.from)
        || left.to.localeCompare(right.to),
    )

  const components = stronglyConnectedComponents(keys, edges, order)
  const componentByKey = new Map<string, number>()
  components.forEach((component, index) => component.forEach(key => componentByKey.set(key, index)))
  const cyclicComponents = new Set<number>()
  components.forEach((component, index) => {
    if (component.length > 1 || edges.some(edge => edge.from === component[0] && edge.to === component[0])) {
      cyclicComponents.add(index)
    }
  })

  const componentEdges = new Map<number, Set<number>>()
  const indegree = new Map(components.map((_component, index) => [index, 0]))
  for (const edge of edges) {
    const from = componentByKey.get(edge.from)!
    const to = componentByKey.get(edge.to)!
    if (from === to) continue
    const targets = componentEdges.get(from) ?? new Set<number>()
    if (!targets.has(to)) {
      targets.add(to)
      componentEdges.set(from, targets)
      indegree.set(to, (indegree.get(to) ?? 0) + 1)
    }
  }
  const componentOrder = (index: number) => order.get(components[index][0]) ?? Number.MAX_SAFE_INTEGER
  const queue = [...indegree.entries()]
    .filter(([, count]) => count === 0)
    .map(([index]) => index)
    .sort((left, right) => componentOrder(left) - componentOrder(right))
  const componentStage = new Map(components.map((_component, index) => [index, 0]))
  while (queue.length) {
    const current = queue.shift()!
    for (const target of [...(componentEdges.get(current) ?? [])].sort((left, right) => componentOrder(left) - componentOrder(right))) {
      componentStage.set(target, Math.max(componentStage.get(target) ?? 0, (componentStage.get(current) ?? 0) + 1))
      indegree.set(target, (indegree.get(target) ?? 0) - 1)
      if (indegree.get(target) === 0) {
        queue.push(target)
        queue.sort((left, right) => componentOrder(left) - componentOrder(right))
      }
    }
  }

  const undirected = new Map(keys.map(key => [key, new Set<string>()]))
  for (const edge of edges) {
    undirected.get(edge.from)?.add(edge.to)
    undirected.get(edge.to)?.add(edge.from)
  }
  const connected: string[][] = []
  const isolated: string[] = []
  const walked = new Set<string>()
  for (const key of keys) {
    if (walked.has(key)) continue
    if (!(undirected.get(key)?.size)) {
      walked.add(key)
      isolated.push(key)
      continue
    }
    const values: string[] = []
    const pending = [key]
    walked.add(key)
    while (pending.length) {
      const current = pending.shift()!
      values.push(current)
      for (const neighbour of [...(undirected.get(current) ?? [])].sort((left, right) =>
        (order.get(left) ?? 0) - (order.get(right) ?? 0) || left.localeCompare(right),
      )) {
        if (walked.has(neighbour)) continue
        walked.add(neighbour)
        pending.push(neighbour)
      }
    }
    connected.push(values)
  }
  connected.sort((left, right) => (order.get(left[0]) ?? 0) - (order.get(right[0]) ?? 0))
  const trackKeys = [...connected, ...(isolated.length ? [isolated] : [])]
  const incoming = new Map(keys.map(key => [key, [] as string[]]))
  const clearedIncoming = new Map(keys.map(key => [key, [] as string[]]))
  const outgoing = new Map(keys.map(key => [key, [] as string[]]))
  for (const edge of edges) {
    if (edge.cleared) {
      clearedIncoming.get(edge.to)?.push(edge.from)
    } else {
      incoming.get(edge.to)?.push(edge.from)
      outgoing.get(edge.from)?.push(edge.to)
    }
  }

  const nodeByKey = new Map<string, IssueDependencyNode>()
  trackKeys.forEach((track, trackIndex) => {
    for (const key of track) {
      const component = componentByKey.get(key)!
      const stats = scopeStats.get(key) ?? {
        total: 1,
        terminal: TERMINAL_ISSUE_STATUSES.has(byKey.get(key)!.status) ? 1 : 0,
        active: byKey.get(key)!.status === 'in-progress' ? 1 : 0,
        blocked: byKey.get(key)!.status === 'blocked' ? 1 : 0,
      }
      nodeByKey.set(key, {
        key,
        issue: byKey.get(key)!,
        external: externalKeys.has(key),
        stage: componentStage.get(component) ?? 0,
        track: trackIndex,
        cyclic: cyclicComponents.has(component),
        hiddenScopeCount: hiddenScopeCount.get(key) ?? 0,
        hiddenDependencyCount: hiddenDependencyCount.get(key) ?? 0,
        scopeTotal: stats.total,
        scopeTerminal: stats.terminal,
        scopeActive: stats.active,
        scopeBlocked: stats.blocked,
        unresolvedBlockers: [...(incoming.get(key) ?? [])].sort(),
        clearedBlockers: [...(clearedIncoming.get(key) ?? [])].sort(),
        blocking: [...(outgoing.get(key) ?? [])].sort(),
      })
    }
  })

  const tracks: IssueDependencyTrack[] = trackKeys.map((track, trackIndex) => {
    const rows = track.map(key => nodeByKey.get(key)!).sort((left, right) =>
      left.stage - right.stage
        || (order.get(left.key) ?? 0) - (order.get(right.key) ?? 0)
        || left.key.localeCompare(right.key),
    )
    const stageIndexes = [...new Set(rows.map(row => row.stage))].sort((left, right) => left - right)
    return {
      id: trackIndex + 1,
      independent: track.length === isolated.length && track.every(key => isolated.includes(key)),
      stages: stageIndexes.map(index => ({ index, nodes: rows.filter(row => row.stage === index) })),
      total: rows.reduce((sum, row) => sum + row.scopeTotal, 0),
      terminal: rows.reduce((sum, row) => sum + row.scopeTerminal, 0),
      active: rows.reduce((sum, row) => sum + row.scopeActive, 0),
      blocked: rows.reduce(
        (sum, row) => sum + row.scopeBlocked + (row.unresolvedBlockers.length && !row.scopeBlocked ? 1 : 0),
        0,
      ),
    }
  })

  return {
    nodes: tracks.flatMap(track => track.stages.flatMap(stage => stage.nodes)),
    edges,
    tracks,
    cycles: [...cyclicComponents]
      .map(index => components[index])
      .sort((left, right) => (order.get(left[0]) ?? 0) - (order.get(right[0]) ?? 0)),
    totalDependencyCount,
    visibleDependencyCount: edges.length,
    openDependencyCount,
    clearedDependencyCount,
    hiddenDependencyCount: [...hiddenDependencyCount.values()].reduce((sum, count) => sum + count, 0),
  }
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
  if (mode === 'dependencies') {
    const plan = buildIssueDependencyPlan(projection, filters)
    const external = new Set(projection.external.map(issue => issue.key))
    for (const node of plan.nodes) {
      const issue = node.issue
      graph.addNode(node.key, {
        label: issue.key,
        displayLabel: `${issue.key} · ${issue.title}`,
        title: issue.title,
        kind: issue.kind,
        status: issue.status,
        depth: (issue as TrackerGraphNode).depth ?? null,
        external: external.has(issue.key),
        dependencyStage: node.stage,
        dependencyTrack: node.track,
        cyclic: node.cyclic,
        hiddenScopeCount: node.hiddenScopeCount,
        size: issue.key === projection.root.key ? 13 : node.hiddenScopeCount ? 9 : external.has(issue.key) ? 5 : 7,
        color: nodeColor(issue, projection, external),
      })
    }
    for (const edge of plan.edges) {
      if (!graph.hasNode(edge.from) || !graph.hasNode(edge.to)) continue
      graph.addEdge(edge.from, edge.to, {
        kind: 'blocks',
        label: `${edge.cleared ? 'cleared ' : ''}blocks: ${edge.from} → ${edge.to}`,
        color: edge.cleared ? '#4b5563' : MAP_EDGE_COLORS.blocks,
        size: edge.cleared ? 1 : 2,
        cleared: edge.cleared,
        type: 'arrow',
      })
    }
    let trackOffset = 0
    for (const track of plan.tracks) {
      const maxRows = Math.max(1, ...track.stages.map(stage => stage.nodes.length))
      for (const stage of track.stages) {
        stage.nodes.forEach((node, index) => {
          graph.setNodeAttribute(node.key, 'x', node.stage * 5)
          graph.setNodeAttribute(node.key, 'y', -(trackOffset + index * 3))
        })
      }
      trackOffset += maxRows * 3 + 4
    }
    return graph
  }
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
