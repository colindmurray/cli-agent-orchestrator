// Shared, pure derivations for the tracker map view (cond-0394).
//
// Everything here is a pure function over the server's map projection: node
// state classification, edge colors, directional link phrasing, and the
// graphology graph itself. Keeping it out of the component files means the
// graph, the legend and the accessible list can never disagree about what a
// color or a phrase means — and jsdom tests can assert the graph's shape
// without a WebGL context.

import Graph from 'graphology'
import { circular } from 'graphology-layout'
import type { TrackerLink, TrackerMapChild, TrackerMapProjection } from '../api'

export type MapNodeState =
  | 'map'
  | 'frontier'
  | 'blocked'
  | 'claimed'
  | 'resolved'
  | 'terminal'
  | 'open'
  | 'external'

/** Node colors by state. Every state also appears in the legend and in the
 * adjacent list as text, so color is never the only carrier of meaning. */
export const MAP_NODE_COLORS: Record<MapNodeState, string> = {
  map: '#f9fafb',
  frontier: '#34d399',
  blocked: '#f87171',
  claimed: '#60a5fa',
  resolved: '#2dd4bf',
  terminal: '#6b7280',
  open: '#38bdf8',
  external: '#fb923c',
}

export const MAP_EDGE_COLORS: Record<string, string> = {
  'part-of': '#10b981',
  blocks: '#f87171',
  relates: '#64748b',
  duplicates: '#a78bfa',
  'caused-by': '#f59e0b',
}

/** The one classification rule for a child node, on top of the server's own
 * `frontier` flag (which already encodes nonterminal + unassigned + unblocked).
 * Precedence: terminal > frontier > blocked > claimed > resolved > open —
 * "cannot be taken" outranks "someone is on it", because that is the fact a
 * map reader acts on. */
export function childState(
  child: Pick<TrackerMapChild, 'status' | 'assignee'> & Partial<TrackerMapChild>,
  terminalStatuses: string[],
): MapNodeState {
  if (terminalStatuses.includes(child.status)) return 'terminal'
  if (child.frontier) return 'frontier'
  if ((child.blocked_by ?? []).length > 0) return 'blocked'
  if (child.assignee) return 'claimed'
  if (child.status === 'resolved') return 'resolved'
  return 'open'
}

// Directional phrasing for a link seen from one of its endpoints. JSON stays
// explicit (from_key/to_key/kind); these phrases are for humans.
export const LINK_PHRASE_OUT: Record<string, string> = {
  blocks: 'blocks',
  'part-of': 'part of',
  relates: 'relates to',
  duplicates: 'duplicates',
  'caused-by': 'caused by',
}
export const LINK_PHRASE_IN: Record<string, string> = {
  blocks: 'blocked by',
  'part-of': 'contains',
  relates: 'relates to',
  duplicates: 'duplicated by',
  'caused-by': 'caused',
}

export function linkPhrase(
  link: Pick<TrackerLink, 'kind' | 'from_key' | 'to_key'>,
  thisKey: string,
): { phrase: string; other: string } {
  if (link.from_key === thisKey) {
    return { phrase: LINK_PHRASE_OUT[link.kind] ?? link.kind, other: link.to_key }
  }
  return { phrase: LINK_PHRASE_IN[link.kind] ?? `${link.kind} (incoming)`, other: link.from_key }
}

/** When two edges share a pair, render the more load-bearing one; the full
 * link set is still listed textually beside the graph. */
const EDGE_PRIORITY: Record<string, number> = {
  blocks: 5,
  'part-of': 4,
  'caused-by': 3,
  duplicates: 2,
  relates: 1,
}

/**
 * Build the map graph: the map as the hub, its children classified by state,
 * every external link endpoint (included by the projection so no returned
 * link is left dangling — blockers and context alike), and every link as a
 * directed, colored edge. `circular` gives deterministic positions — the same
 * projection renders the same layout, which is what makes screenshots
 * reviewable.
 */
export function buildMapGraph(
  projection: TrackerMapProjection,
  terminalStatuses: string[],
): Graph {
  const graph = new Graph()
  const map = projection.map
  graph.addNode(map.key, {
    label: map.key,
    title: map.title,
    size: 13,
    color: MAP_NODE_COLORS.map,
    state: 'map',
  })
  const present = new Set([map.key])
  for (const child of projection.children) {
    const state = childState(child, terminalStatuses)
    graph.addNode(child.key, {
      label: child.key,
      title: child.title,
      size: state === 'frontier' ? 9 : 7,
      color: MAP_NODE_COLORS[state],
      state,
    })
    present.add(child.key)
  }
  for (const ext of projection.external) {
    const state: MapNodeState = terminalStatuses.includes(ext.status) ? 'terminal' : 'external'
    graph.addNode(ext.key, {
      label: ext.key,
      title: ext.title,
      // An actual external blocker (blocking non-empty) draws a touch larger —
      // it holds a ticket back; a context neighbour is just orientation.
      size: ext.blocking.length > 0 ? 6 : 5,
      color: MAP_NODE_COLORS[state],
      state,
      blocking: ext.blocking,
    })
    present.add(ext.key)
  }
  // Keep at most one rendered edge per directed pair — the highest-priority
  // kind. A pair carrying both part-of and blocks still shows its blocks edge.
  const best = new Map<string, TrackerLink>()
  for (const link of projection.links) {
    if (!present.has(link.from_key) || !present.has(link.to_key)) continue
    const pairKey = JSON.stringify([link.from_key, link.to_key])
    const existing = best.get(pairKey)
    if (!existing || (EDGE_PRIORITY[link.kind] ?? 0) > (EDGE_PRIORITY[existing.kind] ?? 0)) {
      best.set(pairKey, link)
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
  circular.assign(graph)
  return graph
}
