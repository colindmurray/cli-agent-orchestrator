// Directed relationship graph for a tracker map (cond-0394).
//
// The graph is a READING aid, not a store: nodes come from the server's map
// projection, edges are the tracker's directed links rendered as arrows
// (blocks: blocker → blocked; part-of: child → map). The adjacent children
// list carries the same state as text, so the canvas is never the only way to
// read the map. WebGL-less environments fall back to the list with a note
// rather than a dead canvas.

import { useEffect, useRef, useState } from 'react'
import Sigma from 'sigma'
import type { TrackerMapProjection } from '../api'
import { buildMapGraph, MAP_EDGE_COLORS, MAP_NODE_COLORS } from '../lib/issueMap'

export function IssueMapGraph({
  projection,
  terminalStatuses,
  selectedKey,
  onSelect,
}: {
  projection: TrackerMapProjection
  terminalStatuses: string[]
  selectedKey: string | null
  onSelect: (key: string) => void
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const sigmaRef = useRef<Sigma | null>(null)
  const [failed, setFailed] = useState(false)
  // The click handler is bound once per projection; route it through a ref so
  // it always reads the current onSelect rather than a stale closure.
  const onSelectRef = useRef(onSelect)
  onSelectRef.current = onSelect

  useEffect(() => {
    if (sigmaRef.current) {
      sigmaRef.current.kill()
      sigmaRef.current = null
    }
    if (!containerRef.current) return
    const graph = buildMapGraph(projection, terminalStatuses)
    let sigma: Sigma
    try {
      sigma = new Sigma(graph, containerRef.current, {
        renderLabels: true,
        labelRenderedSizeThreshold: 5,
        // The dashboard is dark; sigma's default label color assumes light.
        labelColor: { color: '#d1d5db' },
        labelSize: 11,
      })
    } catch {
      // No WebGL (or a failed context): degrade to the list, say so plainly.
      setFailed(true)
      return
    }
    setFailed(false)
    sigma.on('clickNode', ({ node }) => onSelectRef.current(node))
    // Zoom out a touch so edge nodes and their labels fit inside the canvas.
    try {
      sigma.getCamera().setState({ ratio: 1.55 })
    } catch { /* mocked sigma in tests */ }
    sigmaRef.current = sigma
    return () => {
      sigma.kill()
      sigmaRef.current = null
    }
  }, [projection, terminalStatuses])

  // Selection emphasis without rebuilding the graph: the selected node grows.
  useEffect(() => {
    const sigma = sigmaRef.current as unknown as
      | { setSetting: (k: string, v: unknown) => void; refresh: () => void }
      | null
    if (!sigma) return
    sigma.setSetting('nodeReducer', (node: string, data: Record<string, unknown>) =>
      node === selectedKey ? { ...data, size: ((data.size as number) ?? 7) * 1.5 } : data,
    )
    sigma.refresh()
  }, [selectedKey, projection])

  return (
    <div data-testid="map-graph">
      <div
        className={`relative rounded-lg border border-gray-800 bg-gray-950/60 overflow-hidden ${failed ? 'hidden' : ''}`}
      >
        <div
          ref={containerRef}
          data-testid="map-graph-canvas"
          role="img"
          aria-label="Map relationship graph — the list below carries the same information"
          className="h-[380px] w-full"
        />
      </div>
      {failed && (
        <div role="note" className="rounded-lg border border-gray-800 bg-gray-950/60 px-4 py-3 text-xs text-gray-400">
          Graph rendering is unavailable in this browser — the list below carries the same nodes,
          states and relationships.
        </div>
      )}
      {/* Legend: every node state and every edge kind, with text, so meaning
          never depends on color alone. Arrows point in the link's direction:
          blocks goes blocker → blocked, part-of goes child → map. */}
      <div data-testid="map-graph-legend" className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[11px] text-gray-400">
        {Object.entries(MAP_NODE_COLORS).map(([state, color]) => (
          <span key={state} className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: color }} />
            {state}
          </span>
        ))}
        <span className="w-px h-4 bg-gray-800 mx-1" />
        {Object.entries(MAP_EDGE_COLORS).map(([kind, color]) => (
          <span key={kind} className="flex items-center gap-1.5">
            <span className="inline-block w-3.5 border-t-2" style={{ borderColor: color }} />
            {kind} →
          </span>
        ))}
      </div>
    </div>
  )
}
