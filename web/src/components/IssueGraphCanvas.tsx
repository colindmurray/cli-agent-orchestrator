import { useEffect, useRef, useState } from 'react'
import Sigma from 'sigma'
import type { TrackerGraphProjection } from '../api'
import {
  buildIssueGraph,
  ISSUE_GRAPH_NODE_COLORS,
  IssueGraphFilters,
  IssueGraphMode,
} from '../lib/issueGraph'
import { MAP_EDGE_COLORS } from '../lib/issueMap'

export function IssueGraphCanvas({
  projection,
  mode,
  filters,
  selectedKey,
  onSelect,
}: {
  projection: TrackerGraphProjection
  mode: IssueGraphMode
  filters: IssueGraphFilters
  selectedKey: string | null
  onSelect: (key: string) => void
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const sigmaRef = useRef<Sigma | null>(null)
  const onSelectRef = useRef(onSelect)
  const [failed, setFailed] = useState(false)
  onSelectRef.current = onSelect

  useEffect(() => {
    sigmaRef.current?.kill()
    sigmaRef.current = null
    if (!containerRef.current) return
    const graph = buildIssueGraph(projection, mode, filters)
    try {
      const sigma = new Sigma(graph, containerRef.current, {
        renderLabels: true,
        labelRenderedSizeThreshold: 4,
        labelColor: { color: '#d1d5db' },
        labelSize: 10,
      })
      sigma.on('clickNode', ({ node }) => onSelectRef.current(node))
      try { sigma.getCamera().setState({ ratio: mode === 'hierarchy' ? 1.25 : 1.5 }) } catch { /* test double */ }
      sigmaRef.current = sigma
      setFailed(false)
      return () => {
        sigma.kill()
        sigmaRef.current = null
      }
    } catch {
      setFailed(true)
    }
  }, [projection, mode, filters])

  useEffect(() => {
    const sigma = sigmaRef.current as unknown as
      | { setSetting: (key: string, value: unknown) => void; refresh: () => void }
      | null
    if (!sigma) return
    sigma.setSetting('nodeReducer', (node: string, data: Record<string, unknown>) =>
      node === selectedKey ? { ...data, size: ((data.size as number) ?? 7) * 1.5 } : data,
    )
    sigma.refresh()
  }, [selectedKey, projection])

  return (
    <div data-testid="issue-graph-canvas-wrap">
      <div className={`rounded-lg border border-gray-800 bg-gray-950/60 overflow-hidden ${failed ? 'hidden' : ''}`}>
        <div
          ref={containerRef}
          role="img"
          aria-label={`${mode === 'hierarchy' ? 'Issue hierarchy' : 'Issue relationship'} graph; the list below carries the same information`}
          data-testid="issue-graph-canvas"
          className="h-[460px] w-full"
        />
      </div>
      {failed && (
        <div role="note" className="rounded-lg border border-gray-800 bg-gray-950/60 px-4 py-3 text-xs text-gray-400">
          Graph rendering is unavailable in this browser. The structured list below carries the same issues and links.
        </div>
      )}
      <div data-testid="issue-graph-legend" className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-gray-400">
        {Object.entries(ISSUE_GRAPH_NODE_COLORS).map(([state, color]) => (
          <span key={state} className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
            {state}
          </span>
        ))}
        <span className="mx-1 h-4 w-px bg-gray-800" />
        {Object.entries(MAP_EDGE_COLORS)
          .filter(([kind]) => mode === 'relationships' || kind === 'part-of')
          .map(([kind, color]) => (
            <span key={kind} className="flex items-center gap-1.5">
              <span className="inline-block w-3.5 border-t-2" style={{ borderColor: color }} />
              {kind} →
            </span>
          ))}
      </div>
    </div>
  )
}
