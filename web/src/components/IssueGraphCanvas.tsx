import { useEffect, useRef, useState } from 'react'
import { Focus, Minus, Plus } from 'lucide-react'
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
  const [hoveredKey, setHoveredKey] = useState<string | null>(null)
  const [labelMode, setLabelMode] = useState<'keys' | 'focus'>('focus')
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
        // Sigma's default hover painter draws a pale tooltip behind the label.
        // The dark, high-contrast focus card below is the single hover detail
        // surface for this graph, while the node reducer still highlights it.
        defaultDrawNodeHover: () => {},
      })
      sigma.on('clickNode', ({ node }) => onSelectRef.current(node))
      sigma.on('enterNode', ({ node }) => setHoveredKey(node))
      sigma.on('leaveNode', ({ node }) => setHoveredKey(current => current === node ? null : current))
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
    sigma.setSetting('nodeReducer', (node: string, data: Record<string, unknown>) => {
      const focused = node === selectedKey || node === hoveredKey
      return {
        ...data,
        label: focused
          ? data.displayLabel
          : labelMode === 'keys' ? data.label : '',
        size: focused ? ((data.size as number) ?? 7) * 1.5 : data.size,
        zIndex: focused ? 1 : 0,
      }
    })
    sigma.refresh()
  }, [selectedKey, hoveredKey, labelMode, projection])

  const focusKey = hoveredKey ?? selectedKey
  const focusIssue = focusKey
    ? [...projection.nodes, ...projection.external].find(issue => issue.key === focusKey) ?? null
    : null
  const moveCamera = (action: 'zoom' | 'unzoom' | 'reset') => {
    const camera = sigmaRef.current?.getCamera() as unknown as {
      animatedZoom?: (options?: { duration?: number }) => void
      animatedUnzoom?: (options?: { duration?: number }) => void
      animatedReset?: (options?: { duration?: number }) => void
      setState: (state: { ratio?: number; x?: number; y?: number; angle?: number }) => void
    } | undefined
    if (!camera) return
    if (action === 'zoom' && camera.animatedZoom) camera.animatedZoom({ duration: 180 })
    else if (action === 'unzoom' && camera.animatedUnzoom) camera.animatedUnzoom({ duration: 180 })
    else if (action === 'reset' && camera.animatedReset) camera.animatedReset({ duration: 220 })
    else camera.setState({ ratio: mode === 'hierarchy' ? 1.25 : 1.5, x: 0.5, y: 0.5, angle: 0 })
  }

  return (
    <div data-testid="issue-graph-canvas-wrap">
      <div className={`relative rounded-lg border border-gray-800 bg-gray-950/60 overflow-hidden ${failed ? 'hidden' : ''}`}>
        <div
          ref={containerRef}
          role="img"
          aria-label={`${mode === 'hierarchy' ? 'Issue hierarchy' : 'Issue relationship'} graph; the list below carries the same information`}
          data-testid="issue-graph-canvas"
          className="h-[460px] w-full"
        />
        <div className="absolute right-3 top-3 flex items-center gap-1 rounded border border-gray-800 bg-gray-950/90 p-1 shadow-lg">
          <button type="button" onClick={() => moveCamera('zoom')} aria-label="Zoom in" className="rounded p-1.5 text-gray-400 hover:bg-gray-800 hover:text-gray-100"><Plus size={13} /></button>
          <button type="button" onClick={() => moveCamera('unzoom')} aria-label="Zoom out" className="rounded p-1.5 text-gray-400 hover:bg-gray-800 hover:text-gray-100"><Minus size={13} /></button>
          <button type="button" onClick={() => moveCamera('reset')} aria-label="Fit graph" className="rounded p-1.5 text-gray-400 hover:bg-gray-800 hover:text-gray-100"><Focus size={13} /></button>
          <span className="mx-1 h-4 w-px bg-gray-800" />
          {(['keys', 'focus'] as const).map(value => (
            <button
              key={value}
              type="button"
              aria-pressed={labelMode === value}
              onClick={() => setLabelMode(value)}
              className={`rounded px-2 py-1 text-[10px] ${labelMode === value ? 'bg-emerald-600 text-white' : 'text-gray-500 hover:text-gray-200'}`}
            >
              {value === 'keys' ? 'Keys' : 'Focus only'}
            </button>
          ))}
        </div>
        {focusIssue && (
          <div className="pointer-events-none absolute bottom-3 left-3 max-w-[min(680px,calc(100%-24px))] rounded border border-gray-800 bg-gray-950/95 px-3 py-2 shadow-xl">
            <div className="truncate text-xs font-medium text-gray-100">{focusIssue.key} · {focusIssue.title}</div>
            <div className="mt-0.5 text-[10px] text-gray-500">{focusIssue.kind} · {focusIssue.status}</div>
          </div>
        )}
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
