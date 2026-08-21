import { useEffect, useRef, useState } from 'react'
import { Focus, Minus, Plus } from 'lucide-react'
import Sigma from 'sigma'
import type { TrackerGraphProjection } from '../api'
import {
  buildIssueGraph,
  ISSUE_GRAPH_NODE_COLORS,
  IssueGraphFilters,
  IssueGraphMode,
  IssueGraphNodeState,
  IssueGraphVisibility,
} from '../lib/issueGraph'
import { MAP_EDGE_COLORS } from '../lib/issueMap'

export function IssueGraphCanvas({
  projection,
  mode,
  filters,
  visibility,
  selectedKey,
  onSelect,
  onToggleNodeState,
  onToggleEdgeKind,
  onToggleUnconnected,
}: {
  projection: TrackerGraphProjection
  mode: IssueGraphMode
  filters: IssueGraphFilters
  visibility: IssueGraphVisibility
  selectedKey: string | null
  onSelect: (key: string) => void
  onToggleNodeState: (state: IssueGraphNodeState) => void
  onToggleEdgeKind: (kind: string) => void
  onToggleUnconnected: () => void
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
    const graph = buildIssueGraph(projection, mode, filters, visibility)
    try {
      const sigma = new Sigma(graph, containerRef.current, {
        renderLabels: true,
        labelRenderedSizeThreshold: 4,
        labelColor: { color: '#d1d5db' },
        labelSize: 10,
        zIndex: true,
        // Sigma's default hover painter draws a pale tooltip behind the label.
        // The dark, high-contrast focus card below is the single hover detail
        // surface for this graph, while the node reducer still highlights it.
        defaultDrawNodeHover: () => {},
      })
      sigma.on('clickNode', ({ node }) => onSelectRef.current(node))
      sigma.on('enterNode', ({ node }) => setHoveredKey(node))
      sigma.on('leaveNode', ({ node }) => setHoveredKey(current => current === node ? null : current))
      try { sigma.getCamera().setState({ ratio: mode === 'relationships' ? 1.5 : 1.25 }) } catch { /* test double */ }
      sigmaRef.current = sigma
      setFailed(false)
      return () => {
        sigma.kill()
        sigmaRef.current = null
      }
    } catch {
      setFailed(true)
    }
  }, [projection, mode, filters, visibility])

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
    else camera.setState({ ratio: mode === 'relationships' ? 1.5 : 1.25, x: 0.5, y: 0.5, angle: 0 })
  }

  return (
    <div data-testid="issue-graph-canvas-wrap">
      <div className={`relative rounded-lg border border-gray-800 bg-gray-950/60 overflow-hidden ${failed ? 'hidden' : ''}`}>
        <div
          ref={containerRef}
          role="img"
          aria-label={`${mode === 'hierarchy' ? 'Issue hierarchy' : mode === 'dependencies' ? 'Issue dependency DAG' : 'Issue relationship'} graph; the structured view below carries the same information`}
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
      <div data-testid="issue-graph-legend" aria-label="Graph visibility" className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1.5 text-[11px] text-gray-400">
        {Object.entries(ISSUE_GRAPH_NODE_COLORS).map(([state, color]) => (
          <button
            key={state}
            type="button"
            aria-pressed={!visibility.hiddenNodeStates.has(state as IssueGraphNodeState)}
            aria-label={`${visibility.hiddenNodeStates.has(state as IssueGraphNodeState) ? 'Show' : 'Hide'} ${state} nodes`}
            onClick={() => onToggleNodeState(state as IssueGraphNodeState)}
            className={`flex items-center gap-1.5 rounded border px-2 py-1 transition ${
              visibility.hiddenNodeStates.has(state as IssueGraphNodeState)
                ? 'border-gray-900 bg-gray-950 text-gray-700 line-through'
                : 'border-gray-800 bg-gray-950/50 text-gray-400 hover:border-gray-700 hover:text-gray-200'
            }`}
          >
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${state === 'active' ? 'motion-safe:animate-pulse ring-2 ring-green-400/30' : ''}`}
              style={{ background: color }}
            />
            {state}
          </button>
        ))}
        <span className="mx-1 h-4 w-px bg-gray-800" />
        <button
          type="button"
          aria-pressed={visibility.hideUnconnected}
          aria-label="Hide unconnected nodes"
          onClick={onToggleUnconnected}
          className={`flex items-center gap-1.5 rounded border px-2 py-1 transition ${
            visibility.hideUnconnected
              ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300'
              : 'border-gray-800 bg-gray-950/50 text-gray-400 hover:border-gray-700 hover:text-gray-200'
          }`}
        >
          <span className="relative inline-block h-3 w-3">
            <span className="absolute left-0 top-1 h-1.5 w-1.5 rounded-full border border-current" />
            <span className="absolute bottom-0 right-0 h-1.5 w-1.5 rounded-full border border-current" />
          </span>
          hide unconnected
        </button>
        <span className="mx-1 h-4 w-px bg-gray-800" />
        {Object.entries(MAP_EDGE_COLORS).map(([kind, color]) => (
          <button
            key={kind}
            type="button"
            aria-pressed={!visibility.hiddenEdgeKinds.has(kind)}
            aria-label={`${visibility.hiddenEdgeKinds.has(kind) ? 'Show' : 'Hide'} ${kind} edges`}
            onClick={() => onToggleEdgeKind(kind)}
            className={`flex items-center gap-1.5 rounded border px-2 py-1 transition ${
              visibility.hiddenEdgeKinds.has(kind)
                ? 'border-gray-900 bg-gray-950 text-gray-700 line-through'
                : 'border-gray-800 bg-gray-950/50 text-gray-400 hover:border-gray-700 hover:text-gray-200'
            }`}
          >
            <span className="inline-block w-3.5 border-t-2" style={{ borderColor: color }} />
            {kind} →
          </button>
        ))}
        {mode === 'hierarchy' && (
          <span className="ml-auto text-[10px] text-gray-600">scope tree · relationship context lanes</span>
        )}
      </div>
    </div>
  )
}
