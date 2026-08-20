import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, Focus, GitFork, Loader2, Network, Search, Workflow } from 'lucide-react'
import type {
  TrackerGraphProjection,
  TrackerIssue,
  TrackerVocabulary,
} from '../api'
import { api, errorText } from '../api'
import {
  buildIssueDependencyPlan,
  IssueGraphMode,
  orderIssueHierarchyNodes,
  visibleIssueGraphKeys,
} from '../lib/issueGraph'
import { linkPhrase } from '../lib/issueMap'
import { useStore } from '../store'
import { SearchableOption, SearchableSelect } from './SearchablePicker'
import { IssueGraphCanvas } from './IssueGraphCanvas'
import { IssueDependencyTracks } from './IssueDependencyTracks'

export function IssueGraphPanel({
  projectId,
  vocab,
  rootKey,
  onSelectRoot,
  selectedKey,
  onSelectIssue,
  refreshSignal,
}: {
  projectId: string
  vocab: TrackerVocabulary
  rootKey: string | null
  onSelectRoot: (key: string | null) => void
  selectedKey: string | null
  onSelectIssue: (key: string) => void
  refreshSignal: number
}) {
  const { showSnackbar } = useStore()
  const [projection, setProjection] = useState<TrackerGraphProjection | null>(null)
  const [projectRoots, setProjectRoots] = useState<TrackerIssue[]>([])
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState<IssueGraphMode>('hierarchy')
  const [query, setQuery] = useState('')
  const [kinds, setKinds] = useState<string[]>([])
  const [statuses, setStatuses] = useState<string[]>([])
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [dependencyCollapsed, setDependencyCollapsed] = useState<Set<string>>(new Set())
  const dependencyRootRef = useRef<string | null>(null)

  const loadIssueOptions = useCallback(async (needle: string): Promise<SearchableOption[]> => {
    const result = await api.listTrackerIssues({
      projectId,
      kind: 'all',
      q: needle.trim() || undefined,
      openOnly: false,
      limit: 16,
      order: 'severity',
    })
    return result.issues.map(issue => ({
      value: issue.key,
      label: `${issue.key} — ${issue.title}`,
      description: `${issue.kind} · ${issue.status}`,
    }))
  }, [projectId])

  useEffect(() => {
    let active = true
    Promise.all([
      api.listTrackerIssues({ projectId, kind: 'project', openOnly: false, limit: 20 }),
      rootKey ? Promise.resolve(null) : loadIssueOptions(''),
    ]).then(([projects, fallback]) => {
      if (!active) return
      setProjectRoots(projects.issues)
      if (!rootKey) {
        onSelectRoot(projects.issues[0]?.key ?? fallback?.[0]?.value ?? null)
      }
    }).catch(err => {
      if (active) showSnackbar({ type: 'error', message: `Could not load graph roots: ${errorText(err)}` })
    })
    return () => { active = false }
  }, [projectId, loadIssueOptions]) // root selection is intentionally one-time per project

  useEffect(() => {
    setCollapsed(new Set())
    setDependencyCollapsed(new Set())
    setQuery('')
    setKinds([])
    setStatuses([])
  }, [rootKey])

  useEffect(() => {
    if (!rootKey) { setProjection(null); return }
    let active = true
    setLoading(true)
    api.getTrackerGraph(rootKey)
      .then(result => {
        if (!active) return
        setProjection(result)
        if (dependencyRootRef.current !== result.root.key) {
          dependencyRootRef.current = result.root.key
          setDependencyCollapsed(new Set(
            result.nodes
              .filter(node => node.key !== result.root.key && node.child_count > 0)
              .map(node => node.key),
          ))
        }
      })
      .catch(err => {
        if (active) {
          setProjection(null)
          showSnackbar({ type: 'error', message: `Could not load issue graph: ${errorText(err)}` })
        }
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [rootKey, refreshSignal, showSnackbar])

  const activeCollapsed = mode === 'dependencies' ? dependencyCollapsed : collapsed
  const filters = useMemo(
    () => ({ query, kinds, statuses, collapsed: activeCollapsed }),
    [query, kinds, statuses, activeCollapsed],
  )
  const visible = useMemo(
    () => projection ? visibleIssueGraphKeys(projection, mode, filters) : new Set<string>(),
    [projection, mode, filters],
  )
  const dependencyPlan = useMemo(
    () => projection ? buildIssueDependencyPlan(projection, filters) : null,
    [projection, filters],
  )
  const selected = projection
    ? [...projection.nodes, ...projection.external].find(issue => issue.key === selectedKey) ?? null
    : null

  const toggle = (value: string, values: string[], setValues: (next: string[]) => void) => {
    setValues(values.includes(value) ? values.filter(item => item !== value) : [...values, value])
  }
  const toggleCollapsed = (key: string) => {
    const setter = mode === 'dependencies' ? setDependencyCollapsed : setCollapsed
    setter(current => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  return (
    <div className="mt-4 space-y-4" data-testid="issue-graph-view">
      <div className="rounded-lg border border-gray-800 bg-gray-950/40 p-4 space-y-3">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-end">
          <label className="min-w-0 flex-1">
            <span className="mb-1.5 block text-[11px] font-medium uppercase tracking-wide text-gray-500">Graph root</span>
            <SearchableSelect
              value={rootKey ?? ''}
              onChange={value => onSelectRoot(value || null)}
              loadOptions={loadIssueOptions}
              placeholder="Search issue key or title"
              ariaLabel="Issue graph root"
              emptyMessage="No matching issues"
            />
          </label>
          <div>
            <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-gray-500">View</div>
            <div className="flex gap-1" role="tablist" aria-label="Issue graph mode">
              {([
                { key: 'hierarchy', label: 'Hierarchy', icon: <GitFork size={12} /> },
                { key: 'dependencies', label: 'Dependencies', icon: <Workflow size={12} /> },
                { key: 'relationships', label: 'Relationships', icon: <Network size={12} /> },
              ] as const).map(item => (
                <button
                  key={item.key}
                  type="button"
                  role="tab"
                  aria-selected={mode === item.key}
                  onClick={() => setMode(item.key)}
                  className={`inline-flex items-center gap-1.5 rounded border px-3 py-2 text-xs ${
                    mode === item.key
                      ? 'border-emerald-500 bg-emerald-600 text-white'
                      : 'border-gray-800 text-gray-400 hover:border-gray-700'
                  }`}
                >
                  {item.icon} {item.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {projectRoots.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5" aria-label="Project issue roots">
            <span className="mr-1 text-[11px] text-gray-600">Project roots:</span>
            {projectRoots.map(issue => (
              <button
                key={issue.key}
                type="button"
                onClick={() => onSelectRoot(issue.key)}
                className={`rounded border px-2 py-1 text-[11px] ${
                  rootKey === issue.key
                    ? 'border-amber-500/60 bg-amber-500/10 text-amber-200'
                    : 'border-gray-800 text-gray-500 hover:text-gray-300'
                }`}
              >
                {issue.key} · {issue.title}
              </button>
            ))}
          </div>
        )}

        <div className="relative">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600" />
          <input
            value={query}
            onChange={event => setQuery(event.target.value)}
            aria-label="Filter graph issues"
            placeholder="Filter visible issues by key or title"
            className="w-full rounded border border-gray-800 bg-gray-950 py-2 pl-9 pr-3 text-sm text-gray-200 placeholder-gray-600 focus:border-emerald-600/60 focus:outline-none"
          />
        </div>
        <div className="grid gap-3 lg:grid-cols-2">
          <GraphFilter label="Type" values={vocab.item_kinds ?? []} selected={kinds} onToggle={value => toggle(value, kinds, setKinds)} />
          <GraphFilter label="Status" values={vocab.statuses} selected={statuses} onToggle={value => toggle(value, statuses, setStatuses)} />
        </div>
      </div>

      {!rootKey && (
        <div className="rounded-lg border border-dashed border-gray-800 px-6 py-10 text-center text-sm text-gray-500">
          Choose any issue as the graph root.
        </div>
      )}
      {loading && (
        <div className="flex items-center justify-center gap-2 rounded-lg border border-gray-800 py-16 text-sm text-gray-500">
          <Loader2 size={14} className="animate-spin" /> Loading issue graph…
        </div>
      )}
      {!loading && projection && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-gray-800 bg-gray-950/30 px-4 py-3">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-gray-100">{projection.root.key} · {projection.root.title}</div>
              <div className="mt-0.5 text-[11px] text-gray-500">
                {projection.stats.descendants} descendants · {projection.stats.external} related · {projection.stats.links} links · depth {projection.stats.depth}
              </div>
            </div>
            {selected && selected.key !== rootKey && (
              <button
                type="button"
                onClick={() => onSelectRoot(selected.key)}
                className="inline-flex items-center gap-1.5 rounded border border-gray-700 px-2.5 py-1.5 text-xs text-gray-300 hover:border-emerald-600/60"
              >
                <Focus size={12} /> Focus {selected.key}
              </button>
            )}
          </div>
          {projection.bounds.truncated && (
            <div role="note" className="rounded border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              This graph hit its safety bound ({projection.bounds.reasons.join(', ')}). Focus a child to continue from there.
            </div>
          )}
          <IssueGraphCanvas
            projection={projection}
            mode={mode}
            filters={filters}
            selectedKey={selectedKey}
            onSelect={onSelectIssue}
          />
          {mode === 'hierarchy' ? (
            <div className="rounded-lg border border-gray-800 overflow-hidden" aria-label="Issue hierarchy">
              {orderIssueHierarchyNodes(projection, visible).map(node => (
                <div key={node.key} className="flex items-center gap-2 border-b border-gray-800/70 px-3 py-2 last:border-b-0" style={{ paddingLeft: `${12 + node.depth * 22}px` }}>
                  {node.child_count > 0 ? (
                    <button
                      type="button"
                      onClick={() => toggleCollapsed(node.key)}
                      aria-label={`${collapsed.has(node.key) ? 'Expand' : 'Collapse'} ${node.key}`}
                      className="text-gray-500 hover:text-gray-200"
                    >
                      {collapsed.has(node.key) ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                    </button>
                  ) : <span className="w-[13px]" />}
                  <button type="button" onClick={() => onSelectIssue(node.key)} className="min-w-0 flex-1 text-left">
                    <span className="font-mono text-[11px] text-gray-500">{node.key}</span>
                    <span className="ml-2 text-xs text-gray-200">{node.title}</span>
                  </button>
                  <span className="text-[10px] text-gray-600">{node.kind} · {node.status}</span>
                </div>
              ))}
            </div>
          ) : mode === 'dependencies' && dependencyPlan ? (
            <IssueDependencyTracks
              plan={dependencyPlan}
              collapsed={dependencyCollapsed}
              onToggleScope={toggleCollapsed}
              onSelectIssue={onSelectIssue}
            />
          ) : (
            <div className="rounded-lg border border-gray-800 overflow-hidden" aria-label="Issue relationships">
              {projection.links.filter(link => visible.has(link.from_key) && visible.has(link.to_key)).map(link => {
                const phrase = linkPhrase(link, link.from_key)
                return (
                  <div key={link.id} className="flex flex-wrap items-center gap-2 border-b border-gray-800/70 px-3 py-2 text-xs last:border-b-0">
                    <button type="button" onClick={() => onSelectIssue(link.from_key)} className="font-mono text-gray-300 hover:text-emerald-300">{link.from_key}</button>
                    <span className="text-gray-500">{phrase.phrase}</span>
                    <button type="button" onClick={() => onSelectIssue(link.to_key)} className="font-mono text-gray-300 hover:text-emerald-300">{link.to_key}</button>
                  </div>
                )
              })}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function GraphFilter({
  label,
  values,
  selected,
  onToggle,
}: {
  label: string
  values: string[]
  selected: string[]
  onToggle: (value: string) => void
}) {
  return (
    <div>
      <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-gray-600">{label}</div>
      <div className="flex flex-wrap gap-1">
        {values.map(value => (
          <button
            key={value}
            type="button"
            aria-pressed={selected.includes(value)}
            onClick={() => onToggle(value)}
            className={`rounded border px-2 py-0.5 text-[10px] ${
              selected.includes(value)
                ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300'
                : 'border-gray-800 text-gray-500 hover:text-gray-300'
            }`}
          >
            {value}
          </button>
        ))}
      </div>
    </div>
  )
}
