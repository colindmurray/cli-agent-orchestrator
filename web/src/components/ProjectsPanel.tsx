import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import {
  api, ApiError, errorText, conflictDetail, TrackerProject, TrackerIssue, TrackerIssuePage,
  TrackerVocabulary, TrackerScope, TrackerOptionField, TrackerIssueBrief, TrackerProjectHome,
  TrackerProjectSessions, TrackerProjectSessionDetail, TrackerProjectSessionSummary,
} from '../api'
import { linkPhrase } from '../lib/issueMap'
import { useStore } from '../store'
import { ConfirmModal } from './ConfirmModal'
import { WayfinderPanel } from './WayfinderPanel'
import { SearchableMultiSelect, SearchableOption, SearchableSelect } from './SearchablePicker'
import {
  FolderGit2, Plus, Search, Trash2, X, Loader2, Archive, ChevronRight, MessageSquare,
  History, Link2, Save, FileDown, CircleDot, CheckCircle2, Lightbulb, Compass, List,
  SlidersHorizontal, ChevronDown, Star, Home, Activity, Users, GitBranch, TerminalSquare,
  ExternalLink, Clock3, Box,
} from 'lucide-react'

/**
 * Projects + issue tracker.
 *
 * A project here is a declared grouping — one name over many directories,
 * sessions and git remotes — so the header leads with its SCOPES rather than
 * with a single path. Getting a scope wrong is how issues end up filed in the
 * wrong log, and the only way to notice is to see the whole set at once.
 *
 * Every enumeration (statuses, severities, scope kinds, link kinds) is fetched
 * from /tracker/vocabulary rather than hard-coded, so a dropdown can never
 * offer a value the server will reject.
 */

const SEVERITY_CLASS: Record<string, string> = {
  P0: 'bg-red-500/15 text-red-300 border-red-500/30',
  P1: 'bg-orange-500/15 text-orange-300 border-orange-500/30',
  P2: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  P3: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
  P4: 'bg-slate-500/15 text-slate-300 border-slate-500/30',
  unset: 'bg-gray-700/40 text-gray-400 border-gray-600/40',
}

const STATUS_CLASS: Record<string, string> = {
  open: 'bg-emerald-500/15 text-emerald-300',
  triage: 'bg-violet-500/15 text-violet-300',
  'in-progress': 'bg-blue-500/15 text-blue-300',
  blocked: 'bg-red-500/15 text-red-300',
  resolved: 'bg-teal-500/15 text-teal-300',
  closed: 'bg-gray-600/30 text-gray-400',
  wontfix: 'bg-gray-600/30 text-gray-400',
  duplicate: 'bg-gray-600/30 text-gray-400',
}

const KIND_CLASS: Record<string, string> = {
  issue: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
  bug: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
  feature: 'bg-violet-500/15 text-violet-300 border-violet-500/30',
  project: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  milestone: 'bg-orange-500/15 text-orange-300 border-orange-500/30',
  goal: 'bg-teal-500/15 text-teal-300 border-teal-500/30',
  epic: 'bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/30',
  story: 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30',
  task: 'bg-gray-500/15 text-gray-300 border-gray-500/30',
}

const PAGE_SIZE = 50

const FEATURE_BODY_STARTER = `## Problem / opportunity

## Desired outcome

## Acceptance criteria

## Constraints / alternatives
`

function shortDate(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().slice(0, 10)
}

function Pill({ text, className }: { text: string; className: string }) {
  return <span className={`px-1.5 py-0.5 rounded text-[11px] font-medium border border-transparent ${className}`}>{text}</span>
}

// ---------------------------------------------------------------------------
// Kind presentation descriptor — one vocabulary for bugs and planning items.
// Relationships stay permissive: these types describe intent, not a mandatory
// hierarchy.

const ITEM_KINDS = ['project', 'bug', 'feature', 'milestone', 'goal', 'epic', 'story', 'task'] as const
type ItemKind = typeof ITEM_KINDS[number]
type TabKind = ItemKind | 'all'
type ProjectTab = 'home' | 'issues' | 'sessions'

const KIND_LABEL: Record<ItemKind, string> = {
  project: 'Project',
  bug: 'Bug',
  feature: 'Feature',
  milestone: 'Milestone',
  goal: 'Goal',
  epic: 'Epic',
  story: 'Story',
  task: 'Task',
}

interface KindPresentation {
  kind: ItemKind
  createButtonLabel: string
  modalTitle: (projectName: string) => string
  createActionLabel: string
  severityLabel: string
  reporterLabel: string
  assigneeLabel: string
  resolutionLabel: string
  bodyLabel: string
  bodyStarter: string
  showFailingCommandInCreate: boolean
  closingOptions: Array<{ uiLabel: string; status: string; needsDuplicateOf?: boolean }>
}

function planningPresentation(kind: ItemKind): KindPresentation {
  const label = KIND_LABEL[kind]
  return {
    kind,
    createButtonLabel: `Add ${label.toLowerCase()}`,
    modalTitle: (name: string) => `Add a ${label.toLowerCase()} to ${name}`,
    createActionLabel: `Create ${label.toLowerCase()}`,
    severityLabel: kind === 'bug' ? 'Severity' : 'Priority',
    reporterLabel: kind === 'feature' ? 'Requester' : 'Reporter',
    assigneeLabel: kind === 'feature' ? 'Owner' : 'Assignee',
    resolutionLabel: kind === 'feature' ? 'Outcome' : 'Resolution',
    bodyLabel: kind === 'feature' ? 'Proposal' : 'Description',
    bodyStarter: kind === 'feature' ? FEATURE_BODY_STARTER : '',
    showFailingCommandInCreate: kind === 'bug',
    closingOptions: kind === 'feature' ? [
      { uiLabel: 'Shipped', status: 'closed' },
      { uiLabel: 'Declined', status: 'wontfix' },
      { uiLabel: 'Withdrawn', status: 'wontfix' },
      { uiLabel: 'Duplicate', status: 'duplicate', needsDuplicateOf: true },
    ] : [],
  }
}

const KIND_PRESENTATION: Record<ItemKind, KindPresentation> = Object.fromEntries(
  ITEM_KINDS.map(kind => [kind, planningPresentation(kind)]),
) as Record<ItemKind, KindPresentation>

KIND_PRESENTATION.bug = {
    kind: 'bug',
    createButtonLabel: 'Log bug',
    modalTitle: (name: string) => `Log a bug against ${name}`,
    createActionLabel: 'File bug',
    severityLabel: 'Severity',
    reporterLabel: 'Reporter',
    assigneeLabel: 'Assignee',
    resolutionLabel: 'Resolution',
    bodyLabel: 'What happened',
    bodyStarter: '',
    showFailingCommandInCreate: true,
    closingOptions: [],
}

function presentationFor(kind: string | undefined): KindPresentation {
  const canonical = kind === 'issue' ? 'bug' : kind
  return KIND_PRESENTATION[ITEM_KINDS.includes(canonical as ItemKind) ? canonical as ItemKind : 'bug']
}

// ---------------------------------------------------------------------------
// Per-tab filter state — search/open-only/filters/pagination are independent
// per tab as D6 requires. Changing tab does not leak the previous tab's query.

interface TabFilters {
  query: string
  statusFilter: string[]
  severityFilter: string[]
  openOnly: boolean
  /** Repeated exact labels compose as AND on the server. */
  labels: string[]
  component: string
  assignee: string
  reporter: string
  unlabeled: boolean
  offset: number
}

function defaultTabFilters(): TabFilters {
  return {
    query: '',
    statusFilter: [],
    severityFilter: [],
    openOnly: true,
    labels: [],
    component: '',
    assignee: '',
    reporter: '',
    unlabeled: false,
    offset: 0,
  }
}

// ---------------------------------------------------------------------------

export function ProjectsPanel() {
  const { showSnackbar } = useStore()

  const [vocab, setVocab] = useState<TrackerVocabulary | null>(null)
  const [projects, setProjects] = useState<TrackerProject[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [project, setProject] = useState<TrackerProject | null>(null)
  const [loading, setLoading] = useState(true)

  const [page, setPage] = useState<TrackerIssuePage | null>(null)
  const [issuesLoading, setIssuesLoading] = useState(false)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [projectTab, setProjectTab] = useState<ProjectTab>('home')
  const [homeDashboard, setHomeDashboard] = useState<TrackerProjectHome | null>(null)
  const [sessionDashboard, setSessionDashboard] = useState<TrackerProjectSessions | null>(null)
  const [dashboardLoading, setDashboardLoading] = useState(false)

  const [kind, setKind] = useState<TabKind>('bug')
  const [filtersByKind, setFiltersByKind] = useState<Record<TabKind, TabFilters>>(
    () => Object.fromEntries(
      [...ITEM_KINDS, 'all'].map(itemKind => [itemKind, defaultTabFilters()]),
    ) as Record<TabKind, TabFilters>,
  )
  // List ⇄ Wayfinder view and the open map.
  const [view, setView] = useState<'list' | 'wayfinder'>('list')
  const [mapKey, setMapKey] = useState<string | null>(null)
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false)
  // Bumped whenever tracker state changes so the map projection re-reads.
  const [trackerVersion, setTrackerVersion] = useState(0)

  const currentFilters = filtersByKind[kind]

  const updateCurrentFilters = useCallback((patch: Partial<TabFilters>) => {
    setFiltersByKind(prev => ({ ...prev, [kind]: { ...prev[kind], ...patch } }))
  }, [kind])

  // URL state: ?project=cao-system&kind=feature&key=cond-0342 plus
  // &view=wayfinder&map=cond-0001&label=effort:x&unlabeled=1 — shareable and
  // Back/Forward traverses it.
  const urlSyncRef = useRef(false)

  // Initialize from URL on first load
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search)
      const urlProject = params.get('project')
      const urlKind = params.get('kind') as TabKind | null
      const urlKey = params.get('key')
      const urlSection = params.get('section') as ProjectTab | null
      const urlView = params.get('view')
      const urlMap = params.get('map')
      const urlLabels = params.getAll('label')
      const urlStatuses = params.getAll('status')
      const urlSeverities = params.getAll('severity')
      const urlUnlabeled = params.get('unlabeled') === '1'
      const tab: TabKind =
        urlKind && ([...ITEM_KINDS, 'all'] as TabKind[]).includes(urlKind) ? urlKind : 'bug'
      if (urlKind && ([...ITEM_KINDS, 'all'] as TabKind[]).includes(urlKind)) {
        setKind(urlKind)
      }
      if (urlKey) setSelectedKey(urlKey)
      if (urlKey || urlView === 'wayfinder') setProjectTab('issues')
      else if (urlSection && ['home', 'issues', 'sessions'].includes(urlSection)) setProjectTab(urlSection)
      if (urlView === 'wayfinder') setView('wayfinder')
      if (urlMap) setMapKey(urlMap)
      setFiltersByKind(prev => ({
        ...prev,
        [tab]: {
          ...prev[tab],
          labels: urlLabels,
          statusFilter: urlStatuses,
          severityFilter: urlSeverities,
          component: params.get('component') ?? '',
          assignee: params.get('assignee') ?? '',
          reporter: params.get('reporter') ?? '',
          unlabeled: urlUnlabeled,
          openOnly: params.get('open') !== '0',
          offset: 0,
        },
      }))
      // Defer project override until after projects load; store intent
      if (urlProject) {
        ;(window as unknown as { __caoInitialProject?: string }).__caoInitialProject = urlProject
      }
    } catch { /* non-browser test env */ }
  }, [])

  const [showNewProject, setShowNewProject] = useState(false)
  const [showNewIssue, setShowNewIssue] = useState(false)
  const [showScopes, setShowScopes] = useState(false)
  const [pendingArchive, setPendingArchive] = useState<TrackerProject | null>(null)

  const loadProjects = useCallback(async (preferId?: string) => {
    try {
      const rows = await api.listTrackerProjects(true)
      setProjects(rows)
      // Honor URL project if present
      const urlInitial = (window as unknown as { __caoInitialProject?: string }).__caoInitialProject
      const desired = preferId ?? urlInitial ?? null
      if (desired && rows.some(r => r.id === desired)) {
        setActiveId(desired)
        delete (window as unknown as { __caoInitialProject?: string }).__caoInitialProject
      } else {
        setActiveId(current => preferId ?? current ?? (rows.length ? rows[0].id : null))
      }
    } catch (err) {
      showSnackbar({ type: 'error', message: `Could not load projects: ${errorText(err)}` })
    } finally {
      setLoading(false)
    }
  }, [showSnackbar])

  useEffect(() => {
    api.getTrackerVocabulary().then(setVocab).catch(() => {})
    loadProjects()
  }, [loadProjects])

  useEffect(() => {
    if (!activeId) { setProject(null); return }
    api.getTrackerProject(activeId).then(setProject).catch(() => setProject(null))
  }, [activeId])

  const loadProjectDashboards = useCallback(async () => {
    if (!activeId) {
      setHomeDashboard(null)
      setSessionDashboard(null)
      return
    }
    setDashboardLoading(true)
    try {
      const [home, sessions] = await Promise.all([
        api.getTrackerProjectHome(activeId),
        api.getTrackerProjectSessions(activeId),
      ])
      if (!home?.issues || !home?.sessions || !Array.isArray(sessions?.sessions)) {
        throw new Error('project dashboard response is incomplete')
      }
      setHomeDashboard(home)
      setSessionDashboard(sessions)
    } catch (err) {
      showSnackbar({ type: 'error', message: `Could not load project activity: ${errorText(err)}` })
    } finally {
      setDashboardLoading(false)
    }
  }, [activeId, showSnackbar])

  useEffect(() => { loadProjectDashboards() }, [loadProjectDashboards])

  const loadIssues = useCallback(async () => {
    if (!activeId) { setPage(null); return }
    setIssuesLoading(true)
    const f = filtersByKind[kind]
    const shared = {
      projectId: activeId,
      q: f.query.trim() || undefined,
      status: f.statusFilter.length ? f.statusFilter : undefined,
      severity: f.severityFilter.length ? f.severityFilter : undefined,
      label: f.labels.length ? f.labels : undefined,
      component: f.component || undefined,
      assignee: f.assignee || undefined,
      reporter: f.reporter || undefined,
      unlabeled: f.unlabeled || undefined,
      openOnly: f.openOnly,
      limit: PAGE_SIZE,
      offset: f.offset,
      order: 'severity',
    }
    try {
      const result = await api.listTrackerIssues({ ...shared, kind })
      setPage(result)
    } catch (err) {
      showSnackbar({ type: 'error', message: `Could not load issues: ${errorText(err)}` })
      setPage(null)
    } finally {
      setIssuesLoading(false)
    }
  }, [activeId, filtersByKind, kind, showSnackbar])

  useEffect(() => { loadIssues() }, [loadIssues])

  // Sync URL on project/kind/key/view/map/label/unlabeled changes — pushState
  // so Back/Forward traverses history. `kind` is written only when it differs
  // from the default: the bare entry every session starts from must
  // round-trip byte-identically, or restoring it would push a duplicate.
  useEffect(() => {
    if (!urlSyncRef.current) {
      // Skip first render until projects have loaded, to avoid clobbering incoming URL
      urlSyncRef.current = true
      return
    }
    try {
      const params = new URLSearchParams(window.location.search)
      if (activeId) params.set('project', activeId)
      else params.delete('project')
      if (projectTab !== 'home') params.set('section', projectTab)
      else params.delete('section')
      if (kind !== 'bug') params.set('kind', kind)
      else params.delete('kind')
      if (selectedKey) params.set('key', selectedKey)
      else params.delete('key')
      if (view === 'wayfinder') params.set('view', 'wayfinder')
      else params.delete('view')
      if (view === 'wayfinder' && mapKey) params.set('map', mapKey)
      else params.delete('map')
      params.delete('label')
      for (const label of currentFilters.labels) params.append('label', label)
      params.delete('status')
      for (const status of currentFilters.statusFilter) params.append('status', status)
      params.delete('severity')
      for (const severity of currentFilters.severityFilter) params.append('severity', severity)
      for (const [name, value] of [
        ['component', currentFilters.component],
        ['assignee', currentFilters.assignee],
        ['reporter', currentFilters.reporter],
      ] as const) {
        if (value) params.set(name, value)
        else params.delete(name)
      }
      if (currentFilters.openOnly) params.delete('open')
      else params.set('open', '0')
      if (currentFilters.unlabeled) params.set('unlabeled', '1')
      else params.delete('unlabeled')
      const newSearch = params.toString()
      const newUrl = `${window.location.pathname}${newSearch ? '?' + newSearch : ''}`
      // Never push the URL the browser is already on: a popstate restore runs
      // this effect too, and re-pushing the restored entry would duplicate it
      // and truncate the forward stack.
      if (newUrl === window.location.pathname + window.location.search) return
      window.history.pushState(null, '', newUrl)
    } catch { /* test env */ }
  }, [activeId, projectTab, kind, selectedKey, view, mapKey, currentFilters])

  // Cleanup stale key on unmount so next test starts collapsed
  useEffect(() => {
    return () => {
      try {
        const params = new URLSearchParams(window.location.search)
        params.delete('key')
        const newSearch = params.toString()
        window.history.replaceState(null, '', `${window.location.pathname}${newSearch ? '?' + newSearch : ''}`)
        delete (window as unknown as { __caoInitialProject?: string }).__caoInitialProject
      } catch { /* test env */ }
    }
  }, [])

  // Handle back/forward: the URL is the whole state, so restore every param —
  // clearing the ones the entry does not carry, project and kind included.
  // (The sync effect then computes the URL the browser is already on and
  // pushes nothing, which is what keeps traversal from duplicating entries.)
  useEffect(() => {
    const handler = () => {
      try {
        const params = new URLSearchParams(window.location.search)
        const urlProject = params.get('project')
        const urlKind = params.get('kind') as TabKind | null
        const urlKey = params.get('key')
        const urlSection = params.get('section') as ProjectTab | null
        const urlView = params.get('view')
        const urlMap = params.get('map')
        const urlLabels = params.getAll('label')
        const urlStatuses = params.getAll('status')
        const urlSeverities = params.getAll('severity')
        const urlUnlabeled = params.get('unlabeled') === '1'
        const tab: TabKind =
          urlKind && ([...ITEM_KINDS, 'all'] as TabKind[]).includes(urlKind) ? urlKind : 'bug'
        setActiveId(urlProject)
        setKind(tab)
        setSelectedKey(urlKey)
        if (urlKey || urlView === 'wayfinder') setProjectTab('issues')
        else setProjectTab(urlSection && ['home', 'issues', 'sessions'].includes(urlSection) ? urlSection : 'home')
        setView(urlView === 'wayfinder' ? 'wayfinder' : 'list')
        setMapKey(urlMap)
        setFiltersByKind(prev => ({
          ...prev,
          [tab]: {
            ...prev[tab],
            labels: urlLabels,
            statusFilter: urlStatuses,
            severityFilter: urlSeverities,
            component: params.get('component') ?? '',
            assignee: params.get('assignee') ?? '',
            reporter: params.get('reporter') ?? '',
            unlabeled: urlUnlabeled,
            openOnly: params.get('open') !== '0',
            offset: 0,
          },
        }))
      } catch { /* */ }
    }
    window.addEventListener('popstate', handler)
    return () => window.removeEventListener('popstate', handler)
  }, [])

  // Changing project resets all tab offsets/selection? Spec says changing project or tab resets offset and stale selection.
  // Tab offset is per-tab so tab change keeps other tab's offset; but selection always resets.
  // Project change should reset all offsets.
  const handleSelectProject = useCallback((id: string) => {
    setActiveId(id)
    setProjectTab('home')
    setSelectedKey(null)
    setFiltersByKind(prev => Object.fromEntries(
      Object.entries(prev).map(([itemKind, filters]) => [
        itemKind,
        { ...filters, offset: 0 },
      ]),
    ) as Record<TabKind, TabFilters>)
  }, [])

  const handleSelectKind = useCallback((k: TabKind) => {
    setKind(k)
    setSelectedKey(null)
    // reset offset for that tab? Keep per-tab offset but ensure current page starts at 0 if we want; spec says changing tab resets offset
    setFiltersByKind(prev => ({ ...prev, [k]: { ...prev[k], offset: 0 } }))
  }, [])

  const refreshAfterIssueChange = useCallback(async () => {
    await loadIssues()
    if (activeId) api.getTrackerProject(activeId).then(setProject).catch(() => {})
    api.listTrackerProjects(true).then(setProjects).catch(() => {})
    loadProjectDashboards()
    setTrackerVersion(v => v + 1)
  }, [loadIssues, activeId, loadProjectDashboards])

  const updateStatusFilter = useCallback((value: string) => {
    const list = currentFilters.statusFilter
    const next = list.includes(value) ? list.filter(v => v !== value) : [...list, value]
    updateCurrentFilters({ statusFilter: next, offset: 0 })
  }, [currentFilters.statusFilter, updateCurrentFilters])

  const updateSeverityFilter = useCallback((value: string) => {
    const list = currentFilters.severityFilter
    const next = list.includes(value) ? list.filter(v => v !== value) : [...list, value]
    updateCurrentFilters({ severityFilter: next, offset: 0 })
  }, [currentFilters.severityFilter, updateCurrentFilters])

  const loadFieldOptions = useCallback(async (
    field: TrackerOptionField,
    query: string,
  ): Promise<SearchableOption[]> => {
    if (!activeId) return []
    const result = await api.getTrackerFieldOptions(activeId, field, query, 16)
    return result.options.map(option => ({
      value: option.value,
      label: option.value,
      description: `${option.open} open · ${option.total} total`,
    }))
  }, [activeId, trackerVersion])
  const loadLabelOptions = useCallback(
    (query: string) => loadFieldOptions('label', query), [loadFieldOptions],
  )
  const loadComponentOptions = useCallback(
    (query: string) => loadFieldOptions('component', query), [loadFieldOptions],
  )
  const loadAssigneeOptions = useCallback(
    (query: string) => loadFieldOptions('assignee', query), [loadFieldOptions],
  )
  const loadReporterOptions = useCallback(
    (query: string) => loadFieldOptions('reporter', query), [loadFieldOptions],
  )

  const advancedFilterCount = currentFilters.statusFilter.length
    + currentFilters.severityFilter.length
    + currentFilters.labels.length
    + Number(Boolean(currentFilters.component))
    + Number(Boolean(currentFilters.assignee))
    + Number(Boolean(currentFilters.reporter))
    + Number(currentFilters.unlabeled)

  if (loading) {
    return <div className="text-gray-500 text-sm py-12 text-center">Loading projects…</div>
  }

  const headerPresentation = presentationFor(kind === 'all' ? 'bug' : kind)

  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start">
      {/* Project rail */}
      <aside className="w-full lg:w-60 shrink-0 space-y-2">
        <button
          onClick={() => setShowNewProject(true)}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition-colors"
        >
          <Plus size={15} /> New project
        </button>
        {projects.length === 0 && (
          <p className="text-xs text-gray-500 px-1 py-3">
            No projects yet. A project groups any number of directories, tmux sessions and git
            remotes under one issue log.
          </p>
        )}
        {projects.map(p => {
          const byKind = p.counts?.by_kind
          const hasKindCounts = !!byKind
          return (
            <button
              key={p.id}
              onClick={() => handleSelectProject(p.id)}
              className={`w-full text-left px-3 py-2 rounded-lg border transition-colors ${
                activeId === p.id
                  ? 'bg-gray-800 border-emerald-600/50'
                  : 'bg-gray-900/50 border-gray-800 hover:border-gray-700'
              }`}
            >
              <div className="flex items-center gap-2">
                <FolderGit2 size={14} className={activeId === p.id ? 'text-emerald-400' : 'text-gray-500'} />
                <span className="text-sm text-gray-200 truncate">{p.name}</span>
                {p.status === 'archived' && <Archive size={12} className="text-gray-600 shrink-0" />}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5 pl-6">
                {hasKindCounts ? (
                  <>
                    <span title="Bugs open">B {byKind!.bug?.open ?? 0}</span>
                    <span className="mx-1">·</span>
                    <span title="Features open">F {byKind!.feature?.open ?? 0}</span>
                    <span className="mx-1 text-gray-600">/</span>
                    <span className="text-gray-600">{p.counts?.open ?? 0} open</span>
                    <span className="mx-1 text-gray-600">/</span>
                    <span className="text-gray-600">{p.counts?.total ?? 0}</span>
                  </>
                ) : (
                  <>{p.counts?.open ?? 0} open / {p.counts?.total ?? 0}</>
                )}
              </div>
            </button>
          )
        })}
      </aside>

      {/* Project detail */}
      <section className="flex-1 min-w-0">
        {!project && (
          <div className="text-gray-500 text-sm py-12 text-center">
            Select a project, or create one to start an issue log.
          </div>
        )}

        {project && (
          <>
            <ProjectHeader
              project={project}
              kind={kind}
              onToggleScopes={() => setShowScopes(v => !v)}
              scopesOpen={showScopes}
              onArchive={() => setPendingArchive(project)}
              onChanged={async () => { await loadProjects(project.id); const fresh = await api.getTrackerProject(project.id); setProject(fresh) }}
            />

            {showScopes && vocab && (
              <ScopeEditor
                project={project}
                kinds={vocab.scope_kinds}
                onChanged={async () => setProject(await api.getTrackerProject(project.id))}
              />
            )}

            <div className="mt-4 flex items-center gap-1 rounded-lg border border-gray-800 bg-gray-950/40 p-1" role="tablist" aria-label="Project section">
              {([
                { key: 'home', label: 'Home', icon: <Home size={14} /> },
                { key: 'issues', label: 'Issues', icon: <CircleDot size={14} /> },
                { key: 'sessions', label: 'Sessions', icon: <Activity size={14} /> },
              ] as const).map(tab => (
                <button
                  key={tab.key}
                  role="tab"
                  aria-selected={projectTab === tab.key}
                  onClick={() => {
                    setProjectTab(tab.key)
                    if (tab.key !== 'issues') setSelectedKey(null)
                  }}
                  className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-xs font-medium transition-colors ${
                    projectTab === tab.key
                      ? 'bg-gray-800 text-white shadow-sm'
                      : 'text-gray-500 hover:bg-gray-900 hover:text-gray-300'
                  }`}
                >
                  {tab.icon} {tab.label}
                  {tab.key === 'issues' && (
                    <span className="text-[10px] text-gray-500">{project.counts?.all_open ?? project.counts?.open ?? 0}</span>
                  )}
                  {tab.key === 'sessions' && sessionDashboard && (
                    <span className="text-[10px] text-gray-500">{sessionDashboard.total}</span>
                  )}
                </button>
              ))}
            </div>

            {projectTab === 'home' && (
              <ProjectHomePanel
                dashboard={homeDashboard}
                loading={dashboardLoading}
                onOpenIssue={key => { setSelectedKey(key); setProjectTab('issues') }}
                onOpenSessions={() => setProjectTab('sessions')}
              />
            )}

            {projectTab === 'sessions' && activeId && (
              <ProjectSessionsPanel
                projectId={activeId}
                data={sessionDashboard}
                loading={dashboardLoading}
                onOpenIssue={key => { setSelectedKey(key); setProjectTab('issues') }}
              />
            )}

            {projectTab === 'issues' && (
            <>
            {/* View switch — Wayfinder maps are first-class, not an easter egg
                in the flat list. */}
            <div className="mt-4 flex gap-1.5" role="tablist" aria-label="Tracker view">
              {([
                { key: 'list', label: 'List', icon: <List size={12} /> },
                { key: 'wayfinder', label: 'Wayfinder', icon: <Compass size={12} /> },
              ] as const).map(v => (
                <button
                  key={v.key}
                  role="tab"
                  aria-selected={view === v.key}
                  onClick={() => setView(v.key)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
                    view === v.key
                      ? 'bg-emerald-600 border-emerald-500 text-white'
                      : 'border-gray-800 text-gray-500 hover:text-gray-300 hover:border-gray-700'
                  }`}
                >
                  {v.icon} {v.label}
                </button>
              ))}
            </div>

            {view === 'list' && (
            <>
            {/* Filters */}
            <div className="mt-4 space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <div className="relative flex-1 min-w-[220px]">
                  <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                  <input
                    value={currentFilters.query}
                    onChange={e => updateCurrentFilters({ query: e.target.value, offset: 0 })}
                    placeholder={kind === 'all' ? 'Search all project work' : `Search ${KIND_LABEL[kind].toLowerCase()}s`}
                    aria-label={kind === 'all' ? 'Search all' : `Search ${KIND_LABEL[kind]}s`}
                    className="w-full pl-9 pr-3 py-2 rounded-lg bg-gray-900 border border-gray-800 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-emerald-600/50"
                  />
                </div>
                <label className="flex items-center gap-2 text-xs text-gray-400 px-2">
                  <input type="checkbox" checked={currentFilters.openOnly} onChange={e => updateCurrentFilters({ openOnly: e.target.checked, offset: 0 })} className="accent-emerald-600" />
                  Open only
                </label>
                <button
                  onClick={() => setShowNewIssue(true)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition-colors"
                >
                  <Plus size={15} /> {headerPresentation.createButtonLabel}
                </button>
              </div>

              <div className="flex flex-wrap gap-1.5 items-center">
                <span className="text-[11px] font-medium text-gray-500 mr-1">Type:</span>
                {(['all', ...(vocab?.item_kinds ?? ITEM_KINDS)] as TabKind[]).map(k => {
                  const label = k === 'all' ? 'All' : `${KIND_LABEL[k]}s`
                  const count = k === 'all'
                    ? (project.counts?.all_open ?? project.counts?.open ?? 0)
                    : (project.counts?.by_kind?.[k]?.open ?? 0)
                  const active = kind === k
                  return (
                    <button
                      key={k}
                      onClick={() => handleSelectKind(k)}
                      aria-pressed={active}
                      aria-label={`Show ${label}`}
                      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-[11px] border transition-colors ${
                        active
                          ? 'bg-emerald-600 border-emerald-500 text-white'
                          : 'border-gray-800 text-gray-500 hover:text-gray-300 hover:border-gray-700'
                      }`}
                    >
                      {k === 'bug' && <CircleDot size={11} />}
                      {k === 'feature' && <Lightbulb size={11} />}
                      {label} <span className={`ml-1 text-[10px] ${active ? 'text-emerald-200' : 'text-gray-600'}`}>{count}</span>
                    </button>
                  )
                })}
                <span className="w-px h-4 bg-gray-800 mx-2" />
                <button
                  type="button"
                  aria-expanded={showAdvancedFilters}
                  aria-controls="tracker-advanced-filters"
                  onClick={() => setShowAdvancedFilters(open => !open)}
                  className={`inline-flex items-center gap-1.5 rounded border px-2.5 py-1 text-[11px] transition-colors ${
                    showAdvancedFilters || advancedFilterCount
                      ? 'border-emerald-600/50 bg-emerald-500/10 text-emerald-300'
                      : 'border-gray-800 text-gray-500 hover:border-gray-700 hover:text-gray-300'
                  }`}
                >
                  <SlidersHorizontal size={12} />
                  Advanced filters
                  {advancedFilterCount > 0 && (
                    <span className="rounded-full bg-emerald-500/20 px-1.5 text-[10px]">{advancedFilterCount}</span>
                  )}
                  <ChevronDown size={12} className={`transition-transform ${showAdvancedFilters ? 'rotate-180' : ''}`} />
                </button>
                {advancedFilterCount > 0 && (
                  <button
                    type="button"
                    onClick={() => updateCurrentFilters({
                      statusFilter: [], severityFilter: [], labels: [], component: '',
                      assignee: '', reporter: '', unlabeled: false, offset: 0,
                    })}
                    className="text-[11px] text-gray-500 hover:text-gray-300"
                  >
                    Clear filters
                  </button>
                )}
              </div>

              {showAdvancedFilters && (
                <div
                  id="tracker-advanced-filters"
                  data-testid="advanced-filters"
                  className="grid gap-4 rounded-lg border border-gray-800 bg-gray-950/50 p-4 lg:grid-cols-2"
                >
                  <div className="space-y-4">
                    <div>
                      <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-gray-600">Priority</div>
                      <div className="flex flex-wrap gap-1.5">
                        {(vocab?.severities ?? []).map(s => (
                          <button
                            key={s}
                            type="button"
                            onClick={() => updateSeverityFilter(s)}
                            aria-pressed={currentFilters.severityFilter.includes(s)}
                            className={`rounded border px-2 py-0.5 text-[11px] transition-colors ${
                              currentFilters.severityFilter.includes(s)
                                ? SEVERITY_CLASS[s] ?? SEVERITY_CLASS.unset
                                : 'border-gray-800 text-gray-500 hover:text-gray-300'
                            }`}
                          >
                            {s}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-gray-600">Status</div>
                      <div className="flex flex-wrap gap-1.5">
                        {(vocab?.statuses ?? []).map(s => (
                          <button
                            key={s}
                            type="button"
                            onClick={() => updateStatusFilter(s)}
                            aria-pressed={currentFilters.statusFilter.includes(s)}
                            className={`rounded px-2 py-0.5 text-[11px] transition-colors ${
                              currentFilters.statusFilter.includes(s)
                                ? STATUS_CLASS[s] ?? 'bg-gray-700 text-gray-200'
                                : 'text-gray-500 hover:text-gray-300'
                            }`}
                          >
                            {s}
                          </button>
                        ))}
                      </div>
                    </div>
                    <label className="block">
                      <span className="mb-1.5 block text-[11px] font-medium uppercase tracking-wide text-gray-600">Labels</span>
                      <SearchableMultiSelect
                        values={currentFilters.labels}
                        onChange={labels => updateCurrentFilters({ labels, unlabeled: false, offset: 0 })}
                        loadOptions={loadLabelOptions}
                        placeholder="Search labels"
                        ariaLabel="Filter labels"
                        emptyMessage="No matching labels"
                      />
                    </label>
                    <label className="flex items-center gap-2 text-xs text-gray-400">
                      <input
                        type="checkbox"
                        checked={currentFilters.unlabeled}
                        onChange={event => updateCurrentFilters({
                          unlabeled: event.target.checked,
                          labels: event.target.checked ? [] : currentFilters.labels,
                          offset: 0,
                        })}
                        className="accent-emerald-600"
                      />
                      Only items without labels
                    </label>
                  </div>
                  <div className="space-y-3">
                    <label className="block">
                      <span className="mb-1.5 block text-[11px] font-medium uppercase tracking-wide text-gray-600">Component</span>
                      <SearchableSelect value={currentFilters.component} onChange={component => updateCurrentFilters({ component, offset: 0 })}
                        loadOptions={loadComponentOptions} placeholder="Search components" ariaLabel="Filter component" />
                    </label>
                    <label className="block">
                      <span className="mb-1.5 block text-[11px] font-medium uppercase tracking-wide text-gray-600">Assignee / owner</span>
                      <SearchableSelect value={currentFilters.assignee} onChange={assignee => updateCurrentFilters({ assignee, offset: 0 })}
                        loadOptions={loadAssigneeOptions} placeholder="Search assignees" ariaLabel="Filter assignee" />
                    </label>
                    <label className="block">
                      <span className="mb-1.5 block text-[11px] font-medium uppercase tracking-wide text-gray-600">Reporter / requester</span>
                      <SearchableSelect value={currentFilters.reporter} onChange={reporter => updateCurrentFilters({ reporter, offset: 0 })}
                        loadOptions={loadReporterOptions} placeholder="Search reporters" ariaLabel="Filter reporter" />
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Issue list */}
            <div className="mt-4 rounded-lg border border-gray-800 overflow-hidden">
              {issuesLoading && (
                <div className="px-4 py-8 text-center text-sm text-gray-500 flex items-center justify-center gap-2">
                  <Loader2 size={14} className="animate-spin" /> Loading {kind === 'feature' ? 'features' : kind === 'all' ? 'items' : 'issues'}…
                </div>
              )}
              {!issuesLoading && page && page.issues.length === 0 && (
                <div className="px-4 py-8 text-center text-sm text-gray-500">No {kind === 'feature' ? 'features' : kind === 'all' ? 'items' : 'issues'} match these filters.</div>
              )}
              {!issuesLoading && page?.issues.map(issue => (
                <div key={issue.key} className="border-b border-gray-800/70 last:border-b-0">
                  <button
                    onClick={() => setSelectedKey(selectedKey === issue.key ? null : issue.key)}
                    aria-expanded={selectedKey === issue.key}
                    className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-gray-900/60 transition-colors"
                  >
                    <ChevronRight
                      size={14}
                      className={`text-gray-600 shrink-0 transition-transform ${selectedKey === issue.key ? 'rotate-90' : ''}`}
                    />
                    <code className="text-xs text-gray-500 w-24 shrink-0">{issue.key}</code>
                    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium border ${KIND_CLASS[issue.kind] ?? (issue.kind === 'feature' ? 'bg-violet-500/15 text-violet-300 border-violet-500/30' : 'bg-sky-500/15 text-sky-300 border-sky-500/30')}`}>
                      {issue.kind === 'feature' ? <Lightbulb size={10} /> : <CircleDot size={10} />}
                      {KIND_LABEL[presentationFor(issue.kind).kind].toLowerCase()}
                    </span>
                    <Pill text={issue.severity} className={SEVERITY_CLASS[issue.severity] ?? SEVERITY_CLASS.unset} />
                    <Pill text={issue.status} className={STATUS_CLASS[issue.status] ?? 'bg-gray-700/40 text-gray-300'} />
                    <span className="text-sm text-gray-200 truncate flex-1">{issue.title}</span>
                    {issue.component && <span className="text-[11px] text-gray-500 shrink-0">{issue.component}</span>}
                    <span className="text-[11px] text-gray-600 shrink-0 w-20 text-right">{shortDate(issue.created_at)}</span>
                  </button>
                  {selectedKey === issue.key && vocab && (
                    <ItemDetail
                      issueKey={issue.key}
                      initialKind={issue.kind}
                      vocab={vocab}
                      onChanged={refreshAfterIssueChange}
                      onDeleted={() => { setSelectedKey(null); refreshAfterIssueChange() }}
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Deep-link fallback: render selected issue even if closed/off-page or on another page */}
            {selectedKey && vocab && page && !page.issues.some(i => i.key === selectedKey) && (
              <div className="mt-4 rounded-lg border border-gray-800 overflow-hidden">
                <div className="px-3 py-1.5 text-[11px] text-gray-500 bg-gray-900/40 border-b border-gray-800">Deep link: {selectedKey} (not in current page/filters)</div>
                <ItemDetail
                  issueKey={selectedKey}
                  initialKind={kind === 'all' ? undefined : kind}
                  vocab={vocab}
                  onChanged={refreshAfterIssueChange}
                  onDeleted={() => { setSelectedKey(null); refreshAfterIssueChange() }}
                />
              </div>
            )}

            {page && page.total > PAGE_SIZE && (
              <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
                <span>
                  {page.offset + 1}–{Math.min(page.offset + page.issues.length, page.total)} of {page.total}
                </span>
                <div className="flex gap-2">
                  <button
                    disabled={currentFilters.offset === 0}
                    onClick={() => updateCurrentFilters({ offset: Math.max(0, currentFilters.offset - PAGE_SIZE) })}
                    className="px-2 py-1 rounded border border-gray-800 disabled:opacity-30 hover:border-gray-700"
                  >
                    Previous
                  </button>
                  <button
                    disabled={currentFilters.offset + PAGE_SIZE >= page.total}
                    onClick={() => updateCurrentFilters({ offset: currentFilters.offset + PAGE_SIZE })}
                    className="px-2 py-1 rounded border border-gray-800 disabled:opacity-30 hover:border-gray-700"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
            </>
            )}

            {view === 'wayfinder' && activeId && vocab && (
              <WayfinderPanel
                projectId={activeId}
                vocab={vocab}
                mapKey={mapKey}
                onSelectMap={key => { setMapKey(key) }}
                selectedKey={selectedKey}
                onSelectIssue={key => setSelectedKey(key)}
                refreshSignal={trackerVersion}
                onChanged={refreshAfterIssueChange}
              />
            )}

            {/* Selecting a node or row in the map opens the same editable
                detail, below the map — map context is never lost. */}
            {view === 'wayfinder' && selectedKey && vocab && (
              <div className="mt-4 rounded-lg border border-gray-800 overflow-hidden" data-testid="wayfinder-detail">
                <ItemDetail
                  issueKey={selectedKey}
                  vocab={vocab}
                  onChanged={refreshAfterIssueChange}
                  onDeleted={() => { setSelectedKey(null); refreshAfterIssueChange() }}
                  onNavigate={key => setSelectedKey(key)}
                />
              </div>
            )}
            </>
            )}
          </>
        )}
      </section>

      {showNewProject && vocab && (
        <NewProjectModal
          kinds={vocab.scope_kinds}
          onClose={() => setShowNewProject(false)}
          onCreated={async id => { setShowNewProject(false); await loadProjects(id); setSelectedKey(null) }}
        />
      )}

      {showNewIssue && project && vocab && (
        <NewItemModal
          project={project}
          vocab={vocab}
          kind={kind === 'all' ? 'bug' : kind}
          onClose={() => setShowNewIssue(false)}
          onCreated={async key => { setShowNewIssue(false); setSelectedKey(key); await refreshAfterIssueChange() }}
        />
      )}

      <ConfirmModal
        open={pendingArchive !== null}
        title="Archive project"
        message="Archiving hides the project from the default list. Its issues are kept and stay searchable."
        details={pendingArchive ? [{ label: 'Project', value: pendingArchive.name }] : undefined}
        confirmLabel="Archive"
        variant="warning"
        onCancel={() => setPendingArchive(null)}
        onConfirm={async () => {
          const target = pendingArchive!
          setPendingArchive(null)
          try {
            await api.updateTrackerProject(target.id, { status: 'archived' })
            await loadProjects(target.id)
            setProject(await api.getTrackerProject(target.id))
            showSnackbar({ type: 'success', message: `${target.name} archived` })
          } catch (err) {
            showSnackbar({ type: 'error', message: errorText(err) })
          }
        }}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------

function ProjectHomePanel({
  dashboard, loading, onOpenIssue, onOpenSessions,
}: {
  dashboard: TrackerProjectHome | null
  loading: boolean
  onOpenIssue: (key: string) => void
  onOpenSessions: () => void
}) {
  if (loading && !dashboard) {
    return <div className="flex items-center justify-center gap-2 py-16 text-sm text-gray-500"><Loader2 size={14} className="animate-spin" /> Loading project activity…</div>
  }
  if (!dashboard) {
    return <div className="py-16 text-center text-sm text-gray-500">Project activity is unavailable.</div>
  }
  const stats = [
    { label: 'Open work', value: dashboard.issues.open, icon: <CircleDot size={16} />, tone: 'text-emerald-300' },
    { label: 'In progress', value: dashboard.issues.in_progress, icon: <Activity size={16} />, tone: 'text-blue-300' },
    { label: 'Sessions', value: dashboard.sessions.total, icon: <TerminalSquare size={16} />, tone: 'text-violet-300' },
    { label: 'Live now', value: dashboard.sessions.active, icon: <Users size={16} />, tone: 'text-teal-300' },
  ]
  return (
    <div className="mt-4 space-y-5" data-testid="project-home">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map(stat => (
          <div key={stat.label} className="rounded-lg border border-gray-800 bg-gray-900/40 p-4">
            <div className={`mb-3 ${stat.tone}`}>{stat.icon}</div>
            <div className="text-2xl font-semibold text-white">{stat.value}</div>
            <div className="mt-0.5 text-xs text-gray-500">{stat.label}</div>
          </div>
        ))}
      </div>

      {dashboard.issues.favorites.length > 0 && (
        <section className="rounded-lg border border-amber-500/20 bg-amber-500/[0.03] p-4">
          <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-amber-300">
            <Star size={14} fill="currentColor" /> Tracked work
          </div>
          <IssuePreviewList issues={dashboard.issues.favorites} onOpen={onOpenIssue} />
        </section>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        <section className="rounded-lg border border-gray-800 bg-gray-950/30 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-sm font-medium text-gray-200"><Activity size={14} className="text-orange-300" /> Priority attention</h3>
            <span className="text-[11px] text-gray-600">Open P0 / P1</span>
          </div>
          {dashboard.issues.urgent.length > 0
            ? <IssuePreviewList issues={dashboard.issues.urgent} onOpen={onOpenIssue} />
            : <p className="py-5 text-center text-xs text-gray-600">No open P0 or P1 work.</p>}
        </section>
        <section className="rounded-lg border border-gray-800 bg-gray-950/30 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-sm font-medium text-gray-200"><Clock3 size={14} className="text-sky-300" /> Recently updated</h3>
            <span className="text-[11px] text-gray-600">Across every item type</span>
          </div>
          {dashboard.issues.recent.length > 0
            ? <IssuePreviewList issues={dashboard.issues.recent} onOpen={onOpenIssue} />
            : <p className="py-5 text-center text-xs text-gray-600">No work has been filed yet.</p>}
        </section>
      </div>

      <section className="rounded-lg border border-gray-800 bg-gray-950/30 p-4">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h3 className="flex items-center gap-2 text-sm font-medium text-gray-200"><TerminalSquare size={14} className="text-violet-300" /> Session trajectory</h3>
            <p className="mt-0.5 text-[11px] text-gray-600">Current campaigns and durable worker history associated with this project.</p>
          </div>
          <button onClick={onOpenSessions} className="inline-flex items-center gap-1 text-xs text-emerald-400 hover:text-emerald-300">View all <ChevronRight size={13} /></button>
        </div>
        {dashboard.sessions.recent.length > 0 ? (
          <div className="grid gap-2 lg:grid-cols-2">
            {dashboard.sessions.recent.map(session => (
              <button key={session.name} onClick={onOpenSessions} className="rounded-lg border border-gray-800 bg-gray-900/50 p-3 text-left hover:border-gray-700">
                <div className="flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${session.live ? 'bg-emerald-400' : 'bg-gray-600'}`} />
                  <code className="truncate text-xs text-gray-300">{session.name}</code>
                  <span className="ml-auto text-[10px] text-gray-600">{session.worker_count} workers</span>
                </div>
                <div className="mt-2 truncate text-[11px] text-gray-600">{session.associated_by.join(' · ') || 'associated session'}</div>
              </button>
            ))}
          </div>
        ) : <p className="py-6 text-center text-xs text-gray-600">No CAO sessions are associated yet.</p>}
      </section>
    </div>
  )
}

function IssuePreviewList({ issues, onOpen }: { issues: TrackerIssueBrief[]; onOpen: (key: string) => void }) {
  return (
    <div className="space-y-1.5">
      {issues.map(issue => (
        <button key={issue.key} onClick={() => onOpen(issue.key)} className="flex w-full items-center gap-2 rounded-md border border-transparent px-2 py-2 text-left hover:border-gray-800 hover:bg-gray-900/60">
          {issue.favorite && <Star size={11} className="shrink-0 text-amber-300" fill="currentColor" />}
          <code className="w-20 shrink-0 text-[11px] text-gray-600">{issue.key}</code>
          <span className={`rounded border px-1.5 py-0.5 text-[10px] ${KIND_CLASS[issue.kind] ?? KIND_CLASS.task}`}>{issue.kind}</span>
          <span className="min-w-0 flex-1 truncate text-xs text-gray-300">{issue.title}</span>
          <Pill text={issue.severity} className={SEVERITY_CLASS[issue.severity] ?? SEVERITY_CLASS.unset} />
        </button>
      ))}
    </div>
  )
}

function ProjectSessionsPanel({
  projectId, data, loading, onOpenIssue,
}: {
  projectId: string
  data: TrackerProjectSessions | null
  loading: boolean
  onOpenIssue: (key: string) => void
}) {
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<TrackerProjectSessionDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [logTarget, setLogTarget] = useState<{ session: string; terminal: string } | null>(null)

  useEffect(() => {
    setSelected(null)
    setDetail(null)
    setQuery('')
  }, [projectId])

  useEffect(() => {
    if (!selected) { setDetail(null); return }
    let active = true
    setDetailLoading(true)
    api.getTrackerProjectSession(projectId, selected)
      .then(result => { if (active) setDetail(result.session) })
      .catch(() => { if (active) setDetail(null) })
      .finally(() => { if (active) setDetailLoading(false) })
    return () => { active = false }
  }, [projectId, selected])

  if (loading && !data) {
    return <div className="flex items-center justify-center gap-2 py-16 text-sm text-gray-500"><Loader2 size={14} className="animate-spin" /> Loading session history…</div>
  }
  const sessions = (data?.sessions ?? []).filter(session => {
    const haystack = [session.name, ...session.providers, ...session.workdirs, ...session.associated_by].join(' ').toLowerCase()
    return haystack.includes(query.trim().toLowerCase())
  })
  return (
    <div className="mt-4 space-y-4" data-testid="project-sessions">
      <div className="grid gap-3 sm:grid-cols-3">
        {[
          ['All sessions', data?.total ?? 0],
          ['Live now', data?.active ?? 0],
          ['Historical', data?.historical ?? 0],
        ].map(([label, value]) => (
          <div key={String(label)} className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
            <div className="text-lg font-semibold text-white">{value}</div>
            <div className="text-[11px] text-gray-500">{label}</div>
          </div>
        ))}
      </div>
      <div className="relative">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
        <input value={query} onChange={event => setQuery(event.target.value)} aria-label="Search sessions" placeholder="Search session, provider, worktree, or association" className="w-full rounded-lg border border-gray-800 bg-gray-900 py-2 pl-9 pr-3 text-sm text-gray-200 placeholder-gray-600 focus:border-emerald-600/50 focus:outline-none" />
      </div>
      <div className="grid items-start gap-4 xl:grid-cols-[minmax(260px,0.8fr)_minmax(0,1.7fr)]">
        <div className="space-y-2">
          {sessions.map(session => (
            <SessionSummaryButton key={session.name} session={session} selected={selected === session.name} onClick={() => setSelected(session.name)} />
          ))}
          {sessions.length === 0 && <div className="rounded-lg border border-gray-800 py-10 text-center text-xs text-gray-600">No matching sessions.</div>}
        </div>
        <div className="min-w-0 rounded-lg border border-gray-800 bg-gray-950/30">
          {!selected && <div className="py-16 text-center text-sm text-gray-600">Select a session to inspect its workers, lineage, issues, and logs.</div>}
          {selected && detailLoading && <div className="flex items-center justify-center gap-2 py-16 text-sm text-gray-500"><Loader2 size={14} className="animate-spin" /> Loading {selected}…</div>}
          {selected && !detailLoading && detail && (
            <SessionDetailPanel detail={detail} onOpenIssue={onOpenIssue} onOpenLog={terminal => setLogTarget({ session: detail.name, terminal })} />
          )}
          {selected && !detailLoading && !detail && <div className="py-16 text-center text-sm text-gray-600">This session history could not be loaded.</div>}
        </div>
      </div>
      {logTarget && <SessionLogModal projectId={projectId} sessionName={logTarget.session} terminalId={logTarget.terminal} onClose={() => setLogTarget(null)} />}
    </div>
  )
}

function SessionSummaryButton({ session, selected, onClick }: { session: TrackerProjectSessionSummary; selected: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className={`w-full rounded-lg border p-3 text-left transition-colors ${selected ? 'border-emerald-600/50 bg-emerald-500/[0.06]' : 'border-gray-800 bg-gray-900/40 hover:border-gray-700'}`}>
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${session.live ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,.5)]' : 'bg-gray-600'}`} />
        <code className="min-w-0 flex-1 truncate text-xs text-gray-200">{session.name}</code>
        <ChevronRight size={13} className={`text-gray-600 transition-transform ${selected ? 'rotate-90' : ''}`} />
      </div>
      <div className="mt-2 flex flex-wrap gap-2 text-[10px] text-gray-500">
        <span>{session.worker_count} workers</span>
        <span>{session.issue_count} issues</span>
        <span>{session.artifact_count} artifacts</span>
        {session.providers.map(provider => <span key={provider} className="rounded bg-gray-800 px-1.5 py-0.5">{provider}</span>)}
      </div>
      <div className="mt-2 truncate text-[10px] text-gray-600">{session.associated_by.join(' · ')}</div>
    </button>
  )
}

function SessionDetailPanel({
  detail, onOpenIssue, onOpenLog,
}: {
  detail: TrackerProjectSessionDetail
  onOpenIssue: (key: string) => void
  onOpenLog: (terminal: string) => void
}) {
  return (
    <div>
      <div className="border-b border-gray-800 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2"><span className={`h-2 w-2 rounded-full ${detail.live ? 'bg-emerald-400' : 'bg-gray-600'}`} /><code className="text-sm text-gray-200">{detail.name}</code></div>
            <div className="mt-2 flex flex-wrap gap-1.5">{detail.associated_by.map(reason => <span key={reason} className="rounded border border-gray-800 px-1.5 py-0.5 text-[10px] text-gray-500">{reason}</span>)}</div>
          </div>
          <div className="text-right text-[11px] text-gray-600"><div>{detail.worker_count} worker records</div><div>{detail.last_seen ? `Last activity ${shortDate(detail.last_seen)}` : 'No activity timestamp'}</div></div>
        </div>
        {detail.workdirs.length > 0 && <div className="mt-3 space-y-1">{detail.workdirs.slice(0, 4).map(path => <div key={path} className="truncate font-mono text-[10px] text-gray-600">{path}</div>)}</div>}
      </div>
      <div className="p-4">
        <h4 className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-gray-500"><GitBranch size={13} /> Worker lineage</h4>
        <div className="space-y-2">
          {detail.terminals.map(worker => (
            <div key={worker.terminal_id} className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <TerminalSquare size={13} className={worker.wedged ? 'text-red-300' : 'text-gray-500'} />
                <code className="text-xs text-gray-300">{worker.terminal_id}</code>
                <span className="truncate text-xs text-gray-400">{worker.agent_profile || worker.name || 'unprofiled worker'}</span>
                <span className={`ml-auto rounded px-1.5 py-0.5 text-[10px] ${worker.lifecycle_state === 'live' ? STATUS_CLASS.open : 'bg-gray-800 text-gray-500'}`}>{worker.status || worker.lifecycle_state || 'historical'}</span>
              </div>
              <div className="mt-2 grid gap-1 text-[10px] text-gray-600 sm:grid-cols-2">
                <div>Provider: <span className="text-gray-500">{worker.provider || 'unknown'}</span></div>
                <div>Vintage: <span className="text-gray-500">{worker.protocol_vintage || 'unknown'}</span></div>
                {worker.caller_id && <div>Launched by: <code className="text-gray-500">{worker.caller_id}</code></div>}
                {worker.native_session_id && <div className="truncate">Native session: <code className="text-gray-500">{worker.native_session_id}</code></div>}
              </div>
              {worker.working_directory && <div className="mt-2 truncate font-mono text-[10px] text-gray-600">{worker.working_directory}</div>}
              <div className="mt-2 flex flex-wrap items-center gap-2">
                {worker.issue_keys.map(key => <button key={key} onClick={() => onOpenIssue(key)} className="text-[10px] text-sky-400 hover:text-sky-300">{key}</button>)}
                {worker.snapshot_available && <span className="inline-flex items-center gap-1 text-[10px] text-gray-600"><Box size={10} /> snapshot</span>}
                {worker.log_available && <button onClick={() => onOpenLog(worker.terminal_id)} className="ml-auto inline-flex items-center gap-1 text-[10px] text-emerald-400 hover:text-emerald-300"><ExternalLink size={10} /> Logs</button>}
              </div>
            </div>
          ))}
          {detail.terminals.length === 0 && <div className="py-8 text-center text-xs text-gray-600">The session is associated, but no retained worker records were found.</div>}
        </div>
        {detail.issues.length > 0 && (
          <div className="mt-5 border-t border-gray-800 pt-4">
            <h4 className="mb-3 text-xs font-semibold uppercase tracking-wide text-gray-500">Session issues</h4>
            <IssuePreviewList issues={detail.issues} onOpen={onOpenIssue} />
          </div>
        )}
      </div>
    </div>
  )
}

function SessionLogModal({ projectId, sessionName, terminalId, onClose }: { projectId: string; sessionName: string; terminalId: string; onClose: () => void }) {
  const [mode, setMode] = useState<'last' | 'full'>('last')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    let active = true
    setLoading(true)
    api.getTrackerProjectTerminalLog(projectId, sessionName, terminalId, mode)
      .then(result => { if (active) setOutput(result.output) })
      .catch(() => { if (active) setOutput('No captured output is available.') })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [projectId, sessionName, terminalId, mode])
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center">
      <button aria-label="Close session log" className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="relative mx-4 flex max-h-[82vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl border border-gray-700 bg-gray-900 shadow-2xl">
        <div className="flex items-center gap-3 border-b border-gray-800 px-4 py-3"><TerminalSquare size={15} className="text-emerald-400" /><div><div className="text-sm text-gray-200">Captured terminal log</div><code className="text-[10px] text-gray-600">{sessionName} · {terminalId}</code></div><button onClick={onClose} className="ml-auto text-gray-500 hover:text-white"><X size={16} /></button></div>
        <div className="flex gap-1 border-b border-gray-800 px-4 py-2">{(['last', 'full'] as const).map(value => <button key={value} onClick={() => setMode(value)} className={`rounded px-2 py-1 text-[11px] ${mode === value ? 'bg-emerald-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}>{value === 'last' ? 'Last 200 lines' : 'Full capture'}</button>)}</div>
        <pre className="min-h-[240px] flex-1 overflow-auto whitespace-pre-wrap p-4 font-mono text-xs leading-relaxed text-gray-300">{loading ? 'Loading…' : output}</pre>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------

function ProjectHeader({
  project, kind, onToggleScopes, scopesOpen, onArchive, onChanged,
}: {
  project: TrackerProject
  kind: TabKind
  onToggleScopes: () => void
  scopesOpen: boolean
  onArchive: () => void
  onChanged: () => Promise<void>
}) {
  const { showSnackbar } = useStore()
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(project.name)
  const [description, setDescription] = useState(project.description)

  useEffect(() => { setName(project.name); setDescription(project.description) }, [project.id, project.name, project.description])

  const save = async () => {
    try {
      await api.updateTrackerProject(project.id, { name, description })
      setEditing(false)
      await onChanged()
      showSnackbar({ type: 'success', message: 'Project updated' })
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    }
  }

  const scopeCount = project.scopes?.length ?? 0
  const byKind = project.counts?.by_kind

  return (
    <header className="rounded-lg border border-gray-800 bg-gray-900/40 p-4">
      {editing ? (
        <div className="space-y-2">
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            aria-label="Project name"
            className="w-full px-3 py-2 rounded bg-gray-900 border border-gray-700 text-sm text-gray-100"
          />
          <textarea
            value={description}
            onChange={e => setDescription(e.target.value)}
            aria-label="Project description"
            rows={2}
            className="w-full px-3 py-2 rounded bg-gray-900 border border-gray-700 text-sm text-gray-300"
          />
          <div className="flex gap-2">
            <button onClick={save} className="px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs">Save</button>
            <button onClick={() => setEditing(false)} className="px-3 py-1.5 rounded border border-gray-700 text-xs text-gray-400">Cancel</button>
          </div>
        </div>
      ) : (
        <>
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                {project.name}
                {project.status === 'archived' && (
                  <span className="text-[11px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-300">archived</span>
                )}
              </h2>
              <p className="text-xs text-gray-500 mt-0.5">
                <code>{project.id}</code> · keys <code>{project.issue_prefix}-NNNN</code> ·{' '}
                {byKind ? (
                  <>B {byKind.bug?.open ?? 0}/{byKind.bug?.total ?? 0} · F {byKind.feature?.open ?? 0}/{byKind.feature?.total ?? 0} · {project.counts?.open ?? 0} open of {project.counts?.total ?? 0}</>
                ) : (
                  <>{project.counts?.open ?? 0} open of {project.counts?.total ?? 0}</>
                )}
              </p>
              {project.description && <p className="text-sm text-gray-400 mt-2">{project.description}</p>}
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <a
                href={`/tracker/projects/${encodeURIComponent(project.id)}/export?kind=${encodeURIComponent(kind)}`}
                target="_blank"
                rel="noreferrer"
                title={kind === 'all' ? "Render all items as markdown" : `Render ${KIND_LABEL[kind].toLowerCase()} items as markdown`}
                className="p-2 rounded hover:bg-gray-800 text-gray-400 hover:text-gray-200"
              >
                <FileDown size={15} />
              </a>
              <button onClick={() => setEditing(true)} className="px-2 py-1 rounded text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800">
                Edit
              </button>
              {project.status !== 'archived' && (
                <button onClick={onArchive} title="Archive project" className="p-2 rounded hover:bg-gray-800 text-gray-500 hover:text-gray-300">
                  <Archive size={15} />
                </button>
              )}
            </div>
          </div>
          <button
            onClick={onToggleScopes}
            className="mt-3 text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1.5"
          >
            <ChevronRight size={12} className={`transition-transform ${scopesOpen ? 'rotate-90' : ''}`} />
            {scopeCount} scope{scopeCount === 1 ? '' : 's'} — the paths, sessions and remotes that file here
          </button>
        </>
      )}
    </header>
  )
}

// ---------------------------------------------------------------------------

function ScopeEditor({
  project, kinds, onChanged,
}: { project: TrackerProject; kinds: string[]; onChanged: () => Promise<void> }) {
  const { showSnackbar } = useStore()
  const [kind, setKind] = useState(kinds[0] ?? 'path')
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)

  const add = async () => {
    if (!value.trim()) return
    setBusy(true)
    try {
      const row = await api.addTrackerScope(project.id, { kind, value: value.trim() })
      setValue('')
      await onChanged()
      showSnackbar({
        type: row.created ? 'success' : 'info',
        message: row.created ? `Added ${row.kind} scope` : 'That scope was already registered here',
      })
    } catch (err) {
      // A 409 names the project that already owns the value — the most useful
      // thing the operator can be told, so it is surfaced verbatim.
      showSnackbar({ type: 'error', message: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  const remove = async (scope: TrackerScope) => {
    try {
      await api.removeTrackerScope(project.id, scope.id)
      await onChanged()
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    }
  }

  return (
    <div className="mt-2 rounded-lg border border-gray-800 bg-gray-900/20 p-3 space-y-2">
      {(project.scopes ?? []).map(scope => (
        <div key={scope.id} className="flex items-center gap-2 text-xs">
          <span className="w-20 shrink-0 text-gray-500">{scope.kind}</span>
          <code className="flex-1 truncate text-gray-300">{scope.value}</code>
          <button onClick={() => remove(scope)} title="Remove scope" className="p-1 rounded text-gray-600 hover:text-red-400">
            <X size={13} />
          </button>
        </div>
      ))}
      <div className="flex items-center gap-2 pt-1">
        <select
          value={kind}
          onChange={e => setKind(e.target.value)}
          aria-label="Scope kind"
          className="px-2 py-1.5 rounded bg-gray-900 border border-gray-700 text-xs text-gray-300"
        >
          {kinds.map(k => <option key={k} value={k}>{k}</option>)}
        </select>
        <input
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') add() }}
          placeholder={kind === 'path' ? '/Users/you/Projects/repo' : kind === 'session' ? 'cao-p1-closure' : 'github.com/owner/repo'}
          aria-label="Scope value"
          className="flex-1 px-2 py-1.5 rounded bg-gray-900 border border-gray-700 text-xs text-gray-200 placeholder-gray-600"
        />
        <button
          onClick={add}
          disabled={busy || !value.trim()}
          className="px-3 py-1.5 rounded bg-gray-800 hover:bg-gray-700 text-xs text-gray-200 disabled:opacity-40"
        >
          Add
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Item detail — factored around presentation descriptor so the two kinds share
// one implementation and copy is not duplicated.

function ItemDetail({
  issueKey, initialKind, vocab, onChanged, onDeleted, onNavigate,
}: {
  issueKey: string
  initialKind?: string
  vocab: TrackerVocabulary
  onChanged: () => Promise<void>
  onDeleted: () => void
  onNavigate?: (key: string) => void
}) {
  const { showSnackbar } = useStore()
  const [issue, setIssue] = useState<TrackerIssue | null>(null)
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [labelValues, setLabelValues] = useState<string[]>([])
  const [collaboratorValues, setCollaboratorValues] = useState<string[]>([])
  const [branchValues, setBranchValues] = useState<string[]>([])
  const [worktreeValues, setWorktreeValues] = useState<string[]>([])
  const [pullRequestValues, setPullRequestValues] = useState<string[]>([])
  const [retainPreviousAssignee, setRetainPreviousAssignee] = useState(true)
  const [comment, setComment] = useState('')
  const [showHistory, setShowHistory] = useState(false)
  const [saving, setSaving] = useState(false)
  const [claimBusy, setClaimBusy] = useState(false)
  const [pendingDelete, setPendingDelete] = useState(false)
  const [duplicateOf, setDuplicateOf] = useState('')
  const [closingChoice, setClosingChoice] = useState('')
  const [linkTo, setLinkTo] = useState('')
  const [linkKind, setLinkKind] = useState('relates')
  // Set when a body edit lost the optimistic-concurrency race: the current
  // version the server reported. The draft is preserved while this is shown.
  const [casConflict, setCasConflict] = useState<string | null>(null)

  const presentation = useMemo(
    () => presentationFor(draft.kind ?? issue?.kind ?? initialKind),
    [draft.kind, issue?.kind, initialKind],
  )
  const isFeature = presentation.kind === 'feature'
  const isBug = presentation.kind === 'bug'

  const load = useCallback(async (opts?: { preserveDraft?: boolean }) => {
    try {
      // Use key-universal fetch; typed wrappers would 404 on cross-kind
      const row = await api.getTrackerIssue(issueKey)
      setIssue(row)
      if (!opts?.preserveDraft) {
        setDraft({
          kind: row.kind,
          title: row.title,
          body: row.body,
          status: row.status,
          component: row.component ?? '',
          assignee: row.assignee ?? '',
          reporter: row.reporter ?? '',
          failing_command: row.failing_command ?? '',
          reproduction_steps: row.reproduction_steps ?? '',
          expected_outcome: row.expected_outcome ?? '',
          actual_outcome: row.actual_outcome ?? '',
          evidence: row.evidence ?? '',
          resolution: row.resolution ?? '',
        })
        setLabelValues(row.labels)
        setCollaboratorValues(row.collaborators ?? [])
        setBranchValues(row.branches ?? [])
        setWorktreeValues(row.worktrees ?? [])
        setPullRequestValues(row.pull_requests ?? [])
        setRetainPreviousAssignee(true)
        setDuplicateOf(row.duplicate_of ?? '')
        setCasConflict(null)
      }
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    }
  }, [issueKey, showSnackbar])

  useEffect(() => { load() }, [load])

  // The pending change set against a given base. Label edits go out as
  // add/remove DELTAS rather than a full replacement, so a label another
  // actor added concurrently is never silently dropped by this save.
  const computeChanges = useCallback((base: TrackerIssue): Record<string, unknown> => {
    const changes: Record<string, unknown> = {}
    for (const [field, value] of Object.entries(draft)) {
      const current = (base as unknown as Record<string, unknown>)[field]
      if ((current ?? '') !== value) changes[field] = value
    }
    const added = labelValues.filter(l => !base.labels.includes(l))
    const removed = base.labels.filter(l => !labelValues.includes(l))
    if (added.length) changes.add_labels = added
    if (removed.length) changes.remove_labels = removed
    const repeatable: Array<[keyof TrackerIssue, string[]]> = [
      ['collaborators', collaboratorValues],
      ['branches', branchValues],
      ['worktrees', worktreeValues],
      ['pull_requests', pullRequestValues],
    ]
    for (const [field, values] of repeatable) {
      const current = base[field] as string[]
      if (JSON.stringify(current ?? []) !== JSON.stringify(values)) changes[field] = values
    }
    return changes
  }, [draft, labelValues, collaboratorValues, branchValues, worktreeValues, pullRequestValues])

  const dirty = useMemo(() => (issue ? computeChanges(issue) : {}), [issue, computeChanges])

  const hasChanges = Object.keys(dirty).length > 0

  const patch = async (
    extra?: Record<string, unknown>,
    opts?: { expectedUpdatedAt?: string | null; changesOverride?: Record<string, unknown> },
  ) => {
    setSaving(true)
    try {
      const changes = opts?.changesOverride ?? dirty
      const body: Record<string, unknown> = { ...changes, ...extra, actor: 'dashboard' }
      if (
        issue?.assignee
        && typeof changes.assignee === 'string'
        && changes.assignee !== issue.assignee
        && !retainPreviousAssignee
      ) {
        body.drop_previous_assignee = true
      }
      if (extra?.status === 'duplicate' && !duplicateOf.trim()) {
        showSnackbar({ type: 'error', message: 'Duplicate requires a canonical issue' })
        setSaving(false)
        return
      }
      if (extra?.status === 'duplicate') body.duplicate_of = duplicateOf.trim()
      // Body edits carry optimistic concurrency: the draft was written against
      // the loaded version, and a concurrent edit must not be overwritten
      // silently. Other field-only patches stay unconditional, as before.
      const expected =
        opts?.expectedUpdatedAt !== undefined
          ? opts.expectedUpdatedAt
          : 'body' in changes || 'reproduction_steps' in changes
            ? issue?.updated_at ?? null
            : null
      if (expected) body.expected_updated_at = expected
      await api.updateTrackerIssue(issueKey, body)
      setCasConflict(null)
      await load()
      await onChanged()
      showSnackbar({ type: 'success', message: `${issueKey} updated` })
    } catch (err) {
      const current = conflictDetail(err)?.current_updated_at
      if ((err as ApiError).status === 409 && current && !extra) {
        // A stale body write: nothing was applied and the draft stays. The
        // banner below offers re-read & retry / discard.
        setCasConflict(String(current))
      } else {
        showSnackbar({ type: 'error', message: errorText(err) })
      }
    } finally {
      setSaving(false)
    }
  }

  // Re-read the issue (fresh version, server values) WITHOUT touching the
  // draft, then retry the same draft against the fresh version. A further
  // race re-arms the conflict banner.
  const rereadAndRetry = async () => {
    try {
      const fresh = await api.getTrackerIssue(issueKey)
      const changes = computeChanges(fresh)
      setIssue(fresh)
      setCasConflict(null)
      await patch(undefined, { expectedUpdatedAt: fresh.updated_at, changesOverride: changes })
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    }
  }

  // Atomic claim/unclaim — NOT an assignee PATCH: the claim endpoint is the
  // only path that can refuse a second claimant with the observed owner.
  const claim = async () => {
    setClaimBusy(true)
    try {
      await api.claimTrackerIssue(issueKey, 'dashboard')
      await load()
      await onChanged()
      showSnackbar({ type: 'success', message: `${issueKey} claimed` })
    } catch (err) {
      const owner = conflictDetail(err)?.observed_assignee
      if ((err as ApiError).status === 409 && owner) {
        showSnackbar({
          type: 'error',
          message: `${issueKey} is already claimed by ${String(owner)}`,
        })
        // Show the state the server actually holds.
        await load()
        await onChanged()
      } else {
        showSnackbar({ type: 'error', message: errorText(err) })
      }
    } finally {
      setClaimBusy(false)
    }
  }

  const unclaim = async () => {
    setClaimBusy(true)
    try {
      await api.unclaimTrackerIssue(issueKey, 'dashboard')
      await load()
      await onChanged()
      showSnackbar({ type: 'success', message: `${issueKey} released` })
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    } finally {
      setClaimBusy(false)
    }
  }

  const postComment = async () => {
    if (!comment.trim()) return
    try {
      await api.addTrackerComment(issueKey, { body: comment, author: 'dashboard' })
      setComment('')
      await load()
      await onChanged()
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    }
  }

  const addLink = async () => {
    if (!linkTo.trim()) return
    try {
      await api.addTrackerLink(issueKey, { to_key: linkTo.trim(), kind: linkKind })
      setLinkTo('')
      await load()
      await onChanged()
      showSnackbar({ type: 'success', message: `Linked ${issueKey} → ${linkTo.trim()}` })
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    }
  }

  const loadFieldOptions = useCallback(async (
    field: TrackerOptionField,
    query: string,
  ): Promise<SearchableOption[]> => {
    if (!issue?.project_id) return []
    const result = await api.getTrackerFieldOptions(issue.project_id, field, query, 12)
    return result.options.map(option => ({
      value: option.value,
      label: option.value,
      description: `${option.open} open · ${option.total} total`,
    }))
  }, [issue?.project_id, issue?.updated_at])
  const loadLabels = useCallback((query: string) => loadFieldOptions('label', query), [loadFieldOptions])
  const loadComponents = useCallback((query: string) => loadFieldOptions('component', query), [loadFieldOptions])
  const loadAssignees = useCallback((query: string) => loadFieldOptions('assignee', query), [loadFieldOptions])
  const loadReporters = useCallback((query: string) => loadFieldOptions('reporter', query), [loadFieldOptions])
  const loadCollaborators = useCallback((query: string) => loadFieldOptions('collaborator', query), [loadFieldOptions])
  const loadBranches = useCallback((query: string) => loadFieldOptions('branch', query), [loadFieldOptions])
  const loadWorktrees = useCallback((query: string) => loadFieldOptions('worktree', query), [loadFieldOptions])
  const loadPullRequests = useCallback((query: string) => loadFieldOptions('pull_request', query), [loadFieldOptions])
  const loadIssueOptions = useCallback(async (query: string): Promise<SearchableOption[]> => {
    if (!issue?.project_id) return []
    const result = await api.listTrackerIssues({
      projectId: issue.project_id,
      kind: 'all',
      q: query.trim() || undefined,
      limit: 12,
      order: 'updated_desc',
    })
    return result.issues
      .filter(candidate => candidate.key !== issue.key)
      .map(candidate => ({
        value: candidate.key,
        label: candidate.key,
        description: `${candidate.title} · ${candidate.status}`,
      }))
  }, [issue?.project_id, issue?.key, issue?.updated_at])

  const removeLink = async (linkId: number) => {
    try {
      await api.removeTrackerLink(issueKey, linkId)
      await load()
      await onChanged()
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    }
  }

  const handleDelete = async () => {
    setPendingDelete(false)
    try {
      await api.deleteTrackerIssue(issueKey)
      showSnackbar({ type: 'success', message: `${issueKey} deleted` })
      onDeleted()
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    }
  }

  if (!issue) {
    return <div className="px-10 py-4 text-xs text-gray-600">Loading {issueKey}…</div>
  }

  const statusesForKind = vocab.statuses_by_kind?.[presentation.kind] ?? vocab.statuses
  const terminalStatuses = vocab.terminal_statuses_by_kind?.[presentation.kind] ?? vocab.terminal_statuses
  const isTerminal = terminalStatuses.includes(issue.status)
  const activeWithoutAssignee =
    (draft.status ?? issue.status) === 'in-progress' && !(draft.assignee ?? '').trim()
  const bugDetailsIncomplete = isBug && [
    draft.reproduction_steps,
    draft.expected_outcome,
    draft.actual_outcome,
  ].some(value => !value?.trim())
  const bugDetailPolicyApplies = bugDetailsIncomplete && (
    issue.kind !== 'bug'
    || ['reproduction_steps', 'expected_outcome', 'actual_outcome'].some(field => field in dirty)
  )
  const isReassignment = Boolean(
    issue.assignee && (draft.assignee ?? '') !== issue.assignee,
  )

  // Free-form evidence fields stay text inputs; reusable vocabulary fields use
  // searchable, creatable pickers below.
  const editableFields: Array<{ field: keyof TrackerIssue; label: string; mono?: boolean; hideWhenEmpty?: boolean }> = [
    { field: 'failing_command' as keyof TrackerIssue, label: 'Failing command', mono: true, hideWhenEmpty: !isBug },
    { field: 'evidence' as keyof TrackerIssue, label: 'Evidence', mono: true },
  ].filter(f => !(f.hideWhenEmpty && !draft[f.field as string] && !issue[f.field]))

  return (
    <div className="px-10 py-4 bg-gray-950/50 border-t border-gray-800/70 space-y-4">
      <input
        value={draft.title ?? ''}
        onChange={e => setDraft({ ...draft, title: e.target.value })}
        aria-label={`${KIND_LABEL[presentation.kind]} title`}
        className="w-full px-3 py-2 rounded bg-gray-900 border border-gray-800 text-sm text-gray-100 focus:outline-none focus:border-emerald-600/50"
      />

      <div className="flex flex-wrap items-center gap-2">
        <select
          value={draft.kind ?? issue.kind}
          onChange={e => setDraft({ ...draft, kind: e.target.value })}
          aria-label="Type"
          title="Change item type"
          className={`px-2 py-1.5 rounded border text-xs font-medium ${KIND_CLASS[presentation.kind]}`}
        >
          {(vocab.item_kinds ?? ITEM_KINDS).map(k => (
            <option key={k} value={k}>{KIND_LABEL[presentationFor(k).kind]}</option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => patch({ favorite: !issue.favorite })}
          aria-label={issue.favorite ? 'Remove from project Home' : 'Favorite on project Home'}
          title={issue.favorite ? 'Remove from project Home' : 'Favorite on project Home'}
          className={`rounded border p-1.5 ${issue.favorite ? 'border-amber-500/50 bg-amber-500/10 text-amber-300' : 'border-gray-800 text-gray-500 hover:text-amber-300'}`}
        >
          <Star size={13} fill={issue.favorite ? 'currentColor' : 'none'} />
        </button>
        <select
          value={draft.status ?? issue.status}
          onChange={e => {
            if (e.target.value === 'duplicate') setClosingChoice('duplicate')
            else setDraft({ ...draft, status: e.target.value })
          }}
          aria-label="Status"
          className="px-2 py-1.5 rounded bg-gray-900 border border-gray-800 text-xs text-gray-200"
        >
          {statusesForKind.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <div className="min-w-[220px] max-w-sm flex-1 sm:flex-none">
          <SearchableSelect
            value={draft.assignee ?? ''}
            onChange={assignee => setDraft({ ...draft, assignee })}
            loadOptions={loadAssignees}
            placeholder={`Search or create ${presentation.assigneeLabel.toLowerCase()}`}
            ariaLabel={presentation.assigneeLabel}
            allowCreate
          />
          {isReassignment && (
            <label className="mt-1 flex items-center gap-1.5 text-[11px] text-gray-500">
              <input
                type="checkbox"
                checked={retainPreviousAssignee}
                onChange={event => setRetainPreviousAssignee(event.target.checked)}
              />
              Keep {issue.assignee} as a collaborator
            </label>
          )}
        </div>
        <select
          value={issue.severity}
          onChange={e => patch({ severity: e.target.value })}
          aria-label={presentation.severityLabel}
          className="px-2 py-1.5 rounded bg-gray-900 border border-gray-800 text-xs text-gray-200"
        >
          {vocab.severities.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        {/* Atomic claim/unclaim — deliberately not an assignee edit, so a
            second claimant gets the typed conflict naming the owner. */}
        {!isTerminal && !issue.assignee && (
          <button
            onClick={claim}
            disabled={claimBusy}
            aria-label={`Claim ${issue.key}`}
            className="px-2 py-1.5 rounded text-xs border border-emerald-700/60 text-emerald-300 hover:bg-emerald-600/20 disabled:opacity-40"
          >
            Claim
          </button>
        )}
        {issue.assignee && (
          <span className="inline-flex items-center gap-1.5">
            <span className="px-1.5 py-0.5 rounded text-[11px] bg-blue-500/15 text-blue-300 border border-blue-500/30">
              claimed by {issue.assignee}
            </span>
            <button
              onClick={unclaim}
              disabled={claimBusy}
              aria-label={`Unclaim ${issue.key} (claimed by ${issue.assignee})`}
              title="Release the claim — the ordinary recovery exit"
              className="px-2 py-1 rounded text-[11px] border border-gray-700 text-gray-400 hover:text-gray-200 disabled:opacity-40"
            >
              Unclaim
            </button>
          </span>
        )}
        <span className="text-[11px] text-gray-600 flex items-center gap-1.5">
          {isTerminal ? <CheckCircle2 size={12} /> : <CircleDot size={12} />}
          filed {shortDate(issue.created_at)}
          {issue.closed_at && ` · closed ${shortDate(issue.closed_at)}`}
          {issue.origin === 'migration' && ' · migrated from the markdown ledger'}
        </span>
        <span className="flex-1" />
        <button
          onClick={() => setShowHistory(v => !v)}
          title="Audit trail"
          className="p-1.5 rounded text-gray-500 hover:text-gray-300 hover:bg-gray-800"
        >
          <History size={14} />
        </button>
        <button
          onClick={() => setPendingDelete(true)}
          title={isFeature ? 'Delete feature' : 'Delete issue'}
          className="p-1.5 rounded text-gray-600 hover:text-red-400 hover:bg-gray-800"
        >
          <Trash2 size={14} />
        </button>
      </div>

      {/* Feature closing choices */}
      {isFeature && !isTerminal && presentation.closingOptions.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 p-2 rounded bg-gray-900/60 border border-gray-800">
          <span className="text-[11px] text-gray-500">Close as:</span>
          {presentation.closingOptions.map(opt => (
            <button
              key={opt.uiLabel}
              onClick={() => {
                if (opt.needsDuplicateOf) {
                  setClosingChoice(opt.status)
                } else {
                  patch({ status: opt.status })
                }
              }}
              className="px-2 py-1 rounded text-[11px] bg-gray-800 hover:bg-gray-700 text-gray-300"
            >
              {opt.uiLabel}
            </button>
          ))}
        </div>
      )}

      {closingChoice === 'duplicate' && (
        <div className="flex flex-wrap items-end gap-2 rounded border border-violet-700/40 bg-violet-500/10 p-3">
          <label className="min-w-[260px] flex-1">
            <span className="mb-1 block text-[11px] uppercase tracking-wide text-violet-300">Canonical issue</span>
            <SearchableSelect
              value={duplicateOf}
              onChange={setDuplicateOf}
              loadOptions={loadIssueOptions}
              placeholder="Search by issue key or title"
              ariaLabel="Canonical issue"
              emptyMessage="No matching issues"
            />
          </label>
          <button
            onClick={() => { patch({ status: 'duplicate' }); setClosingChoice('') }}
            disabled={!duplicateOf}
            className="rounded bg-violet-600 px-3 py-2 text-xs text-white hover:bg-violet-500 disabled:opacity-40"
          >
            Mark duplicate
          </button>
          <button onClick={() => setClosingChoice('')} aria-label="Cancel duplicate" className="mb-2 text-gray-500 hover:text-gray-300">
            <X size={13} />
          </button>
        </div>
      )}

      {casConflict && (
        <div
          role="alert"
          data-testid="cas-conflict"
          className="rounded-lg border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
        >
          <p>
            {issue.key} changed while you were editing (current version{' '}
            <code>{casConflict}</code>). Nothing was written and your draft is untouched — re-read
            and retry to apply it against the fresh version, or discard it.
          </p>
          <div className="flex gap-2 mt-2">
            <button
              onClick={rereadAndRetry}
              disabled={saving}
              className="px-2.5 py-1 rounded bg-amber-600 hover:bg-amber-500 text-white disabled:opacity-40"
            >
              Re-read &amp; retry
            </button>
            <button
              onClick={() => load()}
              className="px-2.5 py-1 rounded border border-gray-700 text-gray-300"
            >
              Discard draft
            </button>
          </div>
        </div>
      )}

      <textarea
        value={draft.body ?? ''}
        onChange={e => setDraft({ ...draft, body: e.target.value })}
        rows={6}
        aria-label={isFeature ? 'Feature body' : 'Issue body'}
        className="w-full px-3 py-2 rounded bg-gray-900 border border-gray-800 text-sm text-gray-300 font-mono leading-relaxed focus:outline-none focus:border-emerald-600/50"
      />

      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-600">Component</span>
          <SearchableSelect
            value={draft.component ?? ''}
            onChange={component => setDraft({ ...draft, component })}
            loadOptions={loadComponents}
            placeholder="Search or create a component"
            ariaLabel="Component"
            allowCreate
            className="mt-1"
          />
        </label>
        <label className="block col-span-2">
          <span className="text-[11px] uppercase tracking-wide text-gray-600">{presentation.reporterLabel}</span>
          <SearchableSelect
            value={draft.reporter ?? ''}
            onChange={reporter => setDraft({ ...draft, reporter })}
            loadOptions={loadReporters}
            placeholder={`Search or create ${presentation.reporterLabel.toLowerCase()}`}
            ariaLabel={presentation.reporterLabel}
            allowCreate
            className="mt-1"
          />
        </label>
        <label className="block col-span-2">
          <span className="text-[11px] uppercase tracking-wide text-gray-600">Collaborators</span>
          <SearchableMultiSelect
            values={collaboratorValues}
            onChange={setCollaboratorValues}
            loadOptions={loadCollaborators}
            placeholder="Search or add collaborators"
            ariaLabel="Collaborators"
            allowCreate
            className="mt-1"
          />
        </label>
        <label className="block col-span-2">
          <span className="text-[11px] uppercase tracking-wide text-gray-600">Branches</span>
          <SearchableMultiSelect
            values={branchValues}
            onChange={setBranchValues}
            loadOptions={loadBranches}
            placeholder="Search or add implementation branches"
            ariaLabel="Branches"
            allowCreate
            className="mt-1"
          />
        </label>
        <label className="block col-span-2">
          <span className="text-[11px] uppercase tracking-wide text-gray-600">Worktrees</span>
          <SearchableMultiSelect
            values={worktreeValues}
            onChange={setWorktreeValues}
            loadOptions={loadWorktrees}
            placeholder="Search or add worktree paths"
            ariaLabel="Worktrees"
            allowCreate
            className="mt-1"
          />
        </label>
        <label className="block col-span-2">
          <span className="text-[11px] uppercase tracking-wide text-gray-600">Pull requests</span>
          <SearchableMultiSelect
            values={pullRequestValues}
            onChange={setPullRequestValues}
            loadOptions={loadPullRequests}
            placeholder="Search or add PR URLs / references"
            ariaLabel="Pull requests"
            allowCreate
            className="mt-1"
          />
        </label>
        {editableFields.map(({ field, label, mono }) => (
          <label key={field as string} className="block">
            <span className="text-[11px] uppercase tracking-wide text-gray-600">{label}</span>
            <input
              value={draft[field as string] ?? ''}
              onChange={e => setDraft({ ...draft, [field as string]: e.target.value })}
              aria-label={label}
              className={`mt-1 w-full px-2 py-1.5 rounded bg-gray-900 border border-gray-800 text-xs text-gray-200 ${mono ? 'font-mono' : ''}`}
            />
          </label>
        ))}
        <label className="block col-span-2">
          <span className="text-[11px] uppercase tracking-wide text-gray-600">Labels</span>
          <SearchableMultiSelect
            values={labelValues}
            onChange={setLabelValues}
            loadOptions={loadLabels}
            placeholder="Search or create labels"
            ariaLabel="Labels"
            allowCreate
            className="mt-1"
          />
        </label>
        {isBug && (
          <>
            <label className="block col-span-2">
              <span className="text-[11px] uppercase tracking-wide text-gray-600">Reproduction steps</span>
              <textarea
                value={draft.reproduction_steps ?? ''}
                onChange={e => setDraft({ ...draft, reproduction_steps: e.target.value })}
                rows={4}
                aria-label="Reproduction steps"
                placeholder="Setup, exact numbered actions, and the point where behavior diverges"
                className="mt-1 w-full rounded bg-gray-900 border border-gray-800 px-3 py-2 text-xs text-gray-200 font-mono leading-relaxed focus:outline-none focus:border-emerald-600/50"
              />
            </label>
            <label className="block">
              <span className="text-[11px] uppercase tracking-wide text-gray-600">Expected outcome</span>
              <textarea
                value={draft.expected_outcome ?? ''}
                onChange={e => setDraft({ ...draft, expected_outcome: e.target.value })}
                rows={3}
                aria-label="Expected outcome"
                placeholder="What should happen"
                className="mt-1 w-full rounded bg-gray-900 border border-gray-800 px-3 py-2 text-xs text-gray-200 leading-relaxed focus:outline-none focus:border-emerald-600/50"
              />
            </label>
            <label className="block">
              <span className="text-[11px] uppercase tracking-wide text-gray-600">Actual outcome</span>
              <textarea
                value={draft.actual_outcome ?? ''}
                onChange={e => setDraft({ ...draft, actual_outcome: e.target.value })}
                rows={3}
                aria-label="Actual outcome"
                placeholder="What happens instead"
                className="mt-1 w-full rounded bg-gray-900 border border-gray-800 px-3 py-2 text-xs text-gray-200 leading-relaxed focus:outline-none focus:border-emerald-600/50"
              />
            </label>
          </>
        )}
        <label className="block col-span-2">
          <span className="text-[11px] uppercase tracking-wide text-gray-600">{presentation.resolutionLabel}</span>
          <textarea
            value={draft.resolution ?? ''}
            onChange={e => setDraft({ ...draft, resolution: e.target.value })}
            rows={3}
            aria-label={presentation.resolutionLabel}
            className="mt-1 w-full rounded bg-gray-900 border border-gray-800 px-3 py-2 text-xs text-gray-200 leading-relaxed focus:outline-none focus:border-emerald-600/50"
          />
        </label>
        {issue.duplicate_of && (
          <div className="col-span-2 text-xs text-gray-500">
            Duplicate of{' '}
            <button onClick={() => onNavigate?.(issue.duplicate_of!)} className="font-mono text-emerald-400 hover:underline">
              {issue.duplicate_of}
            </button>
          </div>
        )}
      </div>

      {hasChanges && (
        <div className="space-y-2">
          {activeWithoutAssignee && (
            <div role="alert" className="rounded border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              In-progress work should have one primary assignee. Choose an assignee above, or save an explicit exception.
            </div>
          )}
          {bugDetailPolicyApplies && (
            <div role="alert" className="rounded border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              Bugs should include reproduction steps, expected outcome, and actual outcome. Complete the fields above, or save an explicit exception.
            </div>
          )}
          <div className="flex items-center gap-2">
          <button
            onClick={() => patch()}
            disabled={saving || activeWithoutAssignee || bugDetailPolicyApplies}
            className="flex items-center gap-2 px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs disabled:opacity-50"
          >
            {saving ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
            Save {Object.keys(dirty).length} change{Object.keys(dirty).length === 1 ? '' : 's'}
          </button>
          {(activeWithoutAssignee || bugDetailPolicyApplies) && (
            <button
              onClick={() => patch({ force: true })}
              disabled={saving}
              className="rounded border border-amber-600/60 px-3 py-1.5 text-xs text-amber-200 hover:bg-amber-500/10 disabled:opacity-50"
            >
              Save with override
            </button>
          )}
          <button onClick={() => load()} className="px-3 py-1.5 rounded border border-gray-800 text-xs text-gray-400">
            Discard
          </button>
          </div>
        </div>
      )}

      <div className="space-y-2">
        {issue.links && issue.links.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {issue.links.map(link => {
              // Direction, in words: "blocks cond-2" vs "blocked by cond-1",
              // "part of cond-1" vs "contains cond-2". JSON stays explicit;
              // this line is for humans, and the key navigates.
              const { phrase, other } = linkPhrase(link, issue.key)
              return (
                <span key={link.id} className="inline-flex items-center gap-1.5 text-[11px] text-gray-400 px-2 py-1 rounded bg-gray-900 border border-gray-800">
                  <Link2 size={11} />
                  {phrase}{' '}
                  <button
                    onClick={() => onNavigate?.(other)}
                    aria-label={`Open ${other}`}
                    className="font-mono text-emerald-400 hover:underline"
                  >
                    {other}
                  </button>
                  <button onClick={() => removeLink(link.id)} aria-label={`Remove link ${link.id}`} className="ml-1 text-gray-500 hover:text-red-400">
                    <X size={10} />
                  </button>
                </span>
              )
            })}
          </div>
        )}
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start">
          <SearchableSelect
            value={linkTo}
            onChange={setLinkTo}
            loadOptions={loadIssueOptions}
            placeholder="Search issue key or title"
            ariaLabel="Link target issue"
            emptyMessage="No matching issues"
            className="flex-1"
          />
          <select value={linkKind} onChange={e => setLinkKind(e.target.value)} aria-label="Link kind" className="px-2 py-1.5 rounded bg-gray-900 border border-gray-800 text-xs text-gray-200">
            {vocab.link_kinds.map(k => <option key={k} value={k}>{k}</option>)}
          </select>
          <button onClick={addLink} disabled={!linkTo} className="px-3 py-2 rounded bg-gray-800 hover:bg-gray-700 text-xs text-gray-200 disabled:opacity-40">Link</button>
        </div>
      </div>

      <div className="space-y-2">
        {(issue.comments ?? []).map(c => (
          <div key={c.id} className="rounded bg-gray-900/60 border border-gray-800 px-3 py-2">
            <div className="text-[11px] text-gray-600">{c.author ?? 'unknown'} · {shortDate(c.created_at)}</div>
            <div className="text-xs text-gray-300 whitespace-pre-wrap mt-1">{c.body}</div>
          </div>
        ))}
        <div className="flex gap-2">
          <input
            value={comment}
            onChange={e => setComment(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) postComment() }}
            placeholder="Add a comment"
            aria-label="Add a comment"
            className="flex-1 px-3 py-1.5 rounded bg-gray-900 border border-gray-800 text-xs text-gray-200 placeholder-gray-600"
          />
          <button
            onClick={postComment}
            disabled={!comment.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-gray-800 hover:bg-gray-700 text-xs text-gray-200 disabled:opacity-40"
          >
            <MessageSquare size={13} /> Comment
          </button>
        </div>
      </div>

      {showHistory && (
        <ol className="space-y-1 border-t border-gray-800 pt-3">
          {(issue.events ?? []).map(event => (
            <li key={event.id} className="text-[11px] text-gray-500 font-mono">
              {shortDate(event.created_at)} · {event.actor ?? 'unknown'} ·{' '}
              {event.kind === 'field'
                ? `${event.field}: ${event.old_value ?? '∅'} → ${event.new_value ?? '∅'}`
                : `${event.kind}${event.new_value ? `: ${event.new_value.slice(0, 60)}` : ''}`}
            </li>
          ))}
        </ol>
      )}

      <ConfirmModal
        open={pendingDelete}
        title={`Delete ${issue.key}`}
        message={`This removes the ${isFeature ? 'feature request' : 'issue'} and every comment, link and audit event attached to it. The key is never reissued.`}
        details={[{ label: 'Title', value: issue.title }]}
        confirmLabel="Delete"
        onCancel={() => setPendingDelete(false)}
        onConfirm={handleDelete}
      />
    </div>
  )
}

// Backward compat alias — keep old name for any external import
const IssueDetail = ItemDetail

// ---------------------------------------------------------------------------

function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label={title}
        onClick={e => e.stopPropagation()}
        className="w-full max-w-lg rounded-xl border border-gray-800 bg-gray-900 shadow-2xl"
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800">
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          <button onClick={onClose} aria-label="Close" className="p-1 rounded text-gray-500 hover:text-gray-200">
            <X size={16} />
          </button>
        </div>
        <div className="p-5 space-y-3 max-h-[70vh] overflow-y-auto">{children}</div>
      </div>
    </div>
  )
}

function NewProjectModal({
  kinds, onClose, onCreated,
}: { kinds: string[]; onClose: () => void; onCreated: (id: string) => void }) {
  const { showSnackbar } = useStore()
  const [name, setName] = useState('')
  const [id, setId] = useState('')
  const [prefix, setPrefix] = useState('')
  const [description, setDescription] = useState('')
  const [scopes, setScopes] = useState<Array<{ kind: string; value: string }>>([])
  const [kind, setKind] = useState(kinds[0] ?? 'path')
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)

  const create = async () => {
    setBusy(true)
    try {
      const created = await api.createTrackerProject({
        name,
        id: id.trim() || undefined,
        issue_prefix: prefix.trim() || undefined,
        description,
        scopes,
      })
      showSnackbar({ type: 'success', message: `Created ${created.id}` })
      onCreated(created.id)
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal title="New project" onClose={onClose}>
      <label className="block">
        <span className="text-[11px] uppercase tracking-wide text-gray-500">Name</span>
        <input value={name} onChange={e => setName(e.target.value)} aria-label="Name"
          placeholder="CAO System"
          className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-sm text-gray-100" />
      </label>
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">Id (optional)</span>
          <input value={id} onChange={e => setId(e.target.value)} aria-label="Id" placeholder="derived from the name"
            className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-sm text-gray-100 font-mono" />
        </label>
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">Key prefix (optional)</span>
          <input value={prefix} onChange={e => setPrefix(e.target.value)} aria-label="Key prefix" placeholder="cond"
            className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-sm text-gray-100 font-mono" />
        </label>
      </div>
      <label className="block">
        <span className="text-[11px] uppercase tracking-wide text-gray-500">Description</span>
        <textarea value={description} onChange={e => setDescription(e.target.value)} rows={2} aria-label="Description"
          className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-sm text-gray-300" />
      </label>

      <div className="space-y-1.5">
        <span className="text-[11px] uppercase tracking-wide text-gray-500">Scopes</span>
        {scopes.map((s, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className="w-20 text-gray-500">{s.kind}</span>
            <code className="flex-1 truncate text-gray-300">{s.value}</code>
            <button onClick={() => setScopes(scopes.filter((_, j) => j !== i))} aria-label="Remove"
              className="p-1 text-gray-600 hover:text-red-400"><X size={12} /></button>
          </div>
        ))}
        <div className="flex gap-2">
          <select value={kind} onChange={e => setKind(e.target.value)} aria-label="Scope kind"
            className="px-2 py-1.5 rounded bg-gray-950 border border-gray-800 text-xs text-gray-300">
            {kinds.map(k => <option key={k} value={k}>{k}</option>)}
          </select>
          <input value={value} onChange={e => setValue(e.target.value)} aria-label="Scope value"
            onKeyDown={e => { if (e.key === 'Enter' && value.trim()) { setScopes([...scopes, { kind, value: value.trim() }]); setValue('') } }}
            placeholder="/Users/you/Projects/repo"
            className="flex-1 px-2 py-1.5 rounded bg-gray-950 border border-gray-800 text-xs text-gray-200 placeholder-gray-600" />
          <button
            onClick={() => { if (value.trim()) { setScopes([...scopes, { kind, value: value.trim() }]); setValue('') } }}
            className="px-3 py-1.5 rounded bg-gray-800 hover:bg-gray-700 text-xs text-gray-200">Add</button>
        </div>
      </div>

      <button onClick={create} disabled={busy || !name.trim()}
        className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-sm disabled:opacity-50">
        {busy && <Loader2 size={14} className="animate-spin" />} Create project
      </button>
    </Modal>
  )
}

// Generic create modal factored around presentation descriptor — one component
// handles both kinds so fixes to validation, body-file handling or audit actors
// do not diverge.

function NewItemModal({
  project, vocab, kind, onClose, onCreated,
}: {
  project: TrackerProject
  vocab: TrackerVocabulary
  kind: ItemKind
  onClose: () => void
  onCreated: (key: string) => void
}) {
  const presentation = KIND_PRESENTATION[kind]
  const { showSnackbar } = useStore()
  const [title, setTitle] = useState('')
  const [body, setBody] = useState(presentation.bodyStarter)
  const [severity, setSeverity] = useState('unset')
  const [status, setStatus] = useState('open')
  const [component, setComponent] = useState('')
  const [requester, setRequester] = useState('dashboard')
  const [owner, setOwner] = useState('')
  const [labels, setLabels] = useState<string[]>([])
  const [collaborators, setCollaborators] = useState<string[]>([])
  const [branches, setBranches] = useState<string[]>([])
  const [worktrees, setWorktrees] = useState<string[]>([])
  const [pullRequests, setPullRequests] = useState<string[]>([])
  const [evidence, setEvidence] = useState('')
  const [failingCommand, setFailingCommand] = useState('')
  const [reproductionSteps, setReproductionSteps] = useState('')
  const [expectedOutcome, setExpectedOutcome] = useState('')
  const [actualOutcome, setActualOutcome] = useState('')
  const [favorite, setFavorite] = useState(false)
  const [busy, setBusy] = useState(false)

  // Reset body when kind switches (modal is remounted via key, but guard anyway)
  useEffect(() => {
    setBody(presentation.bodyStarter)
  }, [presentation.bodyStarter])

  const create = async (force = false) => {
    setBusy(true)
    try {
      const base: Record<string, unknown> = {
        project_id: project.id,
        kind,
        title,
        body,
        severity,
        status: status || undefined,
        component: component.trim() || undefined,
        evidence: evidence.trim() || undefined,
        labels,
        collaborators,
        branches,
        worktrees,
        pull_requests: pullRequests,
        reporter: requester.trim() || 'dashboard',
        assignee: owner.trim() || undefined,
        favorite,
        origin: 'dashboard',
        force,
      }
      if (kind === 'bug') {
        base.failing_command = failingCommand.trim() || undefined
        base.reproduction_steps = reproductionSteps.trim() || undefined
        base.expected_outcome = expectedOutcome.trim() || undefined
        base.actual_outcome = actualOutcome.trim() || undefined
      }
      const created = await api.createTrackerIssue(base)
      showSnackbar({ type: 'success', message: `Filed ${created.key}` })
      onCreated(created.key)
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  const statusOptions = vocab.statuses_by_kind?.[kind] ?? vocab.statuses
  const activeWithoutOwner = status === 'in-progress' && !owner.trim()
  const bugDetailsIncomplete = kind === 'bug' && [
    reproductionSteps,
    expectedOutcome,
    actualOutcome,
  ].some(value => !value.trim())

  const loadFieldOptions = useCallback(async (
    field: TrackerOptionField,
    query: string,
  ): Promise<SearchableOption[]> => {
    const result = await api.getTrackerFieldOptions(project.id, field, query, 12)
    return result.options.map(option => ({
      value: option.value,
      label: option.value,
      description: `${option.open} open · ${option.total} total`,
    }))
  }, [project.id])
  const loadLabels = useCallback((query: string) => loadFieldOptions('label', query), [loadFieldOptions])
  const loadComponents = useCallback((query: string) => loadFieldOptions('component', query), [loadFieldOptions])
  const loadAssignees = useCallback((query: string) => loadFieldOptions('assignee', query), [loadFieldOptions])
  const loadReporters = useCallback((query: string) => loadFieldOptions('reporter', query), [loadFieldOptions])
  const loadCollaborators = useCallback((query: string) => loadFieldOptions('collaborator', query), [loadFieldOptions])
  const loadBranches = useCallback((query: string) => loadFieldOptions('branch', query), [loadFieldOptions])
  const loadWorktrees = useCallback((query: string) => loadFieldOptions('worktree', query), [loadFieldOptions])
  const loadPullRequests = useCallback((query: string) => loadFieldOptions('pull_request', query), [loadFieldOptions])

  return (
    <Modal title={presentation.modalTitle(project.name)} onClose={onClose}>
      <label className="block">
        <span className="text-[11px] uppercase tracking-wide text-gray-500">Title</span>
        <input value={title} onChange={e => setTitle(e.target.value)} aria-label="Title"
          className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-sm text-gray-100" />
      </label>
      <label className="block">
        <span className="text-[11px] uppercase tracking-wide text-gray-500">{presentation.bodyLabel}</span>
        <textarea value={body} onChange={e => setBody(e.target.value)} rows={6} aria-label="Body"
          className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-sm text-gray-300 font-mono" />
      </label>
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">{presentation.severityLabel}</span>
          <select value={severity} onChange={e => setSeverity(e.target.value)} aria-label={presentation.severityLabel}
            className="mt-1 w-full px-2 py-2 rounded bg-gray-950 border border-gray-800 text-sm text-gray-200">
            {vocab.severities.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">Status</span>
          <select value={status} onChange={e => setStatus(e.target.value)} aria-label="Status"
            className="mt-1 w-full px-2 py-2 rounded bg-gray-950 border border-gray-800 text-sm text-gray-200">
            {statusOptions.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">Component</span>
          <SearchableSelect value={component} onChange={setComponent} loadOptions={loadComponents}
            placeholder="Search or create a component" ariaLabel="Component" allowCreate className="mt-1" />
        </label>
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">Evidence path</span>
          <input value={evidence} onChange={e => setEvidence(e.target.value)} aria-label="Evidence"
            className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-xs text-gray-200 font-mono" />
        </label>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">{presentation.reporterLabel}</span>
          <SearchableSelect value={requester} onChange={setRequester} loadOptions={loadReporters}
            placeholder={`Search or create ${presentation.reporterLabel.toLowerCase()}`} ariaLabel={presentation.reporterLabel} allowCreate className="mt-1" />
        </label>
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">{presentation.assigneeLabel}</span>
          <SearchableSelect value={owner} onChange={setOwner} loadOptions={loadAssignees}
            placeholder={`Search or create ${presentation.assigneeLabel.toLowerCase()}`} ariaLabel={presentation.assigneeLabel} allowCreate className="mt-1" />
        </label>
      </div>
      {kind === 'bug' && (
        <>
          <label className="block">
            <span className="text-[11px] uppercase tracking-wide text-gray-500">Failing command</span>
            <input value={failingCommand} onChange={e => setFailingCommand(e.target.value)} aria-label="Failing command"
              className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-xs text-gray-200 font-mono" />
          </label>
          <label className="block">
            <span className="text-[11px] uppercase tracking-wide text-gray-500">Reproduction steps</span>
            <textarea value={reproductionSteps} onChange={e => setReproductionSteps(e.target.value)} rows={4}
              aria-label="Reproduction steps" placeholder="Numbered steps, required setup, and the observed result"
              className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-xs text-gray-200 font-mono" />
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-[11px] uppercase tracking-wide text-gray-500">Expected outcome</span>
              <textarea value={expectedOutcome} onChange={e => setExpectedOutcome(e.target.value)} rows={3}
                aria-label="Expected outcome" placeholder="What should happen"
                className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-xs text-gray-200" />
            </label>
            <label className="block">
              <span className="text-[11px] uppercase tracking-wide text-gray-500">Actual outcome</span>
              <textarea value={actualOutcome} onChange={e => setActualOutcome(e.target.value)} rows={3}
                aria-label="Actual outcome" placeholder="What happens instead"
                className="mt-1 w-full px-3 py-2 rounded bg-gray-950 border border-gray-800 text-xs text-gray-200" />
            </label>
          </div>
        </>
      )}
      <label className="block">
        <span className="text-[11px] uppercase tracking-wide text-gray-500">Labels</span>
        <SearchableMultiSelect values={labels} onChange={setLabels} loadOptions={loadLabels}
          placeholder="Search or create labels" ariaLabel="Labels" allowCreate className="mt-1" />
      </label>
      <label className="flex items-center gap-2 text-xs text-gray-400">
        <input type="checkbox" checked={favorite} onChange={e => setFavorite(e.target.checked)} />
        Show this item on the project Home dashboard
      </label>
      <label className="block">
        <span className="text-[11px] uppercase tracking-wide text-gray-500">Collaborators</span>
        <SearchableMultiSelect values={collaborators} onChange={setCollaborators} loadOptions={loadCollaborators}
          placeholder="Search or add collaborators" ariaLabel="Collaborators" allowCreate className="mt-1" />
      </label>
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">Branches</span>
          <SearchableMultiSelect values={branches} onChange={setBranches} loadOptions={loadBranches}
            placeholder="Search or add branches" ariaLabel="Branches" allowCreate className="mt-1" />
        </label>
        <label className="block">
          <span className="text-[11px] uppercase tracking-wide text-gray-500">Worktrees</span>
          <SearchableMultiSelect values={worktrees} onChange={setWorktrees} loadOptions={loadWorktrees}
            placeholder="Search or add worktrees" ariaLabel="Worktrees" allowCreate className="mt-1" />
        </label>
      </div>
      <label className="block">
        <span className="text-[11px] uppercase tracking-wide text-gray-500">Pull requests</span>
        <SearchableMultiSelect values={pullRequests} onChange={setPullRequests} loadOptions={loadPullRequests}
          placeholder="Search or add PR URLs / references" ariaLabel="Pull requests" allowCreate className="mt-1" />
      </label>
      {activeWithoutOwner && (
        <div role="alert" className="rounded border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          In-progress work should have one primary {presentation.assigneeLabel.toLowerCase()}.
        </div>
      )}
      {bugDetailsIncomplete && (
        <div role="alert" className="rounded border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          Bugs should include reproduction steps, expected outcome, and actual outcome. Complete those fields, or file an explicit exception.
        </div>
      )}
      <button onClick={() => create()} disabled={busy || !title.trim() || activeWithoutOwner || bugDetailsIncomplete}
        className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-sm disabled:opacity-50">
        {busy && <Loader2 size={14} className="animate-spin" />} {presentation.createActionLabel}
      </button>
      {(activeWithoutOwner || bugDetailsIncomplete) && (
        <button onClick={() => create(true)} disabled={busy || !title.trim()}
          className="w-full rounded border border-amber-600/60 px-3 py-2 text-sm text-amber-200 hover:bg-amber-500/10 disabled:opacity-50">
          {presentation.createActionLabel} with policy override
        </button>
      )}
    </Modal>
  )
}

// Keep old name as alias for backward compat (tests don't import it, but keep for safety)
const NewIssueModal = NewItemModal
