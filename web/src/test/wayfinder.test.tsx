// Wayfinder view tests (cond-0394).
//
// Sigma needs a real WebGL2 context which jsdom lacks, so `sigma` is mocked
// (the pattern from memory-graph.test.tsx): the fake records the graphology
// graph it was constructed with and lets tests simulate clickNode. Assertions
// target the projection rendering, the graph data, the filter wiring, and the
// claim/CAS conflict paths — not canvas pixels.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { useStore } from '../store'

type AnyHandler = (payload: any) => void

const { FakeSigma, getLastSigma, resetLastSigma } = vi.hoisted(() => {
  let last: any
  class FakeSigmaImpl {
    graph: any
    container: HTMLElement
    handlers: Record<string, AnyHandler[]> = {}
    killed = false
    settings: Record<string, unknown> = {}
    constructor(graph: unknown, container: HTMLElement, settings?: Record<string, unknown>) {
      this.graph = graph
      this.container = container
      this.settings = settings ?? {}
      last = this
    }
    on(event: string, handler: AnyHandler) {
      ;(this.handlers[event] ??= []).push(handler)
    }
    emit(event: string, payload: any) {
      for (const h of this.handlers[event] ?? []) h(payload)
    }
    setSetting(key: string, value: unknown) {
      this.settings[key] = value
    }
    refresh() {}
    kill() {
      this.killed = true
    }
  }
  return {
    FakeSigma: FakeSigmaImpl,
    getLastSigma: () => last,
    resetLastSigma: () => {
      last = undefined
    },
  }
})

vi.mock('sigma', () => ({ default: FakeSigma }))

// eslint-disable-next-line import/first
import { ProjectsPanel } from '../components/ProjectsPanel'
// eslint-disable-next-line import/first
import type { TrackerIssue } from '../api'

const VOCAB = {
  statuses: ['open', 'triage', 'in-progress', 'blocked', 'resolved', 'closed', 'wontfix', 'duplicate'],
  terminal_statuses: ['closed', 'duplicate', 'wontfix'],
  item_kinds: ['project', 'bug', 'feature', 'milestone', 'goal', 'epic', 'story', 'task'],
  severities: ['P0', 'P1', 'P2', 'P3', 'P4', 'unset'],
  scope_kinds: ['path', 'session', 'git_remote', 'project_id'],
  link_kinds: ['blocks', 'relates', 'duplicates', 'caused-by', 'part-of'],
  project_statuses: ['active', 'archived'],
}

const PROJECT = {
  id: 'cao-system',
  name: 'CAO System',
  description: '',
  status: 'active',
  issue_prefix: 'cond',
  next_issue_number: 10,
  created_at: '2026-08-01T00:00:00Z',
  updated_at: '2026-08-01T00:00:00Z',
  counts: { total: 5, open: 4, by_kind: { bug: { total: 5, open: 4 }, feature: { total: 0, open: 0 } }, all_total: 5, all_open: 4 },
  scopes: [],
}

function issue(over: Record<string, unknown>): TrackerIssue {
  return {
    key: '',
    title: '',
    project_id: 'cao-system',
    kind: 'bug',
    body: '',
    status: 'open',
    severity: 'unset',
    component: null,
    reporter: null,
    assignee: null,
    labels: [],
    collaborators: [],
    branches: [],
    worktrees: [],
    pull_requests: [],
    failing_command: null,
    reproduction_steps: null,
    expected_outcome: null,
    actual_outcome: null,
    evidence: null,
    resolution: null,
    session_name: null,
    terminal_id: null,
    source_path: null,
    duplicate_of: null,
    origin: 'api',
    favorite: false,
    created_at: '2026-08-10T00:00:00Z',
    updated_at: '2026-08-10T00:00:00Z',
    closed_at: null,
    comments: [],
    events: [],
    links: [],
    ...over,
  } as TrackerIssue
}

const MAP = issue({
  key: 'cond-0001',
  title: 'Find the deploy path',
  labels: ['wayfinder:map', 'effort:deploy'],
  body: '## Destination\n\nDecide the cutover strategy.',
  updated_at: '2026-08-10T00:00:00Z',
})
const T_RESEARCH = issue({ key: 'cond-0002', title: 'research providers', labels: ['wayfinder:research'] })
const T_GRILL = issue({ key: 'cond-0003', title: 'grill the operator', assignee: 'terra', labels: ['wayfinder:grilling'] })
const T_PROTO = issue({ key: 'cond-0004', title: 'prototype the UI', status: 'closed', closed_at: '2026-08-12T00:00:00Z', labels: ['wayfinder:prototype'] })
const T_MIGRATE = issue({ key: 'cond-0005', title: 'migrate the store' })
const EXT = issue({ key: 'cond-0009', title: 'quota repair' })
const EXT2 = issue({ key: 'cond-0010', title: 'prior art survey' })

const ALL_ISSUES = [MAP, T_RESEARCH, T_GRILL, T_PROTO, T_MIGRATE, EXT, EXT2]

const PROJECTION = {
  map: MAP,
  children: [
    { ...T_RESEARCH, blocked_by: [], frontier: true },
    { ...T_GRILL, blocked_by: [], frontier: false },
    { ...T_PROTO, blocked_by: [], frontier: false },
    { ...T_MIGRATE, blocked_by: ['cond-0009'], frontier: false },
  ],
  frontier: ['cond-0002'],
  links: [
    { id: 1, kind: 'part-of', from_key: 'cond-0002', to_key: 'cond-0001' },
    { id: 2, kind: 'part-of', from_key: 'cond-0003', to_key: 'cond-0001' },
    { id: 3, kind: 'part-of', from_key: 'cond-0004', to_key: 'cond-0001' },
    { id: 4, kind: 'part-of', from_key: 'cond-0005', to_key: 'cond-0001' },
    { id: 5, kind: 'blocks', from_key: 'cond-0009', to_key: 'cond-0005' },
    { id: 6, kind: 'relates', from_key: 'cond-0002', to_key: 'cond-0003' },
    { id: 7, kind: 'relates', from_key: 'cond-0002', to_key: 'cond-0010' },
    { id: 8, kind: 'duplicates', from_key: 'cond-0010', to_key: 'cond-0004' },
  ],
  // Every non-member endpoint of the links above: the blocker benching
  // cond-0005 and a context neighbour with relates/duplicates edges.
  external: [
    { ...EXT, blocking: ['cond-0005'] },
    { ...EXT2, blocking: [] },
  ],
  progress: { total: 4, open: 3, terminal: 1, resolved: 0, claimed: 1, frontier: 1 },
}

const GRAPH_PROJECTION = {
  root: MAP,
  nodes: [
    { ...MAP, depth: 0, parent_keys: [], child_count: 1 },
    { ...T_RESEARCH, kind: 'milestone', depth: 1, parent_keys: ['cond-0001'], child_count: 1 },
    { ...T_MIGRATE, kind: 'task', depth: 2, parent_keys: ['cond-0002'], child_count: 0 },
  ],
  external: [EXT, EXT2],
  links: [
    { id: 21, kind: 'part-of', from_key: 'cond-0002', to_key: 'cond-0001' },
    { id: 22, kind: 'part-of', from_key: 'cond-0005', to_key: 'cond-0002' },
    { id: 23, kind: 'blocks', from_key: 'cond-0009', to_key: 'cond-0005' },
    { id: 24, kind: 'relates', from_key: 'cond-0010', to_key: 'cond-0002' },
  ],
  bounds: { max_depth: 8, max_nodes: 300, truncated: false, reasons: [] },
  stats: { nodes: 3, descendants: 2, external: 2, links: 4, depth: 2 },
}

const FACETS = {
  project_id: 'cao-system',
  labels: [
    { label: 'effort:deploy', total: 5, open: 4 },
    { label: 'wayfinder:map', total: 1, open: 1 },
    { label: 'wayfinder:research', total: 1, open: 1 },
  ],
  unlabeled: 2,
  unlabeled_open: 2,
}

describe('Wayfinder view', () => {
  const mockFetch = vi.fn()
  const calls: Array<{ url: string; method: string; body: any }> = []
  /** Per-test overrides keyed by `${method} ${path-prefix}`. */
  let routeOverrides: Record<string, (body?: any) => { status: number; data: unknown }>

  function respond(url: string, method: string): unknown {
    const path = url.split('?')[0]
    if (path === '/tracker/vocabulary') return VOCAB
    if (path === '/tracker/projects/cao-system/dashboard') {
      return {
        project_id: 'cao-system',
        issues: { open: 4, in_progress: 0, favorites: [], urgent: [], recent: [] },
        sessions: { total: 0, active: 0, historical: 0, recent: [] },
      }
    }
    if (path === '/tracker/projects/cao-system/sessions') {
      return { project_id: 'cao-system', total: 0, active: 0, historical: 0, sessions: [] }
    }
    if (path === '/tracker/projects/cao-system/labels') return FACETS
    if (path === '/tracker/projects/cao-system/options') {
      const params = new URLSearchParams(url.split('?')[1] ?? '')
      const field = params.get('field') ?? 'label'
      const options = field === 'label'
        ? FACETS.labels.map(item => ({ value: item.label, total: item.total, open: item.open }))
        : []
      return { project_id: 'cao-system', field, query: params.get('q') ?? '', matching_total: options.length, options }
    }
    if (path === '/tracker/projects/cao-system') return PROJECT
    if (path === '/tracker/projects') return [PROJECT]
    if (path === '/tracker/issues/cond-0001/map') return PROJECTION
    if (path === '/tracker/issues/cond-0001/graph') return GRAPH_PROJECTION
    if (path === '/tracker/issues' && method === 'GET') {
      const params = new URLSearchParams(url.split('?')[1] ?? '')
      let rows = ALL_ISSUES
      if (params.get('label') === 'wayfinder:map') rows = [MAP]
      if (params.getAll('without_label').includes('wayfinder:research')) {
        rows = rows.filter(row => !row.labels.includes('wayfinder:research'))
      }
      if (params.get('unlabeled') === 'true') rows = [T_MIGRATE, EXT]
      return { total: rows.length, limit: 50, offset: 0, issues: rows }
    }
    const detail = ALL_ISSUES.find(i => path === `/tracker/issues/${i.key}`)
    if (detail) return detail
    return {}
  }

  beforeEach(() => {
    calls.length = 0
    routeOverrides = {}
    resetLastSigma()
    useStore.setState({ snackbar: null })
    try {
      window.history.replaceState(null, '', '/')
    } catch { /* */ }
    mockFetch.mockImplementation((url: string, opts?: RequestInit) => {
      const method = opts?.method ?? 'GET'
      const body = opts?.body ? JSON.parse(opts.body as string) : undefined
      calls.push({ url, method, body })
      const path = url.split('?')[0]
      // Exact path match — a prefix match would let /tracker/issues/cond-0001
      // shadow /tracker/issues/cond-0001/map.
      const hit = routeOverrides[`${method} ${path}`]
      if (hit) {
        const { status, data } = hit(body)
        return Promise.resolve({
          ok: status >= 200 && status < 300,
          status,
          statusText: status === 200 ? 'OK' : 'Conflict',
          json: () => Promise.resolve(data),
        })
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: () => Promise.resolve(respond(url, method)),
      })
    })
    vi.stubGlobal('fetch', mockFetch)
  })

  afterEach(() => vi.restoreAllMocks())

  async function openIssues() {
    fireEvent.click(await screen.findByRole('tab', { name: /Issues/ }))
  }

  async function openWayfinder() {
    render(<ProjectsPanel />)
    await openIssues()
    fireEvent.click(await screen.findByRole('tab', { name: 'Wayfinder' }))
  }

  async function openGraph() {
    render(<ProjectsPanel />)
    await openIssues()
    fireEvent.click(await screen.findByRole('tab', { name: 'Graph' }))
  }

  it('opens a root-scoped transitive hierarchy and persists the graph deep link', async () => {
    await openGraph()
    const hierarchy = await screen.findByLabelText('Issue hierarchy')
    expect(within(hierarchy).getByText('Find the deploy path')).toBeInTheDocument()
    expect(within(hierarchy).getByText('research providers')).toBeInTheDocument()
    expect(within(hierarchy).getByText('migrate the store')).toBeInTheDocument()
    expect(screen.getByText('2 descendants · 2 related · 4 links · depth 2')).toBeInTheDocument()
    await waitFor(() => {
      expect(window.location.search).toContain('view=graph')
      expect(window.location.search).toContain('root=cond-0001')
    })

    const sigma = getLastSigma()
    expect(sigma.graph.order).toBe(3)
    expect(sigma.graph.size).toBe(2)
    expect(sigma.graph.getEdgeAttributes('cond-0005', 'cond-0002').kind).toBe('part-of')
    expect(sigma.settings.defaultDrawNodeHover).toBeTypeOf('function')
  })

  it('renders each expanded subtree contiguously when the API returns breadth-first nodes', async () => {
    routeOverrides['GET /tracker/issues/cond-0001/graph'] = () => ({
      status: 200,
      data: {
        ...GRAPH_PROJECTION,
        nodes: [
          { ...MAP, depth: 0, parent_keys: [], child_count: 2 },
          { ...T_RESEARCH, kind: 'story', depth: 1, parent_keys: ['cond-0001'], child_count: 1 },
          { ...T_GRILL, kind: 'story', depth: 1, parent_keys: ['cond-0001'], child_count: 0 },
          { ...T_MIGRATE, kind: 'task', depth: 2, parent_keys: ['cond-0002'], child_count: 0 },
        ],
        links: [
          { id: 21, kind: 'part-of', from_key: 'cond-0002', to_key: 'cond-0001' },
          { id: 22, kind: 'part-of', from_key: 'cond-0003', to_key: 'cond-0001' },
          { id: 23, kind: 'part-of', from_key: 'cond-0005', to_key: 'cond-0002' },
        ],
        stats: { nodes: 4, descendants: 3, external: 2, links: 3, depth: 2 },
      },
    })

    await openGraph()
    const hierarchy = await screen.findByLabelText('Issue hierarchy')
    const titles = Array.from(hierarchy.querySelectorAll('button.min-w-0')).map(row => row.textContent)
    expect(titles).toEqual([
      'cond-0001Find the deploy path',
      'cond-0002research providers',
      'cond-0005migrate the store',
      'cond-0003grill the operator',
    ])
  })

  it('switches to the relationship graph with materialized external context', async () => {
    await openGraph()
    await screen.findByLabelText('Issue hierarchy')
    fireEvent.click(screen.getByRole('tab', { name: 'Relationships' }))
    const relationships = await screen.findByLabelText('Issue relationships')
    expect(within(relationships).getByText('blocks')).toBeInTheDocument()
    expect(within(relationships).getByText('relates to')).toBeInTheDocument()
    await waitFor(() => expect(getLastSigma().graph.order).toBe(5))
    expect(getLastSigma().graph.getEdgeAttributes('cond-0009', 'cond-0005').kind).toBe('blocks')
  })

  it('shows blocker sequencing as staged parallel work tracks', async () => {
    routeOverrides['GET /tracker/issues/cond-0001/graph'] = () => ({
      status: 200,
      data: {
        ...GRAPH_PROJECTION,
        nodes: [
          { ...MAP, depth: 0, parent_keys: [], child_count: 3 },
          { ...T_RESEARCH, kind: 'story', depth: 1, parent_keys: ['cond-0001'], child_count: 0 },
          { ...T_GRILL, kind: 'story', depth: 1, parent_keys: ['cond-0001'], child_count: 0 },
          { ...T_MIGRATE, kind: 'task', depth: 1, parent_keys: ['cond-0001'], child_count: 0 },
        ],
        external: [{ ...EXT, status: 'closed' }],
        links: [
          { id: 31, kind: 'blocks', from_key: EXT.key, to_key: T_RESEARCH.key },
          { id: 32, kind: 'blocks', from_key: T_RESEARCH.key, to_key: T_MIGRATE.key },
          { id: 33, kind: 'blocks', from_key: T_GRILL.key, to_key: T_MIGRATE.key },
        ],
        stats: { nodes: 4, descendants: 3, external: 1, links: 3, depth: 1 },
      },
    })

    await openGraph()
    fireEvent.click(await screen.findByRole('tab', { name: 'Dependencies' }))
    const lanes = await screen.findByLabelText('Dependency work tracks')
    expect(within(lanes).getByText('Parallel work')).toBeInTheDocument()
    expect(within(lanes).getByText('Integration')).toBeInTheDocument()
    expect(within(lanes).getByText('2 open blocker links')).toBeInTheDocument()
    expect(within(lanes).getByText('quota repair')).toBeInTheDocument()
    expect(within(lanes).getByText('research providers')).toBeInTheDocument()
    expect(within(lanes).getByText('grill the operator')).toBeInTheDocument()
    expect(within(lanes).getByText('migrate the store')).toBeInTheDocument()
    expect(within(lanes).getByText('prerequisites cleared')).toBeInTheDocument()
    expect(within(lanes).getByText('external')).toBeInTheDocument()

    const sigma = getLastSigma()
    expect(sigma.graph.order).toBe(4)
    expect(sigma.graph.size).toBe(3)
    expect(sigma.graph.getEdgeAttributes('cond-0009', 'cond-0002').kind).toBe('blocks')
  })

  it('aggregates nested blockers onto a collapsed story and expands to the precise task', async () => {
    routeOverrides['GET /tracker/issues/cond-0001/graph'] = () => ({
      status: 200,
      data: {
        ...GRAPH_PROJECTION,
        nodes: [
          { ...MAP, depth: 0, parent_keys: [], child_count: 1 },
          { ...T_RESEARCH, kind: 'story', depth: 1, parent_keys: ['cond-0001'], child_count: 1 },
          { ...T_MIGRATE, kind: 'task', depth: 2, parent_keys: ['cond-0002'], child_count: 0 },
        ],
        external: [EXT],
        links: [
          { id: 41, kind: 'part-of', from_key: T_RESEARCH.key, to_key: MAP.key },
          { id: 42, kind: 'part-of', from_key: T_MIGRATE.key, to_key: T_RESEARCH.key },
          { id: 43, kind: 'blocks', from_key: EXT.key, to_key: T_MIGRATE.key },
        ],
        stats: { nodes: 3, descendants: 2, external: 1, links: 3, depth: 2 },
      },
    })

    await openGraph()
    fireEvent.click(await screen.findByRole('tab', { name: 'Dependencies' }))
    const lanes = await screen.findByLabelText('Dependency work tracks')
    expect(within(lanes).getByText('research providers')).toBeInTheDocument()
    expect(within(lanes).queryByText('migrate the store')).not.toBeInTheDocument()
    expect(getLastSigma().graph.getEdgeAttributes(EXT.key, T_RESEARCH.key).kind).toBe('blocks')

    fireEvent.click(within(lanes).getByRole('button', { name: 'Expand nested scope for cond-0002' }))
    expect(await within(lanes).findByText('migrate the store')).toBeInTheDocument()
    await waitFor(() => expect(getLastSigma().graph.getEdgeAttributes(EXT.key, T_MIGRATE.key).kind).toBe('blocks'))
  })

  it('collapses descendants and opens the shared editable issue detail', async () => {
    await openGraph()
    const hierarchy = await screen.findByLabelText('Issue hierarchy')
    fireEvent.click(within(hierarchy).getByRole('button', { name: 'Collapse cond-0002' }))
    expect(within(hierarchy).queryByText('migrate the store')).not.toBeInTheDocument()
    await waitFor(() => expect(getLastSigma().graph.order).toBe(2))

    fireEvent.click(within(hierarchy).getByText('research providers'))
    const detail = await screen.findByTestId('issue-graph-detail')
    expect(await within(detail).findByLabelText('Bug title')).toHaveValue('research providers')
    expect(screen.getByTestId('issue-graph-view')).toBeInTheDocument()
  })

  it('switches between List and Wayfinder views and persists it in the URL', async () => {
    render(<ProjectsPanel />)
    await openIssues()
    expect(await screen.findByText('cond-0001')).toBeInTheDocument()  // list view rows
    fireEvent.click(await screen.findByRole('tab', { name: 'Wayfinder' }))
    expect(await screen.findByRole('listbox', { name: 'Wayfinder maps' })).toBeInTheDocument()
    await waitFor(() => expect(window.location.search).toContain('view=wayfinder'))
    fireEvent.click(screen.getByRole('tab', { name: 'List' }))
    expect(await screen.findByText('research providers')).toBeInTheDocument()
  })

  it('discovers labels with counts and filters by exact label', async () => {
    render(<ProjectsPanel />)
    await openIssues()
    expect(screen.queryByText('effort:deploy')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Advanced filters/ }))
    fireEvent.focus(screen.getByRole('combobox', { name: 'Filter labels' }))
    const option = await screen.findByRole('option', { name: /effort:deploy.*4 open.*5 total/ })
    fireEvent.click(option)
    await waitFor(() => expect(calls.some(c => c.url.includes('label=effort%3Adeploy'))).toBe(true))
    await waitFor(() => expect(window.location.search).toContain('label=effort%3Adeploy'))
  })

  it('searches excluded labels and persists the exclusion in the URL', async () => {
    render(<ProjectsPanel />)
    await openIssues()
    fireEvent.click(screen.getByRole('button', { name: /Advanced filters/ }))
    fireEvent.focus(screen.getByRole('combobox', { name: 'Exclude labels' }))
    fireEvent.click(await screen.findByRole('option', { name: /wayfinder:research/ }))
    await waitFor(() => expect(calls.some(c => c.url.includes('without_label=wayfinder%3Aresearch'))).toBe(true))
    await waitFor(() => expect(window.location.search).toContain('without_label=wayfinder%3Aresearch'))
    expect(screen.queryByText('research providers')).not.toBeInTheDocument()
  })

  it('offers the unlabeled bucket with its count', async () => {
    render(<ProjectsPanel />)
    await openIssues()
    fireEvent.click(screen.getByRole('button', { name: /Advanced filters/ }))
    fireEvent.click(screen.getByLabelText('Only items without labels'))
    await waitFor(() => expect(calls.some(c => c.url.includes('unlabeled=true'))).toBe(true))
    expect(await screen.findByText('migrate the store')).toBeInTheDocument()
    expect(screen.queryByText('research providers')).not.toBeInTheDocument()
  })

  it('lists maps and renders the selected map from the projection', async () => {
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    expect(await screen.findByText('Destination & notes')).toBeInTheDocument()
    expect(await screen.findByText(/Decide the cutover strategy/)).toBeInTheDocument()
    expect(screen.getByText('1/4 done · 1 claimed · 1 on the frontier')).toBeInTheDocument()
    await waitFor(() => expect(window.location.search).toContain('map=cond-0001'))
  })

  it('shows the frontier ordered, with claim controls', async () => {
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    const frontier = await screen.findByLabelText('Frontier')
    expect(within(frontier).getByText('research providers')).toBeInTheDocument()
    expect(within(frontier).getByRole('button', { name: 'Claim cond-0002' })).toBeInTheDocument()
    // Claimed and blocked tickets are NOT on the frontier list.
    expect(within(frontier).queryByText('grill the operator')).not.toBeInTheDocument()
    expect(within(frontier).queryByText('migrate the store')).not.toBeInTheDocument()
  })

  it('renders children with their state as text, plus every external endpoint', async () => {
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    const list = await screen.findByTestId('map-children')
    expect(within(list).getByText('claimed by terra')).toBeInTheDocument()
    expect(within(list).getByText('blocked by cond-0009')).toBeInTheDocument()
    expect(within(list).getByText('frontier')).toBeInTheDocument()
    const external = await screen.findByTestId('map-external')
    expect(screen.getByText('External links (2)')).toBeInTheDocument()
    // The blocker and the context neighbour both render, each with its
    // relationship and direction as text. The benching `blocks` phrase is the
    // red one — that is the explicit external-blocker marker.
    expect(within(external).getByText('quota repair')).toBeInTheDocument()
    expect(within(external).getByText('blocks cond-0005').className).toContain('text-red-300')
    expect(within(external).getByText('prior art survey')).toBeInTheDocument()
    expect(within(external).getByText('relates to cond-0002').className).toContain('text-gray-400')
    expect(within(external).getByText('duplicates cond-0004')).toBeInTheDocument()
  })

  it('builds a directed graph with per-kind edge colors and arrow heads', async () => {
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    await screen.findByTestId('map-children')
    const sigma = getLastSigma()
    expect(sigma).toBeTruthy()
    const graph = sigma.graph
    expect(graph.order).toBe(7)  // map + 4 children + both external endpoints
    expect(graph.size).toBe(8)   // every projection link has both endpoints
    const partOf = graph.getEdgeAttributes('cond-0002', 'cond-0001')
    expect(partOf.kind).toBe('part-of')
    expect(partOf.type).toBe('arrow')
    const blocks = graph.getEdgeAttributes('cond-0009', 'cond-0005')
    expect(blocks.kind).toBe('blocks')
    expect(blocks.color).not.toBe(partOf.color)
    // Links to non-blocker external endpoints render too — nothing is dropped
    // because its endpoint fell outside the member set.
    expect(graph.getEdgeAttributes('cond-0002', 'cond-0010').kind).toBe('relates')
    expect(graph.getEdgeAttributes('cond-0010', 'cond-0004').kind).toBe('duplicates')
    // …and the node data keeps which externals actually bench a member.
    expect(graph.getNodeAttributes('cond-0009').blocking).toEqual(['cond-0005'])
    expect(graph.getNodeAttributes('cond-0010').blocking).toEqual([])
    // The legend names every state and every edge kind as text.
    const legend = screen.getByTestId('map-graph-legend')
    for (const word of ['map', 'frontier', 'blocked', 'claimed', 'terminal', 'external', 'blocks →', 'part-of →']) {
      expect(within(legend).getByText(word)).toBeInTheDocument()
    }
  })

  it('selecting a graph node opens the editable issue detail below the map', async () => {
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    await screen.findByTestId('map-children')
    getLastSigma().emit('clickNode', { node: 'cond-0003' })
    const detail = await screen.findByTestId('wayfinder-detail')
    expect(await within(detail).findByLabelText('Bug title')).toHaveValue('grill the operator')
    // The map is still on screen — context is not lost.
    expect(screen.getByTestId('map-view')).toBeInTheDocument()
  })

  it('selecting a row in the accessible list opens the same detail', async () => {
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    const list = await screen.findByTestId('map-children')
    fireEvent.click(within(list).getByText('migrate the store'))
    const detail = await screen.findByTestId('wayfinder-detail')
    expect(await within(detail).findByLabelText('Bug title')).toHaveValue('migrate the store')
  })

  it('a claim conflict surfaces the observed owner from the typed 409', async () => {
    // The structured field, not the prose message, carries the owner — the
    // UI must read the typed detail, not hope the message contains it.
    routeOverrides['POST /tracker/issues/cond-0002/claim'] = () => ({
      status: 409,
      data: { detail: { message: 'cond-0002 could not be claimed', code: 'conflict', observed_assignee: 'terra' } },
    })
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    const frontier = await screen.findByLabelText('Frontier')
    fireEvent.click(within(frontier).getByRole('button', { name: 'Claim cond-0002' }))
    await waitFor(() =>
      expect(useStore.getState().snackbar?.message).toBe('cond-0002 is already claimed by terra'),
    )
    expect(useStore.getState().snackbar?.type).toBe('error')
  })

  it('claim and unclaim hit the atomic endpoints, not an assignee PATCH', async () => {
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    const frontier = await screen.findByLabelText('Frontier')
    fireEvent.click(within(frontier).getByRole('button', { name: 'Claim cond-0002' }))
    await waitFor(() =>
      expect(calls.some(c => c.url === '/tracker/issues/cond-0002/claim' && c.method === 'POST')).toBe(true),
    )
    expect(calls.find(c => c.url.endsWith('/claim'))!.body).toEqual({ claimant: 'dashboard' })

    // The claimed row (terra) offers Unclaim; it posts to /unclaim.
    const list = await screen.findByTestId('map-children')
    fireEvent.click(within(list).getByRole('button', { name: /Unclaim cond-0003/ }))
    await waitFor(() =>
      expect(calls.some(c => c.url === '/tracker/issues/cond-0003/unclaim' && c.method === 'POST')).toBe(true),
    )
    expect(calls.some(c => c.method === 'PATCH' && c.body && 'assignee' in c.body)).toBe(false)
  })

  it('a stale map-body edit preserves the draft and retries after re-read', async () => {
    // First PATCH: the map moved (someone else saved) — 409 with the current
    // version. The re-read returns the fresh version; the retry succeeds.
    let patchCount = 0
    routeOverrides['PATCH /tracker/issues/cond-0001'] = (body: any) => {
      patchCount += 1
      if (patchCount === 1) {
        return {
          status: 409,
          data: { detail: { message: 'changed', code: 'conflict', current_updated_at: '2026-08-11T00:00:00Z' } },
        }
      }
      return { status: 200, data: { ...MAP, body: body.body, updated_at: '2026-08-11T00:00:00Z' } }
    }
    routeOverrides['GET /tracker/issues/cond-0001'] = () => ({
      status: 200,
      data: { ...MAP, body: 'someone elses edit', updated_at: '2026-08-11T00:00:00Z' },
    })

    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    const mapView = await screen.findByTestId('map-view')
    fireEvent.click(within(mapView).getByRole('button', { name: 'Edit' }))
    const textarea = await screen.findByLabelText('Map body')
    fireEvent.change(textarea, { target: { value: '## Destination\n\nMy draft' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    // The conflict banner names the current version; the draft survives.
    const banner = await screen.findByTestId('map-body-conflict')
    expect(banner).toHaveTextContent('2026-08-11T00:00:00Z')
    expect(screen.getByLabelText('Map body')).toHaveValue('## Destination\n\nMy draft')
    const firstPatch = calls.filter(c => c.method === 'PATCH')[0]
    expect(firstPatch.body.expected_updated_at).toBe('2026-08-10T00:00:00Z')

    fireEvent.click(within(banner).getByRole('button', { name: /Re-read & retry/ }))
    await waitFor(() => expect(patchCount).toBe(2))
    const retry = calls.filter(c => c.method === 'PATCH')[1]
    expect(retry.body.body).toBe('## Destination\n\nMy draft')
    expect(retry.body.expected_updated_at).toBe('2026-08-11T00:00:00Z')
  })

  it('renders links directionally in the detail and navigates on click', async () => {
    routeOverrides['GET /tracker/issues/cond-0005'] = () => ({
      status: 200,
      data: {
        ...T_MIGRATE,
        links: [
          { id: 4, kind: 'part-of', from_key: 'cond-0005', to_key: 'cond-0001' },
          { id: 5, kind: 'blocks', from_key: 'cond-0009', to_key: 'cond-0005' },
        ],
      },
    })
    routeOverrides['GET /tracker/issues/cond-0001'] = () => ({
      status: 200,
      data: {
        ...MAP,
        links: [{ id: 4, kind: 'part-of', from_key: 'cond-0005', to_key: 'cond-0001' }],
      },
    })
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    const list = await screen.findByTestId('map-children')
    fireEvent.click(within(list).getByText('migrate the store'))
    const detail = await screen.findByTestId('wayfinder-detail')
    expect(await within(detail).findByText('part of')).toBeInTheDocument()
    expect(within(detail).getByText('blocked by')).toBeInTheDocument()
    // Clicking the map key navigates the detail to the map, which reads
    // "contains" from the other side of the same edge.
    fireEvent.click(within(detail).getByRole('button', { name: 'Open cond-0001' }))
    expect(await within(detail).findByText('contains')).toBeInTheDocument()
  })

  it('label edits go out as add/remove deltas, never a full replacement', async () => {
    routeOverrides['GET /tracker/issues/cond-0005'] = () => ({
      status: 200,
      data: { ...T_MIGRATE, labels: ['needs-triage', 'bug'] },
    })
    await openWayfinder()
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    const list = await screen.findByTestId('map-children')
    fireEvent.click(within(list).getByText('migrate the store'))
    const detail = await screen.findByTestId('wayfinder-detail')
    const labels = await within(detail).findByLabelText('Labels')
    fireEvent.click(within(detail).getByRole('button', { name: 'Remove needs-triage' }))
    fireEvent.change(labels, { target: { value: 'ready-for-agent' } })
    fireEvent.click(await within(detail).findByRole('option', { name: 'Create “ready-for-agent”' }))
    // The delta is two changes (one add, one remove), not one replacement.
    fireEvent.click(within(detail).getByRole('button', { name: /Save 2 changes/ }))
    await waitFor(() => expect(calls.some(c => c.method === 'PATCH')).toBe(true))
    const patch = calls.find(c => c.method === 'PATCH')!
    expect(patch.body.add_labels).toEqual(['ready-for-agent'])
    expect(patch.body.remove_labels).toEqual(['needs-triage'])
    expect(patch.body).not.toHaveProperty('labels')
  })

  it('shows the empty state when a project has no maps', async () => {
    routeOverrides['GET /tracker/issues'] = () => ({ status: 200, data: { total: 0, limit: 50, offset: 0, issues: [] } })
    await openWayfinder()
    expect(await screen.findByTestId('wayfinder-empty')).toHaveTextContent('wayfinder:map')
  })

  // Back/Forward restoration: the URL carries the whole view state, so
  // traversing history must restore it exactly — and push NOTHING, or the
  // forward stack is truncated by a duplicate of the entry just restored.
  it('Back/Forward traverses view, map and detail state without duplicate entries', async () => {
    render(<ProjectsPanel />)
    await screen.findByText('CAO System')
    await waitFor(() => expect(window.location.search).toContain('project=cao-system'))
    const depth = window.history.length

    // home → issues → wayfinder → map → ticket detail: one history entry each.
    await openIssues()
    fireEvent.click(await screen.findByRole('tab', { name: 'Wayfinder' }))
    await waitFor(() => expect(window.location.search).toContain('view=wayfinder'))
    fireEvent.click(await screen.findByRole('option', { name: /Find the deploy path/ }))
    await waitFor(() => expect(window.location.search).toContain('map=cond-0001'))
    const children = await screen.findByTestId('map-children')
    fireEvent.click(within(children).getByText('migrate the store'))
    await waitFor(() => expect(window.location.search).toContain('key=cond-0005'))
    expect(await screen.findByTestId('wayfinder-detail')).toBeInTheDocument()
    expect(window.history.length).toBe(depth + 4)

    // Back: each step restores exactly the earlier state; the entry count
    // never grows (a duplicate push here is the regression this pins).
    window.history.back()
    await waitFor(() => expect(window.location.search).not.toContain('key='))
    expect(window.history.length).toBe(depth + 4)
    await waitFor(() => expect(screen.queryByTestId('wayfinder-detail')).not.toBeInTheDocument())
    expect(screen.getByTestId('map-view')).toBeInTheDocument()

    window.history.back()
    await waitFor(() => expect(window.location.search).not.toContain('map='))
    expect(screen.queryByTestId('map-view')).not.toBeInTheDocument()
    expect(screen.getByRole('listbox', { name: 'Wayfinder maps' })).toBeInTheDocument()

    window.history.back()
    await waitFor(() => expect(window.location.search).not.toContain('view=wayfinder'))
    expect(screen.queryByRole('listbox', { name: 'Wayfinder maps' })).not.toBeInTheDocument()
    expect(await screen.findByText('research providers')).toBeInTheDocument()

    // Back once more returns to the project Home before leaving the project.
    window.history.back()
    await waitFor(() => expect(window.location.search).not.toContain('section=issues'))
    expect(await screen.findByTestId('project-home')).toBeInTheDocument()

    // The bare entry the session started from: every param clears, project
    // included — restoring it must not push a normalized duplicate either.
    window.history.back()
    await waitFor(() => expect(window.location.search).toBe(''))
    expect(window.history.length).toBe(depth + 4)
    expect(await screen.findByText(/Select a project/)).toBeInTheDocument()

    // Forward: the whole chain replays — a duplicate push would have
    // truncated it.
    window.history.forward()
    await waitFor(() => expect(window.location.search).toContain('project=cao-system'))
    expect(await screen.findByTestId('project-home')).toBeInTheDocument()
    window.history.forward()
    await waitFor(() => expect(window.location.search).toContain('section=issues'))
    expect(await screen.findByText('research providers')).toBeInTheDocument()
    window.history.forward()
    await waitFor(() => expect(window.location.search).toContain('view=wayfinder'))
    window.history.forward()
    await waitFor(() => expect(window.location.search).toContain('map=cond-0001'))
    expect(await screen.findByTestId('map-view')).toBeInTheDocument()
    window.history.forward()
    await waitFor(() => expect(window.location.search).toContain('key=cond-0005'))
    const detail = await screen.findByTestId('wayfinder-detail')
    expect(await within(detail).findByLabelText('Bug title')).toHaveValue('migrate the store')
  })

  it('Back/Forward restores label and unlabeled filters', async () => {
    render(<ProjectsPanel />)
    await screen.findByText('CAO System')
    await waitFor(() => expect(window.location.search).toContain('project=cao-system'))
    const depth = window.history.length

    await openIssues()
    fireEvent.click(screen.getByRole('button', { name: /Advanced filters/ }))
    fireEvent.focus(screen.getByRole('combobox', { name: 'Filter labels' }))
    fireEvent.click(await screen.findByRole('option', { name: /effort:deploy/ }))
    await waitFor(() => expect(window.location.search).toContain('label=effort%3Adeploy'))
    fireEvent.click(screen.getByLabelText('Only items without labels'))
    await waitFor(() => expect(window.location.search).toContain('unlabeled=1'))
    expect(window.location.search).not.toContain('label=')
    expect(window.history.length).toBe(depth + 3)

    // Back to the label entry: unlabeled clears, the label filter returns,
    // and the list re-queries with it.
    const labeledCalls = () => calls.filter(c => c.url.includes('label=effort%3Adeploy')).length
    const before = labeledCalls()
    window.history.back()
    await waitFor(() => expect(window.location.search).toContain('label=effort%3Adeploy'))
    expect(window.location.search).not.toContain('unlabeled=')
    expect(window.history.length).toBe(depth + 3)
    await waitFor(() => expect(labeledCalls()).toBeGreaterThan(before))

    // Back to the unfiltered entry, then forward through both again.
    window.history.back()
    await waitFor(() => expect(window.location.search).not.toContain('label='))
    window.history.forward()
    await waitFor(() => expect(window.location.search).toContain('label=effort%3Adeploy'))
    window.history.forward()
    await waitFor(() => expect(window.location.search).toContain('unlabeled=1'))
    expect(await screen.findByText('migrate the store')).toBeInTheDocument()
    expect(screen.queryByText('research providers')).not.toBeInTheDocument()
  })
})
