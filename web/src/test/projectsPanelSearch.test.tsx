import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act, within } from '@testing-library/react'
import { ProjectsPanel } from '../components/ProjectsPanel'
import { RankedSearchResponse, TrackerIssue } from '../api'

/**
 * M1.5 wiring proofs (§12.3): the panel switches between the flat list and
 * the ranked-search route on the debounced query, cancels superseded
 * searches, forwards the active structured filters, restarts pagination when
 * the tracker's content_clock moves under an established offset, and lands a
 * winning-comment hit on the anchored comment inside ItemDetail — where the
 * importance toggle PATCHes set and clear.
 *
 * These go through ProjectsPanel rather than only the result component: an
 * ingredient test cannot prove the call site consumes it.
 */

const VOCAB = {
  statuses: ['open', 'triage', 'in-progress', 'blocked', 'resolved', 'closed', 'wontfix', 'duplicate'],
  terminal_statuses: ['closed', 'duplicate', 'wontfix'],
  severities: ['P0', 'P1', 'P2', 'P3', 'P4', 'unset'],
  scope_kinds: ['path', 'session', 'git_remote', 'project_id'],
  link_kinds: ['blocks', 'relates', 'duplicates', 'caused-by'],
  project_statuses: ['active', 'archived'],
  item_kinds: ['project', 'bug', 'feature', 'milestone', 'goal', 'epic', 'story', 'task'],
}

const PROJECT = {
  id: 'cao-system',
  name: 'CAO System',
  description: '',
  status: 'active',
  issue_prefix: 'cond',
  next_issue_number: 210,
  created_at: '2026-07-20T00:00:00Z',
  updated_at: '2026-08-07T00:00:00Z',
  counts: { total: 209, open: 80, by_status: { open: 80 } },
  scopes: [],
}

const HOME_DASHBOARD = {
  project_id: 'cao-system',
  issues: { open: 80, in_progress: 0, favorites: [], urgent: [], recent: [] },
  sessions: { total: 0, active: 0, historical: 0, recent: [] },
}

function issueRow(key: string, title: string): TrackerIssue {
  return {
    key,
    project_id: 'cao-system',
    kind: 'bug',
    title,
    body: '',
    status: 'open',
    severity: 'P2',
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
    observed_revision: null,
    resolution: null,
    session_name: null,
    terminal_id: null,
    source_path: null,
    duplicate_of: null,
    origin: 'cli',
    favorite: false,
    created_at: '2026-08-01T00:00:00Z',
    updated_at: '2026-08-01T00:00:00Z',
    closed_at: null,
    comments: [],
    events: [],
    links: [],
  }
}

const LIST_ISSUE = issueRow('cond-0700', 'event-mirror lock contention logs a traceback')

const DETAIL_ISSUE = {
  ...issueRow('cond-0638', 'ranked lexical search service with shared filters'),
  comments: [
    { id: 41, author: 'codex:lane', body: 'routine progress note', important: false, created_at: '2026-08-21T10:00:00Z' },
    { id: 77, author: 'colin', body: 'the lease deadlock cleared after restart', important: true, created_at: '2026-08-21T11:00:00Z' },
  ],
}

export function searchResponse(overrides?: Partial<RankedSearchResponse>): RankedSearchResponse {
  return {
    query: 'deadlock',
    scope: { project_ids: ['cao-system'], all_projects: false, subtree_roots: [], subtree_closure_size: 0 },
    mode_requested: 'lexical',
    mode_effective: 'lexical',
    degradation: {
      requested_mode: 'lexical',
      effective_mode: 'lexical',
      reasons: [],
      lanes: {
        'issue-bm25': { available: true },
        'comment-bm25': { available: true },
        exact: { available: true },
        'semantic-issue': { available: false, reason: 'installs at M2' },
        'semantic-comment': { available: false, reason: 'installs at M2' },
      },
    },
    generations: { schema_version: 1, document_schema_version: 1, content_clock: 7 },
    diagnostics: { lane_elapsed_ms: {}, total_elapsed_ms: 1 },
    total: 2,
    limit: 50,
    offset: 0,
    results: [
      {
        issue: DETAIL_ISSUE,
        rank_score: 0.0325,
        contributing_lanes: [{ lane: 'comment-bm25', rank: 1, raw_score: 3.1 }],
        matched_fields: ['comments'],
        snippets: { comments: '…lease deadlock cleared after restart…' },
        winning_comment: {
          comment_id: 77,
          important: true,
          retained_hits: 1,
          additional_comment_ids: [],
          total_matching_comments: 1,
        },
        exact_boosts: [],
        neighborhood: [],
        duplicate_chain: [],
      },
      {
        issue: issueRow('cond-0644', 'fusion lane weights'),
        rank_score: 0.0161,
        contributing_lanes: [{ lane: 'issue-bm25', rank: 1, raw_score: 5 }],
        matched_fields: ['title'],
        snippets: { title: '…fusion lane weights…' },
        winning_comment: null,
        exact_boosts: [],
        neighborhood: [],
        duplicate_chain: [],
      },
    ],
    ...overrides,
  }
}

interface RecordedCall {
  url: string
  method: string
  body: unknown
  signal: AbortSignal | null | undefined
}

describe('ProjectsPanel ranked search (M1.5)', () => {
  const calls: RecordedCall[] = []
  const searchCalls = () => calls.filter(c => c.url.startsWith('/tracker/issues/search'))
  const listCalls = () => calls.filter(c => /^\/tracker\/issues\?/.test(c.url))

  let respondImpl: (url: string, opts?: RequestInit) => unknown

  function json(value: unknown) {
    return { ok: true, status: 200, statusText: 'OK', json: () => Promise.resolve(value) }
  }

  function defaultRespond(url: string, opts?: RequestInit): unknown {
    if (url.startsWith('/tracker/issues/search')) return json(searchResponse())
    if (/^\/tracker\/issues\?/.test(url)) return json({ total: 1, limit: 50, offset: 0, issues: [LIST_ISSUE] })
    if (url.startsWith('/tracker/issues/')) return json(structuredClone(DETAIL_ISSUE))
    if (opts?.method === 'PATCH' && url.includes('/comments/')) {
      const parsed = JSON.parse(String(opts.body)) as { important: boolean }
      return json({ id: Number(url.split('/').pop()), issue_key: 'cond-0638', important: parsed.important, changed: true })
    }
    if (url === '/tracker/vocabulary') return json(VOCAB)
    if (/^\/tracker\/projects\?/.test(url)) return json([PROJECT])
    if (url === '/tracker/projects/cao-system') return json(PROJECT)
    if (url === '/tracker/projects/cao-system/dashboard') return json(HOME_DASHBOARD)
    if (url === '/tracker/projects/cao-system/sessions') return json({ project_id: 'cao-system', total: 0, active: 0, historical: 0, sessions: [] })
    return json({})
  }

  function typeQuery(text: string) {
    fireEvent.change(screen.getByLabelText('Search Bugs'), { target: { value: text } })
  }

  /** Flush pending microtasks without touching the fake clock. */
  async function flush() {
    await act(async () => {})
  }

  /** Advance the fake clock past the debounce window and flush. */
  async function settle(ms = 300) {
    await act(async () => { vi.advanceTimersByTime(ms) })
    await act(async () => {})
  }

  async function openIssuesTab() {
    await flush()
    fireEvent.click(screen.getByRole('tab', { name: /Issues/ }))
    await flush()
  }

  beforeEach(() => {
    vi.useFakeTimers()
    calls.length = 0
    window.history.replaceState(null, '', '/')
    respondImpl = defaultRespond
    vi.stubGlobal('fetch', vi.fn((url: string, opts?: RequestInit) => {
      calls.push({
        url,
        method: opts?.method ?? 'GET',
        body: opts?.body ? JSON.parse(String(opts.body)) : undefined,
        signal: opts?.signal,
      })
      return Promise.resolve(respondImpl(url, opts))
    }))
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('keeps the empty query on the existing list view and never calls search', async () => {
    render(<ProjectsPanel />)
    await openIssuesTab()
    await settle()
    expect(screen.getByText(/event-mirror lock contention/)).toBeInTheDocument()
    expect(listCalls().length).toBeGreaterThan(0)
    expect(searchCalls()).toHaveLength(0)
    expect(screen.queryByTestId('ranked-search')).not.toBeInTheDocument()
  })

  it('debounces input into exactly one ranked search request', async () => {
    render(<ProjectsPanel />)
    await openIssuesTab()
    await settle()
    calls.length = 0

    typeQuery('lease')
    await act(async () => { vi.advanceTimersByTime(100) })
    expect(searchCalls()).toHaveLength(0)
    typeQuery('lease deadlock')
    await act(async () => { vi.advanceTimersByTime(150) })
    expect(searchCalls()).toHaveLength(0)
    await settle()
    expect(searchCalls()).toHaveLength(1)
    expect(screen.getByTestId('search-degradation')).toBeInTheDocument()
    expect(screen.getByText(/ranked lexical search service/)).toBeInTheDocument()
  })

  it('aborts the superseded request and renders only the newest response', async () => {
    let releaseStale: ((value: unknown) => void) | null = null
    let searchCount = 0
    respondImpl = (url, opts) => {
      if (url.startsWith('/tracker/issues/search')) {
        searchCount += 1
        if (searchCount === 1) {
          return new Promise(resolve => { releaseStale = resolve })
        }
        return json(searchResponse())
      }
      return defaultRespond(url, opts)
    }

    render(<ProjectsPanel />)
    await openIssuesTab()
    calls.length = 0

    typeQuery('lease deadlock')
    await settle()
    expect(searchCalls()).toHaveLength(1)
    expect(searchCalls()[0].signal?.aborted).toBe(false)

    typeQuery('lease deadlocks successor')
    await settle()
    expect(searchCalls()).toHaveLength(2)
    // The superseded in-flight request was cancelled at the network layer.
    expect(searchCalls()[0].signal?.aborted).toBe(true)

    // A late resolution of the stale request changes nothing on screen.
    await act(async () => { releaseStale!(json(searchResponse())) })
    await flush()
    expect(screen.getAllByText(/fusion lane weights/).length).toBeGreaterThan(0)
  })

  it('forwards the active structured filters to the ranked search', async () => {
    render(<ProjectsPanel />)
    await openIssuesTab()
    await settle()
    calls.length = 0

    fireEvent.click(screen.getByRole('button', { name: /Advanced filters/ }))
    fireEvent.click(await act(async () => screen.getByRole('button', { name: 'blocked' })))
    typeQuery('deadlock')
    await settle()

    const url = searchCalls()[0]?.url ?? ''
    expect(url).toContain('q=deadlock')
    expect(url).toContain('project_id=cao-system')
    expect(url).toContain('kind=bug')
    expect(url).toContain('status=blocked')
    expect(url).toContain('open_only=true')
  })

  it('falls back to the list view when the query is cleared again', async () => {
    render(<ProjectsPanel />)
    await openIssuesTab()
    await settle()
    calls.length = 0
    typeQuery('deadlock')
    await settle()
    expect(searchCalls().length).toBeGreaterThan(0)
    // While the debounced query is non-empty, ranked search owns the surface:
    // the flat list is not fetched at all (§12.3).
    expect(listCalls()).toHaveLength(0)
    expect(screen.queryByText(/event-mirror lock contention/)).not.toBeInTheDocument()

    typeQuery('')
    calls.length = 0
    await settle()
    expect(searchCalls()).toHaveLength(0)
    expect(listCalls().length).toBeGreaterThan(0)
    expect(screen.getByText(/event-mirror lock contention/)).toBeInTheDocument()
    expect(screen.queryByTestId('ranked-search')).not.toBeInTheDocument()
  })

  it('restarts pagination at offset 0 when content_clock changes mid-paging', async () => {
    respondImpl = (url, opts) => {
      if (url.startsWith('/tracker/issues/search')) {
        const params = new URLSearchParams(url.split('?')[1] ?? '')
        const offset = Number(params.get('offset') ?? 0)
        const clock = offset > 0 ? 8 : 7
        return json(searchResponse({
          offset,
          total: 60,
          generations: { schema_version: 1, document_schema_version: 1, content_clock: clock },
          results: offset > 0 ? [] : searchResponse().results,
        }))
      }
      return defaultRespond(url, opts)
    }

    render(<ProjectsPanel />)
    await openIssuesTab()
    await settle()
    calls.length = 0

    typeQuery('deadlock')
    await settle()
    expect(screen.getByText(/1–2 of 60/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    await settle()

    const offsets = searchCalls().map(c => Number(new URLSearchParams(c.url.split('?')[1]).get('offset')))
    expect(offsets).toEqual([0, 50, 0])
    expect(screen.getByText(/1–2 of 60/)).toBeInTheDocument()
  })

  it('restarts pagination when only active_vector_generation changes (§12.3 keys on both)', async () => {
    // Regression pin for review finding P2: the restart contract originally
    // keyed on content_clock alone and missed a generation rollover.
    respondImpl = (url, opts) => {
      if (url.startsWith('/tracker/issues/search')) {
        const params = new URLSearchParams(url.split('?')[1] ?? '')
        const offset = Number(params.get('offset') ?? 0)
        return json(searchResponse({
          offset,
          total: 60,
          generations: {
            schema_version: 1,
            document_schema_version: 1,
            content_clock: 7,
            active_vector_generation: offset > 0 ? 'gen-2' : null,
          },
          results: offset > 0 ? [] : searchResponse().results,
        }))
      }
      return defaultRespond(url, opts)
    }

    render(<ProjectsPanel />)
    await openIssuesTab()
    await settle()
    calls.length = 0

    typeQuery('deadlock')
    await settle()
    expect(screen.getByText(/1–2 of 60/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    await settle()

    const offsets = searchCalls().map(c => Number(new URLSearchParams(c.url.split('?')[1]).get('offset')))
    expect(offsets).toEqual([0, 50, 0])
  })

  it('keeps pagination stable when neither clock nor generation moved', async () => {
    respondImpl = (url, opts) => {
      if (url.startsWith('/tracker/issues/search')) {
        const params = new URLSearchParams(url.split('?')[1] ?? '')
        const offset = Number(params.get('offset') ?? 0)
        return json(searchResponse({
          offset,
          total: 60,
          generations: { schema_version: 1, document_schema_version: 1, content_clock: 7, active_vector_generation: null },
          results: offset > 0 ? [searchResponse().results[1]] : searchResponse().results,
        }))
      }
      return defaultRespond(url, opts)
    }

    render(<ProjectsPanel />)
    await openIssuesTab()
    await settle()
    calls.length = 0

    typeQuery('deadlock')
    await settle()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    await settle()

    const offsets = searchCalls().map(c => Number(new URLSearchParams(c.url.split('?')[1]).get('offset')))
    expect(offsets).toEqual([0, 50])
    // Page two actually renders; a spurious reset would have re-shown page one.
    expect(screen.getByText(/51–51 of 60/)).toBeInTheDocument()
    expect(screen.getAllByText(/fusion lane weights/).length).toBeGreaterThan(0)
  })

  it('lands a winning-comment hit on the anchored comment and toggles importance via PATCH', async () => {
    render(<ProjectsPanel />)
    await openIssuesTab()
    await settle()

    typeQuery('deadlock')
    await settle()

    fireEvent.click(screen.getByRole('button', { name: /Open matching comment 77 on cond-0638/ }))
    await settle()

    const target = screen.getByTestId('issue-comment-77')
    expect(target.className).toContain('ring-amber-400')

    // Clear: the important comment goes back to routine through the PATCH.
    fireEvent.click(within(target).getByRole('button', { name: /Mark comment 77 routine/ }))
    await settle()
    const clearPatch = calls.find(c => c.method === 'PATCH' && c.url.endsWith('/comments/77'))
    expect(clearPatch).toBeDefined()
    expect((clearPatch!.body as { important: boolean }).important).toBe(false)

    // Set: a routine comment becomes important through the same endpoint.
    const other = screen.getByTestId('issue-comment-41')
    fireEvent.click(within(other).getByRole('button', { name: /Mark comment 41 important/ }))
    await settle()
    const setPatch = calls.find(c => c.method === 'PATCH' && c.url.endsWith('/comments/41'))
    expect(setPatch).toBeDefined()
    expect((setPatch!.body as { important: boolean }).important).toBe(true)
  })
})
