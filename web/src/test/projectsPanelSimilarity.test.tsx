import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { ProjectsPanel } from '../components/ProjectsPanel'
import { SimilarIssuesResponse, TrackerIssue } from '../api'

/**
 * M2.5 wiring proofs (§11/§12.3): while the create modal is open and the
 * draft is meaningful, the panel debounces one explicitly project-scoped
 * POST /tracker/issues/similar, cancels superseded probes, renders explained
 * open/terminal candidates with canonical/duplicate/relationship facts, opens
 * the normal full ItemDetail from a candidate, and — above all — never lets
 * similarity gate filing: a 500 from the probe is a visible advisory while
 * POST /tracker/issues still succeeds, with no link/duplicate/status mutation.
 *
 * These go through ProjectsPanel rather than only the candidates component:
 * an ingredient test cannot prove the call site consumes it.
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
  next_issue_number: 720,
  created_at: '2026-07-20T00:00:00Z',
  updated_at: '2026-08-07T00:00:00Z',
  counts: { total: 719, open: 80, by_status: { open: 80 } },
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

const OPEN_CANDIDATE: TrackerIssue = {
  ...issueRow('cond-0711', 'event-mirror lock contention on reconnect'),
}

const TERMINAL_CANDIDATE: TrackerIssue = {
  ...issueRow('cond-0666', 'lock contention when the mirror restarts'),
  status: 'closed',
  closed_at: '2026-08-10T00:00:00Z',
  resolution: 'fixed in the event-mirror retry loop',
}

function similarResponse(overrides?: Partial<SimilarIssuesResponse>): SimilarIssuesResponse {
  return {
    query_source: { mode: 'draft', issue_key: null, kind: 'bug' },
    query: 'dashboard hangs on reconnect',
    scope: { project_ids: ['cao-system'], all_projects: false, subtree_roots: [], subtree_closure_size: 0 },
    include_comments: false,
    limit: 5,
    total: 2,
    candidates: [
      {
        issue: OPEN_CANDIDATE,
        rank_score: 0.031,
        contributing_lanes: [{ lane: 'issue-bm25', rank: 1, raw_score: 5.2 }],
        matched_fields: ['title'],
        snippets: { title: '…event-mirror lock contention on reconnect…' },
        winning_comment: null,
        exact_boosts: [],
        // The candidate's confirmed relationship neighborhood (§12.3).
        neighborhood: [{ from_key: 'cond-0711', to_key: 'cond-0700', kind: 'blocks' }],
        duplicate_chain: [],
      },
      {
        issue: TERMINAL_CANDIDATE,
        rank_score: 0.017,
        contributing_lanes: [{ lane: 'issue-bm25', rank: 2, raw_score: 3.4 }],
        matched_fields: ['title', 'resolution'],
        snippets: { resolution: '…fixed in the event-mirror retry loop…' },
        winning_comment: null,
        exact_boosts: [],
        neighborhood: [{ from_key: 'cond-0601', to_key: 'cond-0666', kind: 'relates' }],
        // A terminal candidate that is itself a duplicate names its canonical.
        duplicate_chain: [{ canonical_key: 'cond-0600', canonical_title: 'the canonical lock issue', resolved: true }],
      },
    ],
    // Confirmed duplicates of the hits, expanded one level beside their hit.
    duplicate_expansions: [
      { duplicate_of: 'cond-0711', issue: issueRow('cond-0712', 'a confirmed duplicate report') },
    ],
    ...overrides,
  }
}

interface RecordedCall {
  url: string
  method: string
  body: any
  signal: AbortSignal | null | undefined
}

describe('ProjectsPanel pre-filing similar candidates (M2.5)', () => {
  const calls: RecordedCall[] = []
  const similarCalls = () => calls.filter(c => c.url === '/tracker/issues/similar')

  let respondImpl: (url: string, opts?: RequestInit) => unknown

  function json(value: unknown) {
    return { ok: true, status: 200, statusText: 'OK', json: () => Promise.resolve(value) }
  }

  function defaultRespond(url: string, opts?: RequestInit): unknown {
    if (url === '/tracker/issues/similar') return json(similarResponse())
    if (url === '/tracker/issues' && opts?.method === 'POST') return json(issueRow('cond-0720', 'freshly filed bug'))
    if (/^\/tracker\/issues\?/.test(url)) return json({ total: 1, limit: 50, offset: 0, issues: [LIST_ISSUE] })
    if (url.startsWith('/tracker/issues/')) return json(structuredClone(OPEN_CANDIDATE))
    if (url === '/tracker/vocabulary') return json(VOCAB)
    if (/^\/tracker\/projects\?/.test(url)) return json([PROJECT])
    if (url === '/tracker/projects/cao-system') return json(PROJECT)
    if (url === '/tracker/projects/cao-system/dashboard') return json(HOME_DASHBOARD)
    if (url === '/tracker/projects/cao-system/sessions') return json({ project_id: 'cao-system', total: 0, active: 0, historical: 0, sessions: [] })
    return json({})
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

  async function openBugModal() {
    render(<ProjectsPanel />)
    await openIssuesTab()
    fireEvent.click(screen.getByRole('button', { name: /Log bug/ }))
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

  it('debounces rapid draft edits into exactly one project-scoped probe with only declared draft fields', async () => {
    await openBugModal()
    await settle()
    // An untouched empty form is not a meaningful draft: no probe fires.
    expect(similarCalls()).toHaveLength(0)

    const title = screen.getByLabelText('Title')
    fireEvent.change(title, { target: { value: 'd' } })
    await act(async () => { vi.advanceTimersByTime(100) })
    fireEvent.change(title, { target: { value: 'dashboard hangs' } })
    await act(async () => { vi.advanceTimersByTime(150) })
    // Still inside the debounce window: nothing has been sent.
    expect(similarCalls()).toHaveLength(0)
    fireEvent.change(title, { target: { value: 'dashboard hangs on reconnect' } })
    await act(async () => { vi.advanceTimersByTime(100) })
    expect(similarCalls()).toHaveLength(0)

    await settle()
    expect(similarCalls()).toHaveLength(1)
    // Exactly the declared draft fields plus the explicit project scope —
    // no server-owned identity/status/relation fields, ever.
    expect(similarCalls()[0].method).toBe('POST')
    expect(similarCalls()[0].body).toEqual({
      draft: { kind: 'bug', title: 'dashboard hangs on reconnect', reporter: 'dashboard' },
      project_ids: ['cao-system'],
      limit: 5,
    })
  })

  it('does not query for meaningless starter content, and never sends the untouched feature template', async () => {
    render(<ProjectsPanel />)
    await openIssuesTab()
    fireEvent.click(screen.getByRole('button', { name: 'Show Features' }))
    await flush()
    fireEvent.click(screen.getByRole('button', { name: /Add feature/ }))
    await settle()
    // The pre-filled body starter alone is scaffolding, not a draft.
    expect(similarCalls()).toHaveLength(0)

    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'fleet-wide drain status' } })
    await settle()
    expect(similarCalls()).toHaveLength(1)
    expect(similarCalls()[0].body.draft.kind).toBe('feature')
    expect(similarCalls()[0].body.draft.title).toBe('fleet-wide drain status')
    // The untouched template is not sent as the body — it would match every
    // other template-bearing feature.
    expect(similarCalls()[0].body.draft.body).toBeUndefined()
    expect(similarCalls()[0].body.project_ids).toEqual(['cao-system'])
  })

  it('aborts the superseded probe and a late response cannot replace the newest results', async () => {
    let releaseStale: ((value: unknown) => void) | null = null
    let similarCount = 0
    respondImpl = (url, opts) => {
      if (url === '/tracker/issues/similar') {
        similarCount += 1
        if (similarCount === 1) {
          return new Promise(resolve => { releaseStale = resolve })
        }
        return json(similarResponse())
      }
      return defaultRespond(url, opts)
    }

    await openBugModal()
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'lock contention' } })
    await settle()
    expect(similarCalls()).toHaveLength(1)
    expect(similarCalls()[0].signal?.aborted).toBe(false)

    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'lock contention on reconnect' } })
    await settle()
    expect(similarCalls()).toHaveLength(2)
    // The superseded in-flight probe was cancelled at the network layer.
    expect(similarCalls()[0].signal?.aborted).toBe(true)
    expect(screen.getByRole('button', { name: /^Open cond-0711: / })).toBeInTheDocument()

    // A late resolution of the stale probe changes nothing on screen.
    await act(async () => {
      releaseStale!(json(similarResponse({
        candidates: [{
          ...similarResponse().candidates[0],
          issue: issueRow('cond-0999', 'stale candidate from the superseded probe'),
        }],
        duplicate_expansions: [],
      })))
    })
    await flush()
    expect(screen.queryByTestId('similar-candidate-cond-0999')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Open cond-0711: / })).toBeInTheDocument()
  })

  it('renders explained open and terminal candidates with canonical, duplicate, and relationship facts', async () => {
    await openBugModal()
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'dashboard hangs on reconnect' } })
    await settle()

    // Open and terminal candidates both render, with the terminal one marked.
    const openRow = screen.getByTestId('similar-candidate-cond-0711')
    expect(openRow).toHaveTextContent('cond-0711')
    expect(openRow).toHaveTextContent('event-mirror lock contention on reconnect')
    expect(openRow).toHaveTextContent('open')
    const terminalRow = screen.getByTestId('similar-candidate-cond-0666')
    expect(terminalRow).toHaveTextContent('closed · terminal')

    // Explanations: matched-field/lane badges and snippets.
    expect(openRow).toHaveTextContent('title')
    expect(openRow).toHaveTextContent('issue text')
    expect(openRow).toHaveTextContent('…event-mirror lock contention on reconnect…')
    expect(terminalRow).toHaveTextContent('…fixed in the event-mirror retry loop…')

    // Canonical duplicate chain for the terminal candidate.
    expect(terminalRow).toHaveTextContent('duplicate of cond-0600 — the canonical lock issue (resolved)')
    // One-level expansion: confirmed duplicates of a hit.
    expect(openRow).toHaveTextContent('Confirmed duplicates: cond-0712 — a confirmed duplicate report')
    // Confirmed relationship neighborhoods.
    expect(openRow).toHaveTextContent('blocks cond-0700')
    expect(terminalRow).toHaveTextContent('relates to cond-0601')
  })

  it('opens the normal full issue detail from a candidate and closes the create form', async () => {
    await openBugModal()
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'dashboard hangs on reconnect' } })
    await settle()

    fireEvent.click(screen.getByRole('button', { name: /^Open cond-0711: / }))
    await settle()

    // The create modal is gone and the normal ItemDetail surface opened for
    // the candidate — the deep-link path, since cond-0711 is off-page.
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByText('Deep link: cond-0711 (not in current page/filters)')).toBeInTheDocument()
    expect(calls.some(c => c.url === '/tracker/issues/cond-0711' && c.method === 'GET')).toBe(true)
  })

  it('shows a visible advisory when the probe fails and filing stays enabled with no mutation calls', async () => {
    respondImpl = (url, opts) => {
      if (url === '/tracker/issues/similar') {
        return {
          ok: false,
          status: 500,
          statusText: 'Internal Server Error',
          json: () => Promise.resolve({ detail: 'search index offline' }),
        }
      }
      return defaultRespond(url, opts)
    }

    await openBugModal()
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'dashboard hangs on reconnect' } })
    await settle()
    expect(similarCalls()).toHaveLength(1)

    // The failure is a visible advisory — and explicitly non-gating.
    const advisory = screen.getByTestId('similar-unavailable')
    expect(advisory).toHaveTextContent('Similarity check unavailable')
    expect(advisory).toHaveTextContent('search index offline')
    expect(advisory).toHaveTextContent('Filing is unaffected')

    // Normal bug filing is untouched: complete the required diagnostics and file.
    fireEvent.change(screen.getByLabelText('Reproduction steps'), { target: { value: '1. suspend\n2. resume' } })
    fireEvent.change(screen.getByLabelText('Expected outcome'), { target: { value: 'The dashboard reconnects' } })
    fireEvent.change(screen.getByLabelText('Actual outcome'), { target: { value: 'The dashboard stays disconnected' } })
    const fileButton = screen.getByRole('button', { name: /^File bug$/ })
    expect(fileButton).toBeEnabled()
    fireEvent.click(fileButton)
    await settle()

    const created = calls.filter(c => c.url === '/tracker/issues' && c.method === 'POST')
    expect(created).toHaveLength(1)
    expect(created[0].body).toMatchObject({ project_id: 'cao-system', kind: 'bug', title: 'dashboard hangs on reconnect' })

    // The advisory surface never mutates tracker state: the only writes on
    // the wire are the probe itself (re-fired as the draft kept changing)
    // and the operator's own filing — no link, duplicate, or status calls.
    expect(calls.filter(c => c.method === 'PATCH')).toHaveLength(0)
    expect(calls.filter(c => c.url.includes('/links'))).toHaveLength(0)
    expect(
      calls.filter(c =>
        c.method !== 'GET' && c.url !== '/tracker/issues' && c.url !== '/tracker/issues/similar',
      ),
    ).toHaveLength(0)
  })

  it('renders an explicit no-result state when nothing is similar', async () => {
    respondImpl = (url, opts) => {
      if (url === '/tracker/issues/similar') {
        return json(similarResponse({ total: 0, candidates: [], duplicate_expansions: [] }))
      }
      return defaultRespond(url, opts)
    }

    await openBugModal()
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'a wholly novel failure mode' } })
    await settle()

    expect(screen.getByText('No similar issues found in this project.')).toBeInTheDocument()
    expect(screen.queryByTestId('similar-unavailable')).not.toBeInTheDocument()
  })
})
