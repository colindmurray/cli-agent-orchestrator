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
        probe_contributions: [{
          label: 'failing_command',
          query: 'conduct deploy --dry-run',
          weight: 1,
          original_rank: 1,
          original_score: 0.12,
        }],
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
        probe_contributions: [{
          label: 'actual_outcome',
          query: 'the dashboard stays disconnected',
          weight: 0.5,
          original_rank: 2,
          original_score: 0.08,
        }],
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

  it('hides stale candidates immediately while the replacement draft is debouncing and loading', async () => {
    let releaseReplacement: ((value: unknown) => void) | null = null
    let similarCount = 0
    respondImpl = (url, opts) => {
      if (url === '/tracker/issues/similar') {
        similarCount += 1
        if (similarCount === 2) {
          return new Promise(resolve => { releaseReplacement = resolve })
        }
        return json(similarResponse())
      }
      return defaultRespond(url, opts)
    }

    await openBugModal()
    const title = screen.getByLabelText('Title')
    fireEvent.change(title, { target: { value: 'lock contention' } })
    await settle()
    expect(screen.getByRole('button', { name: /^Open cond-0711: / })).toBeEnabled()

    fireEvent.change(title, { target: { value: 'a different reconnect failure' } })

    // The prior answer belongs to the prior draft. It disappears in the same
    // render as the edit, before the debounce expires, so it cannot be acted on.
    expect(screen.queryByRole('button', { name: /^Open cond-0711: / })).not.toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('Updating similar issues')
    expect(similarCalls()).toHaveLength(1)

    await settle()
    expect(similarCalls()).toHaveLength(2)
    expect(screen.queryByRole('button', { name: /^Open cond-0711: / })).not.toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('Updating similar issues')

    await act(async () => { releaseReplacement!(json(similarResponse())) })
    await flush()
    expect(screen.getByRole('button', { name: /^Open cond-0711: / })).toBeEnabled()
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

  it('preserves the complete draft while inspecting a candidate, then returns and files normally', async () => {
    await openBugModal()
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'dashboard hangs on reconnect' } })
    fireEvent.change(screen.getByLabelText('Body'), { target: { value: 'The reconnect path hangs after wake.' } })
    fireEvent.change(screen.getByLabelText('Severity'), { target: { value: 'P1' } })
    fireEvent.change(screen.getByLabelText('Evidence'), { target: { value: '/tmp/reconnect.log' } })
    fireEvent.change(screen.getByLabelText('Failing command'), { target: { value: 'cao web' } })
    fireEvent.change(screen.getByLabelText('Reproduction steps'), { target: { value: '1. suspend\n2. resume' } })
    fireEvent.change(screen.getByLabelText('Expected outcome'), { target: { value: 'The dashboard reconnects' } })
    fireEvent.change(screen.getByLabelText('Actual outcome'), { target: { value: 'The dashboard stays disconnected' } })
    const favorite = screen.getByText('Show this item on the project Home dashboard').querySelector('input')!
    fireEvent.click(favorite)
    await settle()

    fireEvent.click(screen.getByRole('button', { name: /^Open cond-0711: / }))
    await settle()

    // Candidate inspection uses the normal full ItemDetail while keeping the
    // create component mounted, so every draft field survives the round trip.
    expect(screen.getByRole('dialog', { name: 'Inspect cond-0711' })).toBeInTheDocument()
    expect(calls.some(c => c.url === '/tracker/issues/cond-0711' && c.method === 'GET')).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Back to new bug draft' }))
    expect(screen.getByRole('dialog', { name: 'Log a bug against CAO System' })).toBeInTheDocument()
    expect(screen.getByLabelText('Title')).toHaveValue('dashboard hangs on reconnect')
    expect(screen.getByLabelText('Body')).toHaveValue('The reconnect path hangs after wake.')
    expect(screen.getByLabelText('Severity')).toHaveValue('P1')
    expect(screen.getByLabelText('Evidence')).toHaveValue('/tmp/reconnect.log')
    expect(screen.getByLabelText('Failing command')).toHaveValue('cao web')
    expect(screen.getByLabelText('Reproduction steps')).toHaveValue('1. suspend\n2. resume')
    expect(screen.getByLabelText('Expected outcome')).toHaveValue('The dashboard reconnects')
    expect(screen.getByLabelText('Actual outcome')).toHaveValue('The dashboard stays disconnected')
    expect(screen.getByText('Show this item on the project Home dashboard').querySelector('input')).toBeChecked()

    fireEvent.click(screen.getByRole('button', { name: /^File bug$/ }))
    await settle()
    const created = calls.filter(c => c.url === '/tracker/issues' && c.method === 'POST')
    expect(created).toHaveLength(1)
    expect(created[0].body).toMatchObject({
      title: 'dashboard hangs on reconnect',
      body: 'The reconnect path hangs after wake.',
      severity: 'P1',
      evidence: '/tmp/reconnect.log',
      failing_command: 'cao web',
      reproduction_steps: '1. suspend\n2. resume',
      expected_outcome: 'The dashboard reconnects',
      actual_outcome: 'The dashboard stays disconnected',
      favorite: true,
    })
  })

  it('opens canonical and relationship-neighbor issues directly from candidate facts', async () => {
    await openBugModal()
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'dashboard hangs on reconnect' } })
    await settle()

    fireEvent.click(screen.getByRole('button', { name: 'Inspect canonical cond-0600' }))
    await settle()
    expect(screen.getByRole('dialog', { name: 'Inspect cond-0600' })).toBeInTheDocument()
    expect(calls.some(c => c.url === '/tracker/issues/cond-0600' && c.method === 'GET')).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Back to new bug draft' }))
    await settle()
    fireEvent.click(screen.getByRole('button', { name: 'Inspect related cond-0700' }))
    await settle()
    expect(screen.getByRole('dialog', { name: 'Inspect cond-0700' })).toBeInTheDocument()
    expect(calls.some(c => c.url === '/tracker/issues/cond-0700' && c.method === 'GET')).toBe(true)
  })

  it('navigates relationships from an off-page deep-linked ItemDetail', async () => {
    respondImpl = (url, opts) => {
      if (url === '/tracker/issues/cond-0711') {
        return json({
          ...OPEN_CANDIDATE,
          links: [{ id: 1, from_key: 'cond-0711', to_key: 'cond-0700', kind: 'blocks' }],
        })
      }
      return defaultRespond(url, opts)
    }
    window.history.replaceState(null, '', '/?project=cao-system&section=issues&key=cond-0711')

    render(<ProjectsPanel />)
    await settle()
    expect(screen.getByText('Deep link: cond-0711 (not in current page/filters)')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Open cond-0700' }))
    await settle()
    expect(calls.some(c => c.url === '/tracker/issues/cond-0700' && c.method === 'GET')).toBe(true)
  })

  it('removes stale edit and delete actions immediately when relationship navigation fails to load', async () => {
    let releaseTarget: ((value: unknown) => void) | null = null
    respondImpl = (url, opts) => {
      if (url === '/tracker/issues/cond-0711') {
        return json({
          ...OPEN_CANDIDATE,
          links: [{ id: 1, from_key: 'cond-0711', to_key: 'cond-0800', kind: 'blocks' }],
        })
      }
      if (url === '/tracker/issues/cond-0800') {
        return new Promise(resolve => { releaseTarget = resolve })
      }
      return defaultRespond(url, opts)
    }
    window.history.replaceState(null, '', '/?project=cao-system&section=issues&key=cond-0711')

    render(<ProjectsPanel />)
    await settle()
    fireEvent.change(screen.getByLabelText('Bug title'), { target: { value: 'unsaved stale title' } })
    expect(screen.getByRole('button', { name: 'Save 1 change' })).toBeInTheDocument()
    expect(screen.getByTitle('Delete issue')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Open cond-0800' }))

    // Identity changes remove every action backed by cond-0711 before the new
    // GET resolves. There is no window where old draft state targets cond-0800.
    expect(screen.getByText('Loading cond-0800…')).toBeInTheDocument()
    expect(screen.queryByDisplayValue('unsaved stale title')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Save 1 change' })).not.toBeInTheDocument()
    expect(screen.queryByTitle('Delete issue')).not.toBeInTheDocument()

    await act(async () => {
      releaseTarget!({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.resolve({ detail: 'target read failed' }),
      })
    })
    await flush()

    expect(screen.getByRole('alert')).toHaveTextContent('Unable to load cond-0800')
    expect(screen.getByRole('alert')).toHaveTextContent('target read failed')
    expect(screen.queryByDisplayValue('unsaved stale title')).not.toBeInTheDocument()
    expect(calls.filter(c => c.method === 'PATCH' || c.method === 'DELETE')).toHaveLength(0)
  })

  it('keeps the newest rendered issue when rapid navigation responses resolve out of order', async () => {
    let releaseMiddle: ((value: unknown) => void) | null = null
    respondImpl = (url, opts) => {
      if (url === '/tracker/issues/cond-0711') {
        return json({
          ...OPEN_CANDIDATE,
          links: [{ id: 1, from_key: 'cond-0711', to_key: 'cond-0800', kind: 'blocks' }],
        })
      }
      if (url === '/tracker/issues/cond-0800') {
        return new Promise(resolve => { releaseMiddle = resolve })
      }
      if (url === '/tracker/issues/cond-0600') {
        return json(issueRow('cond-0600', 'newest canonical target'))
      }
      return defaultRespond(url, opts)
    }
    window.history.replaceState(null, '', '/?project=cao-system&section=issues&key=cond-0711')

    render(<ProjectsPanel />)
    await settle()
    fireEvent.click(screen.getByRole('button', { name: 'Open cond-0800' }))
    expect(screen.getByText('Loading cond-0800…')).toBeInTheDocument()

    await act(async () => {
      window.history.pushState(null, '', '/?project=cao-system&section=issues&key=cond-0600')
      window.dispatchEvent(new PopStateEvent('popstate'))
    })
    await flush()

    expect(screen.getByText('Deep link: cond-0600 (not in current page/filters)')).toBeInTheDocument()
    expect(screen.getByLabelText('Bug title')).toHaveValue('newest canonical target')
    expect(screen.getByRole('button', { name: 'Claim cond-0600' })).toBeInTheDocument()
    expect(calls.find(c => c.url === '/tracker/issues/cond-0800')?.signal?.aborted).toBe(true)

    // The abandoned middle request resolves last. It must not replace the
    // newest target or expose actions carrying the wrong issue identity.
    await act(async () => {
      releaseMiddle!(json(issueRow('cond-0800', 'late obsolete relationship target')))
    })
    await flush()

    expect(screen.getByLabelText('Bug title')).toHaveValue('newest canonical target')
    expect(screen.getByRole('button', { name: 'Claim cond-0600' })).toBeInTheDocument()
    expect(screen.queryByDisplayValue('late obsolete relationship target')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Claim cond-0800' })).not.toBeInTheDocument()
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

  it('renders probe-level audit and suppresses definitive empty text when coverage is inconclusive', async () => {
    respondImpl = (url, opts) => {
      if (url === '/tracker/issues/similar') {
        return json(similarResponse({
          total: 0,
          candidates: [],
          duplicate_expansions: [],
          mode_requested: 'hybrid',
          mode_effective: 'lexical',
          degradation: {
            requested_mode: 'hybrid',
            effective_mode: 'lexical',
            reasons: ['semantic unavailable'],
            lanes: {},
          },
          coverage: {
            status: 'inconclusive',
            complete: false,
            inconclusive: true,
            partial: false,
            probes_requested: 1,
            probes_completed: 1,
            probes_failed: 0,
            candidate_keys_seen: 0,
          },
          diagnostics: {
            similarity_duplicate_conflicts: [{
              code: 'multiple-native-duplicate-targets',
              message: 'native duplicate source has multiple canonical targets',
              duplicate_key: 'cond-0900',
              canonical_keys: ['cond-0901', 'cond-0902'],
              hit_canonical_keys: [],
            }],
          },
        }))
      }
      return defaultRespond(url, opts)
    }

    await openBugModal()
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'offline retrieval' } })
    await settle()

    expect(screen.getByTestId('similar-degraded')).toHaveTextContent('No candidates is inconclusive')
    expect(screen.getByTestId('similar-degraded')).toHaveTextContent('Filing is unaffected')
    expect(screen.getByTestId('similar-duplicate-conflict')).toHaveTextContent('cond-0900')
    expect(screen.getByTestId('similar-duplicate-conflict')).toHaveTextContent('no canonical was asserted')
    expect(screen.queryByText('No similar issues found in this project.')).not.toBeInTheDocument()
  })

  it('renders which draft probe contributed a candidate', async () => {
    await openBugModal()
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'dashboard hangs on reconnect' } })
    await settle()
    expect(screen.getAllByTestId('similar-probe-contributions')[0]).toHaveTextContent('probe failing_command')
    expect(screen.getAllByTestId('similar-probe-contributions')[0]).toHaveTextContent('conduct deploy --dry-run')
  })
})
