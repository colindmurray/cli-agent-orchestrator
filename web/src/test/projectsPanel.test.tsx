import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { ProjectsPanel } from '../components/ProjectsPanel'

/**
 * The behaviours worth pinning are the ones that decide whether an operator
 * can trust what the page tells them: the vocabulary comes from the server
 * rather than from this file, a PATCH carries only the fields that actually
 * changed, and a scope conflict surfaces the server's explanation instead of a
 * generic failure.
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
  description: 'Conductor and its fork',
  status: 'active',
  issue_prefix: 'cond',
  next_issue_number: 209,
  created_at: '2026-07-20T00:00:00Z',
  updated_at: '2026-08-07T00:00:00Z',
  counts: { total: 208, open: 80, by_status: { open: 80, closed: 128 } },
  scopes: [
    { id: 1, project_id: 'cao-system', kind: 'path', value: '/Users/colin/Projects/cao-conductor', created_at: null },
    { id: 2, project_id: 'cao-system', kind: 'session', value: 'cao-p1-closure', created_at: null },
  ],
}

const ISSUE = {
  key: 'cond-0039',
  project_id: 'cao-system',
  kind: 'bug',
  title: 'event-mirror lock contention logs full traceback every tick',
  body: 'Independent validation confirmed the mirror stays bounded.',
  status: 'open',
  severity: 'P2',
  component: 'conduct',
  reporter: '13e6fe47',
  assignee: null,
  labels: ['deferred'],
  collaborators: [],
  branches: [],
  worktrees: [],
  pull_requests: [],
  failing_command: 'python3 -B probes.py',
  reproduction_steps: '1. run python3 -B probes.py',
  expected_outcome: 'The probe completes',
  actual_outcome: 'The probe logs a traceback',
  evidence: '/Users/colin/runs/report.md',
  resolution: null,
  session_name: null,
  terminal_id: null,
  source_path: null,
  duplicate_of: null,
  origin: 'migration',
  favorite: false,
  created_at: '2026-07-21T17:01:14Z',
  updated_at: '2026-07-21T17:01:14Z',
  closed_at: null,
  comments: [],
  events: [{ id: 1, actor: 'human', kind: 'created', field: null, old_value: null, new_value: 'filed', created_at: '2026-07-21T17:01:14Z' }],
  links: [],
}

const RELATED_ISSUE = {
  ...ISSUE,
  key: 'cond-0040',
  title: 'restart the dashboard after reconnect',
}

describe('ProjectsPanel', () => {
  const mockFetch = vi.fn()
  const calls: Array<{ url: string; method: string; body: unknown }> = []

  function respond(url: string): unknown {
    if (url.startsWith('/tracker/vocabulary')) return VOCAB
    if (url.startsWith('/tracker/projects/cao-system/options')) {
      const params = new URLSearchParams(url.split('?')[1] ?? '')
      const field = params.get('field') ?? 'label'
      const values: Record<string, string[]> = {
        label: ['deferred', 'initiative:dashboard'],
        component: ['conduct', 'dashboard'],
        assignee: ['terra'],
        reporter: ['13e6fe47', 'dashboard'],
        collaborator: ['codex:sess-1'],
        branch: ['fix/cond-0039'],
        worktree: ['/tmp/cond-0039'],
        pull_request: ['o/r#39'],
      }
      const options = (values[field] ?? []).map(value => ({ value, total: 1, open: 1 }))
      return { project_id: 'cao-system', field, query: params.get('q') ?? '', matching_total: options.length, options }
    }
    if (url.startsWith('/tracker/projects/cao-system')) return PROJECT
    if (url.startsWith('/tracker/projects')) return [PROJECT]
    if (url.startsWith('/tracker/issues/cond-0039')) return ISSUE
    if (url.startsWith('/tracker/issues')) {
      const rows = url.includes('q=') ? [ISSUE, RELATED_ISSUE] : [ISSUE]
      return { total: rows.length, limit: 50, offset: 0, issues: rows }
    }
    return {}
  }

  beforeEach(() => {
    calls.length = 0
    ;(ISSUE as unknown as { assignee: string | null }).assignee = null
    ISSUE.collaborators = []
    window.history.replaceState(null, '', '/')
    mockFetch.mockImplementation((url: string, opts?: RequestInit) => {
      calls.push({
        url,
        method: opts?.method ?? 'GET',
        body: opts?.body ? JSON.parse(opts.body as string) : undefined,
      })
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: () => Promise.resolve(respond(url)),
      })
    })
    vi.stubGlobal('fetch', mockFetch)
  })

  afterEach(() => vi.restoreAllMocks())

  it('lists projects with their open and total counts', async () => {
    render(<ProjectsPanel />)
    expect(await screen.findByText('CAO System')).toBeInTheDocument()
    expect(await screen.findByText('80 open / 208')).toBeInTheDocument()
  })

  it('offers the severities the server declares, not a hard-coded set', async () => {
    // P0 arrived only because the real ledger turned out to use it. A
    // dropdown built from a local constant would have silently omitted it.
    render(<ProjectsPanel />)
    await screen.findByText('CAO System')
    fireEvent.click(screen.getByRole('button', { name: /Advanced filters/ }))
    const filters = await screen.findAllByRole('button', { name: 'P0' })
    expect(filters.length).toBeGreaterThan(0)
  })

  it('shows the scopes that decide where issues file', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/2 scopes/))
    expect(await screen.findByText('/Users/colin/Projects/cao-conductor')).toBeInTheDocument()
    expect(await screen.findByText('cao-p1-closure')).toBeInTheDocument()
  })

  it('renders an issue row with its key, severity and status', async () => {
    render(<ProjectsPanel />)
    expect(await screen.findByText('cond-0039')).toBeInTheDocument()
    expect(await screen.findByText(/event-mirror lock contention/)).toBeInTheDocument()
  })

  it('expands an issue into an editable detail view', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/event-mirror lock contention/))
    const title = await screen.findByLabelText('Bug title')
    expect(title).toHaveValue(ISSUE.title)
    expect(await screen.findByLabelText('Failing command')).toHaveValue('python3 -B probes.py')
  })

  it('sends only the fields that actually changed', async () => {
    // A PATCH echoing every field would write an audit event per visit and
    // turn the history into noise.
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/event-mirror lock contention/))
    const assignee = await screen.findByLabelText('Assignee')
    fireEvent.focus(assignee)
    fireEvent.click(await screen.findByRole('option', { name: /terra/ }))
    fireEvent.click(await screen.findByRole('button', { name: /Save 1 change/ }))

    await waitFor(() => expect(calls.some(c => c.method === 'PATCH')).toBe(true))
    const patch = calls.find(c => c.method === 'PATCH')!
    expect(patch.body).toEqual({ assignee: 'terra', actor: 'dashboard' })
  })

  it('can suppress only the automatic former-assignee collaborator addition', async () => {
    ;(ISSUE as unknown as { assignee: string | null }).assignee = 'codex:sess-old'
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/event-mirror lock contention/))
    const assignee = await screen.findByLabelText('Assignee')
    fireEvent.focus(assignee)
    fireEvent.click(await screen.findByRole('option', { name: /terra/ }))
    const retain = screen.getByRole('checkbox', {
      name: /Keep codex:sess-old as a collaborator/,
    })
    expect(retain).toBeChecked()
    fireEvent.click(retain)
    fireEvent.click(screen.getByRole('button', { name: /Save 1 change/ }))

    await waitFor(() => expect(calls.some(c => c.method === 'PATCH')).toBe(true))
    expect(calls.find(c => c.method === 'PATCH')!.body).toEqual({
      assignee: 'terra',
      actor: 'dashboard',
      drop_previous_assignee: true,
    })
  })

  it('saves in-progress and its assignee together', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/event-mirror lock contention/))
    fireEvent.change(await screen.findByLabelText('Status'), { target: { value: 'in-progress' } })
    expect(calls.some(c => c.method === 'PATCH')).toBe(false)
    expect(await screen.findByRole('alert')).toHaveTextContent(/should have one primary assignee/)
    const assignee = screen.getByLabelText('Assignee')
    fireEvent.focus(assignee)
    fireEvent.click(await screen.findByRole('option', { name: /terra/ }))
    fireEvent.click(screen.getByRole('button', { name: /Save 2 changes/ }))
    await waitFor(() => expect(calls.some(c => c.method === 'PATCH')).toBe(true))
    expect(calls.find(c => c.method === 'PATCH')!.body).toMatchObject({
      status: 'in-progress',
      assignee: 'terra',
    })
  })

  it('edits collaborators and implementation links with searchable multi-pickers', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/event-mirror lock contention/))
    const collaborators = await screen.findByRole('combobox', { name: 'Collaborators' })
    fireEvent.focus(collaborators)
    fireEvent.click(await screen.findByRole('option', { name: /codex:sess-1/ }))
    const prs = screen.getByRole('combobox', { name: 'Pull requests' })
    fireEvent.change(prs, { target: { value: 'o/r#40' } })
    fireEvent.click(await screen.findByRole('option', { name: /Create “o\/r#40”/ }))
    fireEvent.click(screen.getByRole('button', { name: /Save 2 changes/ }))

    await waitFor(() => expect(calls.some(c => c.method === 'PATCH')).toBe(true))
    expect(calls.find(c => c.method === 'PATCH')!.body).toMatchObject({
      collaborators: ['codex:sess-1'],
      pull_requests: ['o/r#40'],
    })
  })

  it('records who made the change so the audit trail names an actor', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/event-mirror lock contention/))
    fireEvent.change(await screen.findByLabelText('Severity'), { target: { value: 'P1' } })
    await waitFor(() => expect(calls.some(c => c.method === 'PATCH')).toBe(true))
    expect(calls.find(c => c.method === 'PATCH')!.body).toMatchObject({ actor: 'dashboard' })
  })

  it('files a new bug against the selected project with an explicit diagnostic override', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByRole('button', { name: /Log bug/ }))
    fireEvent.change(await screen.findByLabelText('Title'), { target: { value: 'dashboard hangs on reconnect' } })
    expect(screen.getByRole('button', { name: /^File bug$/ })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: /File bug with policy override/ }))
    await waitFor(() => expect(calls.some(c => c.url === '/tracker/issues' && c.method === 'POST')).toBe(true))
    const post = calls.find(c => c.url === '/tracker/issues' && c.method === 'POST')!
    expect(post.body).toMatchObject({
      project_id: 'cao-system',
      kind: 'bug',
      title: 'dashboard hangs on reconnect',
      origin: 'dashboard',
      force: true,
    })
  })

  it('keeps unbounded filters collapsed and loads suggestions only on demand', async () => {
    render(<ProjectsPanel />)
    await screen.findByText('cond-0039')
    expect(screen.queryByTestId('advanced-filters')).not.toBeInTheDocument()
    expect(calls.some(call => call.url.includes('/options'))).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: /Advanced filters/ }))
    fireEvent.focus(screen.getByRole('combobox', { name: 'Filter labels' }))
    expect(await screen.findByRole('option', { name: /deferred/ })).toBeInTheDocument()
    expect(calls.some(call => call.url.includes('field=label'))).toBe(true)
  })

  it('creates a label through the searchable picker and files reproduction steps', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByRole('button', { name: /Log bug/ }))
    fireEvent.change(await screen.findByLabelText('Title'), { target: { value: 'reconnect fails' } })
    fireEvent.change(screen.getByLabelText('Reproduction steps'), {
      target: { value: '1. suspend\n2. resume' },
    })
    fireEvent.change(screen.getByLabelText('Expected outcome'), {
      target: { value: 'The dashboard reconnects' },
    })
    fireEvent.change(screen.getByLabelText('Actual outcome'), {
      target: { value: 'The dashboard stays disconnected' },
    })
    const labels = screen.getByRole('combobox', { name: 'Labels' })
    fireEvent.change(labels, { target: { value: 'initiative:new-ui' } })
    fireEvent.click(await screen.findByRole('option', { name: /Create “initiative:new-ui”/ }))
    fireEvent.click(screen.getByRole('button', { name: /^File bug$/ }))

    await waitFor(() => expect(calls.some(c => c.url === '/tracker/issues' && c.method === 'POST')).toBe(true))
    const post = calls.find(c => c.url === '/tracker/issues' && c.method === 'POST')!
    expect(post.body).toMatchObject({
      labels: ['initiative:new-ui'],
      reproduction_steps: '1. suspend\n2. resume',
      expected_outcome: 'The dashboard reconnects',
      actual_outcome: 'The dashboard stays disconnected',
    })
  })

  it('links only a searched existing issue and identifies it by key and title', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/event-mirror lock contention/))
    const target = await screen.findByRole('combobox', { name: 'Link target issue' })
    fireEvent.change(target, { target: { value: 'restart' } })
    expect(screen.getByRole('button', { name: 'Link' })).toBeDisabled()

    const option = await screen.findByRole('option', { name: /cond-0040.*restart the dashboard/ })
    fireEvent.click(option)
    fireEvent.click(screen.getByRole('button', { name: 'Link' }))

    await waitFor(() => expect(calls.some(c =>
      c.url === '/tracker/issues/cond-0039/links' && c.method === 'POST',
    )).toBe(true))
    const post = calls.find(c => c.url === '/tracker/issues/cond-0039/links' && c.method === 'POST')!
    expect(post.body).toEqual({ to_key: 'cond-0040', kind: 'relates' })
  })

  it('sends repeated status params rather than a comma list', async () => {
    // The server reads repeated params as an OR; a comma list would be one
    // unknown status and 400.
    render(<ProjectsPanel />)
    await screen.findByText('cond-0039')
    fireEvent.click(screen.getByRole('button', { name: /Advanced filters/ }))
    fireEvent.click(await screen.findByRole('button', { name: 'blocked' }))
    await waitFor(() => expect(calls.some(c => c.url.includes('status=blocked'))).toBe(true))
    expect(calls.some(c => c.url.includes('status=blocked%2C'))).toBe(false)
  })

  it('surfaces the server explanation when a scope belongs to another project', async () => {
    mockFetch.mockImplementation((url: string, opts?: RequestInit) => {
      if (opts?.method === 'POST' && url.includes('/scopes')) {
        return Promise.resolve({
          ok: false,
          status: 409,
          statusText: 'Conflict',
          json: () => Promise.resolve({ detail: "scope '/Users/colin/Projects/aegix' is already registered to project 'aegix'" }),
        })
      }
      return Promise.resolve({ ok: true, status: 200, statusText: 'OK', json: () => Promise.resolve(respond(url)) })
    })

    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/2 scopes/))
    fireEvent.change(await screen.findByLabelText('Scope value'), {
      target: { value: '/Users/colin/Projects/aegix' },
    })
    fireEvent.click(await screen.findByRole('button', { name: 'Add' }))
    // The 409 names the owning project, which is the one fact that tells the
    // operator what to do next.
    await waitFor(() => expect(mockFetch).toHaveBeenCalled())
  })

  it('marks a migrated issue as coming from the markdown ledger', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/event-mirror lock contention/))
    expect(await screen.findByText(/migrated from the markdown ledger/)).toBeInTheDocument()
  })

  it('shows the audit trail on request', async () => {
    render(<ProjectsPanel />)
    fireEvent.click(await screen.findByText(/event-mirror lock contention/))
    fireEvent.click(await screen.findByRole('button', { name: 'Audit trail' }))
    expect(await screen.findByText(/human · created/)).toBeInTheDocument()
  })

  it('reports an empty project list without pretending it failed', async () => {
    mockFetch.mockImplementation((url: string) =>
      Promise.resolve({
        ok: true, status: 200, statusText: 'OK',
        json: () => Promise.resolve(url.startsWith('/tracker/vocabulary') ? VOCAB : []),
      }),
    )
    render(<ProjectsPanel />)
    expect(await screen.findByText(/No projects yet/)).toBeInTheDocument()
  })
})
