// Render the real App so tracker query state exercises both outer-tab routing
// and ProjectsPanel's own URL restoration.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, render, screen } from '@testing-library/react'
import App from '../App'

vi.mock('../components/TerminalView', () => ({ TerminalView: () => null }))
vi.mock('sigma', () => ({
  default: class {
    on() {}
    kill() {}
    refresh() {}
    setSetting() {}
    getCamera() { return { setState() {} } }
  },
}))

const VOCAB = {
  statuses: ['open', 'triage', 'in-progress', 'blocked', 'resolved', 'closed', 'wontfix', 'duplicate'],
  terminal_statuses: ['closed', 'duplicate', 'wontfix'],
  severities: ['P0', 'P1', 'P2', 'P3', 'P4', 'unset'],
  scope_kinds: ['path', 'session', 'git_remote', 'project_id'],
  link_kinds: ['part-of', 'blocks', 'relates'],
  project_statuses: ['active', 'archived'],
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
  scopes: [],
}

function respond(url: string): unknown {
  if (url === '/sessions') return []
  if (url === '/settings/memory') return { enabled: false }
  if (url === '/agents/profiles') return []
  if (url.startsWith('/tracker/vocabulary')) return VOCAB
  if (url.startsWith('/tracker/issues/cond-0001/graph')) {
    return {
      root: { key: 'cond-0001', title: 'Project root' },
      nodes: [], external: [], links: [],
      bounds: { max_depth: 8, max_nodes: 300, truncated: false, reasons: [] },
      stats: { nodes: 1, descendants: 0, external: 0, links: 0, depth: 0 },
    }
  }
  if (url.startsWith('/tracker/issues') || url.startsWith('/tracker/features')) {
    return { total: 0, limit: 50, offset: 0, issues: [] }
  }
  if (url === '/tracker/projects/cao-system/labels') {
    return { project_id: 'cao-system', labels: [], unlabeled: 0, unlabeled_open: 0 }
  }
  if (url === '/tracker/projects/cao-system') return PROJECT
  if (url.startsWith('/tracker/projects')) return [PROJECT]
  return {}
}

function setUrl(url: string) {
  window.history.replaceState(null, '', url)
}

describe('App deep-link restoration', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request) => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: () => Promise.resolve(respond(String(input))),
    })))
  })

  afterEach(() => {
    setUrl('/')
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('opens the Projects tab on the Wayfinder view for a tracker deep link', async () => {
    setUrl('/?project=cao-system&view=wayfinder')
    render(<App />)
    // The outer tab is the regression: pre-fix it stayed Home while the
    // tracker query sat unread in the URL.
    expect(screen.getByRole('tab', { name: 'Projects' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Home' })).toHaveAttribute('aria-selected', 'false')
    // The panel's own restore then lands on the Wayfinder view, not the list.
    const wayfinder = await screen.findByRole('tab', { name: 'Wayfinder' })
    expect(wayfinder).toHaveAttribute('aria-selected', 'true')
  })

  it('opens the Projects tab on the generic Graph view for a root deep link', async () => {
    setUrl('/?project=cao-system&view=graph&root=cond-0001')
    render(<App />)
    expect(screen.getByRole('tab', { name: 'Projects' })).toHaveAttribute('aria-selected', 'true')
    const graph = await screen.findByRole('tab', { name: 'Graph' })
    expect(graph).toHaveAttribute('aria-selected', 'true')
    expect(await screen.findByLabelText('Issue graph root')).toHaveValue('cond-0001')
  })

  it('opens Home on bare /', () => {
    setUrl('/')
    render(<App />)
    expect(screen.getByRole('tab', { name: 'Home' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Projects' })).toHaveAttribute('aria-selected', 'false')
  })

  it('restores the Projects tab when history pops to a tracker URL', async () => {
    setUrl('/')
    render(<App />)
    expect(screen.getByRole('tab', { name: 'Home' })).toHaveAttribute('aria-selected', 'true')
    act(() => {
      window.history.pushState(null, '', '/?project=cao-system&view=wayfinder')
      window.dispatchEvent(new Event('popstate'))
    })
    expect(screen.getByRole('tab', { name: 'Projects' })).toHaveAttribute('aria-selected', 'true')
    const wayfinder = await screen.findByRole('tab', { name: 'Wayfinder' })
    expect(wayfinder).toHaveAttribute('aria-selected', 'true')
  })
})
