import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { RankedSearchResults } from '../components/RankedSearchResults'
import { RankedSearchResponse, TrackerIssue } from '../api'

/**
 * §12.3 V1 presentation contract for the ranked-search surface: evidence
 * (snippets, matched fields, lanes, the winning comment) is visible, and the
 * lexical-only state never looks complete — the degradation banner renders
 * whenever any lane reports unavailable or the server gives reasons.
 */

const ISSUE: TrackerIssue = {
  key: 'cond-0638',
  project_id: 'cao-system',
  kind: 'story',
  title: 'ranked lexical search service with shared filters',
  body: 'BM25 over issues and comments.',
  status: 'closed',
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
  created_at: '2026-08-21T00:00:00Z',
  updated_at: '2026-08-22T00:00:00Z',
  closed_at: null,
}

export function searchFixture(overrides?: Partial<RankedSearchResponse>): RankedSearchResponse {
  return {
    query: 'lease deadlock',
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
    generations: { schema_version: 1, document_schema_version: 1, content_clock: 7, active_vector_generation: null, rebuilt_at: null },
    diagnostics: { lane_elapsed_ms: {}, total_elapsed_ms: 1.2 },
    total: 2,
    limit: 50,
    offset: 0,
    results: [
      {
        issue: ISSUE,
        rank_score: 0.0325,
        contributing_lanes: [
          { lane: 'issue-bm25', rank: 1, raw_score: 5.2 },
          { lane: 'comment-bm25', rank: 2, raw_score: 3.1 },
        ],
        matched_fields: ['title', 'comments'],
        snippets: {
          title: '…successor lease deadlock between two workers…',
          comments: '…the lease deadlock cleared after restart…',
        },
        winning_comment: {
          comment_id: 77,
          important: true,
          retained_hits: 2,
          additional_comment_ids: [81],
          total_matching_comments: 2,
        },
        exact_boosts: [],
        neighborhood: [],
        duplicate_chain: [],
      },
      {
        issue: { ...ISSUE, key: 'cond-0644', title: 'fusion lane weights' },
        rank_score: 0.0161,
        contributing_lanes: [{ lane: 'exact', rank: 1, raw_score: -3 }],
        matched_fields: ['body'],
        snippets: { body: '…mentions deadlock once…' },
        winning_comment: null,
        exact_boosts: ['key'],
        neighborhood: [],
        duplicate_chain: [],
      },
    ],
    ...overrides,
  }
}

function renderResults(response: RankedSearchResponse, handlers?: {
  onOpenComment?: (key: string, id: number) => void
  onSelectIssue?: () => void
  onOffsetChange?: (offset: number) => void
}) {
  return render(
    <RankedSearchResults
      response={response}
      loading={false}
      error={null}
      onRetry={() => {}}
      selectedKey={null}
      onSelectIssue={handlers?.onSelectIssue ?? (() => {})}
      onOpenComment={handlers?.onOpenComment ?? (() => {})}
      onOffsetChange={handlers?.onOffsetChange ?? (() => {})}
    />,
  )
}

describe('RankedSearchResults', () => {
  it('shows snippets under each hit with the field they came from', () => {
    renderResults(searchFixture())
    expect(screen.getByText(/successor lease deadlock/)).toBeInTheDocument()
    const commentsSnippet = screen.getByText(/cleared after restart/)
    expect(commentsSnippet.textContent).toContain('comments:')
  })

  it('badges matched fields and contributing lanes per hit', () => {
    renderResults(searchFixture())
    const badges = screen.getAllByTestId('matched-badges')
    expect(badges[0].textContent).toContain('title')
    expect(badges[0].textContent).toContain('comments')
    expect(badges[0].textContent).toContain('issue text')
    expect(badges[0].textContent).not.toContain('exact')
    expect(badges[1].textContent).toContain('exact')
    expect(badges[1].textContent).not.toContain('title')
  })

  it('offers navigation to the winning comment and says when more match', () => {
    const onOpenComment = vi.fn()
    renderResults(searchFixture(), { onOpenComment })
    fireEvent.click(screen.getByRole('button', { name: /Open matching comment 77 on cond-0638/ }))
    expect(onOpenComment).toHaveBeenCalledWith('cond-0638', 77)
    expect(screen.getByTestId('open-winning-comment').textContent).toContain('+1')
  })

  it('never presents a lexical-only result set as complete', () => {
    renderResults(searchFixture())
    const banner = screen.getByTestId('search-degradation')
    expect(banner.getAttribute('role')).toBe('status')
    expect(banner.textContent).toMatch(/Lexical-only results/)
    expect(banner.textContent).toMatch(/installs at M2|unavailable/)
  })

  it('renders an explicit empty state instead of an empty box', () => {
    renderResults(searchFixture({ results: [], total: 0 }))
    expect(screen.getByText(/No ranked matches for “lease deadlock”/)).toBeInTheDocument()
  })

  it('paginates with Previous and Next over the ranked total', () => {
    const onOffsetChange = vi.fn()
    renderResults(searchFixture({ total: 120 }), { onOffsetChange })
    expect(screen.getByText(/1–2 of 120/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(onOffsetChange).toHaveBeenCalledWith(50)
  })

  it('surfaces search failure with a retry affordance', () => {
    render(
      <RankedSearchResults
        response={null}
        loading={false}
        error="400 Bad Request"
        onRetry={() => {}}
        selectedKey={null}
        onSelectIssue={() => {}}
        onOpenComment={() => {}}
        onOffsetChange={() => {}}
      />,
    )
    expect(screen.getByRole('alert')).toHaveTextContent(/Search failed: 400 Bad Request/)
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('selecting a row hands the issue to the panel for expansion', () => {
    const onSelectIssue = vi.fn()
    renderResults(searchFixture(), { onSelectIssue })
    fireEvent.click(screen.getByText(/ranked lexical search service/))
    expect(onSelectIssue).toHaveBeenCalledWith(expect.objectContaining({ key: 'cond-0638' }))
  })
})
