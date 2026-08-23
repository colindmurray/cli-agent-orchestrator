import { RankedSearchResponse, RankedSearchExplanation, TrackerIssue } from '../api'

/**
 * Ranked-search result surface for the project panel (§12.3).
 *
 * Deliberately separate from the flat list rows: a ranked hit carries evidence
 * a list row does not — where it matched, which lane found it, and which
 * comment carried it — and the semantic lanes are not installed yet, so every
 * response is currently a degraded one. The degradation banner renders
 * whenever any lane is unavailable or the server gives reasons, which means a
 * lexical-only result set never looks complete.
 */

const LANE_LABEL: Record<string, string> = {
  'issue-bm25': 'issue text',
  'comment-bm25': 'comments',
  exact: 'exact',
}

const FIELD_LABEL: Record<string, string> = {
  key: 'key',
  title: 'title',
  body: 'body',
  comments: 'comments',
  failing_command: 'failing command',
  reproduction_steps: 'repro steps',
  expected_outcome: 'expected outcome',
  actual_outcome: 'actual outcome',
  evidence: 'evidence',
  observed_revision: 'observed revision',
  resolution: 'resolution',
  component: 'component',
  assignee: 'assignee',
  reporter: 'reporter',
}

function fieldLabel(field: string): string {
  return FIELD_LABEL[field] ?? field.replace(/_/g, ' ')
}

function laneLabel(lane: string): string {
  return LANE_LABEL[lane] ?? lane
}

function hasSemanticGap(response: RankedSearchResponse): boolean {
  const lanes = response.degradation?.lanes ?? {}
  return (
    Object.values(lanes).some(lane => !lane.available)
    || (response.degradation?.reasons?.length ?? 0) > 0
  )
}

function ExplanationRow({
  hit,
  selected,
  onSelectIssue,
  onOpenComment,
}: {
  hit: RankedSearchExplanation
  selected: boolean
  onSelectIssue: (issue: TrackerIssue) => void
  onOpenComment: (issueKey: string, commentId: number) => void
}) {
  const issue = hit.issue
  if (!issue) return null
  const lanes = [...new Set(hit.contributing_lanes.map(l => l.lane))]
  const win = hit.winning_comment
  return (
    <div className="border-b border-gray-800/70 last:border-b-0">
      {/* The select affordance and the comment-navigation affordance are
          siblings, never parent/child: a button inside a button is invalid
          HTML and screen readers drop the inner control. */}
      <div className="w-full flex items-center gap-3 px-3 py-2 hover:bg-gray-900/60 transition-colors">
        <button
          onClick={() => onSelectIssue(issue)}
          aria-expanded={selected}
          aria-label={`Open ${issue.key}: ${issue.title}`}
          className="flex flex-1 min-w-0 items-center gap-3 text-left"
        >
          <code className="text-xs text-gray-500 w-24 shrink-0">{issue.key}</code>
          <span className="text-sm text-gray-200 truncate flex-1">{issue.title}</span>
          <span
            data-testid="rank-score"
            title="Reciprocal-rank-fusion score — higher is a better match"
            className="text-[10px] font-mono text-gray-600 shrink-0"
          >
            {hit.rank_score.toFixed(4)}
          </span>
        </button>
        {win && (
          <button
            data-testid="open-winning-comment"
            onClick={() => onOpenComment(issue.key, win.comment_id)}
            aria-label={`Open matching comment ${win.comment_id} on ${issue.key}`}
            title={
              win.total_matching_comments > 1
                ? `${win.total_matching_comments} comments match; #${win.comment_id} ranked highest`
                : 'The comment that matched this query'
            }
            className={`flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-[11px] border transition-colors ${
              win.important
                ? 'border-amber-600/50 bg-amber-500/10 text-amber-300'
                : 'border-gray-800 text-gray-400 hover:text-emerald-300 hover:border-gray-700'
            }`}
          >
            ★ comment #{win.comment_id}
            {win.total_matching_comments > 1 && (
              <span className="text-gray-500">+{win.total_matching_comments - 1}</span>
            )}
          </button>
        )}
      </div>
      <div className="px-3 pb-2 pl-[4.75rem] space-y-1">
        <div className="flex flex-wrap gap-1" data-testid="matched-badges">
          {hit.matched_fields.map(field => (
            <span
              key={field}
              className={`rounded px-1.5 py-0.5 text-[10px] border ${
                field === 'comments'
                  ? 'border-violet-500/30 bg-violet-500/10 text-violet-300'
                  : 'border-sky-500/30 bg-sky-500/10 text-sky-300'
              }`}
            >
              {fieldLabel(field)}
            </span>
          ))}
          {lanes.map(lane => (
            <span
              key={lane}
              className="rounded px-1.5 py-0.5 text-[10px] border border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
            >
              {laneLabel(lane)}
            </span>
          ))}
        </div>
        {Object.entries(hit.snippets).map(([field, snippet]) => (
          <p key={field} className="text-[11px] text-gray-400 leading-snug">
            <span className="text-gray-600">{fieldLabel(field)}: </span>
            {snippet}
          </p>
        ))}
      </div>
    </div>
  )
}

export function RankedSearchResults({
  response,
  loading,
  error,
  onRetry,
  selectedKey,
  onSelectIssue,
  onOpenComment,
  onOffsetChange,
}: {
  response: RankedSearchResponse | null
  loading: boolean
  error: string | null
  onRetry: () => void
  selectedKey: string | null
  onSelectIssue: (issue: TrackerIssue) => void
  onOpenComment: (issueKey: string, commentId: number) => void
  onOffsetChange: (offset: number) => void
}) {
  return (
    <div className="mt-4" data-testid="ranked-search" aria-label="Ranked search results">
      {response && hasSemanticGap(response) && (
        <div
          role="status"
          data-testid="search-degradation"
          className="mb-2 rounded-lg border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200"
        >
          <span className="font-medium">Lexical-only results.</span>
          {' '}Semantic lanes are not contributing
          {(response?.degradation?.reasons?.length ?? 0) > 0 && (
            <> — {response!.degradation.reasons.join('; ')}</>
          )}
          {Object.entries(response?.degradation?.lanes ?? {})
            .filter(([, lane]) => !lane.available)
            .map(([name, lane]) => (
              <span key={name} className="ml-2 rounded bg-amber-500/10 px-1.5 py-0.5" title={lane.reason}>
                {laneLabel(name)} unavailable
              </span>
            ))}
        </div>
      )}

      <div className="rounded-lg border border-gray-800 overflow-hidden" aria-busy={loading}>
        {loading && (
          <div role="status" className="px-4 py-8 text-center text-sm text-gray-500 flex items-center justify-center gap-2">
            Searching…
          </div>
        )}
        {!loading && error && (
          <div role="alert" data-testid="search-error" className="px-4 py-6 text-center space-y-2">
            <p className="text-sm text-red-300">Search failed: {error}</p>
            <button
              onClick={onRetry}
              className="rounded border border-gray-700 px-2.5 py-1 text-xs text-gray-300 hover:border-gray-600"
            >
              Retry
            </button>
          </div>
        )}
        {!loading && !error && response && response.results.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-gray-500">
            No ranked matches for “{response.query}”. Results are lexical-only while the semantic
            lanes are unavailable, so try plainer wording or fewer filters.
          </div>
        )}
        {!loading && !error && response && response.results.map(hit => (
          <ExplanationRow
            key={hit.issue?.key ?? hit.rank_score}
            hit={hit}
            selected={hit.issue?.key === selectedKey}
            onSelectIssue={onSelectIssue}
            onOpenComment={onOpenComment}
          />
        ))}
      </div>

      {response && response.total > response.limit && !error && (
        <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
          <span>
            {response.offset + 1}–{Math.min(response.offset + response.results.length, response.total)} of {response.total}
          </span>
          <div className="flex gap-2">
            <button
              disabled={response.offset === 0}
              onClick={() => onOffsetChange(Math.max(0, response.offset - response.limit))}
              className="px-2 py-1 rounded border border-gray-800 disabled:opacity-30 hover:border-gray-700"
            >
              Previous
            </button>
            <button
              disabled={response.offset + response.limit >= response.total}
              onClick={() => onOffsetChange(response.offset + response.limit)}
              className="px-2 py-1 rounded border border-gray-800 disabled:opacity-30 hover:border-gray-700"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
