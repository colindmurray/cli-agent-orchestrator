import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import {
  api, errorText, SimilarIssueDraft, SimilarIssueExplanation, SimilarIssuesResponse, TrackerIssue,
} from '../api'
import { linkPhrase } from '../lib/issueMap'

/**
 * Pre-filing similar-issue candidates for the create form (M2.5, design
 * §11/§12.3).
 *
 * The probe is ADVISORY by contract: it debounces one explicitly
 * project-scoped POST /tracker/issues/similar per settled draft, cancels
 * superseded requests, renders explained open and terminal candidates with
 * their canonical/duplicate/relationship facts, and never disables filing —
 * an unavailable similarity surface is shown as a visible advisory while the
 * create path stays fully enabled.
 */

/** Debounce window for draft similarity probes, ms — same window as the
 * panel's ranked search so typing cadence costs at most one request. */
const SIMILAR_DEBOUNCE_MS = 250
const SIMILAR_LIMIT = 5

const LANE_LABEL: Record<string, string> = {
  'issue-bm25': 'issue text',
  'comment-bm25': 'comments',
  exact: 'exact',
}

function laneLabel(lane: string): string {
  return LANE_LABEL[lane] ?? lane
}

interface SimilarState {
  /** Serialized draft whose response/error this state describes. */
  key: string | null
  response: SimilarIssuesResponse | null
  loading: boolean
  error: string | null
}

function CandidateRow({
  hit,
  expansions,
  terminalStatuses,
  onOpenIssue,
}: {
  hit: SimilarIssueExplanation
  expansions: TrackerIssue[]
  terminalStatuses: string[]
  onOpenIssue: (key: string) => void
}) {
  const issue = hit.issue
  if (!issue) return null
  const terminal = terminalStatuses.includes(issue.status)
  const lanes = [...new Set(hit.contributing_lanes.map(l => l.lane))]
  const probeContributions = hit.probe_contributions ?? []
  return (
    <div className="border-b border-gray-800/70 last:border-b-0 px-3 py-2" data-testid={`similar-candidate-${issue.key}`}>
      {/* One button per candidate row; the badges/facts below are plain text
          siblings — never nested inside the open control. */}
      <button
        onClick={() => onOpenIssue(issue.key)}
        aria-label={`Open ${issue.key}: ${issue.title}`}
        className="w-full flex items-center gap-2 text-left hover:bg-gray-900/60 rounded transition-colors"
      >
        <code className="text-xs text-gray-500 shrink-0">{issue.key}</code>
        <span className="text-sm text-gray-200 truncate flex-1">{issue.title}</span>
        <span
          data-testid="similar-candidate-status"
          className={`px-1.5 py-0.5 rounded text-[10px] border shrink-0 ${
            terminal
              ? 'border-gray-600/40 bg-gray-700/40 text-gray-400'
              : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
          }`}
        >
          {issue.status}{terminal ? ' · terminal' : ''}
        </span>
      </button>
      <div className="pl-1 pt-1 space-y-1">
        {(hit.matched_fields.length > 0 || lanes.length > 0) && (
          <div className="flex flex-wrap gap-1" data-testid="similar-matched-badges">
            {hit.matched_fields.map(field => (
              <span key={field} className="rounded px-1.5 py-0.5 text-[10px] border border-sky-500/30 bg-sky-500/10 text-sky-300">
                {field.replace(/_/g, ' ')}
              </span>
            ))}
            {lanes.map(lane => (
              <span key={lane} className="rounded px-1.5 py-0.5 text-[10px] border border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
                {laneLabel(lane)}
              </span>
            ))}
          </div>
        )}
        {Object.entries(hit.snippets).map(([field, snippet]) => (
          <p key={field} className="text-[11px] text-gray-400 leading-snug">
            <span className="text-gray-600">{field.replace(/_/g, ' ')}: </span>
            {snippet}
          </p>
        ))}
        {probeContributions.length > 0 && (
          <div className="text-[11px] text-gray-500" data-testid="similar-probe-contributions">
            {probeContributions.map((contribution, index) => (
              <p key={`${contribution.label}:${contribution.original_rank}:${index}`}>
                probe {contribution.label} · weight {contribution.weight.toFixed(2)} · rank {contribution.original_rank}
                {contribution.original_score == null ? '' : ` · score ${contribution.original_score}`}
                {` — ${contribution.query}`}
              </p>
            ))}
          </div>
        )}
        {hit.duplicate_chain.length > 0 && (
          <p className="text-[11px] text-gray-500" data-testid="similar-canonical">
            {hit.duplicate_chain.map((link, index) => (
              <span key={link.canonical_key}>
                {index > 0 && '; '}
                duplicate of{' '}
                <button
                  type="button"
                  onClick={() => onOpenIssue(link.canonical_key)}
                  aria-label={`Inspect canonical ${link.canonical_key}`}
                  className="font-mono text-emerald-400 hover:underline"
                >
                  {link.canonical_key}
                </button>
                {link.canonical_title ? ` — ${link.canonical_title}` : ''}
                {link.resolved ? ' (resolved)' : ''}
              </span>
            ))}
          </p>
        )}
        {expansions.length > 0 && (
          <p className="text-[11px] text-gray-500" data-testid="similar-expansion">
            Confirmed duplicates:{' '}
            {expansions.map((dup, index) => (
              <span key={dup.key}>
                {index > 0 && '; '}
                <button
                  type="button"
                  onClick={() => onOpenIssue(dup.key)}
                  aria-label={`Inspect confirmed duplicate ${dup.key}`}
                  className="font-mono text-emerald-400 hover:underline"
                >
                  {dup.key}
                </button>
                {dup.title ? ` — ${dup.title}` : ''}
              </span>
            ))}
          </p>
        )}
        {hit.neighborhood.length > 0 && (
          <p className="text-[11px] text-gray-500" data-testid="similar-neighborhood">
            {hit.neighborhood.map((link, index) => {
              const { phrase, other } = linkPhrase(link, issue.key)
              return (
                <span key={`${link.from_key}:${link.kind}:${link.to_key}`}>
                  {index > 0 && '; '}
                  {phrase}{' '}
                  <button
                    type="button"
                    onClick={() => onOpenIssue(other)}
                    aria-label={`Inspect related ${other}`}
                    className="font-mono text-emerald-400 hover:underline"
                  >
                    {other}
                  </button>
                </span>
              )
            })}
          </p>
        )}
      </div>
    </div>
  )
}

export function SimilarIssueCandidates({
  projectId,
  draft,
  terminalStatuses,
  onOpenIssue,
}: {
  projectId: string
  /** The create-shaped draft to probe, or null while the form carries no
   * meaningful content yet (empty title and untouched body starter). */
  draft: SimilarIssueDraft | null
  terminalStatuses: string[]
  onOpenIssue: (key: string) => void
}) {
  const [state, setState] = useState<SimilarState>({ key: null, response: null, loading: false, error: null })
  // Request identity is serialized scope + CONTENT, not object identity: the
  // form rebuilds the draft object on every render.
  const requestKey = JSON.stringify([projectId, draft])

  useEffect(() => {
    if (!draft) {
      setState({ key: null, response: null, loading: false, error: null })
      return
    }
    const controller = new AbortController()
    // Invalidate the prior answer as soon as this effect observes the new
    // draft. The render below also checks the key, so old candidates are
    // already non-actionable during the render-before-effect window.
    setState({ key: requestKey, response: null, loading: true, error: null })
    const handle = setTimeout(() => {
      api.similarTrackerIssues(
        { draft, project_ids: [projectId], limit: SIMILAR_LIMIT },
        controller.signal,
      )
        .then(response => {
          // A superseded request resolved late: its answer belongs to an
          // older draft and must never replace the newest results.
          if (controller.signal.aborted) return
          if (!response || !Array.isArray(response.candidates) || !Array.isArray(response.duplicate_expansions)) {
            setState({ key: requestKey, response: null, loading: false, error: 'the tracker returned an unrecognized similarity response' })
            return
          }
          setState({ key: requestKey, response, loading: false, error: null })
        })
        .catch(err => {
          if (controller.signal.aborted) return
          setState({ key: requestKey, response: null, loading: false, error: errorText(err) })
        })
    }, SIMILAR_DEBOUNCE_MS)
    return () => {
      clearTimeout(handle)
      controller.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey, projectId])

  if (!draft) return null
  // A draft edit re-renders before its effect runs. Key the visible state to
  // the exact draft so a response from the previous render is never displayed
  // or clickable during that gap, the debounce, or request latency.
  const current = state.key === requestKey
    ? state
    : { key: requestKey, response: null, loading: true, error: null }
  const { response, loading, error } = current
  const duplicateConflicts = response?.diagnostics?.similarity_duplicate_conflicts ?? []
  const coverageNeedsNotice = response?.coverage?.status === 'degraded'
    || response?.coverage?.status === 'partial'
    || response?.coverage?.status === 'inconclusive'
  const emptyInconclusive = Boolean(response && response.candidates.length === 0 && (
    response.coverage?.inconclusive
    || coverageNeedsNotice
    || (response.degradation?.reasons.length ?? 0) > 0
    || ((response.mode_requested === 'semantic' || response.mode_requested === 'hybrid')
      && response.mode_effective === 'lexical')
  ))
  const expansionsByHit = new Map<string, TrackerIssue[]>()
  for (const expansion of response?.duplicate_expansions ?? []) {
    const rows = expansionsByHit.get(expansion.duplicate_of) ?? []
    rows.push(expansion.issue)
    expansionsByHit.set(expansion.duplicate_of, rows)
  }

  return (
    <section
      aria-label="Similar issues"
      data-testid="similar-candidates"
      aria-busy={loading}
      className="rounded-lg border border-gray-800 bg-gray-950/40"
    >
      <div className="px-3 py-1.5 border-b border-gray-800 text-[11px] uppercase tracking-wide text-gray-500">
        Similar issues — advisory only; filing is never blocked
      </div>
      {loading && !response && !error && (
        <div role="status" className="px-3 py-3 text-xs text-gray-500 flex items-center gap-2">
          <Loader2 size={12} className="animate-spin" /> Updating similar issues…
        </div>
      )}
      {error && (
        <div
          role="status"
          data-testid="similar-unavailable"
          className="m-2 rounded border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200"
        >
          <span className="font-medium">Similarity check unavailable.</span>
          {' '}{error} Filing is unaffected.
        </div>
      )}
      {!error && response && response.degradation && (
        (response.degradation.reasons.length > 0 || coverageNeedsNotice || emptyInconclusive)
      ) && (
        <div
          role="status"
          data-testid="similar-degraded"
          className="m-2 rounded border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200"
        >
          Similarity is advisory: mode {response.mode_effective ?? 'degraded'}, coverage {response.coverage?.status ?? 'degraded'}.
          {response.coverage?.inconclusive
            ? ' No candidates is inconclusive while retrieval is degraded. Filing is unaffected.'
            : ' Filing is unaffected.'}
        </div>
      )}
      {!error && response && duplicateConflicts.length > 0 && (
        <div
          role="status"
          data-testid="similar-duplicate-conflict"
          className="m-2 rounded border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200"
        >
          Similarity found conflicting native duplicate targets for{' '}
          {duplicateConflicts.map(conflict => conflict.duplicate_key).join(', ')}; no canonical was asserted.
        </div>
      )}
      {!error && response && response.candidates.length === 0 && !emptyInconclusive && (
        <div role="status" className="px-3 py-3 text-xs text-gray-500">
          No similar issues found in this project.
        </div>
      )}
      {!error && response && response.candidates.map(hit => (
        <CandidateRow
          key={hit.issue?.key ?? hit.rank_score}
          hit={hit}
          expansions={hit.issue ? expansionsByHit.get(hit.issue.key) ?? [] : []}
          terminalStatuses={terminalStatuses}
          onOpenIssue={onOpenIssue}
        />
      ))}
    </section>
  )
}
