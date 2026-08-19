// Wayfinder view for the Projects tracker page (cond-0394).
//
// A wayfinder map is an issue labelled `wayfinder:map`; its decision tickets
// are part-of children. This panel lists a project's maps and renders the
// selected map from the server's one-request projection: destination/body,
// progress, the ordered frontier, every child with its state, every external
// link endpoint (with the actual blockers flagged), and the relationship
// graph. Classification is computed once on the server (see map_projection) —
// this file renders it, it never re-derives blocked/frontier rules of its
// own.
//
// Map-body edits are concurrency-safe: they carry expected_updated_at, and a
// stale 409 keeps the user's draft and offers re-read & retry rather than
// silently overwriting a concurrent session's edit.

import { useCallback, useEffect, useState } from 'react'
import {
  api,
  conflictDetail,
  errorText,
  ApiError,
  TrackerIssue,
  TrackerMapChild,
  TrackerMapProjection,
  TrackerVocabulary,
} from '../api'
import { useStore } from '../store'
import { IssueMapGraph } from './IssueMapGraph'
import { linkPhrase } from '../lib/issueMap'
import {
  Compass,
  Loader2,
  Map as MapIcon,
  RefreshCw,
  Save,
  X,
} from 'lucide-react'

function shortDate(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().slice(0, 10)
}

const MAP_LABEL = 'wayfinder:map'

// ---------------------------------------------------------------------------
// Map browser
// ---------------------------------------------------------------------------

export function WayfinderPanel({
  projectId,
  vocab,
  mapKey,
  onSelectMap,
  selectedKey,
  onSelectIssue,
  refreshSignal,
  onChanged,
}: {
  projectId: string
  vocab: TrackerVocabulary
  mapKey: string | null
  onSelectMap: (key: string | null) => void
  selectedKey: string | null
  onSelectIssue: (key: string | null) => void
  /** Bumped by the parent whenever tracker state changed elsewhere (e.g. an
   * edit in the detail panel) so the projection re-reads. */
  refreshSignal: number
  onChanged: () => Promise<void>
}) {
  const [maps, setMaps] = useState<TrackerIssue[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadMaps = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const page = await api.listTrackerIssues({
        projectId,
        label: [MAP_LABEL],
        kind: 'all',
        openOnly: false,
        limit: 200,
        order: 'created_desc',
      })
      setMaps(page.issues)
    } catch (err) {
      setError(errorText(err))
      setMaps(null)
    } finally {
      setLoading(false)
    }
  }, [projectId])

  useEffect(() => {
    loadMaps()
  }, [loadMaps, refreshSignal])

  if (loading && !maps) {
    return (
      <div className="mt-4 rounded-lg border border-gray-800 px-4 py-8 text-center text-sm text-gray-400 flex items-center justify-center gap-2">
        <Loader2 size={14} className="animate-spin" /> Loading wayfinder maps…
      </div>
    )
  }
  if (error) {
    return (
      <div className="mt-4 rounded-lg border border-gray-800 px-4 py-8 text-center">
        <p className="text-sm text-red-400">{error}</p>
        <button onClick={loadMaps} className="mt-2 text-xs text-emerald-400 hover:underline">
          Retry
        </button>
      </div>
    )
  }
  if (!maps || maps.length === 0) {
    return (
      <div className="mt-4 rounded-lg border border-gray-800 px-4 py-8 text-center" data-testid="wayfinder-empty">
        <Compass size={26} className="mx-auto text-gray-400 mb-2" />
        <p className="text-sm text-gray-400">No wayfinder maps in this project yet.</p>
        <p className="text-xs text-gray-400 mt-1">
          A map is an issue labelled <code className="text-gray-400">{MAP_LABEL}</code>; its
          tickets link to it with <code className="text-gray-400">part-of</code>.
        </p>
      </div>
    )
  }

  return (
    <div className="mt-4 space-y-4">
      {/* Map picker — always visible so switching maps never loses context. */}
      <div className="flex flex-wrap gap-1.5" role="listbox" aria-label="Wayfinder maps">
        {maps.map(m => (
          <button
            key={m.key}
            role="option"
            aria-selected={mapKey === m.key}
            onClick={() => onSelectMap(mapKey === m.key ? null : m.key)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs transition-colors ${
              mapKey === m.key
                ? 'bg-emerald-600/20 border-emerald-600/50 text-emerald-200'
                : 'border-gray-800 text-gray-400 hover:text-gray-200 hover:border-gray-700'
            }`}
          >
            <MapIcon size={11} />
            <code className="text-[11px]">{m.key}</code>
            <span className="max-w-64 truncate">{m.title}</span>
          </button>
        ))}
      </div>

      {mapKey && maps.some(m => m.key === mapKey) && (
        <MapView
          mapKey={mapKey}
          vocab={vocab}
          selectedKey={selectedKey}
          onSelectIssue={onSelectIssue}
          refreshSignal={refreshSignal}
          onChanged={onChanged}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// One map
// ---------------------------------------------------------------------------

function MapView({
  mapKey,
  vocab,
  selectedKey,
  onSelectIssue,
  refreshSignal,
  onChanged,
}: {
  mapKey: string
  vocab: TrackerVocabulary
  selectedKey: string | null
  onSelectIssue: (key: string | null) => void
  refreshSignal: number
  onChanged: () => Promise<void>
}) {
  const { showSnackbar } = useStore()
  const [projection, setProjection] = useState<TrackerMapProjection | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busyKey, setBusyKey] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setProjection(await api.getTrackerMap(mapKey))
    } catch (err) {
      setError(errorText(err))
      setProjection(null)
    } finally {
      setLoading(false)
    }
  }, [mapKey])

  useEffect(() => {
    load()
  }, [load, refreshSignal])

  const changed = useCallback(async () => {
    await load()
    await onChanged()
  }, [load, onChanged])

  const claim = async (key: string) => {
    setBusyKey(key)
    try {
      await api.claimTrackerIssue(key, 'dashboard')
      await changed()
      showSnackbar({ type: 'success', message: `${key} claimed` })
    } catch (err) {
      // A lost claim is a typed 409 naming the observed owner — surface them,
      // then re-read so the row shows the state the server actually holds.
      const owner = conflictDetail(err)?.observed_assignee
      showSnackbar({
        type: 'error',
        message: owner ? `${key} is already claimed by ${String(owner)}` : errorText(err),
      })
      await changed()
    } finally {
      setBusyKey(null)
    }
  }

  const unclaim = async (key: string) => {
    setBusyKey(key)
    try {
      await api.unclaimTrackerIssue(key, 'dashboard')
      await changed()
      showSnackbar({ type: 'success', message: `${key} released` })
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    } finally {
      setBusyKey(null)
    }
  }

  if (loading && !projection) {
    return (
      <div className="rounded-lg border border-gray-800 px-4 py-8 text-center text-sm text-gray-400 flex items-center justify-center gap-2">
        <Loader2 size={14} className="animate-spin" /> Loading {mapKey}…
      </div>
    )
  }
  if (error) {
    return (
      <div className="rounded-lg border border-gray-800 px-4 py-8 text-center">
        <p className="text-sm text-red-400">{error}</p>
        <button onClick={load} className="mt-2 text-xs text-emerald-400 hover:underline">
          Retry
        </button>
      </div>
    )
  }
  if (!projection) return null

  const { map, children, progress } = projection
  const childrenByKey = new Map(children.map(c => [c.key, c]))
  const terminal = vocab.terminal_statuses
  const donePct = progress.total > 0 ? Math.round((progress.terminal / progress.total) * 100) : 0

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/30 p-4 space-y-4" data-testid="map-view">
      {/* Map identity + progress */}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-white flex items-center gap-2 flex-wrap">
            <MapIcon size={15} className="text-emerald-400 shrink-0" />
            <span className="truncate">{map.title}</span>
            <code className="text-xs text-gray-400 font-normal">{map.key}</code>
          </h3>
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            {map.labels.map(l => (
              <span key={l} className="px-1.5 py-0.5 rounded text-[11px] bg-gray-800 text-gray-400 border border-gray-700/60">
                {l}
              </span>
            ))}
          </div>
        </div>
        <button
          onClick={load}
          title="Re-read the map"
          className="p-2 rounded hover:bg-gray-800 text-gray-400 hover:text-gray-200 shrink-0"
        >
          <RefreshCw size={14} />
        </button>
      </div>

      <div>
        <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1">
          <span>
            {progress.terminal}/{progress.total} done · {progress.claimed} claimed ·{' '}
            {progress.frontier} on the frontier
            {progress.resolved > 0 && ` · ${progress.resolved} resolved (unverified)`}
          </span>
          <span>{donePct}%</span>
        </div>
        <div
          className="h-1.5 rounded-full bg-gray-800 overflow-hidden"
          role="progressbar"
          aria-valuenow={progress.terminal}
          aria-valuemax={progress.total}
          aria-label="Map progress"
        >
          <div className="h-full bg-emerald-500 transition-all" style={{ width: `${donePct}%` }} />
        </div>
      </div>

      {/* Destination / body — editable, with optimistic concurrency */}
      <MapBodyEditor
        mapKey={mapKey}
        map={map}
        onSaved={changed}
      />

      {/* Frontier — the takeable edge, oldest first */}
      <section aria-label="Frontier">
        <h4 className="text-[11px] font-medium uppercase tracking-wide text-gray-400 mb-1.5">
          Frontier — takeable now, oldest first ({projection.frontier.length})
        </h4>
        {projection.frontier.length === 0 ? (
          <p className="text-xs text-gray-400 px-1" data-testid="frontier-empty">
            Nothing takeable right now — every open ticket is claimed or blocked.
          </p>
        ) : (
          <ol className="space-y-1">
            {projection.frontier.map((key, i) => {
              const child = childrenByKey.get(key)
              if (!child) return null
              return (
                <li key={key} className="flex items-center gap-2">
                  <span className="text-[11px] text-gray-400 w-5 text-right shrink-0">{i + 1}.</span>
                  <button
                    onClick={() => onSelectIssue(key)}
                    aria-pressed={selectedKey === key}
                    className={`flex-1 min-w-0 flex items-center gap-2 px-2.5 py-1.5 rounded border text-left transition-colors ${
                      selectedKey === key
                        ? 'border-emerald-600/50 bg-emerald-600/10'
                        : 'border-gray-800 hover:border-gray-700 bg-gray-950/40'
                    }`}
                  >
                    <code className="text-[11px] text-gray-400 shrink-0">{key}</code>
                    <span className="text-xs text-gray-200 truncate">{child.title}</span>
                  </button>
                  <button
                    onClick={() => claim(key)}
                    disabled={busyKey === key}
                    aria-label={`Claim ${key}`}
                    className="px-2.5 py-1.5 rounded text-[11px] bg-emerald-700/80 hover:bg-emerald-600 text-white disabled:opacity-40 shrink-0"
                  >
                    Claim
                  </button>
                </li>
              )
            })}
          </ol>
        )}
      </section>

      {/* The graph and the accessible list carry the same information. */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
        <IssueMapGraph
          projection={projection}
          terminalStatuses={terminal}
          selectedKey={selectedKey}
          onSelect={onSelectIssue}
        />
        <section aria-label="Tickets">
          <h4 className="text-[11px] font-medium uppercase tracking-wide text-gray-400 mb-1.5">
            Tickets ({children.length})
          </h4>
          <ul className="space-y-1" data-testid="map-children">
            {children.map(child => (
              <ChildRow
                key={child.key}
                child={child}
                terminal={terminal}
                selected={selectedKey === child.key}
                busy={busyKey === child.key}
                onSelect={() => onSelectIssue(child.key)}
                onClaim={() => claim(child.key)}
                onUnclaim={() => unclaim(child.key)}
              />
            ))}
            {children.length === 0 && (
              <li className="text-xs text-gray-400 px-1">
                No tickets yet — child issues join the map with a part-of link.
              </li>
            )}
          </ul>
          {projection.external.length > 0 && (
            <div className="mt-3">
              <h4 className="text-[11px] font-medium uppercase tracking-wide text-gray-400 mb-1.5">
                External links ({projection.external.length})
              </h4>
              <ul className="space-y-1" data-testid="map-external">
                {projection.external.map(ext => {
                  // Every relationship that pulled this issue into the
                  // projection, in the link's own direction — the list carries
                  // what the graph draws. A `blocks` phrase aimed at a child
                  // this issue actually benches renders red: that is the
                  // explicit external-blocker marker; a landed blocker or a
                  // relates/duplicates neighbour stays gray context.
                  const phrases = new Map<string, { text: string; benching: boolean }>()
                  for (const link of projection.links) {
                    if (link.from_key !== ext.key && link.to_key !== ext.key) continue
                    const { phrase, other } = linkPhrase(link, ext.key)
                    const text = `${phrase} ${other}`
                    if (!phrases.has(text)) {
                      phrases.set(text, {
                        text,
                        benching: phrase === 'blocks' && ext.blocking.includes(other),
                      })
                    }
                  }
                  return (
                    <li key={ext.key}>
                      <button
                        onClick={() => onSelectIssue(ext.key)}
                        aria-pressed={selectedKey === ext.key}
                        className={`w-full flex flex-wrap items-center gap-x-2 gap-y-1 px-2.5 py-1.5 rounded border text-left transition-colors ${
                          selectedKey === ext.key
                            ? 'border-emerald-600/50 bg-emerald-600/10'
                            : 'border-gray-800 hover:border-gray-700 bg-gray-950/40'
                        }`}
                      >
                        <code className="text-[11px] text-gray-400 shrink-0">{ext.key}</code>
                        <span className="text-xs text-gray-300 truncate min-w-0 flex-1 basis-32">
                          {ext.title}
                        </span>
                        <span className="text-[11px] text-gray-400 shrink-0">{ext.status}</span>
                        {[...phrases.values()].map(p => (
                          <span
                            key={p.text}
                            className={`px-1.5 py-0.5 rounded text-[10px] border ${
                              p.benching
                                ? 'bg-red-500/15 text-red-300 border-red-500/30'
                                : 'bg-gray-800/70 text-gray-400 border-gray-700/50'
                            }`}
                          >
                            {p.text}
                          </span>
                        ))}
                      </button>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------

function ChildRow({
  child,
  terminal,
  selected,
  busy,
  onSelect,
  onClaim,
  onUnclaim,
}: {
  child: TrackerMapChild
  terminal: string[]
  selected: boolean
  busy: boolean
  onSelect: () => void
  onClaim: () => void
  onUnclaim: () => void
}) {
  const isTerminal = terminal.includes(child.status)
  return (
    <li className="flex items-center gap-2">
      <button
        onClick={onSelect}
        aria-pressed={selected}
        className={`flex-1 min-w-0 flex flex-wrap items-center gap-x-2 gap-y-1 px-2.5 py-1.5 rounded border text-left transition-colors ${
          selected ? 'border-emerald-600/50 bg-emerald-600/10' : 'border-gray-800 hover:border-gray-700 bg-gray-950/40'
        }`}
      >
        <code className="text-[11px] text-gray-400 shrink-0">{child.key}</code>
        <span className="text-xs text-gray-200 truncate min-w-0 flex-1 basis-40">{child.title}</span>
        <span className="text-[11px] text-gray-400 shrink-0">{child.status}</span>
        {/* State as text — the graph's colors never carry meaning alone. */}
        {child.frontier && (
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
            frontier
          </span>
        )}
        {child.assignee && (
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-blue-500/15 text-blue-300 border border-blue-500/30">
            claimed by {child.assignee}
          </span>
        )}
        {child.blocked_by.length > 0 && (
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-red-500/15 text-red-300 border border-red-500/30">
            blocked by {child.blocked_by.join(', ')}
          </span>
        )}
      </button>
      {!isTerminal && !child.assignee && (
        <button
          onClick={onClaim}
          disabled={busy}
          aria-label={`Claim ${child.key}`}
          className="px-2.5 py-1.5 rounded text-[11px] bg-gray-800 hover:bg-emerald-700 text-gray-300 hover:text-white disabled:opacity-40 shrink-0"
        >
          Claim
        </button>
      )}
      {child.assignee && (
        <button
          onClick={onUnclaim}
          disabled={busy}
          aria-label={`Unclaim ${child.key} (claimed by ${child.assignee})`}
          title="Release the claim — the ordinary recovery exit"
          className="px-2.5 py-1.5 rounded text-[11px] bg-gray-800 hover:bg-gray-700 text-gray-400 disabled:opacity-40 shrink-0"
        >
          Unclaim
        </button>
      )}
    </li>
  )
}

// ---------------------------------------------------------------------------
// Map body editor with optimistic concurrency
// ---------------------------------------------------------------------------

function MapBodyEditor({
  mapKey,
  map,
  onSaved,
}: {
  mapKey: string
  map: TrackerIssue
  onSaved: () => Promise<void>
}) {
  const { showSnackbar } = useStore()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(map.body)
  const [saving, setSaving] = useState(false)
  /** The current version reported by a stale-write 409, while the draft stays. */
  const [conflictVersion, setConflictVersion] = useState<string | null>(null)

  // Switching maps always resets the editor.
  useEffect(() => {
    setEditing(false)
    setDraft(map.body)
    setConflictVersion(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map.key])

  // A fresh body/version from the server syncs the editor only when no draft
  // is in flight — a background reload must never clobber an edit.
  useEffect(() => {
    if (editing) return
    setDraft(map.body)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map.body, map.updated_at])

  const save = async (expectedVersion: string | null) => {
    setSaving(true)
    try {
      await api.updateTrackerIssue(mapKey, {
        body: draft,
        actor: 'dashboard',
        ...(expectedVersion ? { expected_updated_at: expectedVersion } : {}),
      })
      setEditing(false)
      setConflictVersion(null)
      await onSaved()
      showSnackbar({ type: 'success', message: `${mapKey} updated` })
    } catch (err) {
      const current = conflictDetail(err)?.current_updated_at
      if ((err as ApiError).status === 409 && current) {
        // The map changed under this edit. The draft is NOT discarded and
        // nothing was overwritten — offer the re-read/retry path.
        setConflictVersion(String(current))
      } else {
        showSnackbar({ type: 'error', message: errorText(err) })
      }
    } finally {
      setSaving(false)
    }
  }

  const rereadAndRetry = async () => {
    // Re-read the map to learn the fresh version (the textarea keeps the
    // draft), then retry against it. A further race re-arms the banner.
    try {
      const fresh = await api.getTrackerIssue(mapKey)
      setConflictVersion(null)
      await save(fresh.updated_at)
    } catch (err) {
      showSnackbar({ type: 'error', message: errorText(err) })
    }
  }

  if (!editing) {
    return (
      <section aria-label="Map body">
        <div className="flex items-center justify-between mb-1.5">
          <h4 className="text-[11px] font-medium uppercase tracking-wide text-gray-400">
            Destination &amp; notes
          </h4>
          <button
            onClick={() => {
              setDraft(map.body)
              setEditing(true)
            }}
            className="text-[11px] text-gray-400 hover:text-gray-300"
          >
            Edit
          </button>
        </div>
        {map.body ? (
          <pre className="text-xs text-gray-300 whitespace-pre-wrap font-mono leading-relaxed bg-gray-950/50 border border-gray-800/70 rounded-lg p-3 max-h-64 overflow-y-auto">
            {map.body}
          </pre>
        ) : (
          <p className="text-xs text-gray-400 px-1">No destination written yet.</p>
        )}
      </section>
    )
  }

  return (
    <section aria-label="Edit map body">
      <h4 className="text-[11px] font-medium uppercase tracking-wide text-gray-400 mb-1.5">
        Destination &amp; notes — editing
      </h4>
      {conflictVersion && (
        <div
          role="alert"
          data-testid="map-body-conflict"
          className="mb-2 rounded-lg border border-amber-600/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
        >
          <p>
            This map changed while you were editing (current version{' '}
            <code>{conflictVersion}</code>). Your draft below is untouched — re-read and retry to
            apply it against the fresh version, or discard it.
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
              onClick={() => {
                setDraft(map.body)
                setConflictVersion(null)
                setEditing(false)
              }}
              className="px-2.5 py-1 rounded border border-gray-700 text-gray-300"
            >
              Discard draft
            </button>
          </div>
        </div>
      )}
      <textarea
        value={draft}
        onChange={e => setDraft(e.target.value)}
        rows={8}
        aria-label="Map body"
        className="w-full px-3 py-2 rounded bg-gray-900 border border-gray-800 text-sm text-gray-200 font-mono leading-relaxed focus:outline-none focus:border-emerald-600/50"
      />
      <div className="flex gap-2 mt-2">
        <button
          onClick={() => save(map.updated_at)}
          disabled={saving}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs disabled:opacity-40"
        >
          {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
          Save
        </button>
        <button
          onClick={() => {
            setDraft(map.body)
            setConflictVersion(null)
            setEditing(false)
          }}
          className="px-3 py-1.5 rounded border border-gray-800 text-xs text-gray-400"
        >
          Cancel
        </button>
        <span className="text-[11px] text-gray-400 self-center">
          version <code>{map.updated_at ?? '—'}</code>
        </span>
      </div>
    </section>
  )
}
