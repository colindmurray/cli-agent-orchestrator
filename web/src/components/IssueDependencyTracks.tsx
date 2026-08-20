import { AlertTriangle, ChevronDown, ChevronRight } from 'lucide-react'
import type { IssueDependencyPlan, IssueDependencyStage } from '../lib/issueGraph'

export function IssueDependencyTracks({
  plan,
  collapsed,
  onToggleScope,
  onExpandAll,
  onCollapseAll,
  onSelectIssue,
}: {
  plan: IssueDependencyPlan
  collapsed: Set<string>
  onToggleScope: (key: string) => void
  onExpandAll: () => void
  onCollapseAll: () => void
  onSelectIssue: (key: string) => void
}) {
  return (
    <div className="space-y-3" aria-label="Dependency work tracks">
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-gray-800 bg-gray-950/40 px-4 py-3 text-xs">
        <div>
          <div className="font-medium text-gray-200">
            {plan.tracks.length} {plan.tracks.length === 1 ? 'work track' : 'work tracks'}
          </div>
          <div className="mt-0.5 text-[11px] text-gray-500">
            Tracks have no blocker edges between them and can proceed independently.
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          <span className="rounded bg-red-500/10 px-2 py-1 text-red-300">
            {plan.totalDependencyCount} blocker {plan.totalDependencyCount === 1 ? 'link' : 'links'} total
          </span>
          <span className="rounded bg-gray-800 px-2 py-1 text-gray-400">
            {plan.openDependencyCount} open · {plan.clearedDependencyCount} cleared
          </span>
          <span className="rounded bg-sky-500/10 px-2 py-1 text-sky-300">{plan.visibleDependencyCount} visible</span>
          <span className={`rounded px-2 py-1 ${plan.hiddenDependencyCount ? 'bg-amber-500/10 text-amber-300' : 'bg-gray-800 text-gray-500'}`}>
            {plan.hiddenDependencyCount} hidden
          </span>
          <span className="mx-0.5 h-4 w-px bg-gray-800" aria-hidden="true" />
          <button
            type="button"
            onClick={onExpandAll}
            disabled={collapsed.size === 0}
            className="rounded border border-gray-700 px-2 py-1 text-gray-300 hover:border-emerald-600/60 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Expand all scopes
          </button>
          <button
            type="button"
            onClick={onCollapseAll}
            className="rounded border border-gray-700 px-2 py-1 text-gray-300 hover:border-amber-600/60"
          >
            Collapse all scopes
          </button>
        </div>
      </div>

      {plan.cycles.length > 0 && (
        <div role="note" className="flex items-start gap-2 rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          <span>
            Blocker cycle detected: {plan.cycles.map(cycle => cycle.join(' ↔ ')).join('; ')}.
            Nodes in a cycle share a stage because no valid order exists until an edge is removed.
          </span>
        </div>
      )}

      {plan.tracks.map(track => {
        const maxStage = track.stages[track.stages.length - 1]?.index ?? 0
        return <section key={track.id} className="overflow-hidden rounded-lg border border-gray-800 bg-gray-950/20">
          <header className="flex flex-wrap items-center justify-between gap-2 border-b border-gray-800 bg-gray-950/50 px-4 py-3">
            <div>
              <h3 className="text-xs font-medium text-gray-200">
                {track.independent ? 'Independent work' : `Track ${track.id}`}
              </h3>
              <p className="mt-0.5 text-[10px] text-gray-600">
                {track.total} scoped issues · {track.terminal} complete · {track.active} active · {track.blocked} blocked
              </p>
            </div>
            <div className="h-1.5 w-32 overflow-hidden rounded-full bg-gray-800" aria-label={`${track.terminal} of ${track.total} issues complete`}>
              <div
                className="h-full rounded-full bg-emerald-500"
                style={{ width: `${track.total ? (track.terminal / track.total) * 100 : 0}%` }}
              />
            </div>
          </header>
          <div className="overflow-x-auto p-3">
            <div className="flex min-w-max items-start gap-3">
              {track.stages.map(stage => (
                <DependencyStageColumn
                  key={stage.index}
                  stage={stage}
                  maxStage={maxStage}
                  collapsed={collapsed}
                  onToggleScope={onToggleScope}
                  onSelectIssue={onSelectIssue}
                />
              ))}
            </div>
          </div>
        </section>
      })}
    </div>
  )
}

function DependencyStageColumn({
  stage,
  maxStage,
  collapsed,
  onToggleScope,
  onSelectIssue,
}: {
  stage: IssueDependencyStage
  maxStage: number
  collapsed: Set<string>
  onToggleScope: (key: string) => void
  onSelectIssue: (key: string) => void
}) {
  const label = stage.index === 0
    ? 'Parallel work'
    : stage.index === maxStage ? 'Integration' : `Sequence ${stage.index + 1}`
  return (
    <div className="w-64 shrink-0">
      <div className="mb-2 flex items-baseline justify-between gap-2 px-1">
        <span className="text-[10px] font-medium uppercase tracking-wide text-gray-500">{label}</span>
        <span className="font-mono text-[10px] text-gray-700">stage {stage.index}</span>
      </div>
      <div className="space-y-2">
        {stage.nodes.map(node => {
          const childCount = Number((node.issue as { child_count?: number }).child_count ?? 0)
          const hasScope = node.hiddenScopeCount > 0 || childCount > 0
          return (
            <article
              key={node.key}
              className={`rounded border p-3 ${node.cyclic ? 'border-red-500/40 bg-red-500/5' : 'border-gray-800 bg-gray-950/70'}`}
            >
              <button type="button" onClick={() => onSelectIssue(node.key)} className="block w-full text-left">
                <span className="font-mono text-[10px] text-gray-500">{node.key}</span>
                <span className="mt-1 block text-xs font-medium leading-snug text-gray-200">{node.issue.title}</span>
              </button>
              <div className="mt-2 flex flex-wrap items-center gap-1 text-[10px]">
                <span className="rounded bg-gray-800 px-1.5 py-0.5 text-gray-400">{node.issue.kind}</span>
                <span className="rounded bg-gray-800 px-1.5 py-0.5 text-gray-400">{node.issue.status}</span>
                {node.external && (
                  <span className="rounded bg-orange-500/10 px-1.5 py-0.5 text-orange-300">external</span>
                )}
                {node.unresolvedBlockers.length > 0 ? (
                  <span className="rounded bg-red-500/10 px-1.5 py-0.5 text-red-300">
                    blocked by {node.unresolvedBlockers.join(', ')}
                  </span>
                ) : node.clearedBlockers.length > 0 ? (
                  <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-emerald-300">prerequisites cleared</span>
                ) : node.blocking.length > 0 ? (
                  <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-emerald-300">upstream</span>
                ) : (
                  <span className="rounded bg-sky-500/10 px-1.5 py-0.5 text-sky-300">independent</span>
                )}
              </div>
              {hasScope && (
                <button
                  type="button"
                  onClick={() => onToggleScope(node.key)}
                  aria-label={`${collapsed.has(node.key) ? 'Expand' : 'Collapse'} nested scope for ${node.key}`}
                  aria-expanded={!collapsed.has(node.key)}
                  className="mt-2 inline-flex items-center gap-1 rounded border border-gray-800 bg-gray-900/70 px-2 py-1 text-[10px] text-gray-400 hover:border-gray-700 hover:text-gray-100"
                >
                  {collapsed.has(node.key) ? <ChevronRight size={11} /> : <ChevronDown size={11} />}
                  {collapsed.has(node.key) ? 'Expand' : 'Collapse'} {node.hiddenScopeCount || childCount} nested
                  {node.hiddenDependencyCount > 0 ? ` · ${node.hiddenDependencyCount} blocker ${node.hiddenDependencyCount === 1 ? 'link' : 'links'}` : ''}
                </button>
              )}
            </article>
          )
        })}
      </div>
    </div>
  )
}
