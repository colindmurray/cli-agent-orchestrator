const BASE = ''  // Vite proxy handles routing to backend

/**
 * Error thrown by fetchJSON on a non-OK response. Carries the HTTP status and
 * the server's `detail` string so callers can branch on them (e.g. the graph
 * export's 422 secret-gate path surfaces `detail` — the matched PATTERN name
 * only, never the memory bytes). `message` stays "<status> <statusText>" for
 * back-compat with existing callers.
 */
export interface ApiError extends Error {
  status?: number
  detail?: string
  /** The parsed JSON error body, when there was one (e.g. §5.3 `{errors}`). */
  body?: unknown
}

/** The structured detail of a typed conflict (409), when the server sent one:
 * `{message, code, observed_assignee}` on a lost claim, `{message, code,
 * current_updated_at}` on a stale optimistic-concurrency write. */
export function conflictDetail(err: unknown): Record<string, unknown> | null {
  const apiErr = err as ApiError | undefined
  const detail = (apiErr?.body as { detail?: unknown } | undefined)?.detail
  return detail && typeof detail === 'object' ? (detail as Record<string, unknown>) : null
}

/** The server's explanation for a failed call, falling back to the message. */
export function errorText(err: unknown): string {
  const apiErr = err as ApiError | undefined
  return apiErr?.detail || apiErr?.message || String(err)
}

async function fetchJSON<T>(url: string, opts?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), opts?.timeoutMs ?? 10000)
  // A caller-supplied signal (e.g. superseded-search cancellation) composes
  // with the internal timeout instead of being overwritten by it.
  const externalSignal = opts?.signal
  const propagateAbort = () => controller.abort()
  if (externalSignal) {
    if (externalSignal.aborted) controller.abort()
    else externalSignal.addEventListener('abort', propagateAbort)
  }
  try {
    const res = await fetch(`${BASE}${url}`, { ...opts, signal: controller.signal })
    if (!res.ok) {
      // Best-effort read of the JSON error body to expose the server's
      // `detail` without leaking a full response. A non-JSON body is fine —
      // detail just stays undefined.
      let detail: string | undefined
      let body: unknown
      try {
        body = await res.json()
        if (body && typeof (body as { detail?: unknown }).detail === 'string') {
          detail = (body as { detail: string }).detail
        } else if (body && typeof (body as { detail?: unknown }).detail === 'object' && (body as { detail?: unknown }).detail !== null) {
          // Typed conflicts carry structured detail ({message, code, ...}) —
          // e.g. a lost claim names the observed owner, a stale write names the
          // current version. Surface the message as `detail`; the structured
          // fields stay readable on `err.body.detail`.
          const d = (body as { detail: Record<string, unknown> }).detail
          if (typeof d.message === 'string') detail = d.message
        }
      } catch { /* non-JSON error body */ }
      const err: ApiError = new Error(`${res.status} ${res.statusText}`)
      err.status = res.status
      err.detail = detail
      err.body = body
      throw err
    }
    return res.json()
  } finally {
    clearTimeout(timeout)
    if (externalSignal) externalSignal.removeEventListener('abort', propagateAbort)
  }
}

export type SessionLifecycleValue = 'working' | 'pausing' | 'paused' | 'complete' | 'stopped'

// The DECLARED dimension. Deliberately not merged into `Session`, whose
// `status` is tmux attach state (`active`/`detached`) and is a third
// unrelated thing again.
export interface SessionLifecycle {
  session_name: string
  lifecycle: SessionLifecycleValue
  restore_to: string | null
  archived: boolean
  kind: 'campaign' | 'service'
  declared_by: string | null
  note: string | null
  pause_deadline_at: string | null
  epoch: number
  declared: boolean
  suppresses_marshal?: boolean
  pause_overdue?: boolean
  unreadable?: string
}

// ---------------------------------------------------------------------------
// Fleet cohort operations (M3-C)
// ---------------------------------------------------------------------------

/**
 * A whole-fleet Pause/Stop/Resume. Deliberately a different dimension again
 * from both `SessionLifecycle.lifecycle` (what the session has declared) and
 * `Terminal.lifecycle_state` (observed pane liveness): this is *an operation
 * somebody performed*, with its own state machine and its own history.
 */
export type CohortOperationKind = 'pause' | 'stop' | 'resume'
export type CohortMode = 'safe' | 'force'
export type CohortState =
  | 'preparing'
  | 'draining-to-boundary'
  | 'interrupting'
  | 'tearing-down'
  | 'paused'
  | 'stopped'
  | 'restoring'
  | 'reconciliation-required'
  | 'settled'

/**
 * One member's outcome. All of `restored-exact`, `restored-fresh`, `failed`
 * and `unresumable` are *terminal*: a fleet that came back with one worker
 * missing is a settled fleet that lost a worker, not an unresolved one. Only
 * `reconciliation-required` means "we do not know yet".
 */
export type CohortMemberOutcome =
  | 'pending'
  | 'excluded-historical'
  | 'drained'
  | 'interrupted'
  | 'already-idle'
  | 'parked'
  | 'exited'
  | 'stopped'
  | 'restored-exact'
  | 'restored-fresh'
  | 'failed'
  | 'unresumable'
  | 'reconciliation-required'

/** Which stable agent kept which native session across the operation. */
export interface CohortContinuity {
  agent_id: string
  role: 'supervisor' | 'worker'
  included: boolean
  exclusion_reason: string | null
  /** The native-session lineage. Distinct from both ids below. */
  lineage_id: string | null
  harness: string | null
  native_session_id: string | null
  /** One run of a stable agent. A restore makes a new one; the agent persists. */
  incarnation_id: string | null
  terminal_id: string | null
  generation: string | null
  final_state: CohortMemberOutcome
}

export interface CohortProvenance {
  operation_id: string
  session_name: string
  operation_kind: CohortOperationKind
  state: CohortState
  state_epoch: number
  lifecycle_epoch: number
  lifecycle_observation: SessionLifecycleValue
  roster_revision: string
  member_snapshot_digest: string
  requested_mode: CohortMode
  current_mode: CohortMode
  /** Safe never becomes force implicitly; a promotion always has a receipt. */
  promoted_to_force: boolean
  promotion_receipt_digest: string | null
  promoted_by: string | null
  initiator_kind: 'operator' | 'supervisor'
  initiated_by: string
  /** The Stop a Resume descends from. Null for Pause/Stop. */
  source_operation_id: string | null
  resume_target: string | null
  member_outcomes: Partial<Record<CohortMemberOutcome, number>>
  continuity: CohortContinuity[]
  /** True while the operation is in `reconciliation-required` and can be continued. */
  retryable: boolean
  /** Every operator retry already performed, each with its opaque receipt. */
  retries: CohortRetry[]
  /** The durable reason the last attempt stopped short, if it did. */
  reconciliation_reason: string | null
}

export interface CohortRetry {
  transition_id: string
  from_state_epoch: number
  actor: string
  reason: string | null
  receipt_digest: string | null
  created_at: string
}

export interface CohortOperation {
  operation_id: string
  session_name: string
  operation_kind: CohortOperationKind
  requested_mode: CohortMode
  current_mode: CohortMode
  initiator_kind: 'operator' | 'supervisor'
  initiated_by: string
  state: CohortState
  state_epoch: number
  lifecycle_epoch: number
  source_operation_id: string | null
  resume_target: string | null
  created_at: string
  updated_at: string
  /** Only on the single-operation read; the list projection is deliberately light. */
  provenance?: CohortProvenance
}

// ---------------------------------------------------------------------------
// Safe drain, task occurrences, supervisor wakes (M3-D)
// ---------------------------------------------------------------------------

/**
 * How complete the summary/artifact seed a fresh successor would inherit is.
 *
 * `truncated` is the dangerous value and the reason this is a three-state
 * vocabulary rather than a boolean: it reads as context while missing the part
 * that mattered, so a fresh worker started on it continues confidently from
 * the wrong place. Only `complete` is enough.
 */
export type SeedQuality = 'complete' | 'truncated' | 'empty'

export type DrainIntent = 'pause' | 'stop'
export type DrainState = 'pending' | 'complete' | 'reconciliation-required'
export type DrainMemberState =
  | 'pending'
  | 'drained'
  | 'already-idle'
  | 'parked'
  | 'reconciliation-required'

export interface DrainMember {
  drain_id: string
  agent_id: string
  role: 'supervisor' | 'worker'
  terminal_id: string | null
  observed_state: 'active' | 'idle' | 'parked' | 'exited' | 'unknown'
  /** Derived from the drain and the agent, so a retry never steers twice. */
  steer_control_id: string
  steer_state: 'not-required' | 'sent' | 'refused' | 'unproven'
  task_occurrence_id: string | null
  boundary_digest: string | null
  report_digest: string | null
  checkpoint_digest: string | null
  /** Stop only: CAO's teardown is announced before the pane disappears. */
  teardown_state: 'not-required' | 'requested' | 'unproven'
  member_state: DrainMemberState
  detail: string | null
  revision: number
}

export interface DrainProvenance {
  drain_id: string
  session_name: string
  intent: DrainIntent
  state: DrainState
  attempt: number
  member_outcomes: Partial<Record<DrainMemberState, number>>
  undecided: string[]
  steered: string[]
  teardown_requested: string[]
  /** Present only on a `complete` drain. An unfinished one has nothing to spend. */
  receipt_digest: string | null
  retryable: boolean
  reconciliation_reason: string | null
  /** Derived from the *stalled* drain; a drain never promotes itself. */
  force_promotion_receipt: string | null
}

export interface SessionDrain {
  drain_id: string
  session_name: string
  intent: DrainIntent
  state: DrainState
  attempt: number
  lifecycle_epoch: number
  roster_revision: string
  receipt_digest: string | null
  reconciliation_reason: string | null
  initiated_by: string
  created_at: string
  updated_at: string
  members?: DrainMember[]
  provenance?: DrainProvenance
}

/**
 * A task/round occurrence. Deliberately *not* keyed by terminal generation or
 * native conversation: a stable agent outlives many of each, so binding a task
 * to one of them is how a resumed pane inherits a finished round.
 */
export interface TaskOccurrence {
  task_occurrence_id: string
  session_name: string
  agent_id: string
  round_index: number
  state: 'open' | 'finalized'
  /** The exact effect that executed it, carried alongside — never as its id. */
  incarnation_id: string
  terminal_id: string
  generation: string | null
  current: TaskOccurrenceEvidence
  finalized: TaskOccurrenceEvidence & {
    disposition: 'reported' | 'abandoned' | 'superseded' | 'lost' | null
    finalized_by: string | null
    finalized_at: string | null
  }
  revision: number
  extensions?: TaskOccurrenceExtension[]
  seed_verdict?: SeedVerdict
}

export interface TaskOccurrenceEvidence {
  boundary_digest: string | null
  report_digest: string | null
  checkpoint_digest: string | null
  summary_seed_digest: string | null
  artifact_seed_digest: string | null
  seed_quality: SeedQuality | null
}

/** Opaque and versioned. An unrecognised kind is preserved, never interpreted. */
export interface TaskOccurrenceExtension {
  task_occurrence_id: string
  extension_id: string
  extension_kind: string
  extension_version: string
  decider: string
  claims_final: boolean
  recognized: boolean
  routing_state: 'pending-decider' | 'routed'
}

export interface SeedVerdict {
  family: 'current' | 'finalized'
  quality: SeedQuality | null
  sufficient_for_fresh_start: boolean
  reason: string | null
}

export interface ReconciliationWake {
  wake_id: string
  session_name: string
  source_kind: 'resume-and-start' | 'paused-to-working'
  source_operation_id: string
  delivery_state: 'claimed' | 'delivered' | 'undelivered'
  reason_code: string | null
  detail: string | null
  receipt_digest: string | null
  /** The exact text the supervisor was sent, so "it was told" is checkable. */
  message: { text: string; counts: Record<string, number>; truncated: boolean } | null
}

export interface StopImpactWorker {
  terminal_id: string
  provider: string
  agent_profile: string | null
  reason: string | null
  resumable: boolean
}

export interface StopImpact {
  session_name: string
  live_workers: number
  resumable: StopImpactWorker[]
  not_resumable: StopImpactWorker[]
  resume_machinery_available: boolean
  resume_machinery_reason: string
  one_way_for_every_worker?: boolean
  unreadable?: string
}

export interface Session {
  id: string
  name: string
  status: string
}

export interface Terminal {
  id: string
  name: string
  provider: string
  session_name: string
  agent_profile: string | null
  status: string | null
  last_active: string | null
  assigned_model?: string | null
  assigned_effort?: string | null
  assigned_quota_provider?: string | null
  assigned_route_state?: 'present' | 'absent' | 'unreadable' | null
}

export interface SessionDetail {
  session: Session
  terminals: TerminalMeta[]
}

/** One auditable input to the terminal status-fusion decision. */
export interface TerminalStatusSignal {
  /** Open by design: the server may add evidence sources without a web release. */
  name: string
  /** `available` | `absent` | `unreadable` today; open for the same reason. */
  state: string
  value?: string | number | boolean | null
  detail?: string | null
}

/**
 * One row of the shared terminal projection.
 *
 * This is `terminal_projection.project_row`'s ACTUAL output, key for key —
 * `/sessions/{name}` returns `project_session(...)` verbatim
 * (services/session_service.py), so anything this interface omits is a field
 * the dashboard cannot see and anything it invents is a field the server never
 * sends. It previously declared seven keys against the real twenty-five, and
 * one of those seven — `created_at` — is not among them: no `TerminalModel`
 * column and no projection key by that name exists, so `t.created_at` was
 * permanently `undefined` in production while three test fixtures fed it a
 * value and made the dead branch look covered. That is the specific failure
 * mode this shape exists to prevent, so keep it driven from `project_row`.
 *
 * `terminal_id`/`name`/`session_name` are the canonical spellings both human
 * views are required to agree on; `id`/`tmux_session`/`tmux_window` are the
 * pre-existing display keys the projection deliberately keeps alongside them.
 */
export interface TerminalMeta {
  // Identity — canonical spellings.
  terminal_id: string
  name: string | null
  session_name: string | null
  // Identity — the pre-existing display keys, kept by the projection.
  id: string
  tmux_session: string | null
  tmux_window: string | null
  // Provenance.
  provider: string | null
  agent_profile: string | null
  caller_id: string | null
  generation: string | null
  callback_target_generation: string | null
  protocol_vintage: string
  // Recorded pane identity.
  server_socket_path: string | null
  session_id: string | null
  window_id: string | null
  pane_id: string | null
  pane_pid: number | string | null
  native_session_id: string | null
  // Honest requested route — durable reservation, never a footer parse.
  // Harness (provider) and AI provider stay separate labels; model/effort
  // always render with the exact qualifier `requested, not observed`.
  assigned_model: string | null
  assigned_effort: string | null
  assigned_quota_provider: string | null
  assigned_route_state: 'present' | 'absent' | 'unreadable' | null
  // Observed liveness — never the stored lifecycle.
  lifecycle_state: string
  lifecycle_reason: string | null
  superseded_by_terminal_id: string | null
  superseded_by_generation: string | null
  /** Stated rather than inferred from `status`: no classification is coming. */
  fifo_monitored: boolean
  /** Provider status for a live pane; the lifecycle word otherwise. */
  status: string | null
  /** Strength of the evidence behind `status`, not a task-quality judgement. */
  status_confidence: string
  /** Server-authored explanation of how the status was selected. */
  status_reason: string
  /** Every input used by the status-fusion decision. */
  status_signals: TerminalStatusSignal[]
  /** A working claim contradicted by both render and activity clocks. */
  wedged: boolean
  /** When CAO last sent input to this pane; not general model activity. */
  last_active: string | null
}

// ── Conductor annotations (work-state design §9.5) ─────────────────────
// The wire shape of GET /annotations. The fork is a bounded, confined
// pass-through: `kind`, `semantic_role` and `subject.type` are the conductor's
// vocabulary and are deliberately typed as open strings here, so the conductor
// can add to any of them without a change on this side. The renderer resolves
// `semantic_role` against the six roles in design-tokens/tokens.json and falls
// back to `neutral` for anything else.

export interface AnnotationSubject {
  /** "terminal" | "task" | "campaign" today; open by design. */
  type: string
  terminal_id?: string | null
  /**
   * The generation this annotation was derived against. The renderer fences on
   * it: an annotation whose generation differs from the projection row's is
   * dropped, so a stale claim cannot outlive the obligation that produced it.
   */
  generation?: string | null
  task_id?: string | null
  campaign?: string | null
  /**
   * A subject type invented later brings its own identifier, and the route
   * carries a bounded number of them through. Typed open here so the renderer
   * can draw whatever arrived instead of an anonymous chip — placement is
   * durable without this, identity is not.
   */
  [key: string]: string | null | undefined
}

export interface Annotation {
  namespace: string
  kind: string
  version: number
  label: string
  semantic_role: string
  priority: number
  subject: AnnotationSubject
  /** Past this the renderer greys the chip (§9.6). */
  valid_until?: string | null
  /**
   * An OPAQUE identity token. The conductor decides what identity it
   * expresses; this side only hashes it into a palette slot, which is why a
   * change of grouping policy needs no change here.
   *
   * Three states, and the renderer branches on PRESENCE AND EMPTINESS ONLY:
   * absent (or null) is a severity chip coloured by `semantic_role`; `''` is
   * an identity chip with no colour; a non-empty value is an identity chip
   * coloured by the hash.
   */
  colour_key?: string | null
  /** Derived facets only — never worker-authored free text (§7). */
  details?: Record<string, string>
  /** The conductor project directory that published it; never a path. */
  source?: string | null
}

export interface AnnotationSourceReason {
  source: string
  reason: string
}

export interface AnnotationsResponse {
  annotation_schema: string
  /** "complete" | "partial" | "truncated" | "unavailable" */
  coverage: string
  sources_read: number
  sources_failed: number
  items_dropped: number
  items_omitted: number
  /** Facets that did not fit the bounded detail bag, or had no display form. */
  facets_dropped?: number
  reasons: AnnotationSourceReason[]
  annotations: Annotation[]
}

// ── Communications catalog (communication-catalog design §7) ─────────────
// The wire shape of the three /communications routes. As with annotations,
// `kind`, `report_scope`, `content_state`, the actor fields, and every other
// conductor-authored value are OPEN strings here: the conductor owns the
// vocabulary and can extend it without a change on this side. The only closed
// sets are the envelope's own bookkeeping (coverage values, reason codes).

export interface CatalogQuarantineInfo {
  reason: string
  actor?: string | null
  quarantined_at?: string | null
  receipt_sha256?: string | null
}

/** One attachment or body document exactly as the publisher wrote it. */
export interface CatalogDocumentEntry {
  attachment_id: string
  document_id: string
  role: string
  display_name: string
  media_type: string
  sha256: string
  byte_size: number
  blob_id: string
  content_state: string
  capture_kind?: string | null
  redaction_applied?: boolean | null
  provenance?: Record<string, unknown> | null
  quarantine?: CatalogQuarantineInfo | null
  /** The publisher allows extra keys (backend `extra="allow"` wire model, mirrored here). */
  [key: string]: unknown
}

/** Metadata for one communication; the list endpoint never carries bodies. */
export interface CommunicationListItem {
  communication_id: string
  project_id: string
  session_id?: string | null
  lane_id?: string | null
  task_occurrence_id?: string | null
  goal_version?: string | null
  kind?: string | null
  report_scope?: string | null
  authored_by_type?: string | null
  authored_by_id?: string | null
  authored_at?: string | null
  recorded_at?: string | null
  title?: string | null
  delivery_state?: string | null
  visibility?: string | null
  request_key?: string | null
  supersedes_communication_id?: string | null
  superseded_by?: string | null
  body?: CatalogDocumentEntry | null
  documents: CatalogDocumentEntry[]
  /** Extra keys allowed by the backend (`extra="allow"`), mirrored here. */
  [key: string]: unknown
}

/** Why one project contributed nothing or less than it holds. */
export interface CatalogReason {
  source: string
  reason: string
}

export interface CommunicationsListResponse {
  schema: string
  /** "complete" | "partial" | "truncated" | "unavailable" */
  coverage: string
  reasons: CatalogReason[]
  communications: CommunicationListItem[]
  next_cursor: string | null
  total: number
}

/** One communication with its exact UTF-8 body, or null plus a typed reason. */
export interface CommunicationDetailResponse {
  communication: CommunicationListItem
  content: string | null
  reason: string | null
}

/** One attachment with its exact UTF-8 content, or null plus a typed reason. */
export interface AttachmentDetailResponse {
  document: CatalogDocumentEntry
  content: string | null
  reason: string | null
}

/**
 * Known profile source values the backend can emit.
 * Using `string` (not a closed union) so new provider-discovered directories
 * and custom agent directories are accepted without repeated type widening.
 */
export type AgentProfileSource = string

export interface AgentProfileInfo {
  name: string
  description: string
  source: AgentProfileSource
  // Other enabled directories that also define this profile name (the winner
  // above is what loads). Empty/absent when the name is unique. (GH #280)
  duplicated_in?: string[]
}

export interface AgentDirsSettings {
  agent_dirs: Record<string, string>
  extra_dirs: string[]
  // Directory paths toggled OFF: kept in the list but skipped when scanning
  // for agent profiles. (GH #280/#281)
  disabled_dirs?: string[]
}

export interface InboxMessage {
  id: string
  sender_id: string
  receiver_id: string
  message: string
  status: 'pending' | 'delivered' | 'failed'
  created_at: string | null
}

export interface Flow {
  name: string
  file_path: string
  schedule: string
  agent_profile: string
  provider: string
  script: string | null
  last_run: string | null
  next_run: string | null
  enabled: boolean
  prompt_template: string | null
}

export interface ProviderInfo {
  name: string
  binary: string
  installed: boolean
}

export interface MemoryStatus {
  enabled: boolean
}

export interface MemorySummary {
  key: string
  scope: string
  scope_id: string | null
  memory_type: string
  tags: string
  created_at: string
  updated_at: string
}

export interface MemoryDetail extends MemorySummary {
  content: string
}

// ── Graph layer (Issue #348) ────────────────────────────────────────────
// Wire shape of GET /graph/{provider}. Mirrors the server's GraphView.to_dict
// (src/cli_agent_orchestrator/api/main.py get_graph_endpoint). `attrs` is an
// open bag — the renderer reads is_hub / is_orphan but the server may add more.
export interface GraphNode {
  id: string
  kind: string
  label: string
  status: string
  attrs: Record<string, unknown>
}

export interface GraphEdge {
  source: string
  target: string
  type: string
  attrs: Record<string, unknown>
}

export interface GraphView {
  nodes: GraphNode[]
  edges: GraphEdge[]
  meta: Record<string, unknown>
}

// The v3 event wire shape (§3.1): text, key, or chord.
export interface WireEvent {
  type: 'text' | 'key' | 'chord'
  text?: string
  key?: string
  chord?: string
}

// ── Control-input capabilities (§3.5) ─────────────────────────────────
// The deployed keys plus the additive §3.5 blocks Lane B gates on. Unknown
// keys are ignored by construction (the interface is additive); absence of
// a block drives the old-server degradation rows of §3.5.
export interface ProviderControlBlock {
  compact?: { events: WireEvent[] }
  stop?: { events: WireEvent[] }
  steer_chords?: string[]
  dispatch_grace_ms?: number
  // §8.6 additive Lane C blocks (absent on old servers and unproven builds).
  operator_message?: OperatorMessageBlock
  image?: ImageCapabilityBlock
}

// ── Lane C: operator messages + image attachments (§8.3/§8.4/§8.6) ─────

export interface OperatorMessageBlock {
  supported: boolean
  max_text_bytes: number
  multiline: boolean
  max_attachments: number
}

export interface ImageCapabilityBlock {
  supported: boolean
  formats: string[]
  max_bytes: number
  max_width: number
  max_height: number
  mechanism: string
  reference_template?: string
  evidence?: string
}

export interface ImageAttachmentRecord {
  attachment_id: string
  terminal_id: string
  state: 'staging' | 'ready' | 'failed' | 'removed' | 'submitted'
  format: string | null
  content_type: string | null
  width: number | null
  height: number | null
  size_bytes: number
  display_filename: string
  bound_operation_id: string | null
  error: { reason_code: string; detail: string } | null
  created_at: string
  updated_at: string
}

/** The typed refusal body of a failed upload/delete (422/409). */
export interface AttachmentRefusalBody {
  outcome?: string
  reason_code?: string
  detail?: string
  attachment?: ImageAttachmentRecord
}

export interface ControlInputCapabilities {
  protocol: string
  execution_modes: string[]
  literal_write: boolean
  bracketed_paste: boolean
  enter_required: boolean
  request_schema_versions?: number[]
  sequence?: {
    event_types: string[]
    keys: string[]
    max_events: number
    max_text_bytes: number
  }
  streaming?: {
    supported: boolean
    max_in_flight: number
    coalesce_window_ms: number
  }
  provider_controls?: Record<string, ProviderControlBlock>
  command_controls?: {
    composer_nonempty_guard: boolean
  }
}

// ── Operator macro library (§5.4) ─────────────────────────────────────
export interface MacroScope {
  kind: 'global' | 'provider' | 'profile'
  provider?: string
  profile?: string
}

export interface MacroRecord {
  id: string
  name: string
  description: string | null
  scope: MacroScope
  events: WireEvent[]
  favorite: boolean
  origin: 'builtin' | 'user'
  mutable: boolean
  builtin_kind?: 'compact' | 'stop'
  created_at: string | null
  updated_at: string | null
}

export interface MacroListResponse {
  macros: MacroRecord[]
  quarantine?: { count: number | null; path: string }
}

export interface MacroWriteBody {
  name: string
  description?: string
  scope: MacroScope
  events?: WireEvent[]
  notation?: string
  favorite?: boolean
}

export interface MacroNotationParseResult {
  events: WireEvent[]
  preview: string
}

/** The §5.3 422 error body shape (offset may be null for non-notation errors). */
export interface MacroErrorsBody {
  errors?: Array<{ offset: number | null; message: string }>
}

// Request body for POST /graph/{provider}/export. `dest` MUST be a relative
// name; the server confines it under CAO_GRAPH_EXPORT_ROOT and rejects
// absolute/traversal paths with 400.
export interface GraphExportBody {
  sink: string
  dest: string
  options?: Record<string, unknown>
}

export interface GraphExportResult {
  written_files: string[]
  sink: string
  dest: string
}

// ---------------------------------------------------------------------------
// Issue tracker
// ---------------------------------------------------------------------------

/**
 * The enumerations the server will accept, fetched from /tracker/vocabulary.
 *
 * Deliberately not hard-coded here: a dropdown that offers a status the server
 * rejects is a bug the UI cannot detect, and this list has already grown once
 * (P0 arrived when the real ledger turned out to use it).
 */
export interface TrackerVocabulary {
  statuses: string[]
  terminal_statuses: string[]
  item_kinds?: string[]
  statuses_by_kind?: Record<string, string[]>
  terminal_statuses_by_kind?: Record<string, string[]>
  severities: string[]
  scope_kinds: string[]
  link_kinds: string[]
  project_statuses: string[]
}

export interface TrackerScope {
  id: number
  project_id?: string
  kind: string
  value: string
  created_at: string | null
}

export interface TrackerProject {
  id: string
  name: string
  description: string
  status: string
  issue_prefix: string
  next_issue_number: number
  created_at: string | null
  updated_at: string | null
  counts?: { total: number; open: number; by_status?: Record<string, number>; by_kind?: Record<string, { total: number; open: number }>; all_total?: number; all_open?: number }
  scopes?: TrackerScope[]
}

export interface TrackerIssueBrief {
  key: string
  kind: string
  title: string
  status: string
  severity: string
  assignee: string | null
  favorite: boolean
  session_name: string | null
  terminal_id: string | null
  created_at: string | null
  updated_at: string | null
}

export interface TrackerProjectSessionSummary {
  name: string
  status: string
  live: boolean
  associated_by: string[]
  worker_count: number
  active_workers: number
  providers: string[]
  workdirs: string[]
  issue_count: number
  artifact_count: number
  first_seen: string | null
  last_seen: string | null
}

export interface TrackerProjectSessionTerminal {
  terminal_id: string
  session_name: string
  name: string | null
  provider: string | null
  agent_profile: string | null
  caller_id: string | null
  generation: string | null
  native_session_id: string | null
  protocol_vintage: string | null
  lifecycle_state: string | null
  status: string | null
  last_active: string | null
  working_directory?: string | null
  pane_id?: string | null
  wedged?: boolean
  issue_keys: string[]
  snapshot_available: boolean
  log_available: boolean
}

export interface TrackerProjectSessionDetail extends TrackerProjectSessionSummary {
  terminals: TrackerProjectSessionTerminal[]
  issues: TrackerIssueBrief[]
}

export interface TrackerProjectSessions {
  project_id: string
  total: number
  active: number
  historical: number
  sessions: TrackerProjectSessionSummary[]
}

export interface TrackerProjectHome {
  project_id: string
  issues: {
    open: number
    in_progress: number
    favorites: TrackerIssueBrief[]
    urgent: TrackerIssueBrief[]
    recent: TrackerIssueBrief[]
  }
  sessions: {
    total: number
    active: number
    historical: number
    recent: TrackerProjectSessionSummary[]
  }
}

export interface TrackerComment {
  id: number
  author: string | null
  body: string
  important: boolean
  created_at: string | null
  /** Parent issue clock after this durable comment effect committed. */
  updated_at?: string | null
  /** Audit-event identity for this creation effect. */
  effect_id?: number
}

/** Response of the importance PATCH: the transition outcome, not a comment row.
 * `updated_at` is the parent issue's bumped timestamp and is present only on a
 * changed write. */
export interface TrackerCommentImportanceResult {
  id: number
  issue_key: string
  important: boolean
  changed: boolean
  updated_at?: string | null
}

export interface TrackerEvent {
  id: number
  actor: string | null
  kind: string
  field: string | null
  old_value: string | null
  new_value: string | null
  created_at: string | null
}

export interface TrackerLink {
  id: number
  kind: string
  from_key: string
  to_key: string
}

export interface TrackerIssue {
  key: string
  project_id: string
  kind: 'project' | 'bug' | 'feature' | 'milestone' | 'goal' | 'epic' | 'story' | 'task' | 'issue'
  title: string
  body: string
  status: string
  severity: string
  component: string | null
  reporter: string | null
  assignee: string | null
  labels: string[]
  collaborators: string[]
  branches: string[]
  worktrees: string[]
  pull_requests: string[]
  failing_command: string | null
  reproduction_steps: string | null
  expected_outcome: string | null
  actual_outcome: string | null
  evidence: string | null
  observed_revision: string | null
  resolution: string | null
  session_name: string | null
  terminal_id: string | null
  source_path: string | null
  duplicate_of: string | null
  origin: string
  favorite: boolean
  created_at: string | null
  updated_at: string | null
  closed_at: string | null
  /** Present only on the create response: which scope matched. */
  resolved_by?: string | null
  /** Present only on the detail response. */
  comments?: TrackerComment[]
  events?: TrackerEvent[]
  links?: TrackerLink[]
}

export interface TrackerIssuePage {
  total: number
  limit: number
  offset: number
  issues: TrackerIssue[]
}

export interface TrackerIssueFilters {
  projectId?: string
  kind?: 'project' | 'bug' | 'feature' | 'milestone' | 'goal' | 'epic' | 'story' | 'task' | 'issue' | 'all'
  status?: string[]
  severity?: string[]
  component?: string
  assignee?: string
  reporter?: string
  label?: string[]
  withoutLabel?: string[]
  unlabeled?: boolean
  q?: string
  openOnly?: boolean
  limit?: number
  offset?: number
  order?: string
}

export interface TrackerLabelFacet {
  label: string
  total: number
  open: number
}

// ---------------------------------------------------------------------------
// Similar issues — POST /tracker/issues/similar (M2.4a contract, consumed by
// the M2.5 pre-filing candidates panel).

/**
 * The create/search fields a similarity draft may carry, mirroring
 * DRAFT_FIELDS in services/issue_similar.py. Server-owned identity
 * (key/project_id), status, and relation fields are refused by the service,
 * so this interface simply does not declare them.
 */
export interface SimilarIssueDraft {
  title?: string
  kind?: string
  body?: string
  severity?: string
  component?: string
  reporter?: string
  assignee?: string
  labels?: string[]
  failing_command?: string
  reproduction_steps?: string
  expected_outcome?: string
  actual_outcome?: string
  evidence?: string
  observed_revision?: string
}

/** The advisory similar-issues probe: exactly one of issue_key/draft and
 * exactly one of project_ids/all_projects — the service owns both XOR
 * refusals. Read-only by contract: it never files, links, or mutates. */
export interface SimilarIssuesRequest {
  issue_key?: string
  draft?: SimilarIssueDraft
  project_ids?: string[]
  all_projects?: boolean
  limit?: number
  mode?: 'lexical' | 'semantic' | 'hybrid'
}

/** A confirmed duplicate of a returned hit, expanded one level beside it. */
export interface SimilarDuplicateExpansion {
  duplicate_of: string
  issue: TrackerIssue
}

/** One bounded draft probe that contributed a candidate to similarity RRF. */
export interface SimilarProbeContribution {
  label: string
  query: string
  weight: number
  original_rank: number
  original_score: number | null
}

/** A malformed native duplicate source whose canonical target is ambiguous. */
export interface SimilarDuplicateConflict {
  code: 'multiple-native-duplicate-targets' | string
  message: string
  duplicate_key: string
  canonical_keys: string[]
  hit_canonical_keys: string[]
}

export interface SimilarIssuesResponse {
  query_source: { mode: 'issue_key' | 'draft'; issue_key: string | null; kind: string }
  query: string
  scope: RankedSearchResponse['scope']
  include_comments: boolean
  mode_requested?: string
  mode_effective?: string
  degradation?: RankedSearchDegradation
  coverage?: {
    status: 'complete' | 'degraded' | 'inconclusive' | string
    complete: boolean
    inconclusive: boolean
    probes_requested: number
    probes_completed: number
    probes_failed: number
    partial?: boolean
    candidate_keys_seen: number
  }
  diagnostics?: Record<string, unknown> & {
    similarity_duplicate_conflicts?: SimilarDuplicateConflict[]
  }
  generations?: Record<string, unknown>
  limit: number
  total: number
  /** Candidates keep the full ranked-search explanation objects. */
  candidates: SimilarIssueExplanation[]
  duplicate_expansions: SimilarDuplicateExpansion[]
}

// ---------------------------------------------------------------------------
// Ranked issue search — GET /tracker/issues/search (M1.4a contract)

export interface RankedSearchLaneContribution {
  lane: string
  rank: number
  raw_score: number
}

/** The §10.4 comment-lane winner: the single comment that carried the issue
 * into the results, plus the rest of the matching set behind it. */
export interface RankedWinningComment {
  comment_id: number
  important: boolean
  retained_hits: number
  additional_comment_ids: number[]
  total_matching_comments: number
}

export interface RankedSearchExplanation {
  issue: TrackerIssue | null
  rank_score: number
  contributing_lanes: RankedSearchLaneContribution[]
  matched_fields: string[]
  snippets: Record<string, string>
  winning_comment: RankedWinningComment | null
  exact_boosts: string[]
  neighborhood: Array<{ from_key: string; to_key: string; kind: string }>
  duplicate_chain: Array<{ canonical_key: string; canonical_title: string | null; resolved: boolean }>
}

/** Similarity adds the probe-level audit without changing ranked search. */
export interface SimilarIssueExplanation extends RankedSearchExplanation {
  /** Present on the repaired service; optional for older API responses. */
  probe_contributions?: SimilarProbeContribution[]
}

export interface RankedSearchLaneAvailability {
  available: boolean
  reason?: string
}

export interface RankedSearchDegradation {
  requested_mode: string
  effective_mode: string
  reasons: string[]
  lanes: Record<string, RankedSearchLaneAvailability>
}

export interface RankedSearchResponse {
  query: string
  scope: {
    project_ids: string[]
    all_projects: boolean
    subtree_roots: string[]
    subtree_closure_size: number
  }
  mode_requested: string
  mode_effective: string
  degradation: RankedSearchDegradation
  generations: Partial<Record<'schema_version' | 'document_schema_version' | 'content_clock' | 'active_vector_generation' | 'rebuilt_at', number | string | null>>
  diagnostics: {
    lane_elapsed_ms: Record<string, number>
    total_elapsed_ms: number
  }
  total: number
  limit: number
  offset: number
  results: RankedSearchExplanation[]
}

/** Filters for the dashboard's ranked search. Structured filters mirror
 * TrackerIssueFilters so an operator's active list filters survive the switch
 * into search; multi-value families go over the wire as repeated params. */
export interface RankedSearchFilters {
  projectId?: string
  q: string
  kind?: string
  status?: string[]
  severity?: string[]
  component?: string
  assignee?: string
  reporter?: string
  label?: string[]
  withoutLabel?: string[]
  unlabeled?: boolean
  openOnly?: boolean
  limit?: number
  offset?: number
}

export interface TrackerLabelFacets {
  project_id: string
  labels: TrackerLabelFacet[]
  unlabeled: number
  unlabeled_open: number
}

export type TrackerOptionField =
  | 'label'
  | 'component'
  | 'assignee'
  | 'reporter'
  | 'collaborator'
  | 'branch'
  | 'worktree'
  | 'pull_request'

export interface TrackerFieldOption {
  value: string
  total: number
  open: number
}

export interface TrackerFieldOptions {
  project_id: string
  field: TrackerOptionField
  query: string
  matching_total: number
  options: TrackerFieldOption[]
}

/** A child row in the map projection: the issue plus its server-computed
 * classification. `blocked_by` lists the nonterminal blockers benching it;
 * `frontier` is the canonical takeable rule (nonterminal, unassigned,
 * unblocked) — computed once, server-side, never re-derived per widget. */
export interface TrackerMapChild extends TrackerIssue {
  blocked_by: string[]
  frontier: boolean
}

/** An external row in the map projection: an issue that is neither the map
 * nor a child but is named by one of the projection's links, so every
 * returned link has both endpoints on screen. `blocking` lists the member
 * children this issue actually benches (its nonterminal `blocks` edges to
 * them) — a non-empty list is what makes it an external blocker; anything
 * else is context (a relates/duplicates/caused-by neighbour, or a blocker
 * that has already landed). */
export interface TrackerMapExternal extends TrackerIssue {
  blocking: string[]
}

export interface TrackerMapProjection {
  map: TrackerIssue
  children: TrackerMapChild[]
  /** Child keys on the frontier, oldest first. */
  frontier: string[]
  links: TrackerLink[]
  /** Every link endpoint that is neither the map nor a child — blockers and
   * context alike, so no returned link points at an invisible issue. */
  external: TrackerMapExternal[]
  progress: {
    total: number
    open: number
    terminal: number
    resolved: number
    claimed: number
    frontier: number
  }
}

export interface TrackerGraphNode extends TrackerIssue {
  depth: number
  parent_keys: string[]
  child_count: number
}

export interface TrackerGraphProjection {
  root: TrackerIssue
  nodes: TrackerGraphNode[]
  external: TrackerIssue[]
  links: TrackerLink[]
  bounds: {
    max_depth: number
    max_nodes: number
    truncated: boolean
    reasons: string[]
    live_children_beyond_bound?: string[]
  }
  stats: {
    nodes: number
    descendants: number
    external: number
    links: number
    depth: number
  }
}

export interface TrackerClaimResult extends TrackerIssue {
  claimed: boolean
  already_claimed: boolean
}

export interface TrackerUnclaimResult extends TrackerIssue {
  unclaimed: boolean
  was_claimed: boolean
}

export interface TrackerStats {
  project_id: string | null
  total: number
  open: number
  by_status: Record<string, number>
  by_severity: Record<string, number>
  by_component: Record<string, number>
}

function trackerQuery(filters?: TrackerIssueFilters): string {
  if (!filters) return ''
  const parts: string[] = []
  if (filters.projectId) parts.push(`project_id=${encodeURIComponent(filters.projectId)}`)
  // Repeated params, not a comma list: the server reads them as an OR.
  for (const s of filters.status ?? []) parts.push(`status=${encodeURIComponent(s)}`)
  for (const s of filters.severity ?? []) parts.push(`severity=${encodeURIComponent(s)}`)
  if (filters.kind) parts.push(`kind=${encodeURIComponent(filters.kind)}`)
  if (filters.component) parts.push(`component=${encodeURIComponent(filters.component)}`)
  if (filters.assignee) parts.push(`assignee=${encodeURIComponent(filters.assignee)}`)
  if (filters.reporter) parts.push(`reporter=${encodeURIComponent(filters.reporter)}`)
  for (const label of filters.label ?? []) parts.push(`label=${encodeURIComponent(label)}`)
  for (const label of filters.withoutLabel ?? []) parts.push(`without_label=${encodeURIComponent(label)}`)
  if (filters.unlabeled) parts.push('unlabeled=true')
  if (filters.q) parts.push(`q=${encodeURIComponent(filters.q)}`)
  if (filters.openOnly) parts.push('open_only=true')
  if (filters.limit) parts.push(`limit=${filters.limit}`)
  if (filters.offset) parts.push(`offset=${filters.offset}`)
  if (filters.order) parts.push(`order=${encodeURIComponent(filters.order)}`)
  return parts.length ? `?${parts.join('&')}` : ''
}

function linkClockQuery(clocks?: {
  expected_from_updated_at?: string
  expected_to_updated_at?: string
}): string {
  if (!clocks) return ''
  const parts: string[] = []
  if (clocks.expected_from_updated_at) {
    parts.push(`expected_from_updated_at=${encodeURIComponent(clocks.expected_from_updated_at)}`)
  }
  if (clocks.expected_to_updated_at) {
    parts.push(`expected_to_updated_at=${encodeURIComponent(clocks.expected_to_updated_at)}`)
  }
  return parts.length ? `?${parts.join('&')}` : ''
}

/** Query string for GET /tracker/issues/search. Same repeated-param
 * conventions as trackerQuery: multi-value families repeat, scope is exactly
 * one project here (the dashboard is per-project), and `q` is mandatory — the
 * route refuses an empty normalized query, so callers only send non-empty
 * text (the empty-query surface stays on the issue list). */
function rankedSearchQuery(filters: RankedSearchFilters): string {
  const parts: string[] = [`q=${encodeURIComponent(filters.q)}`]
  if (filters.projectId) parts.push(`project_id=${encodeURIComponent(filters.projectId)}`)
  if (filters.kind && filters.kind !== 'all') parts.push(`kind=${encodeURIComponent(filters.kind)}`)
  for (const s of filters.status ?? []) parts.push(`status=${encodeURIComponent(s)}`)
  for (const s of filters.severity ?? []) parts.push(`severity=${encodeURIComponent(s)}`)
  if (filters.component) parts.push(`component=${encodeURIComponent(filters.component)}`)
  if (filters.assignee) parts.push(`assignee=${encodeURIComponent(filters.assignee)}`)
  if (filters.reporter) parts.push(`reporter=${encodeURIComponent(filters.reporter)}`)
  for (const label of filters.label ?? []) parts.push(`label=${encodeURIComponent(label)}`)
  for (const label of filters.withoutLabel ?? []) parts.push(`without_label=${encodeURIComponent(label)}`)
  if (filters.unlabeled) parts.push('unlabeled=true')
  if (filters.openOnly) parts.push('open_only=true')
  if (filters.limit) parts.push(`limit=${filters.limit}`)
  if (filters.offset) parts.push(`offset=${filters.offset}`)
  return `?${parts.join('&')}`
}

export const api = {
  // Agent Profiles & Providers
  listProfiles: () => fetchJSON<AgentProfileInfo[]>('/agents/profiles'),
  listProviders: () => fetchJSON<ProviderInfo[]>('/agents/providers'),

  // Settings
  getAgentDirs: () => fetchJSON<AgentDirsSettings>('/settings/agent-dirs'),
  setAgentDirs: (data: { agent_dirs?: Record<string, string>; extra_dirs?: string[]; disabled_dirs?: string[] }) =>
    fetchJSON<AgentDirsSettings>('/settings/agent-dirs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  // Sessions
  listSessions: () => fetchJSON<Session[]>('/sessions'),
  getSession: (name: string) => fetchJSON<SessionDetail>(`/sessions/${name}`),
  createSession: (provider: string, agentProfile: string, sessionName?: string, workingDirectory?: string) =>
    fetchJSON<Terminal>(`/sessions?provider=${encodeURIComponent(provider)}&agent_profile=${encodeURIComponent(agentProfile)}${sessionName ? `&session_name=${encodeURIComponent(sessionName)}` : ''}${workingDirectory ? `&working_directory=${encodeURIComponent(workingDirectory)}` : ''}`, { method: 'POST', timeoutMs: 90000 }),
  deleteSession: (name: string) => fetchJSON<{ success: boolean; deleted: string[]; errors: any[] }>(`/sessions/${name}`, { method: 'DELETE' }),

  // Session lifecycle — what a session has DECLARED it is doing, which is a
  // different dimension from a terminal's observed `lifecycle_state`. The
  // value sets are disjoint (`working|pausing|paused|complete|stopped` here
  // versus `live|superseded|dead|unknown-liveness` there) and they must never
  // be rendered as one field.
  //
  // Never 404s: an undeclared session reads as `working`, because every
  // session that predates the table would otherwise read as unknown and the
  // fire marshal's suppression has to fail toward watching.
  getSessionLifecycle: (name: string) =>
    fetchJSON<SessionLifecycle>(`/sessions/${encodeURIComponent(name)}/lifecycle`),
  // Only what the dashboard actually calls. The write wrappers were added
  // ahead of the controls that would use them, which is the same shape as
  // the defect this branch exists to fix — a tested surface nothing invokes.
  // They come back with the buttons.
  getStopImpact: (name: string) =>
    fetchJSON<StopImpact>(`/sessions/${encodeURIComponent(name)}/stop-impact`),

  // Fleet cohort operations. Six write methods, not two with a mode argument:
  // the separation that keeps force out of reach of a caller who meant safe
  // has to survive the client, or the client becomes the way around it.
  //
  // `operationId` is caller-supplied so a retried request is the *same*
  // operation and adopts the durable winner instead of starting a second one.
  listCohortOperations: (name: string) =>
    fetchJSON<{ operations: CohortOperation[]; count: number }>(
      `/sessions/${encodeURIComponent(name)}/cohort-operations`),
  getCohortOperation: (operationId: string) =>
    fetchJSON<CohortOperation & { provenance: CohortProvenance }>(
      `/cohort-operations/${encodeURIComponent(operationId)}`),

  cohortPauseForce: (name: string, operationId: string, initiatedBy: string, reason?: string) =>
    fetchJSON<CohortOperation>(`/sessions/${encodeURIComponent(name)}/cohort/pause/force`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeoutMs: 120000,
      body: JSON.stringify({
        operation_id: operationId, initiated_by: initiatedBy, reason: reason ?? null,
        acknowledged_interrupt: true,
      }),
    }),
  cohortStopSafe: (
    name: string, operationId: string, initiatedBy: string,
    drainReceiptDigest: string, reason?: string,
  ) =>
    fetchJSON<CohortOperation>(`/sessions/${encodeURIComponent(name)}/cohort/stop/safe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeoutMs: 120000,
      body: JSON.stringify({
        operation_id: operationId, initiated_by: initiatedBy, reason: reason ?? null,
        drain_receipt_digest: drainReceiptDigest, acknowledged_one_way: true,
      }),
    }),
  cohortStopForce: (name: string, operationId: string, initiatedBy: string, reason?: string) =>
    fetchJSON<CohortOperation>(`/sessions/${encodeURIComponent(name)}/cohort/stop/force`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeoutMs: 120000,
      body: JSON.stringify({
        operation_id: operationId, initiated_by: initiatedBy, reason: reason ?? null,
        acknowledged_one_way: true, acknowledged_force: true,
      }),
    }),
  // Restores the panes and stops. Sends zero input — no keystroke, no
  // supervisor bump — so an operator can look before anything moves.
  cohortResumePaused: (name: string, operationId: string, initiatedBy: string, reason?: string) =>
    fetchJSON<CohortOperation>(`/sessions/${encodeURIComponent(name)}/cohort/resume/paused`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeoutMs: 180000,
      body: JSON.stringify({
        operation_id: operationId, initiated_by: initiatedBy, reason: reason ?? null,
      }),
    }),
  // Restores and wakes the supervisor exactly once, after every member
  // outcome is durable.
  cohortResumeStart: (name: string, operationId: string, initiatedBy: string, reason?: string) =>
    fetchJSON<CohortOperation>(`/sessions/${encodeURIComponent(name)}/cohort/resume/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeoutMs: 180000,
      body: JSON.stringify({
        operation_id: operationId, initiated_by: initiatedBy, reason: reason ?? null,
      }),
    }),
  // Continues an EXISTING Resume out of `reconciliation-required`. The
  // operation id names what to finish rather than minting a new one, which is
  // what keeps this from being a second Resume: no new boundary, no second
  // barrier release, no member re-restored that already has a decided outcome.
  cohortResumeRetry: (name: string, operationId: string, initiatedBy: string, reason?: string) =>
    fetchJSON<CohortOperation>(`/sessions/${encodeURIComponent(name)}/cohort/resume/retry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeoutMs: 180000,
      body: JSON.stringify({
        operation_id: operationId, initiated_by: initiatedBy, reason: reason ?? null,
      }),
    }),

  // Safe drain and its receipt (M3-D). `runSafeDrain` types into every
  // non-idle worker's pane, which is why it is the only M3-D write here that
  // is not a read: everything else in this block observes.
  //
  // `drainId` is caller-supplied for the same reason `operationId` is — a
  // retried request is the *same* drain and adopts it rather than steering
  // the whole fleet a second time.
  runSafeDrain: (
    name: string, drainId: string, intent: DrainIntent, initiatedBy: string, retry = false,
  ) =>
    fetchJSON<SessionDrain>(`/sessions/${encodeURIComponent(name)}/drain/safe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeoutMs: 180000,
      body: JSON.stringify({
        drain_id: drainId, intent, initiated_by: initiatedBy, retry,
      }),
    }),
  getDrain: (drainId: string) =>
    fetchJSON<SessionDrain & { provenance: DrainProvenance }>(
      `/drains/${encodeURIComponent(drainId)}`),
  listDrains: (name: string) =>
    fetchJSON<{ drains: SessionDrain[]; count: number }>(
      `/sessions/${encodeURIComponent(name)}/drains`),

  // A safe Pause names the *drain*, never a digest. The receipt and the
  // per-member classification are then read from one durable row: a client
  // that carried a digest forward while editing the member list would spend a
  // real receipt on a claim it does not describe.
  cohortPauseSafeFromDrain: (
    name: string, operationId: string, drainId: string, initiatedBy: string, reason?: string,
  ) =>
    fetchJSON<CohortOperation>(
      `/sessions/${encodeURIComponent(name)}/cohort/pause/safe-drained`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        timeoutMs: 120000,
        body: JSON.stringify({
          operation_id: operationId, drain_id: drainId,
          initiated_by: initiatedBy, reason: reason ?? null,
        }),
      }),

  // Task occurrences. Read-only here: opening, finalizing, and extending a
  // round belong to the components that own the work, not to a dashboard.
  listTaskOccurrences: (name: string) =>
    fetchJSON<{ occurrences: TaskOccurrence[]; count: number }>(
      `/sessions/${encodeURIComponent(name)}/task-occurrences`),
  getTaskOccurrence: (id: string) =>
    fetchJSON<TaskOccurrence>(`/task-occurrences/${encodeURIComponent(id)}`),
  getAgentOccurrenceHistory: (name: string, agentId: string) =>
    fetchJSON<{ open: TaskOccurrence | null; finalized: TaskOccurrence[] }>(
      `/sessions/${encodeURIComponent(name)}/agents/${encodeURIComponent(agentId)}` +
      '/task-occurrences'),

  // What each resumed supervisor was actually told. The dashboard may only say
  // "the supervisor was told" where a delivered wake says so.
  listReconciliationWakes: (name: string) =>
    fetchJSON<{ wakes: ReconciliationWake[]; count: number }>(
      `/sessions/${encodeURIComponent(name)}/reconciliation-wakes`),

  // Conductor annotations (§9.5). The route takes no parameters — that is the
  // security property, not an omission — and never errors: an absent or
  // unreadable conductor state root answers `coverage: "unavailable"` with an
  // empty list, which renders exactly as the dashboard did before it existed.
  getAnnotations: () => fetchJSON<AnnotationsResponse>('/annotations'),

  // Communications catalog (design §7). The task-scoped list is keyset-paged;
  // pass `next_cursor` back verbatim and render the server's order as
  // returned — the total order is `recorded_at DESC, communication_id ASC`,
  // and a client-side sort on `recorded_at` alone is not that order. Detail
  // responses are `Cache-Control: no-store`; never cache bodies.
  listCommunications: (taskOccurrenceId: string, cursor?: string | null) =>
    fetchJSON<CommunicationsListResponse>(
      `/communications?task_occurrence_id=${encodeURIComponent(taskOccurrenceId)}` +
        (cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''),
    ),
  getCommunication: (id: string) =>
    fetchJSON<CommunicationDetailResponse>(`/communications/${encodeURIComponent(id)}`),
  getCommunicationAttachment: (id: string) =>
    fetchJSON<AttachmentDetailResponse>(`/communications/attachments/${encodeURIComponent(id)}`),

  // Terminals
  getTerminalStatus: (id: string) =>
    fetchJSON<Terminal>(`/terminals/${id}`).then(t => t.status),
  getTerminalOutput: (id: string, mode: 'full' | 'last' = 'full') =>
    fetchJSON<{ output: string; mode: string }>(`/terminals/${id}/output?mode=${mode}`),
  sendInput: (id: string, message: string) =>
    fetchJSON<{ success: boolean }>(`/terminals/${id}/input?message=${encodeURIComponent(message)}`, { method: 'POST' }),
  getManagedControl: (id: string) =>
    fetchJSON<{ managed: boolean; generation?: string; provider?: string; execution_mode?: string; vintage?: string }>(`/terminals/${id}/managed-control`),
  getControlInputCapabilities: () =>
    fetchJSON<ControlInputCapabilities>('/control-input/capabilities'),
  getControlIdentity: (id: string) =>
    fetchJSON<Record<string, unknown>>(`/terminals/${id}/control-identity`),
  sendControlInput: (
    id: string,
    body: {
      control_id: string
      text?: string
      enter?: boolean
      // v3 structured sequences: the request carries events OR the v1/v2
      // fields, never both.
      events?: WireEvent[]
      // v4 declaration carrier. "command" (§4.1): sent ONLY by the registry
      // Compact built-in send (and supervisor/provider command controls),
      // and only after the server advertises the command_controls block.
      // "interactive" (§6.7, r15): sent ONLY by the armed manual streaming
      // capture, and only after the per-terminal, build-exact
      // interactive_streaming block advertises support. Macros, favorites,
      // the prose composer, operator messages, and inbox/automation never
      // set either.
      payload_class?: 'command' | 'interactive'
      expected_identity: Record<string, unknown>
    }
  ) =>
    fetchJSON<Record<string, unknown>>(`/terminals/${id}/control-input`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      timeoutMs: 15000,
    }),
  queryControlInput: (controlId: string) =>
    fetchJSON<Record<string, unknown>>(`/control-input/${encodeURIComponent(controlId)}`),

  // Lane C: operator messages + image attachments (§8.3/§8.4). Typed
  // outcomes travel as 200 like control-input; a 404 on either route is the
  // old-server signal and resolves to `unsupported` — never a resend.
  submitOperatorMessage: (
    id: string,
    body: {
      operation_id: string
      text: string
      attachments: string[]
      token_map: Record<string, string>
      expected_identity: Record<string, unknown>
    }
  ) =>
    fetchJSON<Record<string, unknown>>(`/terminals/${id}/operator-message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      // The server's write deadline is 20s; the client must outlast it.
      timeoutMs: 25000,
    }),
  reconcileOperatorMessage: (operationId: string) =>
    fetchJSON<Record<string, unknown>>(`/operator-message/${encodeURIComponent(operationId)}`),
  uploadAttachment: (id: string, file: File) => {
    const form = new FormData()
    form.append('file', file, file.name)
    // No Content-Type header: the browser sets the multipart boundary.
    return fetchJSON<{ attachment: ImageAttachmentRecord }>(
      `/terminals/${id}/attachments`,
      { method: 'POST', body: form, timeoutMs: 15000 },
    )
  },
  listAttachments: (id: string) =>
    fetchJSON<{ attachments: ImageAttachmentRecord[] }>(`/terminals/${id}/attachments`),
  deleteAttachment: (id: string, attachmentId: string) =>
    fetchJSON<{ deleted: boolean; attachment: ImageAttachmentRecord }>(
      `/terminals/${id}/attachments/${encodeURIComponent(attachmentId)}`,
      { method: 'DELETE' },
    ),

  // Operator macro library (§5.4). Sending a macro is NOT a store
  // operation: the client takes the resolved events and sends an ordinary
  // v3 control-input request (D2) — these routes only manage the library.
  listMacros: (filters?: { provider?: string; profile?: string }) => {
    const params = [
      filters?.provider ? `provider=${encodeURIComponent(filters.provider)}` : '',
      filters?.profile ? `profile=${encodeURIComponent(filters.profile)}` : '',
    ].filter(Boolean).join('&')
    return fetchJSON<MacroListResponse>(`/macros${params ? `?${params}` : ''}`)
  },
  createMacro: (body: MacroWriteBody) =>
    fetchJSON<MacroRecord>('/macros', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  updateMacro: (macroId: string, body: MacroWriteBody) =>
    fetchJSON<MacroRecord>(`/macros/${encodeURIComponent(macroId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  deleteMacro: (macroId: string) =>
    fetchJSON<{ deleted: string }>(`/macros/${encodeURIComponent(macroId)}`, { method: 'DELETE' }),
  duplicateMacro: (macroId: string, name?: string) =>
    fetchJSON<MacroRecord>(`/macros/${encodeURIComponent(macroId)}/duplicate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(name ? { name } : {}),
    }),
  parseMacroNotation: (notation: string) =>
    fetchJSON<MacroNotationParseResult>('/macros/parse-notation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ notation }),
    }),
  beginManagedOperation: (
    id: string,
    body: {
      action: string
      operation_id: string
      generation?: string
      message?: string
      config_id?: string
      value?: string
      instruction?: string
    }
  ) =>
    fetchJSON<{ success: boolean; receipt: Record<string, unknown> }>(
      `/terminals/${id}/managed-operations`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        timeoutMs: 50000,
      }
    ),
  queryManagedOperation: (id: string, operationId: string, generation?: string) =>
    fetchJSON<{ receipt: Record<string, unknown> }>(
      `/terminals/${id}/managed-operations/${encodeURIComponent(operationId)}${generation ? `?generation=${encodeURIComponent(generation)}` : ''}`,
      { timeoutMs: 35000 }
    ),
  exitTerminal: (id: string) =>
    fetchJSON<{ success: boolean }>(`/terminals/${id}/exit`, { method: 'POST' }),
  deleteTerminal: (id: string) => fetchJSON<{ success: boolean }>(`/terminals/${id}`, { method: 'DELETE' }),
  getWorkingDirectory: (id: string) =>
    fetchJSON<{ working_directory: string | null }>(`/terminals/${id}/working-directory`),
  addTerminalToSession: (sessionName: string, provider: string, agentProfile: string, workingDirectory?: string) =>
    fetchJSON<Terminal>(`/sessions/${sessionName}/terminals?provider=${encodeURIComponent(provider)}&agent_profile=${encodeURIComponent(agentProfile)}${workingDirectory ? `&working_directory=${encodeURIComponent(workingDirectory)}` : ''}`, { method: 'POST', timeoutMs: 90000 }),

  // Inbox
  getInboxMessages: (terminalId: string, limit?: number, status?: string) =>
    fetchJSON<InboxMessage[]>(`/terminals/${terminalId}/inbox/messages?limit=${limit || 50}${status ? `&status=${status}` : ''}`),
  sendInboxMessage: (receiverId: string, senderId: string, message: string) =>
    fetchJSON<{ success: boolean }>(`/terminals/${receiverId}/inbox/messages?sender_id=${senderId}&message=${encodeURIComponent(message)}`, { method: 'POST' }),

  // Flows
  listFlows: () => fetchJSON<Flow[]>('/flows'),
  createFlow: (data: { name: string; schedule: string; agent_profile: string; provider?: string; prompt_template: string }) =>
    fetchJSON<Flow>('/flows', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      timeoutMs: 30000,
    }),
  deleteFlow: (name: string) => fetchJSON<{ success: boolean }>(`/flows/${name}`, { method: 'DELETE' }),
  enableFlow: (name: string) => fetchJSON<{ success: boolean }>(`/flows/${name}/enable`, { method: 'POST' }),
  disableFlow: (name: string) => fetchJSON<{ success: boolean }>(`/flows/${name}/disable`, { method: 'POST' }),
  runFlow: (name: string) => fetchJSON<{ executed: boolean }>(`/flows/${name}/run`, { method: 'POST', timeoutMs: 90000 }),

  // Memory
  getMemoryStatus: () => fetchJSON<MemoryStatus>('/settings/memory'),
  listMemories: (filters?: { scope?: string; type?: string; scopeId?: string; limit?: number }) => {
    const params = [
      filters?.scope ? `scope=${encodeURIComponent(filters.scope)}` : '',
      filters?.type ? `type=${encodeURIComponent(filters.type)}` : '',
      filters?.scopeId ? `scope_id=${encodeURIComponent(filters.scopeId)}` : '',
      filters?.limit ? `limit=${filters.limit}` : '',
    ].filter(Boolean).join('&')
    return fetchJSON<MemorySummary[]>(`/memory${params ? `?${params}` : ''}`)
  },
  getMemory: (key: string, scope?: string, scopeId?: string) => {
    const params = [
      scope ? `scope=${encodeURIComponent(scope)}` : '',
      scopeId ? `scope_id=${encodeURIComponent(scopeId)}` : '',
    ].filter(Boolean).join('&')
    return fetchJSON<MemoryDetail>(`/memory/${encodeURIComponent(key)}${params ? `?${params}` : ''}`)
  },
  deleteMemory: (key: string, scope: string, scopeId?: string) =>
    fetchJSON<{ success: boolean }>(`/memory/${encodeURIComponent(key)}?scope=${encodeURIComponent(scope)}${scopeId ? `&scope_id=${encodeURIComponent(scopeId)}` : ''}`, { method: 'DELETE' }),
  clearMemories: (scope: string, scopeId?: string) =>
    fetchJSON<{ success: boolean; deleted_count: number }>(`/memory?scope=${encodeURIComponent(scope)}${scopeId ? `&scope_id=${encodeURIComponent(scopeId)}` : ''}`, { method: 'DELETE' }),

  // Graph (Issue #348). The projection runs wiki_lint (ripgrep detectors)
  // server-side, so both routes get a wide timeout — a populated scope can take
  // ~30s typical, up to ~148s under load. Errors surface as ApiError (status +
  // server detail) for the caller.
  getGraph: (provider = 'memory', scope?: string, scopeId?: string) => {
    const params = [
      scope ? `scope=${encodeURIComponent(scope)}` : '',
      scopeId ? `scope_id=${encodeURIComponent(scopeId)}` : '',
    ].filter(Boolean).join('&')
    return fetchJSON<GraphView>(
      `/graph/${encodeURIComponent(provider)}${params ? `?${params}` : ''}`,
      { timeoutMs: 120000 },
    )
  },
  exportGraph: (provider = 'memory', body: GraphExportBody, scope?: string, scopeId?: string) => {
    const params = [
      scope ? `scope=${encodeURIComponent(scope)}` : '',
      scopeId ? `scope_id=${encodeURIComponent(scopeId)}` : '',
    ].filter(Boolean).join('&')
    return fetchJSON<GraphExportResult>(
      `/graph/${encodeURIComponent(provider)}/export${params ? `?${params}` : ''}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ options: {}, ...body }),
        timeoutMs: 60000,
      },
    )
  },

  // Issue tracker
  getTrackerVocabulary: () => fetchJSON<TrackerVocabulary>('/tracker/vocabulary'),

  listTrackerProjects: (includeArchived = false) =>
    fetchJSON<TrackerProject[]>(`/tracker/projects${includeArchived ? '?include_archived=true' : ''}`),
  getTrackerProject: (id: string) =>
    fetchJSON<TrackerProject>(`/tracker/projects/${encodeURIComponent(id)}`),
  getTrackerProjectHome: (id: string) =>
    fetchJSON<TrackerProjectHome>(`/tracker/projects/${encodeURIComponent(id)}/dashboard`),
  getTrackerProjectSessions: (id: string) =>
    fetchJSON<TrackerProjectSessions>(`/tracker/projects/${encodeURIComponent(id)}/sessions`),
  getTrackerProjectSession: (id: string, sessionName: string) =>
    fetchJSON<{ project_id: string; session: TrackerProjectSessionDetail }>(
      `/tracker/projects/${encodeURIComponent(id)}/sessions/${encodeURIComponent(sessionName)}`,
    ),
  getTrackerProjectTerminalLog: (
    id: string,
    sessionName: string,
    terminalId: string,
    mode: 'last' | 'full' = 'last',
  ) => fetchJSON<{ output: string; mode: string; truncated: boolean; source: string }>(
    `/tracker/projects/${encodeURIComponent(id)}/sessions/${encodeURIComponent(sessionName)}` +
    `/terminals/${encodeURIComponent(terminalId)}/log?mode=${mode}`,
  ),
  createTrackerProject: (body: {
    name: string
    id?: string
    description?: string
    issue_prefix?: string
    scopes?: Array<{ kind: string; value: string }>
  }) =>
    fetchJSON<TrackerProject>('/tracker/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  updateTrackerProject: (
    id: string,
    body: { name?: string; description?: string; status?: string; issue_prefix?: string },
  ) =>
    fetchJSON<TrackerProject>(`/tracker/projects/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  deleteTrackerProject: (id: string, force = false) =>
    fetchJSON<{ id: string; deleted: boolean; issues_deleted: number }>(
      `/tracker/projects/${encodeURIComponent(id)}${force ? '?force=true' : ''}`,
      { method: 'DELETE' },
    ),
  addTrackerScope: (projectId: string, body: { kind: string; value: string }) =>
    fetchJSON<TrackerScope & { created: boolean }>(
      `/tracker/projects/${encodeURIComponent(projectId)}/scopes`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    ),
  removeTrackerScope: (projectId: string, scopeId: number) =>
    fetchJSON<{ id: number; deleted: boolean }>(
      `/tracker/projects/${encodeURIComponent(projectId)}/scopes/${scopeId}`,
      { method: 'DELETE' },
    ),

  listTrackerIssues: (filters?: TrackerIssueFilters) =>
    fetchJSON<TrackerIssuePage>(`/tracker/issues${trackerQuery(filters)}`),
  searchTrackerIssues: (filters: RankedSearchFilters, signal?: AbortSignal) =>
    fetchJSON<RankedSearchResponse>(`/tracker/issues/search${rankedSearchQuery(filters)}`, { signal }),
  // Advisory pre-filing probe (M2.5): the signal lets the form cancel a
  // superseded draft query. A failure here must never gate filing.
  similarTrackerIssues: (body: SimilarIssuesRequest, signal?: AbortSignal) =>
    fetchJSON<SimilarIssuesResponse>('/tracker/issues/similar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    }),
  getTrackerIssue: (key: string, signal?: AbortSignal) =>
    fetchJSON<TrackerIssue>(`/tracker/issues/${encodeURIComponent(key)}`, { signal }),
  createTrackerIssue: (body: Record<string, unknown>) =>
    fetchJSON<TrackerIssue>('/tracker/issues', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ origin: 'dashboard', ...body }),
    }),
  updateTrackerIssue: (key: string, body: Record<string, unknown>) =>
    fetchJSON<TrackerIssue>(`/tracker/issues/${encodeURIComponent(key)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  deleteTrackerIssue: (key: string) =>
    fetchJSON<{ key: string; deleted: boolean }>(`/tracker/issues/${encodeURIComponent(key)}`, {
      method: 'DELETE',
    }),
  addTrackerComment: (key: string, body: {
    body: string
    author?: string
    important?: boolean
    expected_updated_at?: string
  }) =>
    fetchJSON<TrackerComment>(`/tracker/issues/${encodeURIComponent(key)}/comments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  setTrackerCommentImportance: (key: string, commentId: number, important: boolean) =>
    fetchJSON<TrackerCommentImportanceResult>(
      `/tracker/issues/${encodeURIComponent(key)}/comments/${commentId}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ important, actor: 'dashboard' }),
      },
    ),
  addTrackerLink: (key: string, body: {
    to_key: string
    kind: string
    actor?: string
    expected_from_updated_at?: string
    expected_to_updated_at?: string
    action_key?: string
  }) =>
    fetchJSON<TrackerLink & {
      created: boolean
      replayed?: boolean
      action_key?: string
      from_updated_at?: string | null
      to_updated_at?: string | null
      effect_ids?: number[]
    }>(
      `/tracker/issues/${encodeURIComponent(key)}/links`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    ),
  removeTrackerLink: (key: string, linkId: number, clocks?: {
    expected_from_updated_at?: string
    expected_to_updated_at?: string
  }) =>
    fetchJSON<{ id: number; deleted: boolean; from_updated_at?: string | null; to_updated_at?: string | null; effect_ids?: number[] }>(
      `/tracker/issues/${encodeURIComponent(key)}/links/${linkId}${linkClockQuery(clocks)}`,
      { method: 'DELETE' },
    ),
  getTrackerStats: (projectId?: string) =>
    fetchJSON<TrackerStats>(
      `/tracker/issues/stats${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ''}`,
    ),
  listTrackerFeatures: (filters?: TrackerIssueFilters) =>
    fetchJSON<TrackerIssuePage>(`/tracker/features${trackerQuery({ ...filters, kind: 'feature' })}`),
  getTrackerFeature: (key: string) =>
    fetchJSON<TrackerIssue>(`/tracker/features/${encodeURIComponent(key)}`),
  createTrackerFeature: (body: Record<string, unknown>) =>
    fetchJSON<TrackerIssue>('/tracker/features', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ origin: 'dashboard', ...body }),
    }),
  updateTrackerFeature: (key: string, body: Record<string, unknown>) =>
    fetchJSON<TrackerIssue>(`/tracker/features/${encodeURIComponent(key)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  deleteTrackerFeature: (key: string) =>
    fetchJSON<{ key: string; deleted: boolean }>(`/tracker/features/${encodeURIComponent(key)}`, {
      method: 'DELETE',
    }),
  addTrackerFeatureComment: (key: string, body: {
    body: string
    author?: string
    important?: boolean
    expected_updated_at?: string
  }) =>
    fetchJSON<TrackerComment>(`/tracker/features/${encodeURIComponent(key)}/comments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  setTrackerFeatureCommentImportance: (key: string, commentId: number, important: boolean) =>
    fetchJSON<TrackerCommentImportanceResult>(
      `/tracker/features/${encodeURIComponent(key)}/comments/${commentId}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ important, actor: 'dashboard' }),
      },
    ),
  addTrackerFeatureLink: (key: string, body: {
    to_key: string
    kind: string
    actor?: string
    expected_from_updated_at?: string
    expected_to_updated_at?: string
    action_key?: string
  }) =>
    fetchJSON<TrackerLink & {
      created: boolean
      replayed?: boolean
      action_key?: string
      from_updated_at?: string | null
      to_updated_at?: string | null
      effect_ids?: number[]
    }>(
      `/tracker/features/${encodeURIComponent(key)}/links`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    ),
  removeTrackerFeatureLink: (key: string, linkId: number, clocks?: {
    expected_from_updated_at?: string
    expected_to_updated_at?: string
  }) =>
    fetchJSON<{ id: number; deleted: boolean; from_updated_at?: string | null; to_updated_at?: string | null; effect_ids?: number[] }>(
      `/tracker/features/${encodeURIComponent(key)}/links/${linkId}${linkClockQuery(clocks)}`,
      { method: 'DELETE' },
    ),
  getTrackerFeatureStats: (projectId?: string) =>
    fetchJSON<TrackerStats>(`/tracker/features/stats${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ''}`),

  // cond-0394: map membership, frontier, claim lifecycle, discovery
  getTrackerLabels: (projectId: string) =>
    fetchJSON<TrackerLabelFacets>(`/tracker/projects/${encodeURIComponent(projectId)}/labels`),
  getTrackerFieldOptions: (
    projectId: string,
    field: TrackerOptionField,
    query = '',
    limit = 20,
  ) =>
    fetchJSON<TrackerFieldOptions>(
      `/tracker/projects/${encodeURIComponent(projectId)}/options` +
      `?field=${encodeURIComponent(field)}&q=${encodeURIComponent(query)}&limit=${limit}`,
    ),
  listTrackerChildren: (key: string) =>
    fetchJSON<{ parent: string; children: TrackerIssue[] }>(
      `/tracker/issues/${encodeURIComponent(key)}/children`,
    ),
  getTrackerFrontier: (key: string) =>
    fetchJSON<{ parent: string; frontier: TrackerIssue[] }>(
      `/tracker/issues/${encodeURIComponent(key)}/frontier`,
    ),
  getTrackerMap: (key: string) =>
    fetchJSON<TrackerMapProjection>(`/tracker/issues/${encodeURIComponent(key)}/map`),
  getTrackerGraph: (key: string, maxDepth = 8, maxNodes = 300) =>
    fetchJSON<TrackerGraphProjection>(
      `/tracker/issues/${encodeURIComponent(key)}/graph?max_depth=${maxDepth}&max_nodes=${maxNodes}`,
    ),
  claimTrackerIssue: (key: string, claimant: string) =>
    fetchJSON<TrackerClaimResult>(`/tracker/issues/${encodeURIComponent(key)}/claim`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ claimant }),
    }),
  unclaimTrackerIssue: (key: string, actor?: string) =>
    fetchJSON<TrackerUnclaimResult>(`/tracker/issues/${encodeURIComponent(key)}/unclaim`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor }),
    }),
}
