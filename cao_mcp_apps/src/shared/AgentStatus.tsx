// AgentStatus — a single agent (terminal) card with a status badge.
//
// Status text/provider/profile are rendered as escaped React children. The
// status string is also normalized to a CSS modifier class for the badge color.

import React from "react";
import type { TerminalView } from "./types";
import { STATUS } from "./status.generated";

// The known status taxonomy is generated from the shared SSOT
// (design-tokens/status.json) via `node design-tokens/gen.mjs`.
const KNOWN_STATUSES = new Set(Object.keys(STATUS));

export interface AgentStatusProps {
  terminal: TerminalView;
  onOpen?: (terminalId: string) => void;
  /** Render a "supervisor" role badge and accent (the fleet coordinator). */
  isSupervisor?: boolean;
}

export function requestedRouteDisplay(
  value: string | null | undefined,
  state: string | null | undefined,
): string {
  if (state === "unreadable") return "unreadable (requested, not observed)";
  if (!value) return "unavailable (requested, not observed)";
  return `${value} (requested, not observed)`;
}

type RequestedRoute = Pick<
  TerminalView,
  | "provider"
  | "assigned_model"
  | "assigned_effort"
  | "assigned_quota_provider"
  | "assigned_route_state"
>;

export function RequestedRouteEntries({ route }: { route: RequestedRoute }) {
  return (
    <>
      <div>
        <dt>harness</dt>
        <dd data-testid="harness-label">{route.provider}</dd>
      </div>
      <div>
        <dt>AI provider</dt>
        <dd data-testid="ai-provider-label">
          {route.assigned_quota_provider ?? "unavailable"}
        </dd>
      </div>
      <div>
        <dt>model</dt>
        <dd data-testid="requested-model">
          {requestedRouteDisplay(
            route.assigned_model,
            route.assigned_route_state,
          )}
        </dd>
      </div>
      <div>
        <dt>effort</dt>
        <dd data-testid="requested-effort">
          {requestedRouteDisplay(
            route.assigned_effort,
            route.assigned_route_state,
          )}
        </dd>
      </div>
    </>
  );
}

export function AgentStatus({
  terminal,
  onOpen,
  isSupervisor = false,
}: AgentStatusProps): JSX.Element {
  const status = (terminal.status ?? "unknown").toLowerCase();
  const statusClass = KNOWN_STATUSES.has(status) ? status : "unknown";
  // Label/role/pulse are derived from the generated status SSOT; the color is
  // applied through the cao-status-<status> class, which resolves to the
  // role's host-overridable CSS variable in styles.css/tokens.generated.css.
  const semantics = STATUS[statusClass];
  const pulseClass = semantics?.pulse ? " cao-status-pulse" : "";
  return (
    <div
      className={`cao-card${isSupervisor ? " cao-card-supervisor" : ""}`}
      data-testid="agent-card"
      data-terminal-id={terminal.id}
      role={onOpen ? "button" : undefined}
      tabIndex={onOpen ? 0 : undefined}
      onClick={onOpen ? () => onOpen(terminal.id) : undefined}
    >
      <div className="cao-card-head">
        <span className="cao-card-title">
          {terminal.agent_profile ?? terminal.id}
          {isSupervisor && (
            <span className="cao-role-badge" data-testid="role-badge">
              supervisor
            </span>
          )}
        </span>
        <span
          className={`cao-status cao-status-${statusClass}${pulseClass}`}
          data-testid="status-badge"
          title={semantics?.label}
        >
          {status}
        </span>
      </div>
      <dl className="cao-card-meta">
        <RequestedRouteEntries route={terminal} />
        <div>
          <dt>session</dt>
          <dd>{terminal.session_name}</dd>
        </div>
      </dl>
    </div>
  );
}
