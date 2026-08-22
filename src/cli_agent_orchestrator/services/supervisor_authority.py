"""Supervisor authority: the SOURCES high-water, the decision, and phases A/C.

Authority is bound to *which run of a project* it belongs to, not merely to a
project name, and an epoch is **never** reused. Everything here exists to make
those two statements true across crashes, races, and partial state loss.

**Why the work is split into phases rather than one transaction.** Terminal
creation is a settle wait — provider initialization takes 15-30 s — so holding a
write lock on the main database across it would block inbox delivery,
control-input, and managed launch for that whole window. Worse, under one
transaction a crash before the CAS and a lost race would both roll the epoch
allocation *back*, and the guarantee that an allocated epoch is consumed and
never reissued would be false. So:

* **phase A** decides and commits an allocation, on its own, in a short
  transaction. This commit is the durable boundary.
* **phase B** creates the terminal with no transaction held (not in this
  module — the channel owns it).
* **phase C** appends history and CASes the single current row, in a short
  transaction. On any failed precondition it aborts, and the phase-A epoch is
  **consumed, never reissued**. A gap in the sequence is expected and harmless.

**The high-water is a union, and no source is privileged.** Recovery reads every
surviving source that carries a ``(project_incarnation, authority_epoch)`` pair
and takes the maximum over all of them. History is one contributor, not the
answer; the allocation table is the *fastest* source, not a privileged one.
Losing any single store lowers nothing.

**`project.json` is deliberately not a source** — it carries a run identity but
no epoch, so it cannot contribute a high-water. Its *survival* is what
distinguishes a genuinely fresh project from an existing run whose epoch
high-water is unrecoverable, and those two cases have opposite outcomes. The
fork cannot read conductor state, so that fact arrives as an explicit
:class:`ExistingRunWitness` and **`UNKNOWN` fails closed** — never as fresh. See
the note on :class:`ExistingRunWitness` for why that asymmetry matters.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Reason codes, all members of the design's closed v1 vocabulary.
REASON_LIVE_SUPERVISOR_PRESENT = "authority-live-supervisor-present"
REASON_RECOVERY_HIGH_WATER_UNAVAILABLE = "recovery-high-water-unavailable"
REASON_ROTATION_CONFLICT = "authority-rotation-conflict"
REASON_EPOCH_ALLOCATION_CONFLICT = "authority-epoch-allocation-conflict"
REASON_BOOTSTRAP_UNAVAILABLE = "authority-bootstrap-unavailable"

STATE_LIVE = "live"
STATE_REVOKED = "revoked"

CLOSE_ROTATED = "rotated"
CLOSE_REVOKED = "revoked"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_provenance(provenance: Optional[dict]) -> Optional[str]:
    """Serialize the witness provenance for the authority row.

    Stored as JSON text rather than spread across columns because it is an audit
    record read by humans and never queried on, and because the observation's
    shape belongs to the observer.
    """
    if provenance is None:
        return None
    return json.dumps(provenance, separators=(",", ":"), sort_keys=True)


class EpochAllocationConflict(RuntimeError):
    """Another creator allocated this epoch first. Surfaced, never silent."""


class ExistingRunWitness(enum.Enum):
    """Whether a prior run of this project is known to survive.

    Three states, and the third is the point. The conductor owns
    ``project.json``; this server cannot see it. So "no prior run" and "cannot
    tell" are different facts, and collapsing them would resurrect exactly the
    failure the refuse branch exists to prevent: an existing run whose fork-side
    authority artifacts were lost would be classified *fresh* and restart at
    ``(1, 1)``, reissuing epochs already bound into surviving records elsewhere.

    ``UNKNOWN`` therefore refuses. Only a positive ``ABSENT`` — which a
    disposable proof project has by construction, and which an ordinary project
    must have supplied by the lane that owns ``project.json`` — admits the fresh
    branch.
    """

    ABSENT = "absent"
    PRESENT = "present"
    UNKNOWN = "unknown"


class Decision(enum.Enum):
    BOOTSTRAP = "bootstrap"
    RECOVER = "recover"
    NOOP = "noop"
    ADOPT = "adopt"
    REFUSE = "refuse"


@dataclass(frozen=True)
class HighWater:
    """The maxima over the full SOURCES union, and whether anything survived."""

    epoch_max: int
    incarnation_max: int
    any_source: bool


@dataclass(frozen=True)
class AuthorityDecision:
    decision: Decision
    reason_code: Optional[str] = None
    detail: Optional[str] = None
    project_incarnation: Optional[int] = None
    authority_epoch: Optional[int] = None
    #: What the witness observation saw and on what basis. Carried on every
    #: decision, refusals included: an auditor asking why `(1, 1)` was minted —
    #: or why a bring-up refused — needs the basis, not just the verdict.
    witness_provenance: Optional[dict] = None


@dataclass(frozen=True)
class SupervisorTuple:
    """What the *server* derived, never what a caller said."""

    project: str
    supervisor_terminal_id: str
    supervisor_generation: str


def compute_high_water(project: str) -> HighWater:
    """Maximum ``(incarnation, epoch)`` over every surviving source for a project.

    Scoped by ``project`` throughout, which is what makes a disposable proof
    project structurally unable to contaminate a real one: a different key
    cannot contribute to another project's maxima, so no proof run can lower,
    raise, or reuse a real project's epoch.

    **Two named contributors are intentionally omitted, each with the gate that
    creates its store:**

    * ``route_observation_finality`` — **ordered gate 4**, conductor lane. It is a
      conductor-owned JSON file per run; no such store exists on this fork.
    * surviving **grants** (``route_observation_grant``) — added **with the grant
      work**, which belongs to the route-observation operation behind gate 2.

    Each must be added **in the same change that creates its store**, not
    earlier: a reader for a table that does not exist is dead code that looks
    like coverage. Their absence is conservative *only* because every
    contributor can only ever *raise* the maxima — so omitting one can never
    lower a high-water and can never license a reuse. That reasoning stops
    holding the moment either store exists, which is why the gate is named here
    rather than left to a changelog.
    """
    from cli_agent_orchestrator.clients.database import (
        ProjectSupervisorAuthorityHistoryModel,
        ProjectSupervisorAuthorityModel,
        RouteObservationAuthorityEpochAllocationModel,
        SessionLocal,
    )

    epochs: list[int] = []
    incarnations: list[int] = []
    any_source = False

    with SessionLocal() as db:
        current = (
            db.query(ProjectSupervisorAuthorityModel)
            .filter(ProjectSupervisorAuthorityModel.project == project)
            .one_or_none()
        )
        if current is not None:
            any_source = True
            epochs.append(int(current.authority_epoch))
            incarnations.append(int(current.project_incarnation))

        for row in (
            db.query(
                ProjectSupervisorAuthorityHistoryModel.authority_epoch,
                ProjectSupervisorAuthorityHistoryModel.project_incarnation,
            )
            .filter(ProjectSupervisorAuthorityHistoryModel.project == project)
            .all()
        ):
            any_source = True
            epochs.append(int(row[0]))
            incarnations.append(int(row[1]))

        for row in (
            db.query(
                RouteObservationAuthorityEpochAllocationModel.authority_epoch,
                RouteObservationAuthorityEpochAllocationModel.project_incarnation,
            )
            .filter(RouteObservationAuthorityEpochAllocationModel.project == project)
            .all()
        ):
            any_source = True
            epochs.append(int(row[0]))
            incarnations.append(int(row[1]))

    return HighWater(
        epoch_max=max(epochs) if epochs else 0,
        incarnation_max=max(incarnations) if incarnations else 0,
        any_source=any_source,
    )


def is_supervisor_live(terminal_id: str, generation: str) -> bool:
    """Whether the recorded supervisor terminal is provably still alive.

    Conservative in the direction that matters: only a row this server still
    records as live counts as live. An absent row is *not* proof of life, so
    adoption is available; that is the ``revive`` case.
    """
    from cli_agent_orchestrator.clients.database import SessionLocal, TerminalModel

    with SessionLocal() as db:
        row = (
            db.query(TerminalModel)
            .filter(
                TerminalModel.id == terminal_id,
                TerminalModel.generation == generation,
            )
            .one_or_none()
        )
        if row is None:
            return False
        lifecycle = (row.lifecycle_state or "").lower()
        if lifecycle in {"exited", "superseded", "dead", "retired"}:
            return False
        return True


def decide(
    tuple_: SupervisorTuple,
    *,
    witness: ExistingRunWitness = ExistingRunWitness.UNKNOWN,
    witness_provenance: Optional[dict] = None,
) -> AuthorityDecision:
    """The single decision, over server-derived state only.

    Consults the current row **and** the full recovery union — reading only the
    current row would resurrect the reuse the incarnation exists to prevent,
    because losing that row would bootstrap a live project back to ``(1, 1)``
    and reissue epochs already bound elsewhere.
    """
    from cli_agent_orchestrator.clients.database import (
        ProjectSupervisorAuthorityModel,
        SessionLocal,
    )

    with SessionLocal() as db:
        current = (
            db.query(ProjectSupervisorAuthorityModel)
            .filter(ProjectSupervisorAuthorityModel.project == tuple_.project)
            .one_or_none()
        )
        current_snapshot = (
            None
            if current is None
            else (
                current.state,
                int(current.project_incarnation),
                current.supervisor_terminal_id,
                current.supervisor_generation,
                int(current.authority_epoch),
            )
        )

    high_water = compute_high_water(tuple_.project)

    if current_snapshot is None:
        if not high_water.any_source:
            if witness is ExistingRunWitness.ABSENT:
                return AuthorityDecision(
                    decision=Decision.BOOTSTRAP,
                    project_incarnation=1,
                    authority_epoch=1,
                    witness_provenance=witness_provenance,
                )
            # Either a prior run is known to survive, or we cannot tell. Both
            # refuse: an existing run whose epoch high-water is unrecoverable
            # must never be guessed at, and "cannot tell" is not a proof of
            # freshness.
            return AuthorityDecision(
                decision=Decision.REFUSE,
                reason_code=REASON_RECOVERY_HIGH_WATER_UNAVAILABLE,
                detail=(
                    "existing-run-present"
                    if witness is ExistingRunWitness.PRESENT
                    else "existing-run-unknown"
                ),
                witness_provenance=witness_provenance,
            )
        # Ordinary partial loss: the run survived, so the incarnation is
        # preserved and the epoch is allocated strictly above every observed
        # value. A new run requires positive proof, which absence of state is
        # not.
        return AuthorityDecision(
            decision=Decision.RECOVER,
            project_incarnation=high_water.incarnation_max,
            authority_epoch=high_water.epoch_max + 1,
            witness_provenance=witness_provenance,
        )

    state, incarnation, terminal_id, generation, epoch = current_snapshot

    if state == STATE_REVOKED:
        return AuthorityDecision(
            decision=Decision.ADOPT,
            project_incarnation=incarnation,
            authority_epoch=high_water.epoch_max + 1,
            witness_provenance=witness_provenance,
        )

    if terminal_id == tuple_.supervisor_terminal_id and generation == tuple_.supervisor_generation:
        # The exact live tuple: idempotent no-op. Incarnation and epoch both
        # stay put, and no grant is invalidated. Rotating here would revoke
        # valid grants on an ordinary same-session recovery.
        return AuthorityDecision(
            decision=Decision.NOOP,
            project_incarnation=incarnation,
            authority_epoch=epoch,
            witness_provenance=witness_provenance,
        )

    if is_supervisor_live(str(terminal_id), str(generation)):
        return AuthorityDecision(
            decision=Decision.REFUSE,
            reason_code=REASON_LIVE_SUPERVISOR_PRESENT,
            detail=f"live-supervisor:{terminal_id}",
            witness_provenance=witness_provenance,
        )

    return AuthorityDecision(
        decision=Decision.ADOPT,
        project_incarnation=incarnation,
        authority_epoch=high_water.epoch_max + 1,
        witness_provenance=witness_provenance,
    )


def phase_a_allocate(project: str, incarnation: int, epoch: int) -> None:
    """Commit the allocation for the epoch about to be used, on its own.

    This commit is the durable boundary of the whole pipeline. It is
    deliberately *not* joined to phase C: if it were, a lost CAS would roll the
    allocation back and the next creator could reissue this epoch.
    """
    from cli_agent_orchestrator.clients.database import (
        RouteObservationAuthorityEpochAllocationModel,
        SessionLocal,
    )

    with SessionLocal() as db:
        db.add(
            RouteObservationAuthorityEpochAllocationModel(
                project=project,
                authority_epoch=epoch,
                project_incarnation=incarnation,
                allocated_at=_now(),
            )
        )
        try:
            db.commit()
        except Exception as exc:
            # Two legitimate creators can race here, and the primary key is what
            # makes the epoch un-reissuable. The loser gets a named outcome
            # rather than a raw integrity error, because a client is waiting for
            # an answer and an unnamed failure is indistinguishable from a hang.
            db.rollback()
            raise EpochAllocationConflict(
                f"epoch {epoch} is already allocated for {project!r}; it is "
                "consumed and never reissued"
            ) from exc


def phase_c_bind(
    tuple_: SupervisorTuple,
    decision: AuthorityDecision,
    *,
    channel_socket_path: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Append history and CAS the single current row. ``(bound, reason_code)``.

    The CAS is what serializes concurrent creators: both allocate in phase A,
    then exactly one wins here. The loser aborts
    ``authority-rotation-conflict`` and its terminal is torn down by the caller;
    its allocated epoch stays consumed.
    """
    from cli_agent_orchestrator.clients.database import (
        ProjectSupervisorAuthorityHistoryModel,
        ProjectSupervisorAuthorityModel,
        SessionLocal,
    )

    if decision.decision is Decision.NOOP:
        return True, None

    with SessionLocal() as db:
        try:
            current = (
                db.query(ProjectSupervisorAuthorityModel)
                .filter(ProjectSupervisorAuthorityModel.project == tuple_.project)
                .with_for_update()
                .one_or_none()
            )
        except Exception:
            # SQLite has no row locks; the single-row uniqueness constraint plus
            # the transaction is the serialization point.
            current = (
                db.query(ProjectSupervisorAuthorityModel)
                .filter(ProjectSupervisorAuthorityModel.project == tuple_.project)
                .one_or_none()
            )

        if decision.decision in (Decision.BOOTSTRAP, Decision.RECOVER):
            if current is not None:
                # Another creator bound this project between our decision and
                # our CAS.
                db.rollback()
                return False, REASON_ROTATION_CONFLICT
            db.add(
                ProjectSupervisorAuthorityModel(
                    project=tuple_.project,
                    project_incarnation=decision.project_incarnation,
                    supervisor_terminal_id=tuple_.supervisor_terminal_id,
                    supervisor_generation=tuple_.supervisor_generation,
                    authority_epoch=decision.authority_epoch,
                    channel_socket_path=channel_socket_path,
                    state=STATE_LIVE,
                    established_at=_now(),
                    witness_provenance_json=_encode_provenance(decision.witness_provenance),
                )
            )
            db.commit()
            return True, None

        if decision.decision is Decision.ADOPT:
            if current is None:
                db.rollback()
                return False, REASON_ROTATION_CONFLICT
            # The superseded epoch is appended before the row moves, so the
            # high-water can never be lowered by the rotation itself.
            db.add(
                ProjectSupervisorAuthorityHistoryModel(
                    project=tuple_.project,
                    authority_epoch=int(current.authority_epoch),
                    project_incarnation=int(current.project_incarnation),
                    supervisor_terminal_id=str(current.supervisor_terminal_id),
                    supervisor_generation=str(current.supervisor_generation),
                    state_at_close=(
                        CLOSE_REVOKED if current.state == STATE_REVOKED else CLOSE_ROTATED
                    ),
                    opened_at=current.established_at,
                    closed_at=_now(),
                )
            )
            # Updated through the query rather than by attribute assignment, so
            # the whole rotation is one statement on the one row.
            new_values: dict = {
                "project_incarnation": decision.project_incarnation,
                "supervisor_terminal_id": tuple_.supervisor_terminal_id,
                "supervisor_generation": tuple_.supervisor_generation,
                "authority_epoch": decision.authority_epoch,
                "state": STATE_LIVE,
                "rotated_at": _now(),
                "witness_provenance_json": _encode_provenance(decision.witness_provenance),
            }
            if channel_socket_path is not None:
                new_values["channel_socket_path"] = channel_socket_path
            db.query(ProjectSupervisorAuthorityModel).filter(
                ProjectSupervisorAuthorityModel.project == tuple_.project
            ).update(new_values, synchronize_session=False)
            db.commit()
            return True, None

    return False, REASON_ROTATION_CONFLICT


def session_has_other_live_managed_terminal(tmux_session: str, exclude_terminal_id: str) -> bool:
    """Bootstrap's phase-C precondition, with phase B's own terminal excluded.

    Without the exclusion this self-refuses every genuine bootstrap: phase B
    records its terminal in the target session *before* phase C runs, and that
    row is by construction a live managed terminal.
    """
    from cli_agent_orchestrator.clients.database import list_terminals_by_session

    try:
        rows = list_terminals_by_session(tmux_session)
    except Exception:
        logger.exception("supervisor-authority: session terminal lookup failed")
        # Cannot prove the session empty, so bootstrap must not proceed.
        return True
    return any(row.get("id") != exclude_terminal_id for row in rows)
