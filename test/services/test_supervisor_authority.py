"""Supervisor authority: the high-water union, the decision, and phases A/C.

The invariants under test are narrow and absolute: an epoch is never reused, a
surviving run never restarts at ``(1, 1)``, and an unprovable prior run refuses
rather than guessing. Most rows here exist to pin one of those three.
"""

from __future__ import annotations

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.services import supervisor_authority as authority

PROJECT = "cao-proj"


def _tuple(terminal: str = "term0001", generation: str = "gen-0001"):
    return authority.SupervisorTuple(
        project=PROJECT, supervisor_terminal_id=terminal, supervisor_generation=generation
    )


def _current(**overrides):
    fields = dict(
        project=PROJECT,
        project_incarnation=7,
        supervisor_terminal_id="old-term",
        supervisor_generation="old-gen",
        authority_epoch=11,
        state=authority.STATE_LIVE,
        established_at="2026-08-01T00:00:00Z",
    )
    fields.update(overrides)
    return database.ProjectSupervisorAuthorityModel(**fields)


# --------------------------------------------------------------------------
# The high-water is a union, and no single source is privileged.
# --------------------------------------------------------------------------


def test_high_water_is_zero_when_nothing_survives(isolated_memory_db):
    got = authority.compute_high_water(PROJECT)
    assert (got.epoch_max, got.incarnation_max, got.any_source) == (0, 0, False)


def test_allocation_alone_supplies_the_high_water(isolated_memory_db):
    with database.SessionLocal() as db:
        db.add(
            database.RouteObservationAuthorityEpochAllocationModel(
                project=PROJECT, authority_epoch=12, project_incarnation=7
            )
        )
        db.commit()
    got = authority.compute_high_water(PROJECT)
    assert (got.epoch_max, got.incarnation_max, got.any_source) == (12, 7, True)


def test_history_alone_supplies_the_high_water(isolated_memory_db):
    """History is a contributor, never the privileged maximum."""
    with database.SessionLocal() as db:
        db.add(
            database.ProjectSupervisorAuthorityHistoryModel(
                project=PROJECT,
                authority_epoch=11,
                project_incarnation=7,
                supervisor_terminal_id="t",
                supervisor_generation="g",
                state_at_close=authority.CLOSE_ROTATED,
            )
        )
        db.commit()
    got = authority.compute_high_water(PROJECT)
    assert (got.epoch_max, got.incarnation_max) == (11, 7)


def test_high_water_takes_the_max_across_all_sources(isolated_memory_db):
    with database.SessionLocal() as db:
        db.add(_current(authority_epoch=11))
        db.add(
            database.ProjectSupervisorAuthorityHistoryModel(
                project=PROJECT,
                authority_epoch=9,
                project_incarnation=7,
                supervisor_terminal_id="t",
                supervisor_generation="g",
                state_at_close=authority.CLOSE_ROTATED,
            )
        )
        db.add(
            database.RouteObservationAuthorityEpochAllocationModel(
                project=PROJECT, authority_epoch=14, project_incarnation=7
            )
        )
        db.commit()
    got = authority.compute_high_water(PROJECT)
    assert got.epoch_max == 14, "the allocation table is the fastest source, not a lesser one"


def test_high_water_is_scoped_per_project(isolated_memory_db):
    """A disposable proof project cannot contaminate a real one.

    Structural, not procedural: a different project key simply cannot contribute
    to another project's maxima.
    """
    with database.SessionLocal() as db:
        db.add(
            database.RouteObservationAuthorityEpochAllocationModel(
                project="scratch-proof", authority_epoch=999, project_incarnation=42
            )
        )
        db.commit()
    got = authority.compute_high_water(PROJECT)
    assert (got.epoch_max, got.incarnation_max, got.any_source) == (0, 0, False)


# --------------------------------------------------------------------------
# The decision table.
# --------------------------------------------------------------------------


def test_fresh_project_bootstraps_at_one_one(isolated_memory_db):
    got = authority.decide(_tuple(), witness=authority.ExistingRunWitness.ABSENT)
    assert got.decision is authority.Decision.BOOTSTRAP
    assert (got.project_incarnation, got.authority_epoch) == (1, 1)


def test_nothing_survives_but_prior_run_present_refuses(isolated_memory_db):
    """The `project.json`-only case: an existing run with no epoch high-water."""
    got = authority.decide(_tuple(), witness=authority.ExistingRunWitness.PRESENT)
    assert got.decision is authority.Decision.REFUSE
    assert got.reason_code == authority.REASON_RECOVERY_HIGH_WATER_UNAVAILABLE
    assert got.project_incarnation is None and got.authority_epoch is None


def test_unknown_prior_run_refuses_and_never_mints_one_one(isolated_memory_db):
    """`UNKNOWN` must not be read as freshness.

    Collapsing it into `ABSENT` would restart a surviving run at `(1, 1)` and
    reissue epochs already bound elsewhere — the exact reuse the incarnation
    exists to prevent.
    """
    got = authority.decide(_tuple(), witness=authority.ExistingRunWitness.UNKNOWN)
    assert got.decision is authority.Decision.REFUSE
    assert got.reason_code == authority.REASON_RECOVERY_HIGH_WATER_UNAVAILABLE


def test_allocation_absent_finality_style_survivor_recovers_above_observed(isolated_memory_db):
    """A surviving pair recovers: incarnation preserved, epoch strictly above."""
    with database.SessionLocal() as db:
        db.add(
            database.ProjectSupervisorAuthorityHistoryModel(
                project=PROJECT,
                authority_epoch=11,
                project_incarnation=7,
                supervisor_terminal_id="t",
                supervisor_generation="g",
                state_at_close=authority.CLOSE_ROTATED,
            )
        )
        db.commit()
    got = authority.decide(_tuple(), witness=authority.ExistingRunWitness.UNKNOWN)
    assert got.decision is authority.Decision.RECOVER
    assert (got.project_incarnation, got.authority_epoch) == (
        7,
        12,
    ), "never (8, 12), never a restart"


def test_recover_ignores_the_witness_because_a_source_survived(isolated_memory_db):
    """With a surviving pair the run is known to exist; the witness adds nothing."""
    with database.SessionLocal() as db:
        db.add(
            database.RouteObservationAuthorityEpochAllocationModel(
                project=PROJECT, authority_epoch=11, project_incarnation=7
            )
        )
        db.commit()
    for witness in authority.ExistingRunWitness:
        got = authority.decide(_tuple(), witness=witness)
        assert got.decision is authority.Decision.RECOVER
        assert (got.project_incarnation, got.authority_epoch) == (7, 12)


def test_exact_same_live_tuple_is_an_idempotent_noop(isolated_memory_db):
    """Rotating here would revoke valid grants on an ordinary recovery."""
    with database.SessionLocal() as db:
        db.add(_current(supervisor_terminal_id="same-t", supervisor_generation="same-g"))
        db.commit()
    got = authority.decide(_tuple("same-t", "same-g"))
    assert got.decision is authority.Decision.NOOP
    assert (got.project_incarnation, got.authority_epoch) == (7, 11), "epoch stays put"


def test_live_supervisor_refuses(isolated_memory_db, monkeypatch):
    with database.SessionLocal() as db:
        db.add(_current())
        db.commit()
    monkeypatch.setattr(authority, "is_supervisor_live", lambda t, g: True)
    got = authority.decide(_tuple("new-t", "new-g"))
    assert got.decision is authority.Decision.REFUSE
    assert got.reason_code == authority.REASON_LIVE_SUPERVISOR_PRESENT


def test_dead_supervisor_is_adopted_preserving_incarnation(isolated_memory_db, monkeypatch):
    """The `revive` case: rotate forward, same run."""
    with database.SessionLocal() as db:
        db.add(_current())
        db.commit()
    monkeypatch.setattr(authority, "is_supervisor_live", lambda t, g: False)
    got = authority.decide(_tuple("new-t", "new-g"))
    assert got.decision is authority.Decision.ADOPT
    assert (got.project_incarnation, got.authority_epoch) == (7, 12)


def test_revoked_row_is_adopted_forward_never_back_to_zero(isolated_memory_db):
    with database.SessionLocal() as db:
        db.add(_current(state=authority.STATE_REVOKED, authority_epoch=11))
        db.commit()
    got = authority.decide(_tuple("new-t", "new-g"))
    assert got.decision is authority.Decision.ADOPT
    assert got.authority_epoch == 12


def test_absent_terminal_row_is_not_proof_of_life(isolated_memory_db):
    assert authority.is_supervisor_live("nope", "nope") is False


# --------------------------------------------------------------------------
# Phase A commits on its own; a failed phase C cannot roll it back.
# --------------------------------------------------------------------------


def test_phase_a_allocation_is_durable_and_insert_only(isolated_memory_db):
    authority.phase_a_allocate(PROJECT, 1, 1)
    got = authority.compute_high_water(PROJECT)
    assert got.epoch_max == 1
    with pytest.raises(Exception):
        # The primary key is what makes an epoch un-reissuable, not application
        # code remembering to check.
        authority.phase_a_allocate(PROJECT, 1, 1)


def test_epoch_is_consumed_even_when_phase_c_never_binds(isolated_memory_db):
    """The property the phase split exists for.

    Phase A commits, phase C is never reached, and the next decision must skip
    past the consumed epoch rather than reissuing it.
    """
    authority.phase_a_allocate(PROJECT, 1, 1)
    # No phase C. The next creator observes 1 and must allocate 2.
    got = authority.decide(_tuple(), witness=authority.ExistingRunWitness.ABSENT)
    assert got.decision is authority.Decision.RECOVER
    assert got.authority_epoch == 2, "a gap is harmless; reuse is not"


def test_phase_c_bootstrap_inserts_the_live_row(isolated_memory_db):
    decision = authority.AuthorityDecision(
        decision=authority.Decision.BOOTSTRAP, project_incarnation=1, authority_epoch=1
    )
    bound, reason = authority.phase_c_bind(_tuple(), decision)
    assert (bound, reason) == (True, None)
    with database.SessionLocal() as db:
        row = db.query(database.ProjectSupervisorAuthorityModel).one()
        assert row.state == authority.STATE_LIVE
        assert row.authority_epoch == 1
        assert row.supervisor_terminal_id == "term0001"


def test_phase_c_bootstrap_loses_to_a_row_that_appeared_meanwhile(isolated_memory_db):
    """Two creators both allocate in A and serialize here; one loses."""
    with database.SessionLocal() as db:
        db.add(_current())
        db.commit()
    decision = authority.AuthorityDecision(
        decision=authority.Decision.BOOTSTRAP, project_incarnation=1, authority_epoch=1
    )
    bound, reason = authority.phase_c_bind(_tuple(), decision)
    assert bound is False
    assert reason == authority.REASON_ROTATION_CONFLICT


def test_phase_c_adopt_updates_in_place_and_appends_history(isolated_memory_db):
    """One row per project, CAS-updated; the superseded epoch is preserved."""
    with database.SessionLocal() as db:
        db.add(_current())
        db.commit()
    decision = authority.AuthorityDecision(
        decision=authority.Decision.ADOPT, project_incarnation=7, authority_epoch=12
    )
    bound, reason = authority.phase_c_bind(_tuple("new-t", "new-g"), decision)
    assert (bound, reason) == (True, None)
    with database.SessionLocal() as db:
        rows = db.query(database.ProjectSupervisorAuthorityModel).all()
        assert len(rows) == 1, "UNIQUE(project) is never violated by a rotation"
        assert rows[0].authority_epoch == 12
        assert rows[0].supervisor_terminal_id == "new-t"
        history = db.query(database.ProjectSupervisorAuthorityHistoryModel).all()
        assert len(history) == 1
        assert history[0].authority_epoch == 11
        assert history[0].state_at_close == authority.CLOSE_ROTATED


def test_rotation_never_lowers_the_high_water(isolated_memory_db):
    with database.SessionLocal() as db:
        db.add(_current())
        db.commit()
    authority.phase_c_bind(
        _tuple("new-t", "new-g"),
        authority.AuthorityDecision(
            decision=authority.Decision.ADOPT, project_incarnation=7, authority_epoch=12
        ),
    )
    assert authority.compute_high_water(PROJECT).epoch_max == 12


def test_phase_c_noop_binds_without_touching_anything(isolated_memory_db):
    with database.SessionLocal() as db:
        db.add(_current(supervisor_terminal_id="same-t", supervisor_generation="same-g"))
        db.commit()
    bound, reason = authority.phase_c_bind(
        _tuple("same-t", "same-g"),
        authority.AuthorityDecision(
            decision=authority.Decision.NOOP, project_incarnation=7, authority_epoch=11
        ),
    )
    assert (bound, reason) == (True, None)
    with database.SessionLocal() as db:
        assert db.query(database.ProjectSupervisorAuthorityHistoryModel).count() == 0


# --------------------------------------------------------------------------
# The bootstrap precondition excludes phase B's own terminal.
# --------------------------------------------------------------------------


def test_bootstrap_precondition_excludes_phase_b_terminal(isolated_memory_db):
    """Without the exclusion this self-refuses every genuine bootstrap."""
    with database.SessionLocal() as db:
        db.add(
            database.TerminalModel(
                id="phase-b",
                tmux_session="cao-fresh",
                tmux_window="supervisor",
                provider="codex",
            )
        )
        db.commit()
    assert authority.session_has_other_live_managed_terminal("cao-fresh", "phase-b") is False


def test_bootstrap_precondition_sees_a_foreign_live_terminal(isolated_memory_db):
    with database.SessionLocal() as db:
        db.add_all(
            [
                database.TerminalModel(
                    id="phase-b",
                    tmux_session="cao-shared",
                    tmux_window="supervisor",
                    provider="codex",
                ),
                database.TerminalModel(
                    id="someone-else",
                    tmux_session="cao-shared",
                    tmux_window="worker",
                    provider="codex",
                ),
            ]
        )
        db.commit()
    assert authority.session_has_other_live_managed_terminal("cao-shared", "phase-b") is True


def test_unreadable_session_blocks_bootstrap(isolated_memory_db, monkeypatch):
    """Cannot prove the session empty, so bootstrap must not proceed."""
    import cli_agent_orchestrator.clients.database as db_module

    def boom(_session):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(db_module, "list_terminals_by_session", boom)
    assert authority.session_has_other_live_managed_terminal("cao-x", "phase-b") is True


# --------------------------------------------------------------------------
# No bearer credential exists to leak.
# --------------------------------------------------------------------------


def test_no_authority_table_has_a_credential_column():
    """Asserted over the schema, so it cannot regress by accident."""
    forbidden = ("credential", "secret", "token", "hash", "signature", "hmac", "key")
    for model in (
        database.ProjectSupervisorAuthorityModel,
        database.ProjectSupervisorAuthorityHistoryModel,
        database.RouteObservationAuthorityEpochAllocationModel,
    ):
        for column in model.__table__.columns:
            name = column.name.lower()
            assert not any(
                bad in name for bad in forbidden
            ), f"{model.__tablename__}.{column.name} looks like authority material"
