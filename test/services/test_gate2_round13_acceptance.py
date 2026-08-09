"""The writer's round-13 acceptance checks, one section per handoff item.

The load-bearing choice here is in section A: **every referent is hand-written**,
never produced by `emit_receipt`. Building referents through the validating
emitter is exactly what made the original gap invisible — the strongest checks in
the codec only ever ran on documents this process had already written itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cli_agent_orchestrator.cli.commands import gate2_proof_run as runner
from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.services import gate2_proof_receipt as codec
from cli_agent_orchestrator.services import gate2_proof_receipt_state as rs
from cli_agent_orchestrator.services import supervisor_authority as authority
from cli_agent_orchestrator.services import supervisor_create_channel as channel
from cli_agent_orchestrator.services.actor_broker import PeerCredentials

from ._gate2_fixtures import minimal_receipt

CREDS = PeerCredentials(pid=4242, uid=501)


@pytest.fixture
def state_root(monkeypatch, tmp_path):
    import cli_agent_orchestrator.constants as constants

    root = tmp_path / "state"
    root.mkdir()
    monkeypatch.setattr(constants, "CAO_HOME_DIR", root)
    return root


@pytest.fixture(autouse=True)
def _clean_start_state():
    channel._set_designation_for_test(None)
    channel._set_receipt_state_for_test(None)
    channel._set_bindings_for_test(None)
    yield
    channel._set_designation_for_test(None)
    channel._set_receipt_state_for_test(None)
    channel._set_bindings_for_test(None)


def _hand_write(path: Path, document: dict) -> str:
    """Write a referent **without** the validating emitter, and return its digest.

    Plain `json.dumps`, so a document the codec would refuse still lands on disk
    exactly as an operator or a defective tool could place it.
    """
    payload = json.dumps(document).encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _install_hand_written(root: Path, document: dict) -> None:
    sha = _hand_write(root / codec.RECEIPT_BASENAME, document)
    rs.write_receipt_state_for_proof_run(root / rs.RECEIPT_STATE_BASENAME, sha)


# ==========================================================================
# A. P2-1 — full receipt validation at server start (T-A22h(h))
# ==========================================================================


def test_a_hand_written_valid_receipt_still_loads(state_root):
    """The control: a valid hand-written referent must still be accepted."""
    _install_hand_written(state_root, minimal_receipt())
    assert rs.load_receipt_state() is not None


def _mutate(**over) -> dict:
    doc = minimal_receipt()
    for path, value in over.items():
        parts = path.split(".")
        node = doc
        for key in parts[:-1]:
            node = node[key]
        node[parts[-1]] = value
    return doc


@pytest.mark.parametrize(
    "doc,label",
    [
        (_mutate(**{"proofs.lineage_isolation.outcome": "not-observed"}), "lineage unproven"),
        (
            _mutate(**{"proofs.supervisor_creation_discriminator.outcome": "skipped"}),
            "discriminator unproven",
        ),
        (_mutate(**{"isolation.is_disposable_instance": False}), "non-disposable"),
        (_mutate(**{"teardown.instance_destroyed": False}), "instance not destroyed"),
        (_mutate(**{"teardown.state_root_removed": False}), "state root kept"),
        (_mutate(**{"teardown.artifacts_deleted": False}), "artifacts kept"),
        (_mutate(**{"capability_dark.advertisement_enabled": True}), "advertisement on"),
        (_mutate(**{"capability_dark.provider_tuples_enabled": True}), "tuples on"),
        (
            _mutate(**{"capability_dark.listener_enabled_for_ordinary_projects": True}),
            "listener on",
        ),
        (_mutate(**{"ordinary_project_non_effect.authority_rows_created": 1}), "rows created"),
        (_mutate(**{"ordinary_project_non_effect.grants_minted": 2}), "grants minted"),
        (_mutate(**{"identities.tool_version": None}), "null required field"),
        (_mutate(**{"target.server_start_id": 7}), "wrong type"),
    ],
)
def test_start_refuses_a_hand_written_invalid_referent(state_root, doc, label):
    """A document the writer would refuse can no longer be accepted at start."""
    _install_hand_written(state_root, doc)
    with pytest.raises(rs.ReceiptStateError) as excinfo:
        rs.load_receipt_state()
    assert "not a valid canonical gate-2 receipt" in str(excinfo.value), label


def test_start_refuses_an_unknown_key_in_the_referent(state_root):
    doc = minimal_receipt()
    doc["force_enable"] = True
    _install_hand_written(state_root, doc)
    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state()


def test_start_refuses_a_missing_key_in_the_referent(state_root):
    doc = minimal_receipt()
    del doc["teardown"]
    _install_hand_written(state_root, doc)
    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state()


def test_the_reviewers_reproduction_now_inverts(state_root):
    """Digest matches, schema tag matches, and it is still refused.

    Before the fix this exact document was accepted: the referent check was a
    schema-tag-plus-digest comparison, so the codec's strongest rules never ran
    on a document the process had not itself written.
    """
    doc = _mutate(**{"isolation.is_disposable_instance": False})
    sha = _hand_write(state_root / codec.RECEIPT_BASENAME, doc)
    rs.write_receipt_state_for_proof_run(state_root / rs.RECEIPT_STATE_BASENAME, sha)

    import hashlib

    on_disk = (state_root / codec.RECEIPT_BASENAME).read_bytes()
    assert hashlib.sha256(on_disk).hexdigest() == sha, "digest genuinely matches"
    assert json.loads(on_disk)["schema"] == codec.RECEIPT_SCHEMA, "tag genuinely matches"

    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state()


# ==========================================================================
# B. P2-2 — run-began failure writes a partial, and only a partial (T-A22h(g))
# ==========================================================================


def _iso_root(tmp_path) -> Path:
    root = tmp_path / "proof-root"
    root.mkdir()
    return root


def _argv(root: Path, designation: Path) -> list:
    return [
        "--state-root",
        str(root),
        "--project",
        "cao-gate2-scratch",
        "--designation",
        str(designation),
    ]


@pytest.fixture
def designation(tmp_path) -> Path:
    path = tmp_path / "d.json"
    path.write_text("{}")
    return path


def _boom(request):
    raise RuntimeError("provider never settled")


def test_executor_failure_mid_run_writes_a_partial_only(tmp_path, designation):
    root = _iso_root(tmp_path)
    code = runner.run(_argv(root, designation), executor=_boom)

    assert code != runner.EXIT_OK
    assert (root / codec.PARTIAL_BASENAME).exists()
    assert not (root / codec.RECEIPT_BASENAME).exists()

    doc = json.loads((root / codec.PARTIAL_BASENAME).read_bytes())
    assert "proofs" not in doc
    assert "provider never settled" in doc["refusal_reason"]
    assert doc["observed_fact_names"] == []
    assert set(doc["unobserved_fact_names"]) == {
        "lineage_isolation",
        "supervisor_creation_discriminator",
    }
    # The teardown records what actually happened, not an asserted success.
    assert doc["teardown"] == {
        "instance_destroyed": False,
        "state_root_removed": False,
        "artifacts_deleted": False,
    }


def test_an_unknown_executor_schema_also_writes_a_partial(tmp_path, designation):
    root = _iso_root(tmp_path)
    code = runner.run(_argv(root, designation), executor=lambda r: {"schema": "nonsense"})
    assert code != runner.EXIT_OK
    assert (root / codec.PARTIAL_BASENAME).exists()
    assert not (root / codec.RECEIPT_BASENAME).exists()


def test_an_emitter_failure_does_not_escape_run(tmp_path, designation, monkeypatch):
    """An uncaught exception satisfies no row of the exit table."""
    root = _iso_root(tmp_path)

    def exploding_emit(path, document):
        raise OSError("disk full")

    monkeypatch.setattr(codec, "emit_receipt", exploding_emit)
    code = runner.run(_argv(root, designation), executor=lambda r: minimal_receipt())
    assert code == runner.EXIT_REFUSED
    assert not (root / codec.RECEIPT_BASENAME).exists()


def test_a_partial_emitter_failure_does_not_escape_run(tmp_path, designation, monkeypatch):
    root = _iso_root(tmp_path)

    def exploding_emit(path, document):
        raise OSError("disk full")

    monkeypatch.setattr(codec, "emit_partial", exploding_emit)
    code = runner.run(_argv(root, designation), executor=_boom)
    assert code == runner.EXIT_REFUSED


@pytest.mark.parametrize("basename", [codec.RECEIPT_BASENAME, codec.PARTIAL_BASENAME])
def test_a_collision_leaves_the_pre_existing_artifact_byte_identical(
    tmp_path, designation, basename
):
    root = _iso_root(tmp_path)
    original = b'{"prior": "artifact"}'
    (root / basename).write_bytes(original)

    code = runner.run(_argv(root, designation), executor=lambda r: minimal_receipt())

    assert code == runner.EXIT_REFUSED
    assert (root / basename).read_bytes() == original, "never overwritten"


def test_a_pre_execution_refusal_writes_neither_artifact(tmp_path, designation):
    root = _iso_root(tmp_path)
    (root / "cao.db").write_text("")
    code = runner.run(_argv(root, designation), executor=lambda r: minimal_receipt())
    assert code == runner.EXIT_REFUSED
    assert not (root / codec.RECEIPT_BASENAME).exists()
    assert not (root / codec.PARTIAL_BASENAME).exists()


@pytest.mark.parametrize(
    "executor",
    [lambda r: minimal_receipt(), _boom, lambda r: {"schema": "nonsense"}],
)
def test_never_both_artifacts_on_any_path(tmp_path, designation, executor):
    root = _iso_root(tmp_path)
    runner.run(_argv(root, designation), executor=executor)
    both = (root / codec.RECEIPT_BASENAME).exists() and (root / codec.PARTIAL_BASENAME).exists()
    assert not both


# ==========================================================================
# C. P2-3 + P3-1 — normalized A-scope session identity
# ==========================================================================


def test_one_tmux_session_is_one_project_key():
    assert channel.normalize_session_identity("foo") == "cao-foo"
    assert channel.normalize_session_identity("cao-foo") == "cao-foo"
    assert channel.normalize_session_identity("foo") == channel.normalize_session_identity(
        "cao-foo"
    )


@pytest.mark.parametrize("bad", ["", "   ", " padded ", "a\nb", 7, None])
def test_a_non_normalizable_session_name_is_refused_not_defaulted(bad):
    with pytest.raises(channel.SessionIdentityError):
        channel.normalize_session_identity(bad)


def test_both_spellings_share_one_sources_high_water_and_epoch_sequence(isolated_memory_db):
    """An epoch allocated under one spelling is never reissued under the other."""
    key_a = channel.normalize_session_identity("foo")
    key_b = channel.normalize_session_identity("cao-foo")
    assert key_a == key_b

    authority.phase_a_allocate(key_a, 1, 1)
    assert authority.compute_high_water(key_b).epoch_max == 1

    with pytest.raises(authority.EpochAllocationConflict):
        authority.phase_a_allocate(key_b, 1, 1)

    nxt = authority.decide(
        authority.SupervisorTuple(key_b, "t", "g"),
        witness=authority.ExistingRunWitness.ABSENT,
    )
    assert nxt.authority_epoch == 2


def test_the_project_key_is_read_from_the_normalized_value():
    assert channel._project_for({"session_name": "foo"}) == "cao-foo"
    assert channel._project_for({"session_name": "cao-foo"}) == "cao-foo"


def test_artifacts_name_the_normalized_identity(state_root, tmp_path):
    """psu-3: a raw-spelled artifact is inert; a normalized one matches."""
    from cli_agent_orchestrator.services import project_state_binding as psb

    project_dir = tmp_path / "conductor"
    project_dir.mkdir()

    psb.write_binding_for_project(psb.binding_path("foo"), "foo", str(project_dir))
    channel.load_designation_at_start()
    assert channel._bindings().recorded("cao-foo") is False, "raw-spelled binding is inert"

    psb.write_binding_for_project(psb.binding_path("cao-foo"), "cao-foo", str(project_dir))
    channel.load_designation_at_start()
    assert channel._bindings().recorded("cao-foo") is True


# T-A22e as restated: A-launch fields are decision-invariant.


def _install_create(monkeypatch, session="cao-scratch", terminal_id="phase-b"):
    async def fake_create(args):
        with database.SessionLocal() as db:
            db.add(
                database.TerminalModel(
                    id=terminal_id,
                    tmux_session=session,
                    tmux_window="supervisor",
                    provider="codex",
                    generation=f"gen-{terminal_id}",
                )
            )
            db.commit()
        return {
            "id": terminal_id,
            "generation": f"gen-{terminal_id}",
            "tmux_session": session,
        }

    monkeypatch.setattr(channel, "_create_terminal_from_set_a", fake_create)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_profile", "developer"),
        ("working_directory", "/somewhere/else"),
        ("env_vars", {"PATH": "/other"}),
        ("caller_id", "zzzz9999"),
        ("initial_message", "different"),
        ("orchestration_type", "handoff"),
        ("allowed_tools", "x,y"),
        ("defer_init", True),
    ],
)
async def test_varying_any_a_launch_field_leaves_the_decision_identical(
    isolated_memory_db, state_root, monkeypatch, field, value
):
    from cli_agent_orchestrator.services.gate2_proof_designation import Gate2ProofDesignation

    monkeypatch.setattr(
        channel, "classify_peer_origin", lambda pid, m: (channel.PeerOrigin.OPERATOR, None)
    )
    monkeypatch.setattr(
        channel, "managed_pid_set", lambda: channel.ManagedPidSet(frozenset({9}), True)
    )
    channel._set_designation_for_test(
        Gate2ProofDesignation(project="cao-scratch", sha256="0" * 64, path="/tmp/d.json")
    )

    args = {"agent_profile": "code_supervisor", "session_name": "scratch", field: value}
    _install_create(monkeypatch)
    outcome = await channel.handle_supervisor_terminal_create(
        args, credentials=CREDS, managed=channel.ManagedPidSet(frozenset({9}), True)
    )

    assert outcome.authority_granted is True
    with database.SessionLocal() as db:
        row = db.query(database.ProjectSupervisorAuthorityModel).one()
        # The subject is always phase B's terminal, never anything a caller named.
        assert row.supervisor_terminal_id == "phase-b"
        assert row.project == "cao-scratch"
        assert (row.project_incarnation, row.authority_epoch) == (1, 1)


# ==========================================================================
# D. Phase-C same-session proof and teardown
# ==========================================================================


@pytest.mark.asyncio
async def test_phase_c_session_mismatch_tears_down_and_consumes_the_epoch(
    isolated_memory_db, state_root, monkeypatch
):
    """Reachable only by a fork defect, so it is an internal invariant failure.

    It still aborts, tears down, and consumes the epoch — and it is surfaced with
    no section-13 code rather than closed silently.
    """
    from cli_agent_orchestrator.services.gate2_proof_designation import Gate2ProofDesignation

    monkeypatch.setattr(
        channel, "classify_peer_origin", lambda pid, m: (channel.PeerOrigin.OPERATOR, None)
    )
    monkeypatch.setattr(
        channel, "managed_pid_set", lambda: channel.ManagedPidSet(frozenset({9}), True)
    )
    channel._set_designation_for_test(
        Gate2ProofDesignation(project="cao-scratch", sha256="0" * 64, path="/tmp/d.json")
    )
    # Phase B lands the terminal in the wrong session.
    _install_create(monkeypatch, session="cao-somewhere-else")

    torn: list = []

    async def fake_teardown(terminal_id):
        torn.append(terminal_id)

    monkeypatch.setattr(channel, "_teardown", fake_teardown)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": "scratch"},
        credentials=CREDS,
        managed=channel.ManagedPidSet(frozenset({9}), True),
    )

    assert outcome.ok is False
    assert outcome.reason_code is None, "no section-13 code for an internal invariant"
    assert "internal-invariant" in outcome.detail
    assert torn == ["phase-b"]
    with database.SessionLocal() as db:
        assert db.query(database.ProjectSupervisorAuthorityModel).count() == 0
    # Epoch consumed, never reissued.
    assert authority.compute_high_water("cao-scratch").epoch_max == 1
    nxt = authority.decide(
        authority.SupervisorTuple("cao-scratch", "t2", "g2"),
        witness=authority.ExistingRunWitness.ABSENT,
    )
    assert nxt.authority_epoch == 2


# ==========================================================================
# E. Folded and non-widening dispositions
# ==========================================================================


@pytest.mark.parametrize("key", ["terminal_id", "role", "authority_epoch", "anything"])
def test_an_unknown_top_level_request_key_is_refused_not_dropped(key):
    payload = {
        "verb": channel.VERB_SUPERVISOR_TERMINAL_CREATE,
        "args": {"agent_profile": "code_supervisor"},
        key: "value",
    }
    with pytest.raises(channel.SupervisorCreateChannelError) as excinfo:
        channel.validate_request(payload)
    assert key in str(excinfo.value)


def test_a_well_formed_top_level_request_still_validates():
    got = channel.validate_request(
        {
            "verb": channel.VERB_SUPERVISOR_TERMINAL_CREATE,
            "args": {"agent_profile": "code_supervisor"},
        }
    )
    assert got["agent_profile"] == "code_supervisor"


def test_channel_protocol_refusals_carry_no_section_13_code():
    """P3-3: a malformed frame never reached an authority decision."""
    assert channel.REASON_PROTOCOL_REFUSED is None


def test_a_dead_terminal_pid_is_not_in_the_managed_set(isolated_memory_db, monkeypatch):
    """P3-2: the set is scoped to live rows, as the rule words it."""
    monkeypatch.setattr(channel, "_tmux_server_pid", lambda: None)
    with database.SessionLocal() as db:
        db.add_all(
            [
                database.TerminalModel(
                    id="live",
                    tmux_session="s",
                    tmux_window="w",
                    provider="codex",
                    pane_pid=1111,
                ),
                database.TerminalModel(
                    id="dead",
                    tmux_session="s",
                    tmux_window="w",
                    provider="codex",
                    pane_pid=2222,
                    lifecycle_state="exited",
                ),
            ]
        )
        db.commit()

    got = channel.managed_pid_set()
    assert got.enumerable is True
    assert 1111 in got.pids
    assert 2222 not in got.pids, "a retired terminal cannot be a peer's ancestor"


def test_a_historical_null_pane_pid_row_does_not_permanently_disable_the_set(
    isolated_memory_db, monkeypatch
):
    """The NULL limb: one retired row must not render the set unenumerable forever."""
    monkeypatch.setattr(channel, "_tmux_server_pid", lambda: None)
    with database.SessionLocal() as db:
        db.add_all(
            [
                database.TerminalModel(
                    id="historical",
                    tmux_session="s",
                    tmux_window="w",
                    provider="codex",
                    pane_pid=None,
                    lifecycle_state="exited",
                ),
                database.TerminalModel(
                    id="live",
                    tmux_session="s",
                    tmux_window="w",
                    provider="codex",
                    pane_pid=3333,
                ),
            ]
        )
        db.commit()

    got = channel.managed_pid_set()
    assert got.enumerable is True, "a dead NULL-pid row is excluded before the check"
    assert 3333 in got.pids


def test_a_live_null_pane_pid_row_still_makes_the_set_unenumerable(isolated_memory_db, monkeypatch):
    """The conservatism is kept where it still means something."""
    monkeypatch.setattr(channel, "_tmux_server_pid", lambda: None)
    with database.SessionLocal() as db:
        db.add(
            database.TerminalModel(
                id="live-incomplete",
                tmux_session="s",
                tmux_window="w",
                provider="codex",
                pane_pid=None,
            )
        )
        db.commit()
    assert channel.managed_pid_set().enumerable is False


def test_the_allocation_collision_has_a_named_outcome(isolated_memory_db):
    """P3-4: reachable by ordinary concurrency, so it gets a real reason code."""
    authority.phase_a_allocate("cao-p", 1, 1)
    with pytest.raises(authority.EpochAllocationConflict):
        authority.phase_a_allocate("cao-p", 1, 1)
    assert authority.REASON_EPOCH_ALLOCATION_CONFLICT == "authority-epoch-allocation-conflict"


@pytest.mark.asyncio
async def test_an_allocation_collision_is_surfaced_and_creates_nothing(
    isolated_memory_db, state_root, monkeypatch
):
    from cli_agent_orchestrator.services.gate2_proof_designation import Gate2ProofDesignation

    monkeypatch.setattr(
        channel, "classify_peer_origin", lambda pid, m: (channel.PeerOrigin.OPERATOR, None)
    )
    channel._set_designation_for_test(
        Gate2ProofDesignation(project="cao-scratch", sha256="0" * 64, path="/tmp/d.json")
    )
    created: list = []

    async def fake_create(args):
        created.append(args)
        return {}

    monkeypatch.setattr(channel, "_create_terminal_from_set_a", fake_create)

    def colliding(project, incarnation, epoch):
        raise authority.EpochAllocationConflict("epoch 1 already allocated")

    monkeypatch.setattr(authority, "phase_a_allocate", colliding)

    outcome = await channel.handle_supervisor_terminal_create(
        {"agent_profile": "code_supervisor", "session_name": "scratch"},
        credentials=CREDS,
        managed=channel.ManagedPidSet(frozenset({9}), True),
    )

    assert outcome.ok is False
    assert outcome.reason_code == authority.REASON_EPOCH_ALLOCATION_CONFLICT
    assert outcome.terminal_created is False
    assert created == [], "phase A precedes phase B, so nothing was created"


def test_no_handler_failure_closes_the_connection_without_a_response():
    """P3-4's second limb, asserted structurally on the handler."""
    import inspect

    source = inspect.getsource(channel.SupervisorCreateChannel._handle_connection)
    assert "internal-error" in source
    assert "writer.write(encode_response(outcome))" in source


# ==========================================================================
# F-1 — a begun run that produces an INVALID receipt writes a partial only
# ==========================================================================


def _invalid_receipt(request):
    """A receipt-shaped document that fails validation, with disk writes healthy.

    The reviewer's exact case: the run began and produced something, but what it
    produced is not a receipt anyone may act on.
    """
    doc = minimal_receipt()
    doc["proofs"]["lineage_isolation"]["outcome"] = "failed"
    return doc


def test_an_invalid_returned_receipt_writes_a_partial_only(tmp_path, designation):
    """A run that began and then failed validation must not vanish.

    Exiting with neither artifact loses the audit artifact Gate 2 needs, and it
    reports a begun run with the pre-execution exit code.
    """
    root = _iso_root(tmp_path)
    code = runner.run(_argv(root, designation), executor=_invalid_receipt)

    assert code != runner.EXIT_OK
    assert code != runner.EXIT_REFUSED, "a begun run is not a pre-execution refusal"
    assert (root / codec.PARTIAL_BASENAME).exists()
    assert not (root / codec.RECEIPT_BASENAME).exists(), "never a receipt"

    doc = json.loads((root / codec.PARTIAL_BASENAME).read_bytes())
    assert "proofs" not in doc
    assert "lineage_isolation" in doc["refusal_reason"]
    assert "proven" in doc["refusal_reason"], "the validation evidence is preserved"
    assert doc["teardown"] == {
        "instance_destroyed": False,
        "state_root_removed": False,
        "artifacts_deleted": False,
    }, "the actual teardown, never a borrowed success"


def test_a_partial_emitter_io_failure_leaves_no_artifact_without_escaping(
    tmp_path, designation, monkeypatch
):
    """When the artifact itself cannot be written, no-artifact is unavoidable.

    It must still be caught and named as an I/O failure rather than escaping or
    masquerading as a validation outcome.
    """
    root = _iso_root(tmp_path)

    def exploding(path, document):
        raise OSError("disk full")

    monkeypatch.setattr(codec, "emit_partial", exploding)

    code = runner.run(_argv(root, designation), executor=_invalid_receipt)

    assert code == runner.EXIT_REFUSED
    assert not (root / codec.PARTIAL_BASENAME).exists()
    assert not (root / codec.RECEIPT_BASENAME).exists()


def test_an_io_failure_is_distinguished_from_a_validation_outcome(
    tmp_path, designation, monkeypatch, capsys
):
    root = _iso_root(tmp_path)

    def exploding(path, document):
        raise OSError("disk full")

    monkeypatch.setattr(codec, "emit_partial", exploding)
    runner.run(_argv(root, designation), executor=_invalid_receipt)

    err = capsys.readouterr().err
    assert "invalid receipt" in err, "the validation failure is reported"
    assert "I/O failure" in err, "and the write failure is named distinctly"


def test_an_invalid_receipt_never_writes_both_artifacts(tmp_path, designation):
    root = _iso_root(tmp_path)
    runner.run(_argv(root, designation), executor=_invalid_receipt)
    both = (root / codec.RECEIPT_BASENAME).exists() and (root / codec.PARTIAL_BASENAME).exists()
    assert not both


@pytest.mark.parametrize(
    "mutation",
    [
        {"isolation.is_disposable_instance": False},
        {"teardown.instance_destroyed": False},
        {"capability_dark.advertisement_enabled": True},
        {"ordinary_project_non_effect.grants_minted": 3},
        {"proofs.supervisor_creation_discriminator.outcome": "unobserved"},
    ],
)
def test_every_invalid_receipt_shape_routes_to_a_partial(tmp_path, designation, mutation):
    """Not just the reviewer's case: any validation failure is a begun-run failure."""
    root = _iso_root(tmp_path)

    def executor(request):
        return _mutate(**mutation)

    code = runner.run(_argv(root, designation), executor=executor)
    assert code == runner.EXIT_PARTIAL
    assert (root / codec.PARTIAL_BASENAME).exists()
    assert not (root / codec.RECEIPT_BASENAME).exists()


def test_a_valid_receipt_still_succeeds_unchanged(tmp_path, designation):
    """No regression on the success row: validation before write changes nothing."""
    root = _iso_root(tmp_path)
    code = runner.run(_argv(root, designation), executor=lambda r: minimal_receipt())
    assert code == runner.EXIT_OK
    assert (root / codec.RECEIPT_BASENAME).exists()
    assert not (root / codec.PARTIAL_BASENAME).exists()
    codec.validate_receipt(json.loads((root / codec.RECEIPT_BASENAME).read_bytes()))
