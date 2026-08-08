"""T-A22h — the canonical gate-2 receipt codec, runner, and verification.

Clause (a) is the load-bearing one: the digest is a three-party contract, so this
suite carries its **own independent encoder** that shares no code with the
implementation. If the two ever disagree by a byte, the digest a reviewer
computes and the digest the server computes stop matching, and that is exactly
the failure the receipt exists to make impossible.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from cli_agent_orchestrator.cli.commands import gate2_proof_run as runner
from cli_agent_orchestrator.services import gate2_proof_receipt as codec
from cli_agent_orchestrator.services import gate2_proof_receipt_state as rs

# --------------------------------------------------------------------------
# An independent encoder: written from §5.4's rules, importing none of the
# implementation's serialization.
# --------------------------------------------------------------------------


def independent_encode(document: dict) -> bytes:
    """A second implementation of the canonical form, deliberately unshared."""

    def esc(text: str) -> str:
        buf = ""
        for ch in text:
            if ch == '"':
                buf += '\\"'
            elif ch == "\\":
                buf += "\\\\"
            elif ord(ch) < 32:
                buf += "\\u%04x" % ord(ch)
            else:
                buf += ch
        return buf

    def val(v, order=None):
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, int):
            return "%d" % v
        if isinstance(v, str):
            return '"' + esc(v) + '"'
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(val(x) for x in v) + "]"
        if isinstance(v, dict):
            return obj(v, order)
        raise AssertionError("unencodable")

    orders = {
        "target": codec.TARGET_ORDER,
        "isolation": codec.ISOLATION_ORDER,
        "designation": codec.DESIGNATION_ORDER,
        "capability_dark": codec.CAPABILITY_DARK_ORDER,
        "ordinary_project_non_effect": codec.NON_EFFECT_ORDER,
        "proofs": codec.PROOFS_ORDER,
        "lineage_isolation": codec.PROOF_ENTRY_ORDER,
        "supervisor_creation_discriminator": codec.PROOF_ENTRY_ORDER,
        "teardown": codec.TEARDOWN_ORDER,
        "identities": codec.IDENTITIES_ORDER,
    }

    def obj(d: dict, order) -> str:
        pieces = []
        for k in order:
            pieces.append('"' + esc(k) + '":' + val(d[k], orders.get(k)))
        return "{" + ",".join(pieces) + "}"

    top = codec.RECEIPT_ORDER if document["schema"] == codec.RECEIPT_SCHEMA else codec.PARTIAL_ORDER
    pieces = []
    for k in top:
        v = document[k]
        if k == "steps":
            pieces.append('"steps":[' + ",".join(obj(s, codec.STEP_ORDER) for s in v) + "]")
        else:
            pieces.append('"' + esc(k) + '":' + val(v, orders.get(k)))
    return ("{" + ",".join(pieces) + "}\n").encode("utf-8")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _step(seq: int, command: str, outcome: str, **over) -> dict:
    step = {
        "seq": seq,
        "command": command,
        "outcome": outcome,
        "reason_code": "",
        "reason_detail": "",
        "terminal_created": False,
        "terminal_torn_down": False,
        "epoch_allocated": False,
        "epoch_reused": False,
        "observed_at": "2026-08-08T00:00:00Z",
    }
    step.update(over)
    return step


def receipt(**over) -> dict:
    doc = {
        "schema": codec.RECEIPT_SCHEMA,
        "target": {
            "fork_source_sha256": "a" * 64,
            "deployed_artifact_sha256": "b" * 64,
            "server_start_id": "start-1",
        },
        "isolation": {
            "state_root": "/tmp/cao-proof",
            "proof_project": "cao-gate2-scratch",
            "is_disposable_instance": True,
        },
        "designation": {
            "sha256": "c" * 64,
            "project": "cao-gate2-scratch",
            "opened_at": "2026-08-08T00:00:00Z",
            "closed_at": "2026-08-08T00:10:00Z",
        },
        "capability_dark": {
            "advertisement_enabled": False,
            "provider_tuples_enabled": False,
            "listener_enabled_for_ordinary_projects": False,
        },
        "ordinary_project_non_effect": {
            "ordinary_projects_observed": 0,
            "authority_rows_created": 0,
            "epoch_allocations_created": 0,
            "grants_minted": 0,
            "project_json_files_touched": 0,
        },
        "proofs": {
            "lineage_isolation": {
                "outcome": "proven",
                "evidence_refs": ["ref/lineage/1"],
                "observed_at": "2026-08-08T00:01:00Z",
            },
            "supervisor_creation_discriminator": {
                "outcome": "proven",
                "evidence_refs": ["ref/discriminator/1"],
                "observed_at": "2026-08-08T00:02:00Z",
            },
        },
        "steps": [
            _step(1, "probe managed-origin peer", "supervisor-creation-discriminator-absent"),
            _step(2, "probe UNPROVEN peer", "authority-lineage-unproven"),
            _step(
                3,
                "operator-origin admit",
                "bound",
                terminal_created=True,
                epoch_allocated=True,
            ),
            _step(
                4,
                "phase-C re-verify abort",
                "authority-lineage-unproven",
                terminal_created=True,
                terminal_torn_down=True,
                epoch_allocated=True,
                epoch_reused=False,
            ),
            _step(5, "TCP inertness probe", "no-authority"),
        ],
        "teardown": {
            "instance_destroyed": True,
            "state_root_removed": True,
            "artifacts_deleted": True,
        },
        "identities": {
            "operator_ref": "ops@host",
            "tool_version": "1",
            "spec_head": "8a01802f",
            "reviewed_source_head": "401ef0e",
        },
    }
    doc.update(over)
    return doc


def partial(**over) -> dict:
    doc = {
        "schema": codec.PARTIAL_SCHEMA,
        "target": receipt()["target"],
        "isolation": receipt()["isolation"],
        "observed_fact_names": ["lineage_isolation"],
        "unobserved_fact_names": ["supervisor_creation_discriminator"],
        "refusal_reason": "authority-lineage-unproven",
        "steps": [_step(1, "probe", "authority-lineage-unproven")],
        "teardown": {
            "instance_destroyed": True,
            "state_root_removed": True,
            "artifacts_deleted": False,
        },
        "identities": receipt()["identities"],
    }
    doc.update(over)
    return doc


# --------------------------------------------------------------------------
# (a) Codec: two independent encoders, byte-identical.
# --------------------------------------------------------------------------


def test_two_independent_encoders_agree_byte_for_byte():
    doc = codec.redact(receipt())
    assert codec.canonical_bytes(doc) == independent_encode(doc)


def test_digest_is_over_those_exact_bytes_including_the_trailing_lf():
    doc = codec.redact(receipt())
    payload = codec.canonical_bytes(doc)
    assert payload.endswith(b"}\n")
    assert payload.count(b"\n") == 1
    assert codec.digest(payload) == hashlib.sha256(payload).hexdigest()


def test_order_is_pinned_not_lexicographic():
    payload = codec.canonical_bytes(codec.redact(receipt())).decode()
    positions = [payload.index(f'"{key}"') for key in codec.RECEIPT_ORDER]
    assert positions == sorted(positions)
    assert payload.index('"proofs"') < payload.index('"identities"')
    # Lexicographic order would put capability_dark first.
    assert payload.index('"schema"') < payload.index('"capability_dark"')


def test_control_characters_use_the_long_escape():
    doc = receipt()
    doc["identities"] = {**doc["identities"], "tool_version": "a\nb\tc"}
    payload = codec.canonical_bytes(codec.redact(doc)).decode()
    assert "\\u000a" in payload and "\\u0009" in payload
    assert "\\n" not in payload and "\\t" not in payload


def test_encoder_rejects_a_negative_integer():
    doc = receipt()
    doc["ordinary_project_non_effect"] = {
        **doc["ordinary_project_non_effect"],
        "grants_minted": -1,
    }
    with pytest.raises(codec.ReceiptError):
        codec.canonical_bytes(doc)


# --------------------------------------------------------------------------
# (b) Closed key set and pinned success values.
# --------------------------------------------------------------------------


def test_a_complete_receipt_validates():
    codec.validate_receipt(codec.redact(receipt()))


def test_unknown_top_level_key_is_refused():
    doc = receipt()
    doc["force"] = True
    with pytest.raises(codec.ReceiptError) as e:
        codec.validate_receipt(doc)
    assert "force" in str(e.value)


def test_missing_key_is_refused():
    doc = receipt()
    del doc["teardown"]
    with pytest.raises(codec.ReceiptError):
        codec.validate_receipt(doc)


def test_wrong_schema_is_refused():
    with pytest.raises(codec.ReceiptError):
        codec.validate_receipt(receipt(schema="cao-gate2-proof-receipt-v2"))


def test_null_in_any_required_field_is_refused():
    doc = receipt()
    doc["identities"] = {**doc["identities"], "tool_version": None}
    with pytest.raises(codec.ReceiptError) as e:
        codec.validate_receipt(doc)
    assert "null" in str(e.value)


@pytest.mark.parametrize("name", codec.PROOFS_ORDER)
def test_an_unproven_outcome_cannot_be_a_receipt(name):
    doc = receipt()
    doc["proofs"] = {
        **doc["proofs"],
        name: {**doc["proofs"][name], "outcome": "not-observed"},
    }
    with pytest.raises(codec.ReceiptError) as e:
        codec.validate_receipt(doc)
    assert "proven" in str(e.value)


def test_non_disposable_instance_is_refused_not_left_to_a_reader():
    doc = receipt()
    doc["isolation"] = {**doc["isolation"], "is_disposable_instance": False}
    with pytest.raises(codec.ReceiptError) as e:
        codec.validate_receipt(doc)
    assert "disposable" in str(e.value)


@pytest.mark.parametrize("key", codec.TEARDOWN_ORDER)
def test_every_teardown_member_must_be_true(key):
    doc = receipt()
    doc["teardown"] = {**doc["teardown"], key: False}
    with pytest.raises(codec.ReceiptError) as e:
        codec.validate_receipt(doc)
    assert key in str(e.value)


@pytest.mark.parametrize("key", codec.CAPABILITY_DARK_ORDER)
def test_capability_must_be_dark(key):
    doc = receipt()
    doc["capability_dark"] = {**doc["capability_dark"], key: True}
    with pytest.raises(codec.ReceiptError) as e:
        codec.validate_receipt(doc)
    assert key in str(e.value)


@pytest.mark.parametrize("key", codec.NON_EFFECT_ORDER)
def test_any_ordinary_project_effect_is_refused(key):
    doc = receipt()
    doc["ordinary_project_non_effect"] = {**doc["ordinary_project_non_effect"], key: 1}
    with pytest.raises(codec.ReceiptError) as e:
        codec.validate_receipt(doc)
    assert key in str(e.value)


def test_a_true_boolean_is_not_accepted_where_zero_is_required():
    """`True == 1` in Python; the count check must not be fooled by it."""
    doc = receipt()
    doc["ordinary_project_non_effect"] = {
        **doc["ordinary_project_non_effect"],
        "grants_minted": False,
    }
    with pytest.raises(codec.ReceiptError):
        codec.validate_receipt(doc)


def test_empty_steps_is_refused():
    with pytest.raises(codec.ReceiptError):
        codec.validate_receipt(receipt(steps=[]))


# --------------------------------------------------------------------------
# (c) Redaction cannot erase a required fact.
# --------------------------------------------------------------------------


def test_redaction_replaces_only_the_enumerated_fields():
    original = receipt()
    red = codec.redact(original)

    assert all(step["command"] == codec.REDACTED for step in red["steps"])
    assert red["identities"]["operator_ref"] == codec.REDACTED
    for name in codec.PROOFS_ORDER:
        assert red["proofs"][name]["evidence_refs"] == [codec.REDACTED]

    # Everything else survives untouched.
    assert red["identities"]["tool_version"] == original["identities"]["tool_version"]
    assert red["target"] == original["target"]
    assert red["designation"] == original["designation"]
    assert [s["outcome"] for s in red["steps"]] == [s["outcome"] for s in original["steps"]]


def test_redaction_never_deletes_a_key_or_an_object():
    red = codec.redact(receipt())
    assert set(red) == set(codec.RECEIPT_ORDER)
    assert set(red["proofs"]) == set(codec.PROOFS_ORDER)
    for step in red["steps"]:
        assert set(step) == set(codec.STEP_ORDER)
    assert set(red["identities"]) == set(codec.IDENTITIES_ORDER)


def test_a_redacted_receipt_is_still_a_complete_receipt():
    """The property that makes archiving redacted and closing a gate compatible."""
    red = codec.redact(receipt())
    codec.validate_receipt(red)
    assert all(red["proofs"][n]["outcome"] == "proven" for n in codec.PROOFS_ORDER)
    assert all(v is False for v in red["capability_dark"].values())
    assert all(v == 0 for v in red["ordinary_project_non_effect"].values())


def test_redaction_precedes_the_digest_so_only_one_byte_sequence_exists(tmp_path):
    path = tmp_path / codec.RECEIPT_BASENAME
    payload, sha = codec.emit_receipt(path, receipt())
    assert path.read_bytes() == payload
    assert sha == hashlib.sha256(payload).hexdigest()
    assert codec.REDACTED in payload.decode()


# --------------------------------------------------------------------------
# Atomic write and mode.
# --------------------------------------------------------------------------


def test_receipt_is_written_0600_and_atomically(tmp_path):
    path = tmp_path / codec.RECEIPT_BASENAME
    codec.emit_receipt(path, receipt())
    assert stat.S_IMODE(path.lstat().st_mode) == 0o600
    # No temp file is left observable beside it.
    assert [p.name for p in tmp_path.iterdir()] == [codec.RECEIPT_BASENAME]


def test_a_failed_validation_writes_nothing(tmp_path):
    path = tmp_path / codec.RECEIPT_BASENAME
    doc = receipt()
    doc["teardown"] = {**doc["teardown"], "state_root_removed": False}
    with pytest.raises(codec.ReceiptError):
        codec.emit_receipt(path, doc)
    assert not path.exists()


# --------------------------------------------------------------------------
# (e) The partial cannot attest and cannot borrow success values.
# --------------------------------------------------------------------------


def test_partial_has_no_proofs_key():
    doc = partial()
    assert "proofs" not in doc
    codec.validate_partial(codec.redact(doc))


def test_a_partial_carrying_proofs_is_refused():
    with pytest.raises(codec.ReceiptError) as e:
        codec.validate_partial(partial(proofs=receipt()["proofs"]))
    assert "no 'proofs' key" in str(e.value)


def test_partial_teardown_records_what_happened_not_an_asserted_success():
    doc = partial()
    assert doc["teardown"]["artifacts_deleted"] is False
    codec.validate_partial(codec.redact(doc))  # accepted as-is


def test_a_partial_asserting_success_values_is_still_not_a_receipt():
    doc = partial(
        teardown={
            "instance_destroyed": True,
            "state_root_removed": True,
            "artifacts_deleted": True,
        }
    )
    codec.validate_partial(codec.redact(doc))
    with pytest.raises(codec.ReceiptError):
        codec.validate_receipt(doc)


def test_partial_and_receipt_have_different_schema_tags():
    assert codec.PARTIAL_SCHEMA != codec.RECEIPT_SCHEMA
    assert codec.PARTIAL_BASENAME != codec.RECEIPT_BASENAME


# --------------------------------------------------------------------------
# (d) Unconditional verification at server start.
# --------------------------------------------------------------------------


@pytest.fixture
def state_root(monkeypatch, tmp_path):
    import cli_agent_orchestrator.constants as constants

    root = tmp_path / "state"
    root.mkdir()
    monkeypatch.setattr(constants, "CAO_HOME_DIR", root)
    return root


def _install(root: Path, *, receipt_bytes: bytes | None, declared: str) -> None:
    if receipt_bytes is not None:
        codec.write_atomically(root / codec.RECEIPT_BASENAME, receipt_bytes)
    rs.write_receipt_state_for_proof_run(root / rs.RECEIPT_STATE_BASENAME, declared)


def test_matching_receipt_and_state_load(state_root):
    payload, sha = codec.emit_receipt(state_root / codec.RECEIPT_BASENAME, receipt())
    rs.write_receipt_state_for_proof_run(state_root / rs.RECEIPT_STATE_BASENAME, sha)
    got = rs.load_receipt_state()
    assert got is not None and got.records_both_proofs is True


def test_receipt_state_without_its_receipt_refuses_start(state_root):
    """The withdrawn verify-if-present branch would have accepted this."""
    rs.write_receipt_state_for_proof_run(state_root / rs.RECEIPT_STATE_BASENAME, "a" * 64)
    with pytest.raises(rs.ReceiptStateError) as e:
        rs.load_receipt_state()
    assert "absent" in str(e.value)


def test_digest_mismatch_refuses_start(state_root):
    payload, _ = codec.emit_receipt(state_root / codec.RECEIPT_BASENAME, receipt())
    rs.write_receipt_state_for_proof_run(state_root / rs.RECEIPT_STATE_BASENAME, "d" * 64)
    with pytest.raises(rs.ReceiptStateError) as e:
        rs.load_receipt_state()
    assert "hashes to" in str(e.value)


def test_a_partial_cannot_be_a_receipt_state_referent(state_root):
    payload, sha = codec.emit_partial(state_root / codec.RECEIPT_BASENAME, partial())
    rs.write_receipt_state_for_proof_run(state_root / rs.RECEIPT_STATE_BASENAME, sha)
    with pytest.raises(rs.ReceiptStateError) as e:
        rs.load_receipt_state()
    assert "never attest" in str(e.value) or "may be a receipt-state referent" in str(e.value)


def test_unreadable_receipt_refuses_start(state_root):
    (state_root / codec.RECEIPT_BASENAME).mkdir()
    rs.write_receipt_state_for_proof_run(state_root / rs.RECEIPT_STATE_BASENAME, "a" * 64)
    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state()


def test_no_receipt_state_means_absence_of_the_receipt_is_not_an_error(state_root):
    assert rs.load_receipt_state() is None
    assert not (state_root / codec.RECEIPT_BASENAME).exists()


# --------------------------------------------------------------------------
# (f) Runner contract.
# --------------------------------------------------------------------------


def _iso_root(tmp_path) -> Path:
    root = tmp_path / "proof-root"
    root.mkdir()
    return root


def test_relative_state_root_is_refused(tmp_path):
    with pytest.raises(runner.ProofRunRefused) as e:
        runner.validate_request("relative/root", "p", str(tmp_path))
    assert "relative" in str(e.value)


def test_the_default_state_root_is_refused(tmp_path):
    designation = tmp_path / "d.json"
    designation.write_text("{}")
    default = str(Path.home() / runner.DEFAULT_ROOT_SUFFIX)
    with pytest.raises(runner.ProofRunRefused) as e:
        runner.validate_request(default, "p", str(designation))
    assert "default root" in str(e.value)


def test_a_root_holding_another_projects_binding_is_refused(tmp_path):
    root = _iso_root(tmp_path)
    (root / "project-state-bindings").mkdir()
    (root / "project-state-bindings" / "real-project.json").write_text("{}")
    designation = tmp_path / "d.json"
    designation.write_text("{}")
    with pytest.raises(runner.ProofRunRefused) as e:
        runner.validate_request(str(root), "cao-gate2-scratch", str(designation))
    assert "live root is refused" in str(e.value)


def test_a_root_holding_a_database_is_refused(tmp_path):
    root = _iso_root(tmp_path)
    (root / "cao.db").write_text("")
    designation = tmp_path / "d.json"
    designation.write_text("{}")
    with pytest.raises(runner.ProofRunRefused):
        runner.validate_request(str(root), "p", str(designation))


def test_a_missing_designation_is_refused(tmp_path):
    root = _iso_root(tmp_path)
    with pytest.raises(runner.ProofRunRefused) as e:
        runner.validate_request(str(root), "p", str(tmp_path / "nope.json"))
    assert "designation" in str(e.value)


@pytest.mark.parametrize("project", ["", "  ", " padded "])
def test_an_inexact_project_is_refused(tmp_path, project):
    root = _iso_root(tmp_path)
    with pytest.raises(runner.ProofRunRefused):
        runner.validate_request(str(root), project, str(tmp_path))


def test_a_valid_request_resolves(tmp_path):
    root = _iso_root(tmp_path)
    designation = tmp_path / "d.json"
    designation.write_text("{}")
    got = runner.validate_request(str(root), "cao-gate2-scratch", str(designation))
    assert got.project == "cao-gate2-scratch"
    assert got.state_root.is_absolute()


@pytest.mark.parametrize("basename", [codec.RECEIPT_BASENAME, codec.PARTIAL_BASENAME])
def test_an_existing_artifact_is_a_collision_never_an_overwrite(tmp_path, basename):
    root = _iso_root(tmp_path)
    (root / basename).write_text("prior")
    with pytest.raises(runner.ProofRunRefused) as e:
        runner.refuse_on_existing_artifacts(root)
    assert "never overwrites" in str(e.value)


def _argv(root: Path, designation: Path) -> list:
    return [
        "--state-root",
        str(root),
        "--project",
        "cao-gate2-scratch",
        "--designation",
        str(designation),
    ]


def test_exit_zero_iff_a_receipt_is_written(tmp_path):
    root = _iso_root(tmp_path)
    designation = tmp_path / "d.json"
    designation.write_text("{}")

    code = runner.run(_argv(root, designation), executor=lambda request: receipt())

    assert code == runner.EXIT_OK
    assert (root / codec.RECEIPT_BASENAME).exists()
    assert not (root / codec.PARTIAL_BASENAME).exists(), "never both artifacts"


def test_nonzero_iff_a_partial_is_written(tmp_path):
    root = _iso_root(tmp_path)
    designation = tmp_path / "d.json"
    designation.write_text("{}")

    code = runner.run(_argv(root, designation), executor=lambda request: partial())

    assert code != runner.EXIT_OK
    assert (root / codec.PARTIAL_BASENAME).exists()
    assert not (root / codec.RECEIPT_BASENAME).exists(), "never both artifacts"


def test_a_precondition_refusal_writes_neither_artifact(tmp_path):
    root = _iso_root(tmp_path)
    (root / "cao.db").write_text("")
    designation = tmp_path / "d.json"
    designation.write_text("{}")

    code = runner.run(_argv(root, designation), executor=lambda request: receipt())

    assert code == runner.EXIT_REFUSED
    assert not (root / codec.RECEIPT_BASENAME).exists()
    assert not (root / codec.PARTIAL_BASENAME).exists()


def test_the_default_executor_refuses_rather_than_driving_a_real_server(tmp_path):
    """Live execution is ops-only; this build ships the contract, not a driver."""
    root = _iso_root(tmp_path)
    designation = tmp_path / "d.json"
    designation.write_text("{}")

    code = runner.run(_argv(root, designation))

    assert code == runner.EXIT_REFUSED
    assert not (root / codec.RECEIPT_BASENAME).exists()
    assert not (root / codec.PARTIAL_BASENAME).exists()


def test_the_runner_is_reachable_from_no_request_path():
    """No API route, and no server-settable selector, can invoke it."""
    import inspect

    from cli_agent_orchestrator.api.main import app

    for route in app.routes:
        assert "gate2" not in getattr(route, "path", "").lower()
        assert "proof" not in getattr(route, "path", "").lower()

    source = inspect.getsource(runner)
    assert "os.environ" not in source, "no environment selector"
    assert "getenv" not in source

    from cli_agent_orchestrator.services import supervisor_create_channel as channel

    assert "gate2_proof_run" not in inspect.getsource(channel)


def test_the_runner_is_shipped_as_its_own_entry_point():
    """Deliberately not a `cao` subcommand: it must not sit in the server's CLI tree."""
    text = Path("pyproject.toml").read_text()
    assert (
        '"cao-gate2-proof-run" = '
        '"cli_agent_orchestrator.cli.commands.gate2_proof_run:main"' in text
    )
