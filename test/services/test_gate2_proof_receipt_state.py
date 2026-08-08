"""The gate-2 receipt state: presence must not be able to stand in for proof.

Three properties carry that weight and each has rows here: the proof list must
name both proofs, the digest must bind to the canonical receipt, and the key set
is exact. Everything malformed refuses server start, which is the same
server-scoped blast radius a malformed designation has — deliberately unlike a
malformed per-project binding.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat

import pytest

from cli_agent_orchestrator.services import gate2_proof_receipt_state as rs


def _write(path, payload: object, mode: int = 0o600) -> None:
    raw = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
    try:
        os.write(descriptor, raw)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def _valid(digest: str = "a" * 64) -> dict:
    return {
        "schema": rs.RECEIPT_STATE_SCHEMA,
        "receipt_sha256": digest,
        "proofs_recorded": list(rs.REQUIRED_PROOFS),
    }


# --------------------------------------------------------------------------
# Absence keeps the bypass running.
# --------------------------------------------------------------------------


def test_absent_receipt_state_is_none(tmp_path):
    assert rs.load_receipt_state(tmp_path / rs.RECEIPT_STATE_BASENAME) is None


def test_paths_follow_the_state_root(monkeypatch, tmp_path):
    import cli_agent_orchestrator.constants as constants

    monkeypatch.setattr(constants, "CAO_HOME_DIR", tmp_path)
    assert rs.receipt_state_path() == tmp_path / rs.RECEIPT_STATE_BASENAME
    assert rs.canonical_receipt_path() == tmp_path / rs.CANONICAL_RECEIPT_BASENAME


# --------------------------------------------------------------------------
# The valid shape.
# --------------------------------------------------------------------------


def test_valid_state_records_both_proofs(tmp_path):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, _valid())
    got = rs.load_receipt_state(path)
    assert got is not None
    assert got.records_both_proofs is True
    assert got.receipt_sha256 == "a" * 64
    assert got.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_proof_order_does_not_matter(tmp_path):
    """Both named is the requirement; their order is not a wire contract."""
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, {**_valid(), "proofs_recorded": list(reversed(rs.REQUIRED_PROOFS))})
    assert rs.load_receipt_state(path).records_both_proofs is True


def test_write_helper_round_trips(tmp_path):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    written = rs.write_receipt_state_for_proof_run(path, "b" * 64)
    assert stat.S_IMODE(path.lstat().st_mode) == 0o600
    loaded = rs.load_receipt_state(path)
    assert loaded.sha256 == written.sha256
    assert loaded.records_both_proofs is True


# --------------------------------------------------------------------------
# The proof list is what stops presence from standing in for proof.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proofs",
    [
        [],
        ["lineage-isolation"],
        ["supervisor-creation-discriminator"],
        ["lineage-isolation", "something-else"],
        ["lineage-isolation", "supervisor-creation-discriminator", "extra"],
        ["lineage-isolation", "lineage-isolation"],
    ],
)
def test_a_partial_or_padded_proof_list_is_refused(tmp_path, proofs):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, {**_valid(), "proofs_recorded": proofs})
    with pytest.raises(rs.ReceiptStateError) as excinfo:
        rs.load_receipt_state(path)
    assert "must name exactly" in str(excinfo.value)


def test_proofs_must_be_a_list_of_strings(tmp_path):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, {**_valid(), "proofs_recorded": "lineage-isolation"})
    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state(path)


# --------------------------------------------------------------------------
# The digest binds to the canonical receipt.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "digest",
    [
        "A" * 64,  # uppercase hex is refused, not normalized
        "a" * 63,  # too short
        "a" * 65,  # too long
        "z" * 64,  # not hex
        "",
        "0x" + "a" * 62,
    ],
)
def test_a_non_canonical_digest_is_refused(tmp_path, digest):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, {**_valid(), "receipt_sha256": digest})
    with pytest.raises(rs.ReceiptStateError) as excinfo:
        rs.load_receipt_state(path)
    assert "64" in str(excinfo.value) or "hex" in str(excinfo.value)


def test_digest_matching_the_canonical_receipt_is_accepted(monkeypatch, tmp_path):
    import cli_agent_orchestrator.constants as constants

    monkeypatch.setattr(constants, "CAO_HOME_DIR", tmp_path)
    receipt = tmp_path / rs.CANONICAL_RECEIPT_BASENAME
    receipt.write_bytes(b'{"gate": 2}')
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    _write(tmp_path / rs.RECEIPT_STATE_BASENAME, _valid(digest))

    got = rs.load_receipt_state()
    assert got is not None and got.receipt_sha256 == digest


def test_digest_mismatching_the_canonical_receipt_refuses(monkeypatch, tmp_path):
    """A pointer naming the wrong evidence is worse than no pointer."""
    import cli_agent_orchestrator.constants as constants

    monkeypatch.setattr(constants, "CAO_HOME_DIR", tmp_path)
    (tmp_path / rs.CANONICAL_RECEIPT_BASENAME).write_bytes(b'{"gate": 2}')
    _write(tmp_path / rs.RECEIPT_STATE_BASENAME, _valid("c" * 64))

    with pytest.raises(rs.ReceiptStateError) as excinfo:
        rs.load_receipt_state()
    assert "hashes to" in str(excinfo.value)


# --------------------------------------------------------------------------
# Exact key set, mode, and shape.
# --------------------------------------------------------------------------


def test_missing_key_is_refused(tmp_path):
    payload = _valid()
    del payload["proofs_recorded"]
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, payload)
    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state(path)


def test_unknown_key_is_refused_not_ignored(tmp_path):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, {**_valid(), "force_enable": True})
    with pytest.raises(rs.ReceiptStateError) as excinfo:
        rs.load_receipt_state(path)
    assert "force_enable" in str(excinfo.value)


def test_wrong_schema_is_refused(tmp_path):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, {**_valid(), "schema": "cao-gate2-proof-receipt-state-v2"})
    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state(path)


@pytest.mark.parametrize("mode", [0o644, 0o666, 0o640, 0o700])
def test_wrong_mode_is_refused(tmp_path, mode):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, _valid(), mode=mode)
    with pytest.raises(rs.ReceiptStateError) as excinfo:
        rs.load_receipt_state(path)
    assert "0600" in str(excinfo.value)


def test_non_json_is_refused(tmp_path):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, b"receipt: yes")
    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state(path)


def test_json_array_is_refused(tmp_path):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    _write(path, [_valid()])
    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state(path)


def test_directory_in_place_of_the_file_is_refused(tmp_path):
    path = tmp_path / rs.RECEIPT_STATE_BASENAME
    path.mkdir()
    with pytest.raises(rs.ReceiptStateError):
        rs.load_receipt_state(path)


# --------------------------------------------------------------------------
# No endpoint, no selector, no readback.
# --------------------------------------------------------------------------


def test_no_api_or_channel_module_imports_the_writer():
    import inspect

    from cli_agent_orchestrator.api import main as api_main
    from cli_agent_orchestrator.services import supervisor_create_channel as channel

    for module in (api_main, channel):
        assert "write_receipt_state_for_proof_run" not in inspect.getsource(module)


def test_no_http_route_exposes_the_receipt_state():
    from cli_agent_orchestrator.api.main import app

    for route in app.routes:
        path = getattr(route, "path", "").lower()
        assert "receipt-state" not in path
        assert "receipt_state" not in path


def test_no_environment_variable_selects_the_receipt_state():
    """Interval selection comes from durable state, never an env selector."""
    import inspect

    source = inspect.getsource(rs)
    assert "os.environ" not in source
    assert "getenv" not in source


# --------------------------------------------------------------------------
# Audit fields.
# --------------------------------------------------------------------------


def test_receipt_fields_state_absence_positively():
    fields = rs.receipt_fields(None)
    assert fields["gate2_receipt_state_present"] is False
    assert fields["gate2_receipt_sha256"] is None
    assert fields["gate2_proofs_recorded"] is None


def test_receipt_fields_record_both_digests_and_the_proof_list(tmp_path):
    state = rs.write_receipt_state_for_proof_run(tmp_path / rs.RECEIPT_STATE_BASENAME, "d" * 64)
    fields = rs.receipt_fields(state)
    assert fields["gate2_receipt_state_present"] is True
    assert fields["gate2_receipt_sha256"] == "d" * 64
    assert fields["gate2_receipt_state_sha256"] == state.sha256
    assert fields["gate2_proofs_recorded"] == list(rs.REQUIRED_PROOFS)
