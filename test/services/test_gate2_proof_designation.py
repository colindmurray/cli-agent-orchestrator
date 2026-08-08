"""The gate-2 proof designation: absence is safe, anything malformed refuses.

Every row here is a negative case except two, and that ratio is the point. The
designation decides which project's authority pipeline may run before the gate
that authorizes it has closed, so the only acceptable failure direction is
"refuse and say why".
"""

from __future__ import annotations

import hashlib
import json
import os
import stat

import pytest

from cli_agent_orchestrator.services import gate2_proof_designation as gate2


def _write(path, payload: object, mode: int = 0o600) -> None:
    raw = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
    try:
        os.write(descriptor, raw)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


# --------------------------------------------------------------------------
# Absence is the default and the safe state.
# --------------------------------------------------------------------------


def test_absent_designation_is_none_not_an_error(tmp_path):
    """No designation means the pipeline runs nowhere — the ordinary case."""
    assert gate2.load_designation(tmp_path / gate2.DESIGNATION_BASENAME) is None


def test_designation_path_follows_the_state_root(monkeypatch, tmp_path):
    import cli_agent_orchestrator.constants as constants

    monkeypatch.setattr(constants, "CAO_HOME_DIR", tmp_path)
    assert gate2.designation_path() == tmp_path / gate2.DESIGNATION_BASENAME


# --------------------------------------------------------------------------
# The happy path, and the digest a receipt must carry.
# --------------------------------------------------------------------------


def test_exact_single_project_loads_with_its_digest(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    payload = {"schema": gate2.DESIGNATION_SCHEMA, "project": "cao-gate2-scratch"}
    _write(path, payload)

    got = gate2.load_designation(path)
    assert got is not None
    assert got.project == "cao-gate2-scratch"
    assert got.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert got.path == str(path)


def test_write_helper_produces_a_loadable_0600_designation(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    written = gate2.write_designation_for_proof_run(path, "proof-project")
    assert stat.S_IMODE(path.lstat().st_mode) == 0o600
    loaded = gate2.load_designation(path)
    assert loaded is not None
    assert loaded.project == "proof-project"
    assert loaded.sha256 == written.sha256


# --------------------------------------------------------------------------
# Mode: it must not be readable or writable beyond its owner.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [0o644, 0o666, 0o604, 0o640, 0o700])
def test_wrong_mode_is_refused(tmp_path, mode):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, {"schema": gate2.DESIGNATION_SCHEMA, "project": "p"}, mode=mode)
    with pytest.raises(gate2.DesignationError) as excinfo:
        gate2.load_designation(path)
    assert "0600" in str(excinfo.value)


# --------------------------------------------------------------------------
# Malformed, ambiguous, and multi-project designations are refused, never
# interpreted.
# --------------------------------------------------------------------------


def test_non_json_is_refused(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, b"not json at all")
    with pytest.raises(gate2.DesignationError):
        gate2.load_designation(path)


def test_json_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, ["proj-a", "proj-b"])
    with pytest.raises(gate2.DesignationError):
        gate2.load_designation(path)


def test_missing_project_key_is_refused(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, {"schema": gate2.DESIGNATION_SCHEMA})
    with pytest.raises(gate2.DesignationError):
        gate2.load_designation(path)


def test_unexpected_key_is_refused_not_ignored(tmp_path):
    """An extra key is an operator saying something; refusing beats guessing."""
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(
        path,
        {"schema": gate2.DESIGNATION_SCHEMA, "project": "p", "also_allow": "q"},
    )
    with pytest.raises(gate2.DesignationError) as excinfo:
        gate2.load_designation(path)
    assert "also_allow" in str(excinfo.value)


def test_wrong_schema_is_refused(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, {"schema": "something-else-v9", "project": "p"})
    with pytest.raises(gate2.DesignationError):
        gate2.load_designation(path)


def test_multi_project_list_is_refused(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, {"schema": gate2.DESIGNATION_SCHEMA, "project": ["a", "b"]})
    with pytest.raises(gate2.DesignationError) as excinfo:
        gate2.load_designation(path)
    assert "one project" in str(excinfo.value)


def test_comma_joined_projects_are_refused(tmp_path):
    """The other shape an operator reaches for to name two projects."""
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, {"schema": gate2.DESIGNATION_SCHEMA, "project": "a,b"})
    with pytest.raises(gate2.DesignationError) as excinfo:
        gate2.load_designation(path)
    assert "several projects" in str(excinfo.value)


def test_newline_separated_projects_are_refused(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, {"schema": gate2.DESIGNATION_SCHEMA, "project": "a\nb"})
    with pytest.raises(gate2.DesignationError):
        gate2.load_designation(path)


def test_empty_project_is_refused(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, {"schema": gate2.DESIGNATION_SCHEMA, "project": "   "})
    with pytest.raises(gate2.DesignationError):
        gate2.load_designation(path)


def test_whitespace_padded_project_is_refused_not_trimmed(tmp_path):
    """Trimming would silently designate a different string than was written."""
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, {"schema": gate2.DESIGNATION_SCHEMA, "project": " padded "})
    with pytest.raises(gate2.DesignationError) as excinfo:
        gate2.load_designation(path)
    assert "whitespace" in str(excinfo.value)


def test_non_string_project_is_refused(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    _write(path, {"schema": gate2.DESIGNATION_SCHEMA, "project": 7})
    with pytest.raises(gate2.DesignationError):
        gate2.load_designation(path)


def test_directory_in_place_of_the_file_is_refused(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    path.mkdir()
    with pytest.raises(gate2.DesignationError) as excinfo:
        gate2.load_designation(path)
    assert "regular file" in str(excinfo.value)


# --------------------------------------------------------------------------
# There is no endpoint, and no request-reachable reader.
# --------------------------------------------------------------------------


def test_no_api_or_channel_module_imports_the_designation_writer():
    """The write helper must not be reachable from a request path.

    Asserted structurally: if the HTTP app or the channel ever imported the
    writer, a request could be one refactor away from creating a designation.
    """
    import inspect

    from cli_agent_orchestrator.api import main as api_main
    from cli_agent_orchestrator.services import supervisor_create_channel as channel

    for module in (api_main, channel):
        source = inspect.getsource(module)
        assert "write_designation_for_proof_run" not in source


def test_designation_is_not_exposed_on_any_http_route():
    from cli_agent_orchestrator.api.main import app

    paths = {getattr(route, "path", "") for route in app.routes}
    for path in paths:
        assert "designation" not in path.lower()
        assert "gate2" not in path.lower()


# --------------------------------------------------------------------------
# Receipt fields: auditable either way.
# --------------------------------------------------------------------------


def test_receipt_fields_record_digest_project_and_window(tmp_path):
    path = tmp_path / gate2.DESIGNATION_BASENAME
    designation = gate2.write_designation_for_proof_run(path, "scratch")
    fields = gate2.receipt_fields(
        designation, window_opened_at="2026-08-08T00:00:00Z", window_closed_at=None
    )
    assert fields["gate2_proof_designation_present"] is True
    assert fields["gate2_proof_designation_sha256"] == designation.sha256
    assert fields["gate2_proof_designation_project"] == "scratch"
    assert fields["gate2_proof_designation_window_opened_at"] == "2026-08-08T00:00:00Z"


def test_receipt_fields_state_absence_positively():
    """A receipt says "no designation" rather than omitting the fact."""
    fields = gate2.receipt_fields(None)
    assert fields["gate2_proof_designation_present"] is False
    assert fields["gate2_proof_designation_sha256"] is None
    assert fields["gate2_proof_designation_project"] is None


# --------------------------------------------------------------------------
# Disposable-instance practice: the residual is removed by a separate state root.
# --------------------------------------------------------------------------


def test_disposable_instance_isolates_the_designation(monkeypatch, tmp_path):
    """With its own CAO_STATE_ROOT the designation cannot sit beside real state.

    This is the recommended practice that removes the declared same-UID
    residual entirely, exercised rather than only documented.
    """
    import cli_agent_orchestrator.constants as constants

    real_root = tmp_path / "real"
    disposable_root = tmp_path / "disposable"
    real_root.mkdir()
    disposable_root.mkdir()

    gate2.write_designation_for_proof_run(disposable_root / gate2.DESIGNATION_BASENAME, "scratch")

    monkeypatch.setattr(constants, "CAO_HOME_DIR", real_root)
    assert gate2.load_designation() is None, "the real state root must see no designation"

    monkeypatch.setattr(constants, "CAO_HOME_DIR", disposable_root)
    loaded = gate2.load_designation()
    assert loaded is not None and loaded.project == "scratch"
