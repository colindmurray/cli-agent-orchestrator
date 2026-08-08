"""The witness binding: the fork looks, and is never told.

Two properties matter most here. `ABSENT` — the only trit that admits a `(1, 1)`
mint — requires a readable directory demonstrably lacking `project.json`, never a
missing or unreadable path. And a malformed binding degrades **that project only**
to `UNKNOWN` rather than refusing server start, which is the smaller blast radius
of the three state-root artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat

import pytest

from cli_agent_orchestrator.services import project_state_binding as psb

PROJECT = "cao-proj"


@pytest.fixture
def state_root(monkeypatch, tmp_path):
    import cli_agent_orchestrator.constants as constants

    root = tmp_path / "state"
    root.mkdir()
    monkeypatch.setattr(constants, "CAO_HOME_DIR", root)
    return root


def _write_raw(path, payload: object, mode: int = 0o600) -> None:
    raw = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
    try:
        os.write(descriptor, raw)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


# --------------------------------------------------------------------------
# binding_recorded: existence only, and that is deliberate.
# --------------------------------------------------------------------------


def test_no_binding_is_not_recorded(state_root):
    assert psb.binding_recorded(PROJECT) is False


def test_empty_project_is_not_recorded(state_root):
    assert psb.binding_recorded("") is False


def test_a_malformed_binding_still_counts_as_recorded(state_root, tmp_path):
    """Recorded and well-formed are separate questions, on purpose.

    A malformed binding must end the bypass (the gate-6 mechanism has landed)
    *and* observe UNKNOWN (so authority still refuses). Conflating the two would
    let a bad file silently reinstate the bypass and look healthy.
    """
    _write_raw(psb.binding_path(PROJECT), b"not json")
    assert psb.binding_recorded(PROJECT) is True
    assert psb.observe_witness(PROJECT).witness == "unknown"


# --------------------------------------------------------------------------
# ABSENT requires positive proof.
# --------------------------------------------------------------------------


def test_readable_dir_without_project_json_is_absent(state_root, tmp_path):
    project_dir = tmp_path / "conductor-proj"
    project_dir.mkdir()
    psb.write_binding_for_project(psb.binding_path(PROJECT), PROJECT, str(project_dir))

    got = psb.observe_witness(PROJECT)
    assert got.witness == "absent"
    assert got.project_json_observed_absent is True
    assert got.project_json_sha256 is None
    assert got.source_path.endswith("project.json")


def test_existing_project_json_is_present_with_its_digest(state_root, tmp_path):
    project_dir = tmp_path / "conductor-proj"
    project_dir.mkdir()
    payload = b'{"repo_root": "/x", "project_incarnation": 7}'
    (project_dir / psb.PROJECT_JSON_BASENAME).write_bytes(payload)
    psb.write_binding_for_project(psb.binding_path(PROJECT), PROJECT, str(project_dir))

    got = psb.observe_witness(PROJECT)
    assert got.witness == "present"
    assert got.project_json_sha256 == hashlib.sha256(payload).hexdigest()
    assert got.project_json_observed_absent is False


@pytest.mark.parametrize(
    "detail,setup",
    [
        ("binding-absent", "none"),
        ("project-state-dir-unreadable", "missing-dir"),
        ("project-state-dir-not-a-directory", "file-not-dir"),
    ],
)
def test_anything_short_of_positive_proof_is_unknown(state_root, tmp_path, detail, setup):
    """Absence of evidence is never evidence of absence."""
    if setup == "missing-dir":
        psb.write_binding_for_project(psb.binding_path(PROJECT), PROJECT, str(tmp_path / "nope"))
    elif setup == "file-not-dir":
        target = tmp_path / "a-file"
        target.write_text("x")
        psb.write_binding_for_project(psb.binding_path(PROJECT), PROJECT, str(target))

    got = psb.observe_witness(PROJECT)
    assert got.witness == "unknown"
    assert got.detail == detail
    assert got.project_json_observed_absent is False


def test_unreadable_directory_is_unknown_not_absent(state_root, tmp_path):
    project_dir = tmp_path / "locked"
    project_dir.mkdir()
    psb.write_binding_for_project(psb.binding_path(PROJECT), PROJECT, str(project_dir))
    os.chmod(project_dir, 0o000)
    try:
        got = psb.observe_witness(PROJECT)
        assert got.witness == "unknown"
    finally:
        os.chmod(project_dir, 0o700)


# --------------------------------------------------------------------------
# Malformed bindings: UNKNOWN for that project only, never a startup refusal.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": psb.BINDING_SCHEMA, "project": PROJECT},  # missing key
        {"schema": psb.BINDING_SCHEMA, "project": PROJECT, "project_state_dir": "/x", "extra": 1},
        {"schema": "other-v1", "project": PROJECT, "project_state_dir": "/x"},
        {"schema": psb.BINDING_SCHEMA, "project": PROJECT, "project_state_dir": "relative/x"},
        {"schema": psb.BINDING_SCHEMA, "project": PROJECT, "project_state_dir": "  "},
        {"schema": psb.BINDING_SCHEMA, "project": PROJECT, "project_state_dir": " /padded "},
        {"schema": psb.BINDING_SCHEMA, "project": PROJECT, "project_state_dir": ["/a", "/b"]},
        {"schema": psb.BINDING_SCHEMA, "project": "other-project", "project_state_dir": "/x"},
    ],
)
def test_malformed_binding_yields_unknown_and_does_not_raise(state_root, payload):
    _write_raw(psb.binding_path(PROJECT), payload)
    got = psb.observe_witness(PROJECT)
    assert got.witness == "unknown"


def test_wrong_mode_binding_yields_unknown(state_root, tmp_path):
    project_dir = tmp_path / "conductor-proj"
    project_dir.mkdir()
    _write_raw(
        psb.binding_path(PROJECT),
        {
            "schema": psb.BINDING_SCHEMA,
            "project": PROJECT,
            "project_state_dir": str(project_dir),
        },
        mode=0o644,
    )
    assert psb.observe_witness(PROJECT).witness == "unknown"


def test_one_bad_binding_does_not_affect_another_project(state_root, tmp_path):
    """The per-project blast radius, asserted rather than assumed."""
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    psb.write_binding_for_project(psb.binding_path("good-proj"), "good-proj", str(good_dir))
    _write_raw(psb.binding_path("bad-proj"), b"{{{")

    assert psb.observe_witness("bad-proj").witness == "unknown"
    assert psb.observe_witness("good-proj").witness == "absent"


def test_a_binding_cannot_answer_for_a_different_project(state_root, tmp_path):
    """Filename and declared project must agree."""
    project_dir = tmp_path / "conductor-proj"
    project_dir.mkdir()
    _write_raw(
        psb.binding_path("victim"),
        {
            "schema": psb.BINDING_SCHEMA,
            "project": "attacker",
            "project_state_dir": str(project_dir),
        },
    )
    assert psb.observe_witness("victim").witness == "unknown"


# --------------------------------------------------------------------------
# Provenance is what an auditor reads.
# --------------------------------------------------------------------------


def test_provenance_carries_trit_source_and_basis(state_root, tmp_path):
    project_dir = tmp_path / "conductor-proj"
    project_dir.mkdir()
    payload = b'{"x": 1}'
    (project_dir / psb.PROJECT_JSON_BASENAME).write_bytes(payload)
    psb.write_binding_for_project(psb.binding_path(PROJECT), PROJECT, str(project_dir))

    provenance = psb.observe_witness(PROJECT).as_provenance()
    assert provenance["witness"] == "present"
    assert provenance["witness_project_json_sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["witness_source_path"].endswith("project.json")
    assert provenance["witness_project_json_observed_absent"] is False
    assert provenance["witness_detail"] == "project-json-observed"


# --------------------------------------------------------------------------
# No endpoint, no selector.
# --------------------------------------------------------------------------


def test_no_api_or_channel_module_imports_the_binding_writer():
    import inspect

    from cli_agent_orchestrator.api import main as api_main
    from cli_agent_orchestrator.services import supervisor_create_channel as channel

    for module in (api_main, channel):
        assert "write_binding_for_project" not in inspect.getsource(module)


def test_no_http_route_exposes_bindings():
    from cli_agent_orchestrator.api.main import app

    for route in app.routes:
        assert "binding" not in getattr(route, "path", "").lower()


def test_binding_writer_produces_0600(state_root, tmp_path):
    psb.write_binding_for_project(psb.binding_path(PROJECT), PROJECT, str(tmp_path))
    assert stat.S_IMODE(psb.binding_path(PROJECT).lstat().st_mode) == 0o600
