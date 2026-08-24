"""Durable Muse provider-version banner through launch, teardown, restore.

cond-0573 MUSE-VERSION: muse_cli workers were non-resurrectable because the
durable executable fact recorded only ``{path, sha256}``, so the Muse
profile-carrier revalidation typed "the Muse restore contract lacks the
wrapper path/full version banner" and refused every teardown contract.  This
suite pins the additive-optional threading: the full ``[wrapper, --version]``
banner is recorded durably at launch, teardown publishes it inside the
executable fact value, and exact restore revalidates the carrier — while a row
that predates the capture (or a non-Muse row) keeps exactly today's behaviour
and refusal.

No provider, tmux, or network I/O.  The wrapper/inner pair is a real
digest-pinned filesystem fixture (the resume gate re-hashes the bytes and the
carrier gate reads ``.muse-version`` next to the wrapper), and the roster
bind/terminal rows are written against an isolated database.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.managed_launch_v2 import (
    PROTOCOL_VERSION_V2,
    ManagedLaunchV2ReserveRequest,
)
from cli_agent_orchestrator.services import exact_executor as ee
from cli_agent_orchestrator.services import managed_launch_v2 as v2
from cli_agent_orchestrator.services import muse_native_launch as muse
from cli_agent_orchestrator.services import restore_contract as rc
from cli_agent_orchestrator.services import stable_agent_roster as roster
from cli_agent_orchestrator.services import terminal_service as ts

BANNER = "Muse Code 0.1.0 (0.1.0-R708.1)"
REVISION = "0.1.0-R708.1"
_NATIVE_ID = "11111111-2222-4333-8444-555555555555"
_SESSION = "cao-campaign-a"
_MODEL = "muse-spark-1.2-contributor"
_EFFORT = "high"
_REFUSAL_LACKS_BANNER = (
    "the Muse restore contract lacks the wrapper path/full version banner "
    "required to revalidate its exact profile carrier"
)


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.fixture
def muse_install(tmp_path):
    """A real digest-pinned Muse wrapper + ``.muse-version`` + inner pair.

    ``resolve_profile_carrier_inner_executable`` requires the wrapper to be a
    canonical existing file whose ``.muse-version`` sibling matches the
    banner's revision and whose ``muse-bin-<revision>`` inner exists and is
    executable, so the fixture models the real install layout.
    """
    install = os.path.realpath(str(tmp_path / "install"))
    os.makedirs(install, exist_ok=True)
    workdir = os.path.realpath(str(tmp_path / "work"))
    os.makedirs(workdir, exist_ok=True)
    wrapper = os.path.join(install, "muse")
    with open(wrapper, "w", encoding="utf-8") as handle:
        handle.write("#!/bin/sh\nexit 0\n")
    os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    (Path(install) / ".muse-version").write_text(REVISION, encoding="utf-8")
    inner = _write_probe_inner(os.path.join(install, f"muse-bin-{REVISION}"))
    return {
        "workdir": workdir,
        "wrapper": wrapper,
        "wrapper_sha256": hashlib.sha256(open(wrapper, "rb").read()).hexdigest(),
        "inner": inner,
        "banner": BANNER,
    }


def _write_probe_inner(path: str) -> str:
    """A carrier probe stub: refuses with the base-instructions refusal when
    the profile env is set, runs clean otherwise (the PROBED leg)."""
    script = f"""#!{sys.executable}
import os, sys
from pathlib import Path
env_path = os.environ.get("{muse.PROFILE_SYSTEM_PROMPT_ENV}")
if env_path is not None:
    p = Path(env_path)
    if not p.exists():
        sys.stderr.write(f"failed to read {{env_path}}: No such file or directory (os error 2)\\n")
        sys.exit(1)
    if p.stat().st_size > 0:
        sys.stderr.write("{muse.CARRIER_PROBE_REFUSAL}\\n")
        sys.exit(1)
    sys.stderr.write("muse: workspace root: /tmp\\n")
    sys.stdout.write("echo: ping\\n")
    sys.exit(0)
sys.stderr.write("muse: workspace root: /tmp\\n")
sys.stdout.write("echo: ping\\n")
sys.exit(0)
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return path


@pytest.fixture
def plain_binary(tmp_path):
    """A real digest-pinned generic provider binary for the non-Muse path."""
    workdir = os.path.realpath(str(tmp_path / "work"))
    os.makedirs(workdir, exist_ok=True)
    binary = os.path.realpath(str(tmp_path / "bin" / "claude"))
    os.makedirs(os.path.dirname(binary), exist_ok=True)
    with open(binary, "wb") as handle:
        handle.write(b"#!/bin/sh\nsleep 60\n")
    os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return {
        "workdir": workdir,
        "binary": binary,
        "binary_sha256": hashlib.sha256(open(binary, "rb").read()).hexdigest(),
    }


def _v2_muse_reserve_request(muse_install, **changes):
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "reservation_id": str(uuid.uuid4()),
        "session_name": _SESSION,
        "provider": "muse_cli",
        "agent_profile": "developer",
        "caller_id": "deadbeef",
        "working_directory": muse_install["workdir"],
        "trusted_project_root": None,
        "expected_model": _MODEL,
        "expected_effort": _EFFORT,
        "provider_executable": muse_install["wrapper"],
        "provider_executable_sha256": muse_install["wrapper_sha256"],
        "obligation_generation": str(uuid.uuid4()),
        "task_id": "self-heal-demo-task",
        "run_id": str(uuid.uuid4()),
        "delivery_id": str(uuid.uuid4()),
        "launch_nonce": "n" * 40,
        "execution_mode": "native_tui",
    }
    payload.update(changes)
    return ManagedLaunchV2ReserveRequest(**payload)


def _v2_claude_reserve_request(plain_binary, **changes):
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "reservation_id": str(uuid.uuid4()),
        "session_name": _SESSION,
        "provider": "claude_code",
        "agent_profile": "developer",
        "caller_id": "deadbeef",
        "working_directory": plain_binary["workdir"],
        "trusted_project_root": None,
        "expected_model": "claude-sonnet-4-5",
        "expected_effort": _EFFORT,
        "provider_executable": plain_binary["binary"],
        "provider_executable_sha256": plain_binary["binary_sha256"],
        "obligation_generation": str(uuid.uuid4()),
        "task_id": "self-heal-demo-task",
        "run_id": str(uuid.uuid4()),
        "delivery_id": str(uuid.uuid4()),
        "launch_nonce": "n" * 40,
        "execution_mode": "native_tui",
    }
    payload.update(changes)
    return ManagedLaunchV2ReserveRequest(**payload)


def _bind_worker(terminal_id, generation, **changes):
    payload = {
        "agent_id": str(uuid.uuid4()),
        "session_name": _SESSION,
        "role": roster.ROLE_WORKER,
        "profile_family": "developer",
        "harness": "muse_cli",
        "native_session_id": _NATIVE_ID,
        "acquisition_method": "chosen_session_id",
        "route_provenance": {"provider_route": "meta"},
        "terminal_id": terminal_id,
        "generation": generation,
        "pane_id": "%101",
        "pane_pid": 4242,
        "process_identity": {"pid": 4242, "start_marker": "2026-08-09T00:00:00Z"},
        "execution_mode": "native_tui",
    }
    payload.update(changes)
    return roster.bind_generation(roster.BindingContract(**payload))


def _terminal_row(terminal_id, generation, provider="muse_cli"):
    return database.create_terminal(
        terminal_id,
        _SESSION,
        "worker-1",
        provider,
        agent_profile="developer",
        generation=generation,
    )


def _stored_contract_payload(terminal_id, generation):
    stored = rc.get_contract_by_incarnation(terminal_id, generation)
    assert stored is not None
    return stored


def _operation_request(contract):
    from cli_agent_orchestrator.services import operation_journal as oj
    from cli_agent_orchestrator.services import session_lifecycle as sl

    return oj.OperationRequest(
        operation_id=str(uuid.uuid4()),
        session_name=_SESSION,
        agent_id=contract.agent_id,
        roster_revision=0,
        role="worker",
        profile_family="developer",
        lineage_id=contract.lineage_id,
        harness=contract.harness,
        native_session_id=contract.native_session_id or "",
        prior_terminal_id=contract.terminal_id,
        prior_generation=contract.generation,
        prior_incarnation_id=str(uuid.uuid4()),
        lifecycle_epoch=0,
        lifecycle_observation=sl.WORKING,
        restore_contract_id="restore-contract-placeholder",
        restore_contract_digest=contract.digest(),
        restore_contract_schema=rc.SCHEMA_VERSION,
        route_provider="muse_cli",
        model_requested=_MODEL,
        effort_requested=_EFFORT,
        execution_mode_requested="native_tui",
    )


# ---------------------------------------------------------------------------
# 1. Launch-time durability: the full banner lands on the reservation row.
# ---------------------------------------------------------------------------


def test_record_launch_executable_version_persists_the_banner(isolated_memory_db, muse_install):
    record, _created = v2.reserve(_v2_muse_reserve_request(muse_install))
    v2._record_launch_executable_version(record["reservation_id"], BANNER)

    facts = v2.get(record["reservation_id"])["launch_facts"]
    assert facts["provider_executable_version"] == BANNER
    # The pinned facts stay intact; the version is additive.
    assert facts["provider_executable"] == muse_install["wrapper"]
    assert facts["provider_executable_sha256"] == muse_install["wrapper_sha256"]
    with database.SessionLocal() as session:
        row = (
            session.query(database.ManagedLaunchV2ReservationModel)
            .filter(
                database.ManagedLaunchV2ReservationModel.reservation_id == record["reservation_id"]
            )
            .one()
        )
        assert json.loads(row.launch_facts_json)["provider_executable_version"] == BANNER


def test_record_launch_executable_version_ignores_an_empty_banner(isolated_memory_db, muse_install):
    """No fallback version is invented when the probe produced nothing."""
    record, _created = v2.reserve(_v2_muse_reserve_request(muse_install))
    for blank in ("", "   "):
        v2._record_launch_executable_version(record["reservation_id"], blank)
    facts = v2.get(record["reservation_id"])["launch_facts"]
    assert "provider_executable_version" not in facts


def test_record_launch_executable_version_is_best_effort_on_unreadable_facts(
    isolated_memory_db, muse_install
):
    """An unreadable facts payload never raises and never invents a version."""
    record, _created = v2.reserve(_v2_muse_reserve_request(muse_install))
    with database.SessionLocal() as session:
        row = (
            session.query(database.ManagedLaunchV2ReservationModel)
            .filter(
                database.ManagedLaunchV2ReservationModel.reservation_id == record["reservation_id"]
            )
            .one()
        )
        row.launch_facts_json = "not-json"
        session.commit()
    # The record is a no-op, never a launch gate.
    v2._record_launch_executable_version(record["reservation_id"], BANNER)
    with database.SessionLocal() as session:
        row = (
            session.query(database.ManagedLaunchV2ReservationModel)
            .filter(
                database.ManagedLaunchV2ReservationModel.reservation_id == record["reservation_id"]
            )
            .one()
        )
        assert row.launch_facts_json == "not-json"


# ---------------------------------------------------------------------------
# 2. Teardown threading: reservation -> restore-contract executable fact.
# ---------------------------------------------------------------------------


def test_teardown_publishes_the_muse_executable_fact_with_version(isolated_memory_db, muse_install):
    record, _created = v2.reserve(_v2_muse_reserve_request(muse_install))
    v2._record_launch_executable_version(record["reservation_id"], BANNER)
    terminal_id = record["terminal_id"]
    generation = record["generation"]
    _bind_worker(terminal_id, generation)
    _terminal_row(terminal_id, generation)

    ts._roster_retire_incarnation_best_effort(terminal_id, generation)

    stored = _stored_contract_payload(terminal_id, generation)
    payload = stored["contract"]
    assert payload["executable"]["state"] == rc.FACT_PRESENT
    assert payload["executable"]["value"] == {
        "path": muse_install["wrapper"],
        "sha256": muse_install["wrapper_sha256"],
        "version": BANNER,
    }
    # The version is decodable into the same constructor the executor uses.
    contract = rc.decode_stored_contract(payload)
    assert contract is not None
    assert ee._stored_executable_version(contract) == BANNER


def test_teardown_muse_executable_fact_without_version_keeps_sha256_shape(
    isolated_memory_db, muse_install
):
    """A pre-capture row (no recorded banner) keeps ``{path, sha256}``."""
    record, _created = v2.reserve(_v2_muse_reserve_request(muse_install))
    terminal_id = record["terminal_id"]
    generation = record["generation"]
    _bind_worker(terminal_id, generation)
    _terminal_row(terminal_id, generation)

    ts._roster_retire_incarnation_best_effort(terminal_id, generation)

    stored = _stored_contract_payload(terminal_id, generation)
    payload = stored["contract"]
    assert payload["executable"]["state"] == rc.FACT_PRESENT
    assert payload["executable"]["value"] == {
        "path": muse_install["wrapper"],
        "sha256": muse_install["wrapper_sha256"],
    }


def test_non_muse_executable_fact_keeps_the_sha256_shape(isolated_memory_db, plain_binary):
    """claude/codex/kimi lanes gain no version key anywhere."""
    record, _created = v2.reserve(_v2_claude_reserve_request(plain_binary))
    terminal_id = record["terminal_id"]
    generation = record["generation"]
    _bind_worker(terminal_id, generation, harness="claude_code")
    _terminal_row(terminal_id, generation, provider="claude_code")

    ts._roster_retire_incarnation_best_effort(terminal_id, generation)

    stored = _stored_contract_payload(terminal_id, generation)
    payload = stored["contract"]
    assert payload["executable"]["state"] == rc.FACT_PRESENT
    assert payload["executable"]["value"] == {
        "path": plain_binary["binary"],
        "sha256": plain_binary["binary_sha256"],
    }


def test_contract_executable_fact_shape_with_and_without_version():
    """The executable fact stays PRESENT on path+sha256 alone; version rides
    additively iff a non-empty string was durably recorded."""
    base = {
        "provider_executable": "/bin/true",
        "provider_executable_sha256": "a" * 64,
    }
    without = ts._contract_executable_fact(dict(base))
    assert without.state == rc.FACT_PRESENT
    assert without.value == {"path": "/bin/true", "sha256": "a" * 64}

    with_version = ts._contract_executable_fact({**base, "provider_executable_version": BANNER})
    assert with_version.state == rc.FACT_PRESENT
    assert with_version.value == {
        "path": "/bin/true",
        "sha256": "a" * 64,
        "version": BANNER,
    }

    empty_version = ts._contract_executable_fact({**base, "provider_executable_version": ""})
    assert empty_version.value == {"path": "/bin/true", "sha256": "a" * 64}

    absent_executable = ts._contract_executable_fact({})
    assert absent_executable.state == rc.FACT_UNAVAILABLE


# ---------------------------------------------------------------------------
# 3. Resurrection: the Muse profile-carrier revalidation consumes the banner.
# ---------------------------------------------------------------------------


def _retire_and_decode(record):
    terminal_id = record["terminal_id"]
    generation = record["generation"]
    _bind_worker(terminal_id, generation)
    _terminal_row(terminal_id, generation)
    ts._roster_retire_incarnation_best_effort(terminal_id, generation)
    stored = _stored_contract_payload(terminal_id, generation)
    contract = rc.decode_stored_contract(stored["contract"])
    assert contract is not None
    return contract


def test_muse_carrier_revalidates_on_a_teardown_contract_with_version(
    isolated_memory_db, muse_install
):
    """Positive: contract with path/sha256/version revalidates the carrier."""
    record, _created = v2.reserve(_v2_muse_reserve_request(muse_install))
    v2._record_launch_executable_version(record["reservation_id"], BANNER)
    contract = _retire_and_decode(record)

    capability, refusal = ee._muse_profile_carrier(_operation_request(contract), contract)

    assert refusal is None
    assert capability is not None and capability.supported is True
    assert capability.full_banner == BANNER
    assert capability.inner_executable == muse_install["inner"]


def test_muse_carrier_support_failure_yields_the_typed_refusal(isolated_memory_db, muse_install):
    """Negative: the banner is present, but the installed carrier drifted —
    the refusal is typed, never a fabricated pass."""
    record, _created = v2.reserve(_v2_muse_reserve_request(muse_install))
    v2._record_launch_executable_version(record["reservation_id"], BANNER)
    contract = _retire_and_decode(record)

    # The wrapper's active revision moved after the launch recorded its
    # banner, so the recorded contract no longer matches the install.
    install_dir = os.path.dirname(muse_install["wrapper"])
    (Path(install_dir) / ".muse-version").write_text("0.1.0-R999.1", encoding="utf-8")

    capability, refusal = ee._muse_profile_carrier(_operation_request(contract), contract)

    assert refusal is not None
    assert "profile carrier" in refusal
    assert "refusing before any physical effect" in refusal
    assert capability is not None and capability.supported is False
    # The refusal is a typed result, not an exception and not a pass.
    assert "profile_carrier_unverified" in capability.reason


def test_pre_change_row_muse_restoration_keeps_the_exact_typed_refusal(
    isolated_memory_db, muse_install
):
    """Back-compat: a row admitted before the version capture keeps the EXACT
    current typed refusal — no crash, no inference, no fabricated banner."""
    record, _created = v2.reserve(_v2_muse_reserve_request(muse_install))
    # No _record_launch_executable_version call: the pre-change row shape.
    contract = _retire_and_decode(record)

    capability, refusal = ee._muse_profile_carrier(_operation_request(contract), contract)

    assert refusal == _REFUSAL_LACKS_BANNER
    assert capability is None
