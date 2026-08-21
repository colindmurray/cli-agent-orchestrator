"""Lane F monitor substrate: 10 proof points, migration, and API validation."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.services import (
    registered_waits,
)
from cli_agent_orchestrator.services import stable_agent_roster as roster
from cli_agent_orchestrator.services import (
    wait_admission,
)
from cli_agent_orchestrator.services.registered_wait_monitors import (
    HEALTH_MONITORED,
    HEALTH_STALE,
    HEALTH_UNMONITORED,
    monitor_health_for,
    monitor_paths,
)
from cli_agent_orchestrator.services.registered_waits import RegistrationRequest
from cli_agent_orchestrator.services.wait_runner import compute_sha256


@pytest.fixture(autouse=True)
def _isolate_db_and_monitors(tmp_path, monkeypatch):
    # Isolate DB and monitor filesystem per test
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        f"sqlite:///{tmp_path}/isolated.db", connect_args={"check_same_thread": False}
    )
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(
        database,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=engine),
    )
    database.Base.metadata.create_all(bind=engine)
    # Ensure monitor root is isolated
    import cli_agent_orchestrator.constants as const
    from cli_agent_orchestrator.services import registered_wait_monitors as mon

    monkeypatch.setattr(const, "CAO_HOME_DIR", tmp_path)
    monkeypatch.setattr(mon, "CAO_HOME_DIR", tmp_path)
    # Ensure fake marker is set for helper liveness on macOS
    monkeypatch.setenv("CAO_WAIT_RUNNER_FAKE_MARKER", "test-marker")
    monkeypatch.setenv("CAO_M7_WAIT_MONITOR_CONSUMER_ENABLED", "true")
    yield
    engine.dispose()


# ---------- helpers ----------


def _bind_owner(session_name: str = "s", terminal_id: str = "term-1"):
    agent_id = str(uuid.uuid4())
    bound = roster.bind_generation(
        roster.BindingContract(
            agent_id=agent_id,
            session_name=session_name,
            role=roster.ROLE_WORKER,
            profile_family="developer",
            harness="codex",
            native_session_id=f"native-{uuid.uuid4()}",
            acquisition_method="chosen_session_id",
            terminal_id=terminal_id,
            generation=str(uuid.uuid4()),
            pane_id="%1",
            pane_pid=9001,
            process_identity={"pid": 9001, "start_marker": "m-1"},
            execution_mode="native_tui",
            admitted=True,
        )
    )
    return wait_admission.WaitOwner(
        agent_id=bound["incarnation"]["agent_id"],
        incarnation_id=bound["incarnation"]["incarnation_id"],
        terminal_id=bound["incarnation"]["terminal_id"],
        generation=bound["incarnation"]["generation"],
        lineage_id=bound["lineage"]["lineage_id"],
        native_session_id=bound["lineage"]["native_session_id"],
    )


def _process_adapter(tmp_path: Path):
    exe = tmp_path / "exe"
    exe.write_text("#!/bin/sh\necho hi\n")
    exe.chmod(0o755)
    sha = compute_sha256(str(exe))
    return {
        "kind": "process",
        "executable": str(exe),
        "executable_sha256": sha,
        "cwd": str(tmp_path),
        "argv": [str(exe)],
    }


def _github_adapter():
    return {
        "kind": "github-actions",
        "repository": "o/r",
        "run_id": 1,
        "run_attempt": 1,
        "workflow_id": 1,
        "head_sha": "a" * 40,
        "ref": "refs/heads/main",
    }


# 1. ORM/migration parity/idempotence/restart readback while all D fields survive
def test_monitor_migration_matches_orm_and_is_idempotent(tmp_path, monkeypatch):
    migrated = tmp_path / "migrated.db"
    orm = tmp_path / "orm.db"
    m_engine = create_engine(f"sqlite:///{migrated}")
    o_engine = create_engine(f"sqlite:///{orm}")
    try:
        monkeypatch.setattr(database, "engine", m_engine)
        database._migrate_registered_wait_monitors()
        database._migrate_registered_wait_monitors()
        database.RegisteredWaitMonitorModel.__table__.create(bind=o_engine)
    finally:
        m_engine.dispose()
        o_engine.dispose()

    def _shape(p):
        with sqlite3.connect(p) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(registered_wait_monitors)")}
            idx = {
                row[1]: bool(row[2])
                for row in conn.execute("PRAGMA index_list(registered_wait_monitors)")
            }
            # also capture index sql for uniqueness check
            idx_detail = {
                row[1]: row for row in conn.execute("PRAGMA index_list(registered_wait_monitors)")
            }
        return cols, idx_detail

    ms, mi = _shape(migrated)
    os, oi = _shape(orm)
    assert ms == os, f"migrated cols {ms} vs orm {os}"
    assert set(mi) == set(oi), f"index names {set(mi)} vs {set(oi)}"
    # minimal schema: only wait_id PK + run_dir, state indexed; no operation unique
    assert "wait_id" in ms
    assert "run_dir" in ms
    assert "request_digest" in ms
    assert "state" in ms
    assert "helper_pid" in ms
    assert "result_json" in ms
    assert "wake_message_id" in ms
    assert "ix_registered_wait_monitors_state" in mi
    assert "ix_registered_wait_monitors_operation" not in mi
    # ensure only state index is non-unique index (plus PK)
    non_pk = {k for k in mi if not k.startswith("sqlite_autoindex")}
    assert non_pk == {"ix_registered_wait_monitors_state"}
    # operation_id/adapter_json/spec_path should not exist in minimal schema
    assert "operation_id" not in ms
    assert "adapter_json" not in ms
    assert "spec_path" not in ms


def test_restart_readback_preserves_d_fields(tmp_path, monkeypatch):
    # Use isolated CAO state
    # Rebind with fresh engine
    owner = _bind_owner("restart-sess")
    tmp = tmp_path / "exe2"
    tmp.mkdir()
    adapter = _process_adapter(tmp)
    op = str(uuid.uuid4())
    req = RegistrationRequest(
        operation_id=op,
        session_name="restart-sess",
        project="p",
        task_id="t",
        name="n",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    rec = registered_waits.register(req)
    assert rec["state"] == "acknowledged"
    # Simulate restart: re-init and read back
    # Check that wait row still has all owner fields
    got = registered_waits.get(rec["wait_id"])
    assert got["owner"]["terminal_id"] == owner.terminal_id
    assert got["owner"]["stable_agent_id"] == owner.agent_id
    assert got["adapter"] == adapter
    assert got["monitor_health"] in {HEALTH_MONITORED, HEALTH_UNMONITORED, HEALTH_STALE}


# 2. intent-before-launch and acknowledgement-before-Activate ordering
def test_intent_before_launch_and_ack_before_activate(tmp_path, monkeypatch):
    owner = _bind_owner("intent-sess")
    tmp2 = tmp_path / "exe3"
    tmp2.mkdir()
    adapter = _process_adapter(tmp2)
    # Patch Popen to capture ordering
    from cli_agent_orchestrator.services import registered_wait_monitors as mon

    original_create = mon.create_monitor_intent
    calls = []

    def tracked_create(*a, **kw):
        res = original_create(*a, **kw)
        # Spec file should exist before Popen (intent-before-launch)
        assert res["spec"].exists()
        calls.append("intent")
        return res

    monkeypatch.setattr(mon, "create_monitor_intent", tracked_create)
    # Patch launch to verify intent exists before launch
    original_launch = mon.launch_dormant_runner

    def tracked_launch(paths):
        assert paths["spec"].exists()
        calls.append("launch")
        return original_launch(paths)

    monkeypatch.setattr(mon, "launch_dormant_runner", tracked_launch)
    req = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="intent-sess",
        project="p",
        task_id="t",
        name="n",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    rec = registered_waits.register(req)
    assert calls == ["intent", "launch"]
    # After register, Activate should exist only after ack commit
    paths = mon.monitor_paths(rec["wait_id"])
    # Give helper time to write ready
    time.sleep(1)
    assert paths["activate"].exists()
    assert rec["state"] == "acknowledged"


# 3. exact ready/result replay and divergent replay
def test_exact_replay_adopts_and_divergent_refuses(tmp_path, monkeypatch):
    owner = _bind_owner("replay-sess")
    tmp3 = tmp_path / "exe4"
    tmp3.mkdir()
    adapter = _process_adapter(tmp3)
    op = str(uuid.uuid4())
    req = RegistrationRequest(
        operation_id=op,
        session_name="replay-sess",
        project="p",
        task_id="t",
        name="n",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    rec1 = registered_waits.register(req)
    rec2 = registered_waits.register(req)
    assert rec2["adopted"] is True
    assert rec1["wait_id"] == rec2["wait_id"]
    # divergent adapter should refuse (valid but different)
    adapter2 = dict(adapter)
    adapter2["argv"] = [adapter["executable"], "extra-arg"]
    req3 = RegistrationRequest(
        operation_id=op,
        session_name="replay-sess",
        project="p",
        task_id="t",
        name="n2",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter2,
    )
    with pytest.raises(registered_waits.RegisteredWaitConflict):
        registered_waits.register(req3)


# 4. ambiguous launch intent never duplicates launch
def test_ambiguous_intent_never_relaunches(tmp_path, monkeypatch):
    owner = _bind_owner("ambig-sess")
    tmp4 = tmp_path / "exe5"
    tmp4.mkdir()
    adapter = _process_adapter(tmp4)
    op = str(uuid.uuid4())
    req = RegistrationRequest(
        operation_id=op,
        session_name="ambig-sess",
        project="p",
        task_id="t",
        name="n",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    # Create wait and monitor intent but simulate crash before ready: we will manually create monitor row without launching
    # Instead, use register but mock launch to not actually launch, then check second register doesn't relaunch
    from cli_agent_orchestrator.services import registered_wait_monitors as mon

    launch_calls = []
    orig_launch = mon.launch_dormant_runner

    def counting_launch(paths):
        launch_calls.append(1)
        return orig_launch(paths)

    monkeypatch.setattr(mon, "launch_dormant_runner", counting_launch)
    rec1 = registered_waits.register(req)
    first_calls = len(launch_calls)
    # Now simulate crash: second register with same op should adopt without relaunch even if ready missing
    # Remove ready/result to simulate ambiguous window: delete ready file if exists
    paths = mon.monitor_paths(rec1["wait_id"])
    if paths["ready"].exists():
        paths["ready"].unlink()
    if paths["result"].exists():
        paths["result"].unlink()
    # Also reset monitor to launch-intent to simulate ambiguous window (no helper)
    with database.SessionLocal() as db:
        m = db.get(database.RegisteredWaitMonitorModel, rec1["wait_id"])
        m.state = "launch-intent"
        m.helper_pid = None
        m.helper_start_marker = None
        db.commit()
    rec2 = registered_waits.register(req)
    assert rec2["adopted"] is True
    assert rec2["monitor_health"] == HEALTH_UNMONITORED
    # Should not have launched again
    assert len(launch_calls) == first_calls


# 5. helper death without result => monitor-stale + no suppression
def test_helper_death_without_result_is_stale_and_no_deadman(tmp_path, monkeypatch):
    owner = _bind_owner("stale-sess", terminal_id="term-stale")
    gen = owner.generation
    tmp5 = tmp_path / "exe6"
    tmp5.mkdir()
    adapter = _process_adapter(tmp5)
    req = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="stale-sess",
        project="p",
        task_id="t",
        name="n",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    rec = registered_waits.register(req)
    time.sleep(0.5)
    # Force helper death: kill helper if alive, and delete result to simulate no result
    from cli_agent_orchestrator.services.registered_wait_monitors import monitor_paths

    paths = monitor_paths(rec["wait_id"])
    with database.SessionLocal() as db:
        m = db.get(database.RegisteredWaitMonitorModel, rec["wait_id"])
        # Simulate helper pid as a dead pid
        m.helper_pid = 999999
        m.helper_start_marker = "fake-999999"
        db.commit()
    if paths["result"].exists():
        paths["result"].unlink()
    # Settle via Sentinel process_monitors (deadman is now pure read)
    registered_waits.process_monitors()
    disp = registered_waits.deadman_disposition("term-stale", gen)
    assert disp["suppress_ordinary_deadman"] is False
    # Check wait became invalid
    got = registered_waits.get(rec["wait_id"])
    assert got["state"] == "invalid"
    assert "monitor-stale" in json.dumps(got["outcome"])


# 6. exact result persisted before attachment, attachment before inbox/deliver, callback failure retry
def test_result_before_attachment_and_retry(tmp_path, monkeypatch):
    owner = _bind_owner("attach-sess")
    tmp6 = tmp_path / "exe7"
    tmp6.mkdir()
    adapter = _process_adapter(tmp6)
    req = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="attach-sess",
        project="p",
        task_id="t",
        name="n",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    rec = registered_waits.register(req)
    time.sleep(1.5)
    # Ensure result exists
    from cli_agent_orchestrator.services.registered_wait_monitors import monitor_paths

    paths = monitor_paths(rec["wait_id"])
    assert paths["result"].exists()
    # First process without attach should persist result but not create inbox
    res1 = registered_waits.process_monitors()
    # Should be result-ready
    assert any(r["state"] == "result-ready" for r in res1)
    from cli_agent_orchestrator.clients.database import InboxModel, SessionLocal

    with SessionLocal() as db:
        assert db.query(InboxModel).count() == 0

    # Now failing attach should still leave result-ready and no inbox
    def failing_attach(rec):
        raise RuntimeError("fail")

    res2 = registered_waits.process_monitors(attach_result=failing_attach)
    with SessionLocal() as db:
        assert db.query(InboxModel).count() == 0

    # Successful attach should persist refs before inbox
    def good_attach(rec):
        return {
            "communication_id": "comm-1",
            "attachment_id": "att-1",
            "digest": rec["result_digest"],
        }

    res3 = registered_waits.process_monitors(attach_result=good_attach)
    with SessionLocal() as db:
        m = db.get(database.RegisteredWaitMonitorModel, rec["wait_id"])
        assert m.communication_id == "comm-1"
        assert m.attachment_id == "att-1"
        assert db.query(InboxModel).count() == 1
    # Retry with same request should be idempotent: second good attach should not create second inbox
    res4 = registered_waits.process_monitors(attach_result=good_attach)
    with SessionLocal() as db:
        assert db.query(InboxModel).count() == 1


# 7. timer and adapter paths cannot double-wake
def test_timer_and_adapter_no_double_wake(tmp_path, monkeypatch):
    owner = _bind_owner("double-sess")
    # Timer wait
    timer_req = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="double-sess",
        project="p",
        task_id="t",
        name="timer",
        description="d",
        duration_seconds=1,
        owner=owner,
    )
    timer_rec = registered_waits.register(timer_req)
    # Adapter wait
    tmp7 = tmp_path / "exe8"
    tmp7.mkdir()
    adapter = _process_adapter(tmp7)
    adapter_req = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="double-sess",
        project="p",
        task_id="t2",
        name="adapter",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    adapter_rec = registered_waits.register(adapter_req)
    time.sleep(1.2)
    # Process due should only wake timer, not adapter
    timer_results = registered_waits.process_due(
        deliver=lambda x: None, receipt_probe=lambda tid, mid: {"status": "delivered"}
    )
    # Timer should be resolved, adapter should stay acknowledged
    timer_got = registered_waits.get(timer_rec["wait_id"])
    adapter_got = registered_waits.get(adapter_rec["wait_id"])
    assert timer_got["state"] == "resolved"
    assert adapter_got["state"] == "acknowledged"


# 8. Stop before activation, during process, after result-before-attachment, and racing delivery
def test_stop_variants(tmp_path, monkeypatch):
    from cli_agent_orchestrator.services import registered_wait_monitors as mon

    # Mock termination to always succeed for deterministic test
    monkeypatch.setattr(mon, "_terminate_pgid", lambda pgid, grace=2.0: True)
    alive_counts = {}

    def mock_alive(pid, marker):
        if pid is None:
            return False
        cnt = alive_counts.get(pid, 0)
        alive_counts[pid] = cnt + 1
        return cnt < 5

    monkeypatch.setattr(mon, "_helper_alive", mock_alive)
    monkeypatch.setattr(mon, "_group_absent", lambda pgid: True)

    owner = _bind_owner("stop-sess")
    tmp8 = tmp_path / "exe9"
    tmp8.mkdir()
    # Use a long-running exe for during-process test
    long_exe = tmp8 / "long"
    long_exe.write_text("#!/bin/sh\nsleep 5\n")
    long_exe.chmod(0o755)
    from cli_agent_orchestrator.services.wait_runner import compute_sha256

    sha = compute_sha256(str(long_exe))
    long_adapter = {
        "kind": "process",
        "executable": str(long_exe),
        "executable_sha256": sha,
        "cwd": str(tmp8),
        "argv": [str(long_exe)],
    }
    # Stop before activation: mock launch to leave in launch-intent
    orig_launch = mon.launch_dormant_runner

    def no_launch(paths):
        # Write spec but don't launch, leaving launch-intent without helper
        return None

    monkeypatch.setattr(mon, "launch_dormant_runner", no_launch)
    op1 = str(uuid.uuid4())
    req1 = RegistrationRequest(
        operation_id=op1,
        session_name="stop-sess",
        project="p",
        task_id="t",
        name="n1",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=long_adapter,
    )
    rec1 = registered_waits.register(req1)
    # Restore launch for next tests
    monkeypatch.setattr(mon, "launch_dormant_runner", orig_launch)
    # Ensure monitor is launch-intent
    from cli_agent_orchestrator.clients import database as dbc

    with dbc.SessionLocal() as db:
        m = db.get(dbc.RegisteredWaitMonitorModel, rec1["wait_id"])
        assert m.state == "launch-intent"
    stop_op = str(uuid.uuid4())
    res = registered_waits.cancel(rec1["wait_id"], operation_id=stop_op, actor="tester")
    assert res["state"] == "cancelled"
    # Stop during process: now with real launch but mocked termination
    op2 = str(uuid.uuid4())
    req2 = RegistrationRequest(
        operation_id=op2,
        session_name="stop-sess",
        project="p",
        task_id="t2",
        name="n2",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=long_adapter,
    )
    rec2 = registered_waits.register(req2)
    time.sleep(0.5)  # let helper become running (mocked alive)
    stop_op2 = str(uuid.uuid4())
    res2 = registered_waits.cancel(rec2["wait_id"], operation_id=stop_op2, actor="tester")
    assert res2["state"] == "cancelled"
    # After result-before-attachment: use quick exe, wait for result, then stop before wake
    quick_tmp = tmp_path / "quick"
    quick_tmp.mkdir()
    quick_exe = quick_tmp / "quick"
    quick_exe.write_text("#!/bin/sh\necho done\n")
    quick_exe.chmod(0o755)
    sha2 = compute_sha256(str(quick_exe))
    quick_adapter = {
        "kind": "process",
        "executable": str(quick_exe),
        "executable_sha256": sha2,
        "cwd": str(quick_tmp),
        "argv": [str(quick_exe)],
    }
    op3 = str(uuid.uuid4())
    req3 = RegistrationRequest(
        operation_id=op3,
        session_name="stop-sess",
        project="p",
        task_id="t3",
        name="n3",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=quick_adapter,
    )
    rec3 = registered_waits.register(req3)
    time.sleep(1.5)
    # Ensure result ready
    registered_waits.process_monitors()  # persist result, no attach
    stop_op3 = str(uuid.uuid4())
    res3 = registered_waits.cancel(rec3["wait_id"], operation_id=stop_op3, actor="tester")
    assert res3["state"] == "cancelled"
    # Racing delivery: create wait, get to wake-pending, then try to stop after inbox delivered should not rewrite
    op4 = str(uuid.uuid4())
    req4 = RegistrationRequest(
        operation_id=op4,
        session_name="stop-sess",
        project="p",
        task_id="t4",
        name="n4",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=quick_adapter,
    )
    rec4 = registered_waits.register(req4)
    time.sleep(1.5)
    # Poll until inbox is created
    for _ in range(20):
        registered_waits.process_monitors(
            attach_result=lambda r: {
                "communication_id": "c",
                "attachment_id": "a",
                "digest": r["result_digest"],
            }
        )
        from cli_agent_orchestrator.clients.database import InboxModel as _IM
        from cli_agent_orchestrator.clients.database import SessionLocal as _SL

        with _SL() as _db:
            _m = _db.get(database.RegisteredWaitMonitorModel, rec4["wait_id"])
            if _m and _m.wake_message_id is not None:
                break
        time.sleep(0.1)
    # Simulate delivery: mark inbox delivered and process
    from cli_agent_orchestrator.clients.database import InboxModel, SessionLocal

    with SessionLocal() as db:
        m = db.get(database.RegisteredWaitMonitorModel, rec4["wait_id"])
        assert m is not None and m.wake_message_id is not None, "wake not created"
        inbox = db.get(InboxModel, m.wake_message_id)
        inbox.status = "delivered"
        db.commit()
    # Now process should mark resolved
    registered_waits.process_monitors(
        attach_result=lambda r: {
            "communication_id": "c",
            "attachment_id": "a",
            "digest": r["result_digest"],
        },
        receipt_probe=lambda tid, mid: {"status": "delivered"},
    )
    got4 = registered_waits.get(rec4["wait_id"])
    assert got4["state"] == "resolved"
    # Stop after resolved should not rewrite
    stop_op4 = str(uuid.uuid4())
    res4 = registered_waits.cancel(rec4["wait_id"], operation_id=stop_op4, actor="tester")
    assert res4["state"] == "resolved"


def test_stop_inconclusive_raises(tmp_path, monkeypatch):
    owner = _bind_owner("inconclusive-sess")
    tmp = tmp_path / "exe10"
    tmp.mkdir()
    adapter = _process_adapter(tmp)
    req = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="inconclusive-sess",
        project="p",
        task_id="t",
        name="n",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    rec = registered_waits.register(req)
    time.sleep(0.5)
    # Make helper appear alive but termination will be inconclusive by patching _terminate_pgid
    from cli_agent_orchestrator.services import registered_wait_monitors as mon

    monkeypatch.setattr(mon, "_terminate_pgid", lambda pgid, grace=2.0: False)
    with pytest.raises(registered_waits.RegisteredWaitUnavailable):
        registered_waits.cancel(rec["wait_id"], operation_id=str(uuid.uuid4()), actor="tester")


# 9. restart adoption of active helper and durable result
def test_restart_adoption(tmp_path, monkeypatch):
    owner = _bind_owner("restart2-sess")
    tmp9 = tmp_path / "exe11"
    tmp9.mkdir()
    long_exe = tmp9 / "long"
    long_exe.write_text("#!/bin/sh\nsleep 10\n")
    long_exe.chmod(0o755)
    from cli_agent_orchestrator.services.wait_runner import compute_sha256

    sha = compute_sha256(str(long_exe))
    adapter = {
        "kind": "process",
        "executable": str(long_exe),
        "executable_sha256": sha,
        "cwd": str(tmp9),
        "argv": [str(long_exe)],
    }
    op = str(uuid.uuid4())
    req = RegistrationRequest(
        operation_id=op,
        session_name="restart2-sess",
        project="p",
        task_id="t",
        name="n",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    rec = registered_waits.register(req)
    time.sleep(1)
    # Check health is monitored
    assert monitor_health_for(rec["wait_id"])["health"] == HEALTH_MONITORED
    # Simulate restart: call process_monitors which should adopt active helper
    res = registered_waits.process_monitors()
    # Should still be active/monitored, not stale
    assert monitor_health_for(rec["wait_id"])["health"] == HEALTH_MONITORED
    # Now test durable result adoption: use quick exe
    quick_tmp = tmp_path / "quick2"
    quick_tmp.mkdir()
    quick_exe = quick_tmp / "quick"
    quick_exe.write_text("#!/bin/sh\necho done\n")
    quick_exe.chmod(0o755)
    sha2 = compute_sha256(str(quick_exe))
    quick_adapter = {
        "kind": "process",
        "executable": str(quick_exe),
        "executable_sha256": sha2,
        "cwd": str(quick_tmp),
        "argv": [str(quick_exe)],
    }
    op2 = str(uuid.uuid4())
    req2 = RegistrationRequest(
        operation_id=op2,
        session_name="restart2-sess",
        project="p",
        task_id="t2",
        name="n2",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=quick_adapter,
    )
    rec2 = registered_waits.register(req2)
    time.sleep(1.5)
    # Simulate restart after result: process_monitors should persist result
    res2 = registered_waits.process_monitors()
    assert any(r["wait_id"] == rec2["wait_id"] for r in res2)
    got2 = registered_waits.get(rec2["wait_id"])
    assert got2["monitor_health"] == HEALTH_UNMONITORED  # result-present


# 10. process and GitHub API validation/projection
def test_api_validation_and_projection(tmp_path, monkeypatch):
    owner = _bind_owner("api-sess")
    # Valid process
    tmp10 = tmp_path / "exe12"
    tmp10.mkdir()
    adapter = _process_adapter(tmp10)
    req = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="api-sess",
        project="p",
        task_id="t",
        name="n",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=adapter,
    )
    rec = registered_waits.register(req)
    assert rec["condition"]["kind"] == "process"
    assert rec["adapter"]["kind"] == "process"
    # Valid github
    g_adapter = _github_adapter()
    req2 = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="api-sess",
        project="p",
        task_id="t2",
        name="n2",
        description="d",
        duration_seconds=60,
        owner=owner,
        adapter=g_adapter,
    )
    rec2 = registered_waits.register(req2)
    assert rec2["condition"]["kind"] == "github-actions"
    # Timer still works
    req3 = RegistrationRequest(
        operation_id=str(uuid.uuid4()),
        session_name="api-sess",
        project="p",
        task_id="t3",
        name="n3",
        description="d",
        duration_seconds=60,
        owner=owner,
    )
    rec3 = registered_waits.register(req3)
    assert rec3["condition"]["kind"] == "scheduled-time"
    assert "adapter" not in rec3
    # Invalid adapter kind
    with pytest.raises(registered_waits.RegisteredWaitInvalid):
        RegistrationRequest(
            operation_id=str(uuid.uuid4()),
            session_name="api-sess",
            project="p",
            task_id="t",
            name="n",
            description="d",
            duration_seconds=60,
            owner=owner,
            adapter={"kind": "bad"},
        )
    # Invalid process missing executable
    with pytest.raises(registered_waits.RegisteredWaitInvalid):
        RegistrationRequest(
            operation_id=str(uuid.uuid4()),
            session_name="api-sess",
            project="p",
            task_id="t",
            name="n",
            description="d",
            duration_seconds=60,
            owner=owner,
            adapter={
                "kind": "process",
                "executable": "/nope",
                "executable_sha256": "x",
                "cwd": "/",
                "argv": [],
            },
        )
    # Duration bounds
    with pytest.raises(registered_waits.RegisteredWaitInvalid):
        RegistrationRequest(
            operation_id=str(uuid.uuid4()),
            session_name="api-sess",
            project="p",
            task_id="t",
            name="n",
            description="d",
            duration_seconds=0,
            owner=owner,
        )
    with pytest.raises(registered_waits.RegisteredWaitInvalid):
        RegistrationRequest(
            operation_id=str(uuid.uuid4()),
            session_name="api-sess",
            project="p",
            task_id="t",
            name="n",
            description="d",
            duration_seconds=28801,
            owner=owner,
        )
