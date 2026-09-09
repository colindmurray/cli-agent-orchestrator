"""Opt-in native-subject path: real Kimi spawn, real /compact, wire readback.

Scope (cond-0845 actual-Kimi E2E): enroll a real Kimi worker through the
production managed-launch v2 verbs, admit a task through ordinary
``conduct spawn`` admission, drive a real ``/compact`` through the
production operator-message path, and prove restoration with the
production wire scanners — plus parked/stopped no-spurious-turn legs.
No mock terminal or model is substituted anywhere: a missing leg fails
loudly with the exact absent verb named.

RUNTIME IS OPT-IN AND MANUAL. Nothing in this module launches without
``COND0845_NATIVE_SUBJECT=1``; without it every test skips at the gate.
The host driver run WILL spend real model quota: the delivered reminder
is submitted to a live Kimi turn. Provider preflight (``kimi`` on PATH,
pinned build, valid auth) is the driver's job — see HOST DRIVER below.
Do NOT claim pass until the host driver executes Green.

HOST DRIVER (execute verbatim on a host with loopback bind, tmux sockets,
and an authenticated Kimi CLI; no rediscovery needed):

  PREFLIGHT (must all succeed before the run):
    which kimi && kimi --version
      # expect: a kimi-code 0.42.x build; record the exact version
    python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print('bind ok')"
    tmux -S /tmp/cao-driver-probe.sock new-session -d -s probe -- sleep 5 \
      && tmux -S /tmp/cao-driver-probe.sock kill-server && echo 'tmux ok'

  RUN (one command; ~10-20 min wall clock, mostly Kimi readiness/compaction):
    cd /tmp/cond0845-kimi/fork && \\
    SP=/Users/colin/.local/share/uv/tools/cli-agent-orchestrator/lib/python3.13/site-packages && \\
    COND0845_NATIVE_SUBJECT=1 \\
    COND0845_CONDUCTOR_ROOT=/tmp/cond0845-kimi/conductor \\
    PYTHONPATH=src:$SP /Users/colin/.local/bin/pytest \\
      test/integration/test_kimi_postcompact_native_subject.py \\
      -p no:cacheprovider -o addopts="" -v

  On CI (suite interpreter already carries the fork deps) drop ``$SP`` and
  the ``PYTHONPATH`` prefix and run ``pytest`` from the fork root instead.
  ``COND0845_KIMI_BIN`` overrides the ``kimi`` lookup; ``COND0845_PYTHON``
  overrides the spawned-interpreter discovery (default: this interpreter).

STAGE MAP (each stage is one test; order matters):
  preflight_platform/kimi/companions -> enroll_launch_bind ->
  goal_projection_ok -> compact_restore_readback ->
  parked_boundary_refuses -> stopped_terminal_gone.
Cleanup (server stop, owned-tmux teardown, shared-server sentinel,
provider-PID reap check) runs in fixture finalizers even on failure.

KNOWN ADJUDICATION POINT: the reserve requests effort ``high``; if the
wire ``profile.bind`` reports ``max`` (as one earlier driver observed),
the readback test FAILS the mismatch with both values printed. That
failure is the signal — do not coerce it green.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from test.integration.test_kimi_postcompact_isolated_harness import (
    K3_EFFORT,
    K3_MODEL,
    POSTCOMPACT_STDIN,
    _closed_port,
    _require_conductor_root,
    build_child_env,
    discover_conductor_root,
    discover_python,
    write_conduct_entrypoint,
)
from typing import Any, Dict, Iterator, Optional

import pytest
import requests

FORK_ROOT = Path(__file__).resolve().parents[2]

NATIVE_OPT_IN_ENV = "COND0845_NATIVE_SUBJECT"
NATIVE_KIMI_BIN_ENV = "COND0845_KIMI_BIN"

#: Requested route for the whole native run. The readback stage asserts
#: the wire ``profile.bind`` equals exactly these two values.
REQUESTED_MODEL = K3_MODEL
REQUESTED_EFFORT = K3_EFFORT

#: Bounded waits (seconds). Generous: Kimi cold readiness and a real
#: compaction dominate the wall clock.
READY_DEADLINE = 300.0
HOOK_DEADLINE = 240.0
RECONCILE_DEADLINE = 240.0
SETTLE_WINDOW = 60.0
_POLL = 2.0

pytestmark = pytest.mark.integration


def _require_native_subject() -> None:
    if os.environ.get(NATIVE_OPT_IN_ENV) != "1":
        pytest.skip(
            f"native Kimi subject is opt-in: set {NATIVE_OPT_IN_ENV}=1 "
            "on a preflighted host (see module docstring HOST DRIVER)"
        )


def test_native_plan_self_consistent():
    """The opt-in contract cannot drift from its documentation: the gate
    reads the same env name the HOST DRIVER block tells the driver to
    set, and the requested route is the staged K3/high selection."""
    doc = Path(__file__).read_text()
    assert NATIVE_OPT_IN_ENV in doc
    assert NATIVE_KIMI_BIN_ENV in doc
    assert "COND0845_NATIVE_SUBJECT=1" in doc
    assert REQUESTED_MODEL == "kimi-code/k3" and REQUESTED_EFFORT == "high"


# ---------------------------------------------------------------------------
# Preflight (opt-in only; fail loudly, never degrade)
# ---------------------------------------------------------------------------


def _kimi_binary() -> str:
    override = os.environ.get(NATIVE_KIMI_BIN_ENV)
    if override:
        return override
    found = shutil.which("kimi")
    if not found:
        pytest.skip(
            "no Kimi CLI on PATH: install/authenticate Kimi 0.42.x or set "
            f"{NATIVE_KIMI_BIN_ENV} to its absolute path"
        )
    return found


def test_native_preflight_platform():
    _require_native_subject()
    from test.fixtures.tmux_server import real_tmux_binary
    from test.integration.test_kimi_postcompact_isolated_harness import (
        _can_bind_loopback,
        _can_create_tmux_socket,
        _loopback_funnels_to_foreign_server,
    )

    assert _can_bind_loopback(), "host driver needs loopback TCP bind"
    assert real_tmux_binary(), "host driver needs a tmux binary"
    assert _can_create_tmux_socket(), "host driver needs tmux socket creation"
    assert not _loopback_funnels_to_foreign_server(), (
        "loopback funnels to a foreign server here; the native run would "
        "enroll against production state — refusing"
    )


def test_native_preflight_kimi(tmp_path):
    _require_native_subject()
    binary = _kimi_binary()
    assert os.path.isfile(binary) and os.access(binary, os.X_OK)
    proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr[-1000:]
    version = proc.stdout.strip()
    assert version, "kimi --version printed nothing"
    (tmp_path / "kimi-version.txt").write_text(version + "\n")


def test_native_preflight_companions():
    _require_native_subject()
    root = _require_conductor_root()
    assert (root / "conduct" / "cli.py").exists()
    assert (FORK_ROOT / "src" / "cli_agent_orchestrator" / "api" / "main.py").exists()


# ---------------------------------------------------------------------------
# Paired native fixture (owned everything; strict teardown)
# ---------------------------------------------------------------------------


class _NativePair:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _http() -> requests.Session:
    """HTTP client that never consults proxy environment variables.

    The paired server is always direct loopback; ambient ``http_proxy``
    must not route it (the funnel lesson). No parent-environ mutation.
    """
    session = requests.Session()
    session.trust_env = False
    return session


@pytest.fixture(scope="module")
def native_pair(tmp_path_factory):
    """One isolated tmux server + one isolated fork server for the run."""
    _require_native_subject()
    from test.fixtures.cao_server import _pick_free_port, _start_cao_server
    from test.fixtures.tmux_server import (
        isolated_tmux_server,
        shared_server_sentinel,
    )
    from test.integration.test_kimi_postcompact_isolated_harness import (
        _require_fork_dep_path,
        build_child_env,
    )

    scratch = Path(tmp_path_factory.mktemp("native-subject"))
    home = scratch / "home"
    fork_state = scratch / "fork-state"
    fork_state.mkdir()
    conductor_xdg = scratch / "conductor-xdg"
    conductor_xdg.mkdir()
    worktree = scratch / "work"
    worktree.mkdir()
    shim_dir = scratch / "shim"
    port = _pick_free_port()
    dep_path = _require_fork_dep_path()

    pair = _NativePair(
        scratch=scratch,
        home=home,
        fork_state=fork_state,
        conductor_xdg=conductor_xdg,
        worktree=worktree,
        port=port,
        base=f"http://127.0.0.1:{port}",
        server=None,
        tmux=None,
        provider_pids=[],
        http=_http(),
    )
    # The server child's full environment is built, never inherited:
    # shimmed PATH first (owned tmux server), scratch state roots, and
    # the fork sources. Parent environ is untouched (build_child_env
    # copies; the one deliberate XDG override below is restored after).
    child = build_child_env(
        home_dir=home,
        fork_state=fork_state,
        conductor_xdg=conductor_xdg,
        port=port,
        dep_path=dep_path,
    )
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{old_path}"
    try:
        with shared_server_sentinel() as sentinel:
            pair.sentinel = sentinel
            with isolated_tmux_server() as srv:
                pair.tmux = srv
                srv.write_shim(shim_dir)
                child["PATH"] = f"{shim_dir}{os.pathsep}{child.get('PATH', '')}"
                pair.server = _start_cao_server(home, port, extra_env=child, deadline=90.0)
                health = pair.http.get(f"{pair.base}/health", timeout=10).json()
                assert health.get("status") == "ok"
                assert pair.http.get(f"{pair.base}/sessions", timeout=10).json() == []
                yield pair
    finally:
        os.environ["PATH"] = old_path
        if pair.server is not None:
            pair.server.stop()
        try:
            pair.http.get(f"{pair.base}/health", timeout=5)
        except Exception:
            pass
        else:
            raise AssertionError("paired server port still answers after stop()")
        for pid in pair.provider_pids:
            ps = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
            assert ps.returncode != 0, f"owned provider PID {pid} survived teardown"


# ---------------------------------------------------------------------------
# Stage 1: enroll + launch + bind through the production v2 verbs
# ---------------------------------------------------------------------------


def _reserve_payload(kimi_bin: str, worktree: Path) -> Dict[str, Any]:
    digest = hashlib.sha256(Path(kimi_bin).read_bytes()).hexdigest()
    return {
        "protocol_version": "cao-managed-launch-v2",
        "reservation_id": str(uuid.uuid4()),
        "session_name": f"cao-native-{uuid.uuid4().hex[:8]}",
        "provider": "kimi_cli",
        "agent_profile": "reviewer",
        "caller_id": uuid.uuid4().hex[:8],
        "working_directory": str(worktree),
        "trusted_project_root": None,
        "expected_model": REQUESTED_MODEL,
        "expected_effort": REQUESTED_EFFORT,
        "provider_executable": kimi_bin,
        "provider_executable_sha256": digest,
        "obligation_generation": f"obgen-{uuid.uuid4().hex[:8]}",
        "task_id": f"cond0845-native-{uuid.uuid4().hex[:8]}",
        "run_id": f"run-{uuid.uuid4().hex[:8]}",
        "delivery_id": str(uuid.uuid4()),
        "launch_nonce": uuid.uuid4().hex + uuid.uuid4().hex[:8],
        "execution_mode": "native_tui",
    }


@pytest.fixture(scope="module")
def native_worker(native_pair):
    """Enrolled, launched, bound native Kimi worker + its production facts."""
    _require_native_subject()
    base, kimi_bin = native_pair.base, _kimi_binary()
    http = native_pair.http
    payload = _reserve_payload(kimi_bin, native_pair.worktree)
    rid = payload["reservation_id"]
    resp = http.post(f"{base}/managed-launch/v2/reservations", json=payload, timeout=30)
    assert resp.status_code == 201, resp.text[:2000]
    resp = http.post(f"{base}/managed-launch/v2/reservations/{rid}/launch", timeout=60)
    assert resp.status_code == 200, resp.text[:2000]

    deadline = time.monotonic() + READY_DEADLINE
    record: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        got = http.get(f"{base}/managed-launch/v2/reservations/{rid}", timeout=15).json()
        record = got.get("record", got)
        if record.get("readiness", {}).get("native_session_id"):
            break
        if record.get("state") in ("preflight_blocked", "negative"):
            raise AssertionError(f"reservation went {record.get('state')}: {record}")
        time.sleep(_POLL)
    native_session = (record.get("readiness") or {}).get("native_session_id")
    assert native_session, f"no native session proof within {READY_DEADLINE}s: {record}"
    terminal_id = record.get("terminal_id")
    generation = record.get("generation")
    assert terminal_id and generation, f"launch record lacks binding: {record}"

    bind = http.post(
        f"{base}/managed-launch/v2/reservations/{rid}/bind",
        json={
            "protocol_version": "cao-managed-launch-v2",
            "terminal_id": terminal_id,
            "generation": generation,
            "attempt_id": uuid.uuid4().hex,
            "execution_mode": "native_tui",
        },
        timeout=30,
    )
    assert bind.status_code == 200, bind.text[:2000]

    env = record.get("environment") or {}
    kimi_home = env.get("KIMI_CODE_HOME")
    assert kimi_home, f"launch record carries no KIMI_CODE_HOME: {sorted(env)}"
    pid = (record.get("readiness") or {}).get("provider_process_id")
    if pid:
        native_pair.provider_pids.append(int(pid))
    return {
        "reservation_id": rid,
        "payload": payload,
        "terminal_id": terminal_id,
        "generation": generation,
        "native_session_id": native_session,
        "kimi_home": Path(kimi_home),
        "record": record,
    }


def test_native_enroll_launch_bind(native_worker):
    """Production enrollment is real: readiness proof, staged route argv,
    and the installed PostCompact hook all come from the launch record —
    nothing hand-written."""
    _require_native_subject()
    record = native_worker["record"]
    # The persisted reservation route (the argv rendering itself is pinned
    # by test/providers/test_kimi_cli_unit.py; the wire stage below proves
    # the actual). A guessed argv key here would be a placeholder assert.
    assert record.get("expected_model") == REQUESTED_MODEL, record
    assert record.get("expected_effort") == REQUESTED_EFFORT, record

    config = native_worker["kimi_home"] / "config.toml"
    assert config.exists(), f"launch installed no kimi config at {config}"
    text = config.read_text()
    assert "PostCompact" in text
    assert f"--terminal {native_worker['terminal_id']}" in text
    assert native_worker["native_session_id"] in text
    assert native_worker["generation"] in text


# ---------------------------------------------------------------------------
# Stage 2: ordinary admission -> verified goal projection
# ---------------------------------------------------------------------------


def _conduct(args, xdg: Path, timeout: float = 120.0):
    from test.integration.test_kimi_postcompact_isolated_harness import _run_conduct

    return _run_conduct(args, xdg=xdg, timeout=timeout)


def test_native_goal_projection_ok(native_pair, native_worker):
    """Ordinary cond-0842 admission binds the enrolled worker to a goal:
    ``conduct spawn`` with the reservation/delivery ids, then the
    hook-context projection must be ``ok`` with a ``verified`` generation
    fence. A ``no-assignment`` answer fails here naming the exact missing
    leg — never a seeded row."""
    _require_native_subject()
    project = f"cond0845-native-{uuid.uuid4().hex[:8]}"
    payload = native_worker["payload"]
    proc = _conduct(
        [
            "spawn",
            "--project",
            project,
            "--task-class",
            "fix-kimi",
            "--provider",
            "kimi_cli",
            "--profile",
            "reviewer",
            "--model",
            REQUESTED_MODEL,
            "--effort",
            REQUESTED_EFFORT,
            "--execution-mode",
            "native_tui",
            "--reservation-id",
            native_worker["reservation_id"],
            "--delivery-id",
            payload["delivery_id"],
            "--session",
            payload["session_name"],
            "--base-url",
            native_pair.base,
        ],
        xdg=native_pair.conductor_xdg,
        timeout=180.0,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    probe = _conduct(
        [
            "goal",
            "hook-context",
            "--harness",
            "kimi_cli",
            "--native-session-id",
            native_worker["native_session_id"],
            "--terminal",
            native_worker["terminal_id"],
            "--terminal-generation",
            native_worker["generation"],
            "--base-url",
            native_pair.base,
        ],
        xdg=native_pair.conductor_xdg,
    )
    assert probe.returncode == 0, probe.stderr[-3000:]
    answer = json.loads(probe.stdout)
    assert (
        answer.get("result_type") == "ok"
    ), f"ordinary admission produced no projectable goal: {answer}"
    fence = answer.get("delivery_fence") or {}
    assert answer.get("identity", {}).get("generation_fence") == "verified", answer
    assert fence.get("occurrence_id"), f"no occurrence bound: {answer}"
    native_worker["fence"] = fence
    native_worker["projection"] = answer


# ---------------------------------------------------------------------------
# Stage 3: real /compact -> delivery -> wire + model readback
# ---------------------------------------------------------------------------


def _wire_files(kimi_home: Path):
    import glob as _glob
    import os as _os

    patterns = (
        _os.path.join(str(kimi_home), "sessions", "*", "session_*", "agents", "*", "wire.jsonl"),
        _os.path.join(str(kimi_home), "sessions", "*", "agents", "*", "wire.jsonl"),
    )
    found = []
    for pattern in patterns:
        found.extend(_glob.glob(pattern))
    return sorted(set(found))


def _wire_cursor(kimi_home: Path) -> Dict[str, int]:
    import os as _os

    return {p: _os.path.getsize(p) for p in _wire_files(kimi_home)}


def _profile_binds_since(kimi_home: Path, cursor: Dict[str, int]):
    """All ``profile.bind`` events appended after ``cursor``.

    Read-only observation of the provider's own wire files (same layout
    the production ``scan_wire_for_marker`` reads). Returns a list of
    ``(path, event)`` in wire order.
    """
    import json as _json

    binds = []
    for path in _wire_files(kimi_home):
        start = cursor.get(path, 0)
        with open(path, "rb") as handle:
            handle.seek(start)
            for raw in handle.read().splitlines():
                try:
                    event = _json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if isinstance(event, dict) and event.get("type") == "profile.bind":
                    binds.append((path, event))
    return binds


def _user_entries_since(kimi_home: Path, cursor: Dict[str, int]):
    """``context.append_message`` role-``user`` entries appended after
    ``cursor`` — the provider's own record of model-context entry."""
    import json as _json

    entries = []
    for path in _wire_files(kimi_home):
        start = cursor.get(path, 0)
        with open(path, "rb") as handle:
            handle.seek(start)
            for raw in handle.read().splitlines():
                try:
                    event = _json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if (
                    isinstance(event, dict)
                    and event.get("type") == "context.append_message"
                    and isinstance(event.get("message"), dict)
                    and event["message"].get("role") == "user"
                ):
                    entries.append((path, event))
    return entries


def test_native_compact_restore_readback(native_pair, native_worker):
    """Real ``/compact`` -> real PostCompact hook -> real wrapper ->
    real boundary -> wire + model readback. The restoration text carries
    the distinct admitted objective; the wire ``profile.bind`` must equal
    the requested route exactly (a max-vs-high mismatch FAILS here with
    both values — the known adjudication point, never coerced green)."""
    _require_native_subject()
    from cli_agent_orchestrator.services import kimi_native_control as adapter

    base = native_pair.base
    http = native_pair.http
    kimi_home = native_worker["kimi_home"]
    cursor = _wire_cursor(kimi_home)

    submit = http.post(
        f"{base}/terminals/{native_worker['terminal_id']}/operator-message",
        json={"operation_id": f"op-{uuid.uuid4().hex}", "text": "/compact"},
        timeout=30,
    )
    assert submit.status_code == 200, submit.text[:2000]

    operation_id: Optional[str] = None
    hook_deadline = time.monotonic() + HOOK_DEADLINE
    while time.monotonic() < hook_deadline:
        pending = http.get(
            f"{base}/terminals/{native_worker['terminal_id']}/context-restore/pending",
            params={"generation": native_worker["generation"]},
            timeout=15,
        ).json()
        rows = pending.get("unresolved", []) or []
        mine = [
            r
            for r in rows
            if isinstance(r.get("hook_evidence"), dict)
            and r["hook_evidence"].get("trigger") == "manual"
        ]
        if mine:
            operation_id = mine[-1]["operation_id"]
            break
        time.sleep(_POLL)
    assert operation_id, (
        f"no PostCompact hook row within {HOOK_DEADLINE}s; "
        "the /compact text never compacted or the hook never fired"
    )

    rec_deadline = time.monotonic() + RECONCILE_DEADLINE
    settlement: Dict[str, Any] = {}
    while time.monotonic() < rec_deadline:
        settlement = adapter.reconcile_reminder_from_wire(
            operation_id=operation_id,
            marker=operation_id,
            session_home=[str(kimi_home)],
        )
        if settlement.get("reconciled"):
            break
        time.sleep(_POLL)
    assert settlement.get(
        "reconciled"
    ), f"reminder {operation_id} never reconciled from the wire: {settlement}"
    evidence = settlement.get("evidence") or {}
    assert evidence.get(
        "model_context_entry"
    ), f"prompt accepted but never entered model context: {settlement}"

    binds = _profile_binds_since(kimi_home, cursor)
    assert binds, "no profile.bind on the wire after /compact"
    last = binds[-1][1]
    assert (
        last.get("modelAlias") == REQUESTED_MODEL
    ), f"wire model {last.get('modelAlias')!r} != requested {REQUESTED_MODEL!r}"
    assert last.get("thinkingEffort") == REQUESTED_EFFORT, (
        f"wire effort {last.get('thinkingEffort')!r} != requested "
        f"{REQUESTED_EFFORT!r} (known max-vs-high adjudication point)"
    )

    objective = ((native_worker.get("projection") or {}).get("goal") or {}).get("objective", "")
    assert objective.strip(), "admitted goal carries no objective to trace"
    entries = _user_entries_since(kimi_home, cursor)
    texts = [
        (
            (e[1].get("message") or {}).get("content")
            if isinstance((e[1].get("message") or {}).get("content"), str)
            else json.dumps((e[1].get("message") or {}).get("content"))
        )
        for e in entries
    ]
    assert any(
        objective.strip()[:80] in t for t in texts if t
    ), "the distinct admitted objective never reached model context"
    assert any(
        "[cao-context-restoration]" in t for t in texts if t
    ), "no restoration-labeled entry reached model context"
    native_worker["operation_id"] = operation_id


# ---------------------------------------------------------------------------
# Stage 4: parked / stopped workers start no turns
# ---------------------------------------------------------------------------


def test_native_parked_boundary_refuses(native_pair, native_worker):
    """A parked worker gets no reminder-started turn: the boundary refuses
    under the parked fence with zero bytes, and the wire stays still."""
    _require_native_subject()
    base = native_pair.base
    http = native_pair.http
    session = native_worker["payload"]["session_name"]
    pause = http.post(
        f"{base}/sessions/{session}/lifecycle/pause-request",
        json={
            "requested_by": "cond0845-native-driver",
            "note": "native-subject parked no-spurious-turn observation",
        },
        timeout=30,
    )
    assert pause.status_code == 200, pause.text[:2000]

    deadline = time.monotonic() + 120.0
    paused = False
    while time.monotonic() < deadline:
        detail = http.get(f"{base}/sessions/{session}", timeout=15).json()
        if detail.get("lifecycle") in ("pausing", "paused"):
            paused = True
            break
        time.sleep(_POLL)
    assert paused, f"session never parked: {detail}"

    fence = dict(native_worker["fence"])
    refused = http.post(
        f"{base}/terminals/{native_worker['terminal_id']}/context-restore",
        json={
            "operation_id": f"op-{uuid.uuid4().hex}",
            "occurrence_id": fence.get("occurrence_id"),
            "generation": native_worker["generation"],
            "native_session_id": native_worker["native_session_id"],
            "goal_version": fence.get("goal_version"),
            "hold_high_water": fence.get("hold_high_water"),
            "flock_path": fence.get("flock_path"),
            "projection": native_worker.get("projection") or {},
            "hook_evidence": None,
        },
        timeout=30,
    )
    assert refused.status_code == 200
    body = refused.json()
    assert body["new_bytes"] is False
    assert body["status"] in ("refused", "deferred"), body

    cursor = _wire_cursor(native_worker["kimi_home"])
    time.sleep(SETTLE_WINDOW)
    assert (
        _wire_cursor(native_worker["kimi_home"]) == cursor
    ), "wire moved while parked: a spurious turn started"


def test_native_stopped_terminal_gone(native_pair, native_worker):
    """A stopped worker is gone: the boundary refuses unknown-terminal and
    the owned provider process is reaped (owned PID only, never broad)."""
    _require_native_subject()
    base = native_pair.base
    http = native_pair.http
    session = native_worker["payload"]["session_name"]
    gone = http.delete(f"{base}/sessions/{session}", timeout=60)
    assert gone.status_code in (200, 204, 404), gone.text[:2000]

    refused = http.post(
        f"{base}/terminals/{native_worker['terminal_id']}/context-restore",
        json={
            "operation_id": f"op-{uuid.uuid4().hex}",
            "occurrence_id": (native_worker.get("fence") or {}).get("occurrence_id")
            or f"occ-{uuid.uuid4().hex}",
            "generation": native_worker["generation"],
            "native_session_id": native_worker["native_session_id"],
            "goal_version": 1,
            "hold_high_water": 0,
            "flock_path": str(native_pair.scratch / "fences" / "proj" / "goal-effect.lock"),
            "projection": {"result_type": "no-worker"},
            "hook_evidence": None,
        },
        timeout=30,
    )
    assert refused.status_code == 200
    body = refused.json()
    assert body["status"] == "refused" and body["new_bytes"] is False

    for pid in native_pair.provider_pids:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            ps = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
            if ps.returncode != 0:
                break
            time.sleep(_POLL)
        else:
            raise AssertionError(f"owned provider PID {pid} not reaped after stop")
