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
  preflight_receipt -> enroll_via_spawn -> goal_assignment_ok ->
  compact_restore_readback -> parked_no_spurious_turn ->
  stopped_no_spurious_turn.
``native_preflight`` gates every fixture: no proof, no servers, no
launch. Always-run legs (route registry, pause schema+semantics, wiring
entrypoint, reserve schema) execute without opt-in and prove the
request shapes the native stages will send. Preparation passing is NOT
a native pass — only the host driver run is.
Cleanup (server stop, owned-tmux teardown, shared-server sentinel,
provider-PID reap check, owned worktree removal) runs in fixture
finalizers even on partial failure, and addresses owned resources only.

KNOWN ADJUDICATION POINT: the reserve requests effort ``high``; if the
wire ``profile.bind`` reports ``max`` (as one earlier driver observed),
the readback test FAILS the mismatch with both values printed. That
failure is the signal — do not coerce it green.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from test.integration.test_kimi_postcompact_isolated_harness import (
    K3_EFFORT,
    K3_MODEL,
    _require_conductor_root,
    build_child_env,
    discover_python,
    write_conduct_entrypoint,
)
from typing import Any, Dict, Optional

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


# ---------------------------------------------------------------------------
# Always-run regression: native-path route registry + park semantics.
#
# Guards the exact P1 blind spot from review run 140227: a static
# ``app.routes`` read misses every router-included endpoint on FastAPI >=
# 0.141 (lazy ``_IncludedRouter``), which misreported the production
# ``lifecycle/pause-request`` park verb as nonexistent. This test expands
# lazy includes exactly like the framework serves them, pins every
# (path, method) this native path calls, pins the pause request schema,
# and proves the resulting semantic state end to end in-process:
# pause-request -> lifecycle ``pausing`` -> the restoration boundary's
# own lifecycle gate refuses. No server, no model, no tmux.
# ---------------------------------------------------------------------------


def _effective_route_table(app) -> Dict[str, set]:
    """(path, methods) served by ``app``, expanding lazy includes.

    Compatible with both the pre-0.141 flat layout (included routes
    copied into ``app.routes``) and the lazy ``_IncludedRouter`` layout.
    """
    table: Dict[str, set] = {}
    stack = list(app.routes)
    while stack:
        route = stack.pop()
        if type(route).__name__ == "_IncludedRouter":
            stack.extend(route.original_router.routes)
            continue
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if isinstance(path, str) and methods:
            table.setdefault(path, set()).update(methods)
    return table


NATIVE_ROUTE_CONTRACT = {
    "/health": {"GET"},
    "/sessions": {"GET"},
    "/sessions/{session_name}": {"GET", "DELETE"},
    "/sessions/{session_name}/lifecycle": {"GET"},
    "/sessions/{session_name}/lifecycle/pause-request": {"POST"},
    "/sessions/{session_name}/lifecycle/pause-settled": {"POST"},
    "/terminals/{terminal_id}/operator-message": {"POST"},
    "/terminals/{terminal_id}/context-restore": {"POST"},
    "/terminals/{terminal_id}/context-restore/pending": {"GET"},
    "/managed-launch/v2/reservations": {"POST"},
    "/managed-launch/v2/reservations/{reservation_id}": {"GET"},
    "/managed-launch/v2/reservations/{reservation_id}/launch": {"POST"},
    "/managed-launch/v2/reservations/{reservation_id}/bind": {"POST"},
}


def test_native_route_registry_matches_handlers():
    """Every endpoint the native path calls is served, with the method
    the path uses — read off the real candidate app, not a static list."""
    from cli_agent_orchestrator.api.main import app

    table = _effective_route_table(app)
    assert len(table) > len(NATIVE_ROUTE_CONTRACT)
    missing = {
        path: sorted(methods)
        for path, methods in NATIVE_ROUTE_CONTRACT.items()
        if not set(methods) <= set(table.get(path, ()))
    }
    assert not missing, f"native path calls unserved routes: {missing}"


def test_native_wiring_entrypoint_runs_pinned_companion(tmp_path):
    """The pinned ``conduct`` entrypoint executes the companion — the
    exact binary the server child PATH and the launch hook resolve.

    Runs ``conduct --help`` through it (import proof, zero side
    effects, no server contact) and asserts the companion root is
    baked into the script. Skips actionably when the companion clone
    is absent.
    """
    from test.integration.test_kimi_postcompact_isolated_harness import (
        _require_conductor_root,
        write_conduct_entrypoint,
    )

    root = _require_conductor_root()
    entry = write_conduct_entrypoint(
        tmp_path / "bin", conductor_root=root, python=discover_python()
    )
    assert str(root) in entry.read_text()
    proc = subprocess.run([str(entry), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-1000:]
    assert "conduct" in proc.stdout.lower()


def test_native_child_path_resolves_pinned_conduct(tmp_path):
    """PATH-resolution proof the paired server relies on: a bin dir
    holding the pinned entrypoint resolves ``conduct`` to it first."""
    from test.integration.test_kimi_postcompact_isolated_harness import (
        _require_conductor_root,
        write_conduct_entrypoint,
    )

    root = _require_conductor_root()
    entry = write_conduct_entrypoint(
        tmp_path / "bin", conductor_root=root, python=discover_python()
    )
    resolved = shutil.which("conduct", path=str(tmp_path / "bin"))
    assert resolved == str(entry), f"PATH resolves conduct to {resolved}, not the pin"


def test_native_reserve_payload_validates_against_schema(tmp_path):
    """The enrollment payload validates against the production v2
    schema in-process — the no-model dry run of the launch leg (a live
    POST needs a server; the schema gate needs none)."""
    from cli_agent_orchestrator.models.managed_launch_v2 import (
        ManagedLaunchV2ReserveRequest,
    )

    exe = Path(sys.executable)
    payload = {
        "protocol_version": "cao-managed-launch-v2",
        "reservation_id": str(uuid.uuid4()),
        "session_name": "cao-native-dryrun",
        "provider": "kimi_cli",
        "agent_profile": "reviewer",
        "caller_id": uuid.uuid4().hex[:8],
        "working_directory": str(tmp_path),
        "expected_model": REQUESTED_MODEL,
        "expected_effort": REQUESTED_EFFORT,
        "provider_executable": str(exe),
        "provider_executable_sha256": hashlib.sha256(b"dry-run").hexdigest(),
        "obligation_generation": "obgen-dryrun",
        "task_id": "cond0845-native-dryrun",
        "run_id": "run-dryrun",
        "delivery_id": str(uuid.uuid4()),
        "launch_nonce": uuid.uuid4().hex + uuid.uuid4().hex[:8],
        "execution_mode": "native_tui",
    }
    validated = ManagedLaunchV2ReserveRequest.model_validate(payload)
    assert validated.expected_model == REQUESTED_MODEL
    assert validated.expected_effort == REQUESTED_EFFORT
    assert validated.execution_mode == "native_tui"


def test_native_pause_request_schema_and_semantics():
    """Exact pause schema plus the resulting park state, in-process.

    ``requested_by`` is required (a body without it must fail request
    validation); a valid request flips a fresh session to ``pausing``;
    and the restoration boundary's own lifecycle gate then refuses that
    session while an untouched session stays working. This is the same
    verdict the native parked leg asserts over HTTP.
    """
    from pydantic import ValidationError

    from cli_agent_orchestrator.api.session_lifecycle import PauseRequestBody
    from cli_agent_orchestrator.services import kimi_context_restore as kr
    from cli_agent_orchestrator.services import session_lifecycle as sl

    with pytest.raises(ValidationError):
        PauseRequestBody.model_validate({})
    body = PauseRequestBody.model_validate(
        {"requested_by": "cond0845-regression", "note": "park-state proof"}
    )
    assert body.deadline_seconds > 0

    session = f"cao-native-regression-{uuid.uuid4().hex[:8]}"
    assert sl.describe(session).get("lifecycle") == "working"
    assert kr._fork_lifecycle_working(session_name=session) is None
    sl.request_pause(
        session,
        requested_by=body.requested_by,
        deadline_seconds=body.deadline_seconds,
        note=body.note,
    )
    assert sl.describe(session).get("lifecycle") == "pausing"
    refused = kr._fork_lifecycle_working(session_name=session)
    assert refused is not None and refused[0] == "lifecycle_not_working", refused


# ---------------------------------------------------------------------------
# Mandatory preflight fixture (opt-in only; gates EVERY native fixture).
#
# Fail-fast, before any possible launch: isolated scratch, exact Kimi
# build receipt (binary + version + sha256), owned tmux capability,
# hermetic loopback, pinned conductor companion, git, a task-class the
# normal CLI supports, and a VALID real git worktree/branch/task-file
# triple (spawn validates worktree+branch; trunk capture refused).
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


def _run_checked(argv, *, cwd=None, timeout=60, env=None):
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
    assert proc.returncode == 0, f"{argv} failed: {proc.stderr[-1500:]}"
    return proc.stdout.strip()


def _conduct_env(root: Path, xdg: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["XDG_STATE_HOME"] = str(xdg)
    env["no_proxy"] = "127.0.0.1,localhost"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    for var in ("TMUX", "TMUX_PANE", "TMUX_TMPDIR"):
        env.pop(var, None)
    return env


@pytest.fixture(scope="module")
def native_preflight(tmp_path_factory):
    """All-or-nothing gate for the native run. See contract above."""
    _require_native_subject()
    from test.fixtures.tmux_server import real_tmux_binary
    from test.integration.test_kimi_postcompact_isolated_harness import (
        _can_bind_loopback,
        _can_create_tmux_socket,
        _loopback_contact_verdict,
    )

    assert _can_bind_loopback(), "host driver needs loopback TCP bind"
    assert real_tmux_binary(), "host driver needs a tmux binary"
    assert _can_create_tmux_socket(), "host driver needs tmux socket creation"
    assert _loopback_contact_verdict() == "refused", (
        f"loopback contact is {_loopback_contact_verdict()!r}, not proven "
        "hermetic; the native run would risk foreign state — refusing"
    )

    binary = _kimi_binary()
    assert os.path.isfile(binary) and os.access(binary, os.X_OK)
    version = _run_checked([binary, "--version"])
    assert version, "kimi --version printed nothing"
    with open(binary, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()

    root = _require_conductor_root()
    assert (root / "conduct" / "cli.py").exists()
    assert (FORK_ROOT / "src" / "cli_agent_orchestrator" / "api" / "main.py").exists()
    assert shutil.which("git"), "host driver needs git for the worktree triple"

    # Owned scratch comes FIRST: every path below (probe XDG, repo,
    # worktree, receipt) derives from it, and the finally-block cleanup
    # must hold even when an early probe fails.
    scratch = Path(tmp_path_factory.mktemp("native-preflight"))
    repo: Optional[Path] = None
    worktree: Optional[Path] = None
    try:
        supported = _run_checked(
            [discover_python(), "-m", "conduct", "spawn", "--help"],
            timeout=60,
            env=_conduct_env(root, scratch / "xdg"),
        )
        assert "fix-kimi" in supported, "normal CLI lacks task-class fix-kimi"

        repo = scratch / "repo"
        repo.mkdir()
        tag = uuid.uuid4().hex[:8]
        _run_checked(["git", "init", "-q", "-b", "main"], cwd=repo)
        _run_checked(["git", "config", "user.email", "cao-native@example.invalid"], cwd=repo)
        _run_checked(["git", "config", "user.name", "cao-native"], cwd=repo)
        (repo / "task.txt").write_text(f"native probe {tag}\n")
        _run_checked(["git", "add", "."], cwd=repo)
        _run_checked(["git", "commit", "-qm", "seed"], cwd=repo)
        branch = f"cao-native-{tag}"
        worktree = scratch / "worktree"
        _run_checked(["git", "worktree", "add", "-b", branch, str(worktree)], cwd=repo)
        current = _run_checked(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree)
        assert current == branch, f"worktree on {current!r}, not {branch!r}"
        objective = (
            f"cond-0845 native restoration probe {tag}: compact the worker, "
            "restore this exact objective, requirements, and checkpoint."
        )
        task_file = scratch / "task.md"
        task_file.write_text(f"# native probe {tag}\n\n{objective}\n")

        receipt = {
            "kimi_binary": binary,
            "kimi_version": version,
            "kimi_sha256": digest,
            "conductor_root": str(root),
            "repo": str(repo),
            "worktree": str(worktree),
            "branch": branch,
            "task_file": str(task_file),
            "task_class": "fix-kimi",
            "distinct_tag": tag,
            "distinct_objective": objective,
        }
        (scratch / "preflight-receipt.json").write_text(json.dumps(receipt, indent=2))
        yield receipt
    finally:
        # Teardown touches ONLY paths under this fixture's scratch: remove
        # the owned worktree registration first, then the tree. Guards hold
        # when setup failed partway (repo/worktree may be unbound). No
        # prune of foreign repositories, no broad delete.
        if repo is not None and worktree is not None and worktree.exists():
            assert scratch in worktree.parents and scratch in repo.parents
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                capture_output=True,
                timeout=60,
            )
            assert not worktree.exists(), f"owned worktree {worktree} survived teardown"
        shutil.rmtree(scratch, ignore_errors=True)


def test_native_preflight_receipt(native_preflight):
    """The gate's receipt names exact build, companion, and worktree."""
    assert native_preflight["kimi_version"]
    assert len(native_preflight["kimi_sha256"]) == 64
    assert Path(native_preflight["task_file"]).exists()
    assert native_preflight["task_class"] == "fix-kimi"


def test_native_preflight_no_use_before_def():
    """AST regression for the cf4f2f90 P1: ``native_preflight`` used
    ``scratch`` a dozen lines before assigning it, so every opt-in run
    died with UnboundLocalError at gate setup. This enforces
    definition-before-use for every local in that fixture — a revert
    reintroducing the order fails here with the exact name and lines.
    Pure static analysis: no subprocess, no platform capability, and
    nothing here stands in for a native run."""
    import ast

    tree = ast.parse(Path(__file__).read_text())
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "native_preflight"
    )
    stores: Dict[str, int] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores.setdefault(node.id, node.lineno)
    errors = [
        f"{node.id} loaded line {node.lineno} before store line {stores[node.id]}"
        for node in ast.walk(fn)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in stores
        and node.lineno < stores[node.id]
    ]
    assert not errors, errors


# ---------------------------------------------------------------------------
# Paired native fixture (owned everything; strict teardown)
# ---------------------------------------------------------------------------


class _NativePair:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _http() -> requests.Session:
    """HTTP client that never consults proxy environment variables.

    The paired server is always direct loopback; ambient ``http_proxy``
    must not route it (proxy behavior is never contact evidence).
    No parent-environ mutation.
    """
    session = requests.Session()
    session.trust_env = False
    return session


@pytest.fixture(scope="module")
def native_pair(tmp_path_factory, native_preflight):
    """One isolated tmux server + one isolated fork server for the run.

    Gated on the preflight fixture: no bind/tmux/hermetic/kimi/companion
    proof, no servers. The child PATH leads with one owned bin directory
    holding BOTH the tmux shim (owned server) and the pinned companion
    ``conduct`` entrypoint — production PATH-resolution inside the server
    and inside the launched hook therefore cannot reach an ambient
    ``conduct`` or an ambient tmux server.
    """
    _require_native_subject()
    from test.fixtures.cao_server import _pick_free_port, _start_cao_server
    from test.fixtures.tmux_server import (
        isolated_tmux_server,
        shared_server_sentinel,
    )
    from test.integration.test_kimi_postcompact_isolated_harness import (
        _PAIRED_TMUX_ANCHOR,
        _require_fork_dep_path,
        build_child_env,
        write_conduct_entrypoint,
    )

    scratch = Path(tmp_path_factory.mktemp("native-subject"))
    home = scratch / "home"
    fork_state = scratch / "fork-state"
    fork_state.mkdir()
    conductor_xdg = scratch / "conductor-xdg"
    conductor_xdg.mkdir()
    bin_dir = scratch / "bin"
    bin_dir.mkdir()
    port = _pick_free_port()
    dep_path = _require_fork_dep_path()
    conductor_root = Path(native_preflight["conductor_root"])
    entry = write_conduct_entrypoint(
        bin_dir, conductor_root=conductor_root, python=discover_python()
    )

    pair = _NativePair(
        scratch=scratch,
        home=home,
        fork_state=fork_state,
        conductor_xdg=conductor_xdg,
        port=port,
        base=f"http://127.0.0.1:{port}",
        server=None,
        tmux=None,
        provider_pids=[],
        http=_http(),
        conduct_entry=entry,
        bin_dir=bin_dir,
    )
    # The server child's full environment is built, never inherited:
    # owned bin first (tmux shim + pinned conduct), scratch state roots,
    # and the fork sources. Parent environ is untouched.
    child = build_child_env(
        home_dir=home,
        fork_state=fork_state,
        conductor_xdg=conductor_xdg,
        port=port,
        dep_path=dep_path,
    )
    try:
        with shared_server_sentinel() as sentinel:
            pair.sentinel = sentinel
            with isolated_tmux_server(anchor=_PAIRED_TMUX_ANCHOR) as srv:
                pair.tmux = srv
                srv.write_shim(bin_dir)
                assert entry.exists(), "pinned conduct entrypoint missing from bin"
                child["PATH"] = f"{bin_dir}{os.pathsep}{child.get('PATH', '')}"
                pair.server = _start_cao_server(home, port, extra_env=child, deadline=90.0)
                health = pair.http.get(f"{pair.base}/health", timeout=10).json()
                assert health.get("status") == "ok"
                assert pair.http.get(f"{pair.base}/sessions", timeout=10).json() == []
                yield pair
    finally:
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


@pytest.fixture(scope="module")
def native_admission(native_pair, native_preflight):
    """Spawned worker + launch facts. One normal verb (``conduct spawn``
    with the preflighted worktree/branch/task-file triple) carries the
    managed reservation, the native launch, and the conductor-minted
    occurrence id on its receipt. Manual reserve/launch/bind scaffolding
    is gone: the existing workflow covers it."""
    _require_native_subject()
    base, http = native_pair.base, native_pair.http
    tag = native_preflight["distinct_tag"]
    project = f"cond0845-native-{tag}"
    session = f"cao-native-{tag}"
    rid, did = str(uuid.uuid4()), str(uuid.uuid4())

    # Explicit isolated-test caller identity: spawn records a supervisor
    # terminal id for callback routing, and outside a CAO terminal there
    # is no provable one (env/TTY are diagnostics, never identity). A
    # fresh random id per fixture run names the test rig without
    # impersonating any operator or worker terminal; the explicit flag
    # always wins over an ambient $CAO_TERMINAL_ID.
    caller_id = uuid.uuid4().hex[:8]
    proc = _conduct(
        [
            "spawn",
            "--project",
            project,
            "--caller-id",
            caller_id,
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
            "--worktree",
            native_preflight["worktree"],
            "--branch",
            native_preflight["branch"],
            "--task-file",
            native_preflight["task_file"],
            "--pr-action",
            "none",
            "--reservation-id",
            rid,
            "--delivery-id",
            did,
            "--session",
            session,
            "--base-url",
            base,
        ],
        xdg=native_pair.conductor_xdg,
        timeout=300.0,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    receipt = json.loads(proc.stdout)
    assert receipt.get("ok"), f"spawn refused: {receipt}"
    occurrence = receipt.get("task_occurrence_id")
    assert occurrence, f"spawn receipt names no occurrence: {receipt}"
    terminal_id = receipt.get("terminal_id")
    generation = receipt.get("terminal_generation")
    assert terminal_id and generation, f"spawn receipt lacks binding: {receipt}"

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

    env = record.get("environment") or {}
    kimi_home = env.get("KIMI_CODE_HOME")
    assert kimi_home, f"launch record carries no KIMI_CODE_HOME: {sorted(env)}"
    pid = (record.get("readiness") or {}).get("provider_process_id")
    if pid:
        native_pair.provider_pids.append(int(pid))
    return {
        "project": project,
        "session": session,
        "reservation_id": rid,
        "delivery_id": did,
        "task_occurrence_id": occurrence,
        "terminal_id": terminal_id,
        "generation": generation,
        "native_session_id": native_session,
        "kimi_home": Path(kimi_home),
        "record": record,
        "receipt": receipt,
    }


def test_native_enroll_via_spawn(native_admission):
    """The spawn receipt carries the launch route, and the installed
    PostCompact hook is baked from the launch record — nothing staged."""
    _require_native_subject()
    record = native_admission["record"]
    assert record.get("expected_model") == REQUESTED_MODEL, record
    assert record.get("expected_effort") == REQUESTED_EFFORT, record

    config = native_admission["kimi_home"] / "config.toml"
    assert config.exists(), f"launch installed no kimi config at {config}"
    text = config.read_text()
    assert "PostCompact" in text
    assert f"--terminal {native_admission['terminal_id']}" in text
    assert native_admission["native_session_id"] in text
    assert native_admission["generation"] in text


# ---------------------------------------------------------------------------
# Stage 2: ordinary admission -> verified goal projection
# ---------------------------------------------------------------------------


def _conduct(args, xdg: Path, timeout: float = 120.0):
    from test.integration.test_kimi_postcompact_isolated_harness import _run_conduct

    return _run_conduct(args, xdg=xdg, timeout=timeout)


def test_spawn_admission_caller_gate(tmp_path, monkeypatch):
    """Spawn admission validation with the real CLI and owned fixture
    setup — no model, no server. Leg 1 reproduces the host enrollment
    failure exactly (omitted caller, no CAO terminal): typed
    configuration-schema naming the caller contract. Leg 2 carries a
    fresh explicit caller id through the identical argv and must clear
    that gate, failing instead at the next prerequisite (deploy
    receipt); nothing is contacted and nothing launches either way.
    If the fixture ever gains a deploy receipt, leg 2's frontier moves
    past unexpected-source-mutation: update the expected class then,
    never delete the caller legs."""
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "cao-native@example.invalid"],
                   cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "cao-native"], cwd=repo)
    (repo / "task.txt").write_text("admission probe\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "-b", "cao-admission-probe", str(worktree)],
                   cwd=repo, check=True)
    task_file = tmp_path / "task.md"
    task_file.write_text("# admission probe\n")
    # Hermetic omission: no ambient terminal id may mask the missing flag.
    monkeypatch.delenv("CAO_TERMINAL_ID", raising=False)

    def _argv(extra):
        return [
            "spawn",
            "--project", "cond0845-admission-probe",
            "--task-class", "fix-kimi",
            "--provider", "kimi_cli",
            "--profile", "reviewer",
            "--model", REQUESTED_MODEL,
            "--effort", REQUESTED_EFFORT,
            "--execution-mode", "native_tui",
            "--worktree", str(worktree),
            "--branch", "cao-admission-probe",
            "--task-file", str(task_file),
            "--pr-action", "none",
            "--reservation-id", str(uuid.uuid4()),
            "--delivery-id", str(uuid.uuid4()),
            "--session", "cao-admission-probe",
            "--base-url", "http://127.0.0.1:0",
            *extra,
        ]

    denied = _conduct(_argv([]), xdg=xdg, timeout=120.0)
    assert denied.returncode != 0, "spawn without caller id must not admit"
    assert "no caller_id" in denied.stderr, denied.stderr[-2000:]
    assert "configuration-schema" in denied.stderr

    admitted = _conduct(_argv(["--caller-id", uuid.uuid4().hex[:8]]), xdg=xdg, timeout=120.0)
    assert admitted.returncode != 0, "dead base-url must still refuse"
    assert "no caller_id" not in admitted.stderr + admitted.stdout
    # Typed errors print to stderr when captured (success receipts go to
    # stdout, which is why the enrollment fixture reads proc.stdout).
    answer = json.loads(admitted.stderr)
    assert answer.get("error", {}).get("failure_class") == "unexpected-source-mutation", answer


def test_native_goal_assignment_ok(native_pair, native_preflight, native_admission):
    """The occurrence-bound goal via the normal disposition transition.

    Reads the run goal (``goal show``), moves it ``assign-next`` onto the
    conductor-minted occurrence with the distinct objective, records a
    distinct checkpoint, then proves the hook projection is ``ok`` with a
    ``verified`` fence. Any ``no-assignment`` fails here naming the leg.
    """
    _require_native_subject()
    project = native_admission["project"]
    session = native_admission["session"]
    tag = native_preflight["distinct_tag"]

    shown = _conduct(
        ["goal", "show", "--project", project, "--session", session, "--format", "json"],
        xdg=native_pair.conductor_xdg,
    )
    assert shown.returncode == 0, shown.stderr[-3000:]
    goals = json.loads(shown.stdout).get("goals", [])
    assert goals, f"spawn created no goal for session {session!r}"
    prior = goals[0]
    prior_id, prior_version = prior["goal_id"], prior["goal_version"]

    moved = _conduct(
        [
            "goal",
            "disposition",
            "--kind",
            "assign-next",
            "--project",
            project,
            "--goal",
            prior_id,
            "--expect-version",
            str(prior_version),
            "--role",
            "reviewer",
            "--objective",
            native_preflight["distinct_objective"],
            "--task-occurrence",
            native_admission["task_occurrence_id"],
            "--summary",
            f"native probe {tag}: bind assignment to occurrence",
        ],
        xdg=native_pair.conductor_xdg,
    )
    assert moved.returncode == 0, moved.stderr[-3000:]

    bound = _conduct(
        [
            "goal",
            "show",
            "--project",
            project,
            "--task-occurrence",
            native_admission["task_occurrence_id"],
            "--format",
            "json",
        ],
        xdg=native_pair.conductor_xdg,
    )
    assert bound.returncode == 0, bound.stderr[-3000:]
    bound_goals = json.loads(bound.stdout).get("goals", [])
    assert len(bound_goals) == 1, f"expected one occurrence-bound goal: {bound_goals}"
    goal = bound_goals[0]
    assert goal["objective"] == native_preflight["distinct_objective"], goal

    checkpoint_text = f"native probe {tag}: pre-compaction boundary recorded"
    marked = _conduct(
        [
            "goal",
            "checkpoint",
            "--project",
            project,
            "--goal",
            goal["goal_id"],
            "--expect-version",
            str(goal["goal_version"]),
            "--summary",
            checkpoint_text,
        ],
        xdg=native_pair.conductor_xdg,
    )
    assert marked.returncode == 0, marked.stderr[-3000:]

    probe = _conduct(
        [
            "goal",
            "hook-context",
            "--harness",
            "kimi_cli",
            "--native-session-id",
            native_admission["native_session_id"],
            "--terminal",
            native_admission["terminal_id"],
            "--terminal-generation",
            native_admission["generation"],
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
    native_admission["fence"] = fence
    native_admission["projection"] = answer
    native_admission["checkpoint_text"] = checkpoint_text


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


# Event vocabulary below is the provider's own, verified against retained
# Kimi 0.42.0 wire (keys observed, payloads redacted): context.append_message
# carries {type,agentId,message,time} with message.role user and an array
# content of {type:text} parts; turn.prompt carries
# {type,agentId,input,origin,promptId,time}; turn.steer the same minus
# promptId; prompt.accepted carries {type,agentId,promptId,content,time}.
# Bookkeeping (context.append_loop_event, mcp.*, usage.*, token_counting.*,
# llm.*, runtime.*, permission.*, metadata, prompt.completed, turn.ended) is
# never turn activity. Mirrors production scan_wire_for_marker /
# _wire_texts (services/kimi_native_control.py), which own the definitions.
_TURN_TYPES = ("turn.prompt", "turn.steer", "prompt.accepted")


def _append_entries_since(kimi_home: Path, cursor: Dict[str, int]):
    """All user-role ``context.append_message`` events after ``cursor``.

    The provider's own record that text reached model context (the
    model_context_entry leg of ``scan_wire_for_marker``). Returns a list
    of ``(path, event)`` in wire order; malformed lines are skipped, and
    a cursor-known file that vanishes mid-observation fails loud instead
    of passing vacuously.
    """
    import json as _json

    entries = []
    for path in dict.fromkeys([*cursor, *_wire_files(kimi_home)]):
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


def _entry_text(event) -> str:
    """Renderable text of one wire event (mirrors ``_wire_texts``).

    Checks message/input/content: a user-role dict unfolds to its
    content, a list of {type:text} parts yields each text, a bare string
    yields itself. Joined with newlines for substring assertions.
    """
    texts = []
    for key in ("message", "input", "content"):
        payload = event.get(key) if isinstance(event, dict) else None
        if isinstance(payload, dict) and payload.get("role") == "user":
            payload = payload.get("content")
        if isinstance(payload, list):
            for part in payload:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        texts.append(text)
        elif isinstance(payload, str) and payload:
            texts.append(payload)
    return "\n".join(texts)


def _turn_events_since(kimi_home: Path, cursor: Dict[str, int]):
    """New turn activity after ``cursor``: turn.prompt / turn.steer /
    prompt.accepted plus fresh user-role context entries — the exact
    predicate the no-spurious-turn legs assert empty. Stream chatter and
    completion bookkeeping are excluded; malformed lines are skipped.
    """
    import json as _json

    found = []
    for path in dict.fromkeys([*cursor, *_wire_files(kimi_home)]):
        start = cursor.get(path, 0)
        with open(path, "rb") as handle:
            handle.seek(start)
            for raw in handle.read().splitlines():
                try:
                    event = _json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind in _TURN_TYPES:
                    found.append((path, event))
                elif (
                    kind == "context.append_message"
                    and isinstance(event.get("message"), dict)
                    and event["message"].get("role") == "user"
                ):
                    found.append((path, event))
    return found


def _wait_wire_quiet(kimi_home: Path, *, window: float = 10.0, deadline: float = 90.0) -> None:
    """Block until no new turn activity lands for ``window`` seconds.

    Bounded by ``deadline`` and loud on timeout — proceeding past an
    active wire would shift the cursor over real turns and false-pass
    the legs below. A home with no wire files at all is a pointed-at-
    nothing error, not quiet.
    """
    import time as _time

    files = _wire_files(kimi_home)
    assert files, f"no wire files under {kimi_home}; refusing a vacuous cursor"
    start = _time.monotonic()
    while True:
        cursor = _wire_cursor(kimi_home)
        _time.sleep(window)
        if _turn_events_since(kimi_home, cursor) == []:
            return
        assert _time.monotonic() - start < deadline, (
            f"wire still turning after {deadline}s; refusing to observe a moving target"
        )


def test_wire_helpers_read_genuine_redacted_frames(tmp_path):
    """Parser/readback/quiet paths against redacted frames in the genuine
    Kimi 0.42.0 shapes: exact key sets per type as observed on retained
    provider wire (payloads scrubbed, no bulk ingestion). A torn line is
    skipped without losing its neighbours; chatter and completion
    bookkeeping never count as turns; a cursor scopes reads; a deleted
    wire file fails loud instead of passing vacuously."""
    import json as _json

    agent = tmp_path / "sessions" / "wd_probe" / "session_abc123" / "agents" / "main"
    agent.mkdir(parents=True)
    wire = agent / "wire.jsonl"

    def _user(text, time):
        return {
            "type": "context.append_message",
            "agentId": "main",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
            "time": time,
        }

    lines = [
        _json.dumps({"type": "profile.bind", "agentId": "main", "modelAlias": "m",
                     "thinkingEffort": "high", "time": 1}),
        _json.dumps({"type": "mcp.tools_discovered", "agentId": "main", "time": 2}),
        _json.dumps(_user("RESTORATION sentinel-alpha checkpoint check-alpha", 3)),
        "NOT-JSON{{{",
        _json.dumps({"type": "turn.prompt", "agentId": "main", "input": "i",
                     "origin": "o", "promptId": "p1", "time": 4}),
        _json.dumps({"type": "prompt.accepted", "agentId": "main", "promptId": "p1",
                     "content": "i", "time": 5}),
        _json.dumps({"type": "usage.record", "agentId": "main", "time": 6}),
        _json.dumps({"type": "prompt.completed", "agentId": "main", "promptId": "p1",
                     "finishedAt": 7, "reason": "done", "time": 7}),
        _json.dumps({"type": "turn.ended", "agentId": "main", "turnId": "t1",
                     "reason": "done", "durationMs": 1, "time": 8}),
        _json.dumps(_user("later entry beta", 9)),
    ]
    wire.write_text("\n".join(lines) + "\n", encoding="utf-8")

    entries = _append_entries_since(tmp_path, {})
    assert len(entries) == 2, f"torn line or chatter leaked in: {len(entries)}"
    texts = [_entry_text(path_event[1]) for path_event in entries]
    hits = [t for t in texts if "RESTORATION sentinel-alpha" in t]
    assert hits and "check-alpha" in hits[0]
    rest_idx = texts.index(hits[0])
    replies = [t for t in texts[rest_idx + 1 :] if t and t != hits[0]]
    assert replies == ["later entry beta"]

    turns = _turn_events_since(tmp_path, {})
    assert [event[1]["type"] for event in turns] == [
        "context.append_message",
        "turn.prompt",
        "prompt.accepted",
        "context.append_message",
    ], "chatter counted as turns, or real turns missed"

    fresh = _wire_cursor(tmp_path)
    assert _turn_events_since(tmp_path, fresh) == []
    assert _append_entries_since(tmp_path, fresh) == []
    _wait_wire_quiet(tmp_path, window=0.05, deadline=5.0)

    wire.unlink()
    with pytest.raises(OSError):
        _turn_events_since(tmp_path, fresh)


def test_native_compact_restore_readback(native_pair, native_preflight, native_admission):
    """Real ``/compact`` -> real PostCompact hook -> real wrapper ->
    real boundary -> wire + model readback.

    ``profile.bind`` is captured from startup (cursor ``{}``) AND the
    compaction cursor is taken separately. Reconciliation goes through
    the paired HTTP boundary (re-POST the SAME operation id — the
    production adopt-and-settle path), never the adapter's default DB.
    The readback asserts the EXACT full rendered restoration post-event
    (production ``render_restoration`` of a freshly fetched projection),
    the distinct objective, one requirement, and the checkpoint — plus a
    later model reply entry. A max-vs-high effort mismatch FAILS with
    both values (known adjudication point, never coerced green).
    """
    _require_native_subject()
    from cli_agent_orchestrator.services import kimi_context_restore as kr

    base = native_pair.base
    http = native_pair.http
    kimi_home = native_admission["kimi_home"]
    tag = native_preflight["distinct_tag"]

    startup_binds = _profile_binds_since(kimi_home, {})
    assert startup_binds, "no profile.bind since startup"
    for _, event in startup_binds:
        assert (
            event.get("modelAlias") == REQUESTED_MODEL
        ), f"startup wire model {event.get('modelAlias')!r} != {REQUESTED_MODEL!r}"

    fresh = _conduct(
        [
            "goal",
            "hook-context",
            "--harness",
            "kimi_cli",
            "--native-session-id",
            native_admission["native_session_id"],
            "--terminal",
            native_admission["terminal_id"],
            "--terminal-generation",
            native_admission["generation"],
            "--base-url",
            base,
        ],
        xdg=native_pair.conductor_xdg,
    )
    assert fresh.returncode == 0, fresh.stderr[-3000:]
    projection = json.loads(fresh.stdout)
    assert projection.get("result_type") == "ok", projection
    expected_text = kr.render_restoration(projection)
    assert expected_text and expected_text.strip(), "projection renders nothing"
    requirement_lines = [
        line for line in expected_text.splitlines() if line.strip().startswith(("- ", "* ", "1."))
    ]
    assert requirement_lines, "rendered restoration carries no requirement lines"

    compaction_cursor = _wire_cursor(kimi_home)
    submit = http.post(
        f"{base}/terminals/{native_admission['terminal_id']}/operator-message",
        json={"operation_id": f"op-{uuid.uuid4().hex}", "text": "/compact"},
        timeout=30,
    )
    assert submit.status_code == 200, submit.text[:2000]

    operation_id: Optional[str] = None
    row: Dict[str, Any] = {}
    hook_deadline = time.monotonic() + HOOK_DEADLINE
    while time.monotonic() < hook_deadline:
        pending = http.get(
            f"{base}/terminals/{native_admission['terminal_id']}/context-restore/pending",
            params={"generation": native_admission["generation"]},
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
            row = mine[-1]
            operation_id = row["operation_id"]
            break
        time.sleep(_POLL)
    assert operation_id, (
        f"no PostCompact hook row within {HOOK_DEADLINE}s; "
        "the /compact text never compacted or the hook never fired"
    )

    fence = native_admission["fence"]
    repost = {
        "operation_id": operation_id,
        "occurrence_id": fence.get("occurrence_id"),
        "generation": native_admission["generation"],
        "native_session_id": native_admission["native_session_id"],
        "goal_version": fence.get("goal_version"),
        "hold_high_water": fence.get("hold_high_water"),
        "flock_path": fence.get("flock_path"),
        "projection": projection,
        "hook_evidence": row.get("hook_evidence"),
    }
    rec_deadline = time.monotonic() + RECONCILE_DEADLINE
    settled: Dict[str, Any] = {}
    while time.monotonic() < rec_deadline:
        settled = http.post(
            f"{base}/terminals/{native_admission['terminal_id']}/context-restore",
            json=repost,
            timeout=30,
        ).json()
        if settled.get("status") == "completed":
            break
        time.sleep(_POLL)
    assert (
        settled.get("status") == "completed"
    ), f"reminder {operation_id} never settled via paired HTTP: {settled}"
    pending = http.get(
        f"{base}/terminals/{native_admission['terminal_id']}/context-restore/pending",
        params={"generation": native_admission["generation"]},
        timeout=15,
    ).json()
    assert operation_id not in [
        r.get("operation_id") for r in pending.get("unresolved", [])
    ], f"settled operation still pending: {pending}"

    binds = _profile_binds_since(kimi_home, compaction_cursor)
    observed = binds if binds else startup_binds
    last = observed[-1][1]
    assert (
        last.get("modelAlias") == REQUESTED_MODEL
    ), f"wire model {last.get('modelAlias')!r} != requested {REQUESTED_MODEL!r}"
    assert last.get("thinkingEffort") == REQUESTED_EFFORT, (
        f"wire effort {last.get('thinkingEffort')!r} != requested "
        f"{REQUESTED_EFFORT!r} (known max-vs-high adjudication point)"
    )

    entries = _append_entries_since(kimi_home, compaction_cursor)
    texts = [_entry_text(e[1]) for e in entries]
    hits = [t for t in texts if expected_text.strip() in t]
    assert hits, "exact full rendered restoration never reached model context"
    assert tag in hits[0], "distinct probe tag missing from restoration entry"
    assert (
        native_admission["checkpoint_text"] in hits[0]
    ), "checkpoint missing from restoration entry"
    rest_idx = texts.index(hits[0])
    replies = [t for t in texts[rest_idx + 1 :] if t and t != hits[0]]
    assert replies, "no model reply entry after the restoration entry"
    native_admission["operation_id"] = operation_id


# ---------------------------------------------------------------------------
# Stage 4: parked / stopped workers start no turns
# ---------------------------------------------------------------------------


def test_native_parked_no_spurious_turn(native_pair, native_admission):
    """A parked worker starts no turn: idle first, cursor BEFORE the park
    request, then zero NEW TURN events over the window.

    The predicate is turn activity (``turn.prompt``/``turn.steer``/
    ``prompt.accepted``/fresh context entries) — NOT whole-wire equality,
    because legitimate bookkeeping may move the wire. The boundary leg
    proves the parked fence refuses with zero bytes.
    """
    _require_native_subject()
    base = native_pair.base
    http = native_pair.http
    kimi_home = native_admission["kimi_home"]
    session = native_admission["session"]

    _wait_wire_quiet(kimi_home)
    cursor = _wire_cursor(kimi_home)
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
    record: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        record = http.get(f"{base}/sessions/{session}/lifecycle", timeout=15).json()
        if record.get("lifecycle") in ("pausing", "paused"):
            break
        time.sleep(_POLL)
    assert record.get("lifecycle") in ("pausing", "paused"), f"session never parked: {record}"

    fence = dict(native_admission["fence"])
    refused = http.post(
        f"{base}/terminals/{native_admission['terminal_id']}/context-restore",
        json={
            "operation_id": f"op-{uuid.uuid4().hex}",
            "occurrence_id": fence.get("occurrence_id"),
            "generation": native_admission["generation"],
            "native_session_id": native_admission["native_session_id"],
            "goal_version": fence.get("goal_version"),
            "hold_high_water": fence.get("hold_high_water"),
            "flock_path": fence.get("flock_path"),
            "projection": native_admission.get("projection") or {},
            "hook_evidence": None,
        },
        timeout=30,
    )
    assert refused.status_code == 200
    body = refused.json()
    assert body["new_bytes"] is False
    assert body["status"] in ("refused", "deferred"), body

    time.sleep(SETTLE_WINDOW)
    spurious = _turn_events_since(kimi_home, cursor)
    assert spurious == [], f"spurious turn events while parked: {spurious}"


def test_native_stopped_no_spurious_turn(native_pair, native_admission):
    """A stopped worker is gone and starts nothing: cursor BEFORE the
    delete, boundary refuses unknown-terminal, zero new turn events, and
    the owned provider process is reaped (owned PID only, never broad)."""
    _require_native_subject()
    base = native_pair.base
    http = native_pair.http
    kimi_home = native_admission["kimi_home"]
    session = native_admission["session"]

    _wait_wire_quiet(kimi_home)
    cursor = _wire_cursor(kimi_home)
    gone = http.delete(f"{base}/sessions/{session}", timeout=60)
    assert gone.status_code in (200, 204, 404), gone.text[:2000]

    refused = http.post(
        f"{base}/terminals/{native_admission['terminal_id']}/context-restore",
        json={
            "operation_id": f"op-{uuid.uuid4().hex}",
            "occurrence_id": (native_admission.get("fence") or {}).get("occurrence_id")
            or f"occ-{uuid.uuid4().hex}",
            "generation": native_admission["generation"],
            "native_session_id": native_admission["native_session_id"],
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

    time.sleep(SETTLE_WINDOW)
    spurious = _turn_events_since(kimi_home, cursor)
    assert spurious == [], f"spurious turn events after stop: {spurious}"

    for pid in native_pair.provider_pids:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            ps = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
            if ps.returncode != 0:
                break
            time.sleep(_POLL)
        else:
            raise AssertionError(f"owned provider PID {pid} not reaped after stop")
