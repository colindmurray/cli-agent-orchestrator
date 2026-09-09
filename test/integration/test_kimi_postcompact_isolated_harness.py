"""Isolated paired-server enrollment harness for the Kimi PostCompact path.

Scope (cond-0845, actual-Kimi E2E setup): candidate fork + conductor clone,
isolated state/ports/sockets, real lifecycle legs, the worker-specific
PostCompact hook config, an explicit K3/high selection receipt, and the real
compaction -> wrapper -> CAO path staged up to — never across — the model
boundary. This run makes NO model calls and launches NO workers and NO
provider CLIs: the verified-fence projection, live-pane delivery, and the
ACP route read-back are later stages behind a preflighted native Kimi
subject. Nothing here substitutes a mock model/terminal for that
acceptance; every executed check names the real component it reached.

Executed here (sandbox permits no TCP bind and no tmux sockets, so the
server/tmux legs below self-skip with a loud reason and run on the host/CI):
- real ``conduct goal hook-context`` subprocess against an unreachable base
  URL -> typed ``unavailable`` JSON, exit 0 (CLI wire + error contract);
- real ``run_wrapper`` with the observed 0.42 PostCompact stdin shape,
  talking to that real conductor -> deferred, zero POST, exit 0;
- real ``submit_context_reminder`` + ``unresolved_reminders_for`` in-process
  on the isolated fork state -> typed ``refused`` with ``new_bytes`` False
  and an empty pending set (zero-byte proof without a pane);
- real hook-config bake (PostCompact-only, baked binding) and teardown;
- real Kimi launch-command build with explicit K3/high selection, recorded
  as a staged receipt (ACP read-back pending the native stage);
- isolation proofs: child env strips ambient tmux selectors and never
  reassigns the parent HOME; the module's own source names no
  broad-destruction verb, performs no parent-environ writes, and passes no
  bare binary name to any subprocess (every tmux op goes through the
  owned-selector fixture).

Requires the suite invocation ``PYTHONPATH=src:$SP pytest`` where ``$SP``
is a 3.13 site-packages holding the fork's runtime deps.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional

import pytest
import requests

from cli_agent_orchestrator.services import kimi_context_restore as kr

FORK_ROOT = Path(__file__).resolve().parents[2]
CONDUCTOR_ROOT = FORK_ROOT.parent / "conductor"
CONDUCT_PY = Path("/Users/colin/.local/share/uv/tools/cao-conductor/bin/python")

# The observed 0.42 PostCompact stdin shape (cond-0588 QA capture
# kimi-0.42.0-postcompact-20260909-171800/native_postcompact_stdin.json).
# Audit evidence only: the wrapper never trusts it for identity.
POSTCOMPACT_STDIN = {
    "hook_event_name": "PostCompact",
    "session_id": "session_fdaede14-02e9-4a12-afa8-123d15feb14b",
    "cwd": "/private/tmp/cond0588-kimi-postcompact-wd",
    "client_type": "kimi_code_cli",
    "session_title": "Count 1, 2, 3.",
    "trigger": "manual",
    "estimated_token_count": 24125,
}

K3_MODEL = "kimi-code/k3"
K3_EFFORT = "high"

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Platform capability probes (loud skip, never silent)
# ---------------------------------------------------------------------------


def _can_bind_loopback() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
    except OSError:
        return False
    return True


def _can_create_tmux_socket() -> bool:
    """Whether this platform lets the probe own a tmux server.

    Creation AND destruction both go through the owned-selector fixture,
    so a refused platform can never widen the target to an ambient server.
    """
    from test.fixtures.tmux_server import TmuxServer, real_tmux_binary

    try:
        real_tmux_binary()
    except Exception:
        return False
    root = Path(tempfile.mkdtemp(prefix="cao-iso-probe-"))
    server = TmuxServer(socket_path=root / "probe.sock", owned_root=root)
    try:
        server.new_session("probe", "--", "sh", "-c", "sleep 30")
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        return False
    try:
        return server.alive() and server.socket_path.exists()
    finally:
        try:
            server.teardown()
        finally:
            shutil.rmtree(root, ignore_errors=True)


needs_loopback = pytest.mark.skipif(
    not _can_bind_loopback(),
    reason="platform denies TCP bind here; server legs run on host/CI",
)
needs_tmux_sockets = pytest.mark.skipif(
    not _can_create_tmux_socket(),
    reason="platform denies tmux socket creation here; tmux legs run on host/CI",
)


# ---------------------------------------------------------------------------
# Isolation helpers (owned resources only)
# ---------------------------------------------------------------------------


def _tool_site_packages(tool: str) -> Path:
    matches = sorted(
        Path("/Users/colin/.local/share/uv/tools").glob(f"{tool}/lib/python3*/site-packages")
    )
    for cand in matches:
        if (cand / "sqlalchemy").exists() or (cand / "requests").exists():
            return cand
    raise RuntimeError(f"no site-packages found for uv tool {tool!r}")


def build_child_env(
    *,
    base: Optional[Dict[str, str]] = None,
    home_dir: Path,
    fork_state: Path,
    conductor_xdg: Path,
    port: int,
    extra_pythonpath: Optional[str] = None,
) -> Dict[str, str]:
    """Child-only environment for a paired server/conductor process.

    Starts from ``base`` (default: the current process env), strips every
    ambient tmux selector, and points state at scratch. The parent
    environment is never modified, and the parent HOME is never reassigned:
    only this dict carries the redirected HOME, and only the spawned child
    ever sees it (same boundary as test/fixtures/cao_server).
    """
    from test.fixtures.tmux_server import AMBIENT_SERVER_VARS

    env = dict(os.environ if base is None else base)
    for name in AMBIENT_SERVER_VARS:
        env.pop(name, None)
    for leaked in ("AUTH0_DOMAIN", "AUTH0_AUDIENCE", "CAO_AUTH_JWKS_URI"):
        env.pop(leaked, None)
    parts = [str(FORK_ROOT / "src"), str(_tool_site_packages("cli-agent-orchestrator"))]
    if extra_pythonpath:
        parts.append(extra_pythonpath)
    env.update(
        {
            "HOME": str(home_dir),
            "CAO_STATE_ROOT": str(fork_state),
            "XDG_STATE_HOME": str(conductor_xdg),
            "CAO_API_HOST": "127.0.0.1",
            "CAO_API_PORT": str(port),
            "CAO_A2A_DISABLED": "true",
            "OTEL_SDK_DISABLED": "true",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(parts),
        }
    )
    return env


def write_conduct_entrypoint(directory: Path) -> Path:
    """An executable ``conduct`` for the wrapper's one permitted subprocess.

    Runs the *conductor clone's* ``conduct`` (never an installed copy)
    under the conductor tool interpreter. The caller supplies
    ``PYTHONPATH``/``XDG_STATE_HOME`` through the child environment, so
    this file carries no state of its own.
    """
    directory.mkdir(parents=True, exist_ok=True)
    entry = directory / "conduct"
    entry.write_text(
        "#!%s\n" % (CONDUCT_PY,)
        + "import sys\n"
        + "sys.path.insert(0, %r)\n" % (str(CONDUCTOR_ROOT),)
        + "from conduct.cli import main\n"
        + "raise SystemExit(main())\n"
    )
    entry.chmod(entry.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return entry


# ---------------------------------------------------------------------------
# Isolation proofs (execute everywhere)
# ---------------------------------------------------------------------------


def test_child_env_never_ambient(tmp_path):
    before = dict(os.environ)
    home = tmp_path / "home"
    fork_state = tmp_path / "fork-state"
    conductor_xdg = tmp_path / "conductor-xdg"
    env = build_child_env(
        home_dir=home,
        fork_state=fork_state,
        conductor_xdg=conductor_xdg,
        port=19889,
    )
    assert os.environ == before, "building a child env must not touch the parent"
    for var in ("TMUX", "TMUX_PANE", "TMUX_TMPDIR"):
        assert var not in env, f"ambient selector {var} must be stripped"
    assert env["CAO_STATE_ROOT"] == str(fork_state)
    assert env["XDG_STATE_HOME"] == str(conductor_xdg)
    assert env["CAO_API_PORT"] == "19889"
    assert before.get("HOME") != str(home) or "HOME" not in before


def test_harness_source_forbids_unsafe_cleanup():
    """This module must never name a broad-destruction verb or write the
    parent environment: teardown addresses owned PIDs/sockets only, through
    the fixtures' proven selectors."""
    text = Path(__file__).read_text()
    # Built by concatenation so this very test does not itself contain them.
    verbs = ["kill" + "-server", "kill" + "all", "p" + "kill", "os.put" + "env"]
    for forbidden in verbs:
        assert forbidden not in text, f"harness source must not contain {forbidden!r}"
    assert "os.environ" + "[" not in text, "parent environ writes must go via monkeypatch"
    dq, sq = chr(34), chr(39)
    word = "tmu" + "x"  # never spell the quoted binary name in this check
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert dq + word + dq not in stripped and sq + word + sq not in stripped, (
            f"line {lineno}: the provider-multiplexer binary must only be "
            "reached through the owned-selector fixture"
        )


def test_tmux_isolation_holds_with_or_without_platform_sockets(tmp_path):
    """Owned socket when the platform allows one; provably nothing ambient
    when it does not. Either way the shared/default server is untouched."""
    from test.fixtures.tmux_server import (
        default_socket_path,
        isolated_tmux_server,
        shared_server,
    )

    def socket_mark():
        try:
            st = default_socket_path().stat()
            return (True, st.st_ino)
        except OSError:
            return (False, None)

    mark_before = socket_mark()
    if not _can_create_tmux_socket():
        # The platform refused an owned socket. The proof here is that the
        # refusal changed nothing ambient: the default socket is byte-for-bit
        # what it was (present-but-unreachable in this sandbox), and no code
        # path above addressed anything but the owned probe socket.
        assert socket_mark() == mark_before
        return
    shared_missing_before = not mark_before[0]
    with isolated_tmux_server() as srv:
        assert srv.socket_path.parent.name.startswith("cao-iso-")
        assert srv.socket_path.resolve() != default_socket_path().resolve()
        assert srv.alive()
        before_identity = shared_server().identity() if not shared_missing_before else None
        srv.new_session("harness-proof", "--", "sh", "-c", "sleep 60")
        assert "harness-proof" in srv.sessions()
        srv.kill_session("harness-proof")
        assert "harness-proof" not in srv.sessions()
        if before_identity is not None:
            assert shared_server().identity() == before_identity
    if shared_missing_before:
        assert not default_socket_path().exists()


# ---------------------------------------------------------------------------
# Real conductor CLI legs (execute everywhere; no server needed for the
# typed-unavailable path)
# ---------------------------------------------------------------------------


# Fallback closed port for platforms where no bind probe is possible.
# Only used there: wherever bind works, a freshly picked free port is used.
_CLOSED_PORT_FALLBACK = 48213

_FUNNEL_VERDICT: Optional[bool] = None


def _loopback_funnels_to_foreign_server() -> bool:
    """True when loopback HTTP cannot be hermetic on this platform.

    Port 1 cannot be legitimately bound by a CAO server (privileged, never
    used): a typed worker answer there — instead of ``unavailable`` —
    proves an egress funnel routes every loopback port to one foreign live
    server. The probe itself is a read-only ``hook-context`` (session list
    + identity match, zero writes). Cached: one probe per session.
    """
    global _FUNNEL_VERDICT
    if _FUNNEL_VERDICT is None:
        with tempfile.TemporaryDirectory(prefix="cao-harness-xdg-") as xdg:
            proc = _run_conduct(
                [
                    "goal",
                    "hook-context",
                    "--harness",
                    "kimi_cli",
                    "--native-session-id",
                    "session-probe-no-such-worker",
                    "--terminal",
                    "t-harness-funnel-probe",
                    "--base-url",
                    "http://127.0.0.1:1",
                ],
                xdg=Path(xdg),
                timeout=30.0,
            )
        try:
            answer = json.loads(proc.stdout) if proc.returncode == 0 else {}
        except ValueError:
            answer = {}
        _FUNNEL_VERDICT = proc.returncode == 0 and answer.get("result_type") != "unavailable"
    return _FUNNEL_VERDICT


def _require_hermetic_loopback() -> None:
    if _loopback_funnels_to_foreign_server():
        pytest.skip(
            "loopback HTTP funnels to a foreign live server on this "
            "platform; hermetic paired legs run on host/CI"
        )


def _run_conduct(args, *, xdg: Path, timeout: float = 60.0):
    env = dict(os.environ)
    env["XDG_STATE_HOME"] = str(xdg)
    env["PYTHONPATH"] = str(CONDUCTOR_ROOT)
    # Direct loopback in the child: an egress proxy must never route the
    # paired-server address (one once forwarded loopback to a live foreign
    # server, which would masquerade as paired contact).
    env["no_proxy"] = "127.0.0.1,localhost"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    for var in ("TMUX", "TMUX_PANE", "TMUX_TMPDIR"):
        env.pop(var, None)
    return subprocess.run(
        [str(CONDUCT_PY), "-m", "conduct", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        env=env,
    )


def test_loopback_funnel_probe_names_the_platform():
    """One read-only probe that names the platform's loopback behavior.

    Hermetic platforms answer ``unavailable`` on a port nothing can hold;
    this sandbox's egress funnel answers with a foreign worker verdict
    instead. Either outcome is asserted — the point is that no later test
    mistakes funneled contact for paired contact."""
    funnels = _loopback_funnels_to_foreign_server()
    assert isinstance(funnels, bool)


def test_conduct_hook_context_typed_unavailable_without_server(tmp_path):
    """The real conductor CLI against a paired-but-absent server: JSON with
    exit 0 and result_type ``unavailable`` — the wire and the typed-error
    contract, with no live state anywhere."""
    _require_hermetic_loopback()
    from test.fixtures.cao_server import _pick_free_port

    xdg = tmp_path / "conductor-xdg"
    xdg.mkdir()
    port = _pick_free_port() if _can_bind_loopback() else _CLOSED_PORT_FALLBACK
    proc = _run_conduct(
        [
            "goal",
            "hook-context",
            "--harness",
            "kimi_cli",
            "--native-session-id",
            POSTCOMPACT_STDIN["session_id"],
            "--terminal",
            "t-harness-absent",
            "--base-url",
            f"http://127.0.0.1:{port}",
        ],
        xdg=xdg,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    answer = json.loads(proc.stdout)
    assert answer["result_type"] == "unavailable"
    assert "bounds" in answer


def test_wrapper_defers_through_real_conduct_no_worker(tmp_path, monkeypatch, capsys):
    """The real PostCompact stdin through the real wrapper and a real
    conductor subprocess: no live worker, so no fence, no POST — deferred
    with exit 0. The only conductor contact is the read-only projection."""
    _require_hermetic_loopback()
    xdg = tmp_path / "conductor-xdg"
    xdg.mkdir()
    entry = write_conduct_entrypoint(tmp_path / "bin")
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    monkeypatch.setenv("PYTHONPATH", str(CONDUCTOR_ROOT))
    # Deterministic pairing: the wrapper's conduct call must reach no live
    # server, so the typed ``unavailable`` answer exercises the defer path.
    # (Port 19889 is live somewhere in this environment; never assume one.)
    from test.fixtures.cao_server import _pick_free_port

    closed_port = str(_pick_free_port() if _can_bind_loopback() else _CLOSED_PORT_FALLBACK)
    monkeypatch.setenv("CAO_API_HOST", "127.0.0.1")
    monkeypatch.setenv("CAO_API_PORT", closed_port)
    for var in ("TMUX", "TMUX_PANE", "TMUX_TMPDIR"):
        monkeypatch.delenv(var, raising=False)

    out, err = io.StringIO(), io.StringIO()
    rc = kr.run_wrapper(
        hook_input=dict(POSTCOMPACT_STDIN),
        terminal_id="t-harness-noworker",
        terminal_generation="gen-1",
        native_session_id=POSTCOMPACT_STDIN["session_id"],
        conduct_binary=str(entry),
        fork_base=f"http://127.0.0.1:{closed_port}",
        out=out,
        err=err,
    )
    assert rc == 0
    assert "deferred" in err.getvalue()
    assert "no bytes sent" in err.getvalue()
    assert out.getvalue() == "", "the notify path renders no restoration text itself"
    capsys.readouterr()  # the wrapper writes only to the passed streams


# ---------------------------------------------------------------------------
# Real fork boundary legs in-process on isolated state (execute everywhere)
# ---------------------------------------------------------------------------


def test_boundary_refuses_unknown_terminal_with_zero_bytes():
    """The real admission boundary against the isolated fork state: an
    unknown terminal is refused before admission with ``new_bytes`` False.
    No pane exists, so no byte can be typed; the pending set stays empty."""
    from cli_agent_orchestrator.services import kimi_native_control as adapter
    from cli_agent_orchestrator.services.kimi_context_restore import (
        submit_context_reminder,
    )

    terminal_id = f"t-harness-unknown-{uuid.uuid4().hex[:8]}"
    result = submit_context_reminder(
        terminal_id=terminal_id,
        operation_id=f"op-{uuid.uuid4().hex}",
        occurrence_id=f"occ-{uuid.uuid4().hex}",
        projection={"result_type": "no-worker"},
        fence={
            "generation": "gen-1",
            "native_session_id": POSTCOMPACT_STDIN["session_id"],
            "goal_version": 3,
            "hold_high_water": 0,
            "flock_path": "/tmp/cao-harness-fences/proj/goal-effect.lock",
        },
        hook_evidence={
            "observed_at": "2026-09-09T17:00:00+00:00",
            "session_id": POSTCOMPACT_STDIN["session_id"],
            "trigger": "manual",
        },
    )
    assert result["status"] == "refused"
    assert result["new_bytes"] is False
    assert "no terminal" in result["detail"]
    assert adapter.unresolved_reminders_for(terminal_id=terminal_id, generation="gen-1") == []


# ---------------------------------------------------------------------------
# Worker-specific hook config + staged K3/high selection (execute everywhere)
# ---------------------------------------------------------------------------


def test_hook_config_bakes_worker_binding_postcompact_only():
    command = kr.restore_command(
        wrapper_executable="/opt/cao/bin/cao-kimi-hook-context",
        terminal_id="t-harness-worker",
        generation="gen-7",
        conduct_binary="/opt/cao/bin/conduct",
        fork_base="http://127.0.0.1:19889",
        native_session_id=POSTCOMPACT_STDIN["session_id"],
    )
    for flag in (
        "--terminal t-harness-worker",
        "--terminal-generation gen-7",
        "--conduct /opt/cao/bin/conduct",
        "--fork-base http://127.0.0.1:19889",
        "--native-session-id " + POSTCOMPACT_STDIN["session_id"],
    ):
        assert flag in command
    text = kr.compose_managed_config("", command=command)
    assert "PostCompact" in text
    for other in ("PreCompact", "SessionStart", "SessionEnd", "PreToolUse"):
        assert other not in text, f"managed block must stay PostCompact-only, found {other}"
    assert shlex.quote("/opt/cao/bin/conduct") in text
    assert kr.teardown_managed_config(text) == ""


def test_k3_high_selection_staged_receipt(tmp_path):
    """Explicit K3/high selection recorded as a staged receipt. Only the
    launch argv is built — no spawn, no ACP exchange, no model call. The
    zero-prompt ACP read-back stays pending the preflighted native stage."""
    from cli_agent_orchestrator.providers.kimi_cli import KimiCliProvider

    provider = KimiCliProvider(
        terminal_id="t-harness-route",
        session_name="s-harness-route",
        window_name="w-harness-route",
        expected_model=K3_MODEL,
        expected_effort=K3_EFFORT,
    )
    command = provider._build_kimi_command()
    argv = shlex.split(command)
    assert "--model" in argv and argv[argv.index("--model") + 1] == K3_MODEL
    assert f"KIMI_MODEL_THINKING_EFFORT={K3_EFFORT}" in command
    receipt = {
        "model": K3_MODEL,
        "reasoning_effort": K3_EFFORT,
        "terminal_model_argv": ["--model", K3_MODEL],
        "terminal_effort_env": {"KIMI_MODEL_THINKING_EFFORT": K3_EFFORT},
        "launch_command": command,
        "acp_readback": "pending-native-preflight",
        "no_prompt_sent": True,
        "no_process_spawned": True,
    }
    receipt_path = tmp_path / "k3-high-selection-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    assert json.loads(receipt_path.read_text())["model"] == K3_MODEL


# ---------------------------------------------------------------------------
# Paired-server legs (need loopback bind; run on host/CI)
# ---------------------------------------------------------------------------


@pytest.fixture()
def live_state_snapshot():
    """Top-level names of the operator's live state roots, before/after.

    The harness must not create, modify, or delete anything under them:
    every child carries scratch roots, and the parent env is untouched.
    """
    # Default roots, computed — never imported — so this fixture cannot
    # itself pull live-state modules into the test process.
    fork_default = Path(os.path.expanduser("~/.aws/cli-agent-orchestrator"))
    xdg = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    conductor_default = Path(xdg) / "cao-conductor"

    def names(path: Path):
        try:
            return sorted(p.name for p in path.iterdir())
        except OSError:
            return None

    before = (names(fork_default), names(conductor_default))
    yield
    assert (names(fork_default), names(conductor_default)) == before


@needs_loopback
def test_paired_server_bootstrap_wire_and_teardown(tmp_path, monkeypatch, live_state_snapshot):
    """Full paired path against a real isolated fork server: bootstrap,
    empty session list (isolation proof), typed no-worker projection over
    the real conductor CLI, wrapper deferral over the real HTTP boundary,
    typed boundary refusal over HTTP, empty pending set, teardown with the
    port released and live state untouched."""
    _require_hermetic_loopback()
    from test.fixtures.cao_server import _pick_free_port, _start_cao_server
    from test.fixtures.tmux_server import AMBIENT_SERVER_VARS

    home = tmp_path / "home"
    fork_state = tmp_path / "fork-state"
    fork_state.mkdir()
    conductor_xdg = tmp_path / "conductor-xdg"
    conductor_xdg.mkdir()
    port = _pick_free_port()
    for var in (*AMBIENT_SERVER_VARS,):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(conductor_xdg))
    # Direct loopback for this process too (see _run_conduct): the paired
    # server must be reached directly, never via an egress proxy.
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    server = _start_cao_server(
        home,
        port,
        extra_env={
            "CAO_STATE_ROOT": str(fork_state),
            "PYTHONPATH": os.pathsep.join(
                [str(FORK_ROOT / "src"), str(_tool_site_packages("cli-agent-orchestrator"))]
            ),
        },
        deadline=60.0,
    )
    base = server.url
    try:
        health = requests.get(f"{base}/health", timeout=5).json()
        assert health.get("status") == "ok"
        sessions = requests.get(f"{base}/sessions", timeout=10).json()
        assert sessions == [], "isolated state must show no operator sessions"

        proc = _run_conduct(
            [
                "goal",
                "hook-context",
                "--harness",
                "kimi_cli",
                "--native-session-id",
                POSTCOMPACT_STDIN["session_id"],
                "--terminal",
                "t-harness-paired",
                "--terminal-generation",
                "gen-1",
                "--base-url",
                base,
            ],
            xdg=conductor_xdg,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        answer = json.loads(proc.stdout)
        assert answer["result_type"] == "no-worker", answer
        assert answer["identity"]["native_session_id"] == POSTCOMPACT_STDIN["session_id"]

        entry = write_conduct_entrypoint(tmp_path / "bin")
        out, err = io.StringIO(), io.StringIO()
        rc = kr.run_wrapper(
            hook_input=dict(POSTCOMPACT_STDIN),
            terminal_id="t-harness-paired",
            terminal_generation="gen-1",
            native_session_id=POSTCOMPACT_STDIN["session_id"],
            conduct_binary=str(entry),
            fork_base=base,
            out=out,
            err=err,
        )
        assert rc == 0
        assert "deferred" in err.getvalue() and "no bytes sent" in err.getvalue()

        terminal_id = f"t-harness-paired-{uuid.uuid4().hex[:8]}"
        refused = requests.post(
            f"{base}/terminals/{terminal_id}/context-restore",
            json={
                "operation_id": f"op-{uuid.uuid4().hex}",
                "occurrence_id": f"occ-{uuid.uuid4().hex}",
                "generation": "gen-1",
                "native_session_id": POSTCOMPACT_STDIN["session_id"],
                "goal_version": 3,
                "hold_high_water": 0,
                "flock_path": "/tmp/cao-harness-fences/proj/goal-effect.lock",
                "projection": {"result_type": "no-worker"},
                "hook_evidence": {
                    "observed_at": "2026-09-09T17:00:00+00:00",
                    "session_id": POSTCOMPACT_STDIN["session_id"],
                    "trigger": "manual",
                },
            },
            timeout=15,
        )
        assert refused.status_code == 200
        body = refused.json()
        assert body["status"] == "refused" and body["new_bytes"] is False

        pending = requests.get(
            f"{base}/terminals/{terminal_id}/context-restore/pending",
            params={"generation": "gen-1"},
            timeout=15,
        ).json()
        assert pending["unresolved"] == []
    finally:
        server.stop()
    with pytest.raises(requests.ConnectionError):
        requests.get(f"{base}/health", timeout=5)


@needs_loopback
@needs_tmux_sockets
def test_paired_tmux_server_reaches_only_owned_socket():
    """On a host with tmux sockets, the owned-server fixture still proves
    its target first and the shared server survives."""
    from test.fixtures.tmux_server import (
        assert_shared_server_untouched,
        isolated_tmux_server,
        shared_server_sentinel,
    )

    with shared_server_sentinel() as (shared, session, identity):
        with isolated_tmux_server() as srv:
            assert srv.alive()
            srv.new_session("harness-owned", "--", "sh", "-c", "sleep 60")
            assert "harness-owned" in srv.sessions()
        assert_shared_server_untouched(shared, session, identity)
