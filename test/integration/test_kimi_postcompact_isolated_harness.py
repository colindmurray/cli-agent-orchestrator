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
is a 3.13 site-packages holding the fork's runtime deps (in CI the suite
interpreter is the project env and no extra path is needed).

Portability: companion paths are discovered, never hardcoded —
``COND0845_CONDUCTOR_ROOT`` (default: the ``conductor`` sibling checkout),
``COND0845_PYTHON`` (default: this interpreter), and
``COND0845_FORK_SITE_PACKAGES`` (default: none when deps import, else the
standard uv-tools layout). Legs needing an absent companion skip with the
exact variable to set. The opt-in native-subject path (real Kimi spawn,
real /compact, wire + model readback) lives in the companion module
``test_kimi_postcompact_native_subject.py`` and never runs unasked.
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

#: Env overrides (all optional; discovery is the default):
#: - ``COND0845_CONDUCTOR_ROOT``: conductor companion clone (default: the
#:   ``conductor`` sibling of this checkout; must contain ``conduct/cli.py``).
#: - ``COND0845_PYTHON``: interpreter for spawned ``conduct``/server
#:   children (default: this test process's own interpreter; ``conduct``
#:   itself is stdlib-only).
#: - ``COND0845_FORK_SITE_PACKAGES``: extra ``sys.path`` entry carrying the
#:   fork's runtime deps for the server child (default: none when this
#:   interpreter already imports them, else the standard uv-tools layout
#:   under ``~/.local/share``).


def discover_conductor_root() -> Optional[Path]:
    """The conductor companion clone, or None when truly absent.

    An explicit ``COND0845_CONDUCTOR_ROOT`` that does not contain
    ``conduct/cli.py`` is a configuration error, not a guess: it returns
    None rather than silently falling back to the sibling checkout.
    """
    override = os.environ.get("COND0845_CONDUCTOR_ROOT")
    if override:
        cand = Path(override)
        return cand if (cand / "conduct" / "cli.py").exists() else None
    sibling = FORK_ROOT.parent / "conductor"
    if (sibling / "conduct" / "cli.py").exists():
        return sibling
    return None


def _require_conductor_root() -> Path:
    root = discover_conductor_root()
    if root is None:
        pytest.skip(
            "conductor companion clone absent: set COND0845_CONDUCTOR_ROOT "
            "to a checkout containing conduct/cli.py"
        )
    return root


def discover_python() -> str:
    """Interpreter for spawned children (conduct is stdlib-only)."""
    return os.environ.get("COND0845_PYTHON") or sys.executable


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


def _live_dep_dir() -> Optional[str]:
    """Site-packages dir backing the live ``sqlalchemy`` import, if any.

    A bare importability probe cannot tell an interpreter-native install
    from an ambient ``PYTHONPATH`` entry, and the server child only sees
    what ``build_child_env`` passes it — so resolve the directory from the
    imported module itself (then ``sys.path``), never from probe success.
    """
    try:
        import sqlalchemy as _sa
    except ImportError:
        return None
    anchor = getattr(_sa, "__file__", None)
    if anchor:
        cand = Path(anchor).resolve().parent.parent
        if (cand / "sqlalchemy").is_dir():
            return str(cand)
    for entry in sys.path:
        if entry and (Path(entry) / "sqlalchemy").is_dir():
            return entry
    return None


def discover_fork_dep_path() -> Optional[str]:
    """Extra ``sys.path`` entry the server child needs, if any.

    Resolved from the live import location first, so a parent that imports
    the fork's runtime deps only via ambient ``PYTHONPATH`` still yields a
    path the child interpreter can use. Else an explicit override, else
    the standard uv-tools layout — never a hardcoded operator path. None
    only when the deps are nowhere to be found.
    """
    override = os.environ.get("COND0845_FORK_SITE_PACKAGES")
    if override:
        return override
    live = _live_dep_dir()
    if live:
        return live
    roots = sorted(
        (Path.home() / ".local" / "share" / "uv" / "tools").glob(
            "cli-agent-orchestrator/lib/python3*/site-packages"
        )
    )
    for cand in roots:
        if (cand / "sqlalchemy").exists():
            return str(cand)
    return None


def _require_fork_dep_path() -> Optional[str]:
    path = discover_fork_dep_path()
    if path is None:
        pytest.skip(
            "fork runtime deps not importable here: run the suite in the "
            "project env or set COND0845_FORK_SITE_PACKAGES"
        )
    return path


def build_child_env(
    *,
    base: Optional[Dict[str, str]] = None,
    home_dir: Path,
    fork_state: Path,
    conductor_xdg: Path,
    port: int,
    dep_path: Optional[str] = None,
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
    parts = [str(FORK_ROOT / "src")]
    if dep_path:
        parts.append(dep_path)
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


def write_conduct_entrypoint(directory: Path, *, conductor_root: Path, python: str) -> Path:
    """An executable ``conduct`` for the wrapper's one permitted subprocess.

    Runs the *conductor companion's* ``conduct`` (never an installed copy)
    under the discovered interpreter. The caller supplies
    ``PYTHONPATH``/``XDG_STATE_HOME`` through the child environment, so
    this file carries no state of its own.
    """
    directory.mkdir(parents=True, exist_ok=True)
    entry = directory / "conduct"
    entry.write_text(
        "#!%s\n" % (python,)
        + "import sys\n"
        + "sys.path.insert(0, %r)\n" % (str(conductor_root),)
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


def _undefined_names(tree):
    """Names loaded with no binding in scope (a bounded F821).

    Collect-only imports these modules but never executes bodies, so a
    NameError inside a stage helper would surface only on the host after
    a real model run (cond-0845 P1-B). No third-party linter is
    available here; this AST pass is stdlib-only and deliberately
    narrow: nested scopes, comprehensions, and global/nonlocal are
    honoured, dynamic tricks (globals()/eval) are out of scope.
    """
    import ast as _ast
    import builtins as _builtins

    builtin = set(dir(_builtins))
    module_attrs = {
        "__name__",
        "__doc__",
        "__file__",
        "__package__",
        "__spec__",
        "__loader__",
        "__cached__",
        "__path__",
        "__annotations__",
    }
    bad = set()
    _scopes = (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef, _ast.Lambda)

    def stores(node):
        """Names a statement binds: Name Stores, string-held handler and
        pattern names, and import aliases. Never descends into nested
        scopes — their stores belong to them, not to this level."""
        found = set()
        todo = [node]
        while todo:
            child = todo.pop()
            if child is not node and isinstance(child, _scopes):
                if not isinstance(child, _ast.Lambda):
                    found.add(child.name)
                continue
            if isinstance(child, _ast.Name) and isinstance(child.ctx, (_ast.Store, _ast.Del)):
                found.add(child.id)
            elif isinstance(child, _ast.ExceptHandler) and child.name:
                found.add(child.name)
            elif isinstance(child, (_ast.MatchAs, _ast.MatchStar)) and child.name:
                found.add(child.name)
            elif isinstance(child, _ast.MatchMapping) and child.rest:
                found.add(child.rest)
            elif isinstance(child, _ast.Import):
                for alias in child.names:
                    found.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(child, _ast.ImportFrom):
                for alias in child.names:
                    if alias.name != "*":
                        found.add(alias.asname or alias.name)
            todo.extend(_ast.iter_child_nodes(child))
        return found

    def bound_in(stmts):
        bound = set()
        for stmt in stmts:
            if isinstance(stmt, _scopes):
                if not isinstance(stmt, _ast.Lambda):
                    bound.add(stmt.name)
                continue
            bound.update(stores(stmt))
        return bound

    def check(node, scopes):
        if isinstance(node, _ast.Name) and isinstance(node.ctx, _ast.Load):
            if node.id not in builtin and not any(node.id in scope for scope in scopes):
                bad.add(f"{node.id} (line {node.lineno})")
            return
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.Lambda)):
            inner = {a.arg for a in list(node.args.args) + list(node.args.kwonlyargs)}
            if node.args.vararg:
                inner.add(node.args.vararg.arg)
            if node.args.kwarg:
                inner.add(node.args.kwarg.arg)
            for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
                check(default, scopes)
            for decorator in getattr(node, "decorator_list", []):
                check(decorator, scopes)
            body = node.body if isinstance(node.body, list) else [node.body]
            check_body(body, scopes + [inner])
            return
        if isinstance(node, _ast.ClassDef):
            for decorator in node.decorator_list:
                check(decorator, scopes)
            for base in node.bases:
                check(base, scopes)
            check_body(node.body, scopes)
            return
        if isinstance(node, (_ast.ListComp, _ast.SetComp, _ast.GeneratorExp, _ast.DictComp)):
            scope = [set()]
            for gen in node.generators:
                check(gen.iter, scopes + scope)
                scope[0].update(stores(gen.target))
                for cond in gen.ifs:
                    check(cond, scopes + scope)
            if isinstance(node, _ast.DictComp):
                check(node.key, scopes + scope)
                check(node.value, scopes + scope)
            else:
                check(node.elt, scopes + scope)
            return
        if isinstance(node, _ast.Global):
            scopes[0].update(node.names)
            return
        if isinstance(node, _ast.Nonlocal):
            return
        for child in _ast.iter_child_nodes(node):
            check(child, scopes)

    def check_body(stmts, scopes):
        scopes = scopes + [bound_in(stmts)]
        for stmt in stmts:
            check(stmt, scopes)

    check_body(tree.body, [set(module_attrs)])
    return sorted(bad)


def test_acceptance_modules_have_no_undefined_names():
    """Collect-only must not hide another P1-B: every name loaded in the
    two acceptance modules resolves to a binding, import, or builtin."""
    import ast as _ast

    modules = (
        "test/integration/test_kimi_postcompact_isolated_harness.py",
        "test/integration/test_kimi_postcompact_native_subject.py",
    )
    for name in modules:
        tree = _ast.parse((FORK_ROOT / name).read_text(), filename=name)
        assert _undefined_names(tree) == [], f"{name}: undefined names"


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


def _closed_port() -> int:
    """A port nothing legitimate holds, for paired-but-absent legs.

    Freshly picked OS-ephemeral wherever bind works, else the control
    port (:attr:`_CLOSED_PROBE_PORT`). Contact-sensitive legs additionally
    pass :func:`_require_hermetic_loopback`, so a foreign listener can
    never masquerade as paired contact — it skips instead.
    """
    if _can_bind_loopback():
        from test.fixtures.cao_server import _pick_free_port

        return _pick_free_port()
    return _CLOSED_PROBE_PORT


# Control port for the contact probe: privileged, so nothing legitimate
# holds it here. A response on ONE port never proves anything about other
# ports — the probe below reports only what this port does.
_CLOSED_PROBE_PORT = 1

_CONTACT_VERDICT: Optional[str] = None


def _probe_loopback_port(port: int, timeout: float = 5.0) -> str:
    """Direct proxy-disabled observation of one loopback port.

    A raw socket is used precisely so no ``http_proxy``/``all_proxy``
    entry (and no client transport quirk) can answer on the server's
    behalf. Returns ``'refused'`` (nothing there — hermetic),
    ``'answered'`` (a live HTTP server accepted and spoke), or
    ``'error:...'`` (unobservable: timeout, denied, unreadable).
    Absent, error, and answered stay three distinct outcomes.
    """
    import socket as _socket

    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        try:
            sock.connect(("127.0.0.1", port))
        except ConnectionRefusedError:
            return "refused"
        except OSError as exc:
            return f"error:{type(exc).__name__}:{exc}"
        try:
            sock.sendall(b"GET /health HTTP/1.0\r\nHost: probe.invalid\r\n\r\n")
            chunk = sock.recv(4096)
        except OSError as exc:
            return f"error:{type(exc).__name__}:{exc}"
        if chunk.startswith((b"HTTP/", b"RTSP/")) or b"\r\n" in chunk[:64]:
            return "answered"
        return f"error:unreadable:{chunk[:32]!r}"
    finally:
        sock.close()


def _loopback_contact_verdict() -> str:
    """Cached contact verdict for the control port. See
    :func:`_probe_loopback_port` for the outcome vocabulary."""
    global _CONTACT_VERDICT
    if _CONTACT_VERDICT is None:
        _CONTACT_VERDICT = _probe_loopback_port(_CLOSED_PROBE_PORT)
    return _CONTACT_VERDICT


def _require_hermetic_loopback() -> None:
    """Gate paired mutations on PROVEN hermetic loopback.

    Only ``'refused'`` on the control port permits contact-sensitive
    legs. An ``'answered'`` means a genuine unexpected server is live on
    loopback and MUST prevent mutations; an ``'error:...'`` means contact
    is unobservable, which also prevents them. Neither skips silently:
    the reason names the verdict.
    """
    verdict = _loopback_contact_verdict()
    if verdict == "refused":
        return
    if verdict == "answered":
        pytest.skip(
            "unexpected server answers loopback 127.0.0.1:1; paired "
            "mutations refused so they cannot touch foreign state"
        )
    pytest.skip(f"loopback contact unobservable ({verdict}); paired legs need proof")


def _run_conduct(args, *, xdg: Path, timeout: float = 60.0):
    root = _require_conductor_root()
    env = dict(os.environ)
    env["XDG_STATE_HOME"] = str(xdg)
    env["PYTHONPATH"] = str(root)
    # Direct loopback in the child as hygiene: proxies must not route the
    # paired-server address. Proxy behavior is NOT the contact arbiter —
    # only the raw-socket verdict from :func:`_probe_loopback_port` is
    # (a client transport quirk once made a local semantic negative look
    # like foreign contact; see the confound replay test below).
    env["no_proxy"] = "127.0.0.1,localhost"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    for var in ("TMUX", "TMUX_PANE", "TMUX_TMPDIR"):
        env.pop(var, None)
    return subprocess.run(
        [discover_python(), "-m", "conduct", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        env=env,
    )


def test_loopback_contact_verdict_names_the_platform():
    """The contact verdict is always one well-formed outcome.

    ``'refused'`` permits contact-sensitive legs; ``'answered'`` and
    ``'error:...'`` both prevent them, each naming the verdict. The point
    is that no later test mistakes anything else for paired contact."""
    verdict = _loopback_contact_verdict()
    assert verdict == "refused" or verdict == "answered" or verdict.startswith("error:"), verdict


def test_terminal_hint_no_worker_without_contact(tmp_path):
    """Confound replay: the real conductor CLI can answer ``no-worker``
    with zero HTTP contact, so that answer must never count as contact.

    ``hook_context._terminal_candidates`` swallows the transport failure
    in its terminal-hint branch (``except Exception: listed = []``) and
    the empty candidate list then renders the semantic negative
    ``no-worker`` locally. This replays the exact observation that once
    misread that negative as foreign-server contact: the real CLI runs
    against the closed control port, and the verdict that matters is the
    raw-socket one beside it. Skips actionably without the companion.
    """
    _require_conductor_root()
    xdg = tmp_path / "conductor-xdg"
    xdg.mkdir()
    proc = _run_conduct(
        [
            "goal",
            "hook-context",
            "--harness",
            "kimi_cli",
            "--native-session-id",
            "session-probe-no-such-worker",
            "--terminal",
            "t-harness-confound-probe",
            "--base-url",
            f"http://127.0.0.1:{_CLOSED_PROBE_PORT}",
        ],
        xdg=xdg,
        timeout=30.0,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    answer = json.loads(proc.stdout)
    # Both local negatives are possible here depending on transport
    # behavior; neither is consulted as contact evidence either way.
    assert answer.get("result_type") in ("no-worker", "unavailable"), answer
    verdict = _loopback_contact_verdict()
    if verdict != "refused":
        pytest.skip(f"confound replay needs a refused control port, got {verdict}")
    # The CLI may well say no-worker here (the swallowed-transport
    # semantic negative). Whatever it says, the socket proved no contact —
    # and the hermetic gate must agree.
    _require_hermetic_loopback()


def test_hermetic_gate_consults_only_socket_verdict(monkeypatch):
    """Pure-unit counterpart for platforms where the socket is
    unobservable: the hermetic gate maps ONLY the raw-socket verdict —
    ``refused`` proceeds, ``answered`` and ``error:*`` both skip with
    the verdict named. A conduct answer (e.g. the terminal-hint
    ``no-worker``) is never an input to this decision, so the confound
    cannot recur by construction. Stubbing the cached verdict here is
    unit setup for the gate predicate itself — it enables no native
    leg, launches nothing, and touches no network.
    """
    # sys.modules[__name__], not a fresh import: pytest may hold this
    # file under a top-level module name, and the stub must hit the copy
    # whose gate actually runs below.
    import sys as _sys

    harness = _sys.modules[__name__]
    monkeypatch.setattr(harness, "_CONTACT_VERDICT", "refused")
    _require_hermetic_loopback()
    for verdict, reason in (
        ("answered", "unexpected server"),
        ("error:TimeoutError:timed out", "unobservable"),
    ):
        monkeypatch.setattr(harness, "_CONTACT_VERDICT", verdict)
        with pytest.raises(pytest.skip.Exception, match=reason):
            _require_hermetic_loopback()
    # The removed confound helper must stay removed: any verdict shaped
    # like a conduct answer is not a verdict at all.
    assert not hasattr(harness, "_loopback_funnels_to_foreign_server")


def test_probe_detects_owned_server_and_closed_port():
    """The probe reports a real owned loopback server as ``answered``
    and the closed control port as observed — no mocks of the network
    path, just a stdlib server the test itself owns and closes."""
    if not _can_bind_loopback():
        pytest.skip("platform denies TCP bind; owned-server half runs on host/CI")
    import http.server as _http_server
    import threading as _threading

    httpd = _http_server.HTTPServer(("127.0.0.1", 0), _http_server.BaseHTTPRequestHandler)
    port = httpd.server_address[1]
    thread = _threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        assert _probe_loopback_port(port) == "answered"
    finally:
        httpd.shutdown()
        thread.join(timeout=10)
        httpd.server_close()
    assert _probe_loopback_port(port) == "refused"


def test_conduct_hook_context_typed_unavailable_without_server(tmp_path):
    """The real conductor CLI against a paired-but-absent server: JSON with
    exit 0 and result_type ``unavailable`` — the wire and the typed-error
    contract, with no live state anywhere."""
    _require_hermetic_loopback()

    xdg = tmp_path / "conductor-xdg"
    xdg.mkdir()
    port = _closed_port()
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
    root = _require_conductor_root()
    entry = write_conduct_entrypoint(
        tmp_path / "bin", conductor_root=root, python=discover_python()
    )
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    monkeypatch.setenv("PYTHONPATH", str(root))
    # Deterministic pairing: the wrapper's conduct call must reach no live
    # server, so the typed ``unavailable`` answer exercises the defer path.
    # Never assume a port (19889 proved live in one environment already).
    closed_port = str(_closed_port())
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


def test_boundary_refuses_unknown_terminal_with_zero_bytes(tmp_path):
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
            "flock_path": str(tmp_path / "fences" / "proj" / "goal-effect.lock"),
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


def test_wrapper_identity_mismatch_never_spawns_conduct(tmp_path):
    """Mutation on the trust seam: a stdin session that disagrees with the
    baked binding restores nothing AND never spawns the conductor — the
    refusal precedes every subprocess. The tripwire is a conduct stand-in
    that records any invocation to a file; the file must stay absent."""
    log = tmp_path / "conduct-calls.log"
    fake = tmp_path / "conduct"
    fake.write_text("#!/bin/sh\n" + f'echo SPAWNED "$@" >> {shlex.quote(str(log))}\n' + "exit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

    out, err = io.StringIO(), io.StringIO()
    rc = kr.run_wrapper(
        hook_input=dict(POSTCOMPACT_STDIN, session_id="session-impostor-rotated-0000"),
        terminal_id="t-harness-mismatch",
        terminal_generation="gen-1",
        native_session_id=POSTCOMPACT_STDIN["session_id"],
        conduct_binary=str(fake),
        fork_base="http://127.0.0.1:1",
        out=out,
        err=err,
    )
    assert rc == 0
    assert "does not match baked" in err.getvalue()
    assert not log.exists(), "mismatched identity must not spawn conduct"


def test_helpers_ignore_unrelated_cwd(tmp_path, monkeypatch):
    """Discovery and builders use absolute paths only: an unrelated cwd
    changes nothing about the produced env, entrypoint, or resolutions."""
    (tmp_path / "work").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path / "work")
    monkeypatch.delenv("COND0845_PYTHON", raising=False)

    home = tmp_path / "home"
    env = build_child_env(
        home_dir=home,
        fork_state=tmp_path / "fork-state",
        conductor_xdg=tmp_path / "conductor-xdg",
        port=1,
        dep_path="/dep",
    )
    assert env["HOME"] == str(home)
    assert env["CAO_STATE_ROOT"] == str(tmp_path / "fork-state")
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(FORK_ROOT / "src")
    assert env["PYTHONPATH"].split(os.pathsep)[1] == "/dep"

    root = _require_conductor_root()
    entry = write_conduct_entrypoint(
        tmp_path / "bin", conductor_root=root, python=discover_python()
    )
    text = entry.read_text()
    assert str(root) in text
    assert discover_python() == sys.executable
    monkeypatch.setenv("COND0845_PYTHON", "/explicit/python")
    assert discover_python() == "/explicit/python"


def test_server_child_imports_deps_without_parent_pythonpath(tmp_path):
    """The resolved dep dir must import in a real child whose environment
    carries no ambient ``PYTHONPATH`` — a parent that imports the deps via
    its own ``PYTHONPATH`` must not green-light a child that cannot. This
    is the cond-0845 host correction for the ``ModuleNotFoundError:
    fastapi`` bootstrap failure: the child proves the import itself."""
    dep_path = _require_fork_dep_path()
    assert dep_path, "resolved dep dir must be a real path, never None"
    home = tmp_path / "home"
    home.mkdir()
    env = build_child_env(
        base={"PATH": os.environ.get("PATH", "")},
        home_dir=home,
        fork_state=tmp_path / "fork-state",
        conductor_xdg=tmp_path / "conductor-xdg",
        port=1,
        dep_path=dep_path,
    )
    # base carried no PYTHONPATH: only the built entries reach the child.
    assert dep_path in env["PYTHONPATH"].split(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-c", "import fastapi, requests, sqlalchemy, uvicorn; print('deps-ok')"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "deps-ok" in proc.stdout


def test_absent_companion_skips_actionably(tmp_path, monkeypatch):
    """An explicit conductor root pointing nowhere is a configuration
    error with an actionable skip — never a silent fallback, never a
    crash, and never a guess at the sibling checkout."""
    monkeypatch.setenv("COND0845_CONDUCTOR_ROOT", str(tmp_path / "no-such-clone"))
    assert discover_conductor_root() is None
    # Skipped derives from BaseException, so pytest.raises(Exception)
    # cannot catch it — assert the skip explicitly.
    with pytest.raises(pytest.skip.Exception, match="COND0845_CONDUCTOR_ROOT"):
        _require_conductor_root()


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


# Anchor session for paired-server tmux fixtures. It must NOT match the
# product SESSION_PREFIX ("cao-"): the anchor lives on the owned server,
# and any cao- name would self-pollute the exact `sessions == []`
# isolation assertion (filtering it out is forbidden — operator sessions
# must stay visible to that assertion).
_PAIRED_TMUX_ANCHOR = "paired-harness-anchor"


def _paired_server_extra_env(
    *,
    fork_state: Path,
    dep_path: Optional[str],
    shim_dir: Path,
) -> Dict[str, str]:
    """Extra server-child env for a paired fork server.

    Scratch state root, fork sources + deps, and the owned tmux shim
    first on PATH — every tmux invocation inside the child (libtmux
    resolves its binary via PATH lookup, as does the product client)
    lands on the isolated server, never the ambient default socket.
    """
    parts = [str(FORK_ROOT / "src")]
    if dep_path:
        parts.append(dep_path)
    return {
        "CAO_STATE_ROOT": str(fork_state),
        "PYTHONPATH": os.pathsep.join(parts),
        "PATH": f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    }


def test_paired_server_tmux_wiring_reaches_owned_socket(tmp_path, monkeypatch):
    """Wiring, not ingredients: the exact child env a paired server gets,
    built under a hostile ambient (an inherited operator TMUX socket, a
    shimless PATH), must resolve tmux to the owned shim, name the owned
    socket, and strip every ambient selector — through the real fixture
    merge path. No daemon, no network: PATH lookup is pure."""
    from test.fixtures.cao_server import _subprocess_env
    from test.fixtures.tmux_server import (
        TmuxSelectorLost,
        TmuxServer,
        default_socket_path,
        real_tmux_binary,
    )

    from cli_agent_orchestrator.constants import SESSION_PREFIX

    try:
        real_tmux_binary()
    except TmuxSelectorLost:
        pytest.skip("no tmux binary here; shim-wiring leg runs on host/CI")
    # The anchor must stay outside the product session prefix, or the
    # owned server self-pollutes the exact `sessions == []` assertion.
    assert not _PAIRED_TMUX_ANCHOR.startswith(
        SESSION_PREFIX
    ), f"anchor {_PAIRED_TMUX_ANCHOR!r} matches prefix {SESSION_PREFIX!r}"
    monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,12345,0")
    monkeypatch.setenv("TMUX_PANE", "%0")
    monkeypatch.setenv("TMUX_TMPDIR", "/elsewhere")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    owned = TmuxServer(socket_path=tmp_path / "owned.sock", owned_root=tmp_path)
    shim_dir = tmp_path / "bin"
    owned.write_shim(shim_dir)
    env = _subprocess_env(
        tmp_path / "home",
        1,
        extra=_paired_server_extra_env(
            fork_state=tmp_path / "fork-state",
            dep_path="/dep",
            shim_dir=shim_dir,
        ),
    )
    # The interception primitive libtmux and the product client both use.
    # Spelled split: the module source-scan guard forbids the quoted
    # provider-multiplexer binary name outside the owned-selector fixture.
    mux = "tmu" + "x"
    assert shutil.which(mux, path=env["PATH"]) == str(shim_dir / mux)
    shim_text = (shim_dir / mux).read_text()
    assert f"-S {shlex.quote(str(owned.socket_path))}" in shim_text
    assert owned.socket_path.resolve() != default_socket_path(env).resolve()
    for var in ("TMUX", "TMUX_PANE", "TMUX_TMPDIR"):
        assert var not in env, f"ambient selector {var} leaks to the child"


def test_tmux_teardown_refuses_unowned_socket(tmp_path):
    """Destruction proves its target first: handles without ownership, and
    default-named sockets even with it, raise before running anything.
    Daemon-less by construction — the refusal precedes any subprocess."""
    from test.fixtures.tmux_server import TmuxSelectorLost, TmuxServer

    with pytest.raises(TmuxSelectorLost):
        TmuxServer(socket_path=tmp_path / "shared.sock").teardown()
    with pytest.raises(TmuxSelectorLost):
        TmuxServer(socket_path=tmp_path / "default", owned_root=tmp_path).teardown()


def _prove_paired_server(server, port: int) -> None:
    """Identity proof BEFORE any mutation: the answering server is the
    owned test instance. Uses only the fixture's supported handles: our
    spawned child is alive (``is_alive`` closes over the owned Popen),
    and our own child's log records serving this exact port. A foreign
    listener on the port fails here instead of receiving our writes."""
    assert server.is_alive(), "owned server process already dead"
    log_text = server.log_path.read_text(errors="replace")[-4000:]
    assert (
        f"127.0.0.1:{port}" in log_text or f":{port}" in log_text
    ), f"owned server log names no bind on {port}; refusing mutations"


def test_prove_paired_server_pins_liveness_and_bind_log(tmp_path):
    """Both halves of the pre-mutation identity proof, with no network.

    A dead owned child refuses even when the log names the port, and a
    live child whose log never bound the port refuses mutations. The
    handle is built as a real ``CaoServer`` so drift in the fixture
    interface (a removed ``is_alive``) fails here, not behind a mock.
    """
    from test.fixtures.cao_server import CaoServer

    def _server(alive: bool, log_text: str) -> CaoServer:
        log_path = tmp_path / f"server-{alive}-{len(log_text)}.log"
        log_path.write_text(log_text)
        return CaoServer(
            url="http://127.0.0.1:1",
            port=1,
            home_dir=tmp_path,
            db_path=tmp_path / "db.sqlite",
            log_path=log_path,
            stop=lambda: None,
            is_alive=lambda: alive,
        )

    _prove_paired_server(_server(True, "serving on 127.0.0.1:1\n"), 1)
    with pytest.raises(AssertionError, match="already dead"):
        _prove_paired_server(_server(False, "serving on 127.0.0.1:1\n"), 1)
    with pytest.raises(AssertionError, match="refusing mutations"):
        _prove_paired_server(_server(True, "listening elsewhere\n"), 1)


@needs_loopback
@needs_tmux_sockets
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

    dep_path = _require_fork_dep_path()
    import contextlib
    from test.fixtures.tmux_server import isolated_tmux_server

    # The server child enumerates sessions through libtmux, which resolves
    # its tmux binary via PATH lookup with no socket of its own — without
    # an owned server + shim it lands on the ambient default socket and
    # reads operator sessions (cond-0845 host: cao-cond-0588-*). The
    # ExitStack keeps this lifetime beside the server's own try/finally
    # without re-indenting the whole body.
    _owned_tmux = contextlib.ExitStack()
    tmux = _owned_tmux.enter_context(isolated_tmux_server(anchor=_PAIRED_TMUX_ANCHOR))
    shim_dir = tmp_path / "bin"
    tmux.write_shim(shim_dir)
    try:
        server = _start_cao_server(
            home,
            port,
            extra_env=_paired_server_extra_env(
                fork_state=fork_state, dep_path=dep_path, shim_dir=shim_dir
            ),
            deadline=60.0,
        )
    except BaseException:
        _owned_tmux.close()
        raise
    base = server.url
    try:
        _prove_paired_server(server, port)
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

        root = _require_conductor_root()
        entry = write_conduct_entrypoint(
            tmp_path / "bin", conductor_root=root, python=discover_python()
        )
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

        # Valid-but-unknown TerminalId (^[a-f0-9]{8}$): a prefixed fake
        # dies with 422 at FastAPI validation before the handler, while
        # this leg proves the handler's typed refusal for unknown ids.
        terminal_id = uuid.uuid4().hex[:8]
        refused = requests.post(
            f"{base}/terminals/{terminal_id}/context-restore",
            json={
                "operation_id": f"op-{uuid.uuid4().hex}",
                "occurrence_id": f"occ-{uuid.uuid4().hex}",
                "generation": "gen-1",
                "native_session_id": POSTCOMPACT_STDIN["session_id"],
                "goal_version": 3,
                "hold_high_water": 0,
                "flock_path": str(tmp_path / "fences" / "proj" / "goal-effect.lock"),
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
        _owned_tmux.close()
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
