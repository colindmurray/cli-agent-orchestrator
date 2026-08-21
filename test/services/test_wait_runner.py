"""Red-first tests for wait_runner standalone helper (cond-0535/0536)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cli_agent_orchestrator.services import wait_runner
from cli_agent_orchestrator.services.wait_runner import (
    MAX_TIMEOUT_SECONDS,
    RESULT_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    SPEC_SCHEMA_VERSION,
    TAIL_LIMIT,
    atomic_write_json,
    compute_sha256,
    validate_spec,
)

FIXTURE_REPO = "colindmurray/cli-agent-orchestrator"
FIXTURE_SHA40 = "a" * 40
FIXTURE_DIGEST64 = "b" * 64
CONTROL_SCHEMA_VERSION = "cao-wait-runner-control-v1"


def _write_control(path, action, spec):
    path.write_text(
        json.dumps(
            {
                "schema_version": CONTROL_SCHEMA_VERSION,
                "action": action,
                "wait_id": spec["wait_id"],
                "request_digest": spec["request_digest"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_activate(path, spec):
    _write_control(path, "activate", spec)


def _write_stop(path, spec):
    _write_control(path, "stop", spec)


@pytest.fixture(autouse=True)
def _fake_marker_for_tests(request, monkeypatch):
    if request.node.name == "test_unobservable_start_marker_is_not_fabricated":
        return
    monkeypatch.setattr(wait_runner, "_get_start_marker", lambda pid: f"marker-{pid}")
    monkeypatch.setenv("CAO_WAIT_RUNNER_FAKE_MARKER", "test-marker")


def _write_spec(path: Path, spec: dict) -> None:
    path.write_text(json.dumps(spec) + "\n", encoding="utf-8")


def _make_exe(tmp_path: Path, name: str, body: str) -> tuple[Path, str]:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    p.chmod(0o755)
    return p, compute_sha256(str(p))


def _process_spec(
    tmp_path: Path,
    exe: Path,
    sha: str,
    argv: list[str],
    cwd: str | None = None,
    timeout: int = 5,
    wait_id: str = "wait-1",
    digest: str = FIXTURE_DIGEST64,
) -> dict:
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "wait_id": wait_id,
        "request_digest": digest,
        "timeout_seconds": timeout,
        "adapter": {
            "kind": "process",
            "executable": str(exe),
            "executable_sha256": sha,
            "cwd": cwd or str(tmp_path),
            "argv": argv,
        },
    }


def _github_spec(
    timeout: int = 5,
    repository: str = FIXTURE_REPO,
    run_id: int = 123,
    run_attempt: int = 1,
    workflow_id: int = 456,
    head_sha: str = FIXTURE_SHA40,
    ref: str = "refs/heads/main",
    wait_id: str = "wait-gh-1",
    digest: str = FIXTURE_DIGEST64,
) -> dict:
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "wait_id": wait_id,
        "request_digest": digest,
        "timeout_seconds": timeout,
        "adapter": {
            "kind": "github-actions",
            "repository": repository,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "workflow_id": workflow_id,
            "head_sha": head_sha,
            "ref": ref,
        },
    }


def _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path):
    # Runs wait_runner.main in a thread, returns thread and captures SystemExit
    outcome = {}

    def target():
        try:
            wait_runner.main(
                [
                    "--spec",
                    str(spec_path),
                    "--ready",
                    str(ready_path),
                    "--activate",
                    str(activate_path),
                    "--stop",
                    str(stop_path),
                    "--result",
                    str(result_path),
                ]
            )
        except SystemExit as e:
            outcome["code"] = e.code
        except Exception as e:
            outcome["exc"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, outcome


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_spec_validation_rejects_bad_timeout(tmp_path):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\necho hi\n")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=0)
    with pytest.raises(ValueError):
        validate_spec(spec)
    spec["timeout_seconds"] = MAX_TIMEOUT_SECONDS + 1
    with pytest.raises(ValueError):
        validate_spec(spec)
    spec["timeout_seconds"] = MAX_TIMEOUT_SECONDS
    # should pass
    validate_spec(spec)


def test_spec_validation_rejects_unknown_adapter(tmp_path):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\necho hi\n")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)])
    spec["adapter"]["kind"] = "arbitrary-url"
    with pytest.raises(ValueError):
        validate_spec(spec)


# ---------------------------------------------------------------------------
# atomic writes
# ---------------------------------------------------------------------------


def test_atomic_write_creates_0600_and_valid_json(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_json(target, {"a": 1})
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["a"] == 1
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600
    # sibling tmp not left behind
    assert not list(tmp_path.glob("*.tmp"))


def test_result_restoration_after_exception(tmp_path, monkeypatch):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\nexit 0\n")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=2)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    # Cause compute_sha256 to raise after validation but before spawn,
    # then ensure main still writes a durable result via exception handler.
    # We patch inside run_process_adapter path by making exe unreadable after ready.
    # Instead directly test atomic restoration by patching atomic_write_json to fail once.
    calls = {"n": 0}
    orig = wait_runner.atomic_write_json

    def flaky(path, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            # first call is ready write; let it succeed
            return orig(path, payload)
        if calls["n"] == 2 and payload.get("outcome") is None:
            # simulate failure during first result attempt
            raise RuntimeError("injected failure")
        return orig(path, payload)

    monkeypatch.setattr(wait_runner, "atomic_write_json", flaky)

    # also need to make process adapter raise after first result attempt
    # we will not use flaky for github; use process timeout path with injected popen that raises
    def raising_popen(*a, **kw):
        raise RuntimeError("spawn boom")

    monkeypatch.setattr(wait_runner.subprocess, "Popen", raising_popen)

    t, outcome = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    # activate quickly
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    assert result_path.exists(), "result must be restored even after exception"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == RESULT_SCHEMA_VERSION
    # second attempt should have written valid JSON not partial
    json.loads(result_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# activation gate and stop-before-activation
# ---------------------------------------------------------------------------


def test_no_effect_before_activation(tmp_path):
    marker = tmp_path / "marker"
    exe, sha = _make_exe(
        tmp_path,
        "exe",
        f"#!/usr/bin/env python3\nimport pathlib; pathlib.Path('{marker}').write_text('hit')\n",
    )
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=3)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    t, outcome = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.4)
    # ready should be written with helper pid/marker
    assert ready_path.exists()
    rt = json.loads(ready_path.read_text(encoding="utf-8"))
    assert rt["schema_version"] == RUNTIME_SCHEMA_VERSION
    assert "pid" in rt and "start_marker" in rt
    # marker must not exist because activation not yet
    assert not marker.exists(), "no target effect before activation"
    # now stop before activation
    _write_stop(stop_path, spec)
    t.join(timeout=5)
    assert not t.is_alive()
    assert result_path.exists()
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "interrupted"
    assert "stop-file" in res["reason"]
    assert not marker.exists(), "stop-before-activation must not execute target"


def test_stop_before_activation_interrupted_result(tmp_path):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\necho hi\n")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=3)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)
    # pre-create stop before runner starts
    _write_stop(stop_path, spec)
    t, outcome = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    t.join(timeout=5)
    assert result_path.exists()
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "interrupted"
    assert res["reason"] == "stop-file-before-activation"
    # ensure process adapter fields indicate no execution
    if "process" in res and res["process"]:
        assert res["process"]["interrupted"] is True


# ---------------------------------------------------------------------------
# executable digest drift (recheck)
# ---------------------------------------------------------------------------


def test_executable_digest_drift_prevents_spawn(tmp_path):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\necho original\n")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=3)
    # mutate file after spec creation
    exe.write_text("#!/bin/sh\necho mutated\n", encoding="utf-8")
    exe.chmod(0o755)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)
    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "mismatch"
    assert "digest" in res["reason"]


def test_digest_recheck_immediately_before_spawn(tmp_path, monkeypatch):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\necho hi\n")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=3)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)
    # mutate between first check and recheck: patch compute_sha256 to return different on second call
    calls = {"n": 0}
    orig = wait_runner.compute_sha256

    def patched(path):
        calls["n"] += 1
        if calls["n"] == 1:
            return sha
        return "f" * 64

    monkeypatch.setattr(wait_runner, "compute_sha256", patched)
    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "mismatch"
    assert "recheck" in res["reason"]


# ---------------------------------------------------------------------------
# argv without shell interpretation (real subprocess)
# ---------------------------------------------------------------------------


def test_argv_without_shell_interpretation(tmp_path):
    # exe prints its argv as JSON
    exe, sha = _make_exe(
        tmp_path,
        "exe",
        textwrap.dedent("""\
            #!/usr/bin/env python3
            import sys, json
            print(json.dumps(sys.argv))
            """),
    )
    special = "hello; echo pwned && rm -rf /"
    spec = _process_spec(tmp_path, exe, sha, [str(exe), special, "a b"], timeout=5)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.wait_runner",
            "--spec",
            str(spec_path),
            "--ready",
            str(ready_path),
            "--activate",
            str(activate_path),
            "--stop",
            str(stop_path),
            "--result",
            str(result_path),
        ]
    )
    try:
        time.sleep(0.3)
        assert ready_path.exists()
        _write_activate(activate_path, spec)
        proc.wait(timeout=6)
    finally:
        with open(os.devnull, "w"):
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "completed"
    assert res["process"]["exit_code"] == 0
    stdout = res["process"]["stdout_tail"]
    # stdout should contain literal argv, not shell expanded
    assert special in stdout
    assert "pwned" in stdout  # literal, not executed as separate command
    # ensure no shell interpretation: the file system wasn't mutated by shell


# ---------------------------------------------------------------------------
# stdout + stderr capture, bounded tail, ordinary exit (real subprocess)
# ---------------------------------------------------------------------------


def test_stdout_stderr_capture_and_ordinary_exit(tmp_path):
    exe, sha = _make_exe(
        tmp_path,
        "exe",
        textwrap.dedent("""\
            #!/usr/bin/env python3
            import sys
            print("out-line")
            print("err-line", file=sys.stderr)
            sys.exit(7)
            """),
    )
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=5)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.wait_runner",
            "--spec",
            str(spec_path),
            "--ready",
            str(ready_path),
            "--activate",
            str(activate_path),
            "--stop",
            str(stop_path),
            "--result",
            str(result_path),
        ]
    )
    try:
        time.sleep(0.3)
        _write_activate(activate_path, spec)
        proc.wait(timeout=6)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            pass

    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "completed"
    assert res["process"]["exit_code"] == 7
    assert "out-line" in res["process"]["stdout_tail"]
    assert "err-line" in res["process"]["stderr_tail"]
    assert res["process"]["stdout_truncated"] is False
    assert res["process"]["stderr_truncated"] is False
    assert "elapsed_seconds" in res and res["elapsed_seconds"] >= 0
    assert "started_at" in res and "finished_at" in res


def test_bounded_tail_truncation(tmp_path):
    # generate > TAIL_LIMIT output
    exe, sha = _make_exe(
        tmp_path,
        "exe",
        textwrap.dedent("""\
            #!/usr/bin/env python3
            import sys
            chunk = "X" * 1024
            for _ in range(100):
                print(chunk)
                print(chunk, file=sys.stderr)
            """),
    )
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=5)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.wait_runner",
            "--spec",
            str(spec_path),
            "--ready",
            str(ready_path),
            "--activate",
            str(activate_path),
            "--stop",
            str(stop_path),
            "--result",
            str(result_path),
        ]
    )
    try:
        time.sleep(0.3)
        _write_activate(activate_path, spec)
        proc.wait(timeout=6)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            pass

    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "completed"
    assert res["process"]["stdout_truncated"] is True
    assert res["process"]["stderr_truncated"] is True
    assert len(res["process"]["stdout_tail"]) == TAIL_LIMIT
    assert len(res["process"]["stderr_tail"]) == TAIL_LIMIT


# ---------------------------------------------------------------------------
# timeout with process-group termination (real subprocess)
# ---------------------------------------------------------------------------


def test_timeout_with_process_group_termination(tmp_path):
    # exe spawns a child sleep and then sleeps itself; runner must kill pg
    exe, sha = _make_exe(
        tmp_path,
        "exe",
        textwrap.dedent("""\
            #!/usr/bin/env python3
            import subprocess, time
            # spawn grandchild that would outlive parent if only parent killed
            subprocess.Popen(["sleep", "10"])
            time.sleep(10)
            """),
    )
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=2)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    start = time.monotonic()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.wait_runner",
            "--spec",
            str(spec_path),
            "--ready",
            str(ready_path),
            "--activate",
            str(activate_path),
            "--stop",
            str(stop_path),
            "--result",
            str(result_path),
        ]
    )
    try:
        time.sleep(0.3)
        _write_activate(activate_path, spec)
        proc.wait(timeout=6)
        elapsed = time.monotonic() - start
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            pass

    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "timeout"
    assert res["process"]["timed_out"] is True
    assert res["elapsed_seconds"] >= 1.5
    assert elapsed < 5, "should timeout near 2s not wait full 10s"
    # runtime should have been refreshed with child pgid
    rt = json.loads(ready_path.read_text(encoding="utf-8"))
    assert "pgid" in rt
    assert "child_pid" in rt
    assert "child_start_marker" in rt
    # child process group must have been terminated (PID reuse guard)
    child_pid = rt["child_pid"]
    # child should be dead after timeout via pg termination; allow brief race
    time.sleep(0.2)
    try:
        os.kill(child_pid, 0)
        # if still alive, try pg
        try:
            os.killpg(rt["pgid"], 0)
            still_alive = True
        except (ProcessLookupError, PermissionError):
            still_alive = False
        # if still alive, fail - termination was not performed
        assert not still_alive, "child pg should be terminated on timeout"
    except ProcessLookupError:
        pass  # expected dead
    except PermissionError:
        pass


# ---------------------------------------------------------------------------
# signal interruption (real subprocess)
# ---------------------------------------------------------------------------


def test_signal_interruption(tmp_path):
    exe, sha = _make_exe(
        tmp_path,
        "exe",
        textwrap.dedent("""\
            #!/usr/bin/env python3
            import time
            time.sleep(10)
            """),
    )
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=8)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.wait_runner",
            "--spec",
            str(spec_path),
            "--ready",
            str(ready_path),
            "--activate",
            str(activate_path),
            "--stop",
            str(stop_path),
            "--result",
            str(result_path),
        ]
    )
    try:
        time.sleep(0.4)
        _write_activate(activate_path, spec)
        time.sleep(0.4)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=6)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            pass

    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "interrupted"
    assert "signal" in res["reason"]
    assert res["process"]["interrupted"] is True if "process" in res else True


# ---------------------------------------------------------------------------
# minimal env (HOME, PATH only)
# ---------------------------------------------------------------------------


def test_minimal_env_only_home_path(tmp_path, monkeypatch):
    # exe prints its env keys
    exe, sha = _make_exe(
        tmp_path,
        "exe",
        textwrap.dedent("""\
            #!/usr/bin/env python3
            import os
            print(",".join(sorted(os.environ.keys())))
            """),
    )
    monkeypatch.setenv("HOME", "/tmp/fakehome")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("SHOULD_NOT_LEAK", "secret123")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=5)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.wait_runner",
            "--spec",
            str(spec_path),
            "--ready",
            str(ready_path),
            "--activate",
            str(activate_path),
            "--stop",
            str(stop_path),
            "--result",
            str(result_path),
        ]
    )
    try:
        time.sleep(0.3)
        _write_activate(activate_path, spec)
        proc.wait(timeout=6)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            pass

    res = json.loads(result_path.read_text(encoding="utf-8"))
    stdout = res["process"]["stdout_tail"]
    assert "HOME" in stdout
    assert "PATH" in stdout
    assert "SHOULD_NOT_LEAK" not in stdout


# ---------------------------------------------------------------------------
# GitHub adapter tests (injected fetch, no network)
# ---------------------------------------------------------------------------


def test_github_completed_success(tmp_path, monkeypatch):

    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=5)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    def fake_fetch(url, token):
        assert url == f"https://api.github.com/repos/{FIXTURE_REPO}/actions/runs/123"
        assert token == "fake-token"
        return {
            "status_code": 200,
            "data": {
                "status": "completed",
                "conclusion": "success",
                "head_sha": FIXTURE_SHA40,
                "workflow_id": 456,
                "id": 123,
                "run_attempt": 1,
                "repository": {"full_name": FIXTURE_REPO},
                "head_branch": "main",
            },
            "error": None,
        }

    def fake_token():
        return "fake-token"

    monkeypatch.setattr(wait_runner, "default_fetch_github", fake_fetch)
    monkeypatch.setattr(wait_runner, "default_gh_token", fake_token)

    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "completed"
    assert res["github"]["observed_conclusion"] == "success"
    assert res["github"]["http_status"] == 200
    assert res["github"]["last_observation_type"] == "completed"


def test_github_completed_failure(tmp_path, monkeypatch):

    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=5)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    def fake_fetch(url, token):
        return {
            "status_code": 200,
            "data": {
                "status": "completed",
                "conclusion": "failure",
                "head_sha": FIXTURE_SHA40,
                "workflow_id": 456,
                "id": 123,
                "run_attempt": 1,
                "repository": {"full_name": FIXTURE_REPO},
                "head_branch": "main",
            },
            "error": None,
        }

    monkeypatch.setattr(wait_runner, "default_fetch_github", fake_fetch)
    monkeypatch.setattr(wait_runner, "default_gh_token", lambda: "tok")

    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "completed"
    assert res["github"]["observed_conclusion"] == "failure"


def test_github_exact_identity_mismatch(tmp_path, monkeypatch):

    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=5, head_sha=FIXTURE_SHA40)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    def fake_fetch(url, token):
        return {
            "status_code": 200,
            "data": {
                "status": "completed",
                "conclusion": "success",
                "head_sha": "c" * 40,  # mismatch
                "workflow_id": 456,
                "id": 123,
                "run_attempt": 1,
                "repository": {"full_name": FIXTURE_REPO},
                "head_branch": "main",
            },
            "error": None,
        }

    monkeypatch.setattr(wait_runner, "default_fetch_github", fake_fetch)
    monkeypatch.setattr(wait_runner, "default_gh_token", lambda: "tok")

    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "mismatch"
    assert "head_sha" in res["reason"]


def test_github_absent_404_distinct_from_unreadable(tmp_path, monkeypatch):

    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=4)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    def fake_fetch_404(url, token):
        if url == f"https://api.github.com/repos/{FIXTURE_REPO}":
            return {"status_code": 200, "data": {"full_name": FIXTURE_REPO}, "error": None}
        return {"status_code": 404, "data": None, "error": "absent"}

    monkeypatch.setattr(wait_runner, "default_fetch_github", fake_fetch_404)
    monkeypatch.setattr(wait_runner, "default_gh_token", lambda: "tok")

    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "absent"
    assert res["github"]["http_status"] == 404

    # now unreadable should be different outcome
    result_path.unlink()
    ready_path.unlink(missing_ok=True)
    # reset stop/activate for second run
    activate_path.unlink(missing_ok=True)
    stop_path.unlink(missing_ok=True)
    spec["wait_id"] = "wait-gh-2"
    _write_spec(spec_path, spec)

    def fake_fetch_unreadable(url, token):
        return {"status_code": 500, "data": None, "error": "unreadable"}

    # make it timeout quickly with unreadable last observation
    spec["timeout_seconds"] = 1
    _write_spec(spec_path, spec)
    monkeypatch.setattr(wait_runner, "default_fetch_github", fake_fetch_unreadable)

    t2, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t2.join(timeout=5)
    res2 = json.loads(result_path.read_text(encoding="utf-8"))
    assert res2["outcome"] == "timeout"
    assert res2["github"]["last_observation_type"] == "unreadable"
    assert res["outcome"] != res2["outcome"], "absent and unreadable must be distinct"


def test_github_transient_unreadable_then_success(tmp_path, monkeypatch):

    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=5)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    calls = {"n": 0}

    def fake_fetch(url, token):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status_code": 500, "data": None, "error": "unreadable"}
        return {
            "status_code": 200,
            "data": {
                "status": "completed",
                "conclusion": "success",
                "head_sha": FIXTURE_SHA40,
                "workflow_id": 456,
                "id": 123,
                "run_attempt": 1,
                "repository": {"full_name": FIXTURE_REPO},
                "head_branch": "main",
            },
            "error": None,
        }

    monkeypatch.setattr(wait_runner, "default_fetch_github", fake_fetch)
    monkeypatch.setattr(wait_runner, "default_gh_token", lambda: "tok")

    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "completed"
    assert calls["n"] >= 2


def test_github_timeout_retains_last_typed_observation(tmp_path, monkeypatch):

    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=2)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    def fake_fetch(url, token):
        return {
            "status_code": 200,
            "data": {
                "status": "in_progress",
                "conclusion": None,
                "head_sha": FIXTURE_SHA40,
                "workflow_id": 456,
                "id": 123,
                "run_attempt": 1,
                "repository": {"full_name": FIXTURE_REPO},
                "head_branch": "main",
            },
            "error": None,
        }

    monkeypatch.setattr(wait_runner, "default_fetch_github", fake_fetch)
    monkeypatch.setattr(wait_runner, "default_gh_token", lambda: "tok")

    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["outcome"] == "timeout"
    assert res["github"]["last_observation_type"] == "in_progress"
    assert res["elapsed_seconds"] >= 1.5


def test_github_uses_fixed_api_url(tmp_path, monkeypatch):

    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=3, repository="owner/repo", run_id=999)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    captured = {}

    def fake_fetch(url, token):
        captured["url"] = url
        # return mismatch to finish quickly
        return {
            "status_code": 200,
            "data": {
                "status": "completed",
                "conclusion": "success",
                "head_sha": FIXTURE_SHA40,
                "workflow_id": 456,
                "id": 999,
                "run_attempt": 1,
                "repository": {"full_name": "owner/repo"},
                "head_branch": "main",
            },
            "error": None,
        }

    monkeypatch.setattr(wait_runner, "default_fetch_github", fake_fetch)
    monkeypatch.setattr(wait_runner, "default_gh_token", lambda: "tok")

    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    assert captured["url"] == "https://api.github.com/repos/owner/repo/actions/runs/999"


def test_github_token_not_logged(tmp_path, monkeypatch, capsys):

    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=2)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    secret = "ghp_super_secret_token_123"

    def fake_fetch(url, token):
        assert token == secret
        return {
            "status_code": 200,
            "data": {
                "status": "completed",
                "conclusion": "success",
                "head_sha": FIXTURE_SHA40,
                "workflow_id": 456,
                "id": 123,
                "run_attempt": 1,
                "repository": {"full_name": FIXTURE_REPO},
                "head_branch": "main",
            },
            "error": None,
        }

    monkeypatch.setattr(wait_runner, "default_fetch_github", fake_fetch)
    monkeypatch.setattr(wait_runner, "default_gh_token", lambda: secret)

    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=5)
    res_text = result_path.read_text(encoding="utf-8")
    assert secret not in res_text
    # also check stdout/stderr not leaked
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


# ---------------------------------------------------------------------------
# result durability: self-contained for later attachment capture
# ---------------------------------------------------------------------------


def test_result_is_versioned_and_self_contained(tmp_path):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\nexit 0\n")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=5)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.wait_runner",
            "--spec",
            str(spec_path),
            "--ready",
            str(ready_path),
            "--activate",
            str(activate_path),
            "--stop",
            str(stop_path),
            "--result",
            str(result_path),
        ]
    )
    try:
        time.sleep(0.3)
        _write_activate(activate_path, spec)
        proc.wait(timeout=6)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            pass

    res = json.loads(result_path.read_text(encoding="utf-8"))
    assert res["schema_version"] == RESULT_SCHEMA_VERSION
    assert res["wait_id"] == spec["wait_id"]
    assert res["request_digest"] == spec["request_digest"]
    assert res["adapter"]["kind"] == "process"
    assert "started_at" in res and "finished_at" in res and "elapsed_seconds" in res
    assert "process" in res and "exit_code" in res["process"]
    # durable implies result exists after runner exits, no DB or inbox writes attempted
    assert result_path.exists()


# ---------------------------------------------------------------------------
# Coordinator repair reproductions: exact effect, identity, and round bounds
# ---------------------------------------------------------------------------


def test_process_argv_cannot_select_a_different_executable(tmp_path):
    declared, declared_sha = _make_exe(tmp_path, "declared", "#!/bin/sh\nexit 0\n")
    other, _ = _make_exe(tmp_path, "other", "#!/bin/sh\nexit 0\n")
    spec = _process_spec(tmp_path, declared, declared_sha, [str(other)])

    with pytest.raises(ValueError, match="argv"):
        validate_spec(spec)


def test_unobservable_start_marker_is_not_fabricated(monkeypatch):
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no proc")),
    )
    monkeypatch.setattr(
        wait_runner.subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no ps")),
    )

    assert wait_runner._get_start_marker(12345) is None


def test_elapsed_activation_time_cannot_extend_the_process_round(tmp_path):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\nexit 0\n")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=5)
    result_path = tmp_path / "result.json"
    spawned = []

    def forbidden_spawn(*args, **kwargs):
        spawned.append((args, kwargs))
        raise AssertionError("an expired round must not spawn")

    wait_runner.run_process_adapter(
        spec,
        tmp_path / "ready.json",
        result_path,
        tmp_path / "stop.json",
        "2026-08-20T00:00:00Z",
        10.0,
        lambda: False,
        lambda: None,
        123,
        "helper-marker",
        clock_monotonic=lambda: 16.0,
        sleep_fn=lambda _seconds: None,
        popen_cls=forbidden_spawn,
    )

    assert spawned == []
    assert json.loads(result_path.read_text(encoding="utf-8"))["outcome"] == "timeout"


def test_control_file_must_bind_action_wait_and_request(tmp_path):
    marker = tmp_path / "marker"
    exe, sha = _make_exe(
        tmp_path,
        "exe",
        f"#!/usr/bin/env python3\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
    )
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=3)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate.json"
    stop_path = tmp_path / "stop.json"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)
    activate_path.write_text(
        json.dumps(
            {
                "schema_version": "cao-wait-runner-control-v1",
                "action": "activate",
                "wait_id": "a-different-wait",
                "request_digest": spec["request_digest"],
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.wait_runner",
            "--spec",
            str(spec_path),
            "--ready",
            str(ready_path),
            "--activate",
            str(activate_path),
            "--stop",
            str(stop_path),
            "--result",
            str(result_path),
        ]
    )
    try:
        time.sleep(0.4)
        assert not marker.exists(), "a stale/foreign activation must have zero effect"
    finally:
        stop_path.write_text(
            json.dumps(
                {
                    "schema_version": "cao-wait-runner-control-v1",
                    "action": "stop",
                    "wait_id": spec["wait_id"],
                    "request_digest": spec["request_digest"],
                }
            ),
            encoding="utf-8",
        )
        proc.wait(timeout=5)


def test_completed_github_observation_missing_identity_is_inconclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=1)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    def missing_identity(_url, _token):
        return {
            "status_code": 200,
            "data": {"status": "completed", "conclusion": "success"},
            "error": None,
        }

    monkeypatch.setattr(wait_runner, "default_fetch_github", missing_identity)
    monkeypatch.setattr(wait_runner, "default_gh_token", lambda: "token")
    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=4)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["outcome"] != "completed"
    assert result["github"]["last_observation_type"] == "inconclusive"


def test_github_404_is_absent_only_after_repository_access_is_proven(tmp_path, monkeypatch):
    monkeypatch.setattr(wait_runner, "GITHUB_POLL_INTERVAL", 0.05)
    spec = _github_spec(timeout=1)
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate"
    stop_path = tmp_path / "stop"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)

    def concealed_or_absent(url, _token):
        if url.endswith("/actions/runs/123"):
            return {"status_code": 404, "data": None, "error": "absent"}
        return {"status_code": 404, "data": None, "error": "absent"}

    monkeypatch.setattr(wait_runner, "default_fetch_github", concealed_or_absent)
    monkeypatch.setattr(wait_runner, "default_gh_token", lambda: "token")
    t, _ = _run_main_thread(spec_path, ready_path, activate_path, stop_path, result_path)
    time.sleep(0.2)
    _write_activate(activate_path, spec)
    t.join(timeout=4)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["outcome"] != "absent"
    assert result["github"]["last_observation_type"] == "unreadable"


def test_failed_group_signal_is_not_reported_as_terminated(monkeypatch):
    monkeypatch.setattr(
        wait_runner.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError("not observable")),
    )

    assert wait_runner._terminate_pgid(12345) is False


# ---------------------------------------------------------------------------
# Coordinator adjudication coverage: exercise the compact in-process core.
# ---------------------------------------------------------------------------


def test_validation_boundaries_reject_malformed_identity(tmp_path):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\nexit 0\n")

    def invalid_process(field, value):
        adapter = _process_spec(tmp_path, exe, sha, [str(exe)])["adapter"]
        if field is None:
            adapter.pop("cwd")
        else:
            adapter[field] = value
        with pytest.raises(ValueError):
            wait_runner.validate_process_adapter(adapter)

    for field, value in (
        (None, None),
        ("executable", "relative"),
        ("executable_sha256", "bad"),
        ("cwd", "relative"),
        ("argv", []),
    ):
        invalid_process(field, value)

    def invalid_github(field, value):
        adapter = _github_spec()["adapter"]
        if field is None:
            adapter.pop("ref")
        else:
            adapter[field] = value
        with pytest.raises(ValueError):
            wait_runner.validate_github_adapter(adapter)

    for field, value in (
        (None, None),
        ("repository", "not-a-slug"),
        ("run_id", True),
        ("head_sha", "bad"),
        ("ref", "main"),
    ):
        invalid_github(field, value)

    valid = _process_spec(tmp_path, exe, sha, [str(exe)])
    for candidate in (
        None,
        {**valid, "extra": True},
        {**valid, "schema_version": "old"},
        {**valid, "wait_id": ""},
        {**valid, "request_digest": "bad"},
        {**valid, "adapter": []},
    ):
        with pytest.raises(ValueError):
            validate_spec(candidate)

    control = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "action": "activate",
        "wait_id": "wait-1",
        "request_digest": FIXTURE_DIGEST64,
    }
    for candidate in (
        {},
        {**control, "schema_version": "old"},
        {**control, "action": "resume"},
        {**control, "wait_id": ""},
        {**control, "request_digest": "bad"},
    ):
        with pytest.raises(ValueError):
            wait_runner.validate_control(candidate)


def test_default_github_transport_types_http_and_decode_failures(monkeypatch):
    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.body

    monkeypatch.setattr(
        wait_runner.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b'{"status":"completed"}'),
    )
    assert (
        wait_runner.default_fetch_github("https://example.invalid", "secret")["status_code"] == 200
    )

    def http(code):
        return urllib.error.HTTPError("https://example.invalid", code, "error", None, None)

    for code, expected in ((404, "absent"), (403, "unreadable"), (500, "unreadable")):
        monkeypatch.setattr(
            wait_runner.urllib.request,
            "urlopen",
            lambda *_args, _error=http(code), **_kwargs: (_ for _ in ()).throw(_error),
        )
        assert (
            wait_runner.default_fetch_github("https://example.invalid", "secret")["error"]
            == expected
        )

    for error in (
        urllib.error.URLError("offline"),
        json.JSONDecodeError("bad", "{", 0),
    ):
        if isinstance(error, json.JSONDecodeError):
            replacement = lambda *_args, **_kwargs: Response(b"{")
        else:
            replacement = lambda *_args, _error=error, **_kwargs: (_ for _ in ()).throw(_error)
        monkeypatch.setattr(wait_runner.urllib.request, "urlopen", replacement)
        assert (
            wait_runner.default_fetch_github("https://example.invalid", "secret")["error"]
            == "unreadable"
        )


def test_tail_collector_bounds_bytes_and_tolerates_closed_pipe():
    collector = wait_runner.TailCollector(io.BytesIO(b"abcdef"), limit=3)
    collector.start()
    collector.join(1)
    assert collector.buf == "def"
    assert collector.truncated is True

    class Closed:
        def read(self, _amount):
            raise ValueError("closed")

    closed = wait_runner.TailCollector(Closed(), limit=3)
    closed.start()
    closed.join(1)
    assert closed.buf == ""


def test_termination_distinguishes_absence_sigkill_and_failed_sigkill(monkeypatch):
    monkeypatch.setattr(
        wait_runner.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert wait_runner._terminate_pgid(99) is True

    class Reaped:
        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(wait_runner.os, "killpg", lambda *_args: None)
    assert wait_runner._terminate_pgid(99, Reaped(), grace=0, sleep_fn=lambda _s: None)

    def deny_kill(_pgid, signum):
        if signum == signal.SIGKILL:
            raise PermissionError("denied")

    monkeypatch.setattr(wait_runner.os, "killpg", deny_kill)
    assert not wait_runner._terminate_pgid(99, grace=0, sleep_fn=lambda _s: None)


def test_process_adapter_real_success_is_covered_in_process(tmp_path):
    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\nprintf covered-success\n")
    spec = _process_spec(tmp_path, exe, sha, [str(exe)], timeout=3)
    started = time.monotonic()
    wait_runner.run_process_adapter(
        spec,
        tmp_path / "ready.json",
        tmp_path / "result.json",
        tmp_path / "stop.json",
        datetime.now(timezone.utc).isoformat(),
        started,
        lambda: False,
        lambda: None,
        os.getpid(),
        "helper-marker",
        get_marker=lambda pid: f"child-{pid}",
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["outcome"] == "completed"
    assert result["process"]["exit_code"] == 0
    assert "covered-success" in result["process"]["stdout_tail"]


def test_github_adapter_stop_signal_token_and_fetch_failures(tmp_path):
    spec = _github_spec(timeout=1)
    started_at = "2026-08-20T00:00:00Z"

    def run(name, interrupted, fetch, token, clock=lambda: 0.0, stop=False):
        result = tmp_path / f"{name}.json"
        stop_path = tmp_path / f"{name}.stop"
        if stop:
            _write_stop(stop_path, spec)
        wait_runner.run_github_adapter(
            spec,
            tmp_path / f"{name}.ready",
            result,
            stop_path,
            started_at,
            0.0,
            interrupted,
            lambda: signal.SIGTERM,
            1,
            "helper",
            clock_monotonic=clock,
            sleep_fn=lambda _seconds: None,
            fetch_fn=fetch,
            gh_token_fn=token,
            poll_interval=0.01,
        )
        return json.loads(result.read_text())

    assert run("signal", lambda: True, lambda *_a: {}, lambda: "t")["outcome"] == "interrupted"
    assert (
        run("stop", lambda: False, lambda *_a: {}, lambda: "t", stop=True)["reason"] == "stop-file"
    )
    assert (
        run(
            "token",
            lambda: False,
            lambda *_a: {},
            lambda: (_ for _ in ()).throw(RuntimeError("no token")),
            clock=lambda: 2.0,
        )["github"]["last_observation_type"]
        == "unreadable"
    )

    ticks = iter((0.0, 0.0, 2.0, 2.0, 2.0))
    assert (
        run(
            "fetch",
            lambda: False,
            lambda *_a: (_ for _ in ()).throw(OSError("offline")),
            lambda: "t",
            clock=lambda: next(ticks, 2.0),
        )["github"]["last_observation_type"]
        == "unreadable"
    )


def test_main_types_invalid_spec_and_unobservable_helper(tmp_path, monkeypatch):
    def argv(spec_path, result_path):
        return [
            "--spec",
            str(spec_path),
            "--ready",
            str(tmp_path / "ready"),
            "--activate",
            str(tmp_path / "activate"),
            "--stop",
            str(tmp_path / "stop"),
            "--result",
            str(result_path),
        ]

    bad_spec = tmp_path / "bad.json"
    bad_result = tmp_path / "bad-result.json"
    bad_spec.write_text("{}")
    with pytest.raises(SystemExit) as error:
        wait_runner.main(argv(bad_spec, bad_result))
    assert error.value.code == 2
    assert json.loads(bad_result.read_text())["outcome"] == "invalid_spec"

    exe, sha = _make_exe(tmp_path, "exe", "#!/bin/sh\nexit 0\n")
    good_spec = tmp_path / "good.json"
    good_result = tmp_path / "good-result.json"
    _write_spec(good_spec, _process_spec(tmp_path, exe, sha, [str(exe)]))
    monkeypatch.setattr(wait_runner, "_get_start_marker", lambda _pid: None)
    with pytest.raises(SystemExit) as error:
        wait_runner.main(argv(good_spec, good_result))
    assert error.value.code == 1
    assert json.loads(good_result.read_text())["reason"] == "helper-start-marker-unavailable"


def test_stop_wins_when_activate_and_stop_are_both_present(tmp_path):
    marker = tmp_path / "effect"
    exe, sha = _make_exe(
        tmp_path,
        "exe",
        f"#!/bin/sh\necho ran > {marker}\n",
    )
    spec = _process_spec(tmp_path, exe, sha, [str(exe)])
    spec_path = tmp_path / "spec.json"
    ready_path = tmp_path / "ready.json"
    activate_path = tmp_path / "activate.json"
    stop_path = tmp_path / "stop.json"
    result_path = tmp_path / "result.json"
    _write_spec(spec_path, spec)
    _write_activate(activate_path, spec)
    _write_stop(stop_path, spec)

    thread, _outcome = _run_main_thread(
        spec_path, ready_path, activate_path, stop_path, result_path
    )
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert not marker.exists()
    result = json.loads(result_path.read_text())
    assert result["outcome"] == "interrupted"
    assert result["reason"] == "stop-file-before-activation"
