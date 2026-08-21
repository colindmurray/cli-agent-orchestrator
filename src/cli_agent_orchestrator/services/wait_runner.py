"""Run one bounded, identity-bound wait adapter as a standalone process."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

SPEC_SCHEMA_VERSION = "cao-wait-runner-v1"
CONTROL_SCHEMA_VERSION = "cao-wait-runner-control-v1"
RUNTIME_SCHEMA_VERSION = "cao-wait-runner-runtime-v1"
RESULT_SCHEMA_VERSION = "cao-wait-runner-result-v1"
MAX_TIMEOUT_SECONDS = 8 * 60 * 60
TAIL_LIMIT = 32 * 1024
ACTIVATION_POLL_INTERVAL = 0.1
GITHUB_POLL_INTERVAL = 30.0
CHILD_GRACE_SECONDS = 2.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clock_monotonic() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _get_start_marker(pid: int) -> Optional[str]:
    fake = os.environ.get("CAO_WAIT_RUNNER_FAKE_MARKER")
    if fake is not None:
        return f"{fake}-{pid}"
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        return fields[19] if len(fields) > 19 and fields[19] else None
    except (OSError, IndexError):
        pass
    try:
        marker = subprocess.check_output(
            ["ps", "-o", "lstart=", "-p", str(pid)], text=True, timeout=2
        ).strip()
        return marker or None
    except (OSError, subprocess.SubprocessError):
        return None


def compute_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path | str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _strict_int(value: Any) -> bool:
    return type(value) is int


def validate_process_adapter(adapter: dict[str, Any]) -> None:
    required = {"kind", "executable", "executable_sha256", "cwd", "argv"}
    if set(adapter) != required:
        raise ValueError(f"process adapter must have exactly keys {required}")
    executable = adapter["executable"]
    if not isinstance(executable, str) or not executable.startswith("/"):
        raise ValueError("process.executable must be an absolute path")
    digest = adapter["executable_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("process.executable_sha256 must be 64 lowercase hex characters")
    cwd = adapter["cwd"]
    if not isinstance(cwd, str) or not cwd.startswith("/"):
        raise ValueError("process.cwd must be an absolute path")
    argv = adapter["argv"]
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("process.argv must be a non-empty list of strings")
    if argv[0] != executable:
        raise ValueError("process.argv[0] must equal the digest-bound executable")


def validate_github_adapter(adapter: dict[str, Any]) -> None:
    required = {
        "kind",
        "repository",
        "run_id",
        "run_attempt",
        "workflow_id",
        "head_sha",
        "ref",
    }
    if set(adapter) != required:
        raise ValueError(f"github-actions adapter must have exactly keys {required}")
    repository = adapter["repository"]
    if (
        not isinstance(repository, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None
    ):
        raise ValueError("github repository must be an owner/repository slug")
    for field in ("run_id", "run_attempt", "workflow_id"):
        if not _strict_int(adapter[field]) or adapter[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if (
        not isinstance(adapter["head_sha"], str)
        or re.fullmatch(r"[0-9a-f]{40}", adapter["head_sha"]) is None
    ):
        raise ValueError("head_sha must be 40 lowercase hex characters")
    if not isinstance(adapter["ref"], str) or not adapter["ref"].startswith("refs/"):
        raise ValueError("ref must start with refs/")


def validate_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("spec must be an object")
    allowed = {"schema_version", "wait_id", "request_digest", "timeout_seconds", "adapter"}
    if set(value) != allowed:
        raise ValueError(f"spec must have exactly keys {allowed}")
    if value["schema_version"] != SPEC_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SPEC_SCHEMA_VERSION}")
    if not isinstance(value["wait_id"], str) or not value["wait_id"]:
        raise ValueError("wait_id must be a non-empty string")
    digest = value["request_digest"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("request_digest must be 64 lowercase hex characters")
    timeout = value["timeout_seconds"]
    if not _strict_int(timeout) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be an integer in 1..{MAX_TIMEOUT_SECONDS}")
    adapter = value["adapter"]
    if not isinstance(adapter, dict):
        raise ValueError("adapter must be an object")
    if adapter.get("kind") == "process":
        validate_process_adapter(adapter)
    elif adapter.get("kind") == "github-actions":
        validate_github_adapter(adapter)
    else:
        raise ValueError("adapter kind must be process or github-actions")
    return value


def validate_control(value: Any) -> dict[str, Any]:
    required = {"schema_version", "action", "wait_id", "request_digest"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"control must have exactly keys {required}")
    if value["schema_version"] != CONTROL_SCHEMA_VERSION:
        raise ValueError(f"control schema_version must be {CONTROL_SCHEMA_VERSION}")
    if value["action"] not in {"activate", "stop"}:
        raise ValueError("control action must be activate or stop")
    if not isinstance(value["wait_id"], str) or not value["wait_id"]:
        raise ValueError("control wait_id must be a non-empty string")
    digest = value["request_digest"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("control request_digest must be 64 lowercase hex characters")
    return value


def _is_control(path: Path, spec: dict[str, Any], action: str) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            control = validate_control(json.load(handle))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (
        control["action"] == action
        and control["wait_id"] == spec["wait_id"]
        and control["request_digest"] == spec["request_digest"]
    )


def _minimal_env() -> dict[str, str]:
    return {key: os.environ[key] for key in ("HOME", "PATH") if key in os.environ}


def github_api_url(repository: str, run_id: int) -> str:
    return f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"


def github_repo_url(repository: str) -> str:
    return f"https://api.github.com/repos/{repository}"


def default_gh_token() -> str:
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("gh auth token unavailable")
    return result.stdout.strip()


def default_fetch_github(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "cao-wait-runner",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            return {
                "status_code": response.status,
                "data": json.loads(body) if body else {},
                "error": None,
            }
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {"status_code": 404, "data": None, "error": "absent"}
        reason = f"auth-{error.code}" if error.code in {401, 403} else f"http-{error.code}"
        return {
            "status_code": error.code,
            "data": None,
            "error": "unreadable",
            "reason": reason,
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        return {
            "status_code": None,
            "data": None,
            "error": "unreadable",
            "reason": f"transport-{type(error).__name__}",
        }


class TailCollector:
    def __init__(self, pipe: Any, limit: int = TAIL_LIMIT):
        self.pipe = pipe
        self.limit = limit
        self.buf = ""
        self.truncated = False
        self._thread = threading.Thread(target=self._collect, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    def _collect(self) -> None:
        while True:
            try:
                chunk = self.pipe.read(4096)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
            self.buf += chunk
            if len(self.buf) > self.limit:
                self.buf = self.buf[-self.limit :]
                self.truncated = True


def _group_absent(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return False
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False


def _terminate_pgid(
    pgid: int,
    proc: Any | None = None,
    grace: float = CHILD_GRACE_SECONDS,
    sleep_fn: Callable[[float], None] | None = None,
    clock_monotonic: Callable[[], float] | None = None,
) -> bool:
    sleep_fn = sleep_fn or _sleep
    clock_monotonic = clock_monotonic or _clock_monotonic
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    deadline = clock_monotonic() + grace
    while clock_monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            proc.wait()
        if _group_absent(pgid):
            return True
        sleep_fn(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    for _ in range(5):
        sleep_fn(0.05)
        if proc is not None and proc.poll() is not None:
            proc.wait()
            # A successfully delivered group SIGKILL plus a reaped leader
            # proves that no member observed in this group can perform a late
            # effect.  A zombie grandchild may keep the numeric pgid visible
            # briefly while its new parent reaps it, but cannot execute.
            return True
        if _group_absent(pgid):
            return True
    return False


def _write_runtime(
    path: Path,
    spec: dict[str, Any],
    helper_pid: int,
    helper_marker: str,
    phase: str,
    started_at: str,
    **identity: Any,
) -> None:
    payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "wait_id": spec["wait_id"],
        "request_digest": spec["request_digest"],
        "pid": helper_pid,
        "start_marker": helper_marker,
        "phase": phase,
        "started_at": started_at,
        "adapter": spec["adapter"]["kind"],
        "timeout_seconds": spec["timeout_seconds"],
    }
    payload.update({key: value for key, value in identity.items() if value is not None})
    atomic_write_json(path, payload)


def _build_result(
    spec: dict[str, Any],
    started_at: str,
    started_monotonic: float,
    outcome: str,
    reason: str,
    observed: dict[str, Any],
    clock_monotonic: Callable[[], float],
    **section: Any,
) -> dict[str, Any]:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "wait_id": spec.get("wait_id"),
        "request_digest": spec.get("request_digest"),
        "adapter": spec.get("adapter"),
        "outcome": outcome,
        "reason": reason,
        "observed": observed,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "elapsed_seconds": max(0.0, clock_monotonic() - started_monotonic),
    }
    payload.update(section)
    return payload


def _process_fields(
    adapter: dict[str, Any],
    *,
    exit_code: Optional[int] = None,
    stdout_tail: str = "",
    stderr_tail: str = "",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    timed_out: bool = False,
    interrupted: bool = False,
    **identity: Any,
) -> dict[str, Any]:
    fields = {
        "executable": adapter["executable"],
        "cwd": adapter["cwd"],
        "argv": adapter["argv"],
        "exit_code": exit_code,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "timed_out": timed_out,
        "interrupted": interrupted,
    }
    fields.update({key: value for key, value in identity.items() if value is not None})
    return fields


def _github_fields(
    adapter: dict[str, Any],
    api_url: str,
    *,
    http_status: Optional[int] = None,
    last_observation_type: Optional[str] = None,
    last_observation: Optional[dict[str, Any]] = None,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    data = data or {}
    return {
        **{
            key: adapter[key]
            for key in (
                "repository",
                "run_id",
                "run_attempt",
                "workflow_id",
                "head_sha",
                "ref",
            )
        },
        "api_url": api_url,
        "observed_status": data.get("status"),
        "observed_conclusion": data.get("conclusion"),
        "observed_head_sha": data.get("head_sha"),
        "observed_workflow_id": data.get("workflow_id"),
        "observed_ref": data.get("head_branch") or data.get("ref"),
        "http_status": http_status,
        "last_observation_type": last_observation_type,
        "last_observation": last_observation,
    }


def _check_executable(adapter: dict[str, Any]) -> Optional[tuple[str, str, dict[str, Any]]]:
    executable = adapter["executable"]
    cwd = adapter["cwd"]
    if not os.path.isdir(cwd):
        return "invalid", "cwd-not-directory", {"cwd": cwd}
    try:
        metadata = os.stat(executable)
    except FileNotFoundError:
        return "invalid", "executable-not-found", {"executable": executable}
    except OSError as error:
        return "inconclusive", "executable-unreadable", {"error": type(error).__name__}
    if not stat.S_ISREG(metadata.st_mode):
        return "invalid", "executable-not-regular", {"executable": executable}
    if not os.access(executable, os.X_OK):
        return "invalid", "executable-not-executable", {"executable": executable}
    try:
        observed = compute_sha256(executable)
    except OSError as error:
        return "inconclusive", "executable-unreadable", {"error": type(error).__name__}
    if observed != adapter["executable_sha256"]:
        return "mismatch", "executable-digest-mismatch", {"observed_sha256": observed}
    return None


def run_process_adapter(
    spec: dict[str, Any],
    ready_path: Path,
    result_path: Path,
    stop_path: Path,
    started_at_iso: str,
    started_monotonic: float,
    is_interrupted: Callable[[], bool],
    signal_name: Callable[[], Optional[int]],
    helper_pid: int,
    helper_marker: str,
    clock_monotonic: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    popen_cls: Any | None = None,
    get_marker: Callable[[int], Optional[str]] | None = None,
    tail_limit: int = TAIL_LIMIT,
    grace: float = CHILD_GRACE_SECONDS,
) -> None:
    clock = clock_monotonic or _clock_monotonic
    sleep = sleep_fn or _sleep
    popen = popen_cls or subprocess.Popen
    marker_for = get_marker or _get_start_marker
    adapter = spec["adapter"]
    deadline = started_monotonic + spec["timeout_seconds"]

    def finish(
        outcome: str,
        reason: str,
        observed: Optional[dict[str, Any]] = None,
        process: Optional[dict[str, Any]] = None,
    ) -> None:
        atomic_write_json(
            result_path,
            _build_result(
                spec,
                started_at_iso,
                started_monotonic,
                outcome,
                reason,
                observed or {},
                clock,
                process=process or _process_fields(adapter),
            ),
        )

    if clock() >= deadline:
        finish(
            "timeout",
            "process-timeout",
            {"phase": "before-spawn"},
            _process_fields(adapter, timed_out=True),
        )
        return
    if is_interrupted() or _is_control(stop_path, spec, "stop"):
        reason = f"signal-{signal_name()}" if is_interrupted() else "stop-file-before-spawn"
        finish(
            "interrupted",
            reason,
            {"phase": "before-spawn"},
            _process_fields(adapter, interrupted=True),
        )
        return
    invalid = _check_executable(adapter)
    if invalid is not None:
        finish(*invalid)
        return
    try:
        rechecked = compute_sha256(adapter["executable"])
    except OSError as error:
        finish("inconclusive", "executable-recheck-unreadable", {"error": type(error).__name__})
        return
    if rechecked != adapter["executable_sha256"]:
        finish("mismatch", "executable-digest-recheck-mismatch", {"observed_sha256": rechecked})
        return
    if is_interrupted() or _is_control(stop_path, spec, "stop"):
        reason = f"signal-{signal_name()}" if is_interrupted() else "stop-file-before-spawn"
        finish(
            "interrupted",
            reason,
            {"phase": "before-spawn"},
            _process_fields(adapter, interrupted=True),
        )
        return
    if clock() >= deadline:
        finish(
            "timeout",
            "process-timeout",
            {"phase": "before-spawn"},
            _process_fields(adapter, timed_out=True),
        )
        return
    process = popen(
        adapter["argv"],
        cwd=adapter["cwd"],
        env=_minimal_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        start_new_session=True,
    )
    child_pid = process.pid
    pgid = os.getpgid(child_pid)
    child_marker = marker_for(child_pid)
    identity = {"pid": child_pid, "pgid": pgid, "child_start_marker": child_marker}
    if child_marker is None:
        terminated = _terminate_pgid(pgid, process, grace, sleep, clock)
        if process.poll() is not None:
            process.wait()
        reason = "child-start-marker-unavailable" if terminated else "termination-inconclusive"
        finish("inconclusive", reason, identity, _process_fields(adapter, **identity))
        return
    _write_runtime(
        ready_path,
        spec,
        helper_pid,
        helper_marker,
        "running",
        started_at_iso,
        child_pid=child_pid,
        child_start_marker=child_marker,
        pgid=pgid,
    )
    stdout = TailCollector(process.stdout, tail_limit)
    stderr = TailCollector(process.stderr, tail_limit)
    stdout.start()
    stderr.start()
    outcome = reason = ""
    flags: dict[str, bool] = {}
    while process.poll() is None:
        if is_interrupted():
            outcome, reason, flags = "interrupted", f"signal-{signal_name()}", {"interrupted": True}
            break
        if _is_control(stop_path, spec, "stop"):
            outcome, reason, flags = "interrupted", "stop-file", {"interrupted": True}
            break
        if clock() >= deadline:
            outcome, reason, flags = "timeout", "process-timeout", {"timed_out": True}
            break
        sleep(min(0.05, max(0.0, deadline - clock())))
    if outcome:
        if not _terminate_pgid(pgid, process, grace, sleep, clock):
            outcome, reason, flags = "inconclusive", "termination-inconclusive", {}
    else:
        outcome, reason = "completed", "process-exit"
    if process.poll() is not None:
        process.wait()
        stdout.join(1)
        stderr.join(1)
    finish(
        outcome,
        reason,
        {**identity, "exit_code": process.poll()},
        _process_fields(
            adapter,
            exit_code=process.poll(),
            stdout_tail=stdout.buf,
            stderr_tail=stderr.buf,
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
            **flags,
            **identity,
        ),
    )


def _github_identity(data: Any) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    repository = data.get("repository") or data.get("head_repository")
    full_name = repository.get("full_name") if isinstance(repository, dict) else None
    ref = data.get("head_branch") or data.get("ref")
    required = (
        _strict_int(data.get("id")),
        _strict_int(data.get("run_attempt")),
        _strict_int(data.get("workflow_id")),
        isinstance(data.get("status"), str),
        isinstance(ref, str) and bool(ref),
        isinstance(full_name, str),
        isinstance(data.get("head_sha"), str)
        and re.fullmatch(r"[0-9a-f]{40}", data["head_sha"].lower()) is not None,
    )
    if not all(required):
        return None
    return {
        "repository": full_name,
        "run_id": data["id"],
        "run_attempt": data["run_attempt"],
        "workflow_id": data["workflow_id"],
        "head_sha": data["head_sha"].lower(),
        "ref": ref,
        "status": data["status"],
        "conclusion": data.get("conclusion"),
    }


def _ref_matches(expected: str, observed: str) -> bool:
    if observed == expected:
        return True
    for prefix in ("refs/heads/", "refs/tags/"):
        if expected.startswith(prefix):
            return observed == expected.removeprefix(prefix)
    return False


def run_github_adapter(
    spec: dict[str, Any],
    ready_path: Path,
    result_path: Path,
    stop_path: Path,
    started_at_iso: str,
    started_monotonic: float,
    is_interrupted: Callable[[], bool],
    signal_name: Callable[[], Optional[int]],
    helper_pid: int,
    helper_marker: str,
    clock_monotonic: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    fetch_fn: Callable[[str, str], dict[str, Any]] | None = None,
    gh_token_fn: Callable[[], str] | None = None,
    poll_interval: float | None = None,
) -> None:
    clock = clock_monotonic or _clock_monotonic
    sleep = sleep_fn or _sleep
    fetch = fetch_fn or default_fetch_github
    get_token = gh_token_fn or default_gh_token
    poll = GITHUB_POLL_INTERVAL if poll_interval is None else poll_interval
    adapter = spec["adapter"]
    deadline = started_monotonic + spec["timeout_seconds"]
    api_url = github_api_url(adapter["repository"], adapter["run_id"])
    repo_url = github_repo_url(adapter["repository"])
    last_type: Optional[str] = None
    last_observation: Optional[dict[str, Any]] = None
    last_status: Optional[int] = None
    last_data: Optional[dict[str, Any]] = None

    def finish(outcome: str, reason: str, observed: Optional[dict[str, Any]] = None) -> None:
        atomic_write_json(
            result_path,
            _build_result(
                spec,
                started_at_iso,
                started_monotonic,
                outcome,
                reason,
                observed or {},
                clock,
                github=_github_fields(
                    adapter,
                    api_url,
                    http_status=last_status,
                    last_observation_type=last_type,
                    last_observation=last_observation,
                    data=last_data,
                ),
            ),
        )

    _write_runtime(ready_path, spec, helper_pid, helper_marker, "running", started_at_iso)
    try:
        token: Optional[str] = get_token()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        token = None
        last_type = "unreadable"
        last_observation = {"type": "token-unreadable", "error": type(error).__name__}
    while True:
        if is_interrupted():
            last_type = "interrupted"
            finish("interrupted", f"signal-{signal_name()}", {"phase": "running"})
            return
        if _is_control(stop_path, spec, "stop"):
            last_type = "interrupted"
            finish("interrupted", "stop-file", {"phase": "running"})
            return
        if clock() >= deadline:
            finish("timeout", "github-timeout", {"last_observation_type": last_type})
            return
        if token is None:
            sleep(min(poll, max(0.0, deadline - clock())))
            continue
        try:
            response = fetch(api_url, token)
        except Exception as error:  # observation boundary; no verdict was obtained
            last_type = "unreadable"
            last_observation = {"type": "fetch-exception", "error": type(error).__name__}
            sleep(min(poll, max(0.0, deadline - clock())))
            continue
        last_status = response.get("status_code")
        data = response.get("data")
        last_data = data if isinstance(data, dict) else None
        if last_status == 404:
            try:
                repo_response = fetch(repo_url, token)
            except Exception as error:  # observation boundary
                repo_response = {"status_code": None, "error": type(error).__name__}
            repo_data = repo_response.get("data")
            if (
                repo_response.get("status_code") == 200
                and isinstance(repo_data, dict)
                and repo_data.get("full_name") == adapter["repository"]
            ):
                last_type = "absent"
                finish("absent", "github-404-absent", {"http_status": 404})
                return
            last_type = "unreadable"
            last_observation = {
                "run_status": 404,
                "repo_status": repo_response.get("status_code"),
            }
        elif last_status != 200 or response.get("error") is not None:
            last_type = "unreadable"
            last_observation = {
                "status_code": last_status,
                "reason": response.get("reason") or response.get("error"),
            }
        else:
            identity = _github_identity(data)
            if identity is None:
                last_type = "inconclusive"
                last_observation = {"type": "missing-or-malformed-identity"}
            else:
                comparisons = {
                    "repository": adapter["repository"],
                    "run_id": adapter["run_id"],
                    "run_attempt": adapter["run_attempt"],
                    "workflow_id": adapter["workflow_id"],
                    "head_sha": adapter["head_sha"],
                }
                mismatch = next(
                    (
                        field
                        for field, expected in comparisons.items()
                        if identity[field] != expected
                    ),
                    None,
                )
                if mismatch is None and not _ref_matches(adapter["ref"], identity["ref"]):
                    mismatch = "ref"
                if mismatch is not None:
                    last_type = "mismatch"
                    finish("mismatch", f"{mismatch}-mismatch", identity)
                    return
                if identity["status"] == "completed":
                    last_type = "completed"
                    finish(
                        "completed",
                        (
                            f"ci-{identity['conclusion']}"
                            if identity["conclusion"]
                            else "ci-completed"
                        ),
                        identity,
                    )
                    return
                last_type = "in_progress"
                last_observation = {
                    "status": identity["status"],
                    "conclusion": identity["conclusion"],
                }
        sleep(min(poll, max(0.0, deadline - clock())))


def _before_activation_section(spec: dict[str, Any]) -> dict[str, Any]:
    if spec["adapter"]["kind"] == "process":
        return {"process": _process_fields(spec["adapter"], interrupted=True)}
    return {
        "github": _github_fields(
            spec["adapter"],
            github_api_url(spec["adapter"]["repository"], spec["adapter"]["run_id"]),
            last_observation_type="interrupted",
        )
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CAO wait runner")
    for flag in ("spec", "ready", "activate", "stop", "result"):
        parser.add_argument(f"--{flag}", required=True)
    arguments = parser.parse_args(argv)
    paths = {
        name: Path(getattr(arguments, name))
        for name in (
            "spec",
            "ready",
            "activate",
            "stop",
            "result",
        )
    }
    interrupted: dict[str, Any] = {"flag": False, "signum": None}
    originals: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        interrupted.update(flag=True, signum=signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            originals[signum] = signal.signal(signum, handle_signal)
        except ValueError:
            pass
    raw: Any = None
    spec: Optional[dict[str, Any]] = None
    started_at = _now_iso()
    started_monotonic = _clock_monotonic()
    try:
        try:
            with paths["spec"].open(encoding="utf-8") as handle:
                raw = json.load(handle)
            spec = validate_spec(raw)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            invalid = raw if isinstance(raw, dict) else {}
            atomic_write_json(
                paths["result"],
                _build_result(
                    invalid,
                    started_at,
                    started_monotonic,
                    "invalid_spec",
                    str(error)[:500],
                    {},
                    _clock_monotonic,
                ),
            )
            raise SystemExit(2)
        helper_pid = os.getpid()
        helper_marker = _get_start_marker(helper_pid)
        if helper_marker is None:
            atomic_write_json(
                paths["result"],
                _build_result(
                    spec,
                    started_at,
                    started_monotonic,
                    "inconclusive",
                    "helper-start-marker-unavailable",
                    {"helper_pid": helper_pid},
                    _clock_monotonic,
                ),
            )
            raise SystemExit(1)
        _write_runtime(
            paths["ready"],
            spec,
            helper_pid,
            helper_marker,
            "waiting-for-activation",
            started_at,
        )
        deadline = started_monotonic + spec["timeout_seconds"]
        while True:
            reason: Optional[str] = None
            if interrupted["flag"]:
                reason = f"signal-{interrupted['signum']}"
            elif _is_control(paths["stop"], spec, "stop"):
                reason = "stop-file-before-activation"
            elif _clock_monotonic() >= deadline:
                reason = "activation-timeout"
            if reason is not None:
                outcome = "timeout" if reason == "activation-timeout" else "interrupted"
                atomic_write_json(
                    paths["result"],
                    _build_result(
                        spec,
                        started_at,
                        started_monotonic,
                        outcome,
                        reason,
                        {"phase": "waiting-for-activation"},
                        _clock_monotonic,
                        **_before_activation_section(spec),
                    ),
                )
                raise SystemExit(0)
            if _is_control(paths["activate"], spec, "activate"):
                break
            _sleep(ACTIVATION_POLL_INTERVAL)
        common = (
            spec,
            paths["ready"],
            paths["result"],
            paths["stop"],
            started_at,
            started_monotonic,
            lambda: bool(interrupted["flag"]),
            lambda: interrupted["signum"],
            helper_pid,
            helper_marker,
        )
        if spec["adapter"]["kind"] == "process":
            run_process_adapter(*common)
        else:
            run_github_adapter(*common)
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as error:
        if spec is not None and not paths["result"].exists():
            atomic_write_json(
                paths["result"],
                _build_result(
                    spec,
                    started_at,
                    started_monotonic,
                    "error",
                    str(error)[:500],
                    {"exception": type(error).__name__},
                    _clock_monotonic,
                ),
            )
        print(f"wait runner failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        for signum, handler in originals.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
