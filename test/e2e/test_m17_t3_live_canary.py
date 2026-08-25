"""M17 T3 paid, installed Codex route-observation canaries.

This test is the CP1 activation ladder's five-case runner.  Each case gets a
fresh state root, private tmux server, zero-turn native Codex target, and
zero-task ACP Codex/Luna requester.  The fork-side runner drives ``/status``
while the production capability remains dark.  Deliverable wakes traverse the
real inbox bridge, produce the provider-native model-turn receipt, and enter
the real conductor consumer.  The stale-requester case deliberately retires
its requester first and proves zero provider input; by definition it cannot
honestly produce a model-turn receipt.

The file is excluded by the repository's default ``not e2e`` selection and
also requires ``CAO_M17_T3_LIVE=1``.  Run only after CP1 and exact installed
source attestation::

    CAO_M17_T3_LIVE=1 \
    CAO_M17_T3_EVIDENCE_DIR=/absolute/evidence \
    CAO_M17_T3_CONDUCTOR_REPO=/absolute/clean/conductor-main \
      uv run pytest -m e2e test/e2e/test_m17_t3_live_canary.py -v
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from test.e2e.exact_canary.evidence import EvidenceSanitizer
from test.e2e.route_observation_canary import cases
from test.e2e.test_exact_executor_canary import _build_codex_home
from test.fixtures.cao_server import _pick_free_port, _start_cao_server
from test.fixtures.tmux_server import (
    assert_shared_server_untouched,
    isolated_tmux_server,
    shared_server_sentinel,
)
from typing import Any, Mapping

import pytest
import requests

from cli_agent_orchestrator.services import provider_contracts

pytestmark = [pytest.mark.e2e, pytest.mark.requires_tmux]

LIVE_ENV = "CAO_M17_T3_LIVE"
EVIDENCE_ENV = "CAO_M17_T3_EVIDENCE_DIR"
CONDUCTOR_ENV = "CAO_M17_T3_CONDUCTOR_REPO"
V2_ROOT = "/managed-launch/v2/reservations"
PROTOCOL_V2 = "cao-managed-launch-v2"
MODEL = "gpt-5.6-luna"
EFFORT = "high"
PROFILE = "m17-t3-canary"
PROFILE_TEXT = """---
name: m17-t3-canary
description: Disposable M17 T3 route-observation wake receiver
---

This is an installed route-observation canary. When a JSON wake arrives,
reply with exactly `wake received`. Do not use tools.
"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return result.stdout.strip()


def _require_clean_worktree(path: Path, label: str) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    ).stdout
    assert not status.strip(), f"{label} must be clean"


def _require_empty_evidence_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise AssertionError(f"{EVIDENCE_ENV} must name a new or empty evidence directory")
    path.mkdir(parents=True, exist_ok=True)


def _conductor_source() -> Path:
    value = os.environ.get(CONDUCTOR_ENV)
    if not value:
        pytest.skip(f"{CONDUCTOR_ENV} must name the exact clean conductor installation")
    path = Path(value)
    if not (path / "conduct" / "lib" / "route_completion_consumer.py").is_file():
        pytest.fail(f"{CONDUCTOR_ENV} does not contain the route consumer: {path}")
    _require_clean_worktree(path, "the T3 conductor installation")
    return path


def _installed_codex() -> dict[str, str]:
    binary = shutil.which("codex")
    if not binary:
        pytest.skip("Codex CLI is not installed")
    executable = os.path.realpath(binary)
    probe = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    banner = (probe.stdout or probe.stderr or "").strip()
    assert probe.returncode == 0 and banner, f"Codex version probe failed: {banner}"
    version = str(provider_contracts.normalized_version(banner))
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), f"unparseable Codex build {banner!r}"
    return {
        "path": executable,
        "sha256": _sha256_file(Path(executable)),
        "banner": banner,
        "version": version,
    }


def _git_worktree(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "m17-t3@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "m17-t3-canary"], cwd=path, check=True)
    (path / "README.txt").write_text("M17 T3 disposable canary\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initialize canary"], cwd=path, check=True)
    return os.path.realpath(path)


def _request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, url, **kwargs)
    assert response.status_code < 400, f"{method} {url}: {response.status_code} {response.text}"
    value = response.json()
    assert isinstance(value, dict), f"{method} {url} returned a non-object"
    return value


def _launch_bound(
    *,
    server_url: str,
    session_name: str,
    workdir: str,
    installed: Mapping[str, str],
    mode: str,
    case_key: str,
) -> dict[str, Any]:
    reservation_id = str(uuid.uuid4())
    payload = {
        "protocol_version": PROTOCOL_V2,
        "reservation_id": reservation_id,
        "session_name": session_name,
        "provider": "codex",
        "agent_profile": PROFILE,
        "caller_id": uuid.uuid4().hex[:8],
        "working_directory": workdir,
        "trusted_project_root": workdir,
        "expected_model": MODEL,
        "expected_effort": EFFORT,
        "provider_executable": installed["path"],
        "provider_executable_sha256": installed["sha256"],
        "obligation_generation": f"m17-t3-{uuid.uuid4().hex[:12]}",
        "task_id": f"m17-t3-{case_key}-{mode}",
        "run_id": f"m17-t3-{uuid.uuid4().hex[:12]}",
        "delivery_id": str(uuid.uuid4()),
        "launch_nonce": uuid.uuid4().hex + uuid.uuid4().hex,
        "execution_mode": mode,
        "quota_provider": "openai",
    }
    reserved = requests.post(f"{server_url}{V2_ROOT}", json=payload, timeout=30)
    assert reserved.status_code == 201, f"reserve failed: {reserved.status_code} {reserved.text}"
    launched = requests.post(
        f"{server_url}{V2_ROOT}/{reservation_id}/launch",
        timeout=420,
    )
    record = _request_json("GET", f"{server_url}{V2_ROOT}/{reservation_id}", timeout=30)
    assert launched.status_code == 200, (
        f"{mode} launch failed: {launched.status_code} {launched.text}; "
        f"state={record.get('state')} preflight={record.get('preflight_failure')}"
    )
    assert record.get("terminal_id") and record.get("generation")
    assert record.get("native_session_id"), record
    bind = requests.post(
        f"{server_url}{V2_ROOT}/{reservation_id}/bind",
        json={
            "protocol_version": PROTOCOL_V2,
            "terminal_id": record["terminal_id"],
            "generation": record["generation"],
            "attempt_id": str(uuid.uuid4()),
            "execution_mode": mode,
        },
        timeout=180,
    )
    assert bind.status_code == 200, (
        f"{mode} bind failed: {bind.status_code} {bind.text}; "
        f"record={json.dumps(record, sort_keys=True)}"
    )
    bound = bind.json()
    assert bound["state"] == "bound", bound
    assert bound["binding"]["native_session_id"] == record["native_session_id"]
    return {"reserve": payload, "launch": launched.json(), "record": bound}


def _run_module(
    module: str,
    arguments: list[str],
    *,
    env: Mapping[str, str],
    timeout: float = 300,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"{module} failed with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result


def _runner(
    case_key: str,
    command: str,
    *,
    env: Mapping[str, str],
    spec: Path | None = None,
    prepared: Path | None = None,
    output: Path,
    restart_phase: str | None = None,
    replay_phase: str | None = None,
) -> None:
    args = [case_key, command]
    if command == "prepare":
        assert spec is not None
        args.extend(["--spec", str(spec), "--output", str(output)])
    else:
        assert prepared is not None
        args.extend(["--prepared", str(prepared), "--output", str(output)])
        if restart_phase is not None:
            args.extend(["--restart-phase", restart_phase])
        if replay_phase is not None:
            args.extend(["--replay-phase", replay_phase])
    _run_module("test.e2e.route_observation_canary.runner", args, env=env)


def _shareable_json(sanitizer: EvidenceSanitizer, value: Any) -> Any:
    """Hash provider session ids, then apply the ordinary evidence redactions."""

    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            converted: dict[str, Any] = {}
            for raw_key, child in item.items():
                key = str(raw_key)
                if isinstance(child, str) and (
                    key.lower() == "session_id" or key.lower().endswith("_session_id")
                ):
                    converted[f"{key}_sha256"] = _sha256_text(child)
                else:
                    converted[key] = convert(child)
            return converted
        if isinstance(item, (list, tuple)):
            return [convert(child) for child in item]
        return item

    return sanitizer.sanitize_json(convert(value))


def _shareable_installed_codex(installed: Mapping[str, str]) -> dict[str, str]:
    """Keep the installed-build attestation without publishing its local path."""
    executable = installed["path"]
    return {
        "executable_basename": Path(executable).name,
        "executable_path_sha256": _sha256_text(executable),
        "sha256": installed["sha256"],
        "banner": installed["banner"],
        "version": installed["version"],
    }


def _write_bounded_server_log(
    sanitizer: EvidenceSanitizer,
    source: Path,
    destination: Path,
) -> None:
    try:
        content = "\n".join(
            source.read_text(encoding="utf-8", errors="replace").splitlines()[-500:]
        )
    except OSError as exc:
        content = f"<server log read failed: {exc}>"
    sanitizer.write_text(destination, content + "\n")


def _deliver(
    *,
    receiver_id: str,
    message_id: int,
    output: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    _run_module(
        "test.e2e.route_observation_canary.delivery",
        [
            "--receiver-id",
            receiver_id,
            "--message-id",
            str(message_id),
            "--output",
            str(output),
        ],
        env=env,
        timeout=360,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _turn_receipt(
    server_url: str,
    terminal_id: str,
    message_id: int,
    *,
    timeout: float = 180,
) -> dict[str, Any]:
    url = f"{server_url}/terminals/{terminal_id}/inbox/messages/{message_id}/turn-receipt"
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            receipt = response.json()
            assert isinstance(receipt, dict)
            return receipt
        assert response.status_code == 204, f"turn receipt read failed: {response.text}"
        last = response.status_code
        time.sleep(0.5)
    raise AssertionError(f"provider-native turn receipt stayed absent (last={last})")


def _write_posted_control(conductor: Path, psd: str, wake: Mapping[str, str]) -> str:
    root = str(conductor)
    if root not in sys.path:
        sys.path.insert(0, root)
    from conduct.lib import control_input as journal

    os.makedirs(os.path.join(psd, journal.JOURNAL_DIRNAME), exist_ok=True)
    control_id = str(uuid.uuid4())
    record = journal.new_record(
        control_id=control_id,
        terminal_id=wake["target_terminal_id"],
        task_id="m17-t3",
        kind="command",
        command="/observe",
        text="/observe",
        enter=True,
        expected_identity={
            "terminal_id": wake["target_terminal_id"],
            "terminal_generation": wake["target_generation"],
            "native_session_id": wake["native_session_id"],
            "provider": wake["provider"],
        },
    )
    journal.mark_attempt(record)
    journal.mark_transport(record, posted=True)
    journal.apply_transition(record, "posted", evidence="M17 T3 installed transport")
    journal.write_record(psd, record)
    return control_id


def _consume(
    conductor: Path,
    *,
    case_dir: Path,
    wake: dict[str, Any],
    turn_receipt: dict[str, Any],
    route_receipt: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[tuple[str, str, str, dict[str, Any]]], Any, Any]:
    root = str(conductor)
    if root not in sys.path:
        sys.path.insert(0, root)
    from conduct.lib import route_completion_consumer as consumer

    psd = str(case_dir / "project-state")
    _write_posted_control(conductor, psd, wake)
    store_path = case_dir / "consumer.sqlite3"
    consumer.create(
        str(store_path),
        db_uuid=str(uuid.uuid4()),
        created_at="2026-08-25T00:00:00Z",
    )
    connection = consumer.connect(str(store_path))
    store = consumer.RouteCompletionStore(connection)
    wakes: list[tuple[str, str, str, dict[str, Any]]] = []

    def wake_supervisor(
        receiver_id: str,
        receiver_generation: str,
        disposition: str,
        detail: dict[str, Any],
    ) -> None:
        wakes.append((receiver_id, receiver_generation, disposition, detail))

    claim = {"wake": wake, "turn_receipt": turn_receipt}
    if route_receipt is not None:
        claim["route_receipt"] = route_receipt
    response = consumer.consume_wake(
        claim,
        psd=psd,
        store=store,
        wake_supervisor=wake_supervisor,
        env={consumer.ENABLED_ENV: "true"},
    )
    return response, wakes, store, connection


def _retire_requester(server_url: str, requester: Mapping[str, Any]) -> dict[str, Any]:
    record = requester["record"]
    response = requests.delete(
        f"{server_url}/terminals/{record['terminal_id']}",
        params={
            "expected_generation": record["generation"],
            "expected_session": record["session_name"],
        },
        timeout=60,
    )
    assert response.status_code == 200, f"requester retirement failed: {response.text}"
    absent = requests.get(f"{server_url}/terminals/{record['terminal_id']}", timeout=10)
    assert (
        absent.status_code == 404
    ), f"retired requester {record['terminal_id']}/{record['generation']} remained live"
    return response.json()


def _run_case(
    *,
    case: cases.CanaryCase,
    case_dir: Path,
    installed: Mapping[str, str],
    conductor: Path,
    real_home: Path,
) -> dict[str, Any]:
    def progress(stage: str) -> None:
        print(f"[m17-t3] {case.runner_key}: {stage}", flush=True)

    progress("starting isolated runtime")
    case_dir.mkdir(parents=True)
    state_root = case_dir / "state"
    state_root.mkdir()
    profile_store = state_root / "agent-store"
    profile_store.mkdir()
    (profile_store / f"{PROFILE}.md").write_text(PROFILE_TEXT, encoding="utf-8")
    scratch = case_dir / "scratch"
    scratch.mkdir()
    codex_home = _build_codex_home(real_home, scratch)
    workdir = _git_worktree(scratch / "worktree")

    with shared_server_sentinel(f"m17-t3-{case.runner_key[:12]}") as shared_canary:
        shared, shared_session, shared_identity = shared_canary
        with isolated_tmux_server(f"m17-t3-{case.runner_key[:12]}") as tmux:
            assert tmux.owned_root is not None
            shim = tmux.write_shim(tmux.owned_root / "bin")
            target_session = f"cao-t3-t-{uuid.uuid4().hex[:8]}"
            requester_session = f"cao-t3-r-{uuid.uuid4().hex[:8]}"
            tmux.new_session(
                target_session, "-x", "120", "-y", "40", "--", "sh", "-c", "sleep 3600"
            )
            tmux.new_session(
                requester_session, "-x", "120", "-y", "40", "--", "sh", "-c", "sleep 3600"
            )
            child_env = tmux.subprocess_env(shim)
            child_env.update(
                {
                    "HOME": str(real_home),
                    "CODEX_HOME": codex_home,
                    "CAO_STATE_ROOT": str(state_root),
                    "CAO_A2A_DISABLED": "true",
                    "OTEL_SDK_DISABLED": "true",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            sanitizer = EvidenceSanitizer(
                {
                    str(real_home): "<HOME>",
                    str(state_root): "<STATE_ROOT>",
                    str(scratch): "<SCRATCH>",
                    str(tmux.socket_path): "<TMUX_SOCKET>",
                    os.environ.get("USER", ""): "<USER>",
                }
            )
            server_home = scratch / "server-home"
            server = None
            try:
                server = _start_cao_server(
                    server_home,
                    _pick_free_port(),
                    extra_env=child_env,
                    deadline=30,
                )
                child_env["CAO_API_HOST"] = "127.0.0.1"
                child_env["CAO_API_PORT"] = str(server.port)
                target = _launch_bound(
                    server_url=server.url,
                    session_name=target_session,
                    workdir=workdir,
                    installed=installed,
                    mode="native_tui",
                    case_key=case.runner_key,
                )
                progress("native target bound")
                requester = _launch_bound(
                    server_url=server.url,
                    session_name=requester_session,
                    workdir=workdir,
                    installed=installed,
                    mode="acp",
                    case_key=case.runner_key,
                )
                progress("Luna requester bound without task input")
                target_record = target["record"]
                requester_record = requester["record"]
                pane_id = target_record["pane_id"]
                assert pane_id
                window_name = tmux.out("display-message", "-p", "-t", pane_id, "#{window_name}")
                pane_before = str(tmux.out("capture-pane", "-p", "-S-300", "-t", pane_id))
                sanitizer.write_text(case_dir / "pane-before.txt", pane_before)

                retirement = None
                if case is cases.STALE_REQUESTER:
                    retirement = _retire_requester(server.url, requester)
                else:
                    current_requester = _request_json(
                        "GET",
                        f"{server.url}/terminals/{requester_record['terminal_id']}",
                        timeout=10,
                    )
                    assert current_requester["generation"] == requester_record["generation"]

                spec_path = case_dir / "spec.json"
                prepared_path = case_dir / "prepared.json"
                evidence_path = case_dir / "fork-evidence.json"
                event_log = case_dir / "fork-events.jsonl"
                spec = {
                    "operation_id": str(uuid.uuid4()),
                    "target_terminal_id": target_record["terminal_id"],
                    "target_generation": target_record["generation"],
                    "native_session_id": target_record["native_session_id"],
                    "provider": "codex",
                    "provider_version": installed["version"],
                    "provider_artifact_sha256": installed["sha256"],
                    "requester_terminal_id": requester_record["terminal_id"],
                    "requester_generation": requester_record["generation"],
                    "runtime": {
                        "pane_id": pane_id,
                        "event_log": str(event_log),
                        "target_terminal_id": target_record["terminal_id"],
                        "target_session_name": target_record["session_name"],
                        "target_window_name": window_name,
                        "requester_probe_url": (f"{server.url}/terminals/{{terminal_id}}"),
                    },
                }
                spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
                _runner(
                    case.runner_key,
                    "prepare",
                    env=child_env,
                    spec=spec_path,
                    output=prepared_path,
                )
                if case is cases.RESTART_RECOVERY:
                    interrupt = case_dir / "restart-interrupt.json"
                    _runner(
                        case.runner_key,
                        "execute",
                        env=child_env,
                        prepared=prepared_path,
                        output=interrupt,
                        restart_phase="interrupt",
                    )
                    assert (
                        json.loads(interrupt.read_text(encoding="utf-8"))["outcome"]["terminal"]
                        is False
                    )
                    _runner(
                        case.runner_key,
                        "execute",
                        env=child_env,
                        prepared=prepared_path,
                        output=evidence_path,
                        restart_phase="resume",
                    )
                elif case is cases.REPLAY_NO_DUPLICATE:
                    replay_initial = case_dir / "replay-initial.json"
                    _runner(
                        case.runner_key,
                        "execute",
                        env=child_env,
                        prepared=prepared_path,
                        output=replay_initial,
                        replay_phase="initial",
                    )
                    initial_evidence = json.loads(replay_initial.read_text(encoding="utf-8"))
                    _runner(
                        case.runner_key,
                        "execute",
                        env=child_env,
                        prepared=prepared_path,
                        output=evidence_path,
                        replay_phase="retry",
                    )
                    replay_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                    assert replay_evidence["outcome"]["replayed"] is True
                    assert replay_evidence["outcome"]["inbox_message_id"] == (
                        initial_evidence["outcome"]["inbox_message_id"]
                    )
                else:
                    _runner(
                        case.runner_key,
                        "execute",
                        env=child_env,
                        prepared=prepared_path,
                        output=evidence_path,
                    )
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                outcome = evidence["outcome"]
                progress("route observation terminalized")

                delivery = None
                turn_receipt = None
                consumer_response = None
                consumer_wakes: list[Any] = []
                consumer_replay = None
                if case is not cases.STALE_REQUESTER:
                    delivery = _deliver(
                        receiver_id=requester_record["terminal_id"],
                        message_id=int(outcome["inbox_message_id"]),
                        output=case_dir / "delivery.json",
                        env=child_env,
                    )
                    assert delivery["status"] == "delivered", delivery
                    turn_receipt = _turn_receipt(
                        server.url,
                        requester_record["terminal_id"],
                        int(outcome["inbox_message_id"]),
                    )
                    progress("wake delivered and provider turn receipted")
                    route_receipt = outcome.get("receipt")
                    consumer_response, consumer_wakes, store, connection = _consume(
                        conductor,
                        case_dir=case_dir,
                        wake=outcome["wake"],
                        turn_receipt=turn_receipt,
                        route_receipt=route_receipt,
                    )
                    try:
                        expected_consumer = (
                            "woken-failure" if case is cases.AMBIGUOUS_CLOSE else "completed"
                        )
                        assert consumer_response["outcome"] == expected_consumer, consumer_response
                        assert len(consumer_wakes) == 1, consumer_wakes
                        if case is cases.REPLAY_NO_DUPLICATE:
                            from conduct.lib import route_completion_consumer as consumer

                            claim = {
                                "wake": outcome["wake"],
                                "turn_receipt": turn_receipt,
                                "route_receipt": route_receipt,
                            }
                            consumer_replay = consumer.consume_wake(
                                claim,
                                psd=str(case_dir / "project-state"),
                                store=store,
                                wake_supervisor=lambda *args: consumer_wakes.append(args),
                                env={consumer.ENABLED_ENV: "true"},
                            )
                            assert consumer_replay["outcome"] == "replayed"
                            assert len(consumer_wakes) == 1
                        progress("conductor consumer verified")
                    finally:
                        connection.close()
                else:
                    assert evidence["status_command_count"] == 0
                    assert outcome["disposition"] == "requester-stale"
                    assert evidence["inbox_message_status"] == "pending"
                    assert retirement is not None
                    progress("stale generation fenced with zero provider input")

                assert_shared_server_untouched(shared, shared_session, shared_identity)
                progress("passed")
                result = {
                    "case_id": case.case_id,
                    "case": case.runner_key,
                    "target": {
                        "terminal_id": target_record["terminal_id"],
                        "generation": target_record["generation"],
                        "native_session_id_sha256": _sha256_text(
                            target_record["native_session_id"]
                        ),
                        "pane_id": pane_id,
                    },
                    "requester": {
                        "terminal_id": requester_record["terminal_id"],
                        "generation": requester_record["generation"],
                        "retirement": retirement,
                    },
                    "pane_before_sha256": _sha256_text(pane_before),
                    "operation_id": outcome["operation_id"],
                    "result": outcome["result"],
                    "route_receipt_digest": outcome.get("receipt_digest"),
                    "wake_message_id": outcome["inbox_message_id"],
                    "wake_message_status": evidence["inbox_message_status"],
                    "status_command_count": evidence["status_command_count"],
                    "delivery": delivery,
                    "turn_receipt": turn_receipt,
                    "consumer": consumer_response,
                    "consumer_wake_count": len(consumer_wakes),
                    "consumer_replay": consumer_replay,
                }
                return _shareable_json(sanitizer, result)
            finally:
                if server is not None:
                    server.stop()
                _write_bounded_server_log(
                    sanitizer,
                    server.log_path if server is not None else server_home / "server.log",
                    case_dir / "server.log",
                )
                with contextlib.suppress(Exception):
                    tmux.kill_session(target_session, check=False)
                with contextlib.suppress(Exception):
                    tmux.kill_session(requester_session, check=False)


def _preserve_case(
    run_dir: Path,
    preserved: Path,
    *,
    result: Mapping[str, Any] | None = None,
) -> None:
    """Preserve the bounded case record even when the live attempt fails."""
    preserved.mkdir(parents=True, exist_ok=True)
    for name in (
        "spec.json",
        "prepared.json",
        "fork-evidence.json",
        "fork-events.jsonl",
        "pane-before.txt",
        "delivery.json",
        "restart-interrupt.json",
        "replay-initial.json",
        "consumer.sqlite3",
        "server.log",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, preserved / name)
    project_state = run_dir / "project-state"
    if project_state.is_dir():
        shutil.copytree(
            project_state,
            preserved / "project-state",
            dirs_exist_ok=True,
        )
    if result is not None:
        (preserved / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_partial_case_evidence_is_preserved_without_an_outcome(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "fork-events.jsonl").write_text('{"kind":"status-authorized"}\n', encoding="utf-8")
    (run_dir / "server.log").write_text("bounded server evidence\n", encoding="utf-8")
    project_state = run_dir / "project-state"
    project_state.mkdir()
    (project_state / "control.json").write_text("{}\n", encoding="utf-8")
    preserved = tmp_path / "preserved"

    _preserve_case(run_dir, preserved)

    assert (preserved / "fork-events.jsonl").read_text(encoding="utf-8") == (
        '{"kind":"status-authorized"}\n'
    )
    assert (preserved / "project-state" / "control.json").read_text(encoding="utf-8") == "{}\n"
    assert (preserved / "server.log").read_text(encoding="utf-8") == "bounded server evidence\n"
    assert not (preserved / "summary.json").exists()


def test_shareable_json_hashes_session_ids_and_redacts_local_paths(tmp_path: Path) -> None:
    sanitizer = EvidenceSanitizer({str(tmp_path): "<RUN_ROOT>"})
    shared = _shareable_json(
        sanitizer,
        {
            "provider_session_id": "provider-session",
            "nested": {"native_session_id": "native-session"},
            "path": str(tmp_path / "case"),
        },
    )

    assert shared == {
        "provider_session_id_sha256": _sha256_text("provider-session"),
        "nested": {"native_session_id_sha256": _sha256_text("native-session")},
        "path": "<RUN_ROOT>/case",
    }


def test_installed_codex_rollup_hashes_the_local_executable_path() -> None:
    installed = {
        "path": "/Users/operator/bin/codex",
        "sha256": "a" * 64,
        "banner": "codex-cli 0.149.0",
        "version": "0.149.0",
    }

    shared = _shareable_installed_codex(installed)

    assert "path" not in shared
    assert shared["executable_basename"] == "codex"
    assert shared["executable_path_sha256"] == _sha256_text(installed["path"])


def test_bounded_server_log_is_sanitized_and_limited(tmp_path: Path) -> None:
    source = tmp_path / "server-home" / "server.log"
    source.parent.mkdir()
    source.write_text(
        "discarded\n" * 4 + "/Users/operator/private\n" * 500,
        encoding="utf-8",
    )
    destination = tmp_path / "case" / "server.log"

    _write_bounded_server_log(
        EvidenceSanitizer({"/Users/operator": "<HOME>"}),
        source,
        destination,
    )

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 500
    assert set(lines) == {"<HOME>/private"}


def test_server_start_failure_preserves_its_sanitized_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_home = tmp_path / "operator-home"
    real_home.mkdir()
    owned_root = tmp_path / "tmux-owned"
    owned_root.mkdir()

    class FakeTmux:
        socket_path = tmp_path / "tmux.sock"

        def __init__(self) -> None:
            self.owned_root = owned_root

        def write_shim(self, path: Path) -> str:
            path.mkdir(parents=True)
            return str(path / "tmux")

        def new_session(self, *args: Any) -> None:
            return None

        def subprocess_env(self, shim: str) -> dict[str, str]:
            return {}

        def kill_session(self, *args: Any, **kwargs: Any) -> None:
            return None

    @contextlib.contextmanager
    def fake_shared_server_sentinel(name: str):
        yield object(), "shared-session", {"generation": "shared"}

    @contextlib.contextmanager
    def fake_isolated_tmux_server(name: str):
        yield FakeTmux()

    def fail_server_start(home: Path, *args: Any, **kwargs: Any) -> None:
        home.mkdir(parents=True)
        (home / "server.log").write_text(
            f"startup failed under {real_home}\n",
            encoding="utf-8",
        )
        raise RuntimeError("server health check failed")

    monkeypatch.setattr(
        sys.modules[__name__],
        "shared_server_sentinel",
        fake_shared_server_sentinel,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "isolated_tmux_server",
        fake_isolated_tmux_server,
    )
    monkeypatch.setattr(sys.modules[__name__], "_start_cao_server", fail_server_start)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_build_codex_home",
        lambda home, scratch: str(scratch / "codex-home"),
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_git_worktree",
        lambda path: str(path),
    )
    case_dir = tmp_path / "case"

    with pytest.raises(RuntimeError, match="health check failed"):
        _run_case(
            case=cases.POSITIVE_PATH,
            case_dir=case_dir,
            installed={
                "path": "/opt/codex",
                "sha256": "a" * 64,
                "version": "0.149.0",
            },
            conductor=tmp_path / "conductor",
            real_home=real_home,
        )

    assert (case_dir / "server.log").read_text(encoding="utf-8") == (
        "startup failed under <HOME>\n"
    )


def test_evidence_root_and_worktree_preconditions_are_mutation_pinned(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _require_empty_evidence_root(evidence)
    (evidence / "prior.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="new or empty"):
        _require_empty_evidence_root(evidence)

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
    _require_clean_worktree(repo, "test repo")
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="test repo must be clean"):
        _require_clean_worktree(repo, "test repo")


def test_installed_five_case_route_observation_ladder(tmp_path: Path) -> None:
    if os.environ.get(LIVE_ENV) != "1":
        pytest.skip(f"set {LIVE_ENV}=1 only after the CP1 live-testing checkpoint")
    evidence_value = os.environ.get(EVIDENCE_ENV)
    if not evidence_value:
        pytest.skip(f"{EVIDENCE_ENV} must name the preserved T3 evidence directory")
    evidence_root = Path(evidence_value)
    _require_empty_evidence_root(evidence_root)
    conductor = _conductor_source()
    installed = _installed_codex()
    real_home_value = os.environ.get("HOME")
    if not real_home_value or not Path(real_home_value).is_dir():
        pytest.skip("the installed canary requires the operator provider-auth HOME")
    real_home = Path(real_home_value)

    fork_root = Path(__file__).resolve().parents[2]
    _require_clean_worktree(fork_root, "the T3 fork test installation")
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", "origin/main", "--", "src"], cwd=fork_root
    )
    assert (
        source_diff.returncode == 0
    ), "T3 production source differs from origin/main; install/attest the reviewed head first"

    results = []
    for case in cases.CANARY_CASES:
        run_dir = tmp_path / case.runner_key
        preserved = evidence_root / "cases" / case.runner_key
        result = None
        try:
            result = _run_case(
                case=case,
                case_dir=run_dir,
                installed=installed,
                conductor=conductor,
                real_home=real_home,
            )
            results.append(result)
        except BaseException as exc:
            preserved.mkdir(parents=True, exist_ok=True)
            (preserved / "failure.json").write_text(
                json.dumps(
                    {
                        "schema": "cao-m17-t3-case-failure-v1",
                        "case": case.runner_key,
                        "exception_type": type(exc).__name__,
                        "recorded_at": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
        finally:
            _preserve_case(run_dir, preserved, result=result)

    assert [item["case"] for item in results] == [case.runner_key for case in cases.CANARY_CASES]
    assert sum(item["status_command_count"] for item in results) == 4
    assert sum(item["turn_receipt"] is not None for item in results) == 4
    stale_result = next(item for item in results if item["case"] == "stale-requester")
    assert stale_result["wake_message_status"] == "pending"
    manifest = {
        "schema": "cao-m17-t3-installed-canary-report-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fork_runner_head": _git_head(fork_root),
        "fork_production_head": subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=fork_root,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip(),
        "conductor_source_head": _git_head(conductor),
        "installed_codex": _shareable_installed_codex(installed),
        "case_order": [case.runner_key for case in cases.CANARY_CASES],
        "paid_turn_count": 4,
        "stale_requester_turn_count": 0,
        "results": results,
    }
    manifest_path = evidence_root / "m17-t3-installed-canary-report.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_root / "m17-t3-installed-canary-report.sha256").write_text(
        _sha256_file(manifest_path) + "\n",
        encoding="utf-8",
    )
