"""Mint and materialize one persistent Codex thread without a turn.

The app-server ``thread/start`` response is Codex's pre-turn native identity
source.  A zero-turn ``thread/start`` alone is not written to Codex's rollout
store, so an exact CLI resume cannot find it after app-server exits.  Worse,
the metadata-only ``thread/name/set`` method only persists the name to the
SQLite/index metadata: it never creates the rollout file, so the minted id is
still unloadable (``thread/resume`` answers ``no rollout found for thread
id``) once the minter exits.  This bootstrap therefore follows the start with
both ``thread/name/set`` (operator-visible name in the metadata store) and
``thread/inject_items`` carrying a single contentless ``reasoning`` item.
Codex documents ``thread/inject_items`` as appending history "without
starting a user turn", and that append is what forces the exact advertised
rollout file to exist.  The bootstrap then proves the exact rollout exists,
proves a fresh process adopts the exact id with zero turns, sends no
``turn/start`` anywhere, closes and reaps the app-server, and returns the
exact thread id that the native TUI may resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from cli_agent_orchestrator.providers.codex import CODEX_APP_SERVER_FLAGS
from cli_agent_orchestrator.services import (
    codex_native_launch,
    native_attachment,
    provider_contracts,
)
from cli_agent_orchestrator.services.codex_trust import (
    _digest_or_absent,
    _response_by_id,
    _run_app_server_probe,
)

BOOTSTRAP_SCHEMA = "cao-codex-native-bootstrap-v1"
EXIT_PROOF_SCHEMA = "cao-codex-native-bootstrap-exit-v1"
# Naming persists the operator-visible title to the metadata store only; it
# never creates the rollout file, so it is not the materializer.
NAMING_METHOD = "thread/name/set"
# The wire name uses an underscore (``thread/inject_items``); the schema title
# reads ``Thread/injectItemsRequest``.  Bind the wire spelling exactly: the
# server rejects the camelCase guess as an unknown method.
MATERIALIZATION_METHOD = "thread/inject_items"
TURNS_LIST_METHOD = "thread/turns/list"
RESUME_ADOPTION_SCHEMA = "cao-codex-native-resume-adoption-v1"
RESUME_METHOD = "thread/resume"
RESUME_THREAD_ID_PARAM = "threadId"
SCHEMA_PROBE_SCHEMA = "cao-codex-native-schema-probe-v1"

# Capability evidence is scoped to the executable bytes. A vendor version is
# only a label and cannot stand in for the exchange and rollout postconditions.
_CAPABILITY_VERDICTS: dict[str, dict[str, Any]] = {}

_SCHEMA_REQUIREMENTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "config/read": ("ConfigReadParams", "ConfigReadResponse", ("cwd",)),
    "thread/start": (
        "ThreadStartParams",
        "ThreadStartResponse",
        ("cwd", "ephemeral"),
    ),
    "thread/name/set": (
        "ThreadSetNameParams",
        "ThreadSetNameResponse",
        ("threadId", "name"),
    ),
    "thread/inject_items": (
        "ThreadInjectItemsParams",
        "ThreadInjectItemsResponse",
        ("threadId", "items"),
    ),
    "thread/resume": ("ThreadResumeParams", "ThreadResumeResponse", ("threadId",)),
}


class CodexBootstrapError(RuntimeError):
    """The Codex zero-turn bootstrap could not be proven safe."""


class CodexSchemaProbeTransientError(CodexBootstrapError):
    """The schema probe could not execute; retrying may change the result."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _materialization_items() -> list[dict[str, Any]]:
    """Return a fresh copy of the single history item that forces persistence.

    The item is a contentless ``reasoning`` item: model-internal scratch with
    no user-request semantics and zero content bytes.  A contentless ``user``
    message would also satisfy the server's non-empty-items rule, but a
    trailing user message risks reading as a pending prompt; a contentless
    reasoning item can never become a turn request, carries no task text, and
    leaves ``has_user_event`` unset.
    """
    return [{"type": "reasoning", "summary": []}]


def _find_schema_definition(schema: Mapping[str, Any], name: str) -> Optional[Mapping[str, Any]]:
    """Find a named definition in either the v1 or v2 generated bundle."""
    definitions = schema.get("definitions") or schema.get("$defs") or schema
    if not isinstance(definitions, Mapping):
        return None
    candidate = definitions.get(name)
    if isinstance(candidate, Mapping):
        return candidate
    for value in definitions.values():
        if isinstance(value, Mapping):
            nested = _find_schema_definition(value, name)
            if nested is not None:
                return nested
    return None


def _find_method_request(value: Any, method: str) -> Optional[Mapping[str, Any]]:
    """Find a JSON-RPC request shape carrying ``method``."""
    if isinstance(value, Mapping):
        properties = value.get("properties")
        if isinstance(properties, Mapping):
            method_shape = properties.get("method")
            if isinstance(method_shape, Mapping):
                values = method_shape.get("enum")
                if method_shape.get("const") == method or (
                    isinstance(values, list) and method in values
                ):
                    return value
        for child in value.values():
            found = _find_method_request(child, method)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_method_request(child, method)
            if found is not None:
                return found
    return None


def _schema_ref_name(value: Any) -> Optional[str]:
    if not isinstance(value, Mapping):
        return None
    reference = value.get("$ref")
    if not isinstance(reference, str):
        return None
    return reference.rsplit("/", 1)[-1]


def _validate_schema_bundle(schema: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        raise CodexBootstrapError("Codex schema probe failed: generated bundle is not an object")
    methods: dict[str, dict[str, Any]] = {}
    for method, (params_name, response_name, fields) in _SCHEMA_REQUIREMENTS.items():
        request = _find_method_request(schema, method)
        if request is None:
            raise CodexBootstrapError(f"Codex schema probe failed: missing method {method!r}")
        properties = request.get("properties")
        params = properties.get("params") if isinstance(properties, Mapping) else None
        if _schema_ref_name(params) != params_name:
            raise CodexBootstrapError(
                f"Codex schema probe failed: {method} params missing {params_name}"
            )
        params_schema = _find_schema_definition(schema, params_name)
        response_schema = _find_schema_definition(schema, response_name)
        if params_schema is None:
            raise CodexBootstrapError(
                f"Codex schema probe failed: {method} params definition missing {params_name}"
            )
        if response_schema is None:
            raise CodexBootstrapError(
                f"Codex schema probe failed: {method} response definition missing {response_name}"
            )
        parameter_properties = params_schema.get("properties")
        if not isinstance(parameter_properties, Mapping) or any(
            field not in parameter_properties for field in fields
        ):
            missing = [field for field in fields if field not in (parameter_properties or {})]
            raise CodexBootstrapError(
                f"Codex schema probe failed: {method} params missing field(s) {missing!r}"
            )
        if method in {"thread/start", "thread/resume"}:
            response_properties = response_schema.get("properties")
            if not isinstance(response_properties, Mapping) or "thread" not in response_properties:
                raise CodexBootstrapError(
                    f"Codex schema probe failed: {method} response missing field 'thread'"
                )
        methods[method] = {
            "params": params_name,
            "response": response_name,
            "fields": list(fields),
        }
    return {"schema": SCHEMA_PROBE_SCHEMA, "methods": methods}


def _probe_schema_capability(
    binary: str, *, timeout: float = 30.0, environment: Optional[Mapping[str, str]] = None
) -> dict[str, Any]:
    """Prove the required app-server shape without opening a session."""
    with tempfile.TemporaryDirectory(prefix="cao-codex-schema-") as directory:
        output = Path(directory) / "schema"
        try:
            completed = subprocess.run(
                [binary, "app-server", "generate-json-schema", "--out", str(output)],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=dict(environment) if environment is not None else None,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CodexSchemaProbeTransientError(
                f"Codex schema probe failed: could not execute generate-json-schema: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-500:]
            raise CodexBootstrapError(
                "Codex schema probe failed: generate-json-schema exited "
                f"{completed.returncode}: {detail}"
            )
        bundle = output / "codex_app_server_protocol.schemas.json"
        if not bundle.is_file():
            bundle = output / "codex_app_server_protocol.v2.schemas.json"
        if not bundle.is_file():
            raise CodexBootstrapError(
                "Codex schema probe failed: generate-json-schema did not produce "
                "a protocol schema bundle"
            )
        try:
            schema = json.loads(bundle.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CodexBootstrapError(
                f"Codex schema probe failed: generated schema is unreadable: {exc}"
            ) from exc
    return _validate_schema_bundle(schema)


def _validate_binary(
    binary: str,
    digest: str,
    version_output: str,
    *,
    environment: Optional[Mapping[str, str]] = None,
    timeout: float = 30.0,
) -> str:
    if not isinstance(binary, str) or not os.path.isabs(binary):
        raise CodexBootstrapError("codex binary must be a canonical absolute path")
    if os.path.realpath(binary) != binary or not os.path.isfile(binary):
        raise CodexBootstrapError("codex binary must be an existing canonical file")
    observed = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    if observed != digest:
        raise CodexBootstrapError(
            f"codex binary digest changed: expected {digest}, observed {observed}"
        )
    verdict = _CAPABILITY_VERDICTS.get(observed)
    if verdict is None:
        try:
            _CAPABILITY_VERDICTS[observed] = _probe_schema_capability(
                binary, timeout=timeout, environment=environment
            )
        except CodexSchemaProbeTransientError:
            raise
        except CodexBootstrapError as exc:
            _CAPABILITY_VERDICTS[observed] = {"error": str(exc)}
            raise
    elif "error" in verdict:
        raise CodexBootstrapError(str(verdict["error"]))
    return observed


def _rollout_path(codex_home: Path, thread_id: str) -> Path:
    """Return the sole exact rollout for ``thread_id`` or fail closed."""
    sessions = codex_home / "sessions"
    matches = [
        path
        for path in sessions.rglob(f"*-{thread_id}.jsonl")
        if path.is_file() and not path.is_symlink()
    ]
    if len(matches) != 1:
        raise CodexBootstrapError(
            f"Codex {MATERIALIZATION_METHOD} did not leave exactly one resumable rollout "
            f"for {thread_id}: found {len(matches)} under {sessions}"
        )
    return matches[0]


def _prove_resume_adoption(
    argv: list[str],
    native_id: str,
    timeout: float,
    *,
    env: Optional[Mapping[str, str]],
    config_path: pathlib.Path,
) -> dict[str, Any]:
    """Prove a fresh app-server process adopts the minted thread id."""
    requests: list[dict[str, Any]] = [
        {
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "cao-native-resume-adoption",
                    "version": BOOTSTRAP_SCHEMA,
                }
            },
        },
        {"method": "initialized", "params": {}},
        {
            "id": 2,
            "method": RESUME_METHOD,
            "params": {RESUME_THREAD_ID_PARAM: native_id},
        },
        {
            "id": 3,
            "method": TURNS_LIST_METHOD,
            "params": {RESUME_THREAD_ID_PARAM: native_id},
        },
    ]
    before = _digest_or_absent(config_path)
    try:
        stdout, stderr, returncode = _run_app_server_probe(
            argv, requests, timeout, env=dict(env) if env is not None else None
        )
    except Exception as exc:  # noqa: BLE001 - normalize the provider leg
        raise CodexBootstrapError(
            f"Codex {RESUME_METHOD} adoption probe failed for {native_id}: {exc}"
        ) from exc
    if _digest_or_absent(config_path) != before:
        raise CodexBootstrapError(
            "protected Codex user config changed during the resume-adoption probe"
        )
    if returncode not in (0, -15):
        raise CodexBootstrapError(
            f"codex app-server exited {returncode} during resume adoption: {stderr[-500:]}"
        )
    response = _response_by_id(stdout, 2)
    if "error" in response or "result" not in response:
        raise CodexBootstrapError(
            f"codex {RESUME_METHOD} failed for {native_id}: {response.get('error')!r}"
        )
    thread = (response.get("result") or {}).get("thread") or {}
    adopted = thread.get("id")
    if adopted != native_id:
        raise CodexBootstrapError(
            f"codex {RESUME_METHOD} adopted {adopted!r}, not the minted {native_id!r}"
        )
    turns_response = _response_by_id(stdout, 3)
    if "error" in turns_response or "result" not in turns_response:
        raise CodexBootstrapError(
            f"codex {TURNS_LIST_METHOD} failed for {native_id}: {turns_response.get('error')!r}"
        )
    observed_turns = (turns_response.get("result") or {}).get("data")
    if not isinstance(observed_turns, list) or len(observed_turns) != 0:
        raise CodexBootstrapError(
            f"codex {native_id} is not a zero-turn thread: "
            f"{TURNS_LIST_METHOD} observed {observed_turns!r}; refusing the minted id "
            "so no terminal can attach to a thread that already ran"
        )
    inline_turns = thread.get("turns")
    if inline_turns is not None and list(inline_turns) != []:
        raise CodexBootstrapError(
            f"codex {native_id} is not a zero-turn thread: {RESUME_METHOD} carried "
            f"{len(list(inline_turns))} inline turn(s); refusing the minted id "
            "so no terminal can attach to a thread that already ran"
        )
    return {
        "schema": RESUME_ADOPTION_SCHEMA,
        "method": RESUME_METHOD,
        "adopted_session_id": adopted,
        "adopted_in_fresh_process": True,
        "sent_no_turn": True,
        "observed_turns": [],
        "exit_status": returncode,
        "exchange_sha256": _digest(
            {
                "initialize": _response_by_id(stdout, 1),
                "resume": response,
                "turns": turns_response,
            }
        ),
        "protected_config_sha256": before,
    }


def mint_session(
    *,
    codex_binary: str,
    binary_sha256: str,
    version_output: str,
    working_directory: str,
    model: Optional[str],
    effort: Optional[str],
    profile_args: Sequence[str],
    environment: Optional[Mapping[str, str]] = None,
    developer_instructions: Optional[str] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Create one persistent thread (exact-route or provider-default) and prove
    the minter exited.

    The requested ``model`` and ``effort`` are OPTIONAL so
    an ordinary launch can inherit Codex's own configuration/defaults.  An
    unset model is omitted from ``thread/start``; effort is never a
    ``thread/start`` field (Codex's ``ThreadStartParams`` has no
    reasoning-effort field), so it stays a config/argv selection.  The actual
    model, effort (which may truthfully be null/unselected), and cwd returned
    by ``thread/start`` are ALWAYS validated and recorded; when a caller
    supplied a non-empty expected value the exact equality check is retained.
    A sealed managed-v2 route still supplies non-empty values and stays strict.
    """
    child_env = dict(environment) if environment is not None else None
    digest = _validate_binary(
        codex_binary,
        binary_sha256,
        version_output,
        environment=child_env,
        timeout=timeout,
    )
    if (
        not isinstance(working_directory, str)
        or not os.path.isdir(working_directory)
        or os.path.realpath(working_directory) != working_directory
    ):
        raise CodexBootstrapError("working_directory must be an existing canonical directory")
    expected_model = model if isinstance(model, str) and model else None
    expected_effort = effort if isinstance(effort, str) and effort else None

    # The trust override and route are composed once by the shared Codex
    # argument composer and arrive in ``profile_args``; the bootstrap appends
    # only the app-server suffix (one implementation of trust
    # rendering, owned by the composer).
    argv = [
        codex_binary,
        *list(profile_args),
        *CODEX_APP_SERVER_FLAGS,
    ]
    thread_params: dict[str, Any] = {
        "cwd": working_directory,
        "ephemeral": False,
        "approvalPolicy": "never",
        "sandbox": "danger-full-access",
    }
    # Omit an unset model so Codex applies its own default; the actual model
    # is recorded from the thread/start response.
    if expected_model:
        thread_params["model"] = expected_model
    if developer_instructions:
        thread_params["developerInstructions"] = developer_instructions
    requests: list[dict[str, Any]] = [
        {
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "cao-native-bootstrap",
                    "version": BOOTSTRAP_SCHEMA,
                }
            },
        },
        {"method": "initialized", "params": {}},
        {
            "id": 2,
            "method": "config/read",
            "params": {"cwd": working_directory, "includeLayers": True},
        },
        {"id": 3, "method": "thread/start", "params": thread_params},
    ]

    def materialize(
        responses: Mapping[int, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        thread_id = (((responses.get(3) or {}).get("result") or {}).get("thread") or {}).get("id")
        try:
            native_id = codex_native_launch.validate_session_id(thread_id)
        except codex_native_launch.CodexNativeLaunchError as exc:
            raise CodexBootstrapError(
                f"Codex thread/start returned an unusable thread id before materialization: {exc}"
            ) from exc
        # Naming first (metadata store), then the contentless inject that
        # forces the rollout file.  Both stay in the minter process: after it
        # exits the thread must be loadable by id with zero turns.
        return [
            {
                "id": 4,
                "method": NAMING_METHOD,
                "params": {
                    "threadId": native_id,
                    "name": f"CAO managed Codex session {native_id[:8]}",
                },
            },
            {
                "id": 5,
                "method": MATERIALIZATION_METHOD,
                "params": {
                    "threadId": native_id,
                    "items": _materialization_items(),
                },
            },
        ]

    configured_home = (child_env or os.environ).get("CODEX_HOME")
    effective_home = (child_env or os.environ).get("HOME")
    if configured_home:
        codex_home = pathlib.Path(configured_home).expanduser()
    elif effective_home:
        codex_home = pathlib.Path(effective_home).expanduser() / ".codex"
    else:
        codex_home = pathlib.Path.home() / ".codex"
    config_path = codex_home / "config.toml"
    config_before = _digest_or_absent(config_path)
    probe_error: Optional[Exception] = None
    stdout = stderr = ""
    returncode = -1
    try:
        stdout, stderr, returncode = _run_app_server_probe(
            argv,
            requests,
            timeout,
            env=child_env,
            followup_factory=materialize,
        )
    except Exception as exc:  # noqa: BLE001 - config integrity still checked below
        probe_error = exc
    config_after = _digest_or_absent(config_path)
    if config_before != config_after:
        raise CodexBootstrapError("protected Codex user config changed during native bootstrap")
    if probe_error is not None:
        if isinstance(probe_error, CodexBootstrapError):
            raise probe_error
        raise CodexBootstrapError(f"Codex native bootstrap exchange failed: {probe_error}") from (
            probe_error
        )
    if returncode not in (0, -15):
        raise CodexBootstrapError(f"codex app-server exited {returncode}: {stderr[-500:]}")
    for request_id, name in (
        (1, "initialize"),
        (2, "config/read"),
        (3, "thread/start"),
        (4, NAMING_METHOD),
        (5, MATERIALIZATION_METHOD),
    ):
        response = _response_by_id(stdout, request_id)
        if "error" in response or "result" not in response:
            raise CodexBootstrapError(f"codex {name} failed: {response.get('error')!r}")

    config = _response_by_id(stdout, 2)["result"] or {}
    projects = (config.get("config") or {}).get("projects") or {}
    if (projects.get(working_directory) or {}).get("trust_level") != "trusted":
        raise CodexBootstrapError("Codex did not resolve the exact project as trusted")

    thread = _response_by_id(stdout, 3)["result"] or {}
    thread_id = (thread.get("thread") or {}).get("id")
    try:
        native_id = codex_native_launch.validate_session_id(thread_id)
    except codex_native_launch.CodexNativeLaunchError as exc:
        raise CodexBootstrapError(
            f"Codex thread/start returned an unusable thread id: {exc}"
        ) from exc
    actual_model = thread.get("model")
    actual_effort = thread.get("reasoningEffort")
    actual_cwd = thread.get("cwd")
    # cwd is always supplied and always asserted.  Model and effort are
    # asserted only when the caller supplied a non-empty expected value; an
    # ordinary default-route launch records the provider's actual (possibly
    # null) values instead.
    route_mismatch: list[str] = []
    if actual_cwd != working_directory:
        route_mismatch.append(f"cwd={actual_cwd!r}")
    if expected_model is not None and actual_model != expected_model:
        route_mismatch.append(f"model={actual_model!r} (expected {expected_model!r})")
    if expected_effort is not None and actual_effort != expected_effort:
        route_mismatch.append(f"effort={actual_effort!r} (expected {expected_effort!r})")
    if route_mismatch:
        raise CodexBootstrapError(
            "Codex persistent thread resolved the wrong route or working directory: "
            + ", ".join(route_mismatch)
        )
    rollout_path = _rollout_path(codex_home, native_id)
    resume_adoption = _prove_resume_adoption(
        argv, native_id, timeout, env=child_env, config_path=config_path
    )
    schema_capability = _CAPABILITY_VERDICTS[digest]

    return {
        "schema": BOOTSTRAP_SCHEMA,
        "provider": provider_contracts.PROVIDER_CODEX,
        "native_session_id": native_id,
        "id_source": provider_contracts.native_id_source(provider_contracts.PROVIDER_CODEX),
        "provider_version": provider_contracts.normalized_version(version_output),
        "bootstrap_capability": "zero-turn-resume",
        "capability_proof": {
            "schema": schema_capability,
            "binary_sha256": digest,
            "resume_adoption": resume_adoption,
        },
        "resume_adoption_proof": resume_adoption,
        "binary_path": codex_binary,
        "binary_sha256": digest,
        "working_directory": working_directory,
        "model": actual_model,
        "effort": actual_effort,
        "requested_model": expected_model,
        "requested_effort": expected_effort,
        "sent_no_turn": True,
        "materialization_method": MATERIALIZATION_METHOD,
        "materialization_items_sha256": _digest(_materialization_items()),
        "materialization_sent_no_turn": True,
        "rollout_path": str(rollout_path),
        "rollout_sha256": hashlib.sha256(rollout_path.read_bytes()).hexdigest(),
        "detached_before_launch": True,
        "exit_proof": {
            "schema": EXIT_PROOF_SCHEMA,
            "exit_status": returncode,
            "reaped": True,
        },
        "app_server_exchange_sha256": _digest(
            {
                "initialize": _response_by_id(stdout, 1),
                "config": _response_by_id(stdout, 2),
                "thread": _response_by_id(stdout, 3),
                "naming": _response_by_id(stdout, 4),
                "materialization": _response_by_id(stdout, 5),
            }
        ),
        "protected_config_sha256": config_after,
    }


def bootstrap_intent(receipt: Mapping[str, Any], *, note: Optional[str] = None) -> dict[str, Any]:
    rollout_path = receipt.get("rollout_path") if isinstance(receipt, Mapping) else None
    rollout_sha256 = receipt.get("rollout_sha256") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != BOOTSTRAP_SCHEMA
        or receipt.get("sent_no_turn") is not True
        or receipt.get("materialization_method") != MATERIALIZATION_METHOD
        or receipt.get("materialization_items_sha256") != _digest(_materialization_items())
        or receipt.get("materialization_sent_no_turn") is not True
        or not isinstance(rollout_path, str)
        or not os.path.isabs(rollout_path)
        or not isinstance(rollout_sha256, str)
        or len(rollout_sha256) != 64
        or any(character not in "0123456789abcdef" for character in rollout_sha256)
        or receipt.get("detached_before_launch") is not True
        or (receipt.get("exit_proof") or {}).get("reaped") is not True
    ):
        raise CodexBootstrapError(
            "Codex receipt does not prove a materialized, turn-free, detached bootstrap"
        )
    native_session_id = receipt.get("native_session_id")
    if not isinstance(native_session_id, str) or not rollout_path.endswith(
        f"-{native_session_id}.jsonl"
    ):
        raise CodexBootstrapError("Codex receipt rollout path does not bind the native session id")
    return native_attachment.acquire_intent(
        acquisition_method=native_attachment.ACQUISITION_ZERO_TURN_BOOTSTRAP,
        acquisition_receipt=dict(receipt),
        admits_only_new_instructions=True,
        replays_task_bytes=False,
        bootstrap_sent_no_turn=True,
        bootstrap_detached_before_launch=True,
        note=note,
    )
