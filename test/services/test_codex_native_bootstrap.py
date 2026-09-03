"""The Codex app-server bootstrap mints a persistent thread without a turn."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from cli_agent_orchestrator.services import codex_native_bootstrap as cnb
from cli_agent_orchestrator.services import native_attachment, provider_contracts

SESSION = "019fb17d-0c6d-7161-a408-6b1fa61c8f2d"


@pytest.fixture
def codex_binary(tmp_path):
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    path = os.path.realpath(binary)
    return path, hashlib.sha256(binary.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def isolated_capability_probe(monkeypatch):
    cnb._CAPABILITY_VERDICTS.clear()
    monkeypatch.setattr(
        cnb,
        "_probe_schema_capability",
        lambda *_args, **_kwargs: {"schema": cnb.SCHEMA_PROBE_SCHEMA},
    )


def _response(request_id, result):
    return json.dumps({"id": request_id, "result": result})


def test_unlisted_builds_are_not_withheld_by_a_version_allowlist():
    assert not hasattr(cnb, "BOOTSTRAP_CAPABLE_VERSIONS")
    assert not hasattr(cnb, "is_bootstrap_capable_build")
    assert (
        provider_contracts.is_listed_version(provider_contracts.PROVIDER_CODEX, "codex-cli 0.147.0")
        is False
    )


def test_bootstrap_materializes_a_resumable_zero_turn_thread(tmp_path, codex_binary, monkeypatch):
    seen = {}
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()

    def exchange(argv, requests, timeout, *, env=None, followup_factory=None):
        if followup_factory is None:
            seen["adoption_requests"] = requests
            return (
                "\n".join(
                    [
                        _response(1, {"userAgent": "codex-test"}),
                        _response(2, {"thread": {"id": SESSION, "ephemeral": False, "turns": []}}),
                        _response(3, {"data": []}),
                    ]
                ),
                "",
                0,
            )
        seen["argv"] = argv
        seen["requests"] = requests
        seen["env"] = env
        start_response = {
            "id": 3,
            "result": {
                "thread": {"id": SESSION},
                "model": "gpt-5.6-sol",
                "reasoningEffort": "xhigh",
                "cwd": os.path.realpath(tmp_path),
            },
        }
        followups = followup_factory({3: start_response})
        seen["followups"] = followups
        rollout = codex_home / "sessions" / "2026" / "07" / "30"
        rollout.mkdir(parents=True)
        (rollout / f"rollout-test-{SESSION}.jsonl").write_text('{"type":"session_meta"}\n')
        stdout = "\n".join(
            [
                _response(1, {"userAgent": "codex-test"}),
                _response(
                    2,
                    {
                        "config": {
                            "projects": {os.path.realpath(tmp_path): {"trust_level": "trusted"}}
                        }
                    },
                ),
                json.dumps(start_response),
                _response(4, {}),
                _response(5, {}),
            ]
        )
        return stdout, "", -15

    monkeypatch.setattr(cnb, "_run_app_server_probe", exchange)
    path, digest = codex_binary
    receipt = cnb.mint_session(
        codex_binary=path,
        binary_sha256=digest,
        version_output="codex-cli 0.146.0",
        working_directory=os.path.realpath(tmp_path),
        model="gpt-5.6-sol",
        effort="xhigh",
        profile_args=["--yolo"],
        environment={"PATH": os.environ["PATH"], "CODEX_HOME": str(codex_home)},
    )

    methods = [request["method"] for request in seen["requests"]]
    assert methods == ["initialize", "initialized", "config/read", "thread/start"]
    assert seen["followups"] == [
        {
            "id": 4,
            "method": "thread/name/set",
            "params": {
                "threadId": SESSION,
                "name": f"CAO managed Codex session {SESSION[:8]}",
            },
        },
        {
            "id": 5,
            "method": "thread/inject_items",
            "params": {
                "threadId": SESSION,
                "items": [{"type": "reasoning", "summary": []}],
            },
        },
    ]
    thread_params = seen["requests"][-1]["params"]
    assert thread_params["ephemeral"] is False
    assert "turn/start" not in methods
    assert "turn/start" not in [request["method"] for request in seen["followups"]]
    assert receipt["native_session_id"] == SESSION
    assert receipt["sent_no_turn"] is True
    assert receipt["materialization_method"] == "thread/inject_items"
    assert receipt["materialization_items_sha256"] == cnb._digest(
        [{"type": "reasoning", "summary": []}]
    )
    assert receipt["materialization_sent_no_turn"] is True
    assert receipt["rollout_path"].endswith(f"-{SESSION}.jsonl")
    assert receipt["detached_before_launch"] is True
    assert receipt["exit_proof"]["reaped"] is True
    assert receipt["resume_adoption_proof"]["observed_turns"] == []
    assert seen["argv"][-2:] == ["app-server", "--stdio"]
    assert [request["method"] for request in seen["adoption_requests"]] == [
        "initialize",
        "initialized",
        "thread/resume",
        "thread/turns/list",
    ]
    assert seen["adoption_requests"][-2]["params"] == {"threadId": SESSION}
    assert seen["adoption_requests"][-1]["params"] == {"threadId": SESSION}
    assert "turn/start" not in [request["method"] for request in seen["adoption_requests"]]

    intent = cnb.bootstrap_intent(receipt)
    assert intent["acquisition_method"] == native_attachment.ACQUISITION_ZERO_TURN_BOOTSTRAP


def test_default_route_omits_model_and_records_provider_observation(
    tmp_path, codex_binary, monkeypatch
):
    seen = {}
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()

    def exchange(argv, requests, _timeout, *, env=None, followup_factory=None):
        if followup_factory is None:
            return (
                "\n".join(
                    [
                        _response(1, {"userAgent": "codex-test"}),
                        _response(2, {"thread": {"id": SESSION, "ephemeral": False}}),
                        _response(3, {"data": []}),
                    ]
                ),
                "",
                0,
            )
        seen["argv"] = argv
        seen["requests"] = requests
        start_response = {
            "id": 3,
            "result": {
                "thread": {"id": SESSION},
                "model": "gpt-5.6-sol",
                "reasoningEffort": None,
                "cwd": os.path.realpath(tmp_path),
            },
        }
        followup_factory({3: start_response})
        rollout = codex_home / "sessions" / "2026" / "08" / "09"
        rollout.mkdir(parents=True)
        (rollout / f"rollout-test-{SESSION}.jsonl").write_text('{"type":"session_meta"}\n')
        return (
            "\n".join(
                [
                    _response(1, {"userAgent": "codex-test"}),
                    _response(
                        2,
                        {
                            "config": {
                                "projects": {os.path.realpath(tmp_path): {"trust_level": "trusted"}}
                            }
                        },
                    ),
                    json.dumps(start_response),
                    _response(4, {}),
                    _response(5, {}),
                ]
            ),
            "",
            0,
        )

    monkeypatch.setattr(cnb, "_run_app_server_probe", exchange)
    path, digest = codex_binary
    profile_args = [
        "--yolo",
        "-c",
        'projects={"' + os.path.realpath(tmp_path) + '":{trust_level="trusted"}}',
    ]
    receipt = cnb.mint_session(
        codex_binary=path,
        binary_sha256=digest,
        version_output="codex-cli 0.147.0",
        working_directory=os.path.realpath(tmp_path),
        model=None,
        effort=None,
        profile_args=profile_args,
        environment={"CODEX_HOME": str(codex_home)},
    )

    assert seen["argv"] == [path, *profile_args, "app-server", "--stdio"]
    assert seen["requests"][-1] == {
        "id": 3,
        "method": "thread/start",
        "params": {
            "cwd": os.path.realpath(tmp_path),
            "ephemeral": False,
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
        },
    }
    assert receipt["model"] == "gpt-5.6-sol"
    assert receipt["effort"] is None
    assert receipt["requested_model"] is None
    assert receipt["requested_effort"] is None


def test_route_drift_is_refused(tmp_path, codex_binary, monkeypatch):
    seen = {}
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()

    def exchange(_argv, requests, _timeout, *, env=None, followup_factory=None):
        if followup_factory is None:
            return (
                "\n".join(
                    [
                        _response(1, {}),
                        _response(2, {"thread": {"id": SESSION, "ephemeral": False}}),
                        _response(3, {"data": []}),
                    ]
                ),
                "",
                0,
            )
        seen["requests"] = requests
        start_response = {
            "id": 3,
            "result": {
                "thread": {"id": SESSION},
                "model": "wrong-model",
                "reasoningEffort": "xhigh",
                "cwd": os.path.realpath(tmp_path),
            },
        }
        followup_factory({3: start_response})
        stdout = "\n".join(
            [
                _response(1, {}),
                _response(
                    2,
                    {
                        "config": {
                            "projects": {os.path.realpath(tmp_path): {"trust_level": "trusted"}}
                        }
                    },
                ),
                json.dumps(start_response),
                _response(4, {}),
                _response(5, {}),
            ]
        )
        return stdout, "", 0

    monkeypatch.setattr(cnb, "_run_app_server_probe", exchange)
    path, digest = codex_binary
    with pytest.raises(cnb.CodexBootstrapError) as error:
        cnb.mint_session(
            codex_binary=path,
            binary_sha256=digest,
            version_output="codex-cli 0.146.0",
            working_directory=os.path.realpath(tmp_path),
            model="gpt-5.6-sol",
            effort="xhigh",
            profile_args=[],
            environment={"CODEX_HOME": str(codex_home)},
        )
    assert str(error.value) == (
        "Codex persistent thread resolved the wrong route or working directory: "
        "model='wrong-model' (expected 'gpt-5.6-sol')"
    )
    assert seen["requests"][-1]["params"]["model"] == "gpt-5.6-sol"


def test_bootstrap_refuses_when_materialization_leaves_no_rollout(
    tmp_path, codex_binary, monkeypatch
):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()

    def exchange(_argv, _requests, _timeout, *, env=None, followup_factory=None):
        if followup_factory is None:
            return (
                "\n".join(
                    [
                        _response(1, {}),
                        _response(2, {"thread": {"id": SESSION, "ephemeral": False}}),
                        _response(3, {"data": []}),
                    ]
                ),
                "",
                0,
            )
        start_response = {
            "id": 3,
            "result": {
                "thread": {"id": SESSION},
                "model": "gpt-5.6-sol",
                "reasoningEffort": "xhigh",
                "cwd": os.path.realpath(tmp_path),
            },
        }
        followup_factory({3: start_response})
        return (
            "\n".join(
                [
                    _response(1, {}),
                    _response(
                        2,
                        {
                            "config": {
                                "projects": {os.path.realpath(tmp_path): {"trust_level": "trusted"}}
                            }
                        },
                    ),
                    json.dumps(start_response),
                    _response(4, {}),
                    _response(5, {}),
                ]
            ),
            "",
            0,
        )

    monkeypatch.setattr(cnb, "_run_app_server_probe", exchange)
    path, digest = codex_binary
    with pytest.raises(cnb.CodexBootstrapError, match="exactly one resumable rollout"):
        cnb.mint_session(
            codex_binary=path,
            binary_sha256=digest,
            version_output="codex-cli 0.146.0",
            working_directory=os.path.realpath(tmp_path),
            model="gpt-5.6-sol",
            effort="xhigh",
            profile_args=[],
            environment={"CODEX_HOME": str(codex_home)},
        )


def test_bootstrap_checks_protected_config_even_when_exchange_raises(
    tmp_path, codex_binary, monkeypatch
):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    config = codex_home / "config.toml"
    config.write_text('model = "safe"\n')

    def exchange(_argv, _requests, _timeout, *, env=None, followup_factory=None):
        config.write_text('model = "mutated"\n')
        raise RuntimeError("app-server timed out")

    monkeypatch.setattr(cnb, "_run_app_server_probe", exchange)
    path, digest = codex_binary
    with pytest.raises(cnb.CodexBootstrapError, match="protected Codex user config changed"):
        cnb.mint_session(
            codex_binary=path,
            binary_sha256=digest,
            version_output="codex-cli 0.146.0",
            working_directory=os.path.realpath(tmp_path),
            model="gpt-5.6-sol",
            effort="xhigh",
            profile_args=[],
            environment={"CODEX_HOME": str(codex_home)},
        )


def test_bootstrap_intent_refuses_an_unmaterialized_legacy_receipt():
    with pytest.raises(cnb.CodexBootstrapError, match="materialized"):
        cnb.bootstrap_intent(
            {
                "schema": cnb.BOOTSTRAP_SCHEMA,
                "provider": "codex",
                "native_session_id": SESSION,
                "sent_no_turn": True,
                "detached_before_launch": True,
                "exit_proof": {"reaped": True},
            }
        )


def test_materialization_binds_the_exact_underscore_wire_method():
    # The schema title reads Thread/injectItemsRequest but the wire method is
    # thread/inject_items; the camelCase guess is an unknown method server-side.
    assert cnb.MATERIALIZATION_METHOD == "thread/inject_items"
    assert cnb.NAMING_METHOD == "thread/name/set"
    assert cnb._SCHEMA_REQUIREMENTS["thread/inject_items"] == (
        "ThreadInjectItemsParams",
        "ThreadInjectItemsResponse",
        ("threadId", "items"),
    )


def test_materialization_payload_is_contentless_and_fresh_per_mint():
    first = cnb._materialization_items()
    assert first == [{"type": "reasoning", "summary": []}]
    first[0]["summary"].append({"text": "mutated"})
    assert cnb._materialization_items() == [{"type": "reasoning", "summary": []}]


def _mint_exchange_with_adoption(tmp_path, codex_home, *, resume_thread, turns_data):
    """Fake both legs: the minter succeeds with a rollout file on disk, and
    the fresh-process adoption leg replays the given resume thread/turns."""

    def exchange(_argv, _requests, _timeout, *, env=None, followup_factory=None):
        if followup_factory is None:
            return (
                "\n".join(
                    [
                        _response(1, {}),
                        _response(2, {"thread": resume_thread}),
                        _response(3, {"data": turns_data}),
                    ]
                ),
                "",
                0,
            )
        start_response = {
            "id": 3,
            "result": {
                "thread": {"id": SESSION},
                "model": "gpt-5.6-sol",
                "reasoningEffort": "xhigh",
                "cwd": os.path.realpath(tmp_path),
            },
        }
        followup_factory({3: start_response})
        rollout = codex_home / "sessions" / "2026" / "09" / "03"
        rollout.mkdir(parents=True)
        (rollout / f"rollout-test-{SESSION}.jsonl").write_text('{"type":"session_meta"}\n')
        return (
            "\n".join(
                [
                    _response(1, {}),
                    _response(
                        2,
                        {
                            "config": {
                                "projects": {os.path.realpath(tmp_path): {"trust_level": "trusted"}}
                            }
                        },
                    ),
                    json.dumps(start_response),
                    _response(4, {}),
                    _response(5, {}),
                ]
            ),
            "",
            0,
        )

    return exchange


def _mint_kwargs(tmp_path, codex_home, codex_binary):
    path, digest = codex_binary
    return {
        "codex_binary": path,
        "binary_sha256": digest,
        "version_output": "codex-cli 0.151.0",
        "working_directory": os.path.realpath(tmp_path),
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "profile_args": [],
        "environment": {"CODEX_HOME": str(codex_home)},
    }


def test_adoption_refuses_a_thread_with_listed_turns(tmp_path, codex_binary, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setattr(
        cnb,
        "_run_app_server_probe",
        _mint_exchange_with_adoption(
            tmp_path,
            codex_home,
            resume_thread={"id": SESSION},
            turns_data=[{"id": "turn-1"}],
        ),
    )
    with pytest.raises(cnb.CodexBootstrapError, match="not a zero-turn thread"):
        cnb.mint_session(**_mint_kwargs(tmp_path, codex_home, codex_binary))


def test_adoption_refuses_inline_turns_on_the_resumed_thread(tmp_path, codex_binary, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setattr(
        cnb,
        "_run_app_server_probe",
        _mint_exchange_with_adoption(
            tmp_path,
            codex_home,
            resume_thread={"id": SESSION, "turns": [{"id": "turn-1"}]},
            turns_data=[],
        ),
    )
    with pytest.raises(cnb.CodexBootstrapError, match="not a zero-turn thread"):
        cnb.mint_session(**_mint_kwargs(tmp_path, codex_home, codex_binary))


def test_bootstrap_refuses_when_inject_reports_an_error(tmp_path, codex_binary, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()

    def exchange(_argv, _requests, _timeout, *, env=None, followup_factory=None):
        if followup_factory is None:
            raise AssertionError("mint must fail before the adoption probe")
        start_response = {
            "id": 3,
            "result": {
                "thread": {"id": SESSION},
                "model": "gpt-5.6-sol",
                "reasoningEffort": "xhigh",
                "cwd": os.path.realpath(tmp_path),
            },
        }
        followup_factory({3: start_response})
        return (
            "\n".join(
                [
                    _response(1, {}),
                    _response(
                        2,
                        {
                            "config": {
                                "projects": {os.path.realpath(tmp_path): {"trust_level": "trusted"}}
                            }
                        },
                    ),
                    json.dumps(start_response),
                    _response(4, {}),
                    json.dumps(
                        {
                            "id": 5,
                            "error": {"code": -32600, "message": "items must not be empty"},
                        }
                    ),
                ]
            ),
            "",
            0,
        )

    monkeypatch.setattr(cnb, "_run_app_server_probe", exchange)
    path, digest = codex_binary
    with pytest.raises(cnb.CodexBootstrapError, match="thread/inject_items"):
        cnb.mint_session(**_mint_kwargs(tmp_path, codex_home, codex_binary))


def _valid_intent_receipt():
    return {
        "schema": cnb.BOOTSTRAP_SCHEMA,
        "provider": "codex",
        "native_session_id": SESSION,
        "sent_no_turn": True,
        "materialization_method": cnb.MATERIALIZATION_METHOD,
        "materialization_items_sha256": cnb._digest(cnb._materialization_items()),
        "materialization_sent_no_turn": True,
        "rollout_path": f"/tmp/rollout-test-{SESSION}.jsonl",
        "rollout_sha256": "b" * 64,
        "detached_before_launch": True,
        "exit_proof": {"reaped": True},
    }


def test_bootstrap_intent_binds_the_contentless_materialization_payload():
    intent = cnb.bootstrap_intent(_valid_intent_receipt())
    assert intent["acquisition_method"] == native_attachment.ACQUISITION_ZERO_TURN_BOOTSTRAP


def test_bootstrap_intent_refuses_a_legacy_name_set_receipt():
    receipt = _valid_intent_receipt()
    receipt["materialization_method"] = "thread/name/set"
    with pytest.raises(cnb.CodexBootstrapError, match="materialized"):
        cnb.bootstrap_intent(receipt)


def test_bootstrap_intent_refuses_a_non_canonical_materialization_payload():
    receipt = _valid_intent_receipt()
    receipt["materialization_items_sha256"] = cnb._digest(
        [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}]
    )
    with pytest.raises(cnb.CodexBootstrapError, match="materialized"):
        cnb.bootstrap_intent(receipt)


def _schema_fixture():
    definitions = {}
    requests = []
    for method, (params, response, fields) in cnb._SCHEMA_REQUIREMENTS.items():
        definitions[params] = {"properties": {field: {} for field in fields}}
        definitions[response] = {
            "properties": {"thread": {}} if "thread" in response.lower() else {}
        }
        requests.append(
            {
                "properties": {
                    "method": {"enum": [method]},
                    "params": {"$ref": f"#/definitions/v2/{params}"},
                }
            }
        )
    return {"definitions": {"v2": definitions}, "oneOf": requests}


def test_schema_probe_requires_every_load_bearing_method_and_field():
    assert set(cnb._validate_schema_bundle(_schema_fixture())["methods"]) == set(
        cnb._SCHEMA_REQUIREMENTS
    )
    broken = _schema_fixture()
    del broken["definitions"]["v2"]["ThreadResumeParams"]["properties"]["threadId"]
    with pytest.raises(cnb.CodexBootstrapError, match="thread/resume.*threadId"):
        cnb._validate_schema_bundle(broken)


def test_schema_capability_is_cached_by_binary_digest(tmp_path, codex_binary, monkeypatch):
    path, digest = codex_binary
    calls = []

    def probe(*args, **kwargs):
        calls.append((args, kwargs))
        return {"schema": cnb.SCHEMA_PROBE_SCHEMA}

    monkeypatch.setattr(cnb, "_probe_schema_capability", probe)
    assert cnb._validate_binary(path, digest, "codex-cli 99.99.0") == digest
    assert cnb._validate_binary(path, digest, "codex-cli 99.99.0") == digest
    assert len(calls) == 1


def test_transient_schema_probe_failure_is_retried_for_the_same_digest(
    tmp_path, codex_binary, monkeypatch
):
    path, digest = codex_binary
    calls = []
    transient = cnb.CodexSchemaProbeTransientError("timed out")

    def probe(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise transient
        return {"schema": cnb.SCHEMA_PROBE_SCHEMA}

    monkeypatch.setattr(cnb, "_probe_schema_capability", probe)
    with pytest.raises(cnb.CodexSchemaProbeTransientError, match="timed out"):
        cnb._validate_binary(path, digest, "codex-cli 99.99.0")
    assert cnb._validate_binary(path, digest, "codex-cli 99.99.0") == digest
    assert len(calls) == 2


def test_deterministic_schema_probe_failure_is_cached_by_binary_digest(
    tmp_path, codex_binary, monkeypatch
):
    path, digest = codex_binary
    calls = []
    deterministic = cnb.CodexBootstrapError("missing method thread/resume")

    def probe(*args, **kwargs):
        calls.append((args, kwargs))
        raise deterministic

    monkeypatch.setattr(cnb, "_probe_schema_capability", probe)
    for _ in range(2):
        with pytest.raises(cnb.CodexBootstrapError, match="thread/resume"):
            cnb._validate_binary(path, digest, "codex-cli 99.99.0")
    assert len(calls) == 1


def test_repeated_mints_keep_the_digest_schema_verdict_and_bindable_proofs(
    tmp_path, codex_binary, monkeypatch
):
    path, digest = codex_binary
    schema_verdict = cnb._validate_schema_bundle(_schema_fixture())
    monkeypatch.setattr(cnb, "_probe_schema_capability", lambda *_a, **_k: schema_verdict)
    homes = [tmp_path / "codex-home-one", tmp_path / "codex-home-two"]
    for home in homes:
        home.mkdir()
    ids = [SESSION, "01a01a53-0000-7000-8000-000000000000"]

    def exchange(_argv, requests, _timeout, *, env=None, followup_factory=None):
        index = 0 if env["CODEX_HOME"] == str(homes[0]) else 1
        native_id = ids[index]
        if followup_factory is None:
            return (
                "\n".join(
                    [
                        _response(1, {}),
                        _response(2, {"thread": {"id": native_id, "ephemeral": False}}),
                        _response(3, {"data": []}),
                    ]
                ),
                "",
                0,
            )
        start = {
            "id": 3,
            "result": {
                "thread": {"id": native_id},
                "model": "gpt-5.6-sol",
                "reasoningEffort": "xhigh",
                "cwd": os.path.realpath(tmp_path),
            },
        }
        followup_factory({3: start})
        rollout = Path(env["CODEX_HOME"]) / "sessions" / "2026" / "08" / "25"
        rollout.mkdir(parents=True)
        (rollout / f"rollout-test-{native_id}.jsonl").write_text("{}\n")
        return (
            "\n".join(
                [
                    _response(1, {}),
                    _response(
                        2,
                        {
                            "config": {
                                "projects": {os.path.realpath(tmp_path): {"trust_level": "trusted"}}
                            }
                        },
                    ),
                    json.dumps(start),
                    _response(4, {}),
                    _response(5, {}),
                ]
            ),
            "",
            0,
        )

    monkeypatch.setattr(cnb, "_run_app_server_probe", exchange)
    receipts = [
        cnb.mint_session(
            codex_binary=path,
            binary_sha256=digest,
            version_output="codex-cli 0.999.0",
            working_directory=os.path.realpath(tmp_path),
            model="gpt-5.6-sol",
            effort="xhigh",
            profile_args=[],
            environment={"CODEX_HOME": str(home)},
        )
        for home in homes
    ]
    assert all(receipt["capability_proof"]["schema"] == schema_verdict for receipt in receipts)
    assert [receipt["resume_adoption_proof"]["adopted_session_id"] for receipt in receipts] == ids
