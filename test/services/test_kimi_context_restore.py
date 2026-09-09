"""Kimi PostCompact adapter: notify-only hook, fenced delivery, preserved base.

The hook notifies; it never injects (PostCompact output is ignored
provider-side) and never delivers by itself. Delivery happens only
through the fork boundary after a read-only projection with a verified
fence. The managed block composes textually into config.toml: user
hooks stay byte-identical, refresh replaces only our block, teardown
removes exactly it. Main always exits 0 — a notification must never
block or fail a compaction.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
import uuid as _uuid

import pytest

from cli_agent_orchestrator.services import execution_mode as em
from cli_agent_orchestrator.services import kimi_context_restore as kr
from cli_agent_orchestrator.services import kimi_native_control as knc
from cli_agent_orchestrator.services import native_attachment as na


@pytest.fixture(autouse=True)
def _db(isolated_memory_db):
    return isolated_memory_db


def test_hook_toml_is_postcompact_only():
    toml = kr.restore_hook_toml(command="/bin/cao-kimi-hook-context --terminal t")
    assert 'event = "PostCompact"' in toml
    assert "SessionStart" not in toml
    assert "UserPromptSubmit" not in toml
    assert "timeout =" in toml


def test_compose_preserves_user_hooks_byte_identical():
    base = (
        '[general]\ntheme = "dark"\n\n[[hooks]]\nevent = "PreToolUse"\n'
        'command = "/home/u/my-hook"\n'
    )
    composed = kr.compose_managed_config(base, command="/bin/w --terminal t")
    assert composed.startswith(base)
    assert kr.MANAGED_BLOCK_BEGIN in composed
    assert "/home/u/my-hook" in composed


def test_refresh_replaces_only_managed_block():
    first = kr.compose_managed_config("", command="/bin/w --terminal t")
    second = kr.compose_managed_config(first, command="/bin/w --terminal t2")
    assert second.count(kr.MANAGED_BLOCK_BEGIN) == 1
    assert "--terminal t2" in second


def test_teardown_removes_exactly_managed_block():
    base = '[[hooks]]\nevent = "PreToolUse"\ncommand = "/home/u/my-hook"\n'
    composed = kr.compose_managed_config(base, command="/bin/w --terminal t")
    assert kr.teardown_managed_config(composed) == base
    assert kr.teardown_managed_config(base) == base


def test_conduct_argv_is_read_only_projection():
    argv = kr.build_conduct_argv(
        conduct_binary="/usr/bin/conduct",
        native_session_id="sess-1",
        terminal_id="term-1",
        terminal_generation="gen-3",
    )
    assert argv[:3] == ["/usr/bin/conduct", "goal", "hook-context"]
    assert "--harness" in argv and "kimi_cli" in argv
    joined = " ".join(argv)
    for forbidden in (
        "resume",
        "steer",
        "send",
        "release",
        "approve",
        "checkpoint",
        "claim",
        "satisfy",
    ):
        assert forbidden not in joined


def test_parse_hook_input_accepts_kimi_snake_case():
    raw = json.dumps(
        {
            "session_id": "sess-9",
            "cwd": "/tmp/w",
            "hook_event_name": "PostCompact",
            "trigger": "auto",
            "estimated_token_count": 120000,
        }
    ).encode()
    parsed = kr.parse_hook_input(raw)
    assert parsed["session_id"] == "sess-9"
    assert parsed["hook_event_name"] == "PostCompact"
    assert kr.parse_hook_input(b"not json") == {}


def test_render_restoration_requires_verified_ok():
    assert kr.render_restoration({"result_type": "stale"}) is None
    assert (
        kr.render_restoration(
            {
                "result_type": "ok",
                "identity": {"generation_fence": "unverified"},
                "goal": {"objective": "x"},
            }
        )
        is None
    )
    text = kr.render_restoration(_projection())
    assert text is not None
    assert "cao-context-restoration" in text
    assert "ship it" in text
    assert "not a new assignment" in text
    assert len(text) <= kr.MAX_CONTEXT_CHARS + len("\n[truncated]")


def test_render_reads_full_compact_projection():
    # P1-4: completion_requirements (not the nonexistent
    # "requirements"), checkpoint/evidence refs, hold owner, next
    # action — one renderer for event and periodic alike.
    text = kr.render_restoration(_projection())
    assert "final-report" in text and "required-artifacts" in text
    assert "ev-9" in text and "halfway there" in text
    assert "evidence entries: 2" in text
    assert "draft the report" in text
    held = kr.render_restoration(
        _projection(
            goal={
                "active_hold": {
                    "hold_id": "h1",
                    "state": "active",
                    "reason_kind": "human-approval",
                    "release_kind": "gate",
                    "release_id": "g7",
                    "requested_by_role": "supervisor",
                    "requested_by_agent_id": "sup-1",
                    "decision_authority": "operator",
                }
            }
        )
    )
    assert "human-approval" in held
    assert "gate:g7" in held
    assert "supervisor/sup-1" in held


def test_render_refuses_blank_objective():
    assert kr.render_restoration(_projection(goal={"objective": "  "})) is None


def test_render_ignores_legacy_requirements_field():
    # The capsule never carries "requirements": a legacy field must
    # not leak into restoration text.
    text = kr.render_restoration(
        _projection(
            goal={
                "completion_requirements": [],
                "requirements": [{"id": "r-legacy", "summary": "ghost"}],
            }
        )
    )
    assert "r-legacy" not in text
    assert "completion requirements" not in text


def test_render_mutations_change_sent_text():
    # Mutation tripwire: requirement, checkpoint, and next-action
    # edits must all move the rendered bytes.
    base = kr.render_restoration(_projection())
    assert (
        kr.render_restoration(
            _projection(
                goal={
                    "completion_requirements": [{"id": "changed-req", "kind": "report-occurrence"}]
                }
            )
        )
        != base
    )
    assert (
        kr.render_restoration(
            _projection(
                goal={
                    "latest_checkpoint": {
                        "event_id": "ev-10",
                        "kind": "checkpoint",
                        "goal_version": 3,
                        "recorded_at": "t2",
                        "summary": "moved on",
                    }
                }
            )
        )
        != base
    )
    assert kr.render_restoration(_projection(next_action="stand down")) != base


def test_wrapper_delivers_nothing_without_verified_fence(monkeypatch):
    calls = {}

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        return {"result_type": "stale"}

    monkeypatch.setattr(kr, "_run_conduct", fake_run)
    monkeypatch.setattr(
        kr, "_post_boundary", lambda **k: (_ for _ in ()).throw(AssertionError("must not post"))
    )
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "s"},
            terminal_id="t",
            terminal_generation="g",
            native_session_id="s",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )


def _never_post(**kwargs):
    raise AssertionError("deferred run must not POST")


def test_wrapper_busy_lock_defers_without_post(tmp_path, monkeypatch):
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    flock_path = str(lockdir / "goal-effect.lock")
    from cli_agent_orchestrator.services import goal_effect_flock as flockmod

    def busy(path, **kwargs):
        raise flockmod.GoalEffectBusy("held")

    monkeypatch.setattr(
        kr, "_run_conduct", lambda argv, timeout_seconds=None, **k: _fenced_answer(1, flock_path)
    )
    monkeypatch.setattr(flockmod, "hold_path", busy)
    monkeypatch.setattr(kr, "_post_boundary", _never_post)
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "sess-1"},
            terminal_id="t",
            terminal_generation="gen-1",
            native_session_id="sess-1",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    assert "deferred" in err.getvalue()
    assert "no bytes sent" in err.getvalue()


def test_wrapper_fresh_transport_error_defers(tmp_path, monkeypatch):
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    flock_path = str(lockdir / "goal-effect.lock")
    calls = iter([_fenced_answer(1, flock_path), {"transport_error": "conduct died"}])
    monkeypatch.setattr(kr, "_run_conduct", lambda argv, timeout_seconds=None, **k: next(calls))
    monkeypatch.setattr(kr, "_post_boundary", _never_post)
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "sess-1"},
            terminal_id="t",
            terminal_generation="gen-1",
            native_session_id="sess-1",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    assert "fresh projection unreadable" in err.getvalue()
    assert "no bytes sent" in err.getvalue()


def test_wrapper_fresh_missing_fence_defers(tmp_path, monkeypatch):
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    flock_path = str(lockdir / "goal-effect.lock")
    bare = {
        "result_type": "ok",
        "identity": {"generation_fence": "verified"},
        "goal": {"objective": "ship it", "goal_version": 2},
    }
    calls = iter([_fenced_answer(1, flock_path), bare])
    monkeypatch.setattr(kr, "_run_conduct", lambda argv, timeout_seconds=None, **k: next(calls))
    monkeypatch.setattr(kr, "_post_boundary", _never_post)
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "sess-1"},
            terminal_id="t",
            terminal_generation="gen-1",
            native_session_id="sess-1",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    assert "carries no delivery fence" in err.getvalue()


def test_wrapper_changed_lock_path_defers(tmp_path, monkeypatch):
    old = tmp_path / "projA"
    old.mkdir()
    new = tmp_path / "projB"
    new.mkdir()
    calls = iter(
        [
            _fenced_answer(1, str(old / "goal-effect.lock")),
            _fenced_answer(2, str(new / "goal-effect.lock")),
        ]
    )
    monkeypatch.setattr(kr, "_run_conduct", lambda argv, timeout_seconds=None, **k: next(calls))
    monkeypatch.setattr(kr, "_post_boundary", _never_post)
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "sess-1"},
            terminal_id="t",
            terminal_generation="gen-1",
            native_session_id="sess-1",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    assert "different project lock" in err.getvalue()


def test_wrapper_rotated_native_defers(tmp_path, monkeypatch):
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    flock_path = str(lockdir / "goal-effect.lock")
    second = _fenced_answer(2, flock_path)
    second["delivery_fence"]["native_session_id"] = "sess-rotated"
    calls = iter([_fenced_answer(1, flock_path), second])
    monkeypatch.setattr(kr, "_run_conduct", lambda argv, timeout_seconds=None, **k: next(calls))
    monkeypatch.setattr(kr, "_post_boundary", _never_post)
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "sess-1"},
            terminal_id="t",
            terminal_generation="gen-1",
            native_session_id="sess-1",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    assert "rotated" in err.getvalue()


def test_wrapper_hold_activation_between_reads_delivers_current_only(tmp_path, monkeypatch):
    # Locator read showed goal v1; a hold activation committed before
    # the shared hold; the fresh re-read shows v1→v2 with raised water.
    # Only the current fence is ever POSTed — the stale v1 snapshot is
    # not delivery, even though it was read first.
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    flock_path = str(lockdir / "goal-effect.lock")
    first = _fenced_answer(1, flock_path)
    first["delivery_fence"]["hold_high_water"] = 0
    second = _fenced_answer(2, flock_path)
    second["delivery_fence"]["hold_high_water"] = 5
    calls = iter([first, second])
    posted = {}
    monkeypatch.setattr(kr, "_run_conduct", lambda argv, timeout_seconds=None, **k: next(calls))
    monkeypatch.setattr(
        kr,
        "_post_boundary",
        lambda **k: (
            posted.update(
                fence=k["fence"],
                projection=k["projection"],
                hook_evidence=k["hook_evidence"],
                request_id=k["request_id"],
            )
            or {"status": "posted"}
        ),
    )
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "sess-1", "trigger": "auto", "estimated_token_count": 42},
            terminal_id="t",
            terminal_generation="gen-1",
            native_session_id="sess-1",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    assert posted["fence"]["goal_version"] == 2
    assert posted["fence"]["hold_high_water"] == 5
    assert "cao-context-restoration" in kr.render_restoration(posted["projection"])
    # One run is one request identity, bound to native evidence.
    _uuid.UUID(posted["request_id"], version=4)
    assert posted["hook_evidence"]["session_id"] == "sess-1"
    assert posted["hook_evidence"]["trigger"] == "auto"
    assert posted["hook_evidence"]["estimated_token_count"] == 42


def test_wrapper_refuses_stdin_identity_mismatch(monkeypatch):
    monkeypatch.setattr(
        kr, "_run_conduct", lambda **k: (_ for _ in ()).throw(AssertionError("must not run"))
    )
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "other"},
            terminal_id="t",
            terminal_generation="g",
            native_session_id="baked",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    assert "does not match" in err.getvalue()


def test_wrapper_degrades_when_boundary_unreachable(tmp_path, monkeypatch):
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    flock_path = str(lockdir / "goal-effect.lock")
    monkeypatch.setattr(
        kr, "_run_conduct", lambda argv, timeout_seconds=None, **k: _fenced_answer(2, flock_path)
    )
    monkeypatch.setattr(kr, "_post_boundary", lambda **k: {"transport_error": "down"})
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "sess-1"},
            terminal_id="t",
            terminal_generation="gen-1",
            native_session_id="sess-1",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    assert "unreachable" in err.getvalue()


def test_restore_command_bakes_all_binding():
    cmd = kr.restore_command(
        wrapper_executable="/bin/w",
        terminal_id="t",
        generation="g",
        conduct_binary="/bin/conduct",
        fork_base="http://b:1",
        native_session_id="sess-x",
    )
    assert "--terminal t" in cmd and "--conduct /bin/conduct" in cmd
    assert "sess-x" in cmd and "http://b:1" in cmd


def test_default_fork_base_matches_conductor_rule(monkeypatch):
    monkeypatch.delenv("CAO_API_HOST", raising=False)
    monkeypatch.delenv("CAO_API_PORT", raising=False)
    assert kr.default_fork_base() == "http://127.0.0.1:9889"
    monkeypatch.setenv("CAO_API_PORT", "1234")
    assert kr.default_fork_base() == "http://127.0.0.1:1234"


def test_prepare_degrades_never_refuses(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(kr, "resolve_wrapper_executable", lambda **k: None)
    env, installation = kr.prepare_kimi_restoration(
        record={"terminal_id": "t", "generation": "g"},
        base_environment={"PATH": "/bin"},
        companion_dir=str(tmp_path),
    )
    assert installation["mechanism"] is None
    assert "degraded_reason" in installation and installation["degraded_reason"]
    assert "KIMI_CODE_HOME" not in env
    _ = calls


def test_prepare_composes_private_home(tmp_path, monkeypatch):
    monkeypatch.setattr(kr, "resolve_wrapper_executable", lambda **k: "/bin/w")
    monkeypatch.setattr(kr, "resolve_conduct_binary", lambda **k: "/bin/conduct")
    env, installation = kr.prepare_kimi_restoration(
        record={"terminal_id": "t9", "generation": "g2", "native_session_id": "s"},
        base_environment={"KIMI_CODE_HOME": str(tmp_path / "opaque"), "PATH": "/bin"},
        companion_dir=str(tmp_path / "companion"),
    )
    assert installation["mechanism"] == kr.MECHANISM
    home = installation["kimi_home"]
    assert home is not None and "t9-g2" in home
    assert env["KIMI_CODE_HOME"] == home
    text = open(home + "/config.toml").read()
    assert 'event = "PostCompact"' in text
    assert "--terminal t9" in text


def test_post_boundary_passes_request_identity_through(monkeypatch):
    # The run mints the request id once; the POST transports it with
    # the projection and the native evidence — it never mints its own
    # (a POST-local UUID would sever retries from their event).
    import urllib.request

    bodies = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"status": "posted"}).encode()

    def fake(req, timeout=None):
        bodies.append(json.loads(req.data.decode()))
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fence = {
        "occurrence_id": "o",
        "terminal_generation": "g",
        "native_session_id": "n",
        "goal_version": 1,
        "hold_high_water": 0,
        "flock_path": "/tmp/x/goal-effect.lock",
    }
    projection = _projection()
    evidence = _evidence()
    kr._post_boundary(
        fork_base="http://x",
        terminal_id="t",
        fence=fence,
        request_id="req-run-1",
        projection=projection,
        hook_evidence=evidence,
        timeout_seconds=1,
    )
    kr._post_boundary(
        fork_base="http://x",
        terminal_id="t",
        fence=fence,
        request_id="req-run-1",
        projection=projection,
        hook_evidence=evidence,
        timeout_seconds=1,
    )
    assert [b["operation_id"] for b in bodies] == ["req-run-1"] * 2
    assert bodies[0]["projection"] == projection
    assert bodies[0]["hook_evidence"] == evidence
    assert "context" not in bodies[0]


def test_two_runs_mint_distinct_request_ids(tmp_path, monkeypatch):
    # Mutation tripwire: sharing one id across runs would merge two
    # compactions into one event.
    ids = []

    def fake_post(**kwargs):
        ids.append(kwargs["request_id"])
        return {"status": "posted"}

    flock_path = str(tmp_path / "goal-effect.lock")
    answers = iter([_fenced_answer(1, flock_path) for _ in range(4)])
    monkeypatch.setattr(kr, "_run_conduct", lambda argv, timeout_seconds=None, **k: next(answers))
    monkeypatch.setattr(kr, "_post_boundary", fake_post)
    out, err = io.StringIO(), io.StringIO()
    for _ in range(2):
        assert (
            kr.run_wrapper(
                hook_input={"session_id": "sess-1"},
                terminal_id="t",
                terminal_generation="gen-1",
                native_session_id="sess-1",
                conduct_binary="/usr/bin/conduct",
                fork_base="http://x",
                out=out,
                err=err,
            )
            == 0
        )
    assert len(ids) == 2 and ids[0] != ids[1]


def _fenced_answer(version, flock_path):
    return {
        "result_type": "ok",
        "identity": {"generation_fence": "verified"},
        "goal": {"objective": "ship it", "goal_version": version},
        "delivery_fence": {
            "occurrence_id": "occ-1",
            "terminal_generation": "gen-1",
            "native_session_id": "sess-1",
            "goal_version": version,
            "hold_high_water": version,
            "flock_path": flock_path,
        },
    }


def test_wrapper_rereads_fence_under_shared_hold(tmp_path, monkeypatch):
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    flock_path = str(lockdir / "goal-effect.lock")
    events = []
    answers = iter([_fenced_answer(1, flock_path), _fenced_answer(2, flock_path)])

    def fake_run(argv, timeout_seconds=None, **kwargs):
        events.append("conduct")
        return next(answers)

    from cli_agent_orchestrator.services import goal_effect_flock as flockmod

    real_hold = flockmod.hold_path

    @contextlib.contextmanager
    def tracking(path, **kwargs):
        events.append("enter")
        with real_hold(path, **kwargs):
            yield
        events.append("exit")

    posted = {}

    def fake_post(
        *, fork_base, terminal_id, fence, request_id, projection, hook_evidence, timeout_seconds
    ):
        events.append("post")
        posted["fence"] = fence
        posted["request_id"] = request_id
        posted["projection"] = projection
        return {"status": "posted"}

    monkeypatch.setattr(kr, "_run_conduct", fake_run)
    monkeypatch.setattr(flockmod, "hold_path", tracking)
    monkeypatch.setattr(kr, "_post_boundary", fake_post)
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={"session_id": "sess-1"},
            terminal_id="t",
            terminal_generation="gen-1",
            native_session_id="sess-1",
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    # The fresh re-read and the POST both happen inside the shared
    # hold; the POSTed fence carries the updated goal version.
    assert events == ["conduct", "enter", "conduct", "post", "exit"]
    assert posted["fence"]["goal_version"] == 2
    assert posted["fence"]["hold_high_water"] == 2


def test_wrapper_without_flock_pointer_defers_without_post(monkeypatch):
    answer = _fenced_answer(1, None)
    del answer["delivery_fence"]["flock_path"]
    monkeypatch.setattr(kr, "_run_conduct", lambda argv, timeout_seconds=None, **k: answer)
    monkeypatch.setattr(kr, "_post_boundary", _never_post)
    out, err = io.StringIO(), io.StringIO()
    assert (
        kr.run_wrapper(
            hook_input={},
            terminal_id="t",
            terminal_generation="g",
            native_session_id=None,
            conduct_binary="/usr/bin/conduct",
            fork_base="http://x",
            out=out,
            err=err,
        )
        == 0
    )
    assert "deferred" in err.getvalue()
    assert "no bytes sent" in err.getvalue()


def test_managed_wire_roots_prefers_managed(tmp_path, monkeypatch):
    import cli_agent_orchestrator.constants as constants

    managed = tmp_path / "t1" / "g1" / "kimi-home"
    managed.mkdir(parents=True)
    legacy = tmp_path / "kimi-homes" / "t1-g1"
    legacy.mkdir(parents=True)
    monkeypatch.setattr(constants, "COMPANION_DIR", str(tmp_path))
    roots = kr.managed_wire_roots(terminal_id="t1", generation="g1")
    assert roots[0] == str(managed)
    assert roots[1] == str(legacy)


class _RecorderTransport:
    def __init__(self):
        self.calls = []

    def send_literal(self, text):
        self.calls.append(f"literal:{len(text)}")

    def send_enter(self):
        self.calls.append("enter")

    def send_key(self, keystroke):
        self.calls.append(f"key:{keystroke}")


def _install_submit_world(monkeypatch, *, legs=None, transport=None):
    """Stub every submit collaborator except the adapter and the flock."""
    resolved = types.SimpleNamespace(
        provider="kimi_cli",
        terminal_generation="g1",
        native_session_id="sess-n",
        pane_id="%1",
        pane_dead=False,
        window_id="w1",
        pane_pid=111,
        session_name="sess-tmux",
        execution_mode=em.NATIVE_TUI,
        provider_version="0.29.2",
        bound_server_socket_path="/tmp/x.sock",
    )
    live = types.SimpleNamespace(dead=False, window_id="w1", pane_pid=111)
    box = {"transport": transport or _RecorderTransport()}

    client = types.SimpleNamespace(
        pane_control_identity=lambda pane_id, deadline_monotonic=None: live
    )
    svc = types.ModuleType("cli_agent_orchestrator.services.control_input_service")
    svc.resolve_control_identity = lambda terminal_id: resolved
    svc.provider_byte_admission = lambda *a, **k: contextlib.nullcontext()
    svc._tmux_client = lambda: client
    svc._NativeComposerTransport = lambda *a, **k: box["transport"]
    monkeypatch.setitem(sys.modules, "cli_agent_orchestrator.services.control_input_service", svc)

    journal = types.ModuleType("cli_agent_orchestrator.services.cohort_journal")
    journal.SessionEffectRefused = type("SessionEffectRefused", (Exception,), {})
    journal.session_effect_admission = lambda *a, **k: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "cli_agent_orchestrator.services.cohort_journal", journal)

    arbiter = types.ModuleType("cli_agent_orchestrator.services.pane_input_arbiter")
    arbiter.PaneBusyError = type("PaneBusyError", (Exception,), {})
    arbiter.pane_input_lease = lambda *a, **k: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "cli_agent_orchestrator.services.pane_input_arbiter", arbiter)

    uterm = types.ModuleType("cli_agent_orchestrator.utils.terminal")
    uterm.managed_window_name = lambda terminal_id, generation: "w"
    monkeypatch.setitem(sys.modules, "cli_agent_orchestrator.utils.terminal", uterm)

    cij = types.ModuleType("cli_agent_orchestrator.services.control_input_journal")
    cij.ControlInputBinding = lambda **k: types.SimpleNamespace(**k)
    monkeypatch.setitem(sys.modules, "cli_agent_orchestrator.services.control_input_journal", cij)

    # A bare sys.modules swap loses to the parent package attribute
    # once any collected module imports the real submodule: ``from
    # package import name`` prefers the bound attribute. Pin the
    # attributes too (monkeypatch reverts both layers per test).
    import cli_agent_orchestrator.services as _services_pkg
    import cli_agent_orchestrator.utils as _utils_pkg

    monkeypatch.setattr(_services_pkg, "control_input_service", svc, raising=False)
    monkeypatch.setattr(_services_pkg, "cohort_journal", journal, raising=False)
    monkeypatch.setattr(_services_pkg, "pane_input_arbiter", arbiter, raising=False)
    monkeypatch.setattr(_services_pkg, "control_input_journal", cij, raising=False)
    monkeypatch.setattr(_utils_pkg, "terminal", uterm, raising=False)

    legs = legs or {}
    monkeypatch.setattr(kr, "_fork_lifecycle_working", lambda **k: legs.get("lifecycle"))
    monkeypatch.setattr(kr, "_fork_occurrence_current", lambda **k: legs.get("occurrence"))
    monkeypatch.setattr(kr, "_fork_wait_cover", lambda **k: legs.get("wait"))
    monkeypatch.setattr(kr, "_observe_branch", lambda **k: ("idle", "test idle"))
    return resolved, box


def _attach_submit_world():
    kw = dict(
        provider="kimi_cli",
        native_session_id="sess-n",
        terminal_id="t-submit",
        generation="g1",
        execution_mode=em.NATIVE_TUI,
    )
    na.declare(
        **kw,
        intent=na.acquire_intent(
            acquisition_method=na.ACQUISITION_ACP_BOOTSTRAP,
            acquisition_receipt={"kind": "kimi-acp-session-new", "session_id": "sess-n"},
            admits_only_new_instructions=True,
            replays_task_bytes=False,
            bootstrap_sent_no_turn=True,
            bootstrap_detached_before_launch=True,
        ),
        pane_id="%1",
    )
    na.mark_starting(**kw, pane_id="%1")
    na.mark_attached(
        **kw, pane_id="%1", process_identity=na.process_identity(pid=4242, start_marker="1")
    )


def _submit_fence(flock_path, **over):
    fence = {
        "generation": "g1",
        "native_session_id": "sess-n",
        "goal_version": 3,
        "hold_high_water": 0,
        "flock_path": flock_path,
    }
    fence.update(over)
    return fence


def _projection(**over):
    """A canonical hook-context answer (verified fence, full fields)."""
    goal = {
        "goal_id": "goal-1",
        "goal_version": 3,
        "objective": "ship it",
        "completion_requirements": [
            {"id": "final-report", "kind": "report-occurrence"},
            {"id": "required-artifacts", "kind": "artifact-receipt"},
        ],
        "requirements_outstanding": ["final-report"],
        "evidence_count": 2,
        "latest_checkpoint": {
            "event_id": "ev-9",
            "kind": "checkpoint",
            "goal_version": 3,
            "recorded_at": "t",
            "summary": "halfway there",
        },
        "active_hold": None,
        "next_action": "draft the report",
    }
    goal.update(over.pop("goal", {}))
    answer = {
        "result_type": "ok",
        "identity": {"generation_fence": "verified"},
        "goal": goal,
        "next_action": over.pop("next_action", "draft the report"),
        "delivery_fence": {
            "occurrence_id": "occ-1",
            "terminal_generation": "g1",
            "native_session_id": "sess-n",
            "goal_version": 3,
            "hold_high_water": 0,
            "flock_path": over.pop("flock_path", "/x/goal-effect.lock"),
        },
    }
    answer.update(over)
    return answer


def _evidence(**over):
    """One wrapper run's native invocation evidence (audit only)."""
    ev = {
        "session_id": "sess-n",
        "trigger": "auto",
        "estimated_token_count": 120000,
        "observed_at": "2026-09-09T00:00:00Z",
    }
    ev.update(over)
    return ev


def _empty_box():
    """A proven-empty Kimi composer box (pinned 0.29.2 layout)."""
    return ["transcript above", "╭────╮", "│>     │", "╰────╯"]


def _full_box(*content_rows):
    """A proven-nonempty Kimi composer box (marker-free partial)."""
    return ["transcript above", "╭────╮", *["│" + row + "│" for row in content_rows], "╰────╯"]


def _wire_echo(home, marker):
    wire = home / "sessions" / "wd_1" / "session_s" / "agents" / "main" / "wire.jsonl"
    wire.write_text(
        json.dumps(
            {
                "type": "context.append_message",
                "agentId": "main",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"x [cao-context-restoration marker:{marker}]"}
                    ],
                    "origin": {"kind": "user"},
                },
                "time": 5,
            }
        )
        + "\n"
    )


def test_same_event_retry_adopts_once(tmp_path, monkeypatch):
    # Same operation id re-POSTed (transport retry, periodic
    # rendezvous): adopt + settle, zero new bytes, no new row.
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    home = tmp_path / "kh"
    (home / "sessions" / "wd_1" / "session_s" / "agents" / "main").mkdir(parents=True)
    _, box = _install_submit_world(monkeypatch)
    monkeypatch.setattr(kr, "managed_wire_roots", lambda **k: [str(home)])
    _stub_capture(monkeypatch, rows=_empty_box())
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    first = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_retry_A",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(),
    )
    assert first["status"] == "posted", first
    assert first["new_bytes"] is True
    typed = list(box["transport"].calls)
    assert typed
    second = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_retry_A",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(),
    )
    assert second["status"] == "pending", second
    assert second["new_bytes"] is False
    assert second["record"]["operation_id"] == "op_retry_A"
    assert box["transport"].calls == typed
    # The wire now proves model entry; the same retry completes it.
    _wire_echo(home, "op_retry_A")
    third = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_retry_A",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(),
    )
    assert third["status"] == "completed", third
    assert third["new_bytes"] is False
    assert third["record"]["operation_id"] == "op_retry_A"
    assert box["transport"].calls == typed


def test_two_invocations_identical_native_fields_deliver_twice(tmp_path, monkeypatch):
    # Parent gate: every native hook invocation is a new request. Two
    # invocations with identical ALL native fields and identical
    # rendered text restore twice — no native-field or content
    # equality across invocations ever coalesces. Only an exact
    # request-id re-POST is a retry.
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    home = tmp_path / "kh"
    home.mkdir()
    _, box = _install_submit_world(monkeypatch)
    monkeypatch.setattr(kr, "managed_wire_roots", lambda **k: [str(home)])
    _stub_capture(monkeypatch, rows=_empty_box())
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    same_evidence = _evidence()
    first = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_same_A",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=dict(same_evidence),
    )
    assert first["status"] == "posted", first
    assert first["new_bytes"] is True
    typed = list(box["transport"].calls)
    # The old row's marker reaches the wire — that proves A's bytes,
    # never B's invalidation. The distinct new id still delivers.
    (home / "sessions" / "wd_1" / "session_s" / "agents" / "main").mkdir(parents=True)
    _wire_echo(home, "op_same_A")
    second = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_same_B",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=dict(same_evidence),
    )
    assert second["status"] == "posted", second
    assert second["new_bytes"] is True
    assert second["record"]["operation_id"] == "op_same_B"
    assert len(box["transport"].calls) > len(typed)
    # A settled honestly from its own wire echo; B delivered anew.
    assert knc.get("op_same_A")["state"] == "completed"


def test_distinct_same_text_events_deliver_twice(tmp_path, monkeypatch):
    # Two invocations, byte-identical restoration text — the second
    # delivers anew with new bytes (the old row settles honestly
    # first). Native fields never coalesce.
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    home = tmp_path / "kh"
    home.mkdir()
    _, box = _install_submit_world(monkeypatch)
    monkeypatch.setattr(kr, "managed_wire_roots", lambda **k: [str(home)])
    _stub_capture(monkeypatch, rows=_empty_box())
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    first = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_2x_A",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=100),
    )
    assert first["status"] == "posted", first
    assert first["new_bytes"] is True
    typed = list(box["transport"].calls)
    assert typed
    second = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_2x_B",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=250),
    )
    assert second["status"] == "posted", second
    assert second["new_bytes"] is True
    assert second["record"]["operation_id"] == "op_2x_B"
    assert len(box["transport"].calls) > len(typed)
    # The first row settled honestly (wire silent, marker absent).
    assert knc.get("op_2x_A")["state"] == "ambiguous"


def test_submit_journals_wait_cover_refusal(tmp_path, monkeypatch):
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    _, box = _install_submit_world(monkeypatch, legs={"wait": ("wait_cover", "covering wait")})
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    result = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_sub_W",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(),
    )
    assert result["status"] == "refused", result
    assert result["record"]["state"] == "refused"
    assert result["record"]["refusal_reason"] == "wait_cover_active"
    assert box["transport"].calls == []


def _stub_capture(monkeypatch, rows=None, exc=None):
    import cli_agent_orchestrator.services.native_pane_input as pane

    def fake(pane_id, timeout=None):
        if exc is not None:
            raise exc
        return list(rows or [])

    monkeypatch.setattr(pane, "capture_pane_screen", fake)


def test_two_compactions_same_goal_delivers_new_context(tmp_path, monkeypatch):
    # Compaction 1 delivers v1; compaction 2 carries new context while
    # v1 is live but unproven. The boundary must settle v1 honestly
    # (ambiguous: marker absent, wire silent) and deliver v2 as a new
    # row — never swallow v2 by adopting.
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    home = tmp_path / "kh"
    home.mkdir()
    _, box = _install_submit_world(monkeypatch)
    monkeypatch.setattr(kr, "managed_wire_roots", lambda **k: [str(home)])
    _stub_capture(monkeypatch, rows=_empty_box())
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    first = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_2c_A",
        occurrence_id="occ-1",
        projection=_projection(goal={"objective": "ship it v1"}),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=11),
    )
    assert first["status"] == "posted", first
    assert first["new_bytes"] is True
    typed = list(box["transport"].calls)
    assert typed
    second = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_2c_B",
        occurrence_id="occ-1",
        projection=_projection(goal={"objective": "ship it v2!!"}),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=22),
    )
    assert second["status"] == "posted", second
    assert second["new_bytes"] is True
    assert second["record"]["operation_id"] == "op_2c_B"
    assert len(box["transport"].calls) > len(typed)
    assert knc.get("op_2c_A")["state"] == "ambiguous"


def test_partial_composer_defers_new_compaction(tmp_path, monkeypatch):
    # v1 bytes sit unsubmitted in the composer (partial delivery);
    # a new compaction must defer, never compound, and the old row
    # goes honestly ambiguous. A later operator op still passes the
    # gate: partial owned bytes never freeze unrelated messaging.
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    home = tmp_path / "kh"
    home.mkdir()
    _, box = _install_submit_world(monkeypatch)
    monkeypatch.setattr(kr, "managed_wire_roots", lambda **k: [str(home)])
    _stub_capture(monkeypatch, rows=_empty_box())
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    first = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_pc_A",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=31),
    )
    assert first["status"] == "posted", first
    typed = list(box["transport"].calls)
    _stub_capture(
        monkeypatch,
        rows=["k> current goal: v1 partial", "[cao-context-restoration marker:op_pc_A]"],
    )
    second = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_pc_B",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=32),
    )
    assert second["status"] == "deferred", second
    assert second["new_bytes"] is False
    assert box["transport"].calls == typed
    assert knc.get("op_pc_A")["state"] == "ambiguous"
    assert knc.get("op_pc_B") is None
    knc._assert_session_unblocked(native_session_id="sess-n", operation_id="op_operator_1")


def test_marker_free_partial_blocks_until_proven_empty(tmp_path, monkeypatch):
    # P1-3 decisive: a marker-free partial (our marker never landed —
    # the failure hit earlier text — or an operator draft) is NOT
    # composer-clear. The new event refuses this session with zero
    # bytes; after the operator clears and the composer proves empty,
    # the next event delivers. The draft is never erased by us.
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    home = tmp_path / "kh"
    home.mkdir()
    _, box = _install_submit_world(monkeypatch)
    monkeypatch.setattr(kr, "managed_wire_roots", lambda **k: [str(home)])
    _stub_capture(monkeypatch, rows=_empty_box())
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    first = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_pf_A",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=41),
    )
    assert first["status"] == "posted", first
    typed = list(box["transport"].calls)
    # Marker-free prefill appears; the old row's marker is absent and
    # the wire is silent.
    _stub_capture(
        monkeypatch, rows=_full_box("> truncated reminder without marker", "operator draft line")
    )
    second = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_pf_B",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=42),
    )
    assert second["status"] == "refused", second
    assert second["new_bytes"] is False
    assert second["record"]["refusal_reason"] == knc.REFUSED_COMPOSER_NONEMPTY
    assert box["transport"].calls == typed
    assert knc.get("op_pf_A")["state"] == "ambiguous"
    # The operator submits/clears the draft; the composer proves
    # empty; the next event delivers with new bytes.
    _stub_capture(monkeypatch, rows=_empty_box())
    third = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_pf_C",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=43),
    )
    assert third["status"] == "posted", third
    assert third["new_bytes"] is True
    assert third["record"]["operation_id"] == "op_pf_C"
    assert len(box["transport"].calls) > len(typed)


def test_unreadable_composer_refuses_only_that_session(tmp_path, monkeypatch):
    # Capture failure refuses the blind session with a typed row; a
    # session whose composer proves empty still delivers.
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    home = tmp_path / "kh"
    home.mkdir()
    _, box = _install_submit_world(monkeypatch)
    monkeypatch.setattr(kr, "managed_wire_roots", lambda **k: [str(home)])
    _stub_capture(monkeypatch, exc=OSError("tmux down"))
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    bad = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_un_r",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(),
    )
    assert bad["status"] == "refused", bad
    assert bad["new_bytes"] is False
    assert bad["record"]["refusal_reason"] == knc.REFUSED_COMPOSER_NONEMPTY
    assert box["transport"].calls == []
    _stub_capture(monkeypatch, rows=_empty_box())
    good = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_un_g",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(estimated_token_count=55),
    )
    assert good["status"] == "posted", good
    assert good["new_bytes"] is True


def test_unpinned_build_refuses_with_pin_recovery(tmp_path, monkeypatch):
    # No composer-emptiness pin for this build: refuse the session
    # naming the missing pin — never guess at the region.
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    home = tmp_path / "kh"
    home.mkdir()
    _, box = _install_submit_world(monkeypatch)
    monkeypatch.setattr(kr, "_fork_lifecycle_working", lambda **k: None)
    monkeypatch.setattr(kr, "_fork_occurrence_current", lambda **k: None)
    monkeypatch.setattr(kr, "_fork_wait_cover", lambda **k: None)
    monkeypatch.setattr(kr, "_observe_branch", lambda **k: ("idle", "test idle"))
    import cli_agent_orchestrator.services.control_input_service as svc_mod

    resolved = svc_mod.resolve_control_identity("t-submit")
    resolved.provider_version = "9.9.9-unproven"
    monkeypatch.setattr(kr, "managed_wire_roots", lambda **k: [str(home)])
    _stub_capture(monkeypatch, rows=_empty_box())
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    result = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_up_X",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(),
    )
    assert result["status"] == "refused", result
    assert result["new_bytes"] is False
    assert result["record"]["refusal_reason"] == knc.REFUSED_COMPOSER_UNPINNED
    assert "9.9.9-unproven" in result["detail"]
    assert box["transport"].calls == []


def test_submit_refuses_missing_fence_native(tmp_path, monkeypatch):
    _attach_submit_world()
    lockdir = tmp_path / "proj"
    lockdir.mkdir()
    _install_submit_world(monkeypatch)
    fence = _submit_fence(str(lockdir / "goal-effect.lock"))
    del fence["native_session_id"]
    result = kr.submit_context_reminder(
        terminal_id="t-submit",
        operation_id="op_sub_N",
        occurrence_id="occ-1",
        projection=_projection(),
        fence=fence,
        hook_evidence=_evidence(),
    )
    assert result["status"] == "refused", result
    assert "native_session_id" in result["detail"]
    assert knc.get("op_sub_N") is None
