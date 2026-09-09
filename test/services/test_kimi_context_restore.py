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

import io
import json

from cli_agent_orchestrator.services import kimi_context_restore as kr


def test_hook_toml_is_postcompact_only():
    toml = kr.restore_hook_toml(command="/bin/cao-kimi-hook-context --terminal t")
    assert 'event = "PostCompact"' in toml
    assert "SessionStart" not in toml
    assert "UserPromptSubmit" not in toml
    assert "timeout =" in toml


def test_compose_preserves_user_hooks_byte_identical():
    base = ('[general]\ntheme = "dark"\n\n[[hooks]]\nevent = "PreToolUse"\n'
            'command = "/home/u/my-hook"\n')
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
        conduct_binary="/usr/bin/conduct", native_session_id="sess-1",
        terminal_id="term-1", terminal_generation="gen-3")
    assert argv[:3] == ["/usr/bin/conduct", "goal", "hook-context"]
    assert "--harness" in argv and "kimi_cli" in argv
    joined = " ".join(argv)
    for forbidden in ("resume", "steer", "send", "release", "approve",
                      "checkpoint", "claim", "satisfy"):
        assert forbidden not in joined


def test_parse_hook_input_accepts_kimi_snake_case():
    raw = json.dumps({
        "session_id": "sess-9", "cwd": "/tmp/w",
        "hook_event_name": "PostCompact", "trigger": "auto",
        "estimated_token_count": 120000}).encode()
    parsed = kr.parse_hook_input(raw)
    assert parsed["session_id"] == "sess-9"
    assert parsed["hook_event_name"] == "PostCompact"
    assert kr.parse_hook_input(b"not json") == {}


def test_render_restoration_requires_verified_ok():
    assert kr.render_restoration({"result_type": "stale"}) is None
    assert kr.render_restoration({
        "result_type": "ok",
        "identity": {"generation_fence": "unverified"},
        "goal": {"objective": "x"}}) is None
    text = kr.render_restoration({
        "result_type": "ok",
        "identity": {"generation_fence": "verified"},
        "goal": {"objective": "ship it", "goal_version": 4,
                 "requirements": [{"id": "r1", "summary": "draft"}]},
        "next_action": "draft the report"})
    assert text is not None
    assert "cao-context-restoration" in text
    assert "ship it" in text
    assert "not a new assignment" in text
    assert len(text) <= kr.MAX_CONTEXT_CHARS + len("\n[truncated]")


def _ok_answer():
    return {
        "result_type": "ok",
        "identity": {"generation_fence": "verified"},
        "goal": {"objective": "ship it", "goal_version": 2},
        "delivery_fence": {
            "occurrence_id": "occ-1", "terminal_generation": "gen-1",
            "flock_path": "/tmp/x/goal-effect.lock"},
    }


def test_wrapper_delivers_nothing_without_verified_fence(monkeypatch):
    calls = {}

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        return {"result_type": "stale"}
    monkeypatch.setattr(kr, "_run_conduct", fake_run)
    monkeypatch.setattr(kr, "_post_boundary",
                        lambda **k: (_ for _ in ()).throw(AssertionError("must not post")))
    out, err = io.StringIO(), io.StringIO()
    assert kr.run_wrapper(
        hook_input={"session_id": "s"}, terminal_id="t",
        terminal_generation="g", native_session_id="s",
        conduct_binary="/usr/bin/conduct", fork_base="http://x",
        out=out, err=err) == 0


def test_wrapper_posts_fence_and_context_on_ok(monkeypatch):
    posted = {}

    def fake_run(argv, **kwargs):
        assert argv[1:3] == ["goal", "hook-context"]
        return _ok_answer()
    def fake_post(fork_base, terminal_id, fence, context, timeout_seconds):
        posted.update(fence=fence, context=context, terminal=terminal_id)
        return {"status": "posted"}
    monkeypatch.setattr(kr, "_run_conduct", fake_run)
    monkeypatch.setattr(kr, "_post_boundary", fake_post)
    out, err = io.StringIO(), io.StringIO()
    assert kr.run_wrapper(
        hook_input={"session_id": "s"}, terminal_id="term-9",
        terminal_generation="g", native_session_id="s",
        conduct_binary="/usr/bin/conduct", fork_base="http://base",
        out=out, err=err) == 0
    assert posted["terminal"] == "term-9"
    assert posted["fence"]["occurrence_id"] == "occ-1"
    assert "cao-context-restoration" in posted["context"]


def test_wrapper_refuses_stdin_identity_mismatch(monkeypatch):
    monkeypatch.setattr(kr, "_run_conduct",
                        lambda **k: (_ for _ in ()).throw(AssertionError("must not run")))
    out, err = io.StringIO(), io.StringIO()
    assert kr.run_wrapper(
        hook_input={"session_id": "other"}, terminal_id="t",
        terminal_generation="g", native_session_id="baked",
        conduct_binary="/usr/bin/conduct", fork_base="http://x",
        out=out, err=err) == 0
    assert "does not match" in err.getvalue()


def test_wrapper_degrades_when_boundary_unreachable(monkeypatch):
    monkeypatch.setattr(kr, "_run_conduct", lambda argv, **k: _ok_answer())
    monkeypatch.setattr(kr, "_post_boundary",
                        lambda **k: {"transport_error": "down"})
    out, err = io.StringIO(), io.StringIO()
    assert kr.run_wrapper(
        hook_input={}, terminal_id="t", terminal_generation="g",
        native_session_id=None, conduct_binary="/usr/bin/conduct",
        fork_base="http://x", out=out, err=err) == 0
    assert "unreachable" in err.getvalue()


def test_restore_command_bakes_all_binding():
    cmd = kr.restore_command(
        wrapper_executable="/bin/w", terminal_id="t", generation="g",
        conduct_binary="/bin/conduct", fork_base="http://b:1",
        native_session_id="sess-x")
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
        companion_dir=str(tmp_path))
    assert installation["mechanism"] is None
    assert "degraded_reason" in installation and installation["degraded_reason"]
    assert "KIMI_CODE_HOME" not in env
    _ = calls


def test_prepare_composes_private_home(tmp_path, monkeypatch):
    monkeypatch.setattr(kr, "resolve_wrapper_executable", lambda **k: "/bin/w")
    monkeypatch.setattr(kr, "resolve_conduct_binary", lambda **k: "/bin/conduct")
    env, installation = kr.prepare_kimi_restoration(
        record={"terminal_id": "t9", "generation": "g2", "native_session_id": "s"},
        base_environment={"KIMI_CODE_HOME": str(tmp_path / "opaque"),
                          "PATH": "/bin"},
        companion_dir=str(tmp_path / "companion"))
    assert installation["mechanism"] == kr.MECHANISM
    home = installation["kimi_home"]
    assert home is not None and "t9-g2" in home
    assert env["KIMI_CODE_HOME"] == home
    text = open(home + "/config.toml").read()
    assert 'event = "PostCompact"' in text
    assert "--terminal t9" in text
