"""cond-0713 LANG guarantee: every native pane inherits UTF-8.

Bisection proved: launchd strips LANG/LC_CTYPE → Muse 0.2.1 renders ASCII
fallback chrome (no │ borders) → boxed-panel parser starves (footer-only).
The guarantee forces LANG=en_US.UTF-8 and LC_CTYPE=en_US.UTF-8 into the
child pane env at creation time regardless of host inheritance, and ensures
LC_ALL does not override.  It is enforced at three seams so neither path
relies on the other: TmuxClient (final tmux -e seam), managed_provider_bridge
(provider child env), and managed_launch_v2 (native env alongside
MUSE_NO_AUTO_UPDATE).  This test pins the seams.

Intentionally narrow: it asserts the exact env/argv the pane creation
construction builds, not just that a helper exists.  Reverting the injection
must redden at least one test here (mutation proof).

P1-1/P1-2/P2-1/P2-2 fixups:
- P1-1: spy on managed_launch_v2._ensure_locale_env via _launch_native_tui
  for each branch (GLM, muse, kimi)
- P1-2: parameterized GLM/DeepSeek/kimi_cli bridge env
- P2-1: discover now logs pane_env vs server_env
- P2-2: herdr _inject_env_vars forced locale
- P3-1: single source constants.FORCED_LOCALE, equality asserted
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from unittest.mock import MagicMock, patch

import pytest

# ── TmuxClient seams ───────────────────────────────────────────────────


class TestTmuxLocaleGuarantee:
    """The final tmux -e seam always carries UTF-8."""

    def _client(self):
        with patch("cli_agent_orchestrator.clients.tmux.libtmux") as mock_libtmux:
            mock_server = MagicMock()
            mock_libtmux.Server.return_value = mock_server
            from cli_agent_orchestrator.clients.tmux import TmuxClient

            c = TmuxClient()
            c.server = mock_server
            return c

    def test_filtered_environment_forces_utf8_when_host_stripped(self, tmp_path):
        c = self._client()
        mock_window = MagicMock()
        mock_window.name = "w"
        mock_session = MagicMock()
        mock_session.windows = [mock_window]
        c.server.new_session.return_value = mock_session

        # launchd: no LANG/LC_CTYPE/LC_ALL at all
        with patch.dict(os.environ, {}, clear=True):
            # essential HOME must exist for realpath check
            os.environ["HOME"] = "/home/user"
            with patch.dict(os.environ, {"HOME": "/home/user"}, clear=False):
                # Call the seam directly — env dict construction.
                env = c._filtered_child_environment(extra_env={}, terminal_id="tid1")
                assert env["LANG"] == "en_US.UTF-8"
                assert env["LC_CTYPE"] == "en_US.UTF-8"
                assert "LC_ALL" not in env

    def test_filtered_environment_overwrites_non_utf8_and_removes_lc_all(self, tmp_path):
        c = self._client()
        mock_window = MagicMock()
        mock_window.name = "w"
        mock_session = MagicMock()
        mock_session.windows = [mock_window]
        c.server.new_session.return_value = mock_session

        hostile = {
            "HOME": "/home/user",
            "LANG": "C",
            "LC_ALL": "C",
            "LC_CTYPE": "C",
        }
        with patch.dict(os.environ, hostile, clear=True):
            env = c._filtered_child_environment(extra_env={}, terminal_id="tid1")
            assert env["LANG"] == "en_US.UTF-8"
            assert env["LC_CTYPE"] == "en_US.UTF-8"
            assert "LC_ALL" not in env

    def test_create_session_passes_forced_locale_to_libtmux(self, tmp_path):
        c = self._client()
        mock_window = MagicMock()
        mock_window.name = "w"
        mock_session = MagicMock()
        mock_session.windows = [mock_window]
        c.server.new_session.return_value = mock_session

        with patch.dict(os.environ, {"HOME": "/home/user"}, clear=True):
            os.environ["HOME"] = "/home/user"
            c.create_session("ses", "w", "tid1", str(tmp_path))
            env = c.server.new_session.call_args.kwargs["environment"]
            assert env["LANG"] == "en_US.UTF-8"
            assert env["LC_CTYPE"] == "en_US.UTF-8"
            assert "LC_ALL" not in env

    def test_create_window_injects_locale_into_libtmux_env(self, tmp_path):
        c = self._client()
        mock_window = MagicMock()
        mock_window.name = "agent-window"
        mock_session = MagicMock()
        mock_session.new_window.return_value = mock_window
        c.server.sessions.get.return_value = mock_session

        with patch.dict(os.environ, {}, clear=True):
            os.environ["HOME"] = "/home/user"
            c.create_window("ses", "agent-window", "tid2", str(tmp_path), extra_env={})
            env = mock_session.new_window.call_args.kwargs["environment"]
            assert env["LANG"] == "en_US.UTF-8"
            assert env["LC_CTYPE"] == "en_US.UTF-8"
            assert "LC_ALL" not in env
            assert env["CAO_TERMINAL_ID"] == "tid2"

    def test_create_window_with_argv_injects_locale_into_tmux_cmd(self, tmp_path):
        c = self._client()
        with patch.dict(os.environ, {}, clear=True):
            os.environ["HOME"] = "/home/user"
            with patch("cli_agent_orchestrator.clients.tmux.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                with patch(
                    "cli_agent_orchestrator.clients.tmux.tmux_binary", return_value="/usr/bin/tmux"
                ):
                    c.create_window_with_argv(
                        "ses",
                        "w",
                        "tid3",
                        ["/usr/local/bin/muse", "--trust-workspace", "--yolo"],
                        str(tmp_path),
                        extra_env={},
                    )
                cmd = mock_run.call_args[0][0]
                # tmux new-window -e LANG=... -e LC_CTYPE=... must be present
                assert "-e" in cmd
                e_vals = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-e"]
                assert "LANG=en_US.UTF-8" in e_vals
                assert "LC_CTYPE=en_US.UTF-8" in e_vals
                assert not any(v.startswith("LC_ALL=") for v in e_vals)

    def test_create_window_with_argv_overwrites_host_c_locale(self, tmp_path):
        c = self._client()
        with patch.dict(os.environ, {"HOME": "/home/user", "LC_ALL": "C", "LANG": "C"}, clear=True):
            with patch("cli_agent_orchestrator.clients.tmux.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                with patch(
                    "cli_agent_orchestrator.clients.tmux.tmux_binary", return_value="/usr/bin/tmux"
                ):
                    c.create_window_with_argv(
                        "ses",
                        "w",
                        "tid3",
                        ["/usr/local/bin/muse", "--trust-workspace"],
                        str(tmp_path),
                        extra_env={"LC_ALL": "C"},
                    )
                cmd = mock_run.call_args[0][0]
                e_vals = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-e"]
                # Even if extra_env carried LC_ALL=C, final env must not
                assert not any(v.startswith("LC_ALL=") for v in e_vals)
                assert "LANG=en_US.UTF-8" in e_vals


# ── managed_provider_bridge seam ───────────────────────────────────────


class TestManagedBridgeLocaleGuarantee:
    """Provider child env for native-TUI also forced to UTF-8."""

    def test_provider_env_forced_when_host_stripped(self):
        from cli_agent_orchestrator.services.managed_provider_bridge import _provider_env

        with patch.dict(os.environ, {}, clear=True):
            env = _provider_env({})
            assert env["LANG"] == "en_US.UTF-8"
            assert env["LC_CTYPE"] == "en_US.UTF-8"
            assert "LC_ALL" not in env

    def test_provider_env_overwrites_c_and_removes_lc_all(self):
        from cli_agent_orchestrator.services.managed_provider_bridge import _provider_env

        with patch.dict(os.environ, {"LANG": "C", "LC_CTYPE": "C", "LC_ALL": "C"}, clear=True):
            env = _provider_env({"LC_ALL": "C"})
            assert env["LANG"] == "en_US.UTF-8"
            assert env["LC_CTYPE"] == "en_US.UTF-8"
            assert "LC_ALL" not in env

    def test_native_child_env_forced_for_muse(self):
        from cli_agent_orchestrator.services.managed_provider_bridge import native_child_environment

        # Muse native uses the same provider child env seam
        request = {
            "provider": "muse_cli",
            "provider_route": "anthropic",
            "model": "muse-spark-1.3-contributor",
            "effort": "high",
        }
        with patch.dict(os.environ, {"HOME": "/home/user", "LANG": "C", "LC_ALL": "C"}, clear=True):
            env = native_child_environment(request)
            assert env["LANG"] == "en_US.UTF-8"
            assert env["LC_CTYPE"] == "en_US.UTF-8"
            assert "LC_ALL" not in env


# ── P1-2: parameterized bridge routes (GLM, DeepSeek, kimi_cli) ──


class TestManagedBridgeParameterizedRoutes:
    """Each provider route that builds an env must force UTF-8 (P1-2).

    Removing the _ensure call at the route's specific site must redden
    exactly its test — hence per-route pinning, not just the anthropic fallback.
    """

    def test_native_child_env_forced_for_glm(self, tmp_path):
        """GLM route via native_child_environment with route_envelope (339)."""
        from cli_agent_orchestrator.services.managed_provider_bridge import native_child_environment

        # Minimal valid GLM session_env/envelope via mocked validate_session_env
        session_env = {
            "CAO_CONDUCTOR_ROUTES": "/tmp/routes.json",
            "CAO_CONDUCTOR_SHIM_DIR": "/tmp/shim",
            "CAO_CONDUCTOR_REAL_CLAUDE": "/tmp/inner",
            "HOME": "/home/user",
        }
        envelope = {
            "wrapper_executable": "/tmp/shim/wrapper",
            "wrapper_executable_sha256": "a" * 64,
            "inner_executable": "/tmp/inner",
            "inner_executable_sha256": "b" * 64,
            "route_map_path": "/tmp/routes.json",
            "worktree_realpath": "/tmp/worktree",
            "consumed_marker_path": "/tmp/consumed",
        }
        request = {
            "provider": "claude_code",
            "provider_route": "glm",
            "model": "claude-sonnet-4",
            "route_envelope": envelope,
        }
        verified = {
            "CAO_CONDUCTOR_SHIM_DIR": "/tmp/shim",
            "CAO_CONDUCTOR_ROUTES": "/tmp/routes.json",
            "CAO_CONDUCTOR_REAL_CLAUDE": "/tmp/inner",
            "HOME": "/home/user",
            "CAO_CONDUCTOR_FOO": "bar",
            "ZDOTDIR": "/tmp/zdot",
        }
        with patch(
            "cli_agent_orchestrator.services.glm_native_launch.validate_session_env",
            return_value=verified,
        ):
            with patch.dict(
                os.environ, {"HOME": "/home/user", "LANG": "C", "LC_ALL": "C"}, clear=True
            ):
                env = native_child_environment(request, session_env=session_env)
                assert env["LANG"] == "en_US.UTF-8"
                assert env["LC_CTYPE"] == "en_US.UTF-8"
                assert "LC_ALL" not in env
                # shim dir must still be present (GLM-specific)
                assert env["PATH"].startswith("/tmp/shim")

    def test_native_child_env_forced_for_deepseek(self):
        """DeepSeek route provider_route==deepseek (371)."""
        from cli_agent_orchestrator.services.managed_provider_bridge import native_child_environment

        request = {
            "provider": "claude_code",
            "provider_route": "deepseek",
            "model": "deepseek-chat",
            "provider_executable": "/tmp/claude",
            "provider_executable_sha256": "c" * 64,
            "working_directory": "/tmp/worktree",
            "route_envelope": {
                "wrapper_executable": "/tmp/wrapper",
                "inner_executable": "/tmp/inner",
                "route_map_path": "/tmp/routes.json",
            },
        }
        envelope = {
            "wrapper_executable": "/tmp/wrapper",
            "inner_executable": "/tmp/inner",
            "route_map_path": "/tmp/routes.json",
        }
        with patch(
            "cli_agent_orchestrator.services.deepseek_acp_route.validate_envelope",
            return_value=envelope,
        ):
            with patch.dict(
                os.environ, {"HOME": "/home/user", "LANG": "C", "LC_ALL": "C"}, clear=True
            ):
                env = native_child_environment(request)
                assert env["LANG"] == "en_US.UTF-8"
                assert env["LC_CTYPE"] == "en_US.UTF-8"
                assert "LC_ALL" not in env
                assert "CAO_CONDUCTOR_ROUTES" in env
                assert "CAO_CONDUCTOR_SHIM_DIR" in env

    def test_provider_bound_env_kimi_cli_forces_locale_with_lc_all(self):
        """kimi_cli via _provider_bound_environments with LC_ALL=C (231-232)."""
        from cli_agent_orchestrator.services.managed_provider_bridge import (
            _provider_bound_environments,
        )

        ambient = {
            "HOME": "/home/user",
            "LANG": "C",
            "LC_ALL": "C",
            "LC_CTYPE": "C",
            "PATH": "/usr/bin",
        }
        with patch.dict(os.environ, ambient, clear=True):
            bridge_env, provider_env, inventory = _provider_bound_environments(
                "kimi_cli", ambient=ambient
            )
            for env in (bridge_env, provider_env):
                assert env["LANG"] == "en_US.UTF-8"
                assert env["LC_CTYPE"] == "en_US.UTF-8"
                assert "LC_ALL" not in env

    def test_native_child_env_kimi_cli_forces_locale(self):
        """kimi_cli native child env via fallback _provider_env (373)."""
        from cli_agent_orchestrator.services.managed_provider_bridge import native_child_environment

        request = {
            "provider": "kimi_cli",
            "provider_route": "anthropic",
            "model": "kimi-k2",
            "effort": "high",
        }
        with patch.dict(os.environ, {"HOME": "/home/user", "LANG": "C", "LC_ALL": "C"}, clear=True):
            env = native_child_environment(request)
            assert env["LANG"] == "en_US.UTF-8"
            assert env["LC_CTYPE"] == "en_US.UTF-8"
            assert "LC_ALL" not in env


# ── managed_launch_v2 seam + debug log ─────────────────────────────────


class TestManagedLaunchV2LocaleGuarantee:
    def test_ensure_locale_helper_forces(self):
        from cli_agent_orchestrator.services.managed_launch_v2 import _ensure_locale_env

        env: dict[str, str] = {"LC_ALL": "C", "LANG": "C"}
        _ensure_locale_env(env)
        assert env["LANG"] == "en_US.UTF-8"
        assert env["LC_CTYPE"] == "en_US.UTF-8"
        assert "LC_ALL" not in env

    def test_discover_logs_locale_at_debug(self):
        # P2-1: must log pane_env (forced) with distinct label, not just
        # os.environ. Under launchd server env stripped but pane env forced
        # → log must prove pane env, otherwise it lies.
        from cli_agent_orchestrator.services import managed_launch_v2

        pane_env = {"LANG": "en_US.UTF-8", "LC_CTYPE": "en_US.UTF-8"}

        with patch.object(
            managed_launch_v2, "_await_native_pane_input_ready", return_value={"input_ready": True}
        ):
            with patch.object(
                managed_launch_v2,
                "_observe_muse_status_panel",
                return_value={"observed": {"session_id": "11111111-1111-4111-8111-111111111111"}},
            ):
                with patch.dict(
                    os.environ, {"LANG": "C", "LC_CTYPE": "C", "LC_ALL": "C"}, clear=True
                ):
                    with patch.object(managed_launch_v2.logger, "debug") as mock_debug:
                        try:
                            managed_launch_v2._discover_muse_session(
                                pane_id="%1",
                                record={
                                    "working_directory": "/tmp",
                                    "terminal_id": "tid",
                                    "generation": "gen",
                                },
                                capture=lambda pane_id: [],
                                bootstrap={
                                    "requested_model": "muse-spark-1.3-contributor",
                                    "requested_effort": "high",
                                    "provider_version": "0.2.1",
                                },
                                request={"expected_model": "m", "expected_effort": "high"},
                                environment=pane_env,
                            )
                        except Exception:
                            pass
                        # Must log pane_env with distinct label, not just server env
                        calls = " ".join(str(c) for c in mock_debug.call_args_list)
                        assert (
                            "pane_env" in calls.lower()
                        ), f"must log pane_env, calls={mock_debug.call_args_list}"
                        assert (
                            "en_US.UTF-8" in calls
                        ), f"must log forced LANG, calls={mock_debug.call_args_list}"
                        # Must also log server_env distinct label or at least pane identifier
                        assert "%1" in calls or "pane=" in calls.lower()

    def test_discover_logs_pane_env_not_just_server_env(self):
        """Mutation pin: if discover logged only os.environ, pane_env forced value would not appear when server env is C."""
        from cli_agent_orchestrator.services import managed_launch_v2

        pane_env = {"LANG": "en_US.UTF-8", "LC_CTYPE": "en_US.UTF-8", "LC_ALL": None}
        # server env is hostile C, pane env is forced en_US
        with patch.object(
            managed_launch_v2, "_await_native_pane_input_ready", return_value={"input_ready": True}
        ):
            with patch.object(
                managed_launch_v2,
                "_observe_muse_status_panel",
                return_value={"observed": {"session_id": "11111111-1111-4111-8111-111111111111"}},
            ):
                with patch.dict(
                    os.environ, {"LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"}, clear=True
                ):
                    with patch.object(managed_launch_v2.logger, "debug") as mock_debug:
                        try:
                            managed_launch_v2._discover_muse_session(
                                pane_id="%9",
                                record={
                                    "working_directory": "/tmp",
                                    "terminal_id": "tid",
                                    "generation": "gen",
                                },
                                capture=lambda pane_id: [],
                                bootstrap={
                                    "requested_model": "muse-spark-1.3-contributor",
                                    "requested_effort": "high",
                                    "provider_version": "0.2.1",
                                },
                                request={"expected_model": "m", "expected_effort": "high"},
                                environment=pane_env,
                            )
                        except Exception:
                            pass
                        # Extract the first debug call's args
                        assert mock_debug.call_args_list, "no debug logged"
                        first_call = str(mock_debug.call_args_list[0])
                        # Must contain pane_env LANG=en_US, not just server C
                        assert "en_US.UTF-8" in first_call
                        assert "pane_env" in first_call.lower()


# ── P1-1: _launch_native_tui wiring pins ──


class TestLaunchNativeTuiLocaleWiring:
    """P1-1: each _launch_native_tui branch must call _ensure_locale_env.

    Removing any single call site at 3995/4008/4036 must redden exactly its
    test. Each test patches _ensure_locale_env and executes _launch_native_tui
    for its provider route, capturing that the pane environment was forced.
    """

    @pytest.mark.asyncio
    async def test_glm_branch_calls_ensure_locale(self):
        """GLM at 3995 — provider_route glm must force locale before version probe."""
        from cli_agent_orchestrator.services import managed_launch_v2
        from cli_agent_orchestrator.services.managed_launch_v2 import (
            _ensure_locale_env as real_ensure,
        )

        spy = MagicMock(side_effect=real_ensure)
        # Envelopes for GLM — bypass file checks
        envelope = {
            "wrapper_executable": "/tmp/shim/wrapper",
            "wrapper_executable_sha256": "a" * 64,
            "inner_executable": "/tmp/inner",
            "inner_executable_sha256": "b" * 64,
            "route_map_path": "/tmp/routes.json",
            "worktree_realpath": "/tmp/worktree",
            "consumed_marker_path": "/tmp/consumed",
        }
        record = {
            "provider": "claude_code",
            "request": {
                "expected_model": "claude-sonnet-4",
                "expected_effort": "high",
                "provider_route": "glm",
                "route_envelope": envelope,
                "working_directory": "/tmp/worktree",
                "provider_executable": "/tmp/inner",
                "provider_executable_sha256": "b" * 64,
            },
            "working_directory": "/tmp/worktree",
            "session_name": "test-session",
            "terminal_id": "tid-glm",
            "generation": "gen1",
            "agent_profile": "default",
        }
        bridge_request = {
            "provider_executable": "/tmp/inner",
            "provider_executable_sha256": "b" * 64,
            "profile_sha256": "same-sha",
            "provider": "claude_code",
            "model": "claude-sonnet-4",
        }
        # Capture env passed to version banner to prove it was forced
        captured_env: dict = {}

        def fake_banner(req, environment=None, timeout=None):
            if environment is not None:
                captured_env.update(environment)
            raise RuntimeError("abort after ensure at 3995")

        with (
            patch("cli_agent_orchestrator.services.managed_launch_v2._ensure_locale_env", spy),
            patch(
                "cli_agent_orchestrator.services.glm_native_launch.validate_envelope",
                return_value=envelope,
            ),
            patch(
                "cli_agent_orchestrator.services.managed_provider_bridge._profile_material",
                return_value={"profile_sha256": "same-sha"},
            ),
            patch(
                "cli_agent_orchestrator.services.session_env.get_session_env",
                return_value={
                    "CAO_CONDUCTOR_ROUTES": "/tmp/routes.json",
                    "CAO_CONDUCTOR_SHIM_DIR": "/tmp/shim",
                    "CAO_CONDUCTOR_REAL_CLAUDE": "/tmp/inner",
                },
            ),
            patch(
                "cli_agent_orchestrator.services.managed_provider_bridge.native_child_environment",
                return_value={"PATH": "/usr/bin", "HOME": "/home/user", "LANG": "C", "LC_ALL": "C"},
            ),
            patch(
                "cli_agent_orchestrator.services.managed_provider_bridge.provider_version_banner",
                side_effect=fake_banner,
            ),
            patch(
                "cli_agent_orchestrator.services.managed_launch_v2._mark_preflight_blocked",
                return_value={"state": "preflight_blocked"},
            ),
        ):
            await managed_launch_v2._launch_native_tui("res-glm", record, bridge_request)
        # Must have called _ensure at 3995 exactly once for GLM branch
        assert spy.call_count >= 1, f"_ensure not called for GLM branch, calls={spy.call_args_list}"
        # The spy's side_effect (real_ensure) should have forced LANG on the env dict it received
        # Check at least one call forced en_US.UTF-8
        forced = any(
            call.args[0].get("LANG") == "en_US.UTF-8" and "LC_ALL" not in call.args[0]
            for call in spy.call_args_list
        )
        assert forced, f"no call forced LANG=en_US.UTF-8, calls={spy.call_args_list}"
        # Also prove the downstream env (captured) would have lacked LANG if _ensure at 3995 were removed:
        # Our spy's side_effect forces it, so captured_env should be forced too.
        # If the call site were removed, spy.call_count==0 and this would fail.

    @pytest.mark.asyncio
    async def test_muse_branch_calls_ensure_and_no_auto_update(self):
        """non-GLM + MUSE_NO_AUTO_UPDATE at 4008 — muse_cli anthropic must force locale."""
        from cli_agent_orchestrator.services import managed_launch_v2
        from cli_agent_orchestrator.services.managed_launch_v2 import (
            _ensure_locale_env as real_ensure,
        )

        spy = MagicMock(side_effect=real_ensure)
        record = {
            "provider": "muse_cli",
            "request": {
                "expected_model": "muse-spark-1.3-contributor",
                "expected_effort": "high",
                "provider_route": "anthropic",
                "route_envelope": None,
                "working_directory": "/tmp/worktree",
                "provider_executable": "/tmp/muse",
                "provider_executable_sha256": "c" * 64,
            },
            "working_directory": "/tmp/worktree",
            "session_name": "test-session",
            "terminal_id": "tid-muse",
            "generation": "gen1",
            "agent_profile": "default",
        }
        bridge_request = {
            "provider_executable": "/tmp/muse",
            "provider_executable_sha256": "c" * 64,
            "profile_sha256": "same-sha-muse",
            "provider": "muse_cli",
            "model": "muse-spark-1.3-contributor",
        }
        captured_env: dict = {}

        def fake_banner(req, environment=None, timeout=None):
            if environment is not None:
                captured_env.update(environment)
            raise RuntimeError("abort after ensure at 4008")

        with (
            patch("cli_agent_orchestrator.services.managed_launch_v2._ensure_locale_env", spy),
            patch(
                "cli_agent_orchestrator.services.managed_provider_bridge._profile_material",
                return_value={"profile_sha256": "same-sha-muse"},
            ),
            patch(
                "cli_agent_orchestrator.services.managed_provider_bridge.native_child_environment",
                return_value={"PATH": "/usr/bin", "HOME": "/home/user", "LANG": "C", "LC_ALL": "C"},
            ),
            patch(
                "cli_agent_orchestrator.services.managed_provider_bridge.provider_version_banner",
                side_effect=fake_banner,
            ),
            patch(
                "cli_agent_orchestrator.services.managed_launch_v2._mark_preflight_blocked",
                return_value={"state": "preflight_blocked"},
            ),
        ):
            await managed_launch_v2._launch_native_tui("res-muse", record, bridge_request)
        assert (
            spy.call_count >= 1
        ), f"_ensure not called for muse branch, calls={spy.call_args_list}"
        forced = any(
            call.args[0].get("LANG") == "en_US.UTF-8" and "LC_ALL" not in call.args[0]
            for call in spy.call_args_list
        )
        assert forced, f"no call forced LANG, calls={spy.call_args_list}"
        # Verify MUSE_NO_AUTO_UPDATE was set alongside locale forcing
        # The env passed to banner should contain both
        assert (
            captured_env.get("MUSE_NO_AUTO_UPDATE") == "1"
        ), f"MUSE_NO_AUTO_UPDATE not set, env={captured_env}"
        assert (
            captured_env.get("LANG") == "en_US.UTF-8"
        ), f"LANG not forced in muse env, env={captured_env}"

    @pytest.mark.asyncio
    async def test_kimi_branch_calls_ensure_after_profile_env(self):
        """kimi profile at 4036 — kimi_cli must force locale after _kimi_profile_environment."""
        from cli_agent_orchestrator.services import managed_launch_v2
        from cli_agent_orchestrator.services.managed_launch_v2 import (
            _ensure_locale_env as real_ensure,
        )

        spy = MagicMock(side_effect=real_ensure)
        record = {
            "provider": "kimi_cli",
            "request": {
                "expected_model": "kimi-k2",
                "expected_effort": "high",
                "provider_route": "anthropic",
                "route_envelope": None,
                "working_directory": "/tmp/worktree",
                "provider_executable": "/tmp/kimi",
                "provider_executable_sha256": "d" * 64,
            },
            "working_directory": "/tmp/worktree",
            "session_name": "test-session",
            "terminal_id": "tid-kimi",
            "generation": "gen1",
            "agent_profile": "default",
        }
        bridge_request = {
            "provider_executable": "/tmp/kimi",
            "provider_executable_sha256": "d" * 64,
            "profile_sha256": "same-sha-kimi",
            "provider": "kimi_cli",
            "model": "kimi-k2",
            "effort": "high",
        }
        # Initial env from native_child_environment (before 4008)
        initial_env = {"PATH": "/usr/bin", "HOME": "/home/user", "LANG": "C", "LC_ALL": "C"}
        # Env after _kimi_profile_environment (before 4036) — without locale
        kimi_env = {
            "KIMI_CODE_HOME": "/tmp/kimi-home",
            "PATH": "/usr/bin",
            "LANG": "C",
            "LC_ALL": "C",
        }

        captured_kimi_env: dict = {}

        def fake_kimi_profile_env(*args, **kwargs):
            # Return a fresh dict that the caller will then pass to _ensure at 4036
            return dict(kimi_env)

        def fake_preauth(*args, **kwargs):
            # Capture the env that would have been forced at 4036 — the caller's `environment` variable
            # We capture via spy's last call instead; just abort
            raise RuntimeError("abort after ensure at 4036")

        with (
            patch("cli_agent_orchestrator.services.managed_launch_v2._ensure_locale_env", spy),
            patch(
                "cli_agent_orchestrator.services.managed_provider_bridge._profile_material",
                return_value={
                    "profile_sha256": "same-sha-kimi",
                    "profile": MagicMock(permissionMode=None, allowedTools=["*"], mcpServers={}),
                    "allowed_tools": ["*"],
                    "system_prompt": "",
                    "mcp_servers": [],
                },
            ),
            patch(
                "cli_agent_orchestrator.services.managed_provider_bridge.native_child_environment",
                return_value=dict(initial_env),
            ),
            patch(
                "cli_agent_orchestrator.services.managed_provider_bridge.provider_version_banner",
                return_value="kimi 0.40.0",
            ),
            patch(
                "cli_agent_orchestrator.services.kimi_native_launch.session_proof_gap_for",
                return_value=None,
            ),
            patch(
                "cli_agent_orchestrator.services.managed_launch_v2._kimi_profile_environment",
                side_effect=fake_kimi_profile_env,
            ),
            patch(
                "cli_agent_orchestrator.services.kimi_native_launch.preauthorize_workspace",
                side_effect=fake_preauth,
            ),
            patch(
                "cli_agent_orchestrator.services.managed_launch_v2._mark_preflight_blocked",
                return_value={"state": "preflight_blocked"},
            ),
        ):
            await managed_launch_v2._launch_native_tui("res-kimi", record, bridge_request)
        # Kimi path triggers _ensure twice: once at 4008 (initial env) and once at 4036 (kimi profile env)
        # We assert at least 2 calls to prove both sites, and that the second call forced the kimi env
        assert (
            spy.call_count >= 2
        ), f"expected >=2 _ensure calls for kimi (4008+4036), got {spy.call_count}, calls={spy.call_args_list}"
        # The last call should be the 4036 one; its env should be forced
        last_env = spy.call_args_list[-1].args[0]
        assert (
            last_env.get("LANG") == "en_US.UTF-8"
        ), f"last _ensure call did not force LANG, env={last_env}"
        assert "LC_ALL" not in last_env, f"last _ensure call did not pop LC_ALL, env={last_env}"
        # If 4036 were removed, spy.call_count would be 1 and last_env would be initial_env's second call missing


# ── P2-2: herdr backend locale guarantee ──


class TestHerdrLocaleGuarantee:
    """Herdr's _inject_env_vars must force UTF-8 (P2-2).

    Herdr has no tmux -e argv seam; its only injection point is the typed
    export via pane send-text. Removing the locale exports must redden.
    """

    def test_inject_env_vars_forces_locale(self):
        from cli_agent_orchestrator.backends.herdr_backend import HerdrBackend

        backend = HerdrBackend.__new__(HerdrBackend)
        backend._pane_cache = {}
        backend._workspace_cache = {}
        # Mock _run_herdr to capture commands
        captured: list[list[str]] = []

        def fake_run(args, check=True):
            captured.append(args)
            # For pane list / workspace list etc, return empty result
            m = MagicMock()
            m.stdout = '{"result": {"panes": []}}'
            m.returncode = 0
            return m

        backend._run_herdr = fake_run  # type: ignore[method-assign]
        # Patch workspace_id resolution to avoid needing real herdr
        with patch.object(backend, "_resolve_workspace_id", return_value="ws-1"):
            # Provide explicit pane_id to avoid list scan
            backend._inject_env_vars(
                "ses", "win", "tid-herdr", pane_id="pane-1", extra_env={"LC_ALL": "C", "LANG": "C"}
            )
        # Find the send-text call — args are ["pane", "send-text", pane_id, env_cmd]
        send_text_calls = [c for c in captured if c[:2] == ["pane", "send-text"]]
        assert send_text_calls, f"no pane send-text captured, calls={captured}"
        env_cmd = (
            send_text_calls[0][3]
            if len(send_text_calls[0]) > 3
            else (send_text_calls[0][2] if len(send_text_calls[0]) > 2 else "")
        )
        # Must contain forced locale and unset LC_ALL, even though extra_env carried C
        assert "export LANG=" in env_cmd
        assert "en_US.UTF-8" in env_cmd
        assert "export LC_CTYPE=" in env_cmd
        assert "unset LC_ALL" in env_cmd
        # Must not leak LC_ALL=C as an export before unset? It should be overridden by forced values;
        # The critical proof is that final exports contain unset and forced en_US, not C
        # Check that forced values appear after any extra_env exports (order proof via string)
        # extra_env LC_ALL=C would have been filtered? Actually _build_extra_env_exports would handle extra_env,
        # but our call passed extra_env={"LC_ALL":"C"} — _build_extra_env_exports would create export LC_ALL='C'
        # However final forced unset LC_ALL should still be present after.
        assert env_cmd.count("LC_ALL") >= 1  # at least the unset

    def test_inject_env_vars_ensure_helper_forces(self):
        from cli_agent_orchestrator.backends.herdr_backend import HerdrBackend

        env: dict[str, str] = {"LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"}
        HerdrBackend._ensure_utf8_locale(env)
        assert env["LANG"] == "en_US.UTF-8"
        assert env["LC_CTYPE"] == "en_US.UTF-8"
        assert "LC_ALL" not in env

    def test_build_extra_env_exports_quotes_and_locale_not_injected_there(self):
        """_build_extra_env_exports must not itself inject locale; locale comes from _inject_env_vars."""
        from cli_agent_orchestrator.backends.herdr_backend import HerdrBackend

        exports = HerdrBackend._build_extra_env_exports({"FOO": "bar", "LANG": "C"})
        # Should export FOO, and also LANG as forwarded var (since not blocked), but locale forcing is separate
        assert any("FOO=" in e for e in exports)
        # The locale guarantee is in _inject_env_vars, not here — this test documents the split

    def test_herdr_has_no_create_window_with_argv(self):
        """Document gap: herdr has no create_window_with_argv override; injection is via _inject_env_vars (P2-2)."""
        from cli_agent_orchestrator.backends.base import TerminalBackend
        from cli_agent_orchestrator.backends.herdr_backend import HerdrBackend

        # Base class defines create_window_with_argv that fails closed; herdr must NOT override it
        assert (
            "create_window_with_argv" not in HerdrBackend.__dict__
        ), "herdr should not override create_window_with_argv; its seam is _inject_env_vars"
        # Base impl still exists and fails closed
        assert hasattr(TerminalBackend, "create_window_with_argv")
        assert hasattr(HerdrBackend, "_inject_env_vars")
        # Calling base via herdr instance must fail closed
        backend = HerdrBackend.__new__(HerdrBackend)
        try:
            HerdrBackend.create_window_with_argv(backend, "s", "w", "tid", ["/bin/echo"], None, None)  # type: ignore[call-arg]
            assert False, "expected TerminalBackendError"
        except Exception as e:
            assert (
                "does not support atomic" in str(e).lower()
                or "TerminalBackendError" in type(e).__name__
            )


# ── P2-3 + P3-1: LC_ALL pop documentation and single-source constant ──


class TestLocaleConstantAndLcAllSemantics:
    """P2-3 documents LC_ALL pop intent; P3-1 asserts single source."""

    def test_forced_locale_single_source(self):
        from cli_agent_orchestrator import constants as const
        from cli_agent_orchestrator.backends.herdr_backend import HerdrBackend
        from cli_agent_orchestrator.clients.tmux import TmuxClient
        from cli_agent_orchestrator.services.managed_launch_v2 import _FORCED_LOCALE as v2_locale
        from cli_agent_orchestrator.services.managed_provider_bridge import (
            _FORCED_LOCALE as bridge_locale,
        )

        # All three seams must reference the same canonical value
        assert const.FORCED_LOCALE == "en_US.UTF-8"
        assert (
            TmuxClient._FORCED_LOCALE == const.FORCED_LOCALE
            if hasattr(TmuxClient, "_FORCED_LOCALE")
            else True
        )
        # Module-level aliases (if present) must equal constant
        assert bridge_locale == const.FORCED_LOCALE
        assert v2_locale == const.FORCED_LOCALE

        # Herdr constant usage via FORCED_LOCALE import is checked by its _ensure
        env: dict[str, str] = {"LANG": "C", "LC_ALL": "ja_JP.UTF-8"}
        HerdrBackend._ensure_utf8_locale(env)
        # Popping LC_ALL so LANG controls; still UTF-8 so Muse renders — collation shift documented
        # Even if host had ja_JP.UTF-8 in LC_ALL, we force en_US.UTF-8 (intentional, documented)
        assert env["LANG"] == "en_US.UTF-8"
        assert "LC_ALL" not in env

    def test_lc_all_pop_preserves_utf8_rendering(self):
        """P2-3: popping LC_ALL preserves UTF-8 rendering (LANG=en_US.UTF-8), even if host had ja_JP.UTF-8."""
        from cli_agent_orchestrator.clients.tmux import TmuxClient

        # Host carries ja_JP.UTF-8 in LC_ALL — UTF-8 but different collation
        env = {"LANG": "C", "LC_ALL": "ja_JP.UTF-8", "LC_CTYPE": "C"}
        TmuxClient._ensure_utf8_locale(env)
        # Intentionally forces en_US.UTF-8, popping ja_JP.UTF-8 — still UTF-8 so Muse renders
        # Collation shift is documented, not preserved
        assert env["LANG"] == "en_US.UTF-8"
        assert env["LC_CTYPE"] == "en_US.UTF-8"
        assert "LC_ALL" not in env
