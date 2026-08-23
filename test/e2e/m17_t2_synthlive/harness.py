"""M17 T2 synthetic-live harness: real tmux panes + the conductor's launcher.

Everything here drives REAL terminals, REAL processes, and the REAL
``fire-marshal.sh`` launch path.  Nothing is monkeypatched; the only fake is
the ``codex`` provider binary (see :mod:`stub_codex`), which renders captured
fixture bytes as real pane content.  ``enabled()`` in the dark
route-observation modules is never flipped — the adapter is driven directly,
exactly as the future driver loop would, because the dark gate is a marker,
not an enforcement point.

The harness owns one isolated tmux server (``TMUX_TMPDIR``) and one isolated
conductor state root, so no test touches the operator's live tmux or live
``~/.local/state/cao-conductor``.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

#: Where the conductor checkout is looked for.  Set ``T2_CONDUCTOR_REPO`` to
#: point at another checkout; the conventional path covers the standard
#: layout.  The harness never mutates that repo.
T2_CONDUCTOR_REPO = "T2_CONDUCTOR_REPO"

#: The attested ``cli-agent-orchestrator`` uv-tool install layout where the
#: stub ``codex`` binary must live (acceptance 1).  Overridable for CI/scratch
#: runs; the default is the real attested path.
T2_ATTESTED_BIN = "T2_ATTESTED_BIN"

#: The two bashes ``fire-marshal.sh`` claims to support.
HOMEBREW_BASH = Path("/opt/homebrew/bin/bash")
SEALED_BASH = Path("/bin/bash")

XDG_STATE_HOME = "XDG_STATE_HOME"
CONDUCTOR_STATE_SUBDIR = "cao-conductor"


@dataclasses.dataclass(frozen=True)
class LaunchResult:
    """One ``fire-marshal.sh`` invocation and what it produced."""

    exit_code: int
    output: str
    launch_dir: Optional[Path]
    pane_id: Optional[str]
    session_name: Optional[str]


def conductor_root() -> Optional[Path]:
    """The conductor checkout the harness shells out to, or None."""
    env = os.environ.get(T2_CONDUCTOR_REPO)
    if env:
        root = Path(env).expanduser()
        if (root / "scripts" / "fire-marshal.sh").is_file():
            return root
        return None
    conventional = Path.home() / "Projects" / "cao-conductor"
    if (conventional / "scripts" / "fire-marshal.sh").is_file():
        return conventional
    return None


def attested_bin_dir() -> Path:
    env = os.environ.get(T2_ATTESTED_BIN)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".local" / "share" / "uv" / "tools" / "cli-agent-orchestrator" / "bin"


def supported_bashes() -> list[Path]:
    """The bash binaries fire-marshal.sh supports that are present here."""
    present: list[Path] = []
    if HOMEBREW_BASH.is_file():
        present.append(HOMEBREW_BASH)
    if SEALED_BASH.is_file():
        present.append(SEALED_BASH)
    return present


class T2Harness:
    """Owns the isolated tmux server, state root, and the stub install."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.conductor = conductor_root()
        # tmux refuses a socket path longer than sockaddr_un allows, so the
        # socket dir is short by construction (a nested pytest tmp path blows
        # that budget) — the same constraint fire-marshal-e2e.sh documents.
        import uuid as uuidlib

        self.tmux_tmp = (
            Path(tempfile.gettempdir()) / f"t2-tmux-{os.getpid()}-{uuidlib.uuid4().hex[:6]}"
        )
        self.tmux_tmp.mkdir(parents=True, exist_ok=True)
        self.state_root = workdir / "state"
        self.cao_home = workdir / "cao-home"
        self.shim_bin = workdir / "bin"
        self.quota_skill = workdir / "quota-skill"
        self.fakes_root = workdir / "fakes"
        self.incidents_dir = self.state_root / CONDUCTOR_STATE_SUBDIR / "fire-marshal" / "incidents"
        self.attested = attested_bin_dir()
        self._stub_backup: Optional[bytes] = None
        self._stub_installed = False
        self._tmux_env_restore: dict[str, Optional[str]] = {}

    # -- lifecycle ------------------------------------------------------------

    def __enter__(self) -> "T2Harness":
        for key in ("TMUX_TMPDIR", "TMUX", XDG_STATE_HOME, "CAO_STATE_ROOT"):
            self._tmux_env_restore[key] = os.environ.get(key)
        os.environ["TMUX_TMPDIR"] = str(self.tmux_tmp)
        os.environ.pop("TMUX", None)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.cao_home.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.kill_tmux_server()
        self.uninstall_stub()
        shutil.rmtree(self.tmux_tmp, ignore_errors=True)
        for key, value in self._tmux_env_restore.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # -- the stub binary -------------------------------------------------------

    def stub_source(self) -> Path:
        return Path(__file__).resolve().parent / "stub_codex.py"

    def stub_bin(self) -> Path:
        return self.attested / "codex"

    def attested_writable(self) -> bool:
        if not self.attested.is_dir():
            return False
        return os.access(self.attested, os.W_OK)

    def install_stub(self) -> Path:
        """Render the committed stub template with the real fixtures and
        install it at the attested layout.  Returns the installed path."""
        self.attested.mkdir(parents=True, exist_ok=True)
        target = self.stub_bin()
        if target.exists():
            self._stub_backup = target.read_bytes()
            target.unlink()
        from test.e2e.route_observation_canary import fixtures as fx

        positive = _positive_panel_rows(fx)
        rendered = self.stub_source().read_text(encoding="utf-8")
        rendered = (
            rendered.replace("_STUB_POSITIVE_ROWS_", repr(positive))
            .replace("_STUB_FIX80_ROWS_", repr(list(fx.CAPTURED_STATUS_80X30_ROWS)))
            .replace("_STUB_FIX100_ROWS_", repr(list(fx.CAPTURED_STATUS_100X30_ROWS)))
            .replace("_STUB_GARBAGE_ROWS_", repr(_garbage_rows()))
        )
        target.write_text(rendered, encoding="utf-8")
        target.chmod(0o755)
        self._stub_installed = True
        return target

    def uninstall_stub(self) -> None:
        if not self._stub_installed:
            return
        target = self.stub_bin()
        try:
            if self._stub_backup is not None:
                target.write_bytes(self._stub_backup)
            elif target.exists():
                target.unlink()
        finally:
            self._stub_installed = False
            self._stub_backup = None

    # -- tmux ------------------------------------------------------------------

    def _env(self, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
        env = dict(os.environ)
        env["TMUX_TMPDIR"] = str(self.tmux_tmp)
        env.pop("TMUX", None)
        if extra:
            env.update(extra)
        return env

    def _run(
        self, argv: list[str], *, extra: Optional[dict[str, str]] = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=60, check=False, env=self._env(extra)
        )

    def new_pane(
        self, *, width: int, height: int, command: str, session: Optional[str] = None
    ) -> str:
        """Create one real tmux pane running ``command``; return its pane id."""
        if session is None:
            import uuid as uuidlib

            session = f"t2-{uuidlib.uuid4().hex[:8]}"
        created = self._run(
            [
                "tmux",
                "new-session",
                "-d",
                "-P",
                "-F",
                "#{session_name}",
                "-s",
                session,
                "-x",
                str(width),
                "-y",
                str(height),
                command,
            ]
        )
        if created.returncode != 0:
            raise RuntimeError(f"tmux new-session failed: {created.stderr.strip()}")
        pane = self._run(["tmux", "display", "-p", "-t", f"{session}:.0", "#{pane_id}"])
        if pane.returncode != 0:
            raise RuntimeError(f"tmux pane lookup failed: {pane.stderr.strip()}")
        pane_id = (pane.stdout or "").strip()
        if not pane_id:
            raise RuntimeError("tmux returned no pane id")
        return pane_id

    def capture(self, pane_id: str) -> list[str]:
        result = self._run(["tmux", "capture-pane", "-p", "-t", pane_id])
        if result.returncode != 0:
            raise RuntimeError(f"tmux capture-pane failed: {result.stderr.strip()}")
        return (result.stdout or "").splitlines()

    def pane_width(self, pane_id: str) -> int:
        result = self._run(["tmux", "display", "-p", "-t", pane_id, "#{pane_width}"])
        if result.returncode != 0:
            raise RuntimeError(f"tmux width failed: {result.stderr.strip()}")
        return int((result.stdout or "").strip())

    def pane_pid(self, pane_id: str) -> int:
        result = self._run(["tmux", "display", "-p", "-t", pane_id, "#{pane_pid}"])
        if result.returncode != 0:
            raise RuntimeError(f"tmux pid failed: {result.stderr.strip()}")
        return int((result.stdout or "").strip())

    def pane_alive(self, pane_id: str) -> bool:
        result = self._run(["tmux", "display", "-p", "-t", pane_id, "#{pane_dead}"])
        return result.returncode == 0 and (result.stdout or "").strip() == "0"

    def kill_pane(self, pane_id: str) -> None:
        self._run(["tmux", "kill-pane", "-t", pane_id])

    def kill_session(self, session: str) -> None:
        self._run(["tmux", "kill-session", "-t", session])

    def kill_tmux_server(self) -> None:
        self._run(["tmux", "kill-server"])

    # -- conductor launch surface ------------------------------------------------

    def write_incident(self, incident_id: str, evidence: dict) -> Path:
        incident_dir = self.incidents_dir / incident_id
        incident_dir.mkdir(parents=True, exist_ok=True)
        path = incident_dir / "wake-evidence.json"
        path.write_text(
            _json(evidence) + "\n",
            encoding="utf-8",
        )
        return path

    def setup_launch_surface(self, *, stub_bin: Path) -> dict[str, str]:
        """Write the shimmed conduct, quota stub, and stub codex-headless
        launcher, and return the environment the launch must run under."""
        if self.conductor is None:
            raise RuntimeError("conductor checkout is required for a marshal launch")
        self.shim_bin.mkdir(parents=True, exist_ok=True)
        conduct = self.shim_bin / "conduct"
        conduct.write_text(
            f'#!/bin/sh\ncd "{self.conductor}" && exec python3 -m conduct.cli "$@"\n',
            encoding="utf-8",
        )
        conduct.chmod(0o755)

        self.quota_skill.mkdir(parents=True, exist_ok=True)
        quota = self.quota_skill / "scripts" / "check-ai-quota.sh"
        quota.parent.mkdir(parents=True, exist_ok=True)
        quota.write_text(
            '#!/bin/sh\nprintf \'{"ok":true,"pressure_level":"green"}\\n\'\nexit 0\n',
            encoding="utf-8",
        )
        quota.chmod(0o755)

        self.fakes_root.mkdir(parents=True, exist_ok=True)
        headless = self.fakes_root / "codex-headless" / "scripts" / "codex-headless.sh"
        headless.parent.mkdir(parents=True, exist_ok=True)
        headless.write_text(_codex_headless_stub(stub_bin), encoding="utf-8")
        headless.chmod(0o755)

        return {
            "PATH": f"{self.shim_bin}:{os.environ.get('PATH', '')}",
            XDG_STATE_HOME: str(self.state_root),
            "CAO_STATE_ROOT": str(self.cao_home),
            "HEADLESS_SKILLS_ROOT": str(self.fakes_root),
            "CHECK_AI_QUOTA_SKILL": str(self.quota_skill),
            "TMUX_TMPDIR": str(self.tmux_tmp),
        }

    def launch_marshal(
        self, *, bash: Path, incident_id: str, extra_env: dict[str, str]
    ) -> LaunchResult:
        """Run the real ``fire-marshal.sh`` launch under ``bash``."""
        if self.conductor is None:
            raise RuntimeError("conductor checkout is required for a marshal launch")
        script = self.conductor / "scripts" / "fire-marshal.sh"
        env = self._env(extra_env)
        result = subprocess.run(
            [str(bash), str(script), "--incident", incident_id, "--harness", "codex"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=env,
        )
        output = result.stdout + result.stderr
        launch_dir = None
        pane_id = None
        session_name = None
        run_dir = _first_line_matching(output, "RUN_DIR")
        if run_dir:
            launch_dir = Path(run_dir)
            pane_id = _read_file_quietly(launch_dir / "pane_id")
            session_name = _read_file_quietly(launch_dir / "session_name")
        return LaunchResult(
            exit_code=result.returncode,
            output=output,
            launch_dir=launch_dir,
            pane_id=pane_id,
            session_name=session_name,
        )


def _json(value: dict) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def _first_line_matching(output: str, key: str) -> Optional[str]:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(key):
            rest = stripped[len(key) :].strip()
            # fire-marshal's banner is "RUN_DIR : <path>"; drop the separator.
            if rest.startswith(":"):
                rest = rest[1:].strip()
            return rest if rest else None
    return None


def _read_file_quietly(path: Path) -> Optional[str]:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def _garbage_rows() -> list[str]:
    return [f"garbage-line-{index:02d}: not-a-codex-status-panel" for index in range(30)]


def _positive_panel_rows(fx) -> list[str]:
    """The full 100-column captured context with a parseable Model row, plus
    the Codex composer prompt line the close proof needs to read restored."""
    rows: list[str] = []
    for row in fx.CAPTURED_STATUS_100X30_ROWS:
        if "gpt-5.6-luna (reasoning medium, summaries auto)" in row:
            row = row.replace("(reasoning medium, summaries auto)", "(reasoning high)")
        rows.append(row)
    rows.append("› ")
    return rows


def _codex_headless_stub(stub_bin: Path) -> str:
    """The stub codex-headless launcher: starts the stub codex in a REAL
    detached tmux pane and reports RUN_DIR/PID/session_id exactly as the
    production launcher family does, so fire-marshal's session capture and
    single-root holder see a real process with a real start marker."""
    return f"""#!/bin/sh
RD=$(mktemp -d "${{TMPDIR:-/tmp}}/t2-launch-XXXXXX")
SESS=$(tmux new-session -d -P -F '#{{session_name}}' -s "fm-$$" -x 100 -y 31 "exec {stub_bin}" 2>/dev/null)
PANE_ID=$(tmux display -p -t "$SESS:.0" '#{{pane_id}}' 2>/dev/null)
PID=$(tmux display -p -t "$PANE_ID" '#{{pane_pid}}' 2>/dev/null)
printf '%s\\n' "$PANE_ID" > "$RD/pane_id"
printf '%s\\n' "$SESS" > "$RD/session_name"
printf '%s\\n' "$PID" > "$RD/pane_pid"
printf '{{"session_id_file":"%s/session_id"}}\\n' "$RD" > "$RD/meta.json"
printf 'stub-session-%s\\n' "$$" > "$RD/session_id"
printf 'RUN_DIR : %s\\n' "$RD"
printf 'PID     : %s\\n' "$PID"
"""
