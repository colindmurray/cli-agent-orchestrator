"""Supervisor launch-resolved profile receipts (cond-0817).

Covers the runtime half of the launch-resolved profile contract: the single
launch-boundary read with installed-store precedence, contract validation
(match / divergence / malformed), the threaded no-reload wiring through
session and terminal creation into provider construction, durable receipt
persistence and projection (including legacy absence and post-launch drift),
and the HTTP conflict/retry mapping.
"""

import hashlib
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.models.agent_profile import AgentProfile
from cli_agent_orchestrator.providers.base import SealedLaunchMaterial
from cli_agent_orchestrator.services import session_service
from cli_agent_orchestrator.services import supervisor_profile_receipt as spr
from cli_agent_orchestrator.services.supervisor_profile_receipt import (
    PROFILE_LAUNCH_CONTRACT_SCHEMA,
    PROFILE_RECEIPT_SCHEMA,
    ProfileLaunchConflict,
    build_profile_receipt,
    load_supervisor_launch_context,
    validate_profile_contract,
)
from cli_agent_orchestrator.utils import agent_profiles
from cli_agent_orchestrator.utils.skills import build_skill_catalog

_SERVICE = "cli_agent_orchestrator.services.terminal_service"


def _write_profile(
    store: Path,
    name: str,
    *,
    provider: str = "mock_cli",
    model: str | None = "test-model-1",
    effort: str | None = None,
    role: str = "supervisor",
    body: str = "Do supervision.",
    extra: tuple = (),
) -> Path:
    """Write one flat ``<name>.md`` profile into a store directory.

    ``extra`` appends verbatim frontmatter lines (e.g.
    ``('allowedTools: ["*"]', "skills: []")``) for fields without a
    dedicated knob.
    """
    lines = ["---", f"name: {name}", f"description: {name} profile"]
    if provider is not None:
        lines.append(f"provider: {provider}")
    if role is not None:
        lines.append(f"role: {role}")
    if model is not None:
        lines.append(f"model: {model}")
    if effort is not None:
        lines.extend(["codexConfig:", f"  model_reasoning_effort: {effort}"])
    lines.extend(extra)
    lines.append("---")
    lines.append(body)
    path = store / f"{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def profile_store(tmp_path, monkeypatch):
    """Isolate profile resolution to one scratch local store."""
    from cli_agent_orchestrator.services import settings_service

    store = tmp_path / "agent-store"
    store.mkdir()
    monkeypatch.setattr(agent_profiles, "LOCAL_AGENT_STORE_DIR", store)
    monkeypatch.setattr(settings_service, "get_agent_dirs", lambda: {})
    monkeypatch.setattr(settings_service, "get_extra_agent_dirs", lambda: [])
    monkeypatch.setattr(settings_service, "get_disabled_agent_dirs", lambda: [])
    return store


@pytest.fixture
def launched_provider():
    provider = AsyncMock()
    provider.initialize.return_value = True
    provider.shell_baseline = None
    return provider


def _hermetic_backend():
    backend = MagicMock()
    backend.session_exists.return_value = False
    backend.window_identity.return_value = {
        "pane_id": "%1",
        "window_id": "@1",
        "server_socket_path": "/tmp/tmux",
        "session_id": "$1",
        "pane_pid": 12345,
    }
    backend.supports_event_inbox.return_value = False
    backend.supports_pane_identity.return_value = False
    return backend


class _HermeticLaunch(ExitStack):
    """Backend + FIFO + status patches for a hermetic create_terminal run."""

    def __init__(self, *, provider_manager, launched_provider=None, terminal_id, session):
        super().__init__()
        backend = _hermetic_backend()
        self.enter_context(patch("cli_agent_orchestrator.backends.registry._backend", backend))
        self.enter_context(patch(f"{_SERVICE}.provider_manager", provider_manager))
        self.enter_context(patch(f"{_SERVICE}.fifo_manager"))
        self.enter_context(patch(f"{_SERVICE}.status_monitor"))
        self.enter_context(patch(f"{_SERVICE}.clear_session_env"))
        self.enter_context(patch(f"{_SERVICE}._register_incarnation"))
        self.enter_context(patch(f"{_SERVICE}.generate_terminal_id", return_value=terminal_id))
        self.enter_context(patch(f"{_SERVICE}.generate_session_name", return_value=session))
        self.enter_context(patch(f"{_SERVICE}.generate_window_name", return_value="w-sup"))
        # A None launched_provider leaves construction alone, so a test can
        # run the real ProviderManager (wrapping create_provider itself to
        # observe the call) instead of a stubbed instance.
        if launched_provider is not None:
            provider_manager.create_provider.return_value = launched_provider


def _contract_for(context) -> dict:
    return {
        "schema": PROFILE_LAUNCH_CONTRACT_SCHEMA,
        "profile": context.profile_name,
        "role": "supervisor",
        "provider": context.provider,
        "model": context.model,
        "effort": context.effort,
        "provenance": context.provenance,
        "source_path": context.source_path,
        "sha256": context.sha256,
    }


class TestLaunchContext:
    def test_single_read_supplies_profile_source_and_digest(self, profile_store):
        path = _write_profile(profile_store, "sup")
        byte_reads = []
        text_reads = []
        real_read_bytes = Path.read_bytes
        real_read_text = Path.read_text

        def _counting_bytes(self, *args, **kwargs):
            if self == path:
                byte_reads.append(self)
            return real_read_bytes(self, *args, **kwargs)

        def _counting_text(self, *args, **kwargs):
            if self == path:
                text_reads.append(self)
            return real_read_text(self, *args, **kwargs)

        with (
            patch.object(Path, "read_bytes", _counting_bytes),
            patch.object(Path, "read_text", _counting_text),
        ):
            context = load_supervisor_launch_context("sup")

        # Exactly one binary content read; no text-mode (newline-normalising)
        # read of the profile — the digest is over the exact source bytes.
        assert byte_reads == [path]
        assert text_reads == []
        assert context.profile_name == "sup"
        assert context.source_path == str(path)
        assert context.provenance == "installed-agent-store"
        assert context.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert isinstance(context.profile, AgentProfile)
        assert context.profile.system_prompt == "Do supervision."
        assert context.provider == "mock_cli"
        assert context.model == "test-model-1"
        assert context.effort is None

    def test_installed_store_takes_precedence_over_builtin(self, profile_store):
        _write_profile(profile_store, "developer", model="shadow-model")
        context = load_supervisor_launch_context("developer")
        assert context.provenance == "installed-agent-store"
        assert context.model == "shadow-model"

    def test_builtin_resolves_with_builtin_provenance(self, profile_store):
        context = load_supervisor_launch_context("developer")
        assert context.provenance == "built-in"
        assert context.source_path == "built-in:developer.md"

    def test_codex_route_applies_config_seam(self, profile_store):
        _write_profile(
            profile_store,
            "codex-sup",
            provider="codex",
            model="profile-model",
            effort="xhigh",
        )
        context = load_supervisor_launch_context("codex-sup")
        assert context.provider == "codex"
        assert context.model == "profile-model"
        assert context.effort == "xhigh"

    def test_codex_config_model_wins_over_bare_model(self, profile_store, tmp_path):
        path = tmp_path / "agent-store" / "codex-cfg.md"
        path.write_text(
            "---\nname: codex-cfg\ndescription: x\nprovider: codex\n"
            "model: bare-model\ncodexConfig:\n  model: cfg-model\n"
            "  model_reasoning_effort: xhigh\n---\nbody\n",
            encoding="utf-8",
        )
        context = load_supervisor_launch_context("codex-cfg")
        assert context.model == "cfg-model"
        assert context.effort == "xhigh"

    def test_explicit_provider_wins_and_invalid_profile_provider_falls_back(self, profile_store):
        _write_profile(profile_store, "odd", provider="not-a-provider", model="m")
        assert (
            load_supervisor_launch_context("odd", explicit_provider="kimi_cli").provider
            == "kimi_cli"
        )
        assert load_supervisor_launch_context("odd").provider == "kiro_cli"

    def test_missing_profile_raises_before_any_effect(self, profile_store):
        with pytest.raises(spr.ProfileNotFoundError):
            load_supervisor_launch_context("ghost")

    def test_missing_profile_error_stays_a_filenotfound(self, profile_store):
        """The typed error remains a FileNotFoundError for existing handlers."""
        with pytest.raises(FileNotFoundError):
            load_supervisor_launch_context("ghost")

    def test_unparseable_profile_raises_typed_invalid(self, profile_store):
        (profile_store / "bad.md").write_text("---\nfoo: [unclosed\n---\nbody\n", encoding="utf-8")
        with pytest.raises(spr.ProfileInvalidError):
            load_supervisor_launch_context("bad")

    def test_unparseable_profile_error_stays_a_valueerror(self, profile_store):
        (profile_store / "bad.md").write_text("---\nfoo: [unclosed\n---\nbody\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_supervisor_launch_context("bad")

    def test_receipt_is_runtime_authored_not_request_echo(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        receipt = build_profile_receipt(context)
        assert receipt == {
            "schema": PROFILE_RECEIPT_SCHEMA,
            "profile": "sup",
            "role": "supervisor",
            "provider": "mock_cli",
            "model": "test-model-1",
            "effort": None,
            "provenance": "installed-agent-store",
            "source_path": context.source_path,
            "sha256": context.sha256,
        }


class TestContractValidation:
    @pytest.fixture
    def context(self, profile_store):
        _write_profile(profile_store, "sup")
        return load_supervisor_launch_context("sup")

    def test_matching_contract_validates(self, context):
        validate_profile_contract(_contract_for(context), context)

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("profile", "other"),
            ("role", "worker"),
            ("provider", "kimi_cli"),
            ("model", "other-model"),
            ("effort", "xhigh"),
            ("provenance", "built-in"),
            ("source_path", "/elsewhere/sup.md"),
            ("sha256", "0" * 64),
        ],
    )
    def test_every_compared_field_divergence_conflicts(self, context, field, bad):
        """Each compared field is pinned: dropping any one comparison breaks this."""
        contract = _contract_for(context)
        if field == "role":
            with pytest.raises(ValueError):
                validate_profile_contract({**contract, field: bad}, context)
            return
        if field == "effort":
            # The fixture profile declares no effort; a null->value flip
            # must still conflict rather than pass as "undeclared".
            assert context.effort is None
        with pytest.raises(ProfileLaunchConflict) as exc_info:
            validate_profile_contract({**contract, field: bad}, context)
        assert field in [entry["field"] for entry in exc_info.value.divergent_fields]
        assert exc_info.value.retry

    @pytest.mark.parametrize(
        "contract",
        [
            "not-a-mapping",
            {"schema": "wrong-schema"},
            {"schema": PROFILE_LAUNCH_CONTRACT_SCHEMA, "role": "worker"},
            {"schema": PROFILE_LAUNCH_CONTRACT_SCHEMA, "role": "supervisor"},
            {
                "schema": PROFILE_LAUNCH_CONTRACT_SCHEMA,
                "role": "supervisor",
                "profile": "sup",
                "provider": "mock_cli",
                "model": 42,
                "effort": None,
                "provenance": "local",
                "source_path": "x",
                "sha256": "y",
            },
        ],
    )
    def test_malformed_contracts_are_refused(self, context, contract):
        with pytest.raises(ValueError):
            validate_profile_contract(contract, context)


def _write_bytes_profile(store: Path, name: str, raw: bytes) -> Path:
    """Write exact profile bytes (line endings preserved) into a store."""
    path = store / f"{name}.md"
    path.write_bytes(raw)
    return path


def _write_mcp_profile(store: Path, name: str) -> Path:
    """Write an Antigravity profile carrying one MCP server entry."""
    return _write_bytes_profile(
        store,
        name,
        b"---\nname: sup\ndescription: mcp profile\nprovider: antigravity_cli\n"
        b"role: supervisor\nmodel: test-model-1\nmcpServers:\n"
        b"  cao-mcp-server:\n    command: cao-mcp-server\n---\nDo supervision.\n",
    )


class TestExactByteLoading:
    _CRLF = (
        b"---\r\nname: sup\r\ndescription: crlf profile\r\nprovider: codex\r\n"
        b"role: supervisor\r\nmodel: frozen-model\r\ncodexConfig:\r\n"
        b"  model_reasoning_effort: xhigh\r\n---\r\nDo supervision.\r\n"
    )

    def test_crlf_sha_matches_raw_bytes_and_contract_validates(self, profile_store):
        """A CRLF profile hashes as read: no newline normalisation.

        The runtime digest must equal sha256 of the exact file bytes (what
        the conductor preflighted), and the conductor-style contract built
        from the context validates — the round-5 text-mode read hashed the
        LF-normalised copy and could never agree here.
        """
        path = _write_bytes_profile(profile_store, "sup", self._CRLF)
        context = load_supervisor_launch_context("sup")
        assert context.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert context.sha256 == hashlib.sha256(self._CRLF).hexdigest()
        validate_profile_contract(_contract_for(context), context)
        receipt = build_profile_receipt(context)
        assert receipt["sha256"] == hashlib.sha256(self._CRLF).hexdigest()

    def test_crlf_parsed_fields_come_from_the_same_snapshot(self, profile_store):
        """Model, provider, effort, and prompt parse from the decoded bytes."""
        _write_bytes_profile(profile_store, "sup", self._CRLF)
        context = load_supervisor_launch_context("sup")
        assert context.provider == "codex"
        assert context.model == "frozen-model"
        assert context.effort == "xhigh"
        assert context.profile.system_prompt == "Do supervision."

    def test_crlf_to_lf_rewrite_conflicts_on_sha(self, profile_store):
        """Changing only line endings changes the digest: the old contract
        becomes a typed 409 naming sha256, not a silent agreement."""
        _write_bytes_profile(profile_store, "sup", self._CRLF)
        context = load_supervisor_launch_context("sup")
        stale_contract = _contract_for(context)
        _write_bytes_profile(profile_store, "sup", self._CRLF.replace(b"\r\n", b"\n"))
        fresh = load_supervisor_launch_context("sup")
        assert fresh.sha256 != context.sha256
        with pytest.raises(ProfileLaunchConflict) as exc_info:
            validate_profile_contract(stale_contract, fresh)
        assert "sha256" in [entry["field"] for entry in exc_info.value.divergent_fields]
        assert exc_info.value.retry


class TestStrictContractParsing:
    def _valid_raw(self, context):
        return dict(_contract_for(context))

    def test_non_mapping_contracts_are_malformed(self, profile_store):
        _write_profile(profile_store, "sup")
        load_supervisor_launch_context("sup")
        for raw in (["not", "an", "object"], "a-string", 42, None):
            with pytest.raises(spr.ProfileContractMalformed):
                spr.parse_profile_contract(raw)

    def test_missing_and_extra_fields_are_malformed(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        raw = self._valid_raw(context)
        del raw["sha256"]
        with pytest.raises(spr.ProfileContractMalformed):
            spr.parse_profile_contract(raw)
        raw = self._valid_raw(context)
        raw["unexpected"] = 1
        with pytest.raises(spr.ProfileContractMalformed):
            spr.parse_profile_contract(raw)

    def test_wrong_schema_role_and_types_are_malformed(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        for field, value in (
            ("schema", "wrong-schema"),
            ("role", "developer"),
            ("profile", 42),
            ("provider", ""),
            ("provenance", ""),
            ("model", 42),
            ("effort", 42),
        ):
            raw = self._valid_raw(context)
            raw[field] = value
            with pytest.raises(spr.ProfileContractMalformed):
                spr.parse_profile_contract(raw)

    def test_invalid_source_path_forms_are_malformed(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        for bad in ("relative/path.md", "", "built-in:", "built-in:no-suffix", 42):
            raw = self._valid_raw(context)
            raw["source_path"] = bad
            with pytest.raises(spr.ProfileContractMalformed):
                spr.parse_profile_contract(raw)
        raw = self._valid_raw(context)
        raw["source_path"] = "built-in:developer.md"
        spr.parse_profile_contract(raw)

    def test_sha_must_be_exactly_64_hex(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        for bad in ("", "xyz", "a" * 63, "a" * 65, "g" * 64, "a" * 63 + " "):
            raw = self._valid_raw(context)
            raw["sha256"] = bad
            with pytest.raises(spr.ProfileContractMalformed):
                spr.parse_profile_contract(raw)

    def test_uppercase_sha_normalizes_and_validates(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        raw = self._valid_raw(context)
        raw["sha256"] = raw["sha256"].upper()
        parsed = spr.parse_profile_contract(raw)
        assert parsed["sha256"] == context.sha256
        validate_profile_contract(parsed, context)
        assert build_profile_receipt(context)["sha256"] == context.sha256

    def test_well_formed_drift_is_conflict_not_malformed(self, profile_store):
        """A valid-shape value that differs is 409 territory, not 400:
        the parser accepts it and the validator names the divergence."""
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        raw = self._valid_raw(context)
        raw["provenance"] = "composed-registry"
        parsed = spr.parse_profile_contract(raw)
        with pytest.raises(ProfileLaunchConflict) as exc_info:
            validate_profile_contract(parsed, context)
        assert "provenance" in [entry["field"] for entry in exc_info.value.divergent_fields]


class TestInstalledStoreBoundary:
    def test_sibling_prefix_is_outside_the_store(self, tmp_path, monkeypatch):
        """Direct pin for the containment boundary.

        ``tmp/foo2/sup.md`` shares a string prefix with ``tmp/foo`` but is
        not inside it: containment is a ``commonpath`` equality, never a
        bare prefix match — mutating the boundary to plain ``startswith``
        turns this red.
        """
        from cli_agent_orchestrator.utils import agent_profiles

        monkeypatch.setattr(agent_profiles, "LOCAL_AGENT_STORE_DIR", tmp_path / "foo")
        inside = os.path.realpath(tmp_path / "foo" / "sup.md")
        sibling = os.path.realpath(tmp_path / "foo2" / "sup.md")
        assert spr._is_within_installed_store(inside) is True
        assert spr._is_within_installed_store(sibling) is False
        assert spr._is_within_installed_store("built-in:sup.md") is False


class TestCanonicalProvenance:
    def test_installed_store_emits_canonical_provenance(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        assert context.provenance == "installed-agent-store"
        assert build_profile_receipt(context)["provenance"] == "installed-agent-store"

    def test_legacy_local_contract_accepted_within_installed_store(self, profile_store):
        """An older conductor-era contract carrying "local" converges when
        the runtime source really is the installed store and the digest
        matches — compatibility without a second alias table."""
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        contract["provenance"] = "local"
        validate_profile_contract(contract, context)

    def test_legacy_local_contract_refused_for_builtin_source(self, profile_store):
        """ "local" over packaged bytes is a divergence, not a spelling."""
        context = load_supervisor_launch_context("developer")
        assert context.provenance == "built-in"
        contract = _contract_for(context)
        contract["provenance"] = "local"
        with pytest.raises(ProfileLaunchConflict) as exc_info:
            validate_profile_contract(contract, context)
        assert "provenance" in [entry["field"] for entry in exc_info.value.divergent_fields]

    def test_legacy_local_contract_refused_for_sibling_prefix_store(
        self, profile_store, tmp_path, monkeypatch
    ):
        """A sibling directory sharing a string prefix is not the store.

        ``tmp/foo2/sup.md`` must not satisfy the installed-store check for
        ``tmp/foo``: containment is a ``commonpath`` equality, never a bare
        prefix match — mutating the boundary to plain ``startswith`` turns
        this red.
        """
        from cli_agent_orchestrator.services import settings_service
        from cli_agent_orchestrator.utils import agent_profiles

        monkeypatch.setattr(agent_profiles, "LOCAL_AGENT_STORE_DIR", tmp_path / "foo")
        sibling = tmp_path / "foo2"
        sibling.mkdir()
        _write_profile(sibling, "sup")
        monkeypatch.setattr(settings_service, "get_extra_agent_dirs", lambda: [str(sibling)])
        context = load_supervisor_launch_context("sup")
        assert context.provenance == "custom"
        contract = _contract_for(context)
        contract["provenance"] = "local"
        with pytest.raises(ProfileLaunchConflict) as exc_info:
            validate_profile_contract(contract, context)
        assert "provenance" in [entry["field"] for entry in exc_info.value.divergent_fields]

    def test_legacy_local_contract_refused_for_custom_store(
        self, profile_store, tmp_path, monkeypatch
    ):
        """Custom directories are never aliased to the installed store."""
        from cli_agent_orchestrator.services import settings_service

        custom = tmp_path / "custom"
        custom.mkdir()
        _write_profile(custom, "ext")
        monkeypatch.setattr(settings_service, "get_extra_agent_dirs", lambda: [str(custom)])
        context = load_supervisor_launch_context("ext")
        assert context.provenance == "custom"
        contract = _contract_for(context)
        contract["provenance"] = "local"
        with pytest.raises(ProfileLaunchConflict) as exc_info:
            validate_profile_contract(contract, context)
        assert "provenance" in [entry["field"] for entry in exc_info.value.divergent_fields]


class TestCanonicalSourcePath:
    def test_aliased_spelling_of_same_profile_agrees(self, profile_store, tmp_path):
        """A symlinked spelling of the same physical file validates.

        The conductor may preflight through an aliased path while the
        runtime resolves the canonical one; with matching bytes the
        contract must agree rather than diverge forever.
        """
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        alias_dir = tmp_path / "alias-store"
        alias_dir.symlink_to(profile_store, target_is_directory=True)
        contract = _contract_for(context)
        contract["source_path"] = str(alias_dir / "sup.md")
        validate_profile_contract(contract, context)

    def test_distinct_canonical_path_diverges_despite_matching_sha(self, profile_store, tmp_path):
        """Same bytes at a genuinely different path still conflict.

        Path identity is not advisory and sha alone is not identity: the
        receipt must name the source the runtime actually loaded.
        """
        path = _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "sup.md").write_bytes(path.read_bytes())
        contract = _contract_for(context)
        contract["source_path"] = str(elsewhere / "sup.md")
        with pytest.raises(ProfileLaunchConflict) as exc_info:
            validate_profile_contract(contract, context)
        assert "source_path" in [entry["field"] for entry in exc_info.value.divergent_fields]

    def test_builtin_pseudo_path_compares_exactly(self, profile_store):
        context = load_supervisor_launch_context("developer")
        assert context.source_path == "built-in:developer.md"
        validate_profile_contract(_contract_for(context), context)
        contract = _contract_for(context)
        contract["source_path"] = "built-in:other.md"
        with pytest.raises(ProfileLaunchConflict):
            validate_profile_contract(contract, context)


class TestProviderStoreProvenance:
    def test_provider_and_custom_store_branches(self, tmp_path, monkeypatch):
        """Each on-disk store branch reports its documented provenance."""
        from cli_agent_orchestrator.services import settings_service

        installed = tmp_path / "agent-context"
        kiro = tmp_path / "kiro-agents"
        custom = tmp_path / "custom"
        for directory in (installed, kiro, custom):
            directory.mkdir()
        _write_profile(installed, "sup")
        nested = kiro / "sup"
        nested.mkdir()
        (nested / "agent.md").write_text(
            "---\nname: sup\ndescription: nested\nprovider: mock_cli\n---\nbody\n",
            encoding="utf-8",
        )
        _write_profile(custom, "sup")

        from cli_agent_orchestrator.utils import agent_profiles as ap

        monkeypatch.setattr(ap, "LOCAL_AGENT_STORE_DIR", tmp_path / "empty-store")
        monkeypatch.setattr(
            settings_service,
            "get_agent_dirs",
            lambda: {"cao_installed": str(installed), "kiro_cli": str(kiro)},
        )
        monkeypatch.setattr(settings_service, "get_extra_agent_dirs", lambda: [str(custom)])
        monkeypatch.setattr(settings_service, "get_disabled_agent_dirs", lambda: [])

        assert load_supervisor_launch_context("sup").provenance == "installed"
        (installed / "sup.md").unlink()
        assert load_supervisor_launch_context("sup").provenance == "kiro"
        (nested / "agent.md").unlink()
        assert load_supervisor_launch_context("sup").provenance == "custom"


class TestSessionWiring:
    @pytest.mark.asyncio
    async def test_context_threaded_and_no_second_read(self, profile_store):
        """Real boundary load; everything below consumes it without reloading."""
        # kimi_cli is sealed-capable, so the frozen context threads through.
        _write_profile(profile_store, "sup", provider="kimi_cli")
        with (
            patch.object(
                agent_profiles, "load_agent_profile", side_effect=AssertionError("reload")
            ),
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-sup")
            await session_service.create_session(provider=None, agent_profile="sup")

        kwargs = mock_create.call_args.kwargs
        assert kwargs["provider"] == "kimi_cli"
        assert kwargs["expected_model"] == "test-model-1"
        assert kwargs["expected_effort"] is None
        context = kwargs["profile_launch_context"]
        assert isinstance(context, spr.ProfileLaunchContext)
        assert context.profile.system_prompt == "Do supervision."
        # The same frozen material the gate decided on threads through.
        material = kwargs["sealed_launch_material"]
        assert isinstance(material, SealedLaunchMaterial)
        assert material.profile is context.profile
        assert material.model == context.model
        assert material.effort == context.effort
        assert material.system_prompt == "Do supervision."
        assert material.skill_text == build_skill_catalog(None)
        assert material.allowed_tools == ("@cao-mcp-server", "fs_read", "fs_list")

    @pytest.mark.asyncio
    async def test_conflict_has_zero_effects(self, profile_store):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        contract["sha256"] = "f" * 64  # bytes changed after the preflight
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(ProfileLaunchConflict):
                await session_service.create_session(
                    provider=None, agent_profile="sup", profile_contract=contract
                )
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_contract_refused_before_effects(self, profile_store):
        _write_profile(profile_store, "sup")
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(ValueError):
                await session_service.create_session(
                    provider=None,
                    agent_profile="sup",
                    profile_contract={"schema": "nope"},
                )
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_with_fresh_contract_launches(self, profile_store):
        _write_profile(profile_store, "sup", provider="kimi_cli", model="model-one")
        stale = _contract_for(load_supervisor_launch_context("sup"))
        # The profile bytes change between preflight and retry...
        _write_profile(profile_store, "sup", provider="kimi_cli", model="model-two")
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-sup")
            with pytest.raises(ProfileLaunchConflict):
                await session_service.create_session(
                    provider=None, agent_profile="sup", profile_contract=stale
                )
            fresh = _contract_for(load_supervisor_launch_context("sup"))
            await session_service.create_session(
                provider=None, agent_profile="sup", profile_contract=fresh
            )
        assert mock_create.call_count == 1
        assert mock_create.call_args.kwargs["expected_model"] == "model-two"


class TestSealedRefusal:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["opencode_cli", "copilot_cli", "kiro_cli"])
    async def test_sealed_contract_on_named_agent_adapter_refuses_with_zero_effects(
        self, profile_store, provider
    ):
        """A sealed contract on OpenCode/Copilot/Kiro refuses pre-effect.

        The runtime would otherwise validate/persist CAO profile A while
        the supervisor consumes the mutable provider-native --agent <name>
        (native profile B). create_terminal owns every effect (window, DB
        row, provider), so never calling it proves zero effects.
        """
        _write_profile(profile_store, "sup", provider=provider)
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported) as exc_info:
                await session_service.create_session(
                    provider=None, agent_profile="sup", profile_contract=contract
                )
        mock_create.assert_not_called()
        assert exc_info.value.provider == provider
        assert exc_info.value.source_path == context.source_path
        assert exc_info.value.reason
        assert exc_info.value.recovery

    @pytest.mark.asyncio
    async def test_claude_native_wrapper_refuses_even_with_native_profile(self, profile_store):
        """native_agent B is named in the refusal; nothing native is invoked."""
        _write_profile(
            profile_store,
            "sup",
            provider="claude_code",
            body="supervise",
        )
        path = profile_store / "sup.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("role: supervisor", "role: supervisor\nnative_agent: B"))
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported) as exc_info:
                await session_service.create_session(
                    provider=None, agent_profile="sup", profile_contract=contract
                )
        mock_create.assert_not_called()
        assert "B" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_hermes_wrapper_refuses_even_with_native_profile(self, profile_store):
        _write_profile(profile_store, "sup", provider="hermes")
        path = profile_store / "sup.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("role: supervisor", "role: supervisor\nhermesProfile: B"))
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported) as exc_info:
                await session_service.create_session(
                    provider=None, agent_profile="sup", profile_contract=contract
                )
        mock_create.assert_not_called()
        assert "B" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_codex_named_profile_refuses_with_no_row(self, profile_store, isolated_memory_db):
        """codexProfile B names the mutable native [profiles.B] block.

        The sealed gate refuses before create_terminal owns any effect;
        the rigged row write is a backstop (reaching it raises), and the
        terminal table stays empty so no receipt exists to recover.
        """
        from cli_agent_orchestrator.clients.database import SessionLocal, TerminalModel
        from cli_agent_orchestrator.services import terminal_service

        _write_profile(profile_store, "sup", provider="codex")
        path = profile_store / "sup.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("role: supervisor", "role: supervisor\ncodexProfile: B"))
        context = load_supervisor_launch_context("sup")
        with (
            patch.object(terminal_service, "db_create_terminal", side_effect=AssertionError("row")),
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported) as exc_info:
                await session_service.create_session(
                    provider=None,
                    agent_profile="sup",
                    profile_contract=_contract_for(context),
                )
        assert exc_info.value.provider == "codex"
        assert "B" in exc_info.value.reason
        assert exc_info.value.recovery
        with SessionLocal() as db:
            assert db.query(TerminalModel).count() == 0

    @pytest.mark.asyncio
    async def test_antigravity_mcp_profile_refuses_before_any_effect(
        self, profile_store, isolated_memory_db
    ):
        """Nonempty mcpServers refuse: the shared file is unwritable-sealed.

        The gate fires before create_terminal owns any effect and before
        any provider exists to merge into ~/.gemini/config/mcp_config.json;
        the terminal table stays empty so no receipt exists to recover.
        """
        from cli_agent_orchestrator.clients.database import SessionLocal, TerminalModel
        from cli_agent_orchestrator.providers.antigravity_cli import AntigravityCliProvider
        from cli_agent_orchestrator.services import terminal_service

        _write_mcp_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        assert context.provider == "antigravity_cli"
        assert context.profile.mcpServers == {"cao-mcp-server": {"command": "cao-mcp-server"}}
        with (
            patch.object(terminal_service, "db_create_terminal", side_effect=AssertionError("row")),
            patch.object(
                AntigravityCliProvider,
                "_register_mcp_servers",
                side_effect=AssertionError("mcp-write"),
            ),
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported) as exc_info:
                await session_service.create_session(
                    provider=None,
                    agent_profile="sup",
                    profile_contract=_contract_for(context),
                )
        assert exc_info.value.provider == "antigravity_cli"
        assert "mcp_config.json" in exc_info.value.reason
        assert exc_info.value.recovery
        with SessionLocal() as db:
            assert db.query(TerminalModel).count() == 0

    @pytest.mark.asyncio
    async def test_antigravity_ab_refusal_writes_no_shared_state(
        self, profile_store, tmp_path, monkeypatch
    ):
        """Composed A/B reproduction: the sealed A request refuses before
        the shared MCP file — currently holding B's content — is touched."""
        import json

        from cli_agent_orchestrator.providers.antigravity_cli import AntigravityCliProvider

        fake_home = tmp_path / "home"
        shared = fake_home / ".gemini" / "config" / "mcp_config.json"
        shared.parent.mkdir(parents=True)
        shared.write_text(
            json.dumps({"mcpServers": {"server-B": {"command": "server-B"}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(fake_home))
        _write_mcp_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch.object(AntigravityCliProvider, "_register_mcp_servers") as mock_register,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported):
                await session_service.create_session(
                    provider=None,
                    agent_profile="sup",
                    profile_contract=_contract_for(context),
                )
        mock_create.assert_not_called()
        mock_register.assert_not_called()
        assert json.loads(shared.read_text(encoding="utf-8")) == {
            "mcpServers": {"server-B": {"command": "server-B"}}
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["cursor_cli", "muse_cli", "hermes"])
    async def test_dropped_prompt_refuses_with_zero_effects(self, profile_store, provider):
        """Cursor/Muse/Hermes never receive the frozen prompt: a nonempty
        system prompt refuses pre-effect even under a wildcard policy with
        no skills — the receipt must not claim a role the process lacks."""
        _write_profile(
            profile_store,
            "sup",
            provider=provider,
            extra=('allowedTools: ["*"]', "skills: []"),
        )
        context = load_supervisor_launch_context("sup")
        assert context.profile.system_prompt == "Do supervision."
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported) as exc_info:
                await session_service.create_session(
                    provider=None, agent_profile="sup", profile_contract=_contract_for(context)
                )
        mock_create.assert_not_called()
        assert exc_info.value.provider == provider
        assert "system_prompt" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_hermes_model_only_content_free_launch_passes_gate(self, profile_store):
        """The Hermes default path stays sealed-capable when empty: model
        only, wildcard policy, no skills, no prompt — the gate threads the
        material instead of refusing."""
        _write_profile(
            profile_store,
            "sup",
            provider="hermes",
            body="",
            extra=('allowedTools: ["*"]', "skills: []"),
        )
        context = load_supervisor_launch_context("sup")
        assert context.profile.system_prompt == ""
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-hermes-empty")
            await session_service.create_session(
                provider=None, agent_profile="sup", profile_contract=_contract_for(context)
            )
        kwargs = mock_create.call_args.kwargs
        material = kwargs["sealed_launch_material"]
        # Intra-launch identity: the threaded context and material share the
        # one frozen profile object (a second test-side load would compare
        # equal but never identical — reads are per-load snapshots).
        assert kwargs["profile_launch_context"].profile is material.profile
        assert material.profile == context.profile
        assert material.system_prompt == ""
        assert material.allowed_tools == ("*",)

    @pytest.mark.asyncio
    async def test_cursor_mcp_only_launch_passes_gate(self, profile_store):
        """Cursor stays sealed-capable for model plus per-launch MCP with
        every dropped field empty: the gate threads the material."""
        _write_bytes_profile(
            profile_store,
            "sup",
            b"---\nname: sup\ndescription: x\nprovider: cursor_cli\n"
            b"role: supervisor\nmodel: test-model-1\n"
            b"mcpServers:\n  frozen-srv:\n    command: frozen-srv\n"
            b'allowedTools: ["*"]\nskills: []\n---\n',
        )
        context = load_supervisor_launch_context("sup")
        assert context.profile.system_prompt == ""
        assert context.profile.mcpServers == {"frozen-srv": {"command": "frozen-srv"}}
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-cursor-mcp")
            await session_service.create_session(
                provider=None, agent_profile="sup", profile_contract=_contract_for(context)
            )
        kwargs = mock_create.call_args.kwargs
        material = kwargs["sealed_launch_material"]
        assert kwargs["profile_launch_context"].profile is material.profile
        assert material.profile == context.profile
        assert material.profile.mcpServers == {"frozen-srv": {"command": "frozen-srv"}}

    @pytest.mark.asyncio
    async def test_material_frozen_across_store_mutation(self, profile_store):
        """Mutating the store after the context read cannot move the gate.

        The material holds the frozen object and decided values; with every
        content read rigged to raise, the gate still decides identically —
        it performs no IO of its own.
        """
        from cli_agent_orchestrator.providers.manager import ProviderManager

        _write_profile(profile_store, "sup", provider="kimi_cli", model="model-A", body="PROMPT-A")
        context = load_supervisor_launch_context("sup")
        material = spr.build_sealed_launch_material(context)
        assert material.profile is context.profile
        assert material.model == "model-A"
        assert material.system_prompt == "PROMPT-A"
        before = ProviderManager().sealed_launch_support("kimi_cli", material)
        assert before.supported is True

        _write_profile(profile_store, "sup", provider="kimi_cli", model="model-B", body="PROMPT-B")
        with patch.object(Path, "read_bytes", side_effect=AssertionError("reread")):
            after = ProviderManager().sealed_launch_support("kimi_cli", material)
        assert after == before
        assert material.model == "model-A"
        assert material.system_prompt == "PROMPT-A"

    @pytest.mark.asyncio
    async def test_material_mcp_and_policy_frozen_across_store_mutation(self, profile_store):
        """MCP servers and effective policy freeze with the material: a
        store rewrite between admission and launch cannot move them."""
        from cli_agent_orchestrator.providers.manager import ProviderManager

        _write_bytes_profile(
            profile_store,
            "sup",
            b"---\nname: sup\ndescription: x\nprovider: kimi_cli\n"
            b"role: supervisor\nmodel: mcp-model-a\nmcpServers:\n"
            b'  server-a:\n    command: server-a\nallowedTools: ["fs_read"]\n'
            b"---\nDo A.\n",
        )
        context = load_supervisor_launch_context("sup")
        material = spr.build_sealed_launch_material(context)
        assert material.profile.mcpServers == {"server-a": {"command": "server-a"}}
        # Effective policy: explicit tools plus the MCP server grant.
        assert material.allowed_tools == ("fs_read", "@server-a")
        before = ProviderManager().sealed_launch_support("kimi_cli", material)
        assert before.supported is True

        _write_bytes_profile(
            profile_store,
            "sup",
            b"---\nname: sup\ndescription: x\nprovider: kimi_cli\n"
            b"role: supervisor\nmodel: mcp-model-b\nmcpServers:\n"
            b'  server-b:\n    command: server-b\nallowedTools: ["*"]\n'
            b"---\nDo B.\n",
        )
        after = ProviderManager().sealed_launch_support("kimi_cli", material)
        assert after == before
        assert material.model == "mcp-model-a"
        assert material.system_prompt == "Do A."
        assert material.profile.mcpServers == {"server-a": {"command": "server-a"}}
        assert material.allowed_tools == ("fs_read", "@server-a")

    @pytest.mark.asyncio
    async def test_legacy_no_contract_unsupported_still_launches_without_receipt_claim(
        self, profile_store
    ):
        """No contract on an unsupported adapter: ordinary legacy launch.

        The frozen context is not threaded and no exact receipt is
        claimed — the response projects null/unknown, not profile A.
        """
        _write_profile(profile_store, "sup", provider="kiro_cli")
        with (
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-sup")
            await session_service.create_session(provider=None, agent_profile="sup")
        kwargs = mock_create.call_args.kwargs
        assert "profile_launch_context" not in kwargs
        assert "sealed_launch_material" not in kwargs
        assert "expected_model" not in kwargs
        assert "expected_effort" not in kwargs

    @pytest.mark.asyncio
    async def test_unsupported_refusal_writes_no_row(self, profile_store, isolated_memory_db):
        """End-to-end refusal through real create_terminal mock boundary.

        With create_terminal real but its effectful collaborators absent,
        the refusal must fire before the terminal row exists. The DB stays
        empty: GET recovery can only return stored receipts, and there is
        none to invent.
        """
        from cli_agent_orchestrator.clients.database import SessionLocal, TerminalModel
        from cli_agent_orchestrator.services import terminal_service

        _write_profile(profile_store, "sup", provider="kiro_cli")
        context = load_supervisor_launch_context("sup")
        with (
            patch.object(terminal_service, "db_create_terminal", side_effect=AssertionError("row")),
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            with pytest.raises(spr.ProfileLaunchUnsupported):
                await session_service.create_session(
                    provider=None,
                    agent_profile="sup",
                    profile_contract=_contract_for(context),
                )
        with SessionLocal() as db:
            assert db.query(TerminalModel).count() == 0


class TestTerminalReceiptWiring:
    @pytest.mark.asyncio
    async def test_receipt_persisted_projected_and_exact(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """Real create_terminal run: no reload, durable receipt, exact route."""
        from cli_agent_orchestrator.services import terminal_projection
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        service_provider_manager = MagicMock()
        with (
            _HermeticLaunch(
                provider_manager=service_provider_manager,
                launched_provider=launched_provider,
                terminal_id="abcd1234",
                session="cao-sup",
            ),
            # The launch-boundary read already happened: any by-name reload
            # below (Step 3, bootstrap, adapter) fails the test outright.
            patch(
                f"{_SERVICE}.load_agent_profile",
                side_effect=AssertionError("profile reloaded by name"),
            ),
        ):
            terminal = await create_terminal(
                provider="mock_cli",
                agent_profile="sup",
                new_session=True,
                profile_launch_context=context,
                expected_model=context.model,
                expected_effort=context.effort,
            )

        expected_receipt = build_profile_receipt(context)
        # POST /sessions shape: the same receipt rides the response, and it
        # survives the response_model round-trip the route applies.
        assert terminal.profile_receipt == expected_receipt
        from cli_agent_orchestrator.models.terminal import Terminal

        assert Terminal.model_validate(terminal.model_dump()).profile_receipt == expected_receipt
        # Provider construction consumed the exact context object.
        create_kwargs = service_provider_manager.create_provider.call_args.kwargs
        assert create_kwargs["launch_profile"] is context.profile
        assert create_kwargs["expected_model"] == "test-model-1"
        assert create_kwargs["expected_effort"] is None
        # Durable row carries the receipt...
        stored = database.get_terminal_metadata("abcd1234")
        assert stored is not None
        assert stored["profile_receipt"] == expected_receipt
        # ...and every read surface agrees: session listing, single
        # terminal projection, and the legacy get_terminal fallback.
        from cli_agent_orchestrator.services.terminal_service import get_terminal

        with (
            patch("cli_agent_orchestrator.services.session_service.get_backend") as mock_backend,
            patch(
                "cli_agent_orchestrator.services.terminal_projection.get_backend",
                return_value=_hermetic_backend(),
            ),
        ):
            mock_backend.return_value.session_exists.return_value = True
            mock_backend.return_value.list_sessions.return_value = [{"id": "cao-sup"}]
            session_view = session_service.get_session("cao-sup")
            assert session_view["terminals"][0]["profile_receipt"] == expected_receipt
            assert (
                terminal_projection.project_terminal("abcd1234")["profile_receipt"]
                == expected_receipt
            )
        with patch(f"{_SERVICE}.status_monitor") as mock_status:
            mock_status.get_status.return_value = MagicMock(value="idle")
            assert get_terminal("abcd1234")["profile_receipt"] == expected_receipt

    @pytest.mark.asyncio
    async def test_post_launch_drift_does_not_change_reads(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        from cli_agent_orchestrator.services import terminal_projection
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup", model="launch-model")
        context = load_supervisor_launch_context("sup")
        service_provider_manager = MagicMock()
        with _HermeticLaunch(
            provider_manager=service_provider_manager,
            launched_provider=launched_provider,
            terminal_id="abcd1235",
            session="cao-drift",
        ):
            terminal = await create_terminal(
                provider="mock_cli",
                agent_profile="sup",
                new_session=True,
                profile_launch_context=context,
            )

        launched_receipt = dict(terminal.profile_receipt)
        assert launched_receipt["model"] == "launch-model"
        # The profile changes after the launch...
        _write_profile(profile_store, "sup", model="drifted-model")
        assert load_supervisor_launch_context("sup").model == "drifted-model"
        # ...but every durable surface still reports the launch truth.
        assert database.get_terminal_metadata("abcd1235")["profile_receipt"] == launched_receipt
        with patch(
            "cli_agent_orchestrator.services.terminal_projection.get_backend",
            return_value=_hermetic_backend(),
        ):
            assert (
                terminal_projection.project_terminal("abcd1235")["profile_receipt"]
                == launched_receipt
            )

    @pytest.mark.asyncio
    async def test_legacy_row_stays_missing(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """A launch without a context persists no receipt: reads stay absent."""
        from cli_agent_orchestrator.services import terminal_projection
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup")
        service_provider_manager = MagicMock()
        with _HermeticLaunch(
            provider_manager=service_provider_manager,
            launched_provider=launched_provider,
            terminal_id="abcd1236",
            session="cao-legacy",
        ):
            terminal = await create_terminal(
                provider="mock_cli",
                agent_profile="sup",
                new_session=True,
            )

        assert terminal.profile_receipt is None
        assert database.get_terminal_metadata("abcd1236")["profile_receipt"] is None
        with patch(
            "cli_agent_orchestrator.services.terminal_projection.get_backend",
            return_value=_hermetic_backend(),
        ):
            assert terminal_projection.project_terminal("abcd1236")["profile_receipt"] is None

    @pytest.mark.asyncio
    async def test_stored_legacy_receipt_echoed_unchanged(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """A row written by an older runtime keeps its receipt verbatim.

        The stored "local"-provenance receipt is echoed exactly — never
        rewritten to the canonical label, never re-derived from the live
        profile — so pre-upgrade launches stay auditable as recorded.
        """
        import json

        from cli_agent_orchestrator.clients.database import SessionLocal, TerminalModel
        from cli_agent_orchestrator.services import terminal_projection
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        service_provider_manager = MagicMock()
        with _HermeticLaunch(
            provider_manager=service_provider_manager,
            launched_provider=launched_provider,
            terminal_id="abcd1237",
            session="cao-legacy-echo",
        ):
            await create_terminal(
                provider="mock_cli",
                agent_profile="sup",
                new_session=True,
                profile_launch_context=context,
                expected_model=context.model,
                expected_effort=context.effort,
            )

        legacy = dict(build_profile_receipt(context))
        legacy["provenance"] = "local"
        with SessionLocal() as db:
            row = db.query(TerminalModel).filter_by(id="abcd1237").one()
            row.profile_receipt = json.dumps(legacy, sort_keys=True)
            db.commit()
        with patch(
            "cli_agent_orchestrator.services.terminal_projection.get_backend",
            return_value=_hermetic_backend(),
        ):
            projected = terminal_projection.project_terminal("abcd1237")["profile_receipt"]
        assert projected == legacy

    @pytest.mark.asyncio
    async def test_empty_antigravity_sealed_launch_persists_receipt_without_shared_write(
        self, profile_store, isolated_memory_db, launched_provider, tmp_path, monkeypatch
    ):
        """An MCP-free Antigravity profile stays sealed-capable end to end.

        The ambient shared MCP file is present (another launch's content),
        yet the sealed launch persists an exact receipt carrying no MCP
        material and leaves the shared file untouched: ambient Antigravity
        configuration is outside the CAO receipt, by explicit decision.
        """
        import json

        from cli_agent_orchestrator.services import native_attachment, unmanaged_native_identity
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        fake_home = tmp_path / "home"
        shared = fake_home / ".gemini" / "config" / "mcp_config.json"
        shared.parent.mkdir(parents=True)
        shared.write_text(
            json.dumps({"mcpServers": {"ambient": {"command": "ambient"}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(fake_home))
        _write_profile(profile_store, "sup", provider="antigravity_cli")
        context = load_supervisor_launch_context("sup")
        assert context.profile.mcpServers is None
        validate_profile_contract(_contract_for(context), context)

        def _frozen_agy_identity(**kwargs):
            # The bootstrap consumes the frozen snapshot, never the store.
            assert kwargs["launch_profile"] is context.profile
            assert kwargs["expected_model"] == "test-model-1"
            return {
                "native_session_id": "agy-frozen-native",
                "acquisition_method": (native_attachment.ACQUISITION_ZERO_TURN_BOOTSTRAP),
                "model": "test-model-1",
                "effort": None,
                "binary_path": None,
            }

        service_provider_manager = MagicMock()
        with (
            _HermeticLaunch(
                provider_manager=service_provider_manager,
                launched_provider=launched_provider,
                terminal_id="abcd1239",
                session="cao-agy-empty",
            ),
            patch.object(
                unmanaged_native_identity,
                "resolve_pre_task_identity",
                side_effect=_frozen_agy_identity,
            ),
        ):
            terminal = await create_terminal(
                provider="antigravity_cli",
                agent_profile="sup",
                new_session=True,
                profile_launch_context=context,
                expected_model=context.model,
                expected_effort=context.effort,
            )
        assert terminal.profile_receipt == build_profile_receipt(context)
        assert "mcpServers" not in terminal.profile_receipt
        assert json.loads(shared.read_text(encoding="utf-8")) == {
            "mcpServers": {"ambient": {"command": "ambient"}}
        }

    @pytest.mark.asyncio
    async def test_persistence_failure_leaves_no_successful_launch(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """A row write that cannot persist the receipt fails the launch.

        The provider is never constructed or initialized afterwards: there
        is no successful launch without its durable receipt, and no live
        supervisor without a row.
        """
        from cli_agent_orchestrator.services import terminal_service
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        service_provider_manager = MagicMock()
        with (
            _HermeticLaunch(
                provider_manager=service_provider_manager,
                launched_provider=launched_provider,
                terminal_id="abcd1237",
                session="cao-noreceipt",
            ),
            # terminal_service holds its own reference to the db writer.
            patch.object(
                terminal_service,
                "db_create_terminal",
                side_effect=RuntimeError("store down"),
            ),
        ):
            with pytest.raises(RuntimeError, match="store down"):
                await create_terminal(
                    provider="mock_cli",
                    agent_profile="sup",
                    new_session=True,
                    profile_launch_context=context,
                )
        assert database.get_terminal_metadata("abcd1237", warn_if_missing=False) is None
        service_provider_manager.create_provider.assert_not_called()
        launched_provider.initialize.assert_not_called()

    def test_migration_adds_nullable_receipt_to_legacy_table(self, tmp_path):
        """A pre-receipt terminals table gains the column; old rows read NULL."""
        import sqlite3

        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE terminals ("
            "id TEXT PRIMARY KEY, tmux_session TEXT, tmux_window TEXT, "
            "provider TEXT, agent_profile TEXT)"
        )
        conn.execute(
            "INSERT INTO terminals (id, tmux_session, tmux_window, provider, agent_profile)"
            " VALUES ('deadbeef', 'cao-old', 'w', 'mock_cli', 'sup')"
        )
        conn.commit()
        conn.close()

        from cli_agent_orchestrator import constants

        with patch.object(constants, "DATABASE_FILE", db_path):
            database._migrate_terminals_schema()

        conn = sqlite3.connect(str(db_path))
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(terminals)")}
            assert "profile_receipt" in columns
            row = conn.execute(
                "SELECT profile_receipt FROM terminals WHERE id = 'deadbeef'"
            ).fetchone()
            assert row[0] is None
        finally:
            conn.close()


class TestComposedFrozenContextWiring:
    @pytest.mark.asyncio
    async def test_same_context_drives_contract_receipt_response_and_argv(
        self, profile_store, isolated_memory_db
    ):
        """One frozen context end to end, with a real adapter underneath.

        The context is loaded once from the store; the contract is built
        from that live context (never a fixture-encoded receipt); the
        launch runs through the real ``ProviderManager`` into a real Kimi
        adapter whose ``initialize`` is stubbed but whose argv builder runs
        for real — with every by-name store load rigged to raise. If any
        downstream stage ignores or reloads the frozen context instead of
        consuming it, this fails: at Step 3, in the bootstrap, or in the
        argv builder.
        """
        from cli_agent_orchestrator.models.terminal import Terminal
        from cli_agent_orchestrator.providers import kimi_cli
        from cli_agent_orchestrator.providers.manager import ProviderManager
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup", provider="kimi_cli", model="composed-model-7")
        context = load_supervisor_launch_context("sup")
        assert context.provider == "kimi_cli"

        # The contract the conductor would preflight, derived from the live
        # context rather than encoded expectations.
        validate_profile_contract(_contract_for(context), context)

        real_manager = ProviderManager()
        built = []
        real_create = real_manager.create_provider

        def _capture(*args, **kwargs):
            instance = real_create(*args, **kwargs)
            built.append((instance, kwargs))
            return instance

        real_manager.create_provider = MagicMock(side_effect=_capture)
        with (
            _HermeticLaunch(
                provider_manager=real_manager,
                terminal_id="abcd1238",
                session="cao-composed",
            ),
            patch.object(
                kimi_cli.KimiCliProvider,
                "initialize",
                new=AsyncMock(return_value=True),
            ),
            patch(
                f"{_SERVICE}.load_agent_profile",
                side_effect=AssertionError("terminal reloaded by name"),
            ),
            patch.object(
                kimi_cli, "load_agent_profile", side_effect=AssertionError("adapter reloaded")
            ),
        ):
            terminal = await create_terminal(
                provider="kimi_cli",
                agent_profile="sup",
                new_session=True,
                profile_launch_context=context,
                expected_model=context.model,
                expected_effort=context.effort,
            )

        # The receipt is authored from the live context, not a fixture.
        receipt = build_profile_receipt(context)
        assert receipt["model"] == "composed-model-7"
        assert terminal.profile_receipt == receipt
        assert Terminal.model_validate(terminal.model_dump()).profile_receipt == receipt
        assert database.get_terminal_metadata("abcd1238")["profile_receipt"] == receipt

        # Provider construction consumed the exact frozen object...
        assert len(built) == 1
        instance, create_kwargs = built[0]
        assert create_kwargs["launch_profile"] is context.profile
        assert create_kwargs["expected_model"] == "composed-model-7"
        # ...and the real argv builder renders the frozen route with the
        # store loader still rigged to raise.
        assert instance._launch_profile is context.profile
        with patch.object(
            kimi_cli, "load_agent_profile", side_effect=AssertionError("argv reloaded")
        ):
            command = instance._build_kimi_command()
        assert "--model" in command and "composed-model-7" in command

    @pytest.mark.asyncio
    async def test_sealed_codex_composes_skill_and_policy_exactly_once(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """Gate, prepare, bootstrap, and TUI share one skill/policy composition.

        The catalog builder answers CATALOG-A on its first call and raises
        on any second: with the frozen skill threaded through, preparation
        and the bridge builder never rescan, so the bootstrap observes
        CATALOG-A and the launch performs exactly one skill composition
        and one policy resolution. A rebuild (second call) errors the
        launch instead of silently swapping in CATALOG-B.
        """
        import cli_agent_orchestrator.services.managed_provider_bridge as bridge
        from cli_agent_orchestrator.providers.manager import ProviderManager
        from cli_agent_orchestrator.services import terminal_service, unmanaged_native_identity
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        _write_profile(profile_store, "sup", provider="codex", model="once-model-9")
        context = load_supervisor_launch_context("sup")
        validate_profile_contract(_contract_for(context), context)

        catalog_calls = []
        policy_calls = []
        real_resolve = spr.resolve_allowed_tools

        def _catalog_once(*args, **kwargs):
            catalog_calls.append(1)
            if len(catalog_calls) > 1:
                raise AssertionError("skill catalog recomposed after the gate")
            return "CATALOG-A"

        def _policy_once(*args, **kwargs):
            policy_calls.append(1)
            return real_resolve(*args, **kwargs)

        seen = {}

        def _capture_bootstrap(**kwargs):
            from cli_agent_orchestrator.services import native_attachment

            seen.update(kwargs)
            assert kwargs["launch_profile"] is context.profile
            composed = kwargs["codex_profile_material"]["system_prompt"]
            assert "CATALOG-A" in composed
            return {
                "native_session_id": "codex-once-native",
                "acquisition_method": (native_attachment.ACQUISITION_ZERO_TURN_BOOTSTRAP),
                "working_directory": "/tmp",
                "model": "once-model-9",
                "effort": None,
                "binary_path": None,
            }

        service_provider_manager = MagicMock()
        with (
            _HermeticLaunch(
                provider_manager=service_provider_manager,
                launched_provider=launched_provider,
                terminal_id="abcd1244",
                session="cao-once-codex",
            ),
            patch.object(spr, "build_skill_catalog", side_effect=_catalog_once),
            patch.object(terminal_service, "build_skill_catalog", side_effect=_catalog_once),
            patch.object(bridge, "build_skill_catalog", side_effect=_catalog_once),
            patch.object(spr, "resolve_allowed_tools", side_effect=_policy_once),
            patch.object(
                unmanaged_native_identity,
                "resolve_pre_task_identity",
                side_effect=_capture_bootstrap,
            ),
            patch(
                f"{_SERVICE}.load_agent_profile",
                side_effect=AssertionError("terminal reloaded by name"),
            ),
        ):
            # The gate-equivalent composition: the single counted build.
            material = spr.build_sealed_launch_material(context)
            assert material.skill_text == "CATALOG-A"
            # The session-boundary preparation: validates and serializes
            # without recomposing (a second catalog/policy build would
            # raise through the rigged builders).
            prepared = ProviderManager().prepare_sealed_launch("codex", material)
            assert prepared.codex_material["system_prompt"].count("CATALOG-A") == 1
            terminal = await create_terminal(
                provider="codex",
                agent_profile="sup",
                new_session=True,
                profile_launch_context=context,
                sealed_launch_material=material,
                prepared_sealed_launch=prepared,
                expected_model=context.model,
                expected_effort=context.effort,
            )
        assert len(catalog_calls) == 1
        assert len(policy_calls) == 1
        assert terminal.profile_receipt == build_profile_receipt(context)


class TestFrozenSurvivesStoreMutation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider,terminal_id,session,effort",
        [
            ("kimi_cli", "abcd1241", "cao-frozen-kimi", None),
            ("codex", "abcd1242", "cao-frozen-codex", "xhigh"),
            ("claude_code", "abcd1243", "cao-frozen-claude", None),
        ],
    )
    async def test_supported_adapter_launches_frozen_a_after_mutation_to_b(
        self, profile_store, isolated_memory_db, provider, terminal_id, session, effort
    ):
        """The store mutates A->B between the frozen read and provider build.

        Every launch input the adapter consumes — model, prompt/config,
        effort — must still be A. Any reload observes B (or raises through
        the rigged loader); only the frozen context yields A.
        """
        import shlex

        from cli_agent_orchestrator.providers import claude_code, codex, kimi_cli
        from cli_agent_orchestrator.providers.manager import ProviderManager
        from cli_agent_orchestrator.services.terminal_service import create_terminal

        adapter_modules = {
            "kimi_cli": kimi_cli,
            "codex": codex,
            "claude_code": claude_code,
        }
        adapter_classes = {
            "kimi_cli": kimi_cli.KimiCliProvider,
            "codex": codex.CodexProvider,
            "claude_code": claude_code.ClaudeCodeProvider,
        }
        module = adapter_modules[provider]

        _write_profile(
            profile_store,
            "sup",
            provider=provider,
            model="frozen-model-A",
            effort=effort,
            body="FROZEN-PROMPT-A",
        )
        context = load_supervisor_launch_context("sup")
        assert context.model == "frozen-model-A"
        # Drift between the frozen read and provider construction.
        _write_profile(
            profile_store,
            "sup",
            provider=provider,
            model="mutated-model-B",
            effort="low" if effort else None,
            body="MUTATED-PROMPT-B",
        )

        real_manager = ProviderManager()
        built = []
        real_create = real_manager.create_provider

        def _capture(*args, **kwargs):
            instance = real_create(*args, **kwargs)
            built.append((instance, kwargs))
            return instance

        real_manager.create_provider = MagicMock(side_effect=_capture)
        stack = ExitStack()
        stack.enter_context(
            _HermeticLaunch(
                provider_manager=real_manager,
                terminal_id=terminal_id,
                session=session,
            )
        )
        stack.enter_context(
            patch.object(adapter_classes[provider], "initialize", new=AsyncMock(return_value=True))
        )
        stack.enter_context(
            patch(
                f"{_SERVICE}.load_agent_profile",
                side_effect=AssertionError("terminal reloaded by name"),
            )
        )
        stack.enter_context(
            patch.object(
                module, "load_agent_profile", side_effect=AssertionError("adapter reloaded")
            )
        )
        if provider == "codex":
            # The codex pre-task bootstrap needs its native binary; stub
            # only that exchange with the frozen route. Everything else —
            # material composition, adapter construction, argv build —
            # still runs for real against the frozen context.
            from cli_agent_orchestrator.services import native_attachment, unmanaged_native_identity

            def _frozen_codex_identity(**kwargs):
                assert kwargs["launch_profile"] is context.profile
                assert kwargs["expected_model"] == "frozen-model-A"
                assert kwargs["expected_effort"] == "xhigh"
                return {
                    "native_session_id": "codex-frozen-native",
                    "acquisition_method": (native_attachment.ACQUISITION_ZERO_TURN_BOOTSTRAP),
                    "model": "frozen-model-A",
                    "effort": "xhigh",
                    "binary_path": None,
                }

            stack.enter_context(
                patch.object(
                    unmanaged_native_identity,
                    "resolve_pre_task_identity",
                    side_effect=_frozen_codex_identity,
                )
            )
        with stack:
            terminal = await create_terminal(
                provider=provider,
                agent_profile="sup",
                new_session=True,
                profile_launch_context=context,
                expected_model=context.model,
                expected_effort=context.effort,
            )

        receipt = build_profile_receipt(context)
        assert receipt["model"] == "frozen-model-A"
        assert terminal.profile_receipt == receipt
        assert database.get_terminal_metadata(terminal_id)["profile_receipt"] == receipt

        assert len(built) == 1
        instance, create_kwargs = built[0]
        assert create_kwargs["launch_profile"] is context.profile
        assert create_kwargs["expected_model"] == "frozen-model-A"
        if effort:
            assert create_kwargs["expected_effort"] == effort

        if provider == "kimi_cli":
            command = instance._build_kimi_command()
            assert "frozen-model-A" in command
            assert "mutated-model-B" not in command
            prompt_file = Path(instance._temp_dir) / "system.md"
            content = prompt_file.read_text(encoding="utf-8")
            assert "FROZEN-PROMPT-A" in content
            assert "MUTATED-PROMPT-B" not in content
        elif provider == "codex":
            material = instance._resolve_codex_profile_material()
            assert material["profile"] is context.profile
            assert "FROZEN-PROMPT-A" in material["system_prompt"]
            assert "MUTATED-PROMPT-B" not in material["system_prompt"]
        elif provider == "claude_code":
            command = instance._build_claude_command()
            assert "frozen-model-A" in command
            assert "mutated-model-B" not in command
            parts = shlex.split(command)
            prompt_path = parts[parts.index("--append-system-prompt-file") + 1]
            content = Path(prompt_path).read_text(encoding="utf-8")
            assert "FROZEN-PROMPT-A" in content
            assert "MUTATED-PROMPT-B" not in content


class TestHttpMapping:
    @pytest.mark.asyncio
    async def test_conflict_maps_to_409_with_retry(self, profile_store):
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        contract["provider"] = "kimi_cli"
        with (
            patch.object(
                main.session_service,
                "create_session",
                side_effect=ProfileLaunchConflict(
                    "diverged", divergent_fields=["provider"], retry="re-preflight"
                ),
            ),
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                    profile_contract=contract,
                )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["retry"] == "re-preflight"
        assert exc_info.value.detail["divergent_fields"] == ["provider"]

    @pytest.mark.asyncio
    async def test_unsupported_contract_maps_to_422(self, profile_store):
        """A sealed refusal is an operation-scoped 422 with recovery."""
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main
        from cli_agent_orchestrator.services.supervisor_profile_receipt import (
            ProfileLaunchUnsupported,
        )

        _write_profile(profile_store, "sup", provider="kiro_cli")
        context = load_supervisor_launch_context("sup")
        with (
            patch.object(
                main.session_service,
                "create_session",
                side_effect=ProfileLaunchUnsupported(
                    "provider 'kiro_cli' cannot launch exactly",
                    provider="kiro_cli",
                    source_path=context.source_path,
                    reason="Kiro launches kiro --agent <name>",
                    recovery="use a sealed-capable provider or drop the contract",
                ),
            ),
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                    profile_contract=_contract_for(context),
                )
        assert exc_info.value.status_code == 422
        detail = exc_info.value.detail
        assert detail["provider"] == "kiro_cli"
        assert detail["source_path"] == context.source_path
        assert "Kiro launches" in detail["reason"]
        assert detail["recovery"]

    @pytest.mark.asyncio
    async def test_codex_named_profile_refusal_maps_to_422(self, profile_store):
        """The codex codexProfile refusal reaches the operator as HTTP 422.

        Runs the real service (not a mocked side effect) through the real
        endpoint mapping: the gate refuses, the endpoint reports provider,
        source, reason, and recovery — with no launch behind it.
        """
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup", provider="codex")
        path = profile_store / "sup.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("role: supervisor", "role: supervisor\ncodexProfile: B"))
        context = load_supervisor_launch_context("sup")
        with patch.object(main, "get_plugin_registry", return_value=MagicMock()):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                    profile_contract=_contract_for(context),
                )
        assert exc_info.value.status_code == 422
        detail = exc_info.value.detail
        assert detail["provider"] == "codex"
        assert detail["source_path"] == context.source_path
        assert "B" in detail["reason"]
        assert detail["recovery"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", [["not", "an", "object"], "a-string", {"unexpected": "shape"}])
    async def test_non_mapping_and_extra_field_contracts_map_to_400(self, profile_store, raw):
        """A JSON list/string (or unknown shape) reaches the endpoint as
        ``Optional[Any]`` instead of dying in FastAPI's generic 422: the
        strict parser classifies it as a typed 400."""
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup")
        with patch.object(main, "get_plugin_registry", return_value=MagicMock()):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                    profile_contract=raw,
                )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_uppercase_sha_contract_passes_boundary_lowercase(self, profile_store):
        """An uppercase digest is accepted, normalized for comparison, and
        the launch proceeds on the lowercase receipt digest."""
        from fastapi import BackgroundTasks

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup", provider="kimi_cli")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        contract["sha256"] = contract["sha256"].upper()
        with (
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
            patch(
                "cli_agent_orchestrator.services.session_service.create_terminal",
                new=AsyncMock(),
            ) as mock_create,
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event"),
        ):
            mock_create.return_value = MagicMock(session_name="cao-upper-sha")
            await main.create_session(
                request=MagicMock(),
                background_tasks=BackgroundTasks(),
                agent_profile="sup",
                profile_contract=contract,
            )
        assert mock_create.call_args.kwargs["sealed_launch_material"].profile is (
            mock_create.call_args.kwargs["profile_launch_context"].profile
        )
        assert (
            mock_create.call_args.kwargs["profile_launch_context"].sha256
            == context.sha256
            == contract["sha256"].lower()
        )

    @pytest.mark.asyncio
    async def test_malformed_contract_maps_to_400(self, profile_store):
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup")
        with (
            patch.object(
                main.session_service,
                "create_session",
                side_effect=ValueError("profile_contract schema must be 'cao-...'"),
            ),
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                    profile_contract={"schema": "nope"},
                )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_profile_maps_to_400(self, profile_store):
        """The launch-boundary typed lookup error is a client error."""
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main
        from cli_agent_orchestrator.services.supervisor_profile_receipt import (
            ProfileNotFoundError,
        )

        with (
            patch.object(
                main.session_service,
                "create_session",
                side_effect=ProfileNotFoundError("Agent profile not found: ghost"),
            ),
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="ghost",
                )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unparseable_profile_maps_to_400(self, profile_store):
        """Present-but-unparseable content is client-fixable, not a 500."""
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main
        from cli_agent_orchestrator.services.supervisor_profile_receipt import (
            ProfileInvalidError,
        )

        with (
            patch.object(
                main.session_service,
                "create_session",
                side_effect=ProfileInvalidError("agent profile 'bad' does not parse"),
            ),
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="bad",
                )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_late_filenotfound_stays_500(self, profile_store):
        """A late tmux/FIFO/store FileNotFoundError is not a client error.

        The boundary catch is narrowed to the typed launch-context error,
        so an unrelated FileNotFoundError raised mid-launch keeps its 500
        classification instead of masquerading as a bad profile name.
        """
        from fastapi import BackgroundTasks, HTTPException

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup")
        with (
            patch.object(
                main.session_service,
                "create_session",
                side_effect=FileNotFoundError("tmux pipe gone mid-launch"),
            ),
            patch.object(main, "get_plugin_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await main.create_session(
                    request=MagicMock(),
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                )
        assert exc_info.value.status_code == 500
