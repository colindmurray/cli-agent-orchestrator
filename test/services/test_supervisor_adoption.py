"""Create-or-adopt for contract-bearing supervisor retries (cond-0817 A).

A retry that carries the same profile_contract must converge on the
live winner instead of launching twice: the first launch persists the
receipt, the retry returns the same terminal id with the exact stored
receipt, and exactly one effect sequence runs. Anything that is not
one exact live winner refuses with the typed 409 and zero effects.
"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent_orchestrator.services import session_service
from cli_agent_orchestrator.services import supervisor_profile_receipt as spr
from cli_agent_orchestrator.services.supervisor_profile_receipt import (
    PROFILE_LAUNCH_CONTRACT_SCHEMA,
    load_supervisor_launch_context,
)
from cli_agent_orchestrator.utils import agent_profiles

_SERVICE = "cli_agent_orchestrator.services.terminal_service"


def _write_profile(store: Path, name: str, *, provider: str = "kimi_cli", body="Do supervision."):
    path = store / f"{name}.md"
    path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {name} profile",
                f"provider: {provider}",
                "role: supervisor",
                "model: adopt-model-1",
                "---",
                body,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def profile_store(tmp_path, monkeypatch):
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
    """A backend mock that models reality: sessions exist once created."""
    backend = MagicMock()
    created: set = set()
    backend.session_exists.side_effect = lambda name: name in created
    backend.window_identity.return_value = {
        "pane_id": "%1",
        "window_id": "@1",
        "server_socket_path": "/tmp/tmux",
        "session_id": "$1",
        "pane_pid": 12345,
    }
    backend.supports_event_inbox.return_value = False
    backend.supports_pane_identity.return_value = False
    backend.created_sessions = created
    return backend


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


class _LaunchRig:
    """Hermetic first-creation rig with effect counters."""

    def __init__(self, launched_provider):
        self.launched_provider = launched_provider
        self.backend = _hermetic_backend()
        self.stack = ExitStack()
        self.prepare_calls: list = []
        self.events: list = []
        self.last_result = None

    def __enter__(self):
        from cli_agent_orchestrator.providers.manager import ProviderManager
        from cli_agent_orchestrator.services import terminal_service

        real_prepare = ProviderManager.prepare_sealed_launch

        def _counting_prepare(this, provider_type, material):
            self.prepare_calls.append(provider_type)
            return real_prepare(this, provider_type, material)

        real_create = terminal_service.provider_manager.create_provider

        def _counting_create(*args, **kwargs):
            self.events.append("provider")
            return self.launched_provider

        del real_create
        service_provider_manager = MagicMock()
        service_provider_manager.create_provider.side_effect = _counting_create

        def _record_create(*args, **kwargs):
            self.events.append("tmux")
            if args:
                self.backend.created_sessions.add(args[0])

        self.backend.create_session.side_effect = _record_create

        self.stack.enter_context(
            patch("cli_agent_orchestrator.backends.registry._backend", self.backend)
        )
        self.stack.enter_context(patch(f"{_SERVICE}.provider_manager", service_provider_manager))
        self.stack.enter_context(patch(f"{_SERVICE}.fifo_manager"))
        self.stack.enter_context(patch(f"{_SERVICE}.status_monitor"))
        self.stack.enter_context(patch(f"{_SERVICE}.clear_session_env"))
        # Registration runs for real: it stamps the row lifecycle live,
        # which is part of what makes the winner adoptable.
        self.stack.enter_context(patch(f"{_SERVICE}.generate_terminal_id", return_value="abcd4242"))
        self.stack.enter_context(patch(f"{_SERVICE}.generate_window_name", return_value="w-sup"))
        self.stack.enter_context(
            patch.object(ProviderManager, "prepare_sealed_launch", _counting_prepare)
        )
        self.dispatch = self.stack.enter_context(
            patch("cli_agent_orchestrator.services.session_service.dispatch_plugin_event")
        )
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    async def create(self, **kwargs):
        result = await session_service.create_session(**kwargs)
        self.last_result = result
        return result.terminal

    def row_count(self):
        from cli_agent_orchestrator.clients.database import SessionLocal, TerminalModel

        with SessionLocal() as db:
            return db.query(TerminalModel).count()

    def post_create_events(self):
        return [
            call for call in self.dispatch.call_args_list if call.args[1] == "post_create_session"
        ]


class TestAdoptRetry:
    @pytest.mark.asyncio
    async def test_retry_returns_same_terminal_and_exact_receipt(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """First launch persists; the lost-response retry adopts it.

        The retry returns the same terminal id with the exact stored
        receipt while running no second effect sequence: one tmux
        creation, one row, one preparation, one provider construction,
        one creation event.
        """
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        expected = spr.build_profile_receipt(context)

        with _LaunchRig(launched_provider) as rig:
            first = await rig.create(
                provider=None,
                agent_profile="sup",
                session_name="adopt",
                profile_contract=contract,
            )
            assert first.profile_receipt == expected
            assert rig.last_result.adopted is False
            retry = await rig.create(
                provider=None,
                agent_profile="sup",
                session_name="adopt",
                profile_contract=contract,
            )
            assert rig.last_result.adopted is True

        assert retry.id == first.id
        assert retry.profile_receipt == first.profile_receipt == expected
        assert rig.row_count() == 1
        assert rig.backend.create_session.call_count == 1
        assert rig.prepare_calls == ["kimi_cli"]
        assert len(rig.post_create_events()) == 1

    @pytest.mark.asyncio
    async def test_concurrent_callers_share_one_effect_sequence(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """Two same-candidate callers converge on one winner, one sequence.

        The claim serializes the pair: the winner prepares, effects,
        and dispatches exactly once, while the loser blocks on the
        claim and is surfaced as adopted without preparing or
        dispatching.
        """
        import asyncio

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)

        def _call():
            return session_service.create_session(
                provider=None,
                agent_profile="sup",
                session_name="adopt-race",
                profile_contract=contract,
            )

        with _LaunchRig(launched_provider) as rig:
            winner_result, loser_result = await asyncio.gather(_call(), _call())
            first, second = winner_result.terminal, loser_result.terminal

        assert first.id == second.id
        assert first.profile_receipt == second.profile_receipt
        assert rig.row_count() == 1
        assert rig.backend.create_session.call_count == 1
        assert rig.prepare_calls == ["kimi_cli"]
        assert len(rig.post_create_events()) == 1
        assert sorted([winner_result.adopted, loser_result.adopted]) == [False, True]

    @pytest.mark.asyncio
    async def test_adoption_runs_no_preparation_or_effect(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """The retry path never prepares, clears, or constructs."""
        from cli_agent_orchestrator.providers.manager import ProviderManager
        from cli_agent_orchestrator.services import terminal_service

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)

        with _LaunchRig(launched_provider) as rig:
            first = await rig.create(
                provider=None,
                agent_profile="sup",
                session_name="adopt-pure",
                profile_contract=contract,
            )
            tmux_calls = rig.backend.create_session.call_count
            with (
                patch.object(
                    ProviderManager,
                    "prepare_sealed_launch",
                    side_effect=AssertionError("prepared on adopt"),
                ),
                patch.object(
                    terminal_service,
                    "clear_session_env",
                    side_effect=AssertionError("cleared on adopt"),
                ),
                patch.object(
                    ProviderManager,
                    "create_provider",
                    side_effect=AssertionError("constructed on adopt"),
                ),
            ):
                retry = await rig.create(
                    provider=None,
                    agent_profile="sup",
                    session_name="adopt-pure",
                    profile_contract=contract,
                )

        assert retry.id == first.id
        assert retry.profile_receipt == first.profile_receipt
        assert rig.backend.create_session.call_count == tmux_calls
        assert rig.row_count() == 1


def _launch_kimi(rig, profile_store, session_name, contract):
    return rig.create(
        provider=None,
        agent_profile="sup",
        session_name=session_name,
        profile_contract=contract,
    )


def _retry(rig, session_name, contract):
    return rig.create(
        provider=None,
        agent_profile="sup",
        session_name=session_name,
        profile_contract=contract,
    )


def _tweak_row(terminal_id, **columns):
    """Direct durable-state tweak between launch and retry."""
    from cli_agent_orchestrator.clients.database import SessionLocal, TerminalModel

    with SessionLocal() as db:
        db.query(TerminalModel).filter(TerminalModel.id == terminal_id).update(
            columns, synchronize_session=False
        )
        db.commit()


class TestAdoptMismatch:
    async def _launched(self, rig, profile_store, session_name="adopt-x"):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        first = await _launch_kimi(rig, profile_store, session_name, contract)
        return first, contract

    @pytest.mark.asyncio
    async def test_model_change_refuses(self, profile_store, isolated_memory_db, launched_provider):
        """Profile edited between response loss and retry: 409, zero effects."""
        import json as _json

        with _LaunchRig(launched_provider) as rig:
            first, _ = await self._launched(rig, profile_store)
            # The operator edits the profile; the conductor preflights fresh.
            path = profile_store / "sup.md"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "model: adopt-model-1", "model: adopt-model-2"
                ),
                encoding="utf-8",
            )
            context = load_supervisor_launch_context("sup")
            contract = _contract_for(context)
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "model" in exc_info.value.reason
            assert rig.row_count() == 1
            assert rig.backend.create_session.call_count == 1
            # The stored receipt is untouched.
            from cli_agent_orchestrator.clients import database

            stored = database.get_terminal_adoption_row(first.id)["profile_receipt"]
            assert stored["model"] == "adopt-model-1"
            assert (
                _json.loads(
                    database.SessionLocal()
                    .query(database.TerminalModel)
                    .filter(database.TerminalModel.id == first.id)
                    .one()
                    .profile_receipt
                )["model"]
                == "adopt-model-1"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "column,contract_field,bad",
        [
            ("assigned_model", "model", "other-model"),
            ("assigned_effort", "effort", "high"),
            ("provider", "provider", "codex"),
            ("agent_profile", "profile", "other"),
        ],
    )
    async def test_row_pin_mismatch_refuses(
        self, profile_store, isolated_memory_db, launched_provider, column, contract_field, bad
    ):
        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            _tweak_row(first.id, **{column: bad})
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert contract_field in exc_info.value.reason
            assert rig.row_count() == 1
            assert rig.backend.create_session.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field,bad", [("sha256", "0" * 64), ("model", "other")])
    async def test_receipt_field_mismatch_refuses(
        self, profile_store, isolated_memory_db, launched_provider, field, bad
    ):
        import json as _json

        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            from cli_agent_orchestrator.clients import database

            with database.SessionLocal() as db:
                row = (
                    db.query(database.TerminalModel)
                    .filter(database.TerminalModel.id == first.id)
                    .one()
                )
                receipt = _json.loads(row.profile_receipt)
                receipt[field] = bad
                row.profile_receipt = _json.dumps(receipt)
                db.commit()
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert field in exc_info.value.reason
            assert rig.row_count() == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", [None, "{not-json", "[1,2]"])
    async def test_missing_or_corrupt_receipt_refuses(
        self, profile_store, isolated_memory_db, launched_provider, raw
    ):
        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            _tweak_row(first.id, profile_receipt=raw)
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "receipt" in exc_info.value.reason
            assert rig.row_count() == 1

    @pytest.mark.asyncio
    async def test_pending_launch_refuses_as_in_flight(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            _tweak_row(first.id, pre_task_identity_state="pending")
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "pending" in exc_info.value.reason
            assert "in flight" in exc_info.value.recovery
            assert rig.row_count() == 1
            assert rig.backend.create_session.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("lifecycle", [None, "dead"])
    async def test_partial_or_stale_lifecycle_refuses(
        self, profile_store, isolated_memory_db, launched_provider, lifecycle
    ):
        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            _tweak_row(first.id, lifecycle_state=lifecycle)
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "lifecycle" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_dead_session_refuses(self, profile_store, isolated_memory_db, launched_provider):
        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            rig.backend.session_exists.side_effect = lambda name: False
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "gone" in exc_info.value.reason
            assert rig.row_count() == 1

    @pytest.mark.asyncio
    async def test_dead_window_refuses(self, profile_store, isolated_memory_db, launched_provider):
        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            rig.backend.window_identity.return_value = None
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "no live identity" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_replaced_window_refuses(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            rig.backend.window_identity.return_value = {
                **rig.backend.window_identity.return_value,
                "window_id": "@replaced",
            }
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "replaced" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_superseded_row_refuses(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            _tweak_row(first.id, superseded_by_terminal_id="ffffffff")
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "superseded" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_ambiguous_duplicates_refuse(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        import uuid as _uuid

        from cli_agent_orchestrator.clients import database

        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            # A pathological double creation: a second row claiming the
            # same session and receipt. Adoption must not pick one.
            with database.SessionLocal() as db:
                row = (
                    db.query(database.TerminalModel)
                    .filter(database.TerminalModel.id == first.id)
                    .one()
                )
                twin = database.TerminalModel(
                    **{
                        column.name: getattr(row, column.name)
                        for column in database.TerminalModel.__table__.columns
                        if column.name != "id"
                    },
                    id="b1b2b3b4",
                )
                twin.callback_target_generation = _uuid.uuid4().hex
                db.add(twin)
                db.commit()
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "ambiguous" in exc_info.value.reason
            assert rig.row_count() == 2

    @pytest.mark.asyncio
    async def test_retired_incarnation_refuses(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        from cli_agent_orchestrator.clients import database

        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            with database.SessionLocal() as db:
                db.query(database.StableAgentIncarnationModel).filter(
                    database.StableAgentIncarnationModel.terminal_id == first.id
                ).update({"disposition": "retired"}, synchronize_session=False)
                db.commit()
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "no live incarnation" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_missing_incarnation_refuses(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        from cli_agent_orchestrator.clients import database

        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            with database.SessionLocal() as db:
                db.query(database.StableAgentIncarnationModel).filter(
                    database.StableAgentIncarnationModel.terminal_id == first.id
                ).delete(synchronize_session=False)
                db.commit()
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "no live incarnation" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_wrong_roster_role_refuses(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        from cli_agent_orchestrator.clients import database

        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            with database.SessionLocal() as db:
                db.query(database.StableAgentModel).update(
                    {"role": "worker"}, synchronize_session=False
                )
                db.commit()
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "role" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_moved_on_incarnation_refuses(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        from cli_agent_orchestrator.clients import database

        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            with database.SessionLocal() as db:
                db.query(database.StableAgentModel).update(
                    {"current_incarnation_id": "00000000-0000-0000-0000-000000000000"},
                    synchronize_session=False,
                )
                db.commit()
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await _retry(rig, "adopt-x", contract)
            assert "superseded" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_roster_conflict_refuses(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        from cli_agent_orchestrator.services import stable_agent_roster

        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(rig, profile_store)
            with patch.object(
                stable_agent_roster,
                "get_incarnation_by_terminal",
                side_effect=stable_agent_roster.StableAgentConflict("two live"),
            ):
                with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                    await _retry(rig, "adopt-x", contract)
            assert "ambiguous" in exc_info.value.reason


class TestAdoptRequestMismatch:
    async def _launched(self, rig, profile_store, session_name="adopt-req", **kwargs):
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        first = await rig.create(
            provider=None,
            agent_profile="sup",
            session_name=session_name,
            profile_contract=contract,
            **kwargs,
        )
        return first, contract

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field,first_value,second_value",
        [
            ("working_directory", "/tmp/first-cwd", "/tmp/second-cwd"),
            ("allowed_tools", ["Read"], ["*"]),
            ("env_vars", {"SHARED": "one"}, {"SHARED": "two"}),
            ("memory_manager", None, "true"),
        ],
    )
    async def test_request_only_difference_refuses(
        self, profile_store, isolated_memory_db, launched_provider, field, first_value, second_value
    ):
        """Same contract, different request: 409 with zero new effects.

        The stored receipt still matches the contract — only the
        request identity (cwd, tools, env, sidecar input) differs — so
        the retry must refuse rather than silently receive the first
        request's working directory, policy, or environment.
        """
        with _LaunchRig(launched_provider) as rig:
            first_kwargs = {field: first_value} if first_value is not None else {}
            first, contract = await self._launched(rig, profile_store, **first_kwargs)
            prepares_before_retry = list(rig.prepare_calls)
            posts_before_retry = len(rig.post_create_events())
            second_kwargs = {field: second_value} if second_value is not None else {}
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await rig.create(
                    provider=None,
                    agent_profile="sup",
                    session_name="adopt-req",
                    profile_contract=contract,
                    **second_kwargs,
                )
            assert "fingerprint" in exc_info.value.reason
            assert rig.row_count() == 1
            assert rig.backend.create_session.call_count == 1
            assert rig.prepare_calls == prepares_before_retry
            assert len(rig.post_create_events()) == posts_before_retry

    @pytest.mark.asyncio
    async def test_reordered_env_adopts(self, profile_store, isolated_memory_db, launched_provider):
        """Key order is not identity: reordered env adopts the winner."""
        with _LaunchRig(launched_provider) as rig:
            first, contract = await self._launched(
                rig, profile_store, env_vars={"B": "2", "A": "1"}
            )
            retry = await rig.create(
                provider=None,
                agent_profile="sup",
                session_name="adopt-req",
                profile_contract=contract,
                env_vars={"A": "1", "B": "2"},
            )
            assert rig.last_result.adopted is True
        assert retry.id == first.id
        assert rig.row_count() == 1


class TestAdoptReadinessGating:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("readiness", ["pending", "failed"])
    async def test_unsettled_readiness_refuses(
        self, profile_store, isolated_memory_db, launched_provider, readiness
    ):
        """A pending/failed row is never adopted, however exact the match."""
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        with _LaunchRig(launched_provider) as rig:
            first = await rig.create(
                provider=None,
                agent_profile="sup",
                session_name="adopt-ready",
                profile_contract=contract,
            )
            _tweak_row(first.id, provider_readiness=readiness)
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await rig.create(
                    provider=None,
                    agent_profile="sup",
                    session_name="adopt-ready",
                    profile_contract=contract,
                )
            assert "readiness" in exc_info.value.reason
            assert rig.row_count() == 1
            assert rig.backend.create_session.call_count == 1

    @pytest.mark.asyncio
    async def test_failed_init_marks_row_and_refuses_retry(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """A provider-init failure with failed cleanup refuses the retry.

        The first call raises; its surviving row is marked failed; the
        retry refuses instead of adopting a launch that never completed
        provider initialization — with no second effect sequence.
        """
        from cli_agent_orchestrator.services import terminal_service

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        rig = _LaunchRig(launched_provider)
        rig.launched_provider.initialize = AsyncMock(
            side_effect=RuntimeError("provider init failed")
        )
        with rig:
            # Cleanup fails too, so the dead row survives teardown —
            # the failure the failed-marker exists for.
            with patch.object(
                terminal_service,
                "db_delete_terminal",
                side_effect=RuntimeError("db cleanup failed"),
            ):
                with pytest.raises(RuntimeError, match="provider init failed"):
                    await rig.create(
                        provider=None,
                        agent_profile="sup",
                        session_name="adopt-fail",
                        profile_contract=contract,
                    )
            from cli_agent_orchestrator.clients import database

            row = database.get_terminal_adoption_row("abcd4242")
            assert row is not None
            assert row["provider_readiness"] == "failed"
            tmux_calls = rig.backend.create_session.call_count
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await rig.create(
                    provider=None,
                    agent_profile="sup",
                    session_name="adopt-fail",
                    profile_contract=contract,
                )
            assert "readiness" in exc_info.value.reason
            assert rig.backend.create_session.call_count == tmux_calls
            assert len(rig.post_create_events()) == 0


class TestAdoptSourceDrift:
    @pytest.mark.asyncio
    async def test_exact_winner_adopts_despite_source_drift(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """The durable winner is validated before mutable source is read.

        Launch profile A, overwrite the source with B, retry A's exact
        contract: the retry adopts A's winner with A's stored receipt —
        no Conflict, no B launch, no new effects, and the source is
        never re-read for the verdict.
        """
        from cli_agent_orchestrator.services import terminal_service

        _write_profile(profile_store, "sup")
        context_a = load_supervisor_launch_context("sup")
        contract_a = _contract_for(context_a)
        expected_a = spr.build_profile_receipt(context_a)
        with _LaunchRig(launched_provider) as rig:
            first = await rig.create(
                provider=None,
                agent_profile="sup",
                session_name="adopt-drift",
                profile_contract=contract_a,
            )
            assert first.profile_receipt == expected_a
            # The operator replaces the profile source after response loss.
            path = profile_store / "sup.md"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "model: adopt-model-1", "model: adopt-model-2"
                ),
                encoding="utf-8",
            )
            with patch.object(
                terminal_service,
                "load_agent_profile",
                side_effect=AssertionError("mutable source re-read on adopt"),
            ):
                retry = await rig.create(
                    provider=None,
                    agent_profile="sup",
                    session_name="adopt-drift",
                    profile_contract=contract_a,
                )
            assert rig.last_result.adopted is True
        assert retry.id == first.id
        assert retry.profile_receipt == first.profile_receipt == expected_a
        assert rig.row_count() == 1
        assert rig.backend.create_session.call_count == 1
        assert rig.prepare_calls == ["kimi_cli"]
        assert len(rig.post_create_events()) == 1


class TestAdoptBoundaries:
    @pytest.mark.asyncio
    async def test_no_contract_duplicate_remains_400(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """The ordinary duplicate without a contract keeps ValueError/400."""
        _write_profile(profile_store, "sup")
        with _LaunchRig(launched_provider) as rig:
            first = await rig.create(provider=None, agent_profile="sup", session_name="dup")
            assert first.id == "abcd4242"
            with pytest.raises(ValueError, match="already exists"):
                await rig.create(provider=None, agent_profile="sup", session_name="dup")
            assert rig.row_count() == 1

    @pytest.mark.asyncio
    async def test_legacy_row_never_adopts(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """A contract retry over a legacy row refuses 409 — never adopts.

        The first launch ran without a contract, so its row carries no
        request fingerprint or readiness: the retry's receipt may agree,
        but a contract-bearing retry cannot adopt a row with no durable
        request identity. Zero new effects.
        """
        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        with _LaunchRig(launched_provider) as rig:
            first = await rig.create(provider=None, agent_profile="sup", session_name="rej")
            prepares_before_retry = list(rig.prepare_calls)
            with pytest.raises(spr.ProfileAdoptionMismatch) as exc_info:
                await rig.create(
                    provider=None,
                    agent_profile="sup",
                    session_name="rej",
                    profile_contract=contract,
                )
            # The legacy row records no readiness or request identity:
            # the retry refuses with zero new effects.
            assert "readiness" in exc_info.value.reason
            assert rig.row_count() == 1
            assert rig.backend.create_session.call_count == 1
            assert rig.prepare_calls == prepares_before_retry
            assert len(rig.post_create_events()) == 1

    @pytest.mark.asyncio
    async def test_mismatch_maps_to_409_end_to_end(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """The typed mismatch surfaces as HTTP 409 with reason/recovery."""
        from fastapi import BackgroundTasks, HTTPException, Response

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        with _LaunchRig(launched_provider) as rig:
            first = await rig.create(
                provider=None,
                agent_profile="sup",
                session_name="http409",
                profile_contract=contract,
            )
            _tweak_row(first.id, assigned_model="other-model")
            with patch.object(main, "get_plugin_registry", return_value=MagicMock()):
                with pytest.raises(HTTPException) as exc_info:
                    await main.create_session(
                        request=MagicMock(),
                        response=Response(),
                        background_tasks=BackgroundTasks(),
                        agent_profile="sup",
                        session_name="http409",
                        env_vars=None,
                        profile_contract=contract,
                    )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["session_name"] == "cao-http409"
        assert "model" in exc_info.value.detail["reason"]
        assert exc_info.value.detail["recovery"]
        assert rig.row_count() == 1

    @pytest.mark.asyncio
    async def test_adopted_retry_answers_200_with_indicator(
        self, profile_store, isolated_memory_db, launched_provider
    ):
        """An adopted retry answers HTTP 200 with the adoption indicator.

        The body is the stored terminal with its stored receipt — the
        same terminal id, byte-identical receipt, no new effects.
        """
        from fastapi import BackgroundTasks, Response

        from cli_agent_orchestrator.api import main

        _write_profile(profile_store, "sup")
        context = load_supervisor_launch_context("sup")
        contract = _contract_for(context)
        expected = spr.build_profile_receipt(context)
        with _LaunchRig(launched_provider) as rig:
            first = await rig.create(
                provider=None,
                agent_profile="sup",
                session_name="http200",
                profile_contract=contract,
            )
            with patch.object(main, "get_plugin_registry", return_value=MagicMock()):
                response = Response()
                second = await main.create_session(
                    request=MagicMock(),
                    response=response,
                    background_tasks=BackgroundTasks(),
                    agent_profile="sup",
                    session_name="http200",
                    env_vars=None,
                    profile_contract=contract,
                )
        assert response.status_code == 200
        assert response.headers["x-cao-adopted"] == "exact-ready"
        assert second.id == first.id
        assert second.profile_receipt == first.profile_receipt == expected
        assert rig.row_count() == 1
        assert rig.backend.create_session.call_count == 1
        assert rig.prepare_calls == ["kimi_cli"]
        assert len(rig.post_create_events()) == 1
