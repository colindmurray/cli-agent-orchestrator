"""The Muse route attestor: verdicts travel verbatim, refusals are typed."""

from __future__ import annotations

import sys

import pytest

from cli_agent_orchestrator.services import muse_native_launch as muse
from cli_agent_orchestrator.services.muse_route import (
    MuseRouteProbeError,
    attest_muse_route,
)

BANNER = "Muse Code 0.2.1 (0.2.1-R1215.1)"
REVISION = "0.2.1-R1215.1"


def _make_install(
    tmp_path,
    *,
    behavior: str = "normal",
    revision: str = REVISION,
    banner: str = BANNER,
):
    """A launcher layout that models real Muse: wrapper, .muse-version, inner bin.

    The wrapper answers ``--version`` with ``banner`` so the attestor's
    version observation runs for real; the inner binary is the same
    truth-table stub the carrier-probe tests use, so the two-leg probe runs
    for real against a process rather than being mocked away.
    """
    wrapper = tmp_path / "muse"
    wrapper.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then\n  echo "{banner}"\nfi\n')
    wrapper.chmod(0o755)
    (tmp_path / ".muse-version").write_text(revision)
    inner = tmp_path / f"muse-bin-{revision}"

    if behavior == "normal":
        script = f"""#!{sys.executable}
import os, sys
from pathlib import Path
env_path = os.environ.get("{muse.PROFILE_SYSTEM_PROMPT_ENV}")
if env_path is not None:
    p = Path(env_path)
    if not p.exists():
        sys.stderr.write("failed to read path: No such file or directory\\n")
        sys.exit(1)
    if p.stat().st_size > 0:
        sys.stderr.write("{muse.CARRIER_PROBE_REFUSAL}\\n")
        sys.exit(1)
sys.exit(0)
"""
    elif behavior == "disproved":
        script = f"""#!{sys.executable}
import sys
sys.exit(0)
"""
    elif behavior == "usage_dump":
        script = f"""#!{sys.executable}
import sys
sys.stderr.write("unknown preset <name>; expected native-basic|miniswe\\n")
sys.exit(2)
"""
    else:
        raise ValueError(f"Unknown behavior: {behavior}")

    inner.write_text(script)
    inner.chmod(0o755)
    return wrapper, inner


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    muse._PROBE_CACHE.clear()
    yield
    muse._PROBE_CACHE.clear()


def test_a_probed_carrier_produces_an_honest_receipt(tmp_path, monkeypatch):
    wrapper, inner = _make_install(tmp_path, behavior="normal")
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which", lambda name: str(wrapper)
    )

    receipt = attest_muse_route(
        str(tmp_path), expected_model="muse-spark-1.3-contributor", expected_effort="high"
    )

    assert receipt["probe_version"] == "muse-carrier-route-v1"
    assert receipt["carrier_verdict"] == "probed"
    assert receipt["carrier_verdict_detail"] == ""
    assert receipt["muse_version"] == "0.2.1"
    assert receipt["full_banner"] == BANNER
    assert receipt["wrapper_executable"] == str(wrapper)
    # The receipt names the exact inner binary the launcher resolves, not
    # the update-capable wrapper.
    assert receipt["inner_executable"] == str(inner)
    assert receipt["inner_executable_sha256"] == muse._sha256_file(str(inner))
    assert receipt["project_root"] == str(tmp_path)
    assert receipt["no_managed_session_started"] is True
    assert receipt["no_task_bytes_submitted"] is True


def test_requested_is_recorded_and_observed_stays_null_with_the_reason(tmp_path, monkeypatch):
    """The probe pins neither model nor effort; the receipt says so."""
    wrapper, _ = _make_install(tmp_path, behavior="normal")
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which", lambda name: str(wrapper)
    )

    receipt = attest_muse_route(
        str(tmp_path), expected_model="muse-spark-1.3-contributor", expected_effort="high"
    )

    assert receipt["requested_model"] == "muse-spark-1.3-contributor"
    assert receipt["requested_effort"] == "high"
    assert receipt["observed_model"] is None
    assert receipt["observed_effort"] is None
    assert receipt["pre_turn_route_surface"] is False
    assert "--reasoning-effort" in receipt["unobserved_reason"]
    assert receipt["terminal_route_argv_pins"] == [
        "--model",
        "muse-spark-1.3-contributor",
        "--reasoning-effort",
        "high",
    ]


def test_a_disproved_carrier_refuses(tmp_path, monkeypatch):
    """A build that ignores base instructions proves its own unhealthiness."""
    wrapper, _ = _make_install(tmp_path, behavior="disproved")
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which", lambda name: str(wrapper)
    )

    with pytest.raises(MuseRouteProbeError, match="non-empty base instructions present"):
        attest_muse_route(str(tmp_path), expected_model="m", expected_effort="high")


def test_an_unproven_verdict_travels_verbatim_never_upgraded(tmp_path, monkeypatch):
    """Inconclusive is recorded as inconclusive — never read as health."""
    wrapper, _ = _make_install(tmp_path, behavior="usage_dump")
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which", lambda name: str(wrapper)
    )

    receipt = attest_muse_route(str(tmp_path), expected_model="m", expected_effort="high")

    assert receipt["carrier_verdict"] == "unproven"
    assert receipt["carrier_verdict_detail"] == (
        "profile_carrier_unproven: unknown preset <name>; expected native-basic|miniswe"
    )


def test_an_operator_pinned_build_attests_as_probed_by_operator(tmp_path, monkeypatch):
    """The documented override reads identically on both surfaces.

    A persistent-disproved build cleared by CAO_MUSE_PROFILE_CARRIER_PROVEN
    launches as probed_by_operator; if attest-route refused it anyway, a
    tripped breaker could never be re-armed for a route that launches fine.
    The stub is the disproved build: only the pin gets it through.
    """
    wrapper, inner = _make_install(tmp_path, behavior="disproved")
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which", lambda name: str(wrapper)
    )
    digest = muse._sha256_file(str(inner))
    monkeypatch.setenv(muse.CAO_MUSE_PROFILE_CARRIER_PROVEN_ENV, digest)

    receipt = attest_muse_route(str(tmp_path), expected_model="m", expected_effort="high")

    assert receipt["carrier_verdict"] == "probed_by_operator"
    assert muse.CAO_MUSE_PROFILE_CARRIER_PROVEN_ENV in receipt["carrier_verdict_detail"]
    assert receipt["inner_executable"] == str(inner)
    assert receipt["inner_executable_sha256"] == digest
    # The live probe never ran: a disproved stub would have refused.
    assert receipt["no_managed_session_started"] is True


def test_a_mismatched_operator_pin_falls_through_to_the_live_probe(tmp_path, monkeypatch):
    """A pin naming some other binary proves nothing about this one."""
    wrapper, _ = _make_install(tmp_path, behavior="normal")
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which", lambda name: str(wrapper)
    )
    monkeypatch.setenv(muse.CAO_MUSE_PROFILE_CARRIER_PROVEN_ENV, "f" * 64)

    receipt = attest_muse_route(str(tmp_path), expected_model="m", expected_effort="high")

    # The live two-leg probe ran and answered, rather than the pin speaking.
    assert receipt["carrier_verdict"] == "probed"
    assert receipt["carrier_verdict_detail"] == ""


def test_an_absent_wrapper_is_a_typed_probe_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which", lambda name: None
    )

    with pytest.raises(MuseRouteProbeError, match="not on PATH"):
        attest_muse_route(str(tmp_path), expected_model="m", expected_effort="high")


@pytest.mark.parametrize("banner", ["garbage banner\n", "\n"])
def test_an_unparseable_banner_is_a_failed_observation(tmp_path, monkeypatch, banner):
    """Unparseable is distinct from unlisted: the observation itself failed."""
    _make_install(tmp_path, banner=banner)
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which",
        lambda name: str(tmp_path / "muse"),
    )

    with pytest.raises(MuseRouteProbeError, match="semver-shaped"):
        attest_muse_route(str(tmp_path), expected_model="m", expected_effort="high")


def test_a_wrapper_whose_active_revision_disagrees_refuses_with_the_reason(tmp_path, monkeypatch):
    """The launcher-layout gate fails closed with what it observed."""
    _make_install(tmp_path, revision="0.1.0-R708.1")
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which",
        lambda name: str(tmp_path / "muse"),
    )

    with pytest.raises(MuseRouteProbeError, match="active revision differs"):
        attest_muse_route(str(tmp_path), expected_model="m", expected_effort="high")


def test_strict_quarantine_mode_refuses_an_unlisted_build(tmp_path, monkeypatch):
    """The opt-in quarantine still gates the probe at the boundary."""
    wrapper, _ = _make_install(tmp_path, banner="Muse Code 9.9.9 (9.9.9-R9999.0)")
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.muse_route.shutil.which", lambda name: str(wrapper)
    )
    monkeypatch.setenv("CAO_PROVIDER_VERSION_ENFORCEMENT_MUSE", "strict")

    with pytest.raises(MuseRouteProbeError, match="unsupported Muse version"):
        attest_muse_route(str(tmp_path), expected_model="m", expected_effort="high")


def test_a_noncanonical_project_root_refuses_before_any_probe(tmp_path):
    linked = tmp_path / "link"
    linked.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(MuseRouteProbeError, match="canonical"):
        attest_muse_route(str(linked), expected_model="m", expected_effort="high")
