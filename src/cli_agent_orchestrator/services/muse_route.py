"""Zero-task Muse route attestation via the two-leg profile-carrier probe.

The route receipt a launch breaker reads must gather its evidence from the
provider it names, so the Muse attestor runs exactly the probe a managed
Muse launch runs (:func:`muse_native_launch.probe_profile_carrier`) against
exactly the binary a managed Muse launch executes (the ``.muse-version`` +
``muse-bin-<revision>`` inner binary, never the update-capable wrapper).

The carrier verdict travels verbatim and is never upgraded: ``probed`` is
the only verdict that asserts the base-instructions surface works,
``unproven`` records that this probe established nothing, and ``disproved``
— a build that ran a clean echo turn with base instructions present, so it
ignores them — refuses here exactly as it fails closed at launch. An
operator who has pinned ``CAO_MUSE_PROFILE_CARRIER_PROVEN`` to the inner
binary's digest attests as ``probed_by_operator``, the same verdict the
launch path records, so the two surfaces cannot disagree about a build.

Resolution and proof go through
:func:`muse_native_launch.profile_carrier_capability` — the same authority a
managed Muse launch consults — rather than a private re-derivation, so the
override, the launcher-layout gate, and the probe have exactly one reader.

What this receipt deliberately does not claim: an observed model or effort.
The zero-task probe pins neither — model and effort are carried by launch
argv (``--model``/``--reasoning-effort``) and read back from the ``/status``
panel during a real managed launch, which creates a pane this endpoint must
never create. The requested values are recorded as requested; the observed
values are null with the reason stated, the same discipline the Claude
receipt follows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from cli_agent_orchestrator.services import muse_native_launch, provider_contracts

PROBE_VERSION = "muse-carrier-route-v1"

#: Exact Muse builds this probe accepts in strict quarantine mode — never a
#: range. Read from the one acceptance authority so the probe cannot drift
#: from the contract it is attesting against. In open mode any parseable
#: build is probed: the two-leg carrier probe below reads the route back
#: from the provider itself, so an unlisted build proves — or fails — the
#: carrier surface at runtime rather than inheriting a neighbour's
#: attestation.
SUPPORTED_MUSE_VERSIONS = provider_contracts.SUPPORTED_VERSIONS[provider_contracts.PROVIDER_MUSE]

UNOBSERVED_REASON = (
    "Muse resolves no pre-turn model or effort surface this probe can read: "
    "the values are pinned by launch argv (--model, --reasoning-effort) and "
    "observed from the /status panel after a real managed pane starts, which "
    "this zero-task attestation never does. The requested route is recorded "
    "as requested, never as resolved."
)

#: Why the receipt's session flags are honest rather than copied from the
#: other providers' receipts. The carrier probe runs throwaway one-shot echo
#: turns in a temp directory; naming that effect beats borrowing a field
#: whose meaning ("no prompt was sent") would be false by the letter.
PROBE_EFFECT = (
    "the carrier probe runs `muse exec --provider echo --no-session-log` "
    "against the resolved inner binary in a temp directory: throwaway "
    "echo-provider turns only"
)


class MuseRouteProbeError(RuntimeError):
    pass


def attest_muse_route(
    project_root: str,
    *,
    expected_model: str,
    expected_effort: str,
    muse_bin: str = "muse",
) -> dict[str, Any]:
    """Return a provider-native Muse route receipt without starting a session.

    What is proven: the working directory is a canonical existing directory,
    the installed launcher wrapper is present and parseable, and the
    carrier-capability authority a managed launch consults resolves this
    installation to a verdict, carried through verbatim. A ``disproved``
    capability raises, as it fails closed at launch; ``unproven`` produces
    a receipt that says so.
    """
    if not os.path.isdir(project_root) or os.path.realpath(project_root) != project_root:
        raise MuseRouteProbeError("project_root must be an existing canonical directory")

    try:
        muse_native_launch.validate_requested_model(expected_model)
    except muse_native_launch.MuseNativeLaunchError as exc:
        raise MuseRouteProbeError(str(exc)) from exc

    resolved = shutil.which(muse_bin)
    if resolved is None:
        raise MuseRouteProbeError(f"Muse wrapper {muse_bin!r} is not on PATH")
    wrapper = os.path.realpath(resolved)
    if not os.path.isabs(wrapper) or not os.path.isfile(wrapper):
        raise MuseRouteProbeError("Muse wrapper must be an existing canonical file")

    try:
        version_proc = subprocess.run(
            [wrapper, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
            env={**os.environ, "MUSE_NO_AUTO_UPDATE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MuseRouteProbeError(f"could not execute Muse version probe: {exc}") from exc

    banner = (version_proc.stdout or version_proc.stderr or "").strip()
    normalized = provider_contracts.normalized_version(banner)
    if version_proc.returncode != 0 or not normalized:
        # Unparseable is a failed observation, distinct from unlisted.
        raise MuseRouteProbeError(
            f"unsupported Muse version {banner!r}; expected a semver-shaped version"
        )
    # Open enforcement: any semver-shaped version is probed — the carrier
    # probe below is the route proof. Strict enforcement: must be an exact
    # listed build (the quarantine set).
    if (
        provider_contracts.version_enforcement_mode(provider_contracts.PROVIDER_MUSE)
        != (provider_contracts.VERSION_ENFORCEMENT_OPEN)
        and normalized not in SUPPORTED_MUSE_VERSIONS
    ):
        raise MuseRouteProbeError(
            f"unsupported Muse version {banner!r}; expected one of "
            f"{list(SUPPORTED_MUSE_VERSIONS)!r}"
        )

    # Resolve, override, and probe through the one capability authority a
    # managed launch consults. An operator pin on
    # CAO_MUSE_PROFILE_CARRIER_PROVEN therefore reads identically here and at
    # launch: a build that launches as probed_by_operator also attests as it,
    # and a disproved build refuses on both surfaces instead of wedging a
    # tripped breaker that launch can clear.
    capability = muse_native_launch.profile_carrier_capability(
        wrapper_executable=wrapper, full_banner=banner
    )
    if not capability.supported:
        raise MuseRouteProbeError(f"Muse route attestation failed: {capability.reason}")

    detail = capability.reason
    if capability.proof == muse_native_launch.PROOF_PROBED_BY_OPERATOR:
        detail = (
            "operator attestation: "
            f"{muse_native_launch.CAO_MUSE_PROFILE_CARRIER_PROVEN_ENV} is pinned to "
            "this inner binary's sha256"
        )

    return {
        "probe_version": PROBE_VERSION,
        "muse_version": normalized,
        # This banner comes from the wrapper resolved on the ambient PATH by
        # ``shutil.which`` above — NOT the reserve-pinned provider_executable
        # that a managed launch re-verifies by digest.  It must therefore never
        # be recorded as the durable ``provider_executable_version`` in the
        # launch facts: that would be exactly the forbidden ambient inference.
        # v1 Muse rows stay non-resurrectable by design; only the v2 native
        # launch records its digest-verified wrapper's banner (managed_launch_v2).
        "full_banner": banner,
        "wrapper_executable": wrapper,
        "inner_executable": capability.inner_executable,
        "inner_executable_sha256": capability.inner_executable_sha256,
        "project_root": project_root,
        # Verbatim from the authority: unproven travels as unproven and an
        # operator attestation travels as probed_by_operator — never
        # upgraded to probed, never silently dropped.
        "carrier_verdict": capability.proof,
        "carrier_verdict_detail": detail,
        "route_source": "launcher-resolved-inner-binary",
        # Requested, not resolved — the two are different claims and the
        # field names say which one this is.
        "requested_model": expected_model,
        "requested_effort": expected_effort,
        "observed_model": None,
        "observed_effort": None,
        "pre_turn_route_surface": False,
        "unobserved_reason": UNOBSERVED_REASON,
        "terminal_route_argv_pins": [
            "--model",
            expected_model,
            "--reasoning-effort",
            expected_effort,
        ],
        "probe_effect": PROBE_EFFECT,
        "no_managed_session_started": True,
        "no_task_bytes_submitted": True,
    }
