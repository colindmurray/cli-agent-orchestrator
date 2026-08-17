"""Composer and rendering facts read from the *installed* provider bundle.

The per-build tables in the native-control adapters (``_PROVEN_COMPOSER_NEWLINE``,
``_PROVEN_STEER_CHORDS``, ``_RENDERED_SESSION_PROVEN_BUILDS``) record what was
proven against builds someone read by hand.  That knowledge stays valuable, but
under the unpinned provider-version policy a missing table row is no longer a
capability refusal: the same facts the pins recorded as evidence are readable at
runtime from whatever build is installed, and a successful read is the override.

This module is that read.  Each extractor looks for the exact byte pattern a
stage-verification recorded — never a looser pattern "close to" it — and every
answer is bound to the build being driven: the bundle's own version (its sibling
``package.json``, or the inner-binary name for Muse) must equal the normalized
provider version, or the answer is ``None``.  ``None`` is not a capability
denial by itself; the caller falls back to its typed refusal, so an unreadable
bundle fails loudly rather than inheriting a neighbouring build's behaviour.

Nothing here executes the provider: a bundle is read as data, never run.

Measured readability (2026-08-17, operator's installation):

- Claude 2.1.233 ``bin/claude.exe``: contains ``ctrl+j for newline`` (twice).
- Kimi 0.36.1 ``dist/main.mjs``: declares ``"tui.input.newLine":
  {defaultKeys: ["shift+enter", "ctrl+j"]}`` and the steer dispatch
  ``matchesKey(normalized, Key.ctrl("s"))``; ``main()`` begins with
  ``process.title = PROCESS_NAME``.
- Muse 0.1.0-R708.1 inner binary: its keymap table is readable as
  ``newline`` + ``shift+enter``/``ctrl+j``/``ctrl+m`` + ``Insert a newline
  without submitting``.
- Codex 0.147.0: NOT readable — the editor keymap defaults are compiled as
  structured data; the binary contains the action name ``insert_newline`` and
  its description but no binding string (``ctrl+j``/``ctrl-j`` absent).  Codex
  therefore keeps the typed refusal for unlisted builds.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from cli_agent_orchestrator.services import provider_contracts

#: The one keystroke every readable hint below names for a composer newline:
#: a single control byte every terminal transmits identically.  The extractors
#: only ever yield it when the build's own bundle says so; it is never
#: assumed.
_CTRL_J = "C-j"

#: Wire provider key -> the executable name looked up on PATH.  Codex is
#: deliberately absent: its binary carries no readable binding string (see
#: the module docstring), so there is nothing to derive.
_EXECUTABLE_FOR_PROVIDER = {
    "kimi_cli": "kimi",
    "claude_code": "claude",
    "muse_cli": "muse",
}

#: How far before/after a chunk boundary a needle may span.  Every needle and
#: every extracted region below is well under one KiB, so a generous overlap
#: keeps the streaming search exact without holding a 300 MB bundle in memory.
_CHUNK_BYTES = 4 * 1024 * 1024
_OVERLAP_BYTES = 64 * 1024


@dataclass(frozen=True)
class BundleFact:
    """One fact read from the installed bundle of the exact build being driven.

    ``bundle_version`` is the bundle's own declared version, which the reader
    required to equal the build the caller is driving — the fact is about this
    build because the bytes it was read from are this build's bytes.
    """

    provider: str
    keystroke: str
    bundle_path: str
    bundle_version: str
    description: str


def _stream_search(path: str, needles: tuple[bytes, ...]) -> tuple[bool, ...]:
    """Which of ``needles`` appear in the file, read once in overlapping chunks."""
    found = [False] * len(needles)
    remaining = len(needles)
    tail = b""
    with open(path, "rb") as handle:
        while remaining:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            window = tail + chunk
            for index, needle in enumerate(needles):
                if not found[index] and needle in window:
                    found[index] = True
                    remaining -= 1
            tail = window[-_OVERLAP_BYTES:]
    return tuple(found)


def _stream_search_regex(path: str, pattern: "re.Pattern[bytes]") -> Optional[re.Match]:
    """The first regex match in the file, or ``None``.

    A match that begins inside the trailing overlap is deferred to the next
    window, where it begins before the overlap; on the final window every
    match is admissible.  Patterns used here bind regions far smaller than
    the overlap.
    """
    tail = b""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                return None
            window = tail + chunk
            final = len(chunk) < _CHUNK_BYTES
            match = pattern.search(window)
            if match is not None and (final or match.start() < len(window) - _OVERLAP_BYTES):
                return match
            if final:
                return None
            tail = window[-_OVERLAP_BYTES:]
    return None


def _package_json_version(bundle_path: str, levels_up: int) -> Optional[str]:
    """The ``version`` of the package.json ``levels_up`` directories above."""
    directory = bundle_path
    for _ in range(levels_up):
        directory = os.path.dirname(directory)
    manifest = os.path.join(directory, "package.json")
    try:
        with open(manifest, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    version = data.get("version")
    return version if isinstance(version, str) and version else None


def _resolve_kimi_bundle() -> Optional[tuple[str, str]]:
    """(main.mjs path, its package version) for the installed Kimi, or None."""
    resolved = shutil.which("kimi")
    if resolved is None:
        return None
    path = os.path.realpath(resolved)
    if os.path.basename(path) != "main.mjs" or not os.path.isfile(path):
        return None
    version = _package_json_version(path, 2)
    return (path, version) if version else None


def _resolve_claude_bundle() -> Optional[tuple[str, str]]:
    """(claude.exe path, its package version) for the installed Claude, or None."""
    resolved = shutil.which("claude")
    if resolved is None:
        return None
    path = os.path.realpath(resolved)
    if not os.path.isfile(path):
        return None
    version = _package_json_version(path, 2)
    return (path, version) if version else None


def _resolve_muse_bundle(semver: str) -> Optional[tuple[str, str]]:
    """(inner-binary path, full version) for the installed Muse, or None.

    The ``muse`` on PATH is the update-capable launcher script, which is never
    evidence (the profile-carrier rule); the inner ``muse-bin-<full version>``
    siblings of the launcher are the real binaries.  Exactly one sibling may
    match the requested semver prefix — zero means the build is not installed,
    and more than one is an ambiguity this read refuses to resolve by guessing.
    """
    resolved = shutil.which("muse")
    if resolved is None:
        return None
    launcher = os.path.realpath(resolved)
    directory = os.path.dirname(launcher)
    try:
        siblings = os.listdir(directory)
    except OSError:
        return None
    matches = sorted(
        name
        for name in siblings
        if name.startswith("muse-bin-")
        and name[len("muse-bin-") :].split("-R", 1)[0] == semver
        and os.path.isfile(os.path.join(directory, name))
    )
    if len(matches) != 1:
        return None
    return os.path.join(directory, matches[0]), matches[0][len("muse-bin-") :]


def _bundle_for(provider: str, normalized_version: str) -> Optional[tuple[str, str]]:
    """(path, bundle version) for ``provider``'s installed bundle, or None.

    The bundle's own version must equal the normalized build being driven:
    reading a hint from a *different* installed build would be the
    neighbour-inheritance the per-build tables existed to prevent, moved to a
    new address.
    """
    if not normalized_version:
        return None
    if provider == "kimi_cli":
        resolved = _resolve_kimi_bundle()
    elif provider == "claude_code":
        resolved = _resolve_claude_bundle()
    elif provider == "muse_cli":
        resolved = _resolve_muse_bundle(normalized_version)
    else:
        return None
    if resolved is None:
        return None
    path, bundle_version = resolved
    if provider == "muse_cli":
        # The inner binary is named by its full version (``0.1.0-R708.1``)
        # while the build being driven is known by its semver; the resolver
        # already required exactly one installed sibling for this semver.
        # The full revision is recorded in the evidence, so the fact names
        # the exact binary it was read from.
        if bundle_version.split("-R", 1)[0] != normalized_version:
            return None
    elif provider_contracts.normalized_version(bundle_version) != normalized_version:
        return None
    return path, bundle_version


@lru_cache(maxsize=64)
def _extract(
    provider: str, path: str, bundle_version: str, mtime_ns: int, size: int
) -> tuple[bool, ...]:
    """The provider's fact needles, answered from one read of the bundle.

    Cached by content identity (path + mtime + size) rather than by provider:
    a bundle replaced on disk — the auto-update case — is a new cache key, so
    a stale answer cannot survive the build it described.  The boolean tuple
    positions are per-provider needle orders, fixed below.
    """
    if provider == "kimi_cli":
        return _stream_search(
            path,
            (
                b'"tui.input.newLine"',
                b'defaultKeys: ["shift+enter", "ctrl+j"]',
                b'matchesKey(normalized, Key.ctrl("s"))',
                b"process.title = PROCESS_NAME",
                b'label: "Directory"',
                b'label: "Session"',
                b'label: "Model"',
                b'label: "Version"',
            ),
        )
    if provider == "claude_code":
        return _stream_search(path, (b"ctrl+j for newline",))
    if provider == "muse_cli":
        match = _stream_search_regex(
            path,
            re.compile(
                rb"newline(?:shift\+enter|ctrl\+[jm]|alt\+enter)+"
                rb"Insert a newline without submitting"
            ),
        )
        if match is None:
            return (False,)
        return (b"ctrl+j" in match.group(0),)
    return ()


def _facts(
    provider: str, normalized_version: str
) -> Optional[tuple[tuple[str, str], tuple[bool, ...]]]:
    """((path, bundle version), needle answers) for the driven build, or None."""
    bundle = _bundle_for(provider, normalized_version)
    if bundle is None:
        return None
    path, bundle_version = bundle
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (path, bundle_version), _extract(
        provider, path, bundle_version, stat.st_mtime_ns, stat.st_size
    )


def newline_keystroke_hint(provider: str, provider_version: Optional[str]) -> Optional[BundleFact]:
    """The composer-newline keystroke read from the installed bundle, or None.

    This is the §3 override for a missing ``_PROVEN_COMPOSER_NEWLINE`` row: the
    hint the stage-verified pins recorded as evidence (``ctrl+j for newline``,
    the ``tui.input.newLine`` default keys, the Muse keymap table) is read from
    the exact build being driven.  ``None`` means no hint could be read —
    unreadable bundle, version mismatch, or a provider (Codex) whose binary
    carries no readable binding — and the caller keeps its typed refusal.
    """
    if provider not in _EXECUTABLE_FOR_PROVIDER or not isinstance(provider_version, str):
        return None
    normalized = provider_contracts.normalized_version(provider_version)
    found = _facts(provider, normalized)
    if found is None:
        return None
    (path, bundle_version), answers = found
    if provider == "kimi_cli":
        if not (answers[0] and answers[1]):
            return None
        description = (
            'installed bundle declares "tui.input.newLine" defaultKeys '
            '["shift+enter", "ctrl+j"] (runtime bundle read, not a '
            "stage-verified pin)"
        )
    elif provider == "claude_code":
        if not answers[0]:
            return None
        description = (
            'installed bundle advertises "ctrl+j for newline" in its composer '
            "hint list (runtime bundle read, not a stage-verified pin)"
        )
    elif provider == "muse_cli":
        if not answers[0]:
            return None
        description = (
            "installed inner binary keymap binds ctrl+j to "
            '"Insert a newline without submitting" (runtime bundle read, not '
            "a stage-verified pin)"
        )
    else:  # pragma: no cover - guarded by _EXECUTABLE_FOR_PROVIDER above
        return None
    return BundleFact(
        provider=provider,
        keystroke=_CTRL_J,
        bundle_path=path,
        bundle_version=bundle_version,
        description=description,
    )


def kimi_steer_chords_hint(provider_version: Optional[str]) -> frozenset[str]:
    """The steer chords read from the installed Kimi bundle, or an empty set.

    The override for a missing ``_PROVEN_STEER_CHORDS`` row: the
    ``matchesKey(normalized, Key.ctrl("s"))`` dispatch the pins recorded is
    looked for in the exact build being driven.  An empty set is the same
    answer an unreadable bundle always gave — the chord is refused locally at
    capture time with zero POSTs, which is a loud, typed refusal.
    """
    if not isinstance(provider_version, str):
        return frozenset()
    found = _facts("kimi_cli", provider_contracts.normalized_version(provider_version))
    if found is None:
        return frozenset()
    _, answers = found
    return frozenset({"C-s"}) if answers[2] else frozenset()


def kimi_rendered_header_hint(provider_version: Optional[str]) -> Optional[str]:
    """Evidence that the installed Kimi build renders the known header, or None.

    The override for a missing ``_RENDERED_SESSION_PROVEN_BUILDS`` row: the
    build provably rewrites its process title (``process.title =
    PROCESS_NAME``, which erases the resumed session id from the kernel argv)
    *and* declares the four header labels the strict parser requires.  Both
    must be present: a title rewrite with a different header layout means
    neither the argv proof nor the rendered proof is known to apply, and the
    caller's fail-closed path is the honest answer.  ``None`` also covers a
    build with no title rewrite, where the argv proof still works.
    """
    if not isinstance(provider_version, str):
        return None
    normalized = provider_contracts.normalized_version(provider_version)
    found = _facts("kimi_cli", normalized)
    if found is None:
        return None
    (path, bundle_version), answers = found
    title_rewrite, labels = answers[3], answers[4:]
    if not (title_rewrite and all(labels)):
        return None
    return (
        f"runtime read of the installed Kimi {bundle_version} bundle ({path}): "
        "the build rewrites process.title to PROCESS_NAME after parsing its "
        "argv, and its bundle declares the Directory/Session/Model/Version "
        "header labels the strict parser requires. Bundle-derived, not a "
        "stage-verified live proof: the rendered header on the admitted pane "
        "remains the session proof, and a build whose painted header does not "
        "match freezes exactly as an unproven build does."
    )
