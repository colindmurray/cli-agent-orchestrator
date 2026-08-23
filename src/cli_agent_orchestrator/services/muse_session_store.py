"""The Muse on-disk session store, read as a fallback identity surface.

A fresh Muse TUI registers its provider-generated session at cold start —
before any prompt or command is typed — in two synchronized places under
``${XDG_DATA_HOME:-$HOME/.local/share}/muse``:

* a per-session directory ``sessions/YYYY/MM/DD/<session-id>/`` whose name
  IS the canonical-UUID session id, created synchronously with the TUI
  process start (verified live against the installed 0.2.1-R1215.1 build:
  a launch in ``/private/tmp/muse-probe-cond0713`` produced the directory
  and its ``session.jsonl`` in the same second the composer rendered); and
* a row in the async indexer's ``session-index.db``, which lands seconds
  to minutes later and is therefore deliberately NOT consulted here.

The first record of ``session.jsonl`` is the provider's own
``runtime.session.metadata`` event naming the workspace root, provider id,
and build that created the session.  Discovery diffs the directory set
against a snapshot taken before the pane process exists, requires the new
candidate's own metadata record to describe exactly this launch
(workspace root equal to the reserved working directory, the expected
provider), and adopts only an exactly-one match: zero candidates keeps
polling until the deadline, two or more is ambiguity that refuses rather
than picks.

This is a fallback identity surface for builds whose ``/status`` panel
does not (or does not yet) render the SESSION row.  The panel remains the
primary observation; identity adopted from the store is validated as
strictly as panel-discovered identity and never carries route facts the
store cannot prove.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

#: Schema of a store-discovered identity observation.
MUSE_SESSION_STORE_SCHEMA = "cao-muse-session-store-v1"

#: The metadata event type every adoptable session log opens with.
_METADATA_PAYLOAD_TYPE = "runtime.session.metadata"

#: How long one discovery pass waits between store scans.
DEFAULT_POLL_SECONDS = 0.25


class MuseSessionStoreError(ValueError):
    """The Muse session store could not prove exactly one new session."""


class MuseSessionStoreAmbiguous(MuseSessionStoreError):
    """More than one new session matches; adopting either would be a guess."""


class MuseSessionStoreUnavailable(MuseSessionStoreError):
    """No new session matched within the deadline."""


def sessions_root() -> Path:
    """The installed layout's session-directory root.

    Mirrors the provider's own resolution — ``${XDG_DATA_HOME:-$HOME
    /.local/share}/muse/sessions`` — so a redirected XDG data home is
    followed rather than assumed away.
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "muse" / "sessions"


def _canonical_uuid(value: str) -> Optional[str]:
    """The value if it is a canonical lowercase UUID string, else None."""
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return None
    return value if str(parsed) == value else None


def _iter_session_dirs() -> list[tuple[str, Path]]:
    """Every per-session directory as ``(session_id, path)``, depth-guarded.

    Directory names that are not canonical lowercase UUIDs are not session
    directories and are skipped, never half-read.
    """
    root = sessions_root()
    found: list[tuple[str, Path]] = []
    try:
        # The layout is sessions/YYYY/MM/DD/<session-id>/ — four levels
        # below ``muse``, so the leaf name IS the candidate id.
        day_dirs = sorted(root.glob("*/*/*/*"))
    except OSError:
        return found
    for path in day_dirs:
        if not path.is_dir():
            continue
        session_id = _canonical_uuid(path.name)
        if session_id is not None:
            found.append((session_id, path))
    return found


def snapshot_known_session_ids() -> frozenset[str]:
    """The session ids already on disk, taken BEFORE the pane process starts.

    A snapshot taken any later would fold this launch's own session into
    the known set and make the diff empty, so callers must take this
    before spawning the fresh TUI.  An absent store root is true absence
    (no session can exist yet) and yields an empty set; a partially
    readable tree can only undercount, which at worst turns a stale
    session into a spurious candidate that its own metadata record then
    refuses.
    """
    return frozenset(session_id for session_id, _ in _iter_session_dirs())


def _read_session_metadata(path: Path) -> tuple[Optional[dict[str, Any]], bool]:
    """Read one candidate's cold-start metadata record.

    Returns ``(record_or_None, readable)``: a missing or still-empty
    ``session.jsonl`` is *pending* (the TUI has not flushed yet), not
    evidence; anything present but not the expected metadata event is
    unreadable evidence that can never be adopted.
    """
    log_path = path / "session.jsonl"
    try:
        with open(log_path, encoding="utf-8") as handle:
            first = handle.readline()
    except FileNotFoundError:
        return None, False
    except OSError:
        return None, True
    if not first.strip():
        return None, False
    try:
        event = json.loads(first)
    except ValueError:
        return None, True
    if not isinstance(event, dict) or event.get("payload_type") != _METADATA_PAYLOAD_TYPE:
        return None, True
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None, True
    record = payload.get("record")
    if not isinstance(record, dict):
        return None, True
    return record, True


def discover_new_session_id(
    snapshot: frozenset[str],
    *,
    working_directory: str,
    provider_id: str = "meta",
    deadline_monotonic: float,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> dict[str, Any]:
    """Adopt the exactly-one new session this launch created, or refuse.

    A candidate is a session directory absent from the snapshot whose own
    cold-start metadata names exactly ``working_directory`` and
    ``provider_id``.  Adoption additionally requires the ENTIRE new-dir
    set to be resolved — no sibling directory still pending its first log
    write or carrying unreadable evidence.  A concurrent launch in this
    workspace registers the same kind of directory ours does, and whichever
    flushes first must never be adopted as THIS pane's identity merely
    because it won the flush race; an unresolved sibling keeps the window
    open until it resolves into a second match (refused as ambiguous), a
    non-match (ours is then adoptable), or nothing at all by the deadline.

    More than one resolved match raises :class:`MuseSessionStoreAmbiguous`
    immediately — resolved matches do not un-resolve, so waiting cannot
    make two candidates into one, and picking among them is how a task
    gets typed into someone else's session.  Hitting the deadline with the
    set still unresolved raises :class:`MuseSessionStoreUnavailable` naming
    the unresolved counts.
    """
    while True:
        candidates: list[tuple[str, dict[str, Any], Path]] = []
        pending = 0
        unreadable = 0
        for session_id, path in _iter_session_dirs():
            if session_id in snapshot:
                continue
            record, readable = _read_session_metadata(path)
            if record is None:
                if readable:
                    unreadable += 1
                else:
                    pending += 1
                continue
            if (
                record.get("workspace_root") == working_directory
                and record.get("provider_id") == provider_id
            ):
                candidates.append((session_id, record, path))
        if len(candidates) > 1:
            raise MuseSessionStoreAmbiguous(
                f"the Muse session store holds {len(candidates)} new "
                f"{provider_id!r} sessions for {working_directory!r} since the "
                "launch snapshot; it cannot prove which one this pane runs, so "
                "refusing rather than choosing a value"
            )
        if len(candidates) == 1 and pending == 0 and unreadable == 0:
            session_id, record, path = candidates[0]
            build = record.get("build") if isinstance(record.get("build"), dict) else {}
            return {
                "schema": MUSE_SESSION_STORE_SCHEMA,
                "session_id": session_id,
                "metadata": {
                    "workspace_root": record.get("workspace_root"),
                    "provider_id": record.get("provider_id"),
                    "build_semver": build.get("semver"),
                    "build_sha": build.get("sha"),
                },
                "session_dir": str(path),
            }
        if time.monotonic() >= deadline_monotonic:
            raise MuseSessionStoreUnavailable(
                f"no new {provider_id!r} session for {working_directory!r} appeared "
                f"in the Muse session store before the deadline "
                f"({pending} candidate(s) still pending their first log write, "
                f"{unreadable} unreadable); refusing to adopt an identity the "
                "store cannot prove"
            )
        time.sleep(poll_seconds)
