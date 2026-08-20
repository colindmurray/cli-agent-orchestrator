"""Stable identity for one CAO state installation.

The project id and issue key namespace are intentionally local to an
installation, so neither can distinguish two machines that happen to carry
the same project.  This UUID is minted once beneath the state root and then
reported by the server.  It is observability and accidental-target protection,
not authentication: the trusted local-process threat model does not treat the
file as a credential.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from cli_agent_orchestrator.constants import CAO_HOME_DIR

INSTALLATION_ID_FILE = CAO_HOME_DIR / "installation-id"


class InstallationIdentityError(RuntimeError):
    """The durable identity exists but cannot be trusted as a canonical UUID."""


def _parse(raw: str, *, path: Path) -> str:
    value = raw.strip()
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise InstallationIdentityError(
            f"CAO installation identity at {path} is malformed"
        ) from exc
    canonical = str(parsed)
    if value != canonical:
        raise InstallationIdentityError(f"CAO installation identity at {path} is malformed")
    return canonical


def _read(path: Path) -> str:
    try:
        return _parse(path.read_text(encoding="utf-8"), path=path)
    except OSError as exc:
        raise InstallationIdentityError(
            f"CAO installation identity at {path} is unreadable: {exc}"
        ) from exc


def get_installation_id() -> str:
    """Read the stable UUID, atomically minting it when this install is new."""

    path = INSTALLATION_ID_FILE
    if path.exists():
        return _read(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = str(uuid.uuid4())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        # Another startup won the first-initialization race. Its durable value
        # is the identity; the unused candidate has no meaning.
        return _read(path)
    except OSError as exc:
        raise InstallationIdentityError(
            f"could not create CAO installation identity at {path}: {exc}"
        ) from exc

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(candidate + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # The exclusive file may now be partial. Preserve it for diagnosis;
        # silently replacing an identity after a failed write would be worse.
        raise
    return candidate
