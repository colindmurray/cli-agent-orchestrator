"""Project goal-effect flock — fork-side vendored twin (cond-0845).

Canonical source: ``cao-conductor: conduct/lib/goal_effect_flock.py``.
This copy is byte-equivalent in behavior (path scheme, modes, timeout,
reentrancy, no-upgrade rule); it exists because the fork venv cannot
import conductor code and the plugin ships inside cao-server's venv.
The cross-repo parity test pins the contract — change the canonical
module and the twins together, never one alone.

The fork holds this lock SHARED across a restoration effect (never a
database transaction across I/O); conductor semantic writers hold it
EXCLUSIVELY around short transactions. The file is named by the
conductor caller (absolute ``goal-effect.lock`` path in the fence);
this module never invents the path.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import threading
import time
from typing import Dict, Iterator, Tuple

#: Lock file name inside the project state directory.
LOCK_FILE_NAME = "goal-effect.lock"

#: How long a writer waits for the flock before deferring (seconds).
#: Writers hold the lock only across short transactions, so anything
#: beyond this means another process wedged mid-write, not contention.
DEFAULT_TIMEOUT_SECONDS = 5.0

_state = threading.local()


def _held() -> Dict[Tuple[str, bool], Tuple[int, object]]:
    """This thread's held locks, dropped on fork.

    ``os.fork`` duplicates memory AND file descriptors, so a child
    inherits both this table and the open lock fds. Entries are tagged
    with the owning pid; a child (different pid) discards them and
    re-acquires from the kernel, which still excludes it while the
    parent holds the lock. Without the tag a forked child would
    mistake inherited table entries for locks it owns.
    """
    current = getattr(_state, "held", None)
    if current is None:
        current = {}
        _state.held = current
        _state.pid = os.getpid()
    elif getattr(_state, "pid", None) != os.getpid():
        current.clear()
        _state.pid = os.getpid()
    return current


class GoalEffectBusy(Exception):
    """The flock could not be acquired before the deadline; defer, retry."""


def lock_path(project_state_dir: str) -> str:
    """The single flock file for one project state directory."""
    return os.path.join(project_state_dir, LOCK_FILE_NAME)


@contextlib.contextmanager
def hold_path(path: str, *, shared: bool = False,
              timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> Iterator[None]:
    """Hold the flock at an explicit absolute path (fork-side entry).

    Same contract as :func:`hold`; the path comes from the conductor
    fence rather than local resolution. Refuses when the parent
    directory cannot host exclusion (fail closed — an effect that
    cannot prove exclusion must not run).
    """
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise GoalEffectBusy(
            f"goal-effect flock parent {parent} is not a directory; "
            "refusing an effect that cannot prove exclusion"
        )
    fd = _acquire_fd(path, shared=shared, timeout_seconds=timeout_seconds)
    held = _held()
    key = (path, shared)
    if key in held:
        # Same-thread re-entry through two spellings of one lock: fold
        # into the existing entry instead of self-blocking on Linux.
        os.close(fd)
        depth, fd0 = held[key]
        held[key] = (depth + 1, fd0)
        try:
            yield
        finally:
            _release_entry(held, path, shared)
        return
    held[key] = (1, fd)
    try:
        yield
    finally:
        del held[key]
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _acquire_fd(path: str, *, shared: bool, timeout_seconds: float) -> int:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                return fd
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GoalEffectBusy(
                    f"goal-effect flock at {path} is held by another process; "
                    "deferring rather than waiting behind an unknown write"
                )
            time.sleep(min(0.05, remaining))
    except Exception:
        os.close(fd)
        raise


@contextlib.contextmanager
def hold(
    project_state_dir: str,
    *,
    shared: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Hold the project goal-effect flock for one bounded critical section.

    Writers take it exclusively around short semantic transactions.
    Restoration delivery takes it shared across its final re-read and the
    bounded provider write (an exclusive writer — a hold activation, a
    completion claim, a succession — still excludes delivery entirely).
    Nested acquisition from the same thread is a no-op re-entry; the
    outermost exit releases. Never hold this across unbounded I/O: the
    provider write it spans is deadline-bounded by the caller.
    """
    if not project_state_dir or project_state_dir == ":memory:":
        yield
        return
    path = lock_path(project_state_dir)
    held = _held()
    exclusive = _held().get((path, False))
    if shared and exclusive is not None:
        # Covered by this thread's own exclusive hold: bump its depth so
        # the outermost exit releases. No second fd (a same-process SH
        # flock against our own EX would self-block on Linux).
        depth, fd = exclusive
        held[(path, False)] = (depth + 1, fd)
        try:
            yield
        finally:
            _release_entry(held, path, False)
        return
    key = (path, shared)
    if key in held:
        depth, fd = held[key]
        held[key] = (depth + 1, fd)
        try:
            yield
        finally:
            _release_entry(held, path, shared)
        return
    if not shared and (path, True) in held:
        # A shared hold must never silently cover an exclusive request
        # from the same thread — that would be a lock upgrade (ABBA
        # against a concurrent exclusive waiter), so refuse loudly.
        raise GoalEffectBusy(
            "cannot upgrade a shared goal-effect hold to exclusive on "
            "the same thread; release first, then re-acquire"
        )
    try:
        os.makedirs(project_state_dir, exist_ok=True)
    except OSError:
        # A state directory we cannot create (read-only fixture, missing
        # project) cannot host cross-process exclusion either; proceed
        # without the lock rather than refusing a read-shaped path. Writers
        # will fail on their own transaction against the missing store.
        yield
        return
    fd = _acquire_fd(path, shared=shared, timeout_seconds=timeout_seconds)
    held[key] = (1, fd)
    try:
        yield
    finally:
        del held[key]
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _release_entry(
    held: Dict[Tuple[str, bool], Tuple[int, object]], path: str, shared: bool
) -> None:
    depth, _fd = held[(path, shared)]
    if depth <= 1:
        del held[(path, shared)]
    else:
        held[(path, shared)] = (depth - 1, _fd)


def held_depth(project_state_dir: str, *, shared: bool) -> int:
    """Test hook: current re-entry depth for one lock (0 when unheld)."""
    entry = _held().get((lock_path(project_state_dir), shared))
    return entry[0] if entry else 0
