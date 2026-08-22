"""Dedicated injectable sqlite-vec search connection factory.

sqlite-vec registers its SQL functions on ONE connection at load time. The
authoritative store therefore keeps two strictly separated access paths:

* the pooled ``SessionLocal`` engine in ``clients/database.py``, which never
  loads extensions and never sees a single ``vec_*`` function; and
* THIS factory, which opens a dedicated raw ``sqlite3`` connection per use,
  enables extension loading, calls ``sqlite_vec.load(...)``, and disables
  extension loading again before handing the connection out.

The hybrid issue-search design fixes this boundary in §7.2: ordinary source
transactions stay on the pool, vector ranking happens only on connections
this factory produced. Because sqlite-vec is pre-1.0, every successful open
pins the observed ``vec_version()`` against :data:`PINNED_VEC_VERSION` and
refuses mismatches rather than ranking against unknown semantics.

The factory is injectable at two seams so tests and later search lanes can
supply their own database file or an entirely foreign connection producer
(any DBAPI connection callable) without touching authoritative state.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

from cli_agent_orchestrator.constants import DATABASE_FILE

logger = logging.getLogger(__name__)

#: The exact sqlite-vec release measured and pinned for v1 (design §7.2).
#: sqlite-vec is pre-1.0: minor releases change semantics freely, so the
#: observed version must equal this string exactly or the connection refuses
#: to serve. The pyproject ``[search]`` extra carries the matching ``==`` pin.
PINNED_VEC_VERSION = "v0.1.9"


class SearchEngineError(Exception):
    """Base class for typed search-engine failures.

    ``reason`` carries a stable machine-readable classification so callers
    can branch on failure kind instead of parsing messages:
    ``runtime-missing`` (the sqlite-vec package is not installed),
    ``extension-api-unavailable`` (this Python's sqlite3 cannot load
    extensions), ``open-failed`` (the connection could not be opened, or an
    injected producer failed), ``load-failed`` (the connection opened with
    extensions enabled but ``sqlite_vec.load`` itself threw), and
    ``version-mismatch`` (loaded vec_version() differs from the pin).
    """

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        observed_vec_version: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        #: Positive observation carried by ``version-mismatch`` refusals: the
        #: version string the loaded engine actually reported.
        self.observed_vec_version = observed_vec_version


@dataclass
class SearchConnection:
    """A live search connection whose ``vec_*`` functions are verified.

    ``connection`` is a plain DBAPI connection suitable for scalar-distance
    KNN over ordinary BLOB rows (design §7.2). ``vec_version`` records the
    version observed on THIS connection at open time — the positive signal
    diagnostics and receipts quote. Close it via the context manager or
    :meth:`close`; closing is idempotent.
    """

    connection: Any
    vec_version: str
    db_path: Optional[str]

    def close(self) -> None:
        try:
            self.connection.close()
        except Exception:  # noqa: BLE001 - closing must never mask the caller's error
            logger.debug("search connection already closed", exc_info=True)

    def __enter__(self) -> "SearchConnection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _load_sqlite_vec_module() -> Any:
    """Import sqlite-vec lazily, translating absence into a typed failure.

    The package lives in the optional ``[search]`` extra; importing at module
    scope would make every base install pay for it and would turn "not
    installed" into an opaque ImportError far from the operator's decision.
    """
    try:
        import sqlite_vec  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SearchEngineError(
            "runtime-missing",
            "the sqlite-vec package is not installed; install the [search] "
            f"extra to enable vector search ({exc})",
        ) from exc
    return sqlite_vec


def _enable_extension_loading(connection: Any) -> None:
    """Enable loadable extensions, typing unavailable sqlite builds."""
    enable = getattr(connection, "enable_load_extension", None)
    if enable is None:
        raise SearchEngineError(
            "extension-api-unavailable",
            "this Python's sqlite3 module does not expose enable_load_extension; "
            "a CPython built without loadable-extension support cannot serve vector search",
        )
    enable(True)


def open_search_connection(
    *,
    db_path: Union[str, Path, None] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    expected_vec_version: str = PINNED_VEC_VERSION,
) -> SearchConnection:
    """Open one dedicated sqlite-vec connection through the pinned sequence.

    Either supply ``db_path`` (defaulting to the authoritative tracker
    database FILE, deliberately bypassing the pooled engine) or a
    ``connection_factory`` returning any DBAPI connection — tests inject
    temporary databases, later lanes may inject read-only or alternate-store
    producers. Supplying both is a caller ambiguity and refuses. The
    loaded-then-disabled extension sequence and the ``vec_version()`` pin run
    identically regardless of where the connection came from, so an injected
    producer can never skip the boundary checks.

    Refusals are typed (:class:`SearchEngineError`): the sqlite-vec runtime
    is absent, this sqlite build cannot load extensions, the connection
    cannot be opened or loaded, or the observed ``vec_version()`` differs
    from ``expected_vec_version``. A refused connection is closed before
    raising; no half-loaded handle escapes.
    """
    if connection_factory is not None and db_path is not None:
        raise SearchEngineError(
            "open-failed",
            "pass either db_path or connection_factory, not both; the source "
            "of the search connection must be unambiguous",
        )
    sqlite_vec = _load_sqlite_vec_module()

    if connection_factory is not None:
        try:
            raw = connection_factory()
        except Exception as exc:
            raise SearchEngineError(
                "open-failed", f"injected connection factory failed: {exc}"
            ) from exc
        path_label = "<injected>"
    else:
        resolved = Path(db_path) if db_path is not None else DATABASE_FILE
        if not resolved.exists():
            raise SearchEngineError(
                "open-failed",
                f"database file does not exist yet: {resolved}; run a tracker "
                "command once to initialize the store before opening vector search",
            )
        try:
            raw = sqlite3.connect(str(resolved))
        except sqlite3.Error as exc:
            raise SearchEngineError(
                "open-failed", f"could not open search connection to {resolved}: {exc}"
            ) from exc
        path_label = str(resolved)

    try:
        _enable_extension_loading(raw)
        sqlite_vec.load(raw)
        # Close the door immediately after loading (the SQL load_extension()
        # path now refuses with "not authorized"); the vec_* functions stay
        # registered on this connection only.
        raw.enable_load_extension(False)
        row = raw.execute("SELECT vec_version()").fetchone()
        observed = row[0] if row else None
        if observed != expected_vec_version:
            raise SearchEngineError(
                "version-mismatch",
                f"vec_version() reported {observed!r} but this build pins "
                f"{expected_vec_version!r}; upgrade or downgrade the sqlite-vec "
                "package to match the pinned generation",
                observed_vec_version=observed,
            )
    except SearchEngineError:
        raw.close()
        raise
    except Exception as exc:
        raw.close()
        # Distinct from open-failed: the connection opened and extension
        # loading was enabled, but sqlite_vec.load itself failed — the
        # extension API demonstrably exists.
        raise SearchEngineError("load-failed", f"loading sqlite-vec failed: {exc}") from exc

    return SearchConnection(connection=raw, vec_version=observed, db_path=path_label)


def describe_search_engine(
    *, connection_factory: Optional[Callable[[], Any]] = None
) -> dict[str, Any]:
    """Observe the engine leg WITHOUT raising, for capability diagnostics.

    Probes through an in-memory connection by default (pass
    ``connection_factory`` to observe against a real store instead), so the
    observation never requires a database file to exist and never touches
    authoritative state. Returns positive signals only — what was actually
    OBSERVED at each stage, with explicit ``False``/``None`` rather than
    absence:

    * ``runtime_present`` — the sqlite-vec distribution imports;
    * ``extension_api_available`` — this sqlite build exposes the extension
      loading API and a load actually succeeded on the probe connection;
    * ``vec_version_observed`` — the exact ``vec_version()`` the loaded
      engine reported (``None`` when loading never got that far);
    * ``vec_version_pinned`` — the version this build requires.

    A version mismatch is still a fully observed engine: both versions are
    reported and the caller decides what the mismatch means.
    """
    signals: dict[str, Any] = {
        "runtime_present": False,
        "extension_api_available": False,
        "vec_version_observed": None,
        "vec_version_pinned": PINNED_VEC_VERSION,
    }
    try:
        _load_sqlite_vec_module()
    except SearchEngineError:
        return signals
    signals["runtime_present"] = True

    if connection_factory is None:
        # In-memory probe store: pure capability observation with zero
        # filesystem footprint.
        def connection_factory() -> Any:
            return sqlite3.connect(":memory:")
    try:
        probe = open_search_connection(connection_factory=connection_factory)
    except SearchEngineError as exc:
        if exc.reason == "extension-api-unavailable":
            return signals
        # The sequence got far enough to attempt a load (a mismatch refusal,
        # or the load itself throwing with the API already enabled): the
        # extension API exists even though no version was observed.
        signals["extension_api_available"] = exc.reason in (
            "version-mismatch",
            "load-failed",
        )
        if exc.observed_vec_version is not None:
            signals["vec_version_observed"] = exc.observed_vec_version
        return signals
    with probe:
        signals["extension_api_available"] = True
        signals["vec_version_observed"] = probe.vec_version
    return signals
