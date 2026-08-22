"""Search connection factory: isolation, pinning, injection, and the bypass gate.

The boundary under test (hybrid search design §7.2): sqlite-vec loads ONLY on
connections this factory produced, NEVER on the authoritative pooled engine,
and every successful open proves the loaded ``vec_version()`` against the pin.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest
import sqlalchemy.exc

from cli_agent_orchestrator.clients import database as db_module
from cli_agent_orchestrator.services.search_engine_factory import (
    PINNED_VEC_VERSION,
    SearchEngineError,
    describe_search_engine,
    open_search_connection,
)


def _blob(vec):
    return struct.pack("<4f", *vec)


def test_open_search_connection_pins_vec_version(tmp_path):
    db = tmp_path / "store.db"
    sqlite3.connect(str(db)).close()
    with open_search_connection(db_path=db) as handle:
        assert handle.vec_version == PINNED_VEC_VERSION
        row = handle.connection.execute("SELECT vec_version()").fetchone()
        assert row[0] == PINNED_VEC_VERSION


def test_extension_loading_disabled_after_open(tmp_path):
    """The door closes after loading: the SQL load_extension() path refuses.

    (The C API can be re-enabled by an explicit enable_load_extension call —
    which is exactly the act the source-wide bypass gate below refuses to let
    any production module perform outside this factory.)
    """
    db = tmp_path / "store.db"
    sqlite3.connect(str(db)).close()
    with open_search_connection(db_path=db) as handle:
        with pytest.raises(sqlite3.OperationalError, match="not authorized"):
            handle.connection.execute("SELECT load_extension('no/such/thing')").fetchone()


def test_plain_connection_on_same_file_never_gains_vec_functions(tmp_path):
    """Registration is per-connection: an ordinary connection sees no vec_*."""
    db = tmp_path / "store.db"
    sqlite3.connect(str(db)).close()
    with open_search_connection(db_path=db):
        pass
    plain = sqlite3.connect(str(db))
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function"):
            plain.execute("SELECT vec_version()").fetchone()
    finally:
        plain.close()


def test_source_sessionlocal_pool_never_gains_vec_functions():
    """Acceptance: the authoritative SessionLocal pool has NO vec functions.

    The pool here is the suite's own scratch-root engine (see conftest), the
    same module-global pool every authoritative tracker write flows through.
    """
    with db_module.SessionLocal() as session:
        raw = session.connection()
        with pytest.raises(sqlalchemy.exc.OperationalError, match="no such function"):
            raw.exec_driver_sql("SELECT vec_version()").fetchone()


def test_injected_connection_factory_receives_loaded_connection():
    """Any producer works — and gets the identical load+pin treatment."""
    holder = {}

    def factory():
        conn = sqlite3.connect(":memory:")
        holder["conn"] = conn
        return conn

    with open_search_connection(connection_factory=factory) as handle:
        assert holder["conn"] is handle.connection
        assert handle.vec_version == PINNED_VEC_VERSION


def test_scalar_distance_knn_over_ordinary_blob_table(tmp_path):
    """Exact KNN follows the §7.2 regular-table pattern end to end."""
    db = tmp_path / "knn.db"
    setup = sqlite3.connect(str(db))
    setup.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding BLOB NOT NULL "
        "CHECK (length(embedding) = 16))"
    )
    rows = [(1, (1.0, 0.0, 0.0, 0.0)), (2, (0.0, 1.0, 0.0, 0.0)), (3, (0.9, 0.1, 0.0, 0.0))]
    for row_id, vec in rows:
        setup.execute("INSERT INTO docs VALUES (?, ?)", (row_id, _blob(vec)))
    setup.commit()
    setup.close()

    query = _blob((0.95, 0.05, 0.0, 0.0))
    with open_search_connection(db_path=db) as handle:
        ranked = handle.connection.execute(
            "SELECT id, vec_distance_cosine(embedding, :q) AS d " "FROM docs ORDER BY d",
            {"q": query},
        ).fetchall()
    # Row 1 points almost exactly where the query points; row 3 is close;
    # row 2 is orthogonal and ranks last.
    assert [row[0] for row in ranked] == [1, 3, 2]


def test_missing_runtime_is_typed(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refusing_import(name, *args, **kwargs):
        if name == "sqlite_vec":
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refusing_import)
    with pytest.raises(SearchEngineError) as excinfo:
        open_search_connection(connection_factory=lambda: sqlite3.connect(":memory:"))
    assert excinfo.value.reason == "runtime-missing"


def test_version_mismatch_is_typed_with_observed_version(tmp_path):
    db = tmp_path / "store.db"
    sqlite3.connect(str(db)).close()
    with pytest.raises(SearchEngineError) as excinfo:
        open_search_connection(db_path=db, expected_vec_version="v9.9.9")
    assert excinfo.value.reason == "version-mismatch"
    assert excinfo.value.observed_vec_version == PINNED_VEC_VERSION


def test_missing_database_file_is_typed(tmp_path):
    with pytest.raises(SearchEngineError) as excinfo:
        open_search_connection(db_path=tmp_path / "absent.db")
    assert excinfo.value.reason == "open-failed"
    assert "does not exist" in excinfo.value.message


def test_describe_reports_positive_signals_without_raising():
    signals = describe_search_engine()
    assert signals["runtime_present"] is True
    assert signals["extension_api_available"] is True
    assert signals["vec_version_observed"] == PINNED_VEC_VERSION
    assert signals["vec_version_pinned"] == PINNED_VEC_VERSION


def test_describe_reports_absent_runtime_as_observed_false(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sqlite_vec", None)  # blocks import
    signals = describe_search_engine()
    assert signals["runtime_present"] is False
    assert signals["extension_api_available"] is False
    assert signals["vec_version_observed"] is None


# ---------------------------------------------------------------------------
# §19.7-style mutation gate: bypassing this factory anywhere in production
# code must fail THIS test. It scans every module under src/ for the loading
# primitives and refuses any occurrence outside search_engine_factory.py, so
# a second call site cannot merge quietly — the gate fires on every future
# occurrence, not just the first.
# ---------------------------------------------------------------------------


_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "cli_agent_orchestrator"
_FACTORY_MODULE = "search_engine_factory.py"
_LOADING_MARKERS = (
    "sqlite_vec.load(",
    "enable_load_extension(",
    "import sqlite_vec",
    "from sqlite_vec",
)


def test_sqlite_vec_load_only_through_factory():
    offenders: dict[str, list[str]] = {}
    for py_file in sorted(_SRC_ROOT.rglob("*.py")):
        if py_file.name == _FACTORY_MODULE:
            continue
        hits = [
            f"{lineno}: {line.strip()}"
            for lineno, line in enumerate(py_file.read_text().splitlines(), start=1)
            if any(marker in line for marker in _LOADING_MARKERS)
        ]
        if hits:
            offenders[str(py_file.relative_to(_SRC_ROOT))] = hits
    assert not offenders, (
        "sqlite-vec loading escaped the dedicated factory; these modules must "
        f"go through services/{_FACTORY_MODULE} instead: {offenders}"
    )
