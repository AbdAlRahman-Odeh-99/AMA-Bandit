"""Transactional, pickle-free training states in one SQLite file per run."""

from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STATE_VERSION = 2
STORE_SUFFIX = ".training_states.sqlite3"
_SAVE_RETRY_DELAYS = (2.0, 4.0, 8.0, 16.0, 32.0)
_TRANSIENT_OS_ERRNOS = {
    errno.EIO,
    errno.EBUSY,
    errno.ETIMEDOUT,
    getattr(errno, "ESTALE", 116),
}
_TRANSIENT_SQLITE_MESSAGES = (
    "disk i/o error",
    "unable to open database file",
    "database is locked",
)
_log = logging.getLogger("afa.training_state")


@dataclass(frozen=True)
class TrainingStateRef:
    store_path: Path
    run_id: str
    state_key: str

    @property
    def name(self):
        return self.state_key

    def __str__(self):
        return f"{self.store_path}#{self.state_key}"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _number_tag(value):
    return format(float(value), ".17g").replace("-", "m").replace(".", "p")


def state_directory(run=None, state_dir=None, method="training"):
    """Resolve the single state-store path (legacy function name retained)."""
    if state_dir is not None:
        path = Path(state_dir)
        if path.exists() and path.is_dir():
            path = path / f"{method}{STORE_SUFFIX}"
        elif not path.suffix:
            path = path.with_name(path.name + STORE_SUFFIX)
    elif run is not None:
        path = Path(run.results_dir) / f"{run.run_id}{STORE_SUFFIX}"
    else:
        token = uuid.uuid4().hex[:12]
        path = Path("results") / (
            f"{method}__{time.strftime('%Y%m%d-%H%M%S')}__"
            f"pid{os.getpid()}__{token}{STORE_SUFFIX}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect(path):
    con = sqlite3.connect(path, timeout=60.0)
    # DELETE journaling is transactional and leaves one durable file after
    # each commit (WAL would expose temporary -wal/-shm sidecars).
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS states (
            run_id TEXT NOT NULL,
            state_key TEXT NOT NULL,
            method TEXT NOT NULL,
            seed INTEGER NOT NULL,
            budget_fraction TEXT NOT NULL,
            init_fraction TEXT,
            metadata_json TEXT NOT NULL,
            arrays_npz BLOB NOT NULL,
            saved_at REAL NOT NULL,
            PRIMARY KEY (run_id, state_key),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
        CREATE INDEX IF NOT EXISTS states_lookup
            ON states(run_id, method, seed, budget_fraction, init_fraction);
    """)
    return con


@contextmanager
def _db(path):
    con = _connect(path)
    try:
        with con:
            yield con
    finally:
        con.close()


def _is_transient_storage_error(exc):
    """Whether a failed state write is worth retrying on shared storage."""
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).lower()
        return any(token in message for token in _TRANSIENT_SQLITE_MESSAGES)
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_OS_ERRNOS


def save_training_state(*, method, cell, arrays, metadata, rng, run=None,
                        state_dir=None):
    """Transactionally upsert one post-training/pre-inference cell."""
    # The runners resolve and create this exact SQLite path once before the
    # sweep. Reuse it directly instead of stat'ing and mkdir'ing its parent
    # again for every cell -- those redundant metadata operations are an
    # unnecessary failure point on shared filesystems such as Gadi's.
    if run is not None and state_dir is not None and Path(state_dir).suffix:
        store = Path(state_dir)
    else:
        store = state_directory(run=run, state_dir=state_dir, method=method)
    run_id = getattr(run, "run_id", None) or metadata.get("run_id") or store.stem
    identity = {
        "method": method,
        "run_id": run_id,
        "cell": _jsonable(cell),
        "dataset": _jsonable(metadata.get("dataset")),
        "dataset_config": _jsonable(metadata.get("dataset_config", {})),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    parts = [method, f"seed{int(cell['seed'])}",
             f"bf{_number_tag(cell['budget_fraction'])}"]
    if "init_fraction" in cell:
        parts.append(f"init{_number_tag(cell['init_fraction'])}")
    state_key = "__".join(parts + [digest])

    envelope = dict(metadata)
    envelope.update({
        "state_version": STATE_VERSION,
        "method": method,
        "run_id": run_id,
        "cell": dict(cell),
        "rng_state": copy.deepcopy(rng.bit_generator.state),
    })
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{str(k): np.asarray(v)
                                  for k, v in arrays.items()})
    now = time.time()
    for attempt in range(len(_SAVE_RETRY_DELAYS) + 1):
        try:
            # Retry the whole transaction: a shared-filesystem outage can
            # surface during connect, schema setup, write, or commit. The
            # INSERT OR REPLACE below makes repeating it idempotent.
            with _db(store) as con:
                con.execute(
                    "INSERT OR IGNORE INTO runs(run_id, created_at) VALUES (?, ?)",
                    (run_id, now),
                )
                con.execute("""
                    INSERT OR REPLACE INTO states(
                        run_id, state_key, method, seed, budget_fraction,
                        init_fraction, metadata_json, arrays_npz, saved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, state_key, method, int(cell["seed"]),
                    format(float(cell["budget_fraction"]), ".17g"),
                    (format(float(cell["init_fraction"]), ".17g")
                     if "init_fraction" in cell else None),
                    json.dumps(_jsonable(envelope), sort_keys=True,
                               allow_nan=True),
                    sqlite3.Binary(buffer.getvalue()), now,
                ))
            break
        except (OSError, sqlite3.OperationalError) as exc:
            if (not _is_transient_storage_error(exc)
                    or attempt == len(_SAVE_RETRY_DELAYS)):
                raise
            delay = _SAVE_RETRY_DELAYS[attempt]
            _log.warning(
                "transient training-state write failure for %s (%s); "
                "retrying in %.0fs [%d/%d]",
                store, exc, delay, attempt + 1,
                len(_SAVE_RETRY_DELAYS),
            )
            time.sleep(delay)
    return TrainingStateRef(store, run_id, state_key)


def _refs_from_store(path, method=None):
    with _db(path) as con:
        latest = con.execute(
            "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        if latest is None:
            return []
        run_id = latest[0]
        sql = "SELECT state_key FROM states WHERE run_id = ?"
        params = [run_id]
        if method is not None:
            sql += " AND method = ?"
            params.append(method)
        sql += " ORDER BY saved_at, state_key"
        keys = [row[0] for row in con.execute(sql, params)]
    return [TrainingStateRef(Path(path), run_id, key) for key in keys]


def find_training_states(source, method=None):
    source = Path(source)
    if source.is_file():
        stores = [source]
    elif source.is_dir():
        stores = sorted(source.rglob(f"*{STORE_SUFFIX}"))
    else:
        raise FileNotFoundError(source)
    refs = []
    for store in stores:
        refs.extend(_refs_from_store(store, method=method))
    if not refs:
        raise FileNotFoundError(f"no saved training cells in {source}")
    return refs


def _coerce_ref(ref):
    if isinstance(ref, TrainingStateRef):
        return ref
    refs = find_training_states(ref)
    if len(refs) != 1:
        raise ValueError(f"{ref} contains {len(refs)} cells; select via discovery")
    return refs[0]


def load_training_state(ref, expected_method=None):
    ref = _coerce_ref(ref)
    with _db(ref.store_path) as con:
        row = con.execute(
            "SELECT metadata_json, arrays_npz FROM states "
            "WHERE run_id = ? AND state_key = ?",
            (ref.run_id, ref.state_key),
        ).fetchone()
    if row is None:
        raise FileNotFoundError(ref)
    metadata = json.loads(row[0])
    with np.load(io.BytesIO(row[1]), allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    if metadata.get("state_version") != STATE_VERSION:
        raise ValueError(f"unsupported training state version in {ref}")
    if expected_method is not None and metadata.get("method") != expected_method:
        raise ValueError(
            f"{ref} contains method={metadata.get('method')!r}, "
            f"expected {expected_method!r}")
    return metadata, arrays


def peek_training_state(ref):
    ref = _coerce_ref(ref)
    with _db(ref.store_path) as con:
        row = con.execute(
            "SELECT metadata_json FROM states WHERE run_id = ? AND state_key = ?",
            (ref.run_id, ref.state_key),
        ).fetchone()
    if row is None:
        raise FileNotFoundError(ref)
    metadata = json.loads(row[0])
    if metadata.get("state_version") != STATE_VERSION:
        raise ValueError(f"unsupported training state version in {ref}")
    return metadata


def restored_rng(metadata):
    rng = np.random.default_rng()
    rng.bit_generator.state = metadata["rng_state"]
    return rng
