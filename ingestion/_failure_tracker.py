"""Persistent ingestion failure counter.

Lets `ingest_file` survive transient and permanent failures without
blocking the batch. A file that fails N times in a row becomes a
"poison pill" — logged CRITICAL, swallowed, and CocoIndex then
marks it processed so the rest of the batch proceeds.

Counts persist across process restarts in a SQLite file under
`{STATE_DIR}/ingestion_failures.db` (`FAILURE_DB_PATH` to pin it
directly). Successful ingestion resets the count for that file.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _default_db_path() -> Path:
    """Resolve the tracker's path from settings, at call time.

    This was a module-level `Path("state") / "ingestion_failures.db"`
    constant, which made it the one piece of runtime state `STATE_DIR` could
    not relocate — so a sandbox or CI checkout that redirected everything
    else still wrote its poison-pill counts into the shared tree.

    Resolved per call rather than at import so a test (or a process that
    reconfigures settings before first use) is not bound by import order.
    """
    return Path(settings.failure_db_path)


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ingestion_failures ("
        "source_file TEXT PRIMARY KEY, "
        "fail_count INTEGER NOT NULL DEFAULT 0, "
        "last_error TEXT, "
        "last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    return conn


class FailureTracker:
    """Sqlite-backed failure counter. Thread-safe via a module lock."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _default_db_path()

    def record_failure(self, source_file: str, error: str = "") -> int:
        """Increment the fail count for `source_file`; return the new count."""
        with _lock, _connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO ingestion_failures (source_file, fail_count, last_error) "
                "VALUES (?, 1, ?) "
                "ON CONFLICT(source_file) DO UPDATE SET "
                "fail_count = fail_count + 1, "
                "last_error = excluded.last_error, "
                "last_seen = CURRENT_TIMESTAMP",
                (source_file, error[:500]),
            )
            row = conn.execute(
                "SELECT fail_count FROM ingestion_failures WHERE source_file = ?",
                (source_file,),
            ).fetchone()
            return int(row[0]) if row else 0

    def reset(self, source_file: str) -> None:
        """Clear the counter for `source_file` — call after successful ingest."""
        with _lock, _connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM ingestion_failures WHERE source_file = ?",
                (source_file,),
            )

    def fail_count(self, source_file: str) -> int:
        """Return the current count without mutating it."""
        with _lock, _connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT fail_count FROM ingestion_failures WHERE source_file = ?",
                (source_file,),
            ).fetchone()
            return int(row[0]) if row else 0

    def should_poison(self, source_file: str, max_retries: int) -> bool:
        """True when fail count has reached or exceeded max_retries."""
        return self.fail_count(source_file) >= max_retries


_tracker: FailureTracker | None = None


def get_tracker() -> FailureTracker:
    """Module-level singleton."""
    global _tracker  # noqa: PLW0603
    if _tracker is None:
        _tracker = FailureTracker()
    return _tracker
