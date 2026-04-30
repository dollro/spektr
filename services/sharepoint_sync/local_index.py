from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    local_path TEXT NOT NULL,
    etag TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_path_idx ON items (local_path);
"""


class LocalIndex:
    """SQLite-backed map of (Graph item_id) -> (local_path, etag)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        with self._connect() as cx:
            cx.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def upsert(self, item_id: str, local_path: str, *, etag: str) -> None:
        with self._connect() as cx:
            cx.execute(
                "INSERT INTO items (item_id, local_path, etag) VALUES (?, ?, ?) "
                "ON CONFLICT(item_id) DO UPDATE SET "
                "local_path=excluded.local_path, etag=excluded.etag",
                (item_id, local_path, etag),
            )

    def get_path(self, item_id: str) -> str | None:
        with self._connect() as cx:
            row = cx.execute(
                "SELECT local_path FROM items WHERE item_id=?", (item_id,)
            ).fetchone()
        return row[0] if row else None

    def get_etag(self, item_id: str) -> str | None:
        with self._connect() as cx:
            row = cx.execute("SELECT etag FROM items WHERE item_id=?", (item_id,)).fetchone()
        return row[0] if row else None

    def delete(self, item_id: str) -> None:
        with self._connect() as cx:
            cx.execute("DELETE FROM items WHERE item_id=?", (item_id,))

    def iter_under(self, prefix: str) -> Iterator[tuple[str, str]]:
        like = prefix.rstrip("/") + "/%"
        with self._connect() as cx:
            for row in cx.execute(
                "SELECT item_id, local_path FROM items WHERE local_path LIKE ?",
                (like,),
            ):
                yield row[0], row[1]
