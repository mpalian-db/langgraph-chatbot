"""SQLite-backed conversation memory adapter.

Implements both ConversationReaderPort and ConversationWriterPort. Uses the
stdlib `sqlite3` module run on a worker thread via `asyncio.to_thread`, so
the route handlers stay async without bringing in `aiosqlite` as a new
dependency.

The schema is a single append-only table; reads are always ordered by
insertion id, so a sliding-window load is just a `LIMIT N` over a
descending sort plus a Python-side reversal to oldest-first.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path

from app.ports.conversation import Role, Turn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    CHECK (role IN ('user', 'assistant'))
);
CREATE INDEX IF NOT EXISTS idx_conversation_id_id
    ON conversation_turns (conversation_id, id);
"""


class SQLiteConversationStore:
    """SQLite-backed implementation of ConversationReaderPort + WriterPort.

    Holds a single shared `sqlite3.Connection` for the lifetime of the
    adapter, with a threading lock to serialize access. This is necessary
    because:
      * `:memory:` databases are per-connection -- opening a fresh connection
        per operation would create a new empty database every time.
      * sqlite3 connections are not safe for concurrent use across threads
        without explicit locking.
    For a single-process app with the access pattern of "one read + one
    write per chat request", a single-connection design is the right shape.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        # Ensure the parent directory exists for file-backed databases. ":memory:"
        # is a special sentinel for an in-memory database used by tests.
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False allows the connection to travel between the
        # event loop thread and the worker thread used by asyncio.to_thread.
        # _lock then serializes the actual SQLite operations.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # ConversationWriterPort
    # ------------------------------------------------------------------

    async def append(self, conversation_id: str, role: Role, content: str) -> None:
        await asyncio.to_thread(self._append_sync, conversation_id, role, content)

    def _append_sync(self, conversation_id: str, role: Role, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversation_turns "
                "(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, time.time()),
            )
            self._conn.commit()

    async def append_pair(
        self, conversation_id: str, user_content: str, assistant_content: str
    ) -> None:
        await asyncio.to_thread(
            self._append_pair_sync, conversation_id, user_content, assistant_content
        )

    def _append_pair_sync(
        self, conversation_id: str, user_content: str, assistant_content: str
    ) -> None:
        # Single transaction over both rows. Either both commit or neither does,
        # so a chat round can never be half-persisted (no orphan user-without-reply).
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO conversation_turns "
                    "(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (conversation_id, "user", user_content, now),
                )
                self._conn.execute(
                    "INSERT INTO conversation_turns "
                    "(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (conversation_id, "assistant", assistant_content, now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # ConversationReaderPort
    # ------------------------------------------------------------------

    async def load(self, conversation_id: str, limit: int = 20) -> list[Turn]:
        return await asyncio.to_thread(self._load_sync, conversation_id, limit)

    def _load_sync(self, conversation_id: str, limit: int) -> list[Turn]:
        # Pull the most recent `limit` rows, then reverse to oldest-first so
        # the LLM prompt reads chronologically.
        with self._lock:
            cursor = self._conn.execute(
                "SELECT role, content FROM conversation_turns "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            )
            rows = cursor.fetchall()
        rows.reverse()
        return [Turn(role=row["role"], content=row["content"]) for row in rows]
