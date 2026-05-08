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

from app.ports.conversation import ConversationOverview, Role, StoredTurn, Turn

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

-- Rolling summary per conversation. One row per conversation_id, replaced
-- in place by the summarisation service when it triggers (lazy, on read).
-- summarised_through_turn_id marks the boundary: turns with id > that value
-- are NOT in the summary and remain verbatim in conversation_turns.
CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    summarised_through_turn_id INTEGER NOT NULL,
    updated_at REAL NOT NULL
);

-- Conversation-level metadata (title, timestamps). Separate from turns
-- and summaries so read-only metadata reads stay cheap and so a title is
-- stable across summarisation rounds (deriving it at read time from the
-- earliest verbatim turn would cause the title to silently change once
-- old turns get folded into the summary).
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

# How many characters of the first user message to use as the auto-title.
# Long enough to be meaningful, short enough not to sprawl the sidebar.
_AUTO_TITLE_MAX_CHARS = 60


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
            self._upsert_conversation_metadata(conversation_id, role, content)
            self._conn.execute(
                "INSERT INTO conversation_turns "
                "(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, time.time()),
            )
            self._conn.commit()

    def _upsert_conversation_metadata(self, conversation_id: str, role: Role, content: str) -> None:
        """Maintain the conversations metadata row for `conversation_id`.

        Behaviour:
          * If the row doesn't exist, create it with `title` set to the
            first `_AUTO_TITLE_MAX_CHARS` of `content` when the incoming
            turn is a user message, else NULL.
          * If the row exists, refresh `updated_at` and only set `title`
            if it was NULL (so the first user turn wins; later messages
            never silently rename a conversation).

        Caller must hold the connection lock; this method does not commit
        -- it's part of a larger transaction in append/append_pair."""
        now = time.time()
        candidate_title = content[:_AUTO_TITLE_MAX_CHARS] if role == "user" else None
        self._conn.execute(
            "INSERT INTO conversations "
            "(conversation_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(conversation_id) DO UPDATE SET "
            "  updated_at = excluded.updated_at, "
            "  title = COALESCE(conversations.title, excluded.title)",
            (conversation_id, candidate_title, now, now),
        )

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
                # Maintain the conversations metadata row (title from the
                # user content on first turn, updated_at refreshed on every
                # turn). All three writes (metadata + user turn + assistant
                # turn) commit atomically below.
                self._upsert_conversation_metadata(conversation_id, "user", user_content)
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

    async def load_summary_and_turns(
        self, conversation_id: str
    ) -> tuple[str | None, list[StoredTurn]]:
        return await asyncio.to_thread(self._load_summary_and_turns_sync, conversation_id)

    def _load_summary_and_turns_sync(
        self, conversation_id: str
    ) -> tuple[str | None, list[StoredTurn]]:
        with self._lock:
            summary_row = self._conn.execute(
                "SELECT summary, summarised_through_turn_id "
                "FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if summary_row is not None:
                summary_text = summary_row["summary"]
                boundary_id = summary_row["summarised_through_turn_id"]
                turn_rows = self._conn.execute(
                    "SELECT id, role, content FROM conversation_turns "
                    "WHERE conversation_id = ? AND id > ? ORDER BY id ASC",
                    (conversation_id, boundary_id),
                ).fetchall()
            else:
                summary_text = None
                turn_rows = self._conn.execute(
                    "SELECT id, role, content FROM conversation_turns "
                    "WHERE conversation_id = ? ORDER BY id ASC",
                    (conversation_id,),
                ).fetchall()
        turns = [
            StoredTurn(id=row["id"], role=row["role"], content=row["content"]) for row in turn_rows
        ]
        return summary_text, turns

    async def list_conversations(self) -> list[ConversationOverview]:
        return await asyncio.to_thread(self._list_conversations_sync)

    def _list_conversations_sync(self) -> list[ConversationOverview]:
        # SQLite doesn't support FULL OUTER JOIN, so we UNION ALL two
        # GROUP-BY views -- one per source table -- and re-aggregate. This
        # guarantees a conversation surfaces in the list whether it has
        # only turns, only a summary (rare but reachable), or both.
        # Re-aggregation pattern:
        #   * SUM over turn_count: turns-side rows contribute the real count;
        #     summary-side rows contribute 0.
        #   * MAX over has_summary: 1 if any source row sets it.
        #   * MAX over last_at: the most recent activity, taking either the
        #     last turn's created_at or the summary's updated_at, whichever
        #     is newer. Ordering by this column means "most recently active
        #     first" works for both turn updates and summary updates.
        # The outer LEFT JOIN brings in the title from the conversations
        # metadata table; a missing metadata row means title is NULL.
        sql = """
            SELECT
                agg.conversation_id,
                agg.turn_count,
                agg.has_summary,
                agg.last_at,
                conv.title
            FROM (
                SELECT
                    conversation_id,
                    SUM(turn_count) AS turn_count,
                    MAX(has_summary) AS has_summary,
                    MAX(last_at) AS last_at
                FROM (
                    SELECT
                        conversation_id,
                        COUNT(*) AS turn_count,
                        0 AS has_summary,
                        MAX(created_at) AS last_at
                    FROM conversation_turns
                    GROUP BY conversation_id
                    UNION ALL
                    SELECT
                        conversation_id,
                        0 AS turn_count,
                        1 AS has_summary,
                        updated_at AS last_at
                    FROM conversation_summaries
                )
                GROUP BY conversation_id
            ) AS agg
            LEFT JOIN conversations AS conv
                ON conv.conversation_id = agg.conversation_id
            ORDER BY agg.last_at DESC
        """
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [
            ConversationOverview(
                conversation_id=row["conversation_id"],
                title=row["title"],
                turn_count=row["turn_count"],
                has_summary=bool(row["has_summary"]),
                last_updated_at=row["last_at"],
            )
            for row in rows
        ]

    async def get_conversation_title(self, conversation_id: str) -> str | None:
        return await asyncio.to_thread(self._get_conversation_title_sync, conversation_id)

    def _get_conversation_title_sync(self, conversation_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT title FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return row["title"] if row else None

    async def upsert_summary(
        self,
        conversation_id: str,
        summary: str,
        summarised_through_turn_id: int,
    ) -> None:
        await asyncio.to_thread(
            self._upsert_summary_sync,
            conversation_id,
            summary,
            summarised_through_turn_id,
        )

    def _upsert_summary_sync(
        self,
        conversation_id: str,
        summary: str,
        summarised_through_turn_id: int,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversation_summaries "
                "(conversation_id, summary, summarised_through_turn_id, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET "
                "summary = excluded.summary, "
                "summarised_through_turn_id = excluded.summarised_through_turn_id, "
                "updated_at = excluded.updated_at",
                (conversation_id, summary, summarised_through_turn_id, time.time()),
            )
            self._conn.commit()

    async def delete_conversation(self, conversation_id: str) -> None:
        await asyncio.to_thread(self._delete_conversation_sync, conversation_id)

    def _delete_conversation_sync(self, conversation_id: str) -> None:
        # Atomic across all three tables: turns, summaries, and the
        # metadata row. A single transaction; if any DELETE fails, the
        # others roll back so a partially-deleted conversation can never
        # surface in subsequent reads (orphan title pointing at
        # nothing, or surviving summary referencing deleted turns).
        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM conversation_turns WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self._conn.execute(
                    "DELETE FROM conversation_summaries WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self._conn.execute(
                    "DELETE FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
