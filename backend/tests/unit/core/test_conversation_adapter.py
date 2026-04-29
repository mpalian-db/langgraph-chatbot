"""Unit tests for the SQLite conversation adapter.

Uses an in-memory SQLite database (`:memory:`) so tests are hermetic and fast.
"""

from __future__ import annotations

import pytest

from app.adapters.conversation.sqlite import SQLiteConversationStore


@pytest.fixture
def store() -> SQLiteConversationStore:
    return SQLiteConversationStore(":memory:")


# ---------------------------------------------------------------------------
# append + load round-trip
# ---------------------------------------------------------------------------


async def test_append_then_load_returns_turns_in_chronological_order(
    store: SQLiteConversationStore,
):
    await store.append("conv-1", "user", "hello")
    await store.append("conv-1", "assistant", "hi there")
    await store.append("conv-1", "user", "how are you?")

    turns = await store.load("conv-1")

    assert len(turns) == 3
    assert turns[0].role == "user" and turns[0].content == "hello"
    assert turns[1].role == "assistant" and turns[1].content == "hi there"
    assert turns[2].role == "user" and turns[2].content == "how are you?"


async def test_load_returns_empty_list_for_unknown_conversation(
    store: SQLiteConversationStore,
):
    """A first-turn user implicitly creates a conversation; we never raise on
    unknown ids."""
    turns = await store.load("never-seen")
    assert turns == []


# ---------------------------------------------------------------------------
# Sliding window: limit truncates oldest, preserves order
# ---------------------------------------------------------------------------


async def test_load_limit_keeps_most_recent_n_turns(store: SQLiteConversationStore):
    for i in range(10):
        await store.append("conv-1", "user", f"turn-{i}")

    turns = await store.load("conv-1", limit=3)

    # Last 3 turns, oldest-first.
    assert [t.content for t in turns] == ["turn-7", "turn-8", "turn-9"]


async def test_load_with_limit_larger_than_history_returns_all(
    store: SQLiteConversationStore,
):
    await store.append("conv-1", "user", "only one")

    turns = await store.load("conv-1", limit=100)

    assert len(turns) == 1


# ---------------------------------------------------------------------------
# Conversation isolation
# ---------------------------------------------------------------------------


async def test_load_isolates_by_conversation_id(store: SQLiteConversationStore):
    await store.append("conv-A", "user", "alpha")
    await store.append("conv-B", "user", "bravo")
    await store.append("conv-A", "assistant", "alpha-reply")

    a = await store.load("conv-A")
    b = await store.load("conv-B")

    assert [t.content for t in a] == ["alpha", "alpha-reply"]
    assert [t.content for t in b] == ["bravo"]


# ---------------------------------------------------------------------------
# Persistence on disk
# ---------------------------------------------------------------------------


async def test_disk_backed_store_persists_across_instances(tmp_path):
    """A new adapter pointed at the same file should see prior turns. Pins
    the persistence contract for restart durability."""
    db = tmp_path / "convos.sqlite"

    store_a = SQLiteConversationStore(db)
    await store_a.append("conv-1", "user", "before restart")

    store_b = SQLiteConversationStore(db)
    turns = await store_b.load("conv-1")

    assert [t.content for t in turns] == ["before restart"]


async def test_role_check_constraint_rejects_invalid_role(
    store: SQLiteConversationStore,
):
    """SQLite's CHECK constraint guards against typos / mis-injection at the
    storage boundary, so a buggy caller cannot poison the table."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        # Bypass the typed protocol to confirm the storage layer's defence.
        await store.append("conv-1", "system", "should not be allowed")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# append_pair: atomicity
# ---------------------------------------------------------------------------


async def test_append_pair_persists_both_rows_in_order(store: SQLiteConversationStore):
    """The (user, assistant) pair must land as two rows, user first, then
    assistant, both readable in a subsequent load."""
    await store.append_pair("conv-1", "user query", "assistant reply")

    turns = await store.load("conv-1")

    assert [t.role for t in turns] == ["user", "assistant"]
    assert [t.content for t in turns] == ["user query", "assistant reply"]


async def test_append_pair_is_atomic_on_failure(store: SQLiteConversationStore):
    """If the second insert fails, the first must be rolled back. Otherwise
    the conversation is left with an orphan user-without-reply row that
    would corrupt every future history load.

    Trigger the failure by passing None as `assistant_content`, which
    violates the schema's `content TEXT NOT NULL` constraint on the second
    insert. The first insert (`user_content="user q"`) is valid on its own;
    only the rollback contract prevents it from persisting."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        await store.append_pair("conv-1", "user q", None)  # type: ignore[arg-type]

    # The failed pair must have rolled back -- no orphan user row.
    turns = await store.load("conv-1")
    assert turns == []
