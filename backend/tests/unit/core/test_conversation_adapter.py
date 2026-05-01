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


async def test_load_summary_and_turns_returns_none_summary_for_fresh_conversation(
    store: SQLiteConversationStore,
):
    """No summary exists yet -- caller gets None and all turns since the
    beginning of the conversation."""
    await store.append("conv-1", "user", "first")
    await store.append("conv-1", "assistant", "second")

    summary, turns = await store.load_summary_and_turns("conv-1")

    assert summary is None
    assert [(t.role, t.content) for t in turns] == [
        ("user", "first"),
        ("assistant", "second"),
    ]
    # Stored turns expose ids -- needed by the summariser to mark a boundary.
    assert all(t.id > 0 for t in turns)
    assert turns[0].id < turns[1].id


async def test_upsert_summary_creates_then_replaces(store: SQLiteConversationStore):
    """First upsert creates the row; the second replaces it. Boundary id
    moves forward as more turns get folded into the summary."""
    await store.append("conv-1", "user", "old")
    _, turns = await store.load_summary_and_turns("conv-1")
    first_boundary = turns[-1].id

    await store.upsert_summary("conv-1", "summary v1", first_boundary)

    summary, after = await store.load_summary_and_turns("conv-1")
    assert summary == "summary v1"
    # Nothing past the boundary -- that "old" turn is now folded in.
    assert after == []

    # Add more turns; trigger a new summary that supersedes the first.
    await store.append("conv-1", "user", "newer")
    await store.append("conv-1", "assistant", "newest")
    _, all_since = await store.load_summary_and_turns("conv-1")
    second_boundary = all_since[-1].id

    await store.upsert_summary("conv-1", "summary v2", second_boundary)

    summary, after = await store.load_summary_and_turns("conv-1")
    assert summary == "summary v2"
    assert after == []


async def test_load_summary_and_turns_returns_only_post_boundary_turns(
    store: SQLiteConversationStore,
):
    """After summarisation, the summary boundary masks earlier turns from
    the load. Only turns added AFTER the boundary should surface."""
    await store.append("conv-1", "user", "old-1")
    await store.append("conv-1", "assistant", "old-2")
    _, before = await store.load_summary_and_turns("conv-1")
    boundary = before[-1].id

    await store.upsert_summary("conv-1", "summary of old", boundary)

    # Now add fresh turns past the boundary.
    await store.append("conv-1", "user", "new-1")
    await store.append("conv-1", "assistant", "new-2")

    summary, recent = await store.load_summary_and_turns("conv-1")

    assert summary == "summary of old"
    assert [(t.role, t.content) for t in recent] == [
        ("user", "new-1"),
        ("assistant", "new-2"),
    ]


async def test_list_conversations_returns_overview_with_metadata(
    store: SQLiteConversationStore,
):
    """The list endpoint needs lightweight aggregate data: turn_count,
    has_summary, last_updated_at -- without pulling turn content."""
    await store.append("conv-A", "user", "alpha-1")
    await store.append("conv-A", "assistant", "alpha-2")
    await store.append("conv-B", "user", "bravo-only")
    await store.upsert_summary("conv-A", "summary of alpha", 0)

    overviews = await store.list_conversations()

    assert len(overviews) == 2
    by_id = {o.conversation_id: o for o in overviews}
    assert by_id["conv-A"].turn_count == 2
    assert by_id["conv-A"].has_summary is True
    assert by_id["conv-A"].last_updated_at is not None
    assert by_id["conv-B"].turn_count == 1
    assert by_id["conv-B"].has_summary is False


async def test_list_conversations_orders_most_recent_first(
    store: SQLiteConversationStore,
):
    """Ordering pins the debug-tooling expectation: the conversation a
    developer last touched should appear at the top of the list."""
    import time as _time

    await store.append("older", "user", "x")
    _time.sleep(0.01)  # SQLite created_at is a real-number epoch; force ordering
    await store.append("newer", "user", "y")

    overviews = await store.list_conversations()

    assert overviews[0].conversation_id == "newer"
    assert overviews[1].conversation_id == "older"


async def test_list_conversations_returns_empty_when_no_turns(
    store: SQLiteConversationStore,
):
    overviews = await store.list_conversations()
    assert overviews == []


async def test_list_conversations_includes_summary_only_rows(
    store: SQLiteConversationStore,
):
    """A conversation that exists only as a summary row (no turns currently
    survive past the boundary) must still appear in the list. Otherwise
    the detail endpoint and the list endpoint disagree about what counts
    as an existing conversation."""
    # Construct a summary-only state by upserting a summary without
    # appending any post-boundary turns.
    await store.upsert_summary("summary-only-conv", "the only summary", 0)

    overviews = await store.list_conversations()

    assert len(overviews) == 1
    only = overviews[0]
    assert only.conversation_id == "summary-only-conv"
    assert only.turn_count == 0
    assert only.has_summary is True
    assert only.last_updated_at is not None  # the summary's updated_at


async def test_summary_is_isolated_per_conversation(store: SQLiteConversationStore):
    """Two conversations' summaries do not interfere with each other."""
    await store.upsert_summary("conv-A", "alpha summary", 0)
    await store.upsert_summary("conv-B", "bravo summary", 0)

    a_summary, _ = await store.load_summary_and_turns("conv-A")
    b_summary, _ = await store.load_summary_and_turns("conv-B")

    assert a_summary == "alpha summary"
    assert b_summary == "bravo summary"


# ---------------------------------------------------------------------------
# Auto-title: first user content seeds the conversation title; later
# messages don't silently rename it.
# ---------------------------------------------------------------------------


async def test_first_user_turn_sets_conversation_title(
    store: SQLiteConversationStore,
):
    """The first user message becomes the auto-title (truncated to the
    configured maximum). Pin the contract: title appears in
    list_conversations after the first append."""
    await store.append("conv-1", "user", "What is LangGraph?")

    overviews = await store.list_conversations()

    assert overviews[0].title == "What is LangGraph?"


async def test_long_first_user_content_is_truncated_for_title(
    store: SQLiteConversationStore,
):
    """A 200-char first message must produce a title at most
    _AUTO_TITLE_MAX_CHARS (60) long, so a sidebar row stays readable."""
    long = "A " * 100  # 200 characters
    await store.append("conv-1", "user", long)

    overviews = await store.list_conversations()

    title = overviews[0].title or ""
    assert 0 < len(title) <= 60


async def test_title_is_stable_across_subsequent_messages(
    store: SQLiteConversationStore,
):
    """The title is set once at creation. Later user turns must NOT
    silently rename the conversation -- otherwise the sidebar entry
    keeps shifting and the user can't recognise their own threads."""
    await store.append("conv-1", "user", "Original title")
    await store.append("conv-1", "assistant", "reply")
    await store.append("conv-1", "user", "completely different topic")

    overviews = await store.list_conversations()

    assert overviews[0].title == "Original title"


async def test_assistant_only_first_turn_leaves_title_null(
    store: SQLiteConversationStore,
):
    """Defensive case: a conversation that opens with an assistant turn
    (e.g. system-injected) shouldn't get a title from that content. The
    next user turn is what should set it."""
    await store.append("conv-1", "assistant", "Welcome!")

    overviews = await store.list_conversations()
    assert overviews[0].title is None

    # User message arrives -- now the title is set.
    await store.append("conv-1", "user", "Hello there")
    overviews = await store.list_conversations()
    assert overviews[0].title == "Hello there"


async def test_append_pair_sets_title_from_user_content(
    store: SQLiteConversationStore,
):
    """append_pair is the dominant call site; the title must come from
    the user side of the pair, not the assistant side."""
    await store.append_pair("conv-1", "user query here", "assistant reply")

    overviews = await store.list_conversations()
    assert overviews[0].title == "user query here"


async def test_get_conversation_title_returns_value(
    store: SQLiteConversationStore,
):
    await store.append("conv-1", "user", "ask something")
    title = await store.get_conversation_title("conv-1")
    assert title == "ask something"


async def test_get_conversation_title_returns_none_for_unknown(
    store: SQLiteConversationStore,
):
    title = await store.get_conversation_title("never-seen")
    assert title is None


async def test_delete_conversation_clears_metadata_row_too(
    store: SQLiteConversationStore,
):
    """The metadata row must be deleted alongside turns and the summary,
    so the deleted conversation can never resurface in list_conversations
    via an orphan title."""
    await store.append("conv-1", "user", "to be deleted")

    await store.delete_conversation("conv-1")

    overviews = await store.list_conversations()
    assert overviews == []
    assert await store.get_conversation_title("conv-1") is None


# ---------------------------------------------------------------------------
# delete_conversation: atomic over both tables
# ---------------------------------------------------------------------------


async def test_delete_conversation_removes_turns_and_summary(
    store: SQLiteConversationStore,
):
    """Both turn rows and the summary row must be cleared in one go."""
    await store.append("conv-1", "user", "first")
    await store.append("conv-1", "assistant", "reply")
    _, turns = await store.load_summary_and_turns("conv-1")
    await store.upsert_summary("conv-1", "the summary", turns[-1].id)

    await store.delete_conversation("conv-1")

    summary, recent = await store.load_summary_and_turns("conv-1")
    assert summary is None
    assert recent == []


async def test_delete_conversation_is_idempotent_for_unknown_id(
    store: SQLiteConversationStore,
):
    """Calling delete on a never-seen conversation must not raise -- the
    caller's desired end state ('absent') is already true."""
    await store.delete_conversation("never-seen")  # must not raise

    # Storage is still empty after the no-op.
    overviews = await store.list_conversations()
    assert overviews == []


async def test_delete_conversation_does_not_touch_other_conversations(
    store: SQLiteConversationStore,
):
    """Isolation: deleting conv-A must leave conv-B's data untouched."""
    await store.append("conv-A", "user", "alpha")
    await store.append("conv-B", "user", "bravo")
    await store.upsert_summary("conv-B", "B's summary", 0)

    await store.delete_conversation("conv-A")

    # A is gone.
    a_summary, a_turns = await store.load_summary_and_turns("conv-A")
    assert a_summary is None and a_turns == []

    # B is intact.
    b_summary, b_turns = await store.load_summary_and_turns("conv-B")
    assert b_summary == "B's summary"
    assert len(b_turns) == 1
    assert b_turns[0].content == "bravo"


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
