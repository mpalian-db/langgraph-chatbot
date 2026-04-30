"""Tests for the conversation-memory orchestration service.

Covers the lazy-on-load summarisation trigger: below threshold no LLM call
fires; above threshold the service generates a new summary, persists it
via the writer port, and returns the trimmed view.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.adapters.conversation.sqlite import SQLiteConversationStore
from app.core.config.models import SummariserConfig
from app.core.operations.conversation_memory import load_with_summary


@pytest.fixture
def store() -> SQLiteConversationStore:
    return SQLiteConversationStore(":memory:")


@pytest.fixture
def summary_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value={
            "text": "MOCK SUMMARY",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 30},
        }
    )
    return llm


def _config(
    *, summarise_threshold: int = 20, keep_recent: int = 10, enabled: bool = True
) -> SummariserConfig:
    return SummariserConfig(
        enabled=enabled,
        summarise_threshold=summarise_threshold,
        keep_recent=keep_recent,
    )


# ---------------------------------------------------------------------------
# Below threshold: no LLM call, no new summary
# ---------------------------------------------------------------------------


async def test_returns_view_unchanged_below_threshold(
    store: SQLiteConversationStore, summary_llm: AsyncMock
):
    # Five turns < threshold of 20 -> service must NOT summarise.
    for i in range(5):
        await store.append("conv-1", "user" if i % 2 == 0 else "assistant", f"turn-{i}")

    view = await load_with_summary(
        "conv-1",
        reader=store,
        writer=store,
        llm=summary_llm,
        config=_config(),
    )

    assert view.summary is None
    assert len(view.recent) == 5
    assert [t.content for t in view.recent] == [f"turn-{i}" for i in range(5)]
    summary_llm.complete.assert_not_called()


async def test_returns_existing_summary_below_threshold(
    store: SQLiteConversationStore, summary_llm: AsyncMock
):
    """If a prior summary already exists, it should pass through unchanged
    when the post-summary tail is below threshold."""
    await store.append("conv-1", "user", "old turn")
    _, before = await store.load_summary_and_turns("conv-1")
    await store.upsert_summary("conv-1", "the old summary", before[-1].id)
    await store.append("conv-1", "user", "new turn 1")
    await store.append("conv-1", "assistant", "new turn 2")

    view = await load_with_summary(
        "conv-1",
        reader=store,
        writer=store,
        llm=summary_llm,
        config=_config(),
    )

    assert view.summary == "the old summary"
    assert [t.content for t in view.recent] == ["new turn 1", "new turn 2"]
    summary_llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Above threshold: summarisation triggers
# ---------------------------------------------------------------------------


async def test_triggers_summarisation_when_threshold_exceeded(
    store: SQLiteConversationStore, summary_llm: AsyncMock
):
    """21 turns with threshold 20 + keep_recent 10: service summarises the
    first 11, keeps the last 10 verbatim, persists the new summary."""
    for i in range(21):
        await store.append("conv-1", "user" if i % 2 == 0 else "assistant", f"turn-{i}")

    view = await load_with_summary(
        "conv-1",
        reader=store,
        writer=store,
        llm=summary_llm,
        config=_config(summarise_threshold=20, keep_recent=10),
    )

    # LLM was called exactly once with the older 11 turns in the prompt.
    assert summary_llm.complete.await_count == 1
    user_content = summary_llm.complete.call_args.kwargs["messages"][0]["content"]
    assert "turn-0" in user_content
    assert "turn-10" in user_content
    # The most recent 10 must NOT have been summarised.
    assert "turn-11" not in user_content

    # Returned view: new summary + last 10 turns verbatim.
    assert view.summary == "MOCK SUMMARY"
    assert len(view.recent) == 10
    assert [t.content for t in view.recent] == [f"turn-{i}" for i in range(11, 21)]

    # Summary was persisted with a boundary id matching the 11th turn.
    summary, post = await store.load_summary_and_turns("conv-1")
    assert summary == "MOCK SUMMARY"
    # After upsert, only 10 turns sit past the boundary.
    assert len(post) == 10


async def test_summariser_integrates_prior_summary_rather_than_restarting(
    store: SQLiteConversationStore, summary_llm: AsyncMock
):
    """When a prior summary exists, the prompt must include it so the
    summariser folds new turns INTO the existing summary rather than
    starting fresh and losing distant context."""
    # Seed a prior summary at turn 0.
    await store.append("conv-1", "user", "ancient turn")
    _, before = await store.load_summary_and_turns("conv-1")
    await store.upsert_summary("conv-1", "PRIOR SUMMARY TEXT", before[-1].id)

    # Add 21 more turns post-boundary to trigger another round.
    for i in range(21):
        await store.append("conv-1", "user" if i % 2 == 0 else "assistant", f"new-{i}")

    await load_with_summary(
        "conv-1",
        reader=store,
        writer=store,
        llm=summary_llm,
        config=_config(),
    )

    # The prior summary appears in the LLM prompt verbatim.
    user_content = summary_llm.complete.call_args.kwargs["messages"][0]["content"]
    assert "PRIOR SUMMARY TEXT" in user_content
    assert "Prior summary" in user_content


# ---------------------------------------------------------------------------
# Disabled: never summarise regardless of length
# ---------------------------------------------------------------------------


async def test_disabled_summariser_never_calls_llm(
    store: SQLiteConversationStore, summary_llm: AsyncMock
):
    """Operator can turn the summariser off entirely. With enabled=False
    even a 100-turn conversation should not trigger an LLM call."""
    for i in range(100):
        await store.append("conv-1", "user", f"turn-{i}")

    view = await load_with_summary(
        "conv-1",
        reader=store,
        writer=store,
        llm=summary_llm,
        config=_config(enabled=False),
    )

    summary_llm.complete.assert_not_called()
    # All 100 turns come back verbatim -- caller takes the full prompt cost.
    assert len(view.recent) == 100
