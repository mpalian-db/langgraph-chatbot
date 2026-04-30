"""Tests for the conversation-introspection endpoints.

Uses a real in-memory SQLite store via dependency_overrides -- the routes
are thin wrappers over the port, so testing through the real adapter
catches both the route serialisation contract and the adapter behaviour
in one pass.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.conversation.sqlite import SQLiteConversationStore
from app.api.dependencies import (
    get_agents_config,
    get_conversation_reader,
    get_system_config,
)
from app.core.config.models import AgentsConfig, SystemConfig
from app.main import create_app


@pytest.fixture
def store() -> SQLiteConversationStore:
    return SQLiteConversationStore(":memory:")


@pytest.fixture
def test_app(store: SQLiteConversationStore):
    app = create_app()
    app.dependency_overrides[get_system_config] = lambda: SystemConfig()
    app.dependency_overrides[get_agents_config] = lambda: AgentsConfig()
    app.dependency_overrides[get_conversation_reader] = lambda: store
    return app


@pytest.fixture
async def client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/conversations
# ---------------------------------------------------------------------------


async def test_list_returns_empty_array_when_no_conversations(client: AsyncClient):
    resp = await client.get("/api/conversations")

    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_returns_overview_per_conversation(
    client: AsyncClient, store: SQLiteConversationStore
):
    await store.append("conv-A", "user", "hello")
    await store.append("conv-A", "assistant", "hi back")
    await store.append("conv-B", "user", "different chat")
    await store.upsert_summary("conv-A", "summary", 0)

    resp = await client.get("/api/conversations")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    by_id = {row["conversation_id"]: row for row in body}
    assert by_id["conv-A"]["turn_count"] == 2
    assert by_id["conv-A"]["has_summary"] is True
    assert by_id["conv-A"]["last_updated_at"] is not None
    assert by_id["conv-B"]["turn_count"] == 1
    assert by_id["conv-B"]["has_summary"] is False


# ---------------------------------------------------------------------------
# GET /api/conversations/{id}
# ---------------------------------------------------------------------------


async def test_detail_returns_summary_and_turns(
    client: AsyncClient, store: SQLiteConversationStore
):
    await store.append("conv-1", "user", "first")
    await store.append("conv-1", "assistant", "first reply")
    _, before = await store.load_summary_and_turns("conv-1")
    await store.upsert_summary("conv-1", "context summary", before[-1].id)
    await store.append("conv-1", "user", "second")
    await store.append("conv-1", "assistant", "second reply")

    resp = await client.get("/api/conversations/conv-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "conv-1"
    assert body["summary"] == "context summary"
    # Only post-boundary turns -- the summarised ones are folded into the summary.
    assert [(t["role"], t["content"]) for t in body["turns"]] == [
        ("user", "second"),
        ("assistant", "second reply"),
    ]


async def test_detail_returns_404_for_unknown_conversation(client: AsyncClient):
    resp = await client.get("/api/conversations/never-seen")

    assert resp.status_code == 404
    assert "never-seen" in resp.json()["detail"]


async def test_detail_returns_200_with_null_summary_when_only_turns_exist(
    client: AsyncClient, store: SQLiteConversationStore
):
    """A fresh conversation has turns but no summary yet. The detail
    endpoint should still return 200 -- 404 is only for genuinely
    unknown conversation_ids."""
    await store.append("conv-1", "user", "just one turn")

    resp = await client.get("/api/conversations/conv-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"] is None
    assert len(body["turns"]) == 1
