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
    get_conversation_writer,
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
    app.dependency_overrides[get_conversation_writer] = lambda: store
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
    # Auto-title from first user turn -- pinned at the route boundary so
    # a regression in either the adapter SQL or the route serialisation
    # would surface here.
    assert by_id["conv-A"]["title"] == "hello"
    assert by_id["conv-B"]["turn_count"] == 1
    assert by_id["conv-B"]["has_summary"] is False
    assert by_id["conv-B"]["title"] == "different chat"


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
    # Title is the first user turn's content, stable across summarisation:
    # even though "first" is now folded into the summary, the title still
    # comes from it.
    assert body["title"] == "first"
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


async def test_list_and_detail_agree_on_summary_only_conversation(
    client: AsyncClient, store: SQLiteConversationStore
):
    """Regression: the list endpoint previously dropped summary-only rows
    while detail returned 200 for them. Pin that they now agree -- both
    surfaces report the same conversation as existing."""
    await store.upsert_summary("summary-only-conv", "all of it", 0)

    list_resp = await client.get("/api/conversations")
    detail_resp = await client.get("/api/conversations/summary-only-conv")

    list_ids = [row["conversation_id"] for row in list_resp.json()]
    assert "summary-only-conv" in list_ids
    assert detail_resp.status_code == 200
    assert detail_resp.json()["summary"] == "all of it"


# ---------------------------------------------------------------------------
# DELETE /api/conversations/{id}
# ---------------------------------------------------------------------------


async def test_delete_conversation_removes_turns_and_summary(
    client: AsyncClient, store: SQLiteConversationStore
):
    await store.append("conv-doomed", "user", "first")
    await store.append("conv-doomed", "assistant", "reply")
    await store.upsert_summary("conv-doomed", "to be deleted", 0)

    resp = await client.delete("/api/conversations/conv-doomed")

    assert resp.status_code == 204
    summary, turns = await store.load_summary_and_turns("conv-doomed")
    assert summary is None and turns == []


async def test_delete_conversation_is_idempotent_for_unknown_id(
    client: AsyncClient,
):
    """Repeating DELETE on a never-seen id stays at 204. The desired end
    state ('absent') is already true; surfacing 404 would force callers
    to special-case the harmless no-op."""
    resp = await client.delete("/api/conversations/never-seen")

    assert resp.status_code == 204


async def test_delete_does_not_touch_other_conversations(
    client: AsyncClient, store: SQLiteConversationStore
):
    await store.append("conv-A", "user", "alpha")
    await store.append("conv-B", "user", "bravo")

    resp = await client.delete("/api/conversations/conv-A")
    assert resp.status_code == 204

    list_ids = [row["conversation_id"] for row in (await client.get("/api/conversations")).json()]
    assert "conv-A" not in list_ids
    assert "conv-B" in list_ids


async def test_delete_then_get_returns_404(client: AsyncClient, store: SQLiteConversationStore):
    """The lifecycle: create via append, delete via DELETE, then GET 404s.
    Pins that the introspection endpoint sees the deletion immediately."""
    await store.append("conv-x", "user", "hi")
    detail_before = await client.get("/api/conversations/conv-x")
    assert detail_before.status_code == 200

    delete_resp = await client.delete("/api/conversations/conv-x")
    assert delete_resp.status_code == 204

    detail_after = await client.get("/api/conversations/conv-x")
    assert detail_after.status_code == 404
