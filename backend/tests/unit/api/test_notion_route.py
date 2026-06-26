"""Unit tests for the Notion sync endpoint.

Mocks the NotionPort, EmbeddingPort, and VectorStorePort so the route logic
can be exercised in CI without real credentials or a running Qdrant.

The integration test in tests/integration/test_api.py covers the same route
against real services and skips when NOTION_TOKEN / NOTION_DATABASE_ID are
absent; this file covers the gap left by that skip.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import (
    get_agents_config,
    get_embedding,
    get_notion,
    get_system_config,
    get_vector_store,
)
from app.core.config.models import AgentsConfig, SystemConfig
from app.main import create_app
from app.ports.notion import NotionPage


def _system_config() -> SystemConfig:
    return SystemConfig()


def _agents_config() -> AgentsConfig:
    return AgentsConfig()


@pytest.fixture
def fake_notion():
    """A NotionPort double with sensible defaults; tests can override per-case."""
    notion = AsyncMock()
    notion.list_pages = AsyncMock(
        return_value=[
            NotionPage(id="page-1", title="Alpha", text="", url="https://notion.so/page-1"),
            NotionPage(id="page-2", title="Beta", text="", url="https://notion.so/page-2"),
        ]
    )
    notion.get_page_content = AsyncMock(
        side_effect=lambda page_id: NotionPage(
            id=page_id,
            title=f"Title for {page_id}",
            # Long enough to produce at least one chunk under default chunk_size.
            text=f"Content for {page_id}. " * 50,
            url=f"https://notion.so/{page_id}",
        )
    )
    return notion


@pytest.fixture
def fake_embedding():
    emb = AsyncMock()
    # Return a vector per text passed in, regardless of count.
    emb.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    return emb


@pytest.fixture
def fake_vectorstore():
    vs = AsyncMock()
    vs.upsert = AsyncMock()
    return vs


@pytest.fixture
def test_app(fake_notion, fake_embedding, fake_vectorstore):
    app = create_app()
    app.dependency_overrides[get_system_config] = _system_config
    app.dependency_overrides[get_agents_config] = _agents_config
    app.dependency_overrides[get_notion] = lambda: fake_notion
    app.dependency_overrides[get_embedding] = lambda: fake_embedding
    app.dependency_overrides[get_vector_store] = lambda: fake_vectorstore
    return app


@pytest.fixture
async def client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_sync_chunks_pages_and_upserts(
    client: AsyncClient,
    fake_notion,
    fake_embedding,
    fake_vectorstore,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NOTION_DATABASE_ID", "db-123")

    resp = await client.post("/api/collections/test-coll/sync-notion")

    assert resp.status_code == 200
    body = resp.json()
    assert body["collection"] == "test-coll"
    assert body["pages_synced"] == 2
    assert body["total_chunks"] >= 2  # at least one chunk per page

    # list_pages called once with the database id from the env var.
    fake_notion.list_pages.assert_awaited_once_with("db-123")

    # get_page_content called once per page returned.
    assert fake_notion.get_page_content.await_count == 2

    # One upsert per non-empty page.
    assert fake_vectorstore.upsert.await_count == 2

    # Each upsert must carry matching chunk and vector counts.
    for call in fake_vectorstore.upsert.await_args_list:
        coll, chunks, vectors = call.args
        assert coll == "test-coll"
        assert len(chunks) == len(vectors)
        assert all(c.metadata["source"] == "notion" for c in chunks)


# ---------------------------------------------------------------------------
# Empty-content handling
# ---------------------------------------------------------------------------


async def test_sync_skips_pages_with_empty_text(
    client: AsyncClient,
    fake_notion,
    fake_vectorstore,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NOTION_DATABASE_ID", "db-123")

    fake_notion.get_page_content = AsyncMock(
        side_effect=[
            NotionPage(id="page-1", title="Empty", text="   \n  ", url="https://notion.so/page-1"),
            NotionPage(
                id="page-2",
                title="Has content",
                text="Real content goes here. " * 50,
                url="https://notion.so/page-2",
            ),
        ]
    )

    resp = await client.post("/api/collections/test-coll/sync-notion")

    assert resp.status_code == 200
    body = resp.json()
    # pages_synced counts only pages that produced chunks; the empty page is
    # skipped before upsert.
    assert body["pages_synced"] == 1
    assert fake_vectorstore.upsert.await_count == 1


# ---------------------------------------------------------------------------
# Missing env var
# ---------------------------------------------------------------------------


async def test_sync_returns_400_when_database_id_missing(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)

    resp = await client.post("/api/collections/test-coll/sync-notion")

    assert resp.status_code == 400
    assert "NOTION_DATABASE_ID" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Idempotency: chunk IDs are deterministic per (page, chunk index)
# ---------------------------------------------------------------------------


async def test_sync_produces_deterministic_chunk_ids(
    client: AsyncClient,
    fake_vectorstore,
    monkeypatch: pytest.MonkeyPatch,
):
    """Re-running sync over the same pages produces the same chunk IDs.

    This is the contract that lets unchanged pages overwrite themselves
    cleanly on re-sync rather than producing duplicates. It does NOT
    cover the case where a page shrinks (orphaned chunks at higher
    indexes are not deleted by upsert alone) -- that would need a
    delete-by-page-id sweep before re-upserting."""
    monkeypatch.setenv("NOTION_DATABASE_ID", "db-123")

    await client.post("/api/collections/test-coll/sync-notion")
    first_ids = [
        chunk.id for call in fake_vectorstore.upsert.await_args_list for chunk in call.args[1]
    ]

    fake_vectorstore.upsert.reset_mock()
    await client.post("/api/collections/test-coll/sync-notion")
    second_ids = [
        chunk.id for call in fake_vectorstore.upsert.await_args_list for chunk in call.args[1]
    ]

    assert first_ids == second_ids
    assert len(first_ids) > 0
