"""Unit tests for the GET / DELETE single-document endpoints.

Mocks the VectorStorePort via dependency_overrides; no Qdrant or Vectorize
required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import (
    get_collection_port,
    get_embedding,
    get_storage,
    get_system_config,
    get_vector_store,
)
from app.core.config.models import SystemConfig
from app.core.models.types import Chunk
from app.main import create_app


@pytest.fixture
def fake_vectorstore():
    vs = AsyncMock()
    vs.get_chunk = AsyncMock(return_value=None)
    vs.delete = AsyncMock()
    return vs


@pytest.fixture
def test_app(fake_vectorstore):
    app = create_app()
    app.dependency_overrides[get_system_config] = lambda: SystemConfig()
    app.dependency_overrides[get_vector_store] = lambda: fake_vectorstore
    app.dependency_overrides[get_collection_port] = lambda: AsyncMock()
    app.dependency_overrides[get_embedding] = lambda: AsyncMock()
    app.dependency_overrides[get_storage] = lambda: AsyncMock()
    return app


@pytest.fixture
async def client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /collections/{coll}/documents/{id}
# ---------------------------------------------------------------------------


async def test_get_document_returns_chunk_when_found(client: AsyncClient, fake_vectorstore):
    fake_vectorstore.get_chunk = AsyncMock(
        return_value=Chunk(
            id="c1",
            text="LangGraph is stateful.",
            collection="docs",
            metadata={"source": "notion"},
        )
    )

    resp = await client.get("/api/collections/docs/documents/c1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "c1"
    assert body["text"] == "LangGraph is stateful."
    assert body["collection"] == "docs"
    assert body["metadata"] == {"source": "notion"}
    fake_vectorstore.get_chunk.assert_awaited_once_with("docs", "c1")


async def test_get_document_returns_404_when_missing(client: AsyncClient, fake_vectorstore):
    fake_vectorstore.get_chunk = AsyncMock(return_value=None)

    resp = await client.get("/api/collections/docs/documents/never-existed")

    assert resp.status_code == 404
    assert "never-existed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /collections/{coll}/documents/{id}
# ---------------------------------------------------------------------------


async def test_delete_document_returns_204_and_calls_delete(client: AsyncClient, fake_vectorstore):
    resp = await client.delete("/api/collections/docs/documents/c1")

    assert resp.status_code == 204
    fake_vectorstore.delete.assert_awaited_once_with("docs", ["c1"])


async def test_delete_document_is_idempotent_for_unknown_id(client: AsyncClient, fake_vectorstore):
    """Repeated DELETE on an already-removed id must still return 204.
    The vectorstore.delete contract is "remove ids if present"; absence
    is not an error."""
    resp = await client.delete("/api/collections/docs/documents/already-gone")

    assert resp.status_code == 204
    fake_vectorstore.delete.assert_awaited_once_with("docs", ["already-gone"])
