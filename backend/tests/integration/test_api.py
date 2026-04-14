"""Integration tests for the FastAPI endpoints.

These require Qdrant (Docker) and Ollama running locally.
Run with: pytest tests/integration/ -v
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

pytestmark = pytest.mark.integration


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_endpoint(client: AsyncClient) -> None:
    response = await client.get("/api/system/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_config_endpoint(client: AsyncClient) -> None:
    response = await client.get("/api/system/config")
    assert response.status_code == 200
    data = response.json()
    assert "environment" in data
    assert "llm" in data
    assert "agents" in data


async def test_list_collections(client: AsyncClient) -> None:
    response = await client.get("/api/collections")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


async def test_chat_endpoint(client: AsyncClient) -> None:
    response = await client.post("/api/chat", json={"query": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "route" in data
