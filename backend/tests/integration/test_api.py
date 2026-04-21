"""Integration tests for the FastAPI endpoints.

These require Qdrant (Docker) and Ollama running locally.
Run with: pytest tests/integration/ -v
"""

from __future__ import annotations

import time
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def temp_collection(client: AsyncClient):
    """Create a uniquely named collection and delete it after the test."""
    name = f"test-{uuid.uuid4().hex[:8]}"
    resp = await client.post("/api/collections", json={"name": name, "vector_size": 768})
    assert resp.status_code == 201, resp.text
    yield name
    await client.delete(f"/api/collections/{name}")


# ---------------------------------------------------------------------------
# System endpoints (pre-existing, kept for regression)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Collection CRUD
# ---------------------------------------------------------------------------


async def test_list_collections(client: AsyncClient) -> None:
    response = await client.get("/api/collections")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


async def test_create_and_delete_collection(client: AsyncClient) -> None:
    name = f"test-{uuid.uuid4().hex[:8]}"

    create_resp = await client.post("/api/collections", json={"name": name, "vector_size": 768})
    assert create_resp.status_code == 201
    assert create_resp.json()["name"] == name

    stats_resp = await client.get(f"/api/collections/{name}")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["name"] == name

    delete_resp = await client.delete(f"/api/collections/{name}")
    assert delete_resp.status_code == 204


async def test_get_stats_for_nonexistent_collection(client: AsyncClient) -> None:
    resp = await client.get("/api/collections/does-not-exist-xyz")
    assert resp.status_code == 404


async def test_rebuild_collection(client: AsyncClient, temp_collection: str) -> None:
    resp = await client.post(f"/api/collections/{temp_collection}/rebuild")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == temp_collection
    assert data["status"] == "rebuilt"

    # Collection should still be queryable after rebuild.
    stats_resp = await client.get(f"/api/collections/{temp_collection}")
    assert stats_resp.status_code == 200


async def test_rebuild_creates_collection_if_missing(client: AsyncClient) -> None:
    name = f"test-rebuild-{uuid.uuid4().hex[:8]}"
    try:
        resp = await client.post(f"/api/collections/{name}/rebuild")
        assert resp.status_code == 200
        assert resp.json()["status"] == "rebuilt"
    finally:
        await client.delete(f"/api/collections/{name}")


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


async def test_chat_endpoint_returns_answer(client: AsyncClient) -> None:
    response = await client.post("/api/chat", json={"query": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


async def test_chat_endpoint_returns_route(client: AsyncClient) -> None:
    response = await client.post("/api/chat", json={"query": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] in ("chat", "rag", "tool", None)


async def test_chat_endpoint_includes_trace(client: AsyncClient) -> None:
    response = await client.post("/api/chat", json={"query": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "trace" in data
    assert isinstance(data["trace"], list)
    # At minimum the router node should appear.
    nodes = [t["node"] for t in data["trace"]]
    assert "router" in nodes


async def test_chat_rag_query(client: AsyncClient) -> None:
    # A question that should trigger the RAG path if the index is populated.
    response = await client.post(
        "/api/chat",
        json={"query": "What is LangGraph?", "collection": "langgraph-docs"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)


# ---------------------------------------------------------------------------
# Chat streaming endpoint
# ---------------------------------------------------------------------------


async def test_chat_stream_endpoint_returns_ndjson(client: AsyncClient) -> None:
    import json

    response = await client.post("/api/chat/stream", json={"query": "Hello"})
    assert response.status_code == 200
    assert "ndjson" in response.headers.get("content-type", "")

    lines = [ln for ln in response.text.splitlines() if ln.strip()]
    assert len(lines) > 0

    # Every line must be valid JSON.
    parsed = [json.loads(ln) for ln in lines]

    # The last line should be the result event.
    result_events = [e for e in parsed if e.get("event") == "result"]
    assert len(result_events) == 1
    assert "answer" in result_events[0]["data"]


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------


async def test_webhook_note_created(client: AsyncClient) -> None:
    payload = {
        "event": "note.created",
        "note_id": f"test-note-{uuid.uuid4().hex[:8]}",
        "title": "Test integration note",
        "content": "This is a test note for integration testing purposes.",
        "tags": ["test"],
        "timestamp": int(time.time()),
    }
    response = await client.post("/api/webhooks/notes", json=payload)
    assert response.status_code in (200, 204)


async def test_webhook_note_deleted(client: AsyncClient) -> None:
    # Create first so there is something to delete.
    note_id = f"test-note-{uuid.uuid4().hex[:8]}"
    create_payload = {
        "event": "note.created",
        "note_id": note_id,
        "title": "Note to delete",
        "content": "Will be deleted by the next webhook call.",
        "tags": [],
        "timestamp": int(time.time()),
    }
    await client.post("/api/webhooks/notes", json=create_payload)

    delete_payload = {
        "event": "note.deleted",
        "note_id": note_id,
        "title": "",
        "content": "",
        "tags": [],
        "timestamp": int(time.time()),
    }
    response = await client.post("/api/webhooks/notes", json=delete_payload)
    assert response.status_code in (200, 204)


async def test_webhook_rejects_invalid_secret(client: AsyncClient) -> None:
    # Only tested when a secret is configured; if no secret is set the check is skipped.
    import pathlib

    from app.core.config.loader import load_system_config

    config_path = pathlib.Path(__file__).resolve().parents[4] / "config" / "config.toml"
    config = load_system_config(config_path)
    if not config.webhooks.edgenotes_secret:
        pytest.skip("No webhook secret configured -- secret validation not active")

    payload = {
        "event": "note.created",
        "note_id": "x",
        "title": "x",
        "content": "x",
        "tags": [],
        "timestamp": int(time.time()),
    }
    response = await client.post(
        "/api/webhooks/notes",
        json=payload,
        headers={"X-Webhook-Secret": "wrong-secret"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Notion sync endpoint
# ---------------------------------------------------------------------------


async def test_notion_sync_returns_400_without_database_id(
    client: AsyncClient, temp_collection: str
) -> None:
    import os

    if os.environ.get("NOTION_DATABASE_ID"):
        pytest.skip("NOTION_DATABASE_ID is set -- skipping missing-var test")

    resp = await client.post(f"/api/collections/{temp_collection}/sync-notion")
    assert resp.status_code == 400
    assert "NOTION_DATABASE_ID" in resp.json()["detail"]


async def test_notion_sync_response_schema(client: AsyncClient, temp_collection: str) -> None:
    import os

    if not os.environ.get("NOTION_TOKEN") or not os.environ.get("NOTION_DATABASE_ID"):
        pytest.skip("NOTION_TOKEN and NOTION_DATABASE_ID required for this test")

    resp = await client.post(f"/api/collections/{temp_collection}/sync-notion")
    assert resp.status_code == 200
    data = resp.json()
    assert data["collection"] == temp_collection
    assert isinstance(data["pages_synced"], int)
    assert isinstance(data["total_chunks"], int)
    assert data["pages_synced"] >= 0
    assert data["total_chunks"] >= 0


# ---------------------------------------------------------------------------
# Worklog agent via chat
# ---------------------------------------------------------------------------


async def test_chat_worklog_route_fallback(client: AsyncClient) -> None:
    """Worklog queries succeed even when WORKLOG_WORKER_URL is not set (falls back to chat)."""
    response = await client.post("/api/chat", json={"query": "show my worklog plans"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


async def test_chat_worklog_route_with_worker(client: AsyncClient) -> None:
    import os

    if not os.environ.get("WORKLOG_WORKER_URL"):
        pytest.skip("WORKLOG_WORKER_URL required for worklog route test")

    response = await client.post("/api/chat", json={"query": "list my worklog plans"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "worklog"
    assert isinstance(data["answer"], str)
