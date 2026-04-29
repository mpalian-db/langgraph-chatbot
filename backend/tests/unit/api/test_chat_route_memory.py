"""Integration-style tests for conversation memory at the chat route.

Mocks the LLM and conversation store via dependency_overrides so the round
trip can be exercised in CI without touching real LLM or the on-disk SQLite
file. Pins the contract:

  * First request without conversation_id: server generates a uuid, returns it.
  * Second request with that id: history is loaded and fed to chat_agent.
  * Both turns persist (user query + assistant answer).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.conversation.sqlite import SQLiteConversationStore
from app.api.dependencies import (
    get_agents_config,
    get_collection_port,
    get_conversation_reader,
    get_conversation_writer,
    get_embedding,
    get_llm_registry,
    get_system_config,
    get_vector_store,
    get_worklog,
)
from app.core.config.models import AgentsConfig, SystemConfig
from app.main import create_app


@pytest.fixture
def store() -> SQLiteConversationStore:
    """Each test gets its own in-memory store -- isolation by construction."""
    return SQLiteConversationStore(":memory:")


@pytest.fixture
def chat_llm() -> AsyncMock:
    """LLM that returns a deterministic chat-path response."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=lambda **kwargs: {
            # Echo back the messages count so tests can assert how much
            # history reached the LLM.
            "text": f"answer:{len(kwargs['messages'])}",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    return llm


@pytest.fixture
def chat_router_llm() -> AsyncMock:
    """Router LLM that always picks the chat route."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value={
            "text": "chat",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    )
    return llm


@pytest.fixture
def test_app(store: SQLiteConversationStore, chat_llm: AsyncMock):
    app = create_app()

    # Dispatch on the actual system-prompt prefix so a router/chat_agent prompt
    # change is observable here -- raising on unknown prevents the previous
    # heuristic from silently collapsing into router.py's "default to chat"
    # fallback and masking a real regression.
    ROUTER_PROMPT_PREFIX = "You are a routing agent"
    CHAT_PROMPT_PREFIX = "You are a helpful assistant"

    def _llm_side_effect(**kwargs):
        system = kwargs.get("system") or ""
        if system.startswith(ROUTER_PROMPT_PREFIX):
            return {
                "text": "chat",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        if system.startswith(CHAT_PROMPT_PREFIX):
            # Echo the message count so tests can assert how much history
            # reached the chat agent.
            return {
                "text": f"answer:{len(kwargs['messages'])}",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        msg = (
            f"unexpected LLM call from a node that is neither router nor "
            f"chat_agent. system prompt prefix: {system[:60]!r}"
        )
        raise AssertionError(msg)

    chat_llm.complete = AsyncMock(side_effect=_llm_side_effect)

    # Use a realistic router prompt so the strict dispatch in `_llm_side_effect`
    # can identify router-vs-chat-agent calls. The default AgentsConfig() has
    # an empty router prompt which would defeat the dispatch.
    agents_config = AgentsConfig()
    agents_config.router.prompt = "You are a routing agent. Pick chat, rag, or tool."
    agents_config.chat_agent.system_prompt = "You are a helpful assistant."

    app.dependency_overrides[get_system_config] = lambda: SystemConfig()
    app.dependency_overrides[get_agents_config] = lambda: agents_config
    app.dependency_overrides[get_llm_registry] = lambda: {"ollama": chat_llm}
    # Wrap AsyncMock in a lambda -- using AsyncMock directly as a dependency
    # factory makes FastAPI introspect its `*args, **kwargs` signature and
    # treat them as required query parameters, returning 422.
    app.dependency_overrides[get_vector_store] = lambda: AsyncMock()
    app.dependency_overrides[get_collection_port] = lambda: AsyncMock()
    app.dependency_overrides[get_embedding] = lambda: AsyncMock()
    app.dependency_overrides[get_worklog] = lambda: None
    app.dependency_overrides[get_conversation_reader] = lambda: store
    app.dependency_overrides[get_conversation_writer] = lambda: store

    return app


@pytest.fixture
async def client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


async def test_first_request_returns_generated_conversation_id(client: AsyncClient):
    """No client-supplied id -> server generates one and returns it."""
    resp = await client.post("/api/chat", json={"query": "hello"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"]
    assert len(body["conversation_id"]) >= 32  # uuid4 is 36 chars including hyphens


async def test_second_request_uses_history_from_first(
    client: AsyncClient, chat_llm: AsyncMock, store: SQLiteConversationStore
):
    """Pin the round-trip: turn 2 must see turn 1 in its prompt."""
    # Turn 1: server generates an id.
    r1 = await client.post("/api/chat", json={"query": "first message"})
    cid = r1.json()["conversation_id"]

    # Confirm turn 1 was persisted: user + assistant rows.
    persisted = await store.load(cid)
    assert len(persisted) == 2
    assert persisted[0].role == "user" and persisted[0].content == "first message"
    assert persisted[1].role == "assistant"

    # Turn 2: use the same conversation id.
    r2 = await client.post("/api/chat", json={"query": "second message", "conversation_id": cid})
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == cid

    # The chat_agent received history. Inspect the second LLM call for the
    # chat_agent path: messages should include user/assistant from turn 1
    # plus the current user query. The mock returns "answer:N" reflecting
    # the total messages count.
    answer = r2.json()["answer"]
    # Turn 2 chat_agent should see: prior user, prior assistant, current user = 3 messages.
    assert answer == "answer:3"


async def test_supplied_unknown_conversation_id_starts_empty_history(
    client: AsyncClient, chat_llm: AsyncMock, store: SQLiteConversationStore
):
    """A client-supplied id that's never been seen before is valid -- it just
    starts a new conversation under that id."""
    resp = await client.post("/api/chat", json={"query": "hello", "conversation_id": "fresh-id"})

    assert resp.status_code == 200
    assert resp.json()["conversation_id"] == "fresh-id"

    # chat_agent saw only the current query (1 message) since history was empty.
    assert resp.json()["answer"] == "answer:1"

    # The turn was now stored under the supplied id.
    persisted = await store.load("fresh-id")
    assert len(persisted) == 2


async def test_separate_conversations_do_not_leak_history(
    client: AsyncClient, store: SQLiteConversationStore
):
    """Two different conversation_ids must not share state."""
    await client.post("/api/chat", json={"query": "alpha message", "conversation_id": "A"})
    await client.post("/api/chat", json={"query": "bravo message", "conversation_id": "B"})

    a = await store.load("A")
    b = await store.load("B")

    assert all("alpha" in t.content or "answer" in t.content for t in a)
    assert all("bravo" in t.content or "answer" in t.content for t in b)
    assert len(a) == 2
    assert len(b) == 2


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------


async def test_stream_endpoint_returns_conversation_id_in_result_event(
    client: AsyncClient, store: SQLiteConversationStore
):
    """The streaming path is the one with the historic state-recovery bug:
    LangGraph's on_chain_end events deliver per-node deltas, not full state,
    so a naive `GraphState(**delta)` would default conversation_id to None
    and the result event would carry an empty id. Pin that the explicit
    `_state_to_response(state, conversation_id)` parameterisation produces
    a correctly populated id."""
    import json

    resp = await client.post(
        "/api/chat/stream",
        json={"query": "hello", "conversation_id": "stream-1"},
    )
    assert resp.status_code == 200
    assert "ndjson" in resp.headers.get("content-type", "")

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    result_events = [e for e in events if e.get("event") == "result"]

    assert len(result_events) == 1
    assert result_events[0]["data"]["conversation_id"] == "stream-1"


async def test_stream_endpoint_persists_turn_pair_atomically(
    client: AsyncClient, store: SQLiteConversationStore
):
    """The streaming path must use append_pair so a cancellation between
    user and assistant writes can never produce an orphan user row."""
    await client.post(
        "/api/chat/stream",
        json={"query": "first stream", "conversation_id": "stream-2"},
    )

    persisted = await store.load("stream-2")
    assert len(persisted) == 2
    assert persisted[0].role == "user" and persisted[0].content == "first stream"
    assert persisted[1].role == "assistant"
