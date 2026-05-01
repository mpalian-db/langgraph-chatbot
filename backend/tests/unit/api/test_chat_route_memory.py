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
    # chat_agent may prepend "Summary of earlier conversation: ..." when a
    # rolling summary exists, so match by substring rather than prefix.
    CHAT_PROMPT_MARKER = "You are a helpful assistant"
    SUMMARISER_PROMPT_PREFIX = "Summarise the following conversation"

    def _llm_side_effect(**kwargs):
        system = kwargs.get("system") or ""
        if system.startswith(ROUTER_PROMPT_PREFIX):
            return {
                "text": "chat",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        if system.startswith(SUMMARISER_PROMPT_PREFIX):
            return {
                "text": "MOCK ROLLED-UP SUMMARY",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 30},
            }
        if CHAT_PROMPT_MARKER in system:
            # Echo the message count so tests can assert how much history
            # reached the chat agent.
            return {
                "text": f"answer:{len(kwargs['messages'])}",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        msg = (
            f"unexpected LLM call from a node that is neither router, "
            f"chat_agent, nor summariser. system prompt prefix: {system[:60]!r}"
        )
        raise AssertionError(msg)

    chat_llm.complete = AsyncMock(side_effect=_llm_side_effect)

    # Use realistic prompts so the strict dispatch in `_llm_side_effect`
    # can identify router-vs-chat-agent-vs-summariser calls. The default
    # AgentsConfig() has an empty router prompt which would defeat the dispatch.
    agents_config = AgentsConfig()
    agents_config.router.prompt = "You are a routing agent. Pick chat, rag, or tool."
    agents_config.chat_agent.system_prompt = "You are a helpful assistant."
    # The default SummariserConfig.prompt already starts with "Summarise the
    # following conversation"; assigning it explicitly here makes the test
    # robust against future default changes that might break the dispatch.
    agents_config.summariser.prompt = (
        "Summarise the following conversation between a user and an assistant."
    )

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


async def test_stream_emits_memory_load_events_before_router(
    client: AsyncClient, store: SQLiteConversationStore
):
    """Pin the TTFB contract: the very first events on the wire are the
    memory_load start/end pair, BEFORE any LangGraph node events. This
    means the client never sees a silent gap during the summariser LLM
    call -- it sees `node_start: memory_load` immediately."""
    import json

    await store.append("ttfb-conv", "user", "old user")
    await store.append("ttfb-conv", "assistant", "old assistant")

    resp = await client.post(
        "/api/chat/stream",
        json={"query": "follow up", "conversation_id": "ttfb-conv"},
    )
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    nodes_in_order = [e.get("node") for e in events if "node" in e]

    # Memory_load start + end are the very first wire events. Anything that
    # comes after them is a graph node event (LangGraph itself emits a
    # top-level "LangGraph" wrapper event we don't filter out, but it
    # follows memory_load -- the TTFB invariant).
    assert nodes_in_order[0] == "memory_load"
    assert nodes_in_order[1] == "memory_load"
    assert any(n not in ("memory_load",) for n in nodes_in_order[2:]), (
        "expected at least one graph-side event after memory_load"
    )

    # The memory_load node_end carries the same data shape the trace will.
    [memory_end] = [
        e for e in events if e.get("event") == "node_end" and e.get("node") == "memory_load"
    ]
    assert memory_end["data"]["history_turns"] == 2
    assert memory_end["data"]["summary_present"] is False
    assert memory_end["data"]["summarisation_triggered"] is False


async def test_stream_endpoint_preserves_memory_load_trace_entry(
    client: AsyncClient, store: SQLiteConversationStore
):
    """The synthetic memory_load TraceEntry is prepended to the initial
    GraphState's execution_trace BEFORE the graph runs. Node deltas
    accumulate further trace entries via `state.execution_trace + [...]`,
    so the memory entry should survive through to the streamed result
    event.

    Pin this contract because the streaming endpoint reconstructs
    `final_state` from per-node on_chain_end deltas; a regression that
    overwrote rather than accumulated trace would silently drop the
    memory_load entry from the response."""
    import json

    # Prime a warm conversation so memory_load reports 2 history turns.
    await store.append("stream-trace", "user", "old user")
    await store.append("stream-trace", "assistant", "old assistant")

    resp = await client.post(
        "/api/chat/stream",
        json={"query": "follow up", "conversation_id": "stream-trace"},
    )
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    [result] = [e for e in events if e.get("event") == "result"]
    trace = result["data"]["trace"]

    memory_entries = [t for t in trace if t["node"] == "memory_load"]
    assert len(memory_entries) == 1
    assert memory_entries[0]["data"]["history_turns"] == 2
    assert memory_entries[0]["data"]["summary_present"] is False
    assert memory_entries[0]["data"]["summarisation_triggered"] is False


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


# ---------------------------------------------------------------------------
# Summarisation trigger via the route
# ---------------------------------------------------------------------------


async def test_chat_response_trace_includes_memory_load_entry(
    client: AsyncClient, store: SQLiteConversationStore
):
    """The chat route emits a `memory_load` trace entry recording history
    size, summary presence, and whether summarisation triggered. This
    surface lets the frontend show what happened during memory load
    alongside the LangGraph node trace -- a unified observability view."""
    # Two-turn warm conversation: one prior round persisted.
    await store.append("trace-conv", "user", "old user")
    await store.append("trace-conv", "assistant", "old assistant")

    resp = await client.post("/api/chat", json={"query": "next", "conversation_id": "trace-conv"})
    assert resp.status_code == 200

    trace = resp.json()["trace"]
    memory_entries = [t for t in trace if t["node"] == "memory_load"]
    assert len(memory_entries) == 1
    entry = memory_entries[0]
    assert entry["data"]["history_turns"] == 2
    assert entry["data"]["summary_present"] is False
    assert entry["data"]["summarisation_triggered"] is False


async def test_chat_response_trace_records_summarisation_trigger(
    client: AsyncClient, store: SQLiteConversationStore
):
    """When this load triggers summarisation, the trace entry must reflect
    it -- otherwise the operator can't tell from the response whether the
    chat round paid the summariser cost."""
    cid = "trace-summary"
    for i in range(25):
        role = "user" if i % 2 == 0 else "assistant"
        await store.append(cid, role, f"seed {i}")

    resp = await client.post("/api/chat", json={"query": "trigger summary", "conversation_id": cid})
    assert resp.status_code == 200

    trace = resp.json()["trace"]
    [memory_entry] = [t for t in trace if t["node"] == "memory_load"]
    assert memory_entry["data"]["summarisation_triggered"] is True
    assert memory_entry["data"]["summary_present"] is True
    assert memory_entry["data"]["history_turns"] == 10  # keep_recent default


async def test_summariser_triggers_when_history_crosses_threshold(
    client: AsyncClient, chat_llm: AsyncMock, store: SQLiteConversationStore
):
    """End-to-end: seed a conversation with enough turns to cross the
    summariser threshold, fire one more chat request, and assert that
    a summary was persisted by the time the response returns.

    This pins the lazy-on-load contract: summarisation runs as part of the
    chat route's history load, transparently from the client's perspective."""
    cid = "sum-conv-1"
    # Seed 25 turns directly via the store so we don't pay the per-turn LLM
    # cost via the route. Need > summarise_threshold (default 20) so the
    # next chat-route call triggers the summariser.
    for i in range(25):
        role = "user" if i % 2 == 0 else "assistant"
        await store.append(cid, role, f"seeded turn {i}")

    # Confirm pre-conditions: no summary yet, 25 turns in the post-summary tail.
    summary_before, turns_before = await store.load_summary_and_turns(cid)
    assert summary_before is None
    assert len(turns_before) == 25

    # Fire a chat request -- the summariser should run during history load.
    resp = await client.post("/api/chat", json={"query": "next turn", "conversation_id": cid})
    assert resp.status_code == 200

    # Post-condition: a summary now exists; the post-boundary tail is
    # exactly the SummariserConfig.keep_recent (default 10) plus the new
    # round that the route just persisted (user + assistant = 2).
    summary_after, turns_after = await store.load_summary_and_turns(cid)
    assert summary_after is not None
    assert len(turns_after) == 12  # 10 kept verbatim + 2 from the new round
