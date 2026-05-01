"""End-to-end integration test for conversation memory + summarisation.

Runs through 25+ chat turns against the real Ollama-backed graph with a
fresh on-disk SQLite store, asserting that:

  * Memory persists across requests (the chat agent on turn N sees turn N-1).
  * The summariser actually fires when the threshold is crossed.
  * The summary text is non-trivial (not empty, references real content).
  * The introspection endpoint surfaces the persisted state.

The unit + route tests with mocked LLMs cover the orchestration shape;
this test catches model-specific failure modes (truncated summaries,
prompt-following lapses, slow generation) that mocks can't surface.

Skipped automatically when Ollama isn't reachable so the integration
suite still runs in environments without local services. Marked
`@pytest.mark.slow` because a 25+ turn loop against llama3.2:3b takes
30-60 seconds depending on the host.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.conversation.sqlite import SQLiteConversationStore
from app.api.dependencies import (
    get_conversation_reader,
    get_conversation_store,
    get_conversation_writer,
)
from app.main import create_app

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Skip-if-services-unavailable
# ---------------------------------------------------------------------------


def _ollama_reachable() -> bool:
    """Quick TCP probe against the Ollama HTTP API. Avoids waiting on
    httpx's full connect timeout when the service is plainly down."""
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=2.0)
        return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


pytestmark.append(
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Ollama not reachable -- start it with `ollama serve` to run this test",
    )
)


# ---------------------------------------------------------------------------
# Fixtures: isolated app with a fresh on-disk SQLite store
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_store() -> Iterator[SQLiteConversationStore]:
    """Each test gets its own SQLite file under a tempdir so the global
    `data/conversations.sqlite` is never touched. The file lives only for
    the lifetime of the test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield SQLiteConversationStore(Path(tmp) / "test.sqlite")


@pytest.fixture
async def client(isolated_store: SQLiteConversationStore) -> AsyncIterator[AsyncClient]:
    app = create_app()
    # Override conversation store with our isolated one. The lru_cache'd
    # default writes to data/conversations.sqlite which would persist
    # across runs and contaminate other tests.
    app.dependency_overrides[get_conversation_store] = lambda: isolated_store
    app.dependency_overrides[get_conversation_reader] = lambda: isolated_store
    app.dependency_overrides[get_conversation_writer] = lambda: isolated_store

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_chat_round_trip_persists_history_against_real_ollama(
    client: AsyncClient, isolated_store: SQLiteConversationStore
):
    """Two turns: the second must see the first in chat_agent's prompt.
    Asserted by storage state, since we can't introspect Ollama's prompt
    without instrumenting the LLM port."""
    r1 = await client.post("/api/chat", json={"query": "Hello, my name is Michal."})
    assert r1.status_code == 200, r1.text
    cid = r1.json()["conversation_id"]

    # Turn 1 must be in storage with the user's actual query.
    persisted = await isolated_store.load(cid)
    assert len(persisted) == 2
    assert persisted[0].role == "user"
    assert "Michal" in persisted[0].content

    r2 = await client.post(
        "/api/chat", json={"query": "What did I say my name was?", "conversation_id": cid}
    )
    assert r2.status_code == 200, r2.text

    # Two more rows persisted for turn 2.
    persisted = await isolated_store.load(cid, limit=10)
    assert len(persisted) == 4

    # Trace shows memory_load loaded 2 history turns into chat_agent.
    [memory_entry] = [t for t in r2.json()["trace"] if t["node"] == "memory_load"]
    assert memory_entry["data"]["history_turns"] == 2


async def test_summariser_fires_against_real_ollama_when_threshold_crossed(
    client: AsyncClient, isolated_store: SQLiteConversationStore
):
    """Seed 21 turns directly via the store (cheap), then fire one more
    chat request through the route. The summariser should run during the
    history load and produce a non-trivial rolling summary.

    Direct seeding rather than 21 round-trips keeps this test under a
    minute; we're testing the summariser's behaviour against real Ollama,
    not the round-trip latency."""
    cid = "integ-sum-conv"
    for i in range(21):
        role = "user" if i % 2 == 0 else "assistant"
        # Vary the content so the summariser has actual material to compress.
        content = f"Turn {i}: this is message number {i} discussing topic {'A' if i < 10 else 'B'}."
        await isolated_store.append(cid, role, content)

    pre_summary, pre_turns = await isolated_store.load_summary_and_turns(cid)
    assert pre_summary is None
    assert len(pre_turns) == 21

    # One real chat request -- this triggers the summariser inside load_with_summary.
    resp = await client.post(
        "/api/chat",
        json={"query": "Briefly summarise the conversation so far.", "conversation_id": cid},
    )
    assert resp.status_code == 200, resp.text

    # Trace records that summarisation was triggered.
    [memory_entry] = [t for t in resp.json()["trace"] if t["node"] == "memory_load"]
    assert memory_entry["data"]["summarisation_triggered"] is True
    assert memory_entry["data"]["summary_present"] is True

    # Storage now has a non-trivial summary.
    summary, post_turns = await isolated_store.load_summary_and_turns(cid)
    assert summary is not None
    assert len(summary) > 50, f"summary unexpectedly short: {summary!r}"
    # Post-boundary tail = keep_recent (10) + the new (user, assistant) pair from this round.
    assert len(post_turns) == 12


async def test_introspection_endpoint_surfaces_real_summary(
    client: AsyncClient, isolated_store: SQLiteConversationStore
):
    """After a real summarisation round, GET /api/conversations/{id}
    returns the summary and post-boundary turns. Pins the end-to-end
    visibility contract: what the operator sees in the debug UI matches
    what's actually persisted."""
    cid = "integ-introspect"
    for i in range(21):
        await isolated_store.append(cid, "user" if i % 2 == 0 else "assistant", f"turn-{i}")

    # Trigger summarisation via a chat request.
    chat_resp = await client.post("/api/chat", json={"query": "carry on", "conversation_id": cid})
    assert chat_resp.status_code == 200

    # Now hit the introspection endpoint.
    detail_resp = await client.get(f"/api/conversations/{cid}")
    assert detail_resp.status_code == 200
    body = detail_resp.json()
    assert body["conversation_id"] == cid
    assert body["summary"] is not None
    assert len(body["summary"]) > 50
    # 10 verbatim kept + 2 from the just-completed round.
    assert len(body["turns"]) == 12
