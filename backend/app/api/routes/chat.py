"""Chat endpoints -- synchronous and streaming."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.dependencies import (
    AgentsConfigDep,
    CollectionDep,
    ConversationReaderDep,
    ConversationWriterDep,
    EmbeddingDep,
    LLMRegistryDep,
    SystemConfigDep,
    VectorStoreDep,
    WorklogDep,
)
from app.core.graph.graph import _resolve_llm, build_graph
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry
from app.core.operations.conversation_memory import MemoryView, load_with_summary

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str
    collection: str | None = None
    # Client-supplied id to bind this turn to an existing conversation. When
    # absent the server generates a new uuid and returns it in the response,
    # so the client can persist it for follow-up turns.
    conversation_id: str | None = None


class CitationOut(BaseModel):
    chunk_id: str
    text: str
    collection: str


class TraceEntryOut(BaseModel):
    node: str
    duration_ms: float
    data: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    route: str | None = None
    citations: list[CitationOut] = []
    trace: list[TraceEntryOut] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _memory_trace(memory: MemoryView, duration_ms: float) -> TraceEntry:
    """Build a synthetic trace entry recording what happened during memory
    load. Memory runs OUTSIDE the LangGraph graph (before it executes), but
    we project it into the trace so the UI's TraceView can show it
    alongside graph nodes for a unified observability surface."""
    return TraceEntry(
        node="memory_load",
        duration_ms=duration_ms,
        data={
            "history_turns": len(memory.recent),
            "summary_present": memory.summary is not None,
            "summarisation_triggered": memory.summarised_this_load,
        },
    )


def _state_to_response(state: GraphState, conversation_id: str) -> ChatResponse:
    """Build the response. `conversation_id` is taken from request scope, not
    from `state.conversation_id`, because LangGraph's streaming `on_chain_end`
    events deliver per-node deltas (not the full state) and a naive
    `GraphState(**delta)` reconstruction would default conversation_id to
    None. Passing it explicitly keeps the contract honest."""
    citations = [
        CitationOut(
            chunk_id=c.chunk_id,
            text=c.text,
            collection=c.collection,
        )
        for c in state.citations
    ]
    trace = [
        TraceEntryOut(
            node=t.node,
            duration_ms=t.duration_ms,
            data=t.data,
        )
        for t in state.execution_trace
    ]
    return ChatResponse(
        answer=state.final_answer or state.draft_answer or "",
        conversation_id=conversation_id,
        route=state.route,
        citations=citations,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    body: ChatRequest,
    agents_config: AgentsConfigDep,
    system_config: SystemConfigDep,
    llms: LLMRegistryDep,
    vectorstore: VectorStoreDep,
    collection_store: CollectionDep,
    embedding: EmbeddingDep,
    worklog: WorklogDep,
    conversation_reader: ConversationReaderDep,
    conversation_writer: ConversationWriterDep,
) -> ChatResponse:
    """Run the full graph synchronously and return the result."""
    import uuid

    start = time.monotonic()

    graph = build_graph(
        agents_config=agents_config,
        llms=llms,
        default_provider=system_config.llm.provider,
        vectorstore=vectorstore,
        collection_store=collection_store,
        embedding=embedding,
        worklog=worklog,
    )

    # Resolve conversation_id: client-supplied or server-generated.
    conversation_id = body.conversation_id or str(uuid.uuid4())
    # Pull the summary-aware view of memory so distant context survives
    # compression. The summariser may issue an LLM call here when the
    # post-summary tail crosses the threshold; that's the lazy-on-load
    # trigger documented in conversation_memory.py.
    summariser_llm = _resolve_llm(agents_config.summariser, llms, system_config.llm.provider)
    memory_start = time.monotonic()
    memory = await load_with_summary(
        conversation_id,
        reader=conversation_reader,
        writer=conversation_writer,
        llm=summariser_llm,
        config=agents_config.summariser,
    )
    memory_ms = (time.monotonic() - memory_start) * 1000

    initial_state = GraphState(
        query=body.query,
        collection=body.collection,
        conversation_id=conversation_id,
        history=memory.recent,
        conversation_summary=memory.summary,
        execution_trace=[_memory_trace(memory, memory_ms)],
    )

    result = await graph.ainvoke(initial_state)
    elapsed = time.monotonic() - start

    # LangGraph may return a dict or a GraphState depending on version.
    if isinstance(result, dict):
        state = GraphState(**result)
    else:
        state = result

    # Persist this turn atomically: either both rows commit or neither, so
    # the conversation can never be left in a "user message with no reply"
    # state on cancellation.
    answer = state.final_answer or state.draft_answer or ""
    if answer:
        await conversation_writer.append_pair(conversation_id, body.query, answer)

    logger.info("Chat completed in %.2fs (route=%s)", elapsed, state.route)
    return _state_to_response(state, conversation_id)


@router.post("/chat/stream")
async def chat_stream_endpoint(
    body: ChatRequest,
    agents_config: AgentsConfigDep,
    system_config: SystemConfigDep,
    llms: LLMRegistryDep,
    vectorstore: VectorStoreDep,
    collection_store: CollectionDep,
    embedding: EmbeddingDep,
    worklog: WorklogDep,
    conversation_reader: ConversationReaderDep,
    conversation_writer: ConversationWriterDep,
) -> StreamingResponse:
    """Stream graph execution events as newline-delimited JSON (NDJSON)."""
    import json
    import uuid

    graph = build_graph(
        agents_config=agents_config,
        llms=llms,
        default_provider=system_config.llm.provider,
        vectorstore=vectorstore,
        collection_store=collection_store,
        embedding=embedding,
        worklog=worklog,
    )

    conversation_id = body.conversation_id or str(uuid.uuid4())
    summariser_llm = _resolve_llm(agents_config.summariser, llms, system_config.llm.provider)
    memory_start = time.monotonic()
    memory = await load_with_summary(
        conversation_id,
        reader=conversation_reader,
        writer=conversation_writer,
        llm=summariser_llm,
        config=agents_config.summariser,
    )
    memory_ms = (time.monotonic() - memory_start) * 1000

    initial_state = GraphState(
        query=body.query,
        collection=body.collection,
        conversation_id=conversation_id,
        history=memory.recent,
        conversation_summary=memory.summary,
        execution_trace=[_memory_trace(memory, memory_ms)],
    )

    async def event_generator():
        final_state = None
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start":
                payload = {"event": "node_start", "node": name}
                yield json.dumps(payload) + "\n"
            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict):
                    final_state = output
                payload = {"event": "node_end", "node": name}
                yield json.dumps(payload) + "\n"

        # Emit the final result and persist the turn. Use append_pair so a
        # cancellation between the two writes cannot leave a half-persisted
        # round. We also persist BEFORE yielding the result so a client
        # disconnect after the result event still keeps the data.
        if final_state is not None:
            state = GraphState(**final_state) if isinstance(final_state, dict) else final_state
            answer = state.final_answer or state.draft_answer or ""
            if answer:
                await conversation_writer.append_pair(conversation_id, body.query, answer)
            response = _state_to_response(state, conversation_id)
            yield json.dumps({"event": "result", "data": response.model_dump()}) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )
