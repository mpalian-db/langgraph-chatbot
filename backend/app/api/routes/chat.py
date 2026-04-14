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
    EmbeddingDep,
    LLMDep,
    VectorStoreDep,
)
from app.core.graph.graph import build_graph
from app.core.graph.state import GraphState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str
    collection: str | None = None


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
    route: str | None = None
    citations: list[CitationOut] = []
    trace: list[TraceEntryOut] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_to_response(state: GraphState) -> ChatResponse:
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
    llm: LLMDep,
    vectorstore: VectorStoreDep,
    collection_store: CollectionDep,
    embedding: EmbeddingDep,
) -> ChatResponse:
    """Run the full graph synchronously and return the result."""
    start = time.monotonic()

    graph = build_graph(
        agents_config=agents_config,
        llm=llm,
        vectorstore=vectorstore,
        collection_store=collection_store,
        embedding=embedding,
    )

    initial_state = GraphState(query=body.query)
    if body.collection:
        initial_state.metadata_filters["collection"] = body.collection

    result = await graph.ainvoke(initial_state)
    elapsed = time.monotonic() - start

    # LangGraph may return a dict or a GraphState depending on version.
    if isinstance(result, dict):
        state = GraphState(**result)
    else:
        state = result

    logger.info("Chat completed in %.2fs (route=%s)", elapsed, state.route)
    return _state_to_response(state)


@router.post("/chat/stream")
async def chat_stream_endpoint(
    body: ChatRequest,
    agents_config: AgentsConfigDep,
    llm: LLMDep,
    vectorstore: VectorStoreDep,
    collection_store: CollectionDep,
    embedding: EmbeddingDep,
) -> StreamingResponse:
    """Stream graph execution events as newline-delimited JSON (NDJSON)."""
    import json

    graph = build_graph(
        agents_config=agents_config,
        llm=llm,
        vectorstore=vectorstore,
        collection_store=collection_store,
        embedding=embedding,
    )

    initial_state = GraphState(query=body.query)
    if body.collection:
        initial_state.metadata_filters["collection"] = body.collection

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

        # Emit the final result.
        if final_state is not None:
            state = GraphState(**final_state) if isinstance(final_state, dict) else final_state
            response = _state_to_response(state)
            yield json.dumps({"event": "result", "data": response.model_dump()}) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )
