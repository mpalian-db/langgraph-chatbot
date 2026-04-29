from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.config.models import AgentsConfig
from app.core.graph.graph import build_graph
from app.core.graph.state import GraphState


@pytest.mark.asyncio
async def test_graph_compiles(mock_llm, mock_vectorstore, mock_collection_store, mock_embedding):
    config = AgentsConfig()
    graph = build_graph(
        agents_config=config,
        llms={"ollama": mock_llm, "anthropic": mock_llm},
        default_provider="ollama",
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )
    # Graph should be a compiled LangGraph object with an ainvoke method
    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_graph_chat_path(mock_llm, mock_vectorstore, mock_collection_store, mock_embedding):
    # Router returns "chat", chat agent returns a final answer
    mock_llm.complete = AsyncMock(
        side_effect=[
            {  # router
                "text": "chat",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
            {  # chat agent
                "text": "Hello! I can help with that.",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 8},
            },
        ]
    )
    config = AgentsConfig()
    graph = build_graph(
        agents_config=config,
        llms={"ollama": mock_llm, "anthropic": mock_llm},
        default_provider="ollama",
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    result = await graph.ainvoke(GraphState(query="hello"))

    assert result["route"] == "chat"
    assert result["final_answer"] == "Hello! I can help with that."


@pytest.mark.asyncio
async def test_graph_rag_path_revises_when_citation_coverage_low(
    mock_llm, mock_vectorstore, mock_collection_store, mock_embedding
):
    """End-to-end: a low-citation draft triggers the verifier's
    citation_coverage check, sends control back to answer_generation, and
    the second draft (with a citation) is accepted.

    This exercises the revise loop introduced by enabling the
    citation_coverage check in the default verifier config."""
    cid = "abcd1234-5678-90ab-cdef-1234567890ab"
    mock_vectorstore.search = AsyncMock(
        return_value=[
            __import__("app.core.models.types", fromlist=["Chunk"]).Chunk(
                id=cid,
                text="LangGraph is a stateful agent framework.",
                collection="docs",
                score=0.9,
            ),
        ]
    )

    mock_llm.complete = AsyncMock(
        side_effect=[
            {  # router -> rag
                "text": "rag",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
            {  # answer_generation: no citations -- coverage 0.0
                "text": (
                    "LangGraph is a stateful agent framework. "
                    "It supports conditional edges and revise loops."
                ),
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": 20},
            },
            {  # answer_generation revised: cites the chunk
                "text": (
                    f"LangGraph is a stateful agent framework [{cid}]. "
                    f"It supports conditional edges and revise loops [{cid}]."
                ),
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 60, "output_tokens": 25},
            },
        ]
    )

    config = AgentsConfig()
    # Mirror the production default introduced in agents.toml.
    config.verifier.checks = ["score_threshold", "citation_coverage"]
    config.verifier.citation_coverage_min = 0.5
    config.verifier.max_retries = 2

    graph = build_graph(
        agents_config=config,
        llms={"ollama": mock_llm, "anthropic": mock_llm},
        default_provider="ollama",
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    result = await graph.ainvoke(GraphState(query="What is LangGraph?", collection="docs"))

    assert result["route"] == "rag"
    # Second draft was accepted -- final_answer matches the revised draft.
    assert cid in result["final_answer"]
    assert result["retry_count"] == 1
    # Citations populated from the accepted draft.
    assert len(result["citations"]) == 1
    assert result["citations"][0].chunk_id == cid


# ---------------------------------------------------------------------------
# Per-node provider routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_routes_each_node_to_its_configured_provider(
    mock_vectorstore, mock_collection_store, mock_embedding
):
    """Distinct LLM ports are wired per node when each node config sets a
    provider override. The router (no override) falls back to the default
    provider; the verifier explicitly uses anthropic.

    This pins the architectural seam introduced for Phase 2: nodes consume
    LLMs through a registry, not a single shared port."""
    ollama_llm = AsyncMock()
    ollama_llm.complete = AsyncMock(
        return_value={
            "text": "chat",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    )
    anthropic_llm = AsyncMock()
    anthropic_llm.complete = AsyncMock(
        return_value={
            "text": "Hi from anthropic.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )

    config = AgentsConfig()
    # router has no override -> falls back to default ("ollama"); chat_agent
    # explicitly requests anthropic. The chat path is router -> chat_agent.
    config.chat_agent.provider = "anthropic"

    graph = build_graph(
        agents_config=config,
        llms={"ollama": ollama_llm, "anthropic": anthropic_llm},
        default_provider="ollama",
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    result = await graph.ainvoke(GraphState(query="hello"))

    assert result["final_answer"] == "Hi from anthropic."
    # Router used the default (ollama) port.
    assert ollama_llm.complete.await_count == 1
    # Chat agent used the anthropic port.
    assert anthropic_llm.complete.await_count == 1


@pytest.mark.asyncio
async def test_graph_build_fails_loudly_when_provider_override_missing(
    mock_llm, mock_vectorstore, mock_collection_store, mock_embedding
):
    """A node config requesting an unregistered provider must fail at graph
    build time with a clear ValueError, not silently fall back to the default
    (which would mask the misconfiguration). Note: build_graph() runs inside
    the chat handler, so this surfaces on the first chat request rather than
    at application startup."""
    config = AgentsConfig()
    # Verifier asks for anthropic, but the registry only has ollama.
    config.verifier.provider = "anthropic"

    with pytest.raises(ValueError, match="anthropic"):
        build_graph(
            agents_config=config,
            llms={"ollama": mock_llm},
            default_provider="ollama",
            vectorstore=mock_vectorstore,
            collection_store=mock_collection_store,
            embedding=mock_embedding,
        )


@pytest.mark.asyncio
async def test_graph_rag_path_routes_verifier_to_anthropic_separately(
    mock_vectorstore, mock_collection_store, mock_embedding
):
    """Pin the motivating verifier case end-to-end: the RAG path runs
    answer_generation on the default provider and verifier on Anthropic.

    Counts the calls per port to prove the per-node split, not just one node
    happening to land on the right LLM."""
    cid = "abcd1234-5678-90ab-cdef-1234567890ab"
    mock_vectorstore.search = AsyncMock(
        return_value=[
            __import__("app.core.models.types", fromlist=["Chunk"]).Chunk(
                id=cid,
                text="LangGraph is stateful.",
                collection="docs",
                score=0.9,
            ),
        ]
    )

    ollama_llm = AsyncMock()
    ollama_llm.complete = AsyncMock(
        side_effect=[
            {  # router -> rag (called on ollama)
                "text": "rag",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
            {  # answer_generation (called on ollama)
                "text": f"LangGraph is stateful [{cid}].",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": 20},
            },
        ]
    )
    anthropic_llm = AsyncMock()
    anthropic_llm.complete = AsyncMock(
        return_value={  # verifier support_analysis (called on anthropic)
            "text": "OUTCOME: accept\nSCORE: 0.9\nREASON: Well supported.\nUNSUPPORTED: NONE",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
    )

    config = AgentsConfig()
    config.verifier.provider = "anthropic"
    # Enable support_analysis so the verifier actually calls its LLM.
    config.verifier.checks = ["score_threshold", "support_analysis"]

    graph = build_graph(
        agents_config=config,
        llms={"ollama": ollama_llm, "anthropic": anthropic_llm},
        default_provider="ollama",
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    result = await graph.ainvoke(GraphState(query="What is LangGraph?", collection="docs"))

    # Router + answer_generation called on ollama (default).
    assert ollama_llm.complete.await_count == 2
    # Verifier called once on anthropic.
    assert anthropic_llm.complete.await_count == 1
    assert result["final_answer"] == f"LangGraph is stateful [{cid}]."


@pytest.mark.asyncio
async def test_graph_build_does_not_mutate_input_router_routes(
    mock_llm, mock_vectorstore, mock_collection_store, mock_embedding
):
    """Regression: get_agents_config() in dependencies.py is lru_cached, so
    build_graph() mutating agents_config.router.routes would leak across
    requests. The "drop worklog when no port" branch must not touch the
    caller's config."""
    config = AgentsConfig()
    original_routes = list(config.router.routes)

    # Build once with worklog=None -- this is the branch that previously
    # mutated the input config.
    build_graph(
        agents_config=config,
        llms={"ollama": mock_llm},
        default_provider="ollama",
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
        worklog=None,
    )

    assert config.router.routes == original_routes, (
        "build_graph mutated agents_config.router.routes; this leaks across "
        "requests because get_agents_config() returns an lru_cached singleton"
    )
