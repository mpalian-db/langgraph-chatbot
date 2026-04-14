from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.config.models import (
    AnswerGenerationConfig,
    ChatAgentConfig,
    RetrievalConfig,
    RouterConfig,
    ToolAgentConfig,
)
from app.core.graph.nodes import answer_generation, chat_agent, retrieval, router, tool_agent
from app.core.graph.state import GraphState
from app.core.models.types import Chunk


@pytest.mark.asyncio
async def test_router_sets_route(mock_llm):
    mock_llm.complete = AsyncMock(
        return_value={
            "text": "rag",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    )
    state = GraphState(query="What is LangGraph?")
    config = RouterConfig(prompt="Route this.", model="llama3.2:3b")

    result = await router.run(state, config=config, llm=mock_llm)

    assert result["route"] == "rag"
    assert len(result["execution_trace"]) == 1
    assert result["execution_trace"][0].node == "router"


@pytest.mark.asyncio
async def test_router_defaults_to_chat_on_unknown_route(mock_llm):
    mock_llm.complete = AsyncMock(
        return_value={
            "text": "unknown_xyz",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    )
    state = GraphState(query="hello")
    config = RouterConfig()

    result = await router.run(state, config=config, llm=mock_llm)

    assert result["route"] == "chat"


@pytest.mark.asyncio
async def test_chat_agent_sets_final_answer(mock_llm):
    mock_llm.complete = AsyncMock(
        return_value={
            "text": "Hello! How can I help?",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
    )
    state = GraphState(query="hello", route="chat")
    config = ChatAgentConfig()

    result = await chat_agent.run(state, config=config, llm=mock_llm)

    assert result["final_answer"] == "Hello! How can I help?"


@pytest.mark.asyncio
async def test_retrieval_searches_vectorstore(mock_vectorstore, mock_embedding):
    state = GraphState(query="What is LangGraph?", route="rag")
    config = RetrievalConfig(top_k=5, score_threshold=0.7, default_collection="docs")

    result = await retrieval.run(
        state,
        config=config,
        vectorstore=mock_vectorstore,
        embedding=mock_embedding,
    )

    mock_embedding.embed.assert_called_once_with(["What is LangGraph?"])
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieval_scores"][0] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_answer_generation_produces_draft(mock_llm):
    mock_llm.complete = AsyncMock(
        return_value={
            "text": "LangGraph is a library [chunk-1] for building stateful agents.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
    )
    state = GraphState(
        query="What is LangGraph?",
        route="rag",
        retrieved_chunks=[
            Chunk(
                id="chunk-1", text="LangGraph builds stateful agents.", collection="docs", score=0.9
            ),
        ],
        retrieval_scores=[0.9],
    )
    config = AnswerGenerationConfig()

    result = await answer_generation.run(state, config=config, llm=mock_llm)

    assert "LangGraph" in result["draft_answer"]
    assert len(result["execution_trace"]) == 1


@pytest.mark.asyncio
async def test_tool_agent_calls_list_collections(
    mock_llm,
    mock_vectorstore,
    mock_collection_store,
    mock_embedding,
):
    # First call: LLM requests tool use. Second call: LLM produces final text.
    mock_llm.complete = AsyncMock(
        side_effect=[
            {
                "text": "",
                "tool_use": [{"name": "list_collections", "input": {}, "id": "tool_1"}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            {
                "text": "Available collections: docs, test",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        ]
    )
    state = GraphState(query="list collections", route="tool")
    config = ToolAgentConfig(allowed_tools=["list_collections"], max_tool_calls=3)

    result = await tool_agent.run(
        state,
        config=config,
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    assert result["final_answer"] == "Available collections: docs, test"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].name == "list_collections"
