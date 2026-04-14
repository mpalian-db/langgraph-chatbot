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
        llm=mock_llm,
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
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    result = await graph.ainvoke(GraphState(query="hello"))

    assert result["route"] == "chat"
    assert result["final_answer"] == "Hello! I can help with that."
