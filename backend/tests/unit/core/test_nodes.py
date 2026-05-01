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
async def test_chat_agent_includes_history_in_prompt_messages(mock_llm):
    """Conversation memory: prior turns must be fed to the LLM as messages,
    in chronological order, with the new query appended last. This is what
    makes follow-up references like "what about the second one?" work."""
    from app.ports.conversation import Turn

    mock_llm.complete = AsyncMock(
        return_value={
            "text": "The second item is X.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 8},
        }
    )
    state = GraphState(
        query="what about the second one?",
        route="chat",
        conversation_id="conv-1",
        history=[
            Turn(role="user", content="list three items"),
            Turn(role="assistant", content="A, B, C"),
        ],
    )
    config = ChatAgentConfig()

    await chat_agent.run(state, config=config, llm=mock_llm)

    call_kwargs = mock_llm.complete.call_args.kwargs
    messages = call_kwargs["messages"]
    # History first (oldest -> newest), then the new query.
    assert messages[0] == {"role": "user", "content": "list three items"}
    assert messages[1] == {"role": "assistant", "content": "A, B, C"}
    assert messages[2] == {"role": "user", "content": "what about the second one?"}


@pytest.mark.asyncio
async def test_chat_agent_omits_history_when_empty(mock_llm):
    """Backward compat for stateless single-turn chats: empty history must
    produce a single-message prompt, identical to pre-memory behaviour."""
    state = GraphState(query="hi", route="chat", history=[])
    config = ChatAgentConfig()

    await chat_agent.run(state, config=config, llm=mock_llm)

    messages = mock_llm.complete.call_args.kwargs["messages"]
    assert messages == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_chat_agent_prepends_conversation_summary_to_system_prompt(mock_llm):
    """The summariser may compress distant turns into a rolling summary.
    chat_agent must surface that summary to the model as system-level prior
    context, otherwise the compression silently strips information from
    the conversation."""
    state = GraphState(
        query="what about the second one?",
        route="chat",
        history=[],
        conversation_summary="Earlier: user listed 3 items A, B, C; assistant explained A.",
    )
    config = ChatAgentConfig(system_prompt="You are a helpful assistant.")

    await chat_agent.run(state, config=config, llm=mock_llm)

    system_arg = mock_llm.complete.call_args.kwargs["system"]
    assert "Earlier: user listed 3 items" in system_arg
    assert "You are a helpful assistant." in system_arg
    # Summary must appear BEFORE the system prompt so model reads context first.
    assert system_arg.index("Earlier:") < system_arg.index("You are a helpful")


@pytest.mark.asyncio
async def test_chat_agent_uses_unmodified_system_prompt_when_no_summary(mock_llm):
    """No summary -> system prompt is exactly what config provides; no
    leading "Summary of earlier..." preamble that would confuse the model."""
    state = GraphState(query="hi", history=[], conversation_summary=None)
    config = ChatAgentConfig(system_prompt="You are a helpful assistant.")

    await chat_agent.run(state, config=config, llm=mock_llm)

    assert mock_llm.complete.call_args.kwargs["system"] == "You are a helpful assistant."


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
async def test_retrieval_reranks_by_score_descending(mock_embedding):
    # Return chunks in ascending score order; after rerank they should be descending.
    from unittest.mock import AsyncMock as _AM

    low = Chunk(id="low", text="less relevant content here", collection="docs", score=0.5)
    high = Chunk(id="high", text="more relevant content here", collection="docs", score=0.9)

    vs = _AM()
    vs.search = _AM(return_value=[low, high])  # low score first, as if vectorstore returned them

    state = GraphState(query="What is LangGraph?", route="rag")
    config = RetrievalConfig(top_k=5, score_threshold=0.4, rerank=True)

    result = await retrieval.run(state, config=config, vectorstore=vs, embedding=mock_embedding)

    assert result["retrieved_chunks"][0].id == "high"
    assert result["retrieved_chunks"][1].id == "low"


@pytest.mark.asyncio
async def test_retrieval_skips_rerank_when_disabled(mock_embedding):
    from unittest.mock import AsyncMock as _AM

    low = Chunk(id="low", text="less relevant content here", collection="docs", score=0.5)
    high = Chunk(id="high", text="more relevant content here", collection="docs", score=0.9)

    vs = _AM()
    vs.search = _AM(return_value=[low, high])

    state = GraphState(query="What is LangGraph?", route="rag")
    config = RetrievalConfig(top_k=5, score_threshold=0.4, rerank=False)

    result = await retrieval.run(state, config=config, vectorstore=vs, embedding=mock_embedding)

    # Original order preserved when rerank=False.
    assert result["retrieved_chunks"][0].id == "low"
    assert result["retrieved_chunks"][1].id == "high"


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


# ---------------------------------------------------------------------------
# Citation extraction across multiple chunk ID formats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_generation_extracts_uuid_citations(mock_llm):
    """Chunker-produced IDs are uuid4 (36 chars, hex with hyphens)."""
    uuid_id = "abcd1234-5678-90ab-cdef-1234567890ab"
    mock_llm.complete = AsyncMock(
        return_value={
            "text": f"LangGraph [{uuid_id}] is stateful.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
    )
    state = GraphState(
        query="What is LangGraph?",
        retrieved_chunks=[
            Chunk(id=uuid_id, text="LangGraph is stateful.", collection="docs", score=0.9),
        ],
    )

    result = await answer_generation.run(state, config=AnswerGenerationConfig(), llm=mock_llm)

    assert len(result["citations"]) == 1
    assert result["citations"][0].chunk_id == uuid_id


@pytest.mark.asyncio
async def test_answer_generation_extracts_sha256_prefix_citations(mock_llm):
    """Notion sync produces IDs as sha256 hexdigest()[:32] -- 32 hex chars,
    no hyphens. The previous regex r'[a-f0-9\\-]{36}' missed these entirely,
    so Notion answers never populated state.citations."""
    sha_id = "a" * 32  # 32 hex chars, no hyphens, like _deterministic_id().
    mock_llm.complete = AsyncMock(
        return_value={
            "text": f"Per the source [{sha_id}], LangGraph is stateful.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
    )
    state = GraphState(
        query="What is LangGraph?",
        retrieved_chunks=[
            Chunk(id=sha_id, text="LangGraph is stateful.", collection="notion", score=0.9),
        ],
    )

    result = await answer_generation.run(state, config=AnswerGenerationConfig(), llm=mock_llm)

    assert len(result["citations"]) == 1
    assert result["citations"][0].chunk_id == sha_id


@pytest.mark.asyncio
async def test_answer_generation_ignores_brackets_that_are_not_real_chunks(mock_llm):
    """Bracketed tokens that aren't in the retrieved_chunks must not produce
    citations. The LLM may invent IDs; we never trust them blindly."""
    real_id = "real-chunk-001"
    mock_llm.complete = AsyncMock(
        return_value={
            "text": f"Real [{real_id}] and fake [hallucinated-id] citations mixed.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
    )
    state = GraphState(
        query="x",
        retrieved_chunks=[
            Chunk(id=real_id, text="Real content.", collection="docs", score=0.9),
        ],
    )

    result = await answer_generation.run(state, config=AnswerGenerationConfig(), llm=mock_llm)

    assert len(result["citations"]) == 1
    assert result["citations"][0].chunk_id == real_id


@pytest.mark.asyncio
async def test_answer_generation_deduplicates_repeated_citations(mock_llm):
    """If the LLM cites the same chunk multiple times, the citation list
    should still contain it only once."""
    cid = "abcd1234-5678-90ab-cdef-1234567890ab"
    mock_llm.complete = AsyncMock(
        return_value={
            "text": f"First mention [{cid}]. Second mention [{cid}].",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
    )
    state = GraphState(
        query="x",
        retrieved_chunks=[Chunk(id=cid, text="Content.", collection="docs", score=0.9)],
    )

    result = await answer_generation.run(state, config=AnswerGenerationConfig(), llm=mock_llm)

    assert len(result["citations"]) == 1


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


@pytest.mark.asyncio
async def test_tool_agent_calls_upload_document(
    mock_llm,
    mock_vectorstore,
    mock_collection_store,
    mock_embedding,
):
    """upload_document chunks the text, embeds it, and upserts vectors."""
    mock_llm.complete = AsyncMock(
        side_effect=[
            {
                "text": "",
                "tool_use": [
                    {
                        "name": "upload_document",
                        "input": {
                            "collection": "docs",
                            "filename": "note.md",
                            "text": "LangGraph is a framework for stateful agents. " * 10,
                        },
                        "id": "tool_1",
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            {
                "text": "Uploaded successfully.",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        ]
    )
    state = GraphState(query="upload this", route="tool")
    config = ToolAgentConfig(allowed_tools=["upload_document"], max_tool_calls=3)

    result = await tool_agent.run(
        state,
        config=config,
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    # Embedding port called once with the chunk texts.
    mock_embedding.embed.assert_called_once()
    embedded_texts = mock_embedding.embed.call_args.args[0]
    assert len(embedded_texts) >= 1
    assert all(isinstance(t, str) for t in embedded_texts)

    # Vectorstore upsert called with the same collection and matching counts.
    mock_vectorstore.upsert.assert_called_once()
    coll_arg, chunks_arg, vectors_arg = mock_vectorstore.upsert.call_args.args
    assert coll_arg == "docs"
    assert len(chunks_arg) == len(vectors_arg)
    assert len(chunks_arg) == len(embedded_texts)

    # The tool result is recorded with the chunk count.
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].name == "upload_document"
    assert "chunks_ingested" in result["tool_calls"][0].result


@pytest.mark.asyncio
async def test_tool_agent_calls_delete_document(
    mock_llm,
    mock_vectorstore,
    mock_collection_store,
    mock_embedding,
):
    """delete_document forwards the chunk IDs to the vectorstore."""
    mock_llm.complete = AsyncMock(
        side_effect=[
            {
                "text": "",
                "tool_use": [
                    {
                        "name": "delete_document",
                        "input": {"collection": "docs", "ids": ["c1", "c2", "c3"]},
                        "id": "tool_1",
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            {
                "text": "Deleted 3 chunks.",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        ]
    )
    state = GraphState(query="delete those chunks", route="tool")
    config = ToolAgentConfig(allowed_tools=["delete_document"], max_tool_calls=3)

    result = await tool_agent.run(
        state,
        config=config,
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    mock_vectorstore.delete.assert_called_once_with("docs", ["c1", "c2", "c3"])
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].name == "delete_document"
    assert "deleted" in result["tool_calls"][0].result


@pytest.mark.asyncio
async def test_tool_agent_soft_denies_disallowed_tool(
    mock_llm,
    mock_vectorstore,
    mock_collection_store,
    mock_embedding,
):
    """A tool name not in allowed_tools is soft-denied: no port is touched and
    the LLM receives a denial string in place of a real tool result."""
    mock_llm.complete = AsyncMock(
        side_effect=[
            {
                "text": "",
                "tool_use": [
                    {
                        "name": "delete_document",
                        "input": {"collection": "docs", "ids": ["c1"]},
                        "id": "tool_1",
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            {
                "text": "Sorry, I cannot delete documents.",
                "tool_use": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        ]
    )
    state = GraphState(query="delete c1", route="tool")
    # delete_document is NOT in allowed_tools.
    config = ToolAgentConfig(allowed_tools=["list_collections"], max_tool_calls=3)

    result = await tool_agent.run(
        state,
        config=config,
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    # No outbound side effects.
    mock_vectorstore.delete.assert_not_called()
    mock_vectorstore.upsert.assert_not_called()

    # The attempt is still recorded in the trace, with a denial message.
    assert len(result["tool_calls"]) == 1
    recorded = result["tool_calls"][0]
    assert recorded.name == "delete_document"
    assert "not permitted" in recorded.result.lower()
    assert result["final_answer"] == "Sorry, I cannot delete documents."
