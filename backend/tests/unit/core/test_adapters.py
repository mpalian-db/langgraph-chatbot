from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.adapters.storage.local import LocalFileStorageAdapter
from app.adapters.llm.ollama import OllamaLLMAdapter


@pytest.mark.asyncio
async def test_store_and_retrieve(tmp_path):
    adapter = LocalFileStorageAdapter(base_dir=tmp_path)
    path = await adapter.store("test.txt", b"hello world")
    content = await adapter.retrieve(path)
    assert content == b"hello world"


@pytest.mark.asyncio
async def test_delete_removes_file(tmp_path):
    adapter = LocalFileStorageAdapter(base_dir=tmp_path)
    path = await adapter.store("test.txt", b"data")
    await adapter.delete(path)
    with pytest.raises(FileNotFoundError):
        await adapter.retrieve(path)


@pytest.mark.asyncio
async def test_ollama_adapter_formats_response():
    mock_message = MagicMock()
    mock_message.content = "Hello there"
    mock_message.tool_calls = None

    mock_response = MagicMock()
    mock_response.message = mock_message
    mock_response.prompt_eval_count = 12
    mock_response.eval_count = 6

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)

    adapter = OllamaLLMAdapter(client=mock_client)
    result = await adapter.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="llama3.2:3b",
        system="Be helpful.",
        max_tokens=100,
    )

    assert result["text"] == "Hello there"
    assert result["stop_reason"] == "end_turn"
    assert result["tool_use"] == []
    assert result["usage"]["input_tokens"] == 12


@pytest.mark.asyncio
async def test_ollama_adapter_passes_tools():
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "search"
    mock_tool_call.function.arguments = {"query": "test"}

    mock_message = MagicMock()
    mock_message.content = ""
    mock_message.tool_calls = [mock_tool_call]

    mock_response = MagicMock()
    mock_response.message = mock_message
    mock_response.prompt_eval_count = 8
    mock_response.eval_count = 4

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)

    adapter = OllamaLLMAdapter(client=mock_client)
    tools = [{"name": "search", "description": "search docs", "parameters": {"type": "object"}}]
    result = await adapter.complete(
        messages=[{"role": "user", "content": "search something"}],
        model="llama3.2:3b",
        tools=tools,
        max_tokens=100,
    )

    assert result["stop_reason"] == "tool_use"
    assert result["tool_use"][0]["name"] == "search"


@pytest.mark.asyncio
async def test_anthropic_adapter_formats_response():
    from app.adapters.llm.anthropic import AnthropicLLMAdapter

    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Hello there")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    adapter = AnthropicLLMAdapter(client=mock_client)
    result = await adapter.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="claude-haiku-4-5-20251001",
        system="Be helpful.",
        max_tokens=100,
    )

    assert result["text"] == "Hello there"
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 10


@pytest.mark.asyncio
async def test_anthropic_adapter_passes_tools():
    from app.adapters.llm.anthropic import AnthropicLLMAdapter

    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="result")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=5, output_tokens=3)

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    adapter = AnthropicLLMAdapter(client=mock_client)
    tools = [{"name": "search", "description": "search docs", "input_schema": {"type": "object"}}]
    await adapter.complete(
        messages=[{"role": "user", "content": "search something"}],
        model="claude-haiku-4-5-20251001",
        tools=tools,
        max_tokens=100,
    )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tools"] == tools


@pytest.mark.asyncio
async def test_workers_ai_embedding_returns_vectors():
    from app.adapters.embeddings.workers_ai import WorkersAIEmbeddingAdapter

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {"data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
        "success": True,
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    adapter = WorkersAIEmbeddingAdapter(
        account_id="test-account",
        api_token="test-token",
        model="@cf/baai/bge-small-en-v1.5",
        http_client=mock_client,
    )
    vectors = await adapter.embed(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert vectors[0][0] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_ollama_embedding_returns_vectors():
    from app.adapters.embeddings.ollama import OllamaEmbeddingAdapter

    mock_response = MagicMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=mock_response)

    adapter = OllamaEmbeddingAdapter(
        model="nomic-embed-text",
        client=mock_client,
    )
    vectors = await adapter.embed(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert vectors[0][0] == pytest.approx(0.1)
    mock_client.embed.assert_awaited_once_with(model="nomic-embed-text", input=["hello", "world"])
