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
