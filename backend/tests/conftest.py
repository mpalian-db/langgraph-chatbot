from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.config.models import AgentsConfig, SystemConfig
from app.core.models.types import Chunk


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value={
            "text": "mock response",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
    )
    return llm


@pytest.fixture
def mock_vectorstore():
    vs = AsyncMock()
    vs.search = AsyncMock(
        return_value=[
            Chunk(
                id="chunk-1",
                text="LangGraph is a library for building stateful agents.",
                collection="test",
                score=0.9,
            ),
        ]
    )
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()
    return vs


@pytest.fixture
def mock_collection_store():
    cs = AsyncMock()
    cs.create = AsyncMock()
    cs.list_collections = AsyncMock(return_value=["docs", "test"])
    cs.delete_collection = AsyncMock()
    cs.get_stats = AsyncMock(
        return_value={"name": "docs", "vectors_count": 100, "points_count": 100}
    )
    return cs


@pytest.fixture
def mock_embedding():
    emb = AsyncMock()
    emb.embed = AsyncMock(return_value=[[0.1] * 384])
    return emb


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.store = AsyncMock(return_value="/tmp/test.txt")
    storage.retrieve = AsyncMock(return_value=b"content")
    storage.delete = AsyncMock()
    return storage


@pytest.fixture
def agents_config():
    return AgentsConfig()


@pytest.fixture
def system_config():
    return SystemConfig()
