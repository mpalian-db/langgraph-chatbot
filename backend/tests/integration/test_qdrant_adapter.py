from __future__ import annotations

import uuid

import pytest

from app.adapters.vectorstore.qdrant import QdrantVectorStoreAdapter
from app.core.models.types import Chunk

VECTOR_SIZE = 4


@pytest.fixture
async def qdrant():
    adapter = QdrantVectorStoreAdapter(url="http://localhost:6333")
    yield adapter


@pytest.fixture
async def test_collection(qdrant):
    name = f"test-{uuid.uuid4().hex[:8]}"
    await qdrant.create(name, VECTOR_SIZE)
    yield name
    await qdrant.delete_collection(name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_and_search(qdrant, test_collection):
    chunks = [
        Chunk(id=str(uuid.uuid4()), text="LangGraph builds stateful agents.", collection=test_collection),
    ]
    vectors = [[0.1, 0.2, 0.3, 0.4]]
    await qdrant.upsert(test_collection, chunks, vectors)

    results = await qdrant.search(
        query_vector=[0.1, 0.2, 0.3, 0.4],
        top_k=5,
        collection=test_collection,
    )
    assert len(results) == 1
    assert results[0].text == "LangGraph builds stateful agents."


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_collections(qdrant, test_collection):
    names = await qdrant.list_collections()
    assert test_collection in names
