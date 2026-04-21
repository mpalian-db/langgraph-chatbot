"""Integration tests for the end-to-end ingestion pipeline.

Requires Qdrant (Docker) and Ollama running locally.
Run with: just test-all  or  pytest -m integration
"""

from __future__ import annotations

import uuid

import pytest

from app.adapters.embeddings.ollama import OllamaEmbeddingAdapter
from app.adapters.storage.local import LocalFileStorageAdapter
from app.adapters.vectorstore.qdrant import QdrantVectorStoreAdapter
from app.ingestion.pipeline import ingest_document

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768


@pytest.fixture
async def qdrant():
    return QdrantVectorStoreAdapter(url=QDRANT_URL)


@pytest.fixture
async def embedding():
    return OllamaEmbeddingAdapter(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)


@pytest.fixture
async def storage(tmp_path):
    return LocalFileStorageAdapter(base_dir=tmp_path)


@pytest.fixture
async def test_collection(qdrant):
    name = f"test-ingest-{uuid.uuid4().hex[:8]}"
    await qdrant.create(name, VECTOR_SIZE)
    yield name
    await qdrant.delete_collection(name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_stores_chunks_and_makes_them_searchable(
    qdrant, embedding, storage, test_collection
):
    content = b"LangGraph is a library for building stateful, multi-actor applications with LLMs."

    chunk_count = await ingest_document(
        filename="test.txt",
        content=content,
        collection=test_collection,
        storage=storage,
        embedding=embedding,
        vectorstore=qdrant,
    )

    assert chunk_count >= 1

    query_vector = (await embedding.embed(["LangGraph stateful applications"]))[0]
    results = await qdrant.search(
        query_vector=query_vector,
        top_k=5,
        collection=test_collection,
    )
    assert len(results) >= 1
    assert any("LangGraph" in r.text for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_attaches_filename_metadata(qdrant, embedding, storage, test_collection):
    content = b"Nodes in LangGraph are ordinary Python functions."

    await ingest_document(
        filename="langgraph-intro.md",
        content=content,
        collection=test_collection,
        storage=storage,
        embedding=embedding,
        vectorstore=qdrant,
    )

    query_vector = (await embedding.embed(["LangGraph nodes"]))[0]
    results = await qdrant.search(
        query_vector=query_vector,
        top_k=5,
        collection=test_collection,
    )
    assert results
    assert results[0].metadata.get("filename") == "langgraph-intro.md"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_multi_chunk_document(qdrant, embedding, storage, test_collection):
    # A document long enough to produce multiple chunks at the default chunk_size of 512.
    paragraph = "LangGraph enables building complex agentic workflows. " * 20
    content = (paragraph * 3).encode()

    chunk_count = await ingest_document(
        filename="long-doc.txt",
        content=content,
        collection=test_collection,
        storage=storage,
        embedding=embedding,
        vectorstore=qdrant,
        chunk_size=256,
        chunk_overlap=32,
    )

    assert chunk_count > 1

    query_vector = (await embedding.embed(["agentic workflows"]))[0]
    results = await qdrant.search(
        query_vector=query_vector,
        top_k=10,
        collection=test_collection,
    )
    assert len(results) > 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_stores_file_on_disk(qdrant, embedding, storage, test_collection, tmp_path):
    content = b"Edges in LangGraph connect nodes."

    await ingest_document(
        filename="edges.txt",
        content=content,
        collection=test_collection,
        storage=storage,
        embedding=embedding,
        vectorstore=qdrant,
    )

    stored = await storage.retrieve("edges.txt")
    assert stored == content
