from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.ingestion.pipeline import ingest_document


@pytest.mark.asyncio
async def test_ingest_document_stores_chunks_and_embedds():
    mock_storage = AsyncMock()
    mock_storage.store = AsyncMock(return_value="/tmp/test.md")

    mock_embedding = AsyncMock()
    mock_embedding.embed = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])

    mock_vectorstore = AsyncMock()
    mock_vectorstore.upsert = AsyncMock()

    count = await ingest_document(
        filename="test.md",
        content=b"A " * 600,
        collection="docs",
        storage=mock_storage,
        embedding=mock_embedding,
        vectorstore=mock_vectorstore,
        chunk_size=512,
        chunk_overlap=64,
    )

    assert count >= 2
    mock_storage.store.assert_called_once()
    mock_embedding.embed.assert_called_once()
    mock_vectorstore.upsert.assert_called_once()
