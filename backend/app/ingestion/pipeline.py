from __future__ import annotations

from app.core.models.types import Chunk
from app.ingestion.chunker import chunk_text
from app.ports.embedding import EmbeddingPort
from app.ports.storage import DocumentStoragePort
from app.ports.vectorstore import VectorStorePort


async def ingest_document(
    filename: str,
    content: bytes,
    collection: str,
    storage: DocumentStoragePort,
    embedding: EmbeddingPort,
    vectorstore: VectorStorePort,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    """Store, chunk, embed, and upsert a document. Returns number of chunks created."""
    await storage.store(filename, content)

    text = content.decode("utf-8", errors="replace")
    raw_chunks = chunk_text(text, filename, collection, chunk_size, chunk_overlap)

    chunks = [
        Chunk(
            id=c["id"],
            text=c["text"],
            collection=collection,
            metadata=c["metadata"],
        )
        for c in raw_chunks
    ]

    texts = [c.text for c in chunks]
    vectors = await embedding.embed(texts)
    await vectorstore.upsert(collection, chunks, vectors)

    return len(chunks)
