"""Notion sync endpoint -- fetches pages from a Notion database and ingests them into Qdrant."""

from __future__ import annotations

import hashlib
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.dependencies import (
    EmbeddingDep,
    NotionDep,
    SystemConfigDep,
    VectorStoreDep,
)
from app.core.models.types import Chunk
from app.ingestion.chunker import chunk_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["notion"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class NotionSyncResponse(BaseModel):
    collection: str
    pages_synced: int
    total_chunks: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deterministic_id(page_id: str, chunk_index: int) -> str:
    raw = f"notion:{page_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/{collection}/sync-notion", response_model=NotionSyncResponse, status_code=200)
async def sync_notion(
    collection: str,
    system_config: SystemConfigDep,
    notion: NotionDep,
    embedding: EmbeddingDep,
    vectorstore: VectorStoreDep,
) -> NotionSyncResponse:
    """Fetch all pages from the configured Notion database and ingest into a collection."""
    database_id = os.environ.get("NOTION_DATABASE_ID", "")
    if not database_id:
        raise HTTPException(
            status_code=400, detail="NOTION_DATABASE_ID environment variable is not set"
        )

    pages = await notion.list_pages(database_id)
    total_chunks = 0

    for page in pages:
        full_page = await notion.get_page_content(page.id)

        if not full_page.text.strip():
            logger.info("Skipping empty page: %s (%s)", full_page.title, page.id)
            continue

        raw_chunks = chunk_text(
            full_page.text,
            full_page.title or page.id,
            collection,
            system_config.ingestion.chunk_size,
            system_config.ingestion.chunk_overlap,
        )

        chunks = [
            Chunk(
                id=_deterministic_id(page.id, i),
                text=c["text"],
                collection=collection,
                metadata={
                    "source": "notion",
                    "page_id": page.id,
                    "title": full_page.title,
                    "url": full_page.url,
                    "chunk_index": i,
                },
            )
            for i, c in enumerate(raw_chunks)
        ]

        if not chunks:
            continue

        texts = [c.text for c in chunks]
        vectors = await embedding.embed(texts)
        await vectorstore.upsert(collection, chunks, vectors)
        total_chunks += len(chunks)

    logger.info(
        "Notion sync: %d pages, %d chunks into collection %s",
        len(pages),
        total_chunks,
        collection,
    )
    return NotionSyncResponse(
        collection=collection,
        pages_synced=len(pages),
        total_chunks=total_chunks,
    )
