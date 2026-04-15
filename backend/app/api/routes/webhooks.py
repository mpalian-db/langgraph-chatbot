"""Webhook endpoints for external integrations."""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.api.dependencies import (
    EmbeddingDep,
    SystemConfigDep,
    VectorStoreDep,
)
from app.core.models.types import Chunk
from app.ingestion.chunker import chunk_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class NoteWebhookPayload(BaseModel):
    event: Literal["note.created", "note.updated", "note.deleted", "note.restored"]
    note_id: str
    title: str
    content: str
    tags: list[str] = []
    timestamp: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deterministic_id(note_id: str, chunk_index: int) -> str:
    """Generate a stable UUID-shaped ID from note_id and chunk index.

    Using a deterministic scheme means upserting the same note twice produces
    the same point IDs, so Qdrant overwrites in place rather than leaving
    orphaned chunks.
    """
    raw = f"{note_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _chunk_note(
    note_id: str,
    title: str,
    content: str,
    tags: list[str],
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk a note's content and return Chunks with deterministic IDs."""
    full_text = f"{title}\n\n{content}" if title else content
    raw_chunks = chunk_text(full_text, note_id, collection, chunk_size, chunk_overlap)

    return [
        Chunk(
            id=_deterministic_id(note_id, i),
            text=c["text"],
            collection=collection,
            metadata={
                "note_id": note_id,
                "title": title,
                "tags": tags,
                "chunk_index": i,
            },
        )
        for i, c in enumerate(raw_chunks)
    ]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/notes", status_code=204)
async def handle_note_webhook(
    payload: NoteWebhookPayload,
    system_config: SystemConfigDep,
    embedding: EmbeddingDep,
    vectorstore: VectorStoreDep,
    x_webhook_secret: str | None = Header(default=None),
) -> None:
    """Receive note lifecycle events from EdgeNotes and sync to Qdrant."""
    expected_secret = system_config.webhooks.edgenotes_secret
    if expected_secret and x_webhook_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    collection = system_config.webhooks.edgenotes_collection

    if payload.event == "note.deleted":
        # Generate the same deterministic IDs used at insert time.
        # 200 is a generous upper bound -- Qdrant ignores missing IDs.
        chunk_ids = [_deterministic_id(payload.note_id, i) for i in range(200)]
        try:
            await vectorstore.delete(collection, ids=chunk_ids)
        except Exception:
            logger.warning(
                "Failed to delete chunks for note %s -- collection may not exist yet",
                payload.note_id,
                exc_info=True,
            )
        return

    # For create, update, and restore: upsert chunks.
    chunks = _chunk_note(
        note_id=payload.note_id,
        title=payload.title,
        content=payload.content,
        tags=payload.tags,
        collection=collection,
        chunk_size=system_config.ingestion.chunk_size,
        chunk_overlap=system_config.ingestion.chunk_overlap,
    )

    if not chunks:
        logger.info("Note %s produced no chunks -- skipping", payload.note_id)
        return

    texts = [c.text for c in chunks]
    vectors = await embedding.embed(texts)
    await vectorstore.upsert(collection, chunks, vectors)
    logger.info(
        "Webhook %s: upserted %d chunks for note %s into %s",
        payload.event,
        len(chunks),
        payload.note_id,
        collection,
    )
