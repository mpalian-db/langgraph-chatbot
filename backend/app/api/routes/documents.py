"""Document ingestion and listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.dependencies import (
    EmbeddingDep,
    StorageDep,
    SystemConfigDep,
    VectorStoreDep,
)
from app.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/collections", tags=["documents"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    filename: str
    collection: str
    chunk_count: int


class DocumentOut(BaseModel):
    id: str
    text: str
    collection: str
    metadata: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/{collection}/documents", response_model=IngestResponse, status_code=201)
async def upload_document(
    collection: str,
    file: UploadFile,
    system_config: SystemConfigDep,
    storage: StorageDep,
    embedding: EmbeddingDep,
    vectorstore: VectorStoreDep,
) -> IngestResponse:
    """Ingest a document: store, chunk, embed, and upsert into the vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()

    chunk_count = await ingest_document(
        file.filename,
        content,
        collection,
        storage,
        embedding,
        vectorstore,
        chunk_size=system_config.ingestion.chunk_size,
        chunk_overlap=system_config.ingestion.chunk_overlap,
    )

    return IngestResponse(
        filename=file.filename,
        collection=collection,
        chunk_count=chunk_count,
    )


@router.get("/{collection}/documents", response_model=list[DocumentOut])
async def list_documents(
    collection: str,
    vectorstore: VectorStoreDep,
    limit: int = 100,
    offset: int = 0,
) -> list[DocumentOut]:
    """List documents (chunks) in a collection."""
    chunks = await vectorstore.list_documents(collection, limit=limit, offset=offset)
    return [
        DocumentOut(
            id=c.id,
            text=c.text,
            collection=c.collection,
            metadata=c.metadata,
        )
        for c in chunks
    ]


@router.get("/{collection}/documents/{chunk_id}", response_model=DocumentOut)
async def get_document(
    collection: str,
    chunk_id: str,
    vectorstore: VectorStoreDep,
) -> DocumentOut:
    """Retrieve a single chunk by its id. Returns 404 when not found.

    Note on naming: the system stores chunks (sub-document pieces produced by
    the chunker), not whole documents -- a single uploaded file becomes many
    chunks. The path uses `documents` for spec compatibility but the unit of
    retrieval is a chunk."""
    chunk = await vectorstore.get_chunk(collection, chunk_id)
    if chunk is None:
        raise HTTPException(
            status_code=404, detail=f"chunk {chunk_id!r} not found in {collection!r}"
        )
    return DocumentOut(
        id=chunk.id,
        text=chunk.text,
        collection=chunk.collection,
        metadata=chunk.metadata,
    )


@router.delete("/{collection}/documents/{chunk_id}", status_code=204)
async def delete_document(
    collection: str,
    chunk_id: str,
    vectorstore: VectorStoreDep,
) -> None:
    """Delete a single chunk by id. Idempotent -- a missing chunk is treated
    as already-deleted (204), matching how vectorstore.delete() handles
    unknown ids."""
    await vectorstore.delete(collection, [chunk_id])
