"""Collection management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.dependencies import CollectionDep

router = APIRouter(prefix="/collections", tags=["collections"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateCollectionRequest(BaseModel):
    name: str
    vector_size: int = 768


class CollectionStatsResponse(BaseModel):
    name: str
    vectors_count: int
    points_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_collections(collection_store: CollectionDep) -> list[str]:
    """Return all collection names."""
    return await collection_store.list_collections()


@router.post("", status_code=201)
async def create_collection(
    body: CreateCollectionRequest,
    collection_store: CollectionDep,
) -> dict[str, str]:
    """Create a new vector collection."""
    try:
        await collection_store.create(body.name, body.vector_size)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"name": body.name, "status": "created"}


@router.get("/{name}", response_model=CollectionStatsResponse)
async def get_collection_stats(
    name: str,
    collection_store: CollectionDep,
) -> CollectionStatsResponse:
    """Return statistics for a single collection."""
    try:
        stats = await collection_store.get_stats(name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CollectionStatsResponse(**stats)


@router.delete("/{name}", status_code=204)
async def delete_collection(
    name: str,
    collection_store: CollectionDep,
) -> None:
    """Delete a collection and all its vectors."""
    try:
        await collection_store.delete_collection(name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{name}/rebuild", status_code=200)
async def rebuild_collection(
    name: str,
    collection_store: CollectionDep,
    vector_size: int = 768,
) -> dict[str, str]:
    """Delete and recreate a collection, discarding all existing vectors.

    The vector dimensionality defaults to 768 (nomic-embed-text).  Pass
    ``vector_size`` as a query parameter to override.
    """
    try:
        await collection_store.delete_collection(name)
    except Exception:
        # Collection may not exist yet -- proceed to create.
        pass
    try:
        await collection_store.create(name, vector_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"name": name, "status": "rebuilt"}
