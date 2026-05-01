from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.core.models.types import Chunk


@runtime_checkable
class VectorStorePort(Protocol):
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        collection: str,
        filters: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[Chunk]: ...

    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None: ...

    async def delete(self, collection: str, ids: list[str]) -> None: ...

    async def get_chunk(self, collection: str, chunk_id: str) -> Chunk | None:
        """Retrieve a single chunk by its id. Returns None when not found.

        This is the read-by-id counterpart to `delete()` -- useful for the
        `GET /collections/{name}/documents/{id}` endpoint where we want
        404 semantics rather than the current "list and filter" approach."""
        ...

    async def list_documents(
        self,
        collection: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]: ...


@runtime_checkable
class CollectionPort(Protocol):
    async def create(self, name: str, vector_size: int) -> None: ...
    async def list_collections(self) -> list[str]: ...
    async def delete_collection(self, name: str) -> None: ...
    async def get_stats(self, name: str) -> dict: ...
