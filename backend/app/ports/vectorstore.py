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


@runtime_checkable
class CollectionPort(Protocol):
    async def create(self, name: str, vector_size: int) -> None: ...
    async def list_collections(self) -> list[str]: ...
    async def delete(self, name: str) -> None: ...
    async def get_stats(self, name: str) -> dict: ...
