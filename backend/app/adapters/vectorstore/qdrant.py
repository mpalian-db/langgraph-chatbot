from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.core.models.types import Chunk


class QdrantVectorStoreAdapter:
    def __init__(self, url: str) -> None:
        self._client = AsyncQdrantClient(url=url)

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        collection: str,
        filters: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[Chunk]:
        qdrant_filter = _build_filter(filters) if filters else None
        results = await self._client.search(  # type: ignore[attr-defined]
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [_scored_point_to_chunk(r, collection) for r in results]

    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={"text": chunk.text, "chunk_id": chunk.id, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        await self._client.upsert(collection_name=collection, points=points)

    async def delete(self, collection: str, ids: list[str]) -> None:
        await self._client.delete(collection_name=collection, points_selector=ids)  # type: ignore[arg-type]

    async def create(self, name: str, vector_size: int) -> None:
        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    async def list_collections(self) -> list[str]:
        result = await self._client.get_collections()
        return [c.name for c in result.collections]

    async def delete_collection(self, name: str) -> None:
        await self._client.delete_collection(collection_name=name)

    async def list_documents(
        self,
        collection: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        """Scroll through points in a collection and return them as Chunks."""
        result = await self._client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, _next_offset = result
        return [
            Chunk(
                id=str(p.id),
                text=(p.payload or {}).get("text", ""),
                collection=collection,
                metadata={
                    k: v for k, v in (p.payload or {}).items() if k not in ("text", "chunk_id")
                },
            )
            for p in points
        ]

    async def get_stats(self, name: str) -> dict:
        info = await self._client.get_collection(collection_name=name)
        return {
            "name": name,
            "vectors_count": info.vectors_count,  # type: ignore[attr-defined]
            "points_count": info.points_count,  # type: ignore[attr-defined]
        }


def _build_filter(filters: dict) -> Filter:
    conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
    return Filter(must=conditions)  # type: ignore[arg-type]


def _scored_point_to_chunk(point: ScoredPoint, collection: str) -> Chunk:
    payload = point.payload or {}
    return Chunk(
        id=str(point.id),
        text=payload.get("text", ""),
        collection=collection,
        score=point.score,
        metadata={k: v for k, v in payload.items() if k not in ("text", "chunk_id")},
    )
