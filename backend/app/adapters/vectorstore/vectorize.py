"""Cloudflare Vectorize adapter -- implements VectorStorePort and CollectionPort.

A Vectorize index does not have sub-collections; logical collection separation is
achieved by storing a `collection` field in each vector's metadata and filtering on it
at query time. The CollectionPort management operations (create/delete) are no-ops
because Vectorize index provisioning is done via Wrangler outside the application.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.models.types import Chunk

logger = logging.getLogger(__name__)

_VECTORIZE_BASE = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/vectorize/v2/indexes/{index_name}"
)


class VectorizeAdapter:
    """Cloudflare Vectorize implementation of VectorStorePort + CollectionPort."""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        index_name: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = _VECTORIZE_BASE.format(account_id=account_id, index_name=index_name)
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._client = client or httpx.AsyncClient(headers=headers, timeout=30.0)

    # ------------------------------------------------------------------
    # VectorStorePort
    # ------------------------------------------------------------------

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        collection: str,
        filters: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[Chunk]:
        # Merge caller filters first so the collection isolation constraint always wins.
        filter_payload: dict[str, Any] = {**(filters or {}), "collection": {"$eq": collection}}

        payload: dict[str, Any] = {
            "vector": query_vector,
            "topK": top_k,
            "filter": filter_payload,
            "returnMetadata": "all",
        }
        resp = await self._client.post(f"{self._base}/query", json=payload)
        resp.raise_for_status()
        body = resp.json()

        matches = body.get("result", {}).get("matches", [])
        chunks: list[Chunk] = []
        for m in matches:
            score = m.get("score", 0.0)
            if score < score_threshold:
                continue
            metadata = m.get("metadata") or {}
            chunks.append(
                Chunk(
                    id=str(m["id"]),
                    text=metadata.get("text", ""),
                    collection=collection,
                    score=score,
                    metadata={k: v for k, v in metadata.items() if k != "text"},
                )
            )
        return chunks

    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        vectors_payload = [
            {
                "id": chunk.id,
                "values": vector,
                "metadata": {"text": chunk.text, "collection": collection, **chunk.metadata},
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        resp = await self._client.post(
            f"{self._base}/upsert",
            json={"vectors": vectors_payload},
        )
        resp.raise_for_status()

    async def delete(self, collection: str, ids: list[str]) -> None:
        resp = await self._client.post(
            f"{self._base}/delete-by-ids",
            json={"ids": ids},
        )
        resp.raise_for_status()

    async def list_documents(
        self,
        collection: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        # Vectorize does not support full scroll; return empty list with a warning.
        # Use the search endpoint with a zero vector as a workaround only when needed.
        logger.warning(
            "list_documents is not efficiently supported on Vectorize; returning empty list"
        )
        return []

    # ------------------------------------------------------------------
    # CollectionPort -- no-ops on Vectorize (index provisioned externally)
    # ------------------------------------------------------------------

    async def create(self, name: str, vector_size: int) -> None:
        logger.info(
            "Vectorize: create(%s, %d) is a no-op -- provision indexes via Wrangler",
            name,
            vector_size,
        )

    async def list_collections(self) -> list[str]:
        # Query distinct collection metadata values -- not natively supported;
        # return a static list from config would require additional wiring.
        # Return empty for now; the app startup check tolerates this.
        return []

    async def delete_collection(self, name: str) -> None:
        logger.info(
            "Vectorize: delete_collection(%s) is a no-op -- delete indexes via Wrangler", name
        )

    async def get_stats(self, name: str) -> dict:
        resp = await self._client.get(f"{self._base}/info")
        resp.raise_for_status()
        info = resp.json().get("result", {})
        count = info.get("vectorCount", 0)
        return {
            "name": name,
            "vectors_count": count,
            "points_count": count,
        }
