"""Cloudflare Vectorize adapter -- implements VectorStorePort and CollectionPort.

A Vectorize index does not have sub-collections; logical collection separation is
achieved by storing a `collection` field in each vector's metadata and filtering on it
at query time. The CollectionPort management operations (create/delete) are no-ops
because Vectorize index provisioning is done via Wrangler outside the application.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.models.types import Chunk

logger = logging.getLogger(__name__)

_VECTORIZE_BASE = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/vectorize/v2/indexes/{index_name}"
)


class VectorizeAdapter:
    """Cloudflare Vectorize implementation of VectorStorePort + CollectionPort.

    Vectorize has no native sub-collection concept. Logical collections are
    implemented via a `collection` metadata field on each vector. Because
    Vectorize cannot enumerate distinct metadata values, the set of known
    collections must be supplied at construction time via `known_collections`.
    """

    def __init__(
        self,
        account_id: str,
        api_token: str,
        index_name: str,
        *,
        known_collections: list[str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = _VECTORIZE_BASE.format(account_id=account_id, index_name=index_name)
        self._known_collections: list[str] = known_collections or []
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
        # Fail loudly on mismatched inputs rather than silently truncating
        # via zip() -- silent data loss is the worst kind in an ingest path.
        if len(chunks) != len(vectors):
            msg = (
                f"chunks and vectors must have equal length; "
                f"got {len(chunks)} chunks and {len(vectors)} vectors"
            )
            raise ValueError(msg)

        # The Vectorize upsert REST endpoint expects a multipart-form upload
        # of newline-delimited JSON, NOT a JSON request body. See:
        #   https://developers.cloudflare.com/vectorize/best-practices/insert-vectors
        # The form-field name used here is `vectors`, matching the documented
        # Python example for /insert. The /upsert REST reference page shows a
        # curl example using `body` as the field name; the API may accept
        # either, but this has not been validated against a live account. If
        # /upsert returns a 400 in production, try renaming the field to
        # `body` before debugging further.
        # Reserved keys (text, collection) MUST win over chunk metadata so a
        # rogue metadata blob cannot land vectors in a foreign collection.
        records = [
            {
                "id": chunk.id,
                "values": vector,
                "metadata": {**chunk.metadata, "text": chunk.text, "collection": collection},
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        ndjson_body = "\n".join(json.dumps(r) for r in records).encode()
        resp = await self._client.post(
            f"{self._base}/upsert",
            files={"vectors": ("vectors.ndjson", ndjson_body, "application/x-ndjson")},
        )
        resp.raise_for_status()

    async def get_chunk(self, collection: str, chunk_id: str) -> Chunk | None:
        # Cloudflare endpoint: POST /get_by_ids with {"ids": [...]}. Returns
        # an array of {id, metadata, values}. See:
        # https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/get_by_ids/
        # Note: Vectorize has no native sub-collections; we additionally
        # check that the returned vector's metadata.collection matches the
        # logical collection we were asked about, otherwise we return None
        # (the caller didn't specify a chunk in this collection).
        resp = await self._client.post(
            f"{self._base}/get_by_ids",
            json={"ids": [chunk_id]},
        )
        resp.raise_for_status()
        result = resp.json().get("result") or []
        if not result:
            return None
        v = result[0]
        metadata = v.get("metadata") or {}
        if metadata.get("collection") != collection:
            return None
        return Chunk(
            id=str(v["id"]),
            text=metadata.get("text", ""),
            collection=collection,
            metadata={k: val for k, val in metadata.items() if k not in ("text", "collection")},
        )

    async def delete(self, collection: str, ids: list[str]) -> None:
        # Vectorize has no native sub-collections, and its delete_by_ids
        # endpoint accepts no metadata filter -- so cross-collection isolation
        # must be enforced client-side. Fetch each id first and delete only
        # those whose metadata.collection matches the caller's logical
        # collection. Without this, a caller who learned an id from
        # collection A could delete it via the collection B path.
        # See:
        #   https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/delete_by_ids/
        if not ids:
            return
        get_resp = await self._client.post(
            f"{self._base}/get_by_ids",
            json={"ids": ids},
        )
        get_resp.raise_for_status()
        found = get_resp.json().get("result") or []
        matching = [
            str(v["id"]) for v in found if (v.get("metadata") or {}).get("collection") == collection
        ]
        if not matching:
            return
        del_resp = await self._client.post(
            f"{self._base}/delete_by_ids",
            json={"ids": matching},
        )
        del_resp.raise_for_status()

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
        # Vectorize cannot enumerate distinct metadata values natively.
        # Return the list supplied at construction time (from config).
        return list(self._known_collections)

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
