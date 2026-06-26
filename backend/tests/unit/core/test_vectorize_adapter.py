"""Unit tests for the Cloudflare Vectorize adapter.

Uses httpx.MockTransport to intercept HTTP calls so the adapter's request
shape and response parsing can be verified without a real Cloudflare account.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import httpx
import pytest

from app.adapters.vectorstore.vectorize import VectorizeAdapter
from app.core.models.types import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "http://vectorize-test"),
    )


def _adapter(
    responses: list[httpx.Response],
    seen: list[httpx.Request] | None = None,
    *,
    known_collections: list[str] | None = None,
) -> VectorizeAdapter:
    """Build a VectorizeAdapter backed by a queue of pre-loaded responses.

    `seen` (if provided) records each intercepted request for assertion."""
    queue = deque(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return queue.popleft()

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, headers={})
    return VectorizeAdapter(
        account_id="acc-1",
        api_token="token-1",
        index_name="idx-1",
        known_collections=known_collections,
        client=client,
    )


def _match(
    vid: str,
    score: float,
    *,
    text: str = "",
    extra_metadata: dict | None = None,
) -> dict:
    metadata = {"text": text, "collection": "docs"}
    if extra_metadata:
        metadata.update(extra_metadata)
    return {"id": vid, "score": score, "metadata": metadata}


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_posts_to_query_with_collection_filter():
    seen: list[httpx.Request] = []
    body = {"result": {"matches": [_match("v1", 0.9, text="hello")]}}
    adapter = _adapter([_resp(body)], seen=seen)

    chunks = await adapter.search(
        query_vector=[0.1, 0.2, 0.3],
        top_k=4,
        collection="docs",
    )

    assert len(chunks) == 1
    assert chunks[0].id == "v1"
    assert chunks[0].text == "hello"
    assert chunks[0].score == pytest.approx(0.9)
    assert chunks[0].collection == "docs"
    # The `text` key must be stripped from the propagated metadata dict.
    assert "text" not in chunks[0].metadata

    # Assert request shape: URL, top_k, vector, filter.
    assert len(seen) == 1
    req = seen[0]
    assert req.url.path.endswith("/query")
    payload = json.loads(req.content)
    assert payload["vector"] == [0.1, 0.2, 0.3]
    assert payload["topK"] == 4
    assert payload["filter"] == {"collection": {"$eq": "docs"}}
    assert payload["returnMetadata"] == "all"


async def test_search_merges_caller_filters_with_collection_winning():
    """If a caller passes filters (e.g. metadata constraint) the collection
    filter must still apply -- the latter overrides any caller-supplied
    `collection` key by virtue of being placed last in the dict."""
    seen: list[httpx.Request] = []
    adapter = _adapter([_resp({"result": {"matches": []}})], seen=seen)

    await adapter.search(
        query_vector=[0.1],
        top_k=5,
        collection="docs",
        filters={"source": {"$eq": "notion"}, "collection": {"$eq": "wrong"}},
    )

    payload = json.loads(seen[0].content)
    assert payload["filter"]["source"] == {"$eq": "notion"}
    # Collection isolation always wins.
    assert payload["filter"]["collection"] == {"$eq": "docs"}


async def test_search_applies_score_threshold_client_side():
    body = {
        "result": {
            "matches": [
                _match("low", 0.1, text="below threshold"),
                _match("high", 0.95, text="above threshold"),
            ]
        }
    }
    adapter = _adapter([_resp(body)])

    chunks = await adapter.search(
        query_vector=[0.1],
        top_k=10,
        collection="docs",
        score_threshold=0.5,
    )

    assert [c.id for c in chunks] == ["high"]


async def test_search_returns_empty_when_no_matches():
    adapter = _adapter([_resp({"result": {"matches": []}})])

    chunks = await adapter.search(query_vector=[0.1], top_k=5, collection="docs")

    assert chunks == []


async def test_search_raises_on_http_error():
    err_resp = httpx.Response(
        status_code=500,
        content=b"server exploded",
        request=httpx.Request("POST", "http://vectorize-test"),
    )
    adapter = _adapter([err_resp])

    with pytest.raises(httpx.HTTPStatusError):
        await adapter.search(query_vector=[0.1], top_k=5, collection="docs")


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


def _parse_ndjson(raw: bytes) -> list[dict]:
    """Parse a multipart-uploaded ndjson body. The body arrives wrapped in
    multipart/form-data; we extract the inner ndjson payload by stripping the
    multipart envelope, then split on newlines."""
    text = raw.decode()
    # Multipart bodies start with `--boundary\r\n` and have headers ending in
    # a blank line before the payload. Find the payload region.
    if "Content-Disposition" in text:
        # Skip past the headers (delimited by blank line) up to the trailing boundary.
        payload_start = text.index("\r\n\r\n") + 4
        payload_end = text.rindex("\r\n--")
        text = text[payload_start:payload_end]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


async def test_upsert_posts_ndjson_vectors_to_upsert_endpoint():
    """Cloudflare Vectorize upsert expects multipart-form upload of an ndjson
    file under the `vectors` field. See:
    https://developers.cloudflare.com/vectorize/best-practices/insert-vectors"""
    seen: list[httpx.Request] = []
    adapter = _adapter([_resp({"result": {"mutationId": "m1"}})], seen=seen)

    chunks = [
        Chunk(
            id="c1",
            text="hello world",
            collection="docs",
            metadata={"source": "notion", "page_id": "p1"},
        ),
        Chunk(
            id="c2",
            text="second chunk",
            collection="docs",
            metadata={"source": "notion", "page_id": "p2"},
        ),
    ]
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    await adapter.upsert("docs", chunks, vectors)

    req = seen[0]
    assert req.url.path.endswith("/upsert")
    # multipart boundary in Content-Type confirms file upload.
    assert req.headers["content-type"].startswith("multipart/form-data")

    records = _parse_ndjson(req.content)
    assert len(records) == 2

    assert records[0]["id"] == "c1"
    assert records[0]["values"] == [0.1, 0.2, 0.3]
    assert records[0]["metadata"]["text"] == "hello world"
    assert records[0]["metadata"]["collection"] == "docs"
    assert records[0]["metadata"]["source"] == "notion"
    assert records[0]["metadata"]["page_id"] == "p1"

    assert records[1]["id"] == "c2"
    assert records[1]["values"] == [0.4, 0.5, 0.6]


async def test_upsert_chunk_metadata_cannot_override_reserved_keys():
    """Tenant isolation: `collection` and `text` are reserved keys set from
    the route's explicit arguments. A chunk's own metadata must not be able
    to overwrite them, otherwise vectors could land in a different logical
    collection than the caller requested."""
    seen: list[httpx.Request] = []
    adapter = _adapter([_resp({"result": {}})], seen=seen)

    chunks = [
        Chunk(
            id="c1",
            text="actual chunk text",
            collection="docs",
            metadata={"collection": "ROGUE", "text": "spoofed", "other": "ok"},
        )
    ]

    await adapter.upsert("docs", chunks, [[0.1]])

    record = _parse_ndjson(seen[0].content)[0]
    assert record["metadata"]["collection"] == "docs"
    assert record["metadata"]["text"] == "actual chunk text"
    # Non-reserved keys still propagate.
    assert record["metadata"]["other"] == "ok"


async def test_upsert_raises_on_length_mismatch_without_calling_api():
    """If chunks and vectors are unequal length the adapter must fail loudly,
    not silently truncate via zip()."""
    seen: list[httpx.Request] = []
    adapter = _adapter([], seen=seen)

    chunks = [
        Chunk(id="c1", text="a", collection="docs"),
        Chunk(id="c2", text="b", collection="docs"),
    ]
    vectors = [[0.1]]  # Only one vector for two chunks.

    with pytest.raises(ValueError, match="length"):
        await adapter.upsert("docs", chunks, vectors)

    # No HTTP call should have been issued.
    assert seen == []


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_get_chunk_returns_chunk_when_metadata_collection_matches():
    """The Cloudflare Vectorize index is shared across logical collections;
    a chunk only counts as belonging to a collection when its metadata says so."""
    seen: list[httpx.Request] = []
    body = {
        "result": [
            {
                "id": "c1",
                "metadata": {"text": "hello world", "collection": "docs", "page_id": "p1"},
            }
        ]
    }
    adapter = _adapter([_resp(body)], seen=seen)

    chunk = await adapter.get_chunk("docs", "c1")

    assert chunk is not None
    assert chunk.id == "c1"
    assert chunk.text == "hello world"
    assert chunk.collection == "docs"
    assert chunk.metadata == {"page_id": "p1"}

    req = seen[0]
    assert req.url.path.endswith("/get_by_ids")
    payload = json.loads(req.content)
    assert payload == {"ids": ["c1"]}


async def test_get_chunk_returns_none_when_id_not_found():
    adapter = _adapter([_resp({"result": []})])

    result = await adapter.get_chunk("docs", "missing")

    assert result is None


async def test_get_chunk_returns_none_when_metadata_collection_mismatches():
    """Cross-collection isolation: a chunk in `notes` must not surface when
    queried under `docs`, even though Vectorize itself doesn't enforce
    sub-collections."""
    body = {
        "result": [
            {
                "id": "c1",
                "metadata": {"text": "private", "collection": "notes"},
            }
        ]
    }
    adapter = _adapter([_resp(body)])

    result = await adapter.get_chunk("docs", "c1")

    assert result is None


async def test_delete_fetches_and_deletes_only_matching_collection():
    """Cross-collection isolation: delete must check each id's metadata
    collection before issuing delete_by_ids. Two HTTP calls: first
    get_by_ids, then delete_by_ids with the filtered subset."""
    seen: list[httpx.Request] = []
    get_body = {
        "result": [
            {"id": "c1", "metadata": {"collection": "docs"}},
            {"id": "c2", "metadata": {"collection": "notes"}},  # different collection
        ]
    }
    adapter = _adapter([_resp(get_body), _resp({"result": {}})], seen=seen)

    await adapter.delete("docs", ["c1", "c2"])

    # Two requests in order: get_by_ids, then delete_by_ids.
    assert len(seen) == 2
    assert seen[0].url.path.endswith("/get_by_ids")
    assert seen[1].url.path.endswith("/delete_by_ids")
    # delete_by_ids only includes the id whose metadata.collection matched.
    delete_payload = json.loads(seen[1].content)
    assert delete_payload["ids"] == ["c1"]


async def test_delete_skips_delete_call_when_no_ids_match_collection():
    """If none of the ids belong to the requested collection, the adapter
    must NOT issue a delete_by_ids call at all -- a no-op delete is the
    correct end state."""
    seen: list[httpx.Request] = []
    get_body = {
        "result": [
            {"id": "c1", "metadata": {"collection": "notes"}},
        ]
    }
    adapter = _adapter([_resp(get_body)], seen=seen)

    await adapter.delete("docs", ["c1"])

    # Only the get_by_ids call -- no delete_by_ids.
    assert len(seen) == 1
    assert seen[0].url.path.endswith("/get_by_ids")


async def test_delete_short_circuits_on_empty_id_list():
    """An empty id list must not even attempt the get_by_ids call -- this
    avoids gratuitous network round-trips on no-op deletes."""
    seen: list[httpx.Request] = []
    adapter = _adapter([], seen=seen)

    await adapter.delete("docs", [])

    assert seen == []


# ---------------------------------------------------------------------------
# list_documents -- documented no-op
# ---------------------------------------------------------------------------


async def test_list_documents_returns_empty_without_calling_api():
    seen: list[httpx.Request] = []
    adapter = _adapter([], seen=seen)

    result = await adapter.list_documents("docs")

    assert result == []
    assert seen == []


# ---------------------------------------------------------------------------
# CollectionPort: no-ops + known_collections
# ---------------------------------------------------------------------------


async def test_list_collections_returns_known_set_from_constructor():
    adapter = _adapter([], known_collections=["docs", "notes"])

    cols = await adapter.list_collections()

    assert cols == ["docs", "notes"]


async def test_list_collections_returns_a_copy():
    """The internal list should not be mutable from the outside via the return value."""
    adapter = _adapter([], known_collections=["docs"])

    cols = await adapter.list_collections()
    cols.append("rogue")

    cols_again = await adapter.list_collections()
    assert cols_again == ["docs"]


async def test_create_and_delete_collection_are_noops():
    seen: list[httpx.Request] = []
    adapter = _adapter([], seen=seen)

    await adapter.create("docs", 768)
    await adapter.delete_collection("docs")

    assert seen == []  # Both must be no-ops on the wire.


async def test_get_stats_reads_vector_count_from_info_endpoint():
    seen: list[httpx.Request] = []
    body = {"result": {"vectorCount": 42, "dimensions": 768}}
    adapter = _adapter([_resp(body)], seen=seen)

    stats = await adapter.get_stats("docs")

    assert seen[0].url.path.endswith("/info")
    assert stats == {"name": "docs", "vectors_count": 42, "points_count": 42}


async def test_get_stats_defaults_to_zero_when_field_missing():
    adapter = _adapter([_resp({"result": {}})])

    stats = await adapter.get_stats("docs")

    assert stats["vectors_count"] == 0
