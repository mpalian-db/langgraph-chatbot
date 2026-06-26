"""Unit tests for the Cloudflare Workers AI embedding adapter.

Uses httpx.MockTransport to intercept HTTP calls so the adapter's request
shape and response parsing can be verified without a real Cloudflare account.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import httpx
import pytest

from app.adapters.embeddings.workers_ai import WorkersAIEmbeddingAdapter


def _resp(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "http://workers-ai-test"),
    )


def _client(responses: list[httpx.Response], seen: list[httpx.Request]) -> httpx.AsyncClient:
    queue = deque(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.popleft()

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.cloudflare.com/client/v4/accounts/acc-1",
        headers={"Authorization": "Bearer token-1"},
    )


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


async def test_embed_posts_texts_to_run_endpoint_and_returns_vectors():
    seen: list[httpx.Request] = []
    body = {"result": {"data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}}
    client = _client([_resp(body)], seen=seen)
    adapter = WorkersAIEmbeddingAdapter(
        account_id="acc-1",
        api_token="token-1",
        model="@cf/baai/bge-small-en-v1.5",
        http_client=client,
    )

    vectors = await adapter.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    # URL should include the model identifier verbatim.
    req = seen[0]
    assert req.url.path.endswith("/ai/run/@cf/baai/bge-small-en-v1.5")
    payload = json.loads(req.content)
    assert payload == {"text": ["hello", "world"]}


async def test_embed_handles_empty_list():
    """An empty input list still hits the API and returns an empty result.

    This is a behavioural choice: the adapter does not short-circuit. If empty
    input becomes a hot path, short-circuiting in the adapter would save a
    round-trip but is not currently implemented."""
    seen: list[httpx.Request] = []
    client = _client([_resp({"result": {"data": []}})], seen=seen)
    adapter = WorkersAIEmbeddingAdapter("acc-1", "token-1", http_client=client)

    vectors = await adapter.embed([])

    assert vectors == []
    assert len(seen) == 1


async def test_embed_raises_on_http_error():
    err_resp = httpx.Response(
        status_code=429,
        content=b"rate limited",
        request=httpx.Request("POST", "http://workers-ai-test"),
    )
    client = _client([err_resp], seen=[])
    adapter = WorkersAIEmbeddingAdapter("acc-1", "token-1", http_client=client)

    with pytest.raises(httpx.HTTPStatusError):
        await adapter.embed(["x"])


async def test_embed_raises_on_unexpected_response_shape():
    """If the response is missing `result.data`, the current adapter raises
    KeyError. Test pins this contract: callers see a hard failure rather than
    a silent empty-list return."""
    client = _client([_resp({"errors": [{"message": "bad"}]})], seen=[])
    adapter = WorkersAIEmbeddingAdapter("acc-1", "token-1", http_client=client)

    with pytest.raises(KeyError):
        await adapter.embed(["x"])


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------


async def test_embed_does_not_close_caller_supplied_client():
    """When the caller injects an httpx.AsyncClient, the adapter must NOT close
    it -- ownership stays with the caller. This matters for long-lived FastAPI
    apps that share a single client across requests."""
    seen: list[httpx.Request] = []
    client = _client([_resp({"result": {"data": [[0.1]]}})], seen=seen)
    adapter = WorkersAIEmbeddingAdapter("acc-1", "token-1", http_client=client)

    await adapter.embed(["x"])

    # Client is still usable -- not closed by the adapter.
    assert not client.is_closed
    await client.aclose()


async def test_embed_closes_client_when_constructed_internally(monkeypatch: pytest.MonkeyPatch):
    """When no client is injected, the adapter constructs one per call and
    must close it via the `finally` block.

    We monkeypatch `httpx.AsyncClient` in the adapter module so the
    internally-constructed client is a TrackingClient backed by a MockTransport.
    This verifies the cleanup contract directly: aclose() must be called exactly
    once, AFTER the response is parsed."""
    closed: list[int] = []

    class TrackingClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            # Force a MockTransport so no real network call is attempted.
            kwargs["transport"] = httpx.MockTransport(
                lambda _req: _resp({"result": {"data": [[0.1, 0.2]]}})
            )
            super().__init__(*args, **kwargs)

        async def aclose(self) -> None:
            closed.append(1)
            await super().aclose()

    monkeypatch.setattr(
        "app.adapters.embeddings.workers_ai.httpx.AsyncClient",
        TrackingClient,
    )

    adapter = WorkersAIEmbeddingAdapter("acc-1", "token-1")
    vectors = await adapter.embed(["hello"])

    # Response was parsed correctly...
    assert vectors == [[0.1, 0.2]]
    # ...and the client was closed exactly once.
    assert closed == [1]
