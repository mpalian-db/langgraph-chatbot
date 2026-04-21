from __future__ import annotations

import json

import httpx
import pytest

from app.adapters.worklog.http import WorklogHTTPAdapter


def _mock_response(data: dict | list, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "http://test"),
    )


@pytest.fixture
def adapter():
    transport = httpx.MockTransport(lambda _: _mock_response([]))
    client = httpx.AsyncClient(transport=transport)
    return WorklogHTTPAdapter(base_url="http://test", api_key="key", client=client)


@pytest.mark.asyncio
async def test_list_plans_parses_array():
    plans_json = [
        {"key": "2026-W16", "created_at": "2026-04-14", "total_hours": 40.0},
        {"key": "2026-W17", "createdAt": "2026-04-21", "totalHours": 37.5},
    ]
    transport = httpx.MockTransport(lambda _: _mock_response(plans_json))
    client = httpx.AsyncClient(transport=transport)
    adapter = WorklogHTTPAdapter(base_url="http://test", api_key="key", client=client)

    result = await adapter.list_plans()

    assert len(result) == 2
    assert result[0].key == "2026-W16"
    assert result[0].total_hours == 40.0
    assert result[1].created_at == "2026-04-21"
    assert result[1].total_hours == 37.5


@pytest.mark.asyncio
async def test_list_plans_parses_wrapped_object():
    body = {"plans": [{"key": "2026-W16", "created_at": "2026-04-14", "total_hours": 40.0}]}
    transport = httpx.MockTransport(lambda _: _mock_response(body))
    client = httpx.AsyncClient(transport=transport)
    adapter = WorklogHTTPAdapter(base_url="http://test", api_key="key", client=client)

    result = await adapter.list_plans()

    assert len(result) == 1
    assert result[0].key == "2026-W16"


@pytest.mark.asyncio
async def test_get_plan_returns_entries():
    plan_json = {
        "key": "2026-W16",
        "created_at": "2026-04-14",
        "total_hours": 40.0,
        "entries": [{"issue": "PROJ-1", "hours": 8}],
    }
    transport = httpx.MockTransport(lambda _: _mock_response(plan_json))
    client = httpx.AsyncClient(transport=transport)
    adapter = WorklogHTTPAdapter(base_url="http://test", api_key="key", client=client)

    result = await adapter.get_plan("2026-W16")

    assert result.key == "2026-W16"
    assert len(result.entries) == 1
    assert result.entries[0]["issue"] == "PROJ-1"


@pytest.mark.asyncio
async def test_generate_plan_posts():
    plan_json = {
        "key": "2026-W18",
        "created_at": "2026-04-28",
        "total_hours": 40.0,
        "entries": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return _mock_response(plan_json)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    adapter = WorklogHTTPAdapter(base_url="http://test", api_key="key", client=client)

    result = await adapter.generate_plan()

    assert result.key == "2026-W18"


@pytest.mark.asyncio
async def test_adapter_raises_on_http_error():
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            status_code=500,
            content=b"Internal Server Error",
            request=httpx.Request("GET", "http://test"),
        )
    )
    client = httpx.AsyncClient(transport=transport)
    adapter = WorklogHTTPAdapter(base_url="http://test", api_key="key", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        await adapter.list_plans()
