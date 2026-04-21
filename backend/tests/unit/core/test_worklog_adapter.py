from __future__ import annotations

import json

import httpx
import pytest

from app.adapters.worklog.http import WorklogHTTPAdapter


def _resp(data: dict | list, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "http://test"),
    )


def _err_resp(status: int) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=b"error",
        request=httpx.Request("GET", "http://test"),
    )


def _adapter(handler) -> WorklogHTTPAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return WorklogHTTPAdapter(base_url="http://test", api_key="key", client=client)


# ---------------------------------------------------------------------------
# list_plans
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_plans_parses_array():
    plans_json = [
        {"key": "2026-W16", "created_at": "2026-04-14", "total_hours": 40.0},
        {"key": "2026-W17", "createdAt": "2026-04-21", "totalHours": 37.5},
    ]
    adapter = _adapter(lambda _: _resp(plans_json))

    result = await adapter.list_plans()

    assert len(result) == 2
    assert result[0].key == "2026-W16"
    assert result[0].total_hours == 40.0
    assert result[1].created_at == "2026-04-21"
    assert result[1].total_hours == 37.5


@pytest.mark.asyncio
async def test_list_plans_parses_wrapped_object():
    body = {"plans": [{"key": "2026-W16", "created_at": "2026-04-14", "total_hours": 40.0}]}
    adapter = _adapter(lambda _: _resp(body))

    result = await adapter.list_plans()

    assert len(result) == 1
    assert result[0].key == "2026-W16"


@pytest.mark.asyncio
async def test_list_plans_handles_unexpected_shape_gracefully():
    # A scalar body should return empty list, not crash
    adapter = _adapter(lambda _: _resp("not a list or dict"))  # type: ignore[arg-type]
    result = await adapter.list_plans()
    assert result == []


@pytest.mark.asyncio
async def test_list_plans_skips_non_dict_items():
    # Mixed array -- only dicts should be parsed
    adapter = _adapter(lambda _: _resp([{"key": "2026-W16", "total_hours": 8}, "bad"]))
    result = await adapter.list_plans()
    assert len(result) == 1


@pytest.mark.asyncio
async def test_list_plans_bad_total_hours_defaults_to_zero():
    adapter = _adapter(
        lambda _: _resp([{"key": "2026-W16", "created_at": "", "total_hours": "oops"}])
    )
    result = await adapter.list_plans()
    assert result[0].total_hours == 0.0


@pytest.mark.asyncio
async def test_list_plans_raises_on_http_error():
    adapter = _adapter(lambda _: _err_resp(500))
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.list_plans()


# ---------------------------------------------------------------------------
# get_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_plan_returns_entries():
    plan_json = {
        "key": "2026-W16",
        "created_at": "2026-04-14",
        "total_hours": 40.0,
        "entries": [{"issue": "PROJ-1", "hours": 8}],
    }
    adapter = _adapter(lambda _: _resp(plan_json))

    result = await adapter.get_plan("2026-W16")

    assert result.key == "2026-W16"
    assert len(result.entries) == 1
    assert result.entries[0]["issue"] == "PROJ-1"


@pytest.mark.asyncio
async def test_get_plan_raises_on_http_error():
    adapter = _adapter(lambda _: _err_resp(404))
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.get_plan("2026-W16")


@pytest.mark.asyncio
async def test_get_plan_raises_on_non_dict_body():
    adapter = _adapter(lambda _: _resp([1, 2, 3]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await adapter.get_plan("2026-W16")


# ---------------------------------------------------------------------------
# generate_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_plan_posts():
    plan_json = {"key": "2026-W18", "created_at": "2026-04-28", "total_hours": 40.0, "entries": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return _resp(plan_json)

    adapter = _adapter(handler)
    result = await adapter.generate_plan()
    assert result.key == "2026-W18"


@pytest.mark.asyncio
async def test_generate_plan_raises_on_http_error():
    adapter = _adapter(lambda _: _err_resp(500))
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.generate_plan()


@pytest.mark.asyncio
async def test_generate_plan_raises_on_non_dict_body():
    adapter = _adapter(lambda _: _resp("string body"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await adapter.generate_plan()
