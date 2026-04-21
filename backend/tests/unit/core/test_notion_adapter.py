from __future__ import annotations

import json
from collections import deque
from typing import Any

import httpx
import pytest

from app.adapters.ingestion.notion import NotionAdapter


def _resp(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "http://notion-test"),
    )


def _err_resp(status: int) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=b"error",
        request=httpx.Request("GET", "http://notion-test"),
    )


def _adapter(responses: list) -> NotionAdapter:
    """Build a NotionAdapter backed by a queue of pre-loaded responses."""
    queue = deque(responses)

    def handler(_request: httpx.Request) -> httpx.Response:
        return queue.popleft()

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, headers={})
    return NotionAdapter(token="test-token", client=client)


def _page_result(page_id: str, title: str = "Test Page") -> dict:
    return {
        "id": page_id,
        "url": f"https://notion.so/{page_id}",
        "properties": {
            "Name": {
                "type": "title",
                "title": [{"plain_text": title}],
            }
        },
    }


def _block(block_id: str, block_type: str, text: str, has_children: bool = False) -> dict:
    return {
        "id": block_id,
        "type": block_type,
        block_type: {"rich_text": [{"plain_text": text}]},
        "has_children": has_children,
    }


# ---------------------------------------------------------------------------
# list_pages
# ---------------------------------------------------------------------------


async def test_list_pages_single_page():
    body = {
        "results": [_page_result("page-1", "Alpha"), _page_result("page-2", "Beta")],
        "has_more": False,
    }
    adapter = _adapter([_resp(body)])

    pages = await adapter.list_pages("db-123")

    assert len(pages) == 2
    assert pages[0].id == "page-1"
    assert pages[0].title == "Alpha"
    assert pages[1].title == "Beta"
    assert pages[0].text == ""  # list_pages does not fetch content


async def test_list_pages_pagination():
    page1 = {
        "results": [_page_result("page-1")],
        "has_more": True,
        "next_cursor": "cursor-abc",
    }
    page2 = {
        "results": [_page_result("page-2")],
        "has_more": False,
    }
    adapter = _adapter([_resp(page1), _resp(page2)])

    pages = await adapter.list_pages("db-123")

    assert len(pages) == 2
    assert pages[0].id == "page-1"
    assert pages[1].id == "page-2"


async def test_list_pages_empty():
    body = {"results": [], "has_more": False}
    adapter = _adapter([_resp(body)])

    pages = await adapter.list_pages("db-123")

    assert pages == []


async def test_list_pages_raises_on_http_error():
    adapter = _adapter([_err_resp(403)])

    with pytest.raises(httpx.HTTPStatusError):
        await adapter.list_pages("db-123")


# ---------------------------------------------------------------------------
# get_page_content
# ---------------------------------------------------------------------------


async def test_get_page_content_returns_rendered_text():
    page_resp = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "properties": {"Name": {"type": "title", "title": [{"plain_text": "My Page"}]}},
    }
    blocks_resp = {
        "results": [_block("b1", "paragraph", "Hello world")],
        "has_more": False,
    }
    # get_page_content makes two requests: GET /pages/{id}, GET /blocks/{id}/children
    adapter = _adapter([_resp(page_resp), _resp(blocks_resp)])

    page = await adapter.get_page_content("page-1")

    assert page.id == "page-1"
    assert page.title == "My Page"
    assert "Hello world" in page.text
    assert page.url == "https://notion.so/page-1"


async def test_get_page_content_raises_on_http_error():
    adapter = _adapter([_err_resp(404)])

    with pytest.raises(httpx.HTTPStatusError):
        await adapter.get_page_content("missing-page")


async def test_get_page_content_fetches_children_recursively():
    page_resp = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "properties": {"Name": {"type": "title", "title": [{"plain_text": "Parent"}]}},
    }
    # First blocks call: one block with has_children=True
    blocks_resp = {
        "results": [_block("b1", "toggle", "Parent toggle", has_children=True)],
        "has_more": False,
    }
    # Second blocks call: children of b1
    children_resp = {
        "results": [_block("b2", "paragraph", "Child content")],
        "has_more": False,
    }
    adapter = _adapter([_resp(page_resp), _resp(blocks_resp), _resp(children_resp)])

    page = await adapter.get_page_content("page-1")

    assert "Parent toggle" in page.text
    assert "Child content" in page.text


# ---------------------------------------------------------------------------
# _fetch_blocks_recursive
# ---------------------------------------------------------------------------


async def test_fetch_blocks_single_page():
    blocks_resp = {
        "results": [_block("b1", "paragraph", "Line one"), _block("b2", "paragraph", "Line two")],
        "has_more": False,
    }
    adapter = _adapter([_resp(blocks_resp)])

    blocks = await adapter._fetch_blocks_recursive("page-1")

    assert len(blocks) == 2
    assert blocks[0]["id"] == "b1"
    assert blocks[1]["id"] == "b2"


async def test_fetch_blocks_pagination():
    page1 = {
        "results": [_block("b1", "paragraph", "First")],
        "has_more": True,
        "next_cursor": "cursor-xyz",
    }
    page2 = {
        "results": [_block("b2", "paragraph", "Second")],
        "has_more": False,
    }
    adapter = _adapter([_resp(page1), _resp(page2)])

    blocks = await adapter._fetch_blocks_recursive("page-1")

    assert len(blocks) == 2
    assert blocks[0]["id"] == "b1"
    assert blocks[1]["id"] == "b2"


async def test_fetch_blocks_children_stored_in_block():
    parent_resp = {
        "results": [_block("b1", "toggle", "Toggle", has_children=True)],
        "has_more": False,
    }
    child_resp = {
        "results": [_block("b2", "paragraph", "Under toggle")],
        "has_more": False,
    }
    adapter = _adapter([_resp(parent_resp), _resp(child_resp)])

    blocks = await adapter._fetch_blocks_recursive("page-1")

    assert len(blocks) == 1
    assert "_children" in blocks[0]
    assert blocks[0]["_children"][0]["id"] == "b2"


async def test_fetch_blocks_raises_on_http_error():
    adapter = _adapter([_err_resp(500)])

    with pytest.raises(httpx.HTTPStatusError):
        await adapter._fetch_blocks_recursive("page-1")
