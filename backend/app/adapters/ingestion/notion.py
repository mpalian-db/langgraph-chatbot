"""Notion API adapter -- fetches pages from a database and renders block content to plain text."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.ports.notion import NotionPage

logger = logging.getLogger(__name__)

_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


class NotionAdapter:
    def __init__(self, token: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }
        self._client = client or httpx.AsyncClient(headers=self._headers, timeout=30.0)

    async def list_pages(self, database_id: str) -> list[NotionPage]:
        pages: list[NotionPage] = []
        payload: dict[str, Any] = {}

        while True:
            resp = await self._client.post(
                f"{_API_BASE}/databases/{database_id}/query",
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()

            for result in body.get("results", []):
                page_id = result["id"]
                title = _extract_title(result.get("properties", {}))
                url = result.get("url", "")
                pages.append(NotionPage(id=page_id, title=title, text="", url=url))

            if not body.get("has_more"):
                break
            payload["start_cursor"] = body["next_cursor"]

        return pages

    async def get_page_content(self, page_id: str) -> NotionPage:
        resp = await self._client.get(f"{_API_BASE}/pages/{page_id}")
        resp.raise_for_status()
        page_data = resp.json()

        title = _extract_title(page_data.get("properties", {}))
        url = page_data.get("url", "")
        blocks = await self._fetch_blocks_recursive(page_id)
        text = render_blocks(blocks)

        return NotionPage(id=page_id, title=title, text=text, url=url)

    async def _fetch_blocks_recursive(self, block_id: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        start_cursor: str | None = None

        while True:
            params: dict[str, str] = {}
            if start_cursor:
                params["start_cursor"] = start_cursor

            resp = await self._client.get(
                f"{_API_BASE}/blocks/{block_id}/children",
                params=params,
            )
            resp.raise_for_status()
            body = resp.json()

            for block in body.get("results", []):
                blocks.append(block)
                if block.get("has_children"):
                    children = await self._fetch_blocks_recursive(block["id"])
                    block["_children"] = children

            if not body.get("has_more"):
                break
            start_cursor = body["next_cursor"]

        return blocks


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------


def render_blocks(blocks: list[dict[str, Any]], depth: int = 0) -> str:
    """Flatten a Notion block tree into plain text suitable for chunking."""
    lines: list[str] = []

    for block in blocks:
        block_type = block.get("type", "")
        content = block.get(block_type, {})

        line = _render_single_block(block_type, content, depth)
        if line is not None:
            lines.append(line)

        children = block.get("_children", [])
        if children:
            lines.append(render_blocks(children, depth + 1))

    return "\n".join(lines)


def _render_single_block(
    block_type: str,
    content: dict[str, Any],
    depth: int,
) -> str | None:
    """Render one block to a text line. Returns None for unsupported types."""
    rich_text = content.get("rich_text", [])
    text = _rich_text_to_str(rich_text)

    if block_type in ("paragraph", "quote", "callout"):
        return text if text else None

    if block_type in ("heading_1", "heading_2", "heading_3"):
        level = int(block_type[-1])
        prefix = "#" * level
        return f"{prefix} {text}" if text else None

    if block_type == "bulleted_list_item":
        indent = "  " * depth
        return f"{indent}- {text}" if text else None

    if block_type == "numbered_list_item":
        indent = "  " * depth
        return f"{indent}1. {text}" if text else None

    if block_type == "to_do":
        checked = content.get("checked", False)
        marker = "[x]" if checked else "[ ]"
        return f"- {marker} {text}" if text else None

    if block_type == "code":
        lang = content.get("language", "")
        return f"```{lang}\n{text}\n```" if text else None

    if block_type == "divider":
        return "---"

    if block_type == "toggle":
        return text if text else None

    return text if text else None


def _rich_text_to_str(rich_text: list[dict[str, Any]]) -> str:
    return "".join(span.get("plain_text", "") for span in rich_text)


def _extract_title(properties: dict[str, Any]) -> str:
    for prop in properties.values():
        if prop.get("type") == "title":
            return _rich_text_to_str(prop.get("title", []))
    return ""
