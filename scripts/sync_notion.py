"""Bulk sync Notion pages into the RAG vector store.

Calls the chatbot's /api/collections/{collection}/sync-notion endpoint,
which fetches pages from the configured Notion database, chunks them,
and upserts into Qdrant.

Usage
-----
  # From the project root (backend must be running):
  uv run python scripts/sync_notion.py

Environment variables
---------------------
CHATBOT_URL         Base URL of this chatbot backend (default: http://localhost:8000)
NOTION_COLLECTION   Target Qdrant collection (default: notion-docs)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHATBOT_URL: str = os.getenv("CHATBOT_URL", "http://localhost:8000")
NOTION_COLLECTION: str = os.getenv("NOTION_COLLECTION", "notion-docs")


async def main() -> None:
    url = f"{CHATBOT_URL}/api/collections/{NOTION_COLLECTION}/sync-notion"
    logger.info("Starting Notion sync: POST %s", url)

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, timeout=300.0)

    if resp.status_code != 200:
        logger.error("Sync failed (HTTP %d): %s", resp.status_code, resp.text)
        sys.exit(1)

    data = resp.json()
    logger.info(
        "Sync complete: %d pages, %d chunks into collection %s",
        data["pages_synced"],
        data["total_chunks"],
        data["collection"],
    )


if __name__ == "__main__":
    asyncio.run(main())
