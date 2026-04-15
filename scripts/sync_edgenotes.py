"""Bulk sync all existing EdgeNotes into the RAG vector store.

This script fetches notes from an EdgeNotes API (or a local JSON export) and
pushes each one through the chatbot's /api/webhooks/notes endpoint so that
Qdrant is populated without waiting for individual webhook deliveries.

Usage
-----
  # From the project root:
  uv run python scripts/sync_edgenotes.py

Environment variables
---------------------
EDGENOTES_API_URL   Base URL of the EdgeNotes API, e.g. https://notes.example.com
EDGENOTES_API_KEY   Bearer token for the EdgeNotes API (if required)
CHATBOT_URL         Base URL of this chatbot backend (default: http://localhost:8000)
WEBHOOK_SECRET      Matches config/config.toml webhooks.edgenotes_secret (optional)
NOTES_JSON          Path to a local JSON export to use instead of calling the API

Local JSON export format
------------------------
A JSON array of note objects, each matching NoteWebhookPayload minus the
'event' and 'timestamp' fields:

  [
    {
      "note_id": "abc-123",
      "title": "My note",
      "content": "Full markdown content...",
      "tags": ["tag1", "tag2"]
    },
    ...
  ]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

EDGENOTES_API_URL: str = os.getenv("EDGENOTES_API_URL", "")
EDGENOTES_API_KEY: str = os.getenv("EDGENOTES_API_KEY", "")
CHATBOT_URL: str = os.getenv("CHATBOT_URL", "http://localhost:8000")
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")
NOTES_JSON: str = os.getenv("NOTES_JSON", "")

BATCH_DELAY_S: float = 0.1      # pause between webhook calls to avoid overloading Qdrant
MAX_RETRIES: int = 3
RETRY_DELAY_S: float = 2.0
NOTES_PER_PAGE: int = 100       # page size when fetching from the EdgeNotes API


# ---------------------------------------------------------------------------
# Fetching notes
# ---------------------------------------------------------------------------


def _load_from_json(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path!r}, got {type(data).__name__}")
    return data


async def _fetch_from_api(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Paginate through the EdgeNotes /notes endpoint and return all notes."""
    headers: dict[str, str] = {}
    if EDGENOTES_API_KEY:
        headers["Authorization"] = f"Bearer {EDGENOTES_API_KEY}"

    notes: list[dict[str, Any]] = []
    page = 1

    while True:
        resp = await client.get(
            f"{EDGENOTES_API_URL}/notes",
            params={"page": page, "per_page": NOTES_PER_PAGE},
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        body = resp.json()

        # Support both {"notes": [...]} envelope and plain array responses.
        if isinstance(body, list):
            batch = body
        elif isinstance(body, dict) and "notes" in body:
            batch = body["notes"]
        else:
            raise ValueError(f"Unexpected response shape from EdgeNotes API: {body!r}")

        notes.extend(batch)
        logger.info("Fetched page %d -- %d notes so far", page, len(notes))

        if len(batch) < NOTES_PER_PAGE:
            break
        page += 1

    return notes


# ---------------------------------------------------------------------------
# Pushing to the chatbot webhook
# ---------------------------------------------------------------------------


async def _push_note(
    client: httpx.AsyncClient,
    note: dict[str, Any],
    index: int,
    total: int,
) -> bool:
    """Send a single note to the webhook endpoint.  Returns True on success."""
    payload: dict[str, Any] = {
        "event": "note.created",
        "note_id": note.get("note_id") or note.get("id", f"note-{index}"),
        "title": note.get("title", ""),
        "content": note.get("content", ""),
        "tags": note.get("tags", []),
        "timestamp": note.get("timestamp") or int(time.time()),
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if WEBHOOK_SECRET:
        headers["X-Webhook-Secret"] = WEBHOOK_SECRET

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.post(
                f"{CHATBOT_URL}/api/webhooks/notes",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            if resp.status_code in (200, 204):
                logger.info(
                    "[%d/%d] Synced note %s (%s)",
                    index,
                    total,
                    payload["note_id"],
                    payload["title"][:50] or "(no title)",
                )
                return True
            logger.warning(
                "[%d/%d] Webhook returned %d for note %s (attempt %d/%d)",
                index,
                total,
                resp.status_code,
                payload["note_id"],
                attempt,
                MAX_RETRIES,
            )
        except httpx.RequestError as exc:
            logger.warning(
                "[%d/%d] Request error for note %s (attempt %d/%d): %s",
                index,
                total,
                payload["note_id"],
                attempt,
                MAX_RETRIES,
                exc,
            )

        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY_S)

    logger.error("[%d/%d] Failed to sync note %s after %d attempts", index, total, payload["note_id"], MAX_RETRIES)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    if NOTES_JSON:
        logger.info("Loading notes from local file: %s", NOTES_JSON)
        notes = _load_from_json(NOTES_JSON)
    elif EDGENOTES_API_URL:
        logger.info("Fetching notes from EdgeNotes API: %s", EDGENOTES_API_URL)
        async with httpx.AsyncClient() as client:
            notes = await _fetch_from_api(client)
    else:
        logger.error(
            "Set either EDGENOTES_API_URL or NOTES_JSON to specify the note source."
        )
        sys.exit(1)

    total = len(notes)
    logger.info("Found %d notes to sync", total)

    if total == 0:
        logger.info("Nothing to do.")
        return

    succeeded = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for i, note in enumerate(notes, start=1):
            ok = await _push_note(client, note, i, total)
            if ok:
                succeeded += 1
            else:
                failed += 1
            if BATCH_DELAY_S > 0 and i < total:
                await asyncio.sleep(BATCH_DELAY_S)

    logger.info(
        "Sync complete: %d succeeded, %d failed (total %d)",
        succeeded,
        failed,
        total,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
