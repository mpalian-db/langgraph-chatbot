"""Conversation introspection endpoints.

Read-only debug surface for the conversation-memory feature: list every
known conversation with lightweight metadata, or fetch a single
conversation's summary + verbatim turns.

This is operator/developer-facing -- there's no auth here. If the chat
endpoint ever ships beyond a personal local-first setup, these routes
should be gated behind admin auth or moved under /api/system.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.dependencies import ConversationReaderDep, ConversationWriterDep

router = APIRouter(prefix="/conversations", tags=["conversations"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ConversationOverviewOut(BaseModel):
    conversation_id: str
    turn_count: int
    has_summary: bool
    last_updated_at: float | None


class TurnOut(BaseModel):
    role: str
    content: str


class ConversationDetailOut(BaseModel):
    conversation_id: str
    summary: str | None
    turns: list[TurnOut]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ConversationOverviewOut])
async def list_conversations(
    reader: ConversationReaderDep,
) -> list[ConversationOverviewOut]:
    """Return every known conversation with lightweight metadata, most
    recently active first."""
    overviews = await reader.list_conversations()
    return [
        ConversationOverviewOut(
            conversation_id=o.conversation_id,
            turn_count=o.turn_count,
            has_summary=o.has_summary,
            last_updated_at=o.last_updated_at,
        )
        for o in overviews
    ]


@router.get("/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(
    conversation_id: str,
    reader: ConversationReaderDep,
) -> ConversationDetailOut:
    """Return the rolling summary (if any) and every verbatim turn since
    the summary boundary for `conversation_id`. 404 when the conversation
    has no turns and no summary."""
    summary, turns = await reader.load_summary_and_turns(conversation_id)
    if summary is None and not turns:
        raise HTTPException(status_code=404, detail=f"conversation {conversation_id!r} not found")
    return ConversationDetailOut(
        conversation_id=conversation_id,
        summary=summary,
        turns=[TurnOut(role=t.role, content=t.content) for t in turns],
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    writer: ConversationWriterDep,
) -> None:
    """Erase a conversation: turns and rolling summary, atomically.

    Idempotent: a missing conversation returns 204, matching the existing
    pattern for `DELETE /collections/{name}/documents/{id}`. The caller's
    desired end state ('absent') is already true."""
    await writer.delete_conversation(conversation_id)
