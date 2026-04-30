"""Conversation-memory orchestration: combines reader, writer, and an LLM
summariser into a single read-time operation that returns the right
context for the chat node.

The "lazy on load" trigger lives here: when the post-summary tail exceeds
`config.summarise_threshold`, fold everything except the last
`config.keep_recent` turns into a new summary and persist it before
returning the trimmed view to the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config.models import SummariserConfig
from app.ports.conversation import ConversationReaderPort, ConversationWriterPort, Turn
from app.ports.llm import LLMPort

logger = logging.getLogger(__name__)


@dataclass
class MemoryView:
    """What the chat route hands to the graph: a synthesised summary string
    (or None) plus the verbatim recent turns."""

    summary: str | None
    recent: list[Turn]


async def load_with_summary(
    conversation_id: str,
    *,
    reader: ConversationReaderPort,
    writer: ConversationWriterPort,
    llm: LLMPort,
    config: SummariserConfig,
) -> MemoryView:
    """Load the chat-ready memory for a conversation.

    Reads the rolling summary plus all turns since its boundary, decides
    whether to compress further, persists any new summary, and returns the
    summary + recent-window pair for the caller to feed into the graph."""
    summary, all_recent = await reader.load_summary_and_turns(conversation_id)

    # Convert StoredTurn -> Turn at the boundary (callers don't need ids).
    def _strip(turns):
        return [Turn(role=t.role, content=t.content) for t in turns]

    if not config.enabled or len(all_recent) <= config.summarise_threshold:
        # Below threshold: hand back what we have, no LLM call.
        return MemoryView(summary=summary, recent=_strip(all_recent))

    # Above threshold: summarise everything except the last `keep_recent`.
    # Defensive guard: even though SummariserConfig's validator enforces
    # summarise_threshold >= keep_recent, a programmatically-constructed
    # config could bypass it. An empty to_summarise slice would have no
    # boundary id to mark, so skip the round and return the current view.
    to_summarise = all_recent[: -config.keep_recent]
    if not to_summarise:
        logger.warning(
            "Summariser triggered with no older turns to fold "
            "(keep_recent=%d, threshold=%d, len=%d) -- skipping",
            config.keep_recent,
            config.summarise_threshold,
            len(all_recent),
        )
        return MemoryView(summary=summary, recent=_strip(all_recent))

    keep = all_recent[-config.keep_recent :]
    # Best-effort summarisation: if the LLM call or the storage write fails,
    # fall back to returning the unsummarised history rather than 500'ing
    # the chat request. Summarisation is an enhancement; chat must still work
    # without it. The next chat request will retry the same trigger.
    try:
        new_summary_text = await _summarise(
            prior_summary=summary,
            turns=_strip(to_summarise),
            llm=llm,
            config=config,
        )
        boundary_id = to_summarise[-1].id
        await writer.upsert_summary(conversation_id, new_summary_text, boundary_id)
    except Exception:
        logger.warning(
            "Summarisation failed for conversation %s; falling back to "
            "unsummarised history (the next request will retry).",
            conversation_id,
            exc_info=True,
        )
        return MemoryView(summary=summary, recent=_strip(all_recent))

    return MemoryView(summary=new_summary_text, recent=_strip(keep))


async def _summarise(
    *,
    prior_summary: str | None,
    turns: list[Turn],
    llm: LLMPort,
    config: SummariserConfig,
) -> str:
    """Build a prompt that integrates the prior summary (if any) with the
    new turns and ask the LLM to produce an updated rolling summary.

    The prompt explicitly tells the model to integrate rather than restart,
    so distant context isn't lost across multiple summarisation cycles."""
    transcript = "\n".join(f"{t.role}: {t.content}" for t in turns)
    if prior_summary:
        user_content = f"Prior summary:\n{prior_summary}\n\nNew turns to integrate:\n{transcript}"
    else:
        user_content = f"Conversation to summarise:\n{transcript}"

    response = await llm.complete(
        messages=[{"role": "user", "content": user_content}],
        model=config.model,
        system=config.prompt,
        max_tokens=config.max_tokens,
    )
    return response["text"].strip()
