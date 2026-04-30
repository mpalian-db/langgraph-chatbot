"""Ports for conversation memory.

Conversation history is split into a reader port and a writer port, in keeping
with the project rule "never aggregate read, write, and mutation behind one
interface". A single SQLite-backed adapter typically implements both, but
nodes that only read history (e.g. chat_agent) take the reader port; the
chat route handler that persists turns takes the writer port. This makes the
data flow legible at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Role = Literal["user", "assistant"]


@dataclass
class Turn:
    role: Role
    content: str


@dataclass
class StoredTurn:
    """A `Turn` with its storage-assigned id.

    Used by the conversation-memory service when it needs to mark a boundary
    for summarisation -- the id of the last summarised turn becomes the
    "everything before this is now in the summary" cutoff."""

    id: int
    role: Role
    content: str


@dataclass
class ConversationOverview:
    """Lightweight conversation metadata for the list endpoint.

    Excludes turn content so a "list all conversations" call stays cheap
    even when individual conversations have hundreds of turns."""

    conversation_id: str
    turn_count: int
    has_summary: bool
    last_updated_at: float | None  # epoch seconds, None when no turns yet


@runtime_checkable
class ConversationReaderPort(Protocol):
    async def load(self, conversation_id: str, limit: int = 20) -> list[Turn]:
        """Return the last `limit` turns for `conversation_id`, oldest first.

        An unknown conversation_id returns an empty list (not an error) -- a
        first-turn user implicitly creates the conversation by sending a query."""
        ...

    async def load_summary_and_turns(
        self, conversation_id: str
    ) -> tuple[str | None, list[StoredTurn]]:
        """Return the rolling summary (if any) and all turns since the
        summary's boundary, oldest-first with their storage ids.

        Used by the conversation-memory service to decide whether to trigger
        further summarisation. The first element is the summary text or None
        if no summary has been recorded yet; the second is every turn whose
        id is strictly greater than the summary's boundary id (or all turns
        if no summary exists)."""
        ...

    async def list_conversations(self) -> list[ConversationOverview]:
        """Return every known conversation as a lightweight overview row,
        most-recently-active first.

        Used by debug/inspection tooling. Implementations should return
        cheap metadata only -- no turn content -- so the result remains
        small even when conversations are long."""
        ...


@runtime_checkable
class ConversationWriterPort(Protocol):
    async def append(self, conversation_id: str, role: Role, content: str) -> None:
        """Append a single turn. Implementations must be append-only -- prior
        turns are never modified."""
        ...

    async def append_pair(
        self, conversation_id: str, user_content: str, assistant_content: str
    ) -> None:
        """Append a (user, assistant) turn pair atomically.

        A chat round is one logical operation: the user query and the
        assistant response are recorded together, or not at all. Two separate
        `append()` calls would leave the conversation half-written if the
        process were cancelled between them, producing a "user message with
        no reply" ghost row that would corrupt future history loads.

        Implementations must commit both rows in a single transaction."""
        ...

    async def upsert_summary(
        self,
        conversation_id: str,
        summary: str,
        summarised_through_turn_id: int,
    ) -> None:
        """Set the rolling summary for this conversation, replacing any prior
        summary. The boundary id marks the cutoff: a subsequent
        `load_summary_and_turns` will return only turns with id strictly
        greater than `summarised_through_turn_id`."""
        ...
