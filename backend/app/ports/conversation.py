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


@runtime_checkable
class ConversationReaderPort(Protocol):
    async def load(self, conversation_id: str, limit: int = 20) -> list[Turn]:
        """Return the last `limit` turns for `conversation_id`, oldest first.

        An unknown conversation_id returns an empty list (not an error) -- a
        first-turn user implicitly creates the conversation by sending a query."""
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
