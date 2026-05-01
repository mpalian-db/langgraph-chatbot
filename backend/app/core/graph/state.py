from __future__ import annotations

from dataclasses import dataclass, field

from app.core.models.types import Chunk, Citation, ToolCall, TraceEntry, VerifierResult
from app.ports.conversation import Turn


@dataclass
class GraphState:
    query: str
    conversation_id: str | None = None
    history: list[Turn] = field(default_factory=list)
    # Rolling summary of conversation turns older than `history`. When set,
    # nodes that consume conversation context (currently chat_agent) prepend
    # it as system-level prior context so the model has access to long-ago
    # facts that have been compressed out of the verbatim history window.
    conversation_summary: str | None = None
    route: str | None = None
    collection: str | None = None
    retrieval_query: str | None = None
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    metadata_filters: dict = field(default_factory=dict)
    draft_answer: str | None = None
    verifier_result: VerifierResult | None = None
    final_answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    execution_trace: list[TraceEntry] = field(default_factory=list)
    retry_count: int = 0
