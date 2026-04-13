from __future__ import annotations

from dataclasses import dataclass, field

from app.core.models.types import Chunk, Citation, ToolCall, TraceEntry, VerifierResult


@dataclass
class GraphState:
    query: str
    route: str | None = None
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
