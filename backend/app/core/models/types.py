from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VerifierOutcome = Literal["accept", "revise", "refuse"]


@dataclass
class Chunk:
    id: str
    text: str
    collection: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Citation:
    chunk_id: str
    text: str
    collection: str


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: str | None = None


@dataclass
class TraceEntry:
    node: str
    duration_ms: float
    data: dict = field(default_factory=dict)


@dataclass
class VerifierResult:
    outcome: VerifierOutcome
    score: float
    reason: str
    unsupported_claims: list[str] = field(default_factory=list)
