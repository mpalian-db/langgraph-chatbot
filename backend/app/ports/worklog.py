from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class WorklogPlanSummary:
    key: str
    created_at: str
    total_hours: float


@dataclass
class WorklogPlan:
    key: str
    created_at: str
    total_hours: float
    entries: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class WorklogPort(Protocol):
    async def list_plans(self) -> list[WorklogPlanSummary]: ...

    async def get_plan(self, key: str) -> WorklogPlan: ...

    async def generate_plan(self) -> WorklogPlan: ...
