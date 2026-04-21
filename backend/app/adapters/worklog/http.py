from __future__ import annotations

import logging
from typing import Any

import httpx

from app.ports.worklog import WorklogPlan, WorklogPlanSummary

logger = logging.getLogger(__name__)


class WorklogHTTPAdapter:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = client or httpx.AsyncClient(headers=headers, timeout=30.0)

    async def list_plans(self) -> list[WorklogPlanSummary]:
        resp = await self._client.get(f"{self._base_url}/api/plans")
        resp.raise_for_status()
        body = resp.json()
        plans = body if isinstance(body, list) else body.get("plans", [])
        return [_parse_plan_summary(p) for p in plans]

    async def get_plan(self, key: str) -> WorklogPlan:
        resp = await self._client.get(f"{self._base_url}/api/plans/{key}")
        resp.raise_for_status()
        return _parse_plan(resp.json())

    async def generate_plan(self) -> WorklogPlan:
        resp = await self._client.post(f"{self._base_url}/api/plans")
        resp.raise_for_status()
        return _parse_plan(resp.json())


def _parse_plan_summary(data: dict[str, Any]) -> WorklogPlanSummary:
    return WorklogPlanSummary(
        key=data.get("key", ""),
        created_at=data.get("created_at", data.get("createdAt", "")),
        total_hours=float(data.get("total_hours", data.get("totalHours", 0))),
    )


def _parse_plan(data: dict[str, Any]) -> WorklogPlan:
    return WorklogPlan(
        key=data.get("key", ""),
        created_at=data.get("created_at", data.get("createdAt", "")),
        total_hours=float(data.get("total_hours", data.get("totalHours", 0))),
        entries=data.get("entries", []),
    )
