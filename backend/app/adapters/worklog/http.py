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
        if isinstance(body, list):
            plans = body
        elif isinstance(body, dict):
            plans = body.get("plans", [])
        else:
            logger.warning("Unexpected list_plans response shape: %s", type(body))
            return []
        return [_parse_plan_summary(p) for p in plans if isinstance(p, dict)]

    async def get_plan(self, key: str) -> WorklogPlan:
        resp = await self._client.get(f"{self._base_url}/api/plans/{key}")
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise ValueError(f"Expected dict for get_plan, got {type(body)}")
        return _parse_plan(body)

    async def generate_plan(self) -> WorklogPlan:
        resp = await self._client.post(f"{self._base_url}/api/plans")
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise ValueError(f"Expected dict for generate_plan, got {type(body)}")
        return _parse_plan(body)


def _parse_plan_summary(data: dict[str, Any]) -> WorklogPlanSummary:
    return WorklogPlanSummary(
        key=str(data.get("key", "")),
        created_at=str(data.get("created_at", data.get("createdAt", ""))),
        total_hours=_to_float(data.get("total_hours", data.get("totalHours", 0))),
    )


def _parse_plan(data: dict[str, Any]) -> WorklogPlan:
    return WorklogPlan(
        key=str(data.get("key", "")),
        created_at=str(data.get("created_at", data.get("createdAt", ""))),
        total_hours=_to_float(data.get("total_hours", data.get("totalHours", 0))),
        entries=data.get("entries", []),
    )


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Could not parse total_hours value %r, defaulting to 0.0", value)
        return 0.0
