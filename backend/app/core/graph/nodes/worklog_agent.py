from __future__ import annotations

import json
import time
from typing import Any

from app.core.config.models import WorklogAgentConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry
from app.ports.llm import LLMPort
from app.ports.worklog import WorklogPort


async def run(
    state: GraphState,
    *,
    config: WorklogAgentConfig,
    llm: LLMPort,
    worklog: WorklogPort,
) -> dict[str, Any]:
    start = time.monotonic()

    context = await _fetch_worklog_context(state.query, worklog)

    messages = [
        {
            "role": "user",
            "content": (f"User question: {state.query}\n\nWorklog data:\n{context}"),
        },
    ]

    response = await llm.complete(
        messages=messages,
        model=config.model,
        system=config.system_prompt,
        max_tokens=config.max_tokens,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "final_answer": response["text"],
        "execution_trace": state.execution_trace
        + [
            TraceEntry(
                node="worklog_agent",
                duration_ms=elapsed_ms,
                data={"tokens": response["usage"]},
            )
        ],
    }


async def _fetch_worklog_context(query: str, worklog: WorklogPort) -> str:
    query_lower = query.lower()

    if "generate" in query_lower or "create" in query_lower or "new plan" in query_lower:
        plan = await worklog.generate_plan()
        return json.dumps(
            {
                "action": "generate_plan",
                "plan": {
                    "key": plan.key,
                    "created_at": plan.created_at,
                    "total_hours": plan.total_hours,
                    "entries": plan.entries,
                },
            },
            indent=2,
        )

    if "plan " in query_lower or "plan:" in query_lower:
        key = _extract_plan_key(query)
        if key:
            plan = await worklog.get_plan(key)
            return json.dumps(
                {
                    "action": "get_plan",
                    "plan": {
                        "key": plan.key,
                        "created_at": plan.created_at,
                        "total_hours": plan.total_hours,
                        "entries": plan.entries,
                    },
                },
                indent=2,
            )

    plans = await worklog.list_plans()
    return json.dumps(
        {
            "action": "list_plans",
            "plans": [
                {
                    "key": p.key,
                    "created_at": p.created_at,
                    "total_hours": p.total_hours,
                }
                for p in plans
            ],
        },
        indent=2,
    )


def _extract_plan_key(query: str) -> str | None:
    for word in query.split():
        if word.startswith("plan:"):
            return word.split(":", 1)[1]
        if "-" in word and any(c.isdigit() for c in word):
            return word
    return None
