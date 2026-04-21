from __future__ import annotations

import json
import re
import time
from typing import Any

from app.core.config.models import WorklogAgentConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry
from app.ports.llm import LLMPort
from app.ports.worklog import WorklogPort

# ISO week key pattern: YYYY-Www (e.g. 2026-W16)
_PLAN_KEY_RE = re.compile(r"\b(\d{4}-W\d{1,2})\b")

# Phrases that signal the user explicitly wants a *new* plan generated,
# not a report or summary referencing an existing one.
_GENERATE_PHRASES = ("generate a new plan", "create a new plan", "new worklog plan")


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
    # Check for a specific plan key first -- if found, retrieve it regardless
    # of other keywords in the query (avoids false-positive generate triggers).
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

    query_lower = query.lower()

    if any(phrase in query_lower for phrase in _GENERATE_PHRASES):
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
    """Extract an ISO week plan key (YYYY-Www) from the query, or None."""
    match = _PLAN_KEY_RE.search(query)
    return match.group(1) if match else None
