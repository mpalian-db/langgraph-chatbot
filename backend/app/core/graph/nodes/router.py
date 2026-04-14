from __future__ import annotations

import time
from typing import Any

from app.core.config.models import RouterConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry
from app.ports.llm import LLMPort


async def run(
    state: GraphState,
    *,
    config: RouterConfig,
    llm: LLMPort,
) -> dict[str, Any]:
    start = time.monotonic()

    response = await llm.complete(
        messages=[{"role": "user", "content": state.query}],
        model=config.model,
        system=config.prompt,
        max_tokens=16,
    )

    route = response["text"].strip().lower()
    if route not in config.routes:
        route = "chat"

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "route": route,
        "execution_trace": state.execution_trace
        + [TraceEntry(node="router", duration_ms=elapsed_ms, data={"route": route})],
    }
