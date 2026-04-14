from __future__ import annotations

import time
from typing import Any

from app.core.config.models import ChatAgentConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry
from app.ports.llm import LLMPort


async def run(
    state: GraphState,
    *,
    config: ChatAgentConfig,
    llm: LLMPort,
) -> dict[str, Any]:
    start = time.monotonic()

    response = await llm.complete(
        messages=[{"role": "user", "content": state.query}],
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
                node="chat_agent", duration_ms=elapsed_ms, data={"tokens": response["usage"]}
            )
        ],
    }
