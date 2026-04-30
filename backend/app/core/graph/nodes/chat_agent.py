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

    # Feed prior turns chronologically so the LLM has conversational context.
    # The current query is appended last as the new user turn. When `history`
    # is empty the prompt collapses to single-turn behaviour.
    messages = [{"role": turn.role, "content": turn.content} for turn in state.history]
    messages.append({"role": "user", "content": state.query})

    # When a rolling summary exists (conversation has been long enough to
    # trigger compression), prepend it to the system prompt as recovered
    # context. This gives the model access to facts and decisions from
    # earlier turns that no longer survive in the verbatim history window.
    system_prompt = config.system_prompt
    if state.conversation_summary:
        system_prompt = (
            f"Summary of earlier conversation: {state.conversation_summary}\n\n"
            f"{config.system_prompt}"
        )

    response = await llm.complete(
        messages=messages,
        model=config.model,
        system=system_prompt,
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
