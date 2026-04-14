from __future__ import annotations

import re
import time
from typing import Any

from app.core.config.models import AnswerGenerationConfig
from app.core.graph.state import GraphState
from app.core.models.types import Citation, TraceEntry
from app.ports.llm import LLMPort


async def run(
    state: GraphState,
    *,
    config: AnswerGenerationConfig,
    llm: LLMPort,
) -> dict[str, Any]:
    start = time.monotonic()

    evidence = "\n\n".join(
        f"[{chunk.id}] {chunk.text}" for chunk in state.retrieved_chunks
    )
    prompt = config.prompt_template.format(evidence=evidence, query=state.query)

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

    # If revising, include previous verifier feedback
    if state.verifier_result and state.verifier_result.outcome == "revise":
        messages.append({"role": "assistant", "content": state.draft_answer or ""})
        messages.append({
            "role": "user",
            "content": (
                f"Please revise. Issues: {state.verifier_result.reason}. "
                f"Unsupported claims: {', '.join(state.verifier_result.unsupported_claims)}"
            ),
        })

    response = await llm.complete(
        messages=messages,
        model=config.model,
        max_tokens=config.max_tokens,
    )

    draft = response["text"]
    citations = _extract_citations(draft, state.retrieved_chunks)

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "draft_answer": draft,
        "citations": citations,
        "execution_trace": state.execution_trace + [
            TraceEntry(
                node="answer_generation",
                duration_ms=elapsed_ms,
                data={"tokens": response["usage"]},
            )
        ],
    }


def _extract_citations(text: str, chunks: list) -> list[Citation]:
    """Extract citations for chunk IDs referenced in square brackets."""
    cited_ids = set(re.findall(r"\[([a-f0-9\-]{36})\]", text))
    return [
        Citation(chunk_id=c.id, text=c.text[:200], collection=c.collection)
        for c in chunks
        if c.id in cited_ids
    ]
