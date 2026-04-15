from __future__ import annotations

import re
import time
from typing import Any

from app.core.config.models import VerifierConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry, VerifierResult
from app.ports.llm import LLMPort

# Matches inline citations of the form [chunk-id] or [abc-123].
_CITATION_RE = re.compile(r"\[\w[\w-]*\]")


async def run(
    state: GraphState,
    *,
    config: VerifierConfig,
    llm: LLMPort,
) -> dict[str, Any]:
    start = time.monotonic()

    # Check 0: no evidence retrieved at all -- refuse immediately.
    if not state.retrieved_chunks:
        result = VerifierResult(
            outcome="refuse",
            score=0.0,
            reason="No evidence retrieved from the knowledge base",
        )
        return _build_return(state, result, start, refuse=True)

    # Check 1: retrieval score threshold
    if "score_threshold" in config.checks and state.retrieval_scores:
        max_score = max(state.retrieval_scores)
        if max_score < config.score_threshold:
            result = VerifierResult(
                outcome="refuse",
                score=max_score,
                reason=(
                    f"Max retrieval score {max_score:.2f} below threshold {config.score_threshold}"
                ),
            )
            return _build_return(state, result, start, refuse=True)

    # Check 2: citation coverage (deterministic, no LLM call)
    if "citation_coverage" in config.checks and state.draft_answer:
        coverage = _citation_coverage(state.draft_answer)
        if coverage < config.citation_coverage_min:
            result = VerifierResult(
                outcome="revise",
                score=coverage,
                reason=(
                    f"Citation coverage {coverage:.0%} is below the required "
                    f"{config.citation_coverage_min:.0%}"
                ),
            )
            if state.retry_count < config.max_retries:
                return _build_return(state, result, start, revise=True)
            else:
                return _build_return(state, result, start, refuse=True)

    # Check 3: LLM-based support analysis
    if "support_analysis" in config.checks:
        evidence = "\n\n".join(f"[{c.id}] {c.text}" for c in state.retrieved_chunks)
        prompt = (
            "You are a grounding verifier. Determine if the answer "
            "is supported by the evidence.\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Answer to verify:\n{state.draft_answer}\n\n"
            "Respond in this EXACT format:\n"
            "OUTCOME: accept|revise|refuse\n"
            "SCORE: 0.0-1.0\n"
            "REASON: one sentence\n"
            "UNSUPPORTED: comma-separated list of unsupported claims, or NONE"
        )
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            model=config.model,
            max_tokens=256,
        )
        result = _parse_verifier_response(response["text"])
    else:
        result = VerifierResult(outcome="accept", score=1.0, reason="checks skipped")

    # Decide outcome
    if result.outcome == "accept":
        return _build_return(state, result, start, accept=True)
    elif result.outcome == "revise" and state.retry_count < config.max_retries:
        return _build_return(state, result, start, revise=True)
    else:
        return _build_return(state, result, start, refuse=True)


def _build_return(
    state: GraphState,
    result: VerifierResult,
    start: float,
    *,
    accept: bool = False,
    revise: bool = False,
    refuse: bool = False,
) -> dict[str, Any]:
    elapsed_ms = (time.monotonic() - start) * 1000
    trace = TraceEntry(
        node="verifier",
        duration_ms=elapsed_ms,
        data={"outcome": result.outcome, "score": result.score},
    )
    update: dict[str, Any] = {
        "verifier_result": result,
        "execution_trace": state.execution_trace + [trace],
    }
    if accept:
        update["final_answer"] = state.draft_answer
    elif revise:
        update["retry_count"] = state.retry_count + 1
    elif refuse:
        update["final_answer"] = f"I cannot provide a fully supported answer. {result.reason}"
    return update


def _citation_coverage(text: str) -> float:
    """Return the fraction of non-trivial sentences that contain at least one citation.

    A sentence is non-trivial when it contains five or more words.  If there are
    no non-trivial sentences the answer is considered fully covered (1.0).
    """
    sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
    non_trivial = [s for s in sentences if len(s.split()) >= 5]
    if not non_trivial:
        return 1.0
    cited = sum(1 for s in non_trivial if _CITATION_RE.search(s))
    return cited / len(non_trivial)


def _parse_verifier_response(text: str) -> VerifierResult:
    outcome_match = re.search(r"OUTCOME:\s*(accept|revise|refuse)", text, re.IGNORECASE)
    score_match = re.search(r"SCORE:\s*([\d.]+)", text)
    reason_match = re.search(r"REASON:\s*(.+)", text)
    unsupported_match = re.search(r"UNSUPPORTED:\s*(.+)", text)

    outcome = outcome_match.group(1).lower() if outcome_match else "refuse"
    score = float(score_match.group(1)) if score_match else 0.0
    reason = reason_match.group(1).strip() if reason_match else "Unable to verify"
    unsupported_raw = unsupported_match.group(1).strip() if unsupported_match else "NONE"
    unsupported = (
        [] if unsupported_raw.upper() == "NONE" else [c.strip() for c in unsupported_raw.split(",")]
    )

    return VerifierResult(
        outcome=outcome,  # type: ignore[arg-type]
        score=score,
        reason=reason,
        unsupported_claims=unsupported,
    )
