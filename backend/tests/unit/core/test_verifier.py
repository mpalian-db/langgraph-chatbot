from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from app.core.config.models import VerifierConfig
from app.core.graph.nodes.verifier import _parse_verifier_response, run
from app.core.graph.state import GraphState
from app.core.models.types import Chunk


@pytest.fixture
def rag_state():
    return GraphState(
        query="What is LangGraph?",
        retrieved_chunks=[
            Chunk(
                id="abc-123",
                text="LangGraph is a library for building stateful agents.",
                collection="docs",
                score=0.85,
            ),
        ],
        retrieval_scores=[0.85],
        draft_answer="LangGraph is a library [abc-123] for building stateful agents.",
    )


def test_parse_verifier_response_accept():
    text = "OUTCOME: accept\nSCORE: 0.9\nREASON: Well supported.\nUNSUPPORTED: NONE"
    result = _parse_verifier_response(text)
    assert result.outcome == "accept"
    assert result.score == pytest.approx(0.9)
    assert result.unsupported_claims == []


def test_parse_verifier_response_revise():
    text = "OUTCOME: revise\nSCORE: 0.5\nREASON: Some claims unsupported.\nUNSUPPORTED: claim A, claim B"
    result = _parse_verifier_response(text)
    assert result.outcome == "revise"
    assert result.unsupported_claims == ["claim A", "claim B"]


def test_parse_verifier_response_garbage_defaults_to_refuse():
    result = _parse_verifier_response("garbled output with no structure")
    assert result.outcome == "refuse"


@pytest.mark.asyncio
async def test_verifier_refuses_below_score_threshold():
    low_score_state = GraphState(
        query="What is X?",
        retrieved_chunks=[Chunk(id="c1", text="unrelated", collection="docs", score=0.3)],
        retrieval_scores=[0.3],
        draft_answer="Some answer.",
    )
    config = VerifierConfig(score_threshold=0.7, checks=["score_threshold"])
    mock_llm = AsyncMock()

    result = await run(low_score_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "refuse"
    assert "final_answer" in result
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_accepts_when_llm_says_accept(rag_state, mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "OUTCOME: accept\nSCORE: 0.9\nREASON: Well supported.\nUNSUPPORTED: NONE",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 30},
    })
    config = VerifierConfig(score_threshold=0.5, checks=["score_threshold", "support_analysis"])

    result = await run(rag_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "accept"
    assert result["final_answer"] == rag_state.draft_answer


@pytest.mark.asyncio
async def test_verifier_increments_retry_on_revise(rag_state, mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "OUTCOME: revise\nSCORE: 0.6\nREASON: Missing citation.\nUNSUPPORTED: LangGraph is stateful",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 30},
    })
    config = VerifierConfig(score_threshold=0.5, checks=["support_analysis"], max_retries=2)

    result = await run(rag_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "revise"
    assert result["retry_count"] == 1
    assert "final_answer" not in result


@pytest.mark.asyncio
async def test_verifier_refuses_when_retries_exhausted(rag_state, mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "OUTCOME: revise\nSCORE: 0.6\nREASON: Still unsupported.\nUNSUPPORTED: some claim",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 30},
    })
    rag_state.retry_count = 2  # already at max
    config = VerifierConfig(score_threshold=0.5, checks=["support_analysis"], max_retries=2)

    result = await run(rag_state, config=config, llm=mock_llm)

    assert "final_answer" in result
    assert "cannot" in result["final_answer"].lower()
