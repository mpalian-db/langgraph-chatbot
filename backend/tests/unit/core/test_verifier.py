from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.config.models import VerifierConfig
from app.core.graph.nodes.verifier import _citation_coverage, _parse_verifier_response, run
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
    text = (
        "OUTCOME: revise\nSCORE: 0.5\nREASON: Some claims unsupported."
        "\nUNSUPPORTED: claim A, claim B"
    )
    result = _parse_verifier_response(text)
    assert result.outcome == "revise"
    assert result.unsupported_claims == ["claim A", "claim B"]


def test_parse_verifier_response_garbage_defaults_to_refuse():
    result = _parse_verifier_response("garbled output with no structure")
    assert result.outcome == "refuse"


# ---------------------------------------------------------------------------
# _citation_coverage unit tests
# ---------------------------------------------------------------------------


def test_citation_coverage_all_cited():
    text = (
        "LangGraph is a stateful agent framework [chunk-1]."
        " It supports conditional edges [chunk-2]."
    )
    assert _citation_coverage(text, valid_ids=["chunk-1", "chunk-2"]) == pytest.approx(1.0)


def test_citation_coverage_none_cited():
    text = "LangGraph is a stateful agent framework. It supports conditional edges and loops."
    assert _citation_coverage(text, valid_ids=["chunk-1"]) == pytest.approx(0.0)


def test_citation_coverage_partial():
    # Three non-trivial sentences, one cited -> 1/3
    text = (
        "LangGraph is a stateful agent library [abc-1]. "
        "It was created to solve complex workflows. "
        "It supports both sync and async execution."
    )
    assert _citation_coverage(text, valid_ids=["abc-1"]) == pytest.approx(1 / 3)


def test_citation_coverage_empty_or_trivial_sentences_returns_one():
    # Short fragments under 5 words should not count as non-trivial.
    assert _citation_coverage("", valid_ids=[]) == pytest.approx(1.0)
    assert _citation_coverage("OK. Yes. No.", valid_ids=[]) == pytest.approx(1.0)


def test_citation_coverage_ignores_abbreviations_and_decimals():
    # "e.g." and "0.85" should not be treated as sentence boundaries.
    text = "See e.g. LangGraph with score 0.85 for building agents [chunk-1]."
    assert _citation_coverage(text, valid_ids=["chunk-1"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# citation_coverage rejects hallucinated and structural false-positive IDs
# ---------------------------------------------------------------------------


def test_citation_coverage_rejects_hallucinated_ids():
    """Coverage must validate bracketed tokens against the real chunk IDs.
    A bracketed string that looks like a citation but does not match any
    retrieved chunk is not a real citation."""
    text = "LangGraph is a stateful framework [hallucinated-id-not-in-chunks]."
    assert _citation_coverage(text, valid_ids=["chunk-1"]) == pytest.approx(0.0)


def test_citation_coverage_ignores_markdown_links():
    """`[text](url)` is a markdown hyperlink, not a citation. The bracketed
    token may even equal a real chunk id by coincidence in `text`, but the
    presence of `(` immediately after the closing bracket disambiguates."""
    text = "See [click here](https://example.com) for details on LangGraph."
    # No real citations, even though "click here" is a bracketed token.
    assert _citation_coverage(text, valid_ids=["chunk-1"]) == pytest.approx(0.0)


def test_citation_coverage_handles_nested_brackets():
    """Nested brackets like `[[chunk-1]]` (wiki-link style) should still
    extract the real id `chunk-1` and count as a citation."""
    text = "LangGraph is a stateful agent framework [[chunk-1]]."
    assert _citation_coverage(text, valid_ids=["chunk-1"]) == pytest.approx(1.0)


def test_citation_coverage_ignores_inline_code_spans():
    """A bracketed id wrapped in backticks is part of code/syntax discussion,
    not a real citation. `[chunk-1]` does not ground a claim."""
    text = "To cite a chunk, write `[chunk-1]` like that in your answer."
    assert _citation_coverage(text, valid_ids=["chunk-1"]) == pytest.approx(0.0)


def test_citation_coverage_ignores_fenced_code_blocks():
    """Bracketed ids inside fenced code blocks are example syntax, not
    citations of the surrounding answer."""
    text = (
        "LangGraph supports stateful agents and conditional edges. "
        "Here is example syntax:\n```\nresult = invoke([chunk-1])\n```"
    )
    # The first sentence has no citation. The fenced block must not count.
    assert _citation_coverage(text, valid_ids=["chunk-1"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# citation_coverage check integration tests (via run())
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_revises_when_citation_coverage_low(mock_llm):
    # draft_answer has two non-trivial sentences, neither cited -> coverage = 0.0
    state = GraphState(
        query="What is LangGraph?",
        retrieved_chunks=[
            Chunk(id="c1", text="LangGraph is a library.", collection="docs", score=0.9)
        ],
        retrieval_scores=[0.9],
        draft_answer=(
            "LangGraph is a library for building stateful agents. "
            "It supports conditional edges and loops."
        ),
    )
    config = VerifierConfig(
        score_threshold=0.5,
        citation_coverage_min=0.8,
        checks=["citation_coverage"],
        max_retries=2,
    )

    result = await run(state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "revise"
    assert result["retry_count"] == 1
    assert "final_answer" not in result
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_refuses_when_coverage_low_and_retries_exhausted(mock_llm):
    state = GraphState(
        query="What is LangGraph?",
        retrieved_chunks=[
            Chunk(id="c1", text="LangGraph is a library.", collection="docs", score=0.9)
        ],
        retrieval_scores=[0.9],
        draft_answer=(
            "LangGraph is a library for building stateful agents. "
            "It supports conditional edges and loops."
        ),
        retry_count=2,
    )
    config = VerifierConfig(
        score_threshold=0.5,
        citation_coverage_min=0.8,
        checks=["citation_coverage"],
        max_retries=2,
    )

    result = await run(state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "revise"
    assert "final_answer" in result
    assert "cannot" in result["final_answer"].lower()
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_skips_coverage_check_when_not_in_checks(rag_state, mock_llm):
    # rag_state.draft_answer has one cited sentence -- coverage = 1.0 anyway,
    # but we confirm the check is skipped entirely by omitting it from checks.
    mock_llm.complete = AsyncMock(
        return_value={
            "text": "OUTCOME: accept\nSCORE: 0.9\nREASON: Well supported.\nUNSUPPORTED: NONE",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
    )
    config = VerifierConfig(
        score_threshold=0.5,
        checks=["support_analysis"],  # citation_coverage deliberately absent
    )

    result = await run(rag_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "accept"
    mock_llm.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------


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
    mock_llm.complete = AsyncMock(
        return_value={
            "text": "OUTCOME: accept\nSCORE: 0.9\nREASON: Well supported.\nUNSUPPORTED: NONE",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
    )
    config = VerifierConfig(score_threshold=0.5, checks=["score_threshold", "support_analysis"])

    result = await run(rag_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "accept"
    assert result["final_answer"] == rag_state.draft_answer


@pytest.mark.asyncio
async def test_verifier_increments_retry_on_revise(rag_state, mock_llm):
    mock_llm.complete = AsyncMock(
        return_value={
            "text": (
                "OUTCOME: revise\nSCORE: 0.6\nREASON: Missing citation."
                "\nUNSUPPORTED: LangGraph is stateful"
            ),
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
    )
    config = VerifierConfig(score_threshold=0.5, checks=["support_analysis"], max_retries=2)

    result = await run(rag_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "revise"
    assert result["retry_count"] == 1
    assert "final_answer" not in result


@pytest.mark.asyncio
async def test_verifier_refuses_when_retries_exhausted(rag_state, mock_llm):
    mock_llm.complete = AsyncMock(
        return_value={
            "text": (
                "OUTCOME: revise\nSCORE: 0.6\nREASON: Still unsupported.\nUNSUPPORTED: some claim"
            ),
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
    )
    rag_state.retry_count = 2  # already at max
    config = VerifierConfig(score_threshold=0.5, checks=["support_analysis"], max_retries=2)

    result = await run(rag_state, config=config, llm=mock_llm)

    assert "final_answer" in result
    assert "cannot" in result["final_answer"].lower()


@pytest.mark.asyncio
async def test_verifier_refuses_when_no_chunks_retrieved():
    empty_state = GraphState(
        query="What is X?",
        retrieved_chunks=[],
        retrieval_scores=[],
        draft_answer=None,
    )
    config = VerifierConfig(score_threshold=0.5, checks=["score_threshold", "support_analysis"])
    mock_llm = AsyncMock()

    result = await run(empty_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "refuse"
    assert "final_answer" in result
    mock_llm.complete.assert_not_called()
