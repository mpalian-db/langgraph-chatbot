from app.core.models.types import Chunk, Citation, ToolCall, TraceEntry, VerifierResult


def test_chunk_defaults():
    chunk = Chunk(id="c1", text="hello", collection="docs")
    assert chunk.score == 0.0
    assert chunk.metadata == {}


def test_verifier_result_fields():
    result = VerifierResult(outcome="accept", score=0.9, reason="well grounded")
    assert result.unsupported_claims == []


def test_tool_call_result_defaults():
    call = ToolCall(name="search", arguments={"query": "test"})
    assert call.result is None
