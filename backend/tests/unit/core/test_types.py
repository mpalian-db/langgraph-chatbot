from app.core.models.types import Chunk, Citation, ToolCall, TraceEntry, VerifierResult
from app.core.graph.state import GraphState


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


def test_graph_state_defaults():
    state = GraphState(query="what is langgraph?")
    assert state.route is None
    assert state.retrieved_chunks == []
    assert state.retry_count == 0
    assert state.final_answer is None


def test_graph_state_with_route():
    state = GraphState(query="hello", route="chat")
    assert state.route == "chat"
