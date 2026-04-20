# LangGraph Low-Level Concepts

## StateGraph

`StateGraph` is the core primitive in LangGraph. It compiles a directed graph of nodes and edges into a runnable that manages state across invocations.

```python
from langgraph.graph import StateGraph

builder = StateGraph(MyState)
builder.add_node("router", router_fn)
builder.add_node("retrieval", retrieval_fn)
builder.add_edge("router", "retrieval")
graph = builder.compile()
```

The graph is immutable after compilation. All state transitions happen through node return values.

## State

State is a typed dataclass or TypedDict that is passed between nodes. Each node receives the full state and returns a partial update. LangGraph merges the returned dict into the running state.

```python
from dataclasses import dataclass, field

@dataclass
class GraphState:
    query: str
    route: str | None = None
    retrieved_chunks: list = field(default_factory=list)
    final_answer: str | None = None
```

State fields must be serialisable. Avoid storing non-serialisable objects (e.g. open file handles) in state.

## Nodes

A node is any async (or sync) callable that accepts state and returns a dict of updates.

```python
async def my_node(state: GraphState) -> dict:
    result = await some_async_operation(state.query)
    return {"final_answer": result}
```

Nodes must return a plain dict. LangGraph uses the keys to determine which fields to update.

## Edges

Edges connect nodes. There are two kinds:

- **Static edges** (`add_edge`): always route from source to target.
- **Conditional edges** (`add_conditional_edges`): call a function that returns the name of the next node.

```python
def route_decision(state: GraphState) -> str:
    if state.route == "rag":
        return "retrieval"
    return "chat_agent"

builder.add_conditional_edges("router", route_decision)
```

## Entry and Finish Points

Every graph needs a defined starting node and one or more terminal nodes.

```python
from langgraph.graph import END

builder.set_entry_point("router")
builder.add_edge("chat_agent", END)
builder.add_edge("verifier", END)
```

`END` is a sentinel that signals graph termination.

## Compilation

Calling `builder.compile()` validates the graph structure (no unreachable nodes, no missing edges) and returns a `CompiledGraph` object with `invoke`, `ainvoke`, `stream`, and `astream` methods.

```python
graph = builder.compile()
result = await graph.ainvoke({"query": "What is LangGraph?"})
```

## Config Injection

To inject per-node configuration at construction time without polluting state, use `functools.partial`:

```python
from functools import partial

async def retrieval_node(state: GraphState, config: RetrievalConfig) -> dict:
    chunks = await vectorstore.search(state.query, top_k=config.top_k)
    return {"retrieved_chunks": chunks}

builder.add_node("retrieval", partial(retrieval_node, config=retrieval_cfg))
```

This keeps node functions pure and testable in isolation.

## Cycles and Retry Loops

LangGraph supports cycles, which enables retry loops. A verifier node can route back to an earlier node if verification fails, up to a maximum retry count tracked in state.

```python
def verifier_decision(state: GraphState) -> str:
    if state.verifier_result.outcome == "revise" and state.retry_count < 3:
        return "answer_generation"
    return END

builder.add_conditional_edges("verifier", verifier_decision)
```

## Streaming

Compiled graphs support streaming execution. Use `astream_events` to receive events as nodes execute:

```python
async for event in graph.astream_events({"query": "..."}, version="v2"):
    if event["event"] == "on_chain_end":
        print(event["data"])
```

Event types include `on_chain_start`, `on_chain_end`, `on_llm_start`, `on_llm_end`, and `on_tool_end`.
