# LangGraph High-Level Overview

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain's expression language by adding first-class support for cycles, controllable state, and multi-agent coordination.

The key insight behind LangGraph is that complex LLM applications require more than a linear chain of operations. Real-world agents need to loop, branch, coordinate between specialised sub-agents, and maintain persistent state across multiple interactions.

## Why LangGraph?

**Linear chains are not enough.** Most LLM frameworks model pipelines as directed acyclic graphs (DAGs) -- each step runs once and passes output to the next. This is suitable for simple question-answering but breaks down for:

- Retrieval-augmented generation with verification and retry
- Multi-step reasoning with reflection
- Tool-using agents that may call multiple tools in sequence
- Multi-agent systems where a supervisor routes to specialists

**LangGraph adds cycles.** By modelling the workflow as a stateful graph with conditional routing, LangGraph supports all of the above patterns without abandoning the composability benefits of the LangChain ecosystem.

## Core Abstractions

### Graphs

A `StateGraph` defines the topology of the application. Nodes are processing units (LLM calls, tool calls, transformations). Edges define control flow, which can be static or conditional.

### State

State is the single source of truth passed between nodes. It is a typed dataclass or TypedDict. Every node reads from and writes to state. LangGraph handles merging partial updates, making the data flow explicit and inspectable.

### Nodes

Nodes are ordinary Python functions (sync or async) that take state and return a partial state update. They are the primary unit of work and can wrap any operation: LLM inference, vector search, API calls, or custom logic.

### Edges

Conditional edges implement routing logic. A router function inspects the current state and returns the name of the next node. This is how LangGraph supports classification, retry loops, and multi-agent delegation.

## Execution Model

When `graph.ainvoke(state)` is called, LangGraph:

1. Runs the entry node.
2. Evaluates outgoing edges (static or conditional) to determine the next node.
3. Merges the node's return dict into the running state.
4. Repeats until a node routes to `END`.

Steps are sequential within a single thread. Parallel execution across branches is possible with `Send` and fan-out patterns.

## Comparison to LangChain LCEL

| Feature | LCEL | LangGraph |
|---|---|---|
| Topology | DAG only | Arbitrary graphs with cycles |
| State | Implicit (piped values) | Explicit typed state object |
| Branching | Limited | Full conditional routing |
| Retry loops | Not supported | First-class |
| Multi-agent | Not supported | Built-in via subgraphs |
| Streaming | Event streaming | Fine-grained event streaming |

LangGraph is the recommended approach for any application that requires agent-like behaviour, retry loops, or multi-step reasoning.

## Typical Patterns

### RAG with Verification

```
user_query -> router -> retrieval -> answer_generation -> verifier
                                          ^                    |
                                          |__ revise loop _____|
                                                               v
                                                          final_response
```

### Tool-Using Agent

```
user_query -> router -> tool_agent -> (tool calls) -> tool_agent -> final_response
```

### Multi-Agent Supervisor

```
user_query -> supervisor -> researcher -> supervisor
                         -> writer     -> supervisor -> final_response
                         -> reviewer   -> supervisor
```

## Integration with LangChain

LangGraph is part of the LangChain ecosystem but does not require using LangChain's model wrappers. Any async callable can be a node. The main integration point is the tracing and observability layer via LangSmith or Langfuse.
