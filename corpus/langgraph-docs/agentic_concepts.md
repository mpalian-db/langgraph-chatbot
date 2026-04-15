# Agentic Concepts in LangGraph

## What is an Agent?

In LangGraph, an agent is a graph node (or a subgraph) that uses an LLM to decide what to do next. Unlike a fixed pipeline, an agent can call tools, observe results, and decide whether to call more tools or produce a final response -- all within a single invocation.

## The ReAct Pattern

The most common agentic pattern is ReAct (Reason + Act):

1. The LLM receives the current state (query + any prior tool results).
2. It reasons about what to do and produces a structured response containing either a tool call or a final answer.
3. If a tool call, the tool is executed and the result is appended to state.
4. Steps 1-3 repeat until the LLM produces a final answer or the maximum number of tool calls is reached.

```python
async def tool_agent(state: GraphState, config: ToolAgentConfig) -> dict:
    messages = [{"role": "user", "content": state.query}]

    response = await llm.complete(
        messages=messages,
        tools=config.allowed_tools,
        model=config.model,
    )

    if response["stop_reason"] == "tool_use":
        # Execute tools and loop back
        tool_results = await execute_tools(response["tool_use"])
        return {"tool_calls": tool_results, "retry_count": state.retry_count + 1}

    return {"final_answer": response["text"]}
```

## Tool Definitions

Tools are described to the LLM using a JSON schema. Each tool has a name, description, and input schema:

```python
SEARCH_COLLECTION_SCHEMA = {
    "name": "search_collection",
    "description": "Search a vector collection for documents relevant to a query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "collection": {"type": "string", "description": "Collection name."},
            "top_k": {"type": "integer", "description": "Number of results.", "default": 5},
        },
        "required": ["query", "collection"],
    },
}
```

The LLM receives these schemas and produces tool call objects matching the schema when it wants to invoke a tool.

## Tool Execution

Tool calls returned by the LLM must be dispatched to actual Python functions. A common pattern is a registry dict mapping tool names to callables:

```python
TOOL_REGISTRY = {
    "search_collection": search_collection,
    "list_collections": list_collections,
    "get_collection_stats": get_collection_stats,
}

async def execute_tools(tool_calls: list[dict]) -> list[dict]:
    results = []
    for call in tool_calls:
        fn = TOOL_REGISTRY[call["name"]]
        result = await fn(**call["input"])
        results.append({"tool_use_id": call["id"], "result": result})
    return results
```

## Multi-Agent Systems

LangGraph supports multi-agent systems where a supervisor agent routes tasks to specialist agents.

### Supervisor Pattern

```
user_query
    |
    v
supervisor (decides which agent to call)
    |------> researcher (retrieves and summarises documents)
    |------> writer (drafts content)
    |------> reviewer (checks quality)
    |
    v
final_response
```

Each specialist is a subgraph compiled separately and added as a node in the supervisor graph:

```python
researcher_graph = build_researcher_graph()
writer_graph = build_writer_graph()

supervisor = StateGraph(SupervisorState)
supervisor.add_node("researcher", researcher_graph)
supervisor.add_node("writer", writer_graph)
```

### Handoff Pattern

Agents can hand off control to each other by returning a special state value:

```python
async def researcher(state: AgentState) -> dict:
    result = await do_research(state.query)
    # Signal handoff to writer
    return {"research_result": result, "next_agent": "writer"}
```

## Guardrails and Limits

### Maximum Tool Calls

Always cap the number of tool calls to prevent infinite loops:

```python
if state.retry_count >= config.max_tool_calls:
    return {"final_answer": "Could not complete the task within the allowed steps."}
```

### Allowed Tools

Restrict which tools an agent can call to the minimum set required for its task. Pass `allowed_tools` from config so the restriction is configurable without code changes.

### Input Validation

Validate tool inputs before execution. The LLM may produce malformed inputs, especially for edge cases:

```python
async def safe_search(query: str, collection: str, top_k: int = 5) -> list:
    if not query.strip():
        return []
    if top_k > 20:
        top_k = 20
    return await vectorstore.search(query, top_k=top_k, collection=collection)
```

## RAG as a Specialised Agent Pattern

Retrieval-Augmented Generation (RAG) is a specific agentic pattern where the agent:

1. Reformulates the query for retrieval (optional)
2. Retrieves relevant document chunks from a vector store
3. Generates an answer grounded in the retrieved chunks
4. Verifies that the answer is supported by the retrieved evidence
5. Returns the answer with citations, or loops back to revise if verification fails

The verifier step is what distinguishes a RAG pipeline from simple retrieval + generation. It prevents hallucination by refusing or revising answers that are not supported by the retrieved evidence.
