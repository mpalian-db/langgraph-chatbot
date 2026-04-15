# Streaming in LangGraph

## Overview

LangGraph supports streaming execution so applications can display partial results as they are produced, rather than waiting for the entire graph to finish. This is important for interactive chat applications where latency is noticeable.

## Streaming Modes

LangGraph graphs compiled with `builder.compile()` expose four streaming interfaces:

- `graph.stream(input)` -- synchronous generator yielding state updates per node
- `graph.astream(input)` -- async generator yielding state updates per node
- `graph.astream_events(input, version="v2")` -- async generator yielding fine-grained events (LLM tokens, tool calls, node starts/ends)

For most chat applications, `astream_events` is the most useful because it gives access to LLM token-by-token output.

## Using astream_events

```python
async for event in graph.astream_events({"query": "What is LangGraph?"}, version="v2"):
    event_type = event["event"]

    if event_type == "on_chat_model_stream":
        # LLM is producing tokens
        chunk = event["data"]["chunk"]
        print(chunk.content, end="", flush=True)

    elif event_type == "on_chain_end":
        # A node finished
        node_name = event["name"]
        output = event["data"]["output"]
```

### Event Types

| Event | When it fires |
|---|---|
| `on_chain_start` | A node begins executing |
| `on_chain_end` | A node finishes executing |
| `on_chat_model_start` | An LLM call begins |
| `on_chat_model_stream` | An LLM token is produced |
| `on_chat_model_end` | An LLM call completes |
| `on_tool_start` | A tool call begins |
| `on_tool_end` | A tool call completes |

## Streaming via FastAPI

To stream graph output over HTTP, use Server-Sent Events (SSE) or newline-delimited JSON (NDJSON):

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def stream_chat(body: ChatRequest):
    async def generate():
        async for event in graph.astream_events(
            {"query": body.query}, version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

For NDJSON (easier to parse on the frontend):

```python
async def generate():
    async for event in graph.astream_events({"query": body.query}, version="v2"):
        if event["event"] == "on_chain_end" and event["name"] == "verifier":
            data = event["data"]["output"]
            yield json.dumps(data) + "\n"
```

## Filtering Events

`astream_events` produces many events. Filter by name or type to reduce noise:

```python
async for event in graph.astream_events(input, version="v2"):
    # Only care about LLM token output
    if event["event"] != "on_chat_model_stream":
        continue
    # Only from the answer generation node
    if "answer_generation" not in event.get("tags", []):
        continue
    token = event["data"]["chunk"].content
```

Tags are set at graph construction time by passing `tags=["node_name"]` when adding nodes.

## State Streaming with astream

For simpler use cases where you only need the state after each node completes (not individual tokens), `astream` is more convenient:

```python
async for state_update in graph.astream({"query": "..."}):
    # state_update is a dict of {node_name: partial_state}
    for node_name, partial_state in state_update.items():
        print(f"Node {node_name} completed")
        if "final_answer" in partial_state:
            print("Final answer:", partial_state["final_answer"])
```

## Frontend Integration

On the React frontend, use `fetch` with a `ReadableStream` to consume NDJSON:

```typescript
const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
});

const reader = response.body!.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const lines = decoder.decode(value).split("\n").filter(Boolean);
    for (const line of lines) {
        const data = JSON.parse(line);
        // handle data
    }
}
```
