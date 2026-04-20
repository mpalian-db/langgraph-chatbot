# Persistence in LangGraph

## Overview

LangGraph supports persistence through checkpointers. A checkpointer saves the graph state after each node execution so that:

- Graphs can be interrupted and resumed (human-in-the-loop patterns)
- Conversation history is preserved across multiple invocations
- State can be inspected or replayed for debugging

Persistence is optional. Graphs without a checkpointer run statelessly -- each invocation starts from scratch.

## Checkpointers

A checkpointer is an object that implements the `BaseCheckpointSaver` interface. LangGraph ships with several implementations:

| Checkpointer | Storage | When to use |
|---|---|---|
| `MemorySaver` | In-process dict | Development and testing |
| `SqliteSaver` | SQLite file | Single-machine persistence |
| `PostgresSaver` | PostgreSQL | Production, multi-instance |
| `RedisSaver` | Redis | High-throughput production |

### Using MemorySaver

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

### Using SqliteSaver

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

## Thread IDs

When a checkpointer is configured, every invocation must include a `thread_id` in the config. This identifies the conversation thread whose state is being read and written.

```python
config = {"configurable": {"thread_id": "user-123-session-abc"}}

result = await graph.ainvoke({"query": "What is LangGraph?"}, config=config)

# The second invocation resumes from the saved state
result2 = await graph.ainvoke({"query": "Can you elaborate?"}, config=config)
```

If no `thread_id` is provided and a checkpointer is configured, LangGraph raises an error.

## State History

With a checkpointer, you can inspect the full history of state snapshots for a thread:

```python
history = list(graph.get_state_history(config))
for snapshot in history:
    print(snapshot.values)  # GraphState at that point in time
    print(snapshot.next)     # Which node runs next
```

## Human-in-the-Loop

Persistence enables human-in-the-loop patterns. A graph can be interrupted before a specific node and resumed after a human approves the action.

### Interrupting Before a Node

```python
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["tool_agent"],  # pause before tool calls
)

# First invocation -- stops before tool_agent
result = await graph.ainvoke({"query": "Delete the docs collection"}, config=config)
# result["__interrupt__"] describes the pending node

# Human approves -- resume
result = await graph.ainvoke(None, config=config)
```

### Updating State Before Resuming

Before resuming, a human (or automated check) can modify the pending state:

```python
graph.update_state(config, {"approved": True})
result = await graph.ainvoke(None, config=config)
```

## Conversation Memory

For multi-turn chat, the simplest approach is to include the conversation history in state and append each new message before invoking the graph:

```python
@dataclass
class ChatState:
    messages: list[dict] = field(default_factory=list)
    final_answer: str | None = None

async def chat_node(state: ChatState) -> dict:
    response = await llm.complete(messages=state.messages)
    new_message = {"role": "assistant", "content": response["text"]}
    return {"messages": state.messages + [new_message], "final_answer": response["text"]}
```

With a checkpointer, the `messages` list grows across invocations automatically, giving the LLM full conversation context.

## Clearing State

To start a fresh conversation on an existing thread, clear its state:

```python
graph.update_state(config, GraphState(query="", messages=[]))
```

Or simply use a new `thread_id` for each conversation session.
