from __future__ import annotations

import time
from typing import Any

from app.core.config.models import ToolAgentConfig
from app.core.graph.state import GraphState
from app.core.models.types import ToolCall, TraceEntry
from app.core.operations.collections import rebuild_collection
from app.ports.embedding import EmbeddingPort
from app.ports.llm import LLMPort
from app.ports.vectorstore import CollectionPort, VectorStorePort
from app.tools.definitions import get_tools_for_agent


async def run(
    state: GraphState,
    *,
    config: ToolAgentConfig,
    llm: LLMPort,
    vectorstore: VectorStorePort,
    collection_store: CollectionPort,
    embedding: EmbeddingPort,
) -> dict[str, Any]:
    start = time.monotonic()

    tools = get_tools_for_agent(config.allowed_tools)
    messages: list[dict] = [{"role": "user", "content": state.query}]
    tool_calls_made: list[ToolCall] = []
    final_text = ""

    for _ in range(config.max_tool_calls):
        response = await llm.complete(
            messages=messages,
            model=config.model,
            tools=tools,
            max_tokens=1024,
        )

        if response["stop_reason"] == "end_turn" or not response["tool_use"]:
            final_text = response["text"]
            break

        for tool_use in response["tool_use"]:
            tool_result = await _execute_tool(
                tool_use["name"],
                tool_use["input"],
                vectorstore=vectorstore,
                collection_store=collection_store,
                embedding=embedding,
            )
            tool_calls_made.append(
                ToolCall(
                    name=tool_use["name"], arguments=tool_use["input"], result=str(tool_result)
                )
            )
            messages.append({"role": "assistant", "content": response["text"] or ""})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": str(tool_result),
                        },
                    ],
                }
            )

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "final_answer": final_text,
        "tool_calls": state.tool_calls + tool_calls_made,
        "execution_trace": state.execution_trace
        + [
            TraceEntry(
                node="tool_agent", duration_ms=elapsed_ms, data={"tool_calls": len(tool_calls_made)}
            )
        ],
    }


async def _execute_tool(
    name: str,
    args: dict,
    *,
    vectorstore: VectorStorePort,
    collection_store: CollectionPort,
    embedding: EmbeddingPort,
) -> Any:
    if name == "list_collections":
        return await collection_store.list_collections()
    if name == "get_collection_stats":
        return await collection_store.get_stats(args["collection"])
    if name == "search_collection":
        [query_vector] = await embedding.embed([args["query"]])
        chunks = await vectorstore.search(
            query_vector=query_vector,
            top_k=args.get("top_k", 5),
            collection=args["collection"],
        )
        return [{"id": c.id, "text": c.text[:200], "score": c.score} for c in chunks]
    if name == "rebuild_index":
        coll = args["collection"]
        vector_size = args.get("vector_size", 768)
        await rebuild_collection(collection_store, coll, vector_size)
        return {"collection": coll, "status": "rebuilt"}
    return f"Unknown tool: {name}"
