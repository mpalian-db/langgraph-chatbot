from __future__ import annotations

from functools import partial
from typing import Literal

from langgraph.graph import END, StateGraph

from app.core.config.models import AgentsConfig
from app.core.graph.nodes import (
    answer_generation,
    chat_agent,
    retrieval,
    router,
    tool_agent,
    verifier,
    worklog_agent,
)
from app.core.graph.state import GraphState
from app.ports.embedding import EmbeddingPort
from app.ports.llm import LLMPort
from app.ports.vectorstore import CollectionPort, VectorStorePort
from app.ports.worklog import WorklogPort


def build_graph(
    agents_config: AgentsConfig,
    llm: LLMPort,
    vectorstore: VectorStorePort,
    collection_store: CollectionPort,
    embedding: EmbeddingPort,
    worklog: WorklogPort | None = None,
):
    builder = StateGraph(GraphState)

    builder.add_node(
        "router",
        partial(router.run, config=agents_config.router, llm=llm),
    )
    builder.add_node(
        "chat_agent",
        partial(chat_agent.run, config=agents_config.chat_agent, llm=llm),
    )
    builder.add_node(
        "retrieval",
        partial(
            retrieval.run,
            config=agents_config.retrieval,
            vectorstore=vectorstore,
            embedding=embedding,
        ),
    )
    builder.add_node(
        "answer_generation",
        partial(answer_generation.run, config=agents_config.answer_generation, llm=llm),
    )
    builder.add_node(
        "verifier",
        partial(verifier.run, config=agents_config.verifier, llm=llm),
    )
    builder.add_node(
        "tool_agent",
        partial(
            tool_agent.run,
            config=agents_config.tool_agent,
            llm=llm,
            vectorstore=vectorstore,
            collection_store=collection_store,
            embedding=embedding,
        ),
    )

    if worklog is not None:
        builder.add_node(
            "worklog_agent",
            partial(
                worklog_agent.run,
                config=agents_config.worklog_agent,
                llm=llm,
                worklog=worklog,
            ),
        )

    builder.set_entry_point("router")

    route_map: dict[str, str] = {
        "chat": "chat_agent",
        "rag": "retrieval",
        "tool": "tool_agent",
    }
    if worklog is not None:
        route_map["worklog"] = "worklog_agent"

    builder.add_conditional_edges(
        "router",
        lambda state: state.route or "chat",
        route_map,
    )
    builder.add_edge("chat_agent", END)
    builder.add_edge("tool_agent", END)
    if worklog is not None:
        builder.add_edge("worklog_agent", END)
    builder.add_edge("retrieval", "answer_generation")
    builder.add_edge("answer_generation", "verifier")
    builder.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"revise": "answer_generation", "__end__": END},
    )

    return builder.compile()


def _route_after_verifier(state: GraphState) -> Literal["revise", "__end__"]:
    if (
        state.verifier_result is not None
        and state.verifier_result.outcome == "revise"
        and state.final_answer is None
    ):
        return "revise"
    return "__end__"
