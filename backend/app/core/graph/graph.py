from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

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


def _resolve_llm(agent_cfg: Any, llms: Mapping[str, LLMPort], default_provider: str) -> LLMPort:
    """Pick the LLM port for an agent: its `provider` override, else default.

    Fails loudly if the requested provider isn't in the registry -- that means
    the operator asked for an LLM (e.g. anthropic) without supplying credentials,
    and silently falling back to ollama would mask the misconfiguration. Direct
    `.provider` access (no `getattr` default) makes a future config refactor
    that drops the field surface as AttributeError here rather than silently
    routing through the default."""
    provider = agent_cfg.provider or default_provider
    if provider not in llms:
        msg = (
            f"agent config requests LLM provider {provider!r} but it is not "
            f"registered. Available: {sorted(llms.keys())}. Check that any "
            "required API keys are set in the environment."
        )
        raise ValueError(msg)
    return llms[provider]


def build_graph(
    agents_config: AgentsConfig,
    llms: Mapping[str, LLMPort],
    default_provider: str,
    vectorstore: VectorStorePort,
    collection_store: CollectionPort,
    embedding: EmbeddingPort,
    worklog: WorklogPort | None = None,
):
    builder = StateGraph(GraphState)

    pick = lambda cfg: _resolve_llm(cfg, llms, default_provider)  # noqa: E731

    # Drop the worklog route from the router's known options when no
    # WorklogPort is configured. Use model_copy() rather than mutating the
    # input config -- get_agents_config() returns an lru_cached singleton, so
    # in-place mutation would leak across requests.
    if worklog is None:
        router_cfg = agents_config.router.model_copy(
            update={"routes": [r for r in agents_config.router.routes if r != "worklog"]}
        )
    else:
        router_cfg = agents_config.router

    builder.add_node(
        "router",
        partial(router.run, config=router_cfg, llm=pick(router_cfg)),
    )
    builder.add_node(
        "chat_agent",
        partial(
            chat_agent.run, config=agents_config.chat_agent, llm=pick(agents_config.chat_agent)
        ),
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
        partial(
            answer_generation.run,
            config=agents_config.answer_generation,
            llm=pick(agents_config.answer_generation),
        ),
    )
    builder.add_node(
        "verifier",
        partial(verifier.run, config=agents_config.verifier, llm=pick(agents_config.verifier)),
    )
    builder.add_node(
        "tool_agent",
        partial(
            tool_agent.run,
            config=agents_config.tool_agent,
            llm=pick(agents_config.tool_agent),
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
                llm=pick(agents_config.worklog_agent),
                worklog=worklog,
            ),
        )

    builder.set_entry_point("router")

    route_map: dict[Hashable, str] = {
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
