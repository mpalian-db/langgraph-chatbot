"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import (
    get_agents_config,
    get_collection_port,
    get_llm_registry,
    get_system_config,
)
from app.api.routes import chat, collections, documents, notion, system, webhooks
from app.core.graph.graph import validate_llm_providers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: configure tracing on startup, clean up on shutdown."""
    config = get_system_config()

    # Validate LLM provider wiring against agents.toml at startup so a missing
    # ANTHROPIC_API_KEY (or any other unregistered provider override) crashes
    # the app immediately with a clear message, instead of returning 500 on
    # the first chat request.
    agents_config = get_agents_config()
    llm_registry = get_llm_registry(system_config=config)
    # Skip agents that won't be wired into the graph at runtime -- otherwise a
    # bad worklog_agent.provider would crash startup even when WORKLOG_WORKER_URL
    # is unset and the node is never instantiated.
    import os as _os

    skip_agents: set[str] = set()
    if not _os.environ.get("WORKLOG_WORKER_URL"):
        skip_agents.add("worklog_agent")

    validate_llm_providers(
        agents_config=agents_config,
        llms=llm_registry,
        default_provider=config.llm.provider,
        skip_agents=skip_agents,
    )
    logger.info(
        "LLM provider validation passed (registry: %s, default: %s, skipped: %s)",
        sorted(llm_registry.keys()),
        config.llm.provider,
        sorted(skip_agents) or "none",
    )

    # Initialise Langfuse tracing when enabled.
    if config.tracing.langfuse_enabled:
        try:
            from langfuse import Langfuse

            app.state.langfuse = Langfuse(
                host=config.tracing.langfuse_host,
            )
            logger.info(
                "Langfuse tracing enabled (host=%s, project=%s)",
                config.tracing.langfuse_host,
                config.tracing.langfuse_project,
            )
        except Exception:
            logger.warning("Failed to initialise Langfuse -- tracing disabled", exc_info=True)
            app.state.langfuse = None
    else:
        app.state.langfuse = None

    # Ensure required collections exist (idempotent; no-op for Vectorize).
    try:
        collection_port = get_collection_port(system_config=config)
        existing = await collection_port.list_collections()
        for name in (
            "langgraph-docs",
            config.webhooks.edgenotes_collection,
            config.notion.default_collection,
        ):
            if name not in existing:
                await collection_port.create(name, vector_size=768)
                logger.info("Created collection: %s", name)
    except Exception:
        logger.warning("Could not ensure collections on startup", exc_info=True)

    yield

    # Shutdown: flush Langfuse if active.
    if getattr(app.state, "langfuse", None) is not None:
        try:
            app.state.langfuse.flush()
        except Exception:
            logger.warning("Error flushing Langfuse on shutdown", exc_info=True)


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="LangGraph RAG Chatbot",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS -- permissive for local dev, tighten for production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers under /api prefix.
    app.include_router(chat.router, prefix="/api")
    app.include_router(collections.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(system.router, prefix="/api")
    app.include_router(webhooks.router, prefix="/api")
    app.include_router(notion.router, prefix="/api")

    return app


app = create_app()
