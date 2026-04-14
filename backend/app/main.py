"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import get_system_config
from app.api.routes import chat, collections, documents, system

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: configure tracing on startup, clean up on shutdown."""
    config = get_system_config()

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

    return app


app = create_app()
