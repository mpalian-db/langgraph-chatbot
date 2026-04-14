"""System endpoints -- health check and configuration."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.dependencies import AgentsConfigDep, SystemConfigDep
from app.core.config.models import AgentsConfig

router = APIRouter(prefix="/system", tags=["system"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str


class ConfigResponse(BaseModel):
    environment: dict[str, Any]
    llm: dict[str, Any]
    embeddings: dict[str, Any]
    vectorstore: dict[str, Any]
    tracing: dict[str, Any]
    agents: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Lightweight liveness probe."""
    return HealthResponse(status="ok")


@router.get("/config", response_model=ConfigResponse)
async def show_config(
    system_config: SystemConfigDep,
    agents_config: AgentsConfigDep,
) -> ConfigResponse:
    """Return the active (non-secret) system and agent configuration."""
    # Determine the active embedding model based on provider.
    if system_config.embeddings.provider == "ollama":
        embedding_model = system_config.embeddings.ollama_model
    else:
        embedding_model = system_config.embeddings.workers_ai_model

    return ConfigResponse(
        environment={
            "mode": system_config.environment.mode,
            "log_level": system_config.environment.log_level,
        },
        llm={
            "provider": system_config.llm.provider,
        },
        embeddings={
            "provider": system_config.embeddings.provider,
            "model": embedding_model,
        },
        vectorstore={
            "provider": system_config.vectorstore.provider,
        },
        tracing={
            "enabled": system_config.tracing.langfuse_enabled,
            "host": system_config.tracing.langfuse_host,
        },
        agents=list(AgentsConfig.model_fields),
    )
