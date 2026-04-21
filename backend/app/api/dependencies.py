"""FastAPI dependency providers for adapter injection.

Each provider resolves to an adapter instance driven by config/config.toml
and config/agents.toml.  All ports are satisfied here so that route handlers
never construct adapters directly.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.adapters.embeddings.ollama import OllamaEmbeddingAdapter
from app.adapters.llm.anthropic import AnthropicLLMAdapter
from app.adapters.llm.ollama import OllamaLLMAdapter
from app.adapters.storage.local import LocalFileStorageAdapter
from app.adapters.vectorstore.qdrant import QdrantVectorStoreAdapter
from app.core.config.loader import load_agents_config, load_system_config
from app.core.config.models import AgentsConfig, SystemConfig
from app.ports.embedding import EmbeddingPort
from app.ports.llm import LLMPort
from app.ports.notion import NotionPort
from app.ports.storage import DocumentStoragePort
from app.ports.vectorstore import CollectionPort, VectorStorePort

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]  # backend/app/api -> repo root
_SYSTEM_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.toml"
_AGENTS_CONFIG_PATH = _PROJECT_ROOT / "config" / "agents.toml"

# ---------------------------------------------------------------------------
# Config loaders (cached -- parsed once per process)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_system_config() -> SystemConfig:
    return load_system_config(_SYSTEM_CONFIG_PATH)


@lru_cache(maxsize=1)
def get_agents_config() -> AgentsConfig:
    return load_agents_config(_AGENTS_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Adapter factories
# ---------------------------------------------------------------------------


def get_llm(
    system_config: SystemConfig = Depends(get_system_config),
) -> LLMPort:
    provider = system_config.llm.provider
    if provider == "ollama":
        return OllamaLLMAdapter(base_url=system_config.llm.ollama_base_url)
    if provider == "anthropic":
        return AnthropicLLMAdapter()
    msg = f"Unknown LLM provider: {provider}"
    raise ValueError(msg)


def get_vector_store(
    system_config: SystemConfig = Depends(get_system_config),
) -> VectorStorePort:
    provider = system_config.vectorstore.provider
    if provider == "qdrant":
        return QdrantVectorStoreAdapter(url=system_config.vectorstore.qdrant_url)
    msg = f"Unknown vector store provider: {provider}"
    raise ValueError(msg)


def get_collection_port(
    system_config: SystemConfig = Depends(get_system_config),
) -> CollectionPort:
    # The Qdrant adapter implements both VectorStorePort and CollectionPort.
    provider = system_config.vectorstore.provider
    if provider == "qdrant":
        return QdrantVectorStoreAdapter(url=system_config.vectorstore.qdrant_url)
    msg = f"Unknown collection provider: {provider}"
    raise ValueError(msg)


def get_embedding(
    system_config: SystemConfig = Depends(get_system_config),
) -> EmbeddingPort:
    provider = system_config.embeddings.provider
    if provider == "ollama":
        return OllamaEmbeddingAdapter(
            model=system_config.embeddings.ollama_model,
            base_url=system_config.embeddings.ollama_base_url,
        )
    if provider == "workers-ai":
        # Workers AI requires account credentials from environment variables.
        import os

        from app.adapters.embeddings.workers_ai import WorkersAIEmbeddingAdapter

        return WorkersAIEmbeddingAdapter(
            account_id=os.environ["CF_ACCOUNT_ID"],
            api_token=os.environ["CF_API_TOKEN"],
            model=system_config.embeddings.workers_ai_model,
        )
    msg = f"Unknown embedding provider: {provider}"
    raise ValueError(msg)


def get_storage() -> DocumentStoragePort:
    storage_dir = _PROJECT_ROOT / "data" / "documents"
    return LocalFileStorageAdapter(base_dir=storage_dir)


def get_notion() -> NotionPort:
    import os

    from app.adapters.ingestion.notion import NotionAdapter

    token = os.environ.get("NOTION_TOKEN", "")
    return NotionAdapter(token=token)


# ---------------------------------------------------------------------------
# Typed dependency aliases for route signatures
# ---------------------------------------------------------------------------

SystemConfigDep = Annotated[SystemConfig, Depends(get_system_config)]
AgentsConfigDep = Annotated[AgentsConfig, Depends(get_agents_config)]
LLMDep = Annotated[LLMPort, Depends(get_llm)]
VectorStoreDep = Annotated[VectorStorePort, Depends(get_vector_store)]
CollectionDep = Annotated[CollectionPort, Depends(get_collection_port)]
EmbeddingDep = Annotated[EmbeddingPort, Depends(get_embedding)]
StorageDep = Annotated[DocumentStoragePort, Depends(get_storage)]
NotionDep = Annotated[NotionPort, Depends(get_notion)]
