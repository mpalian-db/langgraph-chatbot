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
from app.ports.conversation import ConversationReaderPort, ConversationWriterPort
from app.ports.embedding import EmbeddingPort
from app.ports.llm import LLMPort
from app.ports.notion import NotionPort
from app.ports.storage import DocumentStoragePort
from app.ports.vectorstore import CollectionPort, VectorStorePort
from app.ports.worklog import WorklogPort

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


def get_llm_registry(
    system_config: SystemConfig = Depends(get_system_config),
) -> dict[str, LLMPort]:
    """Build the LLM registry exposing every available provider.

    Per-node `provider` overrides in agents.toml resolve against this map.
    Ollama is always present (it's the local-dev default). Anthropic is only
    included when ANTHROPIC_API_KEY is set, so a node that requests
    `provider = "anthropic"` without credentials fails loudly at graph build
    with a clear "not registered" error, instead of a silent fallback or
    an opaque adapter-level crash. The graph is built per-request inside
    the chat handler, so the failure surfaces on first chat -- a future
    improvement is to validate the registry at application startup."""
    import os

    registry: dict[str, LLMPort] = {
        "ollama": OllamaLLMAdapter(base_url=system_config.llm.ollama_base_url),
    }
    if os.environ.get("ANTHROPIC_API_KEY"):
        registry["anthropic"] = AnthropicLLMAdapter()
    return registry


def get_vector_store(
    system_config: SystemConfig = Depends(get_system_config),
) -> VectorStorePort:
    provider = system_config.vectorstore.provider
    if provider == "qdrant":
        return QdrantVectorStoreAdapter(url=system_config.vectorstore.qdrant_url)
    if provider == "vectorize":
        import os

        from app.adapters.vectorstore.vectorize import VectorizeAdapter

        return VectorizeAdapter(
            account_id=os.environ.get("CF_ACCOUNT_ID") or os.environ["CLOUDFLARE_ACCOUNT_ID"],
            api_token=os.environ.get("CF_API_TOKEN") or os.environ["CLOUDFLARE_API_TOKEN"],
            index_name=system_config.vectorstore.vectorize_index_name,
            known_collections=system_config.vectorstore.known_collections,
        )
    msg = f"Unknown vector store provider: {provider}"
    raise ValueError(msg)


def get_collection_port(
    system_config: SystemConfig = Depends(get_system_config),
) -> CollectionPort:
    # The Qdrant adapter implements both VectorStorePort and CollectionPort.
    # The Vectorize adapter also implements both -- index provisioned externally.
    provider = system_config.vectorstore.provider
    if provider == "qdrant":
        return QdrantVectorStoreAdapter(url=system_config.vectorstore.qdrant_url)
    if provider == "vectorize":
        import os

        from app.adapters.vectorstore.vectorize import VectorizeAdapter

        return VectorizeAdapter(
            account_id=os.environ.get("CF_ACCOUNT_ID") or os.environ["CLOUDFLARE_ACCOUNT_ID"],
            api_token=os.environ.get("CF_API_TOKEN") or os.environ["CLOUDFLARE_API_TOKEN"],
            index_name=system_config.vectorstore.vectorize_index_name,
            known_collections=system_config.vectorstore.known_collections,
        )
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
            account_id=os.environ.get("CF_ACCOUNT_ID") or os.environ["CLOUDFLARE_ACCOUNT_ID"],
            api_token=os.environ.get("CF_API_TOKEN") or os.environ["CLOUDFLARE_API_TOKEN"],
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


def get_worklog() -> WorklogPort | None:
    import os

    worker_url = os.environ.get("WORKLOG_WORKER_URL", "")
    api_key = os.environ.get("WORKLOG_API_KEY", "")
    if not worker_url:
        return None

    from app.adapters.worklog.http import WorklogHTTPAdapter

    return WorklogHTTPAdapter(base_url=worker_url, api_key=api_key)


@lru_cache(maxsize=1)
def get_conversation_store():
    """Single shared SQLite-backed conversation store for the application.

    Cached because SQLiteConversationStore holds an open connection -- one
    instance per process is the right shape, and lru_cache here gives us
    that without smuggling globals into the module namespace.

    Note: lru_cache is not strictly thread-safe at cold-start. Under
    concurrent first-request initialization, two store instances may briefly
    be created. With file-backed SQLite this is benign (both connections
    address the same file). For ":memory:" this would matter -- only used
    in tests, which are single-process and not affected."""
    from app.adapters.conversation.sqlite import SQLiteConversationStore

    db_path = _PROJECT_ROOT / "data" / "conversations.sqlite"
    return SQLiteConversationStore(db_path=db_path)


def get_conversation_reader() -> ConversationReaderPort:
    return get_conversation_store()


def get_conversation_writer() -> ConversationWriterPort:
    return get_conversation_store()


# ---------------------------------------------------------------------------
# Typed dependency aliases for route signatures
# ---------------------------------------------------------------------------

SystemConfigDep = Annotated[SystemConfig, Depends(get_system_config)]
AgentsConfigDep = Annotated[AgentsConfig, Depends(get_agents_config)]
LLMDep = Annotated[LLMPort, Depends(get_llm)]
LLMRegistryDep = Annotated[dict[str, LLMPort], Depends(get_llm_registry)]
VectorStoreDep = Annotated[VectorStorePort, Depends(get_vector_store)]
CollectionDep = Annotated[CollectionPort, Depends(get_collection_port)]
EmbeddingDep = Annotated[EmbeddingPort, Depends(get_embedding)]
StorageDep = Annotated[DocumentStoragePort, Depends(get_storage)]
ConversationReaderDep = Annotated[ConversationReaderPort, Depends(get_conversation_reader)]
ConversationWriterDep = Annotated[ConversationWriterPort, Depends(get_conversation_writer)]
NotionDep = Annotated[NotionPort, Depends(get_notion)]
WorklogDep = Annotated[WorklogPort | None, Depends(get_worklog)]
