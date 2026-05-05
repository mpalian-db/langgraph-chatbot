from __future__ import annotations

import os
import pathlib
import tomllib

from app.core.config.models import AgentsConfig, SystemConfig


def load_system_config(path: pathlib.Path) -> SystemConfig:
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f)
        return SystemConfig.model_validate(data)
    return _system_config_from_env()


def load_agents_config(path: pathlib.Path) -> AgentsConfig:
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f)
        return AgentsConfig.model_validate(data)
    return _agents_config_from_env()


def _system_config_from_env() -> SystemConfig:
    """Build SystemConfig from environment variables (Cloudflare Workers deployment)."""
    return SystemConfig.model_validate(
        {
            "environment": {"mode": "cloudflare", "log_level": os.environ.get("LOG_LEVEL", "info")},
            "llm": {"provider": os.environ.get("LLM_PROVIDER", "anthropic")},
            "tracing": {"langfuse_enabled": False},
            "vectorstore": {
                "provider": os.environ.get("VECTORSTORE_PROVIDER", "vectorize"),
                "vectorize_index_name": os.environ.get("VECTORIZE_INDEX_NAME", "langgraph-chatbot"),
            },
            "embeddings": {
                "provider": os.environ.get("EMBEDDINGS_PROVIDER", "workers-ai"),
                "workers_ai_model": os.environ.get(
                    "WORKERS_AI_MODEL", "@cf/baai/bge-small-en-v1.5"
                ),
            },
            "webhooks": {
                "edgenotes_secret": os.environ.get("LANGGRAPH_WEBHOOK_SECRET", ""),
            },
        }
    )


def _agents_config_from_env() -> AgentsConfig:
    """Build AgentsConfig with Anthropic-appropriate defaults for CF deployment."""
    model = os.environ.get("CF_DEFAULT_MODEL", "claude-haiku-4-5-20251001")
    large_model = os.environ.get("CF_LARGE_MODEL", "claude-sonnet-4-6-20250514")
    return AgentsConfig.model_validate(
        {
            "router": {"model": model},
            "chat_agent": {"model": model},
            "retrieval": {},
            "answer_generation": {"model": large_model},
            "verifier": {"model": large_model},
            "tool_agent": {"model": model},
            "worklog_agent": {"model": model},
        }
    )
