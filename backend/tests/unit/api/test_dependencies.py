"""Tests for the dependency providers in app.api.dependencies.

Focus on the LLM registry lifecycle: which providers are exposed under what
environment conditions. This pins the behaviour `_resolve_llm` in graph.py
relies on -- if the registry omits a key, the graph build must fail loudly,
not silently fall back."""

from __future__ import annotations

import pytest

from app.api.dependencies import get_llm_registry
from app.core.config.models import LLMConfig, SystemConfig


def _system_config() -> SystemConfig:
    """A minimal system config with ollama as the default provider."""
    config = SystemConfig()
    config.llm = LLMConfig(provider="ollama", ollama_base_url="http://localhost:11434")
    return config


def test_registry_always_contains_ollama(monkeypatch: pytest.MonkeyPatch):
    """Ollama is the local-dev default; it must always be available regardless
    of which other API keys are present."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    registry = get_llm_registry(_system_config())

    assert "ollama" in registry


def test_registry_excludes_anthropic_when_api_key_missing(monkeypatch: pytest.MonkeyPatch):
    """Without ANTHROPIC_API_KEY, the registry must not register an Anthropic
    adapter -- a node requesting `provider = "anthropic"` then fails loudly
    at graph build with a clear "not registered" error."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    registry = get_llm_registry(_system_config())

    assert "anthropic" not in registry


def test_registry_includes_anthropic_when_api_key_present(monkeypatch: pytest.MonkeyPatch):
    """With ANTHROPIC_API_KEY set, the Anthropic adapter is constructed and
    exposed so per-node provider overrides can resolve to it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")

    registry = get_llm_registry(_system_config())

    assert "anthropic" in registry
