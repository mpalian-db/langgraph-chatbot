"""Unit tests for the system endpoints (health, config).

These do not require external services -- config providers are overridden
with in-memory defaults.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_agents_config, get_system_config
from app.core.config.models import AgentsConfig, SystemConfig
from app.main import create_app


def _override_system_config() -> SystemConfig:
    return SystemConfig()


def _override_agents_config() -> AgentsConfig:
    return AgentsConfig()


@pytest.fixture
def test_app():
    app = create_app()
    app.dependency_overrides[get_system_config] = _override_system_config
    app.dependency_overrides[get_agents_config] = _override_agents_config
    return app


@pytest.fixture
async def client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/api/system/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_config_returns_expected_keys(client: AsyncClient) -> None:
    response = await client.get("/api/system/config")
    assert response.status_code == 200
    data = response.json()
    assert data["environment"]["mode"] == "local"
    assert data["llm"]["provider"] == "ollama"
    assert data["embeddings"]["provider"] == "ollama"
    assert data["embeddings"]["model"] == "nomic-embed-text"
    assert isinstance(data["agents"], list)
    assert "router" in data["agents"]
    assert "verifier" in data["agents"]
