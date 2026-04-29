import pathlib

import pytest
from pydantic import ValidationError

from app.core.config.loader import load_agents_config, load_system_config
from app.core.config.models import AgentsConfig, SystemConfig


def test_system_config_defaults():
    config = SystemConfig()
    assert config.environment.mode == "local"
    assert config.vectorstore.provider == "qdrant"
    assert config.ingestion.chunk_size == 512


def test_agents_config_defaults():
    config = AgentsConfig()
    assert config.router.model == "llama3.2:3b"
    assert config.verifier.max_retries == 2
    assert config.verifier.score_threshold == 0.75
    assert config.tool_agent.max_tool_calls == 5


def test_system_config_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        SystemConfig.model_validate({"environment": {"mode": "local"}, "unknown_key": "bad"})


def test_verifier_checks_defaults():
    config = AgentsConfig()
    assert "score_threshold" in config.verifier.checks
    assert "support_analysis" in config.verifier.checks
    assert "citation_coverage" in config.verifier.checks


def _find_project_root() -> pathlib.Path:
    """Walk up from this file until a directory containing both `config/` and
    `pyproject.toml` is found. More robust than hardcoded `parents[N]` -- it
    survives test relocations and split-out backend layouts."""
    here = pathlib.Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (
            (candidate / "config" / "agents.toml").is_file()
            and (candidate / "pyproject.toml").is_file()
            or (candidate / "config" / "agents.toml").is_file()
            and (candidate / "backend" / "pyproject.toml").is_file()
        ):
            return candidate
    msg = "could not find project root with config/agents.toml"
    raise RuntimeError(msg)


def test_production_thresholds_align_retrieval_with_verifier():
    """Invariant: retrieval.score_threshold >= verifier.score_threshold.

    A chunk that passes retrieval must not be refused by the verifier on
    score alone -- otherwise the answer-generation LLM call is burnt for a
    verifier refusal that was inevitable from the moment of retrieval. If
    this test fails, either lower the verifier threshold or raise retrieval
    to match. See agents.toml for the rationale."""
    project_root = _find_project_root()
    config = load_agents_config(project_root / "config" / "agents.toml")

    assert config.retrieval.score_threshold >= config.verifier.score_threshold, (
        f"retrieval threshold {config.retrieval.score_threshold} is below "
        f"verifier threshold {config.verifier.score_threshold} -- "
        "answer-generation LLM call would be burnt for inevitable refusal"
    )


def test_load_system_config_from_toml(tmp_path):
    toml_content = """\
[environment]
mode = "local"
log_level = "debug"

[llm]
provider = "ollama"
ollama_base_url = "http://localhost:11434"

[tracing]
langfuse_enabled = false
langfuse_host = "http://localhost:3000"
langfuse_project = "test"

[vectorstore]
provider = "qdrant"
qdrant_url = "http://localhost:6333"

[embeddings]
provider = "ollama"
ollama_model = "nomic-embed-text"
ollama_base_url = "http://localhost:11434"

[ingestion]
chunk_size = 256
chunk_overlap = 32
supported_formats = ["md", "txt"]
"""
    config_file = tmp_path / "config.toml"
    config_file.write_text(toml_content)

    config = load_system_config(config_file)
    assert config.environment.mode == "local"
    assert config.ingestion.chunk_size == 256


def test_load_agents_config_from_toml(tmp_path):
    toml_content = """\
[router]
enabled = true
model = "llama3.2:3b"
prompt = "Route this query."
routes = ["chat", "rag", "tool"]

[chat_agent]
enabled = true
model = "llama3.2:3b"
system_prompt = "Be helpful."
max_tokens = 1024

[retrieval]
enabled = true
top_k = 5
score_threshold = 0.6
rerank = false
default_collection = "test-docs"

[answer_generation]
enabled = true
model = "llama3.1:8b"
prompt_template = "Answer: {query} Evidence: {evidence}"
max_tokens = 512

[verifier]
enabled = true
model = "llama3.1:8b"
score_threshold = 0.8
citation_coverage_min = 0.7
max_retries = 1
checks = ["score_threshold", "support_analysis"]

[tool_agent]
enabled = true
model = "llama3.2:3b"
allowed_tools = ["search_collection"]
max_tool_calls = 3
"""
    agents_file = tmp_path / "agents.toml"
    agents_file.write_text(toml_content)

    config = load_agents_config(agents_file)
    assert config.retrieval.top_k == 5
    assert config.verifier.max_retries == 1
    assert config.tool_agent.allowed_tools == ["search_collection"]
