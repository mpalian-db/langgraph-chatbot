# Multi-Agent RAG Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first multi-agent RAG chat application with LangGraph, grounded verification, TOML-configured agents, and tool calling. Ollama is the default local model runtime; Anthropic API is the alternative configurable via TOML.

**Architecture:** Hexagonal Architecture -- FastAPI inbound adapters invoke a fixed LangGraph graph; all I/O (LLM, vector DB, embeddings, file storage) crosses port protocols injected at startup. Config drives node behaviour, not graph topology.

**Tech Stack:** Python 3.12+, uv, LangGraph >=0.2, Ollama (local LLM runtime, default) / Anthropic API (alternative), Qdrant (Docker), Ollama embeddings (`nomic-embed-text`, default) / Cloudflare Workers AI (alternative), Langfuse (observability, self-hosted), n8n (automation sidecar), Tailscale (private dev access), FastAPI, pytest + pytest-asyncio, React + TypeScript + Vite + Tailwind CSS, justfile, Docker Compose.

---

## File Map

### Backend

| File | Responsibility |
|---|---|
| `backend/app/core/models/types.py` | Domain value types: Chunk, Citation, ToolCall, TraceEntry, VerifierResult |
| `backend/app/core/graph/state.py` | GraphState dataclass |
| `backend/app/core/graph/graph.py` | LangGraph graph construction |
| `backend/app/core/graph/nodes/router.py` | Router node |
| `backend/app/core/graph/nodes/chat_agent.py` | Chat agent node |
| `backend/app/core/graph/nodes/retrieval.py` | Retrieval node |
| `backend/app/core/graph/nodes/answer_generation.py` | Answer generation node |
| `backend/app/core/graph/nodes/verifier.py` | Verification node |
| `backend/app/core/graph/nodes/tool_agent.py` | Tool agent node |
| `backend/app/core/config/models.py` | Pydantic config models for TOML |
| `backend/app/core/config/loader.py` | TOML file loader |
| `backend/app/ports/llm.py` | LLMPort Protocol |
| `backend/app/ports/vectorstore.py` | VectorStorePort + CollectionPort Protocols |
| `backend/app/ports/embedding.py` | EmbeddingPort Protocol |
| `backend/app/ports/storage.py` | DocumentStoragePort Protocol |
| `backend/app/adapters/llm/ollama.py` | OllamaLLMAdapter (default) |
| `backend/app/adapters/llm/anthropic.py` | AnthropicLLMAdapter (alternative provider) |
| `backend/app/adapters/embeddings/ollama.py` | OllamaEmbeddingAdapter (default) |
| `backend/app/adapters/vectorstore/qdrant.py` | QdrantVectorStoreAdapter |
| `backend/app/adapters/embeddings/workers_ai.py` | WorkersAIEmbeddingAdapter |
| `backend/app/adapters/storage/local.py` | LocalFileStorageAdapter |
| `backend/app/tools/definitions.py` | Tool function definitions for tool-using agents |
| `backend/app/ingestion/chunker.py` | Markdown/text chunker |
| `backend/app/ingestion/pipeline.py` | Ingestion pipeline: chunk, embed, upsert |
| `backend/app/api/dependencies.py` | FastAPI dependency injection setup |
| `backend/app/api/routes/chat.py` | POST /api/chat, POST /api/chat/stream |
| `backend/app/api/routes/collections.py` | Collection CRUD |
| `backend/app/api/routes/documents.py` | Document CRUD + upload |
| `backend/app/api/routes/system.py` | /api/system/health, /api/system/config |
| `backend/app/main.py` | FastAPI app entry point |

### Tests

| File | Responsibility |
|---|---|
| `backend/tests/conftest.py` | Shared fixtures (mock ports, config) |
| `backend/tests/unit/core/test_types.py` | Domain type construction |
| `backend/tests/unit/core/test_config.py` | TOML loading, Pydantic validation |
| `backend/tests/unit/core/test_nodes.py` | Node functions with mocked ports |
| `backend/tests/unit/core/test_verifier.py` | Verifier logic in isolation |
| `backend/tests/unit/core/test_adapters.py` | Adapter unit tests with mocked clients |
| `backend/tests/unit/ingestion/test_chunking.py` | Chunking strategies |
| `backend/tests/integration/test_api.py` | FastAPI routes via httpx |
| `backend/tests/integration/test_qdrant_adapter.py` | Against real Qdrant |

### Frontend

| File | Responsibility |
|---|---|
| `frontend/src/api/client.ts` | fetch-based API client |
| `frontend/src/api/types.ts` | API response types |
| `frontend/src/components/ChatView.tsx` | Chat UI |
| `frontend/src/components/CollectionsView.tsx` | Collections management UI |
| `frontend/src/components/TraceView.tsx` | Execution trace display |
| `frontend/src/hooks/useChat.ts` | Chat state management hook |
| `frontend/src/hooks/useCollections.ts` | Collections state management hook |
| `frontend/src/App.tsx` | Root component with routing |

### Infrastructure

| File | Responsibility |
|---|---|
| `config/config.toml` | System-wide config |
| `config/agents.toml` | Per-agent config |
| `docker-compose.yml` | Qdrant + n8n + Langfuse (Postgres + server) for local dev |
| `n8n/workflows/` | Exported n8n workflow JSON files (version-controlled) |
| `justfile` | Task runner |
| `.env.example` | Environment variable template |
| `.github/workflows/ci.yml` | Lint + unit tests + frontend build |

---

## Phase 1: Project Scaffold

### Task 1: Python project and directory structure

**Files:**
- Create: `backend/pyproject.toml`
- Create: all `__init__.py` files
- Create: `.env.example`, `.gitignore`

- [ ] **Step 1: Initialise the Python project with uv**

```bash
mkdir -p backend
cd backend && uv init --no-workspace
```

- [ ] **Step 2: Replace the generated pyproject.toml**

```toml
[project]
name = "langgraph-chatbot-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",          # alternative LLM provider
    "ollama>=0.3.0",               # default local LLM + embedding runtime
    "langfuse>=2.0.0",             # observability and tracing
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "langgraph>=0.2.50",
    "qdrant-client>=1.12.0",
    "pydantic>=2.10.0",
    "httpx>=0.28.0",
    "python-multipart>=0.0.20",
    "aiofiles>=24.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.0.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: mark test as integration test (requires Qdrant + Ollama running locally)",
]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

- [ ] **Step 3: Install dependencies**

Run: `cd backend && uv sync --extra dev`

Expected: creates `backend/.venv/` and installs all packages.

- [ ] **Step 4: Create the directory structure**

```bash
mkdir -p backend/app/{api/routes,core/{graph/nodes,agents,config,models},ports,adapters/{llm,vectorstore,embeddings,storage},tools,ingestion}
mkdir -p backend/tests/{unit/{core,ingestion},integration}
mkdir -p config corpus/langgraph-docs
```

Create `__init__.py` in every Python package directory:

```bash
find backend/app backend/tests -type d -exec touch {}/__init__.py \;
```

- [ ] **Step 4b: Populate corpus/langgraph-docs/ with sample documents**

The RAG pipeline needs real documents to be useful during development and integration testing. LangGraph's own documentation is the intended sample dataset -- it is factual, version-controlled, and dense enough for grounding verification to be meaningful.

Fetch a useful subset of the LangGraph documentation from the GitHub repository:

```bash
curl -L https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/low_level.md \
     -o corpus/langgraph-docs/low_level.md

curl -L https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/high_level.md \
     -o corpus/langgraph-docs/high_level.md

curl -L https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/streaming.md \
     -o corpus/langgraph-docs/streaming.md

curl -L https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/persistence.md \
     -o corpus/langgraph-docs/persistence.md

curl -L https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/agentic_concepts.md \
     -o corpus/langgraph-docs/agentic_concepts.md
```

These five files (~30--50 KB total) are enough to test the full RAG and verification pipeline. Add more as needed later. The corpus is gitignored by default -- add `corpus/langgraph-docs/*.md` explicitly if you want it version-controlled.

- [ ] **Step 5: Create .env.example**

```
# Ollama (default LLM + embedding runtime -- runs locally, no key needed)
OLLAMA_BASE_URL=http://localhost:11434

# Anthropic API (alternative LLM provider -- set to use instead of Ollama)
ANTHROPIC_API_KEY=your-key-here

# Langfuse (self-hosted observability -- keys pre-configured for local Docker setup)
LANGFUSE_PUBLIC_KEY=pk-lf-local-dev
LANGFUSE_SECRET_KEY=sk-lf-local-dev
LANGFUSE_HOST=http://localhost:3000
# Required for the Langfuse Docker container:
LANGFUSE_NEXTAUTH_SECRET=change-this-32-char-minimum-string
LANGFUSE_SALT=change-this-32-char-minimum-string
LANGFUSE_ADMIN_PASSWORD=changeme

# Cloudflare Workers AI (alternative embedding provider -- for Cloudflare deployment)
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_API_TOKEN=your-api-token

# n8n
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=change-this-immediately
# Generate with: python -c "import secrets; print(secrets.token_hex(24))"
N8N_ENCRYPTION_KEY=replace-with-48-char-hex-string
N8N_BASE_URL=http://localhost:5678
```

- [ ] **Step 6: Update .gitignore**

```
.venv/
__pycache__/
*.pyc
.env
.python-version
dist/
node_modules/
.DS_Store
qdrant_data/
n8n_data/
uploads/
```

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/app/ backend/tests/ .env.example .gitignore config/ corpus/
git commit -m "chore: scaffold Python project with uv and directory structure"
```

---

### Task 2: Justfile and Docker Compose

**Files:**
- Create: `justfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Write the justfile**

```just
set dotenv-load := true

# Start Qdrant + backend + frontend for local dev
dev:
    docker compose up qdrant -d
    cd backend && uv run uvicorn app.main:app --reload --port 8000 &
    cd frontend && npm run dev

# Run unit tests only
test:
    cd backend && uv run pytest tests/unit -v

# Run all tests including integration
test-all:
    cd backend && uv run pytest tests/ -v

# Run a single test file
test-one FILE:
    cd backend && uv run pytest {{FILE}} -v

# Lint (no auto-fix)
lint:
    cd backend && uv run ruff check . && uv run ruff format --check .

# Auto-format
format:
    cd backend && uv run ruff format . && uv run ruff check --fix .

# Stop local services
stop:
    docker compose down
```

- [ ] **Step 2: Write docker-compose.yml**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

- [ ] **Step 3: Verify Qdrant starts**

Run: `docker compose up qdrant -d && sleep 3 && curl -s http://localhost:6333/healthz`

Expected: `{"title":"qdrant - vectorass engine","version":"..."}` or similar healthy response.

- [ ] **Step 4: Commit**

```bash
git add justfile docker-compose.yml
git commit -m "chore: add justfile and docker-compose for local dev"
```

---

## Phase 2: Domain Layer

### Task 3: Domain value types

**Files:**
- Create: `backend/app/core/models/types.py`
- Create: `backend/tests/unit/core/test_types.py`

- [ ] **Step 1: Write failing tests for domain types**

```python
# backend/tests/unit/core/test_types.py
from app.core.models.types import Chunk, Citation, ToolCall, TraceEntry, VerifierResult


def test_chunk_defaults():
    chunk = Chunk(id="c1", text="hello", collection="docs")
    assert chunk.score == 0.0
    assert chunk.metadata == {}


def test_verifier_result_fields():
    result = VerifierResult(outcome="accept", score=0.9, reason="well grounded")
    assert result.unsupported_claims == []


def test_tool_call_result_defaults():
    call = ToolCall(name="search", arguments={"query": "test"})
    assert call.result is None
```

- [ ] **Step 2: Run test -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_types.py -v`

Expected: `ModuleNotFoundError: No module named 'app.core.models.types'`

- [ ] **Step 3: Implement domain types**

```python
# backend/app/core/models/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VerifierOutcome = Literal["accept", "revise", "refuse"]


@dataclass
class Chunk:
    id: str
    text: str
    collection: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Citation:
    chunk_id: str
    text: str
    collection: str


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: str | None = None


@dataclass
class TraceEntry:
    node: str
    duration_ms: float
    data: dict = field(default_factory=dict)


@dataclass
class VerifierResult:
    outcome: VerifierOutcome
    score: float
    reason: str
    unsupported_claims: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run test -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_types.py -v`

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/models/types.py backend/tests/unit/core/test_types.py
git commit -m "feat: add domain value types (Chunk, Citation, ToolCall, TraceEntry, VerifierResult)"
```

---

### Task 4: GraphState

**Files:**
- Create: `backend/app/core/graph/state.py`
- Modify: `backend/tests/unit/core/test_types.py`

- [ ] **Step 1: Write failing tests for GraphState**

Add to `backend/tests/unit/core/test_types.py`:

```python
from app.core.graph.state import GraphState


def test_graph_state_defaults():
    state = GraphState(query="what is langgraph?")
    assert state.route is None
    assert state.retrieved_chunks == []
    assert state.retry_count == 0
    assert state.final_answer is None


def test_graph_state_with_route():
    state = GraphState(query="hello", route="chat")
    assert state.route == "chat"
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_types.py -v -k "graph_state"`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement GraphState**

```python
# backend/app/core/graph/state.py
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.models.types import Chunk, Citation, ToolCall, TraceEntry, VerifierResult


@dataclass
class GraphState:
    query: str
    route: str | None = None
    retrieval_query: str | None = None
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    metadata_filters: dict = field(default_factory=dict)
    draft_answer: str | None = None
    verifier_result: VerifierResult | None = None
    final_answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    execution_trace: list[TraceEntry] = field(default_factory=list)
    retry_count: int = 0
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_types.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/graph/state.py backend/tests/unit/core/test_types.py
git commit -m "feat: add GraphState dataclass"
```

---

### Task 5: Port protocols

**Files:**
- Create: `backend/app/ports/llm.py`
- Create: `backend/app/ports/vectorstore.py`
- Create: `backend/app/ports/embedding.py`
- Create: `backend/app/ports/storage.py`

Ports are Python Protocols -- no implementation logic, just contracts.

- [ ] **Step 1: Write LLMPort**

```python
# backend/app/ports/llm.py
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]: ...
```

- [ ] **Step 2: Write VectorStorePort and CollectionPort**

```python
# backend/app/ports/vectorstore.py
from typing import Protocol, runtime_checkable

from app.core.models.types import Chunk


@runtime_checkable
class VectorStorePort(Protocol):
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        collection: str,
        filters: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[Chunk]: ...

    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None: ...

    async def delete(self, collection: str, ids: list[str]) -> None: ...


@runtime_checkable
class CollectionPort(Protocol):
    async def create(self, name: str, vector_size: int) -> None: ...
    async def list_collections(self) -> list[str]: ...
    async def delete(self, name: str) -> None: ...
    async def get_stats(self, name: str) -> dict: ...
```

- [ ] **Step 3: Write EmbeddingPort**

```python
# backend/app/ports/embedding.py
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingPort(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

- [ ] **Step 4: Write DocumentStoragePort**

```python
# backend/app/ports/storage.py
from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentStoragePort(Protocol):
    async def store(self, filename: str, content: bytes) -> str: ...
    async def retrieve(self, path: str) -> bytes: ...
    async def delete(self, path: str) -> None: ...
```

- [ ] **Step 5: Verify imports compile**

Run: `cd backend && uv run python -c "from app.ports.llm import LLMPort; from app.ports.vectorstore import VectorStorePort, CollectionPort; from app.ports.embedding import EmbeddingPort; from app.ports.storage import DocumentStoragePort; print('all ports import ok')"`

Expected: `all ports import ok`

- [ ] **Step 6: Commit**

```bash
git add backend/app/ports/
git commit -m "feat: add port protocols (LLMPort, VectorStorePort, CollectionPort, EmbeddingPort, DocumentStoragePort)"
```

---

## Phase 3: TOML Configuration

### Task 6: Pydantic config models

**Files:**
- Create: `backend/app/core/config/models.py`
- Create: `backend/tests/unit/core/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/unit/core/test_config.py
import pytest

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
    with pytest.raises(Exception):
        SystemConfig.model_validate({"environment": {"mode": "local"}, "unknown_key": "bad"})


def test_verifier_checks_defaults():
    config = AgentsConfig()
    assert "score_threshold" in config.verifier.checks
    assert "support_analysis" in config.verifier.checks
    assert "citation_coverage" in config.verifier.checks
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_config.py -v`

- [ ] **Step 3: Implement config models**

```python
# backend/app/core/config/models.py
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvironmentConfig(_StrictModel):
    mode: Literal["local", "cloudflare"] = "local"
    log_level: str = "info"


class LLMConfig(_StrictModel):
    provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"


class TracingConfig(_StrictModel):
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_project: str = "langgraph-chatbot"


class VectorStoreConfig(_StrictModel):
    provider: Literal["qdrant", "vectorize"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"


class EmbeddingsConfig(_StrictModel):
    provider: Literal["ollama", "workers-ai"] = "ollama"
    # Ollama settings (default for local dev)
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    # Workers AI settings (for Cloudflare deployment)
    workers_ai_model: str = "@cf/baai/bge-small-en-v1.5"
    workers_ai_base_url: str = "https://api.cloudflare.com/client/v4/accounts"


class IngestionConfig(_StrictModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    supported_formats: list[str] = ["md", "txt", "pdf"]


class SystemConfig(_StrictModel):
    environment: EnvironmentConfig = EnvironmentConfig()
    llm: LLMConfig = LLMConfig()
    tracing: TracingConfig = TracingConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    ingestion: IngestionConfig = IngestionConfig()


class RouterConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.2:3b"   # fast; handles classification well
    prompt: str = ""
    routes: list[str] = ["chat", "rag", "tool"]


class ChatAgentConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.2:3b"
    system_prompt: str = "You are a helpful assistant. Answer clearly and concisely."
    max_tokens: int = 2048


class RetrievalConfig(_StrictModel):
    enabled: bool = True
    top_k: int = 10
    score_threshold: float = 0.7
    rerank: bool = True
    default_collection: str = "langgraph-docs"


class AnswerGenerationConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.1:8b"   # more capable; handles evidence synthesis better
    prompt_template: str = (
        "Answer the user's question using only the evidence provided below. "
        "Cite chunk IDs inline where you use them.\n\nEvidence:\n{evidence}\n\nQuestion: {query}"
    )
    max_tokens: int = 2048


class VerifierConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.1:8b"   # same reasoning requirement as answer generation
    score_threshold: float = 0.75
    citation_coverage_min: float = 0.8
    max_retries: int = 2
    checks: list[str] = ["score_threshold", "support_analysis", "citation_coverage"]


class ToolAgentConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.2:3b"
    allowed_tools: list[str] = []
    max_tool_calls: int = 5


class AgentsConfig(_StrictModel):
    router: RouterConfig = RouterConfig()
    chat_agent: ChatAgentConfig = ChatAgentConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    answer_generation: AnswerGenerationConfig = AnswerGenerationConfig()
    verifier: VerifierConfig = VerifierConfig()
    tool_agent: ToolAgentConfig = ToolAgentConfig()
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_config.py -v`

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/config/models.py backend/tests/unit/core/test_config.py
git commit -m "feat: add Pydantic config models for SystemConfig and AgentsConfig"
```

---

### Task 7: TOML loader and config files

**Files:**
- Create: `backend/app/core/config/loader.py`
- Create: `config/config.toml`
- Create: `config/agents.toml`
- Modify: `backend/tests/unit/core/test_config.py`

- [ ] **Step 1: Write failing tests for the loader**

Add to `backend/tests/unit/core/test_config.py`:

```python
from app.core.config.loader import load_agents_config, load_system_config


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
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_config.py -v -k "load"`

- [ ] **Step 3: Implement the TOML loader**

```python
# backend/app/core/config/loader.py
import pathlib
import tomllib

from app.core.config.models import AgentsConfig, SystemConfig


def load_system_config(path: pathlib.Path) -> SystemConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return SystemConfig.model_validate(data)


def load_agents_config(path: pathlib.Path) -> AgentsConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return AgentsConfig.model_validate(data)
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_config.py -v`

Expected: all 6 tests PASS.

- [ ] **Step 5: Write config/config.toml**

```toml
[environment]
mode = "local"
log_level = "debug"

[llm]
provider = "ollama"                     # "ollama" (default) or "anthropic"
ollama_base_url = "http://localhost:11434"

[tracing]
langfuse_enabled = true
langfuse_host = "http://localhost:3000" # self-hosted; or "https://cloud.langfuse.com"
langfuse_project = "langgraph-chatbot"

[vectorstore]
provider = "qdrant"
qdrant_url = "http://localhost:6333"

[embeddings]
provider = "ollama"                     # "ollama" (default) or "workers-ai"
ollama_model = "nomic-embed-text"       # 768-dimension embeddings
ollama_base_url = "http://localhost:11434"
# workers-ai settings (uncomment for Cloudflare deployment):
# workers_ai_model = "@cf/baai/bge-small-en-v1.5"
# workers_ai_base_url = "https://api.cloudflare.com/client/v4/accounts"

[ingestion]
chunk_size = 512
chunk_overlap = 64
supported_formats = ["md", "txt", "pdf"]
```

- [ ] **Step 6: Write config/agents.toml**

```toml
[router]
enabled = true
model = "llama3.2:3b"
# Anthropic alternative: model = "claude-haiku-4-5-20251001"
prompt = """You are a routing agent. Given the user's message, classify it into one of these routes:
- "chat": general conversation, greetings, meta questions about the system
- "rag": questions that likely require searching indexed knowledge
- "tool": requests to inspect collections, upload documents, or run system operations

Respond with ONLY the route name: chat, rag, or tool."""
routes = ["chat", "rag", "tool"]

[chat_agent]
enabled = true
model = "llama3.2:3b"
# Anthropic alternative: model = "claude-haiku-4-5-20251001"
system_prompt = "You are a helpful assistant. Answer the user's questions clearly and concisely."
max_tokens = 2048

[retrieval]
enabled = true
top_k = 10
score_threshold = 0.7
rerank = true
default_collection = "langgraph-docs"

[answer_generation]
enabled = true
model = "llama3.1:8b"
# Anthropic alternative: model = "claude-sonnet-4-6-20250514"
prompt_template = """Answer the user's question using ONLY the evidence below. Include inline citations using chunk IDs like [chunk-id].

Evidence:
{evidence}

Question: {query}

If the evidence does not support a complete answer, say what you can support and indicate what is missing."""
max_tokens = 2048

[verifier]
enabled = true
model = "llama3.1:8b"
# Anthropic alternative: model = "claude-sonnet-4-6-20250514"
score_threshold = 0.75
citation_coverage_min = 0.8
max_retries = 2
checks = ["score_threshold", "support_analysis", "citation_coverage"]

[tool_agent]
enabled = true
model = "llama3.2:3b"
# Anthropic alternative: model = "claude-haiku-4-5-20251001"
allowed_tools = [
    "search_collection",
    "list_collections",
    "get_collection_stats",
    "upload_document",
    "delete_document",
]
max_tool_calls = 5
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/core/config/loader.py backend/tests/unit/core/test_config.py config/
git commit -m "feat: add TOML config loader and default config files"
```

---

**CHECKPOINT 1:** At this point `just test` should pass with 11 unit tests. The domain layer, ports, and config system are complete. Nothing imports from `adapters/` yet.

---

## Phase 4: Adapters

### Task 8: LocalFileStorageAdapter

**Files:**
- Create: `backend/app/adapters/storage/local.py`
- Create: `backend/tests/unit/core/test_adapters.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/unit/core/test_adapters.py
import pytest

from app.adapters.storage.local import LocalFileStorageAdapter


@pytest.mark.asyncio
async def test_store_and_retrieve(tmp_path):
    adapter = LocalFileStorageAdapter(base_dir=tmp_path)
    path = await adapter.store("test.txt", b"hello world")
    content = await adapter.retrieve(path)
    assert content == b"hello world"


@pytest.mark.asyncio
async def test_delete_removes_file(tmp_path):
    adapter = LocalFileStorageAdapter(base_dir=tmp_path)
    path = await adapter.store("test.txt", b"data")
    await adapter.delete(path)
    with pytest.raises(FileNotFoundError):
        await adapter.retrieve(path)
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v`

- [ ] **Step 3: Implement LocalFileStorageAdapter**

```python
# backend/app/adapters/storage/local.py
from __future__ import annotations

import pathlib

import aiofiles
import aiofiles.os


class LocalFileStorageAdapter:
    def __init__(self, base_dir: pathlib.Path) -> None:
        self._base = pathlib.Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    async def store(self, filename: str, content: bytes) -> str:
        path = self._base / filename
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)
        return str(path)

    async def retrieve(self, path: str) -> bytes:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        async with aiofiles.open(p, "rb") as f:
            return await f.read()

    async def delete(self, path: str) -> None:
        await aiofiles.os.remove(path)
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/adapters/storage/local.py backend/tests/unit/core/test_adapters.py
git commit -m "feat: add LocalFileStorageAdapter"
```

---

### Task 9: OllamaLLMAdapter (default provider)

**Files:**
- Create: `backend/app/adapters/llm/ollama.py`
- Modify: `backend/tests/unit/core/test_adapters.py`

- [ ] **Step 1: Write failing tests using a mock Ollama client**

Add to `backend/tests/unit/core/test_adapters.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

from app.adapters.llm.ollama import OllamaLLMAdapter


@pytest.mark.asyncio
async def test_ollama_adapter_formats_response():
    mock_message = MagicMock()
    mock_message.content = "Hello there"
    mock_message.tool_calls = None

    mock_response = MagicMock()
    mock_response.message = mock_message
    mock_response.prompt_eval_count = 12
    mock_response.eval_count = 6

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)

    adapter = OllamaLLMAdapter(client=mock_client)
    result = await adapter.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="llama3.2:3b",
        system="Be helpful.",
        max_tokens=100,
    )

    assert result["text"] == "Hello there"
    assert result["stop_reason"] == "end_turn"
    assert result["tool_use"] == []
    assert result["usage"]["input_tokens"] == 12


@pytest.mark.asyncio
async def test_ollama_adapter_passes_tools():
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "search"
    mock_tool_call.function.arguments = {"query": "test"}

    mock_message = MagicMock()
    mock_message.content = ""
    mock_message.tool_calls = [mock_tool_call]

    mock_response = MagicMock()
    mock_response.message = mock_message
    mock_response.prompt_eval_count = 8
    mock_response.eval_count = 4

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)

    adapter = OllamaLLMAdapter(client=mock_client)
    tools = [{"name": "search", "description": "search docs", "parameters": {"type": "object"}}]
    result = await adapter.complete(
        messages=[{"role": "user", "content": "search something"}],
        model="llama3.2:3b",
        tools=tools,
        max_tokens=100,
    )

    assert result["stop_reason"] == "tool_use"
    assert result["tool_use"][0]["name"] == "search"
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v -k "ollama_adapter"`

- [ ] **Step 3: Implement OllamaLLMAdapter**

```python
# backend/app/adapters/llm/ollama.py
from __future__ import annotations

from typing import Any

from ollama import AsyncClient


class OllamaLLMAdapter:
    """LLMPort implementation backed by a local Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        client: AsyncClient | None = None,
    ) -> None:
        self._client = client or AsyncClient(host=base_url)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        ollama_messages: list[dict[str, Any]] = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(messages)

        kwargs: dict[str, Any] = {"model": model, "messages": ollama_messages}
        if tools:
            # Ollama accepts OpenAI-compatible tool schemas directly.
            kwargs["tools"] = tools

        response = await self._client.chat(**kwargs)

        text = response.message.content or ""
        tool_use: list[dict[str, Any]] = []
        if response.message.tool_calls:
            tool_use = [
                {
                    "name": tc.function.name,
                    "input": tc.function.arguments,
                    "id": f"ollama-tool-{i}",
                }
                for i, tc in enumerate(response.message.tool_calls)
            ]

        return {
            "text": text,
            "tool_use": tool_use,
            "stop_reason": "tool_use" if tool_use else "end_turn",
            "usage": {
                "input_tokens": response.prompt_eval_count or 0,
                "output_tokens": response.eval_count or 0,
            },
        }
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v -k "ollama_adapter"`

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/adapters/llm/ollama.py backend/tests/unit/core/test_adapters.py
git commit -m "feat: add OllamaLLMAdapter as default local LLM provider"
```

---

### Task 9b: AnthropicLLMAdapter (alternative provider)

**Files:**
- Create: `backend/app/adapters/llm/anthropic.py`
- Modify: `backend/tests/unit/core/test_adapters.py`

This adapter is the alternative for when `llm.provider = "anthropic"` in `config/config.toml`. Its response shape is deliberately identical to `OllamaLLMAdapter` so node functions are unaffected by the provider switch.

- [ ] **Step 1: Write failing tests using a mock Anthropic client**

Add to `backend/tests/unit/core/test_adapters.py`:

```python
from unittest.mock import AsyncMock, MagicMock

from app.adapters.llm.anthropic import AnthropicLLMAdapter


@pytest.mark.asyncio
async def test_anthropic_adapter_formats_response():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Hello there")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    adapter = AnthropicLLMAdapter(client=mock_client)
    result = await adapter.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="claude-haiku-4-5-20251001",
        system="Be helpful.",
        max_tokens=100,
    )

    assert result["text"] == "Hello there"
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 10


@pytest.mark.asyncio
async def test_anthropic_adapter_passes_tools():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="result")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=5, output_tokens=3)

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    adapter = AnthropicLLMAdapter(client=mock_client)
    tools = [{"name": "search", "description": "search docs", "input_schema": {"type": "object"}}]
    await adapter.complete(
        messages=[{"role": "user", "content": "search something"}],
        model="claude-haiku-4-5-20251001",
        tools=tools,
        max_tokens=100,
    )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tools"] == tools
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v -k "anthropic"`

- [ ] **Step 3: Implement AnthropicLLMAdapter**

```python
# backend/app/adapters/llm/anthropic.py
from __future__ import annotations

from typing import Any

import anthropic


class AnthropicLLMAdapter:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = client or anthropic.AsyncAnthropic()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_use_blocks = [
            {"name": b.name, "input": b.input, "id": b.id}
            for b in response.content
            if b.type == "tool_use"
        ]

        return {
            "text": text_blocks[0] if text_blocks else "",
            "tool_use": tool_use_blocks,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v`

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/adapters/llm/anthropic.py backend/tests/unit/core/test_adapters.py
git commit -m "feat: add AnthropicLLMAdapter"
```

---

### Task 10: WorkersAIEmbeddingAdapter

**Files:**
- Create: `backend/app/adapters/embeddings/workers_ai.py`
- Modify: `backend/tests/unit/core/test_adapters.py`

- [ ] **Step 1: Write failing test**

Add to `backend/tests/unit/core/test_adapters.py`:

```python
from app.adapters.embeddings.workers_ai import WorkersAIEmbeddingAdapter


@pytest.mark.asyncio
async def test_workers_ai_embedding_returns_vectors():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {"data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
        "success": True,
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    adapter = WorkersAIEmbeddingAdapter(
        account_id="test-account",
        api_token="test-token",
        model="@cf/baai/bge-small-en-v1.5",
        http_client=mock_client,
    )
    vectors = await adapter.embed(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert vectors[0][0] == pytest.approx(0.1)
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v -k "workers_ai"`

- [ ] **Step 3: Implement WorkersAIEmbeddingAdapter**

```python
# backend/app/adapters/embeddings/workers_ai.py
from __future__ import annotations

import httpx


class WorkersAIEmbeddingAdapter:
    def __init__(
        self,
        account_id: str,
        api_token: str,
        model: str = "@cf/baai/bge-small-en-v1.5",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._account_id = account_id
        self._api_token = api_token
        self._model = model
        self._http_client = http_client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        own_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}",
            headers={"Authorization": f"Bearer {self._api_token}"},
        )
        try:
            response = await client.post(
                f"/ai/run/{self._model}",
                json={"text": texts},
            )
            response.raise_for_status()
            data = response.json()
            return data["result"]["data"]
        finally:
            if own_client:
                await client.aclose()
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v`

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/adapters/embeddings/workers_ai.py backend/tests/unit/core/test_adapters.py
git commit -m "feat: add WorkersAIEmbeddingAdapter (alternative provider)"
```

---

### Task 10b: OllamaEmbeddingAdapter (default provider)

**Files:**
- Create: `backend/app/adapters/embeddings/ollama.py`
- Modify: `backend/tests/unit/core/test_adapters.py`

This adapter uses Ollama's embedding endpoint (`/api/embed`) with `nomic-embed-text` to produce 768-dimensional vectors. It is the default when `embeddings.provider = "ollama"` in `config/config.toml`. The response shape is identical to `WorkersAIEmbeddingAdapter` so the ingestion pipeline and retrieval node are unaffected by the provider switch.

- [ ] **Step 1: Write failing test**

Add to `backend/tests/unit/core/test_adapters.py`:

```python
from app.adapters.embeddings.ollama import OllamaEmbeddingAdapter


@pytest.mark.asyncio
async def test_ollama_embedding_returns_vectors():
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=mock_response)

    adapter = OllamaEmbeddingAdapter(
        model="nomic-embed-text",
        client=mock_client,
    )
    vectors = await adapter.embed(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert vectors[0][0] == pytest.approx(0.1)
    mock_client.embed.assert_awaited_once_with(model="nomic-embed-text", input=["hello", "world"])
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v -k "ollama_embedding"`

- [ ] **Step 3: Implement OllamaEmbeddingAdapter**

```python
# backend/app/adapters/embeddings/ollama.py
from __future__ import annotations

from ollama import AsyncClient


class OllamaEmbeddingAdapter:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        client: AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._client = client or AsyncClient(host=base_url)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embed(model=self._model, input=texts)
        return response.embeddings
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_adapters.py -v`

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/adapters/embeddings/ollama.py backend/tests/unit/core/test_adapters.py
git commit -m "feat: add OllamaEmbeddingAdapter (default provider)"
```

---

### Task 11: QdrantVectorStoreAdapter

**Files:**
- Create: `backend/app/adapters/vectorstore/qdrant.py`
- Create: `backend/tests/integration/test_qdrant_adapter.py`

This adapter is tested via integration tests against a real Qdrant instance.

- [ ] **Step 1: Implement QdrantVectorStoreAdapter**

```python
# backend/app/adapters/vectorstore/qdrant.py
from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.core.models.types import Chunk


class QdrantVectorStoreAdapter:
    def __init__(self, url: str) -> None:
        self._client = AsyncQdrantClient(url=url)

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        collection: str,
        filters: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[Chunk]:
        qdrant_filter = _build_filter(filters) if filters else None
        results = await self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [_scored_point_to_chunk(r, collection) for r in results]

    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={"text": chunk.text, "chunk_id": chunk.id, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        await self._client.upsert(collection_name=collection, points=points)

    async def delete(self, collection: str, ids: list[str]) -> None:
        await self._client.delete(collection_name=collection, points_selector=ids)

    async def create(self, name: str, vector_size: int) -> None:
        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    async def list_collections(self) -> list[str]:
        result = await self._client.get_collections()
        return [c.name for c in result.collections]

    async def delete_collection(self, name: str) -> None:
        await self._client.delete_collection(collection_name=name)

    async def get_stats(self, name: str) -> dict:
        info = await self._client.get_collection(collection_name=name)
        return {
            "name": name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
        }


def _build_filter(filters: dict) -> Filter:
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]
    return Filter(must=conditions)


def _scored_point_to_chunk(point: ScoredPoint, collection: str) -> Chunk:
    payload = point.payload or {}
    return Chunk(
        id=str(point.id),
        text=payload.get("text", ""),
        collection=collection,
        score=point.score,
        metadata={k: v for k, v in payload.items() if k not in ("text", "chunk_id")},
    )
```

- [ ] **Step 2: Write integration test**

```python
# backend/tests/integration/test_qdrant_adapter.py
import uuid

import pytest

from app.adapters.vectorstore.qdrant import QdrantVectorStoreAdapter
from app.core.models.types import Chunk

VECTOR_SIZE = 4


@pytest.fixture
async def qdrant():
    adapter = QdrantVectorStoreAdapter(url="http://localhost:6333")
    yield adapter


@pytest.fixture
async def test_collection(qdrant):
    name = f"test-{uuid.uuid4().hex[:8]}"
    await qdrant.create(name, VECTOR_SIZE)
    yield name
    await qdrant.delete_collection(name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_and_search(qdrant, test_collection):
    chunks = [
        Chunk(id=str(uuid.uuid4()), text="LangGraph builds stateful agents.", collection=test_collection),
    ]
    vectors = [[0.1, 0.2, 0.3, 0.4]]
    await qdrant.upsert(test_collection, chunks, vectors)

    results = await qdrant.search(
        query_vector=[0.1, 0.2, 0.3, 0.4],
        top_k=5,
        collection=test_collection,
    )
    assert len(results) == 1
    assert results[0].text == "LangGraph builds stateful agents."


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_collections(qdrant, test_collection):
    names = await qdrant.list_collections()
    assert test_collection in names
```

- [ ] **Step 3: Run unit tests still pass**

Run: `cd backend && uv run pytest tests/unit -v`

Expected: all unit tests PASS (integration tests skipped).

- [ ] **Step 4: Commit**

```bash
git add backend/app/adapters/vectorstore/qdrant.py backend/tests/integration/test_qdrant_adapter.py
git commit -m "feat: add QdrantVectorStoreAdapter with integration tests"
```

---

**CHECKPOINT 2:** All adapters implemented. `just test` passes ~19 unit tests (11 from CHECKPOINT 1 + 8 adapter tests). Integration tests can be run with `just test-all` when Qdrant and Ollama are running.

---

## Phase 5: Ingestion Pipeline

### Task 12: Markdown/text chunker

**Files:**
- Create: `backend/app/ingestion/chunker.py`
- Create: `backend/tests/unit/ingestion/test_chunking.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/unit/ingestion/test_chunking.py
from app.ingestion.chunker import chunk_text


def test_short_text_produces_one_chunk():
    chunks = chunk_text("Hello world", "test.md", "docs", chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world"
    assert chunks[0]["metadata"]["filename"] == "test.md"
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_long_text_produces_multiple_chunks():
    text = "A" * 1200
    chunks = chunk_text(text, "long.txt", "docs", chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2


def test_each_chunk_has_unique_id():
    text = "A" * 1200
    chunks = chunk_text(text, "test.txt", "docs", chunk_size=500, chunk_overlap=50)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_have_collection_set():
    chunks = chunk_text("some text", "f.md", "my-collection")
    assert all(c["collection"] == "my-collection" for c in chunks)
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/ingestion/test_chunking.py -v`

- [ ] **Step 3: Implement chunker**

```python
# backend/app/ingestion/chunker.py
from __future__ import annotations

import uuid


def chunk_text(
    text: str,
    filename: str,
    collection: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """Split text into overlapping character-based chunks."""
    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Avoid cutting mid-word when possible
        if end < len(text) and text[end] not in (" ", "\n"):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        segment = text[start:end].strip()
        if segment:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": segment,
                "collection": collection,
                "metadata": {
                    "filename": filename,
                    "chunk_index": len(chunks),
                },
            })

        if end >= len(text):
            break
        start = end - chunk_overlap

    return chunks
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/ingestion/test_chunking.py -v`

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/chunker.py backend/tests/unit/ingestion/test_chunking.py
git commit -m "feat: add character-based text chunker"
```

---

### Task 13: Ingestion pipeline

**Files:**
- Create: `backend/app/ingestion/pipeline.py`
- Create: `backend/tests/unit/ingestion/test_pipeline.py`

- [ ] **Step 1: Write failing test using mock ports**

```python
# backend/tests/unit/ingestion/test_pipeline.py
import pytest
from unittest.mock import AsyncMock

from app.ingestion.pipeline import ingest_document


@pytest.mark.asyncio
async def test_ingest_document_stores_chunks_and_embedds():
    mock_storage = AsyncMock()
    mock_storage.store = AsyncMock(return_value="/tmp/test.md")

    mock_embedding = AsyncMock()
    mock_embedding.embed = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])

    mock_vectorstore = AsyncMock()
    mock_vectorstore.upsert = AsyncMock()

    count = await ingest_document(
        filename="test.md",
        content=b"A " * 600,
        collection="docs",
        storage=mock_storage,
        embedding=mock_embedding,
        vectorstore=mock_vectorstore,
        chunk_size=512,
        chunk_overlap=64,
    )

    assert count >= 2
    mock_storage.store.assert_called_once()
    mock_embedding.embed.assert_called_once()
    mock_vectorstore.upsert.assert_called_once()
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/ingestion/test_pipeline.py -v`

- [ ] **Step 3: Implement ingestion pipeline**

```python
# backend/app/ingestion/pipeline.py
from __future__ import annotations

from app.core.models.types import Chunk
from app.ingestion.chunker import chunk_text
from app.ports.embedding import EmbeddingPort
from app.ports.storage import DocumentStoragePort
from app.ports.vectorstore import VectorStorePort


async def ingest_document(
    filename: str,
    content: bytes,
    collection: str,
    storage: DocumentStoragePort,
    embedding: EmbeddingPort,
    vectorstore: VectorStorePort,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    """Store, chunk, embed, and upsert a document. Returns number of chunks created."""
    await storage.store(filename, content)

    text = content.decode("utf-8", errors="replace")
    raw_chunks = chunk_text(text, filename, collection, chunk_size, chunk_overlap)

    chunks = [
        Chunk(
            id=c["id"],
            text=c["text"],
            collection=collection,
            metadata=c["metadata"],
        )
        for c in raw_chunks
    ]

    texts = [c.text for c in chunks]
    vectors = await embedding.embed(texts)
    await vectorstore.upsert(collection, chunks, vectors)

    return len(chunks)
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/ingestion/ -v`

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/pipeline.py backend/tests/unit/ingestion/test_pipeline.py
git commit -m "feat: add ingestion pipeline (chunk, embed, upsert)"
```

---

**CHECKPOINT 3:** Foundation complete. `just test` passes ~24 unit tests. Domain types, ports, config, adapters, and ingestion are all in place. The next phase builds the graph nodes.

---

## Phase 6: Graph Nodes

### Task 14: Shared test fixtures

**Files:**
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: Write conftest with mock fixtures**

```python
# backend/tests/conftest.py
import pytest
from unittest.mock import AsyncMock

from app.core.config.models import AgentsConfig, SystemConfig
from app.core.models.types import Chunk


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value={
        "text": "mock response",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    })
    return llm


@pytest.fixture
def mock_vectorstore():
    vs = AsyncMock()
    vs.search = AsyncMock(return_value=[
        Chunk(
            id="chunk-1",
            text="LangGraph is a library for building stateful agents.",
            collection="test",
            score=0.9,
        ),
    ])
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()
    return vs


@pytest.fixture
def mock_collection_store():
    cs = AsyncMock()
    cs.create = AsyncMock()
    cs.list_collections = AsyncMock(return_value=["docs", "test"])
    cs.delete = AsyncMock()
    cs.get_stats = AsyncMock(return_value={"name": "docs", "vectors_count": 100, "points_count": 100})
    return cs


@pytest.fixture
def mock_embedding():
    emb = AsyncMock()
    emb.embed = AsyncMock(return_value=[[0.1] * 384])
    return emb


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.store = AsyncMock(return_value="/tmp/test.txt")
    storage.retrieve = AsyncMock(return_value=b"content")
    storage.delete = AsyncMock()
    return storage


@pytest.fixture
def agents_config():
    return AgentsConfig()


@pytest.fixture
def system_config():
    return SystemConfig()
```

- [ ] **Step 2: Run all unit tests to confirm nothing broke**

Run: `cd backend && uv run pytest tests/unit -v`

Expected: all existing tests PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/conftest.py
git commit -m "feat: add shared test fixtures (mock ports and config)"
```

---

### Task 15: Router node

**Files:**
- Create: `backend/app/core/graph/nodes/router.py`
- Create: `backend/tests/unit/core/test_nodes.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/unit/core/test_nodes.py
import pytest
from unittest.mock import AsyncMock

from app.core.config.models import RouterConfig
from app.core.graph.nodes import router
from app.core.graph.state import GraphState


@pytest.mark.asyncio
async def test_router_sets_route(mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "rag",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    })
    state = GraphState(query="What is LangGraph?")
    config = RouterConfig(prompt="Route this.", model="llama3.2:3b")

    result = await router.run(state, config=config, llm=mock_llm)

    assert result["route"] == "rag"
    assert len(result["execution_trace"]) == 1
    assert result["execution_trace"][0].node == "router"


@pytest.mark.asyncio
async def test_router_defaults_to_chat_on_unknown_route(mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "unknown_xyz",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    })
    state = GraphState(query="hello")
    config = RouterConfig()

    result = await router.run(state, config=config, llm=mock_llm)

    assert result["route"] == "chat"
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_nodes.py -v`

- [ ] **Step 3: Implement router node**

```python
# backend/app/core/graph/nodes/router.py
from __future__ import annotations

import time
from typing import Any

from app.core.config.models import RouterConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry
from app.ports.llm import LLMPort


async def run(
    state: GraphState,
    *,
    config: RouterConfig,
    llm: LLMPort,
) -> dict[str, Any]:
    start = time.monotonic()

    response = await llm.complete(
        messages=[{"role": "user", "content": state.query}],
        model=config.model,
        system=config.prompt,
        max_tokens=16,
    )

    route = response["text"].strip().lower()
    if route not in config.routes:
        route = "chat"

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "route": route,
        "execution_trace": state.execution_trace + [
            TraceEntry(node="router", duration_ms=elapsed_ms, data={"route": route})
        ],
    }
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_nodes.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/graph/nodes/router.py backend/tests/unit/core/test_nodes.py
git commit -m "feat: add router node"
```

---

### Task 16: Chat agent node

**Files:**
- Create: `backend/app/core/graph/nodes/chat_agent.py`
- Modify: `backend/tests/unit/core/test_nodes.py`

- [ ] **Step 1: Write failing test**

Add to `backend/tests/unit/core/test_nodes.py`:

```python
from app.core.config.models import ChatAgentConfig
from app.core.graph.nodes import chat_agent


@pytest.mark.asyncio
async def test_chat_agent_sets_final_answer(mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "Hello! How can I help?",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 8},
    })
    state = GraphState(query="hello", route="chat")
    config = ChatAgentConfig()

    result = await chat_agent.run(state, config=config, llm=mock_llm)

    assert result["final_answer"] == "Hello! How can I help?"
```

- [ ] **Step 2: Implement chat agent node**

```python
# backend/app/core/graph/nodes/chat_agent.py
from __future__ import annotations

import time
from typing import Any

from app.core.config.models import ChatAgentConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry
from app.ports.llm import LLMPort


async def run(
    state: GraphState,
    *,
    config: ChatAgentConfig,
    llm: LLMPort,
) -> dict[str, Any]:
    start = time.monotonic()

    response = await llm.complete(
        messages=[{"role": "user", "content": state.query}],
        model=config.model,
        system=config.system_prompt,
        max_tokens=config.max_tokens,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "final_answer": response["text"],
        "execution_trace": state.execution_trace + [
            TraceEntry(node="chat_agent", duration_ms=elapsed_ms, data={"tokens": response["usage"]})
        ],
    }
```

- [ ] **Step 3: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_nodes.py -v`

Expected: 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/app/core/graph/nodes/chat_agent.py backend/tests/unit/core/test_nodes.py
git commit -m "feat: add chat agent node"
```

---

### Task 17: Retrieval node

**Files:**
- Create: `backend/app/core/graph/nodes/retrieval.py`
- Modify: `backend/tests/unit/core/test_nodes.py`

- [ ] **Step 1: Write failing test**

Add to `backend/tests/unit/core/test_nodes.py`:

```python
from app.core.config.models import RetrievalConfig
from app.core.graph.nodes import retrieval


@pytest.mark.asyncio
async def test_retrieval_searches_vectorstore(mock_vectorstore, mock_embedding):
    state = GraphState(query="What is LangGraph?", route="rag")
    config = RetrievalConfig(top_k=5, score_threshold=0.7, default_collection="docs")

    result = await retrieval.run(
        state, config=config, vectorstore=mock_vectorstore, embedding=mock_embedding,
    )

    mock_embedding.embed.assert_called_once_with(["What is LangGraph?"])
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieval_scores"][0] == pytest.approx(0.9)
```

- [ ] **Step 2: Implement retrieval node**

```python
# backend/app/core/graph/nodes/retrieval.py
from __future__ import annotations

import time
from typing import Any

from app.core.config.models import RetrievalConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry
from app.ports.embedding import EmbeddingPort
from app.ports.vectorstore import VectorStorePort


async def run(
    state: GraphState,
    *,
    config: RetrievalConfig,
    vectorstore: VectorStorePort,
    embedding: EmbeddingPort,
) -> dict[str, Any]:
    start = time.monotonic()

    query_text = state.retrieval_query or state.query
    [query_vector] = await embedding.embed([query_text])

    collection = config.default_collection
    chunks = await vectorstore.search(
        query_vector=query_vector,
        top_k=config.top_k,
        collection=collection,
        filters=state.metadata_filters or None,
        score_threshold=config.score_threshold,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "retrieved_chunks": chunks,
        "retrieval_scores": [c.score for c in chunks],
        "execution_trace": state.execution_trace + [
            TraceEntry(
                node="retrieval",
                duration_ms=elapsed_ms,
                data={"chunks_retrieved": len(chunks), "collection": collection},
            )
        ],
    }
```

- [ ] **Step 3: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_nodes.py -v`

Expected: 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/app/core/graph/nodes/retrieval.py backend/tests/unit/core/test_nodes.py
git commit -m "feat: add retrieval node"
```

---

### Task 18: Answer generation node

**Files:**
- Create: `backend/app/core/graph/nodes/answer_generation.py`
- Modify: `backend/tests/unit/core/test_nodes.py`

- [ ] **Step 1: Write failing test**

Add to `backend/tests/unit/core/test_nodes.py`:

```python
from app.core.config.models import AnswerGenerationConfig
from app.core.graph.nodes import answer_generation
from app.core.models.types import Chunk


@pytest.mark.asyncio
async def test_answer_generation_produces_draft(mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "LangGraph is a library [chunk-1] for building stateful agents.",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 20},
    })
    state = GraphState(
        query="What is LangGraph?",
        route="rag",
        retrieved_chunks=[
            Chunk(id="chunk-1", text="LangGraph builds stateful agents.", collection="docs", score=0.9),
        ],
        retrieval_scores=[0.9],
    )
    config = AnswerGenerationConfig()

    result = await answer_generation.run(state, config=config, llm=mock_llm)

    assert "LangGraph" in result["draft_answer"]
    assert len(result["execution_trace"]) == 1
```

- [ ] **Step 2: Implement answer generation node**

```python
# backend/app/core/graph/nodes/answer_generation.py
from __future__ import annotations

import re
import time
from typing import Any

from app.core.config.models import AnswerGenerationConfig
from app.core.graph.state import GraphState
from app.core.models.types import Citation, TraceEntry
from app.ports.llm import LLMPort


async def run(
    state: GraphState,
    *,
    config: AnswerGenerationConfig,
    llm: LLMPort,
) -> dict[str, Any]:
    start = time.monotonic()

    evidence = "\n\n".join(
        f"[{chunk.id}] {chunk.text}" for chunk in state.retrieved_chunks
    )
    prompt = config.prompt_template.format(evidence=evidence, query=state.query)

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

    # If revising, include previous verifier feedback
    if state.verifier_result and state.verifier_result.outcome == "revise":
        messages.append({"role": "assistant", "content": state.draft_answer or ""})
        messages.append({
            "role": "user",
            "content": (
                f"Please revise. Issues: {state.verifier_result.reason}. "
                f"Unsupported claims: {', '.join(state.verifier_result.unsupported_claims)}"
            ),
        })

    response = await llm.complete(
        messages=messages,
        model=config.model,
        max_tokens=config.max_tokens,
    )

    draft = response["text"]
    citations = _extract_citations(draft, state.retrieved_chunks)

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "draft_answer": draft,
        "citations": citations,
        "execution_trace": state.execution_trace + [
            TraceEntry(
                node="answer_generation",
                duration_ms=elapsed_ms,
                data={"tokens": response["usage"]},
            )
        ],
    }


def _extract_citations(text: str, chunks: list) -> list[Citation]:
    """Extract citations for chunk IDs referenced in square brackets."""
    cited_ids = set(re.findall(r"\[([a-f0-9\-]{36})\]", text))
    return [
        Citation(chunk_id=c.id, text=c.text[:200], collection=c.collection)
        for c in chunks
        if c.id in cited_ids
    ]
```

- [ ] **Step 3: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_nodes.py -v`

Expected: 5 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/app/core/graph/nodes/answer_generation.py backend/tests/unit/core/test_nodes.py
git commit -m "feat: add answer generation node with citation extraction"
```

---

### Task 19: Verifier node

**Files:**
- Create: `backend/app/core/graph/nodes/verifier.py`
- Create: `backend/tests/unit/core/test_verifier.py`

This is the most complex node and gets the most thorough tests.

- [ ] **Step 1: Write comprehensive failing tests**

```python
# backend/tests/unit/core/test_verifier.py
import pytest
from unittest.mock import AsyncMock

from app.core.config.models import VerifierConfig
from app.core.graph.nodes.verifier import _parse_verifier_response, run
from app.core.graph.state import GraphState
from app.core.models.types import Chunk


@pytest.fixture
def rag_state():
    return GraphState(
        query="What is LangGraph?",
        retrieved_chunks=[
            Chunk(
                id="abc-123",
                text="LangGraph is a library for building stateful agents.",
                collection="docs",
                score=0.85,
            ),
        ],
        retrieval_scores=[0.85],
        draft_answer="LangGraph is a library [abc-123] for building stateful agents.",
    )


def test_parse_verifier_response_accept():
    text = "OUTCOME: accept\nSCORE: 0.9\nREASON: Well supported.\nUNSUPPORTED: NONE"
    result = _parse_verifier_response(text)
    assert result.outcome == "accept"
    assert result.score == pytest.approx(0.9)
    assert result.unsupported_claims == []


def test_parse_verifier_response_revise():
    text = "OUTCOME: revise\nSCORE: 0.5\nREASON: Some claims unsupported.\nUNSUPPORTED: claim A, claim B"
    result = _parse_verifier_response(text)
    assert result.outcome == "revise"
    assert result.unsupported_claims == ["claim A", "claim B"]


def test_parse_verifier_response_garbage_defaults_to_refuse():
    result = _parse_verifier_response("garbled output with no structure")
    assert result.outcome == "refuse"


@pytest.mark.asyncio
async def test_verifier_refuses_below_score_threshold():
    low_score_state = GraphState(
        query="What is X?",
        retrieved_chunks=[Chunk(id="c1", text="unrelated", collection="docs", score=0.3)],
        retrieval_scores=[0.3],
        draft_answer="Some answer.",
    )
    config = VerifierConfig(score_threshold=0.7, checks=["score_threshold"])
    mock_llm = AsyncMock()

    result = await run(low_score_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "refuse"
    assert "final_answer" in result
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_accepts_when_llm_says_accept(rag_state, mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "OUTCOME: accept\nSCORE: 0.9\nREASON: Well supported.\nUNSUPPORTED: NONE",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 30},
    })
    config = VerifierConfig(score_threshold=0.5, checks=["score_threshold", "support_analysis"])

    result = await run(rag_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "accept"
    assert result["final_answer"] == rag_state.draft_answer


@pytest.mark.asyncio
async def test_verifier_increments_retry_on_revise(rag_state, mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "OUTCOME: revise\nSCORE: 0.6\nREASON: Missing citation.\nUNSUPPORTED: LangGraph is stateful",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 30},
    })
    config = VerifierConfig(score_threshold=0.5, checks=["support_analysis"], max_retries=2)

    result = await run(rag_state, config=config, llm=mock_llm)

    assert result["verifier_result"].outcome == "revise"
    assert result["retry_count"] == 1
    assert "final_answer" not in result


@pytest.mark.asyncio
async def test_verifier_refuses_when_retries_exhausted(rag_state, mock_llm):
    mock_llm.complete = AsyncMock(return_value={
        "text": "OUTCOME: revise\nSCORE: 0.6\nREASON: Still unsupported.\nUNSUPPORTED: some claim",
        "tool_use": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 30},
    })
    rag_state.retry_count = 2  # already at max
    config = VerifierConfig(score_threshold=0.5, checks=["support_analysis"], max_retries=2)

    result = await run(rag_state, config=config, llm=mock_llm)

    assert "final_answer" in result
    assert "cannot provide" in result["final_answer"].lower() or "cannot" in result["final_answer"].lower()
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `cd backend && uv run pytest tests/unit/core/test_verifier.py -v`

- [ ] **Step 3: Implement verifier node**

```python
# backend/app/core/graph/nodes/verifier.py
from __future__ import annotations

import re
import time
from typing import Any

from app.core.config.models import VerifierConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry, VerifierResult
from app.ports.llm import LLMPort


async def run(
    state: GraphState,
    *,
    config: VerifierConfig,
    llm: LLMPort,
) -> dict[str, Any]:
    start = time.monotonic()

    # Check 1: retrieval score threshold
    if "score_threshold" in config.checks and state.retrieval_scores:
        max_score = max(state.retrieval_scores)
        if max_score < config.score_threshold:
            result = VerifierResult(
                outcome="refuse",
                score=max_score,
                reason=(
                    f"Max retrieval score {max_score:.2f} "
                    f"below threshold {config.score_threshold}"
                ),
            )
            return _build_return(state, result, start, refuse=True)

    # Check 2: LLM-based support analysis
    if "support_analysis" in config.checks:
        evidence = "\n\n".join(
            f"[{c.id}] {c.text}" for c in state.retrieved_chunks
        )
        prompt = (
            "You are a grounding verifier. Determine if the answer "
            "is supported by the evidence.\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Answer to verify:\n{state.draft_answer}\n\n"
            "Respond in this EXACT format:\n"
            "OUTCOME: accept|revise|refuse\n"
            "SCORE: 0.0-1.0\n"
            "REASON: one sentence\n"
            "UNSUPPORTED: comma-separated list of unsupported claims, or NONE"
        )
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            model=config.model,
            max_tokens=256,
        )
        result = _parse_verifier_response(response["text"])
    else:
        result = VerifierResult(outcome="accept", score=1.0, reason="checks skipped")

    # Decide outcome
    if result.outcome == "accept":
        return _build_return(state, result, start, accept=True)
    elif result.outcome == "revise" and state.retry_count < config.max_retries:
        return _build_return(state, result, start, revise=True)
    else:
        return _build_return(state, result, start, refuse=True)


def _build_return(
    state: GraphState,
    result: VerifierResult,
    start: float,
    *,
    accept: bool = False,
    revise: bool = False,
    refuse: bool = False,
) -> dict[str, Any]:
    elapsed_ms = (time.monotonic() - start) * 1000
    trace = TraceEntry(
        node="verifier",
        duration_ms=elapsed_ms,
        data={"outcome": result.outcome, "score": result.score},
    )
    update: dict[str, Any] = {
        "verifier_result": result,
        "execution_trace": state.execution_trace + [trace],
    }
    if accept:
        update["final_answer"] = state.draft_answer
    elif revise:
        update["retry_count"] = state.retry_count + 1
    elif refuse:
        update["final_answer"] = (
            f"I cannot provide a fully supported answer. {result.reason}"
        )
    return update


def _parse_verifier_response(text: str) -> VerifierResult:
    outcome_match = re.search(r"OUTCOME:\s*(accept|revise|refuse)", text, re.IGNORECASE)
    score_match = re.search(r"SCORE:\s*([\d.]+)", text)
    reason_match = re.search(r"REASON:\s*(.+)", text)
    unsupported_match = re.search(r"UNSUPPORTED:\s*(.+)", text)

    outcome = outcome_match.group(1).lower() if outcome_match else "refuse"
    score = float(score_match.group(1)) if score_match else 0.0
    reason = reason_match.group(1).strip() if reason_match else "Unable to verify"
    unsupported_raw = unsupported_match.group(1).strip() if unsupported_match else "NONE"
    unsupported = (
        [] if unsupported_raw.upper() == "NONE"
        else [c.strip() for c in unsupported_raw.split(",")]
    )

    return VerifierResult(
        outcome=outcome,  # type: ignore[arg-type]
        score=score,
        reason=reason,
        unsupported_claims=unsupported,
    )
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_verifier.py -v`

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/graph/nodes/verifier.py backend/tests/unit/core/test_verifier.py
git commit -m "feat: add verifier node with score threshold and LLM support analysis"
```

---

### Task 20: Tool definitions and tool agent node

**Files:**
- Create: `backend/app/tools/definitions.py`
- Create: `backend/app/core/graph/nodes/tool_agent.py`
- Modify: `backend/tests/unit/core/test_nodes.py`

- [ ] **Step 1: Write failing test**

Add to `backend/tests/unit/core/test_nodes.py`:

```python
from app.core.config.models import ToolAgentConfig
from app.core.graph.nodes import tool_agent


@pytest.mark.asyncio
async def test_tool_agent_calls_list_collections(
    mock_llm, mock_vectorstore, mock_collection_store, mock_embedding,
):
    # First call: LLM requests tool use. Second call: LLM produces final text.
    mock_llm.complete = AsyncMock(side_effect=[
        {
            "text": "",
            "tool_use": [{"name": "list_collections", "input": {}, "id": "tool_1"}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {
            "text": "Available collections: docs, test",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        },
    ])
    state = GraphState(query="list collections", route="tool")
    config = ToolAgentConfig(allowed_tools=["list_collections"], max_tool_calls=3)

    result = await tool_agent.run(
        state,
        config=config,
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    assert result["final_answer"] == "Available collections: docs, test"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].name == "list_collections"
```

- [ ] **Step 2: Implement tool definitions**

```python
# backend/app/tools/definitions.py
from __future__ import annotations

TOOL_REGISTRY: dict[str, dict] = {
    "search_collection": {
        "name": "search_collection",
        "description": "Search for documents in a collection by query string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "collection": {"type": "string", "description": "Collection name"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query", "collection"],
        },
    },
    "list_collections": {
        "name": "list_collections",
        "description": "List all available document collections.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_collection_stats": {
        "name": "get_collection_stats",
        "description": "Get statistics for a named collection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["collection"],
        },
    },
}


def get_tools_for_agent(allowed_tools: list[str]) -> list[dict]:
    return [TOOL_REGISTRY[name] for name in allowed_tools if name in TOOL_REGISTRY]
```

- [ ] **Step 3: Implement tool agent node**

```python
# backend/app/core/graph/nodes/tool_agent.py
from __future__ import annotations

import time
from typing import Any

from app.core.config.models import ToolAgentConfig
from app.core.graph.state import GraphState
from app.core.models.types import ToolCall, TraceEntry
from app.ports.embedding import EmbeddingPort
from app.ports.llm import LLMPort
from app.ports.vectorstore import CollectionPort, VectorStorePort
from app.tools.definitions import get_tools_for_agent


async def run(
    state: GraphState,
    *,
    config: ToolAgentConfig,
    llm: LLMPort,
    vectorstore: VectorStorePort,
    collection_store: CollectionPort,
    embedding: EmbeddingPort,
) -> dict[str, Any]:
    start = time.monotonic()

    tools = get_tools_for_agent(config.allowed_tools)
    messages: list[dict] = [{"role": "user", "content": state.query}]
    tool_calls_made: list[ToolCall] = []
    final_text = ""

    for _ in range(config.max_tool_calls):
        response = await llm.complete(
            messages=messages,
            model=config.model,
            tools=tools,
            max_tokens=1024,
        )

        if response["stop_reason"] == "end_turn" or not response["tool_use"]:
            final_text = response["text"]
            break

        for tool_use in response["tool_use"]:
            tool_result = await _execute_tool(
                tool_use["name"],
                tool_use["input"],
                vectorstore=vectorstore,
                collection_store=collection_store,
                embedding=embedding,
            )
            tool_calls_made.append(
                ToolCall(name=tool_use["name"], arguments=tool_use["input"], result=str(tool_result))
            )
            messages.append({"role": "assistant", "content": response["text"] or ""})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use["id"], "content": str(tool_result)},
                ],
            })

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "final_answer": final_text,
        "tool_calls": state.tool_calls + tool_calls_made,
        "execution_trace": state.execution_trace + [
            TraceEntry(node="tool_agent", duration_ms=elapsed_ms, data={"tool_calls": len(tool_calls_made)})
        ],
    }


async def _execute_tool(
    name: str,
    args: dict,
    *,
    vectorstore: VectorStorePort,
    collection_store: CollectionPort,
    embedding: EmbeddingPort,
) -> Any:
    if name == "list_collections":
        return await collection_store.list_collections()
    if name == "get_collection_stats":
        return await collection_store.get_stats(args["collection"])
    if name == "search_collection":
        [query_vector] = await embedding.embed([args["query"]])
        chunks = await vectorstore.search(
            query_vector=query_vector,
            top_k=args.get("top_k", 5),
            collection=args["collection"],
        )
        return [{"id": c.id, "text": c.text[:200], "score": c.score} for c in chunks]
    return f"Unknown tool: {name}"
```

- [ ] **Step 4: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_nodes.py -v`

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/tools/definitions.py backend/app/core/graph/nodes/tool_agent.py backend/tests/unit/core/test_nodes.py
git commit -m "feat: add tool definitions and tool agent node"
```

---

### Task 21: LangGraph graph assembly

**Files:**
- Create: `backend/app/core/graph/graph.py`
- Create: `backend/tests/unit/core/test_graph.py`

- [ ] **Step 1: Write failing test for graph construction**

```python
# backend/tests/unit/core/test_graph.py
import pytest
from unittest.mock import AsyncMock

from app.core.config.models import AgentsConfig
from app.core.graph.graph import build_graph


@pytest.mark.asyncio
async def test_graph_compiles(mock_llm, mock_vectorstore, mock_collection_store, mock_embedding):
    config = AgentsConfig()
    graph = build_graph(
        agents_config=config,
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )
    # Graph should be a compiled LangGraph object with an ainvoke method
    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_graph_chat_path(mock_llm, mock_vectorstore, mock_collection_store, mock_embedding):
    # Router returns "chat", chat agent returns a final answer
    mock_llm.complete = AsyncMock(side_effect=[
        {  # router
            "text": "chat",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        },
        {  # chat agent
            "text": "Hello! I can help with that.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        },
    ])
    config = AgentsConfig()
    graph = build_graph(
        agents_config=config,
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        collection_store=mock_collection_store,
        embedding=mock_embedding,
    )

    from app.core.graph.state import GraphState
    result = await graph.ainvoke(GraphState(query="hello"))

    assert result["route"] == "chat"
    assert result["final_answer"] == "Hello! I can help with that."
```

- [ ] **Step 2: Implement graph construction**

```python
# backend/app/core/graph/graph.py
from __future__ import annotations

from functools import partial
from typing import Literal

from langgraph.graph import END, StateGraph

from app.core.config.models import AgentsConfig
from app.core.graph.nodes import (
    answer_generation,
    chat_agent,
    retrieval,
    router,
    tool_agent,
    verifier,
)
from app.core.graph.state import GraphState
from app.ports.embedding import EmbeddingPort
from app.ports.llm import LLMPort
from app.ports.vectorstore import CollectionPort, VectorStorePort


def build_graph(
    agents_config: AgentsConfig,
    llm: LLMPort,
    vectorstore: VectorStorePort,
    collection_store: CollectionPort,
    embedding: EmbeddingPort,
):
    builder = StateGraph(GraphState)

    builder.add_node(
        "router",
        partial(router.run, config=agents_config.router, llm=llm),
    )
    builder.add_node(
        "chat_agent",
        partial(chat_agent.run, config=agents_config.chat_agent, llm=llm),
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
        partial(answer_generation.run, config=agents_config.answer_generation, llm=llm),
    )
    builder.add_node(
        "verifier",
        partial(verifier.run, config=agents_config.verifier, llm=llm),
    )
    builder.add_node(
        "tool_agent",
        partial(
            tool_agent.run,
            config=agents_config.tool_agent,
            llm=llm,
            vectorstore=vectorstore,
            collection_store=collection_store,
            embedding=embedding,
        ),
    )

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        lambda state: state.route or "chat",
        {"chat": "chat_agent", "rag": "retrieval", "tool": "tool_agent"},
    )
    builder.add_edge("chat_agent", END)
    builder.add_edge("tool_agent", END)
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
```

- [ ] **Step 3: Run -- expect pass**

Run: `cd backend && uv run pytest tests/unit/core/test_graph.py -v`

Expected: 2 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/app/core/graph/graph.py backend/tests/unit/core/test_graph.py
git commit -m "feat: assemble LangGraph graph with router, chat, rag, and tool paths"
```

---

**CHECKPOINT 4:** The graph compiles and routes correctly. `just test` should pass ~38+ unit tests. All core domain logic is complete.

---

## Phase 7: FastAPI Backend

### Task 22: Dependency injection + app entry point

- [ ] **Step 1: Extend VectorStorePort with list_documents**

Open `backend/app/ports/vectorstore.py` and add one method to `VectorStorePort`:

```python
async def list_documents(self, collection: str) -> list[dict[str, Any]]: ...
```

Then implement it in `backend/app/adapters/vectorstore/qdrant.py`:

```python
async def list_documents(self, collection: str) -> list[dict[str, Any]]:
    """Return unique documents (by doc_id) stored in the collection."""
    from qdrant_client.http.models import ScrollRequest

    result, _ = self._client.scroll(
        collection_name=collection,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    seen: set[str] = set()
    docs: list[dict[str, Any]] = []
    for point in result:
        payload = point.payload or {}
        doc_id = payload.get("doc_id", str(point.id))
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(
                {
                    "id": doc_id,
                    "filename": payload.get("filename", "unknown"),
                    "collection": collection,
                }
            )
    return docs
```

- [ ] **Step 2: Write `backend/app/api/dependencies.py`**

```python
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.adapters.storage.local import LocalFileStorageAdapter
from app.adapters.vectorstore.qdrant import QdrantVectorStoreAdapter
from app.core.config.loader import load_config
from app.core.config.models import AgentsConfig, SystemConfig
from app.ports.embedding import EmbeddingPort
from app.ports.llm import LLMPort
from app.ports.storage import DocumentStoragePort
from app.ports.vectorstore import CollectionPort, VectorStorePort


@lru_cache(maxsize=1)
def _cached_config() -> tuple[SystemConfig, AgentsConfig]:
    return load_config()


def get_system_config() -> SystemConfig:
    return _cached_config()[0]


def get_agents_config() -> AgentsConfig:
    return _cached_config()[1]


def get_llm(config: Annotated[SystemConfig, Depends(get_system_config)]) -> LLMPort:
    if config.llm.provider == "anthropic":
        from app.adapters.llm.anthropic import AnthropicLLMAdapter
        return AnthropicLLMAdapter()
    from app.adapters.llm.ollama import OllamaLLMAdapter
    return OllamaLLMAdapter(base_url=config.llm.ollama_base_url)


def get_vector_store(config: Annotated[SystemConfig, Depends(get_system_config)]) -> VectorStorePort:
    return QdrantVectorStoreAdapter(url=config.vectorstore.qdrant_url)


def get_collection_port(config: Annotated[SystemConfig, Depends(get_system_config)]) -> CollectionPort:
    return QdrantVectorStoreAdapter(url=config.vectorstore.qdrant_url)


def get_embedding(config: Annotated[SystemConfig, Depends(get_system_config)]) -> EmbeddingPort:
    if config.embeddings.provider == "workers-ai":
        from app.adapters.embeddings.workers_ai import WorkersAIEmbeddingAdapter
        return WorkersAIEmbeddingAdapter(
            account_id="",  # set via env: CLOUDFLARE_ACCOUNT_ID
            api_token="",   # set via env: CLOUDFLARE_API_TOKEN
            model=config.embeddings.workers_ai_model,
        )
    from app.adapters.embeddings.ollama import OllamaEmbeddingAdapter
    return OllamaEmbeddingAdapter(
        model=config.embeddings.ollama_model,
        base_url=config.embeddings.ollama_base_url,
    )


def get_storage() -> DocumentStoragePort:
    return LocalFileStorageAdapter()


SystemConfigDep = Annotated[SystemConfig, Depends(get_system_config)]
AgentsConfigDep = Annotated[AgentsConfig, Depends(get_agents_config)]
LLMDep = Annotated[LLMPort, Depends(get_llm)]
VectorStoreDep = Annotated[VectorStorePort, Depends(get_vector_store)]
CollectionDep = Annotated[CollectionPort, Depends(get_collection_port)]
EmbeddingDep = Annotated[EmbeddingPort, Depends(get_embedding)]
StorageDep = Annotated[DocumentStoragePort, Depends(get_storage)]
```

- [ ] **Step 3: Write `backend/app/main.py`**

```python
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, collections, documents, system
from app.core.config.loader import load_config

_system_config, _ = load_config()
logging.basicConfig(level=_system_config.environment.log_level.upper())

# Initialise Langfuse tracing when enabled in config.
# The handler is stored on app.state so routes can pass it to graph.ainvoke()
# as a callback without importing at module level.
_langfuse_handler = None
if _system_config.tracing.langfuse_enabled:
    try:
        from langfuse.callback import CallbackHandler as LangfuseCallback  # type: ignore[import]
        _langfuse_handler = LangfuseCallback(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=_system_config.tracing.langfuse_host,
        )
        logging.info("Langfuse tracing enabled (host: %s)", _system_config.tracing.langfuse_host)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Langfuse tracing could not be initialised: %s", exc)

app = FastAPI(title="Multi-Agent RAG Chatbot", version="0.1.0")
app.state.langfuse_handler = _langfuse_handler

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(collections.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(system.router, prefix="/api")
```

Create the routes package init: `backend/app/api/routes/__init__.py` (empty).

- [ ] **Step 4: Commit**

```bash
git add backend/app/ports/vectorstore.py \
        backend/app/adapters/vectorstore/qdrant.py \
        backend/app/api/dependencies.py \
        backend/app/main.py \
        backend/app/api/routes/__init__.py
git commit -m "feat: add FastAPI app entry point and dependency injection"
```

---

### Task 23: Chat endpoints

- [ ] **Step 1: Write `backend/app/api/routes/chat.py`**

```python
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.dependencies import (
    AgentsConfigDep,
    EmbeddingDep,
    LLMDep,
    SystemConfigDep,
    VectorStoreDep,
)
from app.core.graph.graph import build_graph
from app.core.graph.state import GraphState

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    collection: str = "langgraph-docs"


class CitationOut(BaseModel):
    source: str
    chunk_index: int
    text: str


class TraceEntryOut(BaseModel):
    node: str
    duration_ms: float
    detail: str


class ChatResponse(BaseModel):
    answer: str
    route: str | None
    citations: list[CitationOut]
    trace: list[TraceEntryOut]


def _invoke_config(request: Request) -> dict[str, Any]:
    """Build the LangGraph invocation config, attaching Langfuse if available."""
    handler = getattr(request.app.state, "langfuse_handler", None)
    return {"callbacks": [handler]} if handler else {}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    system_config: SystemConfigDep,
    agents_config: AgentsConfigDep,
    llm: LLMDep,
    vector_store: VectorStoreDep,
    embedding: EmbeddingDep,
) -> ChatResponse:
    graph = build_graph(
        system_config=system_config,
        agents_config=agents_config,
        llm=llm,
        vector_store=vector_store,
        embedding=embedding,
    )
    initial: GraphState = GraphState(query=request.query)
    final: GraphState = await graph.ainvoke(initial, config=_invoke_config(http_request))  # type: ignore[arg-type]
    return ChatResponse(
        answer=final.final_answer or "",
        route=final.route,
        citations=[
            CitationOut(source=c.source, chunk_index=c.chunk_index, text=c.text)
            for c in final.citations
        ],
        trace=[
            TraceEntryOut(node=t.node, duration_ms=t.duration_ms, detail=t.detail)
            for t in final.execution_trace
        ],
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    system_config: SystemConfigDep,
    agents_config: AgentsConfigDep,
    llm: LLMDep,
    vector_store: VectorStoreDep,
    embedding: EmbeddingDep,
) -> StreamingResponse:
    # v1: run the full graph then emit the result as a single SSE event.
    # Token-level streaming requires LangGraph astream_events wired through all
    # nodes -- deferred to a later iteration.
    graph = build_graph(
        system_config=system_config,
        agents_config=agents_config,
        llm=llm,
        vector_store=vector_store,
        embedding=embedding,
    )
    initial: GraphState = GraphState(query=request.query)
    final: GraphState = await graph.ainvoke(initial, config=_invoke_config(http_request))  # type: ignore[arg-type]
    answer = final.final_answer or ""

    async def _generate() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'delta': answer})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
```

- [ ] **Step 2: Write integration test skeleton `backend/tests/integration/test_api.py`**

```python
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from app.main import app
from app.api.dependencies import (
    get_llm,
    get_vector_store,
    get_collection_port,
    get_embedding,
    get_storage,
    get_agents_config,
    get_system_config,
)
from app.core.config.models import AgentsConfig, SystemConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system_config() -> SystemConfig:
    return SystemConfig(
        environment={"mode": "local", "log_level": "debug"},
        llm={"provider": "ollama", "ollama_base_url": "http://localhost:11434"},
        tracing={"langfuse_enabled": False, "langfuse_host": "http://localhost:3000", "langfuse_project": "test"},
        vectorstore={"provider": "qdrant", "qdrant_url": "http://localhost:6333"},
        embeddings={
            "provider": "ollama",
            "ollama_model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
        },
        ingestion={"chunk_size": 256, "chunk_overlap": 32, "supported_formats": ["md", "txt"]},
    )


def _make_agents_config() -> AgentsConfig:
    # Minimal valid agents config for testing
    return AgentsConfig(
        router={
            "enabled": True,
            "model": "llama3.2:3b",
            "prompt": "Classify",
            "routes": ["chat", "rag", "tool"],
        },
        chat_agent={
            "enabled": True,
            "model": "llama3.2:3b",
            "system_prompt": "You are helpful.",
            "max_tokens": 256,
        },
        retrieval={
            "enabled": True,
            "top_k": 5,
            "score_threshold": 0.5,
            "rerank": False,
            "default_collection": "test",
        },
        answer_generation={
            "enabled": True,
            "model": "llama3.1:8b",
            "prompt_template": "Answer: {evidence}",
            "max_tokens": 256,
        },
        verifier={
            "enabled": True,
            "model": "llama3.1:8b",
            "score_threshold": 0.5,
            "citation_coverage_min": 0.5,
            "max_retries": 1,
            "checks": ["score_threshold"],
        },
        tool_agent={
            "enabled": True,
            "model": "llama3.2:3b",
            "allowed_tools": [],
            "max_tool_calls": 3,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSystemEndpoints:
    def test_health(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/system/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_config_sanitised(self) -> None:
        system_cfg = _make_system_config()
        agents_cfg = _make_agents_config()

        app.dependency_overrides[get_system_config] = lambda: system_cfg
        app.dependency_overrides[get_agents_config] = lambda: agents_cfg
        try:
            with TestClient(app) as client:
                response = client.get("/api/system/config")
            assert response.status_code == 200
            body = response.json()
            assert body["environment"]["mode"] == "local"
            assert "ANTHROPIC_API_KEY" not in str(body)
        finally:
            app.dependency_overrides.clear()


@pytest.mark.integration
class TestChatEndpointIntegration:
    """Requires Qdrant (Docker) and Ollama running locally."""

    def test_chat_direct_path(self) -> None:
        with TestClient(app) as client:
            response = client.post("/api/chat", json={"query": "Hello, what can you do?"})
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body["answer"], str)
        assert body["route"] in ("chat", "rag", "tool")
```

- [ ] **Step 3: Run -- expect pass (non-integration tests only)**

```bash
cd backend && uv run pytest tests/integration/test_api.py::TestSystemEndpoints -v
```

Expected: `test_health` PASS, `test_config_sanitised` PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/app/api/routes/chat.py backend/tests/integration/test_api.py
git commit -m "feat: add POST /api/chat and POST /api/chat/stream endpoints"
```

---

### Task 24: Collection and document endpoints

- [ ] **Step 1: Write `backend/app/api/routes/collections.py`**

```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.dependencies import CollectionDep

router = APIRouter(tags=["collections"])


class CreateCollectionRequest(BaseModel):
    name: str
    vector_size: int = 768  # nomic-embed-text output dimension (default); 384 for bge-small-en-v1.5


@router.post("/collections", status_code=201)
async def create_collection(
    request: CreateCollectionRequest,
    collection_port: CollectionDep,
) -> dict:
    await collection_port.create(request.name, {"vector_size": request.vector_size})
    return {"name": request.name}


@router.get("/collections")
async def list_collections(collection_port: CollectionDep) -> list[str]:
    return await collection_port.list()


@router.get("/collections/{name}")
async def get_collection_stats(name: str, collection_port: CollectionDep) -> dict:
    stats = await collection_port.get_stats(name)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return stats


@router.delete("/collections/{name}", status_code=204)
async def delete_collection(name: str, collection_port: CollectionDep) -> None:
    await collection_port.delete(name)


@router.post("/collections/{name}/rebuild", status_code=202)
async def rebuild_collection(name: str) -> dict:
    # Full re-embedding is a background task -- deferred to a later iteration.
    return {"status": "accepted", "collection": name}
```

- [ ] **Step 2: Write `backend/app/api/routes/documents.py`**

```python
from __future__ import annotations

from fastapi import APIRouter, UploadFile

from app.api.dependencies import EmbeddingDep, StorageDep, VectorStoreDep
from app.core.config.models import SystemConfig
from app.api.dependencies import SystemConfigDep
from app.ingestion.pipeline import ingest_document

router = APIRouter(tags=["documents"])


@router.post("/collections/{collection}/documents", status_code=201)
async def upload_document(
    collection: str,
    file: UploadFile,
    system_config: SystemConfigDep,
    storage: StorageDep,
    embedding: EmbeddingDep,
    vector_store: VectorStoreDep,
) -> dict:
    content = await file.read()
    filename = file.filename or "upload"
    doc_id = await ingest_document(
        content=content,
        filename=filename,
        collection=collection,
        storage=storage,
        embedding=embedding,
        vector_store=vector_store,
        chunk_size=system_config.ingestion.chunk_size,
        chunk_overlap=system_config.ingestion.chunk_overlap,
    )
    return {"id": doc_id, "filename": filename, "collection": collection}


@router.get("/collections/{collection}/documents")
async def list_documents(collection: str, vector_store: VectorStoreDep) -> list[dict]:
    return await vector_store.list_documents(collection)


@router.get("/collections/{collection}/documents/{doc_id}")
async def get_document(
    collection: str,
    doc_id: str,
    vector_store: VectorStoreDep,
) -> dict:
    docs = await vector_store.list_documents(collection)
    match = next((d for d in docs if d["id"] == doc_id), None)
    if match is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return match


@router.delete("/collections/{collection}/documents/{doc_id}", status_code=204)
async def delete_document(
    collection: str,
    doc_id: str,
    vector_store: VectorStoreDep,
    storage: StorageDep,
) -> None:
    await vector_store.delete(collection, [doc_id])
    try:
        await storage.delete(doc_id)
    except FileNotFoundError:
        pass  # already removed; not an error
```

- [ ] **Step 3: Write `backend/app/api/routes/system.py`**

```python
from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import SystemConfigDep

router = APIRouter(tags=["system"])


@router.get("/system/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/system/config")
async def show_config(system_config: SystemConfigDep) -> dict:
    # Never expose secrets (API keys, tokens).
    return {
        "environment": {
            "mode": system_config.environment.mode,
            "log_level": system_config.environment.log_level,
        },
        "vectorstore": {"provider": system_config.vectorstore.provider},
        "embeddings": {
            "provider": system_config.embeddings.provider,
            "model": system_config.embeddings.model,
        },
        "ingestion": {
            "chunk_size": system_config.ingestion.chunk_size,
            "chunk_overlap": system_config.ingestion.chunk_overlap,
        },
    }
```

- [ ] **Step 4: Run -- expect pass**

```bash
cd backend && uv run pytest tests/integration/test_api.py -v -k "not integration"
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes/collections.py \
        backend/app/api/routes/documents.py \
        backend/app/api/routes/system.py
git commit -m "feat: add collection, document, and system API endpoints"
```

---

**CHECKPOINT 5:** `GET /api/system/health` returns 200. All non-integration API tests pass. `just lint` is clean.

---

## Phase 8: Frontend

`★ Insight ─────────────────────────────────────`
The frontend is intentionally minimal. React Router + useState/useContext only -- no Redux, no Zustand, no React Query. For a v1 learning project, the data flow is simple enough that local component state and a thin fetch client are clearer than adding an abstractions layer. The SSE streaming from `/api/chat/stream` uses the browser-native `EventSource` API, which handles reconnection automatically -- no library needed.
`─────────────────────────────────────────────────`

### Task 25: Vite + React scaffold

- [ ] **Step 1: Scaffold the frontend**

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install react-router-dom
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: Write `frontend/vite.config.ts`**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

The `server.proxy` forwards `/api/*` requests to the FastAPI backend during local development, so the frontend never needs to know the backend URL.

- [ ] **Step 3: Write `frontend/src/main.tsx`**

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>
);
```

- [ ] **Step 4: Write `frontend/src/App.tsx`**

```tsx
import { NavLink, Route, Routes } from "react-router-dom";
import ChatView from "./components/ChatView";
import CollectionsView from "./components/CollectionsView";

function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-4 py-2 rounded text-sm font-medium transition-colors ${
          isActive
            ? "bg-indigo-600 text-white"
            : "text-gray-300 hover:text-white hover:bg-gray-700"
        }`
      }
    >
      {label}
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100">
      <nav className="flex items-center gap-2 px-6 py-3 bg-gray-800 border-b border-gray-700">
        <span className="font-semibold text-white mr-4">RAG Chat</span>
        <NavItem to="/" label="Chat" />
        <NavItem to="/collections" label="Collections" />
      </nav>
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<ChatView />} />
          <Route path="/collections" element={<CollectionsView />} />
        </Routes>
      </main>
    </div>
  );
}
```

- [ ] **Step 5: Write `frontend/src/index.css`**

```css
@import "tailwindcss";
```

- [ ] **Step 6: Verify frontend starts**

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173` -- should show the nav bar with Chat and Collections links.

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold React + Vite + Tailwind CSS frontend with routing"
```

---

### Task 26: API client, types, and hooks

- [ ] **Step 1: Write `frontend/src/api/types.ts`**

```typescript
export interface CitationOut {
  source: string;
  chunk_index: number;
  text: string;
}

export interface TraceEntryOut {
  node: string;
  duration_ms: number;
  detail: string;
}

export interface ChatResponse {
  answer: string;
  route: string | null;
  citations: CitationOut[];
  trace: TraceEntryOut[];
}

export interface Collection {
  name: string;
}

export interface CollectionStats {
  name: string;
  vector_count: number;
}

export interface Document {
  id: string;
  filename: string;
  collection: string;
}
```

- [ ] **Step 2: Write `frontend/src/api/client.ts`**

```typescript
import type {
  ChatResponse,
  CollectionStats,
  Document,
} from "./types";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// Chat

export async function sendChat(query: string, collection = "langgraph-docs"): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, collection }),
  });
}

// Collections

export async function listCollections(): Promise<string[]> {
  return request<string[]>("/collections");
}

export async function createCollection(name: string, vectorSize = 384): Promise<void> {
  await request("/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, vector_size: vectorSize }),
  });
}

export async function deleteCollection(name: string): Promise<void> {
  await request(`/collections/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export async function getCollectionStats(name: string): Promise<CollectionStats> {
  return request<CollectionStats>(`/collections/${encodeURIComponent(name)}`);
}

// Documents

export async function uploadDocument(collection: string, file: File): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  return request<Document>(`/collections/${encodeURIComponent(collection)}/documents`, {
    method: "POST",
    body: form,
  });
}

export async function listDocuments(collection: string): Promise<Document[]> {
  return request<Document[]>(`/collections/${encodeURIComponent(collection)}/documents`);
}

export async function deleteDocument(collection: string, docId: string): Promise<void> {
  await request(
    `/collections/${encodeURIComponent(collection)}/documents/${encodeURIComponent(docId)}`,
    { method: "DELETE" }
  );
}
```

- [ ] **Step 3: Write `frontend/src/hooks/useChat.ts`**

```typescript
import { useState } from "react";
import { sendChat } from "../api/client";
import type { ChatResponse, CitationOut, TraceEntryOut } from "../api/types";

export interface Message {
  role: "user" | "assistant";
  content: string;
  route?: string | null;
  citations?: CitationOut[];
  trace?: TraceEntryOut[];
}

export function useChat(collection: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(query: string): Promise<void> {
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);
    try {
      const response: ChatResponse = await sendChat(query, collection);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.answer,
          route: response.route,
          citations: response.citations,
          trace: response.trace,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return { messages, loading, error, send };
}
```

- [ ] **Step 4: Write `frontend/src/hooks/useCollections.ts`**

```typescript
import { useEffect, useState } from "react";
import {
  createCollection,
  deleteCollection,
  listCollections,
  uploadDocument,
} from "../api/client";

export function useCollections() {
  const [collections, setCollections] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh(): Promise<void> {
    setLoading(true);
    try {
      setCollections(await listCollections());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load collections");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { refresh(); }, []);

  async function create(name: string): Promise<void> {
    await createCollection(name);
    await refresh();
  }

  async function remove(name: string): Promise<void> {
    await deleteCollection(name);
    await refresh();
  }

  async function upload(collection: string, file: File): Promise<void> {
    await uploadDocument(collection, file);
  }

  return { collections, loading, error, create, remove, upload, refresh };
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/ frontend/src/hooks/
git commit -m "feat: add typed API client and useChat/useCollections hooks"
```

---

### Task 27: Chat view and Collections view

- [ ] **Step 1: Write `frontend/src/components/ChatView.tsx`**

```tsx
import { useRef, useState } from "react";
import type { Message } from "../hooks/useChat";
import { useChat } from "../hooks/useChat";

function RouteBadge({ route }: { route?: string | null }) {
  if (!route) return null;
  const colours: Record<string, string> = {
    chat: "bg-blue-600",
    rag: "bg-green-600",
    tool: "bg-purple-600",
  };
  return (
    <span
      className={`ml-2 px-2 py-0.5 rounded text-xs font-mono text-white ${
        colours[route] ?? "bg-gray-600"
      }`}
    >
      {route}
    </span>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-2xl rounded-lg px-4 py-3 text-sm ${
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-gray-700 text-gray-100"
        }`}
      >
        <div className="flex items-center mb-1">
          <span className="font-medium">{isUser ? "You" : "Assistant"}</span>
          {!isUser && <RouteBadge route={message.route} />}
        </div>
        <p className="whitespace-pre-wrap">{message.content}</p>
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-600">
            <p className="text-xs text-gray-400 mb-1">Sources</p>
            {message.citations.map((c, i) => (
              <p key={i} className="text-xs text-gray-400 truncate">
                {c.source} (chunk {c.chunk_index})
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatView() {
  const [input, setInput] = useState("");
  const [collection] = useState("langgraph-docs");
  const { messages, loading, error, send } = useChat(collection);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function handleSend() {
    const query = input.trim();
    if (!query || loading) return;
    setInput("");
    await send(query);
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <p className="text-gray-500 text-sm text-center mt-16">
            Ask a question to get started.
          </p>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {loading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-700 rounded-lg px-4 py-3 text-sm text-gray-400">
              Thinking...
            </div>
          </div>
        )}
        {error && (
          <p className="text-red-400 text-sm text-center">{error}</p>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="px-6 pb-6 pt-2 border-t border-gray-700">
        <div className="flex gap-2">
          <textarea
            className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-sm
                       text-gray-100 placeholder-gray-500 resize-none focus:outline-none
                       focus:ring-1 focus:ring-indigo-500"
            rows={2}
            placeholder="Ask a question... (Enter to send, Shift+Enter for newline)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium
                       hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed
                       transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Write `frontend/src/components/CollectionsView.tsx`**

```tsx
import { useRef, useState } from "react";
import { useCollections } from "../hooks/useCollections";

export default function CollectionsView() {
  const { collections, loading, error, create, remove, upload } = useCollections();
  const [newName, setNewName] = useState("");
  const [selected, setSelected] = useState<string>("");
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploadStatus, setUploadStatus] = useState<string>("");

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    await create(name);
    setNewName("");
  }

  async function handleUpload() {
    const file = fileRef.current?.files?.[0];
    if (!file || !selected) return;
    setUploadStatus("Uploading...");
    try {
      await upload(selected, file);
      setUploadStatus(`Uploaded ${file.name}`);
      if (fileRef.current) fileRef.current.value = "";
    } catch (err) {
      setUploadStatus(err instanceof Error ? err.message : "Upload failed");
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-8">
      <h1 className="text-lg font-semibold mb-6">Collections</h1>

      {error && <p className="text-red-400 text-sm mb-4">{error}</p>}

      {/* Create collection */}
      <form onSubmit={handleCreate} className="flex gap-2 mb-6">
        <input
          className="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm
                     text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-1
                     focus:ring-indigo-500"
          placeholder="New collection name"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
        />
        <button
          type="submit"
          className="px-4 py-2 bg-indigo-600 text-white rounded text-sm font-medium
                     hover:bg-indigo-500 transition-colors"
        >
          Create
        </button>
      </form>

      {/* Collection list */}
      {loading ? (
        <p className="text-gray-500 text-sm">Loading...</p>
      ) : collections.length === 0 ? (
        <p className="text-gray-500 text-sm">No collections yet.</p>
      ) : (
        <ul className="space-y-2 mb-8">
          {collections.map((name) => (
            <li
              key={name}
              className={`flex items-center justify-between px-4 py-3 rounded-lg border
                          cursor-pointer transition-colors ${
                            selected === name
                              ? "border-indigo-500 bg-gray-800"
                              : "border-gray-700 bg-gray-800 hover:border-gray-500"
                          }`}
              onClick={() => setSelected(name)}
            >
              <span className="text-sm font-mono">{name}</span>
              <button
                onClick={(e) => { e.stopPropagation(); remove(name); }}
                className="text-xs text-red-400 hover:text-red-300 transition-colors"
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}

      {/* Upload document */}
      {selected && (
        <div className="border border-gray-700 rounded-lg p-4">
          <p className="text-sm text-gray-400 mb-3">
            Upload document to <span className="font-mono text-white">{selected}</span>
          </p>
          <div className="flex gap-2 items-center">
            <input
              type="file"
              ref={fileRef}
              accept=".md,.txt,.pdf"
              className="text-sm text-gray-300 file:mr-3 file:py-1 file:px-3
                         file:rounded file:border-0 file:bg-gray-700 file:text-sm
                         file:text-gray-200 file:cursor-pointer"
            />
            <button
              onClick={handleUpload}
              className="px-4 py-1.5 bg-green-700 text-white rounded text-sm font-medium
                         hover:bg-green-600 transition-colors"
            >
              Upload
            </button>
          </div>
          {uploadStatus && (
            <p className="mt-2 text-xs text-gray-400">{uploadStatus}</p>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Verify frontend compiles without TypeScript errors**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Smoke test in browser**

```bash
just dev
```

Open `http://localhost:5173`:
- Navigate to Chat, type "Hello", verify a response appears.
- Navigate to Collections, create a test collection, verify it appears in the list.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ frontend/src/api/ frontend/src/hooks/ frontend/src/App.tsx frontend/src/main.tsx frontend/src/index.css
git commit -m "feat: add ChatView and CollectionsView components"
```

---

**CHECKPOINT 6:** The full stack runs locally. `just dev` starts Qdrant, FastAPI, and Vite. Chat and Collections views are functional. `just test` passes all unit tests and non-integration API tests.

---

## Phase 9: CI

### Task 28: GitHub Actions CI workflow

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  backend:
    name: Backend -- lint + unit tests + type check
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - name: Install Python dependencies
        run: uv sync --dev
        working-directory: backend

      - name: Lint (ruff)
        run: uv run ruff check . && uv run ruff format --check .
        working-directory: backend

      - name: Unit tests
        run: uv run pytest tests/unit -v --tb=short
        working-directory: backend

      - name: Type check (mypy)
        run: uv run mypy app --ignore-missing-imports
        working-directory: backend

  frontend:
    name: Frontend -- type check + build
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: npm ci
        working-directory: frontend

      - name: Type check
        run: npx tsc --noEmit
        working-directory: frontend

      - name: Build
        run: npm run build
        working-directory: frontend
```

- [ ] **Step 2: Add mypy to dev dependencies**

```bash
cd backend && uv add --dev mypy
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml backend/pyproject.toml backend/uv.lock
git commit -m "ci: add GitHub Actions workflow for lint, unit tests, and type check"
```

- [ ] **Step 4: Push and verify CI passes**

```bash
git push
```

Open the Actions tab on GitHub. Both `backend` and `frontend` jobs should pass on the first run. If mypy reports issues, fix them before marking this task complete.

---

**CHECKPOINT 7 (Final):** CI is green. `just test` passes locally. `just dev` runs the full stack. The implementation plan is complete.

---

## Phase 10: n8n Local Automation Sidecar

n8n runs as a local Docker service alongside Qdrant. It handles document ingestion coordination, re-indexing, and scheduled maintenance. The core multi-agent graph is never involved in these operational flows -- n8n calls the FastAPI endpoints that already exist, and the graph only runs when a user sends a chat message.

### Architecture position

```
External trigger (webhook call / CRON)
          |
          v
    n8n Workflow Engine  (localhost:5678)
          |
          | HTTP Request nodes call existing FastAPI endpoints
          v
    FastAPI  (/api/collections, /api/chat, ...)
          |
          v
    LangGraph graph  +  Qdrant  +  Ollama
```

n8n never sits in the chat path. It operates exclusively on the ingestion and maintenance pathways. Chat latency is unaffected.

The only optional coupling in the other direction is a `trigger_n8n_workflow` tool that the tool agent may call -- this is additive, listed in `allowed_tools` in TOML, and off by default.

---

### Task 30: Add n8n to Docker Compose

**Files modified:** `docker-compose.yml`, `.env.example`, `justfile`

- [ ] **Step 1: Add n8n service to docker-compose.yml**

Replace the existing `docker-compose.yml` with:

```yaml
services:
  qdrant:
    image: qdrant/qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  n8n:
    image: n8nio/n8n:latest
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_BASIC_AUTH_USER:-admin}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_BASIC_AUTH_PASSWORD}
      - N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY}
      - N8N_HOST=0.0.0.0
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      # When n8n calls FastAPI via HTTP Request nodes, use host.docker.internal
      # (macOS / Windows Docker Desktop). On Linux use 172.17.0.1 or --add-host.
      - BACKEND_URL=http://host.docker.internal:8000
    extra_hosts:
      - "host.docker.internal:host-gateway"  # Linux compatibility
    volumes:
      - n8n_data:/home/node/.n8n
      - ./n8n/workflows:/home/node/workflows:ro  # workflow JSON exports (read-only)

volumes:
  qdrant_data:
  n8n_data:
```

`extra_hosts: host.docker.internal:host-gateway` makes the host machine reachable from inside the n8n container on Linux, where Docker Desktop's `host.docker.internal` alias is not available by default.

- [ ] **Step 2: Add n8n env vars to .env.example**

Append to `.env.example`:

```
# n8n
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=change-this-immediately
# Generate with: python -c "import secrets; print(secrets.token_hex(24))"
N8N_ENCRYPTION_KEY=replace-with-48-char-hex-string
```

`N8N_ENCRYPTION_KEY` must be a stable value -- changing it after saving credentials in n8n makes those credentials unreadable. Generate it once and keep it in `.env` (which is gitignored).

- [ ] **Step 3: Create n8n/workflows/ directory**

```bash
mkdir -p n8n/workflows
touch n8n/workflows/.gitkeep
```

Exported workflow JSON files go here. They are version-controlled so workflows can be re-imported after a fresh n8n setup.

- [ ] **Step 4: Add n8n targets to the justfile**

Add the following block below the existing justfile commands:

```just
# ─── n8n ──────────────────────────────────────────────────────────────────────

# Start the n8n container (UI at http://localhost:5678).
n8n-up:
    docker compose up n8n -d
    @echo "n8n UI: http://localhost:5678"

# Stop the n8n container.
n8n-down:
    docker compose stop n8n

# Tail n8n logs.
n8n-logs:
    docker compose logs n8n -f

# Open n8n UI in the browser.
n8n-open:
    open http://localhost:5678

# Export all n8n workflows to n8n/workflows/ as JSON (requires n8n CLI in container).
n8n-export:
    docker compose exec n8n n8n export:workflow --all --output=/home/node/workflows/
    @echo "Workflows exported to n8n/workflows/"
```

Update the existing `dev` target to start n8n alongside Qdrant:

```just
# Start Qdrant + n8n + backend + frontend for local dev
dev:
    docker compose up qdrant n8n -d
    cd backend && uv run uvicorn app.main:app --reload --port 8000 &
    cd frontend && npm run dev
```

- [ ] **Step 5: Test n8n starts cleanly**

```bash
just n8n-up
```

Open `http://localhost:5678` in a browser. Log in with the credentials from `.env`. The n8n canvas should load empty.

Test that n8n can reach FastAPI: in the n8n UI, create a one-off HTTP Request node pointing to `http://host.docker.internal:8000/api/system/health` and execute it. Expected response: `{"status": "ok"}`.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml .env.example n8n/workflows/.gitkeep justfile
git commit -m "chore: add n8n local automation sidecar to Docker Compose"
```

---

### Task 31: Create the core n8n workflows

Workflows are built in the n8n canvas, then exported to `n8n/workflows/` as JSON. The descriptions below are the blueprint for each workflow.

#### Workflow 1: Document Ingestion via Webhook

**Purpose:** receive an ingest request for a file accessible via URL, download it, and send it to the FastAPI ingestion endpoint.

**Webhook path:** `POST /webhook/ingest-document`

**Input body:**
```json
{
  "collection": "langgraph-docs",
  "source_url": "https://...",
  "filename": "document.md"
}
```

**Nodes:**

| # | Node type | Configuration |
|---|---|---|
| 1 | Webhook | Path: `ingest-document`, Method: POST, Response mode: Last node |
| 2 | HTTP Request | Method: GET, URL: `{{ $json.source_url }}`, Response format: File |
| 3 | HTTP Request | Method: POST, URL: `http://host.docker.internal:8000/api/collections/{{ $('Webhook').item.json.collection }}/documents`, Body content type: Form-Data, field `file` from binary output of node 2 |
| 4 | Respond to Webhook | Return value from node 3 |

Test the workflow once built:

```bash
curl -X POST http://localhost:5678/webhook/ingest-document \
  -H "Content-Type: application/json" \
  -d '{"collection":"langgraph-docs","source_url":"https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/low_level.md","filename":"low_level.md"}'
```

---

#### Workflow 2: Rebuild Collection via Webhook

**Purpose:** trigger a collection re-index on demand (e.g., after bulk document changes).

**Webhook path:** `POST /webhook/rebuild-collection`

**Input body:**
```json
{ "collection": "langgraph-docs" }
```

**Nodes:**

| # | Node type | Configuration |
|---|---|---|
| 1 | Webhook | Path: `rebuild-collection`, Method: POST |
| 2 | HTTP Request | Method: POST, URL: `http://host.docker.internal:8000/api/collections/{{ $json.collection }}/rebuild` |
| 3 | Respond to Webhook | Return value from node 2 |

Test:

```bash
curl -X POST http://localhost:5678/webhook/rebuild-collection \
  -H "Content-Type: application/json" \
  -d '{"collection":"langgraph-docs"}'
```

---

#### Workflow 3: Scheduled Collection Stats Report

**Purpose:** run on a schedule, fetch stats for all collections, and log a summary. Useful for spotting unexpectedly empty or stale collections without manually querying the API.

**Trigger:** Cron node -- `0 9 * * 1` (Monday 09:00 local time, or adjust to taste).

**Nodes:**

| # | Node type | Configuration |
|---|---|---|
| 1 | Schedule Trigger | Rule: every Monday 09:00 |
| 2 | HTTP Request | Method: GET, URL: `http://host.docker.internal:8000/api/collections` |
| 3 | Split In Batches | Batch size: 1 (loop over each collection name) |
| 4 | HTTP Request | Method: GET, URL: `http://host.docker.internal:8000/api/collections/{{ $json }}` |
| 5 | Aggregate | Aggregate all items into a single list |
| 6 | Code | Format the stats as a readable summary string (see snippet below) |

Code node snippet (JavaScript, runs inside n8n):

```javascript
const stats = $input.all().map(item => item.json);
const lines = stats.map(s =>
  `${s.name}: ${s.vector_count ?? 'unknown'} vectors`
);
return [{ json: { report: lines.join('\n'), generated_at: new Date().toISOString() } }];
```

This workflow does not send notifications in v1 -- the output is visible in the n8n execution log. Add a notification node (e.g., Email, Slack, or a webhook to your preferred tool) when needed.

---

- [ ] **Step 1: Build Workflow 1 (Document Ingestion) in the n8n UI**

Create the workflow using the node table above. Run the test `curl` command. Verify the document appears in `GET /api/collections/langgraph-docs/documents`.

- [ ] **Step 2: Build Workflow 2 (Rebuild Collection) in the n8n UI**

Create the workflow. Run the test `curl` command. Verify the response is `{"status": "accepted", ...}`.

- [ ] **Step 3: Build Workflow 3 (Scheduled Stats Report) in the n8n UI**

Create the workflow with a Schedule trigger. Manually execute it (n8n has a "Test workflow" button that bypasses the schedule). Verify the Code node outputs a non-empty report.

- [ ] **Step 4: Export workflows to version control**

```bash
just n8n-export
```

Commit the exported JSON files:

```bash
git add n8n/workflows/
git commit -m "feat: add n8n ingestion, rebuild, and stats report workflows"
```

---

### Task 32 (Optional): Expose n8n trigger to the tool agent

This task adds a `trigger_n8n_workflow` tool so the tool agent can call n8n workflows from inside a chat session. Example: a user asks "re-index the langgraph-docs collection" and the tool agent calls the rebuild webhook.

This is additive: it requires no changes to the graph, the ports, or any core domain code.

- [ ] **Step 1: Add the tool function to `backend/app/tools/definitions.py`**

Add to the existing tool definitions file:

```python
import os
from typing import Any

import httpx


async def trigger_n8n_workflow(
    workflow: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Trigger an n8n workflow by its webhook path name.

    Args:
        workflow: The webhook path registered in n8n (e.g. "rebuild-collection").
        payload: JSON body to send to the n8n webhook.

    Returns:
        The JSON response from n8n.
    """
    base_url = os.environ.get("N8N_BASE_URL", "http://localhost:5678")
    url = f"{base_url}/webhook/{workflow}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
```

- [ ] **Step 2: Register the tool in the tool definitions schema**

In `definitions.py`, add the tool schema alongside the existing tool schemas:

```python
TRIGGER_N8N_WORKFLOW_SCHEMA = {
    "name": "trigger_n8n_workflow",
    "description": (
        "Trigger a named n8n automation workflow via its webhook. "
        "Use 'rebuild-collection' to re-index a collection, "
        "'ingest-document' to ingest a document from a URL."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "description": "The n8n webhook path name (e.g. 'rebuild-collection').",
            },
            "payload": {
                "type": "object",
                "description": "JSON payload to send to the webhook.",
            },
        },
        "required": ["workflow", "payload"],
    },
}
```

- [ ] **Step 3: Add to agents.toml allowed_tools**

In `config/agents.toml`, update the `[tool_agent]` section:

```toml
[tool_agent]
enabled = true
model = "claude-haiku-4-5-20251001"
allowed_tools = [
    "search_collection",
    "list_collections",
    "get_collection_stats",
    "upload_document",
    "delete_document",
    "rebuild_index",
    "trigger_n8n_workflow",
]
max_tool_calls = 5
```

- [ ] **Step 4: Add N8N_BASE_URL to .env.example**

```
N8N_BASE_URL=http://localhost:5678
```

- [ ] **Step 5: Test from the chat UI**

With `just dev` running and n8n running (`just n8n-up`), ask the chat:

> "Re-index the langgraph-docs collection."

Expected: the tool agent routes to the `trigger_n8n_workflow` tool with `workflow = "rebuild-collection"` and `payload = {"collection": "langgraph-docs"}`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/tools/definitions.py config/agents.toml .env.example
git commit -m "feat: add trigger_n8n_workflow tool for agent-initiated automation"
```

---

### n8n + Tailscale

Since n8n runs on port 5678, it can be accessed from other tailnet devices via an additional Tailscale Serve rule. Add this only if you need access to the n8n canvas from a second device:

```bash
tailscale serve https +d:5678 / http://localhost:5678
```

The n8n basic auth (username + password from `.env`) provides the access gate. Do not enable Tailscale Funnel for n8n -- the admin canvas should never be publicly reachable.

---

**CHECKPOINT 8:** `just dev` starts Qdrant, n8n, FastAPI, and Vite. The ingestion webhook responds correctly to the test `curl` command. The stats report workflow executes cleanly in the n8n UI. The tool agent can trigger the rebuild workflow from a chat message.

---

## Phase 11: Tailscale Dev Access

Tailscale is used here strictly as a **development access layer**, not as production infrastructure. It provides private, encrypted access to the locally running stack from any device on your tailnet -- phone, secondary laptop, tablet -- without exposing anything publicly.

This phase has no effect on the multi-agent architecture, graph nodes, adapters, or API design. It is purely a connectivity convenience for the local-first development workflow.

### How the access works

Tailscale Serve proxies inbound tailnet HTTPS requests to the Vite dev server on port 5173. Vite's existing proxy config then forwards `/api/*` to FastAPI on port 8000 -- the same proxy that handles localhost browser access. One Serve rule exposes the full stack.

```
Other device (phone / laptop)
          |
          | HTTPS  (tailnet only, auto-managed cert)
          v
 Tailscale Serve  →  http://localhost:5173
          |
          | Vite proxy: /api  →  http://localhost:8000
          v
       FastAPI  →  LangGraph graph  →  Qdrant / Ollama
```

No CORS changes are needed. From any remote browser's perspective, all requests go to the same origin (the tailnet hostname), and Vite handles the internal routing.

---

### Task 29: Configure Tailscale Serve and add justfile commands

- [ ] **Step 1: Prerequisites**

Tailscale must be installed and authenticated on the development machine. This is a system-level dependency -- it is not a Python package or npm module.

```bash
# macOS (Homebrew)
brew install tailscale

# Authenticate (opens browser, logs in to your tailnet)
tailscale up
```

Verify the machine is connected: `tailscale status` -- your machine should appear with a green status.

Tailscale Funnel (used optionally in a later step) must be enabled for your account in the Tailscale admin console under the ACL / Funnel settings.

- [ ] **Step 2: Configure Tailscale Serve (one-time, persists across restarts)**

```bash
# Expose the Vite dev server privately on your tailnet over HTTPS.
# This configuration survives machine reboots -- you only need to run it once.
tailscale serve https / http://localhost:5173

# Verify
tailscale serve status
```

Expected output:

```
https://[your-machine].ts.net (tailnet only)
|-- / proxy http://127.0.0.1:5173
```

The tailnet hostname (`[your-machine].ts.net`) is where you access the app from other devices.

- [ ] **Step 3: Add Tailscale targets to the justfile**

Add the following block to `justfile`, below the existing commands:

```just
# ─── Tailscale dev access ─────────────────────────────────────────────────────

# Configure Tailscale Serve (idempotent -- persists across restarts, run once).
tailscale-setup:
    tailscale serve https / http://localhost:5173
    tailscale serve status

# Show current Tailscale Serve configuration and the access URL.
tailscale-status:
    tailscale serve status

# Remove all Tailscale Serve configuration.
tailscale-reset:
    tailscale serve off

# Start the full dev stack. Tailscale Serve exposes it on the tailnet.
# Run 'just tailscale-setup' once before using this target.
dev-serve:
    docker compose up qdrant -d
    cd backend && uv run uvicorn app.main:app --reload --port 8000 &
    cd frontend && npm run dev

# Enable Tailscale Funnel for temporary public prototype sharing.
# WARNING: this makes the app reachable by anyone on the internet.
# Disable with 'just funnel-stop' when the session is over.
funnel-start:
    tailscale funnel https / http://localhost:5173
    @echo "Funnel active. Disable with: just funnel-stop"
    tailscale serve status

# Disable Tailscale Funnel -- restore tailnet-only access.
funnel-stop:
    tailscale serve https / http://localhost:5173
    @echo "Funnel disabled. App is tailnet-only."
    tailscale serve status
```

- [ ] **Step 4: Test private access from another device**

Start the dev stack as normal:

```bash
just dev
```

On another device that is a member of your tailnet (phone, secondary laptop), open the URL shown by `tailscale serve status`:

```
https://[your-machine].ts.net/
```

Verify:
- The Chat view loads and a test question gets a response.
- The Collections view loads and lists collections.
- `GET https://[your-machine].ts.net/api/system/health` returns `{"status": "ok"}`.

**Note on Vite HMR:** Hot Module Replacement uses a WebSocket connection initiated by the browser. This WebSocket may not be proxied correctly over Tailscale Serve when accessing from a remote device. This only affects live-reload from remote devices -- development on the local machine is completely unaffected. If live-reload from a remote device is important, configure Vite to use polling (`server.watch: { usePolling: true }`) as a workaround.

- [ ] **Step 5: Commit**

```bash
git add justfile
git commit -m "chore: add Tailscale Serve targets to justfile for private dev access"
```

---

### Tailscale Funnel -- optional temporary prototype sharing

Tailscale Funnel makes the local app reachable at a public HTTPS URL with no deployment. It is useful for sharing a working prototype with a collaborator for a short session.

This is not a substitute for production deployment and should not be left active indefinitely.

```bash
# Enable public access (HTTPS, valid cert, real internet URL)
just funnel-start

# The URL is the same tailnet hostname, now publicly reachable.
tailscale serve status
# https://[your-machine].ts.net  (Funnel: on)
# |-- / proxy http://127.0.0.1:5173

# When the sharing session is over, disable immediately.
just funnel-stop
```

Tailscale Funnel requires that Funnel is enabled for your account in the Tailscale admin console.

---

### Access pattern summary

| Method | Accessible from | HTTPS | When to use |
|---|---|---|---|
| `http://localhost:5173` | Local machine only | No | Normal daily development |
| Tailscale Serve | Your tailnet devices | Yes (auto cert) | Testing from phone / other laptop |
| Tailscale Funnel | Anyone (internet) | Yes | Temporary demo sharing only |
| Cloudflare deployment | Anyone (internet) | Yes | When the app is stable |

The first two cover all normal development needs. Funnel is opt-in and temporary. Cloudflare deployment is the planned production path and remains unchanged.

---

**CHECKPOINT 9:** `just tailscale-setup` runs cleanly. The app is reachable at `https://[your-machine].ts.net/` from another tailnet device. `tailscale-status`, `tailscale-reset`, `funnel-start`, and `funnel-stop` all work.

---

## Phase 12: Langfuse Observability

Langfuse is a self-hosted LLM observability platform. It captures LangGraph traces (node execution, LLM calls, latency, token counts) via a callback handler that is initialised once at FastAPI startup and passed to `graph.ainvoke`. No agent code changes are needed.

This phase adds Langfuse (Postgres backend + Langfuse server) to Docker Compose, wires the callback handler into the FastAPI app (already done in Task 22 / main.py), and documents the one-time key setup.

**Scope:** local dev only. Langfuse does not affect graph topology, node logic, or production deployment paths.

### Task 33: Add Langfuse to Docker Compose and configure tracing

**Files:**
- Modify: `docker-compose.yml`
- Modify: `justfile`
- Modify: `config/config.toml`

- [ ] **Step 1: Extend docker-compose.yml with Langfuse services**

The Langfuse server requires a Postgres database. Add both services to the existing `docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  n8n:
    image: n8nio/n8n:latest
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_BASIC_AUTH_USER:-admin}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_BASIC_AUTH_PASSWORD}
      - N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY}
      - N8N_HOST=0.0.0.0
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - BACKEND_URL=http://host.docker.internal:8000
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - n8n_data:/home/node/.n8n
      - ./n8n/workflows:/home/node/workflows:ro

  langfuse-db:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_db:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 5s
      timeout: 5s
      retries: 5

  langfuse-server:
    image: langfuse/langfuse:latest
    restart: unless-stopped
    depends_on:
      langfuse-db:
        condition: service_healthy
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-db:5432/langfuse
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: ${LANGFUSE_NEXTAUTH_SECRET}
      SALT: ${LANGFUSE_SALT}
      LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: "false"

volumes:
  qdrant_data:
  n8n_data:
  langfuse_db:
```

This is the complete file, superseding the n8n-only version from Task 30. Replace `docker-compose.yml` in full.

- [ ] **Step 2: Add Langfuse justfile targets**

Add to `justfile`:

```just
# Start Langfuse (Postgres + server)
langfuse-up:
    docker compose up langfuse-db langfuse-server -d
    @echo "Langfuse starting at http://localhost:3000 -- allow 20-30 seconds"

# Stop Langfuse
langfuse-down:
    docker compose stop langfuse-db langfuse-server

# Open Langfuse UI in the default browser
langfuse-open:
    open http://localhost:3000
```

Also update the `dev` target to start Langfuse alongside the other services:

```just
dev:
    docker compose up qdrant langfuse-db langfuse-server -d
    cd backend && uv run uvicorn app.main:app --reload --port 8000 &
    cd frontend && npm run dev
```

And add a `dev-full` target that also starts n8n (for the full local stack including automation):

```just
dev-full:
    docker compose up qdrant langfuse-db langfuse-server n8n -d
    cd backend && uv run uvicorn app.main:app --reload --port 8000 &
    cd frontend && npm run dev
```

- [ ] **Step 3: One-time Langfuse project key setup**

This step is performed once per machine after starting Langfuse for the first time. The keys are stable and are stored in `.env`.

Start Langfuse:

```bash
just langfuse-up
# Wait ~30 seconds, then:
just langfuse-open
```

In the Langfuse UI at `http://localhost:3000`:

1. Register a new account (any email/password -- local only).
2. Create a new project named `langgraph-chatbot`.
3. Navigate to **Settings → API Keys** and create a new key pair.
4. Copy the **Public Key** and **Secret Key** into `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

These values persist across container restarts because the Postgres data is in a named volume.

- [ ] **Step 4: Enable tracing in config**

In `config/config.toml`, set:

```toml
[tracing]
langfuse_enabled = true
langfuse_host = "http://localhost:3000"
langfuse_project = "langgraph-chatbot"
```

Restart the FastAPI server (`Ctrl-C` the uvicorn process then `just dev`). The startup log should show:

```
INFO:root:Langfuse tracing enabled (host: http://localhost:3000)
```

- [ ] **Step 5: Verify a trace appears in the UI**

Send a test chat message:

```bash
curl -s -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What is LangGraph?"}' | python3 -m json.tool
```

Open `http://localhost:3000` → **Traces**. A new trace should appear with the graph nodes visible as spans. If no trace appears within 10 seconds, check that `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set correctly in `.env` and that the backend was restarted after editing `config.toml`.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml justfile config/config.toml
git commit -m "feat: add Langfuse self-hosted tracing to Docker Compose and FastAPI"
```

---

**CHECKPOINT 10:** `just dev` starts Qdrant, Langfuse, FastAPI, and Vite. A chat request produces a visible trace in the Langfuse UI at `http://localhost:3000`. `just test` still passes all unit tests unchanged.

---

## Self-review checklist

Run through these before declaring the implementation done:

- [ ] `just test` -- all unit tests pass
- [ ] `just lint` -- ruff reports zero issues
- [ ] `cd frontend && npx tsc --noEmit` -- zero TypeScript errors
- [ ] `cd frontend && npm run build` -- production build succeeds
- [ ] `just dev` -- full stack starts without errors (Qdrant, Langfuse, FastAPI, Vite)
- [ ] POST `/api/chat` with a general question -- routes to `chat`, returns an answer
- [ ] POST `/api/chat` with a document-specific question after ingestion -- routes to `rag`, returns citations
- [ ] POST `/api/collections/{name}/documents` with a Markdown file -- ingests successfully
- [ ] Verifier `refuse` path -- manually craft a query with no matching corpus content and verify the response is a refusal
- [ ] CI on GitHub Actions is green
- [ ] Langfuse UI at `http://localhost:3000` shows a trace with LangGraph node spans after sending a chat message
- [ ] Setting `langfuse_enabled = false` in `config/config.toml` and restarting -- tracing silently disabled, chat still works
- [ ] `just tailscale-setup` runs without error and `tailscale serve status` shows the correct proxy rule
- [ ] App loads at `https://[your-machine].ts.net/` from another tailnet device and the chat responds correctly
- [ ] `just n8n-up` starts n8n and the UI is reachable at `http://localhost:5678`
- [ ] n8n ingestion webhook responds correctly to the test `curl` command and the document appears in `/api/collections/{name}/documents`
- [ ] n8n stats report workflow executes and outputs a non-empty report
