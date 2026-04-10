# Multi-Agent RAG Chatbot -- Technical Design

## Overview

A local-first multi-agent RAG chat application built as a learning and experimentation platform. The system orchestrates multiple specialised agents through a graph-based workflow, supporting retrieval-augmented generation with grounded response verification, configurable agents via TOML, tool calling, and a path to Cloudflare deployment.

## Technology stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (minimal LangChain -- direct adapters only) |
| LLM | Anthropic API (Python SDK, Messages API, tool use) |
| Vector store | Qdrant (local dev via Docker) / Cloudflare Vectorize (production) |
| Embeddings | Cloudflare Workers AI (`bge-small-en-v1.5`, free tier) |
| Backend API | FastAPI (async) |
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Configuration | TOML files |
| Observability | LangSmith |
| Task runner | justfile |
| Dependency management | uv (Python), npm (frontend) |
| Deployment | Cloudflare Pages (frontend) + Python Workers (backend) |
| Containerisation | Docker + Docker Compose (local dev) |
| CI/CD | GitHub Actions |

## Architecture

Hexagonal Architecture (Ports and Adapters): inbound adapters (FastAPI routes) -> core domain (graph, agents, verification) -> outbound adapters (Anthropic SDK, Qdrant client, Workers AI API, filesystem).

The core domain has zero external dependencies. All I/O crosses a port defined as a Python Protocol.

### Approach: static graph skeleton with pluggable node implementations

The graph structure (router -> retrieval -> answer -> verify) is fixed in code. Each node is a thin wrapper that delegates to a pluggable implementation resolved from TOML config. Config controls model, prompt, thresholds, and available tool functions -- but not the graph topology.

This balances readability (graph structure visible in code) with flexibility (behaviour fully configurable). Adding a new graph path requires a code change, but that should be deliberate and rare.

## Project structure

```
langgraph-chatbot/
  backend/
    app/
      api/                        # FastAPI routes, middleware
      core/
        graph/                    # LangGraph graph definition, nodes
        agents/                   # Agent implementations
        config/                   # TOML parsing, Pydantic config models
        models/                   # Domain types
      ports/                      # Protocols (interfaces)
      adapters/                   # Port implementations
        llm/                      # Anthropic SDK adapter
        vectorstore/              # Qdrant adapter + Vectorize adapter
        embeddings/               # Workers AI adapter
        storage/                  # File storage adapter
      tools/                      # Tool definitions for tool-using agents
      ingestion/                  # Chunking, embedding, indexing
    tests/
      unit/
        core/
        ingestion/
      integration/
      conftest.py
    pyproject.toml
  frontend/
    src/
      components/
      hooks/
      api/
    package.json
    vite.config.ts
  config/
    agents.toml                   # Per-agent configuration
    config.toml                   # System-wide config
  corpus/                         # Raw source documents for ingestion
    langgraph-docs/               # Sample dataset
  docs/                           # Detailed project documentation
  docker/                         # Dockerfiles
    Dockerfile.backend
    Dockerfile.frontend
  scripts/                        # Utility scripts
  .github/
    workflows/
      ci.yml
      deploy.yml
  .env.example
  .env                            # Gitignored
  .gitignore
  .dockerignore
  docker-compose.yml
  justfile
  README.md
  CLAUDE.md
```

### Key structural rules

- `core/` never imports from `adapters/` -- only from `ports/`.
- Adapter selection is driven by TOML config (`environment.mode`).
- Adapters are injected at startup via FastAPI's dependency system.
- Config files live at project root in `config/`, shared between backend and scripts.
- Raw documents for ingestion live in `corpus/`, separate from project documentation in `docs/`.

## Graph design

### Execution paths

Three paths through a shared LangGraph graph:

- **Direct chat:** `user_query -> router -> chat_agent -> final_response`
- **RAG:** `user_query -> router -> retrieval -> answer_generation -> verifier -> final_response`
- **Tool use:** `user_query -> router -> tool_agent -> final_response`

### Graph diagram

```
                     +------------+
                     |  __start__ |
                     +-----+------+
                           |
                     +-----v------+
                     |   router   |
                     +-----+------+
                           |
            +--------------+--------------+
            |              |              |
     +------v------+ +----v-----+ +------v-------+
     |  chat_agent | | retrieval| |  tool_agent  |
     +------+------+ +----+-----+ +------+-------+
            |              |              |
            |        +-----v------+       |
            |        |   answer   |       |
            |        +-----+------+       |
            |              |              |
            |        +-----v------+       |
            |        |  verifier  |       |
            |        +-----+------+       |
            |              |              |
            |        +-----+------+       |
            |        |            |       |
            |     accept    revise/refuse |
            |        |            |       |
            +--------+-----+-----+-------+
                           |
                     +-----v------+
                     |  __end__   |
                     +------------+
```

### Node roles

| Node | Responsibility | Model |
|---|---|---|
| Router | Classify intent, decide path (chat/rag/tool) | Haiku 4.5 |
| Chat agent | General conversation, present final responses | Haiku 4.5 |
| Retrieval | Vector search, metadata filtering, reranking, build evidence bundle | N/A (calls ports) |
| Answer generation | Produce draft answer from retrieved chunks with citations | Sonnet 4.6 |
| Verifier | Check answer-to-evidence support; decide accept/revise/refuse | Sonnet 4.6 |
| Tool agent | Execute tools (ingestion, system, utilities) via Anthropic tool-use API | Haiku 4.5 |

### Node function signature

Each node is a pure async function:

```python
async def node_fn(state: GraphState, config: AgentConfig) -> GraphState
```

The graph skeleton wires nodes together. TOML config is injected at construction time via `functools.partial`.

### Revise/refuse flow

- `accept` -- answer passes to `__end__`
- `revise` -- loops back to answer generation with verifier feedback (max retries from TOML)
- `refuse` -- returns a "cannot answer" response with the reason

### GraphState

```python
@dataclass
class GraphState:
    query: str
    route: str | None
    retrieval_query: str | None
    retrieved_chunks: list[Chunk]
    retrieval_scores: list[float]
    metadata_filters: dict
    draft_answer: str | None
    verifier_result: VerifierResult | None
    final_answer: str | None
    citations: list[Citation]
    tool_calls: list[ToolCall]
    execution_trace: list[TraceEntry]
    retry_count: int
```

## Ports and adapters

### Ports

Each port is a single-concern Python Protocol.

| Port | Responsibility | Key methods |
|---|---|---|
| `LLMPort` | Send messages, get completions | `complete(messages, tools?) -> Response` |
| `VectorStorePort` | Store and search vectors | `search(query_vector, top_k, filters) -> list[Chunk]`, `upsert(chunks)`, `delete(ids)` |
| `EmbeddingPort` | Generate embeddings | `embed(texts) -> list[Vector]` |
| `CollectionPort` | Manage collections | `create(name, config)`, `list()`, `delete(name)`, `get_stats(name)` |
| `DocumentStoragePort` | Store raw uploaded files | `store(file) -> path`, `retrieve(path) -> bytes`, `delete(path)` |

### Adapters

| Adapter | Implements | Backing service |
|---|---|---|
| `AnthropicLLMAdapter` | `LLMPort` | Anthropic Python SDK |
| `QdrantVectorStoreAdapter` | `VectorStorePort` + `CollectionPort` | qdrant-client |
| `CloudflareVectorizeAdapter` | `VectorStorePort` + `CollectionPort` | Vectorize binding |
| `WorkersAIEmbeddingAdapter` | `EmbeddingPort` | Cloudflare Workers AI API |
| `LocalFileStorageAdapter` | `DocumentStoragePort` | Local filesystem |

### Adapter swapping

The only things that change between local dev and Cloudflare deployment are the vector store adapter and the deployment wrapper. The core domain is identical.

| Layer | Local dev | Cloudflare |
|---|---|---|
| Frontend | Vite dev server | Cloudflare Pages |
| Backend API | FastAPI (uvicorn) | Python Worker (FastAPI) |
| Vector DB | Qdrant (Docker) | Cloudflare Vectorize |
| LLM | Anthropic API | Anthropic API (same) |

## TOML configuration

### config/config.toml

```toml
[environment]
mode = "local"                          # "local" or "cloudflare"
log_level = "debug"

[tracing]
langsmith_enabled = true
langsmith_project = "langgraph-chatbot"

[vectorstore]
provider = "qdrant"                     # "qdrant" or "vectorize"
qdrant_url = "http://localhost:6333"

[embeddings]
provider = "workers-ai"
model = "@cf/baai/bge-small-en-v1.5"
workers_ai_base_url = "https://api.cloudflare.com/client/v4/accounts"
# Full URL constructed at runtime using CLOUDFLARE_ACCOUNT_ID from .env

[ingestion]
chunk_size = 512
chunk_overlap = 64
supported_formats = ["md", "txt", "pdf"]
```

### config/agents.toml

```toml
[router]
enabled = true
model = "claude-haiku-4-5-20251001"
prompt = "Classify the user query..."
routes = ["chat", "rag", "tool"]

[chat_agent]
enabled = true
model = "claude-haiku-4-5-20251001"
system_prompt = "You are a helpful assistant..."
max_tokens = 2048

[retrieval]
enabled = true
top_k = 10
score_threshold = 0.7
rerank = true
default_collection = "langgraph-docs"

[answer_generation]
enabled = true
model = "claude-sonnet-4-6-20250514"
prompt_template = "Answer based on the following evidence..."
max_tokens = 2048

[verifier]
enabled = true
model = "claude-sonnet-4-6-20250514"
score_threshold = 0.75
citation_coverage_min = 0.8
max_retries = 2
checks = ["score_threshold", "support_analysis", "citation_coverage"]

[tool_agent]
enabled = true
model = "claude-haiku-4-5-20251001"
allowed_tools = ["search_collection", "list_collections", "get_collection_stats",
                 "upload_document", "delete_document", "rebuild_index"]
max_tool_calls = 5
```

### Config flow

1. At startup, both TOML files are parsed into Pydantic models.
2. Config is validated (unknown keys rejected, types checked).
3. Adapters are instantiated based on environment/provider settings.
4. Agent configs are passed to node functions at graph construction time via `functools.partial`.

## Verification approach

Grounding verification goes beyond cosine similarity. The verifier combines multiple checks:

- **Retrieval score thresholds** -- reject chunks below a minimum similarity score
- **Support analysis** -- LLM-based check of whether each claim in the draft answer is supported by retrieved evidence
- **Citation coverage** -- verify that a minimum proportion of answer claims have cited sources
- **Unsupported claim detection** -- identify specific claims with no supporting evidence

Outcomes:
- `accept` -- answer is well-grounded, pass through
- `revise` -- partially grounded, loop back with feedback (up to max_retries)
- `refuse` -- insufficiently grounded, return a "cannot answer" response with the reason

All thresholds are configurable via `agents.toml`.

## API design

### Chat

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Send message, run full graph, return response |
| `POST` | `/api/chat/stream` | Same but streamed via SSE |

### Collection management

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/collections` | Create a collection |
| `GET` | `/api/collections` | List all collections |
| `GET` | `/api/collections/{name}` | Get collection stats |
| `DELETE` | `/api/collections/{name}` | Delete a collection |
| `POST` | `/api/collections/{name}/rebuild` | Rebuild embeddings |

### Document management

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/collections/{name}/documents` | Upload document (file + optional metadata) |
| `GET` | `/api/collections/{name}/documents` | List documents in collection |
| `GET` | `/api/collections/{name}/documents/{id}` | Get document metadata |
| `DELETE` | `/api/collections/{name}/documents/{id}` | Delete document |

### System

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/system/config` | Show active configuration (sanitised) |
| `GET` | `/api/system/health` | Health check |

### Ingestion pipeline

Triggered by document upload:

1. File received via API
2. Stored via `DocumentStoragePort`
3. Chunked based on format and ingestion config (chunk size, overlap)
4. Each chunk embedded via `EmbeddingPort`
5. Chunks + vectors upserted via `VectorStorePort`
6. Metadata (filename, chunk index, timestamps) stored alongside vectors

## Frontend design

Minimal chat UI -- the frontend is not the core learning goal.

### Pages

- **Chat view** -- message input, conversation thread, route indicator (direct/RAG/tool), citations
- **Collections view** -- list/create/delete collections, upload documents, stats
- **Trace view** -- per-response execution trace: nodes run, timing, retrieval scores, verifier decision

### Tech choices

- React + TypeScript + Vite + Tailwind CSS
- React Router for navigation
- Thin `fetch`-based API client (no heavy HTTP library)
- SSE via `EventSource` for streaming chat responses
- No state management library for v1 (`useState`/`useContext` only)

## Testing strategy

### Framework

pytest with pytest-asyncio.

### Structure

```
backend/tests/
  unit/
    core/
      test_graph.py               # Graph construction, routing logic
      test_agents.py              # Node functions with mocked ports
      test_config.py              # TOML parsing, validation, defaults
      test_verifier.py            # Verification checks in isolation
    ingestion/
      test_chunking.py            # Chunking strategies per format
  integration/
    test_api.py                   # FastAPI routes via httpx test client
    test_qdrant_adapter.py        # Against real Qdrant instance
    test_ingestion_pipeline.py    # Full upload -> chunk -> embed -> store
  conftest.py
```

### Approach

- Unit tests mock all ports -- test core logic in isolation. The verifier gets the most thorough tests.
- Integration tests hit real services (Qdrant via Docker, Anthropic API with Haiku). Gated behind a pytest marker.
- No mocking the LLM for integration tests -- real API calls with Haiku keep costs low and catch real issues.

## Docker and CI/CD

### Docker Compose (local dev)

```yaml
services:
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes: [qdrant_data:/qdrant/storage]
  backend:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./config:/app/config
      - ./corpus:/app/corpus
    depends_on: [qdrant]
```

Frontend runs via `vite dev` locally (not containerised -- hot reload is faster outside Docker).

### CI/CD (GitHub Actions)

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | Push / PR (all branches) | Lint (ruff), unit tests (pytest), build frontend, type check (tsc) |
| `deploy.yml` | Push to `main` (after CI passes) | Deploy frontend to Cloudflare Pages + deploy backend to Python Worker |

### Justfile commands

```
test:             # run unit tests only
test-all:         # run unit + integration
test-one FILE:    # run a single test file
lint:             # ruff check + ruff format --check
format:           # ruff format
dev:              # start docker-compose + uvicorn + vite dev
```

## Sample data

The system ships with a curated subset of LangGraph documentation in `corpus/langgraph-docs/`. This provides a meaningful test corpus where grounding verification actually matters (factual, verifiable content). The ingestion pipeline accepts any documents -- users add their own over time.

## Non-goals for v1

- Cloudflare deployment optimisation (architecture supports it, but local-first comes first)
- Multi-user/auth concerns
- Complex state management in frontend
- Model training or fine-tuning
- Kubernetes, MLflow, Prometheus, Grafana
