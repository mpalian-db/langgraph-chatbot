# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Local-first multi-agent RAG chat application. A learning and experimentation platform for graph-based agent orchestration, retrieval-augmented generation, grounded response verification, configurable agents, and tool calling. Designed local-first with a path to Cloudflare deployment.

Full technical design spec: `docs/superpowers/specs/2026-04-10-multi-agent-rag-chatbot-design.md`

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (minimal LangChain -- direct adapters only) |
| LLM | Ollama (default, local) / Anthropic API (alternative, configurable via TOML) |
| Vector store | Qdrant (local dev via Docker) / Cloudflare Vectorize (production) |
| Embeddings | Ollama `nomic-embed-text` (default, 768-dim) / Cloudflare Workers AI (alternative) |
| Backend API | FastAPI (async) |
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Configuration | TOML files (`config/config.toml`, `config/agents.toml`) |
| Observability | Langfuse (self-hosted via Docker) |
| Automation sidecar | n8n (local Docker, ingestion and maintenance workflows) |
| Dev access | Tailscale Serve (private tailnet HTTPS) |
| Task runner | justfile |
| Dependency management | uv (Python), npm (frontend) |

## Commands

```bash
just dev            # start Qdrant + Langfuse + uvicorn + vite dev
just dev-full       # as above, plus n8n
just test           # run backend unit tests only
just test-all       # run backend unit + integration tests
just test-one FILE  # run a single test file
just lint           # ruff + ruff format --check + mypy (matches CI)
just format         # ruff format + ruff --fix
just check          # everything CI runs: backend lint+tests, frontend tsc+test+build
```

Integration tests require Qdrant (Docker) and Ollama running locally, and are gated behind a pytest marker. The frontend runs via `vite dev` outside Docker (not containerised -- hot reload is faster).

## Architecture

Hexagonal Architecture: inbound adapters (FastAPI routes) -> core domain (graph, agents, verification) -> outbound adapters (Ollama client, Qdrant client, filesystem).

**The core domain has zero external dependencies.** All I/O crosses a port defined as a Python Protocol in `backend/app/ports/`. `core/` never imports from `adapters/` -- only from `ports/`.

Adapters are injected at startup via FastAPI's dependency system. `llm.provider` and `embeddings.provider` in `config/config.toml` drive adapter selection in `dependencies.py`. The only things that change between local dev and Cloudflare deployment are the vector store and embedding adapters.

### Graph structure

Three execution paths through a shared LangGraph graph:

- **Direct chat:** `user_query -> router -> chat_agent -> final_response`
- **RAG:** `user_query -> router -> retrieval -> answer_generation -> verifier -> final_response`
- **Tool use:** `user_query -> router -> tool_agent -> final_response`

The graph structure (router -> retrieval -> answer -> verify) is fixed in code. Each node is a thin wrapper delegating to a pluggable implementation resolved from TOML config. Config controls model, prompt, thresholds, and tool permissions -- not graph topology. Adding a new graph path requires a deliberate code change.

### Node function signature

Each node is a pure async function:

```python
async def node_fn(state: GraphState, config: AgentConfig) -> GraphState
```

TOML config is injected at graph construction time via `functools.partial`.

### Node model assignments (defaults)

| Node | Default model |
|---|---|
| Router, Chat agent, Tool agent, Worklog, Summariser | `llama3.2:3b` |
| Answer generation, Verifier | `llama3.1:8b` |

Override by setting `model = "..."` in `config/agents.toml`. Anthropic models (`claude-haiku-4-5-20251001`, `claude-sonnet-4-6-20250514`) work when `llm.provider = "anthropic"`.

### Per-node LLM provider override

Each agent section in `agents.toml` may set `provider = "ollama" | "anthropic"` to use a different LLM than the system default. The verifier's support_analysis check, in particular, benefits from Anthropic's reliable structured output. The system builds a registry (`get_llm_registry`) of available providers; `_resolve_llm` in `graph.py` picks the right port per node. Misconfiguration (e.g. requesting `anthropic` without `ANTHROPIC_API_KEY`) fails at app startup via `validate_llm_providers`. See `docs/adrs/0001-per-node-llm-provider.md`.

### Conversation memory

Multi-turn chat is backed by a SQLite store (`SQLiteConversationStore`) implementing `ConversationReaderPort` + `ConversationWriterPort`. Each chat round persists `(user, assistant)` atomically via `append_pair`. When the unsummarised tail crosses `[summariser].summarise_threshold` turns, `load_with_summary` (in `app.core.operations.conversation_memory`) folds the older slice into a rolling summary, keeping the last `keep_recent` turns verbatim. Summarisation is best-effort: an LLM failure falls back to unsummarised history rather than 500'ing the chat. See `docs/adrs/0002-conversation-memory.md`.

### Read-only debug endpoints

- `GET /api/conversations` -- list every known conversation with metadata (turn_count, has_summary, last_updated_at).
- `GET /api/conversations/{id}` -- summary + post-boundary turns, or 404 when truly unknown.
- `GET /api/collections/{name}/documents/{id}` -- single chunk by id, 404 if not present.
- `DELETE /api/collections/{name}/documents/{id}` -- 204 idempotent.

### Verifier outcomes

`accept` -> pass through. `revise` -> loop back to answer generation with feedback (up to `max_retries` from TOML). `refuse` -> return "cannot answer" with reason. The verifier combines: retrieval score thresholds, LLM-based support analysis, citation coverage, and unsupported claim detection.

### Config files

`config/` lives at the project root, shared between backend and scripts. `corpus/` holds raw source documents for ingestion (separate from `docs/`). The sample dataset is a curated subset of LangGraph documentation (`corpus/langgraph-docs/`).

## Testing

pytest with pytest-asyncio. Unit tests mock all ports and test core logic in isolation. The verifier gets the most thorough unit tests. Integration tests hit real services (Qdrant via Docker, Ollama). Integration tests are gated behind a pytest marker and live in `backend/tests/integration/`.

## Conventions

- British English in all documentation and comments.
- No emojis anywhere (docs, commits, comments, output).
- Keep README concise; detailed docs go under `docs/`. Architectural decisions live in `docs/adrs/`.
- Split ports into single-concern Protocols. Never aggregate read, write, and mutation behind one interface.
- Node functions are pure async; TOML config is injected at construction time via `functools.partial`, never captured from globals.
- Isolate side effects (LLM API calls, vector DB, file I/O) at the boundaries behind ports.
- Cross-cutting concerns (e.g. tenant isolation in the Vectorize adapter, cancellation in the streaming chat hook) belong at the adapter or hook boundary, not in route handlers.
- `agents_config` is `lru_cache`d in `dependencies.py` -- never mutate it in `build_graph`. Use `model_copy(update=...)` for per-request derivations.
