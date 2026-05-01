# langgraph-chatbot

A local-first multi-agent RAG chat application. Learning and experimentation
platform for graph-based agent orchestration, retrieval-augmented generation,
grounded response verification, configurable agents, conversation memory with
summarisation, and tool calling.

Designed local-first, with a path to Cloudflare deployment.

## Stack at a glance

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (minimal LangChain -- direct adapters only) |
| LLM | Ollama (default) / Anthropic API (per-node override) |
| Vector store | Qdrant (local) / Cloudflare Vectorize (production) |
| Embeddings | Ollama `nomic-embed-text` / Cloudflare Workers AI |
| Conversation memory | SQLite (local) with rolling-window summarisation |
| Backend | FastAPI (async) |
| Frontend | React + TypeScript + Vite + Tailwind |
| Configuration | TOML (`config/config.toml`, `config/agents.toml`) |
| Observability | Langfuse (self-hosted, optional) |
| Task runner | justfile |
| Dependency management | uv (Python), npm (frontend) |

## Quick start

Prerequisites: [Docker](https://www.docker.com/), [uv](https://github.com/astral-sh/uv),
[Node 22+](https://nodejs.org/), [Ollama](https://ollama.com/) running locally.

```bash
just ollama-check        # pull required models if missing
just dev                 # Qdrant + Langfuse + backend + frontend
```

Frontend at `http://localhost:5173`, backend at `http://localhost:8000`.

```bash
just check               # backend lint + tests + frontend tsc + tests + build
just test                # backend unit tests only
just test-all            # unit + integration (requires Ollama running)
just lint                # ruff + mypy (matches CI)
just format              # ruff format + auto-fix
```

## What's in here

The graph runs three execution paths through a shared LangGraph:

- **chat** -- direct conversation, no retrieval.
- **rag** -- retrieve, answer, verify (with revise/refuse loop).
- **tool** -- inspect collections, ingest documents, run system operations.

Each node is a pure async function; TOML config is injected at graph
construction via `functools.partial`. The verifier composes a
score-threshold check, a citation-coverage check that validates against
real chunk ids, and an optional LLM-based support analysis.

Multi-turn chat is backed by a SQLite store with atomic
`(user, assistant)` persistence and a rolling-window summariser that
folds older turns into a summary when the conversation grows past a
threshold. Best-effort: a summariser failure falls back to unsummarised
history rather than blocking the chat.

Each agent section in `agents.toml` may override the LLM provider
(`provider = "ollama" | "anthropic"`); a registry in `dependencies.py`
exposes available providers, and the graph routes per node. Startup
validation crashes the app with a clear error if a node requests an
unregistered provider.

A read-only debug surface (`GET /api/conversations`,
`GET /api/conversations/{id}`, plus a frontend sidebar) lets you inspect
what's persisted and switch between conversations.

## Further reading

- `CLAUDE.md` -- working agreements for AI agents and contributors:
  conventions, architecture rules, command index.
- `docs/superpowers/specs/2026-04-10-multi-agent-rag-chatbot-design.md`
  -- full technical design spec.
- `docs/adrs/` -- architectural decision records:
  [0001 per-node LLM provider](docs/adrs/0001-per-node-llm-provider.md),
  [0002 conversation memory](docs/adrs/0002-conversation-memory.md),
  [0003 verifier grounding](docs/adrs/0003-verifier-grounding-strategy.md),
  [0004 atomic conversation persistence](docs/adrs/0004-atomic-conversation-persistence.md).

## Status

Personal learning and experimentation project. Working local-first
deployment; Cloudflare adapters are wired and tested at the HTTP
boundary but await an end-to-end smoke test against a real account.
British English in all documentation.
