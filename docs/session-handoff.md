# Session Handoff — LangGraph Chatbot

> Last updated: 2026-04-17. Use this document to restore full context when starting a new Claude Code session on a different machine.

---

## What this project is

A local-first multi-agent RAG chat application. The user asks questions in a React frontend; a FastAPI backend runs a fixed LangGraph graph that routes to a chat agent, a RAG pipeline (retrieval → answer generation → grounding verifier), or a tool agent. Everything is configurable via TOML files — graph topology is fixed in code, agent behaviour (models, thresholds, prompts) lives in config.

**Design principle:** Hexagonal Architecture — all I/O (LLM calls, vector DB, embeddings, file storage) crosses a port Protocol defined in `backend/app/ports/`. The core domain never imports from adapters.

---

## Repository

- **GitHub:** `github.com/mpalian-db/langgraph-chatbot`
- **Main branch:** `main`
- **Active feature branch:** `feat/missing-features` — pushed, not yet merged, no PR open

---

## Local dev requirements

| Service | How to run |
|---|---|
| Qdrant (vector DB) | `docker compose up qdrant` (port 6333) |
| Langfuse (observability) | `docker compose up langfuse` (port 3000) |
| Ollama (local LLM) | Must be running separately — `ollama serve` |
| Required Ollama models | `ollama pull llama3.2:3b` · `ollama pull llama3.1:8b` · `ollama pull nomic-embed-text` |
| Backend | `just dev` or `cd backend && uv run uvicorn app.main:app --reload` |
| Frontend | `cd frontend && npm run dev` (Vite, port 5173) |

**Full stack in one command:** `just dev` (starts Qdrant + Langfuse + uvicorn + Vite)

**Run tests:**
```bash
cd backend && uv run pytest tests/unit/ -q          # unit only (no services needed)
cd backend && uv run pytest tests/integration/ -v   # needs Qdrant + Ollama running
```

---

## Branch status: `feat/missing-features`

All 7 tasks implemented and committed. **Not yet merged into main.**

### What was implemented on this branch

| Commit | What |
|---|---|
| `e16e645` | `citation_coverage` verifier check — deterministic pre-LLM check that measures inline citation coverage against `citation_coverage_min` threshold. Pure `_citation_coverage()` helper + 7 new unit tests. |
| `ca260f5` | Score-based reranking in retrieval node — sorts retrieved chunks by cosine similarity score descending when `rerank=true` in config. |
| `2944809` | `TraceView` React component — standalone component showing each graph node with duration and data fields. `ChatView` now uses it instead of the old inline `Trace` function. |
| `896ccda` | `scripts/sync_edgenotes.py` — bulk sync script that reads from EdgeNotes API (paginated) or a local JSON export and pushes through the webhook endpoint. Config via env vars. |
| `fd87625` | `POST /api/collections/{name}/rebuild` endpoint + `rebuild_index` tool for the tool agent. |
| `10c6786` | Integration tests expanded from 5 to 20 tests — covers chat, streaming, collection CRUD, rebuild, webhook create/delete, secret rejection. |
| `490e2f5` | Frontend lockfile update. |

### What to do with the branch

- Review and merge into `main` via PR, or just `git merge feat/missing-features` locally
- The worktree at `.worktrees/feat-missing-features/` can be removed after merge: `git worktree remove .worktrees/feat-missing-features`

---

## Architecture overview

```
frontend (React + Vite)
    └── POST /api/chat  ────────────────────────────────────┐
                                                             ▼
backend/app/api/routes/chat.py  →  build_graph()  →  graph.ainvoke(state)
                                                             │
                              ┌──────────────────────────────┘
                              ▼
                          [router node]  ─── LLM classifies query
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                   ▼
         chat_agent       retrieval           tool_agent
              │           → answer_gen            │
              │           → verifier              │
              └─────────────┴─────────────────────┘
                                 ▼
                          GraphState.final_answer
```

**GraphState** key fields: `query`, `route`, `collection`, `retrieved_chunks`, `retrieval_scores`, `draft_answer`, `final_answer`, `citations`, `verifier_result`, `retry_count`, `execution_trace`

**Config files:**
- `config/config.toml` — system config (LLM provider, Qdrant URL, embeddings, ingestion, webhooks)
- `config/agents.toml` — per-agent config (models, prompts, thresholds, tool permissions)

**Default models:**
- Router, Chat agent, Tool agent → `llama3.2:3b`
- Answer generation, Verifier → `llama3.1:8b`
- Embeddings → `nomic-embed-text` (768-dim, cosine similarity)

---

## Verifier pipeline (important detail)

The verifier runs three checks in order — each can short-circuit before the expensive LLM call:

1. **Check 0** — refuse immediately if no chunks retrieved
2. **Check 1** (`score_threshold`) — refuse if max retrieval score < 0.55
3. **Check 2** (`citation_coverage`) — revise/refuse if fewer than 80% of answer sentences have inline citations `[chunk-id]` — **deterministic, no LLM call**
4. **Check 3** (`support_analysis`) — LLM-based grounding analysis → returns `accept | revise | refuse`

Outcomes: `accept` → pass through; `revise` → loop back to answer_generation (max 2 retries); `refuse` → return "cannot answer" message.

Key thresholds in `config/agents.toml`:
```toml
score_threshold = 0.55
citation_coverage_min = 0.8
max_retries = 2
```

---

## Known quirks

- **nomic-embed-text cosine similarity** peaks around 0.64 for relevant content (not 0.7+). Score thresholds were tuned down to 0.5 (retrieval) and 0.55 (verifier) based on real data.
- **Qdrant qdrant-client v1.7+** — `.search()` was removed; adapter uses `.query_points()` which returns `QueryResponse` with `.points` field.
- **Collection routing** — the `collection` field lives on `GraphState` (added in a recent fix); do not pass collection as a metadata filter — it is used to select which Qdrant collection to search.
- **`{} or None` pattern** — `state.metadata_filters or None` correctly passes `None` when the dict is empty (empty dict is falsy in Python), which tells Qdrant to skip filter logic.

---

## Next planned work: Notion + Worklog Agent integrations

These were discussed and designed at the end of the last session but **not yet started**. The branch for this work has not been created yet.

### 1. Notion as a knowledge source

**Approach:** Ingestion-only — no new graph node needed. Notion pages become vectors in Qdrant under a dedicated collection (e.g. `notion-docs`). The existing RAG pipeline handles retrieval.

**Files to create:**
- `backend/app/adapters/ingestion/notion.py` — fetches Notion pages via httpx (direct API, not MCP), chunks and embeds them
- `backend/app/api/routes/notion.py` — `POST /api/collections/{collection}/sync-notion` endpoint
- `config/config.toml` — add `[notion]` section with `api_token`, `database_id`
- `scripts/sync_notion.py` — bulk sync script (mirrors `scripts/sync_edgenotes.py` pattern)

**No changes needed** to the graph, router, or any existing node.

**Notion API notes:**
- Use `https://api.notion.com/v1/databases/{id}/query` to list pages
- Use `https://api.notion.com/v1/blocks/{id}/children` to fetch page content recursively
- Auth: `Authorization: Bearer {NOTION_TOKEN}`, `Notion-Version: 2022-06-28`
- Content comes back as block objects (paragraph, heading, bullet_list, etc.) — needs a renderer to flatten to plain text before chunking

### 2. Worklog Assistant agent node

**Approach:** New `worklog` route in the graph. When user asks about their worklogs, hours, or plans, the router sends to a `worklog_agent` node that calls the worklog-assistant's Cloudflare Workers API.

**Decision made:** Option C — the agent can **query and generate plans** but **cannot apply them** (no Jira/Tempo writes through chat). Applying always happens via `worklog apply` CLI.

**Files to create:**
- `backend/app/ports/worklog.py` — `WorklogPort` Protocol
- `backend/app/adapters/worklog/http.py` — HTTP adapter calling `WORKLOG_WORKER_URL` with bearer token
- `backend/app/core/graph/nodes/worklog_agent.py` — new graph node
- `backend/app/core/config/models.py` — add `WorklogAgentConfig` class + field to `AgentsConfig`

**Files to modify:**
- `backend/app/core/graph/graph.py` — add `worklog_agent` node + conditional edge
- `backend/app/api/dependencies.py` — add `get_worklog_port()` + `WorklogDep` alias
- `config/agents.toml` — add `[worklog_agent]` section
- `config/config.toml` — add `[worklog]` section with `worker_url`, `api_key`
- Router prompt in `agents.toml` — add `"worklog"` route description

**Worklog API available (Cloudflare Workers at `WORKLOG_WORKER_URL`):**
- `GET /api/health` — connectivity check
- `GET /api/plans` — list stored plans
- `GET /api/plans/{key}` — fetch a plan by key
- `POST /api/plans` — generate a new plan (involves git + AI analysis)
- Auth: `Authorization: Bearer {WORKLOG_API_KEY}`

**Worklog-assistant project:** `/Users/michalpalian/Desktop/Projects/worklog-assistant`
- MCP server (for Claude Code sessions only, not runtime): `/Users/michalpalian/Desktop/Projects/mcp-servers/worklog-assistant/server.py`
- Env vars needed at runtime: `WORKLOG_WORKER_URL`, `WORKLOG_API_KEY`

---

## Tool registry (current state)

Defined in `backend/app/tools/definitions.py`, executed in `backend/app/core/graph/nodes/tool_agent.py`:

| Tool | What it does |
|---|---|
| `search_collection` | Vector search by query string |
| `list_collections` | List Qdrant collections |
| `get_collection_stats` | Stats for a named collection |
| `rebuild_index` | Drop and recreate a collection |

(Note: `upload_document` and `delete_document` are in `agents.toml` allowed_tools but **not yet implemented** in `_execute_tool()` — these would need to be added if a user wants the tool agent to ingest documents.)

---

## Test counts

- Unit tests: **51 passing** (as of last session)
- Integration tests: 20 tests defined, require Qdrant + Ollama

---

## MCP servers available in Claude Code

Configured in `~/.claude.json`:

| Server | Purpose |
|---|---|
| `worklog-assistant` | Query worklogs, hours, plans (read-only) |
| `notion-mcp-server` | Notion API access |
| `atlassian` | Jira + Confluence |
| `github` | GitHub |
| `tempo-extension` | Log time directly to Tempo |

---

## Session startup checklist (new machine)

```bash
git clone git@github.com:mpalian-db/langgraph-chatbot.git
cd langgraph-chatbot

# Install Python deps
cd backend && uv sync --extra dev && cd ..

# Install frontend deps
cd frontend && npm install && cd ..

# Pull the active feature branch
git fetch origin
git checkout feat/missing-features

# Start services
docker compose up -d qdrant
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Run baseline tests to verify clean state
cd backend && uv run pytest tests/unit/ -q
```
