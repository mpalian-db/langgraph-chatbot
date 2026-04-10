# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Local-first multi-agent RAG chat application. A learning/experimentation platform for graph-based agent orchestration, retrieval-augmented generation, grounded response verification, configurable agents, and tool calling. Uses the Anthropic API. Designed for local development first with a future Cloudflare deployment path.

Full design spec: `multi-agent-rag-project-description.md`

## Planned stack

- **Orchestration:** LangGraph (graph-based workflow)
- **LLM provider:** Anthropic API (Messages API, tool use, Python SDK)
- **Vector store:** Qdrant (local Docker container for dev, Qdrant Cloud or self-hosted for prod)
- **Configuration:** TOML files for agent behaviour, routing rules, tool permissions, thresholds
- **API:** FastAPI (async)
- **Frontend:** Static SPA (framework TBD), deployed to Cloudflare Pages
- **Language:** Python (backend), TypeScript/JavaScript (frontend)
- **Deployment target:** Cloudflare (Pages for frontend, Workers for API/backend)

## Architecture

Hexagonal Architecture: inbound adapters (API/CLI) -> core domain (graph, agents, verification) -> outbound adapters (Anthropic SDK, vector DB, file storage).

### Graph structure (v1)

Three execution paths through a shared LangGraph graph:

- **Direct chat:** `user_query -> router -> chat_agent -> final_response`
- **RAG:** `user_query -> router -> retrieval_agent -> answer_node -> verification_agent -> final_response`
- **Failed grounding:** same as RAG but verification returns `revise` or `refuse`

### Node roles

| Node | Responsibility |
|---|---|
| Router / orchestrator | Classify intent, decide path (direct chat vs RAG vs tool) |
| Chat agent | Handle general conversation, present final responses |
| Retrieval agent | Vector search, metadata filtering, reranking, build evidence bundle |
| Answer generation | Produce draft answer from retrieved chunks with citations |
| Verification / grounding | Check answer-to-evidence support; decide accept/revise/refuse |
| Tool-using agents | Interact with structured tools (ingestion, system inspection, utilities) |

### Graph state

State flowing through the graph includes: original query, route decision, rewritten retrieval query, retrieved chunks with scores, metadata filters, draft answer, verifier result and grounding status, citations, tool calls made, and execution trace.

### TOML configuration

Agents are configured via TOML: enabled flag, model selection, prompts, routing rules, tool permissions, retrieval/verification thresholds, retry policies, recursion limits, collection bindings, logging verbosity, environment mode.

### Verification approach

Grounding verification goes beyond cosine similarity. Combine: retrieval score thresholds, semantic similarity, answer-to-evidence support checks, citation coverage, unsupported claim detection. Outcomes: `accept`, `revise`, `refuse`.

## Build and run commands

*(To be updated once the project is scaffolded)*

```bash
# TBD: install dependencies
# TBD: run the local server
# TBD: run tests
# TBD: run a single test
# TBD: lint / format
```

## Conventions

- British English in all documentation and comments.
- No emojis anywhere (docs, commits, comments, output).
- Keep README concise; detailed docs go under `docs/`.
- Split interfaces into single-concern protocols. Never aggregate read, write, and mutation behind one interface.
- Pure async functions for graph stages; routing predicates as pure closures with config injected via partial application.
- Stages declare `requires`/`produces`; a runner manages context.
- Isolate side effects (Anthropic API calls, vector DB, file I/O) at the boundaries behind ports.
