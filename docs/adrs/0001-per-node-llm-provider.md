# 0001 -- Per-node LLM provider override

- **Status**: Accepted
- **Date**: 2026-04-29

## Context

Originally a single `LLMPort` was injected into every node of the graph.
The provider (Ollama for local dev, Anthropic for production) was a
system-wide setting in `config/config.toml`. Every node used the same
provider; only the `model` string varied per node.

The verifier's `support_analysis` check exposed a real friction with this
arrangement. It needs reliable structured output (`OUTCOME: ...`,
`SCORE: ...`, `REASON: ...`, `UNSUPPORTED: ...`). Local Ollama models
produce that output unreliably -- the verifier was effectively disabled
when running locally. But forcing the *whole* graph to Anthropic just to
get a working verifier means paying API costs for every chat agent and
router decision, which defeats the project's local-first ethos.

We needed a way for the verifier (and any future node with similar
needs) to use Anthropic while the rest of the graph stayed on Ollama,
without architectural surgery every time we wanted to flip a node.

## Decision

Each agent config section in `agents.toml` may set
`provider = "ollama" | "anthropic"`. `dependencies.py` builds a
`get_llm_registry()` that exposes every available provider as a
`dict[str, LLMPort]` -- always includes `ollama`, includes `anthropic`
only when `ANTHROPIC_API_KEY` is set.

`build_graph()` takes the registry plus a `default_provider` string and
calls `_resolve_llm(agent_cfg, llms, default_provider)` per node, picking
the right port for each. A node that doesn't specify a provider falls
back to the default. A node that requests an unregistered provider
raises `ValueError` immediately.

`validate_llm_providers()` runs at FastAPI startup (in `main.py`'s
lifespan) and walks every LLM-using agent against the registry. A
misconfigured override crashes the app at boot rather than 500'ing on
the first chat.

## Alternatives considered

1. **Same provider for everything; flip to Anthropic when you want
   quality** -- no code change needed, but you can't have a hybrid setup.
   Either you pay Anthropic costs for routine chat or you have an
   unreliable verifier locally. Rejected because hybrid is exactly what
   the project needs.
2. **Aggressive Ollama prompt tuning for the verifier** (few-shot,
   reject malformed outputs as `refuse`). Doable as research but won't
   beat Anthropic on structured-output reliability. Rejected as a
   permanent strategy; might revisit as a v2 enhancement.
3. **Per-node LLM provider via the registry** (this ADR). Real
   architectural improvement. Reuses the per-node `model` field's mental
   model. Adds 30-ish lines across `models.py`, `dependencies.py`, and
   `graph.py`.

## Consequences

**Easier**:
- Production-grade verifier with a one-line config flip:
  `provider = "anthropic"` under `[verifier]`.
- Future nodes can independently choose providers (e.g. summariser
  picked up the same machinery for free in ADR 0002).
- Misconfiguration is loud: startup fails with a clear error pointing at
  the offending agent and the unregistered provider.

**Harder**:
- One more thing to teach a new contributor: the registry, the
  resolver, and where overrides live.
- Test fixtures must pass `llms={...}` and `default_provider=...` to
  `build_graph` instead of a single LLM. The shared `mock_llm_registry`
  fixture in `conftest.py` makes this cheap.
- Running the integration test suite against the real `agents.toml`
  requires `ANTHROPIC_API_KEY` if any node specifies
  `provider = "anthropic"`. In practice the default `agents.toml` keeps
  every node on Ollama; the comment in `[verifier]` documents the
  override as a one-line opt-in.
