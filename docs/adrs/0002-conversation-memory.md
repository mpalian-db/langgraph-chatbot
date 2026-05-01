# 0002 -- Conversation memory architecture

- **Status**: Accepted
- **Date**: 2026-04-29

## Context

The chat endpoint was originally stateless: every `POST /api/chat` ran
the full graph from scratch with the user's single query, returned a
response, and forgot. Multi-turn references ("what about the second
one?") couldn't work because there was no second turn to reference.

For a chat application, this isn't a small gap -- it's a fundamental
missing capability. We needed to add conversation memory in a way that:

- Persists across process restarts (local-first dev includes
  `just dev` cycles).
- Doesn't bloat the prompt as conversations grow.
- Stays consistent with the project's hexagonal architecture.
- Can later run on Cloudflare Durable Objects or D1 without rewriting
  the chat code.

## Decision

The feature has three layered design choices: storage shape, eviction
strategy, and LLM choice.

### Storage: separate summary table (B)

A `conversation_summaries` table keyed by `conversation_id` with one row
per conversation, separate from the existing `conversation_turns`
append-only log. The summary row carries a `summarised_through_turn_id`
boundary; reads return the summary plus all turns past that boundary.

`ConversationReaderPort` and `ConversationWriterPort` are split per the
project rule "never aggregate read, write, and mutation behind one
interface". The writer adds an atomic `append_pair(user, assistant)` so
a chat round can never be left half-persisted.

### Eviction: lazy on load (alpha) with rolling window + summary

When `len(post-boundary turns) > summarise_threshold` (default 20), the
service folds everything except the last `keep_recent` turns (default
10) into a new summary that integrates the prior summary if any.
Triggered during the chat route's history load -- no background job,
no separate operator command.

### LLM: configurable per-node provider (iii)

`SummariserConfig` has its own `provider` override, model, prompt, and
`max_tokens`, reusing the per-node provider machinery from
[ADR 0001](./0001-per-node-llm-provider.md). Defaults to the system LLM
(Ollama locally). One config flip makes it use Anthropic.

### Resilience: best-effort summarisation

If the summariser LLM call or the storage write fails, the service logs
a warning and returns the unsummarised history. The chat request still
completes; storage stays unchanged so the next request retries the
trigger. Summarisation is an enhancement to chat, never a hard
dependency.

## Alternatives considered

### Storage shapes

- **A: Single table with `role = "summary"`** -- one source of truth,
  but the existing CHECK constraint rejects the new role. Either a
  schema migration or a recreate -- friction with existing local
  databases. Rejected for that reason.
- **C: In-memory only, recomputed each turn** -- defeats the
  token-saving point of summarisation by paying the LLM cost on every
  request. Rejected.

### Eviction strategies

- **Sliding window only (no summary)** -- v1 of conversation memory
  shipped this. Simpler but loses distant context abruptly when the
  window slides off it. Accepted as the v1 baseline; this ADR is the
  v2 enhancement.
- **Token budget instead of turn count** -- smarter on prompt-size
  pressure but doesn't solve the UX problem (people remember what they
  said an hour ago). Rejected as a partial solution; could complement
  the rolling summary later.
- **Background summarisation (beta)** -- avoids adding latency to the
  read path, but introduces eventual consistency, scheduling, and
  failure-handling complexity that's not justified for a learning
  project. Rejected for v2.

### LLM choices for summariser

- **System default (i)** -- cheap, but Ollama summaries can be
  unreliable on long inputs. Reasonable for a "best-effort"
  enhancement which is what we settled on.
- **Anthropic Haiku (ii)** -- reliable, paid. Same tradeoff as the
  verifier's support_analysis.
- **Configurable per-node (iii)** -- chosen. Reuses the existing
  machinery; operator picks the trade-off.

## Consequences

**Easier**:
- Multi-turn references in chat work locally with no extra
  infrastructure (single SQLite file under `data/conversations.sqlite`).
- The Cloudflare port-and-adapter pattern still applies: a future
  `D1ConversationStore` or `DOConversationStore` adapter can replace
  SQLite with no changes to chat logic.
- Summary is a first-class part of `GraphState`, so any node can opt
  into using it later.

**Harder**:
- Two HTTP roundtrips per chat-route request when summarisation
  triggers (read + LLM call + upsert + chat). Acceptable because the
  trigger only fires every ~20 turns.
- Streaming endpoint has a TTFB gap: summarisation runs *before* the
  event stream starts emitting. Acceptable for v1; a future enhancement
  could move the call inside the generator with a "preparing memory"
  event.
- The `lru_cache` on `get_conversation_store` is not strictly
  thread-safe at cold-start; under concurrent first-request
  initialisation two stores may briefly be created. Benign for
  file-backed SQLite (both connections address the same file).
