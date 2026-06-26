# 0004 -- Atomic conversation persistence

- **Status**: Accepted
- **Date**: 2026-04-30

## Context

Each chat round produces two related rows in the conversation store: a
user message and an assistant reply. The first version of the
conversation memory writer exposed only `append(conversation_id, role,
content)`, and the chat route called it twice in sequence:

```python
await conversation_writer.append(conversation_id, "user", body.query)
await conversation_writer.append(conversation_id, "assistant", answer)
```

A Codex review on the conversation-memory feature flagged the gap: a
cancellation between the two awaits would leave the user message
persisted with no reply attached. The next history load would hand the
chat agent an "orphan user turn" -- a half-conversation -- and the model
would respond as if it were the first message in a new exchange, ignoring
the orphan.

This isn't theoretical: the streaming endpoint persists turns inside an
async generator that can be cancelled by the client at any await point.
Even the synchronous endpoint can be torn down by the ASGI server during
shutdown.

## Decision

Add `append_pair(conversation_id, user_content, assistant_content)` to
`ConversationWriterPort` as a first-class operation. The contract is in
the port docstring:

> A chat round is one logical operation: the user query and the
> assistant response are recorded together, or not at all. Two separate
> append() calls would leave the conversation half-written if the
> process were cancelled between them, producing a "user message with
> no reply" ghost row that would corrupt future history loads.
>
> Implementations must commit both rows in a single transaction.

The SQLite adapter wraps both inserts in a single transaction with a
`try / commit / except: rollback / raise` block. Both routes (`/chat`
and `/chat/stream`) call `append_pair` at the persistence point.

The original `append` method stays in the port -- it's still useful for
the summariser path and for one-off writes -- but the chat route never
uses it for the round-persist case.

The streaming route persists BEFORE yielding the final result event so
a client disconnect after persistence still keeps the data. Persist
order: `await append_pair(...)` first, then `yield {"event": "result",
...}`. If the client disconnects between persist and yield, the
conversation is intact server-side; the client just doesn't see the
response (and would on reconnect see it via the introspection endpoint).

## Alternatives considered

1. **Keep `append()` only, document the partial-write risk**. Cheapest
   in code but moves the burden onto every caller. Rejected because the
   chat round is the dominant use case and forgetting the contract is
   easy.

2. **Make `append()` itself atomic for a list of turns**:
   `append(conversation_id, [(role, content), ...])`. More general but
   requires the caller to construct a list every time. The two-arg
   convenience method `append_pair` is more direct for the chat-round
   shape that's actually used.

3. **Introduce a transaction context manager** -- `async with
   writer.transaction(): writer.append(...); writer.append(...)`. Most
   flexible but adds a new abstraction (the transaction handle) and
   couples callers to transactional semantics. Rejected as
   over-engineering for the one logical operation we actually need.

4. **Defer persistence until graph completion is acknowledged by the
   client**. Would require a confirmation round-trip from the client
   before persisting -- new protocol, new failure modes, and still no
   atomic guarantee between the user and assistant rows. Rejected.

## Consequences

**Easier**:
- The chat route's persistence step is a single line, with the atomic
  guarantee baked into the port contract. No reviewer has to spot the
  two-await race.
- The unit test surface is small: `test_append_pair_is_atomic_on_failure`
  triggers a NOT NULL violation on the second insert and asserts the
  first row was rolled back. `test_append_pair_persists_both_rows_in_order`
  pins the success path.
- A future Cloudflare adapter (Durable Objects, D1) inherits the same
  contract. D1 supports transactions; DOs can hold both writes in one
  storage transaction.

**Harder**:
- One more port method to learn for new contributors. Mitigated by the
  port docstring spelling out the invariant.
- The `append_pair` SQLite implementation is more code than the simple
  `append`; the rollback path needs explicit handling. Worth the cost
  because the alternative is silent corruption.

## Notes on streaming-side concerns

Two related decisions live in the chat route rather than the port:

- **Persist before yield** in `/chat/stream`: the result event yields
  AFTER `append_pair` completes. A client disconnect after persistence
  still keeps the data; the trade-off is duplicate-on-retry, which
  should be solved with an idempotency key if it matters, not by
  delaying persistence.
- **`conversation_id` is passed to `_state_to_response` as a parameter,
  not read from `state.conversation_id`**: LangGraph's per-node
  on_chain_end deltas don't carry the full state, so reconstructing
  `GraphState(**delta)` would default conversation_id to None. Both
  routes pin the id in request scope and pass it explicitly.

Neither belongs in the port contract -- they're correctness invariants
of the route handler. They're documented in commit messages and pinned
by route-level tests, not duplicated here.
