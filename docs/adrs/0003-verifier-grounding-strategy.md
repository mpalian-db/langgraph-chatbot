# 0003 -- Verifier grounding strategy

- **Status**: Accepted
- **Date**: 2026-04-30

## Context

The verifier is the project's stated reason to exist as more than a "chat
with docs" demo. Its job is to decide whether the answer-generation node's
draft is actually grounded in the retrieved chunks, and to gate it behind
one of three outcomes: `accept`, `revise`, or `refuse`.

Several design questions arose as the verifier accumulated capability:

1. What does "grounded" mean operationally -- one signal or several?
2. When does revise vs. refuse fire, and how does the answer node know
   what to fix?
3. What's a citation, and how do we tell a real one from a hallucinated
   bracketed token?
4. Should the verifier's score threshold differ from the retrieval
   threshold, and if so, by how much?
5. How do we keep the LLM-based check (`support_analysis`) optional so a
   local-only dev loop still gets useful grounding without paying for
   Anthropic on every chat?

Each decision was made independently, but they form a coherent strategy
worth capturing in one place.

## Decision

### Composable checks, declared in config

`VerifierConfig.checks` is a list of named checks. Each check is
deterministic-or-LLM, runs in a fixed order, and can short-circuit the
verifier with a refuse / revise outcome:

- **`score_threshold`** (deterministic) -- `max(retrieval_scores)` must be
  at least `score_threshold`. Refuses if no chunk reached the bar.
- **`citation_coverage`** (deterministic) -- the fraction of non-trivial
  sentences (>= 5 words) that contain a citation matching a real chunk
  id must be at least `citation_coverage_min`. Below that, revise.
- **`support_analysis`** (LLM) -- one prompt asks the verifier model to
  classify the draft against the evidence and return strict
  `OUTCOME / SCORE / REASON / UNSUPPORTED` text. The parser pulls out
  each field and builds a `VerifierResult`.

The config-as-list shape means an operator can disable an unreliable check
(e.g. `support_analysis` on local Ollama where the structured output is
flaky) without code changes -- they comment it out in `agents.toml`.

### Three outcomes, one revise loop

- `accept` -- pass the draft through as the final answer.
- `revise` -- send the draft back to `answer_generation` along with the
  verifier's reason and `unsupported_claims`. The answer node sees the
  prior draft, the verifier's feedback, and the original evidence; it
  produces a new draft that the verifier re-evaluates. Bounded by
  `max_retries` -- once exhausted, escalate to refuse.
- `refuse` -- return a fixed "I cannot provide a fully supported answer"
  message that includes the verifier's reason for transparency.

The revise loop is wired in `graph.py` via a conditional edge: the
verifier's outcome plus a `final_answer is None` check decide whether
control flows back to `answer_generation` or terminates the graph.

### Citations validated against real chunk ids

A bracketed token in the draft is a real citation only if it equals an
id in `state.retrieved_chunks`. The pipeline:

- `_BRACKET_TOKEN_RE = r"\[([^\[\]\s]+)\](?!\()"` -- captures any non-
  whitespace token in brackets, but skips brackets followed by `(`
  (markdown link syntax `[text](url)`).
- The captured token must match a real chunk id verbatim. Format-based
  matches (e.g. UUID-only regexes) are rejected because the chunker emits
  uuid4 (36 chars, hyphens) while the Notion sync emits sha256 hex
  prefixes (32 chars, no hyphens). Validating against the actual id set
  is the only correct policy.

This means a hallucinated citation like `[fake-id]` increments neither
`citation_coverage` nor populates `state.citations` -- it's silently
dropped as if it weren't there.

### Threshold alignment between retrieval and verifier

`retrieval.score_threshold == verifier.score_threshold` (both 0.55 by
default). Earlier these were 0.5 and 0.55 respectively, which created a
dead band: a chunk scoring 0.51 passed retrieval but caused the verifier
to refuse on score alone -- wasting an answer-generation LLM call on an
inevitable refusal.

The new invariant: if a chunk was good enough to retrieve, it's good
enough to ground. The verifier's `score_threshold` check now mostly
serves the empty-retrieval path (`Check 0`) where no chunks came back at
all. A `test_production_thresholds_align_retrieval_with_verifier`
config-level invariant test pins this in `test_config.py`.

### Citation coverage threshold tuned for local Ollama

`citation_coverage_min = 0.5` (not the spec's 0.8). With local Ollama
answer-generation models, citing ~80% of sentences is unreliable; setting
the bar that high turns most chats into `revise -> refuse` cycles. 0.5
is the empirically-found sweet spot: low enough that an "explaining"
answer with one citation per claim passes, high enough that a 0%-cited
"hallucinated paragraph" fails. Comment in `agents.toml` documents the
intent and notes 0.8 as the target with a more capable model.

## Alternatives considered

1. **Single LLM-judge for everything**. A single LLM call that gets the
   evidence, the draft, and asks "is this grounded?" Simple,
   one-pass. Rejected because:
   - Locally-run models can't reliably produce structured output, so
     there's nothing to parse deterministically.
   - It folds three different signals (retrieval quality, citation
     hygiene, semantic support) into one judgment that's hard to
     diagnose when it goes wrong.
   - Operator can't disable the unreliable check without disabling the
     whole verifier.

2. **No revise loop -- just accept or refuse**. Simpler graph, fewer
   edges, no `max_retries` bookkeeping. Rejected because:
   - Most "wrong" drafts have a fixable problem (missing citation,
     under-supported claim) that a second pass can address.
   - Without revise, citation_coverage below threshold turns into
     refusal even on otherwise-good answers. Conversation quality
     suffers.
   - The bounded loop with `max_retries=2` is cheap (at most three
     answer-generation calls per RAG path) and rare in practice.

3. **Looser citation matching (regex format only, no chunk-id check)**.
   The original code used a UUID-shaped regex. Rejected because:
   - It silently dropped citations from Notion-sourced answers (sha256
     prefix ids don't match a UUID regex), so `state.citations` was
     empty even when the LLM had cited correctly.
   - It accepted hallucinated `[fake-id]` tokens that fit the regex
     shape, inflating `citation_coverage` without grounding the answer.

4. **Separate score thresholds for retrieval and verifier with the
   verifier stricter**. The original config had this. Rejected for the
   dead-band reason in the Decision section.

## Consequences

**Easier**:
- Each check is independently testable; the verifier unit test suite
  (`test_verifier.py`) is the most thorough in the project precisely
  because each check is a small pure function.
- An operator can opt out of `support_analysis` without code changes,
  matching the "config controls behaviour, not topology" principle.
- The revise loop creates a natural place for graph-level integration
  tests: `test_graph_rag_path_revises_when_citation_coverage_low` runs
  the full RAG path with a low-citation first draft and asserts the
  second draft gets accepted -- end-to-end pinning of the loop.

**Harder**:
- The verifier code has more moving parts than a single LLM judge would.
  `_citation_coverage`, `_extract_citations` (in `answer_generation.py`),
  `_parse_verifier_response`, the threshold checks, and the revise/refuse
  branching all need to stay coherent.
- The composable-checks pattern is a tax on the operator: they have to
  understand what each check guarantees to know which to enable. The
  comments in `agents.toml` explain the trade-offs but it's still more
  to learn than "verifier on/off".
- Tuning thresholds is a mix of mechanical alignment (retrieval ==
  verifier) and judgement calls (`citation_coverage_min`). The right
  values change with model capability; the comments record the current
  trade-off but the values themselves will need revisiting if/when
  Anthropic models become the default for answer generation.
