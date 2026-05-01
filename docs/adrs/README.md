# Architecture Decision Records

Each ADR captures one architectural decision: the context, the options
considered, the choice, and the consequences. ADRs are immutable once
accepted -- if a decision is later reversed, write a new ADR that
references the prior one rather than editing it.

## Format

Numbered files like `NNNN-short-title.md`, with the front matter:

- **Status**: Accepted | Superseded by NNNN | Deprecated
- **Date**: YYYY-MM-DD
- **Context**: what problem are we solving, and what constraints apply
- **Decision**: what we chose
- **Alternatives considered**: the realistic competing options
- **Consequences**: what becomes easier and harder as a result

## Index

- [0001 -- Per-node LLM provider override](./0001-per-node-llm-provider.md)
- [0002 -- Conversation memory architecture](./0002-conversation-memory.md)
- [0003 -- Verifier grounding strategy](./0003-verifier-grounding-strategy.md)
- [0004 -- Atomic conversation persistence](./0004-atomic-conversation-persistence.md)
