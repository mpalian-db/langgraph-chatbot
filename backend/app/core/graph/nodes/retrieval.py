from __future__ import annotations

import time
from typing import Any

from app.core.config.models import RetrievalConfig
from app.core.graph.state import GraphState
from app.core.models.types import TraceEntry
from app.ports.embedding import EmbeddingPort
from app.ports.vectorstore import VectorStorePort


async def run(
    state: GraphState,
    *,
    config: RetrievalConfig,
    vectorstore: VectorStorePort,
    embedding: EmbeddingPort,
) -> dict[str, Any]:
    start = time.monotonic()

    query_text = state.retrieval_query or state.query
    vectors = await embedding.embed([query_text])
    if not vectors:
        raise RuntimeError(f"Embedding adapter returned no vectors for query: {query_text!r}")
    query_vector = vectors[0]

    collection = state.collection or config.default_collection
    chunks = await vectorstore.search(
        query_vector=query_vector,
        top_k=config.top_k,
        collection=collection,
        filters=state.metadata_filters or None,
        score_threshold=config.score_threshold,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return {
        "retrieved_chunks": chunks,
        "retrieval_scores": [c.score for c in chunks],
        "execution_trace": state.execution_trace
        + [
            TraceEntry(
                node="retrieval",
                duration_ms=elapsed_ms,
                data={"chunks_retrieved": len(chunks), "collection": collection},
            )
        ],
    }
