from __future__ import annotations

import uuid


def chunk_text(
    text: str,
    filename: str,
    collection: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """Split text into overlapping character-based chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )
    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Avoid cutting mid-word when possible
        if end < len(text) and text[end] not in (" ", "\n"):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        segment = text[start:end].strip()
        if segment:
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": segment,
                    "collection": collection,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": len(chunks),
                    },
                }
            )

        if end >= len(text):
            break
        start = end - chunk_overlap

    return chunks
