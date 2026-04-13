from __future__ import annotations

from app.ingestion.chunker import chunk_text


def test_short_text_produces_one_chunk():
    chunks = chunk_text("Hello world", "test.md", "docs", chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world"
    assert chunks[0]["metadata"]["filename"] == "test.md"
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_long_text_produces_multiple_chunks():
    text = "A" * 1200
    chunks = chunk_text(text, "long.txt", "docs", chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2


def test_each_chunk_has_unique_id():
    text = "A" * 1200
    chunks = chunk_text(text, "test.txt", "docs", chunk_size=500, chunk_overlap=50)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_have_collection_set():
    chunks = chunk_text("some text", "f.md", "my-collection")
    assert all(c["collection"] == "my-collection" for c in chunks)
