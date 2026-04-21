from __future__ import annotations

from app.ports.vectorstore import CollectionPort


async def rebuild_collection(
    collection_store: CollectionPort,
    name: str,
    vector_size: int = 768,
) -> None:
    """Delete and recreate a collection, discarding all existing vectors."""
    try:
        await collection_store.delete_collection(name)
    except Exception:
        pass
    await collection_store.create(name, vector_size)
