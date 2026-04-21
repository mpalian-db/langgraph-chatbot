from __future__ import annotations

TOOL_REGISTRY: dict[str, dict] = {
    "search_collection": {
        "name": "search_collection",
        "description": "Search for documents in a collection by query string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "collection": {"type": "string", "description": "Collection name"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query", "collection"],
        },
    },
    "list_collections": {
        "name": "list_collections",
        "description": "List all available document collections.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_collection_stats": {
        "name": "get_collection_stats",
        "description": "Get statistics for a named collection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["collection"],
        },
    },
    "rebuild_index": {
        "name": "rebuild_index",
        "description": (
            "Delete and recreate a collection, discarding all existing vectors. "
            "Use when re-ingestion is required after a schema or embedding change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name to rebuild"},
                "vector_size": {
                    "type": "integer",
                    "description": "Vector dimensionality (default 768 for nomic-embed-text)",
                    "default": 768,
                },
            },
            "required": ["collection"],
        },
    },
    "upload_document": {
        "name": "upload_document",
        "description": (
            "Ingest a plain-text document into a collection. "
            "Chunks the text, embeds it, and upserts into the vector store."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Target collection name"},
                "filename": {"type": "string", "description": "Source filename for metadata"},
                "text": {"type": "string", "description": "Plain text content to ingest"},
            },
            "required": ["collection", "filename", "text"],
        },
    },
    "delete_document": {
        "name": "delete_document",
        "description": "Delete specific chunks from a collection by their chunk IDs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of chunk IDs to delete",
                },
            },
            "required": ["collection", "ids"],
        },
    },
}


def get_tools_for_agent(allowed_tools: list[str]) -> list[dict]:
    return [TOOL_REGISTRY[name] for name in allowed_tools if name in TOOL_REGISTRY]
