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
}


def get_tools_for_agent(allowed_tools: list[str]) -> list[dict]:
    return [TOOL_REGISTRY[name] for name in allowed_tools if name in TOOL_REGISTRY]
