from __future__ import annotations

from typing import Any

from ollama import AsyncClient


class OllamaLLMAdapter:
    """LLM adapter backed by a local Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        client: AsyncClient | None = None,
    ) -> None:
        self._client = client or AsyncClient(host=base_url)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Complete a prompt and return a structured response."""
        ollama_messages: list[dict[str, Any]] = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        for msg in messages:
            ollama_messages.extend(_to_ollama_messages(msg))

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": {"num_predict": max_tokens},
        }
        if tools:
            kwargs["tools"] = [_to_ollama_tool(t) for t in tools]

        response = await self._client.chat(**kwargs)

        text = response.message.content or ""
        tool_use: list[dict[str, Any]] = []
        if response.message.tool_calls:
            tool_use = [
                {
                    "name": tc.function.name,
                    "input": tc.function.arguments,
                    "id": f"ollama-tool-{i}",
                }
                for i, tc in enumerate(response.message.tool_calls)
            ]

        return {
            "text": text,
            "tool_use": tool_use,
            "stop_reason": "tool_use" if tool_use else "end_turn",
            "usage": {
                "input_tokens": response.prompt_eval_count or 0,
                "output_tokens": response.eval_count or 0,
            },
        }


def _to_ollama_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic-style tool definition to Ollama format.

    Anthropic uses 'input_schema'; Ollama expects 'parameters'.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", tool.get("parameters", {})),
        },
    }


def _to_ollama_messages(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a provider-neutral message dict to one or more Ollama messages.

    Handles two conversions:
    - assistant message with _tool_use -> restore tool_calls field Ollama expects
    - user message with tool_result list content -> one role:tool message per result
    """
    role = msg["role"]
    content = msg.get("content", "")

    if role == "assistant":
        out: dict[str, Any] = {"role": "assistant", "content": content}
        tool_use = msg.get("_tool_use")
        if tool_use:
            out["tool_calls"] = [
                {
                    "function": {
                        "name": t["name"],
                        "arguments": t["input"],
                    }
                }
                for t in tool_use
            ]
        return [out]

    if role == "user" and isinstance(content, list):
        # Anthropic-style batched tool results -> separate role:tool messages
        result = []
        for block in content:
            if block.get("type") == "tool_result":
                result.append({"role": "tool", "content": str(block.get("content", ""))})
        return result if result else [{"role": "user", "content": ""}]

    return [{"role": role, "content": content}]
