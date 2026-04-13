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
        ollama_messages.extend(messages)

        kwargs: dict[str, Any] = {"model": model, "messages": ollama_messages}
        if tools:
            kwargs["tools"] = tools

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
