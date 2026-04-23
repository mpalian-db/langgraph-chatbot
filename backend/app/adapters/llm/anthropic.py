from __future__ import annotations

from typing import Any

import anthropic


class AnthropicLLMAdapter:
    """LLM adapter backed by Anthropic API."""

    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = client or anthropic.AsyncAnthropic()

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
        # Strip private keys (e.g. _tool_use) used for cross-provider message routing.
        clean_messages = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": clean_messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_use_blocks = [
            {"name": b.name, "input": b.input, "id": b.id}
            for b in response.content
            if b.type == "tool_use"
        ]

        return {
            "text": text_blocks[0] if text_blocks else "",
            "tool_use": tool_use_blocks,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
