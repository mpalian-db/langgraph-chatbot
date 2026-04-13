from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]: ...
