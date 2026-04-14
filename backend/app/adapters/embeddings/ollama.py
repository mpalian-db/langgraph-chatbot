from __future__ import annotations

from ollama import AsyncClient


class OllamaEmbeddingAdapter:
    """Embedding adapter backed by a local Ollama instance."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        client: AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._client = client or AsyncClient(host=base_url)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        response = await self._client.embed(model=self._model, input=texts)
        return [list(row) for row in response.embeddings]
