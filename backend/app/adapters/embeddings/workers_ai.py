from __future__ import annotations

import httpx


class WorkersAIEmbeddingAdapter:
    """Embedding adapter backed by Cloudflare Workers AI."""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        model: str = "@cf/baai/bge-small-en-v1.5",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._account_id = account_id
        self._api_token = api_token
        self._model = model
        self._http_client = http_client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        own_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}",
            headers={"Authorization": f"Bearer {self._api_token}"},
        )
        try:
            response = await client.post(
                f"/ai/run/{self._model}",
                json={"text": texts},
            )
            response.raise_for_status()
            data = response.json()
            return data["result"]["data"]
        finally:
            if own_client:
                await client.aclose()
