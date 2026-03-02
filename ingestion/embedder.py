from __future__ import annotations

import base64

import httpx

from config.settings import settings


class JinaV4Embedder:
    """Jina v4 embedding client using a shared httpx.AsyncClient."""

    JINA_API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.jina_api_key
        self._model = settings.jina_model
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def embed_text(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
        dimensions: int = 2048,
    ) -> list[list[float]]:
        """Batch text -> list of dense vectors."""
        payload = {
            "model": self._model,
            "task": task,
            "dimensions": dimensions,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": t} for t in texts],
        }
        data = await self._request(payload)
        return [item["embedding"] for item in data["data"]]

    async def embed_text_query(self, query: str, dimensions: int = 2048) -> list[float]:
        """Single query text -> dense vector using retrieval.query LoRA."""
        results = await self.embed_text([query], task="retrieval.query", dimensions=dimensions)
        return results[0]

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[float]:
        """Single image -> dense vector (2048d)."""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self._model,
            "task": "retrieval.passage",
            "dimensions": 2048,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"image": f"data:{media_type};base64,{b64}"}],
        }
        data = await self._request(payload, timeout=120.0)
        return data["data"][0]["embedding"]

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[list[float]]:
        """Single image -> list of ColBERT token vectors (128d each)."""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self._model,
            "task": "retrieval.passage",
            "dimensions": 128,
            "normalized": True,
            "embedding_type": "float",
            "embedding_type_params": {"output_type": "colbert"},
            "input": [{"image": f"data:{media_type};base64,{b64}"}],
        }
        data = await self._request(payload, timeout=120.0)
        return data["data"][0]["embedding"]

    async def embed_query_multi_vector(self, query: str) -> list[list[float]]:
        """Single query text -> list of ColBERT token vectors (128d)."""
        payload = {
            "model": self._model,
            "task": "retrieval.query",
            "dimensions": 128,
            "normalized": True,
            "embedding_type": "float",
            "embedding_type_params": {"output_type": "colbert"},
            "input": [{"text": query}],
        }
        data = await self._request(payload)
        return data["data"][0]["embedding"]

    async def _request(
        self,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """Send request to Jina API and return parsed JSON."""
        kwargs: dict = {"json": payload}  # type: ignore[type-arg]
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await self._client.post(self.JINA_API_URL, **kwargs)
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Jina API error: {resp.text}",
                request=resp.request,
                response=resp,
            )
        return resp.json()  # type: ignore[no-any-return]
