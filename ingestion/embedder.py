from __future__ import annotations

import asyncio
import base64
import logging
import time as _time

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

logger = logging.getLogger(__name__)


def _is_retryable_status(exc: BaseException) -> bool:
    """Return True for HTTP 429 and 5xx errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class _TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, tokens_per_sec: float, burst: int) -> None:
        self._rate = tokens_per_sec
        self._max = float(burst)
        self._tokens = float(burst)
        self._last = _time.monotonic()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        while True:
            now = _time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._max, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)


class JinaV4Embedder:
    """Jina v4 embedding client using a shared httpx.AsyncClient."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.jina_api_key
        self._embeddings_url = f"{settings.jina_api_url}/v1/embeddings"
        self._model = settings.jina_model
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0),
        )
        self._semaphore = asyncio.Semaphore(settings.jina_max_concurrent)
        self._rpm_limiter = _TokenBucket(
            tokens_per_sec=settings.jina_rpm / 60.0,
            burst=settings.jina_max_concurrent,
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
        """Single query text -> dense vector."""
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
        """Single image -> ColBERT token vectors (128d each)."""
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
        """Single query text -> ColBERT token vectors (128d)."""
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
        """Send request to Jina API with rate limiting + concurrency control."""
        await self._rpm_limiter.acquire()
        async with self._semaphore:
            return await self._request_with_retry(payload, timeout)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(_is_retryable_status),
        before_sleep=lambda rs: logger.warning(
            "Jina API retry attempt %d after %s",
            rs.attempt_number,
            rs.outcome.exception(),
        ),
    )
    async def _request_with_retry(
        self,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """Send request with retry on 429/5xx errors."""
        kwargs: dict = {"json": payload}  # type: ignore[type-arg]
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await self._client.post(self._embeddings_url, **kwargs)
        if resp.status_code != 200:
            exc = httpx.HTTPStatusError(
                f"Jina API error: {resp.text}",
                request=resp.request,
                response=resp,
            )
            code = resp.status_code
            if code != 429 and code < 500:
                raise httpx.HTTPStatusError(
                    f"Jina API client error ({code}): {resp.text}",
                    request=resp.request,
                    response=resp,
                ) from exc
            raise exc
        return resp.json()  # type: ignore[no-any-return]
