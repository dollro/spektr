from __future__ import annotations

import asyncio
import base64
import logging
import math

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from ingestion.embedder import TokenBucket

logger = logging.getLogger(__name__)

_TASK_MAP = {"passage": "retrieval.passage", "query": "retrieval.query"}


def _is_retryable(exc: BaseException) -> bool:
    """Return True for HTTP 429/5xx and transient network errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError))


_TILE_SIZE = 28
_TOKENS_PER_TILE = 10
_FALLBACK_TOKENS = 2000


def _estimate_image_tokens(data_uri: str) -> float:
    """Estimate Jina v4 token cost for an image.

    Jina v4 uses a Qwen2.5-VL vision encoder that tiles images
    into 28x28 patches with ~10 tokens per tile.
    """
    comma = data_uri.find(",")
    if comma < 0:
        return _FALLBACK_TOKENS

    b64_data = data_uri[comma + 1 :]
    try:
        import io

        from PIL import Image

        img_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        tiles_w = math.ceil(w / _TILE_SIZE)
        tiles_h = math.ceil(h / _TILE_SIZE)
        return float(tiles_w * tiles_h * _TOKENS_PER_TILE)
    except Exception:
        return _FALLBACK_TOKENS


class JinaV4Embedder:
    """Jina v4 embedding client using a shared httpx.AsyncClient."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.jina_api_key
        self._embeddings_url = f"{settings.jina_api_url}/v1/embeddings"
        self._model = settings.jina_model
        self._dimensions = settings.jina_dense_dimensions
        self._max_concurrent = settings.jina_max_concurrent
        self._rpm_limiter = TokenBucket(
            tokens_per_sec=settings.jina_rpm / 60.0,
            burst=settings.jina_max_concurrent,
        )
        self._tpm_limiter = TokenBucket(
            tokens_per_sec=settings.jina_tpm / 60.0,
            burst=settings.jina_tpm,
        )
        # Loop-bound resources — recreated when the event loop changes
        # because run_async() creates a new loop per call.
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        self._tokens_used: float = 0.0

    def _ensure_loop_resources(self) -> None:
        """Recreate loop-bound async resources when the event loop changes.

        ``run_async`` calls ``asyncio.run()`` which creates a fresh loop each
        time, so the httpx client and semaphore must be re-created to avoid
        "bound to a different event loop" errors.
        """
        loop = asyncio.get_running_loop()
        if self._bound_loop is not loop:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0),
            )
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            self._bound_loop = loop

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()

    @property
    def tokens_used(self) -> float:
        """Total estimated tokens consumed since last reset."""
        return self._tokens_used

    def reset_token_counter(self) -> None:
        """Reset the token consumption counter to zero."""
        self._tokens_used = 0.0

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dimensions

    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,
    ) -> list[list[float]]:
        """Batch text -> list of dense vectors.

        Automatically splits large lists into sub-batches of
        ``settings.jina_batch_size`` to stay within API token limits.

        When *late_chunking* is ``True`` the entire list is sent as a
        single API call (no batching) so the model can apply
        contextualised late-chunking across all texts.
        """
        if late_chunking:
            return await self._embed_text_batch(
                texts,
                task,
                dimensions,
                late_chunking=True,
            )

        batch_size = settings.jina_batch_size
        if len(texts) <= batch_size:
            return await self._embed_text_batch(texts, task, dimensions)

        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(await self._embed_text_batch(batch, task, dimensions))
        return results

    async def _embed_text_batch(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,
    ) -> list[list[float]]:
        """Send a single embed_text API call for one batch."""
        jina_task = _TASK_MAP.get(task, task)
        dims = dimensions if dimensions is not None else self._dimensions
        payload: dict[str, object] = {
            "model": self._model,
            "task": jina_task,
            "dimensions": dims,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": t} for t in texts],
        }
        if late_chunking:
            payload["late_chunking"] = True
        data = await self._request(payload)
        return [item["embedding"] for item in data["data"]]

    async def embed_text_query(self, query: str, dimensions: int | None = None) -> list[float]:
        """Single query text -> dense vector."""
        dims = dimensions if dimensions is not None else self._dimensions
        results = await self.embed_text([query], task="query", dimensions=dims)
        return results[0]

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[float]:
        """Single image -> dense vector (2048d)."""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self._model,
            "task": _TASK_MAP["passage"],
            "dimensions": self._dimensions,
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
            "task": _TASK_MAP["passage"],
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
            "task": _TASK_MAP["query"],
            "dimensions": 128,
            "normalized": True,
            "embedding_type": "float",
            "embedding_type_params": {"output_type": "colbert"},
            "input": [{"text": query}],
        }
        data = await self._request(payload)
        return data["data"][0]["embedding"]

    @staticmethod
    def _estimate_tokens(payload: dict) -> float:  # type: ignore[type-arg]
        """Estimate token count from a Jina API payload.

        Text: len(text) / 4 (standard heuristic).
        Images: Jina v4 tiles into 28x28 patches at ~10 tokens/tile.
        """
        total = 0.0
        for item in payload.get("input", []):
            if "text" in item:
                total += len(item["text"]) / 4.0
            elif "image" in item:
                total += _estimate_image_tokens(item["image"])
        return max(total, 1.0)

    async def _request(
        self,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """Send request with rate limiting + concurrency control."""
        self._ensure_loop_resources()
        estimated_tokens = self._estimate_tokens(payload)
        self._tokens_used += estimated_tokens
        await self._rpm_limiter.acquire()
        await self._tpm_limiter.acquire(estimated_tokens)
        async with self._semaphore:  # type: ignore[union-attr]
            return await self._request_with_retry(payload, timeout)

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(_is_retryable),
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
        resp = await self._client.post(  # type: ignore[union-attr]
            self._embeddings_url, **kwargs
        )
        if resp.status_code != 200:
            exc = httpx.HTTPStatusError(
                f"Jina API error: {resp.text}",
                request=resp.request,
                response=resp,
            )
            code = resp.status_code
            if code == 429:
                retry_after = float(resp.headers.get("Retry-After", 20))
                self._rpm_limiter.pause(retry_after)
                self._tpm_limiter.pause(retry_after)
            if code != 429 and code < 500:
                raise httpx.HTTPStatusError(
                    f"Jina API client error ({code}): {resp.text}",
                    request=resp.request,
                    response=resp,
                ) from exc
            raise exc
        return resp.json()  # type: ignore[no-any-return]
