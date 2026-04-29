from __future__ import annotations

import asyncio
import logging

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


def _is_retryable(exc: BaseException) -> bool:
    """Return True for HTTP 429/5xx and transient network errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError))


class OpenRouterEmbedder:
    """OpenRouter embedding client (OpenAI-compatible /v1/embeddings).

    Default model: google/gemini-embedding-2-preview. Any OpenRouter-served
    embedding model can be selected via OPENROUTER_MODEL.

    Image and ColBERT multi-vector raise NotImplementedError — OpenRouter's
    embeddings endpoint is text-only. Set IMAGE_EMBED_STRATEGY=none, or use
    jina/voyage when image embeddings are required.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.openrouter_api_key
        self._url = f"{settings.openrouter_api_url}/v1/embeddings"
        self._model = settings.openrouter_model
        self._dimensions = settings.openrouter_dense_dimensions
        self._max_concurrent = settings.openrouter_max_concurrent
        self._referer = settings.openrouter_http_referer
        self._title = settings.openrouter_x_title
        self._rpm_limiter = TokenBucket(
            tokens_per_sec=settings.openrouter_rpm / 60.0,
            burst=settings.openrouter_max_concurrent,
        )
        # Loop-bound resources — recreated when the event loop changes
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        self._tokens_used: float = 0.0

    def _ensure_loop_resources(self) -> None:
        loop = asyncio.get_running_loop()
        if self._bound_loop is not loop:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            if self._referer:
                headers["HTTP-Referer"] = self._referer
            if self._title:
                headers["X-Title"] = self._title
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(60.0),
            )
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            self._bound_loop = loop

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    @property
    def tokens_used(self) -> float:
        return self._tokens_used

    def reset_token_counter(self) -> None:
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
        task: str = "passage",  # noqa: ARG002
        dimensions: int | None = None,
        late_chunking: bool = False,  # noqa: ARG002
    ) -> list[list[float]]:
        dims = dimensions if dimensions is not None else self._dimensions
        payload: dict[str, object] = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }
        if dims:
            payload["dimensions"] = dims
        self._tokens_used += sum(len(t) for t in texts) / 4.0
        data = await self._request(payload)
        return [item["embedding"] for item in data["data"]]

    async def embed_text_query(
        self, query: str, dimensions: int | None = None
    ) -> list[float]:
        results = await self.embed_text([query], task="query", dimensions=dimensions)
        return results[0]

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[float]:
        raise NotImplementedError(
            "OpenRouter embeddings endpoint is text-only. "
            "Set IMAGE_EMBED_STRATEGY=none, or use embedding_provider=jina|voyage."
        )

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[list[float]]:
        raise NotImplementedError(
            "OpenRouter does not support ColBERT multi-vector embeddings"
        )

    async def embed_query_multi_vector(self, query: str) -> list[list[float]]:
        raise NotImplementedError(
            "OpenRouter does not support ColBERT multi-vector embeddings"
        )

    async def _request(
        self,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        self._ensure_loop_resources()
        await self._rpm_limiter.acquire()
        async with self._semaphore:  # type: ignore[union-attr]
            return await self._request_with_retry(payload, timeout)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(_is_retryable),
        before_sleep=lambda rs: logger.warning(
            "OpenRouter API retry attempt %d after %s",
            rs.attempt_number,
            rs.outcome.exception() if rs.outcome else None,
        ),
    )
    async def _request_with_retry(
        self,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        kwargs: dict = {"json": payload}  # type: ignore[type-arg]
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await self._client.post(self._url, **kwargs)  # type: ignore[union-attr]
        if resp.status_code != 200:
            exc = httpx.HTTPStatusError(
                f"OpenRouter API error: {resp.text}",
                request=resp.request,
                response=resp,
            )
            code = resp.status_code
            if code == 429:
                retry_after = float(resp.headers.get("Retry-After", 5))
                self._rpm_limiter.pause(retry_after)
            if code != 429 and code < 500:
                raise httpx.HTTPStatusError(
                    f"OpenRouter API client error ({code}): {resp.text}",
                    request=resp.request,
                    response=resp,
                ) from exc
            raise exc
        return resp.json()  # type: ignore[no-any-return]
