from __future__ import annotations

import asyncio
import base64
import logging
import threading
import weakref

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

_TASK_MAP = {"passage": "document", "query": "query"}


def _is_retryable_status(exc: BaseException) -> bool:
    """Return True for HTTP 429 and 5xx errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class VoyageEmbedder:
    """Voyage AI embedding client using a shared httpx.AsyncClient."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.voyage_api_key
        self._text_url = f"{settings.voyage_api_url}/v1/embeddings"
        self._multimodal_url = f"{settings.voyage_api_url}/v1/multimodalembeddings"
        self._text_model = settings.voyage_text_model
        self._multimodal_model = settings.voyage_multimodal_model
        self._dimensions = settings.dense_dimensions
        self._max_concurrent = settings.voyage_max_concurrent
        self._rpm_limiter = TokenBucket(
            tokens_per_sec=settings.voyage_rpm / 60.0,
            burst=settings.voyage_max_concurrent,
        )
        # httpx.AsyncClient and asyncio.Semaphore are bound to the event loop
        # they were created on. CocoIndex runs file components concurrently on
        # multiple worker threads, each with its own loop, all sharing this one
        # embedder instance. A single shared client slot therefore gets clobbered
        # across loops, and an in-flight request ends up awaiting a connection-pool
        # lock owned by a different loop ("RuntimeError: the current task is not
        # holding this lock"). Keep the loop-affine resources isolated per loop so
        # each loop only ever touches the client it created. Keyed weakly so
        # entries drop when a worker loop is garbage-collected.
        self._loop_resources: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, tuple[httpx.AsyncClient, asyncio.Semaphore]
        ] = weakref.WeakKeyDictionary()
        self._resources_lock = threading.Lock()
        # Point at the current loop's resources for introspection/back-compat.
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        self._tokens_used: float = 0.0

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0),
        )

    def _ensure_loop_resources(
        self,
    ) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        """Return the (client, semaphore) bound to the running loop.

        Creates them on first use per loop and reuses them thereafter, so
        connection pooling is preserved within a loop while never sharing a
        client across loops. Thread-safe: multiple worker loops may call this
        concurrently.
        """
        loop = asyncio.get_running_loop()
        with self._resources_lock:
            resources = self._loop_resources.get(loop)
            if resources is None:
                resources = (self._build_client(), asyncio.Semaphore(self._max_concurrent))
                self._loop_resources[loop] = resources
            # Best-effort introspection handles; the request path uses the
            # returned locals, not these, so a concurrent overwrite is harmless.
            self._client, self._semaphore = resources
            self._bound_loop = loop
            return resources

    async def close(self) -> None:
        """Close every per-loop HTTP client (best effort)."""
        for client, _ in list(self._loop_resources.values()):
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 — closing a foreign-loop client may fail
                pass

    @property
    def tokens_used(self) -> float:
        """Total estimated tokens consumed since last reset."""
        return self._tokens_used

    def reset_token_counter(self) -> None:
        """Reset the token consumption counter to zero."""
        self._tokens_used = 0.0

    @property
    def model_name(self) -> str:
        return self._text_model

    @property
    def dim(self) -> int:
        return self._dimensions

    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,  # noqa: ARG002
    ) -> list[list[float]]:
        """Batch text -> list of dense vectors."""
        voyage_task = _TASK_MAP.get(task, task)
        dims = dimensions if dimensions is not None else self._dimensions
        payload = {
            "model": self._text_model,
            "input": texts,
            "input_type": voyage_task,
            "output_dimensions": dims,
        }
        data = await self._request(self._text_url, payload)
        return [item["embedding"] for item in data["data"]]

    async def embed_text_query(self, query: str, dimensions: int | None = None) -> list[float]:
        """Single query text -> dense vector."""
        dims = dimensions if dimensions is not None else self._dimensions
        results = await self.embed_text([query], task="query", dimensions=dims)
        return results[0]

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[float]:
        """Single image -> dense vector."""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self._multimodal_model,
            "input": [
                [
                    {
                        "type": "image_base64",
                        "image_base64": b64,
                        "media_type": media_type,
                    }
                ]
            ],
            "input_type": _TASK_MAP["passage"],
            "output_dimensions": self._dimensions,
        }
        data = await self._request(self._multimodal_url, payload, timeout=120.0)
        embedding: list[float] = data["data"][0]["embedding"]
        return embedding

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[list[float]]:
        """ColBERT multi-vector not supported by Voyage."""
        raise NotImplementedError("Voyage does not support ColBERT multi-vector embeddings")

    async def embed_query_multi_vector(self, query: str) -> list[list[float]]:
        """ColBERT multi-vector not supported by Voyage."""
        raise NotImplementedError("Voyage does not support ColBERT multi-vector embeddings")

    async def _request(
        self,
        url: str,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """Send request with rate limiting + concurrency control."""
        client, semaphore = self._ensure_loop_resources()
        await self._rpm_limiter.acquire()
        async with semaphore:
            return await self._request_with_retry(client, url, payload, timeout)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(_is_retryable_status),
        before_sleep=lambda rs: logger.warning(
            "Voyage API retry attempt %d after %s",
            rs.attempt_number,
            rs.outcome.exception() if rs.outcome else None,
        ),
    )
    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """Send request with retry on 429/5xx errors."""
        kwargs: dict = {"json": payload}  # type: ignore[type-arg]
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await client.post(url, **kwargs)
        if resp.status_code != 200:
            exc = httpx.HTTPStatusError(
                f"Voyage API error: {resp.text}",
                request=resp.request,
                response=resp,
            )
            code = resp.status_code
            if code == 429:
                retry_after = float(resp.headers.get("Retry-After", 5))
                self._rpm_limiter.pause(retry_after)
            if code != 429 and code < 500:
                raise httpx.HTTPStatusError(
                    f"Voyage API client error ({code}): {resp.text}",
                    request=resp.request,
                    response=resp,
                ) from exc
            raise exc
        return resp.json()  # type: ignore[no-any-return]
