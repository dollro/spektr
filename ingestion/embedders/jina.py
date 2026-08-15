from __future__ import annotations

import asyncio
import base64
import logging
import math
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

_TASK_MAP = {"passage": "retrieval.passage", "query": "retrieval.query"}


def _is_retryable(exc: BaseException) -> bool:
    """Return True for HTTP 429/5xx and transient network errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    # httpx.TimeoutException is a SIBLING of NetworkError, not a subclass of
    # ConnectError — so listing ConnectError does not cover ConnectTimeout.
    # Omitting it made a plain connect timeout non-retryable and failed a
    # whole file on one transient blip.
    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ReadError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ),
    )


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
        self._dimensions = settings.dense_dimensions
        self._max_concurrent = settings.jina_max_concurrent
        self._rpm_limiter = TokenBucket(
            tokens_per_sec=settings.jina_rpm / 60.0,
            burst=settings.jina_max_concurrent,
        )
        self._tpm_limiter = TokenBucket(
            tokens_per_sec=settings.jina_tpm / 60.0,
            burst=settings.jina_tpm,
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
        embedding: list[float] = data["data"][0]["embedding"]
        return embedding

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
        vectors: list[list[float]] = data["data"][0]["embedding"]
        return vectors

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
        vectors: list[list[float]] = data["data"][0]["embedding"]
        return vectors

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
        client, semaphore = self._ensure_loop_resources()
        estimated_tokens = self._estimate_tokens(payload)
        self._tokens_used += estimated_tokens
        await self._rpm_limiter.acquire()
        await self._tpm_limiter.acquire(estimated_tokens)
        async with semaphore:
            return await self._request_with_retry(client, payload, timeout)

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(_is_retryable),
        before_sleep=lambda rs: logger.warning(
            "Jina API retry attempt %d after %s",
            rs.attempt_number,
            rs.outcome.exception() if rs.outcome else None,
        ),
    )
    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """Send request with retry on 429/5xx errors."""
        kwargs: dict = {"json": payload}  # type: ignore[type-arg]
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await client.post(self._embeddings_url, **kwargs)
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
