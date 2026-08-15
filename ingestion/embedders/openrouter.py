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


# OpenRouter spells the retrieval role `input_type`; Gemini's own
# `task_type` and Jina's `task` are both accepted and then SILENTLY IGNORED
# by the gateway. Verified against google/gemini-embedding-2: embeddings of
# the same text differ by cos 0.82 between query and document mode, so this
# is worth real retrieval quality.
#
# Unrecognised values are also silently ignored — an `input_type` typo costs
# you the asymmetry with no error — so an unmapped task raises here instead
# of being forwarded and quietly dropped.
_INPUT_TYPE = {"passage": "document", "query": "query"}

# Images are billed on a separate meter ($0.45/M for gemini-2) and there is no
# public token formula, so this is a flat placeholder. It only feeds the
# `tokens_used` report: this client rate-limits on requests per minute, not
# tokens, so an imprecise figure throttles nothing. IMAGE_EMBED_MAX_PX caps
# pages well under one tile, which is where 258 comes from.
_IMAGE_TOKEN_ESTIMATE = 258.0


def _input_type(task: str) -> str:
    """Map the provider-agnostic task name onto OpenRouter's `input_type`."""
    try:
        return _INPUT_TYPE[task]
    except KeyError:
        raise ValueError(
            f"Unknown embedding task {task!r}; expected one of {sorted(_INPUT_TYPE)}"
        ) from None


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


class OpenRouterEmbedder:
    """OpenRouter route: OpenAI-compatible /v1/embeddings.

    Serves whichever model+route pair resolves to an OpenRouter id — today
    `gemini-2` and `voyage-4` (see config/embedding_models.py). This is NOT
    a general client for all 32 models OpenRouter lists: the `input_type`
    contract below holds for the Gemini and Voyage families but not for the
    e5/bge/sentence-transformers ones, which want instruction prefixes in
    the text instead. The registry is what keeps that honest.

    Documents and queries are embedded asymmetrically via `input_type`
    (see _INPUT_TYPE). Changing that mapping changes the vector space and
    requires a full re-ingest.

    Image embedding is supported for models whose registry entry lists this
    route (gemini-2). ColBERT multi-vector is not, and cannot be: gemini-2
    emits a single vector, so `visual_search` stays jina-v4/native only.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.openrouter_api_key
        self._url = f"{settings.openrouter_api_url}/v1/embeddings"
        self._model = settings.embedding_model_id
        self._dimensions = settings.dense_dimensions
        self._batch_size = settings.openrouter_batch_size
        self._max_concurrent = settings.openrouter_max_concurrent
        self._referer = settings.openrouter_http_referer
        self._title = settings.openrouter_x_title
        self._rpm_limiter = TokenBucket(
            tokens_per_sec=settings.openrouter_rpm / 60.0,
            burst=settings.openrouter_max_concurrent,
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
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._referer:
            headers["HTTP-Referer"] = self._referer
        if self._title:
            headers["X-Title"] = self._title
        return httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(60.0))

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
        for client, _ in list(self._loop_resources.values()):
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 — closing a foreign-loop client may fail
                pass

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
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,  # noqa: ARG002
    ) -> list[list[float]]:
        dims = dimensions if dimensions is not None else self._dimensions
        input_type = _input_type(task)
        vectors: list[list[float]] = []
        # Gemini rejects >100 inputs per call with a 400 that names
        # BatchEmbedContentsRequest — a non-retryable client error, so an
        # unbatched long document fails the whole file. Slice here rather
        # than relying on callers to size their chunk lists.
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            payload: dict[str, object] = {
                "model": self._model,
                "input": batch,
                "encoding_format": "float",
                "input_type": input_type,
            }
            if dims:
                payload["dimensions"] = dims
            self._tokens_used += sum(len(t) for t in batch) / 4.0
            data = await self._request(payload)
            # Order matters: chunk N's vector must stay with chunk N. The
            # OpenAI schema carries an explicit index; don't trust position.
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            vectors.extend(item["embedding"] for item in items)
        return vectors

    async def embed_text_query(self, query: str, dimensions: int | None = None) -> list[float]:
        results = await self.embed_text([query], task="query", dimensions=dimensions)
        return results[0]

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[float]:
        """Embed one image into the same vector space as text.

        Deliberately does NOT send `input_type`: verified against
        gemini-2, the gateway accepts it for image input and returns a
        byte-identical vector either way, so passing it would advertise an
        asymmetry the model does not have. `dimensions` IS honoured, which
        is what lets image points share `documents_dense` with text and
        makes cross-modal retrieval work without a second collection.
        """
        if not settings.supports_image_embedding:
            raise NotImplementedError(
                f"Image embedding is not available for {settings.embedding_model} "
                f"via {settings.embedding_route}. Set IMAGE_EMBED_STRATEGY=none."
            )
        b64 = base64.b64encode(image_bytes).decode()
        payload: dict[str, object] = {
            "model": settings.embedding_image_model_id,
            "input": [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        }
                    ]
                }
            ],
            "encoding_format": "float",
        }
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        self._tokens_used += _IMAGE_TOKEN_ESTIMATE
        # A page bitmap is far heavier than a text batch; the 60s default
        # trips on larger pages.
        data = await self._request(payload, timeout=120.0)
        vector: list[float] = data["data"][0]["embedding"]
        return vector

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
        client, semaphore = self._ensure_loop_resources()
        await self._rpm_limiter.acquire()
        async with semaphore:
            return await self._request_with_retry(client, payload, timeout)

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
        client: httpx.AsyncClient,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        kwargs: dict = {"json": payload}  # type: ignore[type-arg]
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await client.post(self._url, **kwargs)
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
