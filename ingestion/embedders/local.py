from __future__ import annotations

import asyncio
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

from config.embedding_models import ModelSpec, spec
from config.settings import settings

logger = logging.getLogger(__name__)


def _prefix(model_spec: ModelSpec, task: str) -> str:
    """Return the instruction prefix this model wants for *task*.

    The counterpart of `_input_type` in the OpenRouter client
    (ingestion/embedders/openrouter.py:42-49), and it raises for the same
    reason. The mechanism differs — there an unmapped `input_type` is
    accepted and silently ignored by the gateway; here an unmapped task
    would simply prepend nothing — but the cost is identical: a query
    embedded in passage mode, no error, no log line, and nothing to see
    but recall that is quietly worse than it should be.

    Note the lookup is on key membership, not truth: `""` is a legitimate
    prefix (Qwen3 wants none on the passage side) and must not be
    mistaken for an unmapped task.
    """
    prefixes = {"passage": model_spec.passage_prefix, "query": model_spec.query_prefix}
    try:
        return prefixes[task]
    except KeyError:
        raise ValueError(
            f"Unknown embedding task {task!r}; expected one of {sorted(prefixes)}"
        ) from None


def _fit(vector: list[float], dims: int) -> list[float]:
    """Truncate a Matryoshka vector to *dims* and renormalise it.

    Truncation happens here rather than on the wire because `dimensions`
    is not a request field we can rely on: TEI ignores it outright and
    vLLM's support varies per model, so asking for a size is not the same
    as getting one. Doing it client-side is the only version that can be
    asserted, and the assert matters — a silently full-width vector
    against a collection sized to the truncated width is a per-point
    Qdrant rejection at ingest, one failed page at a time.

    Renormalising is not cosmetic. The model emits unit-norm vectors and
    the model card's own MRL usage renormalises after slicing; lopping off
    the tail leaves a vector shorter than 1, which changes any dot-product
    comparison and leaves query and stored vectors on different footings.
    """
    if len(vector) < dims:
        raise ValueError(
            f"Local embedding endpoint returned {len(vector)} dimensions, fewer "
            f"than the configured {dims}. Matryoshka truncation can only shrink: "
            f"check EMBEDDING_DIMENSIONS against the served checkpoint."
        )
    head = vector[:dims]
    norm = math.sqrt(sum(v * v for v in head))
    if norm > 0.0:
        head = [v / norm for v in head]
    if len(head) != dims:  # pragma: no cover — belt and braces on the slice above
        raise ValueError(f"Truncation produced {len(head)} dimensions, expected {dims}")
    return head


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


class LocalEmbedder:
    """Local route: an OpenAI-compatible /v1/embeddings you host yourself.

    Serves whichever model+route pair resolves to a `local_model_id` —
    today `qwen3-embedding-8b` (see config/embedding_models.py) — from a
    TEI, vLLM or Infinity container. The wire shape is the same as the
    OpenRouter route's; two things about it are not.

    **The retrieval role is a text prefix, not a request field.** Open
    encoders were trained with the instruction inside the input, so the
    asymmetry comes from `query_prefix` / `passage_prefix` on the registry
    entry rather than from `input_type`. Changing either prefix changes
    the vector space and requires a full re-ingest, exactly as changing
    the `input_type` mapping would.

    **`dimensions` is never sent.** See `_fit`: truncation and
    renormalisation are done here, where the result can be checked.

    Image and ColBERT multi-vector are not implemented and will not be:
    they belong to other slots, not to a text-only dense embedder.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # A local container is normally unauthenticated; the key is here
        # for the case where one is fronted by a proxy that wants a bearer.
        self._api_key = api_key or settings.local_embed_api_key
        self._url = f"{settings.local_embed_api_url}/v1/embeddings"
        self._model = settings.embedding_model_id
        self._dimensions = settings.dense_dimensions
        self._spec = spec(settings.embedding_model)
        self._batch_size = settings.local_batch_size
        self._max_concurrent = settings.local_max_concurrent
        # No TokenBucket, deliberately. A self-hosted endpoint has no RPM
        # quota to respect — it has finite GPU memory, which a concurrency
        # bound expresses and a rate cap does not. Capping requests per
        # minute against your own server is throughput given away for
        # nothing, so the semaphore below is the whole limiter.
        #
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
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # Longer than the hosted routes' 60s: an 8B encoder chewing a full
        # batch on one GPU is slower than a fanned-out managed endpoint,
        # and a timeout here fails the whole file.
        return httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(120.0))

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
        prefix = _prefix(self._spec, task)
        vectors: list[list[float]] = []
        # Batching is a memory bound here rather than an API limit: the
        # server allocates for the whole batch at once, so an unbatched
        # long document is an OOM on the container rather than a 400.
        for start in range(0, len(texts), self._batch_size):
            batch = [f"{prefix}{text}" for text in texts[start : start + self._batch_size]]
            payload: dict[str, object] = {
                "model": self._model,
                "input": batch,
                "encoding_format": "float",
            }
            self._tokens_used += sum(len(t) for t in batch) / 4.0
            data = await self._request(payload)
            # Order matters: chunk N's vector must stay with chunk N. The
            # OpenAI schema carries an explicit index; don't trust position.
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            vectors.extend(_fit(item["embedding"], dims) for item in items)
        return vectors

    async def embed_text_query(self, query: str, dimensions: int | None = None) -> list[float]:
        results = await self.embed_text([query], task="query", dimensions=dimensions)
        return results[0]

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[float]:
        raise NotImplementedError(
            f"{settings.embedding_model} is a text-only encoder — the local route has "
            "no image embedding and its registry entry lists no image_routes. Set "
            "IMAGE_EMBED_STRATEGY=none; page images are reached through the multivec "
            "slot (visual_search) instead of the dense channel."
        )

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[list[float]]:
        raise NotImplementedError(
            "ColBERT multi-vector is not part of the dense slot. Late-interaction "
            "vectors come from the multivec slot's own model and collection, not "
            "from the local dense embedder."
        )

    async def embed_query_multi_vector(self, query: str) -> list[list[float]]:
        raise NotImplementedError(
            "ColBERT multi-vector is not part of the dense slot. Late-interaction "
            "vectors come from the multivec slot's own model and collection, not "
            "from the local dense embedder."
        )

    async def _request(
        self,
        payload: dict,  # type: ignore[type-arg]
        timeout: float | None = None,
    ) -> dict:  # type: ignore[type-arg]
        client, semaphore = self._ensure_loop_resources()
        async with semaphore:
            return await self._request_with_retry(client, payload, timeout)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(_is_retryable),
        before_sleep=lambda rs: logger.warning(
            "Local embedding endpoint retry attempt %d after %s",
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
            # One exception shape, unlike the OpenRouter client: `_is_retryable`
            # reads the status off the response, and there is no rate limiter
            # here that a 429 would need to pause.
            raise httpx.HTTPStatusError(
                f"Local embedding endpoint error ({resp.status_code}): {resp.text}",
                request=resp.request,
                response=resp,
            )
        return resp.json()  # type: ignore[no-any-return]
