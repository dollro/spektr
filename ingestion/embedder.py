"""Embedder protocol, token-bucket rate limiter, and provider factory."""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import Protocol, runtime_checkable

from config.settings import settings

logger = logging.getLogger(__name__)


class TokenBucket:
    """Simple async token-bucket rate limiter.

    Not thread-safe — designed for use within a single asyncio event loop.
    """

    def __init__(self, tokens_per_sec: float, burst: int) -> None:
        self._rate = tokens_per_sec
        self._max = float(burst)
        self._tokens = float(burst)
        self._last = _time.monotonic()
        self._pause_until: float = 0.0

    def pause(self, seconds: float) -> None:
        """Globally pause all token acquisition for *seconds*.

        Called when a 429 response is received so that all concurrent
        requests back off, not just the one that was rate-limited.
        """
        resume_at = _time.monotonic() + seconds
        if resume_at > self._pause_until:
            self._pause_until = resume_at
            logger.warning("Rate limiter paused for %.1fs", seconds)

    async def acquire(self, n: float = 1.0) -> None:
        """Wait until *n* tokens are available, then consume them."""
        # Honour global pause from 429 responses
        now = _time.monotonic()
        if now < self._pause_until:
            await asyncio.sleep(self._pause_until - now)

        while True:
            now = _time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._max, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= n:
                self._tokens -= n
                return
            wait = (n - self._tokens) / self._rate
            await asyncio.sleep(wait)


@runtime_checkable
class Embedder(Protocol):
    """Embedding provider interface.

    Task names use generic labels: ``"passage"`` for indexing,
    ``"query"`` for search.  Concrete implementations map these to
    provider-specific task strings internally.
    """

    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,
    ) -> list[list[float]]: ...

    async def embed_text_query(
        self,
        query: str,
        dimensions: int | None = None,
    ) -> list[float]: ...

    async def embed_image(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
    ) -> list[float]: ...

    async def embed_multi_vector(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
    ) -> list[list[float]]: ...

    async def embed_query_multi_vector(
        self,
        query: str,
    ) -> list[list[float]]: ...

    async def close(self) -> None: ...

    @property
    def tokens_used(self) -> float: ...
    def reset_token_counter(self) -> None: ...

    @property
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. 'jina-embeddings-v4')."""
        ...

    @property
    def dim(self) -> int:
        """Dense vector dimensionality used by this embedder."""
        ...


def create_embedder(api_key: str | None = None) -> Embedder:
    """Instantiate the client for the configured model+route pair.

    Dispatch is on the **route**, because that is what determines the wire
    protocol; the route's client resolves the model id itself. Legality of
    the pair is enforced by Settings, so an illegal combination fails at
    startup rather than here.

    API keys fall back to settings when *api_key* is not given.
    """
    route = settings.embedding_route
    if route == "openrouter":
        from ingestion.embedders.openrouter import OpenRouterEmbedder

        return OpenRouterEmbedder(api_key=api_key)

    # route == "native": one client per vendor API.
    if settings.embedding_model == "jina-v4":
        from ingestion.embedders.jina import JinaV4Embedder

        return JinaV4Embedder(api_key=api_key)

    if settings.embedding_model == "voyage-4":
        from ingestion.embedders.voyage import VoyageEmbedder

        return VoyageEmbedder(api_key=api_key)

    msg = f"No native client for embedding model {settings.embedding_model!r}"
    raise ValueError(msg)
