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


def create_embedder(**kwargs: object) -> Embedder:
    """Instantiate the configured embedding provider.

    Reads ``settings.embedding_provider`` (default ``"jina"``) and lazily
    imports the corresponding implementation.

    Extra *kwargs* are forwarded to the provider constructor.
    """
    provider = settings.embedding_provider
    if provider == "jina":
        from ingestion.embedders.jina import JinaV4Embedder

        return JinaV4Embedder(**kwargs)  # type: ignore[return-value]

    msg = f"Unknown embedding provider: {provider!r}"
    raise ValueError(msg)
