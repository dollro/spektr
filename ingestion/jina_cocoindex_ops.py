from __future__ import annotations

import asyncio

import cocoindex

from config.settings import settings
from ingestion.embedder import JinaV4Embedder

_embedder: JinaV4Embedder | None = None


def _get_embedder() -> JinaV4Embedder:
    """Lazily initialize a shared embedder instance."""
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = JinaV4Embedder(api_key=settings.jina_api_key)
    return _embedder


def _run_async[T](coro: T) -> T:  # type: ignore[type-var]
    """Run an async coroutine from a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    # If already in an event loop, create a new one in a thread
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return loop.run_in_executor(  # type: ignore[return-value]
            pool,
            asyncio.run,
            coro,  # type: ignore[arg-type]
        )


@cocoindex.op.function()
def jina_embed_text(text: str) -> list[float]:
    """CocoIndex op: text -> dense vector (2048d)."""
    embedder = _get_embedder()
    results = _run_async(embedder.embed_text([text]))
    return results[0]


@cocoindex.op.function()
def jina_embed_image(image_bytes: bytes) -> list[float]:
    """CocoIndex op: image -> dense vector (2048d)."""
    embedder = _get_embedder()
    return _run_async(embedder.embed_image(image_bytes))


@cocoindex.op.function()
def jina_embed_image_multivec(image_bytes: bytes) -> list[list[float]]:
    """CocoIndex op: image -> ColBERT multi-vectors (128d each)."""
    embedder = _get_embedder()
    return _run_async(embedder.embed_multi_vector(image_bytes))
