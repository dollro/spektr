from __future__ import annotations

import cocoindex

from ingestion._utils import run_async
from ingestion.embedder import Embedder, create_embedder

_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    """Lazily initialize a shared embedder instance."""
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder


@cocoindex.op.function()
def op_embed_text(text: str) -> list[float]:
    """CocoIndex op: text -> dense vector."""
    embedder = _get_embedder()
    results = run_async(embedder.embed_text([text]))
    return results[0]


@cocoindex.op.function()
def op_embed_image(image_bytes: bytes) -> list[float]:
    """CocoIndex op: image -> dense vector."""
    embedder = _get_embedder()
    return run_async(embedder.embed_image(image_bytes))


@cocoindex.op.function()
def op_embed_image_multivec(image_bytes: bytes) -> list[list[float]]:
    """CocoIndex op: image -> ColBERT multi-vectors."""
    embedder = _get_embedder()
    return run_async(embedder.embed_multi_vector(image_bytes))
