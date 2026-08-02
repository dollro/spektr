"""Standalone embedding helpers for ad-hoc sub-flows.

These were CocoIndex v0 custom ops (``@cocoindex.op.function()``). v1 has no
equivalent decorator for functions that are called outside a component context,
and none of these are wired into the ingestion app — the bulk pipeline embeds
through ``ingestion.page_processor``. They are kept as plain sync helpers so
callers that want a one-shot embedding without an event loop still have one.
"""

from __future__ import annotations

from ingestion._utils import run_async
from ingestion.embedder import Embedder, create_embedder

_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    """Lazily initialize a shared embedder instance."""
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder


def op_embed_text(text: str) -> list[float]:
    """Embed one string to a dense vector."""
    embedder = _get_embedder()
    results = run_async(embedder.embed_text([text]))
    return results[0]


def op_embed_image(image_bytes: bytes) -> list[float]:
    """Embed one image to a dense vector."""
    embedder = _get_embedder()
    return run_async(embedder.embed_image(image_bytes))


def op_embed_image_multivec(image_bytes: bytes) -> list[list[float]]:
    """Embed one image to ColBERT multi-vectors."""
    embedder = _get_embedder()
    return run_async(embedder.embed_multi_vector(image_bytes))
