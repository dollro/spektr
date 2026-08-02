"""miniCOIL sparse encoding for the lexical retrieval channel.

miniCOIL behaves like BM25 that understands word sense — it keeps exact keyword
matching while disambiguating by context. The model runs locally on CPU via
fastembed, so there is no API cost, but the first call pays a load penalty.

Document encoding applies BM25-style length normalisation through ``avg_len``;
query encoding deliberately does not. In fastembed's API, ``avg_len`` is a
constructor argument of ``SparseTextEmbedding`` rather than a per-call option —
the document/query asymmetry is handled internally by the library based on
which of ``embed()`` (documents) or ``query_embed()`` (queries) is called, not
by anything this module passes at call time.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import models

from config.constants import MINICOIL_AVG_LEN
from config.settings import settings

logger = logging.getLogger(__name__)

_model: Any | None = None

SparseVector = models.SparseVector


def _load_model() -> Any:
    """Instantiate the fastembed sparse model. Imported lazily — heavy.

    ``avg_len`` is set here, at construction time: fastembed's
    ``SparseTextEmbedding.embed()``/``query_embed()`` take no ``avg_len``
    kwarg, so it cannot be supplied per call.
    """
    from fastembed import SparseTextEmbedding

    logger.info("Loading sparse model %s", settings.sparse_model)
    return SparseTextEmbedding(model_name=settings.sparse_model, avg_len=MINICOIL_AVG_LEN)


def _get_model() -> Any:
    global _model  # noqa: PLW0603
    if _model is None:
        _model = _load_model()
    return _model


def reset_model() -> None:
    """Drop the cached model. Test hook."""
    global _model  # noqa: PLW0603
    _model = None


def encode_documents(texts: list[str]) -> list[SparseVector]:
    """Encode chunk texts for indexing.

    Args:
        texts: Chunk texts to encode.

    Returns:
        One SparseVector per input text, in order.
    """
    if not texts:
        return []

    embeddings = _get_model().embed(texts)
    return [
        models.SparseVector(indices=list(e.indices), values=list(e.values))
        for e in embeddings
    ]


def encode_query(text: str) -> SparseVector:
    """Encode a query for search. No length normalisation.

    Args:
        text: The query string.

    Returns:
        A single SparseVector.
    """
    embedding = next(iter(_get_model().query_embed(text)))
    return models.SparseVector(
        indices=list(embedding.indices), values=list(embedding.values)
    )
