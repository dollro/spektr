"""Listwise re-scoring of retrieved candidates, over a pluggable reranker.

The reranker used to be Jina, unconditionally: the module opened with
``RERANK_URL = f"{settings.jina_api_url}/v1/rerank"`` computed at import time
and read ``settings.jina_api_key`` on every call. That is what made the
reranker hostage to a vendor unrelated to the active embedding model, and it
is why a deployment following ``.env.example`` — ``RERANK_ENABLED=true``,
``EMBEDDING_MODEL=gemini-2``, ``JINA_API_KEY`` empty — 401'd on every query
and silently degraded to RRF order.

Now the model string selects a wire protocol from ``config/rerank_models.py``
and this module composes it. The split mirrors ``ingestion/embedder.py`` vs
``ingestion/embedders/``: protocol and factory here, implementations in
``retrieval/rerankers/``.

The public surface is unchanged. ``rerank()`` still raises ``RerankError`` and
still leaves the degradation decision to its caller, which is the only layer
that knows whether it can report ``degraded``.

**Scores are not comparable across rerankers.** ``jina-reranker-v3.5`` emits
unbounded logits; a TEI cross-encoder emits [0, 1]. Anything comparing a score
against a threshold must use ``settings.resolved_score_floor``, never a
literal — see ``retrieval/gate.py``.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from config.rerank_models import spec
from config.settings import settings
from retrieval.models import FusedResult

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """Raised when reranking fails and the caller must handle degradation."""


@runtime_checkable
class Reranker(Protocol):
    """One reranker wire protocol.

    ``(original index, score)`` is the whole contract: it is everything both
    callers in this module need, and keeping it that narrow is what let the
    Jina v2 -> v3.5 change of the unused ``document`` field pass unnoticed.
    """

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]: ...

    async def close(self) -> None: ...


_reranker: Reranker | None = None


def create_reranker() -> Reranker:
    """Instantiate the adapter for the configured rerank model+route pair.

    Dispatch is on the **protocol**, because that is what determines the wire
    format; the route only decides where to point it. Legality of the pair and
    the presence of credentials are enforced by ``Settings``, so an
    unreachable slot fails at startup rather than on the first query.
    """
    rerank_model = spec(settings.rerank_model)
    base_url, api_key = settings.rerank_endpoint

    if rerank_model.protocol == "jina":
        from retrieval.rerankers.jina import JinaReranker

        return JinaReranker(base_url=base_url, api_key=api_key, wire_id=rerank_model.wire_id)

    if rerank_model.protocol == "tei":
        from retrieval.rerankers.tei import TEIReranker

        return TEIReranker(base_url=base_url, api_key=api_key, wire_id=rerank_model.wire_id)

    # cohere/voyage are members of RerankProtocol with no adapter yet — a
    # deliberate deferral recorded in config/rerank_models.py. Unreachable
    # today because no registry entry names them.
    msg = (
        f"No reranker adapter for protocol {rerank_model.protocol!r} "
        f"(model {settings.rerank_model!r})."
    )
    raise ValueError(msg)


def _get_reranker() -> Reranker:
    global _reranker  # noqa: PLW0603
    if _reranker is None:
        _reranker = create_reranker()
    return _reranker


def reset_reranker() -> None:
    """Drop the cached reranker. Test hook."""
    global _reranker  # noqa: PLW0603
    _reranker = None


async def _score(query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    """Run the configured reranker, or raise RerankError."""
    try:
        return await _get_reranker().rerank(query, documents, top_n=top_k)
    except Exception as exc:
        logger.exception("Reranking failed")
        raise RerankError(str(exc)) from exc


async def rerank(
    query: str,
    results: list[FusedResult],
    top_k: int = 5,
) -> list[FusedResult]:
    """Re-score fused results against the query.

    Args:
        query: The original user query, never a sub-query.
        results: Fused candidates to rescore.
        top_k: Number of results to return.

    Returns:
        Results ordered by rerank score.

    Raises:
        RerankError: The API call failed. The caller decides how to degrade —
            it is the only layer that knows whether it can report `degraded`.
    """
    if not results:
        return []

    documents = [r.text for r in results]
    if not any(documents):
        return results[:top_k]

    ranked = await _score(query, documents, top_k)
    return [results[index].model_copy(update={"score": score}) for index, score in ranked]


async def rerank_dicts(
    query: str,
    results: list[dict],  # type: ignore[type-arg]
    top_k: int = 5,
) -> list[dict]:  # type: ignore[type-arg]
    """Dict-based reranking for callers that predate the typed pipeline.

    Preserves the original score as 'original_score'. Used by vector_search.
    Unlike `rerank()`, this swallows the failure and returns the input order —
    its caller has no `degraded` channel to report into.
    """
    if not results:
        return results

    documents = [r.get("text", "") for r in results]
    if not any(documents):
        return results[:top_k]

    try:
        ranked = await _score(query, documents, top_k)
    except RerankError:
        return results[:top_k]

    reranked: list[dict] = []  # type: ignore[type-arg]
    for index, score in ranked:
        original = results[index].copy()
        original["original_score"] = original.get("score")
        original["score"] = score
        reranked.append(original)
    return reranked
