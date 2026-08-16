"""Graphiti client lifecycle management.

Provides a singleton async Graphiti client connected to Neo4j
for temporal knowledge graph operations.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from config.settings import settings

# Graphiti reads EMBEDDING_DIM at import time to size any vector index it
# creates, so this must run BEFORE graphiti_core is imported — hence the
# deliberate import ordering and the E402 waivers below.
#
# It must track the active model. A literal here sat at 512 from when that
# was the Jina default, which would size an index for 512 while
# name_embedding holds 768 floats — entity similarity would then silently
# match nothing. No vector index exists today (cosine is computed per node),
# which is the only reason the stale value stayed harmless.
os.environ.setdefault("EMBEDDING_DIM", str(settings.dense_dimensions))

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.client import CrossEncoderClient  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient  # noqa: E402
from graphiti_core.llm_client import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: E402

from ingestion.embedder import create_embedder  # noqa: E402
from retrieval.rerank import create_reranker  # noqa: E402

logger = logging.getLogger(__name__)


class _JinaGraphitiEmbedder(EmbedderClient):
    """Adapts the configured embedder to Graphiti's EmbedderClient interface.

    Named for Jina because that was the only provider when it was written; it
    delegates to whatever `create_embedder()` returns, so graph embeddings
    share the project's rate limiters and API quota rather than opening a
    second client.

    **Both methods pass `task="query"`, including the indexing one.** That is
    deliberate, and it is the opposite of the passage/query asymmetry this
    codebase treats as load-bearing elsewhere (see `_INPUT_TYPE` in
    `ingestion/embedders/openrouter.py`, where the two modes differ by cos
    0.82). Graphiti compares a search query against stored *entity names*, so
    both sides want the same short-text encoding — an asymmetric pair would
    compare a query-mode vector against a passage-mode one and lose
    similarity. It is symmetric on purpose, not by omission.

    Changing it re-embeds every `name_embedding` in the graph, so it is not a
    drop-in adjustment.
    """

    def __init__(self) -> None:
        self._embedder = create_embedder()

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            texts = input_data
        else:
            texts = [str(i) for i in input_data if i is not None]

        if not texts:
            return []
        # Graphiti's EmbedderClient.create contract returns ONE vector, so a
        # multi-item input can only yield the first. Embedding just texts[0]
        # would be cheaper, but the extra items are what a caller passing a
        # list expects to influence the result — warn rather than silently
        # discard, so a misuse is visible instead of merely wrong.
        if len(texts) > 1:
            logger.warning(
                "Graphiti embedder received %d inputs but returns one vector; "
                "using the first and discarding the rest",
                len(texts),
            )
        vectors = await self._embedder.embed_text(texts, task="query")
        return vectors[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []
        return await self._embedder.embed_text(input_data_list, task="query")

    async def close(self) -> None:
        await self._embedder.close()


class _SlotCrossEncoder(CrossEncoderClient):
    """Adapts the configured rerank slot to Graphiti's CrossEncoderClient.

    Graphiti previously took `OpenAIRerankerClient`, which made it a SECOND
    hosted reranker in the stack — unaffected by `RERANK_MODEL`, and pointed
    at OpenAI regardless of what the retrieval path was told to use. Routing
    it through the same slot means one reranker choice is honoured everywhere,
    and a self-hosted deployment does not leak graph queries to a vendor it
    thought it had switched away from.

    Mirrors what `_JinaGraphitiEmbedder` above does for the embedder.
    """

    def __init__(self) -> None:
        self._reranker = create_reranker()

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """Return (passage, score) pairs, most relevant first.

        Graphiti wants the passage text back, not its position, so the
        adapter resolves indices here. Degrading rather than raising is
        deliberate: entity resolution is a *ranking* refinement, and failing
        the whole ingest because a reranker was unreachable would be a far
        worse outcome than falling back to the input order.
        """
        if not passages:
            return []
        try:
            ranked = await self._reranker.rerank(query, passages, top_n=len(passages))
        except Exception:
            logger.exception("Graph cross-encoder failed; falling back to input order")
            return [(passage, 0.0) for passage in passages]
        return [(passages[index], score) for index, score in ranked]

    async def close(self) -> None:
        await self._reranker.close()


_client: Graphiti | None = None
_graphiti_embedder: _JinaGraphitiEmbedder | None = None
_graphiti_cross_encoder: _SlotCrossEncoder | None = None


async def get_graphiti() -> Graphiti:
    """Return (and lazily initialise) the shared Graphiti client."""
    global _client, _graphiti_embedder, _graphiti_cross_encoder  # noqa: PLW0603
    if _client is None:
        os.environ.setdefault("SEMAPHORE_LIMIT", str(settings.graph_semaphore_limit))
        llm_config = LLMConfig(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url or None,
        )
        _graphiti_embedder = _JinaGraphitiEmbedder()
        _graphiti_cross_encoder = _SlotCrossEncoder()
        _client = Graphiti(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            llm_client=OpenAIGenericClient(config=llm_config),
            embedder=_graphiti_embedder,
            cross_encoder=_graphiti_cross_encoder,
        )
        await _client.build_indices_and_constraints()
        logger.info("Graphiti client initialised")
    return _client


async def close_graphiti() -> None:
    """Shut down the shared Graphiti client."""
    global _client, _graphiti_embedder, _graphiti_cross_encoder  # noqa: PLW0603
    if _graphiti_embedder is not None:
        await _graphiti_embedder.close()
        _graphiti_embedder = None
    if _graphiti_cross_encoder is not None:
        await _graphiti_cross_encoder.close()
        _graphiti_cross_encoder = None
    if _client is not None:
        await _client.close()
        _client = None
        logger.info("Graphiti client closed")
