"""Graphiti client lifecycle management.

Provides a singleton async Graphiti client connected to Neo4j
for temporal knowledge graph operations.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

# Graphiti reads EMBEDDING_DIM at import time for vector index sizing
os.environ.setdefault("EMBEDDING_DIM", str(512))

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import (
    OpenAIRerankerClient,
)
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from config.settings import settings
from ingestion.embedder import create_embedder

logger = logging.getLogger(__name__)


class _JinaGraphitiEmbedder(EmbedderClient):
    """Adapts our JinaV4Embedder to Graphiti's EmbedderClient interface."""

    def __init__(self) -> None:
        self._embedder = create_embedder()

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            texts = input_data  # type: ignore[assignment]
        else:
            texts = [str(i) for i in input_data if i is not None]  # type: ignore[union-attr]

        if not texts:
            return []
        vectors = await self._embedder.embed_text(texts, task="query")
        return vectors[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []
        return await self._embedder.embed_text(input_data_list, task="query")

    async def close(self) -> None:
        await self._embedder.close()


_client: Graphiti | None = None
_graphiti_embedder: _JinaGraphitiEmbedder | None = None


async def get_graphiti() -> Graphiti:
    """Return (and lazily initialise) the shared Graphiti client."""
    global _client, _graphiti_embedder  # noqa: PLW0603
    if _client is None:
        llm_config = LLMConfig(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url or None,
        )
        _graphiti_embedder = _JinaGraphitiEmbedder()
        _client = Graphiti(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            llm_client=OpenAIGenericClient(config=llm_config),
            embedder=_graphiti_embedder,
            cross_encoder=OpenAIRerankerClient(config=llm_config),
        )
        await _client.build_indices_and_constraints()
        logger.info("Graphiti client initialised")
    return _client


async def close_graphiti() -> None:
    """Shut down the shared Graphiti client."""
    global _client, _graphiti_embedder  # noqa: PLW0603
    if _graphiti_embedder is not None:
        await _graphiti_embedder.close()
        _graphiti_embedder = None
    if _client is not None:
        await _client.close()
        _client = None
        logger.info("Graphiti client closed")
