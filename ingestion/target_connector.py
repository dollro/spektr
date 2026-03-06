"""CocoIndex custom target connector for RAG pipeline.

Handles deletion of Qdrant points and Graphiti episodes when
source files are removed. Upserts are no-ops here since
ingest_file handles them as a CocoIndex custom op.
"""

import dataclasses
from datetime import UTC, datetime

import cocoindex
from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION, MULTIVEC_COLLECTION
from config.logging import get_logger
from config.settings import settings
from ingestion._utils import run_async

logger = get_logger(__name__)


class RagTarget(cocoindex.op.TargetSpec):
    """Target spec for RAG pipeline cleanup."""

    qdrant_url: str


@dataclasses.dataclass
class RagTargetValues:
    """Value fields from the collector export."""

    result: str


def _delete_qdrant_points(qdrant: QdrantClient, source_key: str) -> None:
    """Delete all Qdrant points for a source file."""
    qdrant.delete(
        collection_name=DENSE_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_file",
                        match=models.MatchValue(value=source_key),
                    ),
                ],
            ),
        ),
    )
    if settings.multivec_enabled:
        qdrant.delete(
            collection_name=MULTIVEC_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=source_key),
                        ),
                    ],
                ),
            ),
        )


async def _remove_graphiti_episodes(source_key: str) -> None:
    """Remove Graphiti episodes matching source_description."""
    from ingestion.graphiti_client import close_graphiti, get_graphiti

    client = await get_graphiti()
    episodes = await client.retrieve_episodes(
        reference_time=datetime.now(tz=UTC),
        last_n=10000,
    )
    removed = 0
    for ep in episodes:
        if ep.source_description == source_key:
            await client.remove_episode(ep.uuid)
            removed += 1
    logger.info(
        "Removed %d Graphiti episodes for %s",
        removed,
        source_key,
        extra={"file_name": source_key},
    )
    await close_graphiti()


async def _remove_gliner_entities(source_key: str) -> None:
    """Remove GLiNER entities and their relationships for a source file."""
    from ingestion.neo4j_setup import get_driver

    driver = get_driver()
    try:
        async with driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity {source: $source}) "
                "DETACH DELETE e RETURN count(e) AS n",
                source=source_key,
            )
            record = await result.single()
            removed = record["n"] if record else 0
        logger.info(
            "Removed %d GLiNER entities for %s",
            removed,
            source_key,
            extra={"file_name": source_key, "entity_count": removed},
        )
    finally:
        await driver.close()


def _handle_delete(source_key: str) -> None:
    """Delete Qdrant points and Graphiti episodes for a source file."""
    logger.info(
        "Handling file delete for %s",
        source_key,
        extra={"file_name": source_key},
    )

    try:
        qdrant = QdrantClient(url=settings.qdrant_url)
        _delete_qdrant_points(qdrant, source_key)
        logger.info(
            "Deleted Qdrant points for %s",
            source_key,
            extra={"file_name": source_key},
        )
    except Exception:
        logger.exception(
            "Failed to delete Qdrant points for %s",
            source_key,
            extra={"file_name": source_key},
        )

    if not settings.graph_enabled:
        return
    try:
        if settings.graph_engine == "gliner":
            run_async(_remove_gliner_entities(source_key))
        else:
            run_async(_remove_graphiti_episodes(source_key))
    except Exception:
        logger.exception(
            "Failed to remove graph data for %s",
            source_key,
            extra={"file_name": source_key},
        )


@cocoindex.op.target_connector(spec_cls=RagTarget)
class RagTargetConnector:
    """CocoIndex target connector that cleans up on file deletion."""

    @staticmethod
    def get_persistent_key(spec: RagTarget, target_name: str) -> str:
        return spec.qdrant_url

    @staticmethod
    def describe(key: str) -> str:
        return f"RAG target (Qdrant: {key})"

    @staticmethod
    def apply_setup_change(
        key: str,
        previous: RagTarget | None,
        current: RagTarget | None,
    ) -> None:
        pass  # Collections managed by ensure_collections()

    @staticmethod
    def mutate(
        *all_mutations: tuple[RagTarget, dict[str, RagTargetValues]],
    ) -> None:
        for _spec, mutations in all_mutations:
            for filename, value in mutations.items():
                if value is None:
                    _handle_delete(filename)


# Backward-compatible aliases
handle_file_delete = _handle_delete
handle_s3_delete = _handle_delete
