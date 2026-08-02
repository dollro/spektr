"""CocoIndex v1 custom target that cleans up graph data on source deletion.

Graphiti is an *episodic* writer: it appends episodes with validity intervals.
That does not fit CocoIndex's declared-target-state model, so graph writes stay
side effects inside the per-file component (see ``pipeline.process_file_impl``)
exactly as they were under v0.

What CocoIndex *is* used for is the one thing a side effect cannot do: noticing
that a source file disappeared. v1 has no ``on_delete`` hook on a plain
component — the only mechanism is a target handler whose ``reconcile`` is called
with ``NON_EXISTENCE`` once nothing declares the key any more. This module is
that handler, and it is the direct successor to v0's ``RagTarget``.

The Qdrant half of ``RagTarget`` is gone: points are declared through the native
connector now, so their deletion is handled per point id by CocoIndex itself.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from typing import NamedTuple

import cocoindex as coco

from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Persistent identity of this target. Must stay stable across runs.
_PROVIDER_NAME = "spektr/graph_source"


class GraphSourceState(NamedTuple):
    """What the pipeline declares for one ingested source file.

    ``content_fingerprint`` exists so that re-ingesting a *changed* file is
    visible to the handler. The handler still does nothing on upsert (the write
    already happened as a side effect), but a changing fingerprint keeps the
    tracking record honest rather than frozen at first-ingest.
    """

    source_key: str
    content_fingerprint: str


class _GraphAction(NamedTuple):
    source_key: str
    delete: bool


async def _remove_graphiti_episodes(source_key: str) -> None:
    """Remove Graphiti episodes matching source_description."""
    from ingestion.graphiti_client import get_graphiti

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


async def _remove_gliner_entities(source_key: str) -> None:
    """Remove GLiNER entities and their relationships for a source file."""
    from ingestion.neo4j_setup import get_driver

    driver = get_driver()
    try:
        async with driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity {source: $source}) DETACH DELETE e RETURN count(e) AS n",
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


async def remove_graph_data(source_key: str) -> None:
    """Remove all graph data for one source file, engine-appropriately.

    Errors are logged, never raised: a graph cleanup failure must not abort the
    rest of the batch's reconciliation. This matches v0's ``_handle_delete``.
    """
    if not settings.graph_enabled:
        return
    try:
        if settings.graph_engine == "gliner":
            await _remove_gliner_entities(source_key)
        else:
            await _remove_graphiti_episodes(source_key)
    except Exception:
        logger.exception(
            "Failed to remove graph data for %s",
            source_key,
            extra={"file_name": source_key},
        )


class GraphSourceHandler(coco.TargetHandler[GraphSourceState, str, None]):
    """Reconciles "graph data exists for this source file" as a target state."""

    def __init__(self) -> None:
        self._sink: coco.TargetActionSink[_GraphAction, None] = (
            coco.TargetActionSink.from_async_fn(self._apply_actions)
        )

    async def _apply_actions(
        self,
        context_provider: object,
        actions: Sequence[_GraphAction],
    ) -> None:
        """Run the deletes for a whole reconcile batch.

        Upsert actions are intentionally no-ops — ``process_file_impl`` already
        wrote the episodes. Batching matters for Graphiti: the shared client is
        closed once here rather than once per deleted file, which is what v0 did
        (and which tore the singleton down mid-run).
        """
        deletes = [a.source_key for a in actions if a.delete]
        if not deletes:
            return
        for source_key in deletes:
            logger.info(
                "Source file removed, cleaning up graph data for %s",
                source_key,
                extra={"file_name": source_key},
            )
            await remove_graph_data(source_key)
        if settings.graph_enabled and settings.graph_engine != "gliner":
            from ingestion.graphiti_client import close_graphiti

            await close_graphiti()

    def reconcile(
        self,
        key: coco.StableKey,
        desired_target_state: GraphSourceState | coco.NonExistenceType,
        prev_possible_records: Collection[str],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[_GraphAction, str] | None:
        source_key = str(key)
        if coco.is_non_existence(desired_target_state):
            if not prev_possible_records and not prev_may_be_missing:
                # Never tracked — nothing of ours to clean up.
                return None
            return coco.TargetReconcileOutput(
                action=_GraphAction(source_key=source_key, delete=True),
                sink=self._sink,
                tracking_record=coco.NON_EXISTENCE,
            )

        fingerprint = desired_target_state.content_fingerprint
        if not prev_may_be_missing and all(p == fingerprint for p in prev_possible_records):
            return None
        return coco.TargetReconcileOutput(
            action=_GraphAction(source_key=source_key, delete=False),
            sink=self._sink,
            tracking_record=fingerprint,
        )


_provider = coco.register_root_target_states_provider(_PROVIDER_NAME, GraphSourceHandler())


def declare_graph_source(source_key: str, content_fingerprint: str) -> None:
    """Declare that graph data exists for ``source_key`` at this fingerprint.

    Call from inside the per-file processing component. When the file later
    disappears from the source, CocoIndex reconciles this key to non-existence
    and the handler removes the episodes/entities.
    """
    coco.declare_target_state(
        _provider.target_state(
            source_key,
            GraphSourceState(source_key=source_key, content_fingerprint=content_fingerprint),
        )
    )


# Supported entry point for out-of-band cleanup (tests, manual repair). v0
# exposed this as ``target_connector.handle_file_delete``.
async def handle_file_delete(source_key: str) -> None:
    """Remove graph data for one source file, outside of a CocoIndex run."""
    await remove_graph_data(source_key)
