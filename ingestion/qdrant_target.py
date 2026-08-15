"""CocoIndex v1 Qdrant target wiring for the bulk ingestion pipeline.

Keeps every reference to the native ``cocoindex.connectors.qdrant`` connector in
one place, so ``pipeline.py`` only ever sees ``CollectionTarget`` handles.

Two invariants are load-bearing here:

* **``managed_by=USER``**. CocoIndex must never create, replace or drop
  ``documents_dense``. A collection *replacement* — which any change to the
  declared vector schema would trigger — drops every point in the collection,
  including Path B's live-session points that share it. Under ``USER`` the
  engine resolves no collection-level action at all, and
  ``ingestion.qdrant_setup.ensure_collections`` stays the sole provisioning
  authority.
* **Per-point reconciliation is keyed on the point id.** Deletes are issued as
  explicit id lists; CocoIndex never enumerates the collection, so points it did
  not declare (again: Path B's live sessions) are invisible to it.
"""

from __future__ import annotations

from typing import Any

import cocoindex as coco
import numpy as np
from cocoindex.connectorkits import target
from cocoindex.connectors import qdrant
from cocoindex.resources.schema import MultiVectorSchema, VectorSchema
from qdrant_client import QdrantClient

from config.constants import (
    DENSE_COLLECTION,
    DENSE_VECTOR_NAME,
    MULTIVEC_COLLECTION,
    MULTIVEC_DIM,
    SPARSE_VECTOR_NAME,
)
from config.settings import settings

# Identity of the Qdrant client in CocoIndex's context registry. The string is
# part of every Qdrant target's persistent key, so it must stay stable forever —
# renaming it orphans the tracking records and re-declares every point.
QDRANT_DB: coco.ContextKey[QdrantClient] = coco.ContextKey("spektr/qdrant")

CollectionTarget = qdrant.CollectionTarget


def _float32(size: int) -> VectorSchema:
    return VectorSchema(dtype=np.dtype(np.float32), size=size)


async def build_dense_schema() -> qdrant.CollectionSchema:
    """Schema for ``documents_dense``: named dense + sparse (IDF) vectors.

    Mirrors :func:`ingestion.qdrant_setup.create_dense_collection`. Because the
    target is user-managed the schema is never applied to Qdrant — it only ends
    up in CocoIndex's tracking record — but it must still describe reality so
    that switching away from ``USER`` could never silently reshape a collection.
    """
    return await qdrant.CollectionSchema.create(
        vectors={
            DENSE_VECTOR_NAME: qdrant.QdrantVectorDef(
                schema=_float32(settings.dense_dimensions),
                distance="cosine",
            ),
            SPARSE_VECTOR_NAME: qdrant.QdrantSparseVectorDef(modifier="idf"),
        },
    )


async def build_multivec_schema() -> qdrant.CollectionSchema:
    """Schema for ``documents_multivec``: one ColBERT multi-vector, MaxSim."""
    return await qdrant.CollectionSchema.create(
        vectors={
            "colbert": qdrant.QdrantVectorDef(
                schema=MultiVectorSchema(vector_schema=_float32(MULTIVEC_DIM)),
                distance="cosine",
                multivector_comparator="max_sim",
            ),
        },
    )


async def mount_dense_target() -> Any:
    """Mount ``documents_dense`` as a user-managed target."""
    return await qdrant.mount_collection_target(
        QDRANT_DB,
        collection_name=DENSE_COLLECTION,
        schema=await build_dense_schema(),
        managed_by=target.ManagedBy.USER,
    )


async def mount_multivec_target() -> Any:
    """Mount ``documents_multivec`` as a user-managed target."""
    return await qdrant.mount_collection_target(
        QDRANT_DB,
        collection_name=MULTIVEC_COLLECTION,
        schema=await build_multivec_schema(),
        managed_by=target.ManagedBy.USER,
    )


def create_qdrant_client() -> QdrantClient:
    """Build the QdrantClient the pipeline provides into CocoIndex's context.

    Deliberately not ``qdrant.create_client``: that defaults to ``prefer_grpc``,
    while the rest of Spektr talks HTTP on ``settings.qdrant_url``.
    """
    return QdrantClient(url=settings.qdrant_url)
