from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from config.constants import (
    DENSE_COLLECTION,
    DENSE_VECTOR_NAME,
    MULTIVEC_COLLECTION,
    MULTIVEC_DIM,
    SPARSE_VECTOR_NAME,
)
from config.settings import settings

logger = logging.getLogger(__name__)


def create_dense_collection(client: QdrantClient) -> None:
    """Create documents_dense collection.

    Named vectors: "dense" (dimensions from settings, provider-dependent,
    COSINE) and "sparse" (miniCOIL, IDF modifier).
    Payload indexes: source_file (KEYWORD), content_type (KEYWORD).
    Idempotent — skips if collection already exists.
    """
    if client.collection_exists(DENSE_COLLECTION):
        logger.info("Collection %s already exists, skipping.", DENSE_COLLECTION)
        return

    dim = settings.dense_dimensions
    client.create_collection(
        collection_name=DENSE_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )
    client.create_payload_index(
        collection_name=DENSE_COLLECTION,
        field_name="source_file",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=DENSE_COLLECTION,
        field_name="content_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    logger.info("Created collection %s.", DENSE_COLLECTION)


def create_multivec_collection(client: QdrantClient) -> None:
    """Create documents_multivec collection.

    Named vector 'colbert': size=128, distance=COSINE, MaxSim comparator.
    Payload index: source_file (KEYWORD).
    Idempotent — skips if collection already exists.
    """
    if client.collection_exists(MULTIVEC_COLLECTION):
        logger.info("Collection %s already exists, skipping.", MULTIVEC_COLLECTION)
        return

    client.create_collection(
        collection_name=MULTIVEC_COLLECTION,
        vectors_config={
            "colbert": models.VectorParams(
                size=MULTIVEC_DIM,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
        },
    )
    client.create_payload_index(
        collection_name=MULTIVEC_COLLECTION,
        field_name="source_file",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    logger.info("Created collection %s.", MULTIVEC_COLLECTION)


def ensure_collections(client: QdrantClient) -> None:
    """Create dense (and optionally multivec) collections (idempotent)."""
    create_dense_collection(client)
    if settings.multivec_enabled:
        create_multivec_collection(client)
