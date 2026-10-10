from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient, models

from config.constants import (
    DENSE_COLLECTION,
    DENSE_DIM,
    DENSE_VECTOR_NAME,
    MULTIVEC_COLLECTION,
    MULTIVEC_DIM,
    SPARSE_VECTOR_NAME,
)
from config.settings import settings
from ingestion.qdrant_setup import (
    CollectionMismatchError,
    create_dense_collection,
    create_multivec_collection,
    ensure_collections,
)


def _existing(vectors: object) -> MagicMock:
    """A client whose collection already exists with the given vectors config."""
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.config.params.vectors = vectors
    return client


def _dense(size: int) -> dict[str, models.VectorParams]:
    return {DENSE_VECTOR_NAME: models.VectorParams(size=size, distance=models.Distance.COSINE)}


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = False
    return client


class TestCreateDenseCollection:
    def test_creates_collection_with_correct_params(self, mock_client: MagicMock) -> None:
        create_dense_collection(mock_client)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == DENSE_COLLECTION
        vec_config = call_kwargs.kwargs["vectors_config"][DENSE_VECTOR_NAME]
        assert isinstance(vec_config, models.VectorParams)
        assert vec_config.size == settings.dense_dimensions
        assert vec_config.distance == models.Distance.COSINE

    def test_dense_collection_uses_named_vectors(self) -> None:
        """Dense and sparse vectors are both named, sparse uses IDF."""
        client = MagicMock()
        client.collection_exists.return_value = False
        create_dense_collection(client)

        kwargs = client.create_collection.call_args.kwargs
        assert kwargs["collection_name"] == DENSE_COLLECTION
        assert DENSE_VECTOR_NAME in kwargs["vectors_config"]
        assert SPARSE_VECTOR_NAME in kwargs["sparse_vectors_config"]
        sparse_cfg = kwargs["sparse_vectors_config"][SPARSE_VECTOR_NAME]
        assert sparse_cfg.modifier == models.Modifier.IDF

    def test_creates_payload_indexes(self, mock_client: MagicMock) -> None:
        create_dense_collection(mock_client)

        index_calls = mock_client.create_payload_index.call_args_list
        assert len(index_calls) == 2
        fields = {c.kwargs["field_name"] for c in index_calls}
        assert fields == {"source_file", "content_type"}
        for c in index_calls:
            assert c.kwargs["field_schema"] == models.PayloadSchemaType.KEYWORD

    def test_idempotent_skips_existing(self) -> None:
        client = _existing(_dense(settings.dense_dimensions))
        create_dense_collection(client)
        client.create_collection.assert_not_called()

    def test_existing_collection_with_other_size_fails_loudly(self) -> None:
        """A model switch without a re-index must not boot.

        Otherwise every write is rejected point by point by Qdrant, and a
        query embedding of the wrong size fails only at search time.
        """
        stale = settings.dense_dimensions + 256
        client = _existing(_dense(stale))

        with pytest.raises(CollectionMismatchError) as exc:
            create_dense_collection(client)

        msg = str(exc.value)
        assert str(stale) in msg
        assert str(settings.dense_dimensions) in msg
        assert "EMBEDDING_MODEL" in msg
        assert "--full-reprocess" in msg
        client.create_collection.assert_not_called()

    def test_existing_collection_without_named_dense_vector_fails(self) -> None:
        client = _existing(
            models.VectorParams(
                size=settings.dense_dimensions, distance=models.Distance.COSINE
            )
        )

        with pytest.raises(CollectionMismatchError, match="named vector"):
            create_dense_collection(client)


class TestCreateMultivecCollection:
    @pytest.fixture(autouse=True)
    def _filled_slot(self):  # type: ignore[no-untyped-def]
        """Size the collection from the slot's model, not a constant.

        `MULTIVEC_DIM` used to be a module constant, so these tests needed no
        configuration. It is now `MultivecSpec.dimensions`, because the width
        is a property of the model — a constant could not vary with the model
        it was sizing. Provisioning therefore requires a filled slot, which
        `ensure_collections` guarantees in production.
        """
        from config.multivec_models import spec

        with patch("ingestion.qdrant_setup.settings") as mock_settings:
            mock_settings.multivec_dimensions = spec("jina-v4-colbert").dimensions
            mock_settings.multivec_slot_filled = True
            yield mock_settings

    def test_creates_collection_with_colbert_config(self, mock_client: MagicMock) -> None:
        from config.multivec_models import spec

        create_multivec_collection(mock_client)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == MULTIVEC_COLLECTION
        vec_config = call_kwargs.kwargs["vectors_config"]
        assert "colbert" in vec_config
        colbert = vec_config["colbert"]
        assert colbert.size == spec("jina-v4-colbert").dimensions
        assert colbert.distance == models.Distance.COSINE
        assert colbert.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM

    def test_creates_source_file_index(self, mock_client: MagicMock) -> None:
        create_multivec_collection(mock_client)

        index_calls = mock_client.create_payload_index.call_args_list
        assert len(index_calls) == 1
        assert index_calls[0].kwargs["field_name"] == "source_file"

    def test_idempotent_skips_existing(self, _filled_slot: MagicMock) -> None:
        size = _filled_slot.multivec_dimensions
        client = _existing(
            {"colbert": models.VectorParams(size=size, distance=models.Distance.COSINE)}
        )
        create_multivec_collection(client)
        client.create_collection.assert_not_called()

    def test_existing_collection_with_other_size_fails_loudly(
        self, _filled_slot: MagicMock
    ) -> None:
        stale = _filled_slot.multivec_dimensions * 2
        client = _existing(
            {"colbert": models.VectorParams(size=stale, distance=models.Distance.COSINE)}
        )

        with pytest.raises(CollectionMismatchError, match="MULTIVEC_MODEL"):
            create_multivec_collection(client)


class TestEnsureCollections:
    def test_creates_dense_only_when_slot_empty(self, mock_client: MagicMock) -> None:
        with patch("ingestion.qdrant_setup.settings") as mock_settings:
            mock_settings.multivec_slot_filled = False
            mock_settings.dense_dimensions = settings.dense_dimensions
            ensure_collections(mock_client)

        exists_calls = mock_client.collection_exists.call_args_list
        names = {c.args[0] for c in exists_calls}
        assert DENSE_COLLECTION in names
        assert MULTIVEC_COLLECTION not in names
        assert mock_client.create_collection.call_count == 1

    def test_creates_both_when_multivec_slot_filled(self, mock_client: MagicMock) -> None:
        with patch("ingestion.qdrant_setup.settings") as mock_settings:
            mock_settings.multivec_slot_filled = True
            mock_settings.dense_dimensions = settings.dense_dimensions
            ensure_collections(mock_client)

        exists_calls = mock_client.collection_exists.call_args_list
        names = {c.args[0] for c in exists_calls}
        assert DENSE_COLLECTION in names
        assert MULTIVEC_COLLECTION in names
        assert mock_client.create_collection.call_count == 2


@pytest.mark.integration
class TestQdrantSetupIntegration:
    @pytest.fixture
    def qdrant_client(self) -> QdrantClient:
        """A REAL client — this class is integration-marked, not mocked.

        The annotation said MagicMock, which is what the unit fixtures above
        return; that made the two look interchangeable when they are not.
        """
        from config.settings import settings

        return QdrantClient(url=settings.qdrant_url)

    def test_create_and_verify_dense(self, qdrant_client: QdrantClient) -> None:
        # Cleanup if exists
        if qdrant_client.collection_exists(DENSE_COLLECTION):
            qdrant_client.delete_collection(DENSE_COLLECTION)

        create_dense_collection(qdrant_client)
        assert qdrant_client.collection_exists(DENSE_COLLECTION)

        info = qdrant_client.get_collection(DENSE_COLLECTION)
        dense = info.config.params.vectors[DENSE_VECTOR_NAME]  # type: ignore[index]
        assert dense.size == DENSE_DIM
        assert SPARSE_VECTOR_NAME in info.config.params.sparse_vectors  # type: ignore[operator]

        # Idempotency
        create_dense_collection(qdrant_client)
        assert qdrant_client.collection_exists(DENSE_COLLECTION)

        # Cleanup
        qdrant_client.delete_collection(DENSE_COLLECTION)

    def test_create_and_verify_multivec(self, qdrant_client: QdrantClient) -> None:
        if qdrant_client.collection_exists(MULTIVEC_COLLECTION):
            qdrant_client.delete_collection(MULTIVEC_COLLECTION)

        create_multivec_collection(qdrant_client)
        assert qdrant_client.collection_exists(MULTIVEC_COLLECTION)

        info = qdrant_client.get_collection(MULTIVEC_COLLECTION)
        colbert = info.config.params.vectors["colbert"]  # type: ignore[index]
        assert colbert.size == MULTIVEC_DIM

        # Idempotency
        create_multivec_collection(qdrant_client)
        assert qdrant_client.collection_exists(MULTIVEC_COLLECTION)

        # Cleanup
        qdrant_client.delete_collection(MULTIVEC_COLLECTION)
