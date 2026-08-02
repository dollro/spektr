from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models

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
    create_dense_collection,
    create_multivec_collection,
    ensure_collections,
)


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

    def test_idempotent_skips_existing(self, mock_client: MagicMock) -> None:
        mock_client.collection_exists.return_value = True
        create_dense_collection(mock_client)
        mock_client.create_collection.assert_not_called()


class TestCreateMultivecCollection:
    def test_creates_collection_with_colbert_config(self, mock_client: MagicMock) -> None:
        create_multivec_collection(mock_client)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == MULTIVEC_COLLECTION
        vec_config = call_kwargs.kwargs["vectors_config"]
        assert "colbert" in vec_config
        colbert = vec_config["colbert"]
        assert colbert.size == MULTIVEC_DIM
        assert colbert.distance == models.Distance.COSINE
        assert colbert.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM

    def test_creates_source_file_index(self, mock_client: MagicMock) -> None:
        create_multivec_collection(mock_client)

        index_calls = mock_client.create_payload_index.call_args_list
        assert len(index_calls) == 1
        assert index_calls[0].kwargs["field_name"] == "source_file"

    def test_idempotent_skips_existing(self, mock_client: MagicMock) -> None:
        mock_client.collection_exists.return_value = True
        create_multivec_collection(mock_client)
        mock_client.create_collection.assert_not_called()


class TestEnsureCollections:
    def test_creates_dense_only_when_multivec_disabled(self, mock_client: MagicMock) -> None:
        with patch("ingestion.qdrant_setup.settings") as mock_settings:
            mock_settings.multivec_enabled = False
            mock_settings.dense_dimensions = settings.dense_dimensions
            ensure_collections(mock_client)

        exists_calls = mock_client.collection_exists.call_args_list
        names = {c.args[0] for c in exists_calls}
        assert DENSE_COLLECTION in names
        assert MULTIVEC_COLLECTION not in names
        assert mock_client.create_collection.call_count == 1

    def test_creates_both_when_multivec_enabled(self, mock_client: MagicMock) -> None:
        with patch("ingestion.qdrant_setup.settings") as mock_settings:
            mock_settings.multivec_enabled = True
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
    def qdrant_client(self) -> MagicMock:
        from qdrant_client import QdrantClient

        from config.settings import settings

        return QdrantClient(url=settings.qdrant_url)

    def test_create_and_verify_dense(self, qdrant_client: MagicMock) -> None:
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

    def test_create_and_verify_multivec(self, qdrant_client: MagicMock) -> None:
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
