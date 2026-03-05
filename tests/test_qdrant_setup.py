from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models

from config.constants import DENSE_COLLECTION, DENSE_DIM, MULTIVEC_COLLECTION, MULTIVEC_DIM
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
        vec_config = call_kwargs.kwargs["vectors_config"]
        assert isinstance(vec_config, models.VectorParams)
        assert vec_config.size == DENSE_DIM
        assert vec_config.distance == models.Distance.COSINE

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
    def test_creates_dense_only_when_multivec_disabled(
        self, mock_client: MagicMock
    ) -> None:
        with patch("config.settings.settings") as mock_settings:
            mock_settings.multivec_enabled = False
            ensure_collections(mock_client)

        exists_calls = mock_client.collection_exists.call_args_list
        names = {c.args[0] for c in exists_calls}
        assert DENSE_COLLECTION in names
        assert MULTIVEC_COLLECTION not in names
        assert mock_client.create_collection.call_count == 1

    def test_creates_both_when_multivec_enabled(
        self, mock_client: MagicMock
    ) -> None:
        with patch("config.settings.settings") as mock_settings:
            mock_settings.multivec_enabled = True
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

        return QdrantClient(url="http://localhost:6333")

    def test_create_and_verify_dense(self, qdrant_client: MagicMock) -> None:
        # Cleanup if exists
        if qdrant_client.collection_exists(DENSE_COLLECTION):
            qdrant_client.delete_collection(DENSE_COLLECTION)

        create_dense_collection(qdrant_client)
        assert qdrant_client.collection_exists(DENSE_COLLECTION)

        info = qdrant_client.get_collection(DENSE_COLLECTION)
        assert info.config.params.vectors.size == DENSE_DIM  # type: ignore[union-attr]

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
