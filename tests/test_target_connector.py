from __future__ import annotations

from unittest.mock import MagicMock, patch

from ingestion.target_connector import (
    RagTarget,
    RagTargetConnector,
    RagTargetValues,
    _delete_qdrant_points,
    _handle_delete,
)


class TestDeleteQdrantPoints:
    def test_deletes_dense_collection(self) -> None:
        mock_qdrant = MagicMock()
        with patch("ingestion.target_connector.settings") as mock_settings:
            mock_settings.multivec_enabled = False
            _delete_qdrant_points(mock_qdrant, "report.pdf")

        mock_qdrant.delete.assert_called_once()
        call_kwargs = mock_qdrant.delete.call_args
        assert call_kwargs.kwargs["collection_name"] == "documents_dense"

    def test_deletes_multivec_when_enabled(self) -> None:
        mock_qdrant = MagicMock()
        with patch("ingestion.target_connector.settings") as mock_settings:
            mock_settings.multivec_enabled = True
            _delete_qdrant_points(mock_qdrant, "report.pdf")

        assert mock_qdrant.delete.call_count == 2


class TestHandleDelete:
    def test_calls_qdrant_and_graphiti(self) -> None:
        with (
            patch("ingestion.target_connector.QdrantClient"),
            patch("ingestion.target_connector._delete_qdrant_points") as mock_del,
            patch("ingestion.target_connector.settings") as mock_settings,
            patch("ingestion.target_connector.run_async") as mock_run,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.graph_enabled = True
            _handle_delete("report.pdf")

        mock_del.assert_called_once()
        mock_run.assert_called_once()

    def test_skips_graphiti_when_disabled(self) -> None:
        with (
            patch("ingestion.target_connector.QdrantClient"),
            patch("ingestion.target_connector._delete_qdrant_points"),
            patch("ingestion.target_connector.settings") as mock_settings,
            patch("ingestion.target_connector.run_async") as mock_run,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.graph_enabled = False
            _handle_delete("report.pdf")

        mock_run.assert_not_called()


class TestRagTargetConnectorMutate:
    def test_delete_calls_handle_delete(self) -> None:
        spec = RagTarget(qdrant_url="http://localhost:6333")
        with patch("ingestion.target_connector._handle_delete") as mock_del:
            RagTargetConnector.mutate((spec, {"report.pdf": None}))
        mock_del.assert_called_once_with("report.pdf")

    def test_upsert_is_noop(self) -> None:
        spec = RagTarget(qdrant_url="http://localhost:6333")
        values = RagTargetValues(result="report.pdf")
        with patch("ingestion.target_connector._handle_delete") as mock_del:
            RagTargetConnector.mutate((spec, {"report.pdf": values}))
        mock_del.assert_not_called()

    def test_multiple_mutations(self) -> None:
        spec = RagTarget(qdrant_url="http://localhost:6333")
        values = RagTargetValues(result="keep.pdf")
        with patch("ingestion.target_connector._handle_delete") as mock_del:
            RagTargetConnector.mutate(
                (spec, {"delete.pdf": None, "keep.pdf": values})
            )
        mock_del.assert_called_once_with("delete.pdf")


class TestRagTargetConnectorSetup:
    def test_get_persistent_key(self) -> None:
        spec = RagTarget(qdrant_url="http://localhost:6333")
        key = RagTargetConnector.get_persistent_key(spec, "rag_target")
        assert key == "http://localhost:6333"

    def test_apply_setup_change_noop(self) -> None:
        RagTargetConnector.apply_setup_change("key", None, None)

    def test_describe(self) -> None:
        desc = RagTargetConnector.describe("http://localhost:6333")
        assert "Qdrant" in desc
