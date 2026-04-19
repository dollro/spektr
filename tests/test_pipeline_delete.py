from __future__ import annotations

from unittest.mock import MagicMock, patch

from ingestion.target_connector import handle_file_delete


class TestHandleFileDelete:
    def test_deletes_dense_points_by_source_file(self) -> None:
        """Qdrant dense collection points are deleted by source_file filter."""
        mock_qdrant = MagicMock()
        with (
            patch(
                "ingestion.target_connector.QdrantClient",
                return_value=mock_qdrant,
            ),
            patch("ingestion.target_connector.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.multivec_enabled = False
            mock_settings.graph_enabled = False
            handle_file_delete("docs/report.pdf")

        mock_qdrant.delete.assert_called_once()

    def test_deletes_multivec_when_enabled(self) -> None:
        """Qdrant multivec collection is also cleaned when enabled."""
        mock_qdrant = MagicMock()
        with (
            patch(
                "ingestion.target_connector.QdrantClient",
                return_value=mock_qdrant,
            ),
            patch("ingestion.target_connector.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.multivec_enabled = True
            mock_settings.graph_enabled = False
            handle_file_delete("docs/report.pdf")

        assert mock_qdrant.delete.call_count == 2

    def test_backward_compat_alias(self) -> None:
        """handle_s3_delete is an alias for handle_file_delete."""
        from ingestion.target_connector import handle_s3_delete

        assert handle_s3_delete is handle_file_delete
