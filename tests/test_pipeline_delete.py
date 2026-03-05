from __future__ import annotations

from unittest.mock import MagicMock, patch

from ingestion.pipeline import handle_file_delete


class TestHandleFileDelete:
    def test_deletes_dense_points_by_source_file(self) -> None:
        """Qdrant dense collection points are deleted by source_file filter."""
        mock_qdrant = MagicMock()
        with patch("ingestion.pipeline._get_qdrant_client", return_value=mock_qdrant):
            with patch("ingestion.pipeline.settings") as mock_settings:
                mock_settings.multivec_enabled = False
                handle_file_delete("docs/report.pdf")

        mock_qdrant.delete.assert_called_once()

    def test_deletes_multivec_when_enabled(self) -> None:
        """Qdrant multivec collection is also cleaned when enabled."""
        mock_qdrant = MagicMock()
        with patch("ingestion.pipeline._get_qdrant_client", return_value=mock_qdrant):
            with patch("ingestion.pipeline.settings") as mock_settings:
                mock_settings.multivec_enabled = True
                handle_file_delete("docs/report.pdf")

        assert mock_qdrant.delete.call_count == 2

    def test_backward_compat_alias(self) -> None:
        """handle_s3_delete is an alias for handle_file_delete."""
        from ingestion.pipeline import handle_s3_delete

        assert handle_s3_delete is handle_file_delete
