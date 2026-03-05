from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestVlmGraphitiIngestion:
    @pytest.mark.asyncio
    async def test_visual_page_caption_sent_to_graphiti(self) -> None:
        """When VLM is enabled, visual page captions are ingested to Graphiti."""
        from ingestion.pipeline import _caption_and_ingest_visual

        mock_graphiti_writer = AsyncMock()
        mock_vlm_caption = AsyncMock(return_value="Chart showing Q3 revenue of $5M")

        with patch(
            "ingestion.pipeline._caption_visual_page",
            mock_vlm_caption,
        ):
            await _caption_and_ingest_visual(
                source_file="report.pdf",
                image_bytes=b"fake-png",
                page_number=3,
                graphiti_writer=mock_graphiti_writer,
            )

        mock_vlm_caption.assert_called_once_with(b"fake-png")
        mock_graphiti_writer.ingest_chunk.assert_called_once()
        call_kwargs = mock_graphiti_writer.ingest_chunk.call_args.kwargs
        assert "Q3 revenue" in call_kwargs["chunk_text"]
        assert call_kwargs["source_key"] == "report.pdf"
        assert call_kwargs["page_number"] == 3

    @pytest.mark.asyncio
    async def test_skipped_when_caption_is_empty(self) -> None:
        """No Graphiti ingestion when VLM returns empty caption."""
        from ingestion.pipeline import _caption_and_ingest_visual

        mock_graphiti_writer = AsyncMock()
        mock_vlm_caption = AsyncMock(return_value="")

        with patch(
            "ingestion.pipeline._caption_visual_page",
            mock_vlm_caption,
        ):
            await _caption_and_ingest_visual(
                source_file="report.pdf",
                image_bytes=b"fake-png",
                page_number=3,
                graphiti_writer=mock_graphiti_writer,
            )

        mock_graphiti_writer.ingest_chunk.assert_not_called()

    @pytest.mark.asyncio
    async def test_skipped_when_vlm_fails(self) -> None:
        """Graceful fallback when VLM captioning fails."""
        from ingestion.pipeline import _caption_and_ingest_visual

        mock_graphiti_writer = AsyncMock()
        mock_vlm_caption = AsyncMock(side_effect=Exception("VLM timeout"))

        with patch(
            "ingestion.pipeline._caption_visual_page",
            mock_vlm_caption,
        ):
            await _caption_and_ingest_visual(
                source_file="report.pdf",
                image_bytes=b"fake-png",
                page_number=3,
                graphiti_writer=mock_graphiti_writer,
            )

        mock_graphiti_writer.ingest_chunk.assert_not_called()
