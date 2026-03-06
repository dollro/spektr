from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestVlmGraphEngineIngestion:
    @pytest.mark.asyncio
    async def test_visual_page_caption_sent_to_graph(self) -> None:
        """When VLM is enabled, visual page captions are ingested via graph engine."""
        from ingestion.pipeline import _caption_and_ingest_visual

        mock_engine = AsyncMock()
        mock_vlm_caption = AsyncMock(return_value="Chart showing Q3 revenue of $5M")

        with patch(
            "ingestion.pipeline._caption_visual_page",
            mock_vlm_caption,
        ):
            await _caption_and_ingest_visual(
                source_file="report.pdf",
                image_bytes=b"fake-png",
                page_number=3,
                graph_engine=mock_engine,
            )

        mock_vlm_caption.assert_called_once_with(b"fake-png")
        mock_engine.ingest.assert_called_once()
        call_args = mock_engine.ingest.call_args
        chunks = call_args[0][0]
        assert len(chunks) == 1
        assert "Q3 revenue" in chunks[0].text
        assert call_args[0][1] == "report.pdf"

    @pytest.mark.asyncio
    async def test_skipped_when_caption_is_empty(self) -> None:
        """No graph ingestion when VLM returns empty caption."""
        from ingestion.pipeline import _caption_and_ingest_visual

        mock_engine = AsyncMock()
        mock_vlm_caption = AsyncMock(return_value="")

        with patch(
            "ingestion.pipeline._caption_visual_page",
            mock_vlm_caption,
        ):
            await _caption_and_ingest_visual(
                source_file="report.pdf",
                image_bytes=b"fake-png",
                page_number=3,
                graph_engine=mock_engine,
            )

        mock_engine.ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_skipped_when_vlm_fails(self) -> None:
        """Graceful fallback when VLM captioning fails."""
        from ingestion.pipeline import _caption_and_ingest_visual

        mock_engine = AsyncMock()
        mock_vlm_caption = AsyncMock(side_effect=Exception("VLM timeout"))

        with patch(
            "ingestion.pipeline._caption_visual_page",
            mock_vlm_caption,
        ):
            await _caption_and_ingest_visual(
                source_file="report.pdf",
                image_bytes=b"fake-png",
                page_number=3,
                graph_engine=mock_engine,
            )

        mock_engine.ingest.assert_not_called()
