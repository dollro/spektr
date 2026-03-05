from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ingestion.file_processor import TextChunk


class TestPipelineBulkGraphiti:
    @pytest.mark.asyncio
    async def test_ingest_to_graphiti_calls_bulk(self) -> None:
        """_ingest_to_graphiti uses ingest_bulk instead of per-chunk calls."""
        from ingestion.pipeline import _ingest_to_graphiti

        chunks = [
            TextChunk(text="chunk 0", chunk_index=0, page_number=1),
            TextChunk(text="chunk 1", chunk_index=1, page_number=1),
            TextChunk(text="chunk 2", chunk_index=2, page_number=2),
        ]

        mock_writer = AsyncMock()
        await _ingest_to_graphiti("doc.pdf", chunks, mock_writer)

        # Should call ingest_bulk once, not ingest_chunk 3 times
        mock_writer.ingest_bulk.assert_called_once()
        call_kwargs = mock_writer.ingest_bulk.call_args.kwargs
        assert call_kwargs["source_key"] == "doc.pdf"
        assert call_kwargs["chunks"] == chunks
        mock_writer.ingest_chunk.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_to_graphiti_empty_chunks(self) -> None:
        """_ingest_to_graphiti with empty chunks is a no-op."""
        from ingestion.pipeline import _ingest_to_graphiti

        mock_writer = AsyncMock()
        await _ingest_to_graphiti("doc.pdf", [], mock_writer)

        mock_writer.ingest_bulk.assert_called_once()
        call_kwargs = mock_writer.ingest_bulk.call_args.kwargs
        assert call_kwargs["chunks"] == []
