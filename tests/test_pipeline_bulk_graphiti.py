from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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

    @pytest.mark.asyncio
    async def test_process_text_page_collects_chunks(self) -> None:
        """_process_text_page appends to chunk_collector instead of calling Graphiti."""
        from ingestion.pipeline import _process_text_page

        mock_writer = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(return_value=[[0.1] * 2048])
        mock_qdrant = MagicMock()

        collected: list[TextChunk] = []

        await _process_text_page(
            source_file="test.pdf",
            text="Some text content here.",
            page_number=1,
            mime="application/pdf",
            now="2026-03-05T00:00:00",
            qdrant=mock_qdrant,
            embedder=mock_embedder,
            graphiti_writer=mock_writer,
            chunk_collector=collected,
        )

        # Should NOT call graphiti directly
        mock_writer.ingest_bulk.assert_not_called()
        mock_writer.ingest_chunk.assert_not_called()
        # Chunks should be collected instead
        assert len(collected) > 0
