from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ingestion.file_processor import TextChunk


class TestPipelineBulkGraphiti:
    @pytest.mark.asyncio
    async def test_ingest_to_graph_calls_engine_ingest(self) -> None:
        """_ingest_to_graph delegates to engine.ingest."""
        from ingestion.pipeline import _ingest_to_graph

        chunks = [
            TextChunk(text="chunk 0", chunk_index=0, page_number=1),
            TextChunk(text="chunk 1", chunk_index=1, page_number=1),
            TextChunk(text="chunk 2", chunk_index=2, page_number=2),
        ]

        mock_engine = AsyncMock()
        await _ingest_to_graph("doc.pdf", chunks, mock_engine)

        mock_engine.ingest.assert_called_once()
        call_kwargs = mock_engine.ingest.call_args.kwargs
        assert call_kwargs["source_key"] == "doc.pdf"
        assert call_kwargs["chunks"] == chunks

    @pytest.mark.asyncio
    async def test_ingest_to_graph_empty_chunks(self) -> None:
        """_ingest_to_graph with empty chunks still calls engine.ingest."""
        from ingestion.pipeline import _ingest_to_graph

        mock_engine = AsyncMock()
        await _ingest_to_graph("doc.pdf", [], mock_engine)

        mock_engine.ingest.assert_called_once()
        call_kwargs = mock_engine.ingest.call_args.kwargs
        assert call_kwargs["chunks"] == []

    @pytest.mark.asyncio
    async def test_process_text_page_collects_chunks(self) -> None:
        """_process_text_page appends to chunk_collector instead of calling graph."""
        from ingestion.pipeline import _process_text_page

        mock_engine = AsyncMock()
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
            graph_engine=mock_engine,
            chunk_collector=collected,
        )

        # Should NOT call graph engine directly
        mock_engine.ingest.assert_not_called()
        # Chunks should be collected instead
        assert len(collected) > 0
