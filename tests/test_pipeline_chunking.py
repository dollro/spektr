from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from ingestion.file_processor import TextChunk


class TestProcessTextPageWithDoclingChunks:
    async def test_uses_docling_chunks_when_provided(self) -> None:
        """_process_text_page uses pre-computed docling chunks."""
        from ingestion.pipeline import _process_text_page

        dl_chunks = [
            TextChunk(
                text="Revenue grew 15%",
                chunk_index=0,
                page_number=1,
                contextualized_text="Financials > Q3\nRevenue grew 15%",
            ),
            TextChunk(
                text="Expenses decreased",
                chunk_index=1,
                page_number=1,
                contextualized_text="Financials > Q3\nExpenses decreased",
            ),
        ]

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(
            return_value=[[0.1] * 10, [0.2] * 10],
        )
        mock_qdrant = MagicMock()

        with patch("ingestion.pipeline.semantic_chunk") as mock_semantic:
            await _process_text_page(
                source_file="report.pdf",
                text="ignored when docling chunks provided",
                page_number=1,
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=mock_qdrant,
                embedder=mock_embedder,
                graphiti_writer=None,
                docling_chunks=dl_chunks,
            )

        # semantic_chunk should NOT have been called
        mock_semantic.assert_not_called()
        # embed_text should have been called with late_chunking=True
        mock_embedder.embed_text.assert_called_once()
        call_kwargs = mock_embedder.embed_text.call_args
        assert call_kwargs.kwargs.get("late_chunking") is True

    async def test_falls_back_to_semantic_chunk(self) -> None:
        """Without docling chunks, uses semantic_chunk."""
        from ingestion.pipeline import _process_text_page

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(
            return_value=[[0.1] * 10],
        )
        mock_qdrant = MagicMock()

        with patch("ingestion.pipeline.semantic_chunk") as mock_semantic:
            mock_semantic.return_value = [
                TextChunk(
                    text="Fallback chunk",
                    chunk_index=0,
                    page_number=1,
                ),
            ]
            await _process_text_page(
                source_file="report.pdf",
                text="Some text content",
                page_number=1,
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=mock_qdrant,
                embedder=mock_embedder,
                graphiti_writer=None,
                docling_chunks=None,
            )

        mock_semantic.assert_called_once()
        call_kwargs = mock_embedder.embed_text.call_args
        assert call_kwargs.kwargs.get("late_chunking", False) is False


class TestQdrantPayloadContextualizedText:
    async def test_stores_contextualized_text_in_payload(self) -> None:
        """Qdrant point payload includes contextualized_text."""
        from ingestion.pipeline import _process_text_page

        dl_chunks = [
            TextChunk(
                text="Raw text",
                chunk_index=0,
                page_number=1,
                contextualized_text="Heading > Sub\nRaw text",
            ),
        ]

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(
            return_value=[[0.1] * 10],
        )
        mock_qdrant = MagicMock()

        await _process_text_page(
            source_file="doc.pdf",
            text="ignored",
            page_number=1,
            mime="application/pdf",
            now="2026-03-05T00:00:00Z",
            qdrant=mock_qdrant,
            embedder=mock_embedder,
            graphiti_writer=None,
            docling_chunks=dl_chunks,
        )

        upsert_call = mock_qdrant.upsert.call_args
        points = upsert_call.kwargs["points"]
        payload = points[0].payload
        assert payload["text_content"] == "Raw text"
        assert payload["contextualized_text"] == "Heading > Sub\nRaw text"


class TestGraphitiContextualizedText:
    async def test_graphiti_receives_contextualized_text(self) -> None:
        """Graphiti bulk ingestion receives chunks with contextualized_text."""
        from ingestion.pipeline import _ingest_to_graphiti

        chunks = [
            TextChunk(
                text="Raw text",
                chunk_index=0,
                page_number=1,
                contextualized_text="Heading > Sub\nRaw text",
            ),
        ]

        mock_writer = AsyncMock()
        await _ingest_to_graphiti("doc.pdf", chunks, mock_writer)

        mock_writer.ingest_bulk.assert_called_once()
        call_kwargs = mock_writer.ingest_bulk.call_args.kwargs
        assert call_kwargs["chunks"][0].contextualized_text == "Heading > Sub\nRaw text"

    async def test_graphiti_falls_back_to_raw_text(self) -> None:
        """Without contextualized_text, chunk has None contextualized_text."""
        from ingestion.pipeline import _ingest_to_graphiti

        chunks = [
            TextChunk(
                text="Raw text only",
                chunk_index=0,
                page_number=1,
            ),
        ]

        mock_writer = AsyncMock()
        await _ingest_to_graphiti("doc.pdf", chunks, mock_writer)

        mock_writer.ingest_bulk.assert_called_once()
        call_kwargs = mock_writer.ingest_bulk.call_args.kwargs
        assert call_kwargs["chunks"][0].text == "Raw text only"
        assert call_kwargs["chunks"][0].contextualized_text is None
