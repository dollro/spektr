from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.file_processor import TextChunk


class TestGraphitiWriterBulk:
    @pytest.mark.asyncio
    async def test_ingest_bulk_builds_raw_episodes(self) -> None:
        """ingest_bulk creates RawEpisode per chunk and calls add_episode_bulk."""
        from ingestion.graph_writer import GraphitiWriter

        chunks = [
            TextChunk(text="chunk zero", chunk_index=0, page_number=1),
            TextChunk(
                text="chunk one raw",
                chunk_index=1,
                page_number=1,
                contextualized_text="## Heading\nchunk one raw",
            ),
            TextChunk(text="chunk two", chunk_index=2, page_number=2),
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(return_value=MagicMock())

        with patch(
            "ingestion.graph_writer.get_graphiti",
            return_value=mock_client,
        ):
            writer = GraphitiWriter()
            await writer.ingest_bulk(
                chunks=chunks,
                source_key="test.pdf",
            )

        mock_client.add_episode_bulk.assert_called_once()
        episodes = mock_client.add_episode_bulk.call_args.args[0]
        assert len(episodes) == 3
        # Chunk 1 should use contextualized_text
        assert episodes[1].content == "## Heading\nchunk one raw"
        # Chunk 0 should use raw text
        assert episodes[0].content == "chunk zero"
        # Names should encode source/page/chunk
        assert episodes[0].name == "test.pdf:p1:c0"
        assert episodes[2].name == "test.pdf:p2:c2"
        # source_description should be source_key
        assert episodes[0].source_description == "test.pdf"

    @pytest.mark.asyncio
    async def test_ingest_bulk_empty_chunks_is_noop(self) -> None:
        """ingest_bulk with empty chunk list does not call Graphiti."""
        from ingestion.graph_writer import GraphitiWriter

        mock_client = AsyncMock()

        with patch(
            "ingestion.graph_writer.get_graphiti",
            return_value=mock_client,
        ):
            writer = GraphitiWriter()
            await writer.ingest_bulk(chunks=[], source_key="test.pdf")

        mock_client.add_episode_bulk.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_bulk_falls_back_on_error(self) -> None:
        """When add_episode_bulk fails, falls back to sequential add_episode."""
        from ingestion.graph_writer import GraphitiWriter

        chunks = [
            TextChunk(text="chunk A", chunk_index=0, page_number=1),
            TextChunk(text="chunk B", chunk_index=1, page_number=1),
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(
            side_effect=Exception("NodeResolutions ValidationError")
        )
        mock_client.add_episode = AsyncMock()

        with patch(
            "ingestion.graph_writer.get_graphiti",
            return_value=mock_client,
        ):
            writer = GraphitiWriter()
            await writer.ingest_bulk(
                chunks=chunks,
                source_key="test.pdf",
            )

        # Bulk failed, so sequential should have been called per chunk
        assert mock_client.add_episode.call_count == 2

    @pytest.mark.asyncio
    async def test_ingest_bulk_fallback_logs_individual_failures(
        self,
    ) -> None:
        """Sequential fallback continues on individual chunk failures."""
        from ingestion.graph_writer import GraphitiWriter

        chunks = [
            TextChunk(text="ok chunk", chunk_index=0, page_number=1),
            TextChunk(text="bad chunk", chunk_index=1, page_number=1),
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(
            side_effect=Exception("bulk failed")
        )
        # Second sequential call fails
        mock_client.add_episode = AsyncMock(
            side_effect=[None, Exception("single failed")]
        )

        with patch(
            "ingestion.graph_writer.get_graphiti",
            return_value=mock_client,
        ):
            writer = GraphitiWriter()
            # Should not raise
            await writer.ingest_bulk(
                chunks=chunks,
                source_key="test.pdf",
            )

        assert mock_client.add_episode.call_count == 2
