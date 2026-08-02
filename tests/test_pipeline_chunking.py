from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models

from ingestion.file_processor import TextChunk


def _fake_sparse(texts: list[str]) -> list[models.SparseVector]:
    return [models.SparseVector(indices=[1], values=[0.5]) for _ in texts]


@pytest.fixture(autouse=True)
def stub_sparse_encoder():
    """Keep unit tests off the real miniCOIL encoder.

    ``encode_documents`` lazily downloads ``Qdrant/minicoil-v1`` via fastembed;
    a unit test must not depend on network access or a model cache.
    """
    with patch("ingestion.page_processor.encode_documents", side_effect=_fake_sparse):
        yield


class TestProcessTextPageWithDoclingChunks:
    async def test_uses_docling_chunks_when_provided(self) -> None:
        """_process_text_page uses pre-computed docling chunks."""
        from ingestion.page_processor import _process_text_page

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
        mock_dense = MagicMock()

        with patch("ingestion.page_processor.semantic_chunk") as mock_semantic:
            await _process_text_page(
                source_file="report.pdf",
                text="ignored when docling chunks provided",
                page_number=1,
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                dense=mock_dense,
                embedder=mock_embedder,
                graph_engine=None,
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
        from ingestion.page_processor import _process_text_page

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(
            return_value=[[0.1] * 10],
        )
        mock_dense = MagicMock()

        with patch("ingestion.page_processor.semantic_chunk") as mock_semantic:
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
                dense=mock_dense,
                embedder=mock_embedder,
                graph_engine=None,
                docling_chunks=None,
            )

        mock_semantic.assert_called_once()
        call_kwargs = mock_embedder.embed_text.call_args
        assert call_kwargs.kwargs.get("late_chunking", False) is False


class TestQdrantPayloadContextualizedText:
    async def test_stores_contextualized_text_in_payload(self) -> None:
        """Qdrant point payload includes contextualized_text."""
        from ingestion.page_processor import _process_text_page

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
        mock_dense = MagicMock()

        await _process_text_page(
            source_file="doc.pdf",
            text="ignored",
            page_number=1,
            mime="application/pdf",
            now="2026-03-05T00:00:00Z",
            dense=mock_dense,
            embedder=mock_embedder,
            graph_engine=None,
            docling_chunks=dl_chunks,
        )

        point = mock_dense.declare_point.call_args.args[0]
        payload = point.payload
        assert payload["text_content"] == "Raw text"
        assert payload["contextualized_text"] == "Heading > Sub\nRaw text"


class TestEmbeddingFailurePropagates:
    """A page we fail to declare is a page CocoIndex deletes.

    Under v0 these points were upserted directly, so swallowing an embedding
    error merely skipped an upsert and previously-written points survived.
    Under v1 the points are declared, so swallowing would (a) reconcile the
    page's existing points to non-existence and (b) memoize the file, so it
    would never be retried. The failure must propagate instead.
    """

    async def test_text_embedding_failure_raises(self) -> None:
        from ingestion.page_processor import _process_text_page

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(side_effect=RuntimeError("429"))
        mock_dense = MagicMock()

        with pytest.raises(RuntimeError):
            await _process_text_page(
                source_file="doc.pdf",
                text="some text that will chunk",
                page_number=1,
                mime="text/plain",
                now="2026-03-05T00:00:00Z",
                dense=mock_dense,
                embedder=mock_embedder,
                graph_engine=None,
            )
        mock_dense.declare_point.assert_not_called()

    async def test_image_embedding_failure_raises(self) -> None:
        from ingestion.page_processor import _process_visual_page

        mock_embedder = AsyncMock()
        mock_embedder.embed_image = AsyncMock(side_effect=RuntimeError("429"))
        mock_dense = MagicMock()

        with pytest.raises(RuntimeError):
            await _process_visual_page(
                "doc.pdf",
                b"fake-png",
                1,
                "image",
                "image/png",
                "2026-03-05T00:00:00Z",
                mock_dense,
                None,
                mock_embedder,
            )
        mock_dense.declare_point.assert_not_called()


class TestSparseEncodeBatching:
    async def test_one_encode_call_per_page_not_per_chunk(self) -> None:
        """miniCOIL is encoded once with every chunk of the page.

        Under v0 this ran inside _build_chunk_point, i.e. once per chunk.
        """
        from ingestion.page_processor import _process_text_page

        dl_chunks = [
            TextChunk(text="one", chunk_index=0, page_number=1),
            TextChunk(text="two", chunk_index=1, page_number=1),
            TextChunk(text="three", chunk_index=2, page_number=1),
        ]
        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(return_value=[[0.1] * 10] * 3)
        mock_dense = MagicMock()

        with patch(
            "ingestion.page_processor.encode_documents", side_effect=_fake_sparse
        ) as enc:
            await _process_text_page(
                source_file="doc.pdf",
                text="ignored",
                page_number=1,
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                dense=mock_dense,
                embedder=mock_embedder,
                graph_engine=None,
                docling_chunks=dl_chunks,
            )

        enc.assert_called_once_with(["one", "two", "three"])
        assert mock_dense.declare_point.call_count == 3


class TestBuildChunkPoint:
    def test_text_chunk_points_carry_named_dense_and_sparse(self) -> None:
        """Declared points use the named-vector dict with a sparse entry."""
        from qdrant_client import models

        from config.constants import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
        from ingestion.page_processor import _build_chunk_point

        point = _build_chunk_point(
            source_file="doc.pdf",
            page_number=1,
            chunk_index=0,
            text="hello",
            contextualized_text=None,
            vector=[0.1] * 512,
            sparse_vector=models.SparseVector(indices=[1], values=[0.5]),
            mime="application/pdf",
            now="2026-07-31T00:00:00Z",
            embedder_model="jina-embeddings-v4",
            embedder_dim=512,
        )

        assert DENSE_VECTOR_NAME in point.vector
        assert SPARSE_VECTOR_NAME in point.vector
        assert point.payload["text_content"] == "hello"


class TestGraphEngineContextualizedText:
    async def test_engine_receives_contextualized_text(self) -> None:
        """Graph engine ingestion receives chunks with contextualized_text."""
        from ingestion.pipeline import _ingest_to_graph

        chunks = [
            TextChunk(
                text="Raw text",
                chunk_index=0,
                page_number=1,
                contextualized_text="Heading > Sub\nRaw text",
            ),
        ]

        mock_engine = AsyncMock()
        await _ingest_to_graph("doc.pdf", chunks, mock_engine)

        mock_engine.ingest.assert_called_once()
        call_kwargs = mock_engine.ingest.call_args.kwargs
        assert call_kwargs["chunks"][0].contextualized_text == "Heading > Sub\nRaw text"

    async def test_engine_falls_back_to_raw_text(self) -> None:
        """Without contextualized_text, chunk has None contextualized_text."""
        from ingestion.pipeline import _ingest_to_graph

        chunks = [
            TextChunk(
                text="Raw text only",
                chunk_index=0,
                page_number=1,
            ),
        ]

        mock_engine = AsyncMock()
        await _ingest_to_graph("doc.pdf", chunks, mock_engine)

        mock_engine.ingest.assert_called_once()
        call_kwargs = mock_engine.ingest.call_args.kwargs
        assert call_kwargs["chunks"][0].text == "Raw text only"
        assert call_kwargs["chunks"][0].contextualized_text is None
