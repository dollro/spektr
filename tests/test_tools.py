"""Integration tests for MCP search tools.

Tests verify tool functions directly (not via MCP protocol) against
seeded Qdrant and Neo4j instances.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.constants import (
    DENSE_COLLECTION,
    DENSE_DIM,
    MULTIVEC_COLLECTION,
    MULTIVEC_DIM,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dense_point(
    idx: int,
    text: str = "sample text",
    source_file: str = "doc.pdf",
    page_number: int = 1,
    content_type: str = "pdf",
) -> dict:
    """Build a Qdrant point payload dict for the dense collection."""
    return {
        "id": idx,
        "vector": [0.1 * (idx + 1)] * DENSE_DIM,
        "payload": {
            "text": text,
            "source_file": source_file,
            "page_number": page_number,
            "content_type": content_type,
            "metadata": {},
        },
    }


def _make_multivec_point(
    idx: int,
    source_file: str = "doc.pdf",
    page_number: int = 1,
    content_type: str = "pdf",
    source_key: str = "documents/doc.pdf",
) -> dict:
    """Build a Qdrant point payload dict for the multivec collection."""
    return {
        "id": idx,
        "vector": {"colbert": [[0.2 * (idx + 1)] * MULTIVEC_DIM] * 5},
        "payload": {
            "source_file": source_file,
            "page_number": page_number,
            "content_type": content_type,
            "source_key": source_key,
            "metadata": {},
        },
    }


def _seed_dense(client, points: list[dict]) -> None:  # type: ignore[no-untyped-def]
    """Upsert points into the dense collection."""
    from qdrant_client.models import PointStruct

    client.upsert(
        collection_name=DENSE_COLLECTION,
        points=[
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points
        ],
    )


def _seed_multivec(client, points: list[dict]) -> None:  # type: ignore[no-untyped-def]
    """Upsert points into the multivec collection."""
    from qdrant_client.models import PointStruct

    client.upsert(
        collection_name=MULTIVEC_COLLECTION,
        points=[
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points
        ],
    )


# ---------------------------------------------------------------------------
# vector_search tests
# ---------------------------------------------------------------------------


class TestVectorSearch:
    """Tests for the vector_search tool."""

    @pytest.mark.integration
    async def test_returns_sorted_results(self, qdrant_client, mock_embedder):
        """Results are returned sorted by score descending."""
        _seed_dense(
            qdrant_client,
            [
                _make_dense_point(1, text="first"),
                _make_dense_point(2, text="second"),
            ],
        )

        with (
            patch(
                "server.tools.vector_search._qdrant_client",
                qdrant_client,
            ),
            patch(
                "server.tools.vector_search._embedder",
                mock_embedder,
            ),
        ):
            from server.tools.vector_search import vector_search

            results = await vector_search("test query", limit=10)

        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]

    @pytest.mark.integration
    async def test_content_type_filter(self, qdrant_client, mock_embedder):
        """content_type filter restricts results."""
        _seed_dense(
            qdrant_client,
            [
                _make_dense_point(1, content_type="pdf"),
                _make_dense_point(2, content_type="text"),
            ],
        )

        with (
            patch(
                "server.tools.vector_search._qdrant_client",
                qdrant_client,
            ),
            patch(
                "server.tools.vector_search._embedder",
                mock_embedder,
            ),
        ):
            from server.tools.vector_search import vector_search

            results = await vector_search("test", content_type="text")

        assert len(results) == 1
        assert results[0]["content_type"] == "text"

    @pytest.mark.integration
    async def test_source_file_filter(self, qdrant_client, mock_embedder):
        """source_file filter restricts results."""
        _seed_dense(
            qdrant_client,
            [
                _make_dense_point(1, source_file="a.pdf"),
                _make_dense_point(2, source_file="b.pdf"),
            ],
        )

        with (
            patch(
                "server.tools.vector_search._qdrant_client",
                qdrant_client,
            ),
            patch(
                "server.tools.vector_search._embedder",
                mock_embedder,
            ),
        ):
            from server.tools.vector_search import vector_search

            results = await vector_search("test", source_file="b.pdf")

        assert len(results) == 1
        assert results[0]["source_file"] == "b.pdf"

    @pytest.mark.integration
    async def test_empty_results(self, qdrant_client, mock_embedder):
        """Empty collection returns empty list."""
        with (
            patch(
                "server.tools.vector_search._qdrant_client",
                qdrant_client,
            ),
            patch(
                "server.tools.vector_search._embedder",
                mock_embedder,
            ),
        ):
            from server.tools.vector_search import vector_search

            results = await vector_search("unrelated query")

        assert results == []


# ---------------------------------------------------------------------------
# visual_search tests
# ---------------------------------------------------------------------------


class TestVisualSearch:
    """Tests for the visual_search tool."""

    @pytest.mark.integration
    async def test_returns_multivec_results(self, qdrant_client, mock_embedder):
        """Returns results from the multivec collection."""
        _seed_multivec(
            qdrant_client,
            [_make_multivec_point(1, source_key="docs/page.pdf")],
        )

        with (
            patch(
                "server.tools.visual_search._qdrant_client",
                qdrant_client,
            ),
            patch(
                "server.tools.visual_search._embedder",
                mock_embedder,
            ),
        ):
            from server.tools.visual_search import visual_search

            results = await visual_search("chart diagram")

        assert len(results) == 1
        assert results[0]["source_key"] == "docs/page.pdf"
        assert "text" not in results[0]


# ---------------------------------------------------------------------------
# graph_search tests
# ---------------------------------------------------------------------------


class TestGraphSearch:
    """Tests for the engine-based graph_search tool."""

    async def test_entity_search_returns_facts(self):
        """Graph search returns fact-based results from graph engine."""
        from server.models import GraphFact

        mock_facts = [
            GraphFact(
                fact="Python is a programming language",
                source="doc1.pdf",
                created_at="2025-01-01T00:00:00",
            ),
        ]

        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=mock_facts)

        with patch(
            "server.tools.graph_search.get_graph_engine",
            return_value=mock_engine,
        ):
            from server.tools.graph_search import graph_search

            results = await graph_search("Python")

        assert len(results) == 1
        assert results[0]["fact"] == "Python is a programming language"
        assert results[0]["source"] == "doc1.pdf"
        assert results[0]["created_at"] is not None

    async def test_entity_search_respects_limit(self):
        """Graph search respects the limit parameter."""
        from server.models import GraphFact

        mock_facts = [
            GraphFact(fact=f"Fact {i}", source=f"doc{i}.pdf", created_at="2025-01-01")
            for i in range(5)
        ]

        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=mock_facts)

        with patch(
            "server.tools.graph_search.get_graph_engine",
            return_value=mock_engine,
        ):
            from server.tools.graph_search import graph_search

            results = await graph_search("test", limit=3)

        # The engine returns 5 but engine.search is called with limit=3
        # The engine mock returns all 5; graph_search trusts engine to limit
        mock_engine.search.assert_called_once_with("test", limit=3)
        assert len(results) == 5  # mock returns all; real engine would limit

    async def test_unsupported_search_type_raises(self):
        """Unsupported search_type raises ValueError."""
        from server.tools.graph_search import graph_search

        with pytest.raises(ValueError, match="not yet implemented"):
            await graph_search("test", search_type="path")

    async def test_graph_search_handles_error(self):
        """Graph search returns error dict on failure."""
        with patch(
            "server.tools.graph_search.get_graph_engine",
            side_effect=RuntimeError("connection failed"),
        ):
            from server.tools.graph_search import graph_search

            results = await graph_search("test")

        assert len(results) == 1
        assert "error" in results[0]


# ---------------------------------------------------------------------------
# hybrid_search tests
# ---------------------------------------------------------------------------


class TestHybridSearch:
    """Tests for the hybrid_search tool."""

    async def test_returns_both_results(self):
        """Returns both vector_results and graph_results."""
        mock_vector = AsyncMock(return_value=[{"score": 0.9, "text": "result"}])
        mock_graph = AsyncMock(return_value=[{"entity": "X", "type": "CONCEPT"}])

        with (
            patch(
                "server.tools.hybrid_search.vector_search",
                mock_vector,
            ),
            patch(
                "server.tools.hybrid_search.graph_search",
                mock_graph,
            ),
        ):
            from server.tools.hybrid_search import hybrid_search

            result = await hybrid_search("test query")

        assert len(result["vector_results"]) == 1
        assert len(result["graph_results"]) == 1
        assert result["query"] == "test query"
        assert result["strategy"] == "parallel"

    async def test_partial_failure_vector(self):
        """If vector search fails, graph results still returned."""
        mock_vector = AsyncMock(side_effect=RuntimeError("Qdrant down"))
        mock_graph = AsyncMock(return_value=[{"entity": "X"}])

        with (
            patch(
                "server.tools.hybrid_search.vector_search",
                mock_vector,
            ),
            patch(
                "server.tools.hybrid_search.graph_search",
                mock_graph,
            ),
        ):
            from server.tools.hybrid_search import hybrid_search

            result = await hybrid_search("test")

        assert result["vector_results"][0]["error"]
        assert len(result["graph_results"]) == 1

    async def test_partial_failure_graph(self):
        """If graph search fails, vector results still returned."""
        mock_vector = AsyncMock(return_value=[{"score": 0.9}])
        mock_graph = AsyncMock(side_effect=RuntimeError("Neo4j down"))

        with (
            patch(
                "server.tools.hybrid_search.vector_search",
                mock_vector,
            ),
            patch(
                "server.tools.hybrid_search.graph_search",
                mock_graph,
            ),
        ):
            from server.tools.hybrid_search import hybrid_search

            result = await hybrid_search("test")

        assert len(result["vector_results"]) == 1
        assert result["graph_results"][0]["error"]


# ---------------------------------------------------------------------------
# MCP server registration test
# ---------------------------------------------------------------------------


class TestMCPServer:
    """Tests for MCP server tool registration."""

    async def test_all_tools_registered(self):
        """Core search tools are always registered; visual_search is conditional."""
        from server.mcp_server import mcp

        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert {
            "vector_search",
            "graph_search",
            "hybrid_search",
            "list_documents",
            "list_document_chunks",
        } <= tool_names
        from config.settings import settings

        if settings.multivec_enabled:
            assert "visual_search" in tool_names
        else:
            assert "visual_search" not in tool_names


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests from spec §13."""

    @pytest.mark.integration
    async def test_empty_corpus_returns_empty_list(
        self,
        qdrant_client,
        mock_embedder,
    ):
        """EC-05: Empty collection returns empty list, not an error."""
        with (
            patch(
                "server.tools.vector_search._qdrant_client",
                qdrant_client,
            ),
            patch(
                "server.tools.vector_search._embedder",
                mock_embedder,
            ),
        ):
            from server.tools.vector_search import vector_search

            results = await vector_search("anything at all")

        assert results == []

    async def test_graph_search_failure_returns_error_dict(self):
        """EC-03: Graph search failure returns error, doesn't crash."""
        with patch(
            "server.tools.graph_search.get_graph_engine",
            side_effect=RuntimeError("Neo4j down"),
        ):
            from server.tools.graph_search import graph_search

            results = await graph_search("test")

        assert len(results) == 1
        assert "error" in results[0]


class TestBearerAuth:
    """EC-08: MCP Bearer token authentication tests."""

    async def test_bearer_auth_middleware_rejects_missing_token(self):
        """Requests without Bearer token are rejected."""

        from server.mcp_server import BearerAuthMiddleware

        middleware = BearerAuthMiddleware()

        # Build a mock context for tools/call with no auth header
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_req_ctx = MagicMock()
        mock_req_ctx.request = mock_request
        mock_fastmcp_ctx = MagicMock()
        mock_fastmcp_ctx.request_context = mock_req_ctx

        context = MagicMock()
        context.method = "tools/call"
        context.fastmcp_context = mock_fastmcp_ctx

        call_next = AsyncMock()

        with patch("server.mcp_server.settings") as mock_settings:
            mock_settings.mcp_api_key = "secret-key"
            with pytest.raises(PermissionError, match="Authentication"):
                await middleware(context, call_next)

    async def test_bearer_auth_middleware_rejects_wrong_token(self):
        """Requests with wrong Bearer token are rejected."""

        from server.mcp_server import BearerAuthMiddleware

        middleware = BearerAuthMiddleware()

        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer wrong-key"}
        mock_req_ctx = MagicMock()
        mock_req_ctx.request = mock_request
        mock_fastmcp_ctx = MagicMock()
        mock_fastmcp_ctx.request_context = mock_req_ctx

        context = MagicMock()
        context.method = "tools/call"
        context.fastmcp_context = mock_fastmcp_ctx

        call_next = AsyncMock()

        with patch("server.mcp_server.settings") as mock_settings:
            mock_settings.mcp_api_key = "secret-key"
            with pytest.raises(PermissionError, match="Invalid token"):
                await middleware(context, call_next)

    async def test_bearer_auth_middleware_passes_valid_token(self):
        """Requests with correct Bearer token pass through."""

        from server.mcp_server import BearerAuthMiddleware

        middleware = BearerAuthMiddleware()

        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer secret-key"}
        mock_req_ctx = MagicMock()
        mock_req_ctx.request = mock_request
        mock_fastmcp_ctx = MagicMock()
        mock_fastmcp_ctx.request_context = mock_req_ctx

        context = MagicMock()
        context.method = "tools/call"
        context.fastmcp_context = mock_fastmcp_ctx

        call_next = AsyncMock(return_value="ok")

        with patch("server.mcp_server.settings") as mock_settings:
            mock_settings.mcp_api_key = "secret-key"
            result = await middleware(context, call_next)

        assert result == "ok"
        call_next.assert_awaited_once()

    async def test_bearer_auth_skipped_when_no_key_configured(self):
        """When mcp_api_key is empty, all requests pass through."""

        from server.mcp_server import BearerAuthMiddleware

        middleware = BearerAuthMiddleware()

        context = MagicMock()
        context.method = "tools/call"

        call_next = AsyncMock(return_value="ok")

        with patch("server.mcp_server.settings") as mock_settings:
            mock_settings.mcp_api_key = ""
            result = await middleware(context, call_next)

        assert result == "ok"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for tool input validation (empty query, limit clamping)."""

    async def test_vector_search_empty_query_returns_empty(self):
        """Empty query string returns empty list."""
        from server.tools.vector_search import vector_search

        results = await vector_search("")
        assert results == []

    async def test_graph_search_empty_query_returns_empty(self):
        """Empty query string returns empty list."""
        from server.tools.graph_search import graph_search

        results = await graph_search("")
        assert results == []

    async def test_hybrid_search_empty_query_returns_empty(self):
        """Empty query returns empty hybrid response."""
        from server.tools.hybrid_search import hybrid_search

        result = await hybrid_search("")
        assert result["vector_results"] == []
        assert result["graph_results"] == []


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------


class TestReranker:
    """Unit tests for the Jina Reranker with mocked HTTP."""

    async def test_rerank_returns_reordered_results(self):
        """Reranker re-scores and reorders results."""
        results = [
            {"score": 0.5, "text": "first doc"},
            {"score": 0.3, "text": "second doc"},
        ]
        mock_api_results = [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.7},
        ]

        with patch(
            "retrieval.rerank._rerank_request",
            new_callable=AsyncMock,
            return_value=mock_api_results,
        ):
            from retrieval.rerank import rerank_dicts

            reranked = await rerank_dicts("query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0]["score"] == 0.9
        assert reranked[0]["text"] == "second doc"
        assert reranked[0]["original_score"] == 0.3

    async def test_rerank_empty_results(self):
        """Empty input returns empty output."""
        from retrieval.rerank import rerank_dicts

        result = await rerank_dicts("query", [], top_k=5)
        assert result == []

    async def test_rerank_fallback_on_failure(self):
        """On API failure, returns original results truncated."""
        results = [
            {"score": 0.5, "text": "doc"},
            {"score": 0.3, "text": "doc2"},
        ]

        with patch(
            "retrieval.rerank._rerank_request",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            from retrieval.rerank import rerank_dicts

            reranked = await rerank_dicts("query", results, top_k=1)

        assert len(reranked) == 1
        assert reranked[0]["text"] == "doc"


# ---------------------------------------------------------------------------
# VLM Generator tests
# ---------------------------------------------------------------------------


class TestVLMGenerator:
    """Unit tests for VLM visual answer generation with mocked deps."""

    async def test_generate_returns_answer(self):
        """VLM generates answer from S3 images."""
        results = [{"source_key": "docs/chart.pdf"}]

        with (
            patch(
                "server.tools.vlm_generator._fetch_s3_image",
                return_value=(b"fake-png", "image/png"),
            ),
            patch(
                "server.tools.vlm_generator._ask_vlm",
                new_callable=AsyncMock,
                return_value="The chart shows growth.",
            ),
        ):
            from server.tools.vlm_generator import (
                generate_visual_answer,
            )

            answer = await generate_visual_answer(
                "What does the chart show?",
                results,
            )

        assert answer == "The chart shows growth."

    async def test_generate_returns_none_on_no_source_keys(self):
        """Returns None when no results have source_key."""
        from server.tools.vlm_generator import generate_visual_answer

        answer = await generate_visual_answer("query", [{"score": 0.5}])
        assert answer is None

    async def test_generate_returns_none_on_s3_error(self):
        """Returns None when S3 fetch fails."""
        results = [{"source_key": "docs/missing.pdf"}]

        with patch(
            "server.tools.vlm_generator._fetch_s3_image",
            side_effect=OSError("S3 unreachable"),
        ):
            from server.tools.vlm_generator import (
                generate_visual_answer,
            )

            answer = await generate_visual_answer("query", results)

        assert answer is None


# ---------------------------------------------------------------------------
# Live ingest model tests
# ---------------------------------------------------------------------------


class TestLiveIngestModels:
    def test_live_chunk_model(self) -> None:
        """LiveChunk validates required fields."""
        from server.models import LiveChunk

        chunk = LiveChunk(
            session_id="session-1",
            text="text content one",
            timestamp="2026-03-06T14:30:00Z",
        )
        assert chunk.session_id == "session-1"
        assert chunk.text == "text content one"

    def test_session_start_request(self) -> None:
        """SessionStartRequest validates fields."""
        from server.models import SessionStartRequest

        req = SessionStartRequest(
            session_id="session-1",
            metadata={"title": "Q1 Review"},
        )
        assert req.session_id == "session-1"

    def test_session_end_request(self) -> None:
        """SessionEndRequest defaults archive to False."""
        from server.models import SessionEndRequest

        req = SessionEndRequest(session_id="session-1")
        assert req.archive is False

    def test_ingest_response(self) -> None:
        """IngestResponse model."""
        from server.models import IngestResponse

        resp = IngestResponse(
            status="accepted",
            vector_indexed=True,
            graph_status="processing",
        )
        assert resp.status == "accepted"

    def test_hybrid_search_response_with_session(self) -> None:
        """HybridSearchResponse supports session_id and live_results."""
        from server.models import HybridSearchResponse

        resp = HybridSearchResponse(
            query="test",
            session_id="session-1",
            live_results=[],
        )
        assert resp.session_id == "session-1"
        assert resp.live_results == []


# ---------------------------------------------------------------------------
# Session-aware vector search tests
# ---------------------------------------------------------------------------


class TestSessionAwareVectorSearch:
    @pytest.mark.asyncio
    async def test_vector_search_with_session_id(self) -> None:
        """When session_id is set, runs dual Qdrant queries."""
        mock_embedder = MagicMock()
        mock_embedder.embed_text_query = AsyncMock(return_value=[0.1] * 512)

        # Mock Qdrant to return different results per call
        live_point = MagicMock()
        live_point.score = 0.9
        live_point.payload = {
            "text_content": "text content one",
            "source_file": "session:session-1",
            "page_number": 0,
            "content_type": "live",
            "metadata": {},
            "timestamp": "2026-03-06T14:32:00Z",
        }
        kb_point = MagicMock()
        kb_point.score = 0.85
        kb_point.payload = {
            "text": "Payment policy doc",
            "source_file": "policies/payment.pdf",
            "page_number": 1,
            "content_type": "text_chunk",
            "metadata": {},
        }

        mock_response_live = MagicMock()
        mock_response_live.points = [live_point]
        mock_response_kb = MagicMock()
        mock_response_kb.points = [kb_point]

        mock_qdrant = MagicMock()
        mock_qdrant.query_points = MagicMock(
            side_effect=[mock_response_live, mock_response_kb]
        )

        with (
            patch("server.tools.vector_search._qdrant_client", mock_qdrant),
            patch("server.tools.vector_search._embedder", mock_embedder),
        ):
            from server.tools.vector_search import vector_search

            results = await vector_search("contract", session_id="session-1")

        assert len(results) == 2
        assert mock_qdrant.query_points.call_count == 2

    @pytest.mark.asyncio
    async def test_vector_search_without_session_unchanged(self) -> None:
        """Without session_id, behavior is unchanged (single query)."""
        mock_embedder = MagicMock()
        mock_embedder.embed_text_query = AsyncMock(return_value=[0.1] * 512)

        mock_response = MagicMock()
        mock_response.points = []

        mock_qdrant = MagicMock()
        mock_qdrant.query_points = MagicMock(return_value=mock_response)

        with (
            patch("server.tools.vector_search._qdrant_client", mock_qdrant),
            patch("server.tools.vector_search._embedder", mock_embedder),
        ):
            from server.tools.vector_search import vector_search

            await vector_search("test")

        assert mock_qdrant.query_points.call_count == 1


# ---------------------------------------------------------------------------
# Session-aware graph search tests
# ---------------------------------------------------------------------------


class TestSessionAwareGraphSearch:
    @pytest.mark.asyncio
    async def test_graph_search_with_session_id(self) -> None:
        """When session_id is set, queries both Graphiti and GLiNER."""
        mock_edge = MagicMock()
        mock_edge.fact = "Contract valued at 1.2M"
        mock_edge.configure_mock(name="graphiti")
        mock_edge.created_at = "2026-03-06T14:32:00Z"
        mock_edge.expired_at = None

        mock_graphiti = AsyncMock()
        mock_graphiti.search = AsyncMock(return_value=[mock_edge])

        from server.models import GraphFact

        gliner_facts = [
            GraphFact(
                fact="Acme Corp (organization)",
                entities=["Acme Corp"],
                confidence=0.9,
            ),
        ]
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=gliner_facts)

        with (
            patch(
                "ingestion.graphiti_client.get_graphiti",
                return_value=mock_graphiti,
            ),
            patch(
                "server.tools.graph_search.get_graph_engine",
                return_value=mock_engine,
            ),
        ):
            from server.tools.graph_search import graph_search

            results = await graph_search("contract", session_id="session-1")

        # Should have results from both engines
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_graph_search_without_session_unchanged(self) -> None:
        """Without session_id, behavior delegates to engine only."""
        from server.models import GraphFact

        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=[GraphFact(fact="Test fact")])

        with patch(
            "server.tools.graph_search.get_graph_engine",
            return_value=mock_engine,
        ):
            from server.tools.graph_search import graph_search

            results = await graph_search("test")

        assert len(results) == 1
        mock_engine.search.assert_called_once()


# ---------------------------------------------------------------------------
# Session-aware hybrid search tests
# ---------------------------------------------------------------------------


class TestSessionAwareHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_with_session_id(self) -> None:
        """Hybrid search passes session_id to both sub-searches."""
        mock_vector = AsyncMock(
            return_value=[
                {
                    "score": 0.9,
                    "text": "live chunk",
                    "metadata": {"source_type": "live"},
                },
                {"score": 0.8, "text": "kb doc", "metadata": {}},
            ]
        )
        mock_graph = AsyncMock(return_value=[{"fact": "X related Y"}])

        with (
            patch("server.tools.hybrid_search.vector_search", mock_vector),
            patch("server.tools.hybrid_search.graph_search", mock_graph),
        ):
            from server.tools.hybrid_search import hybrid_search

            result = await hybrid_search("test", session_id="session-1")

        # vector_search should receive session_id
        mock_vector.assert_called_once()
        assert mock_vector.call_args.kwargs.get("session_id") == "session-1"
        # graph_search should receive session_id
        mock_graph.assert_called_once()
        assert mock_graph.call_args.kwargs.get("session_id") == "session-1"
        assert result["session_id"] == "session-1"

    @pytest.mark.asyncio
    async def test_hybrid_search_separates_live_results(self) -> None:
        """Hybrid search separates live chunks from KB results."""
        mock_vector = AsyncMock(
            return_value=[
                {
                    "score": 0.9,
                    "text": "live chunk",
                    "metadata": {"source_type": "live"},
                },
                {"score": 0.8, "text": "kb doc", "metadata": {}},
            ]
        )
        mock_graph = AsyncMock(return_value=[])

        with (
            patch("server.tools.hybrid_search.vector_search", mock_vector),
            patch("server.tools.hybrid_search.graph_search", mock_graph),
        ):
            from server.tools.hybrid_search import hybrid_search

            result = await hybrid_search("test", session_id="session-1")

        assert len(result["live_results"]) == 1
        assert len(result["vector_results"]) == 1
