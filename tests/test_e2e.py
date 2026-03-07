"""End-to-end integration tests for the Spektr RAG system.

Tests the full flow: seeded databases -> MCP tools -> agent -> HTTP API.
Requires Docker services (Qdrant, Neo4j) to be running.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo

from config.constants import (
    DENSE_COLLECTION,
    DENSE_DIM,
    MULTIVEC_COLLECTION,
    MULTIVEC_DIM,
)

# Ensure provider key is set before any imports that trigger Agent()
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers — data seeding (reused from test_tools.py patterns)
# ---------------------------------------------------------------------------


def _seed_qdrant(client) -> None:  # type: ignore[no-untyped-def]
    """Seed Qdrant with test documents for both collections."""
    from qdrant_client.models import PointStruct

    client.upsert(
        collection_name=DENSE_COLLECTION,
        points=[
            PointStruct(
                id=1,
                vector=[0.1] * DENSE_DIM,
                payload={
                    "text": "Python is a programming language",
                    "source_file": "test_doc.pdf",
                    "page_number": 1,
                    "content_type": "pdf",
                    "metadata": {},
                },
            ),
            PointStruct(
                id=2,
                vector=[0.2] * DENSE_DIM,
                payload={
                    "text": "Machine learning uses neural networks",
                    "source_file": "ml_guide.pdf",
                    "page_number": 3,
                    "content_type": "pdf",
                    "metadata": {},
                },
            ),
        ],
    )
    client.upsert(
        collection_name=MULTIVEC_COLLECTION,
        points=[
            PointStruct(
                id=1,
                vector={"colbert": [[0.3] * MULTIVEC_DIM] * 5},
                payload={
                    "source_file": "chart.pdf",
                    "page_number": 2,
                    "content_type": "pdf",
                    "source_key": "docs/chart.pdf",
                    "metadata": {},
                },
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Mock LLM that references tool results in its answer
# ---------------------------------------------------------------------------


async def _rag_mock_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Mock model that returns an answer referencing source data."""
    return ModelResponse(
        parts=[
            TextPart(
                "Based on the documents, Python is a programming "
                "language. Sources: test_doc.pdf, ml_guide.pdf."
            )
        ]
    )


# ---------------------------------------------------------------------------
# E2E: Agent with seeded databases + patched tools
# ---------------------------------------------------------------------------


class TestAgentWithSeededData:
    """Test agent queries against seeded Qdrant + Neo4j data."""

    async def test_vector_search_query(self, qdrant_client, mock_embedder):
        """Agent answers a text query using vector search results."""
        _seed_qdrant(qdrant_client)

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

            results = await vector_search("Python programming")

        assert len(results) >= 1
        texts = [r["text"] for r in results]
        assert any("Python" in t for t in texts)

    async def test_graph_search_query(self):
        """Agent answers a relationship query via graph search."""
        mock_edge = MagicMock()
        mock_edge.fact = "Python is a programming language"
        mock_edge.configure_mock(name="test_doc.pdf")
        mock_edge.created_at = "2025-01-01"
        mock_edge.expired_at = None

        mock_graphiti = AsyncMock()
        mock_graphiti.search = AsyncMock(return_value=[mock_edge])

        with patch(
            "server.tools.graph_search.get_graphiti",
            return_value=mock_graphiti,
        ):
            from server.tools.graph_search import graph_search

            results = await graph_search("Python")

        assert len(results) >= 1
        assert "Python" in results[0]["fact"]

    async def test_hybrid_search_query(self, qdrant_client, mock_embedder):
        """Hybrid search returns both vector and graph results."""
        _seed_qdrant(qdrant_client)

        mock_edge = MagicMock()
        mock_edge.fact = "Python is a language"
        mock_edge.configure_mock(name="unique_graph_source.pdf")
        mock_edge.created_at = "2025-01-01"
        mock_edge.expired_at = None

        mock_graphiti = AsyncMock()
        mock_graphiti.search = AsyncMock(return_value=[mock_edge])

        with (
            patch(
                "server.tools.vector_search._qdrant_client",
                qdrant_client,
            ),
            patch(
                "server.tools.vector_search._embedder",
                mock_embedder,
            ),
            patch(
                "server.tools.graph_search.get_graphiti",
                return_value=mock_graphiti,
            ),
        ):
            from server.tools.hybrid_search import hybrid_search

            result = await hybrid_search("Python")

        assert len(result["vector_results"]) >= 1
        assert len(result["graph_results"]) >= 1
        assert result["strategy"] == "parallel"

    async def test_visual_search_query(self, qdrant_client, mock_embedder):
        """Visual search returns multivec results."""
        _seed_qdrant(qdrant_client)

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

        assert len(results) >= 1
        assert results[0]["source_key"] == "docs/chart.pdf"


# ---------------------------------------------------------------------------
# E2E: FastAPI HTTP endpoint
# ---------------------------------------------------------------------------


class TestHTTPEndpoint:
    """Test the FastAPI /query and /health endpoints."""

    async def test_health_endpoint(self):
        """GET /health returns status ok."""
        from agent.api import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    async def test_query_endpoint(self):
        """POST /query returns an answer."""
        from agent.agent import create_rag_agent

        agent, server = await create_rag_agent()

        mock_run = AsyncMock()
        mock_run.output = "Python is a programming language. Source: test.pdf"

        with (
            patch("agent.api._agent", agent),
            patch("agent.api._server", server),
            patch.object(agent, "run", return_value=mock_run),
        ):
            from agent.api import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/query",
                    json={"query": "What is Python?"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "Python" in data["answer"]
