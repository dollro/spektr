"""Test script for querying the MCP server tools directly (no server needed).

Calls vector_search, graph_search, and hybrid_search against
the local Qdrant + Neo4j backends to verify end-to-end retrieval.

Usage:
    uv run python scripts/test_mcp_query.py
    uv run python scripts/test_mcp_query.py "your custom query"
"""

from __future__ import annotations

import asyncio
import json
import sys

from server.tools.graph_search import graph_search
from server.tools.hybrid_search import hybrid_search
from server.tools.vector_search import vector_search


def _pretty(label: str, data: list | dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print("=" * 60)
    print(json.dumps(data, indent=2, default=str)[:3000])
    if len(json.dumps(data, default=str)) > 3000:
        print("  ... (truncated)")


async def main(query: str) -> None:
    print(f"Query: {query!r}\n")

    # 1. Vector search
    print("Running vector_search ...")
    vec_results = await vector_search(query, limit=5)
    _pretty("Vector Search Results", vec_results)

    # 2. Graph search
    print("\nRunning graph_search ...")
    graph_results = await graph_search(query, limit=5)
    _pretty("Graph Search Results", graph_results)

    # 3. Hybrid search
    print("\nRunning hybrid_search ...")
    hybrid_results = await hybrid_search(query, limit=5)
    _pretty("Hybrid Search Results", hybrid_results)

    # Summary
    vec_count = len(vec_results) if isinstance(vec_results, list) else 0
    graph_count = len(graph_results) if isinstance(graph_results, list) else 0
    hybrid_vec = len(hybrid_results.get("vector_results", []))
    hybrid_graph = len(hybrid_results.get("graph_results", []))

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  vector_search:  {vec_count} results")
    print(f"  graph_search:   {graph_count} results")
    print(f"  hybrid_search:  {hybrid_vec} vector + {hybrid_graph} graph results")

    has_errors = any(
        isinstance(r, dict) and "error" in r
        for r in [*vec_results, *graph_results]
    )
    if has_errors:
        print("\n  WARNING: Some searches returned errors (see above)")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "machine learning"
    asyncio.run(main(query))
