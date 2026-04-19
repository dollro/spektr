"""Quick retrieval smoke test — hits vector_search directly, no MCP/LLM."""

from __future__ import annotations

import asyncio
import sys

from server.tools.vector_search import vector_search

DEFAULT_QUERIES = [
    "what is this paper about?",
    "methodology",
    "results",
]


async def main(queries: list[str]) -> None:
    for q in queries:
        print(f"\n=== {q} ===")
        results = await vector_search(q, limit=3)
        if not results:
            print("(no results)")
            continue
        for r in results:
            score = r.get("score", 0.0)
            src = r.get("source_file", "?")
            page = r.get("page_number", "?")
            text = (r.get("text") or "").replace("\n", " ")[:120]
            print(f"[{score:.3f}] {src} p{page}: {text}")


if __name__ == "__main__":
    qs = sys.argv[1:] or DEFAULT_QUERIES
    asyncio.run(main(qs))
