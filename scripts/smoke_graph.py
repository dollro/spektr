"""Quick graph-search smoke test — hits graph_search directly, no MCP/LLM."""

from __future__ import annotations

import asyncio
import sys

from server.tools.graph_search import graph_search

DEFAULT_QUERIES = [
    "robot",
    "exploration",
    "LLM",
]


async def main(queries: list[str]) -> None:
    for q in queries:
        print(f"\n=== {q} ===")
        results = await graph_search(q, limit=5)
        if not results:
            print("(no results)")
            continue
        for r in results:
            if "error" in r:
                print(f"ERROR: {r['error']}")
                continue
            fact = (r.get("fact") or "").replace("\n", " ")[:140]
            src = r.get("source") or "?"
            rel = r.get("relation_type") or ""
            conf = r.get("confidence")
            tag = f" [{rel}]" if rel else ""
            conf_s = f" ({conf:.2f})" if isinstance(conf, int | float) else ""
            print(f"- {src}{tag}{conf_s}: {fact}")


if __name__ == "__main__":
    qs = sys.argv[1:] or DEFAULT_QUERIES
    asyncio.run(main(qs))
