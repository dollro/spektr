"""List documents currently in the knowledge base.

Wraps the `list_documents` MCP tool into a CLI so you can inspect
the corpus without spinning up the MCP server or an LLM.

Usage:
    python -m scripts.list_kb                  # human-readable table
    python -m scripts.list_kb --limit 500
    python -m scripts.list_kb --json           # machine-readable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from server.tools.list_documents import list_documents


def _print_table(docs: list[dict]) -> None:  # type: ignore[type-arg]
    if not docs:
        print("(no documents in knowledge base)")
        return
    if "error" in docs[0]:
        print(f"ERROR: {docs[0]['error']}", file=sys.stderr)
        sys.exit(1)

    name_w = max(len(d["source_file"]) for d in docs)
    name_w = min(max(name_w, 12), 80)

    header = f"{'source_file':<{name_w}}  {'chunks':>6}  {'pages':>5}  content_types"
    print(header)
    print("-" * len(header))
    for d in docs:
        src = d["source_file"]
        if len(src) > name_w:
            src = src[: name_w - 1] + "…"
        ct = ",".join(d["content_types"]) or "-"
        print(f"{src:<{name_w}}  {d['chunk_count']:>6}  {d['page_count']:>5}  {ct}")
    print()
    print(f"{len(docs)} document(s)")


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="Max documents to list (1–1000)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of a table")
    args = parser.parse_args()

    docs = await list_documents(limit=args.limit)

    if args.json:
        print(json.dumps(docs, indent=2, sort_keys=True))
    else:
        _print_table(docs)


if __name__ == "__main__":
    asyncio.run(_main())
