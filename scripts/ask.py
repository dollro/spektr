"""Ask a question through the full RAG agent → MCP → LLM stack.

Requires the MCP server to be running (`task serve`).
Uses the LLM configured in .env (LLM_API_TYPE / LLM_MODEL / LLM_API_KEY).
"""

from __future__ import annotations

import asyncio
import sys

from agent.agent import create_rag_agent


async def main(question: str) -> None:
    agent, server = await create_rag_agent()
    async with server:
        result = await agent.run(question)
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.output)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: task ask -- 'your question here'", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(" ".join(sys.argv[1:])))
