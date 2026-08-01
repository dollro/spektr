"""Pydantic AI agent with MCP tool bindings for RAG search."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from config.settings import settings

MCPServer = MCPServerStreamableHTTP | MCPServerSSE

SYSTEM_PROMPT = """\
You are a RAG assistant backed by a dual knowledge store \
(vector database + knowledge graph).

Search tools:
- multi_search: fused keyword + semantic search, reranked. Fast and cheap.
  Use this by default.
- hybrid_search: same as multi_search, plus it splits multi-part questions
  into sub-queries and retries when results look weak. Use for complex or
  compound questions. Slower and costs an extra model call.
- vector_search: semantic-only search. Use when you specifically want
  conceptual similarity without keyword matching.
- visual_search: finds pages by visual layout — charts, diagrams, tables.
- graph_search: entity and relationship facts from the knowledge graph.

Both multi_search and hybrid_search return:
- results: ranked chunks, best first. `channels` shows whether a hit came
  from keyword matching, semantic similarity, or both.
- graph_facts: supporting entity facts, not ranked against the chunks.
- live_results: chunks from the active session, when one is set.
- degraded: present only when part of the pipeline failed. Results are still
  usable but less complete than normal.

Always cite source documents in your answers. If no relevant \
results are found, say so rather than guessing."""


async def create_rag_agent() -> tuple[Agent, MCPServer]:
    """Create a RAG agent connected to the MCP search server.

    Picks the transport based on ``MCP_TRANSPORT`` — ``MCPServerStreamableHTTP``
    for ``http`` / ``streamable-http`` (the default), ``MCPServerSSE`` for
    legacy SSE deployments. Returns a tuple of (agent, mcp_server); the caller
    must manage the server lifecycle via ``async with server:``.
    """
    headers = (
        {"Authorization": f"Bearer {settings.mcp_api_key}"} if settings.mcp_api_key else None
    )
    url = settings.mcp_server_url or f"http://localhost:{settings.mcp_port}{settings.mcp_path}"
    server: MCPServer
    if settings.mcp_transport == "sse":
        server = MCPServerSSE(url, headers=headers)
    else:
        server = MCPServerStreamableHTTP(url, headers=headers)

    if settings.llm_base_url:
        model = OpenAIModel(
            settings.llm_model,
            provider=OpenAIProvider(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
            ),
        )
    else:
        model = f"{settings.llm_api_type}:{settings.llm_model}"

    agent = Agent(
        model,
        system_prompt=SYSTEM_PROMPT,
        toolsets=[server],
    )
    return agent, server
