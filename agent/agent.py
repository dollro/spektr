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
(vector database + knowledge graph). Use these search tools \
based on the type of query:

- vector_search: for text/semantic similarity questions — \
find documents by meaning, keyword relevance, or topic matching.
- visual_search: for visual/layout/image/chart questions — \
find documents by visual content using ColBERT multi-vector search.
- graph_search: for entity relationship/connection questions — \
find entities and how they connect in the knowledge graph.
- hybrid_search: for complex multi-faceted questions — \
runs vector + graph search in parallel and fuses results.

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
    url = f"http://localhost:{settings.mcp_port}{settings.mcp_path}"
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
