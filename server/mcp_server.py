"""FastMCP server exposing RAG search tools to LLM agents."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext

from config.logging import get_logger, setup_logging
from config.observability import setup_observability
from config.settings import settings
from server.tools.graph_search import graph_search
from server.tools.hybrid_search import hybrid_search
from server.tools.list_documents import list_documents
from server.tools.vector_search import vector_search
from server.tools.visual_search import visual_search

setup_logging()
setup_observability()
logger = get_logger(__name__)


class BearerAuthMiddleware(Middleware):
    """Reject unauthenticated MCP tool calls when an API key is configured."""

    async def __call__(
        self,
        context: MiddlewareContext,
        call_next,  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        if settings.mcp_api_key and context.method == "tools/call":
            ctx = context.fastmcp_context
            if ctx.request_context and ctx.request_context.request:
                auth = ctx.request_context.request.headers.get(
                    "Authorization",
                )
                if not auth or not auth.startswith("Bearer "):
                    raise PermissionError("Authentication required")
                if auth.removeprefix("Bearer ") != settings.mcp_api_key:
                    raise PermissionError("Invalid token")
        return await call_next(context)


mcp = FastMCP(
    "rag-knowledge-base",
    middleware=[BearerAuthMiddleware()],
)
mcp.tool()(vector_search)
mcp.tool()(graph_search)
mcp.tool()(hybrid_search)
mcp.tool()(list_documents)

_REGISTERED_TOOLS = [
    "vector_search",
    "graph_search",
    "hybrid_search",
    "list_documents",
]

if settings.multivec_enabled:
    mcp.tool()(visual_search)
    _REGISTERED_TOOLS.append("visual_search")

if __name__ == "__main__":
    transport_kwargs: dict[str, Any] = {}
    if settings.mcp_transport != "stdio":
        transport_kwargs = {
            "host": settings.mcp_host,
            "port": settings.mcp_port,
            "path": settings.mcp_path,
        }
        logger.info(
            "MCP server starting on %s http://%s:%d%s with tools: %s",
            settings.mcp_transport,
            settings.mcp_host,
            settings.mcp_port,
            settings.mcp_path,
            ", ".join(_REGISTERED_TOOLS),
        )
    else:
        logger.info(
            "MCP server starting on stdio with tools: %s",
            ", ".join(_REGISTERED_TOOLS),
        )
    logger.info(
        "Live ingest available on port %d (run separately)",
        settings.live_ingest_port,
    )
    mcp.run(transport=settings.mcp_transport, **transport_kwargs)
