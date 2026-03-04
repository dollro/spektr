"""FastMCP server exposing RAG search tools to LLM agents."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext

from config.logging import get_logger, setup_logging
from config.settings import settings
from server.tools.graph_search import graph_search
from server.tools.hybrid_search import hybrid_search
from server.tools.vector_search import vector_search
from server.tools.visual_search import visual_search

setup_logging()
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
mcp.tool()(visual_search)
mcp.tool()(graph_search)
mcp.tool()(hybrid_search)

_REGISTERED_TOOLS = [
    "vector_search",
    "visual_search",
    "graph_search",
    "hybrid_search",
]

if __name__ == "__main__":
    logger.info(
        "MCP server starting on %s:%d with tools: %s",
        settings.mcp_transport,
        settings.mcp_port,
        ", ".join(_REGISTERED_TOOLS),
    )
    mcp.run(
        transport=settings.mcp_transport,
        port=settings.mcp_port,
    )
