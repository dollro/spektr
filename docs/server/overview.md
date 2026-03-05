# MCP Server Overview

Spektr exposes its RAG search capabilities to LLM agents through a [FastMCP](https://github.com/jlowin/fastmcp) server. The server registers four search tools and optionally enforces Bearer token authentication.

## Server Setup

The server is defined in `server/mcp_server.py`. A single `FastMCP` instance is created with the `BearerAuthMiddleware` attached, and each tool function is registered via `mcp.tool()`:

```python
from fastmcp import FastMCP

mcp = FastMCP(
    "rag-knowledge-base",
    middleware=[BearerAuthMiddleware()],
)
mcp.tool()(vector_search)
mcp.tool()(visual_search)
mcp.tool()(graph_search)
mcp.tool()(hybrid_search)
```

FastMCP inspects each function's signature and docstring to generate the MCP tool schema automatically. No manual schema definitions are needed.

## Registered Tools

| Tool | Description |
|-|-|
| `vector_search` | Dense semantic search via Qdrant |
| `visual_search` | ColBERT multi-vector search for visual content |
| `graph_search` | Knowledge graph search via Graphiti |
| `hybrid_search` | Parallel vector + graph fusion |

See [Search Tools](search-tools.md) for full parameter and return schema documentation.

## Transport Options

The server supports two transports, controlled by the `MCP_TRANSPORT` environment variable:

| Transport | Default | Use Case |
|-|-|-|
| `sse` | Yes | Network clients (agents, HTTP) over Server-Sent Events |
| `stdio` | No | Subprocess communication (Claude Code, local toolchains) |

The listening port for SSE transport is set via `MCP_PORT` (default `8000`).

## Starting the Server

```bash
uv run python -m server.mcp_server
```

On startup the server logs the transport, port, and list of registered tools:

```
MCP server starting on sse:8000 with tools: vector_search, visual_search, graph_search, hybrid_search
```

## Middleware Stack

The server includes one middleware by default:

- **BearerAuthMiddleware** -- Intercepts `tools/call` requests and validates the `Authorization: Bearer <key>` header against `MCP_API_KEY`. See [Authentication](authentication.md) for details.

Additional middleware can be appended to the `middleware` list in the `FastMCP` constructor.

## Client Integration

LLM agents connect to the server as an MCP tool provider. For a working example using Pydantic AI, see the [Agent Overview](../agent/overview.md).
