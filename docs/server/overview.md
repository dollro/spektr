# MCP Server Overview

Spektr exposes its RAG search capabilities to LLM agents through a [FastMCP](https://github.com/jlowin/fastmcp) server. The server registers five tools (search + introspection) and optionally enforces Bearer token authentication.

## Server Setup

The server is defined in `server/mcp_server.py`. A single `FastMCP` instance is created with the `BearerAuthMiddleware` attached, and each tool function is registered via `mcp.tool()`:

```python
from fastmcp import FastMCP

mcp = FastMCP(
    "rag-knowledge-base",
    middleware=[BearerAuthMiddleware()],
)
mcp.tool()(vector_search)
mcp.tool()(graph_search)
mcp.tool()(hybrid_search)
mcp.tool()(list_documents)
# visual_search registered only when MULTIVEC_ENABLED=true
```

FastMCP inspects each function's signature and docstring to generate the MCP tool schema automatically. No manual schema definitions are needed.

## Registered Tools

| Tool | Always on? | Description |
|-|-|-|
| `vector_search` | yes | Dense semantic search via Qdrant; `session_id` aware |
| `graph_search` | yes | Engine-agnostic graph search (GLiNER2 or Graphiti) |
| `hybrid_search` | yes | Parallel vector + graph fusion with session separation |
| `list_documents` | yes | Enumerate ingested docs with chunk + page counts |
| `visual_search` | needs `MULTIVEC_ENABLED=true` | ColBERT multi-vector search for visual content |

See [Search Tools](search-tools.md) for full parameter and return schema documentation. See [Client Setup](client-setup.md) for wiring the server into Claude Code, Cursor, or any other MCP-aware agent.

## Transport Options

The server supports two transports, controlled by the `MCP_TRANSPORT` environment variable:

| Transport | Default | Use Case |
|-|-|-|
| `sse` | Yes | Network clients (agents, HTTP) over Server-Sent Events |
| `stdio` | No | Subprocess communication (Claude Code, local toolchains) |

!!! warning "FastMCP 3.x port quirk"
    FastMCP 3.0.x ignores the `port` kwarg passed to `mcp.run(transport="sse", ...)` and always binds to **8000**. Until the upstream fix lands, point MCP clients at port 8000 in SSE mode regardless of `MCP_PORT`. stdio transport is unaffected.

## Starting the Server

```bash
task serve
# equivalent to:
uv run python -m server.mcp_server
```

On startup the server logs the transport, port, and list of registered tools:

```
MCP server starting on sse:8000 with tools: vector_search, graph_search,
hybrid_search, list_documents
```

## Middleware Stack

The server includes one middleware by default:

- **BearerAuthMiddleware** -- Intercepts `tools/call` requests and validates the `Authorization: Bearer <key>` header against `MCP_API_KEY`. See [Authentication](authentication.md) for details.

Additional middleware can be appended to the `middleware` list in the `FastMCP` constructor.

## Client Integration

LLM agents connect to the server as an MCP tool provider. Two well-worn paths:

- **Pydantic AI** — the built-in `agent/agent.py` and `scripts/ask.py` use `MCPServerSSE` to talk to Spektr over HTTP/SSE. See [Agent Overview](../agent/overview.md).
- **Claude Code / Cursor / other MCP-aware IDEs** — drop a `.mcp.json` into the project root; Claude Code spawns or connects to the Spektr server and exposes its tools as `mcp__spektr__*`. See [Client Setup](client-setup.md) for both stdio and SSE recipes.
