# Connecting MCP Clients

Any MCP-aware coding agent can use Spektr's search tools as first-class functions. This page covers the transport styles and the caveats each brings.

Tested clients: **Claude Code**, **Cursor**, **any Pydantic AI script**. The same `.mcp.json` format works for Claude Code and Cursor; Pydantic AI uses `MCPServerStreamableHTTP` (or `MCPServerSSE` for legacy) in code (see `agent/agent.py`).

!!! tip "Which transport?"
    Prefer **`http`** (streamable-http) — it's the modern MCP default, handles disconnects cleanly, and is what all current clients speak. Use `stdio` for local dev when you want the client to own the server lifecycle. Only use `sse` if a client explicitly requires it.

## Option A: stdio — client spawns the server

**Best for local dev.** No long-running server, no port, no auth token to juggle. The MCP client launches `python -m server.mcp_server` as a subprocess and talks to it over stdin/stdout.

`.mcp.json` (project root):

```json
{
  "mcpServers": {
    "spektr": {
      "command": "uv",
      "args": [
        "--project",
        "/home/YOU/path/to/spektr",
        "run",
        "python",
        "-m",
        "server.mcp_server"
      ],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "MCP_API_KEY": ""
      }
    }
  }
}
```

Notes:

- `--project` is required so `uv` resolves the Spektr venv regardless of the caller's cwd.
- Docker services (`task up`) still need to be running — the MCP server connects to Qdrant / Neo4j over localhost.
- `MCP_API_KEY=""` disables auth (safe for stdio; the client is local).

## Option B: HTTP (streamable-http) — client connects to a running server

**Best for shared or remote setups and the production Docker stack.** Start the server once (`task serve` in dev, or `task prod:up` in production), and any number of agents can connect in parallel.

```bash
task serve
# serves on http://localhost:8080/mcp  (MCP_TRANSPORT=http, MCP_PORT=8080, MCP_PATH=/mcp)
```

`.mcp.json`:

```json
{
  "mcpServers": {
    "spektr": {
      "type": "http",
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_MCP_API_KEY"
      }
    }
  }
}
```

In production behind the Caddy reverse proxy:

```json
{
  "mcpServers": {
    "spektr": {
      "type": "http",
      "url": "https://mcp.yourdomain.tld/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_MCP_API_KEY"
      }
    }
  }
}
```

!!! note "Legacy SSE"
    If a client only speaks SSE, set `MCP_TRANSPORT=sse` in `.env` and change `"type": "http"` → `"type": "sse"` in `.mcp.json`. The URL stays `{scheme}://{host}:{port}{MCP_PATH}`. SSE works but you will see benign `ClosedResourceError` tracebacks in server logs whenever a client disconnects mid-response.

!!! danger "Don't commit the Bearer token"
    The token is your `MCP_API_KEY` from `.env`. `.mcp.json` is gitignored in Spektr precisely because a per-machine token lives inside it. If you want a committable config, either leave auth disabled locally (`MCP_API_KEY=""`) or use env-var expansion if your client supports it.

## Claude Code specifics

- Claude Code reads `.mcp.json` at session start — there is **no hot reload**. After editing the file, exit (`/exit`) and run `claude` again.
- On startup Claude Code prompts once: *"Approve MCP server 'spektr'?"*. Accept → the tools are injected as `mcp__spektr__vector_search`, `mcp__spektr__graph_search`, `mcp__spektr__hybrid_search`, `mcp__spektr__list_documents`.
- `/mcp` slash command shows the active server list and connection state; useful for debugging a broken config.

## Cursor specifics

Cursor uses the same `.mcp.json` schema. Paste it under **Settings → MCP → Add new MCP server** → choose *From file*. Cursor doesn't require a session restart — the server reconnects on save.

## Pydantic AI (for a custom agent)

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP(
    "http://localhost:8080/mcp",
    headers={"Authorization": f"Bearer {settings.mcp_api_key}"} if settings.mcp_api_key else None,
)
agent = Agent(model="...", toolsets=[server])

async with server:
    result = await agent.run("What's in the knowledge base?")
```

For a legacy SSE server, import `MCPServerSSE` instead and point the URL at `http://HOST:PORT{MCP_PATH}` with `MCP_TRANSPORT=sse`.

See `agent/agent.py::create_rag_agent` and `scripts/ask.py` for the complete pattern used by `task ask`.

## Failure modes

| Symptom | Cause | Fix |
|-|-|-|
| Tools missing after restart | `.mcp.json` not in project root, or syntax error | `cat .mcp.json | jq` to validate; place at repo root |
| `connection refused` | `task serve` not running | Start it, or switch to stdio |
| `HTTP 401 Authentication required` | Bearer token wrong or missing | Copy `MCP_API_KEY` from `.env` into headers |
| Tool call returns `{"error": "..."}` | Server up but backend down | `task up`, then `task doctor` |

## Verifying the setup

In a new Claude Code session:

```
/mcp
```

should list `spektr` with status `connected`. Then ask:

> "What documents are in the knowledge base?"

I should call `mcp__spektr__list_documents` and show the per-doc chunk counts. If so, everything is wired.
