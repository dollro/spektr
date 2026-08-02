# Authentication

The MCP server uses a Bearer token middleware to protect tool calls. Authentication is optional and controlled by the `MCP_API_KEY` environment variable.

## Configuration

| Variable | Default | Description |
|-|-|-|
| `MCP_API_KEY` | `""` (empty) | Bearer token for tool call authentication |

- **Key set**: All `tools/call` **and `tools/list`** requests must include a valid `Authorization` header.
- **Key empty**: Authentication is disabled; all requests are allowed through.

## BearerAuthMiddleware

The middleware is implemented as a FastMCP `Middleware` subclass in `server/mcp_server.py`:

```python
_PROTECTED_METHODS = frozenset({"tools/call", "tools/list"})


class BearerAuthMiddleware(Middleware):
    async def __call__(self, context: MiddlewareContext, call_next):
        if settings.mcp_api_key and context.method in _PROTECTED_METHODS:
            auth = context.fastmcp_context.request_context.request.headers.get(
                "Authorization",
            )
            if not auth or not auth.startswith("Bearer "):
                raise PermissionError("Authentication required")
            if auth.removeprefix("Bearer ") != settings.mcp_api_key:
                raise PermissionError("Invalid token")
        return await call_next(context)
```

### Behavior

1. Checks if `MCP_API_KEY` is configured (non-empty).
2. Only intercepts requests whose `context.method` is in `_PROTECTED_METHODS` — `tools/call` and `tools/list`. `tools/list` is gated because it returns every tool name, description and input schema, so leaving it open lets an unauthenticated caller enumerate the whole surface. Other MCP methods (e.g. `initialize`, `ping`) pass through unauthenticated: a client must complete the handshake before it can be told anything, and the response carries none of your data.
3. Extracts the `Authorization` header from the incoming HTTP request.
4. Validates the header starts with `Bearer ` and the token matches `MCP_API_KEY`.
5. Raises `PermissionError` on failure, which FastMCP translates into an MCP error response.

## Client-Side Usage

Include the API key as a Bearer token in the `Authorization` header:

```
Authorization: Bearer <your-mcp-api-key>
```

For agents using the Pydantic AI MCP client, pass headers when constructing the transport. See [Agent Overview](../agent/overview.md) for a complete example.

## Error Responses

When authentication fails, the server raises a `PermissionError`. FastMCP converts this into an MCP-protocol error returned to the client:

| Condition | Error Message |
|-|-|
| Missing or malformed header | `"Authentication required"` |
| Token does not match | `"Invalid token"` |

## Live Ingest Authentication

The live ingest API (`ingestion/live_ingest.py`) uses a two-layer auth scheme: a global API key gates session creation, and per-session tokens gate data ingestion.

### Configuration

| Variable | Default | Description |
|-|-|-|
| `INGEST_API_KEY` | `""` (empty) | API key for `/session/start`. When empty, auth is disabled. |

### Flow

1. **Start session** — `POST /session/start` with `Authorization: Bearer <INGEST_API_KEY>`. Returns a `session_token`.
2. **Ingest chunks** — `POST /ingest/chunk` with `Authorization: Bearer <session_token>`.
3. **End session** — `POST /session/end` with `Authorization: Bearer <session_token>`. Token is invalidated.

### Design

- **Session tokens are ephemeral** — generated via `secrets.token_urlsafe(32)`, scoped to one session, wiped on session end
- **Blast radius is limited** — a leaked session token only grants access to that one session, not the pipeline
- **MCP and ingest auth are independent** — different credentials, different concerns (read vs write)
- **Timing-safe comparison** — both API key and session token validation use `secrets.compare_digest`

## Security Notes

- Store `MCP_API_KEY` and `INGEST_API_KEY` in your `.env` file. Never commit them to version control.
- When running in `stdio` transport mode, the MCP middleware still activates but the request context may not carry HTTP headers. Bearer auth is primarily effective with the `http` (streamable-http) or `sse` transports.
- For production deployments, consider placing both servers behind a reverse proxy with TLS termination.
