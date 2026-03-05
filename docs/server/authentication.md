# Authentication

The MCP server uses a Bearer token middleware to protect tool calls. Authentication is optional and controlled by the `MCP_API_KEY` environment variable.

## Configuration

| Variable | Default | Description |
|-|-|-|
| `MCP_API_KEY` | `""` (empty) | Bearer token for tool call authentication |

- **Key set**: All `tools/call` requests must include a valid `Authorization` header.
- **Key empty**: Authentication is disabled; all requests are allowed through.

## BearerAuthMiddleware

The middleware is implemented as a FastMCP `Middleware` subclass in `server/mcp_server.py`:

```python
class BearerAuthMiddleware(Middleware):
    async def __call__(self, context: MiddlewareContext, call_next):
        if settings.mcp_api_key and context.method == "tools/call":
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
2. Only intercepts requests where `context.method == "tools/call"`. Other MCP methods (e.g. `tools/list`, `ping`) pass through unauthenticated.
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

## Security Notes

- Store `MCP_API_KEY` in your `.env` file. Never commit it to version control.
- When running in `stdio` transport mode, the middleware still activates but the request context may not carry HTTP headers. Bearer auth is primarily effective with the `sse` transport.
- For production deployments, consider placing the MCP server behind a reverse proxy with TLS termination.
