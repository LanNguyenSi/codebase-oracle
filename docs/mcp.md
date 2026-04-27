# MCP server

codebase-oracle ships an MCP (Model Context Protocol) server so any local Claude Code session, or other MCP-capable agent, can query a shared, pre-built index without scanning the filesystem itself, embedding anything, or burning context on grep output.

## Register with Claude Code

```bash
claude mcp add codebase-oracle -- npx tsx src/mcp-server.ts
```

Or run the server standalone for local development:

```bash
npm run mcp
```

From that point on, any Claude Code session on the same machine can call the tools below without a separate scan or API key per session.

## Tools

| Tool | Description |
|------|-------------|
| `oracle_query` | Ask a natural-language question, get an LLM answer with citations |
| `oracle_search` | Raw vector similarity search, returns code chunks with `path:line_start-line_end (repo)` headers |
| `oracle_expand` | Read a window of lines around a position in an indexed file (use after `oracle_search` for more context) |
| `oracle_list_repos` | List repos present in the index with chunk counts, file counts, and the indexed timestamp |

## Example agent prompts

Once the MCP server is registered, an agent can issue calls like the following. The actual tool inputs are `{ question }` for `oracle_query` and `{ query }` for `oracle_search`, both optionally with `repo`.

- `oracle_search` with `query="AGENT_TASKS_TOKEN"`: find every repo that reads the token, across all indexed repos.
- `oracle_query` with `question="how does the audit system work?"`: cross-repo answer with citations.
- `oracle_query` with `question="where is the embedding provider chosen?"`, `repo="codebase-oracle"`: scoped to a single repo.
- `oracle_list_repos`: inventory of what the index actually covers, with freshness per repo.

The returned chunks include file path plus repo name, so the agent can read the full file only when it actually needs to.

## HTTP MCP

`npm run serve` starts the same tools over Streamable HTTP instead of stdio. Defaults: `127.0.0.1:3100`, no authentication. Appropriate for a single local agent on the same machine.

For LAN or remote use, set both `ORACLE_HTTP_BIND` (to e.g. `0.0.0.0`) **and** `ORACLE_HTTP_TOKEN`. The server refuses to start with an off-loopback bind and no token, so there is no accidental-exposure path. Every `POST /mcp` request must then carry `Authorization: Bearer <token>` (constant-time compare). `GET /health` stays open.

The built-in auth is intentionally minimal: one bearer token, constant-time compared. No rate limits, no TLS, no mTLS. If you need those, put codebase-oracle behind a reverse proxy (nginx, Caddy, Cloudflare Tunnel) and let the proxy handle them.

See [docs/configuration.md](configuration.md) for `ORACLE_HTTP_*` env vars.
