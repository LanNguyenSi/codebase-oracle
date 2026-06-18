# MCP server

codebase-oracle ships an MCP (Model Context Protocol) server so any local Claude Code session, or other MCP-capable agent, can query a shared, pre-built index without scanning the filesystem itself, embedding anything, or burning context on grep output.

## Register with Claude Code

After `npm i -g @lannguyensi/codebase-oracle`:

```bash
claude mcp add codebase-oracle -- codebase-oracle mcp
```

From a local source checkout (no global install):

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
| `oracle_search` | Raw vector similarity search, returns code chunks with `path:line_start-line_end (repo)` headers. Accepts an optional `path_glob` filter |
| `oracle_expand` | Read a window of lines around a position in an indexed file (use after `oracle_search` for more context) |
| `oracle_list_repos` | List repos present in the index with chunk counts, file counts, and the indexed timestamp |
| `oracle_reindex` | Rebuild the incremental index from disk; new chunks visible to the next `oracle_search` / `oracle_query` call |

## Example agent prompts

Once the MCP server is registered, an agent can issue calls like the following. The actual tool inputs are `{ question }` for `oracle_query` and `{ query }` for `oracle_search`, both optionally with `repo` plus a `path_glob` for `oracle_search`.

- `oracle_search` with `query="AGENT_TASKS_TOKEN"`: find every repo that reads the token, across all indexed repos.
- `oracle_search` with `query="tag-driven release"`, `path_glob="**/.github/workflows/*.yml"`: scoped to GitHub Actions workflow files only. picomatch semantics: `*` within a segment, `**` recursive, `?` single char, `{a,b}` alternatives.
- `oracle_search` with `query="dockerfile builder stage"`, `path_glob="**/Dockerfile*"`: every Dockerfile or `Dockerfile.prod` across the org.
- `oracle_query` with `question="how does the audit system work?"`: cross-repo answer with citations.
- `oracle_query` with `question="where is the embedding provider chosen?"`, `repo="codebase-oracle"`: scoped to a single repo.
- `oracle_list_repos`: inventory of what the index actually covers, with freshness per repo.
- `oracle_reindex`: when you have just merged a PR and want its new chunks visible to the next query without waiting for the scheduled reindex.

The returned chunks include file path plus repo name, so the agent can read the full file only when it actually needs to.

## HTTP MCP

`npm run serve` starts the MCP server over Streamable HTTP instead of stdio. Defaults: `127.0.0.1:3100`, no authentication. Appropriate for a single local agent on the same machine.

The HTTP transport currently exposes four of the five stdio tools: `oracle_query`, `oracle_search`, `oracle_list_repos`, and `oracle_expand`. `oracle_reindex` is stdio-only, and the HTTP `oracle_search` does not accept the `path_glob` filter (it takes only `query`, `repo`, and `limit`). Use the stdio transport if you need on-demand reindexing or path-glob scoping.

For LAN or remote use, set both `ORACLE_HTTP_BIND` (to e.g. `0.0.0.0`) **and** `ORACLE_HTTP_TOKEN`. The server refuses to start with an off-loopback bind and no token, so there is no accidental-exposure path. Every `POST /mcp` request must then carry `Authorization: Bearer <token>` (constant-time compare). `GET /health` stays open.

The built-in auth is intentionally minimal: one bearer token, constant-time compared. No rate limits, no TLS, no mTLS. If you need those, put codebase-oracle behind a reverse proxy (nginx, Caddy, Cloudflare Tunnel) and let the proxy handle them.

See [docs/configuration.md](configuration.md) for `ORACLE_HTTP_*` env vars.
