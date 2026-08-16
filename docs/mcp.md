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

## Multi-repo workspace recipe (source checkout, user scope)

The battle-tested registration for a multi-repo workspace like `~/git/pandora`, where sessions start in changing directories and the oracle must be available in all of them (verified on the Mac mini, 2026-08-16):

1. **Prerequisites, once per machine:**
   - `.env` in the checkout with at least `ORACLE_SCAN_ROOT` (the directory containing your git repos) and the provider keys you use (`OPENAI_API_KEY`, or `ORACLE_EMBEDDING_PROVIDER` / `ORACLE_LLM_PROVIDER` / `ORACLE_LLM_BASE_URL` / `ORACLE_LLM_API_KEY` / `ORACLE_LLM_MODEL` for a local provider). The server resolves `.env` from its **working directory**, which is why the command below cds first.
   - `npm run build` in the checkout (the recipe runs the compiled `dist/`; rebuild after pulling changes, a stale `dist` serves stale behavior).
   - An index: `npm run index` (or wait for the first `oracle_reindex` call).

2. **Register at user scope** so the entry loads in every session regardless of cwd, with absolute paths only (the spawn environment's `PATH` is not guaranteed):

   ```bash
   claude mcp add -s user codebase-oracle -- sh -c 'cd /abs/path/to/codebase-oracle && exec /abs/path/to/node dist/mcp-server.js'
   ```

   The `sh -c 'cd ... && exec ...'` wrapper exists only to give the server its `.env`/index working directory; both paths inside it must be absolute.

3. **Verify** with `claude mcp list`: the entry must show `✔ Connected` (the health check performs a real stdio handshake). Then restart your Claude Code session: MCP tools only appear in sessions started **after** registration. In the fresh session, `oracle_list_repos` must return your indexed repos; assert that output, not just a silent non-error.

4. **Negative control** (proves the health check cannot silently pass): registering a wrong binary path shows a hard failure, e.g.

   ```
   oracle-negctl: sh -c exec /nonexistent/node ... - ✘ Failed to connect — CONNECTION_CLOSED
   ```

**Avoid local-scope registration for this use case.** A project-local entry (created by `claude mcp add` without `-s user` from inside one directory) connects only for sessions starting in that exact directory, and has been observed in a state where `claude mcp list` / `claude mcp get` could not see or remove it even though sessions still loaded it; the only cleanup path was editing `~/.claude.json` (`.projects["<dir>"].mcpServers`) by hand.

## Tools

| Tool | Description |
|------|-------------|
| `oracle_query` | Ask a natural-language question, get an LLM answer with citations |
| `oracle_search` | Raw vector similarity search, returns code chunks with `path:line_start-line_end (repo)` headers. Accepts optional `path_glob`, `type`, and `tags` filters |
| `oracle_expand` | Read a window of lines around a position in an indexed file (use after `oracle_search` for more context) |
| `oracle_list_repos` | List repos present in the index with chunk counts, file counts, and the indexed timestamp |
| `oracle_reindex` | Rebuild the incremental index from disk; new chunks visible to the next `oracle_search` / `oracle_query` call |

## OKF frontmatter filters (`type` / `tags`)

Markdown files whose content leads with a YAML frontmatter block get `fmType` / `fmTitle` / `fmTags` / `fmSources` chunk metadata (see [docs/architecture.md](architecture.md#chunking)). `oracle_search` accepts two optional filters on this metadata:

- `type` (string): matches chunks whose `fmType` strictly equals the given value.
- `tags` (array of strings): matches chunks whose `fmTags` contains **all** of the listed tags.

Both filters AND-compose with `repo` and `path_glob` (and with each other). A filter only matches chunks that **have** the corresponding field: chunks without frontmatter metadata (or missing that specific field) are excluded whenever `type` or `tags` is set, not treated as a wildcard match.

Like `path_glob`, `type` and `tags` are applied as a post-filter over an over-fetched window: the result count may fall short of `limit` for highly selective filters because that window is capped to keep the SQLite scan bounded. Raise `limit` if you need more matches.

Matching chunks show their `fmType` in the result header (e.g. `[3] docs/okf/backend.md:1-40 (agent-tasks) [module]`) and, when `fmSources` is present, an additional `sources: path1, path2` line. Chunks without frontmatter metadata render exactly as before.

`oracle_query` needs no new params: when any retrieved chunk carries `fmSources`, the answer gets a mechanically assembled `Pointers (from OKF sources metadata):` section appended after the sources list, listing the deduped union of `fmSources` paths in retrieval-rank order (capped at 10, with a `... and N more` note when truncated). No LLM involvement, and the section is omitted entirely when no retrieved chunk has `fmSources`.

## Example agent prompts

Once the MCP server is registered, an agent can issue calls like the following. The actual tool inputs are `{ question }` for `oracle_query` and `{ query }` for `oracle_search`, both optionally with `repo` plus `path_glob` / `type` / `tags` for `oracle_search`.

- `oracle_search` with `query="AGENT_TASKS_TOKEN"`: find every repo that reads the token, across all indexed repos.
- `oracle_search` with `query="tag-driven release"`, `path_glob="**/.github/workflows/*.yml"`: scoped to GitHub Actions workflow files only. picomatch semantics: `*` within a segment, `**` recursive, `?` single char, `{a,b}` alternatives.
- `oracle_search` with `query="dockerfile builder stage"`, `path_glob="**/Dockerfile*"`: every Dockerfile or `Dockerfile.prod` across the org.
- `oracle_search` with `query="backend service invariants"`, `type="module"`, `repo="agent-tasks"`: scoped to OKF docs whose frontmatter declares `type: module`.
- `oracle_search` with `query="okf backend"`, `tags=["okf", "backend"]`: scoped to chunks whose frontmatter `tags` include both `okf` and `backend`.
- `oracle_query` with `question="how does the audit system work?"`: cross-repo answer with citations.
- `oracle_query` with `question="where is the embedding provider chosen?"`, `repo="codebase-oracle"`: scoped to a single repo.
- `oracle_list_repos`: inventory of what the index actually covers, with freshness per repo.
- `oracle_reindex`: when you have just merged a PR and want its new chunks visible to the next query without waiting for the scheduled reindex.

The returned chunks include file path plus repo name, so the agent can read the full file only when it actually needs to.

## HTTP MCP

`npm run serve` starts the MCP server over Streamable HTTP instead of stdio. Defaults: `127.0.0.1:3100`, no authentication. Appropriate for a single local agent on the same machine.

The HTTP transport currently exposes four of the five stdio tools: `oracle_query`, `oracle_search`, `oracle_list_repos`, and `oracle_expand`. `oracle_reindex` is stdio-only, and the HTTP `oracle_search` does not accept the `path_glob` filter (it takes `query`, `repo`, `limit`, `type`, and `tags`). Use the stdio transport if you need on-demand reindexing or path-glob scoping.

For LAN or remote use, set both `ORACLE_HTTP_BIND` (to e.g. `0.0.0.0`) **and** `ORACLE_HTTP_TOKEN`. The server refuses to start with an off-loopback bind and no token, so there is no accidental-exposure path. Every `POST /mcp` request must then carry `Authorization: Bearer <token>` (constant-time compare). `GET /health` stays open.

The built-in auth is intentionally minimal: one bearer token, constant-time compared. No rate limits, no TLS, no mTLS. If you need those, put codebase-oracle behind a reverse proxy (nginx, Caddy, Cloudflare Tunnel) and let the proxy handle them.

See [docs/configuration.md](configuration.md) for `ORACLE_HTTP_*` env vars.
