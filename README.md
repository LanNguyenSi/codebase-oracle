# codebase-oracle

A shared semantic index over your local repos — built for agents first, humans second. Run it once, then let any local Claude Code (or other MCP-capable) session query 48+ repos via `oracle_search` / `oracle_query` without scanning or embedding on its own.

Powered by LangChain.js, with source citations.

[![CI](https://github.com/LanNguyenSi/codebase-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/LanNguyenSi/codebase-oracle/actions/workflows/ci.yml)

## How it works

```
~/git/**/*.ts,md,prisma
        |
        v
   [Scanner]  ── walk repos, filter files
        |
        v
   [Splitter] ── code-aware chunking (function/class boundaries)
        |
        v
  [Embeddings] ── OpenAI text-embedding-3-small
        |
        v
 [Vector Store] ── cosine similarity search (in-memory, persisted to disk)
        |
        v
  [RAG Chain]  ── retrieved chunks + question → Claude/OpenAI/Ollama → cited answer
```

The index pipeline scans all git repos under a root directory, splits source files into chunks with language-aware boundaries, embeds them via OpenAI-compatible APIs (OpenAI or Ollama), and stores the vectors locally. Queries retrieve the most relevant chunks and feed them to an LLM for answer generation with source citations.

## Two use cases

### For agents (primary)

A local Claude Code session (or any MCP client) talks to the oracle's MCP server over stdio. The agent runs `oracle_search` / `oracle_query` / `oracle_list_repos` against a shared, pre-built index — it never has to scan the filesystem, embed anything, or burn its own context on grep output.

**Why an agent would use this:**

- **One scan for everyone.** Index is built once; every session reuses it.
- **Semantic, not grep.** Cross-repo conceptual search ("where do we read `AGENT_TASKS_TOKEN`?") instead of regex chasing.
- **No duplicate embeddings.** Each agent session would otherwise pay embedding cost + time for its own index.
- **MCP-first design.** Tools return compact, citation-ready snippets — friendly to an agent's context window.

See [MCP server](#mcp-server-for-agents) below for setup.

### For humans

There's also a CLI — `npm run query -- "..."` — useful for spot checks, debugging the index, or when you want a terminal-driven answer without going through an agent. See [CLI reference](#cli-reference-for-humans).

## Prerequisites

- Node.js 22+
- `OPENAI_API_KEY` (required if `ORACLE_EMBEDDING_PROVIDER=openai`)
- `ANTHROPIC_API_KEY` (optional, for Claude-powered answers)
- Running Ollama locally if you want `ORACLE_*_PROVIDER=ollama`

## Quick start

```bash
git clone https://github.com/LanNguyenSi/codebase-oracle.git
cd codebase-oracle
npm install

# Set your API key
export OPENAI_API_KEY=sk-...

# Index all repos under ~/git (default)
npm run index
```

From here, either register the MCP server for agents (see [MCP server](#mcp-server-for-agents)) or use the CLI directly (see [CLI reference](#cli-reference-for-humans)).

The CLI and MCP server auto-load `.env` from the repo root if present.

### Ollama as provider

```bash
# Route embeddings + LLM through Ollama's OpenAI-compatible API
export ORACLE_EMBEDDING_PROVIDER=ollama
export ORACLE_LLM_PROVIDER=ollama
export ORACLE_OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_API_KEY=ollama

# Pick local models available in your Ollama instance
export ORACLE_EMBEDDING_MODEL=nomic-embed-text
export ORACLE_LLM_MODEL=llama3.1
```

## MCP server (for agents)

Register the oracle as an MCP tool in Claude Code:

```bash
claude mcp add codebase-oracle -- npx tsx src/mcp-server.ts
```

(Or run the server standalone with `npm run mcp` for local development.)

From that point on, any Claude Code session on the same machine can call the tools below — without its own scan, its own embeddings, or a separate API key per session.

### Tools

| Tool | Description |
|------|-------------|
| `oracle_query` | Ask a natural-language question, get an LLM answer with citations |
| `oracle_search` | Raw vector similarity search, returns code chunks |
| `oracle_list_repos` | List repos actually present in the index, with chunk and file counts |

### Example agent prompts

Once the MCP server is registered, an agent can issue calls like the following (shorthand — the actual tool inputs are `{ question }` for `oracle_query` and `{ query }` for `oracle_search`, both optionally with `repo`):

- `oracle_search` with `query="AGENT_TASKS_TOKEN"` — find every repo that reads the token, across all indexed repos
- `oracle_query` with `question="how does the audit system work?"` — cross-repo answer with citations
- `oracle_query` with `question="where is the embedding provider chosen?"`, `repo="codebase-oracle"` — scoped to a single repo
- `oracle_list_repos` — inventory of what the index actually covers

The returned chunks include file path + repo name, so the agent can read the full file only when it actually needs to.

## CLI reference (for humans)

### `query` — Ask a question

```bash
npm run query -- "what is the performPrMerge helper?"
npm run query -- -r my-repo "how does the audit system work?"
npm run query -- -k 20 "list all API endpoints"
```

| Flag | Description |
|------|-------------|
| `-r, --repo <name>` | Filter results to a specific repo |
| `-k, --limit <n>` | Number of chunks to retrieve (default: 12) |

### `search` — Raw vector search

```bash
npm run dev -- search "evaluateTransitionRules"
npm run dev -- search -r my-repo "Prisma schema"
```

Returns matching code chunks without LLM interpretation.

### `index` — Build the vector index

```bash
npm run index                           # Index ~/git (default scan root)
npm run index -- --path /path/to/repos  # Custom root path
```

Scans all git repos under the root directory. By default loads JS/TS sources (`.ts`, `.tsx`, `.js`, `.jsx`, `.vue`), docs (`.md`), sibling languages (`.py`, `.php`, `.go`, `.rs`, `.java`), config/infra (`.yaml`, `.yml`, `.toml`, `.sql`, `.prisma`, `.sh`), and the `package.json` / `tsconfig.json` manifests. Skips `node_modules`, `.git`, `dist`, `build`, and files over 200 KB. Override the extension allowlist with `ORACLE_INCLUDE_EXTENSIONS` (see below).

Indexing is incremental when `ORACLE_VECTOR_STORE=directory`: unchanged files are reused from persisted vectors (via file hashes), and only new/changed files are re-embedded. Progress is checkpointed batch-by-batch during embedding, so interrupted runs can resume without redoing all completed batches.

### `watch` — Keep the index fresh in the background

```bash
npm run watch                            # ~/git (default scan root), 3s debounce
npm run watch -- --path /path/to/repos   # custom root
npm run watch -- --debounce 5000         # 5s debounce window
```

Runs a [chokidar](https://github.com/paulmillr/chokidar) watcher over the scan root. File add/change/delete events are accumulated and, after a quiet period (default 3 s), processed in one batch: changed files are re-embedded, deleted files drop their vectors, vanished repo roots purge all their vectors. Editor save-storms (e.g. VS Code's atomic-rename trick) collapse into a single re-embed thanks to chokidar's `awaitWriteFinish` + the debounce. Newly dropped `.git` roots are detected and logged; back-fill them with one explicit `npm run index` so the first-time ingestion is consistent, then watch picks up subsequent edits.

Watch mode is additive — it does not replace `npm run index`, which remains the ground-truth bootstrap path.

**Known limitation:** a running stdio or HTTP MCP server does not hot-reload the store. If watch mode updates `embeddings.jsonl` while the MCP server is running, the server will serve stale vectors until it's restarted. A future task will add store-file-change detection to the MCP server side.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ORACLE_EMBEDDING_PROVIDER` | No | `openai` | Embedding provider: `openai` or `ollama` |
| `ORACLE_LLM_PROVIDER` | No | `auto` | LLM provider: `auto`, `anthropic`, `openai`, `ollama` |
| `OPENAI_API_KEY` | Conditionally | — | Required when `ORACLE_EMBEDDING_PROVIDER=openai`; also used for OpenAI LLM |
| `OPENAI_BASE_URL` | No | — | Override OpenAI-compatible base URL for OpenAI provider |
| `ANTHROPIC_API_KEY` | No | — | Anthropic API key for answer generation |
| `OLLAMA_API_KEY` | No | — | Optional API key for Ollama provider (defaults to `ollama`) |
| `ORACLE_OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama OpenAI-compatible base URL |
| `ORACLE_SCAN_ROOT` | No | `~/git` | Root directory to scan for git repos |
| `ORACLE_DATA_DIR` | No | `~/.codebase-oracle` | Directory for persisted index data |
| `ORACLE_EMBEDDING_MODEL` | No | `text-embedding-3-small` (OpenAI) / `nomic-embed-text` (Ollama) | Embedding model name for selected provider |
| `ORACLE_LLM_MODEL` | No | `claude-sonnet-4-20250514` (`auto`/Anthropic), `gpt-4o-mini` (OpenAI), `llama3.1` (Ollama) | LLM model name for selected provider |
| `ORACLE_VECTOR_STORE` | No | `directory` | `directory` (persisted) or `memory` (ephemeral) |
| `ORACLE_INCLUDE_EXTENSIONS` | No | _see scanner defaults_ | Comma-separated extension allowlist, replaces defaults entirely (e.g. `.ts,.py,.rb`). Leading dot optional. If you include `.json`, the built-in manifest filter (only `package.json`/`tsconfig.json`) is bypassed — you'll get every matching JSON file. |
| `ORACLE_HTTP_PORT` | No | `3100` | Port for the HTTP MCP server (`npm run serve`). |
| `ORACLE_HTTP_BIND` | No | `127.0.0.1` | Bind address for the HTTP MCP server. Any non-loopback value (e.g. `0.0.0.0`, LAN IP, IPv6 `::`) requires `ORACLE_HTTP_TOKEN` — the server refuses to start otherwise. |
| `ORACLE_HTTP_TOKEN` | No | — | Bearer token for the HTTP MCP server. When set, every `POST /mcp` request must carry `Authorization: Bearer <token>` (constant-time compare). `GET /health` stays open. |

## HTTP MCP auth

The HTTP MCP server (`npm run serve`) defaults to `127.0.0.1:3100` with no authentication — appropriate for a single local agent on the same machine. For LAN or remote use, set both `ORACLE_HTTP_BIND` (to e.g. `0.0.0.0`) **and** `ORACLE_HTTP_TOKEN`. The server refuses to start with an off-loopback bind and no token, so there is no accidental-exposure path.

The built-in auth is intentionally minimal: one bearer token, constant-time compared. No rate limits, no TLS, no mTLS. If you need those, put codebase-oracle behind a reverse proxy (nginx, Caddy, Cloudflare Tunnel) and let the proxy handle them.

## Embedding fingerprint

The index stores a fingerprint (`embeddingProvider`, `embeddingModel`, `dimension`) in a leading meta line of `embeddings.jsonl`. On load, codebase-oracle refuses to run the index against a different provider/model — you'd get silent garbage otherwise, because the query vector and the stored vectors would live in different embedding spaces (or differ in dimension, producing `NaN` scores).

If you change `ORACLE_EMBEDDING_PROVIDER` or `ORACLE_EMBEDDING_MODEL`, the next `npm run index` / `npm run query` / MCP call will fail fast with a clear message telling you to either delete `~/.codebase-oracle/` (to re-embed with the new model) or revert the env change. There is no automatic migration — the choice is yours.

Indexes created before this check exist without a fingerprint; they load with a warning and should be re-built at your earliest convenience.

## Development

```bash
npm run build          # TypeScript compilation
npm test               # Run tests
npx tsc --noEmit       # Type check only
```

## Tech stack

- **LangChain.js** — document splitting, embeddings orchestration, RAG chain
- **OpenAI-compatible APIs** — OpenAI and Ollama for embeddings/LLM
- **Claude** (Anthropic) — answer generation with source citations
- **MCP SDK** — Model Context Protocol server for Claude Code integration
- **TypeScript** + **Zod** — type-safe configuration and validation

## Credits

The core idea — exposing a semantic index of docs + code to agents through a single MCP endpoint — was inspired by [andrepester/rag-search-mcp](https://github.com/andrepester/rag-search-mcp). codebase-oracle is a Node/LangChain-flavoured take on the same concept, tuned for multi-repo JavaScript/TypeScript workspaces.
