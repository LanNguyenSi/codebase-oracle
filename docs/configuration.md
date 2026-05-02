# Configuration

All configuration is via environment variables. The CLI and MCP server auto-load `.env` from the repo root if present.

## Prerequisites

- Node.js 22+
- `OPENAI_API_KEY` (required if `ORACLE_EMBEDDING_PROVIDER=openai`)
- `ANTHROPIC_API_KEY` (optional, for Claude-powered answers)
- A running Ollama instance locally if you want `ORACLE_*_PROVIDER=ollama`

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ORACLE_EMBEDDING_PROVIDER` | No | `openai` | Embedding provider: `openai` or `ollama`. `stub` exists for tests only (deterministic 8-dim vectors, no network) and is refused under `NODE_ENV=production` |
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
| `ORACLE_INCLUDE_EXTENSIONS` | No | _see scanner defaults_ | Comma-separated extension allowlist, replaces defaults entirely (e.g. `.ts,.py,.rb`). Leading dot optional. If you include `.json`, the built-in manifest filter (only `package.json`/`tsconfig.json`) is bypassed: you'll get every matching JSON file. |
| `ORACLE_HTTP_PORT` | No | `3100` | Port for the HTTP MCP server (`npm run serve`) |
| `ORACLE_HTTP_BIND` | No | `127.0.0.1` | Bind address for the HTTP MCP server. Any non-loopback value (e.g. `0.0.0.0`, LAN IP, IPv6 `::`) requires `ORACLE_HTTP_TOKEN`: the server refuses to start otherwise |
| `ORACLE_HTTP_TOKEN` | No | — | Bearer token for the HTTP MCP server. When set, every `POST /mcp` request must carry `Authorization: Bearer <token>` (constant-time compare). `GET /health` stays open |

## Default scan filters

`npm run index` scans all git repos under `ORACLE_SCAN_ROOT`. By default it loads JS/TS sources (`.ts`, `.tsx`, `.js`, `.jsx`, `.vue`), docs (`.md`), sibling languages (`.py`, `.php`, `.go`, `.rs`, `.java`), config/infra (`.yaml`, `.yml`, `.toml`, `.sql`, `.prisma`, `.sh`), and the `package.json` / `tsconfig.json` manifests. Skips `node_modules`, `.git`, `dist`, `build`, and files over 200 KB. Override the extension allowlist with `ORACLE_INCLUDE_EXTENSIONS`.

## Ollama provider

Route embeddings and LLM through Ollama's OpenAI-compatible API:

```bash
export ORACLE_EMBEDDING_PROVIDER=ollama
export ORACLE_LLM_PROVIDER=ollama
export ORACLE_OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_API_KEY=ollama

# Pick local models available in your Ollama instance
export ORACLE_EMBEDDING_MODEL=nomic-embed-text
export ORACLE_LLM_MODEL=llama3.1
```

Embeddings stay local; nothing leaves your machine.
