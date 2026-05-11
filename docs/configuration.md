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
| `ORACLE_LLM_PROVIDER` | No | `auto` | LLM provider: `auto`, `anthropic`, `openai`, `openai-compatible`, `ollama`. `openai-compatible` is the generic lane for Groq, OpenRouter, Together, vLLM, etc.; `ollama` is kept as a deprecated alias for the same lane (prints a one-shot warning) |
| `ORACLE_LLM_BASE_URL` | Conditionally | — | Required when `ORACLE_LLM_PROVIDER=openai-compatible`. The provider's OpenAI-compatible `/v1` endpoint (e.g. `https://api.groq.com/openai/v1`) |
| `ORACLE_LLM_API_KEY` | Conditionally | — | API key for the `openai-compatible` lane. Kept separate from `OPENAI_API_KEY` so embedding and LLM keys never cross-leak |
| `OPENAI_API_KEY` | Conditionally | — | Required when `ORACLE_EMBEDDING_PROVIDER=openai`; also used for OpenAI LLM |
| `OPENAI_BASE_URL` | No | — | Override OpenAI-compatible base URL for OpenAI provider |
| `ANTHROPIC_API_KEY` | No | — | Anthropic API key for answer generation |
| `OLLAMA_API_KEY` | No | — | Optional API key for Ollama provider (defaults to `ollama`) |
| `ORACLE_OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama OpenAI-compatible base URL |
| `ORACLE_SCAN_ROOT` | No | `~/git` | Root directory to scan for git repos |
| `ORACLE_DATA_DIR` | No | `~/.codebase-oracle` | Directory for persisted index data |
| `ORACLE_EMBEDDING_MODEL` | No | `text-embedding-3-small` (OpenAI) / `nomic-embed-text` (Ollama) | Embedding model name for selected provider |
| `ORACLE_LLM_MODEL` | No | `claude-sonnet-4-6` (`auto`/Anthropic), `gpt-4o-mini` (OpenAI), `llama3.1` (Ollama) | LLM model name for selected provider |
| `ORACLE_VECTOR_STORE` | No | `directory` | `directory` (persisted) or `memory` (ephemeral) |
| `ORACLE_INCLUDE_EXTENSIONS` | No | _see scanner defaults_ | Comma-separated extension allowlist, replaces defaults entirely (e.g. `.ts,.py,.rb`). Leading dot optional. If you include `.json`, the built-in manifest filter (only `package.json`/`tsconfig.json`) is bypassed: you'll get every matching JSON file. |
| `ORACLE_SKIP_DIRS` | No | — | Comma-separated directory names to skip on top of the built-in defaults (see below). Append-only: defaults like `node_modules` and `.git` are always skipped, so this field is for repo-specific additions (`generated`, `fixtures`, etc). |
| `ORACLE_HTTP_PORT` | No | `3100` | Port for the HTTP MCP server (`npm run serve`) |
| `ORACLE_HTTP_BIND` | No | `127.0.0.1` | Bind address for the HTTP MCP server. Any non-loopback value (e.g. `0.0.0.0`, LAN IP, IPv6 `::`) requires `ORACLE_HTTP_TOKEN`: the server refuses to start otherwise |
| `ORACLE_HTTP_TOKEN` | No | — | Bearer token for the HTTP MCP server. When set, every `POST /mcp` request must carry `Authorization: Bearer <token>` (constant-time compare). `GET /health` stays open |

## Default scan filters

`npm run index` scans all git repos under `ORACLE_SCAN_ROOT`. By default it loads JS/TS sources (`.ts`, `.tsx`, `.js`, `.jsx`, `.vue`), docs (`.md`), sibling languages (`.py`, `.php`, `.go`, `.rs`, `.java`), config/infra (`.yaml`, `.yml`, `.toml`, `.sql`, `.prisma`, `.sh`), and the `package.json` / `tsconfig.json` manifests. Files over 200 KB are skipped.

### Default skip directories

The scanner prunes these directory names at any depth so vendored package caches and build output never reach the embedder:

- VCS / language runtimes: `.git`, `__pycache__`, `.venv`
- Build / cache output: `build`, `coverage`, `dist`, `.cache`, `.next`, `.nyc_output`, `.turbo`
- Package managers and their stores: `node_modules`, `.bun`, `.pnpm-store`, `.yarn`, `vendor`
- IDE / workspace caches: `.husky`, `.idea`, `.opencode-home`, `.vscode`

Override the extension allowlist with `ORACLE_INCLUDE_EXTENSIONS`. Add repo-specific directory names to the skip list with `ORACLE_SKIP_DIRS` (appended to the defaults above).

## OpenAI-compatible providers (Groq, OpenRouter, Together, vLLM, Ollama…)

Any inference endpoint that speaks the OpenAI `chat/completions` shape can serve the LLM step. Set the provider to `openai-compatible` and point the new env vars at the endpoint:

```bash
# Groq (fast hosted llama / kimi / gpt-oss)
export ORACLE_LLM_PROVIDER=openai-compatible
export ORACLE_LLM_BASE_URL=https://api.groq.com/openai/v1
export ORACLE_LLM_API_KEY=gsk_...
export ORACLE_LLM_MODEL=llama-3.3-70b-versatile
```

```bash
# Local Ollama (no key needed)
export ORACLE_LLM_PROVIDER=openai-compatible
export ORACLE_LLM_BASE_URL=http://localhost:11434/v1
export ORACLE_LLM_MODEL=llama3.1
```

```bash
# OpenRouter (aggregates many models behind one endpoint)
export ORACLE_LLM_PROVIDER=openai-compatible
export ORACLE_LLM_BASE_URL=https://openrouter.ai/api/v1
export ORACLE_LLM_API_KEY=sk-or-...
export ORACLE_LLM_MODEL=anthropic/claude-sonnet-4-6
```

The embedding lane stays independent. With `ORACLE_EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY=sk-…`, embeddings still hit OpenAI; only the LLM step routes through `ORACLE_LLM_*`. Setting both to point at Ollama keeps the entire pipeline local.

### Legacy ollama variables

`ORACLE_LLM_PROVIDER=ollama` plus `ORACLE_OLLAMA_BASE_URL` and `OLLAMA_API_KEY` still work and resolve to the same lane. The first call against a legacy config logs a one-shot deprecation warning. New env names (`ORACLE_LLM_BASE_URL`, `ORACLE_LLM_API_KEY`) take precedence when both are set, so you can migrate without an outage.
