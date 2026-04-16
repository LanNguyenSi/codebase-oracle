# codebase-oracle

RAG-powered codebase Q&A for multi-repo projects. Ask natural-language questions about your codebase and get answers with source citations — powered by LangChain.js.

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

# Ask a question
npm run query -- "how does the authentication middleware work?"
```

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

## CLI reference

### `index` — Build the vector index

```bash
npm run index                           # Index ~/git (default scan root)
npm run index -- --path /path/to/repos  # Custom root path
```

Scans all git repos under the root directory. Loads `.ts`, `.tsx`, `.js`, `.jsx`, `.md`, `.prisma`, `package.json`, and `tsconfig.json` files. Skips `node_modules`, `.git`, `dist`, `build`, and files over 200KB.

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

## MCP server

Use codebase-oracle as an MCP tool in Claude Code:

```bash
claude mcp add codebase-oracle -- npx tsx src/mcp-server.ts
```

### Tools

| Tool | Description |
|------|-------------|
| `oracle_query` | Ask a natural-language question, get an LLM answer with citations |
| `oracle_search` | Raw vector similarity search, returns code chunks |
| `oracle_list_repos` | List all indexed repos |

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
