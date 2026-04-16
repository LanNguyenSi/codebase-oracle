# codebase-oracle

RAG-powered codebase Q&A for the Pandora ecosystem. Ask natural-language questions about ~50 repos and get answers with source citations — powered by LangChain.js.

[![CI](https://github.com/LanNguyenSi/codebase-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/LanNguyenSi/codebase-oracle/actions/workflows/ci.yml)

## How it works

```
~/git/pandora/**/*.ts,md,prisma
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
  [RAG Chain]  ── retrieved chunks + question → Claude/OpenAI → cited answer
```

The index pipeline scans all git repos under a root directory, splits source files into chunks with language-aware boundaries, embeds them via OpenAI, and stores the vectors locally. Queries retrieve the most relevant chunks and feed them to an LLM for answer generation with source citations.

## Prerequisites

- Node.js 22+
- `OPENAI_API_KEY` (required for embeddings)
- `ANTHROPIC_API_KEY` (optional, for Claude-powered answers — falls back to OpenAI)

## Quick start

```bash
git clone https://github.com/LanNguyenSi/codebase-oracle.git
cd codebase-oracle
npm install

# Set your API key
export OPENAI_API_KEY=sk-...

# Index all repos (takes 2-5 min depending on repo count)
npm run index

# Ask a question
npm run query -- "how does task_finish handle autoMerge?"
```

## CLI reference

### `index` — Build the vector index

```bash
npm run index                           # Index ~/git/pandora (default)
npm run index -- --path /other/root     # Custom root path
```

Scans all git repos, loads `.ts`, `.tsx`, `.js`, `.jsx`, `.md`, `.prisma`, `package.json`, and `tsconfig.json` files. Skips `node_modules`, `.git`, `dist`, `build`, and files over 200KB.

### `query` — Ask a question

```bash
npm run query -- "what is the performPrMerge helper?"
npm run query -- -r agent-tasks "how does the audit system work?"
npm run query -- -k 20 "list all MCP tools"
```

| Flag | Description |
|------|-------------|
| `-r, --repo <name>` | Filter results to a specific repo |
| `-k, --limit <n>` | Number of chunks to retrieve (default: 12) |

### `search` — Raw vector search

```bash
npm run dev -- search "evaluateTransitionRules"
npm run dev -- search -r agent-tasks "Prisma schema"
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
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for embeddings |
| `ANTHROPIC_API_KEY` | No | — | Anthropic API key for answer generation |
| `PANDORA_ROOT` | No | `~/git/pandora` | Root directory to scan for repos |
| `ORACLE_DATA_DIR` | No | `~/.codebase-oracle` | Directory for persisted index data |
| `ORACLE_EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `ORACLE_LLM_MODEL` | No | `claude-sonnet-4-20250514` | LLM model for answer generation |
| `ORACLE_VECTOR_STORE` | No | `directory` | `directory` (persisted) or `memory` (ephemeral) |

## Development

```bash
npm run build          # TypeScript compilation
npm test               # Run tests
npx tsc --noEmit       # Type check only
```

## Tech stack

- **LangChain.js** — document splitting, embeddings orchestration, RAG chain
- **OpenAI** — text-embedding-3-small for vector embeddings
- **Claude** (Anthropic) — answer generation with source citations
- **MCP SDK** — Model Context Protocol server for Claude Code integration
- **TypeScript** + **Zod** — type-safe configuration and validation
